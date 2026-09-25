"""Évaluation : crops automatiques contre un jeu de référence fait à la main.

Un **cas** est un dossier contenant le strip (``strip.webp|png|jpg``) et :

- ``reference.json`` : ``{"crops": [{"y0": .., "y1": .., "label": ".."}]}`` en pixels du strip
  natif ; ou, à défaut,
- ``crops/`` : les crops découpés à la main (images pleine largeur, à n'importe quelle
  échelle), retrouvés dans le strip par corrélation (:func:`locate_crops`) puis écrits
  dans ``reference.json`` ;
- ``spec_manual.json`` (facultatif) : specs écrites à la main, pour ``--spec manual``.

Un dossier qui n'a que le strip est un cas **sans référence** : pas d'IoU, mais les
violations et la planche HTML restent calculées (revue humaine).

Métriques (les boîtes « vérité » sont celles des détecteurs du bloc, faute d'annotation) :

- **violations dures**, cible 0 : tête coupée, bulle (ou narration) coupée ;
- **violations souples** : narration incluse, sujet décentré (centre de la zone à garder à
  plus de 25 % de la hauteur du crop de son centre), onomatopée tranchée ;
- **IoU 1D** avec le crop de référence, métrique secondaire (plusieurs crops sont valides) ;
- une **planche HTML** côte à côte pour la revue humaine.
"""

from __future__ import annotations

import html
import json
import logging
import statistics
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from src.modules.toonsplit.geometry import iou1d, overlap1d
from src.modules.toonsplit.pipeline import BlockResult, SplitResult, load_image, write_json
from src.modules.toonsplit.search import Constraints, window_report

logger = logging.getLogger(__name__)

STRIP_NAMES = ("strip.webp", "strip.png", "strip.jpg", "strip.jpeg")
IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".webp")
#: Sujet décentré au-delà de cette part de la hauteur du crop.
OFFCENTER_LIMIT = 0.25
#: Corrélation minimale pour accepter un crop retrouvé dans le strip.
MIN_MATCH = 0.85


@dataclass(frozen=True)
class RefCrop:
    y0: int
    y1: int
    label: str = ""
    match: float | None = None


@dataclass
class Case:
    name: str
    directory: Path
    strip: Path
    refs: list[RefCrop]
    spec_manual: Path | None = None


# --- Jeu de référence -----------------------------------------------------------------------
def locate_crops(strip: np.ndarray, crops: Sequence[tuple[str, np.ndarray]]) -> list[RefCrop]:
    """Retrouve des crops pleine largeur dans le strip (corrélation normalisée, grossier puis fin)."""
    gray = cv2.cvtColor(strip, cv2.COLOR_BGR2GRAY)
    H, W = gray.shape
    factor = 4
    small = cv2.resize(gray, (W // factor, H // factor), interpolation=cv2.INTER_AREA)
    found: list[RefCrop] = []
    for label, crop in crops:
        c = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
        h = max(1, round(c.shape[0] * W / c.shape[1]))
        c = cv2.resize(c, (W, h), interpolation=cv2.INTER_AREA)
        if h > H:
            raise ValueError(f"{label} : plus haut que le strip une fois mis à sa largeur")
        cs = cv2.resize(c, (W // factor, max(1, h // factor)), interpolation=cv2.INTER_AREA)
        coarse = cv2.matchTemplate(small, cs, cv2.TM_CCOEFF_NORMED)
        y = int(np.argmax(coarse[:, 0])) * factor
        lo, hi = max(0, y - 3 * factor), min(H - h, y + 3 * factor)
        fine = cv2.matchTemplate(gray[lo:hi + h], c, cv2.TM_CCOEFF_NORMED)
        dy = int(np.argmax(fine[:, 0]))
        score = float(fine[dy, 0])
        if score < MIN_MATCH:
            logger.warning("%s : correlation faible (%.2f), position douteuse", label, score)
        found.append(RefCrop(lo + dy, lo + dy + h, label, round(score, 4)))
    return found


def _find_strip(directory: Path) -> Path | None:
    for name in STRIP_NAMES:
        if (directory / name).is_file():
            return directory / name
    return None


def load_case(directory: Path) -> Case | None:
    """Cas d'évaluation d'un dossier (``None`` s'il n'en est pas un)."""
    strip = _find_strip(directory)
    if strip is None:
        return None
    ref_file = directory / "reference.json"
    crops_dir = directory / "crops"
    if ref_file.is_file():
        data = json.loads(ref_file.read_text(encoding="utf-8"))
        refs = [RefCrop(int(c["y0"]), int(c["y1"]), str(c.get("label", "")), c.get("match")) for c in data["crops"]]
    elif crops_dir.is_dir():
        paths = sorted(p for p in crops_dir.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
        refs = locate_crops(load_image(strip), [(p.stem, load_image(p)) for p in paths])
        write_json(ref_file, {"strip": strip.name, "crops": [r.__dict__ for r in refs],
                              "note": "positions retrouvees par correlation depuis crops/"})
    else:
        refs = []  # cas sans référence : violations et planche seulement
    spec = directory / "spec_manual.json"
    return Case(directory.name, directory, strip, sorted(refs, key=lambda r: r.y0), spec if spec.is_file() else None)


def find_cases(root: Path) -> list[Case]:
    """Le dossier lui-même s'il est un cas, sinon ses sous-dossiers (récursivement, ordre alphabétique)."""
    case = load_case(root)
    if case is not None:
        return [case]
    cases: list[Case] = []
    for sub in sorted(p for p in root.iterdir() if p.is_dir()):
        cases += find_cases(sub)
    return cases


# --- Métriques ------------------------------------------------------------------------------
@dataclass
class ShotEval:
    block: int
    y0: int
    y1: int
    pan: bool
    report: dict[str, float]


@dataclass
class RefEval:
    label: str
    y0: int
    y1: int
    block: int | None
    iou: float
    best_shot: tuple[int, int] | None
    report: dict[str, float]


@dataclass
class CaseEval:
    case: str
    refs: list[RefEval]
    shots: list[ShotEval]
    ai_calls: int = 0
    ai_cost_usd: float = 0.0
    notes: list[str] = field(default_factory=list)

    def summary(self) -> dict[str, Any]:
        return summarize([self])


def _block_of(result: SplitResult, y0: int, y1: int) -> BlockResult | None:
    best, best_overlap = None, 0
    for b in result.blocks:
        o = overlap1d(y0, y1, b.block.y0, b.block.y1)
        if o > best_overlap:
            best, best_overlap = b, o
    return best


def _report(cons: Constraints | None, y0: int, y1: int, offset: int) -> dict[str, float]:
    if cons is None:
        return {}
    return window_report(y0 - offset, y1 - offset, cons)


def evaluate(case: Case, result: SplitResult, windows: Sequence[tuple[int, int, int, bool]] | None = None) -> CaseEval:
    """Compare les plans (ceux de ``result``, ou ``windows`` externes ``(bloc, y0, y1, pan)``) aux références.

    Les violations sont mesurées avec les contraintes du bloc calculées par ``result``
    (mêmes détections, même spec) : deux prédictions sont jugées sur la même vérité.
    """
    by_index = {b.block.index: b for b in result.blocks}
    if windows is None:
        windows = [(s.source_block, s.y0, s.y1, s.pan) for s in result.shots if s.has_image]
    shots = []
    for block_index, y0, y1, pan in windows:
        b = by_index.get(block_index)
        cons = b.plan.constraints if b else None
        shots.append(ShotEval(block_index, y0, y1, pan, _report(cons, y0, y1, b.block.y0 if b else 0)))
    refs = []
    for ref in case.refs:
        b = _block_of(result, ref.y0, ref.y1)
        candidates = [(s.y0, s.y1) for s in shots if b is not None and s.block == b.block.index]
        best = max(candidates, key=lambda w: iou1d(w, (ref.y0, ref.y1)), default=None)
        cons = b.plan.constraints if b else None
        refs.append(RefEval(
            ref.label, ref.y0, ref.y1, b.block.index if b else None,
            round(iou1d(best, (ref.y0, ref.y1)), 4) if best else 0.0, best,
            _report(cons, ref.y0, ref.y1, b.block.y0 if b else 0),
        ))
    return CaseEval(case.name, refs, shots, result.ai_calls, result.ai_cost_usd)


def summarize(evals: Sequence[CaseEval]) -> dict[str, Any]:
    ious = [r.iou for e in evals for r in e.refs]
    shots = [s for e in evals for s in e.shots]
    refs = [r for e in evals for r in e.refs]

    def total(items: Sequence[Any], key: str) -> int:
        return int(sum(i.report.get(key, 0) for i in items))

    return {
        "cases": len(evals), "references": len(refs), "shots": len(shots),
        "iou_mean": round(statistics.fmean(ious), 4) if ious else None,
        "iou_median": round(statistics.median(ious), 4) if ious else None,
        "iou_min": round(min(ious), 4) if ious else None,
        "share_iou_ge_0_7": round(sum(i >= 0.7 for i in ious) / len(ious), 4) if ious else None,
        "hard": {"head_cuts": total(shots, "head_cuts"), "bubble_cuts": total(shots, "bubble_cuts")},
        "soft": {
            "narration_included": total(shots, "narration_included"),
            "subject_offcenter": sum(s.report.get("subject_offcenter", 0) > OFFCENTER_LIMIT for s in shots),
            "sfx_cuts": total(shots, "sfx_cuts"),
        },
        "reference_violations": {
            "head_cuts": total(refs, "head_cuts"), "bubble_cuts": total(refs, "bubble_cuts"),
            "narration_included": total(refs, "narration_included"), "sfx_cuts": total(refs, "sfx_cuts"),
        },
        "ai_calls": sum(e.ai_calls for e in evals),
        "ai_cost_usd": round(sum(e.ai_cost_usd for e in evals), 4),
    }


# --- Rendu ----------------------------------------------------------------------------------
COLORS = {
    "head": (0, 0, 230), "person": (0, 200, 230), "bubble": (230, 120, 0), "narration": (200, 0, 160),
    "sfx": (0, 140, 255), "free_text": (0, 140, 255), "auto": (40, 180, 40), "ref": (200, 0, 200),
    "candidate": (150, 150, 150), "keep": (60, 200, 60), "drop": (60, 60, 220),
}


def render_block(img: np.ndarray, result: BlockResult, refs: Sequence[tuple[int, int]] = (), *, width: int = 360) -> np.ndarray:
    """Vue du bloc : détections, zones keep/drop (barres latérales), candidats, plan retenu, référence."""
    b = result.block
    view = img[b.y0:b.y1].copy()
    H, W = view.shape[:2]
    t = max(2, round(W / 160))
    cons = result.plan.constraints
    boxes = list(result.boxes)
    if cons is not None:
        boxes = cons.heads_raw + cons.persons + cons.bubbles + cons.narration + cons.sfx
        bar = max(8, W // 40)
        if cons.keep:
            cv2.rectangle(view, (0, cons.keep[0]), (bar, cons.keep[1] - 1), COLORS["keep"], -1)
        for z in cons.drops:
            cv2.rectangle(view, (W - bar, max(0, z.y0)), (W - 1, min(H, z.y1) - 1), COLORS["drop"], -1)
    for box in boxes:
        cv2.rectangle(view, (box.x0, box.y0), (box.x1 - 1, box.y1 - 1), COLORS.get(box.kind, (90, 90, 90)),
                      t if box.kind != "person" else max(1, t // 2))
    for c in result.plan.candidates:
        cv2.rectangle(view, (3 * t, c.y0), (W - 1 - 3 * t, c.y1 - 1), COLORS["candidate"], max(1, t // 2))
    for y0, y1 in refs:
        cv2.rectangle(view, (6 * t, y0 - b.y0), (W - 1 - 6 * t, y1 - b.y0 - 1), COLORS["ref"], 2 * t)
    for s in result.shots:
        if s.has_image:
            cv2.rectangle(view, (t, s.y0 - b.y0), (W - 1 - t, s.y1 - b.y0 - 1), COLORS["auto"], 2 * t)
    return cv2.resize(view, (width, max(1, round(H * width / W))), interpolation=cv2.INTER_AREA)


def _thumb(img: np.ndarray, height: int = 420) -> np.ndarray:
    h, w = img.shape[:2]
    return cv2.resize(img, (max(1, round(w * height / h)), height), interpolation=cv2.INTER_AREA)


def _save_jpg(path: Path, img: np.ndarray) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, data = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 85])
    if ok:
        path.write_bytes(data.tobytes())
    return path.name


def _fmt(report: dict[str, float]) -> str:
    if not report:
        return ""
    bits = [f"ratio {report.get('ratio', 0):.2f}"]
    for key, label in (("head_cuts", "tete coupee"), ("bubble_cuts", "bulle coupee"),
                       ("narration_included", "narration incluse"), ("sfx_cuts", "sfx tranchee")):
        if report.get(key):
            bits.append(f"<b>{label} x{int(report[key])}</b>")
    if report.get("subject_offcenter", 0) > OFFCENTER_LIMIT:
        bits.append(f"<b>decentre {report['subject_offcenter']:.2f}</b>")
    return ", ".join(bits)


def write_html(out_dir: Path, runs: Sequence[tuple[Case, SplitResult, CaseEval]], title: str, summary: dict[str, Any]) -> Path:
    """Planche HTML : une ligne par bloc (vue annotée, plan auto, crop de référence, métriques)."""
    img_dir = out_dir / "img"
    rows: list[str] = []
    for case, result, ev in runs:
        strip = load_image(case.strip)
        rows.append(f"<h2>{html.escape(case.name)}</h2><pre>{html.escape(json.dumps(ev.summary(), indent=1))}</pre>")
        rows.append("<table><tr><th>bloc</th><th>vue (vert = auto, magenta = reference)</th><th>auto</th>"
                    "<th>reference</th><th>details</th></tr>")
        for b in result.blocks:
            refs = [r for r in ev.refs if r.block == b.block.index]
            view = _save_jpg(img_dir / f"{case.name}_b{b.block.index:02d}_view.jpg",
                             render_block(strip, b, [(r.y0, r.y1) for r in refs]))
            autos = [s for s in ev.shots if s.block == b.block.index]
            auto_cells = "".join(
                f'<img src="img/{_save_jpg(img_dir / f"{case.name}_b{b.block.index:02d}_auto{k}.jpg", _thumb(strip[s.y0:s.y1]))}">'
                f'<div>{s.y0}-{s.y1}{" pan" if s.pan else ""}<br>{_fmt(s.report)}</div>'
                for k, s in enumerate(autos)
            ) or f"<i>pas d'image ({html.escape(b.spec.role)})</i>"
            ref_cells = "".join(
                f'<img src="img/{_save_jpg(img_dir / f"{case.name}_b{b.block.index:02d}_ref{k}.jpg", _thumb(strip[r.y0:r.y1]))}">'
                f'<div>{html.escape(r.label)} {r.y0}-{r.y1}<br>IoU {r.iou:.2f}<br>{_fmt(r.report)}</div>'
                for k, r in enumerate(refs)
            ) or "<i>-</i>"
            verdict = f"juge : {b.verdict.best} ({'sur' if b.verdict.confident else 'incertain'}) - {html.escape(b.verdict.reason)}" if b.verdict else ""
            detail = (f"role <b>{html.escape(b.spec.role)}</b> ({b.spec_source}), mode {b.plan.mode}"
                      f"{', conflit' if b.plan.conflict else ''}{', degrade' if b.plan.relaxed else ''}<br>"
                      f"candidats : {', '.join(f'{c.y0 + b.block.y0}-{c.y1 + b.block.y0} [{c.policy}]' for c in b.plan.candidates)}"
                      f"<br>{verdict}<br>tts : {html.escape(' | '.join(b.spec.tts))}")
            rows.append(f'<tr><td>{b.block.index}<br>{b.block.y0}-{b.block.y1}</td><td><img src="img/{view}"></td>'
                        f"<td>{auto_cells}</td><td>{ref_cells}</td><td class=d>{detail}</td></tr>")
        rows.append("</table>")
    page = f"""<!doctype html><html lang="fr"><head><meta charset="utf-8"><title>{html.escape(title)}</title>
<style>body{{font-family:system-ui,sans-serif;margin:16px;background:#fafafa;color:#222}}
table{{border-collapse:collapse;margin-bottom:32px}}td,th{{border:1px solid #ccc;padding:6px;vertical-align:top}}
td img{{max-height:420px;display:block}}td.d{{max-width:360px;font-size:13px}}pre{{font-size:12px;background:#eee;padding:8px}}
.legend span{{display:inline-block;margin-right:14px}}</style></head><body>
<h1>{html.escape(title)}</h1>
<p class=legend><span style="color:#28b428">vert : plan auto</span><span style="color:#c800c8">magenta : reference</span>
<span style="color:#e60000">rouge : tete</span><span style="color:#0078e6">bleu : bulle</span>
<span style="color:#a000c8">violet : narration</span><span style="color:#ff8c00">orange : onomatopee</span>
<span style="color:#e6c800">jaune : personne</span><span>barre gauche : zone a garder, barre droite : zones a exclure</span></p>
<h2>Synthese</h2><pre>{html.escape(json.dumps(summary, indent=1))}</pre>
{''.join(rows)}</body></html>"""
    path = out_dir / "index.html"
    path.write_text(page, encoding="utf-8")
    return path


__all__ = [
    "RefCrop", "Case", "locate_crops", "load_case", "find_cases", "ShotEval", "RefEval", "CaseEval", "evaluate",
    "summarize", "render_block", "write_html", "OFFCENTER_LIMIT",
]
