"""Cases « personnages » : les zones jaunes de toonsplit deviennent les cases montées.

Deux jeux de cases coexistent dans le dossier d'un chapitre :

- les **cases de lecture** (``panels.json`` + ``panel_NNN.png``, Smart Slicer, inchangées) :
  c'est ce que l'analyse IA lit, bulles et narration comprises. Le script garde donc
  tout le texte du chapitre ;
- les **cases personnages** (``figures/panels.json`` + ``figures/panel_NNN.png``) : un
  recadrage en résolution native autour des personnes détectées d'une même case (tous les
  personnages d'une case regroupés dans une seule image, tête jamais coupée, zones sans
  tête écartées, voir :mod:`src.modules.toonsplit.figures`). Ce sont les seules images
  montées dans la vidéo.

``figures/figures_map.json`` relie chaque case personnage aux cases de lecture qu'elle
recouvre ; :func:`remap_analysis` traduit l'analyse (numéros de cases de lecture) en
numéros de cases personnages juste avant le montage. Rien n'est recalculé côté IA quand
les réglages des personnages changent. :func:`select_figures` écarte au passage les
personnages trop petits pour le cadre et les détections douteuses que le script n'a pas
choisies.

Les cases personnages se calculent sur le strip **reconstitué à partir des cases de
lecture** (gouttières remises en blanc) : même résultat que le chapitre vienne d'être
téléchargé ou non, et aucun retéléchargement pour les chapitres déjà traités.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from src.models.panel import Panel
from src.models.scene import ChapterAnalysis
from src.modules.slicer import GIANT_PANEL_HEIGHT, load_panels, save_panels

logger = logging.getLogger(__name__)

FIGURES_DIRNAME = "figures"
MAP_FILE = "figures_map.json"
PARAMS_FILE = "figures_params.json"
OVERLAY_FILE = "debug_overlay.jpg"
#: Version de l'extraction : l'incrémenter invalide les cases personnages déjà calculées.
FIGURES_VERSION = 2
#: Part minimale de la hauteur d'un personnage qu'une case de lecture doit recouvrir pour
#: lui être rattachée (un personnage à cheval sur deux cases appartient aux deux).
MIN_READING_SHARE = 0.3
#: Agrandissement au-delà duquel un personnage est trop petit pour être monté : il lui
#: faudrait plus de x3 pour remplir 90 % du cadre. Ce sont des silhouettes de fond
#: (villageois au loin), floues même agrandies par IA.
MAX_MONTAGE_FACTOR = 3.0
#: Confiance minimale du détecteur pour un personnage que le script n'a pas choisi (ajouté
#: entre deux cases clés) : en dessous, fausses détections (pointe de bulle prise pour une
#: personne). Pas plus haut : le détecteur note mal les gros plans de visage (0,41 à 0,49
#: mesurés). Les cases clés, choisies par l'IA pour le récit, n'y sont pas soumises.
MIN_EXTRA_SCORE = 0.4


@dataclass(frozen=True)
class FigureOptions:
    """Réglages de l'extraction (voir :func:`src.modules.toonsplit.figures.extract_figures`)."""

    margin: float = 0.0
    bubbles: str = "cut"
    require_head: bool = True
    #: Un seul recadrage par case pour tous ses personnages (boîte englobante).
    group: bool = True

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def strip_from_panels(panels: Sequence[Panel], *, background: int = 255) -> np.ndarray:
    """Strip RGB reconstitué : chaque case à sa place, gouttières retirées remises en fond uni."""
    if not panels:
        raise ValueError("aucune case pour reconstituer le strip")
    height = max(p.y_end for p in panels)
    width = max(p.width for p in panels)
    canvas = np.full((height, width, 3), background, np.uint8)
    for p in sorted(panels, key=lambda p: p.index):
        canvas[p.y_start:p.y_end, : p.width] = p.image
    return canvas


def reading_ids_for(y0: int, y1: int, reading: Sequence[Panel] | Sequence[dict], *, min_share: float = MIN_READING_SHARE) -> list[int]:
    """Cases de lecture recouvrant au moins ``min_share`` de la hauteur ``[y0, y1)`` (sinon la plus recouvrante)."""
    h = max(1, y1 - y0)
    spans = [(int(_get(p, "index")), int(_get(p, "y_start")), int(_get(p, "y_end"))) for p in reading]
    overlaps = [(max(0, min(y1, e) - max(y0, s)), i) for i, s, e in spans]
    ids = sorted(i for o, i in overlaps if o >= min_share * h)
    if not ids:
        best = max(overlaps, default=(0, -1))
        ids = [best[1]] if best[0] > 0 else []
    return ids


def _get(obj: Any, key: str) -> Any:
    return obj[key] if isinstance(obj, dict) else getattr(obj, key)


def build_figure_panels(
    strip_rgb: np.ndarray, options: FigureOptions = FigureOptions(), *,
    detect: Callable[[np.ndarray], list] | None = None,
) -> tuple[list[Panel], list[dict[str, Any]]]:
    """Cases personnages d'un strip RGB : ``(cases, entrées de la carte)``, dans l'ordre de lecture."""
    from src.modules.toonsplit import detectors
    from src.modules.toonsplit.figures import extract_figures
    from src.modules.toonsplit.pipeline import analyze_strip

    if detect is None and options.bubbles == "cut":
        # Bulles inutiles quand la zone jaune est prise telle quelle : têtes et personnes
        # seules, quatre fois plus rapide (le détecteur de bulles RT-DETR est le plus lourd).
        detect = lambda block: detectors.SUBJECT_DETECTOR(block)  # noqa: E731
    bgr = np.ascontiguousarray(strip_rgb[:, :, ::-1])
    result = analyze_strip(bgr, spec_provider=None, judge=None, detect=detect)
    figures = extract_figures(result, margin=options.margin, require_head=options.require_head,
                              bubbles=options.bubbles, group=options.group)  # type: ignore[arg-type]
    panels: list[Panel] = []
    entries: list[dict[str, Any]] = []
    for i, f in enumerate(figures):
        crop = np.ascontiguousarray(strip_rgb[f.y0:f.y1, f.x0:f.x1])
        panels.append(Panel(
            index=i, y_start=f.y0, y_end=f.y1, height=f.h, width=f.w,
            type="scroll_vertical" if f.h > GIANT_PANEL_HEIGHT else "static", image=crop,
        ))
        entries.append({**f.as_dict(), "index": i})
    return panels, entries


def _reading_signature(out_dir: Path) -> str:
    path = out_dir / "panels.json"
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else ""


def load_figures_map(out_dir: str | Path) -> list[dict[str, Any]] | None:
    """Carte des cases personnages déjà calculée, ou ``None``."""
    path = Path(out_dir) / FIGURES_DIRNAME / MAP_FILE
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def ensure_figures(
    out_dir: str | Path, options: FigureOptions = FigureOptions(), *, force: bool = False,
    detect: Callable[[np.ndarray], list] | None = None,
) -> list[dict[str, Any]]:
    """Calcule les cases personnages du chapitre, ou les réutilise si rien n'a changé.

    Réutilisées tant que les réglages, la version de l'extraction et les cases de lecture
    (empreinte de ``panels.json``) sont les mêmes. Renvoie la carte (une entrée par case
    personnage, avec ``reading_panels``) ; liste vide si le chapitre n'a aucun personnage.
    """
    out_dir = Path(out_dir)
    fig_dir = out_dir / FIGURES_DIRNAME
    params = {"version": FIGURES_VERSION, **options.as_dict(), "reading": _reading_signature(out_dir)}
    params_path, map_path = fig_dir / PARAMS_FILE, fig_dir / MAP_FILE
    if not force and params_path.is_file() and map_path.is_file() and (fig_dir / "panels.json").is_file():
        try:
            if json.loads(params_path.read_text(encoding="utf-8")) == params:
                return json.loads(map_path.read_text(encoding="utf-8"))
        except ValueError:
            pass
    reading = load_panels(out_dir)
    strip = strip_from_panels(reading)
    panels, entries = build_figure_panels(strip, options, detect=detect)
    for entry in entries:
        entry["reading_panels"] = reading_ids_for(entry["y0"], entry["y1"], reading)
    fig_dir.mkdir(parents=True, exist_ok=True)
    save_panels(panels, fig_dir)
    map_path.write_text(json.dumps(entries, indent=1), encoding="utf-8")
    params_path.write_text(json.dumps(params, indent=1), encoding="utf-8")
    _write_overlay(strip, entries, fig_dir / OVERLAY_FILE)
    logger.info("Cases personnages : %d sur %d case(s) de lecture (%s)", len(entries), len(reading), fig_dir)
    return entries


def _write_overlay(strip_rgb: np.ndarray, entries: Sequence[dict[str, Any]], path: Path, *, max_height: int = 12000) -> None:
    """Strip réduit avec les cases personnages encadrées et numérotées (contrôle visuel)."""
    scale = min(1.0, max_height / strip_rgb.shape[0], 400 / strip_rgb.shape[1])
    small = cv2.resize(strip_rgb, (max(1, round(strip_rgb.shape[1] * scale)), max(1, round(strip_rgb.shape[0] * scale))),
                       interpolation=cv2.INTER_AREA)
    view = np.ascontiguousarray(small[:, :, ::-1])
    for e in entries:
        p0 = (round(e["x0"] * scale), round(e["y0"] * scale))
        p1 = (round(e["x1"] * scale) - 1, round(e["y1"] * scale) - 1)
        cv2.rectangle(view, p0, p1, (0, 200, 230), 2)
        cv2.putText(view, str(e["index"]), (p0[0] + 3, p0[1] + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 230), 1, cv2.LINE_AA)
    ok, data = cv2.imencode(".jpg", view, [cv2.IMWRITE_JPEG_QUALITY, 80])
    if ok:
        path.write_bytes(data.tobytes())


def remap_analysis(analysis: ChapterAnalysis, figures_map: Sequence[dict[str, Any]]) -> ChapterAnalysis:
    """Analyse traduite en numéros de cases personnages (pour le montage).

    Chaque case de lecture d'une scène est remplacée par ses personnages, dans l'ordre de
    lecture. Une scène dont les cases n'ont aucun personnage reçoit le personnage le plus
    proche dans l'ordre de lecture : aucune scène ne reste sans image.
    """
    by_reading: dict[int, list[int]] = {}
    for entry in sorted(figures_map, key=lambda e: (e["y0"], e["x0"], e["index"])):
        for rid in entry.get("reading_panels", []):
            by_reading.setdefault(int(rid), []).append(int(entry["index"]))

    def translate(pids: Sequence[int]) -> list[int]:
        out: list[int] = []
        for pid in pids:
            for fid in by_reading.get(int(pid), []):
                if fid not in out:
                    out.append(fid)
        return out

    def nearest(pids: Sequence[int]) -> list[int]:
        if not figures_map or not pids:
            return []
        def distance(entry: dict[str, Any]) -> tuple[int, int]:
            reading = entry.get("reading_panels") or []
            return (min((abs(int(r) - int(p)) for r in reading for p in pids), default=10**9), int(entry["index"]))
        return [int(min(figures_map, key=distance)["index"])]

    scenes = []
    for scene in analysis.scenes:
        ids = translate(scene.panel_ids) or nearest(scene.panel_ids)
        heavy = [fid for fid in translate(scene.action_heavy_ids) if fid in ids]
        scenes.append(scene.model_copy(update={"panel_ids": ids, "action_heavy_ids": heavy}))
    beats = [beat.model_copy(update={"panel_ids": translate(beat.panel_ids)}) for beat in analysis.beats]
    return analysis.model_copy(update={"scenes": scenes, "beats": beats, "n_panels": len(figures_map)})


def _size(entry: dict[str, Any]) -> tuple[int, int]:
    return int(entry.get("w", entry["x1"] - entry["x0"])), int(entry.get("h", entry["y1"] - entry["y0"]))


def big_enough(entry: dict[str, Any], frame: tuple[int, int], *, max_factor: float = MAX_MONTAGE_FACTOR) -> bool:
    """Vrai si le personnage remplit 90 % du cadre avec un agrandissement d'au plus ``max_factor``."""
    from src.modules.upscaler import DEFAULT_FILL, target_factor

    width, height = _size(entry)
    return target_factor(width, height, frame[0], frame[1], fill=DEFAULT_FILL, max_factor=math.inf) <= max_factor


def select_figures(
    analysis: ChapterAnalysis, figures_map: Sequence[dict[str, Any]], frame: tuple[int, int], *,
    max_factor: float = MAX_MONTAGE_FACTOR, min_extra_score: float = MIN_EXTRA_SCORE,
) -> tuple[ChapterAnalysis, list[int]]:
    """Personnages à monter : ``(analyse traduite, numéros des cases personnages montables)``.

    Un personnage trop petit pour le cadre (voir :data:`MAX_MONTAGE_FACTOR`) n'est jamais
    monté, même choisi par le script : sa scène reçoit alors le personnage montable le plus
    proche. Parmi les autres, ceux que le script n'a pas choisis (le montage les ajoute
    entre deux cases clés) doivent en plus être détectés avec une confiance d'au moins
    ``min_extra_score``. Liste vide (analyse inchangée) si aucun personnage n'est assez grand.
    """
    big = [entry for entry in figures_map if big_enough(entry, frame, max_factor=max_factor)]
    if not big:
        return analysis, []
    display = remap_analysis(analysis, big)
    keys = {pid for scene in display.scenes for pid in scene.panel_ids}
    ids = sorted(int(e["index"]) for e in big if int(e["index"]) in keys or float(e.get("score", 1.0)) >= min_extra_score)
    return display, ids


__all__ = [
    "FIGURES_DIRNAME", "MAP_FILE", "PARAMS_FILE", "FIGURES_VERSION", "MAX_MONTAGE_FACTOR", "MIN_EXTRA_SCORE",
    "FigureOptions", "strip_from_panels", "reading_ids_for", "build_figure_panels", "load_figures_map",
    "ensure_figures", "remap_analysis", "big_enough", "select_figures",
]
