"""Chaîne complète : strip → blocs → détections → spec IA → recherche → juge → plans.

Point d'entrée public : :func:`split_strip` (``image -> list[Shot]``). :func:`analyze_strip`
rend en plus tout l'intermédiaire (détections, spec, candidats, verdicts) pour le
débogage et l'évaluation.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import cv2
import numpy as np

from src.modules.toonsplit import detectors
from src.modules.toonsplit.ai import AiJudge, AiSpecProvider, BlockSpec, JudgeVerdict, fallback_spec, make_client
from src.modules.toonsplit.blocks import Block, row_background_stats, segment_blocks
from src.modules.toonsplit.geometry import BUBBLE, FREE_TEXT, Box, iou1d
from src.modules.toonsplit.search import POLICIES, BlockPlan, Candidate, SearchParams, plan_block, window_report

logger = logging.getLogger(__name__)

SpecProvider = Callable[[np.ndarray, int], BlockSpec]
#: ``(bloc, fenêtres, spec, index, notes) -> verdict`` ; ``notes`` décrit la politique de chaque candidat.
Judge = Callable[..., JudgeVerdict]
JudgeMode = Literal["never", "auto", "always"]
#: En mode ``auto``, le juge n'est appelé que s'il y a un conflit ou si les deux meilleurs
#: candidats diffèrent vraiment (IoU vertical sous ce seuil) : un appel de moins sinon.
AUTO_JUDGE_IOU = 0.8


@dataclass(frozen=True)
class Shot:
    """Plan vidéo : fenêtre pleine largeur ``[y0, y1)`` du strip (résolution native).

    ``role`` vient de la spec (``key``, ``insert``, ``text_only``, ``transition``) ;
    ``text_only`` n'a pas d'image (texte pour la voix off seulement). ``pan`` : la fenêtre
    est plus haute que le cadre, à parcourir en panoramique (Ken Burns vertical).
    """

    y0: int
    y1: int
    role: str
    tts: tuple[str, ...]
    pan: bool
    source_block: int

    @property
    def has_image(self) -> bool:
        return self.role != "text_only"

    def as_dict(self) -> dict[str, Any]:
        return {"y0": self.y0, "y1": self.y1, "role": self.role, "tts": list(self.tts), "pan": self.pan,
                "source_block": self.source_block}


@dataclass
class BlockResult:
    block: Block
    boxes: list[Box]
    spec: BlockSpec
    spec_source: str
    plan: BlockPlan
    chosen: int = 0
    verdict: JudgeVerdict | None = None
    shots: list[Shot] = field(default_factory=list)

    @property
    def chosen_candidate(self) -> Candidate | None:
        if self.plan.mode != "single" or not self.plan.candidates:
            return None
        return self.plan.candidates[self.chosen]

    def as_dict(self) -> dict[str, Any]:
        cons = self.plan.constraints
        report = []
        if cons is not None:
            report = [window_report(y0, y1, cons) for y0, y1, _ in self._windows()]
        return {
            "index": self.block.index, "y0": self.block.y0, "y1": self.block.y1, "small": self.block.small,
            "spec_source": self.spec_source, "spec": self.spec.model_dump(), "mode": self.plan.mode,
            "conflict": self.plan.conflict, "relaxed": self.plan.relaxed,
            "keep_px": list(cons.keep) if cons is not None and cons.keep else None,
            "candidates": [c.as_dict() for c in self.plan.candidates], "chosen": self.chosen,
            "verdict": self.verdict.model_dump() if self.verdict else None,
            "boxes": [b.as_dict() for b in self.boxes], "window_reports": report,
            "shots": [s.as_dict() for s in self.shots],
        }

    def _windows(self) -> list[tuple[int, int, bool]]:
        return [(s.y0 - self.block.y0, s.y1 - self.block.y0, s.pan) for s in self.shots if s.has_image]


@dataclass
class SplitResult:
    width: int
    height: int
    blocks: list[BlockResult]
    source: str = ""
    ai_calls: int = 0
    #: Coût équivalent au tarif API des appels IA de cette analyse (rapporté par le CLI Claude).
    ai_cost_usd: float = 0.0

    @property
    def shots(self) -> list[Shot]:
        return [s for b in self.blocks for s in b.shots]

    def as_dict(self) -> dict[str, Any]:
        return {
            "source": self.source, "width": self.width, "height": self.height, "ai_calls": self.ai_calls,
            "ai_cost_usd": self.ai_cost_usd,
            "shots": [s.as_dict() for s in self.shots], "blocks": [b.as_dict() for b in self.blocks],
        }


def load_image(image: str | Path | np.ndarray) -> np.ndarray:
    """Image BGR (chemins non ASCII compris ; alpha aplati sur blanc)."""
    if isinstance(image, np.ndarray):
        return image
    data = np.fromfile(str(image), dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError(f"image illisible : {image}")
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif img.shape[2] == 4:
        alpha = img[:, :, 3:4].astype(np.float32) / 255.0
        img = (img[:, :, :3] * alpha + 255 * (1 - alpha)).astype(np.uint8)
    return img


def _needs_judge(plan: BlockPlan, mode: JudgeMode) -> bool:
    if mode == "never" or plan.mode != "single" or len(plan.candidates) < 2:
        return False
    if mode == "always":
        return True
    first, second = plan.candidates[0], plan.candidates[1]
    return plan.conflict or iou1d((first.y0, first.y1), (second.y0, second.y1)) < AUTO_JUDGE_IOU


def analyze_strip(
    image: str | Path | np.ndarray,
    *,
    spec_provider: SpecProvider | None | Literal["claude", "gemini"] = "claude",
    judge: Judge | None | Literal["claude", "gemini"] = "claude",
    judge_mode: JudgeMode = "auto",
    params: SearchParams = SearchParams(),
    detect: Callable[[np.ndarray], list[Box]] | None = None,
) -> SplitResult:
    """Analyse complète d'un strip.

    Args:
        spec_provider: ``"claude"`` (CLI local, défaut), ``"gemini"``, un appelable
            ``(bloc, index) -> BlockSpec`` (spec manuelle, faux client...), ou ``None`` : spec
            déduite des détections.
        judge: ``"claude"``, ``"gemini"``, un appelable, ou ``None`` (candidat n° 1 gardé).
        judge_mode: ``auto`` (conflit ou candidats vraiment différents), ``always``, ``never``.
        detect: détecteur ``bloc -> [Box]`` ; défaut :func:`detectors.detect_all`
            (points d'extension ``SUBJECT_DETECTOR`` et ``BUBBLE_DETECTOR``).
    """
    img = load_image(image)
    H, W = img.shape[:2]
    provider: Any = spec_provider
    if isinstance(spec_provider, str):
        provider = AiSpecProvider(make_client(spec_provider))
    judge_fn: Any = judge
    if isinstance(judge, str):
        shared = getattr(provider, "client", None)
        judge_fn = AiJudge(shared if getattr(shared, "name", None) == judge else make_client(judge))
    detect_fn = detect or detectors.detect_all
    blocks = segment_blocks(img, stats=row_background_stats(img))
    results: list[BlockResult] = []
    for block in blocks:
        crop = img[block.y0:block.y1]
        boxes = detect_fn(crop)
        if block.small and not any(b.kind in (BUBBLE, FREE_TEXT) for b in boxes):
            logger.info("Bloc %d (%d px) ignore : trop petit et sans texte", block.index, block.h)
            continue
        spec, source = _spec_for(provider, crop, block, boxes)
        plan = plan_block(crop.shape, detectors.cut_energy(crop), boxes, spec, params)
        result = BlockResult(block, boxes, spec, source, plan)
        if judge_fn is not None and _needs_judge(plan, judge_mode):
            windows = [(c.y0, c.y1) for c in plan.candidates]
            notes = [POLICIES.get(c.policy.split(":")[0], c.policy) for c in plan.candidates]
            try:
                verdict = judge_fn(crop, windows, spec, block.index, notes)
            except Exception as exc:  # noqa: BLE001 - le juge est facultatif (quota, réseau, réponse invalide)
                logger.warning("Juge du bloc %d indisponible (%s) : candidat 1 garde", block.index, exc)
            else:
                result.verdict = verdict
                if verdict.confident and 1 <= verdict.best <= len(windows):
                    result.chosen = verdict.best - 1
        result.shots = _shots(result)
        results.append(result)
    clients = {id(c): c for c in (getattr(provider, "client", None), getattr(judge_fn, "client", None)) if c is not None}
    calls = sum(getattr(c, "n_calls", 0) for c in clients.values())
    cost = sum(getattr(c, "cost_usd", 0.0) for c in clients.values())
    return SplitResult(W, H, results, str(image) if not isinstance(image, np.ndarray) else "", calls, round(cost, 4))


def _spec_for(provider: SpecProvider | None, crop: np.ndarray, block: Block, boxes: list[Box]) -> tuple[BlockSpec, str]:
    if provider is None:
        return fallback_spec(boxes, block.h), "fallback"
    hits = getattr(provider, "cache_hits", None)
    try:
        spec = provider(crop, block.index)
    except Exception as exc:  # noqa: BLE001 - quota, réseau, spec absente : on découpe quand même
        logger.warning("Spec IA du bloc %d indisponible (%s: %s) : spec deduite des detections",
                       block.index, type(exc).__name__, str(exc)[:200])
        return fallback_spec(boxes, block.h), "fallback"
    source = getattr(provider, "source", "custom")
    if hits is not None and getattr(provider, "cache_hits", None) != hits:
        source = "cache"
    return spec, source


def _shots(result: BlockResult) -> list[Shot]:
    block, spec, plan = result.block, result.spec, result.plan
    tts = tuple(spec.tts)
    if plan.mode == "none":
        return [Shot(block.y0, block.y1, spec.role, tts, False, block.index)]
    if plan.mode == "single":
        c = plan.candidates[result.chosen]
        return [Shot(block.y0 + c.y0, block.y0 + c.y1, spec.role, tts, False, block.index)]
    shots = []
    for i, (y0, y1, pan) in enumerate(plan.windows):
        shots.append(Shot(block.y0 + y0, block.y0 + y1, spec.role, tts if i == 0 else (), pan, block.index))
    return shots


def split_strip(image_path: str | Path | np.ndarray, **kwargs: Any) -> list[Shot]:
    """Découpe un strip en plans vidéo (voir :func:`analyze_strip` pour les options)."""
    return analyze_strip(image_path, **kwargs).shots


def save_shots(img: np.ndarray, shots: Sequence[Shot], out_dir: Path) -> list[Path]:
    """Écrit chaque plan avec image en PNG, en résolution native (aucun redimensionnement)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for n, shot in enumerate(s for s in shots if s.has_image):
        path = out_dir / f"shot_{n:03d}_b{shot.source_block:02d}_{shot.role}{'_pan' if shot.pan else ''}.png"
        ok, data = cv2.imencode(".png", img[shot.y0:shot.y1])
        if ok:
            path.write_bytes(data.tobytes())
            paths.append(path)
    return paths


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")


__all__ = [
    "Shot", "BlockResult", "SplitResult", "SpecProvider", "Judge", "JudgeMode", "AUTO_JUDGE_IOU",
    "load_image", "analyze_strip", "split_strip", "save_shots", "write_json",
]
