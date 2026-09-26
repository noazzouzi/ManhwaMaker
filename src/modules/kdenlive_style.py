"""Style dynamique du rendu Kdenlive : quelle transition à chaque coupe, quel mouvement, quelle couleur.

Décisions de montage seulement, sans XML (voir :mod:`src.modules.kdenlive_builder`) :

- **une transition à chaque coupe**, choisie par l'émotion de la scène qui arrive. Dans une
  scène, elle est courte et discrète ; au changement de scène, plus marquée. Les listes
  sont parcourues à tour de rôle pour ne jamais enchaîner deux fois le même effet ;
- **mouvement varié** des cases : zoom avant ou arrière, vers le centre ou un bord, toujours
  entre 100 et 105 % (règle du format long) ;
- **impact** sur les cases d'action (``punch_in``) : flash, tremblement, décalage des couleurs ;
- **étalonnage** léger par émotion et vignettage, appliqués une fois à l'image de chaque
  case (aucun coût au rendu).
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import count

import numpy as np

from src.models.timeline import PanelClip

__all__ = [
    "Cut", "MIX_KINDS", "plan_cuts", "motion_pattern", "MOTION_PATTERNS", "grade", "GRADES", "INTRA_S", "SCENE_S",
]

#: Transitions qui chevauchent les deux cases (fondu enchaîné sur la même piste).
MIX_KINDS = frozenset({"dissolve", "blur_dissolve", "wipe", "push", "slide", "whip"})
#: Durée visée : coupe à l'intérieur d'une scène, changement de scène, fondu au noir,
#: changement de chapitre (compilation).
INTRA_S, SCENE_S, DIP_S, CHAPTER_S = 0.25, 0.40, 0.50, 1.0
#: Part maximale de la plus courte des deux cases prise par une transition, et durée
#: sous laquelle on préfère une coupe franche.
MAX_CLIP_RATIO, MIN_S = 0.30, 0.12

# (genre, direction ou fichier de volet) : direction pour push / slide / whip.
_INTRA: dict[str, list[tuple[str, str]]] = {
    "action": [("push", "left"), ("whip", "right"), ("push", "up"), ("whip", "left")],
    "epic": [("push", "up"), ("dissolve", ""), ("push", "left")],
    "tension": [("blur_dissolve", ""), ("glitch", ""), ("push", "down")],
    "fear": [("blur_dissolve", ""), ("glitch", "")],
    "mystery": [("blur_dissolve", ""), ("wipe", "radial.pgm"), ("dissolve", "")],
    "sad": [("dissolve", ""), ("blur_dissolve", "")],
    "romance": [("dissolve", ""), ("blur_dissolve", "")],
    "calm": [("dissolve", ""), ("slide", "left"), ("dissolve", "")],
    "happy": [("push", "left"), ("slide", "up"), ("push", "right")],
    "humor": [("push", "left"), ("slide", "up"), ("push", "right")],
    "neutral": [("push", "left"), ("dissolve", ""), ("slide", "right"), ("wipe", "linear_x.pgm")],
}
_SCENE: dict[str, list[tuple[str, str]]] = {
    "action": [("whip", "left"), ("dip_white", ""), ("whip", "up")],
    "epic": [("dip_white", ""), ("whip", "up")],
    "tension": [("glitch", ""), ("dip_black", "")],
    "fear": [("glitch", ""), ("dip_black", "")],
    "mystery": [("wipe", "clock.pgm"), ("dip_black", ""), ("wipe", "radial.pgm")],
    "sad": [("dip_black", ""), ("dissolve", "")],
    "romance": [("dissolve", ""), ("dip_white", "")],
    "calm": [("dip_black", ""), ("dissolve", "")],
    "happy": [("push", "right"), ("wipe", "bi-linear_x.pgm"), ("slide", "down")],
    "humor": [("push", "right"), ("wipe", "bi-linear_x.pgm"), ("slide", "down")],
    "neutral": [("push", "left"), ("wipe", "bi-linear_x.pgm"), ("dip_black", "")],
}


@dataclass(frozen=True)
class Cut:
    """Transition posée entre deux cases.

    Attributes:
        kind: ``dissolve``, ``blur_dissolve``, ``wipe``, ``push``, ``slide``, ``whip`` (chevauchent
            les cases) ou ``dip_black``, ``dip_white``, ``glitch`` (effets de part et d'autre
            d'une coupe franche).
        duration_s: durée totale, centrée sur la coupe.
        option: direction (``left``, ``right``, ``up``, ``down``) ou fichier de volet.
        scene_change: coupe entre deux scènes (le whip s'accompagne alors d'un « swoosh »).
    """

    kind: str
    duration_s: float
    option: str = ""
    scene_change: bool = False


def plan_cuts(clips: list[PanelClip], emotions: list[str], *, chapter_starts: frozenset[int] = frozenset()) -> list[Cut | None]:
    """Transition de chaque coupe : ``result[i]`` va de la case ``i`` à la case ``i + 1`` (``None`` = franche).

    Args:
        clips: cases, dans l'ordre.
        emotions: émotion de la scène de chaque case (``""`` = neutre).
        chapter_starts: rangs des cases qui ouvrent un chapitre (compilation) : fondu au noir long.
    """
    turns = {}
    cuts: list[Cut | None] = []
    for i in range(len(clips) - 1):
        before, after = clips[i], clips[i + 1]
        emotion = emotions[i + 1] or "neutral"
        if i + 1 in chapter_starts:
            kind, option, target = "dip_black", "", CHAPTER_S
        else:
            scene_change = before.scene_index != after.scene_index
            pools = _SCENE if scene_change else _INTRA
            pool = pools.get(emotion) or pools["neutral"]
            key = (scene_change, emotion)
            turn = turns.setdefault(key, count())
            kind, option = pool[next(turn) % len(pool)]
            target = DIP_S if kind.startswith("dip_") else (SCENE_S if scene_change else INTRA_S)
        duration = min(target, MAX_CLIP_RATIO * min(before.duration_s, after.duration_s))
        cuts.append(Cut(kind, round(duration, 3), option, before.scene_index != after.scene_index)
                    if duration >= MIN_S else None)
    return cuts


#: Mouvements des cases, à tour de rôle : sens du zoom et point vers lequel il se fait
#: (fraction de la marge : 0 = bord gauche / haut, 0,5 = centre, 1 = bord droit / bas).
MOTION_PATTERNS: tuple[tuple[str, float, float], ...] = (
    ("in", 0.5, 0.5), ("in", 0.0, 0.5), ("out", 0.5, 0.5), ("in", 1.0, 0.5), ("in", 0.5, 0.0), ("out", 0.5, 1.0),
)


def motion_pattern(index: int) -> tuple[str, float, float]:
    return MOTION_PATTERNS[index % len(MOTION_PATTERNS)]


# --- Étalonnage ------------------------------------------------------------------------------
#: (contraste, saturation, gains R, V, B, luminosité) par émotion : de légers déplacements,
#: la couleur des planches reste reconnaissable.
GRADES: dict[str, tuple[float, float, tuple[float, float, float], float]] = {
    "action": (1.10, 1.15, (1.03, 1.00, 0.97), 1.00),
    "epic": (1.08, 1.10, (1.06, 1.02, 0.94), 1.00),
    "tension": (1.08, 0.85, (0.96, 1.00, 1.06), 0.97),
    "fear": (1.10, 0.80, (0.95, 1.00, 1.08), 0.95),
    "mystery": (1.05, 0.85, (0.94, 1.02, 1.06), 0.93),
    "sad": (0.97, 0.72, (0.95, 0.99, 1.07), 0.97),
    "romance": (1.00, 1.05, (1.06, 1.00, 0.97), 1.02),
    "happy": (1.03, 1.12, (1.05, 1.02, 0.95), 1.03),
    "humor": (1.03, 1.12, (1.04, 1.02, 0.96), 1.03),
}
#: Assombrissement des coins (0 = aucun vignettage).
VIGNETTE = 0.28


def grade(rgb: np.ndarray, emotion: str) -> np.ndarray:
    """Image étalonnée pour l'émotion, avec vignettage (``uint8`` RGB)."""
    contrast, saturation, gains, brightness = GRADES.get(emotion, (1.0, 1.0, (1.0, 1.0, 1.0), 1.0))
    x = rgb.astype(np.float32) / 255.0
    luma = x @ np.array([0.299, 0.587, 0.114], np.float32)
    x = luma[..., None] + (x - luma[..., None]) * saturation
    x = (x - 0.5) * contrast + 0.5
    x *= np.array(gains, np.float32) * brightness
    h, w = rgb.shape[:2]
    ys = np.linspace(-1.0, 1.0, h, dtype=np.float32)[:, None]
    xs = np.linspace(-1.0, 1.0, w, dtype=np.float32)[None, :]
    x *= (1.0 - VIGNETTE * np.clip((xs**2 + ys**2) / 2.0, 0.0, 1.0) ** 1.2)[..., None]
    return (np.clip(x, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)
