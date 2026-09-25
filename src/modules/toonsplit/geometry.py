"""Boîtes de détection et intervalles verticaux partagés par toutes les étapes de toonsplit.

Toutes les coordonnées sont des entiers en pixels, bornes hautes exclusives : une boîte
``(x0, y0, x1, y1)`` couvre les lignes ``y0 .. y1 - 1``. Une coupe horizontale à la ligne
``y`` (frontière entre ``y - 1`` et ``y``) **traverse** une boîte si ``y0 < y < y1``.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, replace
from typing import Any

#: Types de boîtes. ``head`` et ``bubble`` sont des contraintes dures (jamais coupées) ;
#: ``narration`` aussi (texte hors bulle reconnu comme narration par la spec IA) ;
#: ``sfx`` est « tout ou rien » en souple ; ``person`` est une zone à préférer ;
#: ``free_text`` (texte hors bulle non encore classé) devient ``sfx`` ou ``narration``.
HEAD = "head"
PERSON = "person"
BUBBLE = "bubble"
FREE_TEXT = "free_text"
SFX = "sfx"
NARRATION = "narration"
KINDS = (HEAD, PERSON, BUBBLE, FREE_TEXT, SFX, NARRATION)
HARD_KINDS = frozenset({HEAD, BUBBLE, NARRATION})


@dataclass(frozen=True)
class Box:
    """Boîte détectée ; ``source`` liste les détecteurs qui l'ont trouvée (``+``)."""

    x0: int
    y0: int
    x1: int
    y1: int
    kind: str
    score: float = 1.0
    source: str = ""

    @property
    def w(self) -> int:
        return self.x1 - self.x0

    @property
    def h(self) -> int:
        return self.y1 - self.y0

    @property
    def area(self) -> int:
        return max(0, self.w) * max(0, self.h)

    def shifted(self, dy: int) -> Box:
        return replace(self, y0=self.y0 + dy, y1=self.y1 + dy)

    def with_kind(self, kind: str) -> Box:
        return replace(self, kind=kind)

    def as_dict(self) -> dict[str, Any]:
        return {
            "box": [self.x0, self.y0, self.x1, self.y1], "kind": self.kind,
            "score": round(self.score, 3), "source": self.source,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Box:
        x0, y0, x1, y1 = (int(v) for v in data["box"])
        return cls(x0, y0, x1, y1, str(data["kind"]), float(data.get("score", 1.0)), str(data.get("source", "")))


def intersection(a: Box, b: Box) -> int:
    w = min(a.x1, b.x1) - max(a.x0, b.x0)
    h = min(a.y1, b.y1) - max(a.y0, b.y0)
    return max(0, w) * max(0, h)


def iou(a: Box, b: Box) -> float:
    inter = intersection(a, b)
    union = a.area + b.area - inter
    return inter / union if union > 0 else 0.0


def containment(a: Box, b: Box) -> float:
    """Part de la plus petite des deux boîtes couverte par l'autre."""
    small = min(a.area, b.area)
    return intersection(a, b) / small if small > 0 else 0.0


def enclosing(a: Box, b: Box) -> Box:
    """Boîte englobante (score max, sources réunies)."""
    sources = sorted(set(filter(None, a.source.split("+") + b.source.split("+"))))
    return Box(
        min(a.x0, b.x0), min(a.y0, b.y0), max(a.x1, b.x1), max(a.y1, b.y1),
        a.kind, max(a.score, b.score), "+".join(sources),
    )


def fuse_boxes(boxes: Iterable[Box], *, iou_threshold: float = 0.3, contain_threshold: float = 0.6) -> list[Box]:
    """Fusion « union + NMS » de boîtes de même type venant de détecteurs différents.

    Les boîtes qui se recouvrent (IoU ≥ ``iou_threshold`` ou l'une contenue à
    ``contain_threshold`` dans l'autre) sont remplacées par leur boîte englobante :
    pour une contrainte « ne jamais couper », l'union est le choix prudent. Les boîtes
    isolées de chaque détecteur sont gardées telles quelles. Déterministe : l'ordre
    de traitement ne dépend que des coordonnées et des scores.
    """
    ordered = sorted(boxes, key=lambda b: (b.kind, -b.score, b.y0, b.x0, b.y1, b.x1))
    fused: list[Box] = []
    for box in ordered:
        merged = box
        changed = True
        while changed:
            changed = False
            for i, other in enumerate(fused):
                if other.kind != merged.kind:
                    continue
                if iou(other, merged) >= iou_threshold or containment(other, merged) >= contain_threshold:
                    merged = enclosing(other, merged)
                    del fused[i]
                    changed = True
                    break
        fused.append(merged)
    return sorted(fused, key=lambda b: (b.y0, b.x0, b.kind))


def iou1d(a: Sequence[int], b: Sequence[int]) -> float:
    """IoU de deux intervalles verticaux ``(y0, y1)``."""
    inter = max(0, min(a[1], b[1]) - max(a[0], b[0]))
    union = max(a[1], b[1]) - min(a[0], b[0])
    return inter / union if union > 0 else 0.0


def overlap1d(a0: int, a1: int, b0: int, b1: int) -> int:
    return max(0, min(a1, b1) - max(a0, b0))


def cuts(box: Box, y: int) -> bool:
    """La coupe horizontale à la ligne ``y`` traverse-t-elle la boîte ?"""
    return box.y0 < y < box.y1


__all__ = [
    "HEAD", "PERSON", "BUBBLE", "FREE_TEXT", "SFX", "NARRATION", "KINDS", "HARD_KINDS",
    "Box", "intersection", "iou", "containment", "enclosing", "fuse_boxes",
    "iou1d", "overlap1d", "cuts",
]
