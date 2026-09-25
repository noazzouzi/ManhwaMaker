"""Extraction des personnages : une image par zone « personne » détectée (cadres jaunes des planches).

Contrairement aux plans (:mod:`.search`, fenêtres pleine largeur), ces images sont des
recadrages 2D sur les personnages, en résolution native, sans IA (détecteurs seuls).

Règles appliquées à chaque zone jaune :

- une tête qui déborde de la zone l'agrandit : on ne coupe jamais une tête ;
- une zone sans tête détectée est ignorée (faux positifs du détecteur de personnes : hampe
  de drapeau, drapeau), sauf ``require_head=False`` ;
- deux zones quasi identiques (IoU ≥ 0,7) n'en font qu'une ;
- ``bubbles="whole"`` agrandit la zone aux bulles, narrations et onomatopées qu'elle touche
  (tout ou rien) ; par défaut (``"cut"``) la zone jaune est prise telle quelle ;
- ``group=True`` regroupe en une seule image les personnages d'un même bloc qui partagent
  la même bande horizontale (hauteurs qui se recouvrent d'au moins ``GROUP_OVERLAP`` de la
  plus petite) : la boîte englobante, agrandie aux têtes qu'elle touche. Une scène à
  plusieurs personnages reste une seule case, comme dans le webtoon, mais deux cases
  empilées sans gouttière blanche (séparées par un simple trait) ne sont pas fusionnées ;
- ``margin`` ajoute une marge (part de la taille de la zone), bornée au bloc.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import cv2
import numpy as np

from src.modules.toonsplit.geometry import BUBBLE, FREE_TEXT, HEAD, NARRATION, PERSON, SFX, Box, enclosing, intersection, iou
from src.modules.toonsplit.pipeline import SplitResult

TEXT_KINDS = (BUBBLE, FREE_TEXT, NARRATION, SFX)
#: Recouvrement vertical minimal (part de la plus petite hauteur) pour regrouper deux personnages.
GROUP_OVERLAP = 0.25


@dataclass(frozen=True)
class Figure:
    """Recadrage d'un personnage, en coordonnées du strip (``x1``, ``y1`` exclus)."""

    x0: int
    y0: int
    x1: int
    y1: int
    source_block: int
    score: float
    heads: int
    #: Personnages regroupés dans ce recadrage (1 sans regroupement).
    persons: int = 1

    @property
    def w(self) -> int:
        return self.x1 - self.x0

    @property
    def h(self) -> int:
        return self.y1 - self.y0

    def as_dict(self) -> dict[str, Any]:
        return {"x0": self.x0, "y0": self.y0, "x1": self.x1, "y1": self.y1, "w": self.w, "h": self.h,
                "source_block": self.source_block, "score": round(self.score, 3), "heads": self.heads,
                "persons": self.persons}


def _grow(box: Box, others: Sequence[Box]) -> Box:
    """Agrandit ``box`` à toute boîte de ``others`` qu'elle touche, jusqu'à stabilité."""
    changed = True
    while changed:
        changed = False
        for other in others:
            if intersection(box, other) > 0:
                grown = enclosing(box, other)
                if (grown.x0, grown.y0, grown.x1, grown.y1) != (box.x0, box.y0, box.x1, box.y1):
                    box, changed = grown, True
    return box


def _centre_inside(inner: Box, outer: Box) -> bool:
    cx, cy = (inner.x0 + inner.x1) / 2, (inner.y0 + inner.y1) / 2
    return outer.x0 <= cx <= outer.x1 and outer.y0 <= cy <= outer.y1


def _same_row_clusters(found: list[tuple[Box, float, int, int]]) -> list[list[tuple[Box, float, int, int]]]:
    """Groupes de personnages qui partagent une bande horizontale (liaison simple, ordre de lecture)."""
    n = len(found)
    parent = list(range(n))

    def root(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in range(n):
        for j in range(i + 1, n):
            a, b = found[i][0], found[j][0]
            overlap = min(a.y1, b.y1) - max(a.y0, b.y0)
            if overlap >= GROUP_OVERLAP * max(1, min(a.h, b.h)):
                parent[root(j)] = root(i)
    clusters: dict[int, list[tuple[Box, float, int, int]]] = {}
    for i in range(n):
        clusters.setdefault(root(i), []).append(found[i])
    return sorted(clusters.values(), key=lambda c: (min(f[0].y0 for f in c), min(f[0].x0 for f in c)))


def _merge(cluster: list[tuple[Box, float, int, int]], grow_to: Sequence[Box]) -> tuple[Box, float, int, int]:
    """Un seul recadrage pour un groupe : boîte englobante agrandie aux têtes (et bulles) touchées."""
    union = cluster[0][0]
    for other, *_ in cluster[1:]:
        union = enclosing(union, other)
    if len(cluster) > 1:
        union = _grow(union, grow_to)
    return union, max(f[1] for f in cluster), sum(f[2] for f in cluster), sum(f[3] for f in cluster)


def extract_figures(
    result: SplitResult, *, margin: float = 0.0, require_head: bool = True,
    bubbles: Literal["cut", "whole"] = "cut", dedup_iou: float = 0.7, group: bool = False,
) -> list[Figure]:
    """Zones « personne » de chaque bloc, prêtes à recadrer (voir les règles du module)."""
    figures: list[Figure] = []
    for b in result.blocks:
        width, height = result.width, b.block.h
        persons = sorted((x for x in b.boxes if x.kind == PERSON), key=lambda x: (-x.score, x.y0, x.x0))
        heads = [x for x in b.boxes if x.kind == HEAD]
        texts = [x for x in b.boxes if x.kind in TEXT_KINDS]
        kept: list[Box] = []
        for person in persons:
            if not any(iou(person, k) >= dedup_iou for k in kept):
                kept.append(person)
        found: list[tuple[Box, float, int, int]] = []  # (boîte, score, têtes, personnages)
        for person in sorted(kept, key=lambda x: (x.y0, x.x0)):
            inside = [h for h in heads if _centre_inside(h, person)]
            if require_head and not inside:
                continue
            box = _grow(person, heads)
            if bubbles == "whole":
                box = _grow(box, heads + texts)
            found.append((box, person.score, len(inside), 1))
        if group and len(found) > 1:
            grown = heads + texts if bubbles == "whole" else heads
            found = [_merge(cluster, grown) for cluster in _same_row_clusters(found)]
        for box, score, n_heads, n_persons in found:
            mx, my = round(margin * box.w), round(margin * box.h)
            x0, x1 = max(0, box.x0 - mx), min(width, box.x1 + mx)
            y0, y1 = max(0, box.y0 - my), min(height, box.y1 + my)
            if x1 > x0 and y1 > y0:
                figures.append(Figure(x0, b.block.y0 + y0, x1, b.block.y0 + y1, b.block.index, score, n_heads, n_persons))
    return figures


def save_figures(img: np.ndarray, figures: Sequence[Figure], out_dir: Path) -> list[Path]:
    """Écrit chaque personnage en PNG, en résolution native (aucun redimensionnement)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for n, f in enumerate(figures):
        path = out_dir / f"figure_{n:03d}_b{f.source_block:02d}.png"
        ok, data = cv2.imencode(".png", img[f.y0:f.y1, f.x0:f.x1])
        if ok:
            path.write_bytes(data.tobytes())
            paths.append(path)
    return paths


def figures_sheet(img: np.ndarray, figures: Sequence[Figure], *, height: int = 360, max_width: int = 2400) -> np.ndarray:
    """Planche de revue : les personnages extraits côte à côte, numérotés (réduits pour l'affichage)."""
    if not figures:
        return np.full((80, 400, 3), 255, np.uint8)
    tiles = []
    for n, f in enumerate(figures):
        crop = img[f.y0:f.y1, f.x0:f.x1]
        tile = cv2.resize(crop, (max(1, round(crop.shape[1] * height / crop.shape[0])), height), interpolation=cv2.INTER_AREA)
        band = np.full((36, max(tile.shape[1], 150), 3), 255, np.uint8)
        cv2.putText(band, f"{n} (b{f.source_block}) {f.w}x{f.h}", (4, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2,
                    cv2.LINE_AA)
        if tile.shape[1] < band.shape[1]:
            tile = np.hstack([tile, np.full((height, band.shape[1] - tile.shape[1], 3), 255, np.uint8)])
        tiles.append(np.vstack([band, tile]))
    rows, row, row_w = [], [], 0
    for tile in tiles:
        if row and row_w + tile.shape[1] + 12 > max_width:
            rows.append(row)
            row, row_w = [], 0
        row.append(tile)
        row_w += tile.shape[1] + 12
    rows.append(row)
    width = max(sum(t.shape[1] + 12 for t in r) for r in rows)
    lines = []
    for r in rows:
        parts = []
        for t in r:
            parts += [t, np.full((t.shape[0], 12, 3), 255, np.uint8)]
        line = np.hstack(parts)
        lines.append(np.hstack([line, np.full((line.shape[0], width - line.shape[1], 3), 255, np.uint8)]))
        lines.append(np.full((12, width, 3), 255, np.uint8))
    return np.vstack(lines)


__all__ = ["Figure", "extract_figures", "save_figures", "figures_sheet", "TEXT_KINDS"]
