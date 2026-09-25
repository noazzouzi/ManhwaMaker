"""Étape 1 : découpe du strip en blocs aux gouttières.

Le prototype classait chaque ligne en fond / pas fond par un seuil binaire
(écart-type < 5 et moyenne > 235 ou < 20). Ici chaque ligne reçoit un **score de
ressemblance au fond** entre 0 et 1, ce qui gère les fondus vers le blanc ou le noir et
les fonds de couleur unie :

- les couleurs de fond candidates sont le blanc, le noir et les couleurs des lignes
  parfaitement unies du strip (fond coloré propre à une série) ;
- l'écart d'une ligne à une couleur est le 99e centile, sur la ligne, de l'écart
  maximal par canal : quelques pixels de texte suffisent à la disqualifier, ce qui
  garde les lignes de narration hors des gouttières ;
- score = 1 - écart / 48, borné à [0, 1] ;
- une gouttière est une suite de lignes de score ≥ 0,5 dont la **somme** des scores
  atteint ``min_gap`` : 56 lignes de blanc pur, ou une zone de fondu plus longue.

Toutes les longueurs en pixels du prototype sont exprimées à sa largeur de calibration
(575 px) et mises à l'échelle de la largeur réelle : le calcul se fait toujours en
résolution native.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

#: Largeur du strip sur lequel les constantes en pixels du prototype ont été choisies.
PROTO_WIDTH: int = 575
#: Écart (niveaux 0-255) à partir duquel une ligne ne ressemble plus du tout au fond.
BG_DEV_SCALE: float = 48.0
#: Score minimal pour qu'une ligne compte dans une gouttière.
BG_ROW_MIN: float = 0.5
#: Centile de l'écart par ligne (99 : 1 % de pixels « encre » suffit à refuser la ligne).
ROW_PERCENTILE: float = 99.0
#: Écart-type maximal d'une ligne « parfaitement unie » (couleurs de fond propres au strip).
FLAT_STD: float = 5.0
#: Écart d'un pixel au fond à partir duquel c'est de l'encre (bord de bloc, pointe de bulle).
INK_DEV: float = 64.0


def scaled(value: float, width: int) -> int:
    """Longueur ``value`` du prototype (calibré à 575 px) ramenée à la largeur ``width``."""
    return max(1, int(round(value * width / PROTO_WIDTH)))


@dataclass(frozen=True)
class Block:
    """Bloc de contenu entre deux gouttières (coordonnées du strip, ``y1`` exclu)."""

    index: int
    y0: int
    y1: int
    #: Force de la gouttière au-dessus / au-dessous (lignes de fond équivalentes ; 0 au bord du strip).
    gap_above: float = 0.0
    gap_below: float = 0.0
    #: Plus petit que ``min_block`` : gardé seulement s'il contient du texte (narration isolée).
    small: bool = False

    @property
    def h(self) -> int:
        return self.y1 - self.y0


def background_colors(img: np.ndarray, *, flat_std: float = FLAT_STD, min_share: float = 0.02) -> np.ndarray:
    """Couleurs de fond candidates (K, 3) : blanc, noir, et couleurs dominantes des lignes unies."""
    colors = [np.array([255, 255, 255]), np.array([0, 0, 0])]
    std = img.std(axis=1).max(axis=1)
    flat = std < flat_std
    n_flat = int(flat.sum())
    if n_flat:
        quantized = (np.round(img[flat].mean(axis=1) / 16.0) * 16.0).clip(0, 255).astype(int)
        values, counts = np.unique(quantized, axis=0, return_counts=True)
        order = np.lexsort((values[:, 2], values[:, 1], values[:, 0], -counts))
        for k in order:
            if counts[k] < max(8, min_share * n_flat):
                break
            if all(np.abs(values[k] - c).max() > 24 for c in colors):
                colors.append(values[k])
    return np.array(colors, dtype=np.int16)


def row_background_stats(img: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Par ligne : score de ressemblance au fond (0-1) et présence d'au moins un pixel d'encre.

    Le score ignore 1 % des pixels (centile 99) ; l'encre, elle, regarde le pixel le plus
    éloigné du fond : la pointe fine d'une bulle ou d'un trait compte comme contenu.
    """
    pixels = img.astype(np.int16)
    best = np.full(img.shape[0], 255.0)
    worst = np.full(img.shape[0], 255.0)
    for color in background_colors(img):
        dev = np.abs(pixels - color).max(axis=2)
        best = np.minimum(best, np.percentile(dev, ROW_PERCENTILE, axis=1))
        worst = np.minimum(worst, dev.max(axis=1))
    score = np.clip(1.0 - best / BG_DEV_SCALE, 0.0, 1.0).astype(np.float32)
    return score, worst > INK_DEV


def background_likeness(img: np.ndarray) -> np.ndarray:
    """Score (H,) de ressemblance au fond de chaque ligne, entre 0 et 1."""
    return row_background_stats(img)[0]


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """Suites de ``True`` : liste de ``(début, fin exclue)``."""
    padded = np.concatenate([[False], mask, [False]]).astype(np.int8)
    edges = np.flatnonzero(np.diff(padded))
    return list(zip(edges[::2].tolist(), edges[1::2].tolist()))


def segment_blocks(
    img: np.ndarray, *, min_gap: float = 40, min_block: float = 120, min_text_block: float = 16,
    stats: tuple[np.ndarray, np.ndarray] | None = None,
) -> list[Block]:
    """Découpe le strip ``img`` (BGR, H×W×3) en blocs séparés par des gouttières.

    ``min_gap``, ``min_block`` et ``min_text_block`` sont en pixels du prototype (575 px de
    large) et mis à l'échelle. Les blocs plus petits que ``min_block`` mais d'au moins
    ``min_text_block`` sont rendus avec ``small=True`` : c'est souvent une ligne de
    narration isolée, que l'appelant garde s'il y trouve du texte.

    Un bloc s'étend ensuite dans la gouttière tant que les lignes voisines contiennent de
    l'encre (pointe de bulle, trait fin), sans dépasser le milieu de la gouttière : un
    crop calé sur le bord du bloc ne rogne jamais ces quelques lignes.
    """
    height, width = img.shape[:2]
    score, ink = row_background_stats(img) if stats is None else stats
    gap_px = scaled(min_gap, width)
    gutters: list[tuple[int, int, float]] = []
    for start, end in _runs(score >= BG_ROW_MIN):
        strength = float(score[start:end].sum())
        if strength >= gap_px or start == 0 or end == height:
            gutters.append((start, end, strength))
    bounds: list[tuple[int, int, float, float, int, int]] = []
    cursor, above, limit_above = 0, 0.0, 0
    for start, end, strength in gutters:
        if start > cursor:
            bounds.append((cursor, start, above, strength, limit_above, (start + end) // 2))
        cursor, above, limit_above = end, strength, (start + end) // 2
    if cursor < height:
        bounds.append((cursor, height, above, 0.0, limit_above, height))
    blocks: list[Block] = []
    big, tiny = scaled(min_block, width), scaled(min_text_block, width)
    for y0, y1, gap_above, gap_below, lo, hi in bounds:
        while y0 > lo and ink[y0 - 1]:
            y0 -= 1
        while y1 < hi and ink[y1]:
            y1 += 1
        if y1 - y0 < tiny:
            continue
        blocks.append(Block(len(blocks), y0, y1, round(gap_above, 1), round(gap_below, 1), small=y1 - y0 < big))
    return blocks


__all__ = [
    "PROTO_WIDTH", "BG_DEV_SCALE", "BG_ROW_MIN", "ROW_PERCENTILE", "FLAT_STD",
    "INK_DEV", "scaled", "Block", "background_colors", "row_background_stats", "background_likeness",
    "segment_blocks",
]
