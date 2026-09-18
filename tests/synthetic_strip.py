"""Générateur de bandes synthétiques pour tester le Smart Slicer sans réseau.

``make_synthetic_strip(spec)`` construit une bande verticale de largeur fixe à
partir d'une liste de triplets ``(panel_height, gutter_height, gutter_color)`` :
chaque case est remplie d'un contenu texturé (bruit aléatoire + dégradés +
formes dessinées) dont la variance par ligne est nettement supérieure au seuil
de 15.0, et chaque gouttière est une zone uniforme (blanche, noire ou de la
couleur demandée). Utilisé par ``tests/test_slicer.py`` et par le mode
``--synthetic`` de ``tests/test_slicer_local.py``.
"""

from __future__ import annotations

import cv2
import numpy as np
from PIL import Image

#: Couleur de gouttière : nom (``"white"``, ``"black"``, ``"gray"``), niveau de
#: gris (int) ou triplet RGB.
GutterColor = str | int | tuple[int, int, int]

#: Spécification d'une bande : liste de ``(panel_height, gutter_height, gutter_color)``.
#: ``gutter_height = 0`` signifie « pas de gouttière après cette case ».
StripSpec = list[tuple[int, int, GutterColor]]

#: Largeur par défaut d'une bande (largeur des chapitres Webtoons modernes).
DEFAULT_WIDTH: int = 800

#: Bande par défaut du mode ``--synthetic`` : 5 cases dont une géante (1800 px),
#: gouttières blanches et noires, dernière case sans gouttière finale.
DEFAULT_SPEC: StripSpec = [
    (600, 40, "white"),
    (900, 30, "black"),
    (1800, 60, "white"),
    (450, 25, "white"),
    (700, 0, "white"),
]

_NAMED_COLORS: dict[str, tuple[int, int, int]] = {
    "white": (255, 255, 255),
    "black": (0, 0, 0),
    "gray": (128, 128, 128),
    "grey": (128, 128, 128),
}


def resolve_color(color: GutterColor) -> tuple[int, int, int]:
    """Normalise une couleur de gouttière en triplet RGB."""
    if isinstance(color, str):
        try:
            return _NAMED_COLORS[color.lower()]
        except KeyError as exc:
            raise ValueError(f"Couleur inconnue : {color!r}") from exc
    if isinstance(color, (int, np.integer)):
        value = int(np.clip(color, 0, 255))
        return (value, value, value)
    r, g, b = color
    return (int(r), int(g), int(b))


def _make_panel_block(
    rng: np.random.Generator, height: int, width: int, index: int
) -> np.ndarray:
    """Construit un bloc de contenu texturé (H, W, 3) de variance par ligne >> 15."""
    # Dégradé bilinéaire entre deux couleurs aléatoires bornées à [40, 215]
    # (évite l'écrêtage du bruit qui réduirait la variance).
    c_a = rng.integers(40, 216, size=3).astype(np.float64)
    c_b = rng.integers(40, 216, size=3).astype(np.float64)
    ty = np.linspace(0.0, 1.0, height, dtype=np.float64)[:, None, None]
    tx = np.linspace(0.0, 1.0, width, dtype=np.float64)[None, :, None]
    weight = 0.5 * ty + 0.5 * tx
    base = c_a * (1.0 - weight) + c_b * weight

    # Bruit uniforme dans [-40, 40] : variance theorique ~546 par ligne.
    noise = rng.integers(-40, 41, size=(height, width, 3)).astype(np.float64)
    block = np.clip(base + noise, 0, 255).astype(np.uint8)

    # Formes dessinées (contraste fort) pour imiter des traits de dessin.
    n_shapes = 3 + (height // 150)
    for _ in range(n_shapes):
        color = tuple(int(v) for v in rng.integers(0, 256, size=3))
        kind = rng.integers(0, 3)
        x0, x1 = sorted(int(v) for v in rng.integers(0, width, size=2))
        y0, y1 = sorted(int(v) for v in rng.integers(0, height, size=2))
        if kind == 0:
            cv2.rectangle(block, (x0, y0), (x1, y1), color, thickness=int(rng.integers(2, 8)))
        elif kind == 1:
            radius = int(max(5, min(width, height) // int(rng.integers(3, 10))))
            cv2.circle(block, ((x0 + x1) // 2, (y0 + y1) // 2), radius, color, thickness=-1)
        else:
            cv2.line(block, (x0, y0), (x1, y1), color, thickness=int(rng.integers(2, 8)))

    if height >= 40 and width >= 160:
        cv2.putText(
            block, f"PANEL {index}", (10, min(30, height - 5)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA,
        )
    return block


def _make_gutter_block(
    rng: np.random.Generator,
    height: int,
    width: int,
    color: GutterColor,
    noise_amplitude: int,
) -> np.ndarray:
    """Construit une gouttière uniforme (H, W, 3), éventuellement légèrement bruitée."""
    rgb = resolve_color(color)
    block = np.empty((height, width, 3), dtype=np.uint8)
    block[:] = rgb
    if noise_amplitude > 0:
        # Imite le bruit de compression JPEG : petites fluctuations autour de la
        # couleur de fond (variance d'un bruit uniforme +/-a : ((2a+1)^2 - 1) / 12).
        noise = rng.integers(-noise_amplitude, noise_amplitude + 1, size=block.shape)
        block = np.clip(block.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    return block


def make_synthetic_strip(
    spec: StripSpec,
    *,
    width: int = DEFAULT_WIDTH,
    seed: int = 0,
    gutter_noise: int = 0,
) -> Image.Image:
    """Construit une bande synthétique PIL (RGB) à partir d'une spécification.

    Args:
        spec: liste de ``(panel_height, gutter_height, gutter_color)``. Chaque case
            est suivie d'une gouttière de ``gutter_height`` lignes (0 = aucune).
        width: largeur de la bande en pixels.
        seed: graine du générateur aléatoire (reproductibilité).
        gutter_noise: amplitude (0 = aucune) d'un bruit uniforme ajouté aux
            gouttières pour imiter des artefacts JPEG.

    Returns:
        Image PIL en mode ``RGB`` de largeur ``width``.
    """
    rng = np.random.default_rng(seed)
    blocks: list[np.ndarray] = []
    for index, (panel_height, gutter_height, gutter_color) in enumerate(spec):
        if panel_height > 0:
            blocks.append(_make_panel_block(rng, int(panel_height), width, index))
        if gutter_height > 0:
            blocks.append(
                _make_gutter_block(rng, int(gutter_height), width, gutter_color, gutter_noise)
            )
    if not blocks:
        raise ValueError("Specification vide : aucune case ni gouttiere")
    return Image.fromarray(np.vstack(blocks), mode="RGB")


def synthetic_layout(spec: StripSpec) -> list[tuple[int, int]]:
    """Retourne les bornes ``[y_start, y_end)`` du contenu brut de chaque case (sans padding)."""
    layout: list[tuple[int, int]] = []
    cursor = 0
    for panel_height, gutter_height, _ in spec:
        if panel_height > 0:
            layout.append((cursor, cursor + int(panel_height)))
            cursor += int(panel_height)
        cursor += max(int(gutter_height), 0)
    return layout


def total_height(spec: StripSpec) -> int:
    """Hauteur totale (lignes) de la bande décrite par ``spec``."""
    return sum(max(int(p), 0) + max(int(g), 0) for p, g, _ in spec)


__all__ = [
    "DEFAULT_SPEC",
    "DEFAULT_WIDTH",
    "GutterColor",
    "StripSpec",
    "make_synthetic_strip",
    "resolve_color",
    "synthetic_layout",
    "total_height",
]
