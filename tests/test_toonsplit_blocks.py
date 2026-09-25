"""toonsplit, étape 1 : blocs aux gouttières, score de ressemblance au fond."""

from __future__ import annotations

import cv2
import numpy as np

from src.modules.toonsplit.blocks import (
    background_colors, background_likeness, row_background_stats, scaled, segment_blocks,
)
from tests.toonsplit_helpers import art, stack, white


def spans(img: np.ndarray) -> list[tuple[int, int]]:
    return [(b.y0, b.y1) for b in segment_blocks(img)]


def test_scaled_follows_width() -> None:
    assert scaled(40, 575) == 40
    assert scaled(40, 800) == 56
    assert scaled(0.1, 100) == 1  # jamais zéro


def test_white_gutters_split_panels() -> None:
    img = stack(white(50), art(300, seed=1), white(80), art(250, seed=2), white(60))
    assert spans(img) == [(50, 350), (430, 680)]


def test_short_gap_does_not_split() -> None:
    # 10 px de blanc : sous min_gap (40 px à 575 → 28 px à 400).
    img = stack(white(40), art(200, seed=1), white(10), art(200, seed=2), white(40))
    assert spans(img) == [(40, 450)]


def test_black_gutter_splits() -> None:
    img = stack(art(200, seed=1), white(60, value=0), art(200, seed=2))
    assert spans(img) == [(0, 200), (260, 460)]


def test_flat_colored_gutter_is_background() -> None:
    color = np.array([150, 40, 40], np.uint8)
    gutter = np.broadcast_to(color, (70, 400, 3)).copy()
    img = stack(gutter, art(200, seed=1), gutter, art(200, seed=2), gutter)
    assert any(np.abs(c - color).max() <= 16 for c in background_colors(img))
    assert spans(img) == [(70, 270), (340, 540)]


def fade(height: int, down: bool) -> np.ndarray:
    """Dégradé du dessin (gris 150) vers le blanc."""
    ramp = np.linspace(150, 255, height) if down else np.linspace(255, 150, height)
    return np.broadcast_to(ramp[:, None, None], (height, 400, 3)).astype(np.uint8).copy()


def binary_gutter_rows(img: np.ndarray) -> int:
    """Plus longue suite de lignes « fond » au sens du seuil binaire du prototype."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    binary = (gray.std(1) < 5) & ((gray.mean(1) > 235) | (gray.mean(1) < 20))
    return max(len(r) for r in "".join("1" if v else "0" for v in binary).split("0"))


def test_off_white_gutter_splits_where_binary_threshold_would_not() -> None:
    img = stack(art(200, seed=1), white(80, value=226), art(200, seed=2))
    assert binary_gutter_rows(img) == 0  # moyenne 226 < 235 : le prototype ne couperait pas
    assert spans(img) == [(0, 200), (280, 480)]


def test_thin_line_crossing_gutter_does_not_block_split() -> None:
    gutter = white(80)
    gutter[:, 199:201] = 0  # une hampe fine traverse la gouttière (0,5 % des pixels)
    img = stack(art(200, seed=1), gutter, art(200, seed=2))
    assert binary_gutter_rows(img) == 0
    blocks = segment_blocks(img)
    assert len(blocks) == 2
    assert blocks[0].y1 == blocks[1].y0 == 240  # coupe au milieu de la gouttière, à travers la ligne seule


def test_fade_to_white_splits() -> None:
    img = stack(art(200, seed=1), fade(120, down=True), white(20), fade(120, down=False), art(200, seed=2))
    blocks = segment_blocks(img)
    assert len(blocks) == 2
    assert blocks[0].y1 <= 340 <= blocks[1].y0


def test_text_line_is_not_swallowed_by_gutter() -> None:
    caption = white(40)
    cv2.putText(caption, "NARRATION TEXT", (40, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2)
    img = stack(art(200, seed=1), white(100), caption, white(100), art(200, seed=2))
    blocks = segment_blocks(img)
    assert len(blocks) == 3
    text = blocks[1]
    assert text.small and 300 <= text.y0 and text.y1 <= 340


def test_block_grows_over_thin_ink_tail() -> None:
    tail = white(100)
    tail[:12, 198:201] = 0  # pointe fine d'une bulle sous le dessin : < 1 % des pixels
    img = stack(white(60), art(200, seed=1), tail, art(200, seed=2))
    score, ink = row_background_stats(img)
    assert score[262] > 0.9 and ink[262]  # ligne « fond » au sens du score, mais avec de l'encre
    first = segment_blocks(img)[0]
    assert first.y1 == 272


def test_likeness_is_bounded_and_ordered() -> None:
    img = stack(white(10), fade(50, down=False), art(10, seed=4))
    score = background_likeness(img)
    assert score.min() >= 0 and score.max() <= 1
    assert score[:10].min() > 0.95 and score[-10:].max() < 0.2


def test_tiny_noise_block_is_dropped_and_gap_strength_recorded() -> None:
    img = stack(white(80), art(200, seed=1), white(60), art(5, seed=2), white(60), art(200, seed=3), white(20))
    blocks = segment_blocks(img)
    assert [(b.y0, b.y1) for b in blocks] == [(80, 280), (405, 605)]
    assert blocks[0].gap_above > 70 and blocks[1].gap_below > 0
