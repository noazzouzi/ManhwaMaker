"""Tests unitaires du Smart Slicer (``src.modules.slicer``) sur des bandes synthétiques.

Aucun accès réseau : les bandes sont générées par ``tests/synthetic_strip.py``.
"""

from __future__ import annotations

import json
import logging

import numpy as np
import pytest
from PIL import Image

from src.models.panel import Panel
from src.modules.slicer import (
    DEBUG_OVERLAY_MAX_HEIGHT,
    DEBUG_OVERLAY_MIN_WIDTH,
    DEBUG_OVERLAY_TILE_GAP,
    DEFAULT_MARGIN_PADDING,
    DEFAULT_MIN_GAP,
    DEFAULT_VARIANCE_THRESHOLD,
    GIANT_PANEL_HEIGHT,
    compute_row_variance,
    find_gutters,
    render_debug_overlay,
    save_panels,
    slice_panels,
)
from src.utils.image_utils import rgb_to_gray, to_numpy_rgb, to_pil
from tests.synthetic_strip import make_synthetic_strip, synthetic_layout, total_height

PAD = DEFAULT_MARGIN_PADDING


def _noise_rgb(height: int, width: int, seed: int = 0) -> np.ndarray:
    """Bloc RGB de bruit uniforme (variance par ligne ~5461)."""
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, size=(height, width, 3), dtype=np.uint8)


def _flat_rgb(height: int, width: int, value: int = 255) -> np.ndarray:
    """Bloc RGB uniforme (variance par ligne nulle)."""
    return np.full((height, width, 3), value, dtype=np.uint8)


def _png_size(path) -> tuple[int, int]:
    """Taille (largeur, hauteur) d'un PNG, en fermant le fichier."""
    with Image.open(path) as img:
        return img.size


def _png_rgb(path) -> np.ndarray:
    """Pixels RGB d'un PNG, en fermant le fichier."""
    with Image.open(path) as img:
        return np.asarray(img.convert("RGB"))


def _expected_bounds(spec, pad: int = PAD) -> list[tuple[int, int]]:
    """Bornes attendues après padding (bornées à la bande)."""
    height = total_height(spec)
    return [
        (max(0, s - pad), min(height, e + pad)) for s, e in synthetic_layout(spec)
    ]


def _bounds(panels: list[Panel]) -> list[tuple[int, int]]:
    return [(p.y_start, p.y_end) for p in panels]


# --- image_utils -----------------------------------------------------------------
def test_to_numpy_rgb_accepts_pil_gray_and_rgba() -> None:
    pil = Image.new("L", (10, 5), 128)
    arr = to_numpy_rgb(pil)
    assert arr.shape == (5, 10, 3) and arr.dtype == np.uint8
    assert arr.flags["C_CONTIGUOUS"]

    rgba = np.zeros((4, 6, 4), dtype=np.uint8)
    rgba[..., 3] = 255
    assert to_numpy_rgb(rgba).shape == (4, 6, 3)

    gray2d = np.full((3, 7), 200, dtype=np.uint8)
    out = to_numpy_rgb(gray2d)
    assert out.shape == (3, 7, 3) and int(out[0, 0, 0]) == 200

    with pytest.raises(TypeError):
        to_numpy_rgb("not an image")  # type: ignore[arg-type]


def test_to_numpy_rgb_float01_and_bool_are_rescaled(caplog) -> None:
    # Un flottant normalisé dans [0, 1] doit être remis à l'échelle 0-255 (et
    # non écrêté en {0, 1}, ce qui rendait la bande noire et le slicer aveugle).
    base = _noise_rgb(8, 12, seed=3)
    with caplog.at_level(logging.WARNING, logger="src.utils.image_utils"):
        from_float01 = to_numpy_rgb(base.astype(np.float32) / 255.0)
    assert from_float01.dtype == np.uint8
    assert np.array_equal(from_float01, base)
    assert "uint8" in caplog.text  # la conversion implicite est signalée
    # Flottant déjà en échelle 0-255 : simple arrondi/écrêtage.
    assert np.array_equal(to_numpy_rgb(base.astype(np.float64)), base)
    assert np.array_equal(to_numpy_rgb(base.astype(np.float32) + 300.0), np.full_like(base, 255))
    # Booléen : False -> 0, True -> 255.
    mask = base > 127
    assert np.array_equal(to_numpy_rgb(mask), mask.astype(np.uint8) * 255)
    # Entier hors uint8 : borné.
    assert np.array_equal(to_numpy_rgb(base.astype(np.int16) - 500), np.zeros_like(base))
    # uint8 : aucune copie, aucun avertissement.
    caplog.clear()
    with caplog.at_level(logging.WARNING, logger="src.utils.image_utils"):
        same = to_numpy_rgb(base)
    assert same is base and caplog.text == ""
    # Bout en bout : une bande float [0, 1] donne les mêmes cases que l'uint8.
    strip = np.vstack([_noise_rgb(100, 100), _flat_rgb(30, 100), _noise_rgb(100, 100, 1)])
    assert _bounds(slice_panels(strip.astype(np.float32) / 255.0)) == _bounds(slice_panels(strip))
    assert _bounds(slice_panels(strip > 127)) == _bounds(slice_panels(strip))


def test_to_numpy_rgb_accepts_empty_arrays() -> None:
    assert to_numpy_rgb(np.zeros((5, 0), dtype=np.uint8)).shape == (5, 0, 3)
    assert to_numpy_rgb(np.zeros((0, 7, 4), dtype=np.uint8)).shape == (0, 7, 3)
    assert to_numpy_rgb(np.zeros((3, 4, 1), dtype=np.uint8)).shape == (3, 4, 3)


def test_to_pil_roundtrip() -> None:
    arr = _noise_rgb(8, 12)
    pil = to_pil(arr)
    assert pil.mode == "RGB" and pil.size == (12, 8)
    assert np.array_equal(np.asarray(pil), arr)
    # Flottant normalisé : même remise à l'échelle que to_numpy_rgb.
    assert np.array_equal(np.asarray(to_pil(arr.astype(np.float32) / 255.0)), arr)


def test_rgb_to_gray_matches_cv2_shape() -> None:
    arr = _noise_rgb(6, 9)
    gray = rgb_to_gray(arr)
    assert gray.shape == (6, 9) and gray.dtype == np.uint8
    assert np.array_equal(rgb_to_gray(gray), gray)


# --- compute_row_variance ---------------------------------------------------------
def test_compute_row_variance_uniform_rows_are_zero() -> None:
    strip = np.zeros((30, 50, 3), dtype=np.uint8)
    strip[:10] = 255
    strip[10:20] = (37, 120, 200)
    row_var = compute_row_variance(strip)
    assert row_var.shape == (30,) and row_var.dtype == np.float64
    assert np.all(row_var == 0.0)


def test_compute_row_variance_noisy_rows_exceed_threshold() -> None:
    strip = _noise_rgb(64, 200)
    row_var = compute_row_variance(strip)
    assert np.all(row_var > DEFAULT_VARIANCE_THRESHOLD)
    # Référence explicite : np.var(axis=1) sur l'image grise.
    reference = rgb_to_gray(strip).astype(np.float64).var(axis=1)
    assert np.allclose(row_var, reference)


def test_compute_row_variance_accepts_gray_and_pil() -> None:
    gray = np.tile(np.arange(100, dtype=np.uint8), (5, 1))
    row_var = compute_row_variance(gray)
    assert row_var.shape == (5,)
    assert np.allclose(row_var, np.var(np.arange(100, dtype=np.float64)))
    pil = to_pil(_noise_rgb(12, 20))
    assert compute_row_variance(pil).shape == (12,)


def test_compute_row_variance_block_boundary_is_seamless() -> None:
    # Hauteur supérieure à la taille de bloc interne : le résultat doit rester
    # identique à un calcul global.
    strip = _noise_rgb(4096 + 37, 16)
    row_var = compute_row_variance(strip)
    reference = rgb_to_gray(strip).astype(np.float64).var(axis=1)
    assert row_var.shape == (4096 + 37,)
    assert np.allclose(row_var, reference)


def test_compute_row_variance_degenerate_shapes() -> None:
    # Hauteur nulle : tableau vide (pas de mémoire non initialisée).
    empty = compute_row_variance(np.zeros((0, 10, 3), dtype=np.uint8))
    assert empty.shape == (0,) and empty.dtype == np.float64
    assert compute_row_variance(np.zeros((0, 10), dtype=np.uint8)).shape == (0,)
    # Largeur nulle : variance indéfinie -> ValueError (et non une erreur cv2 brute).
    with pytest.raises(ValueError):
        compute_row_variance(np.zeros((5, 0), dtype=np.uint8))
    with pytest.raises(ValueError):
        compute_row_variance(np.zeros((5, 0, 3), dtype=np.uint8))
    with pytest.raises(ValueError):
        compute_row_variance(np.zeros(5, dtype=np.uint8))
    with pytest.raises(TypeError):
        compute_row_variance("strip")  # type: ignore[arg-type]


# --- find_gutters -----------------------------------------------------------------
def test_find_gutters_basic_runs_and_min_gap() -> None:
    row_var = np.full(200, 100.0)
    row_var[50:80] = 0.0  # 30 lignes -> gouttière
    row_var[120:130] = 0.0  # 10 lignes -> trop court avec min_gap=20
    assert find_gutters(row_var) == [(50, 80)]
    assert find_gutters(row_var, min_gap=10) == [(50, 80), (120, 130)]
    assert find_gutters(row_var, min_gap=11) == [(50, 80)]
    assert find_gutters(row_var, min_gap=31) == []


def test_find_gutters_touching_edges_and_full() -> None:
    row_var = np.full(200, 100.0)
    row_var[:25] = 1.0
    row_var[-25:] = 1.0
    assert find_gutters(row_var) == [(0, 25), (175, 200)]
    assert find_gutters(np.zeros(50)) == [(0, 50)]
    assert find_gutters(np.full(50, 100.0)) == []
    assert find_gutters(np.zeros(0)) == []


def test_find_gutters_threshold_is_strict() -> None:
    row_var = np.full(100, DEFAULT_VARIANCE_THRESHOLD)  # == seuil : pas une gouttière
    assert find_gutters(row_var) == []
    row_var[:] = DEFAULT_VARIANCE_THRESHOLD - 0.001
    assert find_gutters(row_var) == [(0, 100)]
    assert find_gutters(np.full(100, 14.0), variance_threshold=14.0) == []
    assert find_gutters(np.full(100, 14.0), variance_threshold=14.5) == [(0, 100)]


def test_find_gutters_rejects_2d_input_and_bad_min_gap() -> None:
    # L'image grise passée par erreur à la place de la variance par ligne.
    with pytest.raises(ValueError):
        find_gutters(np.zeros((10, 10), dtype=np.uint8))
    with pytest.raises(ValueError):
        find_gutters(np.zeros(10), min_gap=0)
    with pytest.raises(ValueError):
        find_gutters(np.zeros(10), min_gap=-3)
    assert find_gutters(np.zeros(10), min_gap=1) == [(0, 10)]
    assert find_gutters([0.0, 0.0, 100.0, 0.0], min_gap=1) == [(0, 2), (3, 4)]


# --- slice_panels : gouttières blanches / noires -----------------------------------
@pytest.mark.parametrize("gutter_color", ["white", "black", 128, (250, 240, 230)])
def test_slice_panels_uniform_gutters(gutter_color) -> None:
    spec = [(300, 40, gutter_color), (500, 25, gutter_color), (400, 0, gutter_color)]
    strip = make_synthetic_strip(spec, seed=3)
    panels = slice_panels(strip)
    assert [p.index for p in panels] == [0, 1, 2]
    assert _bounds(panels) == _expected_bounds(spec)
    assert all(p.type == "static" for p in panels)
    assert all(p.width == 800 for p in panels)
    assert all(p.height == p.y_end - p.y_start for p in panels)


def test_slice_panels_mixed_gutters_and_leading_trailing_gutters() -> None:
    # Gouttière en tête (case de hauteur 0 suivie d'une gouttière) et en queue.
    spec = [(0, 60, "white"), (350, 30, "black"), (420, 45, "white"), (380, 70, "black")]
    strip = make_synthetic_strip(spec, seed=7)
    panels = slice_panels(strip)
    assert len(panels) == 3
    assert _bounds(panels) == _expected_bounds(spec)
    # La première case commence après la gouttière de tête (moins le padding).
    assert panels[0].y_start == 60 - PAD
    assert panels[-1].y_end == total_height(spec) - 70 + PAD


def test_slice_panels_giant_panel_tagged_scroll_vertical() -> None:
    spec = [(600, 40, "white"), (1800, 40, "white"), (500, 0, "white")]
    strip = make_synthetic_strip(spec, seed=1)
    # Sans decoupe des cases hautes : la case de 1830 px est geante (defilement).
    panels = slice_panels(strip, frame_height=0)
    assert [p.type for p in panels] == ["static", "scroll_vertical", "static"]
    assert panels[1].height == 1800 + 2 * PAD
    assert panels[1].height > GIANT_PANEL_HEIGHT
    # Seuil surchargé : tout devient géant.
    panels_low = slice_panels(strip, giant_panel_height=400, frame_height=0)
    assert all(p.type == "scroll_vertical" for p in panels_low)
    # Par defaut (1830 px > 1200) : la case est coupee en deux blocs fixes de pleine largeur.
    split = slice_panels(strip)
    assert [p.type for p in split] == ["static"] * 4
    assert [p.part for p in split] == [None, "top", "bottom", None]
    assert all(p.width == 800 for p in split)


def test_split_segment_to_frame_targets_the_frame_height() -> None:
    from src.modules.slicer import DEFAULT_FRAME_HEIGHT, frame_coverage, split_segment_to_frame

    def heights(y0, y1, **kw):
        return [b - a for a, b, _ in split_segment_to_frame(y0, y1, width=800, **kw)]

    # Sous le cadre l'echelle vaut deja 1 : couper ne montrerait pas un pixel de plus.
    assert heights(0, 600) == [600]
    assert heights(0, DEFAULT_FRAME_HEIGHT) == [DEFAULT_FRAME_HEIGHT]
    # 1430 px : reduite a 0,76x, mais deux moities de 715 px couvriraient MOINS l'ecran.
    assert heights(0, 1430) == [1430]
    # Au-dela, couper gagne, et les blocs sont equilibres autour de la hauteur du cadre.
    for total in (1830, 2500, 4000, 7391):
        pieces = heights(0, total)
        assert sum(pieces) == total
        # Aucun bloc riquiqui : le plus court fait au moins la moitie du plus haut.
        assert min(pieces) >= 0.5 * max(pieces), (total, pieces)
        assert all(600 <= piece <= 1.6 * DEFAULT_FRAME_HEIGHT for piece in pieces), (total, pieces)
        # La decoupe retenue occupe plus d'ecran, en moyenne, que la case entiere.
        avg = sum(frame_coverage(piece, 800, 1920, 1080) for piece in pieces) / len(pieces)
        assert avg > frame_coverage(total, 800, 1920, 1080)
    # Pavage exact : les blocs sont jointifs, du debut a la fin.
    pieces = split_segment_to_frame(100, 4100, width=800)
    assert pieces[0][0] == 100 and pieces[-1][1] == 4100
    assert all(pieces[i][1] == pieces[i + 1][0] for i in range(len(pieces) - 1))
    assert [pieces[0][2], pieces[-1][2]] == ["top", "bottom"]
    assert set(p[2] for p in pieces[1:-1]) <= {"middle"}
    # frame_height=0 : sous-decoupe desactivee (profils qui recadrent, comme le format court).
    assert heights(0, 7391, frame_height=0) == [7391]


def test_split_segment_to_frame_snaps_to_a_real_border() -> None:
    from src.modules.slicer import split_segment_to_frame

    free = split_segment_to_frame(0, 1830, width=800)[0][1]
    # Une frontiere reelle a quelques dizaines de pixels attire la coupe...
    near = split_segment_to_frame(0, 1830, width=800, borders=[(free + 18, 1.0)])[0][1]
    assert near == free + 18
    # ...mais une frontiere lointaine ne la deplace pas jusqu'a desequilibrer les blocs.
    far = split_segment_to_frame(0, 1830, width=800, borders=[(300, 1.0)])[0][1]
    assert abs(far - free) <= 25


def test_compute_row_change_sees_a_full_width_border() -> None:
    from src.modules.slicer import compute_row_change, find_borders

    rng = np.random.default_rng(7)
    top = rng.integers(0, 60, size=(300, 400, 3), dtype=np.uint8)      # zone sombre
    bottom = rng.integers(200, 256, size=(300, 400, 3), dtype=np.uint8)  # zone claire
    change = compute_row_change(np.vstack([top, bottom]))
    assert change.shape == (600,) and change[0] == 0.0
    assert change[300] > 0.9          # la bordure fait changer toute la largeur
    assert change[150] < 0.5 and change[450] < 0.5
    assert [row for row, _ in find_borders(change, 0, 600)] == [300]
    # Une bulle ne change qu'une partie de la largeur : pas une frontiere.
    partial = np.vstack([top.copy(), top.copy()])
    partial[300:, :80] = 255
    assert find_borders(compute_row_change(partial), 0, 600) == []


def test_compute_row_change_is_seamless_across_blocks() -> None:
    """Le calcul est par blocs : une bordure tombant sur une jointure doit rester visible."""
    from src.modules.slicer import _VARIANCE_BLOCK_ROWS, compute_row_change

    seam = _VARIANCE_BLOCK_ROWS + 1
    strip = np.zeros((seam + 200, 300, 3), dtype=np.uint8)
    strip[seam:] = 255
    change = compute_row_change(strip)
    assert change[seam] == pytest.approx(1.0)


def test_slice_panels_splits_tall_panels_full_width(tmp_path) -> None:
    from src.modules.slicer import load_panels

    spec = [(400, 40, "white"), (1800, 40, "white"), (500, 0, "white")]  # 1830 px > cadre
    strip = make_synthetic_strip(spec, seed=11)
    panels = slice_panels(strip)
    rgb = to_numpy_rgb(strip)
    assert [p.index for p in panels] == [0, 1, 2, 3]
    assert [p.part for p in panels] == [None, "top", "bottom", None]
    assert [p.source_index for p in panels] == [None, 1, 1, None]
    assert all(p.width == 800 for p in panels)  # jamais rognee en largeur
    top, bottom = panels[1], panels[2]
    assert top.y_start == 440 - PAD and bottom.y_end == 440 + 1800 + PAD
    assert top.y_end == bottom.y_start
    assert abs(top.height - bottom.height) <= 50  # blocs equilibres autour du cadre
    assert top.height + bottom.height == 1800 + 2 * PAD
    assert np.array_equal(top.image, rgb[top.y_start : top.y_end])
    assert np.array_equal(bottom.image, rgb[bottom.y_start : bottom.y_end])
    assert len(slice_panels(strip, frame_height=0)) == 3
    # Une bande calme trop courte pour etre une gouttiere reste un seul segment, coupe en deux.
    quiet = np.vstack([_noise_rgb(800, 400, seed=3), _flat_rgb(6, 400), _noise_rgb(694, 400, seed=4)])
    halves = slice_panels(quiet)
    assert [p.part for p in halves] == ["top", "bottom"]
    assert halves[0].y_end == halves[1].y_start and halves[1].y_end == 1500
    # Tres haute : plus de trois blocs sont desormais permis, largeur intacte.
    many = slice_panels(_noise_rgb(4000, 800))
    assert len(many) > 3
    assert [many[0].part, many[-1].part] == ["top", "bottom"]
    assert set(p.part for p in many[1:-1]) == {"middle"}
    assert all(p.width == 800 for p in many) and sum(p.height for p in many) == 4000
    assert [p.source_index for p in many] == [0] * len(many)
    # Aller-retour disque : part / source_index conserves dans panels.json.
    out_dir = tmp_path / "split"
    save_panels(panels, out_dir)
    meta = json.loads((out_dir / "panels.json").read_text(encoding="utf-8"))
    assert [m["part"] for m in meta] == [None, "top", "bottom", None]
    assert [m["source_index"] for m in meta] == [None, 1, 1, None]
    assert load_panels(out_dir) == panels


def test_slice_panels_giant_boundary_is_strict() -> None:
    static = slice_panels(_noise_rgb(GIANT_PANEL_HEIGHT, 64), frame_height=0)
    assert len(static) == 1 and static[0].type == "static"
    assert static[0].height == GIANT_PANEL_HEIGHT
    scroll = slice_panels(_noise_rgb(GIANT_PANEL_HEIGHT + 1, 64), frame_height=0)
    assert len(scroll) == 1 and scroll[0].type == "scroll_vertical"


def test_slice_panels_no_gutter_single_panel() -> None:
    strip = _noise_rgb(1000, 800)
    panels = slice_panels(strip)
    assert len(panels) == 1
    panel = panels[0]
    assert (panel.index, panel.y_start, panel.y_end) == (0, 0, 1000)
    assert panel.height == 1000 and panel.width == 800 and panel.type == "static"
    assert np.array_equal(panel.image, strip)


def test_slice_panels_fully_uniform_strip_yields_no_panel(caplog) -> None:
    blank = np.full((500, 300, 3), 255, dtype=np.uint8)
    with caplog.at_level(logging.WARNING, logger="src.modules.slicer"):
        assert slice_panels(blank) == []
    assert "No panel found" in caplog.text
    assert "entirely uniform" in caplog.text


def test_slice_panels_all_dropped_warning_names_min_panel_height(caplog) -> None:
    # Un contenu existe (3 lignes texturées) mais est écarté par min_panel_height :
    # l'avertissement ne doit pas prétendre que la bande est uniforme.
    strip = np.vstack([_flat_rgb(50, 100), _noise_rgb(3, 100), _flat_rgb(50, 100)])
    with caplog.at_level(logging.WARNING, logger="src.modules.slicer"):
        assert slice_panels(strip) == []
    assert "entirely uniform" not in caplog.text
    assert "No panel kept" in caplog.text and "min_panel_height=180" in caplog.text


def test_slice_panels_warns_when_padding_exceeds_min_gap(caplog) -> None:
    strip = np.vstack([_noise_rgb(300, 100), _flat_rgb(10, 100), _noise_rgb(300, 100, 1)])
    with caplog.at_level(logging.WARNING, logger="src.modules.slicer"):
        panels = slice_panels(strip, min_gap=10, frame_height=0)  # padding 15 > min_gap 10
    assert len(panels) == 2
    assert "margin_padding (15) > min_gap (10)" in caplog.text
    # Valeurs par défaut (min_gap >= padding) : aucun avertissement.
    caplog.clear()
    with caplog.at_level(logging.WARNING, logger="src.modules.slicer"):
        slice_panels(np.vstack([_noise_rgb(300, 100), _flat_rgb(30, 100), _noise_rgb(300, 100, 1)]))
    assert caplog.text == ""


def test_merge_small_segments_rules() -> None:
    from src.modules.slicer import merge_small_segments

    # Suite de petites cases proches : fusionnee en un bloc (gouttieres incluses).
    segments = [(0, 100), (130, 230), (260, 400), (450, 900), (930, 1000), (1300, 1400)]
    merged = merge_small_segments(segments, max_height=250, max_gap=150)
    assert merged == [(0, 400), (450, 900), (930, 1000), (1300, 1400)]
    # Une case normale n'absorbe jamais une petite voisine ; un ecart trop grand coupe la suite.
    assert merge_small_segments([(0, 300), (320, 400)]) == [(0, 300), (320, 400)]
    assert merge_small_segments([(0, 100), (300, 400)], max_gap=150) == [(0, 100), (300, 400)]
    assert merge_small_segments([]) == []
    assert merge_small_segments([(0, 100)]) == [(0, 100)]


def test_slice_panels_merges_small_neighbours_and_drops_micro_panels(caplog) -> None:
    # Bulles de chat de 120 px separees par 40 px : une seule case ; onomatopee de 60 px isolee : supprimee.
    spec = [(500, 40, "white"), (120, 40, "white"), (120, 40, "white"), (120, 200, "white"), (60, 200, "white"), (600, 0, "white")]
    strip = make_synthetic_strip(spec, seed=43)
    with caplog.at_level(logging.DEBUG, logger="src.modules.slicer"):
        panels = slice_panels(strip)
    layout = synthetic_layout(spec)
    assert len(panels) == 3
    assert _bounds(panels)[0] == (0, layout[0][1] + PAD)
    # Bloc fusionne : de la premiere bulle a la derniere (gouttieres incluses), puis padding.
    assert _bounds(panels)[1] == (layout[1][0] - PAD, layout[3][1] + PAD)
    assert panels[1].height == (layout[3][1] - layout[1][0]) + 2 * PAD
    assert _bounds(panels)[2] == (layout[5][0] - PAD, total_height(spec))
    assert "2 petite(s) case(s) fusionnee(s)" in caplog.text
    assert "Dropped 1 panel(s) shorter than 180 px" in caplog.text
    # Sans fusion : les bulles restent separees mais sont eliminees par MIN_PANEL_HEIGHT (150 < 180).
    assert len(slice_panels(strip, merge_small_below=0)) == 2
    assert len(slice_panels(strip, merge_small_below=0, min_panel_height=100)) == 5
    with pytest.raises(ValueError):
        slice_panels(strip, merge_small_below=-1)


def test_slice_panels_rejects_invalid_parameters() -> None:
    strip = _noise_rgb(200, 50)
    for kwargs in (
        {"min_gap": 0},
        {"min_gap": -1},
        {"margin_padding": -1},
        {"variance_threshold": -0.1},
        {"giant_panel_height": -1},
        {"min_panel_height": -1},
        {"frame_height": -1},
        {"frame_width": 0},
    ):
        with pytest.raises(ValueError):
            slice_panels(strip, **kwargs)
    # Bornes acceptées.
    assert len(slice_panels(strip, min_gap=1, giant_panel_height=0, min_panel_height=0, frame_height=0)) == 1


def test_slice_panels_margin_padding_clamped_to_image() -> None:
    spec = [(300, 40, "white"), (300, 40, "white"), (300, 0, "white")]
    strip = make_synthetic_strip(spec, seed=5)
    panels = slice_panels(strip)
    assert _bounds(panels) == [(0, 315), (325, 655), (665, 980)]
    # Padding nul : bornes exactes du contenu.
    assert _bounds(slice_panels(strip, margin_padding=0)) == [(0, 300), (340, 640), (680, 980)]
    # Padding énorme : borné à [0, H] et jamais négatif.
    huge = slice_panels(strip, margin_padding=10_000)
    assert all(p.y_start == 0 and p.y_end == 980 for p in huge)
    with pytest.raises(ValueError):
        slice_panels(strip, margin_padding=-1)


def test_slice_panels_min_gap_behaviour() -> None:
    spec = [(300, 10, "white"), (300, 0, "white")]  # gouttière de 10 px seulement
    strip = make_synthetic_strip(spec, seed=9)
    assert len(slice_panels(strip)) == 1  # min_gap=20 : non séparée
    assert len(slice_panels(strip, min_gap=10)) == 2
    assert len(slice_panels(strip, min_gap=11)) == 1
    assert DEFAULT_MIN_GAP == 20


def test_slice_panels_jpeg_like_noise_below_threshold() -> None:
    # Bruit +/-3 : variance ~4 < 15 -> les gouttières restent détectées.
    spec = [(300, 40, (245, 245, 245)), (300, 40, (10, 10, 10)), (300, 0, "white")]
    strip = make_synthetic_strip(spec, seed=11, gutter_noise=3)
    row_var = compute_row_variance(strip)
    gutter_rows = row_var[300:340]
    assert np.all(gutter_rows > 0.0) and np.all(gutter_rows < DEFAULT_VARIANCE_THRESHOLD)
    panels = slice_panels(strip)
    assert _bounds(panels) == _expected_bounds(spec)
    # Un seuil trop strict ne voit plus les gouttières bruitées : une seule case.
    assert len(slice_panels(strip, variance_threshold=0.5)) == 1


def test_slice_panels_drops_panels_shorter_than_min_height(caplog) -> None:
    width = 200
    gutter = np.full((50, width, 3), 255, dtype=np.uint8)
    tiny = _noise_rgb(5, width, seed=1)  # 5 + 2*15 = 35 < 40 -> ignoré
    big = _noise_rgb(300, width, seed=2)
    strip = np.vstack([gutter, tiny, gutter, big, gutter])
    with caplog.at_level(logging.DEBUG, logger="src.modules.slicer"):
        panels = slice_panels(strip, frame_height=0)
    assert len(panels) == 1
    assert (panels[0].index, panels[0].y_start, panels[0].y_end) == (0, 90, 420)
    assert "Dropped 1 panel(s) shorter than 180 px" in caplog.text
    # Seuil abaissé : la petite case est conservée et les index restent contigus.
    kept = slice_panels(strip, min_panel_height=30, frame_height=0)
    assert [p.index for p in kept] == [0, 1]
    assert _bounds(kept) == [(35, 70), (90, 420)]


def test_slice_panels_image_is_copied_crop() -> None:
    spec = [(200, 30, "white"), (250, 0, "white")]
    strip = make_synthetic_strip(spec, seed=13)
    rgb = to_numpy_rgb(strip)
    panels = slice_panels(rgb)
    for panel in panels:
        assert panel.image.shape == (panel.height, panel.width, 3)
        assert panel.image.dtype == np.uint8
        assert np.array_equal(panel.image, rgb[panel.y_start : panel.y_end])
        assert not np.shares_memory(panel.image, rgb)
        assert panel.image.flags["C_CONTIGUOUS"]


def test_slice_panels_pil_and_ndarray_give_same_result() -> None:
    spec = [(300, 40, "white"), (300, 40, "black"), (300, 0, "white")]
    strip = make_synthetic_strip(spec, seed=17)
    from_pil = slice_panels(strip)
    from_arr = slice_panels(np.asarray(strip))
    assert _bounds(from_pil) == _bounds(from_arr)
    assert [p.type for p in from_pil] == [p.type for p in from_arr]


def test_slice_panels_rejects_empty_strip() -> None:
    with pytest.raises(ValueError):
        slice_panels(np.zeros((0, 800, 3), dtype=np.uint8))
    with pytest.raises(ValueError):
        slice_panels(np.zeros((800, 0, 3), dtype=np.uint8))
    with pytest.raises(ValueError):
        slice_panels(np.zeros((800, 0), dtype=np.uint8))


# --- Modèle Panel -------------------------------------------------------------------
def test_panel_model_dump_excludes_image() -> None:
    image = _noise_rgb(10, 20)
    panel = Panel(index=0, y_start=5, y_end=15, height=10, width=20, type="static", image=image)
    dumped = panel.model_dump()
    assert "image" not in dumped
    assert dumped == {
        "index": 0, "y_start": 5, "y_end": 15, "height": 10, "width": 20, "type": "static",
        "part": None, "source_index": None,
    }
    assert panel.to_dict() == dumped
    assert "image" not in json.loads(panel.model_dump_json())
    assert "image" not in repr(panel)


def test_panel_model_validates_consistency() -> None:
    image = _noise_rgb(10, 20)
    with pytest.raises(ValueError):
        Panel(index=0, y_start=5, y_end=15, height=11, width=20, type="static", image=image)
    with pytest.raises(ValueError):
        Panel(index=0, y_start=5, y_end=15, height=10, width=21, type="static", image=image)
    with pytest.raises(ValueError):
        Panel(index=0, y_start=5, y_end=15, height=10, width=20, type="other", image=image)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        Panel(index=0, y_start=5, y_end=15, height=10, width=20, type="static",
              image=image.astype(np.float32))
    with pytest.raises(ValueError):  # pas un ndarray : ValidationError de Pydantic
        Panel(index=0, y_start=5, y_end=15, height=10, width=20, type="static",
              image="pixels")  # type: ignore[arg-type]
    with pytest.raises(ValueError):  # forme (H, W) au lieu de (H, W, 3)
        Panel(index=0, y_start=5, y_end=15, height=10, width=20, type="static",
              image=image[:, :, 0])


def test_panel_equality_with_copied_image() -> None:
    strip = np.vstack([_noise_rgb(300, 100), _flat_rgb(30, 100), _noise_rgb(300, 100, 1)])
    panels = slice_panels(strip)
    first = panels[0]
    # Copies (pixels identiques, tableaux distincts) : égalité sans ValueError.
    rebuilt = Panel(**{**first.to_dict(), "image": first.image.copy()})
    assert rebuilt == first and first == rebuilt
    assert not (rebuilt != first)
    assert rebuilt in panels and panels.index(rebuilt) == 0
    assert slice_panels(strip) == panels
    # Différences : métadonnées ou pixels.
    assert first != panels[1]
    other_pixels = first.image.copy()
    other_pixels[0, 0, 0] ^= 0xFF
    assert Panel(**{**first.to_dict(), "image": other_pixels}) != first
    assert first != "not a panel"
    # Modèle mutable contenant un tableau : non hachable.
    with pytest.raises(TypeError):
        hash(first)


# --- Sorties disque -------------------------------------------------------------------
def test_save_panels_writes_png_and_json(tmp_path) -> None:
    # Derniere case de 200 px : en bas de bande le padding est tronque (200 + 15 >= 180 conservee).
    spec = [(200, 30, "white"), (1600, 30, "black"), (200, 0, "white")]
    strip = make_synthetic_strip(spec, seed=21)
    panels = slice_panels(strip, frame_height=0)
    out_dir = tmp_path / "panels"
    paths = save_panels(panels, out_dir)

    assert [p.name for p in paths] == ["panel_000.png", "panel_001.png", "panel_002.png"]
    assert all(p.exists() for p in paths)
    for panel, path in zip(panels, paths):
        assert np.array_equal(_png_rgb(path), panel.image)

    meta = json.loads((out_dir / "panels.json").read_text(encoding="utf-8"))
    assert len(meta) == 3
    for panel, entry in zip(panels, meta):
        assert set(entry) == {"index", "y_start", "y_end", "height", "width", "type", "file", "part", "source_index"}
        assert entry["part"] is None and entry["source_index"] is None
        assert entry["file"] == f"panel_{panel.index:03d}.png"
        assert entry["type"] == panel.type
        assert (entry["y_start"], entry["y_end"], entry["height"]) == (
            panel.y_start, panel.y_end, panel.height,
        )
    assert meta[1]["type"] == "scroll_vertical"

    # Préfixe personnalisé et liste vide.
    custom = save_panels(panels[:1], tmp_path / "custom", prefix="case")
    assert custom[0].name == "case_000.png"
    assert save_panels([], tmp_path / "empty") == []
    assert json.loads((tmp_path / "empty" / "panels.json").read_text()) == []


def test_save_panels_removes_stale_files_from_previous_run(tmp_path) -> None:
    spec = [(300, 30, "white")] * 4 + [(300, 0, "white")]
    strip = make_synthetic_strip(spec, seed=31)
    panels = slice_panels(strip)
    assert len(panels) == 5
    out_dir = tmp_path / "out"
    save_panels(panels, out_dir)
    # Fichiers étrangers : autre préfixe, autre motif -> jamais touchés.
    (out_dir / "case_000.png").write_bytes(b"x")
    (out_dir / "panel_notes.txt").write_bytes(b"x")
    (out_dir / "panel_0.png").write_bytes(b"x")

    # Deuxième exécution avec moins de cases : les PNG en trop disparaissent.
    save_panels(panels[:2], out_dir)
    names = sorted(p.name for p in out_dir.iterdir())
    assert names == [
        "case_000.png", "panel_0.png", "panel_000.png", "panel_001.png",
        "panel_notes.txt", "panels.json",
    ]
    meta = json.loads((out_dir / "panels.json").read_text(encoding="utf-8"))
    assert [m["file"] for m in meta] == ["panel_000.png", "panel_001.png"]

    # clean=False : comportement historique, les fichiers obsolètes restent.
    save_panels(panels, out_dir)
    save_panels(panels[:1], out_dir, clean=False)
    assert (out_dir / "panel_004.png").exists()
    # Le nettoyage ne concerne que le préfixe demandé.
    save_panels(panels[:1], out_dir, prefix="case")
    assert (out_dir / "panel_004.png").exists() and not (out_dir / "case_001.png").exists()


def test_outputs_support_non_ascii_directories(tmp_path) -> None:
    spec = [(300, 30, "white"), (300, 0, "white")]
    strip = make_synthetic_strip(spec, seed=37)
    panels = slice_panels(strip)
    out_dir = tmp_path / "café 日本"
    paths = save_panels(panels, out_dir)
    assert [np.array_equal(_png_rgb(p), panel.image) for p, panel in zip(paths, panels)] == [True, True]
    overlay = render_debug_overlay(strip, panels, tmp_path / "日本" / "o.png")
    assert _png_size(overlay) == strip.size


def test_render_debug_overlay_same_size_when_short(tmp_path) -> None:
    spec = [(300, 40, "white"), (300, 0, "white")]
    strip = make_synthetic_strip(spec, seed=23)
    panels = slice_panels(strip)
    path = render_debug_overlay(strip, panels, tmp_path / "debug" / "overlay.png")
    assert path.exists()
    assert _png_size(path) == strip.size
    # Un ndarray passé par l'appelant ne doit jamais être modifié (dessin sur copie).
    arr = to_numpy_rgb(strip)
    before = arr.copy()
    render_debug_overlay(arr, panels, tmp_path / "debug" / "overlay_arr.png")
    assert np.array_equal(arr, before)
    view = arr[:, :, :]  # vue partageant la mémoire : idem
    render_debug_overlay(view, panels, tmp_path / "debug" / "overlay_view.png")
    assert np.array_equal(arr, before)
    # L'overlay a bien été dessiné (différent de la bande).
    assert not np.array_equal(_png_rgb(path), before)


def test_render_debug_overlay_tiles_narrow_tall_strips(tmp_path) -> None:
    # Bande plus étroite que DEBUG_OVERLAY_MIN_WIDTH : jamais réduite (les
    # libellés seraient illisibles), mais découpée en colonnes de max_height.
    strip = _noise_rgb(17_000, 100, seed=29)
    panels = slice_panels(strip, frame_height=0)
    assert len(panels) == 1 and panels[0].type == "scroll_vertical"
    path = render_debug_overlay(strip, panels, tmp_path / "tall.png")
    n_tiles = 3  # ceil(17000 / 8000)
    assert _png_size(path) == (
        n_tiles * 100 + (n_tiles - 1) * DEBUG_OVERLAY_TILE_GAP, DEBUG_OVERLAY_MAX_HEIGHT,
    )


def test_render_debug_overlay_keeps_min_width_for_full_chapters(tmp_path) -> None:
    # Chapitre réaliste : 70 000 x 800. Une réduction uniforme à 8000 lignes
    # donnerait 91 px de large ; l'overlay doit garder >= DEBUG_OVERLAY_MIN_WIDTH.
    spec = [(9_000, 40, "white")] * 7 + [(6_720, 0, "white")]  # 7 x 9040 + 6720
    strip = make_synthetic_strip(spec, seed=41)
    assert strip.size == (800, 70_000)
    panels = slice_panels(strip, frame_height=0)
    assert len(panels) == 8
    path = render_debug_overlay(strip, panels, tmp_path / "chapter.png")
    width, height = _png_size(path)
    assert height == DEBUG_OVERLAY_MAX_HEIGHT
    scale = DEBUG_OVERLAY_MIN_WIDTH / 800
    tile_w = round(800 * scale)
    n_tiles = -(-round(70_000 * scale) // DEBUG_OVERLAY_MAX_HEIGHT)  # ceil
    assert tile_w == DEBUG_OVERLAY_MIN_WIDTH and n_tiles == 4
    assert width == n_tiles * tile_w + (n_tiles - 1) * DEBUG_OVERLAY_TILE_GAP
    # Réduction modérée demandée explicitement : une seule colonne réduite.
    single = render_debug_overlay(strip, panels, tmp_path / "single.png", min_width=1)
    assert _png_size(single) == (round(800 * DEBUG_OVERLAY_MAX_HEIGHT / 70_000), DEBUG_OVERLAY_MAX_HEIGHT)
    with pytest.raises(ValueError):
        render_debug_overlay(strip, panels, tmp_path / "bad.png", max_height=0)
