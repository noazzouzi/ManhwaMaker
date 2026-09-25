"""toonsplit : extraction des personnages (zones « personne ») en images, sans IA ni modèle."""

from __future__ import annotations

import json

import cv2
import numpy as np
import pytest

from src.modules.toonsplit import __main__ as cli
from src.modules.toonsplit import detectors
from src.modules.toonsplit.figures import extract_figures, figures_sheet, save_figures
from src.modules.toonsplit.geometry import BUBBLE, HEAD, PERSON, Box
from src.modules.toonsplit.pipeline import analyze_strip
from tests.toonsplit_helpers import art, stack, white

W = 400


def strip() -> np.ndarray:
    return stack(white(60), art(700, seed=1), white(100), art(600, seed=2), white(60))


def detect(block: np.ndarray) -> list[Box]:
    if block.shape[0] == 700:
        return [
            Box(100, 120, 250, 650, PERSON, 0.9, "t"),     # personnage avec tête qui dépasse en haut
            Box(140, 90, 210, 170, HEAD, 0.9, "t"),
            Box(102, 118, 251, 652, PERSON, 0.8, "t"),     # quasi-doublon
            Box(300, 50, 390, 400, PERSON, 0.6, "t"),      # sans tête : faux positif (drapeau)
            Box(200, 20, 330, 140, BUBBLE, 0.9, "t"),      # bulle qui touche le personnage
        ]
    return []


def run():
    return analyze_strip(strip(), spec_provider=None, judge=None, detect=detect)


def test_one_image_per_person_with_head_never_cut() -> None:
    figures = extract_figures(run())
    assert len(figures) == 1  # doublon fusionné, zone sans tête écartée
    f = figures[0]
    assert (f.x0, f.x1) == (100, 250)
    assert f.y0 == 60 + 90 and f.y1 == 60 + 650  # agrandi à la tête (90), coordonnées du strip
    assert f.heads == 1 and f.source_block == 0


def test_options_keep_headless_bubbles_whole_and_margin() -> None:
    result = run()
    assert len(extract_figures(result, require_head=False)) == 2
    whole = extract_figures(result, bubbles="whole")[0]
    assert (whole.y0 - 60, whole.x1) == (20, 330)  # bulle gardée entière
    padded = extract_figures(result, margin=0.1)[0]
    assert padded.x0 == 85 and padded.x1 == 265 and padded.y0 == 60 + 34  # marge bornée au bloc


def test_figures_are_saved_at_native_resolution(tmp_path) -> None:
    img = strip()
    figures = extract_figures(run())
    (path,) = save_figures(img, figures, tmp_path)
    saved = cv2.imread(str(path))
    assert saved.shape == (figures[0].h, figures[0].w, 3)
    assert np.array_equal(saved, img[figures[0].y0:figures[0].y1, figures[0].x0:figures[0].x1])
    assert figures_sheet(img, figures).shape[0] > 0
    assert figures_sheet(img, []).shape == (80, 400, 3)


def test_cli_figures_offline(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(detectors, "detect_all", detect)
    case = tmp_path / "case1"
    case.mkdir()
    cv2.imwrite(str(case / "strip.png"), strip())
    assert cli.main(["figures", str(case / "strip.png"), "--out", str(tmp_path / "out")]) == 0
    data = json.loads((tmp_path / "out" / "case1" / "figures.json").read_text(encoding="utf-8"))
    assert len(data) == 1 and data[0]["w"] == 150
    assert (tmp_path / "out" / "case1" / "figures_sheet.jpg").is_file()
    assert len(list((tmp_path / "out" / "case1" / "figures").glob("*.png"))) == 1


def two_people(block: np.ndarray) -> list[Box]:
    """Deux personnages côte à côte dans la même case, plus une tête sans corps détecté à cheval."""
    if block.shape[0] != 700:
        return []
    return [
        Box(20, 200, 150, 600, PERSON, 0.9, "t"), Box(50, 210, 110, 270, HEAD, 0.9, "t"),
        Box(250, 250, 380, 650, PERSON, 0.8, "t"), Box(280, 260, 340, 320, HEAD, 0.9, "t"),
        Box(140, 150, 200, 215, HEAD, 0.9, "t"),  # tête coupée par la boîte englobante : on l'inclut
    ]


def test_group_merges_characters_of_the_same_panel() -> None:
    result = analyze_strip(strip(), spec_provider=None, judge=None, detect=two_people)
    grouped = extract_figures(result, group=True)
    assert len(grouped) == 1
    g = grouped[0]
    assert (g.x0, g.x1, g.y0 - 60, g.y1 - 60) == (20, 380, 150, 650)  # englobe les deux + la tête à cheval
    assert (g.persons, g.heads) == (2, 2)
    separate = extract_figures(result, group=False)
    assert [(f.x0, f.persons) for f in separate] == [(20, 1), (250, 1)]


def test_cli_figures_separate_flag(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(detectors, "detect_all", two_people)
    case = tmp_path / "case2"
    case.mkdir()
    cv2.imwrite(str(case / "strip.png"), strip())
    assert cli.main(["figures", str(case / "strip.png"), "--out", str(tmp_path / "grouped")]) == 0
    assert cli.main(["figures", str(case / "strip.png"), "--separate", "--out", str(tmp_path / "separate")]) == 0
    assert len(json.loads((tmp_path / "grouped" / "case2" / "figures.json").read_text(encoding="utf-8"))) == 1
    assert len(json.loads((tmp_path / "separate" / "case2" / "figures.json").read_text(encoding="utf-8"))) == 2


def stacked(block: np.ndarray) -> list[Box]:
    """Deux cases empilées sans gouttière : un personnage en haut, un autre en bas."""
    if block.shape[0] != 700:
        return []
    return [
        Box(20, 20, 200, 300, PERSON, 0.9, "t"), Box(60, 30, 120, 90, HEAD, 0.9, "t"),
        Box(150, 380, 390, 690, PERSON, 0.9, "t"), Box(200, 390, 260, 450, HEAD, 0.9, "t"),
    ]


def test_group_keeps_stacked_panels_apart() -> None:
    result = analyze_strip(strip(), spec_provider=None, judge=None, detect=stacked)
    grouped = extract_figures(result, group=True)
    assert [(f.y0 - 60, f.y1 - 60, f.persons) for f in grouped] == [(20, 300, 1), (380, 690, 1)]
