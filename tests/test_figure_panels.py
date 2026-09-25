"""Cases personnages branchées sur le pipeline : strip reconstitué, carte, traduction de l'analyse, montage."""

from __future__ import annotations

import json

import numpy as np
import pytest

from src import pipeline as pipeline_mod
from src.models.panel import Panel
from src.models.scene import Beat, ChapterAnalysis, Scene
from src.modules import figure_panels as fp
from src.modules.slicer import load_panels, save_panels
from src.modules.toonsplit.geometry import BUBBLE, HEAD, PERSON, Box
from src.pipeline import PipelineOptions, PipelineResult
from tests.toonsplit_helpers import art

W = 400


def reading_panels() -> list[Panel]:
    """Trois cases de lecture (RGB) séparées par des gouttières retirées."""
    spans = [(40, 740), (840, 1440), (1540, 1840)]
    return [
        Panel(index=i, y_start=a, y_end=b, height=b - a, width=W, type="static", image=art(b - a, W, seed=i)[:, :, ::-1].copy())
        for i, (a, b) in enumerate(spans)
    ]


def fake_detect(block: np.ndarray) -> list[Box]:
    """Deux personnages dans la 1re case, un dans la 2e, aucun dans la 3e (paysage)."""
    h = block.shape[0]
    boxes: list[Box] = []
    if h == 700:
        boxes += [Box(20, 100, 180, 650, PERSON, 0.9, "t"), Box(60, 110, 120, 180, HEAD, 0.9, "t"),
                  Box(220, 150, 390, 690, PERSON, 0.8, "t"), Box(260, 160, 330, 230, HEAD, 0.9, "t"),
                  Box(100, 5, 300, 90, BUBBLE, 0.9, "t")]
    elif h == 600:
        boxes += [Box(100, 50, 300, 580, PERSON, 0.9, "t"), Box(150, 60, 240, 150, HEAD, 0.9, "t")]
    return boxes


# --- Strip reconstitué et rattachement --------------------------------------------------------
def test_strip_from_panels_puts_each_panel_back_in_place() -> None:
    panels = reading_panels()
    strip = fp.strip_from_panels(panels)
    assert strip.shape == (1840, W, 3)
    assert np.array_equal(strip[840:1440], panels[1].image)
    assert (strip[740:840] == 255).all() and (strip[:40] == 255).all()
    with pytest.raises(ValueError):
        fp.strip_from_panels([])


def test_reading_ids_for_overlap_share() -> None:
    panels = reading_panels()
    assert fp.reading_ids_for(100, 600, panels) == [0]
    assert fp.reading_ids_for(600, 1000, panels) == [0, 1]  # à cheval : 140 et 160 px sur 400
    assert fp.reading_ids_for(700, 1500, panels) == [1]  # 40 px sur 800 ne suffit pas à la case 0
    assert fp.reading_ids_for(760, 800, panels) == []  # dans une gouttière


# --- Calcul et réutilisation ---------------------------------------------------------------------
def test_ensure_figures_writes_panels_map_and_reuses(tmp_path) -> None:
    save_panels(reading_panels(), tmp_path)
    calls = []

    def counting(block):
        calls.append(block.shape[0])
        return fake_detect(block)

    entries = fp.ensure_figures(tmp_path, detect=counting)
    # Les deux personnages de la 1re case sont regroupés dans une seule image (boîte englobante).
    assert [(e["index"], e["reading_panels"], e["persons"]) for e in entries] == [(0, [0], 2), (1, [1], 1)]
    figures = load_panels(tmp_path / fp.FIGURES_DIRNAME)
    assert [(f.width, f.height) for f in figures] == [(370, 590), (200, 530)]
    first = reading_panels()[0]
    assert np.array_equal(figures[0].image, first.image[100:690, 20:390])  # pixels natifs, sans retouche
    assert (tmp_path / fp.FIGURES_DIRNAME / fp.OVERLAY_FILE).is_file()

    n = len(calls)
    assert fp.ensure_figures(tmp_path, detect=counting) == entries and len(calls) == n  # réutilisé
    separate = fp.ensure_figures(tmp_path, fp.FigureOptions(group=False), detect=counting)
    assert len(calls) > n  # réglage changé : recalcul
    assert [(e["reading_panels"], e["persons"]) for e in separate] == [([0], 1), ([0], 1), ([1], 1)]
    assert [(f.width, f.height) for f in load_panels(tmp_path / fp.FIGURES_DIRNAME)] == [(160, 550), (170, 540), (200, 530)]


def test_ensure_figures_recomputes_when_reading_panels_change(tmp_path) -> None:
    save_panels(reading_panels(), tmp_path)
    fp.ensure_figures(tmp_path, detect=fake_detect)
    save_panels(reading_panels()[:1], tmp_path)
    entries = fp.ensure_figures(tmp_path, detect=fake_detect)
    assert len(entries) == 1 and entries[0]["reading_panels"] == [0]


# --- Traduction de l'analyse ----------------------------------------------------------------------
def analysis() -> ChapterAnalysis:
    return ChapterAnalysis(
        model="m", language="en", n_panels=3,
        scenes=[
            Scene(index=0, panel_ids=[0], narration="Two heroes.", emotion="calm", action_heavy_ids=[0]),
            Scene(index=1, panel_ids=[1, 0], narration="Then one.", emotion="tension"),
            Scene(index=2, panel_ids=[2], narration="The valley.", emotion="calm"),
        ],
        beats=[Beat(index=0, panel_ids=[0, 1, 2], summary="s", characters=[], dialogue=[])],
    )


MAP = [
    {"index": 0, "x0": 20, "y0": 140, "x1": 180, "y1": 690, "reading_panels": [0]},
    {"index": 1, "x0": 220, "y0": 190, "x1": 390, "y1": 730, "reading_panels": [0]},
    {"index": 2, "x0": 100, "y0": 890, "x1": 300, "y1": 1420, "reading_panels": [1]},
]


def test_remap_translates_ids_in_reading_order_with_fallback() -> None:
    out = fp.remap_analysis(analysis(), MAP)
    assert [s.panel_ids for s in out.scenes] == [[0, 1], [2, 0, 1], [2]]  # scène 3 : paysage → personnage le plus proche
    assert out.scenes[0].action_heavy_ids == [0, 1]
    assert out.beats[0].panel_ids == [0, 1, 2] and out.n_panels == 3
    assert [s.narration for s in out.scenes] == [s.narration for s in analysis().scenes]  # le texte ne change pas


def test_remap_without_figures_leaves_scenes_empty() -> None:
    out = fp.remap_analysis(analysis(), [])
    assert all(s.panel_ids == [] for s in out.scenes)


# --- Montage ---------------------------------------------------------------------------------------
def test_montage_uses_figure_panels_and_translated_analysis(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    fig_dir = tmp_path / fp.FIGURES_DIRNAME
    fig_dir.mkdir()
    (fig_dir / "panels.json").write_text(json.dumps([{"index": 0}]), encoding="utf-8")
    monkeypatch.setattr(pipeline_mod, "ensure_figures", lambda out_dir, options, force=False: MAP)
    seen = {}

    def fake_timeline(analysis, manifest, panels_meta, *, panels_dir, **kwargs):
        seen.update(analysis=analysis, panels_dir=panels_dir, meta=panels_meta)
        raise StopIteration  # la suite du montage (CapCut, rendu) n'est pas testée ici

    monkeypatch.setattr(pipeline_mod, "build_timeline", fake_timeline)
    options = PipelineOptions(use_sfx=False, use_bgm=False, figure_upscale=False)
    result = PipelineResult(out_dir=tmp_path)
    with pytest.raises(StopIteration):
        pipeline_mod.stage_montage(analysis(), None, None, tmp_path, options, result)
    assert seen["panels_dir"] == fig_dir and seen["meta"] == [{"index": 0}]
    assert [s.panel_ids for s in seen["analysis"].scenes][0] == [0, 1] and result.n_figures == 3


def test_montage_falls_back_to_whole_panels_without_characters(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    (tmp_path / "panels.json").write_text(json.dumps([{"index": 0}]), encoding="utf-8")
    monkeypatch.setattr(pipeline_mod, "ensure_figures", lambda out_dir, options, force=False: [])
    seen = {}

    def fake_timeline(analysis, manifest, panels_meta, *, panels_dir, **kwargs):
        seen.update(analysis=analysis, panels_dir=panels_dir)
        raise StopIteration

    monkeypatch.setattr(pipeline_mod, "build_timeline", fake_timeline)
    with pytest.raises(StopIteration):
        pipeline_mod.stage_montage(analysis(), None, None, tmp_path, PipelineOptions(use_sfx=False, use_bgm=False))
    assert seen["panels_dir"] == tmp_path and seen["analysis"].scenes[0].panel_ids == [0]


def test_slicer_mode_montage_is_unchanged(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    (tmp_path / "panels.json").write_text(json.dumps([{"index": 0}]), encoding="utf-8")
    monkeypatch.setattr(pipeline_mod, "ensure_figures", lambda *a, **k: pytest.fail("pas de personnages en mode slicer"))
    seen = {}

    def fake_timeline(analysis, manifest, panels_meta, *, panels_dir, **kwargs):
        seen.update(panels_dir=panels_dir)
        raise StopIteration

    monkeypatch.setattr(pipeline_mod, "build_timeline", fake_timeline)
    with pytest.raises(StopIteration):
        pipeline_mod.stage_montage(analysis(), None, None, tmp_path,
                                   PipelineOptions(panels="slicer", use_sfx=False, use_bgm=False))
    assert seen["panels_dir"] == tmp_path
