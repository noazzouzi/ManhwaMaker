"""Cablage du profil de format sur la decoupe (``stage_scrape_slice``).

Cette etape n'etait couverte par aucun test : un import manquant y est passe inapercu
alors qu'il aurait plante chaque execution reelle. Les doublures evitent tout reseau.
"""

from __future__ import annotations

import json

import pytest

from src import pipeline as pipeline_mod
from src.models.chapter import ChapterMeta
from src.pipeline import PipelineOptions, stage_scrape_slice


@pytest.fixture
def fake_slicing(monkeypatch: pytest.MonkeyPatch) -> dict:
    """Remplace scraping, decoupe et ecritures ; enregistre les arguments recus."""
    seen: dict = {}
    meta = ChapterMeta(
        url="https://www.webtoons.com/en/x/y/viewer?title_no=1&episode_no=1",
        final_url="https://www.webtoons.com/en/x/y/viewer?title_no=1&episode_no=1",
        series_title="S", episode_title="E", title_no=1, episode_no=1,
        image_urls=["https://example.invalid/a.jpg"], chunk_sizes=[[800, 1280]],
    )
    monkeypatch.setattr(pipeline_mod, "scrape_chapter", lambda url, language=None: ("STRIP", meta))

    def fake_slice(strip, **kwargs):
        seen["strip"], seen["kwargs"] = strip, kwargs
        return ["panel"]

    monkeypatch.setattr(pipeline_mod, "slice_panels", fake_slice)
    monkeypatch.setattr(pipeline_mod, "save_panels", lambda panels, out_dir: seen.setdefault("saved", out_dir))
    monkeypatch.setattr(pipeline_mod, "render_debug_overlay", lambda *a, **k: None)
    monkeypatch.setattr(pipeline_mod, "load_panels_meta", lambda out_dir: ["panel"])

    def fake_figures(out_dir, options, force=False):
        seen.setdefault("figures", []).append((options, force))
        return [{"index": 0}]

    monkeypatch.setattr(pipeline_mod, "ensure_figures", fake_figures)
    monkeypatch.setattr(pipeline_mod, "ensure_upscaled",
                        lambda out_dir, frame, force=False: seen.setdefault("upscaled", []).append((frame, force)))
    return seen


def test_long_profile_asks_the_slicer_for_its_frame(fake_slicing, tmp_path) -> None:
    """Le format long affiche la case entiere : la decoupe vise la hauteur du cadre."""
    stage_scrape_slice("https://www.webtoons.com/en/x/y/viewer?title_no=1&episode_no=1", tmp_path, PipelineOptions())
    assert fake_slicing["kwargs"] == {"frame_height": 1080, "frame_width": 1920}
    assert json.loads((tmp_path / "slice_params.json").read_text(encoding="utf-8")) == fake_slicing["kwargs"]


def test_short_profile_disables_splitting(fake_slicing, tmp_path) -> None:
    """Le format court recadre en 9:16 par saillance : viser une hauteur de cadre n'a pas de sens."""
    options = PipelineOptions(video_format="SHORT")
    stage_scrape_slice("https://www.webtoons.com/en/x/y/viewer?title_no=1&episode_no=1", tmp_path, options)
    assert fake_slicing["kwargs"]["frame_height"] == 0
    assert fake_slicing["kwargs"]["frame_width"] == 1080


def test_a_custom_resolution_reaches_the_slicer(fake_slicing, tmp_path) -> None:
    options = PipelineOptions(width=1280, height=720)
    stage_scrape_slice("https://www.webtoons.com/en/x/y/viewer?title_no=1&episode_no=1", tmp_path, options)
    assert fake_slicing["kwargs"] == {"frame_height": 720, "frame_width": 1280}


def test_changing_the_frame_forces_a_new_slicing(fake_slicing, tmp_path) -> None:
    """Le cache ne regardait que l'existence de panels.json : il aurait garde une decoupe
    faite pour un autre cadre, avec des numeros de case devenus faux."""
    url = "https://www.webtoons.com/en/x/y/viewer?title_no=1&episode_no=1"
    stage_scrape_slice(url, tmp_path, PipelineOptions())
    (tmp_path / "panels.json").write_text("[]", encoding="utf-8")
    fake_slicing.pop("kwargs")

    # Meme cadre : la decoupe est reutilisee, le slicer n'est pas rappele.
    result = pipeline_mod.PipelineResult(out_dir=tmp_path)
    stage_scrape_slice(url, tmp_path, PipelineOptions(), result)
    assert "kwargs" not in fake_slicing and "scrape+slice" in result.reused

    # Cadre different : on re-decoupe.
    stage_scrape_slice(url, tmp_path, PipelineOptions(width=1280, height=720))
    assert fake_slicing["kwargs"] == {"frame_height": 720, "frame_width": 1280}


def test_figures_mode_extracts_characters_after_slicing(fake_slicing, tmp_path) -> None:
    """Mode par defaut : les cases personnages sont calculees apres la decoupe, avec les reglages."""
    options = PipelineOptions(figure_margin=0.05, figure_bubbles="whole")
    result = pipeline_mod.PipelineResult(out_dir=tmp_path)
    stage_scrape_slice("https://www.webtoons.com/en/x/y/viewer?title_no=1&episode_no=1", tmp_path, options, result)
    ((figure_options, force),) = fake_slicing["figures"]
    assert (figure_options.margin, figure_options.bubbles, figure_options.require_head) == (0.05, "whole", True)
    assert result.n_figures == 1 and "figures" in result.timings
    assert fake_slicing["upscaled"] == [((1920, 1080), False)] and "upscale" in result.timings


def test_upscaling_can_be_turned_off(fake_slicing, tmp_path) -> None:
    result = pipeline_mod.PipelineResult(out_dir=tmp_path)
    stage_scrape_slice("https://www.webtoons.com/en/x/y/viewer?title_no=1&episode_no=1", tmp_path,
                       PipelineOptions(figure_upscale=False), result)
    assert "upscaled" not in fake_slicing and "upscale" not in result.timings


def test_slicer_mode_skips_characters(fake_slicing, tmp_path) -> None:
    result = pipeline_mod.PipelineResult(out_dir=tmp_path)
    stage_scrape_slice("https://www.webtoons.com/en/x/y/viewer?title_no=1&episode_no=1", tmp_path,
                       PipelineOptions(panels="slicer"), result)
    assert "figures" not in fake_slicing and "upscaled" not in fake_slicing and result.n_figures is None
