"""Tests de la CLI (``src.main``).

Aucun appel reseau, aucun rendu : ``run_pipeline`` et les rendus sont remplaces par des
doublures. Ces tests couvrent surtout ``_pipeline_options``, la ou les options de la ligne
de commande deviennent des ``PipelineOptions`` -- l'endroit le plus propice a une
regression silencieuse quand on ajoute un drapeau.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from typer.testing import CliRunner

from src import main as main_mod
from src.pipeline import PipelineOptions

runner = CliRunner()


class _DummyManager:
    """Remplace ``GeminiManager`` : aucune cle, aucun appel."""

    def __init__(self, *args, **kwargs) -> None:
        self.kwargs = kwargs


@pytest.fixture
def no_gemini(monkeypatch: pytest.MonkeyPatch) -> None:
    """``GeminiManager`` est importe DANS la fonction : il faut patcher le module source."""
    from src.utils import gemini_manager as gm

    monkeypatch.setattr(gm, "GeminiManager", _DummyManager)


def test_cli_help_lists_every_command() -> None:
    """Le moins cher des tests, et il attrape une signature ``Annotated`` cassee."""
    result = runner.invoke(main_mod.app, ["--help"])
    assert result.exit_code == 0
    for command in ("run", "batch", "merge", "voices", "stats", "preview", "capcut", "thumbnail"):
        assert command in result.stdout


def test_run_maps_the_flags_onto_pipeline_options(monkeypatch: pytest.MonkeyPatch, no_gemini, tmp_path) -> None:
    seen: dict = {}

    def fake_run(url, out_dir, options, manager=None):
        seen["url"], seen["out_dir"], seen["options"] = url, out_dir, options
        return "RESULT"

    monkeypatch.setattr(main_mod, "run_pipeline", fake_run)
    monkeypatch.setattr(main_mod, "format_result", lambda result: f"ok {result}")

    result = runner.invoke(main_mod.app, [
        "run", "https://www.webtoons.com/en/x/y/viewer?title_no=1&episode_no=2",
        "--out", str(tmp_path / "chap"), "--language", "en", "--voice", "am_puck",
        "--speed", "1.1", "--fps", "30", "--format", "SHORT", "--dynamics", "none",
        "--no-preview", "--no-capcut", "--no-bgm", "--no-sfx",
    ])

    assert result.exit_code == 0, result.stdout
    options: PipelineOptions = seen["options"]
    assert seen["out_dir"] == tmp_path / "chap"
    assert options.language == "en" and options.voice == "am_puck"
    assert options.speed == pytest.approx(1.1) and options.fps == 30
    assert options.video_format == "SHORT" and options.dynamics == "none"
    # Les quatre interrupteurs "--no-*" doivent tous etre pris en compte.
    # Attention : --no-bgm / --no-sfx basculent use_bgm / use_sfx, PAS bgm_dir / sfx_dir,
    # qui gardent leur dossier par defaut.
    assert not options.make_preview and not options.make_capcut
    assert not options.use_bgm and not options.use_sfx
    assert options.figure_upscale  # personnages agrandis par defaut


def test_no_upscale_flag_keeps_native_figures(monkeypatch: pytest.MonkeyPatch, no_gemini) -> None:
    seen: dict = {}
    monkeypatch.setattr(main_mod, "run_pipeline", lambda url, out_dir, options, manager=None: seen.setdefault("options", options))
    monkeypatch.setattr(main_mod, "format_result", lambda result: "ok")
    result = runner.invoke(main_mod.app, ["run", "https://www.webtoons.com/en/x/y/viewer?title_no=1&episode_no=2", "--no-upscale"])
    assert result.exit_code == 0, result.stdout
    assert seen["options"].figure_upscale is False


def test_run_prints_failed_and_exits_1(monkeypatch: pytest.MonkeyPatch, no_gemini) -> None:
    def boom(url, out_dir, options, manager=None):
        raise RuntimeError("scraping casse")

    monkeypatch.setattr(main_mod, "run_pipeline", boom)
    result = runner.invoke(main_mod.app, ["run", "https://www.webtoons.com/en/x/y/viewer?title_no=1&episode_no=1"])
    assert result.exit_code == 1
    assert "FAILED" in result.stdout and "scraping casse" in result.stdout


def test_merge_exits_2_when_no_chapter_matches() -> None:
    """Un motif qui ne trouve rien est une erreur d'usage (2), pas un plantage (1)."""
    result = runner.invoke(main_mod.app, ["merge", "--pattern", "output/rien_du_tout_ep*"])
    assert result.exit_code == 2
    assert "FAILED" in result.stdout


def test_merge_crashes_on_an_absolute_pattern(tmp_path) -> None:
    """Fige un DEFAUT connu, pas un comportement voulu.

    ``chapter_folders`` fait un glob relatif : un motif absolu leve NotImplementedError,
    qui n'est pas rattrapee par le ``except ValueError`` de la commande. L'utilisateur
    recoit une trace brute, sans le "FAILED:" des autres erreurs. A corriger : rattraper
    NotImplementedError avec ValueError et sortir en code 2. Ce test devra alors changer.
    """
    result = runner.invoke(main_mod.app, ["merge", "--pattern", str(tmp_path / "chap_*")])
    assert result.exit_code == 1
    assert isinstance(result.exception, NotImplementedError)
    assert "FAILED" not in result.stdout  # <- ce qu'on voudrait voir un jour


def test_preview_reads_timeline_json(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    from src.modules import preview_renderer as pr_mod
    from src.modules import timeline_builder as tb_mod

    read: dict = {}

    def fake_load(path):
        read["path"] = Path(path)
        return "TIMELINE"

    class FakeRenderer:
        def __init__(self, timeline, fps=None) -> None:
            read["timeline"], read["fps"] = timeline, fps

        def render(self, target, max_duration_s=None):
            read["target"], read["max_duration_s"] = Path(target), max_duration_s
            return Path(target)

    monkeypatch.setattr(tb_mod, "load_timeline", fake_load)
    monkeypatch.setattr(pr_mod, "PreviewRenderer", FakeRenderer)

    result = runner.invoke(main_mod.app, ["preview", str(tmp_path), "--seconds", "30"])
    assert result.exit_code == 0, result.stdout
    assert read["path"] == tmp_path / "timeline.json"
    assert read["target"] == tmp_path / "preview_30s.mp4"
    assert read["max_duration_s"] == pytest.approx(30.0)


def test_capcut_reads_timeline_json(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    from src.modules import capcut_builder as cb_mod
    from src.modules import timeline_builder as tb_mod

    read: dict = {}

    def fake_load(path):
        read["path"] = Path(path)
        return "TIMELINE"

    def fake_build(timeline, out_dir, name):
        read["out_dir"], read["name"] = Path(out_dir), name
        return Path(out_dir) / name

    monkeypatch.setattr(tb_mod, "load_timeline", fake_load)
    monkeypatch.setattr(cb_mod, "build_capcut_draft", fake_build)
    monkeypatch.setattr(cb_mod, "detect_capcut_drafts_dir", lambda: None)

    result = runner.invoke(main_mod.app, ["capcut", str(tmp_path), "--name", "Mon Projet"])
    assert result.exit_code == 0, result.stdout
    assert read["path"] == tmp_path / "timeline.json"
    assert read["out_dir"] == tmp_path / "capcut" and read["name"] == "Mon Projet"
