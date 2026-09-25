"""Compilation autonome : médias reliés, mémoire de série archivée, dossiers de chapitres supprimés."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from src import pipeline as pipeline_mod
from src.models.chapter import ChapterMeta
from src.models.scene import CharacterCard
from src.modules.series_memory import load_series_context, save_chapter_sheet, series_key
from src.modules.timeline_builder import build_timeline, load_timeline, save_timeline
from src.pipeline import (
    MEDIA_DIRNAME,
    build_compilation,
    compilation_dirname,
    compilation_label,
    consolidate_media,
    delete_chapter_folders,
)
from tests.test_timeline import BGM, PANELS_META, SFX, _analysis, _manifest


def meta(episode: int | float) -> ChapterMeta:
    return ChapterMeta(url=f"https://asurascans.com/comics/serie-05c7df14/chapter/{episode}", final_url="u",
                       series_title="Serie", episode_title=f"Chapter {episode}", title_no=None, episode_no=episode,
                       image_urls=["a"])


def chapter(root: Path, episode: int | float, hero: str) -> Path:
    """Dossier de chapitre traité : cases, voix, timeline, fiche de série."""
    folder = root / f"serie_ep{episode:g}"
    (folder / "audio").mkdir(parents=True)
    for entry in PANELS_META:
        (folder / entry["file"]).write_bytes(b"png " + entry["file"].encode())
    for name in ("scene_000.wav", "scene_002.wav"):
        (folder / "audio" / name).write_bytes(b"wav " + name.encode())
    (folder / "chapter.json").write_text(meta(episode).model_dump_json(), encoding="utf-8")
    timeline = build_timeline(_analysis(), _manifest(), PANELS_META, panels_dir=folder, audio_dir=folder / "audio",
                              sfx_files=SFX, bgm_files=BGM)
    save_timeline(timeline, folder / "timeline.json")
    analysis = _analysis().model_copy(update={"characters": [CharacterCard(name=hero, also_called=[], who="Hero.")]})
    save_chapter_sheet(analysis, folder, meta(episode))
    return folder


def test_names_carry_the_chapter_range(tmp_path) -> None:
    folders = [chapter(tmp_path, n, "Van") for n in (1, 2, 20)]
    assert compilation_label(folders) == "Chapitres 1-20"
    assert compilation_dirname(folders) == "serie_compilation_ch1-20"  # jamais « _ep » : ce n'est pas un chapitre


def test_consolidate_links_chapter_media_and_leaves_shared_assets(tmp_path) -> None:
    folder = chapter(tmp_path, 1, "Van")
    merged = pipeline_mod.concat_timelines([load_timeline(folder / "timeline.json")])
    media = tmp_path / "compil" / MEDIA_DIRNAME
    local = consolidate_media(merged, [folder], media)
    files = [Path(c.file) for c in local.clips] + [Path(a.file) for a in local.audio]
    assert files and all(f.parent == media.resolve() and f.is_file() for f in files)
    assert os.stat(files[0]).st_nlink == 2  # lien physique : aucun octet de plus sur le disque
    assert [c.file for c in local.sfx] == [c.file for c in merged.sfx]  # bruitages partagés : inchangés
    assert [c.file for c in local.bgm] == [c.file for c in merged.bgm]


def test_compilation_survives_the_deletion_of_its_chapters(tmp_path) -> None:
    folders = [chapter(tmp_path, 1, "Van"), chapter(tmp_path, 2, "Emily")]
    out = tmp_path / compilation_dirname(folders)
    result = build_compilation(folders, out, preview_seconds=None, make_capcut=False, delete_chapters=True)
    assert not any(f.exists() for f in folders)
    timeline = load_timeline(result.timeline_json)
    assert timeline.episode_title == "Chapitres 1-2"
    assert all(Path(c.file).is_file() for c in timeline.clips) and all(Path(a.file).is_file() for a in timeline.audio)
    # La série continue : le chapitre 3 retrouve les personnages des chapitres compilés et supprimés.
    cards, tail = load_series_context(tmp_path, key=series_key(meta(3)), before_episode=3)
    assert sorted(c.name for c in cards) == ["Emily", "Van"] and tail


def test_live_chapter_sheet_wins_over_its_archive(tmp_path) -> None:
    folder = chapter(tmp_path, 1, "Van")
    build_compilation([folder], tmp_path / "compil", preview_seconds=None, make_capcut=False)
    assert folder.exists()  # suppression seulement sur demande
    save_chapter_sheet(_analysis().model_copy(update={"characters": [CharacterCard(name="Van Ecclesia", also_called=[], who="Hero.")]}),
                       folder, meta(1))
    cards, _ = load_series_context(tmp_path, key=series_key(meta(2)), before_episode=2)
    assert [c.name for c in cards] == ["Van Ecclesia"]  # un épisode compté une fois


def test_delete_never_touches_the_compilation(tmp_path) -> None:
    folder = chapter(tmp_path, 1, "Van")
    inside = folder / "compil"
    inside.mkdir()
    delete_chapter_folders([folder, inside], keep=inside)
    assert folder.exists() and inside.exists()


# --- Ligne de commande ---------------------------------------------------------------------------
@pytest.fixture
def seen_compilation(monkeypatch: pytest.MonkeyPatch) -> dict:
    seen: dict = {}

    def fake(chapters, target, **kwargs):
        seen.update(chapters=chapters, target=target, **kwargs)
        return pipeline_mod.PipelineResult(out_dir=target)

    monkeypatch.setattr(pipeline_mod, "build_compilation", fake)
    return seen


def test_merge_renders_an_excerpt_and_deletes_chapters_by_default(tmp_path, seen_compilation) -> None:
    from typer.testing import CliRunner

    from src import main as main_mod

    folders = [chapter(tmp_path, 1, "Van"), chapter(tmp_path, 2, "Emily")]
    runner = CliRunner()
    assert runner.invoke(main_mod.app, ["merge", *map(str, folders)]).exit_code == 0
    assert seen_compilation["preview_seconds"] == 120 and seen_compilation["delete_chapters"] is True
    assert seen_compilation["target"].name == "serie_compilation_ch1-2"
    assert runner.invoke(main_mod.app, ["merge", *map(str, folders), "--keep-chapters"]).exit_code == 0
    assert seen_compilation["delete_chapters"] is False


def test_batch_compile_builds_one_video_without_chapter_previews(tmp_path, monkeypatch, seen_compilation) -> None:
    from typer.testing import CliRunner

    from src import main as main_mod
    from src.modules import batch_processor as bp

    urls = [f"https://asurascans.com/comics/serie-05c7df14/chapter/{n}" for n in (1, 2)]
    for n in (1, 2):
        chapter(tmp_path, n, "Van")
    seen: dict = {}

    def fake_run(urls, options, batch, manager=None):
        seen["options"] = options
        return bp.BatchReport(status=bp.BatchStatus(tmp_path / "status.json"), urls=list(urls))

    monkeypatch.setattr(bp, "run_batch", fake_run)
    monkeypatch.setattr(bp, "resolve_chapter_urls", lambda *a, **k: urls)
    monkeypatch.setattr(bp, "format_report", lambda report: "rapport")
    import src.utils.gemini_manager as gm
    monkeypatch.setattr(gm, "GeminiManager", lambda **k: None)
    result = CliRunner().invoke(main_mod.app, ["batch", urls[0], "--compile", "--out-root", str(tmp_path)])
    assert result.exit_code == 0, result.stdout
    assert seen["options"].make_preview is False and seen["options"].make_capcut is False
    assert [f.name for f in seen_compilation["chapters"]] == ["serie_ep1", "serie_ep2"]
    assert seen_compilation["delete_chapters"] is True and seen_compilation["make_capcut"] is True
    assert seen_compilation["target"] == tmp_path / "serie_compilation_ch1-2"
