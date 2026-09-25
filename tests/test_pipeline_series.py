"""Cablage de la memoire de serie sur l'etape d'analyse (``stage_analyze``).

Ces tests couvrent le raccordement, pas la memoire elle-meme (voir
``test_series_memory.py``) : la fiche du chapitre precedent doit **arriver** a l'analyzer,
et celle du chapitre courant doit **repartir** sur le disque - y compris quand l'analyse
est reutilisee, sans quoi les chapitres produits avant la fonctionnalite resteraient
invisibles pour tous les suivants.

Aucun appel reseau ni Gemini : l'analyzer est remplace par une doublure.
"""

from __future__ import annotations

import json

import pytest

from src import pipeline as pipeline_mod
from src.models.chapter import ChapterMeta
from src.models.scene import ChapterAnalysis, CharacterCard, Scene
from src.modules.series_memory import CHARACTERS_FILE, save_chapter_sheet
from src.pipeline import PipelineOptions, stage_analyze


def _meta(episode: int | None = 2, title_no: int | None = 9674) -> ChapterMeta:
    return ChapterMeta(
        url=f"https://www.webtoons.com/en/x/y/viewer?title_no={title_no}&episode_no={episode}",
        final_url="https://www.webtoons.com/en/x/y/viewer",
        series_title="Mirror World", episode_title=f"Ep. {episode}",
        title_no=title_no, episode_no=episode, image_urls=["a"],
    )


def _analysis(characters: list[CharacterCard] | None = None, last: str = "The gate closes.") -> ChapterAnalysis:
    return ChapterAnalysis(
        series_title="Mirror World", episode_title="Ep. 2", model="m", language="en", n_panels=1,
        scenes=[Scene(index=0, panel_ids=[0], narration=last, emotion="calm")],
        characters=characters or [],
    )


@pytest.fixture
def fake_analyze(monkeypatch: pytest.MonkeyPatch) -> dict:
    """Remplace l'analyzer Gemini ; enregistre les arguments de construction."""
    seen: dict = {}

    class FakeAnalyzer:
        def __init__(self, **kwargs):
            seen["kwargs"] = kwargs
            self.api_seconds = 0.0

        def payload_bytes(self, panels):
            return 0

        def fit_for_single_call(self, panels, *, limit=None):
            seen["fit_called"] = True
            return True

        def analyze_panels_single_call(self, panels, meta=None):
            return _analysis([CharacterCard(name="Hugh", also_called=[], who="The rival.")])

    monkeypatch.setattr(pipeline_mod, "GeminiAnalyzer", FakeAnalyzer)
    monkeypatch.setattr(pipeline_mod, "load_panels", lambda out_dir: ["panel"])
    monkeypatch.setattr(pipeline_mod, "save_analysis", lambda analysis, path: path)
    return seen


def _previous_chapter(root, episode: int = 1) -> None:
    out_dir = root / f"chapter_ep{episode}"
    out_dir.mkdir(parents=True, exist_ok=True)
    save_chapter_sheet(
        _analysis([CharacterCard(name="Eden", also_called=["the revenant"], who="The hero.")],
                  last="He falls through the mirror."),
        out_dir, _meta(episode),
    )


def test_stage_analyze_injects_the_previous_sheet_into_the_analyzer(fake_analyze, tmp_path) -> None:
    _previous_chapter(tmp_path)
    out_dir = tmp_path / "chapter_ep2"
    out_dir.mkdir()
    stage_analyze(_meta(2), out_dir, PipelineOptions())

    kwargs = fake_analyze["kwargs"]
    assert [c.name for c in kwargs["known_characters"]] == ["Eden"]
    assert kwargs["previous_tail"] == "He falls through the mirror."


def test_stage_analyze_writes_the_sheet_of_the_chapter_it_just_analysed(fake_analyze, tmp_path) -> None:
    out_dir = tmp_path / "chapter_ep2"
    out_dir.mkdir()
    stage_analyze(_meta(2), out_dir, PipelineOptions())

    data = json.loads((out_dir / CHARACTERS_FILE).read_text(encoding="utf-8"))
    assert [c["name"] for c in data["characters"]] == ["Hugh"]
    assert data["episode_no"] == 2 and data["series"] == "t9674"


def test_stage_analyze_writes_the_sheet_even_when_the_analysis_is_reused(
    fake_analyze, tmp_path, monkeypatch
) -> None:
    """Un chapitre traite avant la memoire de serie n'a pas de fiche. Sans ecriture dans la
    branche de reutilisation, il resterait invisible et aucun ``--redo`` ne le reparerait."""
    out_dir = tmp_path / "chapter_ep2"
    out_dir.mkdir()
    (out_dir / "scenes.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(
        pipeline_mod, "load_analysis",
        lambda path: _analysis([CharacterCard(name="Nora", also_called=[], who="The captain.")]),
    )
    result = pipeline_mod.PipelineResult(out_dir=out_dir)
    stage_analyze(_meta(2), out_dir, PipelineOptions(), result)

    assert "analyze" in result.reused
    assert "kwargs" not in fake_analyze                       # aucun appel Gemini
    data = json.loads((out_dir / CHARACTERS_FILE).read_text(encoding="utf-8"))
    assert [c["name"] for c in data["characters"]] == ["Nora"]


def test_the_memory_can_be_switched_off(fake_analyze, tmp_path) -> None:
    _previous_chapter(tmp_path)
    out_dir = tmp_path / "chapter_ep2"
    out_dir.mkdir()
    stage_analyze(_meta(2), out_dir, PipelineOptions(series_memory=False))

    assert fake_analyze["kwargs"]["known_characters"] == []
    assert fake_analyze["kwargs"]["previous_tail"] == ""
    assert not (out_dir / CHARACTERS_FILE).exists()           # ni lecture, ni ecriture


def test_a_chapter_without_an_episode_number_never_reads_its_own_sheet(fake_analyze, tmp_path, caplog) -> None:
    """Sans numero d'episode, impossible de savoir quels chapitres precedent celui-ci :
    la memoire se coupe plutot que de risquer un chapitre qui se cite lui-meme."""
    import logging

    _previous_chapter(tmp_path)
    out_dir = tmp_path / "chapter_loose"
    out_dir.mkdir()
    with caplog.at_level(logging.INFO, logger="src.pipeline"):
        stage_analyze(_meta(None), out_dir, PipelineOptions())

    assert fake_analyze["kwargs"]["known_characters"] == []
    assert "numero d'episode inconnu" in caplog.text
    assert not (out_dir / CHARACTERS_FILE).exists()


def test_series_root_points_the_memory_at_the_shared_folder(fake_analyze, tmp_path) -> None:
    """``run --out <ailleurs>`` sort le chapitre du dossier de la serie : sans ``series_root``,
    il n'aurait plus de voisins a lire."""
    shared = tmp_path / "output"
    _previous_chapter(shared)
    elsewhere = tmp_path / "somewhere" / "else"
    elsewhere.mkdir(parents=True)

    stage_analyze(_meta(2), elsewhere, PipelineOptions())
    assert fake_analyze["kwargs"]["known_characters"] == []   # aucun voisin a cet endroit

    stage_analyze(_meta(2), elsewhere, PipelineOptions(series_root=str(shared)))
    assert [c.name for c in fake_analyze["kwargs"]["known_characters"]] == ["Eden"]
