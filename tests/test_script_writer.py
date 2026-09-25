"""Script écrit par Claude : planches d'entrée, prompt, garde-fous, repli sur Gemini dans le pipeline."""

from __future__ import annotations

import numpy as np
import pytest

from src import pipeline as pipeline_mod
from src.models.chapter import ChapterMeta
from src.models.panel import Panel
from src.models.scene import ChapterAnalysis, CharacterCard, Scene
from src.modules import script_writer as sw
from src.modules.analyzer import CTA_TEXTS
from src.modules.toonsplit.ai import AiError, JsonClient
from src.pipeline import PipelineOptions, stage_analyze


def panel(index: int, height: int = 600, width: int = 800) -> Panel:
    image = np.full((height, width, 3), 40 + index % 200, np.uint8)
    return Panel(index=index, y_start=index * 1000, y_end=index * 1000 + height, height=height, width=width,
                 type="static", image=image)


def meta(episode: int | None = 1) -> ChapterMeta:
    return ChapterMeta(url="https://asurascans.com/comics/x-05c7df14/chapter/1", final_url="u", series_title="Bad Born Blood",
                       episode_title=f"Chapter {episode}", title_no=None, episode_no=episode, image_urls=["a"])


class FakeClient(JsonClient):
    """Transport factice : renvoie les réponses prévues, dans l'ordre."""

    def __init__(self, *answers: dict) -> None:
        super().__init__(retries=1)
        self.answers = list(answers)
        self.prompts: list[tuple[str, str, int]] = []

    def _send(self, system, text, images, schema, label):
        self.prompts.append((system, text, len(images)))
        self.cost_usd += 1.5
        return self.answers.pop(0)


def paragraph(text: str, keys: list[int], heavy: list[int] | None = None, emotion: str = "tension") -> dict:
    return {"text": text, "emotion": emotion, "key_panel_ids": keys, "action_heavy_ids": heavy or []}


ANSWER = {
    "paragraphs": [
        paragraph("Our journey begins in a sealed white hall.", [0, 1], [1]),
        paragraph("The rifles open up.", [1, 2, 99]),  # 1 déjà pris, 99 inconnu : retirés
        paragraph("Luka waits his turn.", [3]),
        paragraph("A red light glows.", [4]),
        paragraph('He whispers, "For the empire."', [5]),
    ],
    "characters": [
        {"name": "Luka", "also_called": ["the trainee"], "who": "A boy raised as a weapon."},
        {"name": "luka", "also_called": [], "who": "Doublon."},
    ],
}


# --- Planches ------------------------------------------------------------------------------------
def test_composites_keep_every_panel_in_order_within_the_limits() -> None:
    panels = [panel(i) for i in range(12)] + [panel(12, height=9000)]
    sheets = sw.composites(panels, max_px=1_100_000, max_images=5)
    assert [i for ids, _ in sheets for i in ids] == list(range(13))
    assert len(sheets) <= 5
    assert all(img.shape[0] * img.shape[1] <= 1_100_000 for _, img in sheets)


# --- Prompt --------------------------------------------------------------------------------------
def test_prompt_opens_the_story_on_the_first_chapter_only() -> None:
    first = sw.system_prompt(meta(1))
    assert '"Our journey begins"' in first and "Series: Bad Born Blood" in first
    later = sw.system_prompt(meta(2), known_characters=[CharacterCard(name="Luka", also_called=[], who="The hero.")],
                             previous_tail="The gate closes.")
    assert '"Our hero"' in later and "Luka: The hero." in later and "The gate closes." in later
    assert "{" not in first.split("OUTPUT")[0]  # tous les champs du modèle sont remplis


# --- Rédaction -----------------------------------------------------------------------------------
def test_writer_builds_the_analysis_with_the_usual_safeguards() -> None:
    client = FakeClient(ANSWER)
    writer = sw.ClaudeScriptWriter(client=client)
    analysis = writer.analyze([panel(i) for i in range(6)], meta())
    assert [s.panel_ids for s in analysis.scenes] == [[0, 1], [2], [3], [4], [5]]
    assert analysis.scenes[0].action_heavy_ids == [1]
    assert analysis.model == sw.DEFAULT_MODEL and analysis.n_batches == 1 and analysis.n_panels == 6
    assert [c.name for c in analysis.characters] == ["luka"]  # un nom, une fiche
    # Appel à l'abonnement de la langue, placé vers 40 % (convention du pipeline).
    assert CTA_TEXTS["en"] in analysis.scenes[2].narration
    system, text, n_images = client.prompts[0]
    assert '"Our journey begins"' in system and "numbered 0 to 5" in text and n_images >= 1
    assert writer.cost_usd == 1.5


def test_writer_without_call_to_subscribe() -> None:
    analysis = sw.ClaudeScriptWriter(client=FakeClient(ANSWER), cta_text="").analyze([panel(i) for i in range(6)], meta())
    assert not any("subscribe" in s.narration for s in analysis.scenes)


def test_writer_asks_again_when_the_panel_numbers_are_wrong() -> None:
    wrong = {"paragraphs": [paragraph("A.", [100, 101]), paragraph("B.", [102, 103]), paragraph("C.", [104])], "characters": []}
    client = FakeClient(wrong, ANSWER)
    analysis = sw.ClaudeScriptWriter(client=client).analyze([panel(i) for i in range(6)], meta())
    assert len(client.prompts) == 2 and "rejected" in client.prompts[1][1]
    assert analysis.n_scenes == 5


# --- Pipeline : Claude d'abord, Gemini en repli ----------------------------------------------------
def gemini_analysis() -> ChapterAnalysis:
    return ChapterAnalysis(model="gemini-3.5-flash", language="en", n_panels=1,
                           scenes=[Scene(index=0, panel_ids=[0], narration="Gemini.", emotion="calm")])


@pytest.fixture
def stages(monkeypatch: pytest.MonkeyPatch) -> dict:
    calls: dict = {"claude": 0, "gemini": 0}

    class FakeGemini:
        def __init__(self, **kwargs):
            self.api_seconds = 2.0

        def fit_for_single_call(self, panels):
            return True

        def analyze_panels_single_call(self, panels, meta=None):
            calls["gemini"] += 1
            return gemini_analysis()

    class FakeWriter:
        def __init__(self, **kwargs):
            calls["kwargs"] = kwargs
            self.api_seconds = 5.0

        def analyze(self, panels, meta=None):
            calls["claude"] += 1
            if calls.get("fail"):
                raise AiError("claude CLI : limite d'usage")
            return gemini_analysis().model_copy(update={"model": sw.DEFAULT_MODEL})

    monkeypatch.setattr(pipeline_mod, "GeminiAnalyzer", FakeGemini)
    monkeypatch.setattr(pipeline_mod, "ClaudeScriptWriter", FakeWriter)
    monkeypatch.setattr(pipeline_mod, "load_panels", lambda out_dir: ["panel"])
    return calls


def test_claude_writes_the_script_by_default(stages, tmp_path) -> None:
    analysis = stage_analyze(meta(), tmp_path, PipelineOptions(series_memory=False))
    assert analysis.model == sw.DEFAULT_MODEL and (stages["claude"], stages["gemini"]) == (1, 0)
    assert stages["kwargs"]["model"] == sw.DEFAULT_MODEL
    assert (tmp_path / "scenes.json").is_file()


def test_gemini_takes_over_when_claude_fails(stages, tmp_path) -> None:
    stages["fail"] = True
    result = pipeline_mod.PipelineResult(out_dir=tmp_path)
    analysis = stage_analyze(meta(), tmp_path, PipelineOptions(series_memory=False), result)
    assert analysis.model == "gemini-3.5-flash" and (stages["claude"], stages["gemini"]) == (1, 1)
    assert result.gemini_seconds == 7.0  # temps passé chez les deux IA


def test_gemini_only_never_calls_claude(stages, tmp_path) -> None:
    stage_analyze(meta(), tmp_path, PipelineOptions(series_memory=False, script_ai="gemini"))
    assert (stages["claude"], stages["gemini"]) == (0, 1)
