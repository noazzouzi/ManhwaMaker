"""Tests unitaires de l'Analyzer Gemini en deux étapes (``src.modules.analyzer``) avec un client factice.

Aucun appel réseau : ``FakeClient`` imite ``genai.Client().models.generate_content`` et
répond selon le schéma demandé (beats, script, cases clés) avec de vraies
``types.GenerateContentResponse``.
"""

from __future__ import annotations

import io
import json
import logging
import re
import time

import httpx
import numpy as np
import pytest
from google.genai import errors, types
from PIL import Image

from src.models.chapter import ChapterMeta
from src.models.panel import Panel
from src.models.scene import (
    EMOTIONS,
    Beat,
    BeatBatch,
    BeatDraft,
    ChapterAnalysis,
    KeyframeBatch,
    KeyframeChoice,
    ParagraphDraft,
    Scene,
    SceneBatch,
    SceneDraft,
    ScriptDraft,
)
from src.modules import analyzer as analyzer_mod
from src.modules.analyzer import (
    DEFAULT_MAX_IMAGE_WIDTH,
    DEFAULT_MAX_SLICE_HEIGHT,
    DEFAULT_MODEL,
    MAX_BATCH_SIZE,
    MAX_KEY_PANELS_PER_PARAGRAPH,
    MAX_SLICES_PER_PANEL,
    AnalyzerError,
    GeminiAnalyzer,
    InvalidResponseError,
    coerce_bool,
    coerce_emotion,
    encode_panel_image,
    find_forbidden,
    format_scenes,
    load_analysis,
    make_batches,
    normalize_keyframes,
    normalize_scenes,
    normalize_script,
    parse_response,
    parse_scene_json,
    resolve_model,
    save_analysis,
    scrub_forbidden,
    target_script_words,
)
from src.modules.slicer import load_panels, save_panels, slice_panels
from src.utils import config as config_mod
from tests.synthetic_strip import make_synthetic_strip

_CAPTION = re.compile(r"^Panel (\d+) \(")
_BEAT_LINE = re.compile(r"^Beat (\d+) \(panels ([\d, ]+)\)(\s*\[FILLER[^\]]*\])?:", re.MULTILINE)
_PARAGRAPH_LINE = re.compile(r"^Paragraph (\d+) \(candidate panels: ([\d, ]*)\):", re.MULTILINE)


# --- Outils ------------------------------------------------------------------------
def _panel(index: int, height: int = 80, width: int = 120, kind: str = "static") -> Panel:
    rng = np.random.default_rng(index)
    return Panel(
        index=index, y_start=index * 100, y_end=index * 100 + height, height=height, width=width,
        type=kind,  # type: ignore[arg-type]
        image=rng.integers(0, 256, size=(height, width, 3), dtype=np.uint8),
    )


def _panels(n: int) -> list[Panel]:
    return [_panel(i, height=80 + 10 * (i % 4)) for i in range(n)]


def _jpeg_size(data: bytes) -> tuple[int, int]:
    with Image.open(io.BytesIO(data)) as img:
        assert img.format == "JPEG"
        return img.size


def make_response(payload: str | dict | list, *, prompt_tokens: int = 100, output_tokens: int = 40, thinking_tokens: int = 0):
    """Construit une vraie ``GenerateContentResponse`` contenant ``payload`` en texte."""
    text = payload if isinstance(payload, str) else json.dumps(payload)
    return types.GenerateContentResponse(
        candidates=[types.Candidate(content=types.Content(role="model", parts=[types.Part(text=text)]))],
        usage_metadata=types.GenerateContentResponseUsageMetadata(
            prompt_token_count=prompt_tokens, candidates_token_count=output_tokens, thoughts_token_count=thinking_tokens,
        ),
    )


def all_text(contents: list[types.Part]) -> str:
    return "\n".join(part.text for part in contents if part.text)


def batch_ids_of(contents: list[types.Part]) -> list[int]:
    """Numéros de cases annoncés par les libellés ``Panel N (...)`` d'un prompt."""
    return [int(m.group(1)) for part in contents if part.text for m in [_CAPTION.match(part.text)] if m]


def beats_payload(ids: list[int]) -> dict:
    """Un beat par paire de cases."""
    beats = []
    for i in range(0, len(ids), 2):
        group = ids[i : i + 2]
        beats.append({"panel_ids": group, "summary": f"Beat with panels {group}.", "characters": ["Hero"], "dialogue": []})
    return {"beats": beats}


def script_payload(contents: list[types.Part], *, dirty: bool = False) -> dict:
    """Un paragraphe pour deux beats narratifs, dans l'ordre."""
    story = [int(m.group(1)) for m in _BEAT_LINE.finditer(all_text(contents)) if not m.group(3)]
    paragraphs = []
    for i in range(0, len(story), 2):
        beat_ids = story[i : i + 2]
        text = f"The hero pushes forward through beats {beat_ids}. He refuses to give up."
        if dirty:
            text = f"In this panel, we see the hero pushing through beats {beat_ids}. Here, he refuses to give up."
        paragraphs.append({"text": text, "beat_ids": beat_ids, "emotion": EMOTIONS[i % len(EMOTIONS)]})
    return {"paragraphs": paragraphs}


def keyframes_payload(contents: list[types.Part]) -> dict:
    """La premiere case candidate de chaque paragraphe."""
    choices = []
    for m in _PARAGRAPH_LINE.finditer(all_text(contents)):
        candidates = [int(x) for x in m.group(2).split(",") if x.strip()]
        index = int(m.group(1))
        # Paragraphes impairs : la case cle est un impact (punch-in) ; 99 = numero invente, ignore.
        heavy = candidates[:1] + [99] if index % 2 == 1 else []
        choices.append({"paragraph_index": index, "key_panel_ids": candidates[:1], "action_heavy_ids": heavy})
    return {"choices": choices}


def default_responder(contents, call_no, config):
    schema = config.response_schema
    if schema is BeatBatch:
        return make_response(beats_payload(batch_ids_of(contents)))
    if schema is ScriptDraft:
        return make_response(script_payload(contents), thinking_tokens=7)
    if schema is KeyframeBatch:
        return make_response(keyframes_payload(contents))
    raise AssertionError(f"schema inattendu {schema}")


class FakeModels:
    def __init__(self, responder):
        self.responder = responder
        self.calls: list[dict] = []

    def generate_content(self, *, model, contents, config):
        self.calls.append({"model": model, "contents": contents, "config": config})
        return self.responder(contents, len(self.calls), config)


class FakeClient:
    def __init__(self, responder=default_responder):
        self.models = FakeModels(responder)

    def calls_for(self, schema) -> list[dict]:
        return [c for c in self.models.calls if c["config"].response_schema is schema]


@pytest.fixture
def sleep_calls(monkeypatch) -> list[float]:
    calls: list[float] = []
    monkeypatch.setattr(analyzer_mod, "_sleep", lambda s: calls.append(s))
    return calls


# --- Lots & images ------------------------------------------------------------------
def test_make_batches_balanced_and_bounded() -> None:
    batches = make_batches(_panels(20), 12)
    assert [len(b) for b in batches] == [10, 10]
    assert [len(b) for b in make_batches(_panels(31), 15)] == [11, 10, 10]
    assert make_batches([], 12) == []
    with pytest.raises(ValueError):
        make_batches(_panels(3), MAX_BATCH_SIZE + 1)
    with pytest.raises(ValueError):
        GeminiAnalyzer(client=FakeClient(), batch_size=16)


def test_panel_caption_mentions_split_halves() -> None:
    from src.modules.analyzer import panel_caption

    assert panel_caption(_panel(3)) == "Panel 3 (panel, 120x80 px):"
    assert panel_caption(_panel(9, kind="scroll_vertical"), n_slices=2) == (
        "Panel 9 (giant panel to scroll, 120x80 px, sent in 2 parts from top to bottom):"
    )
    top = _panel(4).model_copy(update={"part": "top", "source_index": 2})
    middle = _panel(5).model_copy(update={"part": "middle", "source_index": 2})
    bottom = _panel(6).model_copy(update={"part": "bottom", "source_index": 2})
    assert panel_caption(top) == "Panel 4 (top part of a tall drawing, continued in panel 5, 120x80 px):"
    assert panel_caption(middle) == (
        "Panel 5 (middle part of the tall drawing started in panel 4, continued in panel 6, 120x80 px):"
    )
    assert panel_caption(bottom) == "Panel 6 (bottom part of the tall drawing started above (panel 5), 120x80 px):"


def test_encode_panel_image_slices_and_width_cap() -> None:
    (data, mime), = encode_panel_image(_panel(0, height=300, width=200))
    assert mime == "image/jpeg" and _jpeg_size(data) == (200, 300)
    (data, _), = encode_panel_image(_panel(1, height=500, width=2000))
    assert _jpeg_size(data) == (DEFAULT_MAX_IMAGE_WIDTH, round(500 * DEFAULT_MAX_IMAGE_WIDTH / 2000))
    slices = encode_panel_image(_panel(3, height=20000, width=800, kind="scroll_vertical"))
    assert len(slices) == MAX_SLICES_PER_PANEL
    assert sum(_jpeg_size(d)[1] for d, _ in slices) == MAX_SLICES_PER_PANEL * DEFAULT_MAX_SLICE_HEIGHT
    with pytest.raises(ValueError):
        encode_panel_image(_panel(0), max_width=0)


# --- Prompts -------------------------------------------------------------------------
def test_prompts_are_ascii_and_well_formed() -> None:
    analyzer = GeminiAnalyzer(client=FakeClient())
    meta = ChapterMeta(url="u", final_url="u", series_title="Tower of God", episode_title="Ep. 0",
                       title_no=95, episode_no=1, image_urls=["a"])
    batch = _panels(3)
    batch[2] = _panel(2, height=5000, width=800, kind="scroll_vertical")
    context = [BeatDraft(panel_ids=[9], summary="Bam enters the tower.", characters=["Bam"], dialogue=[])]
    parts = analyzer.build_beats_contents(batch, context, meta)
    assert len([p for p in parts if p.inline_data is not None]) == 1 + 1 + 3
    assert batch_ids_of(parts) == [0, 1, 2]
    header = parts[0].text
    assert "Tower of God" in header and "Bam enters the tower." in header and "0 to 2" in header
    assert parts[-1].text.startswith("Extract the story beats of these 3 panel(s) (0, 1, 2)")

    beats = [Beat(index=0, panel_ids=[0, 1], summary="Hero wakes.", characters=["Kang"], dialogue=["Wait a sec"]),
             Beat(index=1, panel_ids=[2], summary="Title card.", characters=[], dialogue=[], is_filler=True)]
    script = analyzer.build_script_contents(beats, meta)[0].text
    assert "Beat 0 (panels 0, 1): Hero wakes. | Characters: Kang | Dialogue: \"Wait a sec\"" in script
    assert "Beat 1 (panels 2) [FILLER - do not narrate]: Title card." in script
    assert f"about {target_script_words(beats)} words" in script and "English" in script

    group = [(0, "The hero wakes.", [0, 1]), (1, "He fights.", [2])]
    kf = analyzer.build_keyframes_contents(group, {p.index: p for p in batch}, meta)
    text = all_text(kf)
    assert "Paragraph 0 (candidate panels: 0, 1):\nThe hero wakes." in text
    assert "Paragraph 1 (candidate panels: 2):\nHe fights." in text
    assert batch_ids_of(kf) == [0, 1, 2]
    # Etape 2 : vignettes reduites (largeur <= 640, case geante en 2 tranches max).
    kf_images = [p for p in kf if p.inline_data is not None]
    assert len(kf_images) == 1 + 1 + 2
    assert all(_jpeg_size(p.inline_data.data)[0] <= 640 for p in kf_images)
    assert kf[-1].text.startswith("Return exactly one choice per paragraph (0, 1)")
    for prompt in (analyzer_mod.BEATS_SYSTEM_INSTRUCTION, analyzer.script_config(beats).system_instruction,
                   analyzer.keyframes_config.system_instruction, text, header, script):
        prompt.encode("ascii")
    assert "STRICTLY FORBIDDEN" in analyzer.script_config(beats).system_instruction
    assert GeminiAnalyzer(client=FakeClient()).language == "en"


def test_target_script_words_and_paragraphs_bounds() -> None:
    from src.modules.analyzer import target_paragraphs

    few = [BeatDraft(panel_ids=[i], summary="s", characters=[], dialogue=[]) for i in range(3)]
    assert target_script_words(few) == 250 and target_paragraphs(few) == 4
    many = [BeatDraft(panel_ids=[i, 100 + i], summary="s", characters=[], dialogue=[]) for i in range(100)]
    assert target_script_words(many) == 1500 and target_paragraphs(many) == 40
    # 40 beats dont 20 de remplissage, 3 cases chacun : 60 cases narratives x 11 mots, 10 paragraphes.
    mid = [BeatDraft(panel_ids=[3 * i, 3 * i + 1, 3 * i + 2], summary="s", characters=[], dialogue=[], is_filler=(i % 2 == 0)) for i in range(40)]
    assert target_script_words(mid) == 60 * 11 and target_paragraphs(mid) == 10


# --- Parsing ------------------------------------------------------------------------
def test_parse_scene_json_is_lenient(caplog) -> None:
    fenced = '```json\n{"scenes": [{"panel_ids": ["1", 2], "narration": "x", "emotion": "JOY"}]}\n```'
    with caplog.at_level(logging.WARNING, logger="src.modules.analyzer"):
        scenes = parse_scene_json(fenced)
    assert scenes[0].panel_ids == [1, 2] and scenes[0].emotion == "happy"
    scenes = parse_scene_json('[{"panel_ids": [3], "narration": "y", "emotion": "banana"}]')
    assert scenes[0].emotion == "neutral" and "banana" in caplog.text
    noisy = 'Result:\n{"scenes": [{"panel_ids": [4], "narration": "z", "emotion": "calm"}]}\nEnd.'
    assert parse_scene_json(noisy)[0].panel_ids == [4]
    for bad in ("not json", '{"foo": 1}', '{"scenes": [{"panel_ids": "abc", "narration": "x", "emotion": "calm"}]}'):
        with pytest.raises(InvalidResponseError):
            parse_scene_json(bad)
    assert coerce_emotion("battle") == "action" and coerce_emotion(None) == "neutral"
    assert coerce_bool("TRUE") and not coerce_bool("no")


def test_parse_beats_script_keyframes_tolerant() -> None:
    from src.modules.analyzer import parse_beats, parse_keyframes, parse_script

    beats = parse_beats(make_response({"beats": [{"panel_ids": [0, 1], "summary": "s", "characters": "Kang", "dialogue": None, "is_filler": "yes"}]}))
    assert beats[0].characters == ["Kang"] and beats[0].dialogue == [] and beats[0].is_filler is True
    paragraphs = parse_script(make_response([{"narration": "text", "beat_ids": [0], "emotion": "Joy"}]))
    assert paragraphs[0].text == "text" and paragraphs[0].emotion == "happy"
    choices = parse_keyframes(make_response({"choices": [{"paragraph_index": 0, "panel_ids": [3]}]}))
    assert choices[0].key_panel_ids == [3] and choices[0].action_heavy_ids == []
    choices = parse_keyframes(make_response({"choices": [{"paragraph_index": 1, "key_panel_ids": [3, 4], "action_heavy_ids": ["4"]}]}))
    assert choices[0].action_heavy_ids == [4]
    with pytest.raises(InvalidResponseError):
        parse_beats(make_response({"beats": "nope"}))


def test_parse_response_reports_blocks_and_finish_reasons() -> None:
    parsed = SceneBatch(scenes=[SceneDraft(panel_ids=[0], narration="p", emotion="calm")])
    response = make_response({"scenes": []})
    response.parsed = parsed
    assert [s.narration for s in parse_response(response)] == ["p"]
    blocked = types.GenerateContentResponse(
        prompt_feedback=types.GenerateContentResponsePromptFeedback(block_reason=types.BlockedReason.PROHIBITED_CONTENT)
    )
    with pytest.raises(AnalyzerError, match="PROHIBITED_CONTENT") as info:
        parse_response(blocked)
    assert not isinstance(info.value, InvalidResponseError)
    refused = types.GenerateContentResponse(candidates=[types.Candidate(finish_reason=types.FinishReason.SAFETY)])
    with pytest.raises(AnalyzerError, match="SAFETY"):
        parse_response(refused)
    truncated = types.GenerateContentResponse(candidates=[types.Candidate(finish_reason=types.FinishReason.MAX_TOKENS)])
    with pytest.raises(InvalidResponseError, match="MAX_TOKENS"):
        parse_response(truncated)


# --- Normalisation ----------------------------------------------------------------------
def test_normalize_scenes_partition(caplog) -> None:
    drafts = [
        SceneDraft(panel_ids=[4, 5], narration="  second ", emotion="calm"),
        SceneDraft(panel_ids=[1, 0, 99], narration="first", emotion="action"),
        SceneDraft(panel_ids=[1], narration="dup", emotion="sad"),
        SceneDraft(panel_ids=[7], narration="   ", emotion="sad"),
    ]
    with caplog.at_level(logging.WARNING, logger="src.modules.analyzer"):
        scenes = normalize_scenes(drafts, list(range(8)))
    assert [s.panel_ids for s in scenes] == [[0, 1, 2, 3], [4, 5, 6, 7]]
    assert [s.narration for s in scenes] == ["first", "second"]
    assert "hors de l'ensemble" in caplog.text and "deja attribue" in caplog.text
    interleaved = [SceneDraft(panel_ids=[10, 12], narration="a", emotion="calm"), SceneDraft(panel_ids=[11, 13], narration="b", emotion="calm")]
    assert [s.panel_ids for s in normalize_scenes(interleaved, [10, 11, 12, 13])] == [[10], [11, 12, 13]]
    with pytest.raises(InvalidResponseError):
        normalize_scenes([SceneDraft(panel_ids=[9], narration="n", emotion="calm")], [0, 1])


def test_find_and_scrub_forbidden() -> None:
    dirty = "In this panel, we see the hero. Here, the scene shows a dragon. The camera zooms in on a close-up."
    hits = find_forbidden(dirty)
    assert any("In this panel" in h for h in hits) and any("we see" in h for h in hits)
    assert any(h.lower().startswith("the scene shows") for h in hits) and any("camera" in h for h in hits)
    assert find_forbidden("Kang draws his sword and charges at the dragon.") == []
    assert find_forbidden("Sur cette case, on voit le heros.") != []
    cleaned = scrub_forbidden("In this panel, we see the hero. Here, he refuses to give up.")
    assert cleaned == "The hero. He refuses to give up."
    assert find_forbidden(cleaned) == []


def test_script_conventions_hook_and_cta(caplog) -> None:
    from src.modules.analyzer import (
        CTA_TEXTS,
        cta_paragraph_index,
        enforce_script_conventions,
        generic_intro,
        split_sentences,
    )

    assert generic_intro("In this chapter, humanity falls. Then it rises.") == "In this chapter, humanity falls."
    assert generic_intro("Welcome back everyone! The tower awaits.") == "Welcome back everyone!"
    assert generic_intro("Let's dive into the prologue.") == "Let's dive into the prologue."
    assert generic_intro("Humanity has one hour left. Nobody moves.") is None
    assert split_sentences("A first sentence. A second one! And a third?") == ["A first sentence.", "A second one!", "And a third?"]
    assert [cta_paragraph_index(n) for n in (1, 2, 3, 4, 5, 8, 12)] == [0, 0, 1, 2, 2, 3, 4]

    def para(text: str, i: int) -> ParagraphDraft:
        return ParagraphDraft(text=text, beat_ids=[i], emotion="calm")

    paragraphs = [
        para("In this chapter, the tower opens. Humanity has one hour left.", 0),
        para("The chat panics. Subscribe to see more of this! A stranger appears.", 1),
        para("The dragon roars. The stranger smiles.", 2),
        para("Fire falls. He raises his hand.", 3),
        para("He names himself. The dragon trembles.", 4),
    ]
    with caplog.at_level(logging.WARNING, logger="src.modules.analyzer"):
        result = enforce_script_conventions(paragraphs, language="en")
    assert result[0].text == "Humanity has one hour left."  # intro generique retiree
    assert "Introduction generique supprimee" in caplog.text
    assert result[1].text == "The chat panics. A stranger appears."  # CTA du modele retire
    assert result[2].text == f"The dragon roars. The stranger smiles. {CTA_TEXTS['en']}"  # CTA canonique en 3e paragraphe
    assert sum(text.count("subscribe") for text in (p.text.lower() for p in result)) == 1
    assert [p.beat_ids for p in result] == [[0], [1], [2], [3], [4]]  # brouillons d'entree intacts
    assert paragraphs[0].text.startswith("In this chapter")

    # Script court : le CTA va dans le paragraphe du milieu, avant sa derniere phrase s'il est le dernier.
    short = [para("Danger looms. The end nears.", 0)]
    assert enforce_script_conventions(short, language="fr")[0].text == f"Danger looms. {CTA_TEXTS['fr']} The end nears."
    # CTA personnalise, ou aucun.
    custom = enforce_script_conventions(paragraphs, language="en", cta_text="Like and subscribe!")
    assert custom[2].text.endswith("Like and subscribe!")
    none = enforce_script_conventions(paragraphs, language="en", cta_text="")
    assert not any("subscribe" in p.text.lower() for p in none)
    # Intro generique seule phrase : conservee (avertissement).
    lone = enforce_script_conventions([para("Welcome back to the tower.", 0)], language="en", cta_text="")
    assert lone[0].text == "Welcome back to the tower."


def test_analyze_panels_inserts_cta_and_retries_generic_intro() -> None:
    from src.modules.analyzer import CTA_TEXTS

    state = {"script_calls": 0}

    def responder(contents, call_no, config):
        if config.response_schema is ScriptDraft:
            state["script_calls"] += 1
            payload = script_payload(contents)
            if state["script_calls"] == 1:
                payload["paragraphs"][0]["text"] = "In this chapter, everything changes. " + payload["paragraphs"][0]["text"]
            return make_response(payload)
        return default_responder(contents, call_no, config)

    client = FakeClient(responder)
    analysis = GeminiAnalyzer(client=client).analyze_panels(_panels(20))
    assert state["script_calls"] == 2  # intro generique -> regeneration avec rappel
    assert "generic opening" in client.calls_for(ScriptDraft)[1]["contents"][-1].text
    texts = [s.narration for s in analysis.scenes]
    assert not texts[0].startswith("In this chapter")
    assert sum(CTA_TEXTS["en"] in t for t in texts) == 1 and CTA_TEXTS["en"] in texts[2]
    assert not any(CTA_TEXTS["en"] in t for t in texts[:2])


def test_normalize_script_covers_story_beats_and_skips_filler(caplog) -> None:
    beats = [Beat(index=i, panel_ids=[i], summary=f"b{i}", characters=[], dialogue=[], is_filler=(i == 3)) for i in range(6)]
    paragraphs = [
        ParagraphDraft(text="Second part.", beat_ids=[4, 5], emotion="epic"),
        ParagraphDraft(text="First part.", beat_ids=[0, 3, 42], emotion="calm"),  # 3 = filler, 42 inconnu
    ]
    with caplog.at_level(logging.WARNING, logger="src.modules.analyzer"):
        result = normalize_script(paragraphs, beats)
    assert [(p.text, p.beat_ids) for p in result] == [("First part.", [0, 1, 2]), ("Second part.", [4, 5])]
    assert "hors de l'ensemble" in caplog.text
    with pytest.raises(InvalidResponseError):
        normalize_script([ParagraphDraft(text="   ", beat_ids=[0], emotion="calm")], beats)


def test_normalize_keyframes_rules(caplog) -> None:
    group = [(0, "a", [0, 1, 2, 3, 4, 5]), (1, "b", [6, 7]), (2, "c", [8])]
    heights = {i: 100 + 10 * i for i in range(9)}
    heights[2] = 900
    choices = [
        KeyframeChoice(paragraph_index=0, key_panel_ids=[5, 4, 3, 2, 1, 99, 7], action_heavy_ids=[]),  # 7 non candidate, 99 inconnu, 5 > max
        KeyframeChoice(paragraph_index=1, key_panel_ids=[7, 7], action_heavy_ids=[]),
        # paragraphe 2 : aucun choix -> repli sur la plus grande candidate
    ]
    used: set[int] = set()
    with caplog.at_level(logging.WARNING, logger="src.modules.analyzer"):
        result = normalize_keyframes(choices, group, heights, used, max_key=MAX_KEY_PANELS_PER_PARAGRAPH)
    assert result[0] == [2, 3, 4, 5]  # les 4 plus grandes des 5 valides, en ordre de lecture
    assert result[1] == [7] and result[2] == [8]
    assert used == {2, 3, 4, 5, 7, 8}
    assert "repli sur [8]" in caplog.text and "non candidate ou deja utilisee" in caplog.text
    # Case deja utilisee par un paragraphe precedent : refusee, repli sur une autre candidate.
    result = normalize_keyframes([KeyframeChoice(paragraph_index=3, key_panel_ids=[7], action_heavy_ids=[])], [(3, "d", [7, 6])], heights, used)
    assert result[3] == [6]


def test_normalize_action_heavy_restricted_to_key_panels() -> None:
    from src.modules.analyzer import normalize_action_heavy

    choices = [
        KeyframeChoice(paragraph_index=0, key_panel_ids=[2, 3, 4], action_heavy_ids=[4, 2, 99, 7]),
        KeyframeChoice(paragraph_index=1, key_panel_ids=[7], action_heavy_ids=[]),
    ]
    keyframes = {0: [2, 3, 4], 1: [7], 2: [8]}
    # Ordre de lecture, cases inconnues (99) ou d'un autre paragraphe (7) ignorees, listes vides conservees.
    assert normalize_action_heavy(choices, keyframes) == {0: [2, 4], 1: [], 2: []}


# --- Pipeline en deux etapes avec client factice ------------------------------------------
def test_analyze_panels_two_steps(sleep_calls) -> None:
    client = FakeClient()
    analyzer = GeminiAnalyzer(client=client, model="fake-model", batch_size=12, delay_between_batches=0.5)
    meta = ChapterMeta(url="https://x/viewer?title_no=1&episode_no=2", final_url="f", series_title="S",
                       episode_title="E", title_no=1, episode_no=2, image_urls=["a"])
    panels = _panels(20)
    analysis = analyzer.analyze_panels(panels, meta)

    assert isinstance(analysis, ChapterAnalysis)
    beats_calls, script_calls, kf_calls = client.calls_for(BeatBatch), client.calls_for(ScriptDraft), client.calls_for(KeyframeBatch)
    assert len(beats_calls) == 2 and len(script_calls) == 1
    assert batch_ids_of(beats_calls[0]["contents"]) == list(range(10))
    assert batch_ids_of(beats_calls[1]["contents"]) == list(range(10, 20))
    assert "Context - previous beats" in beats_calls[1]["contents"][0].text
    assert "Context" not in beats_calls[0]["contents"][0].text
    # 10 beats -> 5 paragraphes de 2 beats (4 cases candidates chacun) -> groupes de <= 15 candidates.
    assert len(analysis.beats) == 10 and [b.panel_ids for b in analysis.beats][:2] == [[0, 1], [2, 3]]
    assert analysis.n_scenes == 5 and len(kf_calls) == 2
    assert [s.beat_ids for s in analysis.scenes] == [[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]]
    assert [s.panel_ids for s in analysis.scenes] == [[0], [4], [8], [12], [16]]  # premiere candidate
    assert analysis.n_key_panels == 5 and analysis.covered_panel_ids() == [0, 4, 8, 12, 16]
    # Paragraphes impairs marques action_heavy (le numero invente 99 est ecarte).
    assert [s.action_heavy_ids for s in analysis.scenes] == [[], [4], [], [12], []]
    assert "action_heavy_ids" in kf_calls[0]["config"].system_instruction
    assert all(find_forbidden(s.narration) == [] for s in analysis.scenes)
    assert analysis.n_batches == 5 == len(client.models.calls)
    assert (analysis.prompt_tokens, analysis.output_tokens, analysis.thinking_tokens) == (500, 200, 7)
    assert analysis.series_title == "S" and analysis.model == "fake-model" and analysis.language == "en"
    assert analysis.script_words > 0 and analysis.n_filler == 0
    assert sleep_calls == [0.5] * 4  # pause entre les appels, pas avant le premier
    config = script_calls[0]["config"]
    assert config.response_mime_type == "application/json" and config.response_schema is ScriptDraft
    assert config.automatic_function_calling.disable is True


def test_script_forbidden_phrases_trigger_retry_then_scrub() -> None:
    state = {"script_calls": 0}

    def responder(contents, call_no, config):
        if config.response_schema is ScriptDraft:
            state["script_calls"] += 1
            return make_response(script_payload(contents, dirty=(state["script_calls"] == 1)))
        return default_responder(contents, call_no, config)

    client = FakeClient(responder)
    analysis = GeminiAnalyzer(client=client).analyze_panels(_panels(8))
    assert state["script_calls"] == 2  # regenere une fois avec le rappel
    retry = client.calls_for(ScriptDraft)[1]["contents"]
    assert "broke the rules" in retry[-1].text and "In this panel" in retry[-1].text
    assert all(find_forbidden(s.narration) == [] for s in analysis.scenes)

    always_dirty = FakeClient(lambda c, n, cfg: make_response(script_payload(c, dirty=True)) if cfg.response_schema is ScriptDraft else default_responder(c, n, cfg))
    analysis = GeminiAnalyzer(client=always_dirty).analyze_panels(_panels(8))
    assert len(always_dirty.calls_for(ScriptDraft)) == 2
    assert all(find_forbidden(s.narration) == [] for s in analysis.scenes)  # nettoyage local
    assert analysis.scenes[0].narration.startswith("The hero pushing")


def test_filler_beats_are_never_narrated_nor_shown() -> None:
    def responder(contents, call_no, config):
        if config.response_schema is BeatBatch:
            ids = batch_ids_of(contents)
            payload = beats_payload(ids)
            payload["beats"][-1]["is_filler"] = True  # dernier beat du lot = credits
            return make_response(payload)
        return default_responder(contents, call_no, config)

    analysis = GeminiAnalyzer(client=FakeClient(responder), batch_size=6).analyze_panels(_panels(6))
    assert [b.is_filler for b in analysis.beats] == [False, False, True]
    assert all(4 not in s.panel_ids and 5 not in s.panel_ids for s in analysis.scenes)
    assert all(2 not in s.beat_ids for s in analysis.scenes)


def test_retry_and_error_paths(sleep_calls) -> None:
    def flaky(contents, call_no, config):
        if call_no == 1:
            raise errors.ServerError(503, {"error": {"message": "unavailable"}})
        if call_no == 2:
            raise httpx.ReadTimeout("timed out")
        if call_no == 3:
            return make_response("{not json")
        return default_responder(contents, call_no, config)

    client = FakeClient(flaky)
    analysis = GeminiAnalyzer(client=client, batch_size=5, backoff=2.0, delay_between_batches=0).analyze_panels(_panels(4))
    assert analysis.n_scenes >= 1 and len(client.models.calls) >= 4
    assert sleep_calls[:3] == [2.0, 4.0, 8.0]

    def bad_request(contents, call_no, config):
        raise errors.ClientError(400, {"error": {"message": "bad request"}})

    client = FakeClient(bad_request)
    with pytest.raises(AnalyzerError) as info:
        GeminiAnalyzer(client=client).analyze_panels(_panels(2))
    assert len(client.models.calls) == 1 and isinstance(info.value.__cause__, errors.ClientError)

    client = FakeClient(lambda c, n, cfg: make_response("garbage"))
    with pytest.raises(AnalyzerError, match="Abandon"):
        GeminiAnalyzer(client=client, max_retries=1).analyze_panels(_panels(2))
    assert len(client.models.calls) == 2

    def rate_limited(contents, call_no, config):
        if call_no == 1:
            raise errors.ClientError(429, {"error": {"message": "quota"}})
        if call_no == 2:
            raise errors.ClientError(429, {"error": {"message": "Quota exceeded. Please retry in 37.306310346s.",
                                                      "details": [{"retryDelay": "37s"}]}})
        return default_responder(contents, call_no, config)

    sleep_calls.clear()
    GeminiAnalyzer(client=FakeClient(rate_limited), backoff=1.0, delay_between_batches=0).analyze_panels(_panels(2))
    assert sleep_calls[0] == 10.0  # plancher sans indication du serveur
    assert sleep_calls[1] == pytest.approx(38.3, abs=0.01)  # delai suggere par Gemini + 1 s
    from src.modules.analyzer import suggested_retry_delay

    assert suggested_retry_delay(RuntimeError("no hint")) == 0.0

    analyzer = GeminiAnalyzer(client=FakeClient())
    with pytest.raises(ValueError):
        analyzer.analyze_panels([])
    with pytest.raises(ValueError, match="dupliques"):
        analyzer.analyze_panels([_panel(0), _panel(0)])

    # Quota journalier epuise : echec immediat avec un message d'aide, sans attente.
    from src.modules.analyzer import QuotaExhaustedError

    def daily_quota(contents, call_no, config):
        raise errors.ClientError(429, {"error": {"message": (
            "You exceeded your current quota. Quota exceeded for metric: generate_content_free_tier_requests, "
            "limit: 20, model: gemini-2.5-flash. Please retry in 58.39s."),
            "details": [{"violations": [{"quotaId": "GenerateRequestsPerDayPerProjectPerModel-FreeTier"}]}]}})

    sleep_calls.clear()
    client = FakeClient(daily_quota)
    with pytest.raises(QuotaExhaustedError, match="quota journalier"):
        GeminiAnalyzer(client=client, model="gemini-2.5-flash").analyze_panels(_panels(2))
    assert len(client.models.calls) == 1 and sleep_calls == []


def test_analysis_checkpoint_resumes_after_quota_failure(tmp_path, sleep_calls) -> None:
    """Une analyse coupee par le quota repart des lots deja obtenus, meme avec un autre modele."""
    from src.modules.analyzer import AnalysisCheckpoint, QuotaExhaustedError

    checkpoint = tmp_path / "analysis_checkpoint.json"
    panels = _panels(20)  # 2 lots de beats, 1 script, 2 groupes de cases cles = 5 appels
    reference = GeminiAnalyzer(client=FakeClient(), model="m1", batch_size=12).analyze_panels(panels)

    def dying(contents, call_no, config):
        if config.response_schema is ScriptDraft:  # 3e appel : quota journalier epuise
            raise errors.ClientError(429, {"error": {"message": "Quota exceeded", "details": [
                {"violations": [{"quotaId": "GenerateRequestsPerDayPerProjectPerModel-FreeTier"}]}]}})
        return default_responder(contents, call_no, config)

    first = FakeClient(dying)
    with pytest.raises(QuotaExhaustedError):
        GeminiAnalyzer(client=first, model="m1", batch_size=12).analyze_panels(panels, checkpoint=checkpoint)
    assert len(first.calls_for(BeatBatch)) == 2 and checkpoint.is_file()
    saved = json.loads(checkpoint.read_text(encoding="utf-8"))
    assert len(saved["beat_batches"]) == 2 and saved["paragraphs"] is None and saved["key"]["panel_ids"] == list(range(20))

    # Reprise avec un autre modele : plus aucun appel "beats", le resultat est identique.
    second = FakeClient()
    analysis = GeminiAnalyzer(client=second, model="m2", batch_size=12).analyze_panels(panels, checkpoint=checkpoint)
    assert second.calls_for(BeatBatch) == [] and len(second.calls_for(ScriptDraft)) == 1 and len(second.calls_for(KeyframeBatch)) == 2
    assert analysis.beats == reference.beats and analysis.scenes == reference.scenes and analysis.model == "m2"
    assert analysis.n_batches == 3  # appels reellement passes par ce modele
    assert not checkpoint.exists()  # supprime une fois l'analyse aboutie

    # Point de reprise d'une autre decoupe (cases differentes) : ignore, tout est redemande.
    checkpoint.write_text(json.dumps({**saved, "key": {**saved["key"], "panel_ids": [0, 1]}}), encoding="utf-8")
    third = FakeClient()
    GeminiAnalyzer(client=third, model="m2", batch_size=12).analyze_panels(panels, checkpoint=checkpoint)
    assert len(third.calls_for(BeatBatch)) == 2
    # Fichier illisible : ignore sans erreur.
    checkpoint.write_text("{not json", encoding="utf-8")
    ckpt = AnalysisCheckpoint(checkpoint, panel_ids=[0], batch_size=12, language="en")
    assert ckpt.data["beat_batches"] == [] and ckpt.beats_for(0, [0]) is None and ckpt.paragraphs() is None


def test_candidate_panels_capped_to_tallest() -> None:
    from src.modules.analyzer import MAX_CANDIDATES_PER_PARAGRAPH

    panels = [_panel(i, height=100 + 25 * i) for i in range(12)]
    beats = [Beat(index=0, panel_ids=list(range(12)), summary="s", characters=[], dialogue=[])]
    paragraph = ParagraphDraft(text="t", beat_ids=[0], emotion="calm")
    analyzer = GeminiAnalyzer(client=FakeClient())
    candidates = analyzer.candidate_panels(paragraph, beats, {p.index: p for p in panels})
    assert candidates == list(range(12 - MAX_CANDIDATES_PER_PARAGRAPH, 12))  # les 8 plus grandes, ordre de lecture
    # Deux paragraphes de 8 candidates tiennent dans un seul groupe (<= 15 images) : 1 appel a l'etape 2.
    client = FakeClient()
    two = [ParagraphDraft(text="a", beat_ids=[0], emotion="calm"), ParagraphDraft(text="b", beat_ids=[1], emotion="calm")]
    beats2 = [Beat(index=0, panel_ids=list(range(6)), summary="s", characters=[], dialogue=[]),
              Beat(index=1, panel_ids=list(range(6, 12)), summary="s", characters=[], dialogue=[])]
    scenes = GeminiAnalyzer(client=client).select_keyframes(two, beats2, panels)
    assert len(client.calls_for(KeyframeBatch)) == 1 and [s.panel_ids for s in scenes] == [[0], [6]]


def test_resolve_model_and_missing_key(monkeypatch, tmp_path) -> None:
    monkeypatch.delenv("GEMINI_MODEL", raising=False)
    assert resolve_model(None) == DEFAULT_MODEL
    monkeypatch.setenv("GEMINI_MODEL", "gemini-custom")
    assert resolve_model(None) == "gemini-custom" and resolve_model("explicit") == "explicit"
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.setattr(config_mod, "GEMINI_KEY_FILE", tmp_path / "missing.key")
    with pytest.raises(AnalyzerError, match="Aucune cle Gemini"):
        GeminiAnalyzer()
    key_file = tmp_path / ".gemini_key"
    key_file.write_text("test-key-000001\n", encoding="utf-8")
    monkeypatch.setattr(config_mod, "GEMINI_KEY_FILE", key_file)
    from src.utils.gemini_manager import GeminiManager

    monkeypatch.setattr(GeminiManager, "_default_client", staticmethod(lambda key: FakeClient()))
    analyzer = GeminiAnalyzer()
    assert analyzer.manager is not None and analyzer._client is None
    assert analyzer.manager.keys[0].label.endswith("(...0001)") and analyzer.model == "gemini-custom"


# --- Sorties disque ------------------------------------------------------------------------
def test_save_and_load_analysis_roundtrip(tmp_path) -> None:
    analysis = GeminiAnalyzer(client=FakeClient(), batch_size=4).analyze_panels(_panels(8))
    path = save_analysis(analysis, tmp_path / "out" / "scenes.json")
    raw = json.loads(path.read_text(encoding="utf-8"))
    assert raw["n_panels"] == 8 and len(raw["scenes"]) == 2 and len(raw["beats"]) == 4
    assert set(raw["scenes"][0]) == {"index", "panel_ids", "narration", "emotion", "is_filler", "beat_ids", "action_heavy_ids"}
    assert set(raw["beats"][0]) == {"index", "panel_ids", "summary", "characters", "dialogue", "is_filler"}
    assert load_analysis(path) == analysis
    report = format_scenes(analysis)
    report.encode("ascii")
    assert "2 cles montees, 4 beats" in report and "[  0] cases 0" in report


def test_load_panels_roundtrip(tmp_path) -> None:
    strip = make_synthetic_strip([(300, 40, "white"), (1600, 30, "black"), (200, 0, "white")], seed=5)
    panels = slice_panels(strip)
    save_panels(panels, tmp_path)
    assert load_panels(tmp_path) == panels
    with pytest.raises(FileNotFoundError):
        load_panels(tmp_path / "missing")


def test_response_schemas_are_simple() -> None:
    for model_cls, list_key in ((BeatBatch, "beats"), (ScriptDraft, "paragraphs"), (KeyframeBatch, "choices")):
        schema = model_cls.model_json_schema()
        item = next(iter(schema["$defs"].values()))
        assert list_key in schema["properties"]
        assert "minItems" not in json.dumps(schema) and "maxItems" not in json.dumps(schema)
        assert set(item["required"]) >= {k for k in item["properties"] if k != "is_filler"}
    assert Scene(index=0, panel_ids=[1], narration="n", emotion="calm").beat_ids == []


# --- Gestionnaire multi-cles ------------------------------------------------------------------
def test_analyzer_uses_shared_manager_with_key_rotation_and_batch_delay(sleep_calls) -> None:
    from src.modules.analyzer import BATCH_DELAY_S
    from src.utils.gemini_manager import GeminiManager

    log: list[tuple[str, str]] = []

    def factory(key: str):
        def responder(contents, call_no, config):
            log.append((key, "call"))
            if key == "AIzaSyKEY-ONE-0001":
                raise errors.ClientError(429, {"error": {"message": "limit: 20, model: m1", "details": [
                    {"violations": [{"quotaId": "GenerateRequestsPerDayPerProjectPerModel-FreeTier"}]}]}})
            return default_responder(contents, call_no, config)

        return FakeClient(responder)

    manager = GeminiManager(["AIzaSyKEY-ONE-0001", "AIzaSyKEY-TWO-0002"], models=["m1", "m2"], client_factory=factory, sleep=lambda s: None)
    analyzer = GeminiAnalyzer(manager=manager, batch_size=12)
    assert analyzer.manager is manager and analyzer.model == "m1" and analyzer.delay_between_batches == BATCH_DELAY_S
    analysis = analyzer.analyze_panels(_panels(8))
    assert analysis.n_scenes >= 1 and analysis.model == "m1"
    # La premiere cle a epuise son quota journalier des le premier appel : tout passe par la deuxieme.
    assert log[0][0] == "AIzaSyKEY-ONE-0001" and all(k == "AIzaSyKEY-TWO-0002" for k, _ in log[1:])
    assert manager.status()["rotations"] == 1 and analysis.n_batches == analyzer.n_calls
    # Delai force de 2,5 s entre deux envois d'un meme chapitre (pas avant le premier).
    assert sleep_calls == [BATCH_DELAY_S] * (analyzer.n_calls - 1)

    # Quota epuise partout : QuotaExhaustedError de l'analyzer (sous-classe d'AnalyzerError).
    from src.modules.analyzer import QuotaExhaustedError

    def dead_factory(key: str):
        return FakeClient(lambda c, n, cfg: (_ for _ in ()).throw(errors.ClientError(429, {"error": {"message": "x", "details": [
            {"violations": [{"quotaId": "GenerateRequestsPerDayPerProjectPerModel-FreeTier"}]}]}})))

    dead = GeminiManager(["AIzaSyKEY-ONE-0001"], models=["m1"], client_factory=dead_factory)
    with pytest.raises(QuotaExhaustedError, match="toutes les cles"):
        GeminiAnalyzer(manager=dead).analyze_panels(_panels(2))

    # Reponse invalide : reessayee par l'analyzer (pas par le gestionnaire), puis abandon.
    garbage = GeminiManager(["AIzaSyKEY-ONE-0001"], models=["m1"], client_factory=lambda k: FakeClient(lambda c, n, cfg: make_response("garbage")))
    with pytest.raises(AnalyzerError, match="Abandon"):
        GeminiAnalyzer(manager=garbage, max_retries=1).analyze_panels(_panels(2))
    assert garbage.status()["calls"] == 2


# --- Mode « une requete » --------------------------------------------------------------------
def recap_payload(contents, *, dirty: bool = False, invalid: bool = False) -> dict:
    """Un paragraphe pour 4 cases, la premiere case de chaque groupe en case cle."""
    ids = batch_ids_of(contents)
    paragraphs = []
    # Paragraphes assez longs pour passer le garde-fou de longueur (SCRIPT_MIN_LENGTH_RATIO).
    body = (
        "He steps forward through the ruined courtyard, blade low, breathing hard, while the guards "
        "scatter behind him and the old chief watches from the gate without saying a word about it."
    )
    for i in range(0, len(ids), 4):
        group = ids[i : i + 4]
        text = f"The lord raises his blade over panels {group}. He refuses to yield. {body}"
        if dirty:
            text = f"In this panel, we see the lord over {group}. Here, he refuses to yield. {body}"
        keys = [group[0]] + ([999] if invalid else [])
        paragraphs.append({
            "text": text, "emotion": EMOTIONS[i % len(EMOTIONS)],
            "key_panel_ids": keys, "action_heavy_ids": [group[0]] if i % 8 == 0 else [],
        })
    return {"paragraphs": paragraphs}


def single_call_responder(contents, call_no, config):
    from src.models.scene import RecapDraft

    assert config.response_schema is RecapDraft
    return make_response(recap_payload(contents), thinking_tokens=11)


def test_single_call_produces_the_same_shape_in_one_request() -> None:
    client = FakeClient(single_call_responder)
    analyzer = GeminiAnalyzer(client=client, delay_between_batches=0)
    panels = _panels(24)
    analysis = analyzer.analyze_panels_single_call(panels, ChapterMeta(
        url="https://x/viewer?title_no=1&episode_no=2", final_url="f", series_title="S", episode_title="E",
        title_no=1, episode_no=2, image_urls=["a"],
    ))
    # Un seul appel, toutes les images du chapitre dedans.
    assert len(client.models.calls) == 1 and analysis.n_batches == 1
    sent = batch_ids_of(client.models.calls[0]["contents"])
    assert sent == list(range(24))
    assert analysis.n_scenes == 6 and analysis.beats == []  # pas d'etape beats
    assert [s.panel_ids for s in analysis.scenes] == [[0], [4], [8], [12], [16], [20]]
    assert analysis.covered_panel_ids() == [0, 4, 8, 12, 16, 20]
    assert all(find_forbidden(s.narration) == [] for s in analysis.scenes)
    assert analysis.scenes[0].action_heavy_ids == [0] and analysis.scenes[1].action_heavy_ids == []
    assert (analysis.prompt_tokens, analysis.output_tokens, analysis.thinking_tokens) == (100, 40, 11)
    # Les conventions du script sont appliquees comme en mode deux etapes (un seul CTA).
    from src.modules.analyzer import CTA_PATTERN

    assert sum(1 for s in analysis.scenes if CTA_PATTERN.search(s.narration)) == 1
    # Le prompt annonce la cible de longueur et la plage de numeros valides.
    prompt = all_text(client.models.calls[0]["contents"])
    assert "Target length: about" in prompt and "numbered 0 to 23" in prompt
    instruction = client.models.calls[0]["config"].system_instruction
    assert "key_panel_ids" in instruction and "action_heavy_ids" in instruction and "HOOK" in instruction


def test_single_call_targets_and_normalisation() -> None:
    from src.modules.analyzer import normalize_recap, single_call_targets
    from src.models.scene import RecapParagraph

    words, paragraphs = single_call_targets(_panels(176))
    assert words == 1500 and paragraphs == 25  # bornes du barème, ~1 paragraphe pour 7 cases
    assert single_call_targets(_panels(10)) == (250, 4)  # planchers

    panels = _panels(10)
    drafts = [
        RecapParagraph(text="First.", emotion="calm", key_panel_ids=[0, 999, 1], action_heavy_ids=[1, 999]),
        RecapParagraph(text="Second.", emotion="action", key_panel_ids=[1, 2], action_heavy_ids=[]),  # 1 deja pris
        RecapParagraph(text="   ", emotion="calm", key_panel_ids=[3], action_heavy_ids=[]),  # texte vide
        RecapParagraph(text="Fourth.", emotion="weird", key_panel_ids=[], action_heavy_ids=[]),  # repli
    ]
    result = normalize_recap(drafts, panels, max_key=4)
    # 999 ignore ; la case 1 n'est pas reutilisee ; le paragraphe vide saute ; le dernier
    # se replie sur la plus grande case encore libre (les hauteurs de _panels cyclent, 3 = 110 px).
    assert [keys for _, keys, _ in result] == [[0, 1], [2], [3]]
    assert [heavy for _, _, heavy in result] == [[1], [], []]
    assert [d.emotion for d, _, _ in result] == ["calm", "action", "neutral"]  # emotion inconnue normalisee
    assert all(d.beat_ids == [] for d, _, _ in result)
    # Trop de cases : on garde les plus grandes, en ordre de lecture.
    tall = [_panel(i, height=100 + 10 * i) for i in range(6)]
    single = [RecapParagraph(text="One.", emotion="calm", key_panel_ids=[0, 1, 2, 3, 4, 5], action_heavy_ids=[])]
    assert normalize_recap(single, tall, max_key=2)[0][1] == [4, 5]
    with pytest.raises(InvalidResponseError, match="Aucun paragraphe"):
        normalize_recap([RecapParagraph(text=" ", emotion="calm", key_panel_ids=[], action_heavy_ids=[])], panels)


def test_normalize_recap_restores_reading_order() -> None:
    """Une case en avance est retiree, pas la suite correcte qui la suit."""
    from src.modules.analyzer import normalize_recap
    from src.models.scene import RecapParagraph

    para = lambda text, keys, heavy=(): RecapParagraph(  # noqa: E731
        text=text, emotion="calm", key_panel_ids=list(keys), action_heavy_ids=list(heavy)
    )
    # Cas observe sur un vrai chapitre : la case 8 annoncee au paragraphe 2 alors que le
    # paragraphe 3 raconte les cases 4 a 7. Retirer 8 coute une case, retirer 4-7 en couterait
    # quatre : c'est donc 8 qui part, et son paragraphe garde la case 3.
    panels = [_panel(i, height=100) for i in range(12)]
    result = normalize_recap(
        [para("A.", [0, 1, 2]), para("B.", [3, 8], heavy=[8]), para("C.", [4, 5, 6, 7])], panels
    )
    assert [keys for _, keys, _ in result] == [[0, 1, 2], [3], [4, 5, 6, 7]]
    assert [heavy for _, _, heavy in result] == [[], [], []]  # 8 retiree l'est aussi des action_heavy

    # Un paragraphe vide apres remise en ordre est recase dans la fenetre libre entre ses
    # voisins (ici la seule case < 1), pas sur une case volee a la suite du chapitre.
    tall = [_panel(i, height=100 + 10 * i) for i in range(10)]
    result = normalize_recap([para("A.", [5]), para("B.", [1]), para("C.", [9])], tall)
    assert [keys for _, keys, _ in result] == [[0], [1], [9]]

    # Fenetre libre inexistante : on accepte le recul ponctuel plutot que de narrer un
    # paragraphe sans image a l'ecran.
    result = normalize_recap([para("A.", [0]), para("B.", [5]), para("C.", [1]), para("D.", [6])], tall)
    assert [keys for _, keys, _ in result] == [[0], [5], [1], [6]]
    assert all(keys for _, keys, _ in result)


def test_single_call_retries_on_forbidden_phrases_then_scrubs() -> None:
    state = {"calls": 0}

    def responder(contents, call_no, config):
        state["calls"] += 1
        return make_response(recap_payload(contents, dirty=state["calls"] == 1))

    client = FakeClient(responder)
    analysis = GeminiAnalyzer(client=client, delay_between_batches=0).analyze_panels_single_call(_panels(8))
    assert state["calls"] == 2  # regenere une fois avec le rappel
    retry = client.models.calls[1]["contents"]
    assert "broke the rules" in retry[-1].text and "In this panel" in retry[-1].text
    assert all(find_forbidden(s.narration) == [] for s in analysis.scenes)

    # Toujours fautif : nettoyage local, aucun echec.
    always = FakeClient(lambda c, n, cfg: make_response(recap_payload(c, dirty=True)))
    analysis = GeminiAnalyzer(client=always, delay_between_batches=0).analyze_panels_single_call(_panels(8))
    assert len(always.models.calls) == 2
    assert all(find_forbidden(s.narration) == [] for s in analysis.scenes)


def test_single_call_regenerates_a_script_that_is_too_short() -> None:
    """Le mode une requete a tendance a trop resumer : un script trop court est redemande."""
    from src.modules.analyzer import SCRIPT_MIN_LENGTH_RATIO, single_call_targets

    assert SCRIPT_MIN_LENGTH_RATIO == 0.65
    target, _ = single_call_targets(_panels(40))
    state = {"calls": 0}

    def responder(contents, call_no, config):
        state["calls"] += 1
        ids = batch_ids_of(contents)
        filler = "word " * (5 if state["calls"] == 1 else 120)  # court, puis fourni
        paragraphs = [
            {"text": f"The lord fights on. {filler}".strip() + ".", "emotion": "action",
             "key_panel_ids": [ids[i]], "action_heavy_ids": []}
            for i in range(0, min(len(ids), 8))
        ]
        return make_response({"paragraphs": paragraphs})

    client = FakeClient(responder)
    analysis = GeminiAnalyzer(client=client, delay_between_batches=0).analyze_panels_single_call(_panels(40))
    assert state["calls"] == 2  # une seule reprise
    assert analysis.script_words >= SCRIPT_MIN_LENGTH_RATIO * target
    reminder = client.models.calls[1]["contents"][-1].text
    assert "far too short" in reminder and str(target) in reminder

    # Une reprise encore plus courte est ecartee : on garde la premiere reponse.
    def shrinking(contents, call_no, config):
        ids = batch_ids_of(contents)
        filler = "word " * (30 if call_no == 1 else 2)
        return make_response({"paragraphs": [
            {"text": f"The lord fights. {filler}".strip() + ".", "emotion": "calm",
             "key_panel_ids": [ids[0]], "action_heavy_ids": []},
        ]})

    shrink_client = FakeClient(shrinking)
    analysis = GeminiAnalyzer(client=shrink_client, delay_between_batches=0).analyze_panels_single_call(_panels(40))
    assert len(shrink_client.models.calls) == 2 and analysis.script_words > 25


def test_single_call_rejects_invalid_panel_numbers() -> None:
    client = FakeClient(lambda c, n, cfg: make_response(recap_payload(c, invalid=True)))
    analysis = GeminiAnalyzer(client=client, delay_between_batches=0).analyze_panels_single_call(_panels(8))
    # Le numero invente (999) est ecarte, le reste est conserve.
    assert [s.panel_ids for s in analysis.scenes] == [[0], [4]]
    with pytest.raises(ValueError, match="Aucune case"):
        GeminiAnalyzer(client=client).analyze_panels_single_call([])
    with pytest.raises(ValueError, match="dupliques"):
        GeminiAnalyzer(client=client).analyze_panels_single_call([_panel(0), _panel(0)])


def test_payload_bytes_measures_the_inline_limit() -> None:
    from src.modules.analyzer import MAX_INLINE_PAYLOAD_BYTES

    assert MAX_INLINE_PAYLOAD_BYTES == 16 * 1024 * 1024
    analyzer = GeminiAnalyzer(client=FakeClient())
    small = analyzer.payload_bytes(_panels(4))
    assert 0 < small < MAX_INLINE_PAYLOAD_BYTES
    assert analyzer.payload_bytes(_panels(8)) > small  # croit avec le nombre de cases


def test_keyframe_groups_run_in_parallel_with_identical_result(tmp_path) -> None:
    """L'etape 2 lance ses groupes en parallele : meme resultat qu'en serie, plus vite."""
    import threading

    from src.modules.analyzer import DEFAULT_KEYFRAME_WORKERS

    assert DEFAULT_KEYFRAME_WORKERS == 4
    peak = {"active": 0, "max": 0}
    lock = threading.Lock()

    def slow(contents, call_no, config):
        if config.response_schema is KeyframeBatch:
            with lock:
                peak["active"] += 1
                peak["max"] = max(peak["max"], peak["active"])
            time.sleep(0.05)
            with lock:
                peak["active"] -= 1
        return default_responder(contents, call_no, config)

    panels = _panels(40)  # 4 lots de beats -> 10 beats -> 5 paragraphes -> plusieurs groupes
    serial = GeminiAnalyzer(client=FakeClient(slow), batch_size=12, keyframe_workers=1, delay_between_batches=0)
    reference = serial.analyze_panels(panels)
    assert peak["max"] == 1

    peak["max"] = 0
    parallel_client = FakeClient(slow)
    parallel = GeminiAnalyzer(client=parallel_client, batch_size=12, keyframe_workers=4, delay_between_batches=0)
    analysis = parallel.analyze_panels(panels)
    assert peak["max"] >= 2  # plusieurs groupes en vol en meme temps
    assert [s.panel_ids for s in analysis.scenes] == [s.panel_ids for s in reference.scenes]
    assert [s.narration for s in analysis.scenes] == [s.narration for s in reference.scenes]
    assert [s.action_heavy_ids for s in analysis.scenes] == [s.action_heavy_ids for s in reference.scenes]
    assert analysis.n_batches == reference.n_batches


def test_keyframe_failure_keeps_the_checkpoint_prefix(tmp_path) -> None:
    """Un groupe en echec ne fait pas perdre les groupes deja obtenus (point de reprise)."""
    checkpoint = tmp_path / "cp.json"
    state = {"kf": 0}

    def dying(contents, call_no, config):
        if config.response_schema is KeyframeBatch:
            state["kf"] += 1
            if state["kf"] >= 2:
                raise errors.ClientError(400, {"error": {"message": "bad request"}})
        return default_responder(contents, call_no, config)

    analyzer = GeminiAnalyzer(client=FakeClient(dying), batch_size=12, keyframe_workers=4, delay_between_batches=0)
    with pytest.raises(AnalyzerError):
        analyzer.analyze_panels(_panels(40), checkpoint=checkpoint)
    saved = json.loads(checkpoint.read_text(encoding="utf-8"))
    assert saved["paragraphs"], "le script doit rester dans le point de reprise"
    # Le groupe reussi est conserve quel que soit son rang (stockage par index, pas par prefixe).
    assert len(saved["keyframe_groups"]) == 1
    assert all(entry["choices"] for entry in saved["keyframe_groups"].values())

    # Reprise : le groupe deja obtenu n'est pas redemande.
    resumed = GeminiAnalyzer(client=FakeClient(default_responder), batch_size=12, keyframe_workers=4, delay_between_batches=0)
    analysis = resumed.analyze_panels(_panels(40), checkpoint=checkpoint)
    assert analysis.n_scenes >= 1 and not checkpoint.exists()  # supprime une fois l'analyse aboutie


def test_checkpoint_reads_the_historic_list_format(tmp_path) -> None:
    """Un point de reprise ecrit par l'ancienne version (liste) reste exploitable."""
    from src.modules.analyzer import AnalysisCheckpoint

    path = tmp_path / "cp.json"
    path.write_text(json.dumps({
        "key": {"panel_ids": [0, 1], "batch_size": 12, "language": "en"},
        "beat_batches": [], "paragraphs": None,
        "keyframe_groups": [{"paragraphs": [0], "choices": [{"paragraph_index": 0, "key_panel_ids": [1], "action_heavy_ids": []}]}],
    }), encoding="utf-8")
    checkpoint = AnalysisCheckpoint(path, panel_ids=[0, 1], batch_size=12, language="en")
    choices = checkpoint.choices_for(0, [0])
    assert choices is not None and choices[0].key_panel_ids == [1]
    assert checkpoint.choices_for(1, [1]) is None


def test_analyzer_without_client_builds_manager_from_env(monkeypatch) -> None:
    from src.utils.gemini_manager import GeminiManager

    monkeypatch.setenv("GEMINI_API_KEYS", "AIzaSyKEY-ONE-0001,AIzaSyKEY-TWO-0002")
    monkeypatch.setattr(GeminiManager, "_default_client", staticmethod(lambda key: FakeClient()))
    analyzer = GeminiAnalyzer(model="gemini-3.7-flash")
    assert analyzer.manager is not None and analyzer.manager.models[0] == "gemini-3.7-flash"
    assert len(analyzer.manager.keys) == 2 and analyzer.model == "gemini-3.7-flash"
    monkeypatch.setenv("GEMINI_API_KEYS", "")
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.setattr(config_mod, "GEMINI_KEY_FILE", config_mod.PROJECT_ROOT / "nope.key")
    with pytest.raises(AnalyzerError, match="Aucune cle"):
        GeminiAnalyzer()
