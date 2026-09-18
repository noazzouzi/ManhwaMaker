"""Tests du generateur de miniatures (``src.modules.thumbnail``).

Aucun acces reseau : le gestionnaire Gemini et le backend d'images sont injectes.
"""

from __future__ import annotations

import io
import json

import numpy as np
import pytest
from PIL import Image

from src.models.scene import ChapterAnalysis, Scene
from src.models.thumbnail import MAX_HOOK_CHARS, MAX_HOOK_WORDS, ThumbnailBrief, ThumbnailDraft
from src.modules.thumbnail.analyzer import (
    ThumbnailAnalysisError,
    ThumbnailAnalyzer,
    build_prompt,
    normalize_brief,
    normalize_hook,
    parse_brief,
)
from src.modules.thumbnail.compositor import (
    SATURATION_BOOST,
    TEXT_ANGLE_RANGE,
    TEXT_FILL_WHITE,
    THUMBNAIL_HEIGHT,
    THUMBNAIL_WIDTH,
    ThumbnailCompositor,
    ThumbnailCompositorError,
    build_thumbnail,
    draw_arrow,
    fit_to_frame,
    render_hook,
    text_angle,
)
from src.modules.thumbnail.image_backends import (
    BASE_NEGATIVE,
    STYLE_PROMPT,
    GeminiImageBackend,
    ImageBackendError,
    LocalWebuiBackend,
    StabilityBackend,
    build_image_prompt,
    build_negative_prompt,
    resolve_backend,
    save_image,
)
from src.modules.thumbnail.pipeline import ThumbnailResult, ThumbnailStage


# --- doublures ---------------------------------------------------------------------------------
class FakeResponse:
    def __init__(self, *, parsed=None, text=None, image: bytes | None = None, mime: str = "image/png") -> None:
        self.parsed, self.text = parsed, text
        if image is None:
            self.candidates = []
        else:
            blob = type("Blob", (), {"data": image, "mime_type": mime})()
            part = type("Part", (), {"inline_data": blob})()
            content = type("Content", (), {"parts": [part]})()
            self.candidates = [type("Candidate", (), {"content": content})()]


class FakeManager:
    """Gestionnaire Gemini minimal : renvoie des reponses preparees."""

    def __init__(self, *responses) -> None:
        self.responses = list(responses)
        self.calls: list[dict] = []
        self.current_model = "modele-de-test"

    def generate(self, contents, config, *, label: str = ""):
        self.calls.append({"contents": contents, "config": config, "label": label})
        if not self.responses:
            raise AssertionError("Appel inattendu")
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


def _analysis() -> ChapterAnalysis:
    return ChapterAnalysis(
        series_title="Hidden Rank", episode_title="Ep. 1", source_url="https://exemple/ep1",
        model="fake", language="en", n_panels=10,
        scenes=[
            Scene(index=i, panel_ids=[i], emotion="calm", narration=f"Paragraph number {i} of the recap.")
            for i in range(8)
        ] + [Scene(index=8, panel_ids=[8], emotion="neutral", narration="Title card.", is_filler=True)],
    )


def _png(width: int = 1600, height: int = 900, color: int = 120) -> bytes:
    rng = np.random.default_rng(3)
    pixels = np.clip(rng.integers(color - 40, color + 40, (height, width, 3)), 0, 255).astype(np.uint8)
    buffer = io.BytesIO()
    Image.fromarray(pixels).save(buffer, format="PNG")
    return buffer.getvalue()


def _brief(**kwargs) -> ThumbnailBrief:
    base = {"scene_description": "A tiny swordsman faces a huge dragon", "hook_text": "WAKE UP",
            "arrow_position": "right", "subject_position": "left"}
    return ThumbnailBrief(**{**base, **kwargs})


# --- etape 1 : analyse -------------------------------------------------------------------------
def test_normalize_hook_bounds_words_and_characters() -> None:
    assert normalize_hook("wake up!") == "WAKE UP"
    assert normalize_hook('  "He lied."  ') == "HE LIED"
    # Au-dela de MAX_HOOK_WORDS mots, on tronque.
    assert normalize_hook("one two three four five").split() == ["ONE", "TWO", "THREE"]
    assert len(normalize_hook("one two three four five").split()) == MAX_HOOK_WORDS
    # Trop long en caracteres : on retire des mots plutot que de couper au milieu.
    long_hook = normalize_hook("extraordinary catastrophe imminent")
    assert len(long_hook) <= MAX_HOOK_CHARS and long_hook
    # Ponctuation interne et emojis retires.
    assert normalize_hook("he--lied") == "HE LIED"
    with pytest.raises(ThumbnailAnalysisError, match="vide"):
        normalize_hook("   ...   ")


def test_normalize_brief_puts_the_arrow_opposite_the_subject() -> None:
    # Le modele pose la fleche du meme cote que le sujet : elle le recouvrirait.
    draft = ThumbnailDraft(scene_description="A hero stands", hook_text="too late",
                           arrow_position="left", subject_position="left", avoid="blood")
    brief = normalize_brief(draft, source_url="u", model="m")
    assert brief.arrow_position == "right" and brief.subject_position == "left"
    assert brief.hook_text == "TOO LATE" and brief.avoid == "blood"
    assert brief.source_url == "u" and brief.model == "m"
    # Valeurs invalides : repli sur des positions coherentes.
    odd = ThumbnailDraft(scene_description="x", hook_text="hi", arrow_position="up", subject_position="nowhere", avoid="")
    assert normalize_brief(odd).subject_position == "center"
    assert normalize_brief(odd).arrow_position in ("left", "right")
    with pytest.raises(ThumbnailAnalysisError, match="Description"):
        normalize_brief(ThumbnailDraft(scene_description="  ", hook_text="hi", arrow_position="left",
                                       subject_position="right", avoid=""))


def test_build_prompt_keeps_opening_and_ending() -> None:
    prompt = build_prompt(_analysis(), context_paragraphs=4)
    assert "Hidden Rank" in prompt and "Ep. 1" in prompt
    assert "Opening:" in prompt and "Ending:" in prompt
    # Le remplissage n'est jamais envoye.
    assert "Title card" not in prompt
    # Debut et fin du script, pas le milieu.
    assert "Paragraph number 0" in prompt and "Paragraph number 7" in prompt
    assert "Paragraph number 4" not in prompt


def test_parse_brief_accepts_parsed_object_and_raw_json() -> None:
    draft = ThumbnailDraft(scene_description="A calm face in a burning city", hook_text="he smiles",
                           arrow_position="right", subject_position="center", avoid="gore")
    assert parse_brief(FakeResponse(parsed=draft)).hook_text == "HE SMILES"
    raw = json.dumps(draft.model_dump())
    assert parse_brief(FakeResponse(text=raw)).scene_description.startswith("A calm face")
    with pytest.raises(ThumbnailAnalysisError, match="vide"):
        parse_brief(FakeResponse())
    with pytest.raises(ThumbnailAnalysisError, match="invalide"):
        parse_brief(FakeResponse(text="{pas du json}"))


def test_analyzer_designs_from_the_script() -> None:
    draft = ThumbnailDraft(scene_description="A tiny hero against a huge beast", hook_text="one hit",
                           arrow_position="left", subject_position="left", avoid="")
    manager = FakeManager(FakeResponse(parsed=draft))
    brief = ThumbnailAnalyzer(manager).design(_analysis())
    assert brief.hook_text == "ONE HIT" and brief.model == "modele-de-test"
    assert brief.arrow_position == "right"  # replacee a l'oppose du sujet
    call = manager.calls[0]
    assert call["label"] == "miniature"
    assert call["config"].response_schema is ThumbnailDraft


# --- etape 2 : generation d'image --------------------------------------------------------------
def test_image_prompts_force_the_style_and_the_exclusions() -> None:
    prompt = build_image_prompt(_brief())
    assert STYLE_PROMPT in prompt and "16:9" in prompt
    assert "left third" in prompt  # la position du sujet est rappelee au generateur
    assert "centred" in build_image_prompt(_brief(subject_position="center"))
    negative = build_negative_prompt(_brief(avoid="blood, nudity."))
    assert negative.startswith(BASE_NEGATIVE) and negative.endswith("blood, nudity")
    assert build_negative_prompt(_brief(avoid="")) == BASE_NEGATIVE


def test_gemini_backend_extracts_image_bytes() -> None:
    data = _png(64, 36)
    backend = GeminiImageBackend(FakeManager(FakeResponse(image=data)))
    assert backend.generate("un prompt", negative="du texte") == data
    # Reponse sans image : erreur explicite, pas un plantage obscur.
    with pytest.raises(ImageBackendError, match="sans image"):
        GeminiImageBackend(FakeManager(FakeResponse(text="desole"))).generate("x")
    # Une partie non-image est ignoree.
    empty = FakeResponse(image=b"abc", mime="text/plain")
    with pytest.raises(ImageBackendError):
        GeminiImageBackend(FakeManager(empty)).generate("x")


def test_resolve_backend_and_save_image(tmp_path, monkeypatch) -> None:
    monkeypatch.delenv("THUMBNAIL_IMAGE_BACKEND", raising=False)
    assert isinstance(resolve_backend(), GeminiImageBackend)
    assert isinstance(resolve_backend("stability"), StabilityBackend)
    assert isinstance(resolve_backend("local"), LocalWebuiBackend)
    monkeypatch.setenv("THUMBNAIL_IMAGE_BACKEND", "local")
    assert isinstance(resolve_backend(), LocalWebuiBackend)
    with pytest.raises(ImageBackendError, match="inconnu"):
        resolve_backend("dall-e")
    # Ecriture : les octets doivent etre une image lisible.
    path = save_image(_png(32, 18), tmp_path / "sub" / "base.png")
    assert path.is_file() and Image.open(path).size == (32, 18)
    with pytest.raises(ImageBackendError, match="illisibles"):
        save_image(b"pas une image", tmp_path / "bad.png")
    with pytest.raises(ImageBackendError, match="Aucun octet"):
        save_image(b"", tmp_path / "empty.png")


# --- etape 3 : compositing ---------------------------------------------------------------------
def test_text_angle_is_deterministic_and_within_bounds() -> None:
    low, high = TEXT_ANGLE_RANGE
    for text in ("WAKE UP", "HE LIED", "TOO LATE", "ONE HIT"):
        angle = text_angle(text)
        assert low <= angle <= high
        assert angle == text_angle(text)  # reproductible : deux rendus identiques
    assert len({text_angle(t) for t in ("A", "B", "C", "D", "E")}) > 1  # mais variable


def test_drawn_arrow_and_frame_fitting() -> None:
    arrow = draw_arrow(200, 120)
    assert arrow.size == (200, 120) and arrow.mode == "RGBA"
    pixels = np.asarray(arrow)
    assert pixels[:, :, 3].max() == 255 and pixels[:, :, 3].min() == 0  # silhouette, fond transparent
    # La pointe est a droite : la moitie droite contient la partie la plus haute de la forme.
    opaque = pixels[:, :, 3] > 0
    assert opaque[:, :100].sum() > 0 and opaque[:, 100:].sum() > 0
    # Recadrage : le cadre est rempli exactement, quel que soit le ratio d'entree.
    for size in ((1600, 900), (1000, 1000), (900, 1600)):
        fitted = fit_to_frame(Image.new("RGB", size, (10, 20, 30)), 1280, 720)
        assert fitted.size == (1280, 720) and fitted.mode == "RGB"


def test_render_hook_fits_the_box() -> None:
    image = render_hook("WAKE UP", 900, 220)
    assert image.mode == "RGBA" and image.width <= 900 and image.height <= 220
    # Une accroche plus courte occupe davantage la largeur disponible (police plus grande).
    assert render_hook("NO", 900, 220).height >= image.height * 0.5
    with pytest.raises(ThumbnailCompositorError, match="vide"):
        render_hook("   ", 900, 220)
    with pytest.raises(ThumbnailCompositorError, match="impossible a loger"):
        render_hook("WAKE UP", 4, 4)


def test_build_thumbnail_end_to_end(tmp_path) -> None:
    base = tmp_path / "base.png"
    base.write_bytes(_png(1600, 900))
    out = build_thumbnail(base, "WAKE UP", "right", out_path=tmp_path / "thumb.jpg")
    assert out.is_file()
    with Image.open(out) as image:
        assert image.size == (THUMBNAIL_WIDTH, THUMBNAIL_HEIGHT) and image.mode == "RGB"
    assert out.stat().st_size < 2_000_000  # limite YouTube
    # Le jaune de l'accroche et de la fleche est bien present.
    pixels = np.asarray(Image.open(out).convert("RGB")).astype(int)
    yellow = (pixels[:, :, 0] > 200) & (pixels[:, :, 1] > 200) & (pixels[:, :, 2] < 90)
    assert yellow.sum() > 500
    # Accroche en blanc : plus de jaune que la fleche seule.
    white_out = build_thumbnail(base, "WAKE UP", "left", out_path=tmp_path / "w.jpg", fill=TEXT_FILL_WHITE)
    white_pixels = np.asarray(Image.open(white_out).convert("RGB")).astype(int)
    white_yellow = (white_pixels[:, :, 0] > 200) & (white_pixels[:, :, 1] > 200) & (white_pixels[:, :, 2] < 90)
    assert white_yellow.sum() < yellow.sum()
    with pytest.raises(ThumbnailCompositorError, match="illisible"):
        build_thumbnail(tmp_path / "absente.png", "X", "left", out_path=tmp_path / "x.jpg")


def test_saturation_is_raised_by_fifteen_percent(tmp_path) -> None:
    """Mesure sur une image LISSE et une sortie PNG.

    Une image de bruit aleatoire encodee en JPEG ne convient pas : le sous-echantillonnage
    de la chrominance moyenne le bruit et fait baisser l'ecart entre canaux, ce qui
    masquerait completement l'effet de la saturation.
    """
    ramp = np.zeros((900, 1600, 3), dtype=np.uint8)
    ramp[:, :, 0] = np.linspace(40, 220, 1600, dtype=np.uint8)[None, :]
    ramp[:, :, 1] = 110
    ramp[:, :, 2] = np.linspace(220, 40, 1600, dtype=np.uint8)[None, :]
    base = tmp_path / "smooth.png"
    Image.fromarray(ramp).save(base)

    out = build_thumbnail(base, "NO", "left", out_path=tmp_path / "smooth_out.png")
    after = np.asarray(Image.open(out).convert("RGB")).astype(int)
    before = np.asarray(fit_to_frame(Image.open(base), THUMBNAIL_WIDTH, THUMBNAIL_HEIGHT)).astype(int)
    # Bande basse : ni la fleche (y 319-488) ni l'accroche (en haut) ne s'y trouvent.
    strip = slice(620, 720)
    spread = lambda a: float((a[strip].max(axis=2) - a[strip].min(axis=2)).mean())  # noqa: E731
    assert spread(after) > spread(before) * 1.05
    assert spread(after) == pytest.approx(spread(before) * SATURATION_BOOST, rel=0.05)


def test_compositor_flips_the_arrow_for_the_right_side() -> None:
    compositor = ThumbnailCompositor()
    left, (lx, _) = compositor.arrow_layer("left")
    right, (rx, _) = compositor.arrow_layer("right")
    assert lx < THUMBNAIL_WIDTH / 2 < rx  # chaque fleche du bon cote
    # L'une est le miroir de l'autre : la pointe vise toujours le centre.
    assert np.array_equal(np.asarray(right), np.asarray(left)[:, ::-1])
    assert SATURATION_BOOST == 1.15


# --- orchestration -----------------------------------------------------------------------------
class FakeBackend:
    name = "fake"

    def __init__(self, data: bytes | Exception) -> None:
        self.data = data
        self.prompts: list[str] = []

    def generate(self, prompt, *, negative="", aspect_ratio="16:9"):
        self.prompts.append(prompt)
        if isinstance(self.data, Exception):
            raise self.data
        return self.data


def _stage(response, backend) -> ThumbnailStage:
    return ThumbnailStage(FakeManager(response), backend=backend)


def test_stage_runs_the_three_steps(tmp_path) -> None:
    draft = ThumbnailDraft(scene_description="A hero laughs in a ruin", hook_text="he smiles",
                           arrow_position="right", subject_position="left", avoid="gore")
    backend = FakeBackend(_png(1600, 900))
    result = _stage(FakeResponse(parsed=draft), backend).build(_analysis(), tmp_path)
    assert isinstance(result, ThumbnailResult)
    assert result.thumbnail.is_file() and result.base_image.is_file() and result.brief_json.is_file()
    assert json.loads(result.brief_json.read_text(encoding="utf-8"))["hook_text"] == "HE SMILES"
    assert set(result.seconds) == {"analyse", "image", "compositing"} and result.total_s >= 0
    assert STYLE_PROMPT in backend.prompts[0]


def test_stage_can_reuse_an_existing_illustration(tmp_path) -> None:
    """``base_image`` saute l'etape 2 : aucun quota d'image consomme."""
    draft = ThumbnailDraft(scene_description="x", hook_text="no way", arrow_position="left",
                           subject_position="center", avoid="")
    base = tmp_path / "deja.png"
    base.write_bytes(_png(1280, 720))
    backend = FakeBackend(AssertionError("ne doit pas etre appele"))
    result = _stage(FakeResponse(parsed=draft), backend).build(_analysis(), tmp_path, base_image=base)
    assert result.base_image == base and backend.prompts == []
    assert result.thumbnail.is_file()


def test_stage_run_never_breaks_the_video(tmp_path) -> None:
    """Regle centrale : une miniature ratee n'empeche jamais la video d'exister."""
    draft = ThumbnailDraft(scene_description="x", hook_text="hi", arrow_position="left",
                           subject_position="center", avoid="")
    # Echec du backend d'images.
    assert _stage(FakeResponse(parsed=draft), FakeBackend(ImageBackendError("quota"))).run(_analysis(), tmp_path) is None
    # Echec de l'analyse.
    assert _stage(FakeResponse(text="pas du json"), FakeBackend(_png())).run(_analysis(), tmp_path) is None
    # Erreur inattendue : absorbee elle aussi.
    assert _stage(FakeResponse(parsed=draft), FakeBackend(ZeroDivisionError("boum"))).run(_analysis(), tmp_path) is None
    # Succes : un resultat est bien renvoye.
    assert _stage(FakeResponse(parsed=draft), FakeBackend(_png())).run(_analysis(), tmp_path) is not None


def test_pipeline_stage_is_opt_in(tmp_path) -> None:
    from src.pipeline import PipelineOptions, stage_thumbnail

    options = PipelineOptions()
    assert options.make_thumbnail is False  # desactivee par defaut : la generation d'image a son propre quota
    assert stage_thumbnail(_analysis(), tmp_path, options) is None
