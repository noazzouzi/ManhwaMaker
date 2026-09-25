"""Tests unitaires du moteur Kokoro TTS (``src.modules.tts_engine``) avec un pipeline factice.

Aucun modèle n'est chargé : ``FakePipeline`` imite ``KPipeline.__call__`` et produit
un signal dont la longueur dépend du nombre de mots.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

import numpy as np
import pytest
import soundfile as sf

from src.models.audio import VoiceoverManifest
from src.models.scene import ChapterAnalysis, Scene
from src.modules import tts_engine
from src.modules.tts_engine import (
    DEFAULT_PADDING_S,
    SAMPLE_RATE,
    KokoroTTS,
    TTSError,
    apply_pronunciations,
    compile_pronunciations,
    concat_wavs,
    format_manifest,
    load_manifest,
    load_pronunciations,
    prepare_text,
    read_wav,
    resolve_voice,
    write_wav,
)

SECONDS_PER_WORD = 0.25


@dataclass
class FakeResult:
    graphemes: str
    phonemes: str
    audio: np.ndarray | None


class FakePipeline:
    """Imite ``KPipeline`` : 0,25 s de signal par mot, en deux morceaux."""

    def __init__(self, *, empty: bool = False, as_tuple: bool = False):
        self.calls: list[dict] = []
        self.empty = empty
        self.as_tuple = as_tuple

    def __call__(self, text, *, voice, speed, split_pattern):
        self.calls.append({"text": text, "voice": voice, "speed": speed, "split_pattern": split_pattern})
        if self.empty:
            return iter(())
        n = int(len(text.split()) * SECONDS_PER_WORD * SAMPLE_RATE / speed)
        first, second = n // 2, n - n // 2
        chunks = [np.full(first, 0.25, dtype=np.float32), np.full(second, -0.25, dtype=np.float32)]
        if self.as_tuple:
            return iter([("g1", "p1", chunks[0]), ("g2", "p2", chunks[1])])
        return iter([FakeResult("g1", "p1", chunks[0]), FakeResult("g2", "p2", chunks[1])])


def _scene(index: int, narration: str, *, is_filler: bool = False, emotion: str = "calm") -> Scene:
    return Scene(index=index, panel_ids=[index], narration=narration, emotion=emotion, is_filler=is_filler)


def _analysis(scenes: list[Scene]) -> ChapterAnalysis:
    return ChapterAnalysis(model="fake", language="en", n_panels=len(scenes), scenes=scenes)


# --- Texte -----------------------------------------------------------------------------
def test_prepare_text_normalizes_typography_and_punctuation() -> None:
    curly = chr(0x201C) + "Wait a sec" + chr(0x201D) + chr(0x2026) + " he said " + chr(0x2014) + " twice"
    assert prepare_text(curly) == '"Wait a sec"... he said - twice.'
    assert prepare_text("  Hello   world ") == "Hello world."
    assert prepare_text("Already done!") == "Already done!"
    assert prepare_text("Question?") == "Question?"
    assert prepare_text(chr(0x200B) + "   ") == ""
    assert prepare_text("...") == ""
    assert prepare_text("It" + chr(0x2019) + "s fine") == "It's fine."


def test_pronunciations_word_boundaries_case_and_phonemes() -> None:
    mapping = {"Destia": "Dess-tee-ah", "Kang JinHyeok": "Kang Jin-hyuk", "JinHyeok": "Jin-hyuk", "Kokoro": "/kˈOkəɹO/"}
    text = "DESTIA roars at Kang JinHyeok; destial is not a name. JinHyeok laughs. Kokoro speaks."
    out = apply_pronunciations(text, mapping)
    assert out == (
        "Dess-tee-ah roars at Kang Jin-hyuk; destial is not a name. Jin-hyuk laughs. "
        "[Kokoro](/kˈOkəɹO/) speaks."
    )
    # Les cles longues passent avant les courtes ; les motifs compiles sont reutilisables.
    compiled = compile_pronunciations(mapping)
    assert [p.pattern for p, _ in compiled][0].startswith("(?<!\\w)Kang")
    assert apply_pronunciations("Kang JinHyeok", compiled) == "Kang Jin-hyuk"
    assert apply_pronunciations("nothing here", {}) == "nothing here"
    # Caracteres speciaux dans la cle : echappes.
    assert apply_pronunciations("SSS-rank skill", {"SSS-rank": "triple-S rank"}) == "triple-S rank skill"
    assert apply_pronunciations("Slave. In. Utero", {"Slave. In. Utero": "Slave In Utero"}) == "Slave In Utero"


def test_load_pronunciations_default_file_and_validation(tmp_path, monkeypatch) -> None:
    default = load_pronunciations()  # config/pronunciations.json du projet
    assert default.get("Destia") == "Dess-tee-ah" and not any(k.startswith("_") for k in default)

    custom = tmp_path / "p.json"
    custom.write_text(json.dumps({"_comment": "x", " Bam ": " Bahm "}), encoding="utf-8")
    assert load_pronunciations(custom) == {"Bam": "Bahm"}
    with pytest.raises(FileNotFoundError):
        load_pronunciations(tmp_path / "missing.json")
    bad = tmp_path / "bad.json"
    bad.write_text("[1, 2]", encoding="utf-8")
    with pytest.raises(ValueError):
        load_pronunciations(bad)
    bad.write_text('{"Bam": 3}', encoding="utf-8")
    with pytest.raises(ValueError):
        load_pronunciations(bad)
    bad.write_text("{oops", encoding="utf-8")
    with pytest.raises(ValueError, match="JSON"):
        load_pronunciations(bad)
    # Fichier par defaut absent : dictionnaire vide.
    monkeypatch.setattr(tts_engine, "DEFAULT_PRONUNCIATIONS_FILE", tmp_path / "none.json")
    assert load_pronunciations() == {}


def test_resolve_voice_defaults_to_american_english() -> None:
    assert resolve_voice() == ("a", "am_fenrir,am_michael")
    assert resolve_voice("en", "af_heart") == ("a", "af_heart")
    assert resolve_voice("FR") == ("f", "ff_siwis")
    with pytest.raises(TTSError, match="Langue"):
        resolve_voice("klingon")


# --- WAV ------------------------------------------------------------------------------
def test_write_read_concat_wav(tmp_path) -> None:
    a = np.full(2400, 0.5, dtype=np.float32)
    b = np.full(1200, -2.0, dtype=np.float32)  # ecrete a -1.0
    pa, pb = write_wav(tmp_path / "a.wav", a), write_wav(tmp_path / "sub" / "b.wav", b)
    info = sf.info(str(pa))
    assert info.samplerate == SAMPLE_RATE and info.channels == 1 and info.subtype == "PCM_16"
    samples, rate = read_wav(pb)
    assert rate == SAMPLE_RATE and samples.shape == (1200,) and samples.min() >= -1.0
    joined = concat_wavs([pa, pb], tmp_path / "all.wav")
    js, _ = read_wav(joined)
    assert js.shape == (3600,) and abs(js[:2400].mean() - 0.5) < 0.01 and js[2400:].max() < -0.99
    write_wav(tmp_path / "other.wav", a, 16_000)
    with pytest.raises(TTSError, match="Hz"):
        concat_wavs([pa, tmp_path / "other.wav"], tmp_path / "bad.wav")


def test_export_mp3(tmp_path) -> None:
    from src.modules.tts_engine import export_mp3

    t = np.arange(SAMPLE_RATE, dtype=np.float32) / SAMPLE_RATE
    wav = write_wav(tmp_path / "tone.wav", 0.3 * np.sin(2 * np.pi * 440 * t))
    try:
        mp3 = export_mp3(wav, tmp_path / "out" / "tone.mp3")
    except TTSError:
        pytest.skip("encodage MP3 non supporte par la libsndfile installee")
    info = sf.info(str(mp3))
    assert info.samplerate == SAMPLE_RATE and info.format == "MP3"
    assert info.subtype == "MPEG_LAYER_III" and info.channels == 1
    assert 0 < mp3.stat().st_size < wav.stat().st_size


# --- Moteur ------------------------------------------------------------------------------
def test_synthesize_scene_adds_exact_padding_and_applies_pronunciations(tmp_path) -> None:
    pipeline = FakePipeline()
    pad = 0.18  # le silence de fin n'est plus ajoute par defaut, mais reste reglable
    tts = KokoroTTS(pipeline=pipeline, pronunciations={"Destia": "Dess-tee-ah"}, padding_s=pad)
    scene = _scene(3, "Destia roars at the young man", emotion="epic")
    audio = tts.synthesize_scene(scene, tmp_path / "scene_003.wav")

    assert pipeline.calls[0]["text"] == "Dess-tee-ah roars at the young man."
    assert pipeline.calls[0]["voice"] == "am_fenrir,am_michael" and pipeline.calls[0]["speed"] == 1.0
    n_words = 6
    expected_speech = int(n_words * SECONDS_PER_WORD * SAMPLE_RATE) / SAMPLE_RATE
    assert audio.speech_s == pytest.approx(expected_speech)
    assert audio.duration_s == pytest.approx(expected_speech + pad)
    assert audio.scene_index == 3 and audio.file == "scene_003.wav" and audio.emotion == "epic"
    assert audio.sample_rate == SAMPLE_RATE and audio.text.startswith("Dess-tee-ah")
    samples, rate = read_wav(tmp_path / "scene_003.wav")
    assert rate == SAMPLE_RATE and len(samples) == round(audio.duration_s * SAMPLE_RATE)
    tail = samples[-int(pad * SAMPLE_RATE) :]
    assert np.all(tail == 0.0)  # silence de fin exact
    assert np.abs(samples[: len(samples) - len(tail)]).min() > 0.2  # parole non nulle


def test_sentence_gap_between_sentences(tmp_path) -> None:
    from src.modules.tts_engine import DEFAULT_SENTENCE_GAP_S, split_sentences

    # Une ponctuation forte suivie d'une minuscule (points de suspension, citation) ne coupe pas.
    assert split_sentences('One two. Three four five! "Six" seven? eight... nine.') == [
        "One two.", "Three four five!", '"Six" seven? eight... nine.',
    ]
    assert split_sentences("Mr. Kim waits. He waits again.") == ["Mr.", "Kim waits.", "He waits again."]
    # Pause ajoutee (reglage explicite) : une synthese par phrase, silence exact entre les deux.
    gap = 0.2
    pipeline = FakePipeline()
    tts = KokoroTTS(pipeline=pipeline, pronunciations={}, sentence_gap_s=gap)
    audio = tts.synthesize_scene(_scene(0, "One two. Three four five"), tmp_path / "s.wav")
    assert [c["text"] for c in pipeline.calls] == ["One two.", "Three four five."]  # une synthese par phrase
    expected = int(2 * SECONDS_PER_WORD * SAMPLE_RATE) / SAMPLE_RATE + gap + int(3 * SECONDS_PER_WORD * SAMPLE_RATE) / SAMPLE_RATE
    assert audio.speech_s == pytest.approx(expected)
    samples, _ = read_wav(tmp_path / "s.wav")
    gap_start = int(2 * SECONDS_PER_WORD * SAMPLE_RATE)
    assert np.all(samples[gap_start : gap_start + int(gap * SAMPLE_RATE)] == 0.0)
    # Defaut : la scene est lue d'un seul bloc, les pauses sont laissees a Kokoro (ponctuation),
    # et aucun silence n'est ajoute en fin de scene.
    pipeline = FakePipeline()
    tts = KokoroTTS(pipeline=pipeline, pronunciations={})
    audio = tts.synthesize_scene(_scene(1, "One two. Three four five"), tmp_path / "t.wav")
    assert len(pipeline.calls) == 1 and audio.speech_s == pytest.approx(int(5 * SECONDS_PER_WORD * SAMPLE_RATE) / SAMPLE_RATE)
    assert audio.duration_s == pytest.approx(audio.speech_s)
    with pytest.raises(ValueError):
        KokoroTTS(pipeline=FakePipeline(), sentence_gap_s=-0.1)
    assert DEFAULT_PADDING_S == 0.0 and DEFAULT_SENTENCE_GAP_S == 0.0


def test_synthesize_handles_tuple_results_speed_and_errors(tmp_path) -> None:
    tts = KokoroTTS(pipeline=FakePipeline(as_tuple=True), speed=2.0, padding_s=0.0, pronunciations={})
    audio = tts.synthesize_scene(_scene(0, "one two three four"), tmp_path / "s.wav")
    assert audio.duration_s == pytest.approx(audio.speech_s)
    assert audio.speech_s == pytest.approx(int(4 * SECONDS_PER_WORD * SAMPLE_RATE / 2.0) / SAMPLE_RATE)

    with pytest.raises(TTSError, match="aucun audio"):
        KokoroTTS(pipeline=FakePipeline(empty=True), pronunciations={}).synthesize("hello")
    with pytest.raises(TTSError, match="vide"):
        KokoroTTS(pipeline=FakePipeline(), pronunciations={}).synthesize_scene(_scene(1, " ... "), tmp_path / "e.wav")
    with pytest.raises(ValueError):
        KokoroTTS(pipeline=FakePipeline(), speed=0)
    with pytest.raises(ValueError):
        KokoroTTS(pipeline=FakePipeline(), padding_s=-1)


def test_synthesize_analysis_skips_filler_and_writes_manifest(tmp_path) -> None:
    scenes = [
        _scene(0, "The hero wakes up.", emotion="calm"),
        _scene(1, "Title card.", is_filler=True),
        _scene(2, "He fights the dragon with all his might.", emotion="action"),
    ]
    tts = KokoroTTS(pipeline=FakePipeline(), pronunciations={})
    manifest = tts.synthesize_analysis(_analysis(scenes), tmp_path / "audio")

    assert isinstance(manifest, VoiceoverManifest)
    assert [item.scene_index for item in manifest.items] == [0, 2]
    assert manifest.voice == "am_fenrir,am_michael" and manifest.lang_code == "a" and manifest.language == "en"
    assert manifest.padding_s == DEFAULT_PADDING_S and manifest.sample_rate == SAMPLE_RATE
    assert manifest.sentence_gap_s == 0.0
    assert manifest.total_duration_s == pytest.approx(sum(i.duration_s for i in manifest.items))
    assert manifest.full_file == "voiceover_full.wav"
    files = sorted(p.name for p in (tmp_path / "audio").iterdir())
    assert files == ["scene_000.wav", "scene_002.wav", "voiceover.json", "voiceover_full.wav"]
    full, _ = read_wav(tmp_path / "audio" / "voiceover_full.wav")
    assert len(full) == round(manifest.total_duration_s * SAMPLE_RATE)

    reloaded = load_manifest(tmp_path / "audio" / "voiceover.json")
    assert reloaded == manifest
    assert reloaded.by_scene()[2].emotion == "action"
    report = format_manifest(manifest)
    report.encode("ascii")
    assert "Segments   : 2" in report and "scene_002.wav" in report

    # Remplissage inclus, sans WAV complet.
    with_filler = tts.synthesize_analysis(_analysis(scenes), tmp_path / "audio2", include_filler=True, full_file=None)
    assert [i.scene_index for i in with_filler.items] == [0, 1, 2] and with_filler.full_file is None
    assert not (tmp_path / "audio2" / "voiceover_full.wav").exists()
    with pytest.raises(ValueError):
        tts.synthesize_analysis(_analysis([scenes[1]]), tmp_path / "audio3")


def test_pipeline_is_lazy_and_missing_kokoro_gives_clear_error(monkeypatch) -> None:
    tts = KokoroTTS(pronunciations={})
    assert tts._pipeline is None  # rien charge tant qu'on ne synthetise pas

    import builtins

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "kokoro":
            raise ImportError("no kokoro")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    with pytest.raises(TTSError, match="pip install kokoro"):
        _ = tts.pipeline


def test_default_pronunciations_cover_tested_series() -> None:
    mapping = load_pronunciations()
    text = "Kang JinHyeok defeats Destia with an SSS-rank skill from Etherion."
    out = apply_pronunciations(text, mapping)
    assert re.search(r"Jin-hyuk", out) and "Dess-tee-ah" in out and "triple-S rank" in out
    assert "JinHyeok" not in out and "SSS" not in out
