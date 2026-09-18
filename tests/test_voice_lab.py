"""Tests du banc d'essai des voix Kokoro (``src.modules.voice_lab``) avec un moteur factice."""

from __future__ import annotations

import numpy as np
import pytest
import soundfile as sf

from src.modules.tts_engine import SAMPLE_RATE, TTSError
from src.modules.voice_lab import (
    AMERICAN_FEMALE,
    AMERICAN_MALE,
    BRITISH_FEMALE,
    BRITISH_MALE,
    DEFAULT_SAMPLE_TEXT,
    ENGLISH_VOICES,
    VoiceSample,
    build_voice_report,
    compare_voices,
    format_voice_table,
)


class FakeEngine:
    """Moteur factice : la duree depend de la voix (une voix « rapide », une « lente »)."""

    def __init__(self, missing: set[str] = frozenset()):
        self.calls: list[tuple[str, str]] = []
        self.missing = missing

    def spoken_text(self, text: str) -> str:
        return text.strip()

    def synthesize_text(self, text: str, voice: str | None = None) -> np.ndarray:
        self.calls.append((voice or "?", text))
        if voice in self.missing:
            raise TTSError(f"voix {voice} absente")
        seconds = 3.0 if voice.endswith("fenrir") else 6.0
        return np.full(int(seconds * SAMPLE_RATE), 0.1, dtype=np.float32)


def test_voice_catalogue_is_consistent() -> None:
    assert len(ENGLISH_VOICES) == len(set(ENGLISH_VOICES)) == 28
    assert ENGLISH_VOICES == AMERICAN_FEMALE + AMERICAN_MALE + BRITISH_FEMALE + BRITISH_MALE
    assert "am_fenrir" in AMERICAN_MALE and "af_heart" in AMERICAN_FEMALE
    assert all(v.startswith(("af_", "am_")) for v in AMERICAN_FEMALE + AMERICAN_MALE)
    assert all(v.startswith(("bf_", "bm_")) for v in BRITISH_FEMALE + BRITISH_MALE)
    assert len(DEFAULT_SAMPLE_TEXT.split()) > 30  # assez long pour juger le rythme


def test_compare_voices_writes_wavs_and_measures(tmp_path) -> None:
    engine = FakeEngine()
    samples = compare_voices(
        ["am_fenrir", "af_heart", "bm_george"], tmp_path, text="Two words here. And more.",
        pronunciations={}, engine_factory=lambda: engine,
    )
    assert [s.voice for s in samples] == ["am_fenrir", "af_heart", "bm_george"]
    assert {call[0] for call in engine.calls} == {"am_fenrir", "af_heart", "bm_george"}
    for sample in samples:
        data, rate = sf.read(str(tmp_path / sample.file))
        assert rate == SAMPLE_RATE and len(data) == int(sample.duration_s * SAMPLE_RATE)
    fenrir, heart, george = samples
    assert fenrir.duration_s == pytest.approx(3.0) and heart.duration_s == pytest.approx(6.0)
    # 5 mots en 3 s = 100 mots/min ; en 6 s = 50.
    assert fenrir.words == 5 and fenrir.words_per_minute == pytest.approx(100.0)
    assert heart.words_per_minute == pytest.approx(50.0)
    assert fenrir.accent == "americain" and fenrir.gender == "masculine"
    assert heart.gender == "feminine" and george.accent == "britannique" and george.gender == "masculine"
    assert fenrir.realtime_factor > 0 and fenrir.lang_code == "a"


def test_compare_voices_skips_missing_voices_and_needs_at_least_one(tmp_path) -> None:
    engine = FakeEngine(missing={"am_ghost"})
    samples = compare_voices(["am_ghost", "am_fenrir"], tmp_path, pronunciations={}, engine_factory=lambda: engine)
    assert [s.voice for s in samples] == ["am_fenrir"] and not (tmp_path / "am_ghost.wav").exists()
    with pytest.raises(TTSError, match="Aucune voix"):
        compare_voices(["am_ghost"], tmp_path, pronunciations={}, engine_factory=lambda: FakeEngine(missing={"am_ghost"}))


def test_compare_voices_runs_in_parallel(tmp_path) -> None:
    engine = FakeEngine()
    voices = ["am_fenrir", "af_heart", "bf_emma", "bm_lewis"]
    samples = compare_voices(voices, tmp_path, workers=4, pronunciations={}, engine_factory=lambda: engine)
    assert sorted(s.voice for s in samples) == sorted(voices)  # ordre d'entree preserve par map()
    assert [s.voice for s in samples] == voices


def test_voice_report_and_table(tmp_path) -> None:
    samples = [
        VoiceSample(voice="am_fenrir", lang_code="a", file="am_fenrir.wav", duration_s=3.0, compute_s=1.0, words=5),
        VoiceSample(voice="bf_emma", lang_code="b", file="bf_emma.wav", duration_s=6.0, compute_s=2.0, words=5),
    ]
    path = build_voice_report(samples, tmp_path / "voices.html", text="Two words here. And more.")
    document = path.read_text(encoding="utf-8")
    assert document.count("<audio controls") == 2
    assert 'src="am_fenrir.wav"' in document and 'src="bf_emma.wav"' in document
    assert "am_fenrir" in document and "britannique" in document and "americain" in document
    assert "100" in document and "50" in document  # mots par minute
    table = format_voice_table(samples)
    table.encode("ascii")
    assert table.splitlines()[2].startswith("am_fenrir")  # trie par debit decroissant
    assert "bf_emma" in table
