"""Tests des ressources audio (``src.utils.audio_assets``) : bruitages synthetises et musiques par ambiance."""

from __future__ import annotations

import numpy as np
import pytest
import soundfile as sf

from src.utils.audio_assets import (
    BGM_MOODS,
    SFX_KINDS,
    bgm_for_mood,
    ensure_default_bgm,
    ensure_default_sfx,
    find_bgm,
    find_sfx,
    synthesize_bgm,
    synthesize_sfx,
)


def test_synthesize_sfx_shapes_and_levels() -> None:
    for kind in SFX_KINDS:
        signal = synthesize_sfx(kind, 24000)
        assert signal.dtype == np.float32 and signal.ndim == 1
        assert 0.2 < len(signal) / 24000 < 1.0  # bruitage court
        assert float(np.abs(signal).max()) == pytest.approx(0.8, abs=1e-3)
        assert abs(float(signal[0])) < 0.05 and abs(float(signal[-1])) < 0.05  # enveloppe : pas de clic
    assert not np.array_equal(synthesize_sfx("swoosh"), synthesize_sfx("impact"))
    with pytest.raises(ValueError):
        synthesize_sfx("meow")


def test_ensure_default_sfx_writes_missing_files_and_keeps_existing(tmp_path) -> None:
    sfx_dir = tmp_path / "sfx"
    assert find_sfx(sfx_dir) == {} and find_sfx(None) == {}
    custom = sfx_dir / "impact.mp3"
    sfx_dir.mkdir()
    custom.write_bytes(b"not really audio")  # fichier fourni par l'utilisateur : jamais remplace
    files = ensure_default_sfx(sfx_dir, 24000)
    assert set(files) == set(SFX_KINDS) and files["impact"] == custom
    assert files["swoosh"].name == "swoosh.wav" and files["roar"].name == "roar.wav"
    data, rate = sf.read(str(files["swoosh"]))
    assert rate == 24000 and len(data) > 1000
    mtime = files["swoosh"].stat().st_mtime_ns
    assert ensure_default_sfx(sfx_dir, 24000)["swoosh"].stat().st_mtime_ns == mtime  # reutilise


def test_synthesize_bgm_loops_are_long_and_tame() -> None:
    for mood in BGM_MOODS:
        signal = synthesize_bgm(mood, 24000)
        assert signal.dtype == np.float32 and signal.ndim == 1
        assert 20.0 <= len(signal) / 24000 <= 30.0  # boucle d'une vingtaine de secondes
        assert float(np.abs(signal).max()) == pytest.approx(0.5, abs=1e-3)
        assert abs(float(signal[0])) < 0.02 and abs(float(signal[-1])) < 0.02  # points de boucle sans clic
        assert float(np.abs(signal).mean()) > 0.02  # pas silencieux
    assert not np.array_equal(synthesize_bgm("calm"), synthesize_bgm("tense"))
    with pytest.raises(ValueError):
        synthesize_bgm("disco")


def test_ensure_default_bgm_only_when_no_music_is_provided(tmp_path) -> None:
    bgm_dir = tmp_path / "bgm"
    files = ensure_default_bgm(bgm_dir, 24000)
    assert set(files) == set(BGM_MOODS) and all(files[m].name == f"{m}.wav" for m in BGM_MOODS)
    data, rate = sf.read(str(files["action"]))
    assert rate == 24000 and len(data) > 20 * 24000
    mtime = files["calm"].stat().st_mtime_ns
    assert ensure_default_bgm(bgm_dir, 24000)["calm"].stat().st_mtime_ns == mtime  # reutilise
    # Une musique fournie par l'utilisateur : rien n'est synthetise, elle sert a toutes les ambiances.
    user_dir = tmp_path / "mine"
    user_dir.mkdir()
    (user_dir / "default.mp3").write_bytes(b"x")
    assert ensure_default_bgm(user_dir, 24000) == {"default": user_dir / "default.mp3"}
    assert sorted(p.name for p in user_dir.iterdir()) == ["default.mp3"]


def test_find_bgm_and_fallbacks(tmp_path) -> None:
    assert find_bgm(tmp_path / "nowhere") == {} and bgm_for_mood("calm", {}) is None
    bgm_dir = tmp_path / "bgm"
    bgm_dir.mkdir()
    (bgm_dir / "calm.mp3").write_bytes(b"x")
    (bgm_dir / "calm.wav").write_bytes(b"x")  # .wav prioritaire sur .mp3
    (bgm_dir / "default.ogg").write_bytes(b"x")
    files = find_bgm(bgm_dir)
    assert set(files) == {"calm", "default"} and files["calm"].suffix == ".wav"
    assert bgm_for_mood("calm", files).suffix == ".wav"
    assert bgm_for_mood("action", files).name == "default.ogg"
    only_action = {"action": bgm_dir / "a.wav"}
    assert bgm_for_mood("tense", only_action) == bgm_dir / "a.wav"
    assert BGM_MOODS == ("calm", "tense", "action")
