"""Ressources audio du montage : bruitages (SFX) et musiques de fond (BGM) par ambiance.

- **SFX** : trois bruitages courts (``swoosh``, ``impact``, ``roar``) lus dans
  ``config/sfx/<kind>.(wav|mp3|ogg|flac)``. S'ils manquent, des bruitages de
  substitution sont **synthétisés** (NumPy) et écrits dans ce dossier : ils
  permettent de valider le montage et sont à remplacer par de vrais effets
  (mêmes noms de fichiers).
- **BGM** : musiques par ambiance lues dans ``config/bgm/<mood>.(wav|mp3|ogg|flac)``
  avec ``mood`` parmi ``calm``, ``tense``, ``action`` (une ``default`` sert de
  repli). Si le dossier ne contient aucune musique, trois boucles d'ambiance de
  substitution sont **synthétisées** (à remplacer par de vraies musiques).
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import soundfile as sf

from src.utils.config import PROJECT_ROOT

logger = logging.getLogger(__name__)

SFX_KINDS: tuple[str, ...] = ("swoosh", "impact", "roar")
BGM_MOODS: tuple[str, ...] = ("calm", "tense", "action")
AUDIO_EXTENSIONS: tuple[str, ...] = (".wav", ".mp3", ".ogg", ".flac")
DEFAULT_SFX_DIR: Path = PROJECT_ROOT / "config" / "sfx"
DEFAULT_BGM_DIR: Path = PROJECT_ROOT / "config" / "bgm"
SFX_SAMPLE_RATE: int = 24_000


def _find_named_audio(directory: str | Path | None, names: tuple[str, ...]) -> dict[str, Path]:
    if directory is None:
        return {}
    directory = Path(directory)
    if not directory.is_dir():
        return {}
    found: dict[str, Path] = {}
    for name in names:
        for ext in AUDIO_EXTENSIONS:
            candidate = directory / f"{name}{ext}"
            if candidate.is_file():
                found[name] = candidate
                break
    return found


def _envelope(n: int, attack: float, release: float, sample_rate: int) -> np.ndarray:
    env = np.ones(n, dtype=np.float32)
    a = min(n, max(1, int(attack * sample_rate)))
    r = min(n, max(1, int(release * sample_rate)))
    env[:a] = np.linspace(0.0, 1.0, a, dtype=np.float32)
    env[n - r :] *= np.linspace(1.0, 0.0, r, dtype=np.float32)
    return env


def _lowpass(signal: np.ndarray, cutoff_hz: np.ndarray | float, sample_rate: int) -> np.ndarray:
    """Filtre passe-bas du premier ordre à coupure éventuellement variable (balayage)."""
    cutoff = np.broadcast_to(np.asarray(cutoff_hz, dtype=np.float32), signal.shape)
    alpha = (2 * np.pi * cutoff / sample_rate) / (1 + 2 * np.pi * cutoff / sample_rate)
    out = np.empty_like(signal)
    acc = 0.0
    for i in range(len(signal)):
        acc += alpha[i] * (signal[i] - acc)
        out[i] = acc
    return out


def synthesize_sfx(kind: str, sample_rate: int = SFX_SAMPLE_RATE, seed: int = 7) -> np.ndarray:
    """Synthétise un bruitage de substitution (float32 mono, crête 0,8).

    - ``swoosh`` : bruit blanc filtré avec balayage descendant (0,35 s) ;
    - ``impact`` : coup grave (sinus 110 → 45 Hz) + claquement bruité (0,45 s) ;
    - ``roar`` : grondement grave riche en harmoniques, modulé (0,7 s).

    Raises:
        ValueError: type de bruitage inconnu.
    """
    rng = np.random.default_rng(seed)
    if kind == "swoosh":
        n = int(0.35 * sample_rate)
        noise = rng.standard_normal(n).astype(np.float32)
        cutoff = np.geomspace(6000.0, 800.0, n)
        signal = _lowpass(noise, cutoff, sample_rate) * _envelope(n, 0.06, 0.18, sample_rate)
    elif kind == "impact":
        n = int(0.45 * sample_rate)
        t = np.arange(n, dtype=np.float32) / sample_rate
        freq = 45.0 + 65.0 * np.exp(-t * 14.0)
        phase = 2 * np.pi * np.cumsum(freq) / sample_rate
        thump = np.sin(phase).astype(np.float32) * np.exp(-t * 7.0)
        click = _lowpass(rng.standard_normal(n).astype(np.float32), 3000.0, sample_rate) * np.exp(-t * 60.0)
        signal = (thump + 0.6 * click) * _envelope(n, 0.003, 0.1, sample_rate)
    elif kind == "roar":
        n = int(0.7 * sample_rate)
        t = np.arange(n, dtype=np.float32) / sample_rate
        base = 70.0 + 20.0 * np.sin(2 * np.pi * 1.3 * t)
        phase = 2 * np.pi * np.cumsum(base) / sample_rate
        harmonics = sum(np.sin(k * phase) / k for k in (1, 2, 3, 4, 5))
        growl = 0.5 + 0.5 * np.sin(2 * np.pi * 27.0 * t)
        breath = _lowpass(rng.standard_normal(n).astype(np.float32), 900.0, sample_rate)
        signal = (harmonics * growl + 0.35 * breath) * _envelope(n, 0.05, 0.3, sample_rate)
    else:
        raise ValueError(f"Bruitage inconnu : {kind!r} (attendu {', '.join(SFX_KINDS)})")
    signal = np.asarray(signal, dtype=np.float32)
    peak = float(np.abs(signal).max()) or 1.0
    return (signal / peak * 0.8).astype(np.float32)


def find_sfx(sfx_dir: str | Path | None = DEFAULT_SFX_DIR) -> dict[str, Path]:
    """Bruitages disponibles ``{kind: fichier}`` dans ``sfx_dir``."""
    return _find_named_audio(sfx_dir, SFX_KINDS)


def ensure_default_sfx(sfx_dir: str | Path | None = DEFAULT_SFX_DIR, sample_rate: int = SFX_SAMPLE_RATE) -> dict[str, Path]:
    """Retourne les bruitages de ``sfx_dir``, en synthétisant ceux qui manquent (``<kind>.wav``)."""
    sfx_dir = Path(sfx_dir) if sfx_dir is not None else DEFAULT_SFX_DIR
    found = find_sfx(sfx_dir)
    missing = [kind for kind in SFX_KINDS if kind not in found]
    if missing:
        sfx_dir.mkdir(parents=True, exist_ok=True)
        for kind in missing:
            path = sfx_dir / f"{kind}.wav"
            sf.write(str(path), synthesize_sfx(kind, sample_rate), sample_rate, subtype="PCM_16")
            found[kind] = path
        logger.info("Bruitages de substitution synthetises dans %s : %s", sfx_dir, ", ".join(missing))
    return found


def find_bgm(bgm_dir: str | Path | None = DEFAULT_BGM_DIR) -> dict[str, Path]:
    """Musiques disponibles ``{mood: fichier}`` (``default`` accepté comme repli)."""
    return _find_named_audio(bgm_dir, (*BGM_MOODS, "default"))


# --- Musiques de substitution -----------------------------------------------------------
BGM_SAMPLE_RATE: int = 24_000
BGM_LOOP_SECONDS: float = 24.0


def _hann_window(n: int, center: float, width: float) -> np.ndarray:
    """Fenêtre de Hann de largeur ``width`` centrée en ``center`` (indices, avec bouclage sur ``n``)."""
    idx = np.arange(n, dtype=np.float32)
    delta = (idx - center + n / 2) % n - n / 2
    inside = np.abs(delta) < width / 2
    return np.where(inside, 0.5 * (1 + np.cos(2 * np.pi * delta / width)), 0.0).astype(np.float32)


def _harmonic_tone(freq: float, n: int, sample_rate: int, harmonics: tuple[float, ...]) -> np.ndarray:
    t = np.arange(n, dtype=np.float32) / sample_rate
    return sum(a * np.sin(2 * np.pi * freq * (k + 1) * t) for k, a in enumerate(harmonics)).astype(np.float32)


def _smooth_noise(n: int, sample_rate: int, cutoff_hz: float, rng: np.random.Generator) -> np.ndarray:
    """Bruit blanc lissé par moyenne glissante (≈ passe-bas ``cutoff_hz``), vectorisé."""
    noise = rng.standard_normal(n).astype(np.float32)
    k = max(1, int(sample_rate / cutoff_hz))
    kernel = np.hanning(k + 2)[1:-1].astype(np.float32)
    kernel /= kernel.sum()
    return np.convolve(noise, kernel, mode="same").astype(np.float32)


def synthesize_bgm(mood: str, sample_rate: int = BGM_SAMPLE_RATE, seconds: float = BGM_LOOP_SECONDS, seed: int = 11) -> np.ndarray:
    """Synthétise une musique d'ambiance de substitution, **bouclable** (float32 mono, crête 0,5).

    - ``calm`` : nappe de quatre accords (Am, F, C, G) avec basse douce ;
    - ``tense`` : bourdon grave battant, souffle filtré, note aiguë espacée ;
    - ``action`` : 140 BPM, grosse caisse à chaque temps, charley en contretemps, basse.

    Raises:
        ValueError: ambiance inconnue.
    """
    rng = np.random.default_rng(seed)
    n = int(seconds * sample_rate)
    t = np.arange(n, dtype=np.float32) / sample_rate
    if mood == "calm":
        chords = [(220.0, 261.63, 329.63), (174.61, 220.0, 261.63), (130.81, 329.63, 392.0), (196.0, 246.94, 293.66)]
        signal = np.zeros(n, dtype=np.float32)
        span = n / len(chords)
        for i, chord in enumerate(chords):
            window = _hann_window(n, span * (i + 0.5), span * 1.25)
            pad = sum(_harmonic_tone(f, n, sample_rate, (1.0, 0.35, 0.15)) for f in chord)
            bass = _harmonic_tone(chord[0] / 2, n, sample_rate, (0.6, 0.2))
            signal += window * (pad + bass)
        signal *= 1.0 + 0.08 * np.sin(2 * np.pi * t / seconds * 2)
    elif mood == "tense":
        drone = (
            _harmonic_tone(55.0, n, sample_rate, (1.0, 0.6, 0.3, 0.15))
            + _harmonic_tone(55.6, n, sample_rate, (0.8, 0.4, 0.2))
            + _harmonic_tone(82.41, n, sample_rate, (0.5, 0.25))
        )
        tremolo = 0.7 + 0.3 * np.sin(2 * np.pi * 0.5 * t)
        breath = 0.35 * _smooth_noise(n, sample_rate, 400.0, rng)
        pings = np.zeros(n, dtype=np.float32)
        for k in range(int(seconds // 4)):
            start = int(k * 4 * sample_rate)
            length = min(n - start, int(1.5 * sample_rate))
            tt = np.arange(length, dtype=np.float32) / sample_rate
            pings[start : start + length] += 0.25 * np.sin(2 * np.pi * 659.26 * tt) * np.exp(-tt * 3.0)
        signal = drone * tremolo + breath + pings
    elif mood == "action":
        bpm = 140.0
        beat = 60.0 / bpm
        n_beats = int(seconds / beat)
        n = int(n_beats * beat * sample_rate)  # boucle exacte sur un nombre entier de temps
        t = np.arange(n, dtype=np.float32) / sample_rate
        signal = np.zeros(n, dtype=np.float32)
        bass_notes = (41.2, 41.2, 49.0, 55.0)  # E1 E1 G1 A1, une note par mesure
        for b in range(n_beats):
            start = int(b * beat * sample_rate)
            length = min(n - start, int(0.35 * sample_rate))
            tt = np.arange(length, dtype=np.float32) / sample_rate
            sweep = 45.0 + 105.0 * np.exp(-tt * 40.0)
            phase = 2 * np.pi * np.cumsum(sweep) / sample_rate
            signal[start : start + length] += 0.9 * np.sin(phase) * np.exp(-tt * 9.0)
            hat_start = start + int(beat / 2 * sample_rate)
            hat_len = min(n - hat_start, int(0.05 * sample_rate))
            if hat_len > 0:
                hat = rng.standard_normal(hat_len).astype(np.float32)
                hat -= np.convolve(hat, np.ones(8, dtype=np.float32) / 8, mode="same")
                signal[hat_start : hat_start + hat_len] += 0.12 * hat * np.exp(-np.arange(hat_len) / (0.012 * sample_rate))
            note = bass_notes[(b // 4) % len(bass_notes)] * (1.5 if b % 4 == 3 else 1.0)
            for sub in (0.0, 0.5):
                s0 = start + int(sub * beat * sample_rate)
                s_len = min(n - s0, int(0.22 * sample_rate))
                if s_len > 0:
                    tt = np.arange(s_len, dtype=np.float32) / sample_rate
                    tone = sum(np.sin(2 * np.pi * note * k * tt) / k for k in (1, 2, 3, 4))
                    signal[s0 : s0 + s_len] += 0.3 * tone * np.exp(-tt * 12.0)
    else:
        raise ValueError(f"Ambiance inconnue : {mood!r} (attendu {', '.join(BGM_MOODS)})")
    signal = np.asarray(signal, dtype=np.float32)
    edge = max(1, int(0.02 * sample_rate))
    signal[:edge] *= np.linspace(0.0, 1.0, edge, dtype=np.float32)
    signal[-edge:] *= np.linspace(1.0, 0.0, edge, dtype=np.float32)
    peak = float(np.abs(signal).max()) or 1.0
    return (signal / peak * 0.5).astype(np.float32)


def ensure_default_bgm(bgm_dir: str | Path | None = DEFAULT_BGM_DIR, sample_rate: int = BGM_SAMPLE_RATE) -> dict[str, Path]:
    """Musiques de ``bgm_dir`` ; si le dossier n'en contient **aucune**, synthétise les trois ambiances.

    Dès qu'une musique est fournie par l'utilisateur, rien n'est ajouté : les ambiances
    manquantes se replient sur ``default`` ou sur la première musique disponible.
    """
    bgm_dir = Path(bgm_dir) if bgm_dir is not None else DEFAULT_BGM_DIR
    found = find_bgm(bgm_dir)
    if found:
        return found
    bgm_dir.mkdir(parents=True, exist_ok=True)
    for mood in BGM_MOODS:
        path = bgm_dir / f"{mood}.wav"
        sf.write(str(path), synthesize_bgm(mood, sample_rate), sample_rate, subtype="PCM_16")
        found[mood] = path
    logger.info("Musiques de substitution synthetisees dans %s : %s (a remplacer par de vraies musiques)", bgm_dir, ", ".join(BGM_MOODS))
    return found


def bgm_for_mood(mood: str, bgm_files: Mapping[str, Path]) -> Path | None:
    """Fichier de musique pour une ambiance : exact, sinon ``default``, sinon le premier disponible."""
    if not bgm_files:
        return None
    return bgm_files.get(mood) or bgm_files.get("default") or next(iter(bgm_files.values()))


__all__ = [
    "SFX_KINDS",
    "BGM_MOODS",
    "DEFAULT_SFX_DIR",
    "DEFAULT_BGM_DIR",
    "synthesize_sfx",
    "find_sfx",
    "ensure_default_sfx",
    "find_bgm",
    "synthesize_bgm",
    "ensure_default_bgm",
    "bgm_for_mood",
]
