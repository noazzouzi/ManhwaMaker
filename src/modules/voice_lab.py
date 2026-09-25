"""Banc d'essai des voix Kokoro : même texte, toutes les voix, un rapport HTML comparatif.

Sert à choisir la voix de la chaîne (:data:`~src.modules.tts_engine.LANGUAGE_VOICES`
fixe le mélange ``am_fenrir,am_michael`` par défaut) et, accessoirement, à mesurer le débit réel de Kokoro
sur cette machine (facteur temps réel, gain du parallélisme).

Chaque voix synthétise le même extrait, avec le même dictionnaire phonétique et la
même vitesse que la production. Le rapport HTML aligne un lecteur audio par voix,
la durée obtenue, le débit en mots par minute et le temps de calcul.
"""

from __future__ import annotations

import html
import logging
import threading
import time
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from pydantic import BaseModel, Field

from src.modules.tts_engine import (
    DEFAULT_PADDING_S,
    DEFAULT_SENTENCE_GAP_S,
    DEFAULT_SPEED,
    SAMPLE_RATE,
    KokoroTTS,
    TTSError,
    load_pronunciations,
    resolve_voice,
)

logger = logging.getLogger(__name__)

#: Voix anglaises du dépôt ``hexgrad/Kokoro-82M`` (``a`` = américain, ``b`` = britannique).
#: Préfixe ``f`` = féminin, ``m`` = masculin.
AMERICAN_FEMALE: tuple[str, ...] = (
    "af_heart", "af_alloy", "af_aoede", "af_bella", "af_jessica",
    "af_kore", "af_nicole", "af_nova", "af_river", "af_sarah", "af_sky",
)
AMERICAN_MALE: tuple[str, ...] = (
    "am_adam", "am_echo", "am_eric", "am_fenrir", "am_liam", "am_michael", "am_onyx", "am_puck", "am_santa",
)
BRITISH_FEMALE: tuple[str, ...] = ("bf_alice", "bf_emma", "bf_isabella", "bf_lily")
BRITISH_MALE: tuple[str, ...] = ("bm_daniel", "bm_fable", "bm_george", "bm_lewis")
ENGLISH_VOICES: tuple[str, ...] = AMERICAN_FEMALE + AMERICAN_MALE + BRITISH_FEMALE + BRITISH_MALE

#: Extrait par défaut : une accroche de récap, avec un nom propre et un chiffre pour
#: entendre la prononciation et le rythme.
DEFAULT_SAMPLE_TEXT: str = (
    "Seventy-nine attempts, three hundred thousand dead, and less than an hour before humanity is erased. "
    "A lone swordsman steps into the boss chamber, and the Red Dragon Destia laughs at him. "
    "He has done this before, and he already knows how it ends."
)


class VoiceSample(BaseModel):
    """Résultat de la synthèse d'un extrait par une voix."""

    voice: str
    lang_code: str
    file: str
    duration_s: float = Field(ge=0)
    compute_s: float = Field(ge=0)
    words: int = Field(ge=0)

    @property
    def words_per_minute(self) -> float:
        return self.words * 60 / self.duration_s if self.duration_s > 0 else 0.0

    @property
    def realtime_factor(self) -> float:
        """Secondes d'audio produites par seconde de calcul (plus c'est haut, plus c'est rapide)."""
        return self.duration_s / self.compute_s if self.compute_s > 0 else 0.0

    @property
    def gender(self) -> str:
        return "feminine" if self.voice[1:2] == "f" else "masculine"

    @property
    def accent(self) -> str:
        return {"a": "americain", "b": "britannique"}.get(self.voice[:1], self.voice[:1])


def _local_engine(store: threading.local, language: str, speed: float, pronunciations: dict[str, str]) -> KokoroTTS:
    """Un moteur Kokoro par thread (le pipeline Torch n'est pas partagé entre threads)."""
    engine = getattr(store, "engine", None)
    if engine is None:
        engine = KokoroTTS(language=language, speed=speed, pronunciations=pronunciations)
        store.engine = engine
    return engine


def compare_voices(
    voices: Sequence[str] = ENGLISH_VOICES,
    out_dir: str | Path = "output/voices",
    *,
    text: str = DEFAULT_SAMPLE_TEXT,
    language: str = "en",
    speed: float = DEFAULT_SPEED,
    workers: int = 1,
    pronunciations: dict[str, str] | None = None,
    engine_factory=None,
) -> list[VoiceSample]:
    """Synthétise le même extrait avec chaque voix et renvoie les mesures.

    Args:
        voices: identifiants Kokoro (``am_fenrir``, ``bf_emma``...).
        out_dir: dossier des WAV produits (un par voix).
        text: extrait à lire.
        language: langue (détermine le pipeline Kokoro).
        speed: vitesse de lecture.
        workers: synthèses simultanées (mesure aussi le gain du parallélisme CPU).
        pronunciations: dictionnaire phonétique (défaut : celui du projet).
        engine_factory: fabrique de moteur ``() -> KokoroTTS`` (tests).

    Raises:
        TTSError: aucune voix n'a pu être synthétisée.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    mapping = load_pronunciations() if pronunciations is None else pronunciations
    words = len(text.split())
    store = threading.local()
    lang_code, _ = resolve_voice(language)

    def synthesize(voice: str) -> VoiceSample | None:
        engine = engine_factory() if engine_factory is not None else _local_engine(store, language, speed, mapping)
        path = out_dir / f"{voice}.wav"
        spoken = engine.spoken_text(text)
        started = time.perf_counter()
        try:
            samples = engine.synthesize_text(spoken, voice)
        except Exception as exc:  # noqa: BLE001 - une voix absente ne doit pas tout arreter
            logger.warning("Voix %s ignoree : %s", voice, exc)
            return None
        compute = time.perf_counter() - started
        from src.modules.tts_engine import write_wav

        write_wav(path, samples, SAMPLE_RATE)
        duration = len(samples) / SAMPLE_RATE
        logger.info("Voix %-12s %5.1fs audio en %5.1fs (%.1fx), %3.0f mots/min", voice, duration, compute, duration / compute if compute else 0, words * 60 / duration if duration else 0)
        return VoiceSample(
            voice=voice, lang_code=lang_code, file=path.name, duration_s=duration, compute_s=compute, words=words,
        )

    started = time.perf_counter()
    if workers > 1:
        with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="kokoro") as pool:
            results = list(pool.map(synthesize, voices))
    else:
        results = [synthesize(voice) for voice in voices]
    samples = [s for s in results if s is not None]
    if not samples:
        raise TTSError("Aucune voix n'a pu etre synthetisee")
    elapsed = time.perf_counter() - started
    audio = sum(s.duration_s for s in samples)
    compute = sum(s.compute_s for s in samples)
    logger.info(
        "%d voix en %.0fs (%d en parallele) : %.0fs d'audio, %.0fs de calcul cumule, acceleration %.1fx",
        len(samples), elapsed, workers, audio, compute, compute / elapsed if elapsed else 0,
    )
    return samples


def build_voice_report(
    samples: Sequence[VoiceSample], path: str | Path, *, text: str = DEFAULT_SAMPLE_TEXT, speed: float = DEFAULT_SPEED
) -> Path:
    """Écrit un rapport HTML : un lecteur audio et les mesures par voix, triés par débit."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for sample in sorted(samples, key=lambda s: (s.accent, s.gender, s.voice)):
        rows.append(
            "<tr>"
            f'<td class="v">{html.escape(sample.voice)}</td>'
            f"<td>{html.escape(sample.accent)}</td><td>{html.escape(sample.gender)}</td>"
            f'<td class="n">{sample.duration_s:.1f} s</td>'
            f'<td class="n">{sample.words_per_minute:.0f}</td>'
            f'<td class="n">{sample.realtime_factor:.1f}x</td>'
            f'<td><audio controls preload="none" src="{html.escape(sample.file)}"></audio></td>'
            "</tr>"
        )
    fastest = max(samples, key=lambda s: s.words_per_minute)
    slowest = min(samples, key=lambda s: s.words_per_minute)
    document = f"""<!doctype html>
<html lang="fr"><head><meta charset="utf-8"><title>Voix Kokoro</title>
<style>
 body {{ font-family: system-ui, sans-serif; margin: 2rem; background: #14161a; color: #e8e8e8; }}
 h1 {{ font-size: 1.4rem; }}
 .sample {{ background: #1e222a; padding: 1rem; border-radius: 8px; max-width: 60rem; line-height: 1.5; }}
 table {{ border-collapse: collapse; margin-top: 1.5rem; }}
 th, td {{ padding: .45rem .8rem; border-bottom: 1px solid #2c313a; text-align: left; }}
 th {{ color: #9aa4b2; font-weight: 600; font-size: .85rem; text-transform: uppercase; }}
 td.n {{ text-align: right; font-variant-numeric: tabular-nums; }}
 td.v {{ font-weight: 600; }}
 audio {{ height: 2rem; }}
 p.meta {{ color: #9aa4b2; }}
</style></head><body>
<h1>Comparatif des voix Kokoro ({len(samples)} voix, vitesse {speed:g})</h1>
<p class="sample">{html.escape(text)}</p>
<p class="meta">Debit le plus rapide : <b>{html.escape(fastest.voice)}</b> ({fastest.words_per_minute:.0f} mots/min) ;
le plus lent : <b>{html.escape(slowest.voice)}</b> ({slowest.words_per_minute:.0f} mots/min).
La colonne « calcul » indique combien de secondes d'audio sont produites par seconde de CPU.</p>
<table><thead><tr><th>Voix</th><th>Accent</th><th>Genre</th><th>Duree</th><th>Mots/min</th><th>Calcul</th><th>Ecoute</th></tr></thead>
<tbody>{''.join(rows)}</tbody></table>
</body></html>"""
    path.write_text(document, encoding="utf-8")
    logger.info("Rapport des voix ecrit : %s", path)
    return path


def format_voice_table(samples: Sequence[VoiceSample]) -> str:
    """Tableau ASCII des voix (console Windows cp1252)."""
    lines = [f"{'voix':<13}{'accent':<13}{'genre':<11}{'duree':>7}{'mots/min':>10}{'calcul':>8}"]
    lines.append("-" * len(lines[0]))
    for sample in sorted(samples, key=lambda s: -s.words_per_minute):
        lines.append(
            f"{sample.voice:<13}{sample.accent:<13}{sample.gender:<11}"
            f"{sample.duration_s:>6.1f}s{sample.words_per_minute:>10.0f}{sample.realtime_factor:>7.1f}x"
        )
    return "\n".join(lines).encode("ascii", "replace").decode("ascii")


__all__ = [
    "AMERICAN_FEMALE",
    "AMERICAN_MALE",
    "BRITISH_FEMALE",
    "BRITISH_MALE",
    "ENGLISH_VOICES",
    "DEFAULT_SAMPLE_TEXT",
    "VoiceSample",
    "compare_voices",
    "build_voice_report",
    "format_voice_table",
]
