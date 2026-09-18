"""Module 4 — Kokoro TTS Engine : des scènes narrées aux segments audio.

Pipeline :

1. :func:`prepare_text` nettoie la narration (guillemets typographiques,
   caractères invisibles, espaces) et :func:`apply_pronunciations` applique le
   **dictionnaire de remplacement phonétique** (garde-fou n°3 du PRD) aux noms
   propres, avant toute synthèse ;
2. :class:`KokoroTTS` synthétise chaque scène avec ``KPipeline`` (100 % local,
   24 kHz), concatène les morceaux produits par Kokoro et ajoute
   :data:`DEFAULT_PADDING_S` = 0,18 s de silence en fin de segment ;
3. chaque segment est écrit en WAV 16 bits et décrit par un
   :class:`~src.models.audio.SceneAudio` dont ``duration_s`` est la durée
   **exacte** (nombre d'échantillons / fréquence) qui pilotera la timeline ;
4. :meth:`KokoroTTS.synthesize_analysis` traite toutes les scènes narratives d'un
   :class:`~src.models.scene.ChapterAnalysis` (le remplissage est exclu par
   défaut), écrit ``voiceover.json`` et, optionnellement, la voix off complète
   concaténée pour écoute.

Langue : anglais américain par défaut (voix masculine ``am_puck``, 161 mots/min au
banc d'essai de :mod:`src.modules.voice_lab`), cf.
:data:`LANGUAGE_VOICES` pour les autres langues. Les pipelines non anglais et
les mots hors dictionnaire s'appuient sur espeak-ng (fourni par
``espeakng-loader`` sur Windows).

Le pipeline Kokoro est injectable (``KokoroTTS(pipeline=...)``) : les tests
utilisent un pipeline factice, aucun modèle n'est chargé. Le vrai pipeline est
créé à la première synthèse (import tardif de ``kokoro``), les poids
(~330 Mo) étant téléchargés depuis Hugging Face au premier usage puis mis en cache.
"""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

from src.models.audio import SceneAudio, VoiceoverManifest
from src.models.scene import ChapterAnalysis, Scene
from src.utils.config import DEFAULT_NARRATION_LANGUAGE, PROJECT_ROOT

logger = logging.getLogger(__name__)

# --- Paramètres par défaut (cf. PRD, Module 4) ---------------------------------
#: Identifiant Hugging Face du modèle Kokoro.
KOKORO_REPO_ID: str = "hexgrad/Kokoro-82M"
#: Fréquence d'échantillonnage de Kokoro (Hz).
SAMPLE_RATE: int = 24_000
#: Silence ajouté en fin de chaque segment (secondes) : 0,18 s, assez pour ne jamais
#: chevaucher la phrase suivante tout en resserrant le rythme entre les scènes.
DEFAULT_PADDING_S: float = 0.18
#: Silence inséré entre deux phrases d'un même segment (secondes) : chaque phrase est
#: synthétisée séparément puis assemblée avec cette pause. 0,2 s suffit à marquer la
#: respiration sans casser le rythme (0,4 s donnait une narration traînante).
DEFAULT_SENTENCE_GAP_S: float = 0.2
DEFAULT_SPEED: float = 1.0
#: Fichier du dictionnaire phonétique par défaut (racine du projet).
DEFAULT_PRONUNCIATIONS_FILE: Path = PROJECT_ROOT / "config" / "pronunciations.json"
#: Nom du manifeste écrit à côté des WAV.
MANIFEST_NAME: str = "voiceover.json"
#: Nom du WAV de la voix off complète.
FULL_VOICEOVER_NAME: str = "voiceover_full.wav"

#: Langue de narration -> (code de pipeline Kokoro, voix par défaut).
LANGUAGE_VOICES: dict[str, tuple[str, str]] = {
    "en": ("a", "am_puck"),
    "en-us": ("a", "am_puck"),
    "en-gb": ("b", "bf_emma"),
    "fr": ("f", "ff_siwis"),
    "es": ("e", "ef_dora"),
    "it": ("i", "if_sara"),
    "pt": ("p", "pf_dora"),
    "hi": ("h", "hf_alpha"),
    "ja": ("j", "jf_alpha"),
    "zh": ("z", "zf_xiaobei"),
}

#: Guillemets et tirets typographiques -> équivalents ASCII (prononciation neutre).
_TYPOGRAPHY: dict[int, str] = {
    0x2018: "'", 0x2019: "'", 0x201A: "'", 0x201B: "'",
    0x201C: '"', 0x201D: '"', 0x201E: '"', 0x201F: '"',
    0x2013: "-", 0x2014: " - ", 0x2026: "...", 0x00A0: " ",
}
#: Caractères de largeur nulle supprimés.
_ZERO_WIDTH: dict[int, None] = dict.fromkeys((0x200B, 0x200C, 0x200D, 0xFEFF), None)
_PHONEME_VALUE = re.compile(r"^/.+/$")
#: Frontière de phrase : ponctuation forte suivie d'une majuscule, d'un chiffre ou d'un guillemet
#: (les points de suspension suivis d'une minuscule ne coupent pas).
_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(\[])")


class TTSError(RuntimeError):
    """Erreur définitive du moteur TTS (modèle indisponible, synthèse vide...)."""


# --- Texte ---------------------------------------------------------------------------
def prepare_text(text: str) -> str:
    """Nettoie une narration avant synthèse.

    - guillemets/tirets typographiques remplacés par leurs équivalents ASCII ;
    - caractères de largeur nulle supprimés, espaces compactés ;
    - ponctuation finale ajoutée si absente (intonation de fin de phrase).

    Retourne ``""`` si le texte ne contient aucun caractère alphanumérique.
    """
    cleaned = text.translate(_ZERO_WIDTH).translate(_TYPOGRAPHY)
    cleaned = " ".join(cleaned.split())
    if not any(ch.isalnum() for ch in cleaned):
        return ""
    if cleaned[-1] not in ".!?":
        cleaned += "."
    return cleaned


def split_sentences(text: str) -> list[str]:
    """Découpe un texte préparé en phrases (voir :data:`_SENTENCE_BOUNDARY`)."""
    return [s.strip() for s in _SENTENCE_BOUNDARY.split(text.strip()) if s.strip()]


def load_pronunciations(path: str | Path | None = None) -> dict[str, str]:
    """Charge le dictionnaire phonétique (``{nom: remplacement}``).

    Args:
        path: fichier JSON ; ``None`` = :data:`DEFAULT_PRONUNCIATIONS_FILE` s'il
            existe, sinon dictionnaire vide. Les clés commençant par ``_`` sont
            ignorées (commentaires).

    Raises:
        FileNotFoundError: chemin explicite inexistant.
        ValueError: JSON invalide ou valeurs non textuelles.
    """
    if path is None:
        if not DEFAULT_PRONUNCIATIONS_FILE.is_file():
            return {}
        path = DEFAULT_PRONUNCIATIONS_FILE
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Dictionnaire phonetique introuvable : {path}")
    try:
        data = json.loads(path.read_text(encoding="utf-8-sig"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"{path} n'est pas un JSON valide : {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError(f"{path} doit contenir un objet {{nom: remplacement}}")
    mapping: dict[str, str] = {}
    for key, value in data.items():
        if str(key).startswith("_"):
            continue
        if not isinstance(value, str) or not str(key).strip():
            raise ValueError(f"Entree invalide dans {path} : {key!r} -> {value!r}")
        mapping[str(key).strip()] = value.strip()
    logger.debug("%d remplacement(s) phonetique(s) charges depuis %s", len(mapping), path)
    return mapping


def compile_pronunciations(mapping: Mapping[str, str]) -> list[tuple[re.Pattern[str], str]]:
    """Compile le dictionnaire en motifs (mots entiers, insensibles à la casse).

    Les clés les plus longues sont appliquées en premier (``"Kang JinHyeok"``
    avant ``"JinHyeok"``). Une valeur ``/phonemes/`` devient la syntaxe Kokoro
    ``[mot](/phonemes/)`` ; sinon la valeur remplace le mot tel quel.
    """
    compiled: list[tuple[re.Pattern[str], str]] = []
    for key in sorted(mapping, key=len, reverse=True):
        value = mapping[key]
        pattern = re.compile(rf"(?<!\w){re.escape(key)}(?!\w)", re.IGNORECASE)
        replacement = f"[{key}]({value})" if _PHONEME_VALUE.match(value) else value
        compiled.append((pattern, replacement))
    return compiled


def apply_pronunciations(
    text: str, mapping: Mapping[str, str] | Sequence[tuple[re.Pattern[str], str]]
) -> str:
    """Applique le dictionnaire phonétique à un texte (voir :func:`compile_pronunciations`)."""
    compiled = mapping if isinstance(mapping, list) else compile_pronunciations(mapping)
    for pattern, replacement in compiled:
        text = pattern.sub(lambda _m, r=replacement: r, text)
    return text


def strip_silence(
    samples: np.ndarray, sample_rate: int, max_silence_s: float, *, threshold_db: float = -45.0
) -> np.ndarray:
    """Raccourcit les silences internes dépassant ``max_silence_s``, sans toucher au reste.

    Travaille directement sur les échantillons plutôt qu'avec ``pydub`` : la synthèse nous
    les livre déjà en mémoire, et un aller-retour par un fichier n'apporterait rien.
    ``pydub.speedup`` serait de toute façon à proscrire pour accélérer la voix — c'est un
    rééchantillonnage, qui monte la hauteur ; l'accélération vient du paramètre ``speed``
    de Kokoro, qui préserve le timbre.

    Args:
        samples: audio mono float32.
        sample_rate: fréquence d'échantillonnage.
        max_silence_s: durée de silence conservée là où il en dépasse.
        threshold_db: niveau (dBFS) sous lequel une fenêtre est jugée silencieuse.

    Returns:
        Un nouvel audio, jamais plus long que l'original.
    """
    if max_silence_s is None or max_silence_s < 0 or samples.size == 0:
        return samples
    keep = max(1, int(round(max_silence_s * sample_rate)))
    threshold = 10.0 ** (threshold_db / 20.0)
    # Enveloppe par fenetres courtes : un seuil pixel par pixel couperait dans les
    # passages a faible energie (fins de mots, consonnes sourdes).
    frame = max(1, sample_rate // 200)  # 5 ms
    n_frames = len(samples) // frame
    if n_frames == 0:
        return samples
    blocks = samples[: n_frames * frame].reshape(n_frames, frame)
    loud = np.abs(blocks).max(axis=1) >= threshold
    mask = np.repeat(loud, frame)
    mask = np.concatenate([mask, np.ones(len(samples) - len(mask), dtype=bool)])

    pieces: list[np.ndarray] = []
    index = 0
    removed = 0
    while index < len(samples):
        if mask[index]:
            stop = index + int(np.argmin(mask[index:])) if not mask[index:].all() else len(samples)
            pieces.append(samples[index:stop])
            index = stop
            continue
        stop = index + int(np.argmax(mask[index:])) if mask[index:].any() else len(samples)
        length = stop - index
        pieces.append(samples[index : index + min(length, keep)])
        removed += max(0, length - keep)
        index = stop
    if not removed:
        return samples
    logger.debug("Silences rognes : %.2fs retires", removed / sample_rate)
    return np.concatenate(pieces).astype(np.float32)


def resolve_voice(language: str = DEFAULT_NARRATION_LANGUAGE, voice: str | None = None) -> tuple[str, str]:
    """Retourne ``(lang_code Kokoro, voix)`` pour une langue de narration.

    Raises:
        TTSError: langue sans pipeline Kokoro connu (préciser ``voice`` ne suffit pas).
    """
    key = language.lower()
    if key not in LANGUAGE_VOICES:
        raise TTSError(
            f"Langue {language!r} sans pipeline Kokoro ; langues connues : {', '.join(LANGUAGE_VOICES)}"
        )
    lang_code, default_voice = LANGUAGE_VOICES[key]
    return lang_code, (voice or default_voice)


# --- Audio ---------------------------------------------------------------------------
def _to_numpy_audio(audio: Any) -> np.ndarray:
    """Convertit un morceau audio Kokoro (tenseur torch ou tableau) en float32 mono."""
    if audio is None:
        return np.zeros(0, dtype=np.float32)
    if hasattr(audio, "detach"):
        audio = audio.detach().cpu().numpy()
    array = np.asarray(audio, dtype=np.float32)
    if array.ndim > 1:
        array = array.reshape(array.shape[0], -1).mean(axis=1)
    return np.ascontiguousarray(array)


def _chunk_audio(result: Any) -> Any:
    """Extrait l'audio d'un résultat de ``KPipeline`` (objet ``Result`` ou tuple)."""
    audio = getattr(result, "audio", None)
    if audio is not None or hasattr(result, "graphemes"):
        return audio
    if isinstance(result, (tuple, list)) and len(result) >= 3:
        return result[2]
    return result


def write_wav(path: str | Path, samples: np.ndarray, sample_rate: int = SAMPLE_RATE) -> Path:
    """Écrit un WAV mono 16 bits PCM et renvoie le chemin."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    clipped = np.clip(np.asarray(samples, dtype=np.float32), -1.0, 1.0)
    sf.write(str(path), clipped, sample_rate, subtype="PCM_16")
    return path


def read_wav(path: str | Path) -> tuple[np.ndarray, int]:
    """Lit un WAV en float32 mono ; renvoie ``(échantillons, fréquence)``."""
    samples, sample_rate = sf.read(str(path), dtype="float32", always_2d=True)
    return samples.mean(axis=1).astype(np.float32), int(sample_rate)


def concat_wavs(paths: Iterable[str | Path], out_path: str | Path, sample_rate: int = SAMPLE_RATE) -> Path:
    """Concatène des WAV (même fréquence) en un seul fichier.

    Raises:
        TTSError: fréquences d'échantillonnage différentes.
    """
    pieces: list[np.ndarray] = []
    for path in paths:
        samples, rate = read_wav(path)
        if rate != sample_rate:
            raise TTSError(f"{path} : {rate} Hz au lieu de {sample_rate} Hz")
        pieces.append(samples)
    joined = np.concatenate(pieces) if pieces else np.zeros(0, dtype=np.float32)
    return write_wav(out_path, joined, sample_rate)


def export_mp3(wav_path: str | Path, mp3_path: str | Path) -> Path:
    """Convertit un WAV en MP3 (écoute, partage) via la libsndfile embarquée par ``soundfile``.

    Le WAV reste le format de référence pour le montage ; le MP3 est un aperçu
    léger (~6 fois plus petit).

    Raises:
        TTSError: encodage MP3 non supporté par la libsndfile installée.
    """
    samples, rate = read_wav(wav_path)
    mp3_path = Path(mp3_path)
    mp3_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        sf.write(str(mp3_path), samples, rate, format="MP3")
    except (RuntimeError, ValueError, TypeError) as exc:
        raise TTSError(f"Export MP3 non supporte par libsndfile : {exc}") from exc
    logger.info("MP3 ecrit : %s (%d Ko)", mp3_path, mp3_path.stat().st_size // 1024)
    return mp3_path


# --- Moteur ----------------------------------------------------------------------------
def build_pipeline(lang_code: str, repo_id: str = KOKORO_REPO_ID) -> Any:
    """Crée le ``KPipeline`` Kokoro (import tardif, chargement du modèle).

    Raises:
        TTSError: kokoro non installé ou modèle impossible à charger.
    """
    try:
        from kokoro import KPipeline
    except ImportError as exc:
        raise TTSError(
            "Le paquet 'kokoro' n'est pas installe : .\\.venv\\Scripts\\python.exe -m pip install kokoro soundfile"
        ) from exc
    try:
        return KPipeline(lang_code=lang_code, repo_id=repo_id)
    except Exception as exc:  # noqa: BLE001 - erreurs reseau / HF / torch variees
        raise TTSError(f"Impossible de charger le modele Kokoro ({repo_id}, lang_code={lang_code}) : {exc}") from exc


class KokoroTTS:
    """Synthèse vocale locale des scènes avec Kokoro.

    Args:
        language: langue de la narration (détermine le pipeline et la voix par défaut).
        voice: voix Kokoro (``None`` = voix par défaut de la langue).
        speed: vitesse de lecture (1.0 = normale).
        padding_s: silence ajouté en fin de segment (secondes).
        sentence_gap_s: silence inséré entre deux phrases d'un segment (secondes, 0 = aucun :
            le paragraphe est alors synthétisé d'un bloc).
        pronunciations: dictionnaire phonétique (``dict``), chemin JSON, ou ``None``
            pour le fichier par défaut ``config/pronunciations.json`` (s'il existe).
        pipeline: pipeline injecté (tests) ; sinon créé à la première synthèse.
        repo_id: modèle Hugging Face.
    """

    def __init__(
        self,
        *,
        language: str = DEFAULT_NARRATION_LANGUAGE,
        voice: str | None = None,
        speed: float = DEFAULT_SPEED,
        padding_s: float = DEFAULT_PADDING_S,
        sentence_gap_s: float = DEFAULT_SENTENCE_GAP_S,
        pronunciations: Mapping[str, str] | str | Path | None = None,
        pipeline: Any | None = None,
        repo_id: str = KOKORO_REPO_ID,
        sample_rate: int = SAMPLE_RATE,
        max_silence_s: float | None = None,
        silence_threshold_db: float = -45.0,
    ) -> None:
        if speed <= 0:
            raise ValueError("speed doit etre > 0")
        if padding_s < 0:
            raise ValueError("padding_s doit etre >= 0")
        if sentence_gap_s < 0:
            raise ValueError("sentence_gap_s doit etre >= 0")
        self.language = language
        self.lang_code, self.voice = resolve_voice(language, voice)
        self.speed = speed
        self.padding_s = padding_s
        self.sentence_gap_s = sentence_gap_s
        self.max_silence_s = max_silence_s
        self.silence_threshold_db = silence_threshold_db
        self.repo_id = repo_id
        self.sample_rate = sample_rate
        if isinstance(pronunciations, Mapping):
            mapping = dict(pronunciations)
        else:
            mapping = load_pronunciations(pronunciations)
        self.pronunciations = mapping
        self._compiled = compile_pronunciations(mapping)
        self._pipeline = pipeline

    @property
    def pipeline(self) -> Any:
        """Pipeline Kokoro (créé à la demande)."""
        if self._pipeline is None:
            logger.info("Chargement du modele Kokoro %s (lang_code=%s)", self.repo_id, self.lang_code)
            self._pipeline = build_pipeline(self.lang_code, self.repo_id)
        return self._pipeline

    def spoken_text(self, text: str) -> str:
        """Texte effectivement envoyé au modèle (nettoyage + remplacements phonétiques)."""
        return apply_pronunciations(prepare_text(text), self._compiled)

    def synthesize(self, text: str, voice: str | None = None) -> np.ndarray:
        """Synthétise un texte déjà préparé ; renvoie l'audio float32 mono sans silence.

        Args:
            text: texte préparé (voir :meth:`spoken_text`).
            voice: voix Kokoro pour cet appel (défaut : celle du moteur). Le même
                pipeline sert toutes les voix de la langue : inutile de recharger le modèle.

        Raises:
            TTSError: texte vide ou aucun audio produit.
        """
        if not text.strip():
            raise TTSError("Texte vide : rien a synthetiser")
        pieces: list[np.ndarray] = []
        for result in self.pipeline(text, voice=voice or self.voice, speed=self.speed, split_pattern=r"\n+"):
            chunk = _to_numpy_audio(_chunk_audio(result))
            if chunk.size:
                pieces.append(chunk)
        if not pieces:
            raise TTSError(f"Kokoro n'a produit aucun audio pour : {text[:80]!r}")
        return np.concatenate(pieces)

    def synthesize_text(self, text: str, voice: str | None = None) -> np.ndarray:
        """Synthétise un texte phrase par phrase, avec ``sentence_gap_s`` de silence entre les phrases.

        Avec ``sentence_gap_s = 0`` le texte est synthétisé d'un seul bloc.

        Args:
            text: texte préparé.
            voice: voix Kokoro pour cet appel (défaut : celle du moteur).

        Raises:
            TTSError: texte vide ou aucun audio produit.
        """
        sentences = split_sentences(text) if self.sentence_gap_s > 0 else [text]
        if not sentences:
            raise TTSError("Texte vide : rien a synthetiser")
        gap = np.zeros(int(round(self.sentence_gap_s * self.sample_rate)), dtype=np.float32)
        pieces: list[np.ndarray] = []
        for i, sentence in enumerate(sentences):
            if i > 0 and gap.size:
                pieces.append(gap)
            pieces.append(self.synthesize(sentence, voice))
        return np.concatenate(pieces)

    def synthesize_scene(self, scene: Scene, out_path: str | Path) -> SceneAudio:
        """Synthétise une scène, écrit le WAV (parole + silence) et renvoie ses métadonnées.

        Raises:
            TTSError: narration vide après nettoyage, ou synthèse vide.
        """
        text = self.spoken_text(scene.narration)
        if not text:
            raise TTSError(f"Scene {scene.index} : narration vide apres nettoyage")
        speech = self.synthesize_text(text)
        if self.max_silence_s is not None:
            speech = strip_silence(
                speech, self.sample_rate, self.max_silence_s, threshold_db=self.silence_threshold_db
            )
        padding = np.zeros(int(round(self.padding_s * self.sample_rate)), dtype=np.float32)
        samples = np.concatenate([speech, padding])
        path = write_wav(out_path, samples, self.sample_rate)
        audio = SceneAudio(
            scene_index=scene.index,
            file=path.name,
            duration_s=len(samples) / self.sample_rate,
            speech_s=len(speech) / self.sample_rate,
            sample_rate=self.sample_rate,
            text=text,
            emotion=scene.emotion,
            is_filler=scene.is_filler,
        )
        logger.info(
            "Scene %d : %.2fs de parole + %.2fs de silence -> %s",
            scene.index, audio.speech_s, self.padding_s, path.name,
        )
        return audio

    def synthesize_analysis(
        self,
        analysis: ChapterAnalysis,
        out_dir: str | Path,
        *,
        include_filler: bool = False,
        full_file: str | None = FULL_VOICEOVER_NAME,
        prefix: str = "scene",
    ) -> VoiceoverManifest:
        """Synthétise toutes les scènes d'un chapitre et écrit ``voiceover.json``.

        Args:
            analysis: scènes narrées.
            out_dir: dossier des WAV (créé si besoin).
            include_filler: synthétiser aussi les scènes de remplissage.
            full_file: nom du WAV concaténé (``None`` pour ne pas l'écrire).
            prefix: préfixe des fichiers ``<prefix>_<index:03d>.wav``.

        Raises:
            ValueError: aucune scène à synthétiser.
        """
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        scenes = analysis.scenes if include_filler else analysis.story_scenes()
        if not scenes:
            raise ValueError("Aucune scene a synthetiser")
        logger.info(
            "Synthese de %d scene(s) avec la voix %s (%s), vitesse %.2f, padding %.2fs, pause entre phrases %.2fs",
            len(scenes), self.voice, self.lang_code, self.speed, self.padding_s, self.sentence_gap_s,
        )
        items: list[SceneAudio] = []
        for scene in scenes:
            path = out_dir / f"{prefix}_{scene.index:03d}.wav"
            items.append(self.synthesize_scene(scene, path))

        manifest = VoiceoverManifest(
            language=self.language,
            lang_code=self.lang_code,
            voice=self.voice,
            speed=self.speed,
            padding_s=self.padding_s,
            sentence_gap_s=self.sentence_gap_s,
            sample_rate=self.sample_rate,
            model=self.repo_id,
            items=items,
            total_duration_s=sum(item.duration_s for item in items),
        )
        if full_file:
            concat_wavs((out_dir / item.file for item in items), out_dir / full_file, self.sample_rate)
            manifest.full_file = full_file
        save_manifest(manifest, out_dir / MANIFEST_NAME)
        return manifest


# --- Manifeste ------------------------------------------------------------------------
def save_manifest(manifest: VoiceoverManifest, path: str | Path) -> Path:
    """Écrit le manifeste en JSON (UTF-8) et renvoie le chemin."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(manifest.model_dump_json(indent=2), encoding="utf-8")
    logger.info(
        "Manifeste audio ecrit : %s (%d segment(s), %.1fs)",
        path, manifest.n_items, manifest.total_duration_s,
    )
    return path


def load_manifest(path: str | Path) -> VoiceoverManifest:
    """Relit un manifeste écrit par :func:`save_manifest`."""
    return VoiceoverManifest.model_validate_json(Path(path).read_text(encoding="utf-8"))


def format_manifest(manifest: VoiceoverManifest) -> str:
    """Rendu ASCII multi-lignes du manifeste (console Windows cp1252)."""
    lines = [
        "=" * 72,
        f"Voix       : {manifest.voice} (pipeline {manifest.lang_code}, langue {manifest.language})",
        f"Vitesse    : {manifest.speed:.2f}   Silence fin : {manifest.padding_s:.2f}s   "
        f"Pause phrases : {manifest.sentence_gap_s:.2f}s   {manifest.sample_rate} Hz",
        f"Segments   : {manifest.n_items}   Duree totale : {manifest.total_duration_s:.1f}s",
        "-" * 72,
        f"{'SCENE':>5}  {'PAROLE':>7}  {'TOTAL':>7}  {'MOTS':>4}  FICHIER",
    ]
    for item in manifest.items:
        words = len(item.text.split())
        lines.append(
            f"{item.scene_index:>5}  {item.speech_s:>6.2f}s  {item.duration_s:>6.2f}s  {words:>4}  {item.file}"
        )
    lines.append("=" * 72)
    return "\n".join(lines).encode("ascii", "replace").decode("ascii")


__all__ = [
    "KOKORO_REPO_ID",
    "SAMPLE_RATE",
    "DEFAULT_PADDING_S",
    "DEFAULT_SENTENCE_GAP_S",
    "split_sentences",
    "DEFAULT_SPEED",
    "DEFAULT_PRONUNCIATIONS_FILE",
    "MANIFEST_NAME",
    "FULL_VOICEOVER_NAME",
    "LANGUAGE_VOICES",
    "TTSError",
    "KokoroTTS",
    "prepare_text",
    "load_pronunciations",
    "compile_pronunciations",
    "apply_pronunciations",
    "resolve_voice",
    "build_pipeline",
    "write_wav",
    "read_wav",
    "concat_wavs",
    "export_mp3",
    "save_manifest",
    "load_manifest",
    "format_manifest",
]
