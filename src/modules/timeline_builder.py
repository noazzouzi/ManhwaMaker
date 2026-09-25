"""Construction de la timeline (Module 5, étape 1) : scènes + audio → plan de montage.

Règles :

- seules les scènes qui ont un segment audio sont montées (le remplissage
  ``is_filler`` n'a pas de voix off, il est donc naturellement exclu) ;
- la durée d'affichage d'une scène est **exactement** la durée de son segment
  audio (silence de fin inclus) ; chaque case clé reçoit d'abord
  ``min_clip_s`` puis le reste au prorata des hauteurs ; si la scène a trop de
  cases pour sa durée, les plus petites sont écartées ;
- mouvement : ``punch_in`` pour les cases ``action_heavy``, sinon ``ken_burns`` ;
  dans les deux cas le zoom ne dépasse jamais :data:`MAX_ZOOM` (105 %) de la
  taille native, pour ne jamais pixelliser ;
- sous-titres : narration d'origine découpée en blocs de 2 à 4 mots répartis sur
  la durée de parole au prorata des caractères ;
- design sonore : un bruitage court au début de chaque case des scènes
  ``action`` (cycle impact / swoosh / roar / swoosh) et un ``impact`` au début
  de chaque case ``punch_in`` des autres scènes, musique de fond par ambiance
  (``calm`` / ``tense`` / ``action``) à −22 dB, bouclée, avec fondu enchaîné à
  chaque changement d'ambiance ;
- ``max_duration_s`` tronque la timeline (prévisualisation de la première minute).
"""

from __future__ import annotations

import json
import logging
import math
import re
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np

from src.models.audio import VoiceoverManifest
from src.models.scene import ChapterAnalysis
from src.models.format_profile import FormatProfile
from src.models.timeline import (
    DEFAULT_BGM_GAIN_DB,
    AudioClip,
    BgmClip,
    ClipTransition,
    CueAnimation,
    PanelClip,
    SfxClip,
    SubtitleCue,
    Timeline,
    VfxClip,
    WordTiming,
)
from src.modules.format_factory import VideoConfigFactory
from src.modules.framing import make_framing
from src.modules.pacing import content_height, make_pacing
from src.modules.tts_engine import prepare_text
from src.utils.audio_assets import bgm_for_mood

logger = logging.getLogger(__name__)

DEFAULT_WIDTH: int = 1920
DEFAULT_HEIGHT: int = 1080
DEFAULT_FPS: int = 60
#: Poids minimal d'une case dans la répartition du temps (équivalent hauteur en px).
DEFAULT_MIN_PANEL_WEIGHT: int = 400
#: Durée minimale d'affichage d'une case clé (secondes).
DEFAULT_MIN_CLIP_S: float = 2.5
#: Zoom maximal sur une case, relatif à sa taille native (1,05 = jamais plus de 105 %).
MAX_ZOOM: float = 1.05
#: Amplitude du zoom Ken Burns (100 % → 105 % sur la durée du clip).
KEN_BURNS_ZOOM: float = 0.05
#: Zoom d'impact (punch-in) : 105 % atteint en 0,2 s, puis maintenu.
PUNCH_IN_ZOOM: float = 0.05
PUNCH_IN_S: float = 0.2
#: Fondu enchaîné entre deux ambiances musicales (secondes).
DEFAULT_BGM_CROSSFADE_S: float = 1.5
#: Émotions de scène qui déclenchent des bruitages aux transitions de cases.
SFX_EMOTIONS: frozenset[str] = frozenset({"action"})
#: Cycle des bruitages dans une scène d'action (par ordre de case).
SFX_CYCLE: tuple[str, ...] = ("impact", "swoosh", "roar", "swoosh")
DEFAULT_SFX_GAIN_DB: float = -12.0
#: Ambiance musicale par émotion de scène.
MOOD_BY_EMOTION: dict[str, str] = {
    "neutral": "calm", "calm": "calm", "happy": "calm", "romance": "calm", "humor": "calm",
    "tension": "tense", "mystery": "tense", "fear": "tense", "sad": "tense",
    "action": "action", "epic": "action",
}
#: Nombre maximal de mots par bloc de sous-titre (blocs de 2 à 4 mots).
DEFAULT_MAX_SUBTITLE_WORDS: int = 4
#: Durée minimale d'affichage d'un bloc (secondes).
MIN_CUE_DURATION_S: float = 0.25

# --- Dynamisme du montage --------------------------------------------------------------------
#: Intensité par défaut (voir :data:`~src.models.timeline.DynamicsLevel`).
DEFAULT_DYNAMICS: str = "punchy"

#: Animation d'entrée d'un bloc de sous-titre, par émotion de scène (catalogue CapCut).
#: Les blocs faisant 2 à 4 mots, l'entrée joue déjà le rôle du « pop-up mot par mot » :
#: pas besoin de karaoké, qui demanderait des blocs beaucoup plus longs.
CUE_INTRO_BY_EMOTION: dict[str, str] = {
    "action": "弹入",        # rebond marqué
    "tension": "故障",       # glitch
    "fear": "故障",
    "epic": "放大",          # agrandissement
    "mystery": "模糊",       # apparition floue
    "sad": "渐显",           # fondu doux
    "romance": "渐显",
    "calm": "渐显",
    "happy": "弹入",
    "humor": "弹入",
    "neutral": "逐字",       # mot à mot
}
#: Animation d'entrée de repli quand l'émotion est inconnue.
DEFAULT_CUE_INTRO: str = "逐字"
#: Animation en boucle, réservée aux scènes intenses (ailleurs elle fatigue l'œil).
CUE_LOOP_BY_EMOTION: dict[str, str] = {"action": "心跳", "tension": "颤抖", "fear": "颤抖"}
#: Part de la durée d'un bloc consacrée à son animation d'entrée, et bornes absolues.
CUE_ANIMATION_RATIO: float = 0.35
MIN_CUE_ANIMATION_S: float = 0.12
MAX_CUE_ANIMATION_S: float = 0.40

#: Émotions qui déclenchent une transition au changement de scène.
TRANSITION_EMOTIONS: frozenset[str] = frozenset({"action", "tension"})
#: Transitions candidates par émotion (catalogue CapCut), parcourues cycliquement pour
#: éviter la répétition mécanique d'un même effet sur tout un chapitre.
TRANSITION_BY_EMOTION: dict[str, tuple[str, ...]] = {
    "action": ("快速挥动", "甩鞭转场", "高速滑动"),   # whip pan, coup de fouet, glissement rapide
    "tension": ("故障", "色差故障", "闪黑"),         # glitch, glitch chromatique, flash noir
}
#: Transition de repli si l'émotion n'a pas de liste dédiée.
DEFAULT_TRANSITION: str = "叠化"
#: Durée visée, bornes absolues, et part maximale de la plus courte des deux cases.
#: Les défauts du catalogue CapCut vont jusqu'à 2 s : bien trop long ici, puisque
#: 1131 des 1137 transitions **empiètent** sur les cases voisines.
DEFAULT_TRANSITION_S: float = 0.30
MIN_TRANSITION_S: float = 0.20
MAX_TRANSITION_S: float = 0.40
TRANSITION_MAX_CLIP_RATIO: float = 0.15

#: Effet superposé par émotion de scène, et effet CapCut intégré correspondant.
VFX_KIND_BY_EMOTION: dict[str, str] = {"action": "speed_lines", "epic": "glow", "mystery": "rain"}
VFX_EFFECT_BY_KIND: dict[str, str] = {
    "speed_lines": "冲刺",    # lignes de vitesse
    "glow": "光晕",          # halo lumineux
    "rain": "下雨",          # pluie
    "sparks": "星火",        # étincelles
}
#: Durée minimale d'une scène pour mériter un effet superposé (secondes).
MIN_VFX_SCENE_S: float = 2.0
DEFAULT_VFX_OPACITY: float = 0.85

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")


class TimelineError(RuntimeError):
    """Timeline impossible à construire (aucune scène montable, fichiers manquants...)."""


def expand_panel_ids(
    scenes_panel_ids: Sequence[Sequence[int]], available: Sequence[int]
) -> list[list[int]]:
    """Étend chaque scène aux cases **non retenues** qui la suivent, dans l'ordre de lecture.

    L'analyse ne garde que 1 à 4 cases clés par paragraphe et écarte les autres : sur un
    chapitre mesuré, un peu plus de la moitié des cases découpées n'atteignaient jamais
    l'écran. Les deux formats s'étendent donc aux cases laissées de côté (drapeau
    ``PacingRules.expand_to_unused_panels``) : le format court parce qu'il réclame bien
    plus de plans qu'une scène n'a de cases clés (22 contre 3 sur ce chapitre) et que
    rogner huit cadrages dans la même case revient à montrer huit fois le même dessin ;
    le format long parce qu'afficher une case de plus vaut mieux que la laisser au rebut.

    Chaque scène reçoit donc la plage contiguë allant de sa première case clé à la
    première case clé de la scène suivante. Les plages sont disjointes et ordonnées, les
    numéros de cases clés étant eux-mêmes croissants d'une scène à l'autre.

    Args:
        scenes_panel_ids: cases clés de chaque scène montée, dans l'ordre.
        available: toutes les cases existantes, triées.

    Returns:
        Une liste de cases par scène, cases clés comprises.
    """
    panels = sorted(available)
    if not panels:
        return [list(ids) for ids in scenes_panel_ids]
    starts = [min(ids) if ids else None for ids in scenes_panel_ids]
    result: list[list[int]] = []
    for index, ids in enumerate(scenes_panel_ids):
        if not ids:
            result.append([])
            continue
        low = min(ids)
        following = next((s for s in starts[index + 1:] if s is not None), None)
        high = following if following is not None else panels[-1] + 1
        widened = [pid for pid in panels if low <= pid < high]
        result.append(widened or list(ids))
    return result


def _panel_loader(panels_dir: str | Path):
    """``entrée de panels.json -> image RGB``, pour l'analyse de saillance du mode SHORT.

    Les images ne sont chargées **qu'à la demande** : le mode long n'en ouvre aucune, et
    la stratégie de cadrage met en cache le profil de chaque case plutôt que ses pixels.
    """
    root = Path(panels_dir)

    def load(panel: Mapping):
        from PIL import Image

        path = root / str(panel["file"])
        if not path.is_file():
            return None
        with Image.open(path) as image:
            return np.asarray(image.convert("RGB"))

    return load


def load_panels_meta(panels_dir: str | Path) -> list[dict]:
    """Lit ``panels.json`` (métadonnées seules, sans charger les images)."""
    path = Path(panels_dir) / "panels.json"
    if not path.is_file():
        raise TimelineError(f"panels.json introuvable dans {panels_dir}")
    entries = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(entries, list):
        raise TimelineError(f"{path} doit contenir une liste de cases")
    return entries


# --- Sous-titres ------------------------------------------------------------------------
def split_subtitle_text(text: str, max_words: int = DEFAULT_MAX_SUBTITLE_WORDS) -> list[str]:
    """Découpe une narration en blocs courts de ``max_words`` mots au plus.

    Les blocs ne franchissent pas les fins de phrase et sont équilibrés à
    l'intérieur d'une phrase (9 mots → 3 + 3 + 3, 5 mots → 3 + 2) : jamais un
    bloc d'un seul mot quand la phrase en compte plusieurs.
    """
    cleaned = prepare_text(text)
    if not cleaned:
        return []
    max_words = max(1, max_words)
    pieces: list[str] = []
    for sentence in _SENTENCE_SPLIT.split(cleaned):
        words = sentence.split()
        if not words:
            continue
        n_groups = math.ceil(len(words) / max_words)
        base, extra = divmod(len(words), n_groups)
        cursor = 0
        for g in range(n_groups):
            size = base + (1 if g < extra else 0)
            pieces.append(" ".join(words[cursor : cursor + size]))
            cursor += size
    return pieces


def split_words(text: str, start_s: float, end_s: float) -> list[WordTiming]:
    """Minutage mot à mot d'un bloc, au prorata des caractères.

    Même répartition que celle des blocs sur la scène : la durée d'un mot est
    proportionnelle à sa longueur écrite, faute d'alignement phonétique réel. C'est une
    approximation, suffisante pour un surlignage sur des blocs de 1 à 3 mots.
    """
    words = text.split()
    if not words or end_s <= start_s:
        return []
    total = sum(len(word) for word in words) or 1
    timings: list[WordTiming] = []
    cursor = start_s
    for index, word in enumerate(words):
        stop = end_s if index == len(words) - 1 else min(end_s, cursor + (end_s - start_s) * len(word) / total)
        if stop <= cursor:
            stop = min(end_s, cursor + 1e-3)
        timings.append(WordTiming(text=word, start_s=cursor, end_s=stop))
        cursor = stop
    return timings


def build_subtitle_cues(
    scene_index: int,
    text: str,
    start_s: float,
    speech_s: float,
    *,
    max_words: int = DEFAULT_MAX_SUBTITLE_WORDS,
    emotion: str = "",
    dynamics: str = "none",
    with_words: bool = False,
) -> list[SubtitleCue]:
    """Répartit les blocs d'une scène sur sa durée de parole (prorata des caractères).

    Args:
        emotion: émotion de la scène, qui choisit l'animation des blocs.
        dynamics: ``none`` (blocs fixes, défaut), ``subtle`` ou ``punchy``.
        with_words: calculer aussi le minutage mot à mot (surlignage du format court).
    """
    pieces = split_subtitle_text(text, max_words)
    if not pieces or speech_s <= 0:
        return []
    total_chars = sum(len(p) for p in pieces)
    cues: list[SubtitleCue] = []
    cursor = start_s
    for i, piece in enumerate(pieces):
        share = speech_s * len(piece) / total_chars
        if i == len(pieces) - 1:
            end = start_s + speech_s
        else:
            end = min(cursor + max(share, MIN_CUE_DURATION_S), start_s + speech_s)
        if end <= cursor:
            break
        cues.append(
            SubtitleCue(
                scene_index=scene_index, start_s=cursor, end_s=end, text=piece,
                animation=animation_for_cue(end - cursor, emotion, dynamics=dynamics),
                words=split_words(piece, cursor, end) if with_words else [],
            )
        )
        cursor = end
    return cues


def attach_transitions(
    clips: Sequence[PanelClip], emotions: Mapping[int, str], *, dynamics: str = DEFAULT_DYNAMICS
) -> list[PanelClip]:
    """Pose les transitions aux **changements de scène**, portées par la case précédente.

    Aucune transition à l'intérieur d'une scène : les cases d'un même paragraphe
    s'enchaînent par une coupe franche, comme avant.
    """
    result = list(clips)
    posted = 0
    for i in range(len(result) - 1):
        current, following = result[i], result[i + 1]
        if current.scene_index == following.scene_index:
            continue
        transition = transition_for(
            posted,
            emotions.get(current.scene_index, ""), emotions.get(following.scene_index, ""),
            current.duration_s, following.duration_s, dynamics=dynamics,
        )
        if transition is not None:
            result[i] = current.model_copy(update={"transition": transition})
            posted += 1
    return result


# --- Mouvement -----------------------------------------------------------------------------
def motion_for(entry: Mapping, *, action_heavy: bool = False) -> str:
    """Mouvement d'une case : punch-in (impact marqué par Gemini), sinon Ken Burns doux.

    Aucun défilement ni recadrage : la case est affichée entière, à sa taille
    native, et le zoom reste sous :data:`MAX_ZOOM`.
    """
    return "punch_in" if action_heavy else "ken_burns"


# --- Dynamisme : animations, transitions, effets superposés -----------------------------------
def animation_for_cue(duration_s: float, emotion: str, *, dynamics: str = DEFAULT_DYNAMICS) -> CueAnimation | None:
    """Animation d'un bloc de sous-titre, ou ``None`` si le montage est sobre.

    La durée de l'entrée est une **fraction du bloc** : nos blocs font 2 à 4 mots, donc
    souvent moins de 0,5 s, et l'animation par défaut de CapCut (0,5 s) les consommerait
    entièrement. Un bloc trop court pour :data:`MIN_CUE_ANIMATION_S` reste fixe.

    Args:
        duration_s: durée du bloc.
        emotion: émotion de la scène (pilote le choix de l'animation).
        dynamics: ``none`` (aucune), ``subtle`` (entrée seule) ou ``punchy`` (entrée + boucle).
    """
    if dynamics == "none" or duration_s <= 0:
        return None
    budget = duration_s * CUE_ANIMATION_RATIO
    if budget < MIN_CUE_ANIMATION_S:
        return None
    intro_s = min(MAX_CUE_ANIMATION_S, budget)
    loop = CUE_LOOP_BY_EMOTION.get(emotion, "") if dynamics == "punchy" else ""
    return CueAnimation(
        intro=CUE_INTRO_BY_EMOTION.get(emotion, DEFAULT_CUE_INTRO), loop=loop, duration_s=round(intro_s, 3)
    )


def transition_for(
    index: int,
    from_emotion: str,
    to_emotion: str,
    from_duration_s: float,
    to_duration_s: float,
    *,
    dynamics: str = DEFAULT_DYNAMICS,
) -> ClipTransition | None:
    """Transition à poser entre deux scènes, ou ``None`` pour une coupe franche.

    Une transition n'est posée qu'à un **changement de scène** dont l'une des deux est
    ``action`` ou ``tension``. Sa durée est plafonnée à :data:`TRANSITION_MAX_CLIP_RATIO`
    de la plus courte des deux cases : deux cases brèves valent mieux coupées net
    qu'écrasées par un effet. Comme presque toutes les transitions CapCut **empiètent**
    sur les cases voisines, cette borne limite directement la dérive image/voix.

    Args:
        index: rang du changement de scène (fait varier l'effet retenu).
        from_emotion, to_emotion: émotions des scènes sortante et entrante.
        from_duration_s, to_duration_s: durées des cases de part et d'autre.
        dynamics: ``punchy`` seul active les transitions.
    """
    if dynamics != "punchy":
        return None
    trigger = to_emotion if to_emotion in TRANSITION_EMOTIONS else from_emotion
    if trigger not in TRANSITION_EMOTIONS:
        return None
    budget = TRANSITION_MAX_CLIP_RATIO * min(from_duration_s, to_duration_s)
    duration = min(DEFAULT_TRANSITION_S, MAX_TRANSITION_S, budget)
    if duration < MIN_TRANSITION_S:
        return None
    choices = TRANSITION_BY_EMOTION.get(trigger) or (DEFAULT_TRANSITION,)
    return ClipTransition(kind=choices[index % len(choices)], duration_s=round(duration, 3), overlap=True)


def build_vfx_clips(
    scene_spans: Sequence[tuple[int, float, float, str]],
    *,
    dynamics: str = DEFAULT_DYNAMICS,
    effects: Mapping[str, str] = VFX_EFFECT_BY_KIND,
    assets: Mapping[str, Path] | None = None,
    opacity: float = DEFAULT_VFX_OPACITY,
    min_scene_s: float = MIN_VFX_SCENE_S,
) -> list[VfxClip]:
    """Effets superposés des scènes spectaculaires (``action``, ``epic``, ``mystery``).

    Par défaut on pose un **effet intégré CapCut** : aucun asset à fournir et aucun
    risque de rectangle noir, ``pycapcut`` n'exposant pas de mode de fusion. Si
    ``assets`` fournit une vidéo à canal alpha pour ce type, elle est préférée.

    Args:
        scene_spans: ``(scene_index, début, fin, émotion)`` des scènes montées.
        dynamics: ``punchy`` seul active les effets.
        effects: effet CapCut intégré par type logique.
        assets: vidéos locales à canal alpha, par type logique.
        opacity: opacité de l'asset superposé.
        min_scene_s: durée minimale d'une scène pour recevoir un effet.
    """
    if dynamics != "punchy":
        return []
    assets = assets or {}
    clips: list[VfxClip] = []
    for scene_index, start, end, emotion in scene_spans:
        kind = VFX_KIND_BY_EMOTION.get(emotion)
        duration = end - start
        if kind is None or duration < min_scene_s:
            continue
        asset = assets.get(kind)
        effect = "" if asset else effects.get(kind, "")
        if not asset and not effect:
            continue
        clips.append(
            VfxClip(
                scene_index=scene_index, kind=kind, effect=effect,
                file=str(Path(asset).resolve()) if asset else "",
                start_s=start, duration_s=duration, opacity=opacity,
            )
        )
    return clips


def limit_panels_for_duration(
    panel_ids: Sequence[int], duration_s: float, meta_by_index: Mapping[int, Mapping], min_clip_s: float
) -> list[int]:
    """Écarte les plus petites cases d'une scène tant qu'elles n'auraient pas ``min_clip_s`` chacune."""
    kept = list(panel_ids)
    if min_clip_s <= 0 or len(kept) <= 1:
        return kept
    max_panels = max(1, int(duration_s // min_clip_s))
    if len(kept) > max_panels:
        by_size = sorted(kept, key=lambda pid: (content_height(meta_by_index[pid]), pid), reverse=True)[:max_panels]
        removed = [pid for pid in kept if pid not in by_size]
        kept = [pid for pid in kept if pid in by_size]
        logger.info(
            "Scene de %.1fs : %d case(s) ecartee(s) %s pour garder >= %.1fs par case",
            duration_s, len(removed), removed, min_clip_s,
        )
    return kept


# --- Design sonore ---------------------------------------------------------------------------
def mood_of(emotion: str) -> str:
    """Ambiance musicale (``calm`` / ``tense`` / ``action``) d'une émotion de scène."""
    return MOOD_BY_EMOTION.get(emotion, "calm")


def build_sfx_clips(
    clips: Sequence[PanelClip],
    emotions: Mapping[int, str],
    sfx_files: Mapping[str, Path],
    *,
    gain_db: float = DEFAULT_SFX_GAIN_DB,
    sfx_emotions: frozenset[str] = SFX_EMOTIONS,
) -> list[SfxClip]:
    """Bruitages : un par case des scènes d'action (cycle :data:`SFX_CYCLE`), un ``impact``
    au début de chaque case ``punch_in`` des autres scènes (moment d'impact marqué par Gemini)."""
    if not sfx_files:
        return []
    result: list[SfxClip] = []
    position: dict[int, int] = {}
    for clip in clips:
        if emotions.get(clip.scene_index) in sfx_emotions:
            k = position.get(clip.scene_index, 0)
            position[clip.scene_index] = k + 1
            kind = SFX_CYCLE[k % len(SFX_CYCLE)]
        elif clip.motion == "punch_in":
            kind = "impact"
        else:
            continue
        file = sfx_files.get(kind) or next(iter(sfx_files.values()))
        result.append(
            SfxClip(
                scene_index=clip.scene_index, panel_index=clip.panel_index, start_s=clip.start_s,
                kind=kind, file=str(Path(file).resolve()), gain_db=gain_db,
            )
        )
    return result


def build_bgm_clips(
    scene_spans: Sequence[tuple[int, float, float, str]],
    bgm_files: Mapping[str, Path],
    *,
    gain_db: float = DEFAULT_BGM_GAIN_DB,
    crossfade_s: float = DEFAULT_BGM_CROSSFADE_S,
    single_mood: bool = False,
    max_end_s: float | None = None,
) -> list[BgmClip]:
    """Segments de musique par ambiance, avec chevauchement et fondus aux changements.

    Args:
        scene_spans: ``(scene_index, début, fin, émotion)`` des scènes montées.
        bgm_files: musiques disponibles par ambiance.
        gain_db: gain appliqué à la musique.
        crossfade_s: durée du fondu enchaîné entre deux ambiances.
        single_mood: ignorer les ambiances (une seule musique, un seul segment).
        max_end_s: fin de la timeline (les segments ne la dépassent jamais).
    """
    if not bgm_files or not scene_spans:
        return []
    spans: list[list] = []  # [mood, start, end]
    for _, start, end, emotion in scene_spans:
        mood = "all" if single_mood else mood_of(emotion)
        if spans and spans[-1][0] == mood:
            spans[-1][2] = end
        else:
            spans.append([mood, start, end])
    clips: list[BgmClip] = []
    half = crossfade_s / 2.0
    for i, (mood, start, end) in enumerate(spans):
        file = bgm_for_mood(mood, bgm_files)
        if file is None:
            continue
        clip_start = start if i == 0 else max(0.0, start - half)
        clip_end = end if i == len(spans) - 1 else end + half
        if max_end_s is not None:
            clip_end = min(clip_end, max_end_s)
        if clip_end <= clip_start:
            continue
        clips.append(
            BgmClip(
                mood=mood, file=str(Path(file).resolve()), start_s=clip_start, duration_s=clip_end - clip_start,
                gain_db=gain_db,
                fade_in_s=0.5 if i == 0 else crossfade_s,
                fade_out_s=1.0 if i == len(spans) - 1 else crossfade_s,
            )
        )
    return clips


# --- Timeline --------------------------------------------------------------------------------
def build_timeline(
    analysis: ChapterAnalysis,
    manifest: VoiceoverManifest,
    panels_meta: Sequence[Mapping],
    *,
    panels_dir: str | Path,
    audio_dir: str | Path,
    width: int = DEFAULT_WIDTH,
    height: int = DEFAULT_HEIGHT,
    fps: int = DEFAULT_FPS,
    min_panel_weight: int = DEFAULT_MIN_PANEL_WEIGHT,
    min_clip_s: float = DEFAULT_MIN_CLIP_S,
    max_subtitle_words: int = DEFAULT_MAX_SUBTITLE_WORDS,
    max_duration_s: float | None = None,
    bgm_file: str | Path | None = None,
    bgm_files: Mapping[str, Path] | None = None,
    bgm_gain_db: float = DEFAULT_BGM_GAIN_DB,
    bgm_crossfade_s: float = DEFAULT_BGM_CROSSFADE_S,
    sfx_files: Mapping[str, Path] | None = None,
    sfx_gain_db: float = DEFAULT_SFX_GAIN_DB,
    dynamics: str = DEFAULT_DYNAMICS,
    vfx_assets: Mapping[str, Path] | None = None,
    profile: FormatProfile | None = None,
) -> Timeline:
    """Construit la timeline d'un chapitre.

    Args:
        analysis: scènes narrées (``scenes.json``).
        manifest: segments audio (``voiceover.json``).
        panels_meta: entrées de ``panels.json`` (index, file, width, height, type).
        panels_dir: dossier des PNG des cases.
        audio_dir: dossier des WAV.
        width, height, fps: format de la séquence.
        min_panel_weight: poids minimal d'une case dans le partage du temps.
        min_clip_s: durée minimale d'affichage d'une case (0 = jamais de tri).
        max_subtitle_words: mots par bloc de sous-titre.
        max_duration_s: tronque la timeline à cette durée (``None`` = complète).
        bgm_file: musique unique (compatibilité) si ``bgm_files`` est vide.
        bgm_files: musiques par ambiance (``calm`` / ``tense`` / ``action`` / ``default``).
        bgm_gain_db: gain de la musique (dB).
        bgm_crossfade_s: fondu enchaîné entre ambiances.
        sfx_files: bruitages par type (``swoosh`` / ``impact`` / ``roar``) ; vide = aucun.
        sfx_gain_db: gain des bruitages (dB).
        dynamics: intensité du montage dynamique — ``none`` (sobre), ``subtle``
            (animations de sous-titres seules) ou ``punchy`` (animations, transitions
            aux changements de scène intenses et effets superposés).
        vfx_assets: vidéos à canal alpha par type d'effet ; à défaut, effets CapCut intégrés.
        profile: règles de format (:class:`~src.models.format_profile.FormatProfile`).
            Absent, un profil ``LONG`` est fabriqué à partir de ``width`` / ``height`` /
            ``fps`` / ``min_clip_s`` / ``max_subtitle_words``, si bien que les appelants
            historiques gardent exactement leur comportement. Fourni, **il prime** sur ces
            arguments.

    Raises:
        TimelineError: aucune scène montable, ou case référencée absente de ``panels_meta``.
    """
    if max_duration_s is not None and max_duration_s <= 0:
        raise ValueError("max_duration_s doit etre > 0")
    if min_clip_s < 0:
        raise ValueError("min_clip_s doit etre >= 0")
    meta_by_index = {int(entry["index"]): entry for entry in panels_meta}
    audio_by_scene = manifest.by_scene()

    # Sans profil explicite, on en fabrique un en mode LONG aligne sur les arguments
    # historiques : tous les appelants existants gardent exactement leur comportement.
    if profile is None:
        profile = VideoConfigFactory.create(
            "LONG",
            framing={"width": width, "height": height, "fps": fps},
            pacing={"min_clip_s": min_clip_s},
            subtitles={"max_words": max_subtitle_words},
        )
    framing_rules, subtitle_rules = profile.framing, profile.subtitles
    width, height, fps = framing_rules.width, framing_rules.height, framing_rules.fps
    framing = make_framing(profile, loader=_panel_loader(panels_dir) if framing_rules.saliency_crop else None)
    pacing = make_pacing(profile, framing)

    clips: list[PanelClip] = []
    audio_clips: list[AudioClip] = []
    subtitles: list[SubtitleCue] = []
    emotions: dict[int, str] = {}
    scene_spans: list[tuple[int, float, float, str]] = []
    cursor = 0.0
    # Pre-passage : les scenes montables et leurs cases cles. L'elargissement a besoin de
    # les connaitre toutes d'un coup pour s'etendre aux cases laissees de cote.
    montable: list[tuple] = []
    for scene in analysis.scenes:
        item = audio_by_scene.get(scene.index)
        if item is None:
            continue  # scene sans voix off (remplissage) : non montee
        missing = [pid for pid in scene.panel_ids if pid not in meta_by_index]
        if missing:
            raise TimelineError(f"Scene {scene.index} : cases absentes de panels.json : {missing}")
        panel_ids = [pid for pid in scene.panel_ids if pid in meta_by_index]
        if panel_ids:
            montable.append((scene, item, panel_ids))

    if profile.pacing.expand_to_unused_panels and montable:
        # Les cases des scenes de remplissage (carton de titre, credits, pub) ne sont pas
        # "libres" : l'analyse les a ecartees exprès. Les offrir a l'elargissement les
        # ferait entrer au montage par la bande.
        filler_ids = {pid for scene in analysis.scenes if scene.is_filler for pid in scene.panel_ids}
        available = [pid for pid in meta_by_index if pid not in filler_ids]
        widened = expand_panel_ids([ids for _, _, ids in montable], available)
        before = sum(len(ids) for _, _, ids in montable)
        montable = [(scene, item, ids) for (scene, item, _), ids in zip(montable, widened)]
        logger.info(
            "Format %s : %d cases disponibles pour le montage au lieu de %d (cases cles etendues)",
            profile.name, sum(len(ids) for _, _, ids in montable), before,
        )

    for scene, item, panel_ids in montable:
        if max_duration_s is not None and cursor >= max_duration_s:
            break
        # Le rythme est delegue a la strategie du profil : une case par clip en LONG,
        # plusieurs plans (et re-cadrages) par scene en SHORT.
        t = cursor
        for shot in pacing.plan(scene.index, panel_ids, item.duration_s, meta_by_index, scene.action_heavy_ids):
            clips.append(
                PanelClip(
                    scene_index=shot.scene_index, panel_index=shot.panel_index, file=shot.file,
                    width=shot.panel_width, height=shot.panel_height,
                    start_s=t, duration_s=shot.duration_s, motion=shot.motion,
                    crop=shot.crop, subshot=shot.subshot,
                )
            )
            t += shot.duration_s
        audio_clips.append(AudioClip(scene_index=scene.index, file=item.file, start_s=cursor, duration_s=item.duration_s))
        subtitles.extend(build_subtitle_cues(
            scene.index, scene.narration, cursor, item.speech_s,
            max_words=subtitle_rules.max_words, emotion=scene.emotion, dynamics=dynamics,
            with_words=subtitle_rules.highlight_current_word,
        ))
        emotions[scene.index] = scene.emotion
        scene_spans.append((scene.index, cursor, cursor + item.duration_s, scene.emotion))
        cursor += item.duration_s

    if not clips:
        raise TimelineError("Aucune scene montable : verifier scenes.json et voiceover.json")

    total = cursor
    if max_duration_s is not None and total > max_duration_s:
        total = max_duration_s
        clips = _truncate(clips, total)
        audio_clips = _truncate(audio_clips, total)
        subtitles = [c.model_copy(update={"end_s": min(c.end_s, total)}) for c in subtitles if c.start_s < total]
        scene_spans = [(i, s, min(e, total), m) for i, s, e, m in scene_spans if s < total]

    # Les transitions sont posees apres la troncature : une case coupee par
    # ``max_duration_s`` ne doit pas garder une transition vers une case absente.
    clips = attach_transitions(clips, emotions, dynamics=dynamics)
    vfx = build_vfx_clips(scene_spans, dynamics=dynamics, assets=vfx_assets)
    sfx = build_sfx_clips(clips, emotions, sfx_files or {}, gain_db=sfx_gain_db)
    if bgm_files:
        bgm = build_bgm_clips(scene_spans, bgm_files, gain_db=bgm_gain_db, crossfade_s=bgm_crossfade_s, max_end_s=total)
    elif bgm_file:
        bgm = build_bgm_clips(
            scene_spans, {"default": Path(bgm_file)}, gain_db=bgm_gain_db, crossfade_s=bgm_crossfade_s,
            single_mood=True, max_end_s=total,
        )
    else:
        bgm = []

    timeline = Timeline(
        width=width, height=height, fps=fps,
        series_title=analysis.series_title, episode_title=analysis.episode_title,
        panels_dir=str(Path(panels_dir).resolve()), audio_dir=str(Path(audio_dir).resolve()),
        clips=clips, audio=audio_clips, subtitles=subtitles, sfx=sfx, bgm=bgm, vfx=vfx, total_duration_s=total,
        bgm_file=str(Path(bgm_file).resolve()) if bgm_file else None, bgm_gain_db=bgm_gain_db,
        format=profile.name,
    )
    logger.info(
        "Timeline : %d scene(s), %d case(s) (%d punch-in), %d sous-titre(s), %d bruitage(s), "
        "%d segment(s) musique, %.1fs a %dx%d @ %d fps",
        timeline.n_scenes, len(clips), sum(1 for c in clips if c.motion == "punch_in"),
        len(subtitles), len(sfx), len(bgm), total, width, height, fps,
    )
    if dynamics != "none":
        animated = sum(1 for c in subtitles if c.animation is not None)
        logger.info(
            "Dynamisme (%s) : %d sous-titre(s) anime(s), %d transition(s), %d effet(s) superpose(s)",
            dynamics, animated, timeline.n_transitions, len(vfx),
        )
        if timeline.transition_drift_s > 0:
            # Presque toutes les transitions CapCut empietent sur les cases voisines : si
            # l'editeur applique ce recouvrement, l'image prend cette avance sur la voix.
            logger.info(
                "Transitions : %.1fs de derive image/voix possible dans CapCut (l'apercu ffmpeg, lui, reste cale)",
                timeline.transition_drift_s,
            )
    return timeline


def _truncate(items, total: float):
    """Coupe une liste de clips contigus à ``total`` secondes."""
    kept = []
    for item in items:
        if item.start_s >= total:
            break
        if item.start_s + item.duration_s > total:
            item = item.model_copy(update={"duration_s": total - item.start_s})
        kept.append(item)
    return kept


#: Silence inséré entre deux chapitres d'une compilation (secondes).
DEFAULT_CHAPTER_GAP_S: float = 0.6


def concat_timelines(
    timelines: Sequence[Timeline],
    *,
    gap_s: float = DEFAULT_CHAPTER_GAP_S,
    series_title: str = "",
    episode_title: str = "",
    panels_dir: str | Path | None = None,
    audio_dir: str | Path | None = None,
) -> Timeline:
    """Enchaîne plusieurs timelines de chapitres en une seule (compilation).

    Chaque chapitre est décalé de la durée cumulée des précédents, plus ``gap_s``. Les
    fichiers (cases, voix) sont réécrits en **chemins absolus** : les médias restent dans
    le dossier de leur chapitre, aucune copie n'est faite. Les numéros de scène sont
    décalés pour rester uniques d'un chapitre à l'autre.

    Args:
        timelines: timelines des chapitres, dans l'ordre de lecture.
        gap_s: silence entre deux chapitres.
        series_title, episode_title: titres de la compilation.
        panels_dir, audio_dir: dossiers de référence de la compilation (informatifs :
            les chemins des médias sont absolus).

    Raises:
        TimelineError: liste vide ou formats (résolution / fps) incompatibles.
    """
    timelines = list(timelines)
    if not timelines:
        raise TimelineError("Aucune timeline a fusionner")
    if gap_s < 0:
        raise ValueError("gap_s doit etre >= 0")
    first = timelines[0]
    for other in timelines[1:]:
        if (other.width, other.height, other.fps) != (first.width, first.height, first.fps):
            raise TimelineError(
                f"Formats incompatibles : {first.width}x{first.height}@{first.fps} vs "
                f"{other.width}x{other.height}@{other.fps}"
            )

    clips: list[PanelClip] = []
    audio: list[AudioClip] = []
    subtitles: list[SubtitleCue] = []
    sfx: list[SfxClip] = []
    bgm: list[BgmClip] = []
    vfx: list[VfxClip] = []
    offset = 0.0
    scene_base = 0
    for timeline in timelines:
        panels_root, audio_root = Path(timeline.panels_dir), Path(timeline.audio_dir)
        for k, clip in enumerate(timeline.clips):
            update = {
                "start_s": clip.start_s + offset,
                "scene_index": clip.scene_index + scene_base,
                "file": str((panels_root / clip.file).resolve()),
            }
            # La derniere case d'un chapitre ne doit pas enchainer sur le chapitre
            # suivant : entre les deux il y a le silence de ``gap_s``.
            if k == len(timeline.clips) - 1 and clip.transition is not None:
                update["transition"] = None
            clips.append(clip.model_copy(update=update))
        for item in timeline.audio:
            audio.append(item.model_copy(update={
                "start_s": item.start_s + offset,
                "scene_index": item.scene_index + scene_base,
                "file": str((audio_root / item.file).resolve()),
            }))
        for cue in timeline.subtitles:
            subtitles.append(cue.model_copy(update={
                "start_s": cue.start_s + offset, "end_s": cue.end_s + offset,
                "scene_index": cue.scene_index + scene_base,
            }))
        for clip in timeline.sfx:
            sfx.append(clip.model_copy(update={
                "start_s": clip.start_s + offset, "scene_index": clip.scene_index + scene_base,
            }))
        for clip in timeline.bgm:
            bgm.append(clip.model_copy(update={"start_s": clip.start_s + offset}))
        for clip in timeline.vfx:
            vfx.append(clip.model_copy(update={
                "start_s": clip.start_s + offset, "scene_index": clip.scene_index + scene_base,
            }))
        scene_base += max((c.scene_index for c in timeline.clips), default=-1) + 1
        offset += timeline.total_duration_s + gap_s
    total = offset - gap_s if len(timelines) else 0.0

    merged = Timeline(
        width=first.width, height=first.height, fps=first.fps,
        series_title=series_title or first.series_title,
        episode_title=episode_title or f"{len(timelines)} chapitres",
        panels_dir=str(Path(panels_dir).resolve()) if panels_dir else first.panels_dir,
        audio_dir=str(Path(audio_dir).resolve()) if audio_dir else first.audio_dir,
        clips=clips, audio=audio, subtitles=subtitles, sfx=sfx, bgm=bgm, vfx=vfx,
        total_duration_s=total, bgm_gain_db=first.bgm_gain_db,
    )
    logger.info(
        "Compilation : %d chapitre(s), %d case(s), %d sous-titre(s), %.0f min au total",
        len(timelines), len(clips), len(subtitles), total / 60,
    )
    return merged


def save_timeline(timeline: Timeline, path: str | Path) -> Path:
    """Écrit la timeline en JSON et renvoie le chemin."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(timeline.model_dump_json(indent=2), encoding="utf-8")
    logger.info("Timeline ecrite : %s", path)
    return path


def load_timeline(path: str | Path) -> Timeline:
    """Relit une timeline écrite par :func:`save_timeline`."""
    return Timeline.model_validate_json(Path(path).read_text(encoding="utf-8"))


__all__ = [
    "DEFAULT_WIDTH",
    "DEFAULT_HEIGHT",
    "DEFAULT_FPS",
    "DEFAULT_MIN_PANEL_WEIGHT",
    "DEFAULT_MIN_CLIP_S",
    "MAX_ZOOM",
    "KEN_BURNS_ZOOM",
    "PUNCH_IN_ZOOM",
    "PUNCH_IN_S",
    "DEFAULT_BGM_GAIN_DB",
    "DEFAULT_BGM_CROSSFADE_S",
    "SFX_EMOTIONS",
    "SFX_CYCLE",
    "DEFAULT_SFX_GAIN_DB",
    "MOOD_BY_EMOTION",
    "DEFAULT_MAX_SUBTITLE_WORDS",
    "DEFAULT_DYNAMICS",
    "CUE_INTRO_BY_EMOTION",
    "CUE_LOOP_BY_EMOTION",
    "TRANSITION_EMOTIONS",
    "TRANSITION_BY_EMOTION",
    "DEFAULT_TRANSITION_S",
    "MIN_TRANSITION_S",
    "MAX_TRANSITION_S",
    "TRANSITION_MAX_CLIP_RATIO",
    "VFX_KIND_BY_EMOTION",
    "VFX_EFFECT_BY_KIND",
    "MIN_VFX_SCENE_S",
    "TimelineError",
    "load_panels_meta",
    "split_subtitle_text",
    "build_subtitle_cues",
    "motion_for",
    "animation_for_cue",
    "transition_for",
    "attach_transitions",
    "build_vfx_clips",
    "limit_panels_for_duration",
    "mood_of",
    "build_sfx_clips",
    "build_bgm_clips",
    "build_timeline",
    "DEFAULT_CHAPTER_GAP_S",
    "concat_timelines",
    "save_timeline",
    "load_timeline",
]
