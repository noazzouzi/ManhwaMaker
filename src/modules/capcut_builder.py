"""Module 5 — CapCut Draft Builder : de la timeline au brouillon CapCut.

Génère un projet CapCut (``draft_content.json`` + ``draft_meta_info.json``) via
``pycapcut``, à partir de la :class:`~src.models.timeline.Timeline` :

- **V1** : fond de chaque case, **pré-rendu** en JPEG 1920x1080 (case étirée pour
  couvrir le cadre, flou gaussien, −30 % de luminosité) par le même code que la
  prévisualisation ffmpeg : il comble les bandes 16:9 à gauche et à droite ;
- **V2** : la case PNG **à sa résolution native**, centrée ; une case plus grande
  que le cadre est réduite pour y tenir, jamais agrandie. Images clés : zoom Ken
  Burns 100 % → 105 % sur la durée du clip, ou **punch-in** 105 % en 0,2 s puis
  maintien (cases ``action_heavy``) ; jamais au-delà de 105 % ;
- **A1** : un segment audio par scène (WAV Kokoro) ;
- **A2 / A2b** : musique de fond par ambiance à −22 dB, **bouclée** en segments
  contigus si la musique est plus courte que la vidéo, sur deux pistes alternées
  pour que les fondus enchaînés (``add_fade``) se chevauchent ;
- **A3** : bruitages courts ;
- **T1** : sous-titres en blocs de 2-4 mots, police grasse (Rubik Bold /
  Montserrat), blanc, contour noir de 2 px, centrés en bas.

Convention d'échelle CapCut : une image importée est ajustée (« contain ») dans
le cadre à l'échelle 1.0 ; l'échelle native vaut donc ``1 / contain``.
"""

from __future__ import annotations

import logging
import os
import shutil
import threading
from pathlib import Path

import cv2
from PIL import Image

from src.models.timeline import BgmClip, PanelClip, Timeline
from src.modules.preview_renderer import make_background, native_scale
from src.modules.timeline_builder import KEN_BURNS_ZOOM, MAX_ZOOM, PUNCH_IN_S, PUNCH_IN_ZOOM
from src.utils.image_utils import to_numpy_rgb

logger = logging.getLogger(__name__)

#: Zoom Ken Burns appliqué à toutes les cases fixes pendant la voix off : 100 % → 105 %.
KEN_BURNS_START: float = 1.0
KEN_BURNS_END: float = min(1.0 + KEN_BURNS_ZOOM, MAX_ZOOM)
#: Punch-in des cases ``action_heavy`` : 105 % atteint en 0,2 s, puis maintenu.
PUNCH_IN_END: float = min(1.0 + PUNCH_IN_ZOOM, MAX_ZOOM)
#: Sous-dossier (dans le dossier des cases) des fonds pré-rendus.
BACKGROUNDS_DIRNAME: str = "backgrounds"
BACKGROUND_JPEG_QUALITY: int = 90
#: Sous-titres : taille (unité CapCut), position verticale (demi-hauteur du cadre).
SUBTITLE_FONT_SIZE: float = 9.0
SUBTITLE_TRANSFORM_Y: float = -0.78
#: Contour noir : largeur CapCut (0-100) ; 20 ≈ 2 px en 1080p pour la taille 9 (la valeur
#: 40 par défaut de CapCut donne un contour d'environ 4 px).
SUBTITLE_BORDER_WIDTH: float = 20.0
#: Polices CapCut candidates pour les sous-titres, par ordre de préférence.
SUBTITLE_FONT_CANDIDATES: tuple[str, ...] = ("Rubik_Bold", "Montserrat", "Anton", "BebasNeue")
CAPCUT_DRAFTS_RELATIVE: tuple[str, ...] = ("CapCut", "User Data", "Projects", "com.lveditor.draft")

TRACK_BACKGROUND = "V1 background"
TRACK_PANELS = "V2 panels"
TRACK_VOICE = "A1 voice"
TRACK_BGM = ("A2 bgm", "A2b bgm")
TRACK_SFX = "A3 sfx"
TRACK_SUBTITLES = "T1 subtitles"
#: Piste d'effets (catalogue CapCut) portant les effets superposés des scènes spectaculaires.
TRACK_VFX = "E1 vfx"

#: Compensation du recouvrement des transitions.
#:
#: Dans le catalogue CapCut, 1131 des 1137 transitions ont ``is_overlap`` : l'éditeur
#: rapproche les deux clips, ce qui **raccourcit** le montage et fait prendre de l'avance
#: à l'image sur la voix off. Deux stratégies :
#:
#: - ``"none"`` (défaut) : la timeline est écrite telle quelle. Les transitions étant
#:   bornées à 0,4 s et posées aux seuls changements de scène intenses, la dérive reste
#:   faible et mesurée (:attr:`~src.models.timeline.Timeline.transition_drift_s`).
#: - ``"shift"`` : chaque clip vidéo est déclaré plus tard de la durée cumulée des
#:   transitions qui le précèdent, pour que le recouvrement de CapCut le ramène à sa place.
#:
#: ``"shift"`` reste **à vérifier dans CapCut** : la sémantique exacte de ``is_overlap``
#: n'est pas documentée, et aucun brouillon n'a encore été ouvert dans l'éditeur.
TRANSITION_COMPENSATIONS: tuple[str, ...] = ("none", "shift")
DEFAULT_TRANSITION_COMPENSATION: str = "none"


#: Sérialise la construction des brouillons entre threads.
#:
#: ``pycapcut`` sonde chaque média avec ``pymediainfo.MediaInfo.parse()``, qui n'est pas
#: sûr en concurrence : deux appels simultanés renvoient une sortie vide ou tronquée, d'où
#: des ``ParseError: syntax error: line 1, column 0`` ou des « fichier sans piste video ni
#: image » sur des PNG parfaitement valides. L'étape CapCut ne dure que quelques secondes
#: par chapitre, contre une minute pour le rendu ffmpeg : la sérialiser ne coûte
#: pratiquement rien et rend ``--max-render-workers`` supérieur à 1 utilisable.
_DRAFT_LOCK = threading.Lock()


class CapCutError(RuntimeError):
    """Génération du brouillon impossible (pycapcut absent, fichier manquant...)."""


def _us(value: float) -> int:
    """Convertit des secondes en microsecondes entières (unité de temps de CapCut)."""
    return max(0, int(round(value * 1_000_000)))


def contiguous_ranges(items) -> list[tuple[int, int]]:
    """Bornes ``(début, durée)`` en µs de clips contigus, sans chevauchement ni trou."""
    starts = [_us(item.start_s) for item in items]
    ranges: list[tuple[int, int]] = []
    for i, item in enumerate(items):
        end = starts[i + 1] if i + 1 < len(items) else _us(item.start_s + item.duration_s)
        end = max(end, starts[i] + 1)
        ranges.append((starts[i], end - starts[i]))
    return ranges


def cue_ranges(cues) -> list[tuple[int, int]]:
    """Bornes ``(début, durée)`` en µs des sous-titres (bornés par le bloc suivant)."""
    ranges: list[tuple[int, int]] = []
    for i, cue in enumerate(cues):
        start = _us(cue.start_s)
        end = _us(cue.end_s)
        if i + 1 < len(cues):
            end = min(end, _us(cues[i + 1].start_s))
        ranges.append((start, end - start) if end > start else (start, 0))
    return ranges


def loop_ranges(clip: BgmClip, material_us: int) -> list[tuple[int, int]]:
    """Segments ``(début, durée)`` en µs couvrant un segment de musique, en bouclant la source.

    La musique de ``material_us`` µs est répétée bout à bout jusqu'à la fin du
    segment ; le dernier morceau est tronqué.
    """
    start, end = _us(clip.start_s), _us(clip.start_s + clip.duration_s)
    if material_us <= 0 or end <= start:
        return []
    ranges: list[tuple[int, int]] = []
    t = start
    while t < end:
        length = min(material_us, end - t)
        ranges.append((t, length))
        t += length
    return ranges


def detect_capcut_drafts_dir() -> Path | None:
    """Dossier des projets CapCut sur cette machine, ou ``None`` s'il est introuvable."""
    local = os.environ.get("LOCALAPPDATA")
    if not local:
        return None
    candidate = Path(local).joinpath(*CAPCUT_DRAFTS_RELATIVE)
    return candidate if candidate.is_dir() else None


def ensure_background(clip: PanelClip, timeline: Timeline, backgrounds_dir: Path | None = None) -> Path:
    """Rend (ou réutilise) le fond V1 d'une case en JPEG aux dimensions de la séquence.

    Le fond est écrit **à côté de la case** (``<dossier de la case>/backgrounds/``) : pour
    un chapitre c'est ``<panels_dir>/backgrounds/``, et pour une compilation multi-chapitres
    chaque fond reste dans son chapitre d'origine (pas de collision entre numéros de cases,
    et les fonds déjà calculés sont réutilisés).
    """
    panel_path = Path(timeline.panels_dir) / clip.file
    if not panel_path.is_file():
        raise CapCutError(f"Case introuvable : {panel_path}")
    backgrounds_dir = panel_path.parent / BACKGROUNDS_DIRNAME if backgrounds_dir is None else backgrounds_dir
    backgrounds_dir.mkdir(parents=True, exist_ok=True)
    out_path = backgrounds_dir / f"bg_{clip.panel_index:03d}.jpg"
    if out_path.is_file():
        return out_path
    with Image.open(panel_path) as img:
        rgb = to_numpy_rgb(img).copy()
    background = make_background(rgb, timeline.width, timeline.height)
    ok, buffer = cv2.imencode(".jpg", cv2.cvtColor(background, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, BACKGROUND_JPEG_QUALITY])
    if not ok:
        raise CapCutError(f"Echec de l'encodage du fond pour la case {clip.panel_index}")
    out_path.write_bytes(buffer.tobytes())
    return out_path


def crop_settings(clip: PanelClip, pc) -> object | None:
    """``CropSettings`` CapCut correspondant à la fenêtre de la case, en coordonnées 0-1.

    CapCut exprime le recadrage par les quatre coins de la zone conservée, rapportés aux
    dimensions de la case. Renvoie ``None`` quand la case est affichée entière.
    """
    window = clip.crop
    if window is None or (window.width >= clip.width and window.height >= clip.height):
        return None
    left, right = window.x / clip.width, window.right / clip.width
    top, bottom = window.y / clip.height, window.bottom / clip.height
    return pc.CropSettings(
        upper_left_x=left, upper_left_y=top, upper_right_x=right, upper_right_y=top,
        lower_left_x=left, lower_left_y=bottom, lower_right_x=right, lower_right_y=bottom,
    )


def panel_placement(clip: PanelClip, timeline: Timeline) -> dict:
    """Échelle CapCut d'une case, en tenant compte de sa fenêtre de recadrage.

    Rappel de la convention CapCut : une image importée est ajustée « contain » dans le
    cadre à l'échelle 1,0. L'échelle à écrire est donc toujours le rapport entre
    l'agrandissement voulu et cet ajustement de référence — calculé sur la **fenêtre**
    quand il y en a une, puisque c'est elle que CapCut ajustera après recadrage.

    Returns:
        ``{"scale", "mode", "target"}`` : ``target`` est l'échelle réelle par rapport aux
        pixels affichés, ``scale`` la valeur CapCut correspondante, et ``mode`` vaut
        ``"native"``, ``"contain"`` ou ``"cover"``.
    """
    window = clip.crop
    if window is not None and (window.width < clip.width or window.height < clip.height or window.fit == "cover"):
        source_w, source_h = window.width, window.height
        contain = min(timeline.width / source_w, timeline.height / source_h)
        if window.fit == "cover":
            target = max(timeline.width / source_w, timeline.height / source_h)
            return {"scale": target / contain, "mode": "cover", "target": target}
        return {"scale": 1.0, "mode": "contain", "target": contain}
    contain = min(timeline.width / clip.width, timeline.height / clip.height)
    target = native_scale(clip.width, clip.height, timeline.width, timeline.height)
    return {"scale": target / contain, "mode": "native" if target >= 1.0 else "contain", "target": target}


def compensated_ranges(clips, compensation: str = DEFAULT_TRANSITION_COMPENSATION) -> list[tuple[int, int]]:
    """Bornes ``(début, durée)`` en µs des cases, éventuellement décalées pour les transitions.

    En mode ``"shift"``, chaque case est déclarée plus tard de la durée cumulée des
    transitions qui la précèdent : le recouvrement appliqué par CapCut la ramène alors à
    l'instant voulu par la voix off. En mode ``"none"`` les bornes sont inchangées.

    Voir :data:`TRANSITION_COMPENSATIONS` pour la raison et les limites.
    """
    base = contiguous_ranges(clips)
    if compensation not in TRANSITION_COMPENSATIONS:
        raise CapCutError(f"Compensation inconnue : {compensation} (valeurs : {', '.join(TRANSITION_COMPENSATIONS)})")
    if compensation == "none":
        return base
    ranges: list[tuple[int, int]] = []
    shift = 0
    for clip, (start, duration) in zip(clips, base):
        ranges.append((start + shift, duration))
        transition = getattr(clip, "transition", None)
        if transition is not None and transition.overlap:
            shift += _us(transition.duration_s)
    return ranges


_UNKNOWN_NAMES: set[tuple[str, str]] = set()


def _catalog_member(enum_cls, name: str, label: str):
    """Membre du catalogue CapCut portant ce nom, ou ``None`` (signalé une seule fois).

    Les noms sont stockés en texte dans la timeline : un catalogue ``pycapcut`` plus ancien
    ou plus récent ne doit pas faire échouer tout le montage pour une animation manquante.
    """
    if not name:
        return None
    member = getattr(enum_cls, name, None)
    if member is None and (label, name) not in _UNKNOWN_NAMES:
        _UNKNOWN_NAMES.add((label, name))
        logger.warning("%s absent du catalogue pycapcut : %s (ignore)", label, name)
    return member


def _subtitle_font(pc):
    for name in SUBTITLE_FONT_CANDIDATES:
        font = getattr(pc.FontType, name, None)
        if font is not None:
            return font
    return None


def build_capcut_draft(
    timeline: Timeline,
    out_root: str | Path,
    name: str,
    *,
    subtitle_font_size: float = SUBTITLE_FONT_SIZE,
    dynamics: bool = True,
    transition_compensation: str = DEFAULT_TRANSITION_COMPENSATION,
    punch_in_zoom: float = PUNCH_IN_ZOOM,
) -> Path:
    """Écrit le brouillon CapCut ``<out_root>/<name>/`` et renvoie son dossier.

    L'écriture est **sérialisée entre threads** (voir :data:`_DRAFT_LOCK`) : la sonde de
    médias de ``pycapcut`` n'est pas sûre en concurrence.

    Args:
        timeline: plan de montage.
        out_root: dossier parent du brouillon.
        name: nom du projet.
        subtitle_font_size: taille de la police des sous-titres (unité CapCut).
        dynamics: émettre le dynamisme inscrit dans la timeline (animations de
            sous-titres, transitions aux changements de scène, effets superposés).
        transition_compensation: voir :data:`TRANSITION_COMPENSATIONS`.

    Raises:
        CapCutError: pycapcut absent, timeline vide, média manquant ou compensation inconnue.
    """
    with _DRAFT_LOCK:
        return _build_draft(
            timeline, out_root, name, subtitle_font_size=subtitle_font_size,
            dynamics=dynamics, transition_compensation=transition_compensation,
            punch_in_zoom=punch_in_zoom,
        )


def _build_draft(
    timeline: Timeline,
    out_root: str | Path,
    name: str,
    *,
    subtitle_font_size: float = SUBTITLE_FONT_SIZE,
    dynamics: bool = True,
    transition_compensation: str = DEFAULT_TRANSITION_COMPENSATION,
    punch_in_zoom: float = PUNCH_IN_ZOOM,
) -> Path:
    """Corps de :func:`build_capcut_draft`, à n'appeler que sous :data:`_DRAFT_LOCK`."""
    try:
        import pycapcut as pc
    except ImportError as exc:
        raise CapCutError("pycapcut n'est pas installe : pip install pycapcut") from exc
    if not timeline.clips:
        raise CapCutError("Timeline vide : aucun clip a monter")

    out_root = Path(out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    folder = pc.DraftFolder(str(out_root))
    script = folder.create_draft(name, timeline.width, timeline.height, fps=timeline.fps, allow_replace=True)
    script.add_track(pc.TrackType.video, TRACK_BACKGROUND, relative_index=0)
    script.add_track(pc.TrackType.video, TRACK_PANELS, relative_index=1)
    script.add_track(pc.TrackType.audio, TRACK_VOICE, relative_index=0)
    if timeline.bgm:
        script.add_track(pc.TrackType.audio, TRACK_BGM[0], relative_index=1)
        script.add_track(pc.TrackType.audio, TRACK_BGM[1], relative_index=2)
    if timeline.sfx:
        script.add_track(pc.TrackType.audio, TRACK_SFX, relative_index=3)
    script.add_track(pc.TrackType.text, TRACK_SUBTITLES)
    if dynamics and timeline.vfx:
        script.add_track(pc.TrackType.effect, TRACK_VFX)

    n_transitions = 0
    panel_ranges = compensated_ranges(timeline.clips, transition_compensation if dynamics else "none")
    for clip, (start_us, duration_us) in zip(timeline.clips, panel_ranges):
        panel_path = Path(timeline.panels_dir) / clip.file
        if not panel_path.is_file():
            raise CapCutError(f"Case introuvable : {panel_path}")
        background = ensure_background(clip, timeline)
        timerange = pc.Timerange(start_us, duration_us)
        bg_segment = pc.VideoSegment(pc.VideoMaterial(str(background)), timerange)

        scale = panel_placement(clip, timeline)["scale"]
        # Le recadrage vit dans le materiau : chaque plan a sa propre fenetre, donc son
        # propre materiau, meme quand plusieurs plans viennent de la meme case.
        # Attention : ``crop_settings=None`` ECRASE le defaut de pycapcut et casse
        # l'export ; on ne passe l'argument que lorsqu'il y a vraiment un recadrage.
        crop = crop_settings(clip, pc)
        material = pc.VideoMaterial(str(panel_path), crop_settings=crop) if crop is not None else pc.VideoMaterial(str(panel_path))
        fg_segment = pc.VideoSegment(
            material, timerange, clip_settings=pc.ClipSettings(scale_x=scale, scale_y=scale),
        )
        if clip.motion == "punch_in":
            punch_us = min(_us(PUNCH_IN_S), max(1, duration_us - 1))
            end_scale = scale * (1.0 + punch_in_zoom)
            for prop in (pc.KeyframeProperty.scale_x, pc.KeyframeProperty.scale_y):
                fg_segment.add_keyframe(prop, 0, scale)
                fg_segment.add_keyframe(prop, punch_us, end_scale)
                fg_segment.add_keyframe(prop, duration_us, end_scale)
        elif clip.motion == "fast_pan":
            # Le balayage se traduit par un deplacement horizontal du plan : CapCut
            # n'anime pas le recadrage, seulement la position.
            travel = 0.12 if clip.crop is None else min(0.35, (clip.width - clip.crop.width) / max(clip.width, 1))
            for keyframe_time, value in ((0, -travel / 2), (duration_us, travel / 2)):
                fg_segment.add_keyframe(pc.KeyframeProperty.position_x, keyframe_time, value)
        else:
            for prop in (pc.KeyframeProperty.scale_x, pc.KeyframeProperty.scale_y):
                fg_segment.add_keyframe(prop, 0, scale * KEN_BURNS_START)
                fg_segment.add_keyframe(prop, duration_us, scale * KEN_BURNS_END)

        # La transition est portee par le segment PRECEDENT (convention pycapcut) et posee
        # sur les deux pistes video : sans cela le fond floute couperait net pendant que la
        # case, elle, basculerait. Elle doit etre attachee AVANT ``add_segment`` : c'est
        # l'insertion dans le script qui collecte ``segment.transition`` dans les materials.
        if dynamics and clip.transition is not None:
            member = _catalog_member(pc.TransitionType, clip.transition.kind, "Transition")
            if member is not None:
                for segment in (bg_segment, fg_segment):
                    segment.add_transition(member, duration=_us(clip.transition.duration_s))
                n_transitions += 1
        script.add_segment(bg_segment, TRACK_BACKGROUND)
        script.add_segment(fg_segment, TRACK_PANELS)

    for audio, (start_us, duration_us) in zip(timeline.audio, contiguous_ranges(timeline.audio)):
        wav_path = Path(timeline.audio_dir) / audio.file
        if not wav_path.is_file():
            raise CapCutError(f"Segment audio introuvable : {wav_path}")
        material = pc.AudioMaterial(str(wav_path))
        source_us = min(duration_us, int(material.duration))
        script.add_segment(
            pc.AudioSegment(material, pc.Timerange(start_us, source_us), source_timerange=pc.Timerange(0, source_us)),
            TRACK_VOICE,
        )

    n_bgm_segments = 0
    for i, clip in enumerate(timeline.bgm):
        bgm_path = Path(clip.file)
        if not bgm_path.is_file():
            raise CapCutError(f"Musique de fond introuvable : {bgm_path}")
        material = pc.AudioMaterial(str(bgm_path))
        pieces = loop_ranges(clip, int(material.duration))
        volume = 10 ** (clip.gain_db / 20.0)
        track = TRACK_BGM[i % 2]
        for k, (start_us, duration_us) in enumerate(pieces):
            segment = pc.AudioSegment(
                material, pc.Timerange(start_us, duration_us), source_timerange=pc.Timerange(0, duration_us), volume=volume,
            )
            fade_in = _us(min(clip.fade_in_s, clip.duration_s / 2)) if k == 0 else 0
            fade_out = _us(min(clip.fade_out_s, clip.duration_s / 2)) if k == len(pieces) - 1 else 0
            if fade_in or fade_out:
                segment.add_fade(min(fade_in, duration_us), min(fade_out, duration_us))
            script.add_segment(segment, track)
            n_bgm_segments += 1
        if len(pieces) > 1:
            logger.info("Musique %s (%s) bouclee %d fois sur %.1fs", clip.mood, bgm_path.name, len(pieces), clip.duration_s)

    last_sfx_end = -1
    for clip in timeline.sfx:
        sfx_path = Path(clip.file)
        if not sfx_path.is_file():
            raise CapCutError(f"Bruitage introuvable : {sfx_path}")
        material = pc.AudioMaterial(str(sfx_path))
        start_us = _us(clip.start_s)
        if start_us < last_sfx_end:
            logger.debug("Bruitage %s a %.2fs chevauche le precedent, ignore", clip.kind, clip.start_s)
            continue
        duration_us = int(material.duration)
        if duration_us <= 0:
            continue
        script.add_segment(
            pc.AudioSegment(material, pc.Timerange(start_us, duration_us), source_timerange=pc.Timerange(0, duration_us),
                            volume=10 ** (clip.gain_db / 20.0)),
            TRACK_SFX,
        )
        last_sfx_end = start_us + duration_us

    style = pc.TextStyle(size=subtitle_font_size, bold=True, color=(1.0, 1.0, 1.0), align=1, auto_wrapping=False)
    border = pc.TextBorder(color=(0.0, 0.0, 0.0), width=SUBTITLE_BORDER_WIDTH)
    font = _subtitle_font(pc)
    n_animated = 0
    for cue, (start_us, duration_us) in zip(timeline.subtitles, cue_ranges(timeline.subtitles)):
        if duration_us <= 0:
            continue
        segment = pc.TextSegment(
            cue.text, pc.Timerange(start_us, duration_us), font=font, style=style, border=border,
            clip_settings=pc.ClipSettings(transform_y=SUBTITLE_TRANSFORM_Y),
        )
        if dynamics and cue.animation is not None:
            # pycapcut impose d'ajouter l'entree AVANT la boucle, celle-ci remplissant
            # ensuite le temps restant du bloc.
            intro = _catalog_member(pc.TextIntro, cue.animation.intro, "Animation de texte")
            if intro is not None:
                intro_us = max(1, min(_us(cue.animation.duration_s), duration_us - 1))
                segment.add_animation(intro, intro_us)
                n_animated += 1
            loop = _catalog_member(pc.TextLoopAnim, cue.animation.loop, "Animation de texte en boucle")
            if loop is not None:
                segment.add_animation(loop)
        script.add_segment(segment, TRACK_SUBTITLES)

    n_vfx = 0
    if dynamics and timeline.vfx:
        for clip in timeline.vfx:
            member = _catalog_member(pc.VideoSceneEffectType, clip.effect, "Effet de scene")
            if member is None:
                # Asset local a canal alpha : non pris en charge ici (pycapcut n'expose
                # aucun mode de fusion), l'apercu ffmpeg le rend en revanche.
                continue
            script.add_effect(member, pc.Timerange(_us(clip.start_s), _us(clip.duration_s)), TRACK_VFX)
            n_vfx += 1

    script.save()
    draft_dir = out_root / name
    logger.info(
        "Brouillon CapCut ecrit : %s (%d cases, %d segments voix, %d segments musique, %d bruitages, %d sous-titres, %.1fs)",
        draft_dir, len(timeline.clips), len(timeline.audio), n_bgm_segments, len(timeline.sfx),
        len(timeline.subtitles), timeline.total_duration_s,
    )
    if dynamics and (n_transitions or n_animated or n_vfx):
        logger.info(
            "Dynamisme CapCut : %d transition(s), %d sous-titre(s) anime(s), %d effet(s) superpose(s), compensation '%s'",
            n_transitions, n_animated, n_vfx, transition_compensation,
        )
    return draft_dir


def copy_draft(draft_dir: str | Path, capcut_dir: str | Path) -> Path:
    """Copie un brouillon dans le dossier des projets CapCut (remplace un homonyme).

    Si le projet homonyme est verrouillé (ouvert dans CapCut), la copie est écrite sous un
    nom suffixé ``<nom> (2)``, ``<nom> (3)``... plutôt que d'échouer.

    Raises:
        CapCutError: aucune variante du nom n'a pu être écrite.
    """
    draft_dir, capcut_dir = Path(draft_dir), Path(capcut_dir)
    last_error: OSError | None = None
    for attempt in range(1, 6):
        name = draft_dir.name if attempt == 1 else f"{draft_dir.name} ({attempt})"
        target = capcut_dir / name
        try:
            if target.exists():
                shutil.rmtree(target)
            shutil.copytree(draft_dir, target)
        except OSError as exc:
            last_error = exc
            logger.warning("Projet CapCut %s verrouille ou inaccessible (%s) : essai sous un autre nom", target, exc.strerror or exc)
            continue
        if attempt > 1:
            logger.warning("Le projet %s est ouvert dans CapCut : brouillon copie sous %s", draft_dir.name, name)
        logger.info("Brouillon copie dans CapCut : %s", target)
        return target
    raise CapCutError(f"Impossible de copier le brouillon dans {capcut_dir} (fermer CapCut ?) : {last_error}")


__all__ = [
    "BACKGROUNDS_DIRNAME",
    "KEN_BURNS_START",
    "KEN_BURNS_END",
    "PUNCH_IN_END",
    "SUBTITLE_FONT_SIZE",
    "SUBTITLE_BORDER_WIDTH",
    "SUBTITLE_FONT_CANDIDATES",
    "TRACK_BGM",
    "TRACK_SFX",
    "CapCutError",
    "contiguous_ranges",
    "cue_ranges",
    "loop_ranges",
    "detect_capcut_drafts_dir",
    "ensure_background",
    "panel_placement",
    "build_capcut_draft",
    "copy_draft",
]
