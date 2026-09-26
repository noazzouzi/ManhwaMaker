"""Projet Kdenlive tiré de la timeline, rendu sans interface par ``melt`` (essai, à côté de CapCut).

Même plan de montage que le brouillon CapCut et l'aperçu (``timeline.json``). Le projet
(``<nom>.kdenlive``) s'ouvre dans Kdenlive pour les retouches, et la même description
MLT est rendue directement par ``melt`` : aucun passage manuel dans un éditeur.

Choix de l'essai :

- **une image par case**, composée comme l'aperçu à l'instant 0 : fond flouté et assombri,
  case au premier plan en résolution native, étalonnée selon l'émotion (:func:`render_stills`).
  Le zoom ≤ 105 % est un effet « Transform » (``qtblend``) à images clés sur ce plan : il
  agrandit aussi le fond, déjà flou, ce qui ne se voit pas ;
- style (quelle transition, quel mouvement, quelle couleur) : :mod:`src.modules.kdenlive_style`.
  Fondus, volets, poussées et glissements sont des **fondus enchaînés sur la même piste**
  (« mix » de Kdenlive, deux sous-pistes par piste) ; fondu au noir et glitch, des **effets de
  part et d'autre de la coupe** ; flashs blancs (fondu au blanc, impacts) sur une piste V3 ;
- sous-titres dans un fichier ASS rendu par libass (police de l'aperçu, contour 2 px,
  animations d'entrée approchées par les balises ASS). Présents dans le rendu ``melt``,
  absents du ``.kdenlive`` : Kdenlive plante sur un filtre de sous-titres écrit à la main,
  le fichier s'y importe en deux clics ;
- musique à −22 dB (bouclée, fondus aux changements d'ambiance), bruitages, voix ;
- effets superposés (lueur, pluie, lignes de vitesse, étincelles) : calques transparents
  fabriqués une fois par :func:`render_vfx_loops` à partir des calques de l'aperçu, en boucle
  sur une piste V2. Le ``frei0r.glow`` de MLT coûtait 34 s de rendu pour 24 s de lueur.

Structure du document : format Kdenlive 1.04 (sans séquences), que Kdenlive met à niveau à
l'ouverture ; la copie ``render.mlt`` n'a pas de ``producer="main_bin"`` pour que ``melt``
rende la timeline et non le chutier.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import os
import subprocess
import time
import xml.etree.ElementTree as ET
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf
from PIL import Image

from src.models.timeline import PanelClip, SubtitleCue, Timeline
from src.modules.preview_renderer import (
    FONT_CANDIDATES,
    SUBTITLE_BOTTOM_MARGIN,
    SUBTITLE_FONT_SIZE,
    SUBTITLE_MAX_WIDTH_RATIO,
    SUBTITLE_STROKE_PX,
    VFX_STRENGTH,
    PreviewRenderer,
    apply_vfx,
)
from src.modules.kdenlive_style import GRADES, MIX_KINDS, VIGNETTE, Cut, grade, motion_pattern, plan_cuts
from src.modules.timeline_builder import KEN_BURNS_ZOOM, PUNCH_IN_S, load_timeline
from src.utils.config import PROJECT_ROOT

logger = logging.getLogger(__name__)

__all__ = ["KDENLIVE_BIN", "find_melt", "clip_emotions", "render_stills", "render_vfx_loops", "write_ass", "KdenliveProject",
           "build_project", "render"]

#: Dossier des exécutables de Kdenlive (installation winget par utilisateur).
KDENLIVE_BIN = Path(os.environ.get("LOCALAPPDATA", "")) / "Programs" / "kdenlive" / "bin"
LUMA_DIR = KDENLIVE_BIN / "data" / "kdenlive" / "lumas" / "HD"
#: Bord doux des volets (0 = net).
WIPE_SOFTNESS = 0.2
#: Flou des transitions « whip » et « fondu flou » (frei0r IIRblur, 0 à 1).
WHIP_BLUR, SOFT_BLUR = 0.45, 0.25
OPPOSITE = {"left": "right", "right": "left", "up": "down", "down": "up"}
#: Tremblement après un punch-in : décalages successifs, en fraction de la demi-marge des 5 %.
SHAKE = ((0.9, 0.5), (-0.8, -0.6), (0.6, 0.7), (-0.4, -0.4), (0.2, 0.2), (0.0, 0.0))
SHAKE_STEP_S = 0.05
#: Flash blanc : impact d'un punch-in (durée, opacité de départ) et fondu au blanc.
IMPACT_FLASH_S, IMPACT_FLASH = 0.15, 0.7
#: Bruitages ajoutés par le style : « swoosh » des changements de scène rapides, impact.
SWOOSH_FILE = PROJECT_ROOT / "config" / "sfx" / "swoosh.wav"
IMPACT_FILE = PROJECT_ROOT / "config" / "sfx" / "impact.wav"
SWOOSH_GAIN_DB, IMPACT_GAIN_DB = -14.0, -10.0
#: Boucle de chaque effet superposé (s) : une période exacte du calque de l'aperçu (la pluie
#: n'est pas périodique : saut discret toutes les 5 s).
VFX_LOOP_S = {"glow": 2.5, "speed_lines": 1.25, "sparks": 10.0, "rain": 5.0}
#: libass rend un corps donné 1,5 fois plus petit que PIL (mesuré sur deux sous-titres de
#: l'aperçu) : même taille à l'écran que l'aperçu.
ASS_FONT_SCALE = 1.5
#: Clés d'animation MLT : cubique entrée-sortie (zoom lent), cubique sortie (punch-in).
EASE_IN_OUT, EASE_OUT = "i", "h"
#: Composition interne de Kdenlive (marqueur des pistes et fondus qu'il gère lui-même).
INTERNAL = "237"


def find_melt() -> Path:
    melt = KDENLIVE_BIN / "melt.exe"
    if not melt.is_file():
        raise FileNotFoundError(f"melt introuvable : {melt} (installer Kdenlive : winget install KDE.Kdenlive)")
    return melt


# --- Images des cases ---------------------------------------------------------------------
def clip_emotions(timeline: Timeline, chapter_dir: Path | None = None) -> list[str]:
    """Émotion de chaque case : celle de la timeline, sinon celle de ``scenes.json`` (anciennes timelines)."""
    emotions = [clip.emotion for clip in timeline.clips]
    scenes = chapter_dir / "scenes.json" if chapter_dir else None
    if not any(emotions) and scenes is not None and scenes.is_file():
        by_scene = {s["index"]: s.get("emotion", "") for s in json.loads(scenes.read_text(encoding="utf-8"))["scenes"]}
        emotions = [by_scene.get(clip.scene_index, "") for clip in timeline.clips]
    return emotions


def render_stills(timeline: Timeline, out_dir: Path, emotions: Sequence[str] | None = None) -> list[Path]:
    """Une image par case : l'aperçu à l'instant 0 (fond flouté + case native), étalonné selon l'émotion.

    Le nom porte une empreinte du chemin de la case, de l'émotion et de l'étalonnage : deux
    chapitres d'une compilation ont chacun leur ``panel_001.png``, et une image faite avec un
    autre réglage n'est pas réutilisée.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    renderer = PreviewRenderer(timeline, show_subtitles=False)
    emotions = list(emotions) if emotions else [""] * len(timeline.clips)
    paths = []
    for clip, emotion in zip(timeline.clips, emotions):
        recipe = f"{Path(timeline.panels_dir) / clip.file}|{emotion}|{GRADES.get(emotion)}|{VIGNETTE}|{timeline.width}x{timeline.height}"
        path = out_dir / f"{Path(clip.file).stem}_{hashlib.sha1(recipe.encode()).hexdigest()[:10]}.png"
        if not path.is_file():
            frame = renderer.compose_panel(renderer.prepare_clip(clip), 0.0, 0.0)
            Image.fromarray(grade(frame, emotion)).save(path, compress_level=1)
        paths.append(path)
    return paths


# --- Effets superposés ---------------------------------------------------------------------
def _pulse(t: float) -> float:
    """Battement de la lueur de l'aperçu (entre 0,3 et 1)."""
    return 0.65 + 0.35 * math.sin(2.0 * math.pi * t / VFX_LOOP_S["glow"])


def render_vfx_loops(timeline: Timeline, out_dir: Path) -> dict[tuple[str, float], tuple[Path, int, bool]]:
    """Calque transparent par effet et opacité : ``{(effet, opacité): (fichier, images, image fixe ?)}``.

    Le calque de l'aperçu est ajouté à l'image ; ici il devient un **calque transparent**
    (blanc, transparence = intensité), composé normalement : Kdenlive ne garde pas les modes
    de composition des pistes à l'ouverture. La lueur est une image fixe dont l'opacité bat
    (:func:`_pulse`) ; les particules, une courte vidéo QuickTime RLE avec transparence.
    """
    import imageio_ffmpeg

    out_dir.mkdir(parents=True, exist_ok=True)
    w, h, fps = timeline.width, timeline.height, timeline.fps
    loops: dict[tuple[str, float], tuple[Path, int, bool]] = {}
    for vfx in timeline.vfx:
        key = (vfx.kind, round(vfx.opacity, 2))
        if vfx.kind not in VFX_LOOP_S or key in loops:
            continue
        strength = VFX_STRENGTH * vfx.opacity
        n = round(VFX_LOOP_S[vfx.kind] * fps)
        stem = out_dir / f"{vfx.kind}_{round(vfx.opacity * 100)}_{w}x{h}"
        if vfx.kind == "glow":
            path = stem.with_suffix(".png")
            if not path.is_file():
                alpha = np.zeros((h, w, 3), np.uint8)
                apply_vfx(alpha, "glow", VFX_LOOP_S["glow"] / 4, strength=strength)  # battement au maximum
                rgba = np.dstack([np.full((h, w, 3), 255, np.uint8), alpha[:, :, 0]])
                Image.fromarray(rgba, "RGBA").save(path, compress_level=1)
            loops[key] = (path, n, True)
            continue
        path = stem.parent / f"{stem.name}_{fps}fps.mov"
        if not path.is_file():
            writer = imageio_ffmpeg.write_frames(str(path), (w, h), fps=fps, codec="qtrle", pix_fmt_in="rgba",
                                                 pix_fmt_out="argb", macro_block_size=1)
            writer.send(None)
            for i in range(n):
                mask = np.zeros((h, w, 3), np.uint8)
                apply_vfx(mask, vfx.kind, i / fps, strength=strength)
                writer.send(np.dstack([np.full((h, w, 3), 255, np.uint8), mask[:, :, 0]]))
            writer.close()
        loops[key] = (path, n, False)
    return loops


# --- Sous-titres ASS ----------------------------------------------------------------------
def _ass_time(seconds: float) -> str:
    cs = max(0, round(seconds * 100))
    return f"{cs // 360000}:{cs // 6000 % 60:02d}:{cs // 100 % 60:02d}.{cs % 100:02d}"


def _ass_tags(cue: SubtitleCue) -> str:
    """Animation d'entrée CapCut approchée en balises ASS (même familles que l'aperçu)."""
    anim = cue.animation
    if anim is None or not anim.intro:
        return ""
    ms = max(60, round(anim.duration_s * 1000))
    family = {"弹入": "bounce", "放大": "scale", "渐显": "fade", "模糊": "blur", "故障": "glitch",
              "逐字": "fade", "打字机": "fade"}.get(anim.intro, "fade")
    if family == "bounce":
        return f"{{\\fscx60\\fscy60\\t(0,{ms * 7 // 10},\\fscx112\\fscy112)\\t({ms * 7 // 10},{ms},\\fscx100\\fscy100)}}"
    if family == "scale":
        return f"{{\\fscx70\\fscy70\\t(0,{ms},\\fscx100\\fscy100)}}"
    if family == "blur":
        return f"{{\\fad({ms},0)\\blur8\\t(0,{ms},\\blur0)}}"
    if family == "glitch":
        return f"{{\\fad({ms // 2},0)\\fsp6\\t(0,{ms},\\fsp0)}}"
    return f"{{\\fad({ms},0)}}"


def _font() -> tuple[str, Path]:
    """Famille et dossier de la police des sous-titres (la première trouvée, comme l'aperçu)."""
    from PIL import ImageFont

    for candidate in FONT_CANDIDATES:
        path = Path(candidate)
        if path.is_file():
            family, _ = ImageFont.truetype(str(path), 10).getname()
            return family, path.parent
    return "Arial", Path("C:/Windows/Fonts")


def write_ass(timeline: Timeline, path: Path) -> tuple[Path, Path]:
    """Fichier ASS des sous-titres ; renvoie ``(fichier, dossier de la police)``."""
    family, fonts_dir = _font()
    ratio = timeline.height / 1080
    margin = round(timeline.width * (1 - SUBTITLE_MAX_WIDTH_RATIO) / 2)
    lines = [
        "[Script Info]", "ScriptType: v4.00+", f"PlayResX: {timeline.width}", f"PlayResY: {timeline.height}",
        "WrapStyle: 0", "ScaledBorderAndShadow: yes", "",
        "[V4+ Styles]",
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, "
        "Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, "
        "MarginR, MarginV, Encoding",
        f"Style: Default,{family},{round(SUBTITLE_FONT_SIZE * ASS_FONT_SCALE * ratio)},&H00FFFFFF,&H00FFFFFF,&H00000000,&H00000000,"
        f"-1,0,0,0,100,100,0,0,1,{max(1, round(SUBTITLE_STROKE_PX * ratio))},0,2,{margin},{margin},"
        f"{round(SUBTITLE_BOTTOM_MARGIN * ratio)},1",
        "", "[Events]", "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text",
    ]
    for cue in timeline.subtitles:
        text = cue.text.replace("\n", " ").replace("{", "(").replace("}", ")")
        lines.append(f"Dialogue: 0,{_ass_time(cue.start_s)},{_ass_time(cue.end_s)},Default,,0,0,0,,{_ass_tags(cue)}{text}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8-sig")
    return path, fonts_dir


# --- Document MLT / Kdenlive --------------------------------------------------------------
def _props(element: ET.Element, **props) -> ET.Element:
    for name, value in props.items():
        ET.SubElement(element, "property", name=name.replace("__", ".").replace("_colon_", ":")).text = str(value)
    return element


def _prop(element: ET.Element, name: str, value) -> None:
    ET.SubElement(element, "property", name=name).text = str(value)


class _Doc:
    """Assemble le XML : producteurs (un par usage, partageant le ``kdenlive:id`` du chutier)."""

    def __init__(self, timeline: Timeline) -> None:
        self.fps = timeline.fps
        self.root = ET.Element("mlt", LC_NUMERIC="C", producer="main_bin", version="7.41.0", root="")
        ET.SubElement(self.root, "profile", description=f"HD {timeline.height}p {timeline.fps} fps",
                      width=str(timeline.width), height=str(timeline.height), progressive="1",
                      sample_aspect_num="1", sample_aspect_den="1", display_aspect_num="16", display_aspect_den="9",
                      frame_rate_num=str(timeline.fps), frame_rate_den="1", colorspace="709")
        self.producers: list[ET.Element] = []
        self.body: list[ET.Element] = []  # sous-pistes puis pistes, dans l'ordre de leurs références
        self.n = 0
        self.bin: dict[str, tuple[int, str]] = {}  # fichier -> (kdenlive:id, producteur du chutier)
        self.bin_order: list[tuple[str, int]] = []

    def frames(self, seconds: float) -> int:
        return round(seconds * self.fps)

    def uid(self, prefix: str) -> str:
        self.n += 1
        return f"{prefix}{self.n}"

    def producer(self, resource: Path | str, *, image: bool = False, length: int, color: bool = False) -> str:
        """Nouveau producteur pour ``resource`` (un par usage dans la timeline ; ``color`` : couleur ``0xRRGGBBAA``)."""
        pid = self.uid("producer")
        element = ET.Element("producer", id=pid, **{"in": "0", "out": str(length - 1)})
        self.producers.append(element)
        key = f"color:{resource}" if color else str(resource)
        if key not in self.bin:
            self.bin[key] = (len(self.bin) + 2, pid)
            self.bin_order.append((pid, length))
        kid = self.bin[key][0]
        _prop(element, "length", length)
        _prop(element, "eof", "pause")
        _prop(element, "resource", resource if color else resource.as_posix())
        _prop(element, "mlt_service", "color" if color else "qimage" if image else "avformat-novalidate")
        if color:
            _prop(element, "mlt_image_format", "rgba")
        if image or color:
            _prop(element, "ttl", 25)
            _prop(element, "aspect_ratio", 1)
            _prop(element, "kdenlive:duration", length)
        _prop(element, "kdenlive:clipname", "Flash" if color else resource.name)
        _prop(element, "kdenlive:folderid", -1)
        _prop(element, "kdenlive:id", kid)
        return pid


@dataclass
class _Placed:
    """Une case placée : bornes réelles (chevauchements compris), sous-piste et transitions voisines."""

    clip: PanelClip
    still: Path
    start: int
    end: int
    sub: int
    cut_in: Cut | None = None
    cut_out: Cut | None = None


def _layout(doc: _Doc, timeline: Timeline, stills: list[Path], cuts: list[Cut | None]) -> tuple[list[_Placed], list[dict]]:
    """Place les cases sur les deux sous-pistes et liste les fondus enchaînés (« mix »), centrés sur la coupe."""
    clips = timeline.clips
    placed = [_Placed(c, s, doc.frames(c.start_s), doc.frames(c.end_s) if i + 1 == len(clips) else doc.frames(clips[i + 1].start_s), 0)
              for i, (c, s) in enumerate(zip(clips, stills))]
    mixes = []
    for i in range(1, len(placed)):
        prev, cur = placed[i - 1], placed[i]
        cur.sub = prev.sub
        cut = cuts[i - 1]
        if cut is None:
            continue
        prev.cut_out = cur.cut_in = cut
        if cut.kind in MIX_KINDS:
            m = max(2, doc.frames(cut.duration_s))
            h = m // 2
            at = cur.start
            cur.start, prev.end = at - h, at - h + m
            cur.sub = 1 - prev.sub
            mixes.append({"in": at - h, "out": at - h + m - 1, "reverse": int(cur.sub == 0), "mixcut": h, "cut": cut})
    return placed, mixes


def _motion_filter(entry: ET.Element, placed: _Placed, index: int, width: int, height: int, fps: int) -> None:
    """Effet Transform : zoom 100 <-> 105 % vers un point qui varie, ou punch-in suivi d'un tremblement."""
    length = placed.end - placed.start
    last = length - 1
    z_w, z_h = round(width * (1 + KEN_BURNS_ZOOM)), round(height * (1 + KEN_BURNS_ZOOM))
    mx, my = z_w - width, z_h - height
    small = f"0 0 {width} {height} 1"
    if placed.clip.motion == "punch_in":
        cx, cy = -(mx // 2), -(my // 2)
        big = f"{cx} {cy} {z_w} {z_h} 1"
        hit = min(last, max(1, round(PUNCH_IN_S * fps)))
        keys = [f"0{EASE_OUT}={small}", f"{hit}={big}"]
        step = max(1, round(SHAKE_STEP_S * fps))
        for j, (sx, sy) in enumerate(SHAKE, 1):  # tremblement dans la marge des 5 % : jamais de bord visible
            if hit + j * step >= last:
                break
            keys.append(f"{hit + j * step}={cx + round(sx * mx / 2)} {cy + round(sy * my / 2)} {z_w} {z_h} 1")
        keys.append(f"{last}={big}")
    else:
        direction, ax, ay = motion_pattern(index)
        big = f"{-round(ax * mx)} {-round(ay * my)} {z_w} {z_h} 1"
        start, end = (small, big) if direction == "in" else (big, small)
        keys = [f"0{EASE_IN_OUT}={start}", f"{last}={end}"]
    _props(ET.SubElement(entry, "filter"), mlt_service="qtblend", kdenlive_id="qtblend", rect=";".join(keys),
           rotation=0, compositing=0, distort=0, rotate_center=1)


def _edge_filters(entry: ET.Element, cut: Cut, *, length: int, span: int, outgoing: bool) -> None:
    """Effets d'une case au bord d'une transition, sur ses ``span`` dernières (ou premières) images."""
    span = max(1, min(span, length))
    first, last = (length - span, length - 1) if outgoing else (0, span - 1)

    def add(service: str, **props) -> None:
        _props(ET.SubElement(entry, "filter", **{"in": str(first), "out": str(last)}), mlt_service=service,
               kdenlive_id=service, **props)

    if cut.kind == "dip_black":
        add("brightness", level=f"0=1;{span - 1}=0" if outgoing else f"0=0;{span - 1}=1")
    elif cut.kind == "glitch":
        # Réglé à l'œil : fréquence 0,9 et couleurs 0,4 rendaient l'image illisible.
        add("frei0r.rgbsplit0r", **{"0": 0.51, "1": 0.53})
        add("frei0r.glitch0r", **{"0": 0.25, "1": 0.1, "2": 0.12, "3": 0.03})
    elif cut.kind in ("blur_dissolve", "whip"):
        add("frei0r.IIRblur", **{"0": WHIP_BLUR if cut.kind == "whip" else SOFT_BLUR})


def _mix(track: ET.Element, doc: _Doc, mix: dict) -> None:
    """Fondu enchaîné sur la même piste : volet (``luma``) ou poussée / glissement (``frei0r``)."""
    cut: Cut = mix["cut"]
    m = mix["out"] - mix["in"] + 1
    t = ET.SubElement(track, "transition", id=doc.uid("transition"), **{"in": str(mix["in"]), "out": str(mix["out"])})
    common = {"a_track": 0, "b_track": 1, "kdenlive_colon_mixcut": mix["mixcut"], "reverse": mix["reverse"]}
    if cut.kind in ("push", "slide", "whip"):
        # frei0r va de la sous-piste a (0) vers b (1) : si la case entrante est en a, on joue la
        # transition à l'envers, dans la direction opposée (même mouvement à l'écran).
        direction = OPPOSITE[cut.option] if mix["reverse"] else cut.option
        service = f"frei0r.sleid0r_{'slide' if cut.kind == 'slide' else 'push'}-{direction}"
        position = f"0=1;{m - 1}=0" if mix["reverse"] else f"0=0;{m - 1}=1"
        _props(t, mlt_service=service, kdenlive_id=service, position=position, **common)
    else:
        wipe = cut.kind == "wipe"
        _props(t, factory="loader", mlt_service="luma", kdenlive_id="luma", alpha_over=1, invert=0,
               softness=WIPE_SOFTNESS if wipe else 0, resource=(LUMA_DIR / cut.option).as_posix() if wipe else "",
               **common)


def _track(doc: _Doc, playlists: list[ET.Element], *, audio: bool) -> ET.Element:
    tractor = ET.Element("tractor", id=doc.uid("tractor"), **{"in": "0"})
    doc.body.append(tractor)
    if audio:
        _prop(tractor, "kdenlive:audio_track", 1)
    _props(tractor, **{"kdenlive_colon_trackheight": 67, "kdenlive_colon_timeline_active": 1, "kdenlive_colon_collapsed": 0})
    for playlist in playlists:
        ET.SubElement(tractor, "track", producer=playlist.get("id"), hide="video" if audio else "audio")
    return tractor


def _playlist(doc: _Doc, audio: bool) -> ET.Element:
    playlist = ET.Element("playlist", id=doc.uid("playlist"))
    doc.body.append(playlist)
    if audio:
        _prop(playlist, "kdenlive:audio_track", 1)
    return playlist


class _Cursor:
    """Remplit une sous-piste dans l'ordre : blancs, puis entrées."""

    def __init__(self, playlist: ET.Element) -> None:
        self.playlist, self.at = playlist, 0

    def add(self, producer: str, start: int, length: int, src_in: int = 0) -> ET.Element:
        if start > self.at:
            ET.SubElement(self.playlist, "blank", length=str(start - self.at))
        elif start < self.at:
            raise ValueError(f"chevauchement sur la sous-piste : {start} < {self.at}")
        entry = ET.SubElement(self.playlist, "entry", producer=producer,
                              **{"in": str(src_in), "out": str(src_in + length - 1)})
        self.at = start + length
        return entry


def _audio_length(path: Path, fps: int) -> int:
    info = sf.info(str(path))
    return max(1, int(info.frames / info.samplerate * fps))


@dataclass
class KdenliveProject:
    project: Path
    render_mlt: Path
    subtitles: Path
    stills: list[Path]
    frames: int


def build_project(
    timeline: Timeline, out_dir: Path, *, name: str = "montage", emotions: Sequence[str] | None = None,
    chapter_starts: frozenset[int] = frozenset(),
) -> KdenliveProject:
    """Écrit ``<name>.kdenlive`` (Kdenlive) et ``render.mlt`` (melt) dans ``out_dir``.

    Args:
        emotions: émotion de la scène de chaque case (défaut : :func:`clip_emotions`).
        chapter_starts: rangs des cases qui ouvrent un chapitre (compilation).
    """
    # Chemins absolus : Kdenlive plante sur un projet dont les médias sont en chemin relatif.
    out_dir = Path(out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    doc = _Doc(timeline)
    fps, width, height = timeline.fps, timeline.width, timeline.height
    total = doc.frames(timeline.total_duration_s)
    emotions = list(emotions) if emotions else clip_emotions(timeline)
    clips = timeline.clips
    cuts = plan_cuts(clips, emotions, chapter_starts=chapter_starts)
    stills = render_stills(timeline, out_dir / "stills", emotions)
    ass, fonts_dir = write_ass(timeline, out_dir / f"{name}.kdenlive.ass")

    # Piste noire de fond (convention Kdenlive), puis pistes audio (du bas vers A1), puis vidéo.
    black = ET.Element("producer", id="black_track", **{"in": "0", "out": str(total - 1)})
    _props(black, length=2147483647, eof="continue", resource="black", aspect_ratio=1, mlt_service="color",
           mlt_image_format="rgba", set__test_audio=0)

    audio_tracks = []
    # Bruitages : ceux de la timeline, plus le « swoosh » des changements de scène rapides et
    # l'impact des punch-in qui n'en ont pas. Deux pistes : un bruitage long n'en masque aucun.
    events = [(s.start_s, Path(s.file), s.gain_db) for s in timeline.sfx]
    for i, cut in enumerate(cuts):
        if cut is not None and cut.scene_change and cut.kind in ("whip", "push") and SWOOSH_FILE.is_file():
            events.append((max(0.0, clips[i + 1].start_s - cut.duration_s / 2), SWOOSH_FILE, SWOOSH_GAIN_DB))
    for clip in clips:
        if clip.motion == "punch_in" and IMPACT_FILE.is_file() and all(abs(t - clip.start_s) > 0.4 for t, _, _ in events):
            events.append((clip.start_s, IMPACT_FILE, IMPACT_GAIN_DB))
    sfx_pls = [[_playlist(doc, True), _playlist(doc, True)], [_playlist(doc, True), _playlist(doc, True)]]
    sfx_cursors = [_Cursor(sfx_pls[0][0]), _Cursor(sfx_pls[1][0])]
    for start_s, path, gain_db in sorted(events, key=lambda e: e[0]):
        start = doc.frames(start_s)
        cursor = next((c for c in sfx_cursors if c.at <= start), None)
        length = min(_audio_length(path, fps), total - start)
        if cursor is None or length <= 0:
            continue
        entry = cursor.add(doc.producer(path, image=False, length=_audio_length(path, fps)), start, length)
        _props(ET.SubElement(entry, "filter"), mlt_service="volume", kdenlive_id="volume", level=gain_db)
    audio_tracks += [_track(doc, sfx_pls[1], audio=True), _track(doc, sfx_pls[0], audio=True)]
    # Musique : deux pistes en alternance, pour les fondus enchaînés entre ambiances.
    bgm_pl = [[_playlist(doc, True), _playlist(doc, True)], [_playlist(doc, True), _playlist(doc, True)]]
    cursors = [_Cursor(bgm_pl[0][0]), _Cursor(bgm_pl[1][0])]
    for k, seg in enumerate(timeline.bgm):
        path = Path(seg.file)
        file_len = _audio_length(path, fps)
        start, end = doc.frames(seg.start_s), min(total, doc.frames(seg.end_s))
        pieces, at = [], start
        while at < end:  # musique bouclée : la même piste répétée
            pieces.append((at, min(file_len, end - at)))
            at += pieces[-1][1]
        for p, (at, length) in enumerate(pieces):
            entry = cursors[k % 2].add(doc.producer(path, image=False, length=file_len), at, length)
            level = [f"0={seg.gain_db}"]
            if p == 0 and seg.fade_in_s > 0:
                level = [f"0=-60", f"{min(length - 1, doc.frames(seg.fade_in_s))}={seg.gain_db}"]
            if p == len(pieces) - 1 and seg.fade_out_s > 0:
                level += [f"{max(1, length - 1 - doc.frames(seg.fade_out_s))}={seg.gain_db}", f"{length - 1}=-60"]
            _props(ET.SubElement(entry, "filter"), mlt_service="volume", kdenlive_id="volume", level=";".join(level))
    audio_tracks += [_track(doc, bgm_pl[1], audio=True), _track(doc, bgm_pl[0], audio=True)]
    # Voix off (A1).
    voice_pl = [_playlist(doc, True), _playlist(doc, True)]
    cursor = _Cursor(voice_pl[0])
    for clip in timeline.audio:
        path = Path(timeline.audio_dir) / clip.file
        start = max(cursor.at, doc.frames(clip.start_s))
        length = min(doc.frames(clip.end_s), total) - start
        if length > 0:
            cursor.add(doc.producer(path, image=False, length=max(length, _audio_length(path, fps))), start, length)
    audio_tracks.append(_track(doc, voice_pl, audio=True))

    # Vidéo : une image par case, deux sous-pistes pour les fondus enchaînés.
    video_pl = [_playlist(doc, False), _playlist(doc, False)]
    cursors = [_Cursor(video_pl[0]), _Cursor(video_pl[1])]
    placed, mixes = _layout(doc, timeline, stills, cuts)
    flashes: list[tuple[int, int, str]] = []  # (début, durée, « dip » ou « impact »)
    for index, item in enumerate(placed):
        length = item.end - item.start
        entry = cursors[item.sub].add(doc.producer(item.still, image=True, length=length), item.start, length)
        _motion_filter(entry, item, index, width, height, fps)
        for cut, outgoing in ((item.cut_out, True), (item.cut_in, False)):
            if cut is not None:
                m = max(2, doc.frames(cut.duration_s))
                _edge_filters(entry, cut, length=length, span=m if cut.kind in MIX_KINDS else m // 2, outgoing=outgoing)
        if item.cut_in is not None and item.cut_in.kind == "dip_white":
            m = max(2, doc.frames(item.cut_in.duration_s))
            flashes.append((item.start - m // 2, m, "dip"))
        elif item.clip.motion == "punch_in":
            flashes.append((item.start, max(2, doc.frames(IMPACT_FLASH_S)), "impact"))
            _props(ET.SubElement(entry, "filter", **{"in": "0", "out": str(min(length, doc.frames(IMPACT_FLASH_S)) - 1)}),
                   mlt_service="frei0r.rgbsplit0r", kdenlive_id="frei0r.rgbsplit0r", **{"0": 0.5, "1": 0.54})
    video = _track(doc, video_pl, audio=False)
    for mix in mixes:
        _mix(video, doc, mix)
    video_tracks = [video]
    loops = render_vfx_loops(timeline, out_dir / "vfx")
    if loops:  # piste V2 : calques d'effets transparents, en boucle
        vfx_pl = [_playlist(doc, False), _playlist(doc, False)]
        cursor = _Cursor(vfx_pl[0])
        for vfx in sorted(timeline.vfx, key=lambda v: v.start_s):
            if (vfx.kind, round(vfx.opacity, 2)) not in loops:
                continue
            path, n, still = loops[(vfx.kind, round(vfx.opacity, 2))]
            at, end = max(cursor.at, doc.frames(vfx.start_s)), min(total, doc.frames(vfx.end_s))
            while at < end:
                length = min(n, end - at)
                entry = cursor.add(doc.producer(path, image=still, length=n), at, length)
                if still:  # lueur : opacité qui bat, une période par boucle
                    keys = ";".join(f"{round(k * n / 8)}~=0 0 {width} {height} {_pulse(k * VFX_LOOP_S['glow'] / 8) / _pulse(VFX_LOOP_S['glow'] / 4):.3f}"
                                    for k in range(9))
                    _props(ET.SubElement(entry, "filter"), mlt_service="qtblend", kdenlive_id="qtblend", rect=keys,
                           rotation=0, compositing=0, distort=0)
                at += length
        video_tracks.append(_track(doc, vfx_pl, audio=False))
    if flashes:  # piste V3 : flashs blancs (fondu au blanc, impacts)
        flash_pl = [_playlist(doc, False), _playlist(doc, False)]
        cursor = _Cursor(flash_pl[0])
        for start, length, kind in sorted(flashes):
            start = max(start, cursor.at, 0)
            length = min(length, total - start)
            if length < 2:
                continue
            entry = cursor.add(doc.producer("0xffffffff", color=True, length=length), start, length)
            full = f"0 0 {width} {height}"
            if kind == "dip":
                rect = f"0={full} 0;{length // 2}={full} 1;{length - 1}={full} 0"
            else:
                rect = f"0{EASE_OUT}={full} {IMPACT_FLASH};{length - 1}={full} 0"
            _props(ET.SubElement(entry, "filter"), mlt_service="qtblend", kdenlive_id="qtblend", rect=rect,
                   rotation=0, compositing=0, distort=0)
        video_tracks.append(_track(doc, flash_pl, audio=False))

    # Chutier (Kdenlive) : une entrée par fichier.
    main_bin = ET.Element("playlist", id="main_bin")
    _props(main_bin, **{
        "kdenlive_colon_docproperties__version": "1.04", "kdenlive_colon_docproperties__kdenliveversion": "26.08.1",
        "kdenlive_colon_docproperties__profile": f"atsc_1080p_{fps}", "kdenlive_colon_docproperties__compositing": 1,
        "kdenlive_colon_docproperties__audioChannels": 2, "kdenlive_colon_docproperties__documentid": int(time.time() * 1000),
        "kdenlive_colon_docproperties__videoTarget": len(audio_tracks) + 1, "kdenlive_colon_docproperties__audioTarget": len(audio_tracks),
        "kdenlive_colon_docproperties__activeTrack": len(audio_tracks) + 1, "xml_retain": 1,
    })
    for pid, length in doc.bin_order:
        ET.SubElement(main_bin, "entry", producer=pid, **{"in": "0", "out": str(length - 1)})
    # Ordre du document : chaque élément est défini avant d'être référencé.
    doc.root.extend([*doc.producers, main_bin, black, *doc.body])

    main = ET.SubElement(doc.root, "tractor", id=doc.uid("tractor"), **{"in": "0", "out": str(total - 1)})
    ET.SubElement(main, "track", producer="black_track")
    for tractor in [*audio_tracks, *video_tracks]:
        ET.SubElement(main, "track", producer=tractor.get("id"))
    for index in range(1, len(audio_tracks) + 1):
        _props(ET.SubElement(main, "transition", id=doc.uid("transition")), a_track=0, b_track=index, mlt_service="mix",
               kdenlive_id="mix", internal_added=INTERNAL, always_active=1, accepts_blanks=1, sum=1)
    for k in range(len(video_tracks)):
        _props(ET.SubElement(main, "transition", id=doc.uid("transition")), a_track=0, b_track=len(audio_tracks) + 1 + k,
               mlt_service="qtblend", kdenlive_id="qtblend", internal_added=INTERNAL, always_active=1, compositing=0)
    ET.indent(doc.root, space=" ")
    project = out_dir / f"{name}.kdenlive"
    ET.ElementTree(doc.root).write(project, encoding="utf-8", xml_declaration=True)
    # Sous-titres : seulement dans la copie rendue par melt. Kdenlive 26.08 plante au chargement
    # d'un filtre de sous-titres écrit à la main (il attend sa propre liste de fichiers) : dans
    # Kdenlive, le fichier ASS s'importe (Projet > Sous-titres > Importer).
    _props(ET.SubElement(main, "filter", id=doc.uid("filter")), mlt_service="avfilter.subtitles",
           internal_added=INTERNAL, av__filename=ass.as_posix(), av__fontsdir=fonts_dir.as_posix(), av__alpha=1)
    del doc.root.attrib["producer"]
    render_mlt = out_dir / "render.mlt"
    ET.ElementTree(doc.root).write(render_mlt, encoding="utf-8", xml_declaration=True)
    logger.info("Projet Kdenlive : %s (%d cases, %d fondus enchaines, %d images)", project, len(placed), len(mixes), total)
    return KdenliveProject(project, render_mlt, ass, stills, total)


def render(project: KdenliveProject, out: Path, *, vcodec: str = "h264_amf", threads: int = 0,
           seconds: float | None = None, fps: int = 60) -> float:
    """Rend ``render.mlt`` avec melt ; renvoie la durée du rendu (s)."""
    threads = threads or os.cpu_count() or 4
    codec = (["vcodec=h264_amf", "rc=cqp", "qp_i=20", "qp_p=22", "qp_b=24", "quality=balanced"]
             if vcodec == "h264_amf" else ["vcodec=libx264", "crf=20", "preset=veryfast"])
    cmd = [str(find_melt()), str(project.render_mlt)]
    if seconds:
        cmd += ["out=" + str(round(seconds * fps) - 1)]
    cmd += ["-consumer", f"avformat:{out}", f"real_time=-{threads}", "f=mp4", *codec, "pix_fmt=yuv420p",
            "acodec=aac", "ab=192k", "ar=48000", "channels=2", "movflags=+faststart", "-silent"]
    started = time.perf_counter()
    proc = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    elapsed = time.perf_counter() - started
    if proc.returncode != 0 or not out.is_file():
        raise RuntimeError(f"melt en echec ({proc.returncode}) : {(proc.stderr or proc.stdout)[-800:]}")
    return elapsed


def main() -> None:
    parser = argparse.ArgumentParser(description="Projet Kdenlive + rendu melt d'un chapitre (essai)")
    parser.add_argument("chapter", type=Path, help="dossier du chapitre (contient timeline.json)")
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--seconds", type=float, default=None, help="ne rendre que le début")
    parser.add_argument("--codec", default="h264_amf", choices=["h264_amf", "libx264"])
    parser.add_argument("--fps", type=int, default=None, help="images par seconde (defaut : celles de la timeline)")
    parser.add_argument("--no-render", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    timeline = load_timeline(args.chapter / "timeline.json")
    if args.fps:
        timeline = timeline.model_copy(update={"fps": args.fps})
    out_dir = args.out or args.chapter / "kdenlive"
    started = time.perf_counter()
    project = build_project(timeline, out_dir, name=args.chapter.name, emotions=clip_emotions(timeline, args.chapter))
    logger.info("Projet ecrit en %.1fs", time.perf_counter() - started)
    if args.no_render:
        return
    suffix = (f"_{int(args.seconds)}s" if args.seconds else "") + f"_{timeline.fps}fps"
    out = out_dir / f"melt_{args.codec}{suffix}.mp4"
    elapsed = render(project, out, vcodec=args.codec, seconds=args.seconds, fps=timeline.fps)
    video_s = (args.seconds or timeline.total_duration_s)
    logger.info("Rendu melt (%s) : %.1fs pour %.1fs de video (x%.2f temps reel) -> %s", args.codec, elapsed, video_s,
                video_s / elapsed, out)


if __name__ == "__main__":
    main()
