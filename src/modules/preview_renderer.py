"""Module 5 (prévisualisation) — rendu vidéo ffmpeg de la timeline.

Reproduit la composition prévue dans CapCut pour contrôler le rendu sans ouvrir
l'éditeur :

- **V1** : la case étirée pour couvrir le cadre, floutée (flou gaussien sur une
  version réduite) et assombrie de 30 % : elle comble les bandes 16:9 à gauche
  et à droite ;
- **V2** : la case au premier plan, **à sa résolution native**, centrée ; une
  case plus grande que le cadre est réduite pour y tenir, jamais agrandie
  (aucune pixellisation). Mouvements : zoom Ken Burns 100 % → 105 %, ou punch-in
  105 % en 0,2 s (cases ``action_heavy``) ; jamais au-delà de 105 % ;
- **T1** : sous-titres en blocs de 2-4 mots, police grasse, blanc, contour noir
  de 2 px, centrés en bas ;
- **A1** : voix off ; **A2** : musique de fond par ambiance (bouclée, fondus,
  −22 dB) ; **A3** : bruitages.

Les images sont produites avec OpenCV/NumPy et envoyées à l'exécutable ffmpeg
fourni par ``imageio-ffmpeg`` (H.264 + AAC). Aucune installation système requise.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path

import cv2
import numpy as np
import soundfile as sf
from PIL import Image, ImageDraw, ImageFont

from src.models.timeline import BgmClip, PanelClip, SfxClip, SubtitleCue, Timeline
from src.modules.timeline_builder import KEN_BURNS_ZOOM, MAX_ZOOM, PUNCH_IN_S, PUNCH_IN_ZOOM
from src.modules.tts_engine import read_wav, write_wav
from src.utils.config import PROJECT_ROOT
from src.utils.image_utils import to_numpy_rgb

logger = logging.getLogger(__name__)

#: Facteur d'assombrissement du fond (0,7 = −30 %).
BACKGROUND_DARKEN: float = 0.7
BACKGROUND_DOWNSCALE: int = 8
BACKGROUND_BLUR_SIGMA: float = 6.0
#: Sous-titres : police grasse, contour noir de 2 px (en 1080p), sans bandeau.
SUBTITLE_FONT_SIZE: int = 64
SUBTITLE_STROKE_PX: int = 2
SUBTITLE_BOTTOM_MARGIN: int = 84
SUBTITLE_MAX_WIDTH_RATIO: float = 0.86
#: Hauteur de référence des tailles de sous-titres (px).
REFERENCE_HEIGHT: int = 1080
#: Polices candidates pour les sous-titres (la première trouvée est utilisée).
#:
#: ``config/fonts/`` passe **avant** les polices système : Montserrat et Rubik ne sont pas
#: installées sur Windows, et sans ce dossier l'aperçu retombait silencieusement sur Arial
#: Bold — donc pas dans la police spécifiée pour la chaîne.
FONT_CANDIDATES: tuple[str, ...] = (
    str(PROJECT_ROOT / "config" / "fonts" / "Montserrat-Bold.ttf"),
    str(PROJECT_ROOT / "config" / "fonts" / "Rubik-Bold.ttf"),
    str(PROJECT_ROOT / "config" / "fonts" / "Bangers-Regular.ttf"),
    "C:/Windows/Fonts/Montserrat-Bold.ttf",
    "C:/Windows/Fonts/Montserrat-ExtraBold.ttf",
    "C:/Windows/Fonts/Rubik-Bold.ttf",
    "C:/Windows/Fonts/arialbd.ttf",
    "C:/Windows/Fonts/segoeuib.ttf",
    "C:/Windows/Fonts/bahnschrift.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
)


#: Familles de transitions reconnues, par nom CapCut. L'aperçu ne rejoue pas l'effet exact
#: du catalogue : il en donne une **approximation fidèle en durée et en intention**, ce qui
#: suffit à juger le rythme et à vérifier la synchronisation image / voix.
TRANSITION_FAMILIES: dict[str, str] = {
    "快速挥动": "whip", "甩鞭转场": "whip", "高速滑动": "whip", "Whip_Tear": "whip",
    "故障": "glitch", "色差故障": "glitch", "信号故障": "glitch", "电视故障": "glitch",
    "闪黑": "flash_black", "闪白": "flash_white",
    "叠化": "dissolve", "推近": "dissolve", "拉远": "dissolve",
}
DEFAULT_TRANSITION_FAMILY: str = "dissolve"
#: Amplitude du balayage d'un whip pan, en fraction de la largeur du cadre.
WHIP_TRAVEL_RATIO: float = 0.55
#: Décalage maximal des canaux R et B pendant un glitch (px en 1080p).
GLITCH_SHIFT_PX: int = 18
#: Nombre de bandes horizontales déplacées pendant un glitch.
GLITCH_SLICES: int = 9

#: Familles d'effets superposés, par type logique de :mod:`src.modules.timeline_builder`.
VFX_FAMILIES: tuple[str, ...] = ("speed_lines", "glow", "rain", "sparks")
#: Intensité des effets superposés dans l'aperçu (0 = invisible, 1 = saturé).
VFX_STRENGTH: float = 0.55

#: Part de la marge disponible parcourue par un balayage rapide, et douceur de son départ.
PAN_EASE_POWER: float = 1.6

#: Familles d'animations de sous-titres, par nom CapCut.
CUE_INTRO_FAMILIES: dict[str, str] = {
    "弹入": "bounce", "放大": "scale", "渐显": "fade", "故障": "glitch",
    "逐字": "reveal", "模糊": "fade", "打字机": "reveal",
}
CUE_LOOP_FAMILIES: dict[str, str] = {"心跳": "heartbeat", "颤抖": "shake", "波浪": "heartbeat"}
#: Période des animations en boucle (secondes).
CUE_LOOP_PERIOD_S: float = 0.9


class PreviewError(RuntimeError):
    """Rendu impossible (fichier manquant, ffmpeg indisponible...)."""


# --- Briques de composition (pures, testables) -------------------------------------------
def ease_in_out(p: float) -> float:
    """Interpolation douce (smoothstep) de ``p`` dans [0, 1]."""
    p = min(max(p, 0.0), 1.0)
    return p * p * (3.0 - 2.0 * p)


def ease_out(p: float) -> float:
    """Départ rapide, arrivée douce (zoom d'impact)."""
    p = min(max(p, 0.0), 1.0)
    return 1.0 - (1.0 - p) ** 3


def native_scale(width: int, height: int, frame_w: int, frame_h: int) -> float:
    """Échelle d'affichage d'une case : 1,0 (résolution native) sauf si elle dépasse le cadre.

    Une case plus large ou plus haute que le cadre est réduite pour y tenir ; une
    case plus petite n'est **jamais** agrandie (le fond flouté comble les côtés).
    """
    return min(1.0, frame_w / width, frame_h / height)


def zoom_factor(motion: str, progress: float, elapsed_s: float, zoom: float = KEN_BURNS_ZOOM) -> float:
    """Facteur de zoom du premier plan selon le mouvement, l'avancement et le temps écoulé (≤ :data:`MAX_ZOOM`)."""
    if motion == "punch_in":
        factor = 1.0 + PUNCH_IN_ZOOM * ease_out(elapsed_s / PUNCH_IN_S if PUNCH_IN_S > 0 else 1.0)
    else:
        factor = 1.0 + zoom * ease_in_out(progress)
    return min(factor, MAX_ZOOM)


def make_background(rgb: np.ndarray, width: int, height: int) -> np.ndarray:
    """Fond V1 : image étirée pour couvrir le cadre, floutée et assombrie de 30 %."""
    h, w = rgb.shape[:2]
    scale = max(width / w, height / h)
    cover = cv2.resize(rgb, (max(width, round(w * scale)), max(height, round(h * scale))), interpolation=cv2.INTER_AREA)
    y0 = (cover.shape[0] - height) // 2
    x0 = (cover.shape[1] - width) // 2
    cropped = cover[y0 : y0 + height, x0 : x0 + width]
    small = cv2.resize(
        cropped, (max(1, width // BACKGROUND_DOWNSCALE), max(1, height // BACKGROUND_DOWNSCALE)),
        interpolation=cv2.INTER_AREA,
    )
    blurred = cv2.GaussianBlur(small, (0, 0), BACKGROUND_BLUR_SIGMA)
    back = cv2.resize(blurred, (width, height), interpolation=cv2.INTER_LINEAR)
    return (back.astype(np.float32) * BACKGROUND_DARKEN).astype(np.uint8)


def paste(frame: np.ndarray, image: np.ndarray, x0: int, y0: int) -> None:
    """Colle ``image`` sur ``frame`` en (x0, y0), en rognant ce qui dépasse (en place)."""
    fh, fw = frame.shape[:2]
    ih, iw = image.shape[:2]
    sx0, sy0 = max(0, -x0), max(0, -y0)
    dx0, dy0 = max(0, x0), max(0, y0)
    w = min(iw - sx0, fw - dx0)
    h = min(ih - sy0, fh - dy0)
    if w > 0 and h > 0:
        frame[dy0 : dy0 + h, dx0 : dx0 + w] = image[sy0 : sy0 + h, sx0 : sx0 + w]


def load_font(
    size: int = SUBTITLE_FONT_SIZE, path: str | Path | Sequence[str] | None = None
) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    """Charge la police des sous-titres.

    ``path`` accepte un chemin unique ou une **liste de candidats** par ordre de
    préférence (celle du profil de format) ; sans rien, les candidats par défaut.
    """
    if path is None:
        candidates = list(FONT_CANDIDATES)
    elif isinstance(path, (str, Path)):
        candidates = [str(path)]
    else:
        candidates = [str(p) for p in path] or list(FONT_CANDIDATES)
    for candidate in candidates:
        try:
            return ImageFont.truetype(candidate, size)
        except OSError:
            continue
    logger.warning("Aucune police TrueType trouvee : police bitmap par defaut")
    return ImageFont.load_default()


def render_subtitle_image(
    text: str, max_width: int, font: ImageFont.ImageFont, *, stroke: int = SUBTITLE_STROKE_PX, boxed: bool = False
) -> np.ndarray:
    """Rend un bloc de sous-titre en image RGBA : texte blanc, contour noir net, sans bandeau par défaut."""
    words = text.split()
    lines: list[str] = []
    current: list[str] = []
    probe = ImageDraw.Draw(Image.new("RGBA", (8, 8)))
    for word in words:
        candidate = " ".join([*current, word])
        if current and probe.textlength(candidate, font=font) > max_width - 2 * stroke - 8:
            lines.append(" ".join(current))
            current = [word]
        else:
            current.append(word)
    if current:
        lines.append(" ".join(current))
    lines = lines or [""]
    line_height = int(getattr(font, "size", 20) * 1.3)
    text_w = int(max(probe.textlength(line, font=font) for line in lines)) + 2 * stroke
    pad_x, pad_y = (24, 14) if boxed else (stroke + 2, stroke + 2)
    box_w = min(max_width, text_w + 2 * pad_x)
    box_h = line_height * len(lines) + 2 * pad_y
    canvas = Image.new("RGBA", (box_w, box_h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(canvas)
    if boxed:
        draw.rounded_rectangle((0, 0, box_w - 1, box_h - 1), radius=14, fill=(0, 0, 0, 150))
    y = pad_y
    for line in lines:
        width_line = probe.textlength(line, font=font)
        x = (box_w - width_line) / 2
        draw.text((x, y), line, font=font, fill=(255, 255, 255, 255), stroke_width=stroke, stroke_fill=(0, 0, 0, 255))
        y += line_height
    return np.asarray(canvas, dtype=np.uint8)


def render_words_image(
    words: Sequence[str],
    font: ImageFont.ImageFont,
    *,
    highlight: int = -1,
    fill: tuple[int, int, int] = (255, 255, 255),
    highlight_fill: tuple[int, int, int] = (255, 255, 0),
    stroke: int = SUBTITLE_STROKE_PX,
) -> np.ndarray:
    """Rend un bloc court en RGBA, le mot d'indice ``highlight`` dans une autre couleur.

    Les mots sont dessinés un à un pour pouvoir colorer le mot courant : le surlignage
    progressif du format court n'est pas exprimable autrement, une image de texte unique
    ne portant qu'une seule couleur.
    """
    if not words:
        return np.zeros((1, 1, 4), dtype=np.uint8)
    probe = ImageDraw.Draw(Image.new("RGBA", (8, 8)))
    space = probe.textlength(" ", font=font)
    widths = [probe.textlength(word, font=font) for word in words]
    total = sum(widths) + space * (len(words) - 1)
    height = int(getattr(font, "size", 20) * 1.32)
    canvas = Image.new("RGBA", (int(total) + 2 * stroke + 4, height + 2 * stroke), (0, 0, 0, 0))
    draw = ImageDraw.Draw(canvas)
    x = float(stroke + 2)
    for index, (word, width) in enumerate(zip(words, widths)):
        colour = highlight_fill if index == highlight else fill
        draw.text((x, stroke), word, font=font, fill=(*colour, 255), stroke_width=stroke, stroke_fill=(0, 0, 0, 255))
        x += width + space
    return np.asarray(canvas, dtype=np.uint8)


def blend_rgba(frame: np.ndarray, rgba: np.ndarray, x0: int, y0: int) -> None:
    """Superpose une image RGBA sur ``frame`` (RGB) en (x0, y0), en place."""
    fh, fw = frame.shape[:2]
    ih, iw = rgba.shape[:2]
    x1, y1 = min(fw, x0 + iw), min(fh, y0 + ih)
    if x1 <= x0 or y1 <= y0:
        return
    region = frame[y0:y1, x0:x1].astype(np.float32)
    layer = rgba[: y1 - y0, : x1 - x0].astype(np.float32)
    alpha = layer[:, :, 3:4] / 255.0
    frame[y0:y1, x0:x1] = (region * (1 - alpha) + layer[:, :, :3] * alpha).astype(np.uint8)


# --- Transitions ----------------------------------------------------------------------
def back_out(p: float) -> float:
    """Sortie avec léger dépassement (rebond) : 0 → 1 en passant brièvement au-dessus de 1."""
    p = min(max(p, 0.0), 1.0)
    q = p - 1.0
    return 1.0 + 2.70158 * q**3 + 1.70158 * q**2


def transition_family(kind: str) -> str:
    """Famille de rendu local d'une transition CapCut (repli : fondu enchaîné)."""
    return TRANSITION_FAMILIES.get(kind, DEFAULT_TRANSITION_FAMILY)


def _mix(before: np.ndarray, after: np.ndarray, k: float) -> np.ndarray:
    """Fondu entre deux images. ``cv2.addWeighted`` est ~10 fois plus rapide qu'un
    mélange en ``float32`` NumPy sur une image 1080p (mesuré : 30 ms → 3 ms)."""
    return cv2.addWeighted(before, 1.0 - k, after, k, 0.0)


def _to_color(image: np.ndarray, color: float, k: float) -> np.ndarray:
    """Fondu vers le noir (``color`` = 0) ou le blanc (255), en une seule passe SIMD."""
    return cv2.convertScaleAbs(image, alpha=1.0 - k, beta=color * k)


def _slide(before: np.ndarray, after: np.ndarray, shift: int) -> np.ndarray:
    """Balayage horizontal : l'image sortante part à gauche, l'entrante arrive par la droite."""
    height, width = before.shape[:2]
    frame = np.empty_like(before)
    keep = max(0, width - shift)
    if keep:
        frame[:, :keep] = before[:, shift:]
    if shift:
        frame[:, keep:] = after[:, : width - keep]
    return frame


def _glitch(image: np.ndarray, amount: float, seed: int) -> np.ndarray:
    """Décale les canaux R et B et déplace des bandes horizontales (défaut de signal)."""
    if amount <= 0:
        return image
    height, width = image.shape[:2]
    ratio = height / REFERENCE_HEIGHT
    out = image.copy()
    shift = max(1, int(round(GLITCH_SHIFT_PX * ratio * amount)))
    out[:, shift:, 0] = image[:, : width - shift, 0]
    out[:, : width - shift, 2] = image[:, shift:, 2]
    rng = np.random.default_rng(seed)
    bounds = np.sort(rng.integers(0, height, size=GLITCH_SLICES * 2))
    for top, bottom in zip(bounds[::2], bounds[1::2]):
        if bottom <= top:
            continue
        offset = int(rng.integers(-shift * 3, shift * 3 + 1))
        out[top:bottom] = np.roll(out[top:bottom], offset, axis=1)
    return out


def blend_transition(family: str, before: np.ndarray, after: np.ndarray, progress: float) -> np.ndarray:
    """Compose l'image sortante et l'image entrante à l'avancement ``progress`` (0 → 1).

    La transition **ne consomme aucun temps** : elle se joue à l'intérieur de la fin du
    clip sortant, si bien que la timeline reste calée sur la voix off. C'est là que
    l'aperçu diffère volontairement de CapCut, où presque toutes les transitions
    empiètent sur les clips voisins.
    """
    p = min(max(progress, 0.0), 1.0)
    if family == "whip":
        eased = ease_in_out(p)
        width = before.shape[1]
        frame = _slide(before, after, int(round(width * eased)))
        # Flou de filé maximal au milieu du balayage, nul aux extrémités.
        blur = int(round(WHIP_TRAVEL_RATIO * 60 * math.sin(math.pi * p)))
        if blur >= 2:
            frame = cv2.blur(frame, (blur * 2 + 1, 1))
        return frame
    if family == "glitch":
        base = before if p < 0.5 else after
        return _glitch(base, math.sin(math.pi * p), seed=int(p * 997))
    if family in ("flash_black", "flash_white"):
        color = 255.0 if family == "flash_white" else 0.0
        if p < 0.5:
            return _to_color(before, color, p * 2.0)
        return _to_color(after, color, (1.0 - p) * 2.0)
    return _mix(before, after, ease_in_out(p))


# --- Fenêtre affichée (format court) --------------------------------------------------
def pan_offset(progress: float, travel: int, *, ratio: float = 0.85) -> int:
    """Décalage d'un balayage rapide à l'avancement ``progress``.

    Départ et arrivée adoucis (:data:`PAN_EASE_POWER`) : un balayage strictement linéaire
    sur un plan d'une seconde donne une saccade à la coupe.
    """
    if travel <= 0:
        return 0
    eased = ease_in_out(min(max(progress, 0.0), 1.0)) ** PAN_EASE_POWER
    return int(round(travel * ratio * eased))


def live_window(
    clip: PanelClip, progress: float, elapsed_s: float, *, punch_zoom: float, pan_ratio: float
) -> tuple[int, int, int, int]:
    """Fenêtre réellement affichée à cet instant : ``(x, y, largeur, hauteur)``.

    Le punch-in du format court **resserre la fenêtre** au lieu d'agrandir l'image déjà
    mise à l'échelle : le mouvement est identique à l'écran, mais il consomme des pixels
    de la source au lieu d'en inventer. Le balayage, lui, déplace la fenêtre dans la marge
    restante — horizontale sur une case plus large que le cadre, verticale sinon.
    """
    window = clip.crop
    x, y, width, height = window.x, window.y, window.width, window.height
    if clip.motion == "punch_in" and punch_zoom > 0:
        factor = 1.0 + punch_zoom * ease_out(elapsed_s / PUNCH_IN_S if PUNCH_IN_S > 0 else 1.0)
        new_w, new_h = max(8, round(width / factor)), max(8, round(height / factor))
        x += (width - new_w) // 2
        y += (height - new_h) // 2
        width, height = new_w, new_h
    elif clip.motion == "fast_pan":
        room_x, room_y = clip.width - width, clip.height - height
        if room_x >= room_y:
            x = min(clip.width - width, x + pan_offset(progress, room_x - x, ratio=pan_ratio))
        else:
            y = min(clip.height - height, y + pan_offset(progress, room_y - y, ratio=pan_ratio))
    x = max(0, min(x, clip.width - width))
    y = max(0, min(y, clip.height - height))
    return x, y, width, height


# --- Carte de titre finale ------------------------------------------------------------
#: Longueur du flou de bougé de l'outro, en fraction de la largeur du cadre.
OUTRO_BLUR_RATIO: float = 0.022


def motion_blur(frame: np.ndarray, strength: float) -> np.ndarray:
    """Flou de bougé horizontal, d'intensité ``strength`` (0 = net, 1 = maximal)."""
    if strength <= 0:
        return frame
    length = max(1, int(round(frame.shape[1] * OUTRO_BLUR_RATIO * min(strength, 1.0))))
    if length < 2:
        return frame
    return cv2.blur(frame, (length * 2 + 1, 1))


def draw_title_card(
    frame: np.ndarray, title: str, font: ImageFont.ImageFont, *, stroke: int = 8, opacity: float = 1.0
) -> None:
    """Superpose le titre en gros au centre, sur un voile sombre (en place).

    Le voile garantit la lisibilité quelle que soit la case en fond : sans lui, un titre
    blanc sur un dessin clair disparaît.
    """
    if not title or opacity <= 0:
        return
    height, width = frame.shape[:2]
    veil = int(140 * min(opacity, 1.0))
    if veil:
        frame[:] = cv2.addWeighted(frame, 1.0 - veil / 255.0, np.zeros_like(frame), 0.0, 0.0)
    image = render_subtitle_image(title, int(width * 0.86), font, stroke=stroke)
    if opacity < 0.995:
        image = image.copy()
        image[:, :, 3] = (image[:, :, 3].astype(np.float32) * opacity).astype(np.uint8)
    blend_rgba(frame, image, max(0, (width - image.shape[1]) // 2), max(0, (height - image.shape[0]) // 2))


# --- Effets superposés ----------------------------------------------------------------
_VFX_CACHE: dict[tuple, np.ndarray] = {}


def _radial_glow(width: int, height: int) -> np.ndarray:
    """Halo radial en RGB ``uint8`` (blanc au centre, noir aux bords), mis en cache par format.

    Conservé en ``uint8`` et en trois canaux pour être additionné par ``cv2.addWeighted``
    sans conversion : une passe en ``float32`` coûtait à elle seule 38 ms par image en 1080p.
    """
    key = ("glow", width, height)
    cached = _VFX_CACHE.get(key)
    if cached is None:
        ys = np.linspace(-1.0, 1.0, height, dtype=np.float32)[:, None]
        xs = np.linspace(-1.0, 1.0, width, dtype=np.float32)[None, :]
        radial = np.clip(1.0 - np.sqrt(xs**2 + ys**2), 0.0, 1.0) ** 2
        cached = cv2.cvtColor((radial * 255.0).astype(np.uint8), cv2.COLOR_GRAY2RGB)
        _VFX_CACHE[key] = cached
    return cached


def _vfx_particles(family: str, width: int, height: int, count: int) -> np.ndarray:
    """Positions normalisées et phases des particules d'un effet (déterministes)."""
    key = (family, width, height, count)
    cached = _VFX_CACHE.get(key)
    if cached is None:
        rng = np.random.default_rng(abs(hash(family)) % (2**31))
        cached = rng.random((count, 3), dtype=np.float32)
        _VFX_CACHE[key] = cached
    return cached


def apply_vfx(frame: np.ndarray, family: str, elapsed_s: float, *, strength: float = VFX_STRENGTH) -> None:
    """Superpose un effet lumineux sur l'image, **en place** (mélange additif).

    Les quatre familles sont des approximations locales des effets intégrés de CapCut :
    l'aperçu sert à juger la présence et le rythme de l'effet, pas à le reproduire au pixel.
    """
    if strength <= 0:
        return
    height, width = frame.shape[:2]
    ratio = height / REFERENCE_HEIGHT
    if family == "glow":
        pulse = 0.65 + 0.35 * math.sin(2.0 * math.pi * elapsed_s / 2.5)
        cv2.addWeighted(frame, 1.0, _radial_glow(width, height), strength * pulse * 0.35, 0.0, dst=frame)
        return

    # Les particules sont dessinees directement a l'intensite voulue : le calque n'a
    # alors plus qu'a etre additionne, sans mise a l'echelle ni passage en flottant.
    value = int(round(255 * min(1.0, strength)))
    mask = np.zeros((height, width), dtype=np.uint8)
    if family == "speed_lines":
        particles = _vfx_particles(family, width, height, 56)
        cx, cy = width // 2, height // 2
        for angle_u, radius_u, phase in particles:
            angle = angle_u * 2.0 * math.pi
            travel = (radius_u + elapsed_s * 0.8 + phase) % 1.0
            inner = 0.28 + 0.72 * travel
            outer = inner + 0.16
            dx, dy = math.cos(angle), math.sin(angle)
            reach = width * 0.75
            cv2.line(
                mask,
                (int(cx + dx * inner * reach), int(cy + dy * inner * reach)),
                (int(cx + dx * outer * reach), int(cy + dy * outer * reach)),
                value, max(1, int(round(2 * ratio))), cv2.LINE_AA,
            )
    elif family == "rain":
        particles = _vfx_particles(family, width, height, 140)
        length = int(round(38 * ratio))
        for x_u, y_u, speed_u in particles:
            speed = 0.45 + speed_u
            y = int(((y_u + elapsed_s * speed) % 1.0) * height)
            x = int(x_u * width + y * 0.18)
            cv2.line(mask, (x % width, y), ((x + length // 3) % width, y + length), value, max(1, int(round(1 * ratio))), cv2.LINE_AA)
    elif family == "sparks":
        particles = _vfx_particles(family, width, height, 70)
        for x_u, y_u, phase in particles:
            life = (elapsed_s * 0.9 + phase) % 1.0
            y = int((y_u - life * 0.25) % 1.0 * height)
            radius = max(1, int(round((1.0 - life) * 5 * ratio)))
            cv2.circle(mask, (int(x_u * width), y), radius, value, -1, cv2.LINE_AA)
    else:
        return
    # Addition saturante en uint8 : une seule passe SIMD, contre quatre passes
    # float32 (conversion, multiplication, ecretage, reconversion) auparavant.
    cv2.add(frame, cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB), dst=frame)


# --- Animation des sous-titres --------------------------------------------------------
def cue_animation_state(
    intro: str, loop: str, intro_s: float, elapsed_s: float, *, ratio: float = 1.0
) -> dict:
    """État d'un bloc de sous-titre animé à ``elapsed_s`` secondes de son apparition.

    Returns:
        ``{"scale", "alpha", "dx", "dy", "reveal", "glitch"}`` — ``reveal`` est la
        fraction de mots déjà affichés (animation « mot à mot »).
    """
    state = {"scale": 1.0, "alpha": 1.0, "dx": 0, "dy": 0, "reveal": 1.0, "glitch": 0.0}
    family = CUE_INTRO_FAMILIES.get(intro, "fade" if intro else "")
    if family and intro_s > 0 and elapsed_s < intro_s:
        p = min(max(elapsed_s / intro_s, 0.0), 1.0)
        if family == "bounce":
            state["scale"] = 0.6 + 0.4 * back_out(p)
            state["alpha"] = min(1.0, p * 2.5)
        elif family == "scale":
            state["scale"] = 0.82 + 0.18 * ease_out(p)
            state["alpha"] = min(1.0, p * 2.5)
        elif family == "reveal":
            state["reveal"] = p
        elif family == "glitch":
            state["glitch"] = 1.0 - p
        else:
            state["alpha"] = p
    loop_family = CUE_LOOP_FAMILIES.get(loop, "")
    if loop_family and elapsed_s >= intro_s:
        phase = 2.0 * math.pi * (elapsed_s - intro_s) / CUE_LOOP_PERIOD_S
        if loop_family == "heartbeat":
            state["scale"] *= 1.0 + 0.035 * math.sin(phase)
        elif loop_family == "shake":
            state["dx"] = int(round(3.0 * ratio * math.sin(phase * 3.0)))
            state["dy"] = int(round(1.5 * ratio * math.cos(phase * 5.0)))
    return state


def apply_cue_state(rgba: np.ndarray, state: Mapping) -> np.ndarray:
    """Applique échelle, opacité et glitch à l'image RGBA d'un bloc de sous-titre."""
    out = rgba
    scale = float(state.get("scale", 1.0))
    if abs(scale - 1.0) > 0.005:
        height, width = out.shape[:2]
        out = cv2.resize(
            out, (max(1, round(width * scale)), max(1, round(height * scale))),
            interpolation=cv2.INTER_LINEAR if scale < 1.0 else cv2.INTER_CUBIC,
        )
    glitch = float(state.get("glitch", 0.0))
    if glitch > 0.01:
        rgb = _glitch(np.ascontiguousarray(out[:, :, :3]), glitch, seed=int(glitch * 613))
        out = np.dstack([rgb, out[:, :, 3]])
    alpha = float(state.get("alpha", 1.0))
    reveal = float(state.get("reveal", 1.0))
    if alpha < 0.995 or reveal < 0.999:
        out = out.copy() if out is rgba else out
        if alpha < 0.995:
            out[:, :, 3] = (out[:, :, 3].astype(np.float32) * max(0.0, alpha)).astype(np.uint8)
        if reveal < 0.999:
            # Les blocs faisant 2 a 4 mots tiennent sur une ligne : masquer la partie
            # droite revele les mots l'un apres l'autre sans deplacer le bloc.
            out[:, max(0, int(round(out.shape[1] * reveal))) :, 3] = 0
    return out


# --- Audio ---------------------------------------------------------------------------
def _read_mono(path: str | Path, sample_rate: int) -> np.ndarray:
    """Lit un fichier audio en float32 mono rééchantillonné (interpolation linéaire) si besoin."""
    samples, rate = sf.read(str(path), dtype="float32", always_2d=True)
    mono = samples.mean(axis=1).astype(np.float32)
    if rate != sample_rate and len(mono):
        positions = np.arange(0, len(mono), rate / sample_rate)
        mono = np.interp(positions, np.arange(len(mono)), mono).astype(np.float32)
    return mono


def _apply_fades(samples: np.ndarray, sample_rate: int, fade_in_s: float, fade_out_s: float) -> np.ndarray:
    out = samples.copy()
    n = len(out)
    a = min(n, int(round(fade_in_s * sample_rate)))
    r = min(n, int(round(fade_out_s * sample_rate)))
    if a > 0:
        out[:a] *= np.linspace(0.0, 1.0, a, dtype=np.float32)
    if r > 0:
        out[n - r :] *= np.linspace(1.0, 0.0, r, dtype=np.float32)
    return out


def mix_bgm_clips(voice: np.ndarray, sample_rate: int, clips: list[BgmClip]) -> np.ndarray:
    """Mixe les segments de musique (bouclés si trop courts, fondus, gain) sous la voix."""
    mixed = voice.astype(np.float32).copy()
    for clip in clips:
        music = _read_mono(clip.file, sample_rate)
        if not len(music):
            continue
        n = int(round(clip.duration_s * sample_rate))
        start = int(round(clip.start_s * sample_rate))
        if start >= len(mixed) or n <= 0:
            continue
        n = min(n, len(mixed) - start)
        tiled = np.tile(music, math.ceil(n / len(music)))[:n]
        tiled = _apply_fades(tiled, sample_rate, clip.fade_in_s, clip.fade_out_s) * (10 ** (clip.gain_db / 20.0))
        mixed[start : start + n] += tiled
    return np.clip(mixed, -1.0, 1.0).astype(np.float32)


def mix_sfx_clips(voice: np.ndarray, sample_rate: int, clips: list[SfxClip]) -> np.ndarray:
    """Ajoute les bruitages (gain) aux instants voulus."""
    mixed = voice.astype(np.float32).copy()
    for clip in clips:
        sound = _read_mono(clip.file, sample_rate) * (10 ** (clip.gain_db / 20.0))
        start = int(round(clip.start_s * sample_rate))
        if start >= len(mixed):
            continue
        n = min(len(sound), len(mixed) - start)
        mixed[start : start + n] += sound[:n]
    return np.clip(mixed, -1.0, 1.0).astype(np.float32)


def mix_background_music(voice: np.ndarray, sample_rate: int, bgm_file: str | Path, gain_db: float) -> np.ndarray:
    """Mixe une musique unique (bouclée) sous la voix, sans fondu (compatibilité)."""
    clip = BgmClip(mood="all", file=str(bgm_file), start_s=0.0, duration_s=len(voice) / sample_rate, gain_db=gain_db)
    return mix_bgm_clips(voice, sample_rate, [clip]) if len(voice) else voice


# --- Rendu ---------------------------------------------------------------------------
class PreviewRenderer:
    """Compose les images de la timeline et les encode en MP4 (H.264 + AAC).

    Args:
        timeline: plan de montage.
        fps: images par seconde (défaut : celui de la timeline).
        zoom: amplitude du zoom Ken Burns (bornée par :data:`MAX_ZOOM`).
        font_path: police des sous-titres (défaut : police système).
        subtitle_size: taille de la police des sous-titres (px en 1080p).
        show_subtitles: dessiner la piste T1.
        dynamics: rejouer le dynamisme inscrit dans la timeline (transitions,
            animations de sous-titres, effets superposés). Les transitions sont rendues
            **sans consommer de temps**, à l'intérieur de la fin du clip sortant : la
            synchronisation image / voix de l'aperçu reste donc exacte.
        vfx_strength: intensité des effets superposés (0 = aucun).
    """

    def __init__(
        self,
        timeline: Timeline,
        *,
        fps: int | None = None,
        zoom: float = KEN_BURNS_ZOOM,
        font_path: str | Path | None = None,
        subtitle_size: int = SUBTITLE_FONT_SIZE,
        show_subtitles: bool = True,
        dynamics: bool = True,
        vfx_strength: float = VFX_STRENGTH,
        profile: "FormatProfile | None" = None,
    ) -> None:
        from src.modules.format_factory import VideoConfigFactory

        self.timeline = timeline
        self.fps = fps or timeline.fps
        self.width, self.height = timeline.width, timeline.height
        self.zoom = min(zoom, MAX_ZOOM - 1.0)
        self.show_subtitles = show_subtitles
        self.dynamics = dynamics
        self.vfx_strength = max(0.0, vfx_strength)
        self.profile = profile or VideoConfigFactory.create(timeline.format)
        # Le recadrage par fenetre n'est actif qu'en format court : le mode long garde
        # strictement son chemin de composition d'origine.
        self.cropped_mode = self.profile.framing.fit == "cover_crop"
        self.punch_zoom = self.profile.motion.punch_in_zoom
        self.pan_ratio = self.profile.motion.pan_ratio
        rules = self.profile.subtitles
        if self.cropped_mode:
            # Le profil donne des pixels pour SA resolution : pas de mise a l'echelle.
            self.stroke = max(1, rules.stroke_px)
            size = rules.font_size_px
            candidates = rules.font_candidates
        else:
            ratio = self.height / REFERENCE_HEIGHT
            self.stroke = max(1, round(SUBTITLE_STROKE_PX * ratio))
            size = max(8, round(subtitle_size * ratio))
            candidates = ()
        self.font = load_font(size, font_path or (candidates or None)) if show_subtitles else None
        self.title_font = (
            load_font(self.profile.outro.font_size_px, font_path or (candidates or None))
            if show_subtitles and self.profile.outro.enabled
            else self.font
        )
        self._panel_cache: dict[int, np.ndarray] = {}
        self._subtitle_cache: dict[str, np.ndarray] = {}

    # -- ressources ------------------------------------------------------------------
    def panel_image(self, clip: PanelClip) -> np.ndarray:
        """Pixels RGB d'une case (mis en cache)."""
        cached = self._panel_cache.get(clip.panel_index)
        if cached is not None:
            return cached
        path = Path(self.timeline.panels_dir) / clip.file
        if not path.is_file():
            raise PreviewError(f"Case introuvable : {path}")
        with Image.open(path) as img:
            rgb = to_numpy_rgb(img).copy()
        self._panel_cache[clip.panel_index] = rgb
        return rgb

    def prepare_clip(self, clip: PanelClip) -> dict:
        """Précalcule le fond et le premier plan d'une case (résolution native, ou réduite si trop grande)."""
        rgb = self.panel_image(clip)
        h, w = rgb.shape[:2]
        background = make_background(rgb, self.width, self.height)
        if self.cropped_mode and clip.crop is not None:
            # Format court : la fenetre bouge a chaque image (resserrement, balayage),
            # le recadrage se fait donc au moment de composer, pas ici.
            return {"clip": clip, "bg": background, "source": rgb, "mode": "window", "scale": 1.0}
        scale = native_scale(w, h, self.width, self.height)
        if scale < 1.0:
            foreground = cv2.resize(rgb, (max(1, round(w * scale)), max(1, round(h * scale))), interpolation=cv2.INTER_AREA)
            mode = "contain"
        else:
            foreground, mode = rgb, "native"
        return {"clip": clip, "bg": background, "fg": foreground, "mode": mode, "scale": scale}

    def compose_window(self, prepared: dict, progress: float, elapsed_s: float) -> np.ndarray:
        """Compose une case du format court : fenêtre vivante, puis mise au cadre."""
        clip: PanelClip = prepared["clip"]
        source: np.ndarray = prepared["source"]
        x, y, width, height = live_window(
            clip, progress, elapsed_s, punch_zoom=self.punch_zoom, pan_ratio=self.pan_ratio
        )
        view = source[y : y + height, x : x + width]
        if view.size == 0:
            return prepared["bg"].copy()
        if clip.crop.fit == "cover":
            scale = max(self.width / view.shape[1], self.height / view.shape[0])
            interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_CUBIC
            resized = cv2.resize(
                view, (max(self.width, round(view.shape[1] * scale)), max(self.height, round(view.shape[0] * scale))),
                interpolation=interpolation,
            )
            ox = (resized.shape[1] - self.width) // 2
            oy = (resized.shape[0] - self.height) // 2
            return np.ascontiguousarray(resized[oy : oy + self.height, ox : ox + self.width])
        # Soupape letterbox : la case n'a pas de quoi remplir le cadre, le fond floute comble.
        frame = prepared["bg"].copy()
        scale = min(self.width / view.shape[1], self.height / view.shape[0])
        resized = cv2.resize(
            view, (max(1, round(view.shape[1] * scale)), max(1, round(view.shape[0] * scale))),
            interpolation=cv2.INTER_AREA if scale < 1.0 else cv2.INTER_CUBIC,
        )
        paste(frame, resized, (self.width - resized.shape[1]) // 2, (self.height - resized.shape[0]) // 2)
        return frame

    def subtitle_image(self, cue: SubtitleCue) -> np.ndarray:
        cached = self._subtitle_cache.get(cue.text)
        if cached is None:
            cached = render_subtitle_image(cue.text, int(self.width * SUBTITLE_MAX_WIDTH_RATIO), self.font, stroke=self.stroke)
            self._subtitle_cache[cue.text] = cached
        return cached

    # -- composition -----------------------------------------------------------------
    def compose_panel(self, prepared: dict, progress: float, elapsed_s: float | None = None) -> np.ndarray:
        """Fond flouté + case au premier plan, sans sous-titre ni effet superposé."""
        clip: PanelClip = prepared["clip"]
        if elapsed_s is None:
            elapsed_s = progress * clip.duration_s
        if prepared.get("mode") == "window":
            return self.compose_window(prepared, progress, elapsed_s)
        frame = prepared["bg"].copy()
        fg: np.ndarray = prepared["fg"]
        fh, fw = fg.shape[:2]
        z = zoom_factor(clip.motion, progress, elapsed_s, self.zoom)
        zw, zh = max(1, round(fw * z)), max(1, round(fh * z))
        zoomed = cv2.resize(fg, (zw, zh), interpolation=cv2.INTER_CUBIC) if z != 1.0 else fg
        paste(frame, zoomed, (self.width - zw) // 2, (self.height - zh) // 2)
        return frame

    def draw_subtitle(self, frame: np.ndarray, cue: SubtitleCue, elapsed_s: float | None = None) -> None:
        """Dessine un bloc de sous-titre, animé si ``elapsed_s`` est fourni (en place)."""
        rules = self.profile.subtitles
        if rules.highlight_current_word and cue.words and rules.highlight_color and elapsed_s is not None:
            # Surlignage progressif : le bloc est redessine a chaque changement de mot,
            # et mis en cache sur (texte, mot courant) pour ne pas le refaire par image.
            index = cue.word_at(cue.start_s + elapsed_s)
            key = f"{cue.text}\x00{index}"
            image = self._subtitle_cache.get(key)
            if image is None:
                image = render_words_image(
                    cue.text.split(), self.font, highlight=index, fill=rules.fill,
                    highlight_fill=rules.highlight_color, stroke=self.stroke,
                )
                self._subtitle_cache[key] = image
        else:
            image = self.subtitle_image(cue)
        dx = dy = 0
        if elapsed_s is not None and self.dynamics and cue.animation is not None:
            ratio = self.height / REFERENCE_HEIGHT
            state = cue_animation_state(
                cue.animation.intro, cue.animation.loop, cue.animation.duration_s, elapsed_s, ratio=ratio,
            )
            image = apply_cue_state(image, state)
            dx, dy = int(state["dx"]), int(state["dy"])
        # ``vertical_anchor`` place le BORD BAS du bloc. La valeur du mode long reproduit
        # a l'identique la marge historique de 84 px en 1080p ; celle du format court
        # remonte le texte dans le tiers inferieur, au-dessus des boutons des plateformes.
        bottom = self.height * self.profile.subtitles.vertical_anchor
        x = max(0, (self.width - image.shape[1]) // 2 + dx)
        y = int(min(max(0, bottom - image.shape[0] + dy), max(0, self.height - image.shape[0])))
        blend_rgba(frame, image, x, y)

    def compose(
        self,
        prepared: dict,
        progress: float,
        cue: SubtitleCue | None = None,
        elapsed_s: float | None = None,
        *,
        cue_elapsed_s: float | None = None,
    ) -> np.ndarray:
        """Compose l'image d'une case à l'avancement ``progress`` (0 → 1), ``elapsed_s`` secondes après son début.

        ``cue_elapsed_s`` déclenche l'animation du sous-titre ; sans lui le bloc est
        dessiné dans son état stabilisé (comportement d'origine).
        """
        frame = self.compose_panel(prepared, progress, elapsed_s)
        if cue is not None and self.show_subtitles:
            self.draw_subtitle(frame, cue, cue_elapsed_s)
        return frame

    def frames(self, max_duration_s: float | None = None):
        """Génère les images successives de la timeline (uint8 RGB HxWx3)."""
        total = self.timeline.total_duration_s
        if max_duration_s is not None:
            total = min(total, max_duration_s)
        n_frames = int(round(total * self.fps))
        clips = self.timeline.clips
        cues = self.timeline.subtitles
        vfx = self.timeline.vfx if (self.dynamics and self.vfx_strength > 0) else []
        # L'outro se calcule sur la duree REELLE de la timeline, pas sur l'apercu tronque :
        # un extrait de 30 s ne doit pas afficher la carte de titre du chapitre entier.
        outro_rules = self.profile.outro
        outro_from = (
            self.timeline.total_duration_s - outro_rules.duration_s
            if outro_rules.enabled and self.show_subtitles and self.timeline.series_title
            else None
        )
        clip_pos = 0
        cue_pos = 0
        vfx_pos = 0
        prepared: dict | None = None
        upcoming: dict | None = None
        for i in range(n_frames):
            t = i / self.fps
            # On ne passe a la case suivante qu'a son debut : pendant un intervalle
            # (silence entre deux chapitres d'une compilation) la case precedente reste
            # a l'ecran, exactement comme dans le brouillon CapCut.
            while clip_pos < len(clips) - 1 and t >= clips[clip_pos + 1].start_s:
                clip_pos += 1
                prepared = None
                upcoming = None
            clip = clips[clip_pos]
            if prepared is None:
                prepared = self.prepare_clip(clip)
            elapsed = max(0.0, t - clip.start_s)
            progress = elapsed / clip.duration_s if clip.duration_s > 0 else 0.0
            frame = self.compose_panel(prepared, min(max(progress, 0.0), 1.0), elapsed)

            # Transition vers la case suivante, jouee dans la fin du clip sortant : elle
            # ne decale donc rien, contrairement au recouvrement natif de CapCut.
            transition = clip.transition if self.dynamics else None
            if transition is not None and clip_pos + 1 < len(clips):
                begin = clip.end_s - transition.duration_s
                if t >= begin and transition.duration_s > 0:
                    if upcoming is None:
                        upcoming = self.prepare_clip(clips[clip_pos + 1])
                    incoming = self.compose_panel(upcoming, 0.0, 0.0)
                    frame = blend_transition(
                        transition_family(transition.kind), frame, incoming,
                        (t - begin) / transition.duration_s,
                    )

            while vfx_pos < len(vfx) and t >= vfx[vfx_pos].end_s:
                vfx_pos += 1
            if vfx_pos < len(vfx) and vfx[vfx_pos].start_s <= t:
                active = vfx[vfx_pos]
                apply_vfx(frame, active.kind, t - active.start_s, strength=self.vfx_strength * active.opacity)

            while cue_pos < len(cues) and t >= cues[cue_pos].end_s:
                cue_pos += 1
            if cue_pos < len(cues) and cues[cue_pos].start_s <= t < cues[cue_pos].end_s and self.show_subtitles:
                cue = cues[cue_pos]
                self.draw_subtitle(frame, cue, t - cue.start_s)

            # Carte de titre : les dernieres secondes, sur une case floutee par le bouge.
            if outro_from is not None and t >= outro_from:
                progress = min(1.0, (t - outro_from) / max(self.timeline.total_duration_s - outro_from, 1e-6))
                frame = motion_blur(frame, ease_in_out(min(1.0, progress * 2.5)))
                draw_title_card(
                    frame, self.timeline.series_title, self.title_font,
                    stroke=max(2, self.stroke), opacity=ease_in_out(min(1.0, progress * 3.0)),
                )
            yield frame

    # -- audio -----------------------------------------------------------------------
    def build_audio(self, out_path: str | Path, max_duration_s: float | None = None) -> Path:
        """Assemble voix off (A1), musique (A2) et bruitages (A3) de la timeline en un WAV."""
        total = self.timeline.total_duration_s
        if max_duration_s is not None:
            total = min(total, max_duration_s)
        sample_rate = 24_000
        pieces: list[np.ndarray] = []
        for clip in self.timeline.audio:
            if clip.start_s >= total:
                break
            path = Path(self.timeline.audio_dir) / clip.file
            if not path.is_file():
                raise PreviewError(f"Segment audio introuvable : {path}")
            samples, rate = read_wav(path)
            sample_rate = rate
            keep = int(round(min(clip.duration_s, total - clip.start_s) * rate))
            pieces.append(samples[:keep])
        n_total = int(round(total * sample_rate))
        voice = np.concatenate(pieces) if pieces else np.zeros(0, dtype=np.float32)
        if len(voice) < n_total:
            voice = np.concatenate([voice, np.zeros(n_total - len(voice), dtype=np.float32)])
        voice = voice[:n_total]
        if self.timeline.bgm:
            voice = mix_bgm_clips(voice, sample_rate, [c for c in self.timeline.bgm if c.start_s < total])
        elif self.timeline.bgm_file:
            voice = mix_background_music(voice, sample_rate, self.timeline.bgm_file, self.timeline.bgm_gain_db)
        if self.timeline.sfx:
            voice = mix_sfx_clips(voice, sample_rate, [c for c in self.timeline.sfx if c.start_s < total])
        return write_wav(out_path, voice, sample_rate)

    # -- encodage --------------------------------------------------------------------
    def render(
        self,
        out_path: str | Path,
        *,
        max_duration_s: float | None = None,
        crf: int = 20,
        preset: str = "veryfast",
        progress: Callable[[int, int], None] | None = None,
    ) -> Path:
        """Encode la vidéo (H.264 + AAC) et renvoie son chemin."""
        try:
            import imageio_ffmpeg
        except ImportError as exc:
            raise PreviewError("imageio-ffmpeg n'est pas installe : pip install imageio-ffmpeg") from exc

        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        total = self.timeline.total_duration_s
        if max_duration_s is not None:
            total = min(total, max_duration_s)
        n_frames = int(round(total * self.fps))
        if n_frames <= 0:
            raise PreviewError("Timeline vide : rien a rendre")
        audio_path = out_path.with_name(out_path.stem + "_audio.wav")
        self.build_audio(audio_path, max_duration_s)
        logger.info("Rendu %dx%d @ %d fps, %d images (%.1fs) -> %s", self.width, self.height, self.fps, n_frames, total, out_path)
        writer = imageio_ffmpeg.write_frames(
            str(out_path), (self.width, self.height), fps=self.fps, codec="libx264",
            pix_fmt_in="rgb24", pix_fmt_out="yuv420p", macro_block_size=1,
            audio_path=str(audio_path), audio_codec="aac",
            output_params=["-crf", str(crf), "-preset", preset, "-movflags", "+faststart", "-shortest"],
            ffmpeg_log_level="error",
        )
        writer.send(None)
        try:
            for i, frame in enumerate(self.frames(max_duration_s)):
                writer.send(np.ascontiguousarray(frame))
                if progress is not None and (i % self.fps == 0 or i == n_frames - 1):
                    progress(i + 1, n_frames)
        finally:
            writer.close()
            try:
                audio_path.unlink()
            except OSError:
                pass
        logger.info("Video ecrite : %s (%d Ko)", out_path, out_path.stat().st_size // 1024)
        return out_path


__all__ = [
    "PreviewError",
    "PreviewRenderer",
    "ease_in_out",
    "ease_out",
    "native_scale",
    "zoom_factor",
    "make_background",
    "paste",
    "load_font",
    "render_subtitle_image",
    "blend_rgba",
    "back_out",
    "transition_family",
    "blend_transition",
    "apply_vfx",
    "cue_animation_state",
    "apply_cue_state",
    "TRANSITION_FAMILIES",
    "VFX_FAMILIES",
    "VFX_STRENGTH",
    "CUE_INTRO_FAMILIES",
    "CUE_LOOP_FAMILIES",
    "mix_bgm_clips",
    "mix_sfx_clips",
    "mix_background_music",
]
