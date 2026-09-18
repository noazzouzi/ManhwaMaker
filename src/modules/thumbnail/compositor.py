"""Étape 3 de la miniature : compositing Pillow (saturation, flèche, accroche).

Module **purement local** : aucune requête réseau, aucune dépendance au montage vidéo.
Il prend une image de base, l'amène au format YouTube et y superpose :

- une saturation globale relevée de 15 % (:data:`SATURATION_BOOST`) ;
- une flèche jaune, redimensionnée et posée du côté indiqué par l'analyse, orientée vers
  le sujet ; si l'asset n'existe pas, une flèche est **dessinée** (le pipeline ne doit
  jamais échouer pour un fichier manquant) ;
- l'accroche en police d'affichage, remplissage jaune ou blanc, contour noir de 8 px,
  dans une boîte inclinée de −5 à −10 degrés.
"""

from __future__ import annotations

import logging
from pathlib import Path

from PIL import Image, ImageDraw, ImageEnhance, ImageFont

from src.utils.config import PROJECT_ROOT

logger = logging.getLogger(__name__)

#: Format de sortie YouTube (minimum recommandé par la plateforme).
THUMBNAIL_WIDTH: int = 1280
THUMBNAIL_HEIGHT: int = 720
#: Saturation : +15 % comme demandé (1,0 = image inchangée).
SATURATION_BOOST: float = 1.15
#: Contour noir de l'accroche, en pixels à la hauteur de référence.
TEXT_STROKE_PX: int = 8
#: Inclinaison de la boîte de texte : entre ces deux bornes (degrés, sens antihoraire négatif).
TEXT_ANGLE_RANGE: tuple[float, float] = (-10.0, -5.0)
#: Couleurs de remplissage acceptées pour l'accroche.
TEXT_FILL_YELLOW: tuple[int, int, int] = (255, 255, 0)
TEXT_FILL_WHITE: tuple[int, int, int] = (255, 255, 255)
#: Part de la largeur que l'accroche doit occuper.
TEXT_WIDTH_RATIO: float = 0.72
#: Hauteur maximale de l'accroche, en part de l'image.
TEXT_MAX_HEIGHT_RATIO: float = 0.30
#: Marge des bords.
MARGIN_RATIO: float = 0.05
#: Largeur de la flèche, en part de la largeur de l'image.
ARROW_WIDTH_RATIO: float = 0.20

#: Polices d'affichage, par ordre de préférence. ``Bangers`` n'est pas installée par défaut
#: sur Windows : la déposer dans ``config/fonts/`` pour qu'elle prenne le pas sur Impact.
FONT_CANDIDATES: tuple[str, ...] = (
    str(PROJECT_ROOT / "config" / "fonts" / "Bangers-Regular.ttf"),
    str(PROJECT_ROOT / "config" / "fonts" / "Impact.ttf"),
    "C:/Windows/Fonts/impact.ttf",
    "C:/Windows/Fonts/arialbd.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
)
#: Flèche jaune fournie par l'utilisateur ; à défaut, elle est dessinée.
DEFAULT_ARROW_PATH: Path = PROJECT_ROOT / "config" / "thumbnail" / "arrow_yellow.png"


class ThumbnailCompositorError(RuntimeError):
    """Compositing impossible (image de base illisible, police introuvable...)."""


def text_angle(text: str, *, bounds: tuple[float, float] = TEXT_ANGLE_RANGE) -> float:
    """Inclinaison de la boîte de texte, dans ``bounds``, **déterministe** pour un texte donné.

    Un angle tiré du texte plutôt qu'au hasard : deux rendus du même chapitre sont
    identiques, et deux chapitres différents n'ont pas la même inclinaison.
    """
    low, high = sorted(bounds)
    span = high - low
    if span <= 0:
        return low
    seed = sum(ord(c) for c in text) if text else 0
    steps = max(1, int(span) + 1)
    return low + (seed % steps) * span / steps


def load_display_font(size: int, path: str | Path | None = None) -> ImageFont.FreeTypeFont:
    """Charge la police d'affichage (chemin explicite, sinon première candidate trouvée).

    Raises:
        ThumbnailCompositorError: aucune police TrueType disponible (la police bitmap de
            Pillow ne sait pas dessiner un contour, donc elle ne convient pas ici).
    """
    for candidate in ([str(path)] if path else list(FONT_CANDIDATES)):
        try:
            return ImageFont.truetype(candidate, size)
        except OSError:
            continue
    raise ThumbnailCompositorError(
        "Aucune police d'affichage trouvee : deposer Bangers-Regular.ttf ou Impact.ttf "
        "dans config/fonts/"
    )


def draw_arrow(width: int, height: int, *, color: tuple[int, int, int] = TEXT_FILL_YELLOW) -> Image.Image:
    """Flèche jaune contournée de noir, pointant **vers la droite**, en RGBA.

    Sert de repli quand aucun asset n'est fourni : le pipeline ne doit jamais s'arrêter
    pour une image manquante.
    """
    image = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)
    w, h = width - 1, height - 1
    points = [
        (0, 0.33 * h), (0.55 * w, 0.33 * h), (0.55 * w, 0.05 * h), (w, 0.5 * h),
        (0.55 * w, 0.95 * h), (0.55 * w, 0.67 * h), (0, 0.67 * h),
    ]
    outline = max(2, round(min(width, height) * 0.06))
    draw.polygon(points, fill=(*color, 255), outline=(0, 0, 0, 255), width=outline)
    return image


def load_arrow(path: str | Path | None = None, *, width: int = 240, height: int = 160) -> Image.Image:
    """Charge la flèche (RGBA) ou la dessine si l'asset est absent ou illisible."""
    candidate = Path(path) if path else DEFAULT_ARROW_PATH
    if candidate.is_file():
        try:
            with Image.open(candidate) as img:
                return img.convert("RGBA")
        except OSError as exc:
            logger.warning("Fleche %s illisible (%s) : fleche dessinee", candidate, exc)
    else:
        logger.info("Aucune fleche dans %s : fleche dessinee", candidate)
    return draw_arrow(width, height)


def fit_to_frame(image: Image.Image, width: int, height: int) -> Image.Image:
    """Recadre l'image pour remplir exactement ``width`` x ``height`` (recadrage centré).

    L'image de base est demandée en 16:9, mais un générateur peut livrer un ratio
    légèrement différent : on remplit le cadre plutôt que de laisser des bandes.
    """
    source = image.convert("RGB")
    scale = max(width / source.width, height / source.height)
    resized = source.resize(
        (max(width, round(source.width * scale)), max(height, round(source.height * scale))), Image.LANCZOS
    )
    left = (resized.width - width) // 2
    top = (resized.height - height) // 2
    return resized.crop((left, top, left + width, top + height))


def render_hook(
    text: str,
    max_width: int,
    max_height: int,
    *,
    fill: tuple[int, int, int] = TEXT_FILL_YELLOW,
    stroke: int = TEXT_STROKE_PX,
    font_path: str | Path | None = None,
) -> Image.Image:
    """Rend l'accroche en RGBA, dimensionnée pour tenir dans la boîte donnée.

    La taille de police est trouvée par recherche dichotomique : une accroche de 1 à 3
    mots doit remplir la largeur disponible, ce qu'une taille fixe ne garantit pas.
    """
    words = text.split()
    if not words:
        raise ThumbnailCompositorError("Accroche vide")
    probe = ImageDraw.Draw(Image.new("RGBA", (8, 8)))
    # Deux lignes au plus : une accroche de 3 mots se lit mieux sur deux lignes qu'etiree.
    lines = [text] if len(words) == 1 else [" ".join(words[:-1]), words[-1]] if len(words) == 3 else words
    low, high, best = 8, 400, None
    while low <= high:
        size = (low + high) // 2
        font = load_display_font(size, font_path)
        widths = [probe.textlength(line, font=font) + 2 * stroke for line in lines]
        line_height = round(size * 1.08)
        if max(widths) <= max_width and line_height * len(lines) + 2 * stroke <= max_height:
            best, low = (size, font, widths, line_height), size + 1
        else:
            high = size - 1
    if best is None:
        raise ThumbnailCompositorError(f"Accroche '{text}' impossible a loger dans {max_width}x{max_height}")
    size, font, widths, line_height = best
    canvas_w = round(max(widths))
    canvas_h = line_height * len(lines) + 2 * stroke
    canvas = Image.new("RGBA", (canvas_w, canvas_h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(canvas)
    y = stroke
    for line, line_width in zip(lines, widths):
        draw.text(
            ((canvas_w - line_width) / 2 + stroke, y), line, font=font, fill=(*fill, 255),
            stroke_width=stroke, stroke_fill=(0, 0, 0, 255),
        )
        y += line_height
    return canvas


class ThumbnailCompositor:
    """Assemble l'image de base, la flèche et l'accroche en une miniature YouTube.

    Args:
        width, height: format de sortie.
        saturation: facteur de saturation globale (1,15 = +15 %).
        stroke: épaisseur du contour noir de l'accroche, à la hauteur de référence.
        fill: couleur de remplissage de l'accroche (jaune par défaut, blanc accepté).
        arrow_path: flèche PNG ; à défaut, une flèche est dessinée.
        font_path: police d'affichage ; à défaut, la première candidate trouvée.
    """

    def __init__(
        self,
        *,
        width: int = THUMBNAIL_WIDTH,
        height: int = THUMBNAIL_HEIGHT,
        saturation: float = SATURATION_BOOST,
        stroke: int = TEXT_STROKE_PX,
        fill: tuple[int, int, int] = TEXT_FILL_YELLOW,
        arrow_path: str | Path | None = None,
        font_path: str | Path | None = None,
    ) -> None:
        self.width, self.height = width, height
        self.saturation = saturation
        self.fill = fill
        self.arrow_path = arrow_path
        self.font_path = font_path
        ratio = height / THUMBNAIL_HEIGHT
        self.stroke = max(2, round(stroke * ratio))

    def arrow_layer(self, arrow_pos: str) -> tuple[Image.Image, tuple[int, int]]:
        """Flèche orientée et sa position de collage, pour le côté demandé."""
        target_w = max(32, round(self.width * ARROW_WIDTH_RATIO))
        arrow = load_arrow(self.arrow_path, width=target_w, height=round(target_w * 0.66))
        scale = target_w / arrow.width
        arrow = arrow.resize((target_w, max(8, round(arrow.height * scale))), Image.LANCZOS)
        margin = round(self.width * MARGIN_RATIO)
        if arrow_pos == "right":
            # Posee a droite, elle doit pointer vers la gauche, donc vers le sujet.
            arrow = arrow.transpose(Image.FLIP_LEFT_RIGHT)
            x = self.width - margin - arrow.width
        else:
            x = margin
        y = round((self.height - arrow.height) * 0.58)
        return arrow, (x, y)

    def build(
        self, image_path: str | Path, text: str, arrow_pos: str = "right", out_path: str | Path | None = None
    ) -> Path:
        """Écrit la miniature et renvoie son chemin.

        Raises:
            ThumbnailCompositorError: image de base illisible, police absente ou accroche vide.
        """
        image_path = Path(image_path)
        try:
            with Image.open(image_path) as raw:
                base = fit_to_frame(raw, self.width, self.height)
        except (OSError, ValueError) as exc:
            raise ThumbnailCompositorError(f"Image de base illisible ({image_path}) : {exc}") from exc
        base = ImageEnhance.Color(base).enhance(self.saturation)
        canvas = base.convert("RGBA")

        arrow, position = self.arrow_layer(arrow_pos)
        canvas.alpha_composite(arrow, position)

        margin = round(self.width * MARGIN_RATIO)
        hook = render_hook(
            text,
            max_width=round(self.width * TEXT_WIDTH_RATIO),
            max_height=round(self.height * TEXT_MAX_HEIGHT_RATIO),
            fill=self.fill, stroke=self.stroke, font_path=self.font_path,
        )
        angle = text_angle(text)
        rotated = hook.rotate(angle, expand=True, resample=Image.BICUBIC)
        # L'accroche occupe le haut, du cote oppose au sujet (donc du cote de la fleche).
        x = margin if arrow_pos == "left" else max(margin, self.width - margin - rotated.width)
        x = min(max(0, x), max(0, self.width - rotated.width))
        y = margin
        canvas.alpha_composite(rotated, (x, y))

        out_path = Path(out_path) if out_path else image_path.with_name("thumbnail.jpg")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        final = canvas.convert("RGB")
        if out_path.suffix.lower() in (".jpg", ".jpeg"):
            final.save(out_path, quality=92, optimize=True)
        else:
            final.save(out_path)
        logger.info(
            "Miniature ecrite : %s (%dx%d, accroche '%s' inclinee de %.0f deg, fleche a %s, %d Ko)",
            out_path, self.width, self.height, text, angle, arrow_pos, out_path.stat().st_size // 1024,
        )
        return out_path


def build_thumbnail(
    image_path: str | Path,
    text: str,
    arrow_pos: str = "right",
    *,
    out_path: str | Path | None = None,
    fill: tuple[int, int, int] = TEXT_FILL_YELLOW,
    arrow_path: str | Path | None = None,
    font_path: str | Path | None = None,
) -> Path:
    """Compose une miniature : saturation +15 %, flèche jaune, accroche contournée et inclinée.

    Args:
        image_path: illustration de base (16:9 de préférence).
        text: accroche de 1 à 3 mots.
        arrow_pos: ``"left"`` ou ``"right"``.
        out_path: fichier de sortie (défaut : ``thumbnail.jpg`` à côté de l'image de base).
        fill: :data:`TEXT_FILL_YELLOW` ou :data:`TEXT_FILL_WHITE`.
        arrow_path: flèche PNG ; à défaut, une flèche est dessinée.
        font_path: police d'affichage ; à défaut, la première candidate trouvée.

    Raises:
        ThumbnailCompositorError: image illisible, police absente ou accroche vide.
    """
    compositor = ThumbnailCompositor(fill=fill, arrow_path=arrow_path, font_path=font_path)
    return compositor.build(image_path, text, arrow_pos, out_path)


__all__ = [
    "THUMBNAIL_WIDTH",
    "THUMBNAIL_HEIGHT",
    "SATURATION_BOOST",
    "TEXT_STROKE_PX",
    "TEXT_ANGLE_RANGE",
    "TEXT_FILL_YELLOW",
    "TEXT_FILL_WHITE",
    "FONT_CANDIDATES",
    "DEFAULT_ARROW_PATH",
    "ThumbnailCompositorError",
    "ThumbnailCompositor",
    "build_thumbnail",
    "text_angle",
    "load_display_font",
    "draw_arrow",
    "load_arrow",
    "fit_to_frame",
    "render_hook",
]
