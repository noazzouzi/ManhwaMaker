"""Profils de format vidéo (Module 8) : règles d'assemblage du mode LONG et du mode SHORT.

Le profil est une **donnée immuable** décrivant *quoi* faire ; les stratégies
(:mod:`src.modules.framing`, :mod:`src.modules.pacing`) décrivent *comment* le faire. Le
pipeline ne connaît que le profil, ce qui permet d'ajouter un format sans toucher aux
modules de montage.

- ``LONG`` : 1920x1080, case entière à sa résolution native sur fond flouté, zoom borné à
  105 %, sous-titres de 2 à 4 mots. C'est le comportement historique, inchangé.
- ``SHORT`` : 1080x1920, fenêtre 9:16 recadrée sur la zone la plus dense du dessin,
  cadence de coupe ≤ 1,2 s, punch-in **par resserrement de la fenêtre**, sous-titres de
  1 à 3 mots avec surlignage du mot courant, voix accélérée, carte de titre finale.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, model_validator

#: Format de sortie demandé par l'utilisateur.
VideoFormat = Literal["LONG", "SHORT"]
#: Mode de mise au cadre d'une case.
#:
#: - ``contain`` : la case entre entière dans le cadre, le fond flouté comble les côtés ;
#: - ``cover_crop`` : une fenêtre au ratio du cadre est découpée dans la case, qui remplit
#:   alors l'écran bord à bord (au prix de ce qui sort du cadre).
FitMode = Literal["contain", "cover_crop"]
#: Comment une fenêtre donnée se pose dans le cadre. Distinct de :data:`FitMode`, qui est
#: la règle **du profil** : en ``cover_crop``, une case trop plate peut malgré tout être
#: posée en ``contain`` par la soupape de sécurité.
WindowFit = Literal["cover", "contain"]


class CropWindow(BaseModel):
    """Fenêtre rectangulaire découpée dans une case, en **pixels de la source**.

    Attributes:
        x, y: coin supérieur gauche dans la case.
        width, height: dimensions de la fenêtre.
        fit: comment la poser dans le cadre. ``cover`` = elle le remplit (ce qui dépasse
            est coupé), ``contain`` = elle y entre entière, le fond floutant les côtés.
            L'information est **portée par la fenêtre** et non déduite de sa géométrie :
            une fenêtre repliée en letterbox ressemble à une fenêtre recadrée, et les
            confondre fait lire 14x là où l'agrandissement réel est de 2x.
    """

    x: int = Field(ge=0)
    y: int = Field(ge=0)
    width: int = Field(gt=0)
    height: int = Field(gt=0)
    fit: WindowFit = "cover"

    @property
    def right(self) -> int:
        return self.x + self.width

    @property
    def bottom(self) -> int:
        return self.y + self.height

    @property
    def center(self) -> tuple[float, float]:
        return self.x + self.width / 2, self.y + self.height / 2

    def cover_scale(self, frame_width: int, frame_height: int) -> float:
        """Agrandissement pour que la fenêtre **remplisse** le cadre (mode ``cover_crop``).

        Supérieur à 1 = la source est étirée (perte de netteté) ; inférieur à 1 = réduite.
        """
        return max(frame_width / self.width, frame_height / self.height)

    def contain_scale(self, frame_width: int, frame_height: int) -> float:
        """Agrandissement pour que la fenêtre **entre entière** dans le cadre.

        C'est la bonne mesure pour une case affichée en letterbox : la confondre avec
        :meth:`cover_scale` fait croire à un agrandissement de 9,5x là où il n'est que
        de 1,35x.
        """
        return min(frame_width / self.width, frame_height / self.height)

    def effective_scale(self, frame_width: int, frame_height: int) -> float:
        """Agrandissement réellement subi, selon le mode de pose de la fenêtre."""
        if self.fit == "contain":
            return self.contain_scale(frame_width, frame_height)
        return self.cover_scale(frame_width, frame_height)


class FramingRules(BaseModel):
    """Comment une case occupe le cadre.

    Attributes:
        width, height, fps: format de la séquence.
        fit: ``contain`` (fond flouté) ou ``cover_crop`` (plein cadre).
        saliency_crop: placer la fenêtre sur la zone la plus dessinée plutôt qu'au centre.
        background_blur: remplir les côtés d'un fond flouté et assombri.
        max_upscale: agrandissement **total** toléré par rapport aux pixels d'origine.
            En ``cover_crop`` la mise au cadre impose déjà son propre agrandissement : ce
            plafond gouverne donc ce qui *s'y ajoute* (le punch-in), et l'annule quand la
            case est trop petite pour l'absorber.
        crop_fallback_contain: soupape du mode ``cover_crop``. Une case très plate donne
            une fenêtre 9:16 minuscule : mesuré sur le corpus, 23 % des fenêtres
            dépasseraient le plafond, jusqu'à **9,5x** sur une case de 203 px de haut.
            Au-delà du plafond, la case est alors affichée entière sur fond flouté plutôt
            qu'étirée. Mettre à ``False`` pour imposer le recadrage partout, au prix de
            la netteté.
    """

    width: int = Field(gt=0)
    height: int = Field(gt=0)
    fps: int = Field(gt=0, default=60)
    fit: FitMode = "contain"
    saliency_crop: bool = False
    background_blur: bool = True
    max_upscale: float = Field(gt=0, default=1.05)
    crop_fallback_contain: bool = True

    @property
    def aspect_ratio(self) -> float:
        return self.width / self.height

    @property
    def is_vertical(self) -> bool:
        return self.height > self.width


class PacingRules(BaseModel):
    """Rythme de coupe.

    Attributes:
        min_clip_s: durée minimale d'une case ; en dessous, les plus petites sont écartées
            (``None`` = pas de plancher).
        max_clip_s: durée maximale d'une case ; au-delà, la scène est **subdivisée en
            plusieurs plans** (``None`` = pas de plafond).
        max_subshots_per_panel: nombre de re-cadrages tirés d'une même case quand la scène
            manque de cases pour tenir la cadence.
        min_shot_s: durée plancher d'un sous-plan, pour ne pas produire de clignotement.
    """

    min_clip_s: float | None = 2.5
    max_clip_s: float | None = None
    max_subshots_per_panel: int = Field(ge=1, default=1)
    min_shot_s: float = Field(gt=0, default=0.45)

    @model_validator(mode="after")
    def _check_bounds(self) -> PacingRules:
        if self.min_clip_s is not None and self.max_clip_s is not None and self.min_clip_s > self.max_clip_s:
            raise ValueError("min_clip_s ne peut pas depasser max_clip_s")
        return self


class MotionRules(BaseModel):
    """Mouvements de caméra autorisés et leur amplitude.

    Attributes:
        motions: mouvements piochés cycliquement (hors ``punch_in``, réservé aux cases
            marquées ``action_heavy``).
        punch_in_zoom: amplitude du punch-in (0,05 = 105 %, 0,28 = 128 %).
        punch_in_s: temps pour atteindre le zoom.
        ken_burns_zoom: amplitude du zoom lent.
        pan_ratio: part de la marge disponible parcourue par un balayage.
        pan_axis: ``auto`` choisit l'axe où la fenêtre a de la marge ; en recadrage 9:16
            sur des cases plus larges que hautes, cette marge est **horizontale**.
    """

    motions: tuple[str, ...] = ("ken_burns",)
    punch_in_zoom: float = Field(ge=0, default=0.05)
    punch_in_s: float = Field(gt=0, default=0.2)
    ken_burns_zoom: float = Field(ge=0, default=0.05)
    pan_ratio: float = Field(ge=0, le=1, default=0.8)
    pan_axis: Literal["auto", "horizontal", "vertical"] = "auto"


class SubtitleRules(BaseModel):
    """Mise en forme des sous-titres.

    Attributes:
        min_words, max_words: bornes du découpage en blocs.
        font_candidates: polices par ordre de préférence.
        stroke_px: contour noir, en pixels à la hauteur de référence du profil.
        font_size_px: corps du texte, même référence.
        fill: couleur du texte.
        highlight_color: couleur du mot courant (``None`` = pas de surlignage).
        highlight_current_word: surligner le mot en cours de prononciation.
        vertical_anchor: position du bloc, 0 = haut, 1 = bas de l'image.
    """

    min_words: int = Field(ge=1, default=2)
    max_words: int = Field(ge=1, default=4)
    font_candidates: tuple[str, ...] = ()
    stroke_px: int = Field(ge=0, default=2)
    font_size_px: int = Field(gt=0, default=64)
    fill: tuple[int, int, int] = (255, 255, 255)
    highlight_color: tuple[int, int, int] | None = None
    highlight_current_word: bool = False
    vertical_anchor: float = Field(ge=0, le=1, default=0.92)

    @model_validator(mode="after")
    def _check_words(self) -> SubtitleRules:
        if self.min_words > self.max_words:
            raise ValueError("min_words ne peut pas depasser max_words")
        return self


class AudioRules(BaseModel):
    """Réglages de la voix off.

    Attributes:
        speed: vitesse de lecture Kokoro. L'accélération se fait **à la synthèse** : un
            rééchantillonnage a posteriori (``pydub.speedup``) monterait la hauteur de voix.
        sentence_gap_s: silence entre deux phrases d'une même scène.
        padding_s: silence en fin de scène.
        max_internal_silence_s: silence interne toléré ; au-delà il est rogné après
            synthèse (``None`` = aucun rognage).
        silence_threshold_db: seuil sous lequel un passage est considéré comme silencieux.
    """

    speed: float = Field(gt=0, default=1.0)
    sentence_gap_s: float = Field(ge=0, default=0.2)
    padding_s: float = Field(ge=0, default=0.18)
    max_internal_silence_s: float | None = None
    silence_threshold_db: float = -45.0


class OutroRules(BaseModel):
    """Carte de titre finale.

    Attributes:
        enabled: produire l'outro.
        duration_s: durée, prise sur la fin de la vidéo.
        motion_blur: flou de bougé sur la case de fond.
        font_size_px: corps du titre, à la hauteur de référence du profil.
    """

    enabled: bool = False
    duration_s: float = Field(gt=0, default=5.0)
    motion_blur: bool = True
    font_size_px: int = Field(gt=0, default=110)


class FormatProfile(BaseModel):
    """Jeu complet de règles d'un format de sortie.

    Construit par :class:`~src.modules.format_factory.VideoConfigFactory`, jamais à la main :
    la fabrique est le seul endroit où vivent les valeurs de chaque mode.
    """

    model_config = {"frozen": True}

    name: VideoFormat
    framing: FramingRules
    pacing: PacingRules
    motion: MotionRules
    subtitles: SubtitleRules
    audio: AudioRules
    outro: OutroRules

    @property
    def is_short(self) -> bool:
        return self.name == "SHORT"


__all__ = [
    "VideoFormat",
    "FitMode",
    "CropWindow",
    "FramingRules",
    "PacingRules",
    "MotionRules",
    "SubtitleRules",
    "AudioRules",
    "OutroRules",
    "FormatProfile",
]
