"""Fabrique des profils de format (Module 8).

Seul endroit où vivent les valeurs de chaque mode : le reste du pipeline reçoit un
:class:`~src.models.format_profile.FormatProfile` et n'a plus à savoir s'il monte un
16:9 ou un 9:16.

Les chiffres du mode SHORT sont calés sur la géométrie **réelle** du corpus (1295 cases
mesurées) : toutes les cases font 800 px de large pour une hauteur médiane de 775 px.
Une fenêtre 9:16 y mesure donc ``hauteur x 0,5625`` de large, soit 436 px sur une case
médiane, qu'il faut porter à 1080 px — **2,48x** d'agrandissement (3,74x au p90). C'est
ce qui fixe :data:`SHORT_MAX_UPSCALE`.
"""

from __future__ import annotations

import logging
from typing import Any

from src.models.format_profile import (
    AudioRules,
    FormatProfile,
    FramingRules,
    MotionRules,
    OutroRules,
    PacingRules,
    SubtitleRules,
    VideoFormat,
)
from src.utils.config import PROJECT_ROOT

logger = logging.getLogger(__name__)

_FONTS = PROJECT_ROOT / "config" / "fonts"

#: Plafond d'agrandissement total du mode SHORT.
#:
#: La mise au cadre 9:16 coûte déjà 2,48x sur une case médiane ; 3,2 laisse donc juste
#: la place du punch-in à 128 % (2,48 x 1,28 = 3,17). Sur une case plus petite, le
#: punch-in est **automatiquement réduit**, voire annulé, plutôt que d'empiler les
#: agrandissements — c'est la traduction directe de la règle « jamais de pixellisation »
#: dans un mode qui, lui, ne peut pas éviter d'agrandir.
SHORT_MAX_UPSCALE: float = 3.2
#: Amplitude du punch-in en mode SHORT (128 %), appliquée en **resserrant la fenêtre** de
#: recadrage : le mouvement est identique à l'écran mais consomme des pixels de la source
#: au lieu d'en inventer.
SHORT_PUNCH_IN_ZOOM: float = 0.28
#: Cadence de coupe maximale (secondes) : une phrase de 4 s donne donc 4 plans.
SHORT_MAX_CLIP_S: float = 1.2
#: Accélération de la voix, appliquée par Kokoro à la synthèse (pas de rééchantillonnage,
#: donc pas de montée de la hauteur de voix).
SHORT_SPEED: float = 1.20
#: Silence interne toléré avant rognage.
SHORT_MAX_SILENCE_S: float = 0.1
#: Accélération de la voix en format long. Même mécanisme que :data:`SHORT_SPEED` : Kokoro
#: parle plus vite à la synthèse, la hauteur de voix ne monte donc pas. Elle raccourcit
#: mécaniquement la vidéo, puisque la durée de chaque paragraphe suit sa narration.
#: ``--speed`` sur la ligne de commande continue de primer.
LONG_SPEED: float = 1.15

#: Polices d'affichage du mode SHORT, de la plus grasse à la plus sûre. Futura est une
#: police commerciale, absente de la machine : Montserrat Black (graisse 900, licence
#: SIL OFL) en tient lieu.
SHORT_FONTS: tuple[str, ...] = (
    str(_FONTS / "Montserrat-Black.ttf"),
    str(_FONTS / "Montserrat-Bold.ttf"),
    str(_FONTS / "Bangers-Regular.ttf"),
    "C:/Windows/Fonts/impact.ttf",
    "C:/Windows/Fonts/arialbd.ttf",
)
LONG_FONTS: tuple[str, ...] = (
    str(_FONTS / "Montserrat-Bold.ttf"),
    str(_FONTS / "Rubik-Bold.ttf"),
    "C:/Windows/Fonts/arialbd.ttf",
)

YELLOW: tuple[int, int, int] = (255, 255, 0)
WHITE: tuple[int, int, int] = (255, 255, 255)


def _long_profile() -> FormatProfile:
    """Mode long : comportement historique, à la vitesse de voix près.

    Seule dérogation : :data:`LONG_SPEED`. Tout le reste - cadrage, cadence, zoom,
    sous-titres - garde son réglage d'origine.
    """
    return FormatProfile(
        name="LONG",
        framing=FramingRules(
            width=1920, height=1080, fps=60, fit="contain",
            saliency_crop=False, background_blur=True, max_upscale=1.05,
        ),
        pacing=PacingRules(
            min_clip_s=2.5, max_clip_s=None, max_subshots_per_panel=1,
            expand_to_unused_panels=True,
        ),
        motion=MotionRules(
            motions=("ken_burns",), punch_in_zoom=0.05, punch_in_s=0.2, ken_burns_zoom=0.05,
        ),
        subtitles=SubtitleRules(
            min_words=2, max_words=4, font_candidates=LONG_FONTS, stroke_px=2, font_size_px=64,
            # 0,9222 = (1080 - 84) / 1080 : reproduit au pixel la marge basse historique.
            fill=WHITE, highlight_color=None, highlight_current_word=False, vertical_anchor=0.92222,
        ),
        audio=AudioRules(speed=LONG_SPEED, sentence_gap_s=0.2, padding_s=0.18, max_internal_silence_s=None),
        outro=OutroRules(enabled=False),
    )


def _short_profile() -> FormatProfile:
    """Mode court : 9:16 plein cadre, cadence serrée, voix accélérée, carte de titre."""
    return FormatProfile(
        name="SHORT",
        framing=FramingRules(
            width=1080, height=1920, fps=60, fit="cover_crop",
            saliency_crop=True, background_blur=False, max_upscale=SHORT_MAX_UPSCALE,
        ),
        pacing=PacingRules(
            min_clip_s=None, max_clip_s=SHORT_MAX_CLIP_S, max_subshots_per_panel=6, min_shot_s=0.45,
            expand_to_unused_panels=True,
        ),
        motion=MotionRules(
            motions=("punch_in", "fast_pan"), punch_in_zoom=SHORT_PUNCH_IN_ZOOM, punch_in_s=0.12,
            ken_burns_zoom=0.0, pan_ratio=0.85, pan_axis="auto",
        ),
        subtitles=SubtitleRules(
            min_words=1, max_words=3, font_candidates=SHORT_FONTS, stroke_px=8, font_size_px=92,
            fill=WHITE, highlight_color=YELLOW, highlight_current_word=True, vertical_anchor=0.74,
        ),
        audio=AudioRules(
            speed=SHORT_SPEED, sentence_gap_s=0.05, padding_s=0.05,
            max_internal_silence_s=SHORT_MAX_SILENCE_S,
        ),
        outro=OutroRules(enabled=True, duration_s=5.0, motion_blur=True, font_size_px=120),
    )


_BUILDERS = {"LONG": _long_profile, "SHORT": _short_profile}


class VideoConfigFactory:
    """Fabrique les règles d'assemblage d'un format.

    ``create`` accepte des surcharges par section, ce qui évite de sous-classer un profil
    pour ajuster une valeur depuis la ligne de commande :

    >>> VideoConfigFactory.create("SHORT", framing={"fps": 30}).framing.fps
    30
    """

    @staticmethod
    def available() -> tuple[str, ...]:
        """Formats reconnus."""
        return tuple(_BUILDERS)

    @staticmethod
    def create(fmt: VideoFormat | str = "LONG", **overrides: Any) -> FormatProfile:
        """Profil du format demandé.

        Args:
            fmt: ``"LONG"`` ou ``"SHORT"`` (insensible à la casse).
            **overrides: surcharges par section, ex. ``framing={"fps": 30}`` ou
                ``audio={"speed": 1.18}``.

        Raises:
            ValueError: format inconnu, section inconnue ou valeur invalide.
        """
        key = str(fmt).strip().upper()
        builder = _BUILDERS.get(key)
        if builder is None:
            raise ValueError(f"Format inconnu : {fmt} (attendu : {', '.join(_BUILDERS)})")
        profile = builder()
        if not overrides:
            return profile
        sections = dict(profile)
        for section, values in overrides.items():
            if section not in sections or section == "name":
                raise ValueError(f"Section de profil inconnue : {section} (attendu : {', '.join(k for k in sections if k != 'name')})")
            if not isinstance(values, dict):
                raise ValueError(f"La surcharge '{section}' doit etre un dictionnaire, recu {type(values).__name__}")
            sections[section] = sections[section].model_copy(update=values)
            # ``model_copy`` ne revalide pas : on reconstruit pour que les bornes tiennent.
            sections[section] = type(sections[section]).model_validate(sections[section].model_dump())
        updated = FormatProfile.model_validate(sections)
        logger.debug("Profil %s surcharge : %s", key, ", ".join(sorted(overrides)))
        return updated


def describe(profile: FormatProfile) -> str:
    """Résumé ASCII d'un profil, pour la console et les journaux."""
    framing, pacing, motion = profile.framing, profile.pacing, profile.motion
    cadence = f"<= {pacing.max_clip_s}s" if pacing.max_clip_s else f">= {pacing.min_clip_s}s"
    text = (
        f"{profile.name} {framing.width}x{framing.height}@{framing.fps} "
        f"({framing.fit}, agrandissement <= {framing.max_upscale:.2f}x), cadence {cadence}, "
        f"mouvements {'/'.join(motion.motions)}, sous-titres {profile.subtitles.min_words}-"
        f"{profile.subtitles.max_words} mots, voix x{profile.audio.speed:.2f}"
    )
    return text.encode("ascii", "replace").decode("ascii")


__all__ = [
    "SHORT_MAX_UPSCALE",
    "SHORT_PUNCH_IN_ZOOM",
    "SHORT_MAX_CLIP_S",
    "SHORT_SPEED",
    "LONG_SPEED",
    "SHORT_FONTS",
    "LONG_FONTS",
    "VideoConfigFactory",
    "describe",
]
