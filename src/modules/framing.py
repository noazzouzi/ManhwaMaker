"""Mise au cadre d'une case (Module 8) : fenêtre affichée, recadrage et sous-plans.

Deux stratégies interchangeables :

- :class:`ContainFraming` (mode LONG) — la case entière, sans rognage : la fenêtre couvre
  toute la case et le fond flouté comble les côtés ;
- :class:`SalientCropFraming` (mode SHORT) — une fenêtre au ratio du cadre, placée sur la
  zone la **plus dessinée** de la case, et déclinée en plusieurs sous-plans quand le
  rythme réclame plus de plans que la scène n'a de cases.

Sur la saillance
----------------
Il ne s'agit **pas** de détection de visage : les cascades d'OpenCV sont entraînées sur
des photographies et échouent sur du trait encré. On mesure une **densité de contours**
lissée, et la fenêtre est posée là où le dessin est le plus dense — en pratique les
personnages et les moments d'impact, les aplats et les fonds vides ne produisant aucun
contour.

Une pondération destinée à fuir les bulles de dialogue (pénaliser les colonnes blanches)
a été essayée puis **retirée** : mesurée sur 117 cases, elle augmentait la part de blanc
dans le cadrage de 2,4 points, n'aidait que dans 39 % des cas et nuisait dans 48 %. La
densité brute lui est préférée, plus simple et sans constante arbitraire. Un vrai gain
demanderait un modèle de détection de visages de dessin animé, à ajouter le cas échéant
comme dépendance optionnelle.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping, Sequence
from typing import Protocol, runtime_checkable

import cv2
import numpy as np

from src.models.format_profile import CropWindow

logger = logging.getLogger(__name__)

#: Lissage du profil de saillance (en fraction de la dimension analysée) : évite qu'un
#: détail isolé n'attire tout le cadrage.
SALIENCY_SMOOTH_RATIO: float = 0.06
#: Écart minimal entre deux pics retenus, en fraction de la largeur de fenêtre.
PEAK_SEPARATION_RATIO: float = 0.6


def _gray(image: np.ndarray) -> np.ndarray:
    """Niveaux de gris. Le test de vacuite vient AVANT la conversion : ``cv2.cvtColor``
    leve une assertion sur une image vide au lieu de renvoyer un tableau vide."""
    if image.size == 0:
        return np.zeros((0, 0), dtype=np.uint8)
    if image.ndim == 3:
        return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    return image


def saliency_profile(image: np.ndarray, axis: int = 0) -> np.ndarray:
    """Densité de dessin le long d'un axe (``0`` = colonnes, ``1`` = lignes).

    Densité de contours lissée, sans pondération : voir l'en-tête du module sur la
    tentative de pénaliser les bulles, mesurée puis abandonnée.

    Returns:
        Vecteur normalisé entre 0 et 1, de la taille de la dimension analysée.
    """
    gray = _gray(image)
    if gray.size == 0:
        return np.zeros(1, dtype=np.float32)
    edges = np.abs(cv2.Laplacian(gray, cv2.CV_32F, ksize=3))
    profile = (edges.sum(axis=0) if axis == 0 else edges.sum(axis=1)).astype(np.float32)
    window = max(1, int(len(profile) * SALIENCY_SMOOTH_RATIO) | 1)
    if window > 1 and len(profile) > window:
        kernel = np.ones(window, dtype=np.float32) / window
        profile = np.convolve(profile, kernel, mode="same")
    peak = float(profile.max())
    return profile / peak if peak > 0 else profile


def window_scores(profile: np.ndarray, size: int) -> np.ndarray:
    """Saillance totale de chaque position possible d'une fenêtre de largeur ``size``.

    Somme glissante en O(n) (cumul puis différence) : le profil d'une case fait quelques
    centaines d'éléments, mais la fonction est appelée pour chaque plan de chaque scène.
    """
    span = len(profile)
    size = max(1, min(size, span))
    cumulative = np.concatenate([[0.0], np.cumsum(profile, dtype=np.float64)])
    return cumulative[size:] - cumulative[:-size]


def best_offset(profile: np.ndarray, size: int) -> int:
    """Position de la fenêtre de largeur ``size`` couvrant le plus de saillance."""
    if len(profile) - size <= 0:
        return 0
    return int(np.argmax(window_scores(profile, size)))


def spread_offsets(profile: np.ndarray, size: int, count: int) -> list[int]:
    """``count`` positions **distinctes**, réparties sur toute la marge disponible.

    La marge est découpée en ``count`` bandes et le meilleur point de chaque bande est
    retenu, puis les bandes sont classées par saillance décroissante : le premier plan
    tombe donc sur la zone la plus dessinée, les suivants explorent le reste de la case.

    Une exclusion par voisinage a été essayée d'abord : sur ces cases, la marge utile
    (~400 px pour une fenêtre de ~400 px) était entièrement absorbée par la première
    zone interdite, et toutes les positions retombaient sur le même point. Le découpage
    en bandes garantit la distinction dès qu'il y a de la marge.
    """
    count = max(1, count)
    limit = len(profile) - size
    if limit <= 0:
        return [0] * count
    scores = window_scores(profile, size)
    edges = np.linspace(0, len(scores), count + 1).astype(int)
    picked: list[tuple[float, int]] = []
    for start, end in zip(edges[:-1], edges[1:]):
        end = max(end, start + 1)
        band = scores[start:end]
        offset = start + int(np.argmax(band))
        picked.append((float(band.max()), offset))
    picked.sort(key=lambda item: -item[0])
    return [min(offset, limit) for _, offset in picked]


def base_window(panel_width: int, panel_height: int, frame_ratio: float) -> tuple[int, int]:
    """Plus grande fenêtre de ratio ``frame_ratio`` tenant dans la case.

    Returns:
        ``(largeur, hauteur)`` de la fenêtre.
    """
    if panel_width / panel_height > frame_ratio:
        height = panel_height
        width = max(1, round(panel_height * frame_ratio))
    else:
        width = panel_width
        height = max(1, round(panel_width / frame_ratio))
    return min(width, panel_width), min(height, panel_height)


def allowed_tighten(window: CropWindow, frame_width: int, frame_height: int, max_upscale: float) -> float:
    """Resserrement encore permis avant d'atteindre le plafond d'agrandissement.

    En recadrage plein cadre, la mise au cadre impose déjà son propre agrandissement :
    cette marge est donc ce qui reste pour le punch-in. Elle vaut 0 quand la case est
    trop petite, ce qui **annule** le punch-in au lieu d'empiler les agrandissements.
    """
    base = window.cover_scale(frame_width, frame_height)
    if base <= 0:
        return 0.0
    return max(0.0, max_upscale / base - 1.0)


def tighten_window(window: CropWindow, amount: float, bounds: tuple[int, int]) -> CropWindow:
    """Resserre la fenêtre de ``amount`` (0,28 = 128 %) autour de son centre.

    Le resserrement est borné par la case : une fenêtre ne sort jamais du dessin.
    """
    if amount <= 0:
        return window
    panel_width, panel_height = bounds
    factor = 1.0 / (1.0 + amount)
    width = max(8, round(window.width * factor))
    height = max(8, round(window.height * factor))
    center_x, center_y = window.center
    x = int(round(center_x - width / 2))
    y = int(round(center_y - height / 2))
    x = max(0, min(x, panel_width - width))
    y = max(0, min(y, panel_height - height))
    return CropWindow(x=x, y=y, width=width, height=height, fit=window.fit)


@runtime_checkable
class FramingStrategy(Protocol):
    """Fournit les fenêtres d'affichage d'une case."""

    name: str

    def windows(self, panel: Mapping, count: int = 1) -> list[CropWindow]:
        """``count`` fenêtres successives pour cette case, dans l'ordre de montage."""
        ...


class ContainFraming:
    """Mode LONG : la case entière, jamais rognée.

    La fenêtre couvre toute la case ; c'est le rendu qui la réduit si elle dépasse le
    cadre, et le fond flouté qui comble les côtés.
    """

    name = "contain"

    def windows(self, panel: Mapping, count: int = 1) -> list[CropWindow]:
        width, height = int(panel["width"]), int(panel["height"])
        full = CropWindow(x=0, y=0, width=width, height=height, fit="contain")
        return [full] * max(1, count)


class SalientCropFraming:
    """Mode SHORT : fenêtre au ratio du cadre, posée sur la zone la plus dessinée.

    Args:
        frame_width, frame_height: dimensions du cadre de sortie.
        max_upscale: plafond d'agrandissement total (voir :func:`allowed_tighten`).
        loader: ``index de case -> image RGB``. Injectable pour les tests ; sans lui, la
            fenêtre est simplement centrée (repli déterministe, jamais une erreur).
        tighten_step: resserrement appliqué d'un sous-plan au suivant.
    """

    name = "salient_crop"

    def __init__(
        self,
        frame_width: int,
        frame_height: int,
        *,
        max_upscale: float = 3.2,
        loader: Callable[[Mapping], np.ndarray | None] | None = None,
        tighten_step: float = 0.16,
        fallback_contain: bool = True,
    ) -> None:
        self.frame_width, self.frame_height = frame_width, frame_height
        self.frame_ratio = frame_width / frame_height
        self.max_upscale = max_upscale
        self.fallback_contain = fallback_contain
        self.loader = loader
        self.tighten_step = tighten_step
        self._profiles: dict[int, tuple[np.ndarray, np.ndarray]] = {}

    def _profile_for(self, panel: Mapping) -> tuple[np.ndarray, np.ndarray] | None:
        """Profils de saillance (colonnes, lignes) de la case, mis en cache."""
        if self.loader is None:
            return None
        index = int(panel.get("index", -1))
        cached = self._profiles.get(index)
        if cached is not None:
            return cached
        try:
            image = self.loader(panel)
        except Exception as exc:  # noqa: BLE001 - une case illisible ne bloque pas le montage
            logger.warning("Saillance indisponible pour la case %s (%s) : cadrage centre", index, exc)
            image = None
        if image is None:
            return None
        pair = (saliency_profile(image, axis=0), saliency_profile(image, axis=1))
        self._profiles[index] = pair
        return pair

    def windows(self, panel: Mapping, count: int = 1) -> list[CropWindow]:
        panel_width, panel_height = int(panel["width"]), int(panel["height"])
        width, height = base_window(panel_width, panel_height, self.frame_ratio)
        bounds = (panel_width, panel_height)
        count = max(1, count)

        # Soupape : une case tres plate donne une fenetre 9:16 minuscule, qu'il faudrait
        # etirer bien au-dela du plafond (jusqu'a 9,5x sur une case de 203 px de haut).
        # On l'affiche alors entiere, quitte a laisser le fond flouter les cotes.
        probe = CropWindow(x=0, y=0, width=width, height=height)
        if self.fallback_contain and probe.cover_scale(self.frame_width, self.frame_height) > self.max_upscale:
            full = CropWindow(x=0, y=0, width=panel_width, height=panel_height, fit="contain")
            logger.debug(
                "Case %s (%dx%d) : recadrage 9:16 abandonne (%.1fx > %.1fx), affichage entier a %.2fx",
                panel.get("index"), panel_width, panel_height,
                probe.cover_scale(self.frame_width, self.frame_height), self.max_upscale,
                full.contain_scale(self.frame_width, self.frame_height),
            )
            # Sans marge de recadrage, la variation entre plans vient du zoom : sinon les
            # sous-plans seraient identiques et la coupe ressemblerait a un arret sur image.
            headroom = max(0.0, self.max_upscale / max(full.contain_scale(self.frame_width, self.frame_height), 1e-6) - 1.0)
            return [
                tighten_window(full, min(self.tighten_step * shot, headroom), bounds)
                for shot in range(count)
            ]

        profiles = self._profile_for(panel)

        # L'axe qui porte la variation est celui ou la fenetre a de la marge. Sur des
        # cases plus larges que le 9:16 c'est l'horizontale ; la verticale n'en offre
        # que sur les cases tres hautes.
        room_x, room_y = panel_width - width, panel_height - height
        if profiles is None:
            xs, ys = [room_x // 2] * count, [room_y // 2] * count
        else:
            columns, rows = profiles
            xs = spread_offsets(columns, width, count) if room_x > 0 else [0] * count
            ys = spread_offsets(rows, height, count) if room_y > 0 else [0] * count
            if room_x > 0 and room_y > 0:
                # Les deux axes ont de la marge : on fige la verticale sur son meilleur
                # point et on fait varier l'horizontale, plus lisible en 9:16.
                ys = [ys[0]] * count

        results: list[CropWindow] = []
        for shot in range(count):
            window = CropWindow(
                x=max(0, min(xs[shot], room_x)), y=max(0, min(ys[shot], room_y)),
                width=width, height=height,
            )
            # Un plan sur deux est resserre : la variation d'echelle s'ajoute a celle de
            # position, sans jamais depasser le plafond d'agrandissement du profil.
            if shot % 2 == 1:
                room = allowed_tighten(window, self.frame_width, self.frame_height, self.max_upscale)
                window = tighten_window(window, min(self.tighten_step, room), bounds)
            results.append(window)
        return results


def make_framing(profile, loader: Callable[[Mapping], np.ndarray | None] | None = None) -> FramingStrategy:
    """Stratégie de cadrage correspondant à un :class:`~src.models.format_profile.FormatProfile`."""
    framing = profile.framing
    if framing.fit == "cover_crop":
        return SalientCropFraming(
            framing.width, framing.height, max_upscale=framing.max_upscale,
            loader=loader if framing.saliency_crop else None,
            fallback_contain=framing.crop_fallback_contain,
        )
    return ContainFraming()


__all__ = [
    "SALIENCY_SMOOTH_RATIO",
    "PEAK_SEPARATION_RATIO",
    "saliency_profile",
    "window_scores",
    "best_offset",
    "spread_offsets",
    "base_window",
    "allowed_tighten",
    "tighten_window",
    "FramingStrategy",
    "ContainFraming",
    "SalientCropFraming",
    "make_framing",
]
