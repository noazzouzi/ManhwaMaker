"""Rythme de coupe (Module 8) : d'une scène narrée à une suite de plans.

Un **plan** (:class:`ShotPlan`) associe une case, une fenêtre de cadrage, un mouvement et
une durée. C'est le maillon qui manquait entre la scène et la timeline : jusqu'ici une
case donnait exactement un clip, ce qui interdisait la cadence serrée du format court.

- :class:`LongPacing` — comportement historique : une case par clip, durée plancher
  ``min_clip_s``, et les plus petites cases sont écartées quand la scène est trop courte.
- :class:`ShortPacing` — plafond ``max_clip_s`` : une phrase de 4 s donne 4 plans. Quand
  la scène n'a pas assez de cases, les cases restantes sont **redécoupées** en plusieurs
  cadrages successifs (jusqu'à ``max_subshots_per_panel``).
"""

from __future__ import annotations

import logging
import math
from collections.abc import Mapping, Sequence
from typing import Protocol, runtime_checkable

from pydantic import BaseModel, Field

from src.models.format_profile import CropWindow, FormatProfile
from src.modules.framing import FramingStrategy

logger = logging.getLogger(__name__)

#: Passes de rattrapage du partage du temps : chacune reborne les durées puis redonne
#: l'écart aux cases encore libres. Le compte converge en deux ou trois passes ; la limite
#: n'est qu'un garde-fou contre une oscillation.
_SHARE_PASSES: int = 12


def content_height(entry: Mapping) -> int:
    """Hauteur de dessin d'une case : celle d'origine, avant agrandissement IA.

    Le temps de lecture dépend de ce que la case contient, pas de sa taille à l'écran :
    un personnage agrandi (:mod:`src.modules.upscaler`) garde le poids de sa case d'origine.
    """
    return int(entry.get("native_height", entry["height"]))


class ShotPlan(BaseModel):
    """Un plan : une case, un cadrage, un mouvement, une durée.

    Attributes:
        scene_index: scène d'origine.
        panel_index: case affichée.
        file: fichier de la case.
        panel_width, panel_height: dimensions de la case source.
        crop: fenêtre affichée (``None`` = la case entière).
        motion: mouvement de caméra.
        duration_s: durée du plan.
        subshot: rang du plan parmi ceux tirés de la même case (0 = premier).
    """

    scene_index: int = Field(ge=0)
    panel_index: int = Field(ge=0)
    file: str
    panel_width: int = Field(gt=0)
    panel_height: int = Field(gt=0)
    crop: CropWindow | None = None
    motion: str = "ken_burns"
    duration_s: float = Field(gt=0)
    subshot: int = Field(ge=0, default=0)


@runtime_checkable
class PacingStrategy(Protocol):
    """Découpe la durée d'une scène en plans."""

    name: str

    def plan(
        self,
        scene_index: int,
        panel_ids: Sequence[int],
        duration_s: float,
        meta_by_index: Mapping[int, Mapping],
        heavy_ids: Sequence[int] = (),
    ) -> list[ShotPlan]:
        """Plans couvrant exactement ``duration_s`` secondes."""
        ...


def _shot(
    scene_index: int, panel_id: int, entry: Mapping, crop: CropWindow | None, motion: str,
    duration: float, subshot: int,
) -> ShotPlan:
    return ShotPlan(
        scene_index=scene_index, panel_index=panel_id, file=str(entry["file"]),
        panel_width=int(entry["width"]), panel_height=int(entry["height"]),
        crop=crop, motion=motion, duration_s=duration, subshot=subshot,
    )


class LongPacing:
    """Une case par clip, durée plancher, poids proportionnel à la hauteur.

    Reproduit exactement le comportement du mode long : les plus petites cases sont
    écartées tant qu'elles n'auraient pas ``min_clip_s`` chacune, puis le temps est
    réparti au prorata des hauteurs après une base égale.
    """

    name = "long"

    def __init__(self, profile: FormatProfile, framing: FramingStrategy, *, min_panel_weight: int = 400) -> None:
        self.profile = profile
        self.framing = framing
        self.rules = profile.pacing
        self.min_clip_s = profile.pacing.min_clip_s or 0.0
        self.min_panel_weight = min_panel_weight
        #: Au-delà, une case ne tient pas dans le cadre en résolution native : il faut plus
        #: de temps pour la parcourir des yeux.
        self.frame_height = profile.framing.height

    def plan(self, scene_index, panel_ids, duration_s, meta_by_index, heavy_ids=()) -> list[ShotPlan]:
        # Import tardif : ``timeline_builder`` importe ce module, la reference croisee ne
        # peut donc pas etre resolue au chargement.
        from src.modules.timeline_builder import limit_panels_for_duration

        kept = limit_panels_for_duration(panel_ids, duration_s, meta_by_index, self.min_clip_s)
        if not kept:
            return []
        heavy = set(heavy_ids)
        n = len(kept)
        weights = [self._weight_of(pid, meta_by_index, heavy) for pid in kept]
        durations = self._share(duration_s, weights)
        shots: list[ShotPlan] = []
        spent = 0.0
        for k, pid in enumerate(kept):
            entry = meta_by_index[pid]
            duration = duration_s - spent if k == n - 1 else durations[k]
            window = self.framing.windows(entry, 1)[0]
            motion = "punch_in" if pid in heavy else "ken_burns"
            shots.append(_shot(scene_index, pid, entry, window, motion, duration, 0))
            spent += duration
        return shots

    def _weight_of(self, pid: int, meta_by_index, heavy: set[int]) -> float:
        """Part de la durée de la scène revenant à une case.

        La hauteur reste la base - une grande case demande plus de temps de lecture - mais
        elle ne suffisait pas : les hauteurs se ressemblent trop pour produire du rythme.
        L'impact marqué par le modèle (``action_heavy_ids``) est le seul signal qui dit
        *ce moment compte*, et il était jusqu'ici ignoré dans le partage du temps.
        """
        height = content_height(meta_by_index[pid])
        weight = float(max(height, self.min_panel_weight))
        if pid in heavy:
            weight *= self.rules.heavy_emphasis
        if height > self.frame_height:
            weight *= self.rules.tall_emphasis
        return weight

    def _share(self, duration_s: float, weights: Sequence[float]) -> list[float]:
        """Répartit ``duration_s`` au prorata des poids, borné, **sans en perdre une miette**.

        Le plancher et le plafond sont appliqués puis l'écart est redonné aux cases encore
        libres, jusqu'à ce que la somme retombe exactement sur la durée parlée : l'image ne
        doit jamais glisser par rapport à la voix.
        """
        n = len(weights)
        if n == 1:
            return [duration_s]
        floor_s = min(self.rules.emphasis_floor_s, duration_s / n)
        ceiling_s = max(self.rules.emphasis_ceiling_s, duration_s / n)
        total = float(sum(weights)) or 1.0
        shares = [duration_s * w / total for w in weights]
        for _ in range(_SHARE_PASSES):
            shares = [min(ceiling_s, max(floor_s, s)) for s in shares]
            gap = duration_s - sum(shares)
            free = [i for i, s in enumerate(shares) if floor_s < s < ceiling_s]
            if abs(gap) < 1e-9 or not free:
                break
            for i in free:
                shares[i] += gap / len(free)
        return shares


class ShortPacing:
    """Cadence serrée : aucun plan ne dépasse ``max_clip_s``.

    Le nombre de plans est déduit de la durée (4 s à 1,2 s de plafond → 4 plans), puis
    réparti sur les cases disponibles. Si la scène en manque, chaque case fournit
    plusieurs cadrages successifs — c'est le ``FramingStrategy`` qui les produit.
    """

    name = "short"

    def __init__(self, profile: FormatProfile, framing: FramingStrategy) -> None:
        self.profile = profile
        self.framing = framing
        self.max_clip_s = profile.pacing.max_clip_s or 1.2
        self.min_shot_s = profile.pacing.min_shot_s
        self.max_subshots = profile.pacing.max_subshots_per_panel
        self.motions = profile.motion.motions or ("punch_in",)

    def shot_count(self, duration_s: float, n_panels: int) -> int:
        """Nombre de plans visé, borné par ce que les cases peuvent fournir."""
        wanted = max(1, math.ceil(duration_s / self.max_clip_s - 1e-9))
        # Ne jamais descendre sous la duree plancher d'un plan.
        affordable = max(1, int(duration_s / self.min_shot_s))
        capacity = max(1, n_panels * self.max_subshots)
        return max(1, min(wanted, affordable, capacity))

    def _spread(self, total_shots: int, n_panels: int) -> list[int]:
        """Nombre de plans par case, réparti au plus égal."""
        base, extra = divmod(total_shots, n_panels)
        return [min(self.max_subshots, base + (1 if i < extra else 0)) for i in range(n_panels)]

    def plan(self, scene_index, panel_ids, duration_s, meta_by_index, heavy_ids=()) -> list[ShotPlan]:
        panels = list(panel_ids)
        if not panels:
            return []
        heavy = set(heavy_ids)
        total_shots = self.shot_count(duration_s, len(panels))
        per_panel = self._spread(total_shots, len(panels))
        total_shots = sum(per_panel) or 1
        duration = duration_s / total_shots

        shots: list[ShotPlan] = []
        spent = 0.0
        placed = 0
        for pid, count in zip(panels, per_panel):
            if count <= 0:
                continue
            entry = meta_by_index[pid]
            windows = self.framing.windows(entry, count)
            for sub in range(count):
                placed += 1
                # Le dernier plan absorbe l'arrondi : la scene dure exactement sa voix off.
                length = duration_s - spent if placed == total_shots else duration
                motion = "punch_in" if pid in heavy and sub == 0 else self.motions[(placed - 1) % len(self.motions)]
                shots.append(_shot(scene_index, pid, entry, windows[sub], motion, max(length, 1e-3), sub))
                spent += length
        if shots and duration > self.max_clip_s + 1e-6:
            logger.debug(
                "Scene %s : %d plan(s) de %.2fs, au-dessus du plafond %.2fs (cases insuffisantes)",
                scene_index, total_shots, duration, self.max_clip_s,
            )
        return shots


def make_pacing(profile: FormatProfile, framing: FramingStrategy) -> PacingStrategy:
    """Stratégie de rythme correspondant au profil."""
    if profile.pacing.max_clip_s is not None:
        return ShortPacing(profile, framing)
    return LongPacing(profile, framing)


__all__ = ["ShotPlan", "PacingStrategy", "LongPacing", "ShortPacing", "content_height", "make_pacing"]
