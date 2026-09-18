"""Schéma Pydantic d'une case (panel) produite par le Smart Slicer.

Une case est une tranche horizontale de la bande continue d'un chapitre :
``image`` contient les pixels RGB (H x W x 3, uint8) et les autres champs
décrivent sa position dans la bande. L'image est exclue de ``model_dump`` /
``model_dump_json`` (``Field(exclude=True)``) pour que les métadonnées restent
sérialisables sans embarquer les pixels.
"""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

#: Type d'affichage d'une case dans CapCut :
#: - ``static`` : case affichée entière (zoom Ken Burns doux) ;
#: - ``scroll_vertical`` : case géante (> ``GIANT_PANEL_HEIGHT``) parcourue par
#:   un défilement vertical automatique.
PanelType = Literal["static", "scroll_vertical"]


class Panel(BaseModel):
    """Case individuelle découpée dans la bande verticale d'un chapitre.

    Attributes:
        index: position de la case dans l'ordre de lecture (0 = haut de la bande).
        y_start: ligne de début (incluse) dans la bande d'origine, après padding.
        y_end: ligne de fin (exclue) dans la bande d'origine, après padding.
        height: hauteur en pixels (``y_end - y_start``).
        width: largeur en pixels (largeur de la bande).
        type: ``"static"`` ou ``"scroll_vertical"`` (case géante).
        image: pixels RGB de la case, ``np.ndarray`` de forme (height, width, 3),
            dtype ``uint8``. Exclue des exports Pydantic.

    Deux cases sont égales (``==``) si leurs métadonnées sont identiques et si
    leurs pixels sont identiques (``np.array_equal``) ; l'égalité par défaut de
    Pydantic comparerait les tableaux NumPy et lèverait une ``ValueError``.
    Les instances ne sont pas hachables (modèle mutable contenant un tableau).
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    index: int = Field(ge=0)
    y_start: int = Field(ge=0)
    y_end: int = Field(gt=0)
    height: int = Field(gt=0)
    width: int = Field(gt=0)
    type: PanelType
    #: ``"top"`` / ``"middle"`` / ``"bottom"`` si la case est un bloc d'une case très haute
    #: coupée horizontalement en 2 ou 3 (largeur d'origine conservée).
    part: Literal["top", "middle", "bottom"] | None = None
    #: Numéro du segment d'origine partagé par les blocs d'une case coupée.
    source_index: int | None = Field(default=None, ge=0)
    image: np.ndarray = Field(exclude=True, repr=False)

    @field_validator("image")
    @classmethod
    def _validate_image(cls, value: np.ndarray) -> np.ndarray:
        """Vérifie que l'image est bien un tableau RGB uint8 de forme (H, W, 3).

        Le type ``np.ndarray`` est déjà garanti par Pydantic (``is_instance_of``)
        avant l'appel de ce validateur ``after``.
        """
        if value.ndim != 3 or value.shape[2] != 3:
            raise ValueError(
                f"Panel.image doit etre de forme (H, W, 3), recu {value.shape}"
            )
        if value.dtype != np.uint8:
            raise ValueError(f"Panel.image doit etre uint8, recu {value.dtype}")
        return value

    @model_validator(mode="after")
    def _validate_consistency(self) -> Panel:
        """Vérifie la cohérence entre les bornes, la hauteur/largeur et l'image."""
        if self.y_end <= self.y_start:
            raise ValueError(
                f"y_end ({self.y_end}) doit etre > y_start ({self.y_start})"
            )
        if self.height != self.y_end - self.y_start:
            raise ValueError(
                f"height ({self.height}) != y_end - y_start "
                f"({self.y_end - self.y_start})"
            )
        if self.image.shape[0] != self.height or self.image.shape[1] != self.width:
            raise ValueError(
                f"image.shape {self.image.shape[:2]} incoherent avec "
                f"(height, width) = ({self.height}, {self.width})"
            )
        return self

    def to_dict(self) -> dict[str, Any]:
        """Retourne les métadonnées de la case sous forme de ``dict`` (sans l'image)."""
        return self.model_dump()

    def __eq__(self, other: object) -> bool:
        """Égalité par métadonnées + pixels (``np.array_equal``), sans ambiguïté NumPy."""
        if not isinstance(other, Panel):
            return NotImplemented
        if self.to_dict() != other.to_dict():
            return False
        return bool(np.array_equal(self.image, other.image))

    # Un modèle mutable qui définit ``__eq__`` ne doit pas être hachable.
    __hash__ = None  # type: ignore[assignment]
