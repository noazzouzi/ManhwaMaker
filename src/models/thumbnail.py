"""Schémas Pydantic de la miniature YouTube (Module 7).

Deux niveaux, comme pour les scènes (:mod:`src.models.scene`) :

- :class:`ThumbnailDraft` : schéma **strict et minimal** envoyé au modèle comme
  ``response_schema`` (aucune valeur par défaut, aucune contrainte, pour rester
  compatible avec la conversion JSON-Schema du SDK) ;
- :class:`ThumbnailBrief` : le modèle du projet, normalisé et validé (accroche en
  majuscules bornée à 3 mots, positions contraintes), qui pilote la génération
  d'image puis le compositing.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

#: Côté de l'écran où poser la flèche.
ArrowPosition = Literal["left", "right"]
#: Emplacement du sujet principal dans l'image générée.
SubjectPosition = Literal["left", "center", "right"]
#: Nombre maximal de mots de l'accroche (au-delà, elle cesse d'être lisible en vignette).
MAX_HOOK_WORDS: int = 3
#: Longueur maximale de l'accroche en caractères.
MAX_HOOK_CHARS: int = 18


class ThumbnailDraft(BaseModel):
    """Réponse structurée attendue du modèle pour une miniature.

    Attributes:
        scene_description: description d'une scène visuellement **contrastée ou
            ironique** du chapitre, rédigée comme un prompt d'illustration.
        hook_text: accroche de 1 à 3 mots maximum (ex. « WAKE UP »).
        arrow_position: côté où poser la flèche jaune.
        subject_position: où se trouve le sujet, pour que la flèche le désigne.
        avoid: éléments à exclure de l'image (prompt négatif).
    """

    scene_description: str
    hook_text: str
    arrow_position: str
    subject_position: str
    avoid: str


class ThumbnailBrief(BaseModel):
    """Consignes normalisées d'une miniature, prêtes à être exécutées.

    Attributes:
        scene_description: prompt d'illustration.
        hook_text: accroche en majuscules, 1 à :data:`MAX_HOOK_WORDS` mots.
        arrow_position: côté de la flèche.
        subject_position: emplacement du sujet.
        avoid: prompt négatif.
        source_url: chapitre d'origine (traçabilité).
        model: modèle ayant produit la consigne.
    """

    scene_description: str = Field(min_length=1)
    hook_text: str = Field(min_length=1, max_length=MAX_HOOK_CHARS)
    arrow_position: ArrowPosition = "right"
    subject_position: SubjectPosition = "center"
    avoid: str = ""
    source_url: str = ""
    model: str = ""

    @property
    def hook_words(self) -> list[str]:
        return self.hook_text.split()


__all__ = [
    "ArrowPosition",
    "SubjectPosition",
    "MAX_HOOK_WORDS",
    "MAX_HOOK_CHARS",
    "ThumbnailDraft",
    "ThumbnailBrief",
]
