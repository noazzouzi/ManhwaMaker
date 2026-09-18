"""Schémas Pydantic des scènes narratives produites par l'Analyzer Gemini (Module 3).

Une *scène* regroupe des cases consécutives (``panel_ids``) qui forment un même
moment, avec une ``narration`` (présent, 3ᵉ personne) destinée à la synthèse
vocale et une ``emotion`` (ton dominant) utilisée en aval pour le montage.

Deux niveaux de modèles :

- :class:`SceneDraft` / :class:`SceneBatch` : schéma **strict et minimal** envoyé à
  Gemini comme ``response_schema`` (pas de valeur par défaut ni de contrainte,
  pour rester compatible avec la conversion JSON-Schema du SDK) ;
- :class:`Scene` / :class:`ChapterAnalysis` : modèles du projet, enrichis
  (index global, métadonnées du chapitre, consommation de tokens).
"""

from __future__ import annotations

from typing import Literal, get_args

from pydantic import BaseModel, Field

#: Ton dominant d'une scène (valeurs imposées à Gemini via le schéma de réponse).
Emotion = Literal[
    "neutral",
    "calm",
    "tension",
    "action",
    "sad",
    "happy",
    "humor",
    "romance",
    "fear",
    "mystery",
    "epic",
]

#: Tuple des émotions autorisées (dérivé de :data:`Emotion`).
EMOTIONS: tuple[str, ...] = get_args(Emotion)


class SceneDraft(BaseModel):
    """Scène telle que renvoyée par Gemini (schéma de réponse).

    Attributes:
        panel_ids: numéros (``Panel.index``) des cases regroupées, ordre de lecture.
        narration: texte de récap au présent, 3ᵉ personne, lu par la voix off.
        emotion: ton dominant de la scène.
        is_filler: ``True`` si la scène n'apporte rien au récit (carte-titre, logo,
            crédits, mention de l'éditeur, publicité, « à suivre », note de l'auteur) :
            elle est conservée pour couvrir toutes les cases mais ignorée au montage.
    """

    panel_ids: list[int]
    narration: str
    emotion: Emotion
    is_filler: bool = False


class SceneBatch(BaseModel):
    """Réponse structurée attendue de Gemini pour un lot de cases."""

    scenes: list[SceneDraft]


class Scene(SceneDraft):
    """Scène du chapitre, numérotée globalement (``index`` = ordre de lecture).

    Depuis la génération en deux étapes, ``panel_ids`` ne contient que les
    **cases clés** retenues pour illustrer le paragraphe ``narration`` ; les cases
    de transition ne sont montées nulle part. ``beat_ids`` renvoie aux beats
    (:class:`Beat`) couverts par le paragraphe.
    """

    index: int = Field(ge=0)
    beat_ids: list[int] = Field(default_factory=list)
    #: Cases clés « action_heavy » (impact décisif) : zoom punch-in au montage.
    action_heavy_ids: list[int] = Field(default_factory=list)


# --- Génération en deux étapes -------------------------------------------------------
class BeatDraft(BaseModel):
    """Beat narratif extrait d'un lot de cases (étape 1a, schéma de réponse Gemini).

    Attributes:
        panel_ids: cases consécutives couvertes par le beat.
        summary: résumé factuel (1 à 2 phrases) de ce qui se passe.
        characters: noms des personnages tels qu'écrits dans les dialogues.
        dialogue: répliques importantes (citations courtes).
        is_filler: beat sans récit (titre, crédits, publicité).
    """

    panel_ids: list[int]
    summary: str
    characters: list[str]
    dialogue: list[str]
    is_filler: bool = False


class BeatBatch(BaseModel):
    """Réponse structurée de l'étape 1a pour un lot de cases."""

    beats: list[BeatDraft]


class Beat(BeatDraft):
    """Beat du chapitre, numéroté globalement."""

    index: int = Field(ge=0)


class ParagraphDraft(BaseModel):
    """Paragraphe du script global (étape 1b, schéma de réponse Gemini).

    Attributes:
        text: texte narré (storytelling rétrospectif, sans formule visuelle).
        beat_ids: beats consécutifs couverts par le paragraphe.
        emotion: ton dominant.
    """

    text: str
    beat_ids: list[int]
    emotion: Emotion


class ScriptDraft(BaseModel):
    """Réponse structurée de l'étape 1b : le script complet du chapitre."""

    paragraphs: list[ParagraphDraft]


class RecapParagraph(BaseModel):
    """Paragraphe complet du mode « une requête » : narration **et** cases à monter.

    Attributes:
        text: le paragraphe narré.
        emotion: émotion dominante.
        key_panel_ids: 1 à 4 cases illustrant ce paragraphe, dans l'ordre de lecture.
        action_heavy_ids: sous-ensemble montrant un impact décisif (punch-in au montage).
    """

    text: str
    emotion: str
    key_panel_ids: list[int]
    action_heavy_ids: list[int]


class RecapDraft(BaseModel):
    """Schéma de réponse du mode « une requête » : tout le récap en un seul appel."""

    paragraphs: list[RecapParagraph]


class KeyframeChoice(BaseModel):
    """Cases clés retenues pour un paragraphe (étape 2, schéma de réponse Gemini).

    Attributes:
        paragraph_index: paragraphe concerné.
        key_panel_ids: cases clés, dans l'ordre de lecture.
        action_heavy_ids: sous-ensemble des cases clés montrant un impact décisif
            (coup porté, explosion, rugissement) : elles reçoivent un punch-in.
    """

    paragraph_index: int
    key_panel_ids: list[int]
    action_heavy_ids: list[int]


class KeyframeBatch(BaseModel):
    """Réponse structurée de l'étape 2 pour un groupe de paragraphes."""

    choices: list[KeyframeChoice]


class ChapterAnalysis(BaseModel):
    """Résultat complet de l'analyse d'un chapitre.

    Attributes:
        series_title: titre de la série (vide si inconnu).
        episode_title: titre de l'épisode (vide si inconnu).
        source_url: URL du chapitre analysé (vide si les cases viennent d'un dossier).
        model: nom du modèle Gemini utilisé.
        language: code de langue de la narration (``"fr"``, ``"en"``...).
        n_panels: nombre de cases soumises.
        n_batches: nombre de lots (appels API) effectués.
        scenes: scènes dans l'ordre de lecture.
        prompt_tokens: total des tokens d'entrée facturés.
        output_tokens: total des tokens de sortie (réponse) facturés.
        thinking_tokens: total des tokens de réflexion facturés (modèles « thinking »).
    """

    series_title: str = ""
    episode_title: str = ""
    source_url: str = ""
    model: str
    language: str
    n_panels: int = Field(ge=0)
    n_batches: int = Field(ge=0, default=0)
    scenes: list[Scene]
    beats: list[Beat] = Field(default_factory=list)
    prompt_tokens: int = Field(ge=0, default=0)
    output_tokens: int = Field(ge=0, default=0)
    thinking_tokens: int = Field(ge=0, default=0)

    @property
    def n_scenes(self) -> int:
        """Nombre de scènes (remplissage inclus)."""
        return len(self.scenes)

    @property
    def n_key_panels(self) -> int:
        """Nombre de cases distinctes réellement montées."""
        return len(self.covered_panel_ids())

    @property
    def script_words(self) -> int:
        """Nombre de mots du script narré (scènes narratives)."""
        return sum(len(scene.narration.split()) for scene in self.story_scenes())

    @property
    def n_filler(self) -> int:
        """Nombre de scènes de remplissage (``is_filler``)."""
        return sum(1 for scene in self.scenes if scene.is_filler)

    def story_scenes(self) -> list[Scene]:
        """Scènes narratives à monter (sans le remplissage), ordre de lecture conservé."""
        return [scene for scene in self.scenes if not scene.is_filler]

    def covered_panel_ids(self) -> list[int]:
        """Numéros de cases couverts par au moins une scène, triés."""
        return sorted({pid for scene in self.scenes for pid in scene.panel_ids})


__all__ = [
    "Emotion",
    "EMOTIONS",
    "SceneDraft",
    "SceneBatch",
    "Scene",
    "BeatDraft",
    "BeatBatch",
    "Beat",
    "ParagraphDraft",
    "ScriptDraft",
    "RecapParagraph",
    "RecapDraft",
    "KeyframeChoice",
    "KeyframeBatch",
    "ChapterAnalysis",
]
