"""Schémas Pydantic des segments audio produits par le moteur Kokoro TTS (Module 4).

Chaque scène narrée donne un fichier WAV (:class:`SceneAudio`) dont la durée
exacte, silence de fin inclus, pilote la timeline du montage CapCut. Le
:class:`VoiceoverManifest` regroupe tous les segments d'un chapitre et est écrit
en ``voiceover.json`` à côté des WAV.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class SceneAudio(BaseModel):
    """Segment audio d'une scène.

    Attributes:
        scene_index: ``Scene.index`` de la scène narrée.
        file: nom du fichier WAV, relatif au dossier du manifeste.
        duration_s: durée totale en secondes (parole + silence de fin), calculée
            sur le nombre exact d'échantillons.
        speech_s: durée de la parole seule (sans le silence de fin).
        sample_rate: fréquence d'échantillonnage (24 000 Hz pour Kokoro).
        text: texte réellement synthétisé (après remplacements phonétiques).
        emotion: ton de la scène (repris de l'analyse).
        is_filler: scène de remplissage (normalement exclue de la synthèse).
    """

    scene_index: int = Field(ge=0)
    file: str
    duration_s: float = Field(ge=0)
    speech_s: float = Field(ge=0)
    sample_rate: int = Field(gt=0)
    text: str
    emotion: str = "neutral"
    is_filler: bool = False


class VoiceoverManifest(BaseModel):
    """Ensemble des segments audio d'un chapitre.

    Attributes:
        language: code de langue de la narration (``"en"``...).
        lang_code: code de pipeline Kokoro (``"a"`` = anglais américain...).
        voice: nom de la voix Kokoro (``"af_heart"``...).
        speed: vitesse de lecture.
        padding_s: silence ajouté à la fin de chaque segment.
        sentence_gap_s: silence inséré entre les phrases d'un segment.
        sample_rate: fréquence d'échantillonnage des WAV.
        model: identifiant du modèle Kokoro.
        items: segments dans l'ordre des scènes.
        total_duration_s: somme des durées des segments.
        full_file: WAV de la voix off complète (concaténation), s'il a été écrit.
    """

    language: str
    lang_code: str
    voice: str
    speed: float = Field(gt=0)
    padding_s: float = Field(ge=0)
    sentence_gap_s: float = Field(ge=0, default=0.0)
    sample_rate: int = Field(gt=0)
    model: str = "hexgrad/Kokoro-82M"
    items: list[SceneAudio]
    total_duration_s: float = Field(ge=0)
    full_file: str | None = None

    @property
    def n_items(self) -> int:
        """Nombre de segments."""
        return len(self.items)

    def by_scene(self) -> dict[int, SceneAudio]:
        """Index ``scene_index -> SceneAudio``."""
        return {item.scene_index: item for item in self.items}


__all__ = ["SceneAudio", "VoiceoverManifest"]
