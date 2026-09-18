"""Étape 1 de la miniature : choisir la scène, l'accroche et la flèche.

L'analyse travaille sur le **script déjà écrit** (``scenes.json``), pas sur les images :
aucune case n'est renvoyée à Gemini, donc une miniature ne coûte qu'un appel de texte,
négligeable devant l'analyse du chapitre.

Le transport passe par le :class:`~src.utils.gemini_manager.GeminiManager` partagé : la
rotation de clés, la cascade de modèles et la limite de requêtes par minute s'appliquent
donc aussi aux miniatures.
"""

from __future__ import annotations

import logging
import re
from typing import Any

from google.genai import types
from pydantic import ValidationError

from src.models.scene import ChapterAnalysis
from src.models.thumbnail import MAX_HOOK_CHARS, MAX_HOOK_WORDS, ThumbnailBrief, ThumbnailDraft
from src.utils.gemini_manager import GeminiManager

logger = logging.getLogger(__name__)

#: Température : un peu plus haute que l'analyse du chapitre, l'accroche gagnant à être vive.
DEFAULT_TEMPERATURE: float = 0.7
DEFAULT_TIMEOUT_MS: int = 60_000
#: Nombre de paragraphes du script envoyés en contexte (début + fin : l'accroche et le
#: cliffhanger sont les deux endroits où se trouve le contraste le plus fort).
CONTEXT_PARAGRAPHS: int = 6

SYSTEM_INSTRUCTION: str = """You are the thumbnail designer of a YouTube channel that recaps manhwa chapters.
From the recap script of one chapter, design ONE thumbnail that maximises click-through rate.

Return:
- "scene_description": a single vivid image prompt (25 to 45 words) describing ONE moment with a
  strong VISUAL CONTRAST or IRONY taken from the chapter: the weakest character about to win, a
  calm face in front of a catastrophe, a tiny hero against a huge monster, a smile in a ruin.
  Describe only what is visible: characters, poses, expressions, setting, lighting, colours.
  One subject must clearly dominate the frame. Never mention text, letters, logos or UI.
- "hook_text": 1 to %(max_words)d words, %(max_chars)d characters maximum, upper case, no final
  punctuation. It must provoke curiosity or shock ("WAKE UP", "HE LIED", "TOO LATE"). Never a full
  sentence, never the chapter title, never "EPISODE" or a number alone.
- "arrow_position": "left" or "right" - the side where a yellow arrow will be drawn.
- "subject_position": "left", "center" or "right" - where the main subject sits in your described
  image. The arrow must point AT the subject, so choose arrow_position on the opposite side when
  the subject is on a side, and either side when it is centred.
- "avoid": short comma-separated list of things the illustration must not contain.

Answer only with the requested JSON.""" % {"max_words": MAX_HOOK_WORDS, "max_chars": MAX_HOOK_CHARS}

#: Ponctuation finale et décorations retirées de l'accroche.
_HOOK_STRIP = re.compile(r"^[\s\"'«»\-–—]+|[\s\"'«»\-–—.,;:!?]+$")
_HOOK_INNER = re.compile(r"[^A-Z0-9' ]+")


class ThumbnailAnalysisError(RuntimeError):
    """Le modèle n'a pas produit de consigne de miniature exploitable."""


def build_prompt(analysis: ChapterAnalysis, *, context_paragraphs: int = CONTEXT_PARAGRAPHS) -> str:
    """Contexte envoyé au modèle : titres puis début et fin du script.

    Le milieu du script est volontairement omis : le contraste exploitable se trouve
    presque toujours dans l'accroche ou dans le retournement final.
    """
    scenes = analysis.story_scenes()
    half = max(1, context_paragraphs // 2)
    head, tail = scenes[:half], scenes[-half:] if len(scenes) > half else []
    lines: list[str] = []
    if analysis.series_title or analysis.episode_title:
        lines.append(f"Series: {analysis.series_title} | Chapter: {analysis.episode_title}")
    lines.append(f"The chapter has {len(scenes)} narrated paragraphs. Opening:")
    lines.extend(f"- {scene.narration}" for scene in head)
    if tail:
        lines.append("Ending:")
        lines.extend(f"- {scene.narration}" for scene in tail)
    lines.append("Design the thumbnail now. Answer in JSON.")
    return "\n".join(lines)


def normalize_hook(text: str) -> str:
    """Accroche en majuscules, bornée à :data:`MAX_HOOK_WORDS` mots et :data:`MAX_HOOK_CHARS`.

    Raises:
        ThumbnailAnalysisError: accroche vide après nettoyage.
    """
    cleaned = _HOOK_STRIP.sub("", (text or "").strip()).upper()
    cleaned = _HOOK_INNER.sub(" ", cleaned)
    words = cleaned.split()[:MAX_HOOK_WORDS]
    hook = " ".join(words)
    while len(hook) > MAX_HOOK_CHARS and len(words) > 1:
        words.pop()
        hook = " ".join(words)
    hook = hook[:MAX_HOOK_CHARS].strip()
    if not hook:
        raise ThumbnailAnalysisError("Accroche vide apres nettoyage")
    return hook


def normalize_brief(draft: ThumbnailDraft, *, source_url: str = "", model: str = "") -> ThumbnailBrief:
    """Valide et normalise la réponse du modèle.

    La flèche est replacée **à l'opposé du sujet** si le modèle l'a posée du même côté :
    une flèche qui recouvre le personnage ne désigne plus rien.
    """
    subject = draft.subject_position.strip().lower()
    if subject not in ("left", "center", "right"):
        subject = "center"
    arrow = draft.arrow_position.strip().lower()
    if arrow not in ("left", "right"):
        arrow = "left" if subject == "right" else "right"
    if subject == arrow:
        arrow = "left" if subject == "right" else "right"
        logger.debug("Fleche replacee a l'oppose du sujet (%s)", arrow)
    description = " ".join((draft.scene_description or "").split())
    if not description:
        raise ThumbnailAnalysisError("Description de scene vide")
    return ThumbnailBrief(
        scene_description=description,
        hook_text=normalize_hook(draft.hook_text),
        arrow_position=arrow,
        subject_position=subject,
        avoid=" ".join((draft.avoid or "").split()),
        source_url=source_url,
        model=model,
    )


def parse_brief(response: Any, *, source_url: str = "", model: str = "") -> ThumbnailBrief:
    """Transforme la réponse du SDK en :class:`ThumbnailBrief`.

    Raises:
        ThumbnailAnalysisError: réponse vide, JSON invalide ou champs inexploitables.
    """
    parsed = getattr(response, "parsed", None)
    if isinstance(parsed, ThumbnailDraft):
        return normalize_brief(parsed, source_url=source_url, model=model)
    text = getattr(response, "text", None)
    if not text:
        raise ThumbnailAnalysisError("Reponse vide du modele")
    try:
        return normalize_brief(ThumbnailDraft.model_validate_json(text), source_url=source_url, model=model)
    except ValidationError as exc:
        raise ThumbnailAnalysisError(f"Reponse invalide : {exc}") from exc


class ThumbnailAnalyzer:
    """Produit la consigne de miniature d'un chapitre (un appel de texte).

    Args:
        manager: gestionnaire Gemini partagé ; construit depuis l'environnement si absent.
        temperature: température d'échantillonnage.
        context_paragraphs: paragraphes du script envoyés en contexte.
    """

    def __init__(
        self,
        manager: GeminiManager | None = None,
        *,
        temperature: float = DEFAULT_TEMPERATURE,
        context_paragraphs: int = CONTEXT_PARAGRAPHS,
    ) -> None:
        self._manager = manager
        self.temperature = temperature
        self.context_paragraphs = context_paragraphs

    @property
    def manager(self) -> GeminiManager:
        if self._manager is None:
            self._manager = GeminiManager()
        return self._manager

    def config(self) -> types.GenerateContentConfig:
        return types.GenerateContentConfig(
            system_instruction=SYSTEM_INSTRUCTION,
            temperature=self.temperature,
            response_mime_type="application/json",
            response_schema=ThumbnailDraft,
            http_options=types.HttpOptions(timeout=DEFAULT_TIMEOUT_MS),
        )

    def design(self, analysis: ChapterAnalysis) -> ThumbnailBrief:
        """Consigne de miniature pour ce chapitre.

        Raises:
            ThumbnailAnalysisError: le modèle n'a rien d'exploitable renvoyé.
        """
        prompt = build_prompt(analysis, context_paragraphs=self.context_paragraphs)
        response = self.manager.generate(
            [types.Part.from_text(text=prompt)], self.config(), label="miniature",
        )
        brief = parse_brief(response, source_url=analysis.source_url, model=self.manager.current_model)
        logger.info(
            "Miniature : accroche '%s', fleche a %s, sujet %s", brief.hook_text, brief.arrow_position, brief.subject_position
        )
        return brief


__all__ = [
    "SYSTEM_INSTRUCTION",
    "CONTEXT_PARAGRAPHS",
    "ThumbnailAnalysisError",
    "ThumbnailAnalyzer",
    "build_prompt",
    "normalize_hook",
    "normalize_brief",
    "parse_brief",
]
