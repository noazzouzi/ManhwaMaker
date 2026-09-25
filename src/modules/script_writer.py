"""Script du chapitre écrit par Claude (CLI local, abonnement) : prompt narrateur, cases en planches.

Rédacteur par défaut depuis le 25/09/2026. Banc d'essai sur deux chapitres, même prompt
et mêmes images, notés à l'aveugle par un juge qui lit le chapitre : Opus 8,5/10,
Sonnet 6,5 (répliques mises dans la mauvaise bouche), Gemini 5, Haiku 2,5 (chapitre
survolé). Opus est aussi le plus rapide des trois Claude. Opus 5.5 en production : 81 s et
1,04 $ au tarif API pour un chapitre de 182 cases, décomptés de l'abonnement. Gemini reste le repli
quand Claude échoue (voir :func:`src.pipeline.stage_analyze`).

Entrée : toutes les cases de lecture, empilées dans l'ordre en planches d'au plus
:data:`MAX_IMAGE_PX` pixels (au-delà, Claude réduit l'image), chaque case surmontée d'un
bandeau « Panel N » ; au plus :data:`MAX_IMAGES` images (Claude en accepte 100 par
requête). Sortie : paragraphes (texte, émotion, cases clés, cases d'impact) et fiche des
personnages, validés par ``--json-schema``, puis les mêmes garde-fous que le mode Gemini
(cases valides et jamais réutilisées, ordre de lecture, appel à l'abonnement).

Le prompt (:data:`PROMPT_FILE`) fait parler un conteur qui ne brise jamais le 4e mur :
ouverture « Our journey begins » au premier chapitre, « Our hero » ensuite.
"""

from __future__ import annotations

import logging
import math
import time
from collections.abc import Sequence
from pathlib import Path

import cv2
import numpy as np
from pydantic import BaseModel

from src.models.chapter import ChapterMeta
from src.models.panel import Panel
from src.models.scene import EMOTIONS, ChapterAnalysis, CharacterCard, Emotion, RecapParagraph, Scene
from src.modules.analyzer import (
    DEFAULT_LANGUAGE,
    MAX_KEY_PANELS_PER_PARAGRAPH,
    InvalidResponseError,
    _chapter_line,
    character_sheet,
    enforce_script_conventions,
    language_name,
    normalize_recap,
)
from src.modules.toonsplit.ai import ClaudeCliJson, JsonClient

logger = logging.getLogger(__name__)

__all__ = [
    "DEFAULT_MODEL", "DEFAULT_EFFORT", "PROMPT_FILE", "MAX_IMAGE_PX", "MAX_IMAGES", "OPENING_FIRST", "OPENING_NEXT", "ScriptParagraph",
    "ChapterScript", "composites", "is_first_chapter", "system_prompt", "ClaudeScriptWriter",
]

#: Opus 5.5 (CLI Claude Code 2.1.282 ou plus récent ; le 2.1.270 le refuse). ID explicite :
#: l'alias ``opus`` change de modèle quand le CLI est mis à jour.
DEFAULT_MODEL = "claude-opus-5-5"
PROMPT_FILE = Path(__file__).with_name("script_prompt.md")
#: Pixels par planche : au-delà, Claude réduit l'image (1,15 Mpx).
MAX_IMAGE_PX = 1_100_000
#: Planches par requête (plafond de Claude : 100 images).
MAX_IMAGES = 95
LABEL_H = 30
JPEG_QUALITY = 85
#: Un chapitre dense prend 2 à 4 min ; au-delà de 25 min, le CLI est considéré bloqué.
TIMEOUT_S = 1500.0
#: Effort d'Opus 5.5 : sa réflexion est toujours active, l'effort en règle la quantité (donc la
#: durée et le coût). Fixé ici plutôt que laissé au défaut du CLI, que rien ne garantit. Valeur à
#: confirmer au banc d'essai : ``run_bench.py <chapitre> --models opus@low,opus@medium,opus@high --judge``.
DEFAULT_EFFORT = "medium"

OPENING_FIRST = ('This is the very first chapter of the story. Your first words are exactly "Our journey begins", '
                 "followed at once by the most gripping fact of the chapter: a place, a danger, a loss. Within the "
                 "first forty words the listener knows who our hero is and what he stands to lose.")
OPENING_NEXT = ('This chapter continues the story. Your first words are "Our hero" or "When we last left our hero", '
                "followed at once by where the previous chapter ended and the new danger. Within the first forty "
                "words the listener knows what is at stake now.")


class ScriptParagraph(BaseModel):
    text: str
    emotion: Emotion
    key_panel_ids: list[int]
    action_heavy_ids: list[int]


class ChapterScript(BaseModel):
    """Schéma de réponse : le script complet et la fiche des personnages du chapitre."""

    paragraphs: list[ScriptParagraph]
    characters: list[CharacterCard]


def composites(
    panels: Sequence[Panel], *, max_px: int = MAX_IMAGE_PX, max_images: int = MAX_IMAGES,
) -> list[tuple[list[int], np.ndarray]]:
    """Cases empilées en planches (BGR) : ``[(numéros des cases, image)]``, dans l'ordre de lecture.

    Largeur commune choisie pour que tout tienne en ``max_images`` planches ; une case trop
    haute pour une planche est réduite. Chaque case est précédée d'un bandeau « Panel N ».
    """
    total = sum(p.width * (p.height + LABEL_H) for p in panels)
    widest = max(p.width for p in panels)
    width = min(widest, int(math.sqrt(max_px * max_images * 0.92 / total) * widest))
    while True:
        max_h = max_px // width
        sheets: list[tuple[list[int], np.ndarray]] = []
        current: list[np.ndarray] = []
        ids: list[int] = []
        used = 0
        for p in panels:
            w, h = width, round(p.height * width / p.width)
            if h + LABEL_H > max_h:  # case très haute : réduite pour tenir dans une planche
                h = max_h - LABEL_H
                w = round(p.width * h / p.height)
            img = cv2.resize(np.ascontiguousarray(p.image[:, :, ::-1]), (w, h), interpolation=cv2.INTER_AREA)
            if img.shape[1] < width:
                img = cv2.copyMakeBorder(img, 0, 0, 0, width - img.shape[1], cv2.BORDER_CONSTANT, value=(255, 255, 255))
            label = np.zeros((LABEL_H, width, 3), np.uint8)
            part = {"top": " (top part, continues below)", "middle": " (middle part)", "bottom": " (bottom part)"}.get(p.part or "", "")
            cv2.putText(label, f"Panel {p.index}{part}", (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
            block = np.vstack([label, img])
            if current and used + block.shape[0] > max_h:
                sheets.append((ids, np.vstack(current)))
                current, ids, used = [], [], 0
            current.append(block)
            ids.append(p.index)
            used += block.shape[0]
        if current:
            sheets.append((ids, np.vstack(current)))
        if len(sheets) <= max_images:
            return sheets
        width = int(width * 0.95)


def is_first_chapter(meta: ChapterMeta | None, known_characters: Sequence[CharacterCard], previous_tail: str) -> bool:
    """Premier chapitre de l'histoire : aucune mémoire de série et numéro 1 (ou inconnu)."""
    episode = meta.episode_no if meta is not None and meta.episode_no is not None else 1
    return not previous_tail and not known_characters and episode <= 1


def system_prompt(
    meta: ChapterMeta | None, *, language: str = DEFAULT_LANGUAGE, known_characters: Sequence[CharacterCard] = (),
    previous_tail: str = "", max_key: int = MAX_KEY_PANELS_PER_PARAGRAPH,
) -> str:
    """Prompt du conteur (:data:`PROMPT_FILE`), avec l'ouverture et la mémoire de série du chapitre."""
    first = is_first_chapter(meta, known_characters, previous_tail)
    return PROMPT_FILE.read_text(encoding="utf-8").format(
        language=language_name(language), opening_rule=OPENING_FIRST if first else OPENING_NEXT,
        chapter_line=_chapter_line(meta), character_sheet=character_sheet(known_characters),
        previous_tail=previous_tail or "nothing - this is where the story starts.", max_key=max_key,
        emotions=", ".join(EMOTIONS),
    )


def _images(panels: Sequence[Panel]) -> list[tuple[str, bytes]]:
    sheets = composites(panels)
    images = []
    for k, (ids, img) in enumerate(sheets):
        ok, data = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
        if not ok:
            raise ValueError(f"encodage JPEG impossible (planche {k})")
        images.append((f"Image {k + 1}/{len(sheets)}: panels {ids[0]} to {ids[-1]}", data.tobytes()))
    return images


def _recap_paragraphs(script: ChapterScript) -> list[RecapParagraph]:
    return [RecapParagraph(text=p.text, emotion=p.emotion, key_panel_ids=p.key_panel_ids, action_heavy_ids=p.action_heavy_ids)
            for p in script.paragraphs]


class ClaudeScriptWriter:
    """Écrit le script d'un chapitre en un appel au CLI Claude (voir le module).

    Args:
        model: modèle Claude (ID explicite, voir :data:`DEFAULT_MODEL`).
        effort: effort de réflexion (``low`` à ``max``, voir :data:`DEFAULT_EFFORT`).
        language: langue du script.
        max_key_panels: cases clés maximales par paragraphe.
        cta_text: appel à l'abonnement (``None`` : celui de la langue ; ``""`` : aucun).
        known_characters, previous_tail: mémoire de série (chapitres précédents).
        client: transport JSON (tests) ; défaut : :class:`ClaudeCliJson`.
    """

    def __init__(
        self, *, model: str = DEFAULT_MODEL, effort: str | None = DEFAULT_EFFORT, language: str = DEFAULT_LANGUAGE,
        max_key_panels: int = MAX_KEY_PANELS_PER_PARAGRAPH, cta_text: str | None = None,
        known_characters: Sequence[CharacterCard] = (), previous_tail: str = "",
        client: JsonClient | None = None, retries: int = 1, timeout_s: float = TIMEOUT_S,
    ) -> None:
        self.model = model
        self.language = language
        self.max_key_panels = max_key_panels
        self.cta_text = cta_text
        self.known_characters = list(known_characters)
        self.previous_tail = previous_tail
        self.client = client or ClaudeCliJson(model=model, effort=effort, retries=retries, timeout_s=timeout_s)
        #: Secondes passées dans l'appel (même en cas d'échec).
        self.api_seconds = 0.0

    @property
    def cost_usd(self) -> float:
        return self.client.cost_usd

    def analyze(self, panels: Sequence[Panel], meta: ChapterMeta | None = None) -> ChapterAnalysis:
        """Script et cases clés du chapitre.

        Raises:
            ValueError: aucune case.
            AiError: CLI introuvable, limite d'usage atteinte, réponse invalide après les essais.
        """
        panels = sorted(panels, key=lambda p: p.index)
        if not panels:
            raise ValueError("aucune case a analyser")
        system = system_prompt(meta, language=self.language, known_characters=self.known_characters,
                               previous_tail=self.previous_tail, max_key=self.max_key_panels)
        images = _images(panels)
        text = (f"{_chapter_line(meta)}\nThe chapter has {len(panels)} panels, numbered {panels[0].index} to "
                f"{panels[-1].index}, packed into {len(images)} images in reading order. Write the script.")

        def check(script: ChapterScript) -> tuple[ChapterScript, list[str]]:
            # Numérotation fausse (cases inconnues en masse) : on redemande plutôt que de monter au hasard.
            try:
                normalize_recap(_recap_paragraphs(script), panels, max_key=self.max_key_panels)
            except InvalidResponseError as exc:
                return script, [str(exc)]
            return script, []

        logger.info("Script Claude (%s) : %d case(s) en %d planche(s)", self.model, len(panels), len(images))
        started = time.perf_counter()
        try:
            script = self.client.ask(system, text, images, ChapterScript, label=f"script {self.model}", check=check)
        finally:
            self.api_seconds += time.perf_counter() - started
        entries = normalize_recap(_recap_paragraphs(script), panels, max_key=self.max_key_panels)
        paragraphs = enforce_script_conventions([draft for draft, _, _ in entries], language=self.language, cta_text=self.cta_text)
        scenes = [
            Scene(index=i, panel_ids=keys, narration=paragraph.text, emotion=paragraph.emotion,
                  is_filler=False, beat_ids=[], action_heavy_ids=heavy)
            for i, (paragraph, (_, keys, heavy)) in enumerate(zip(paragraphs, entries))
        ]
        characters: dict[str, CharacterCard] = {}
        for card in script.characters:
            if card.name.strip():
                characters[card.name.strip().casefold()] = card
        logger.info(
            "Script Claude : %d paragraphe(s), %d mots, %d case(s) cle(s) sur %d, %.0fs, %.2f $ (tarif API, abonnement)",
            len(scenes), sum(len(s.narration.split()) for s in scenes), sum(len(s.panel_ids) for s in scenes),
            len(panels), self.api_seconds, self.cost_usd,
        )
        return ChapterAnalysis(
            series_title=meta.series_title if meta else "", episode_title=meta.episode_title if meta else "",
            source_url=meta.url if meta else "", model=self.model, language=self.language, n_panels=len(panels),
            n_batches=self.client.n_calls, scenes=scenes, beats=[], characters=list(characters.values()),
        )
