"""Module 3 — Gemini VLM Analyzer : des cases découpées au script narré, en deux étapes.

Étape 1 — le script global du chapitre :

1a. **Beats** : les cases sont envoyées par lots de :data:`DEFAULT_BATCH_SIZE`
    images (jamais plus de :data:`MAX_BATCH_SIZE`, garde-fou n°2 du PRD) ; pour
    chaque lot Gemini extrait des *beats* factuels et ordonnés (cases couvertes,
    résumé, personnages, répliques, remplissage), avec un contexte glissant des
    derniers beats du lot précédent. Chaque case du lot appartient à exactement
    un beat (:func:`normalize_partition`).
1b. **Script** : à partir de tous les beats (texte seul, donc sans limite de
    lot), Gemini rédige le script complet du récap : storytelling purement
    rétrospectif, accroche, présent, 3ᵉ personne, **sans aucune formule visuelle**
    (« dans cette case », « on voit », « ici »...). Les formules interdites sont
    détectées (:func:`find_forbidden`), le script est régénéré une fois avec un
    rappel, puis nettoyé en dernier recours (:func:`scrub_forbidden`). Chaque
    paragraphe couvre des beats consécutifs ; les beats de remplissage sont ignorés.

Étape 2 — les **cases clés** : pour chaque paragraphe, Gemini choisit parmi les
cases de ses beats (images) 1 à :data:`MAX_KEY_PANELS_PER_PARAGRAPH` cases clés,
les plus fortes visuellement ; les cases de transition (texte seul, bruitages,
fragments) ne sont pas retenues et ne seront pas montées. Repli déterministe
sur la plus grande case candidate si le modèle n'en retient aucune.

Résultat : un :class:`~src.models.scene.ChapterAnalysis` dont chaque
:class:`~src.models.scene.Scene` porte un paragraphe du script et ses cases clés,
plus la liste des beats pour la traçabilité.

Appels Gemini : par défaut ils passent par un
:class:`~src.utils.gemini_manager.GeminiManager` (rotation de clés
``GEMINI_API_KEYS``, backoff 2/4/8/16 s sur 429, cascade de modèles, limite RPM
globale) partagé entre tous les chapitres d'un lot ; un client injecté
(``GeminiAnalyzer(client=...)``, tests) garde la boucle de réessai locale. Les
réponses invalides sont réessayées ; les erreurs définitives (4xx, prompt bloqué
par la sécurité) lèvent :class:`AnalyzerError`. Un délai forcé de
:data:`BATCH_DELAY_S` sépare deux envois d'un même chapitre pour étaler les
jetons (TPM). Narration en anglais par défaut.
"""

from __future__ import annotations

import json
import logging
import math
import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from bisect import bisect_right
from collections.abc import Callable, Mapping, Sequence
from io import BytesIO
from pathlib import Path
from typing import Any

from google.genai import errors, types
from PIL import Image
from pydantic import BaseModel, ValidationError

from src.models.chapter import ChapterMeta
from src.models.panel import Panel
from src.models.scene import (
    EMOTIONS,
    Beat,
    BeatBatch,
    BeatDraft,
    ChapterAnalysis,
    CharacterCard,
    KeyframeBatch,
    KeyframeChoice,
    ParagraphDraft,
    RecapDraft,
    RecapParagraph,
    Scene,
    SceneBatch,
    SceneDraft,
    ScriptDraft,
)
from src.utils import gemini_manager as gm
from src.utils.config import DEFAULT_NARRATION_LANGUAGE, gemini_key_hint, load_gemini_api_key
from src.utils.gemini_manager import GeminiManager, is_daily_quota_error, suggested_retry_delay
from src.utils.image_utils import to_pil

logger = logging.getLogger(__name__)

# --- Paramètres par défaut (cf. PRD, Module 3) ---------------------------------
#: Modèle par défaut : Gemini 3.5 Flash (validé le 2026-09-09 ; 2.5 Flash reste utilisable
#: mais son quota gratuit journalier est très bas). Surchargeable par :data:`MODEL_ENV_VAR`.
DEFAULT_MODEL: str = "gemini-3.5-flash"
MODEL_ENV_VAR: str = "GEMINI_MODEL"
#: Nombre d'images par lot (le PRD impose 10 à 15 images max par requête).
DEFAULT_BATCH_SIZE: int = 12
MAX_BATCH_SIZE: int = 15
#: Langue de la narration (code ISO 639-1) : anglais par défaut (convention du projet).
DEFAULT_LANGUAGE: str = DEFAULT_NARRATION_LANGUAGE
DEFAULT_TEMPERATURE: float = 0.4
#: Largeur maximale (px) d'une image envoyée ; au-delà la case est réduite (ratio conservé).
DEFAULT_MAX_IMAGE_WIDTH: int = 1024
#: Hauteur maximale (px) d'une tranche d'image ; une case plus haute est découpée
#: en tranches consécutives (de haut en bas) envoyées sous le même numéro.
DEFAULT_MAX_SLICE_HEIGHT: int = 2048
MAX_SLICES_PER_PANEL: int = 4
JPEG_QUALITY: int = 88
#: Nombre de beats du lot précédent rappelés en contexte (étape 1a).
CONTEXT_SCENES: int = 3
CONTEXT_BEATS: int = CONTEXT_SCENES
#: Nombre maximal de cases clés retenues par paragraphe (étape 2).
MAX_KEY_PANELS_PER_PARAGRAPH: int = 4
#: Nombre maximal de cases candidates (les plus grandes) envoyées par paragraphe à
#: l'étape 2 : limite les images par requête et le nombre d'appels.
MAX_CANDIDATES_PER_PARAGRAPH: int = 8
#: Vignettes de l'étape 2 (choix des cases clés) : une résolution réduite suffit pour
#: juger l'intérêt visuel et divise les tokens image par trois environ.
KEYFRAME_IMAGE_WIDTH: int = 640
KEYFRAME_SLICE_HEIGHT: int = 1280
KEYFRAME_MAX_SLICES: int = 2
#: Étendue maximale, en identifiants, d'une unité de poids 3 (le sommet du chapitre).
#: Au-delà, le modèle fond plusieurs moments forts dans un seul paragraphe et le pic ne
#: s'entend plus. Ce n'est **pas** un budget de mots : la longueur du script n'est plus
#: pilotée du tout (cf. :data:`MASTER_PROMPT_TEMPLATE`).
PEAK_MAX_SPAN: int = 4
DEFAULT_MAX_RETRIES: int = 3
DEFAULT_BACKOFF: float = 2.0
MAX_BACKOFF: float = 60.0
RATE_LIMIT_MIN_DELAY: float = 10.0
DEFAULT_TIMEOUT_MS: int = 120_000
#: Plafond de jetons de sortie. Sans plafond explicite, une réponse longue est coupée en
#: plein JSON alors que ``finish_reason`` annonce ``STOP`` : le chapitre échoue en
#: « JSON invalide » sans qu'on sache pourquoi. Le plus gros appel observé a produit
#: 3 485 jetons de sortie + 8 123 de réflexion ; 32 768 laisse une marge confortable.
DEFAULT_MAX_OUTPUT_TOKENS: int = 32_768
#: Réessais accordés à une réponse **tronquée**. Une fois le plafond posé, une troncature
#: signale un emballement que le même prompt reproduira : insister brûle du quota.
TRUNCATION_RETRIES: int = 1
#: Nombre de chiffres au-delà duquel un entier de la réponse est tenu pour un emballement
#: du modèle. 18 chiffres couvrent tout entier 64 bits : un identifiant de case en fait 3.
MAX_INT_DIGITS: int = 18
#: Valeur substituée à un entier aberrant : hors de toute plage d'identifiants valide,
#: elle est donc écartée par la validation en aval sans traitement particulier.
RUNAWAY_INT: int = -1
#: Délai forcé (secondes) entre deux envois d'un même chapitre (lots d'images, script,
#: groupes de cases clés) pour étaler la consommation de jetons par minute (TPM).
#: Redondant avec le limiteur RPM du :class:`~src.utils.gemini_manager.GeminiManager` :
#: le mettre à 0 (``--gemini-batch-delay 0``) accélère d'autant quand ``--max-gemini-rpm``
#: est réglé sur la limite réelle du compte.
BATCH_DELAY_S: float = 2.5
#: Appels « cases clés » (étape 2) menés en parallèle : ces groupes sont **indépendants**
#: (la normalisation qui les départage reste séquentielle et déterministe).
DEFAULT_KEYFRAME_WORKERS: int = 4
#: Poids maximal des images inline d'une requête Gemini : la limite dure est de 20 Mo,
#: on garde une marge pour le texte et l'encodage base64 du transport.
MAX_INLINE_PAYLOAD_BYTES: int = 16 * 1024 * 1024
#: Paliers ``(largeur, qualité)`` essayés dans l'ordre pour faire tenir un chapitre en
#: **une seule requête**. Basculer en mode deux étapes coûte vingt fois plus d'appels et,
#: surtout, prive le modèle des images : il ne réécrit alors que des résumés déjà aplatis.
#: Comprimer un peu plus vaut mieux.
#:
#: Mesuré sur un chapitre de 188 cases de 900 px de large (24,6 Mo au premier palier) :
#: la qualité est le vrai levier, la largeur ne mord qu'en dessous de la taille native.
#: La lisibilité des bulles a été vérifiée à l'œil jusqu'à 768 px / qualité 72 - le texte
#: des webtoons est grand et très contrasté, il survit largement à la compression.
INLINE_PAYLOAD_LADDER: tuple[tuple[int, int], ...] = (
    (DEFAULT_MAX_IMAGE_WIDTH, JPEG_QUALITY),
    (DEFAULT_MAX_IMAGE_WIDTH, 75),
    (DEFAULT_MAX_IMAGE_WIDTH, 65),
    (832, 65),
    (768, 62),
    (640, 60),
)
LANGUAGE_NAMES: dict[str, str] = {
    "fr": "francais",
    "en": "English",
    "es": "espanol",
    "de": "Deutsch",
    "it": "italiano",
    "pt": "portugues",
}

_EMOTION_ALIASES: dict[str, str] = {
    "joy": "happy", "joyful": "happy", "cheerful": "happy",
    "sadness": "sad", "melancholy": "sad", "grief": "sad",
    "anger": "tension", "angry": "tension",
    "suspense": "mystery", "mysterious": "mystery",
    "comedy": "humor", "funny": "humor", "humour": "humor",
    "horror": "fear", "scary": "fear", "dread": "fear",
    "peaceful": "calm", "serene": "calm", "quiet": "calm",
    "fight": "action", "battle": "action", "combat": "action",
    "love": "romance", "romantic": "romance",
    "dramatic": "epic", "heroic": "epic", "triumph": "epic",
}

_FATAL_FINISH_REASONS: frozenset[str] = frozenset(
    {"SAFETY", "RECITATION", "BLOCKLIST", "PROHIBITED_CONTENT", "SPII", "IMAGE_SAFETY", "IMAGE_PROHIBITED_CONTENT"}
)
_ZERO_WIDTH: dict[int, None] = dict.fromkeys((0x200B, 0x200C, 0x200D, 0xFEFF), None)

#: Formules visuelles / descriptives interdites dans le script (anglais et francais).
FORBIDDEN_PATTERNS: tuple[str, ...] = (
    r"\bin (?:this|the|that|our) (?:panel|panels|scene|image|images|frame|picture|illustration|shot|drawing|page)\b",
    r"\bwe (?:can |then |now |also |first )?(?:see|watch|witness|observe|are shown|get to see|find)\b",
    r"\bthe (?:panel|image|frame|picture|illustration|artwork|drawing|scene|page|strip) (?:shows|depicts|reveals|displays|features|captures|presents|opens|closes)\b",
    r"\b(?:is|are) (?:shown|depicted|displayed|pictured|illustrated|visible)\b",
    r"\bthe camera\b",
    r"\bclose-?up\b",
    r"\bzoom(?:s|ed|ing)? (?:in|out)\b",
    r"(?:^|[.!?]\s+)here[,:]?\s",
    # Le narrateur parle de lui-même ou de son support plutôt que de l'histoire.
    r"\b(?:the narrator|the protagonist|our (?:hero|protagonist)|the main character)\b",
    r"\bthe (?:scene|panel|image|page|shot) (?:unfolds|cuts|shifts|switches|changes)\b",
    r"\bthis (?:chapter|episode) (?:opens|begins|starts|ends|closes|follows|covers)\b",
    # Volontairement sans « page » : « la dernière page du journal de son père » est du récit.
    r"\bthe (?:final|last|opening|first) (?:panel|frame|shot)\b",
    r"\bsur cette (?:case|image|planche|vignette)\b",
    r"\bon (?:voit|apercoit|aperçoit|decouvre|découvre)\b",
    r"(?:^|[.!?]\s+)ici[,:]?\s",
    r"\bl'image (?:montre|revele|révèle)\b",
)
_FORBIDDEN = [re.compile(p, re.IGNORECASE) for p in FORBIDDEN_PATTERNS]

#: Introductions génériques interdites en ouverture de script (première phrase).
GENERIC_INTRO_PATTERNS: tuple[str, ...] = (
    r"^\s*(?:in|during|for) (?:this|today's|the) (?:chapter|episode|video|recap|prologue)\b",
    r"^\s*(?:welcome|hello|hey|hi|greetings)\b",
    r"^\s*today,? (?:we|let's|let us|i|you)\b",
    r"^\s*let'?s (?:dive|jump|get|start|begin|take|go)\b",
    r"^\s*(?:this|the|our) (?:recap|video|chapter|episode|story|prologue) (?:covers|starts|begins|opens|follows|tells|takes)\b",
    r"^\s*(?:get ready|buckle up|without further ado|before we (?:start|begin))\b",
    r"^\s*(?:dans|pour) (?:ce|cet) (?:chapitre|episode|épisode|recap|récap)\b",
    r"^\s*(?:bienvenue|bonjour|salut)\b",
)
_GENERIC_INTRO = [re.compile(p, re.IGNORECASE) for p in GENERIC_INTRO_PATTERNS]
#: Détection d'un appel à l'abonnement écrit par le modèle (retiré puis replacé par le nôtre).
CTA_PATTERN = re.compile(
    r"\b(?:subscribe|subscribing|subscribed|hit the (?:like|bell)|like (?:this|the) video|"
    r"abonne[sz]?-?(?:toi|vous)?|abonnez|abonner|like et abonne)\b",
    re.IGNORECASE,
)
#: Appel à l'abonnement par défaut, par langue (anglais en repli).
CTA_TEXTS: dict[str, str] = {
    "en": "If you're enjoying this recap, subscribe so you never miss the next chapter.",
    "fr": "Si ce recap te plait, abonne-toi pour ne pas rater le prochain chapitre.",
    "es": "Si te esta gustando este resumen, suscribete para no perderte el proximo capitulo.",
    "de": "Wenn dir dieser Recap gefaellt, abonniere den Kanal, um das naechste Kapitel nicht zu verpassen.",
    "it": "Se questo riassunto ti piace, iscriviti per non perdere il prossimo capitolo.",
    "pt": "Se voce esta gostando deste resumo, inscreva-se para nao perder o proximo capitulo.",
}
#: Position relative du paragraphe qui recoit l'appel a l'abonnement (0 = debut, 1 = fin).
CTA_POSITION: float = 0.4
_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(\[])")

_sleep = time.sleep

try:
    import httpx

    _NETWORK_ERRORS: tuple[type[Exception], ...] = (httpx.TransportError,)
except ImportError:  # pragma: no cover
    _NETWORK_ERRORS = ()


class AnalyzerError(RuntimeError):
    """Erreur définitive de l'analyzer (configuration, refus du modèle, quota épuisé...)."""


class InvalidResponseError(AnalyzerError):
    """Réponse du modèle inexploitable (JSON invalide, aucun élément valide) : réessayable."""


class TruncatedResponseError(InvalidResponseError):
    """Réponse coupée en plein JSON (plafond de jetons atteint, ou fin structurellement absente).

    Sous-classe d':class:`InvalidResponseError` pour rester réessayable, mais nommée à part :
    réessayer une troncature à l'identique produit presque toujours la même troncature, donc
    le nombre de tentatives est plafonné par :data:`TRUNCATION_RETRIES`.
    """


class QuotaExhaustedError(AnalyzerError):
    """Quota journalier Gemini épuisé : inutile de réessayer avant le lendemain (ou changer de modèle)."""


class AnalysisCheckpoint:
    """Point de reprise d'une analyse interrompue (quota, réseau) : rien n'est redemandé deux fois.

    Le fichier JSON conserve, au fil des appels réussis, les beats de chaque lot, le script
    et les choix de cases clés de chaque groupe. Il n'est réutilisé que pour la même liste de
    cases, la même taille de lot et la même langue ; le modèle peut changer (c'est le cas
    d'usage : reprendre avec un autre modèle quand le quota du premier est épuisé).

    Args:
        path: fichier de reprise (``None`` = aucune persistance, tout reste en mémoire).
        panel_ids: numéros des cases analysées.
        batch_size: taille des lots.
        language: langue du script.
    """

    def __init__(self, path: str | Path | None, *, panel_ids: Sequence[int], batch_size: int, language: str) -> None:
        self.path = Path(path) if path else None
        self.key = {"panel_ids": list(panel_ids), "batch_size": batch_size, "language": language}
        self.data: dict[str, Any] = {"key": self.key, "beat_batches": [], "paragraphs": None, "keyframe_groups": []}
        self.reused = 0
        if self.path is not None and self.path.is_file():
            try:
                loaded = json.loads(self.path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                loaded = None
            if isinstance(loaded, dict) and loaded.get("key") == self.key:
                self.data = loaded
                logger.info(
                    "Reprise depuis %s : %d lot(s) de beats, script %s, %d groupe(s) de cases cles",
                    self.path, len(loaded.get("beat_batches", [])), "present" if loaded.get("paragraphs") else "absent",
                    len(loaded.get("keyframe_groups", [])),
                )
            elif loaded is not None:
                logger.info("Point de reprise %s ignore (cases, lot ou langue differents)", self.path)

    def save(self) -> None:
        if self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(json.dumps(self.data, ensure_ascii=False, indent=1), encoding="utf-8")

    def clear(self) -> None:
        if self.path is not None and self.path.is_file():
            self.path.unlink()

    def beats_for(self, b_index: int, ids: Sequence[int]) -> list[BeatDraft] | None:
        batches = self.data.get("beat_batches", [])
        if b_index < len(batches) and batches[b_index].get("ids") == list(ids):
            self.reused += 1
            return [BeatDraft.model_validate(d) for d in batches[b_index]["drafts"]]
        return None

    def store_beats(self, b_index: int, ids: Sequence[int], drafts: Sequence[BeatDraft]) -> None:
        batches = self.data.setdefault("beat_batches", [])
        del batches[b_index:]
        batches.append({"ids": list(ids), "drafts": [d.model_dump() for d in drafts]})
        self.save()

    def paragraphs(self) -> list[ParagraphDraft] | None:
        stored = self.data.get("paragraphs")
        if stored:
            self.reused += 1
            return [ParagraphDraft.model_validate(p) for p in stored]
        return None

    def store_paragraphs(self, paragraphs: Sequence[ParagraphDraft]) -> None:
        self.data["paragraphs"] = [p.model_dump() for p in paragraphs]
        self.data["keyframe_groups"] = {}
        self.save()

    def _keyframe_groups(self) -> dict[str, Any]:
        """Groupes de cases clés, indexés par numéro de groupe.

        Le format historique était une liste ordonnée : un groupe en échec faisait perdre
        les suivants, déjà obtenus. Avec un dictionnaire, **chaque appel réussi est
        conservé**, même quand les groupes partent en parallèle et qu'un seul échoue.
        """
        groups = self.data.get("keyframe_groups")
        if isinstance(groups, list):
            groups = {str(i): entry for i, entry in enumerate(groups)}
            self.data["keyframe_groups"] = groups
        elif not isinstance(groups, dict):
            groups = self.data["keyframe_groups"] = {}
        return groups

    def choices_for(self, g_index: int, paragraph_indexes: Sequence[int]) -> list[KeyframeChoice] | None:
        entry = self._keyframe_groups().get(str(g_index))
        if isinstance(entry, dict) and entry.get("paragraphs") == list(paragraph_indexes):
            self.reused += 1
            return [KeyframeChoice.model_validate(c) for c in entry["choices"]]
        return None

    def store_choices(self, g_index: int, paragraph_indexes: Sequence[int], choices: Sequence[KeyframeChoice]) -> None:
        self._keyframe_groups()[str(g_index)] = {
            "paragraphs": list(paragraph_indexes), "choices": [c.model_dump() for c in choices],
        }
        self.save()


QUOTA_HINT: str = (
    "quota journalier Gemini epuise pour ce modele : activer la facturation sur le projet Google AI, "
    "ou relancer avec un autre modele (--model gemini-3.5-flash-lite, quota distinct), ou attendre le lendemain"
)



# --- Prompts systeme ------------------------------------------------------------------
BEATS_SYSTEM_INSTRUCTION: str = """You are a story analyst for a manhwa / webtoon recap channel.
You are shown, in reading order, the numbered panels of one chapter. Extract the ordered STORY
BEATS of this batch: what happens, to whom, with which words.

Rules:
1. Every panel provided belongs to exactly ONE beat. Use only the panel numbers given in this
   batch: never invent one, never skip one. Text-only panels (sound effects, chat messages,
   captions, titles) attach to the neighbouring beat they belong to.
2. A beat is one unit of action or information, usually 1 to 5 consecutive panels.
3. summary: 1 to 2 factual sentences in English, present tense: who does what, where, and what
   changes. State facts of the story, never describe the drawing or the layout.
4. characters: names as written in the dialogue or captions (empty list if unknown).
5. dialogue: up to 3 short important quotes, verbatim (empty list if none).
6. is_filler: true only for panels with no story at all (title card, series logo, credits,
   author or publisher notice, advertising, preorder promotion, "to be continued", social links).
7. Answer only with the requested JSON (an object with the "beats" key)."""

#: Prompt maitre : un seul gabarit pour les deux modes de generation. Entre le mode
#: « une requete » (images) et le mode en deux etapes (beats), seuls changent la nature des
#: identifiants et la source des dialogues ; tout le reste - le plan, la voix, les regles de
#: retention - est commun. Deux gabarits divergeaient silencieusement a chaque retouche.
#:
#: Aucune consigne de longueur : cinq essais mesures ont montre que le nombre de mots rendu
#: par Gemini ne suit pas la cible demandee. Ce qui se pilote, c'est le *poids* de chaque
#: moment ; la duree se regle en aval, au montage.
MASTER_PROMPT_TEMPLATE: str = """You are the writer and narrator of a YouTube channel that recaps manhwa chapters in {language}.
Your viewer has NOT read this series, watches the episodes in order and days apart, and decides
in thirty seconds whether to stay and in the last five whether to click the next one. You are not
a camera and not a summary: you are someone who read the chapter and is telling it to one person.
Your one job is to keep them watching. Answer with ONE JSON object, fields in the order given
under OUTPUT.

========================= INPUT =========================
{input_block}
=========================================================

MEMORY
{chapter_line}
Characters established in earlier episodes: {character_sheet}
Where the last episode ended: {previous_tail}
Use the canonical name of every character on that sheet, every time. A character not on it gets
ONE label you invent and repeat word for word all episode ("the boy with glasses"). Never two
labels for one person, never one name for two people - an ancestor and his descendant are two
people. If a proper name appears in a bubble and you cannot attach it to anyone, attach it in the
same sentence ("Eden, the name he carried before he died") or do not use it.

------------------------------- STEP 1: THE PLAN -------------------------------
Read the whole chapter before writing one word of prose.

ACTS. Split the ids into four contiguous blocks - setup, turn, climax, cliffhanger - covering
every story id, in order, no hole, no overlap. Name the PEAK: the one moment that matters (a
decision, a reveal, the first blow, the last line). It sits in climax or cliffhanger.

UNITS. A unit is not a chunk of text: it is WHAT THE VIEWER IS LOOKING AT while you speak.
When the pictures should change, a new unit starts. Cut the chapter into units in reading
order, starting a new one roughly every two or three story ids - more often when the images
change fast. Too few units and the viewer stares at the same picture while you talk on; that
is the fastest way to lose them. One unit = one paragraph. Units are contiguous and
together cover every story id exactly once - no story id is left out, ever. Every id you write
anywhere must exist in the input: never write an id past the last one you were given, and
never mix the two numbering systems described in the INPUT block. Give each unit a
weight:
  weight 1  travel, scenery, arriving, inventory, small talk, and every repetition of an attempt
            that already failed
  weight 2  a consequence, a reaction that costs something, or a rule of this world the newcomer
            needs. Explaining is never a 1.
  weight 3  a decision, a reveal, a first blow, the peak, the last line

SHAPE. The weight says how much the moment matters, and therefore how much attention it earns.
Never count words, never aim at a length: the script is exactly as long as the story needs.
- A weight-1 unit is brief and may swallow many ids: that is how you compress a
  carriage ride, or four identical failures into one sentence of pattern - "Stone. Blade. A
  shield. Every time the same three words: insufficient level." Never one paragraph per attempt.
- A weight-3 unit is the one you linger on, and covers at most {max_span} ids. A climax is several short
  units in a row, never one long one.
- The climax and the cliffhanger carry more of the script than the setup. If the setup is the
  part you dwelt on, you planned it wrong.
- Unit 0 is weight 3. The last unit of the chapter is weight 3 and stands alone.
- CTA: name in cta_unit_index the unit sitting closest to 38% of the chapter whose emotion will be
  neutral, calm, mystery or humor. That paragraph carries exactly one sentence asking the viewer
  to subscribe, worded freshly this episode, and never as its last sentence. Nowhere else.

------------------------------- STEP 2: THE WRITING -------------------------------
RETENTION IS THE ONLY CONTRACT. Every paragraph must earn the next one. Cover every unit, in
order. Never pad: no adjective piles, no restating what you just said, no event that is not in the
chapter. And never ration yourself either - if a moment deserves to breathe, let it breathe.

VOICE. Present tense, third person, warm, dry, sometimes amused. You know things the hero does not
and you may tease them.
- Show the thing; never announce that someone perceives it. realizes, observes, notices, notes,
  watches, wonders, muses, deduces, contemplates, reflects, considers, ponders: at most twice in
  the whole script. Not "He realizes the boy is Benny's descendant" - "The boy has Benny's eyes.
  Benny, who died two hundred years ago."
- Never label a fact before the story has shown it: no "the true motive", "the fabricated tale",
  "seemingly", "a startling realization". If the guide lies, let him lie; land the truth after.
- Weight and pace are two different dials. Weight-3 paragraph: you linger, with SHORTER sentences, verb
  first, the blow before the reaction, at least one sentence of six words or fewer. Weight-1
  paragraph: you pass through, dry and quick. Never three sentences of similar length in a row.
- One adjective per noun. Banned: massive, powerful, overwhelming, incredible, glowing, sheer,
  utter, colossal. At most one -ly adverb and one figure per paragraph, the figure only if it
  changes the stakes.

DIALOGUE. The best lines of this chapter are already written, in the bubbles. Steal them.
- Quoted VERBATIM, always, with no option to report them indirectly: every punchline, every
  threat, every reveal spoken out loud, and the final line of the chapter. If the chapter ends on
  a joke, the viewer hears the joke.
- Everything else is narrated. At most two quoted sentences per paragraph, at most 25 words per
  quote. If you find yourself transcribing whole exchanges, you have stopped telling the story.
- NORMALISE FOR THE VOICE. Bubbles are printed in capitals and broken by ellipses; a speech
  synthesiser reads this aloud. "HOW IS... YOUR GREAT-GRANDFATHER..." becomes "How is your
  great-grandfather?" Sentence case, one speakable sentence, no internal ellipses, no printed
  sound effects. Trim and join freely; never add a word that was not said.
- Every quote is attributed to a named speaker inside its own sentence: Hugh sneers, "..." Never a
  disembodied shout. Thoughts are quoted in the first person and attributed too, never melted
  into the narration: Eden thinks, "He's actually Benny's great-grandson?"
- One line, once: never announce a line then quote it, never quote it then explain it. Banned
  around a quote: demanding to know, questioning, claiming, proclaiming, stating, noting,
  explaining, taunting, mocking, pointing out.

CHAINING. This is what keeps the viewer. At least one paragraph in four OPENS on a word reacting
to the one before: But / Then / Yet / Except / Worse / Too late / Instead / Still / Nothing. Ask
the viewer a direct question two to four times, at the moments of highest uncertainty, and never
answer it in the same paragraph. End every paragraph on a new fact, a threat or an open door -
never on a verdict about what you just said ("the display of power is absolute"), never on a
", a X." apposition that restates it. Never open on a scene-setting clause ("As the carriage rolls
on,", "Meanwhile,"): the scene is already there.

CLARITY. When the chapter shows a rule of its world - what a dungeon is, why that class is junk,
what the System does - state it plainly, once, inside the unit that shows it. That is why a
newcomer stays. Never cut it for rhythm.

OPEN AND CLOSE. Start inside the action, on a gesture or a quoted line. The first forty words
contain the hero's canonical name and what he stands to lose. Never a greeting, never "In this
chapter", and never a pitch of the series ("Mirror World follows Eden, who seeks revenge..."). The
last paragraph ends on the chapter's final line, quoted, or on a question you do not answer -
never on a recap of what just happened.

NEVER SPEAK THE MEDIUM. Forbidden anywhere: panel, frame, page, image, artwork, illustration, the
scene, close-up, the camera, zoom, we see, is shown, is depicted, unfolds, the screen, the
interface, a notification, a system message, a display, the narrator, the protagonist, our hero.
(The System as a thing inside the story is allowed; its user interface is not.) No headings, no
bullets, no emojis, no capitals for emphasis. Filler ids are narrated by nobody and chosen by
nobody.

------------------------------- OUTPUT -------------------------------
One JSON object, nothing else. STRICT JSON: every property name in double quotes, every string
in double quotes, no trailing comma, no comment. Fields in this order - a paragraph's weight comes before its text:

{{"plan":{{"acts":[{{"name":"setup","first_id":0,"last_id":23}},{{"name":"turn","first_id":24,
"last_id":52}},{{"name":"climax","first_id":53,"last_id":110}},{{"name":"cliffhanger",
"first_id":111,"last_id":124}}],"peak_id":118,"cta_unit_index":7,
"units":[{{"id":0,"weight":3,"covers":[0,1,2,3,4]}}]}},
"paragraphs":[{{"plan_id":0,"weight":3,"text":"...","emotion":"tension","beat_ids":[],
"key_panel_ids":[1,3],"action_heavy_ids":[3]}}],
"characters":[{{"name":"Eden","also_called":["the man who died and came back"],
"who":"The hero, betrayed and killed by Wright Housman, now in a teenager's body."}}]}}

- "emotion": exactly one of {emotions}.
- "key_panel_ids": 1 to {max_key} panel numbers illustrating THIS paragraph, in reading order,
  taken from the ids it covers. Prefer faces, decisive gestures, large readable artwork; skip
  text-only fragments, chat windows, sound-effect panels, empty backgrounds. Never reuse a number
  in two paragraphs, never invent one, never pick a filler panel.
- "action_heavy_ids": the subset showing a decisive impact - a blow landing, an explosion, a
  reveal. Empty for calm, talking or establishing paragraphs.
- "characters": every character this chapter names, so the next episode can reuse the same names."""

#: Bloc INPUT du mode « une requete » : toutes les cases du chapitre partent en images, et les
#: dialogues se lisent dans les bulles.
INPUT_BLOCK_IMAGES: str = """MODE: IMAGES. Every panel of the chapter follows, in reading order, each preceded by its number
("Panel N"): {n_ids} panels, numbered {first_id} to {last_id}. The ids you plan with - acts, peak,
units, "covers" - are PANEL numbers, and so are key_panel_ids. Your only dialogue source is the
text printed in the bubbles: read it, quote it verbatim, normalise its typography. Ignore filler
panels entirely (title cards, credits, ads, author notes, "to be continued", social links). Leave
"beat_ids" empty in every paragraph."""

#: Bloc INPUT du mode en deux etapes : le chapitre arrive en texte (beats deja extraits), et les
#: deux numerotations coexistent - on planifie en beats, on illustre en cases.
INPUT_BLOCK_BEATS: str = """MODE: BEATS. The chapter arrives as an ordered list of beats, one per line:
  Beat N (panels a, b, c): what happens | Characters: ... | Dialogue: "..." / "..."
{n_ids} beats, numbered {first_id} to {last_id}. The ids you plan with - acts, peak, units,
"covers" - are BEAT numbers, repeated in "beat_ids". key_panel_ids are PANEL numbers, chosen among
the panels listed on the beats that paragraph covers. Your only dialogue source is the Dialogue
field: quote it verbatim, and if a beat lists none, invent none. Beats flagged [FILLER] do not
exist: never narrate them, never cite their panels."""

#: Blocs INPUT par mode, pour :func:`master_instruction`.
INPUT_BLOCKS: dict[str, str] = {"images": INPUT_BLOCK_IMAGES, "beats": INPUT_BLOCK_BEATS}

KEYFRAMES_SYSTEM_INSTRUCTION_TEMPLATE: str = """You are the video editor of a manhwa recap channel. For each paragraph of the recap script you
receive its text and its candidate panels (images). Choose the KEY panels that will stay on
screen while the paragraph is narrated.

Rules:
1. Pick 1 to {max_key} panels per paragraph, only among that paragraph's candidates, in reading
   order. Fewer, stronger panels are better than many weak ones.
2. Prefer the most striking, complete and readable artwork: characters' faces and actions,
   decisive moments, establishing shots, large detailed panels.
3. Ignore transition panels without interest: text-only lines, chat messages, sound effects, tiny
   fragments, empty backgrounds - unless a paragraph has nothing else.
4. Never reuse a panel in two paragraphs. Do not invent panel numbers.
5. "action_heavy_ids": among the key panels you picked, list those showing a DECISIVE IMPACT
   (a blow landing, an explosion, a roar, a sudden reveal). They get a fast punch-in zoom in
   the edit. Leave the list empty for calm, talking or establishing panels - be selective.
6. Answer only with the requested JSON (an object with the "choices" key), exactly one entry per
   paragraph, each with "paragraph_index", "key_panel_ids" and "action_heavy_ids"."""


# --- Lots ----------------------------------------------------------------------------
def make_batches(panels: Sequence[Panel], batch_size: int = DEFAULT_BATCH_SIZE) -> list[list[Panel]]:
    """Répartit les cases en lots équilibrés d'au plus ``batch_size`` images.

    Raises:
        ValueError: ``batch_size`` hors de ``[1, MAX_BATCH_SIZE]``.
    """
    if not 1 <= batch_size <= MAX_BATCH_SIZE:
        raise ValueError(
            f"batch_size doit etre entre 1 et {MAX_BATCH_SIZE} (PRD : 10-15 images max), recu {batch_size}"
        )
    panels = list(panels)
    if not panels:
        return []
    n_batches = math.ceil(len(panels) / batch_size)
    base, extra = divmod(len(panels), n_batches)
    batches: list[list[Panel]] = []
    start = 0
    for i in range(n_batches):
        size = base + (1 if i < extra else 0)
        batches.append(panels[start : start + size])
        start += size
    return batches


# --- Images --------------------------------------------------------------------------
def encode_panel_image(
    panel: Panel,
    *,
    max_width: int = DEFAULT_MAX_IMAGE_WIDTH,
    max_slice_height: int = DEFAULT_MAX_SLICE_HEIGHT,
    max_slices: int = MAX_SLICES_PER_PANEL,
    quality: int = JPEG_QUALITY,
) -> list[tuple[bytes, str]]:
    """Encode une case en une ou plusieurs tranches JPEG lisibles par le modèle.

    Raises:
        ValueError: paramètre < 1.
    """
    if max_width < 1 or max_slice_height < 1 or max_slices < 1:
        raise ValueError("max_width, max_slice_height et max_slices doivent etre >= 1")
    img = to_pil(panel.image)
    width, height = img.size
    scale = min(1.0, max_width / width)
    max_total_height = max_slices * max_slice_height
    if height * scale > max_total_height:
        scale = max_total_height / height
    if scale < 1.0:
        new_size = (max(1, round(width * scale)), max(1, round(height * scale)))
        img = img.resize(new_size, Image.Resampling.LANCZOS)
        width, height = img.size
    n_slices = min(max_slices, max(1, math.ceil(height / max_slice_height)))
    slice_height = math.ceil(height / n_slices)
    slices: list[tuple[bytes, str]] = []
    for k in range(n_slices):
        top = k * slice_height
        bottom = min(height, top + slice_height)
        part = img if n_slices == 1 else img.crop((0, top, width, bottom))
        buffer = BytesIO()
        part.save(buffer, format="JPEG", quality=quality, optimize=True)
        slices.append((buffer.getvalue(), "image/jpeg"))
    return slices


# --- Prompt (texte) --------------------------------------------------------------------
def language_name(code: str) -> str:
    """Nom lisible d'une langue pour le prompt (le code brut sinon)."""
    return LANGUAGE_NAMES.get(code.lower(), code)


def panel_caption(panel: Panel, n_slices: int = 1) -> str:
    """Libellé textuel placé juste avant l'image (ou les tranches) d'une case."""
    kind = "giant panel to scroll" if panel.type == "scroll_vertical" else "panel"
    if panel.part == "top":
        kind = f"top part of a tall drawing, continued in panel {panel.index + 1}"
    elif panel.part == "middle":
        kind = f"middle part of the tall drawing started in panel {panel.index - 1}, continued in panel {panel.index + 1}"
    elif panel.part == "bottom":
        kind = f"bottom part of the tall drawing started above (panel {panel.index - 1})"
    detail = f"{kind}, {panel.width}x{panel.height} px"
    if n_slices > 1:
        detail += f", sent in {n_slices} parts from top to bottom"
    return f"Panel {panel.index} ({detail}):"


def slice_caption(panel: Panel, k: int, n_slices: int) -> str:
    """Libellé d'une tranche d'une case géante."""
    return f"Panel {panel.index} - part {k}/{n_slices}:"


def _chapter_line(meta: ChapterMeta | None) -> str:
    if meta is not None and (meta.series_title or meta.episode_title):
        return f"Series: {meta.series_title or '?'} - Episode: {meta.episode_title or '?'}"
    return ""


def character_sheet(cards: Sequence[CharacterCard]) -> str:
    """Fiche des personnages des épisodes précédents, telle qu'injectée dans le prompt maître."""
    if not cards:
        return "none yet - this is the first episode you narrate, so every name you use is the canonical one."
    lines = []
    for card in cards:
        aka = f" (also called {', '.join(card.also_called)})" if card.also_called else ""
        lines.append(f"  - {card.name}{aka}: {card.who}")
    return "\n" + "\n".join(lines)


def master_instruction(
    *,
    mode: str,
    language: str,
    n_ids: int,
    first_id: int,
    last_id: int,
    meta: ChapterMeta | None = None,
    max_key: int = MAX_KEY_PANELS_PER_PARAGRAPH,
    characters: Sequence[CharacterCard] = (),
    previous_tail: str = "",
) -> str:
    """Prompt système complet, pour ``mode`` ``"images"`` (une requête) ou ``"beats"`` (étape 1b).

    Le formatage se fait en **deux passes** : le bloc INPUT porte ``{n_ids}``, ``{first_id}``
    et ``{last_id}``, il est donc formaté d'abord, puis passé comme *argument* du gabarit.
    L'insérer par concaténation avant une passe unique laisserait ces trois champs littéraux
    dans le prompt envoyé à Gemini - c'est arrivé en bac à sable.

    Raises:
        ValueError: ``mode`` inconnu.
    """
    block = INPUT_BLOCKS.get(mode)
    if block is None:
        raise ValueError(f"mode inconnu : {mode!r} (attendu {' ou '.join(sorted(INPUT_BLOCKS))})")
    return MASTER_PROMPT_TEMPLATE.format(
        language=language_name(language),
        input_block=block.format(n_ids=n_ids, first_id=first_id, last_id=last_id),
        chapter_line=_chapter_line(meta) or "This chapter arrives without a title.",
        character_sheet=character_sheet(characters),
        previous_tail=previous_tail.strip() or "nothing yet - this is the first episode of the series.",
        max_span=PEAK_MAX_SPAN,
        emotions=", ".join(EMOTIONS),
        max_key=max_key,
    )


def build_beats_header(batch: Sequence[Panel], context: Sequence[BeatDraft], meta: ChapterMeta | None) -> str:
    """Texte d'introduction d'un lot (étape 1a) : chapitre, contexte, cases attendues."""
    lines: list[str] = []
    chapter = _chapter_line(meta)
    if chapter:
        lines.append(chapter)
    if context:
        lines.append("Context - previous beats (for continuity only, do not repeat them):")
        for beat in context:
            lines.append(f"- {beat.summary}")
    ids = [panel.index for panel in batch]
    lines.append(f"Batch: {len(ids)} panel(s) numbered {ids[0]} to {ids[-1]}, in reading order.")
    return "\n".join(lines)


def build_beats_footer(batch: Sequence[Panel]) -> str:
    ids = ", ".join(str(panel.index) for panel in batch)
    return (
        f"Extract the story beats of these {len(batch)} panel(s) ({ids}): each number exactly once, "
        "no other number. Answer in JSON."
    )


def build_single_call_header(panels: Sequence[Panel], meta: ChapterMeta | None, language: str) -> str:
    """Texte d'introduction du mode « une requête ».

    Le cadrage (rôle, plan, style, sortie) vit dans l'instruction système
    (:func:`master_instruction`) ; il ne reste ici que l'amorce des images.
    """
    lines: list[str] = []
    chapter = _chapter_line(meta)
    if chapter:
        lines.append(chapter)
    lines.append(f"Script language: {language_name(language)}.")
    lines.append(f"All {len(panels)} panels of the chapter follow, in reading order:")
    return "\n".join(lines)


def build_single_call_footer(panels: Sequence[Panel], max_key: int) -> str:
    """Rappel final du mode « une requête » : c'est le message le plus proche de la
    génération, donc le plus suivi - il rappelle les bornes des numéros et le plan."""
    numbers = f"0 to {panels[-1].index}" if panels else "none"
    return (
        f"End of the chapter ({len(panels)} panels, numbered {numbers}). Plan it first, then write the "
        f"complete recap: every panel number you use anywhere must be in that range. Each paragraph carries "
        f"its plan_id, its weight, its text, its emotion, 1 to {max_key} key panel numbers and the decisive "
        "ones in action_heavy_ids. Answer in JSON."
    )


def _longest_increasing(values: Sequence[int]) -> list[int]:
    """Positions de la plus longue sous-suite strictement croissante de ``values``."""
    tails: list[int] = []
    previous: list[int] = [-1] * len(values)
    for position, value in enumerate(values):
        low, high = 0, len(tails)
        while low < high:
            middle = (low + high) // 2
            if values[tails[middle]] < value:
                low = middle + 1
            else:
                high = middle
        previous[position] = tails[low - 1] if low else -1
        if low == len(tails):
            tails.append(position)
        else:
            tails[low] = position
    order: list[int] = []
    cursor = tails[-1] if tails else -1
    while cursor >= 0:
        order.append(cursor)
        cursor = previous[cursor]
    return order[::-1]


def _enforce_reading_order(
    kept_per_paragraph: list[list[int]], heights: dict[int, int], last_index: int
) -> None:
    """Rétablit un ordre de lecture strictement croissant, sur place.

    Le modèle place parfois une case en avance dans un paragraphe (« case 8 » au paragraphe 2
    alors que le paragraphe 3 raconte les cases 4 à 7), ce qui donne un retour en arrière à
    l'écran. On retire le **minimum** de cases pour rétablir l'ordre — donc l'intruse, pas la
    suite correcte qui la suit — puis on recase les paragraphes devenus vides dans la fenêtre
    libre entre leurs voisins.
    """
    flat = [(index, pid) for index, kept in enumerate(kept_per_paragraph) for pid in kept]
    values = [pid for _, pid in flat]
    if all(before < after for before, after in zip(values, values[1:])):
        return
    keep = set(_longest_increasing(values))
    dropped: dict[int, list[int]] = {}
    for kept in kept_per_paragraph:
        kept.clear()
    for position, (index, pid) in enumerate(flat):
        if position in keep:
            kept_per_paragraph[index].append(pid)
        else:
            dropped.setdefault(index, []).append(pid)
            logger.warning("Case %s hors ordre de lecture, retiree du paragraphe %s", pid, index + 1)
    taken = {pid for kept in kept_per_paragraph for pid in kept}
    for index, kept in enumerate(kept_per_paragraph):
        if kept:
            continue
        lower = max((max(other) for other in kept_per_paragraph[:index] if other), default=-1)
        upper = min((min(other) for other in kept_per_paragraph[index + 1:] if other), default=last_index + 1)
        window = [pid for pid in heights if lower < pid < upper and pid not in taken]
        # Sans fenêtre libre, on garde la plus grande case retirée : un recul ponctuel vaut
        # mieux qu'un paragraphe narré sans image à l'écran.
        choice = max(window or dropped[index], key=lambda pid: heights[pid])
        kept.append(choice)
        taken.add(choice)
        logger.warning("Paragraphe %s recase sur %s apres remise en ordre", index + 1, choice)


#: Au-delà de ce nombre de numéros distincts inconnus, la réponse n'est plus « un id de
#: travers » mais une numérotation entière fausse (le modèle a confondu deux systèmes).
_WRONG_NUMBERING_MIN: int = 2
_WRONG_NUMBERING_RATIO: float = 0.2


def _reject_if_wrong_numbering(
    id_lists: Sequence[Sequence[int]], known: dict[int, int], *, what: str, require_one_valid_each: bool = False
) -> None:
    """Refuse une réponse dont la numérotation des cases est globalement fausse.

    Les numéros inconnus isolés sont normalement retirés un à un, en silence : c'est la
    bonne tolérance pour une coquille. Mais une réponse citant 125 numéros pour un
    chapitre qui en compte 38 — vu une fois, le modèle ayant confondu numéros de case et
    numéros de beat — passait ce filtre et produisait un recap plausible mais faux. Deux
    signes distinguent la coquille de l'erreur de système :

    * trop de numéros distincts inconnus, en valeur absolue comme en proportion ;
    * plus de la moitié des éléments n'ayant plus **aucun** numéro valide.

    Raises:
        InvalidResponseError: la numérotation est fausse ; l'appel est à refaire.
    """
    if not known:
        return
    seen = {pid for ids in id_lists for pid in ids}
    unknown = sorted(seen - known.keys())
    if not unknown:
        return
    limit = max(_WRONG_NUMBERING_MIN, int(_WRONG_NUMBERING_RATIO * len(known)))
    orphans = sum(1 for ids in id_lists if ids and not (set(ids) & known.keys()))
    too_many = len(unknown) > limit
    too_lost = require_one_valid_each and id_lists and orphans > len(id_lists) / 2
    if not (too_many or too_lost):
        return
    raise InvalidResponseError(
        f"Numerotation des cases incoherente ({what}) : {len(unknown)} numero(s) inconnu(s) "
        f"jusqu'a {unknown[-1]}, alors que la derniere case est {max(known)} "
        f"({orphans}/{len(id_lists)} sans aucun numero valide)"
    )


def normalize_recap(
    paragraphs: Sequence[RecapParagraph], panels: Sequence[Panel], *, max_key: int = MAX_KEY_PANELS_PER_PARAGRAPH
) -> list[tuple[ParagraphDraft, list[int], list[int]]]:
    """Valide la réponse du mode « une requête ».

    Écarte les numéros de case inconnus ou déjà pris par un paragraphe précédent, borne à
    ``max_key`` cases (les plus grandes), remplace un paragraphe sans case valide par la plus
    grande case encore libre, supprime les paragraphes au texte vide, puis rétablit un ordre
    de lecture strictement croissant d'un paragraphe au suivant.

    Returns:
        ``[(paragraphe, cases clés, cases action_heavy)]`` dans l'ordre de lecture.
    """
    heights = {panel.index: panel.height for panel in panels}
    _reject_if_wrong_numbering(
        [p.key_panel_ids for p in paragraphs], heights, what="paragraphe", require_one_valid_each=True
    )
    free = [panel.index for panel in panels]
    used: set[int] = set()
    drafts: list[ParagraphDraft] = []
    kept_per_paragraph: list[list[int]] = []
    heavy_per_paragraph: list[list[int]] = []
    for paragraph in paragraphs:
        text = _clean_text(paragraph.text)
        if not text:
            logger.warning("Paragraphe vide ignore (mode une requete)")
            continue
        kept: list[int] = []
        for pid in paragraph.key_panel_ids:
            if pid in heights and pid not in used and pid not in kept:
                kept.append(pid)
            else:
                logger.warning("Case %s inconnue ou deja utilisee, ignoree", pid)
        if len(kept) > max_key:
            kept = sorted(sorted(kept, key=lambda p: -heights[p])[:max_key])
        if not kept:
            fallback = [pid for pid in free if pid not in used]
            if not fallback:
                logger.warning("Plus aucune case disponible : paragraphe ignore")
                continue
            kept = [max(fallback, key=lambda p: heights[p])]
            logger.warning("Paragraphe sans case valide, repli sur %s", kept)
        kept = sorted(kept)
        used.update(kept)
        drafts.append(ParagraphDraft(text=text, beat_ids=[], emotion=coerce_emotion(paragraph.emotion)))
        kept_per_paragraph.append(kept)
        heavy_per_paragraph.append(sorted(set(paragraph.action_heavy_ids)))
    if not drafts:
        raise InvalidResponseError("Aucun paragraphe exploitable dans la reponse")
    if panels:
        _enforce_reading_order(kept_per_paragraph, heights, panels[-1].index)
    return [
        (draft, kept, [pid for pid in heavy if pid in kept])
        for draft, kept, heavy in zip(drafts, kept_per_paragraph, heavy_per_paragraph)
    ]


def build_script_prompt(beats: Sequence[Beat], meta: ChapterMeta | None, language: str) -> str:
    """Prompt utilisateur de l'étape 1b : tous les beats du chapitre en texte."""
    lines: list[str] = []
    chapter = _chapter_line(meta)
    if chapter:
        lines.append(chapter)
    lines.append(f"Script language: {language_name(language)}.")
    lines.append("Story beats in reading order:")
    for beat in beats:
        ids = ", ".join(str(pid) for pid in beat.panel_ids)
        flag = " [FILLER - do not narrate]" if beat.is_filler else ""
        line = f"Beat {beat.index} (panels {ids}){flag}: {beat.summary}"
        if beat.characters:
            line += f" | Characters: {', '.join(beat.characters)}"
        if beat.dialogue:
            quotes = " / ".join(f'"{q}"' for q in beat.dialogue)
            line += f" | Dialogue: {quotes}"
        lines.append(line)
    lines.append(
        "Plan the chapter first, then write the complete recap script: JSON paragraphs, each with its "
        "plan_id, its weight and the beat numbers it covers."
    )
    return "\n".join(lines)


def build_keyframes_header(group: Sequence[tuple[int, str, list[int]]], meta: ChapterMeta | None) -> str:
    """Texte d'introduction d'un groupe de paragraphes (étape 2)."""
    lines: list[str] = []
    chapter = _chapter_line(meta)
    if chapter:
        lines.append(chapter)
    lines.append(f"Script paragraphs of this group ({len(group)}) with their candidate panels:")
    return "\n".join(lines)


def paragraph_line(index: int, text: str, candidates: Sequence[int]) -> str:
    ids = ", ".join(str(pid) for pid in candidates)
    return f"Paragraph {index} (candidate panels: {ids}):\n{text}"


def build_keyframes_footer(group: Sequence[tuple[int, str, list[int]]], max_key: int) -> str:
    ks = ", ".join(str(index) for index, _, _ in group)
    return (
        f"Return exactly one choice per paragraph ({ks}), each with 1 to {max_key} key panel numbers "
        "chosen only among that paragraph's candidates, plus the subset of those key panels that show a "
        "decisive impact in action_heavy_ids (empty if none). Answer in JSON."
    )


# --- Parsing ---------------------------------------------------------------------------
_CODE_FENCE = re.compile(r"^\s*```(?:json)?\s*(.*?)\s*```\s*$", re.DOTALL)


def _strip_code_fence(text: str) -> str:
    match = _CODE_FENCE.match(text)
    return match.group(1) if match else text


#: Clé JSON écrite sans guillemets (``plan_id: 0``), faute la plus fréquente du modèle.
_BARE_KEY = re.compile(r'([{,]\s*)([A-Za-z_][A-Za-z0-9_]*)(\s*:)')
#: Virgule traînante avant une fermeture.
_TRAILING_COMMA = re.compile(r',(\s*[}\]])')
#: Valeur nue glissée entre deux paires d'un objet (``"weight":1,-1,"text":...``). Le
#: lookahead exige une **clé suivie de deux-points** : dans un tableau, ``[1,-1,2]``, le
#: jeton suivant n'est jamais une clé, donc un tableau valide n'est jamais touché.
_STRAY_VALUE = re.compile(r',\s*-?\d+(?:\.\d+)?(?=\s*,\s*"[^"]*"\s*:)')


def _outside_string_spans(text: str) -> list[tuple[int, int]]:
    """Intervalles de ``text`` situés **hors** des chaînes JSON, échappements compris.

    Toute réparation doit se limiter à ces intervalles : une narration contenant
    ``Hugh sneers, "Listen: the gate opens at dawn."`` ressemble à une clé nue, et une
    réparation naïve la réécrit — donc corrompt le script rendu à l'utilisateur.
    """
    spans: list[tuple[int, int]] = []
    start, in_string, escaped = 0, False, False
    for i, char in enumerate(text):
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string, start = False, i + 1
        elif char == '"':
            spans.append((start, i))
            in_string = True
    if not in_string:
        spans.append((start, len(text)))
    return spans


def _repair_json(text: str) -> str:
    """Corrige les fautes de syntaxe courantes du modèle, **hors** du contenu des chaînes.

    Trois corrections, toutes sans ambiguïté : clés sans guillemets, virgules traînantes,
    et valeur nue glissée entre deux paires d'un objet. Chacune est réparée localement
    plutôt que par un second appel Gemini, qui coûterait une unité de quota.
    """
    def apply(pattern: re.Pattern[str], replace: Callable[[re.Match[str]], str], source: str) -> str:
        # Les motifs sont cherchés sur le texte ENTIER (un lookahead doit pouvoir regarder
        # par-dessus une chaîne voisine), mais seule une correspondance qui COMMENCE hors
        # d'une chaîne est appliquée.
        spans = _outside_string_spans(source)
        return pattern.sub(
            lambda m: replace(m) if any(b <= m.start() < e for b, e in spans) else m.group(0),
            source,
        )

    text = apply(_BARE_KEY, lambda m: f'{m.group(1)}"{m.group(2)}"{m.group(3)}', text)
    text = apply(_TRAILING_COMMA, lambda m: m.group(1), text)
    return apply(_STRAY_VALUE, lambda m: "", text)


def _looks_truncated(text: str) -> bool:
    """Vrai si le texte est un JSON **coupé en plein vol**, pas du bavardage.

    Regarder le dernier caractère ne suffit pas : une réponse coupée juste après un objet
    complet se termine par ``}`` tout en laissant le tableau parent ouvert. On compte donc
    les ouvertures non refermées, en ignorant celles qui sont à l'intérieur d'une chaîne.
    """
    stripped = text.strip()
    if not stripped.startswith(("{", "[")):
        return False  # du texte libre : le modèle a répondu autre chose que du JSON
    depth, in_string, escaped = 0, False, False
    for char in stripped:
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
        elif char == '"':
            in_string = True
        elif char in "{[":
            depth += 1
        elif char in "}]":
            depth -= 1
    return in_string or depth > 0


def _safe_int(digits: str) -> int:
    """Entier d'une réponse, ou :data:`RUNAWAY_INT` si le modèle s'est emballé sur un chiffre.

    Vu en production : une réponse contenant un entier de 5 234 chiffres. Python refuse de
    convertir au-delà de 4 300 chiffres (garde-fou contre un déni de service), et
    ``json.loads`` lève alors un :class:`ValueError` ordinaire - pas un
    :class:`json.JSONDecodeError`. Tout le reste du JSON était pourtant valide.

    Plutôt que de jeter la réponse et de rebrûler un appel Gemini, on remplace le nombre
    aberrant par une valeur que la validation des identifiants écarte ensuite d'elle-même.
    """
    if len(digits.lstrip("-")) > MAX_INT_DIGITS:
        logger.warning(
            "Entier aberrant dans la reponse (%d chiffres) : remplace par %d", len(digits), RUNAWAY_INT
        )
        return RUNAWAY_INT
    return int(digits)


def _loads(text: str) -> Any:
    """``json.loads`` avec le garde-fou de :func:`_safe_int` sur les entiers."""
    return json.loads(text, parse_int=_safe_int)


def _extract_json(text: str) -> Any:
    """Décode le JSON d'une réponse, en tolérant du texte parasite autour de l'objet."""
    text = _strip_code_fence(text)
    try:
        return _loads(text)
    # ValueError et non JSONDecodeError : un entier impossible a convertir leve le parent,
    # et la difference a suffi a faire echouer un chapitre entier au lieu d'un reessai.
    except ValueError as first_error:
        for opener, closer in (("{", "}"), ("[", "]")):
            start, end = text.find(opener), text.rfind(closer)
            if 0 <= start < end:
                try:
                    return _loads(text[start : end + 1])
                except ValueError:
                    continue
        # Reparation locale : evite un aller-retour Gemini (et donc une unite de quota)
        # pour une cle sans guillemets ou une virgule en trop.
        repaired = _repair_json(text)
        if repaired != text:
            try:
                return _loads(repaired)
            except ValueError:
                pass
        if _looks_truncated(text):
            raise TruncatedResponseError(
                f"Reponse tronquee : JSON incomplet ({len(text)} caracteres, fin inattendue)"
            ) from first_error
        raise InvalidResponseError(f"JSON invalide dans la reponse du modele : {first_error}") from first_error


def coerce_emotion(value: object) -> str:
    """Ramène une émotion libre vers une valeur de :data:`EMOTIONS` (``neutral`` par défaut)."""
    key = str(value or "").strip().lower()
    if key in EMOTIONS:
        return key
    if key in _EMOTION_ALIASES:
        return _EMOTION_ALIASES[key]
    if key:
        logger.warning("Emotion inconnue %r, remplacee par 'neutral'", key)
    return "neutral"


def coerce_bool(value: object) -> bool:
    """Interprète un booléen renvoyé sous forme libre (``"true"``, ``1``, ``"yes"``...)."""
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        return value.strip().lower() in {"true", "1", "yes", "y", "oui", "vrai"}
    return False


def _coerce_str_list(value: object) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value.strip()] if value.strip() else []
    if isinstance(value, list):
        return [str(v).strip() for v in value if str(v).strip()]
    return []


def _enum_name(value: object) -> str:
    if value is None:
        return ""
    name = getattr(value, "name", None)
    if isinstance(name, str):
        return name
    return str(value).split(".")[-1]


def _finish_reason(response: Any) -> str:
    candidates = getattr(response, "candidates", None) or []
    if not candidates:
        return ""
    return _enum_name(getattr(candidates[0], "finish_reason", None))


def response_data(response: Any, model_cls: type[BaseModel]) -> Any:
    """Contenu exploitable d'une réponse : instance ``parsed`` du bon type, sinon JSON du texte.

    Raises:
        AnalyzerError: prompt bloqué ou réponse refusée par la sécurité (définitif).
        InvalidResponseError: réponse vide/tronquée ou JSON inexploitable (réessayable).
    """
    parsed = getattr(response, "parsed", None)
    if isinstance(parsed, model_cls):
        return parsed
    feedback = getattr(response, "prompt_feedback", None)
    block_reason = _enum_name(getattr(feedback, "block_reason", None))
    if block_reason:
        message = getattr(feedback, "block_reason_message", None) or ""
        raise AnalyzerError(f"Prompt bloque par Gemini (block_reason={block_reason}) {message}".rstrip())
    text = getattr(response, "text", None)
    finish = _finish_reason(response)
    if not text:
        if finish in _FATAL_FINISH_REASONS:
            raise AnalyzerError(f"Reponse refusee par Gemini (finish_reason={finish})")
        raise InvalidResponseError(f"Reponse vide du modele (finish_reason={finish or 'inconnu'})")
    # Une reponse coupee au plafond de jetons arrive AVEC du texte : sans ce test elle
    # ressortirait en « JSON invalide », et on chercherait le bug du mauvais cote.
    if finish == "MAX_TOKENS":
        raise TruncatedResponseError(
            f"Reponse tronquee par le plafond de jetons (finish_reason={finish}, {len(text)} caracteres)"
        )
    return _extract_json(text)


def _validate_list(
    data: Any,
    model_cls: type[BaseModel],
    list_key: str,
    fix_item: Callable[[dict], None],
    fix_root: Callable[[dict], None] | None = None,
) -> BaseModel:
    if isinstance(data, model_cls):
        return data
    if isinstance(data, list):
        data = {list_key: data}
    if not isinstance(data, dict) or not isinstance(data.get(list_key), list):
        raise InvalidResponseError(f"Reponse sans liste '{list_key}'")
    if fix_root is not None:
        fix_root(data)
    for item in data[list_key]:
        if isinstance(item, dict):
            fix_item(item)
    try:
        return model_cls.model_validate(data)
    except ValidationError as exc:
        raise InvalidResponseError(f"Reponse non conforme au schema {model_cls.__name__} : {exc}") from exc


def _fix_scene(item: dict) -> None:
    item["emotion"] = coerce_emotion(item.get("emotion"))
    item["is_filler"] = coerce_bool(item.get("is_filler", False))


def _fix_beat(item: dict) -> None:
    item["characters"] = _coerce_str_list(item.get("characters"))
    item["dialogue"] = _coerce_str_list(item.get("dialogue"))
    item["is_filler"] = coerce_bool(item.get("is_filler", False))
    if "summary" not in item and "narration" in item:
        item["summary"] = item["narration"]


def _coerce_weight(value: object) -> int:
    """Ramène le poids d'un paragraphe dans [1, 3] ; 2 (conséquence) par défaut."""
    try:
        return max(1, min(3, int(value)))  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return 2


def _fix_plan_fields(item: dict) -> None:
    """Complète ``plan_id`` et ``weight``, absents des réponses (et des anciens fichiers)."""
    item["weight"] = _coerce_weight(item.get("weight", 2))
    try:
        item["plan_id"] = int(item.get("plan_id", -1))
    except (TypeError, ValueError):
        item["plan_id"] = -1


def _fix_master(data: dict) -> None:
    """Tolère une réponse sans plan ni fiche personnages plutôt que de la rejeter.

    Rejeter coûterait un appel Gemini entier — cher sur le palier gratuit — pour une
    partie de la réponse dont l'absence ne casse rien en aval : le plan n'est qu'une trace
    du raisonnement, et la fiche des personnages ne sert qu'au chapitre suivant.
    """
    if not isinstance(data.get("plan"), dict):
        logger.warning("Reponse sans plan : accepte tel quel, le plan restera vide")
        data["plan"] = {"acts": [], "peak_id": -1, "cta_unit_index": -1, "units": []}
    characters = data.get("characters")
    if not isinstance(characters, list):
        data["characters"] = []
    else:
        for card in characters:
            if isinstance(card, dict):
                card.setdefault("also_called", [])
                card.setdefault("who", "")


def _fix_paragraph(item: dict) -> None:
    item["emotion"] = coerce_emotion(item.get("emotion"))
    if "text" not in item and "narration" in item:
        item["text"] = item["narration"]
    item.setdefault("beat_ids", [])
    _fix_plan_fields(item)


def _fix_choice(item: dict) -> None:
    item.setdefault("key_panel_ids", item.pop("panel_ids", []))
    heavy = item.get("action_heavy_ids")
    item["action_heavy_ids"] = [int(x) for x in heavy] if isinstance(heavy, list) else []


def parse_scene_json(text: str) -> list[SceneDraft]:
    """Analyse un JSON de scènes (schéma historique ``SceneBatch``), tolérant puis validé."""
    return list(_validate_list(_extract_json(text), SceneBatch, "scenes", _fix_scene).scenes)


def parse_response(response: Any) -> list[SceneDraft]:
    """Extrait des scènes (``SceneBatch``) d'une ``GenerateContentResponse``."""
    return list(_validate_list(response_data(response, SceneBatch), SceneBatch, "scenes", _fix_scene).scenes)


def parse_beats(response: Any) -> list[BeatDraft]:
    """Extrait les beats (``BeatBatch``) d'une réponse."""
    return list(_validate_list(response_data(response, BeatBatch), BeatBatch, "beats", _fix_beat).beats)


def parse_master_script(response: Any) -> ScriptDraft:
    """Réponse complète de l'étape 1b : plan, paragraphes **et** fiche des personnages.

    Une réponse sans plan ni personnages est complétée plutôt que rejetée
    (cf. :func:`_fix_master`) : la refuser coûterait un appel Gemini entier.
    """
    draft = _validate_list(
        response_data(response, ScriptDraft), ScriptDraft, "paragraphs", _fix_paragraph, _fix_master
    )
    assert isinstance(draft, ScriptDraft)
    return draft


def parse_script(response: Any) -> list[ParagraphDraft]:
    """Paragraphes seuls (``ScriptDraft``) : vue étroite de :func:`parse_master_script`."""
    return list(parse_master_script(response).paragraphs)


def _fix_recap(item: dict) -> None:
    item["emotion"] = coerce_emotion(item.get("emotion"))
    if "text" not in item and "narration" in item:
        item["text"] = item["narration"]
    for field in ("key_panel_ids", "action_heavy_ids"):
        value = item.get(field)
        item[field] = [int(x) for x in value] if isinstance(value, list) else []
    _fix_plan_fields(item)


def parse_master_recap(response: Any) -> RecapDraft:
    """Réponse complète du mode « une requête » : plan, paragraphes et personnages."""
    draft = _validate_list(
        response_data(response, RecapDraft), RecapDraft, "paragraphs", _fix_recap, _fix_master
    )
    assert isinstance(draft, RecapDraft)
    return draft


def parse_recap(response: Any) -> list[RecapParagraph]:
    """Paragraphes seuls (``RecapDraft``) : vue étroite de :func:`parse_master_recap`."""
    return list(parse_master_recap(response).paragraphs)


def parse_keyframes(response: Any) -> list[KeyframeChoice]:
    """Extrait les choix de cases clés (``KeyframeBatch``) d'une réponse."""
    return list(_validate_list(response_data(response, KeyframeBatch), KeyframeBatch, "choices", _fix_choice).choices)


# --- Normalisation ----------------------------------------------------------------------
def _clean_text(text: str) -> str:
    cleaned = " ".join(text.translate(_ZERO_WIDTH).split())
    return cleaned if any(ch.isalnum() for ch in cleaned) else ""


def normalize_partition(
    drafts: Sequence[BaseModel],
    ids: Sequence[int],
    *,
    ids_field: str = "panel_ids",
    text_field: str = "narration",
) -> list[BaseModel]:
    """Rend des éléments (scènes, beats, paragraphes) cohérents avec les identifiants soumis.

    Post-conditions : les ``ids_field`` des éléments renvoyés forment une
    **partition** de ``ids`` en suites consécutives triées ; aucun élément n'est
    vide ni sans texte. Les identifiants inconnus ou déjà attribués sont retirés,
    les identifiants oubliés sont rattachés à l'élément qui les précède, les
    éléments entrelacés redeviennent consécutifs, un élément vidé est absorbé.

    Raises:
        InvalidResponseError: aucun élément valide.
    """
    ordered_ids = sorted(set(int(i) for i in ids))
    allowed = set(ordered_ids)
    cleaned: list[BaseModel] = []
    seen: set[int] = set()
    for draft in drafts:
        kept: list[int] = []
        for value in getattr(draft, ids_field):
            if value not in allowed:
                logger.warning("%s %d hors de l'ensemble %s, ignore", ids_field, value, ordered_ids[:8])
                continue
            if value in seen or value in kept:
                logger.warning("%s %d deja attribue, ignore", ids_field, value)
                continue
            kept.append(value)
        text = _clean_text(getattr(draft, text_field))
        if not kept:
            logger.warning("Element sans identifiant valide ignore : %r", text[:60])
            continue
        if not text:
            logger.warning("Element sans texte exploitable ignore (%s %s)", ids_field, kept)
            continue
        seen.update(kept)
        cleaned.append(draft.model_copy(update={ids_field: sorted(kept), text_field: text}))
    if not cleaned:
        raise InvalidResponseError(f"Aucun element valide pour {ordered_ids[:10]}")
    cleaned.sort(key=lambda d: getattr(d, ids_field)[0])
    starts = [getattr(d, ids_field)[0] for d in cleaned]
    assigned: list[list[int]] = [[] for _ in cleaned]
    for value in ordered_ids:
        assigned[max(0, bisect_right(starts, value) - 1)].append(value)
    result: list[BaseModel] = []
    for draft, got in zip(cleaned, assigned):
        proposed = getattr(draft, ids_field)
        if not got:
            logger.warning("Element absorbe par son voisin (%s %s deja attribues)", ids_field, proposed)
            continue
        if got != proposed:
            logger.warning(
                "%s %s -> %s : rattaches %s, retires %s (elements consecutifs)",
                ids_field, proposed, got, sorted(set(got) - set(proposed)), sorted(set(proposed) - set(got)),
            )
        result.append(draft.model_copy(update={ids_field: got}))
    return result


def normalize_scenes(scenes: Sequence[SceneDraft], batch_ids: Sequence[int]) -> list[SceneDraft]:
    """Partition de cases par scènes (schéma historique)."""
    return normalize_partition(scenes, batch_ids, ids_field="panel_ids", text_field="narration")  # type: ignore[return-value]


def normalize_beats(beats: Sequence[BeatDraft], batch_ids: Sequence[int]) -> list[BeatDraft]:
    """Partition des cases d'un lot par beats (étape 1a)."""
    return normalize_partition(beats, batch_ids, ids_field="panel_ids", text_field="summary")  # type: ignore[return-value]


def find_forbidden(text: str) -> list[str]:
    """Extraits du texte qui utilisent une formule visuelle interdite (vide si aucun)."""
    hits: list[str] = []
    for pattern in _FORBIDDEN:
        for match in pattern.finditer(text):
            hits.append(match.group(0).strip())
    return hits


def scrub_forbidden(text: str) -> str:
    """Nettoyage de dernier recours des formules visuelles les plus courantes."""
    cleaned = re.sub(
        r"(?i)(^|[.!?]\s+)in (?:this|the|that) (?:panel|panels|scene|image|frame|picture|illustration|shot|page),?\s+",
        r"\1", text,
    )
    cleaned = re.sub(r"(?i)(^|[.!?]\s+)(?:here|ici),?\s+", r"\1", cleaned)
    cleaned = re.sub(r"(?i)(^|[.!?]\s+)we (?:can |then |now )?(?:see|watch|observe) (?:that |how )?", r"\1", cleaned)
    cleaned = re.sub(r"(?i)\bwe (?:can |then |now )?(?:see|watch) (?:that |how )?", "", cleaned)
    cleaned = re.sub(r"(^|[.!?]\s+)([a-z])", lambda m: m.group(1) + m.group(2).upper(), cleaned)
    return " ".join(cleaned.split())


def split_sentences(text: str) -> list[str]:
    """Découpe un texte en phrases (ponctuation forte suivie d'une majuscule, d'un chiffre ou d'un guillemet)."""
    return [s.strip() for s in _SENTENCE_BOUNDARY.split(text.strip()) if s.strip()]


def generic_intro(text: str) -> str | None:
    """Première phrase du texte si elle est une introduction générique, sinon ``None``."""
    sentences = split_sentences(text)
    if not sentences:
        return None
    first = sentences[0]
    return first if any(pattern.search(first) for pattern in _GENERIC_INTRO) else None


def cta_text_for(language: str, cta_text: str | None = None) -> str:
    """Phrase d'appel à l'abonnement : explicite, sinon celle de la langue (anglais en repli)."""
    if cta_text is not None:
        return cta_text.strip()
    return CTA_TEXTS.get(language.lower(), CTA_TEXTS["en"])


def cta_paragraph_index(n_paragraphs: int, position: float = CTA_POSITION) -> int:
    """Indice du paragraphe qui reçoit l'appel : vers 40 %, jamais dans les deux premiers ni le dernier."""
    if n_paragraphs <= 0:
        return 0
    wanted = max(2, round(position * (n_paragraphs - 1)))
    return max(0, min(n_paragraphs - 2, wanted))


def _model_cta_is_well_placed(paragraphs: Sequence[ParagraphDraft]) -> bool:
    """``True`` si le modèle a écrit **un seul** appel à l'abonnement, et au bon endroit.

    Le prompt maître demande une formulation neuve à chaque épisode : quand le modèle obéit,
    sa phrase vaut mieux que la nôtre, toujours identique. On ne la remplace donc que si elle
    est mal placée - dans les deux premiers paragraphes, dans le dernier, ou en clôture de son
    propre paragraphe, les trois endroits où elle coupe la rétention.
    """
    hits = [
        (i, position, len(sentences))
        for i, paragraph in enumerate(paragraphs)
        for sentences in [split_sentences(paragraph.text)]
        for position, sentence in enumerate(sentences)
        if CTA_PATTERN.search(sentence)
    ]
    if len(hits) != 1 or len(paragraphs) < 4:
        return False
    index, position, n_sentences = hits[0]
    return 2 <= index <= len(paragraphs) - 2 and position < n_sentences - 1


def enforce_script_conventions(
    paragraphs: Sequence[ParagraphDraft], *, language: str = DEFAULT_LANGUAGE, cta_text: str | None = None
) -> list[ParagraphDraft]:
    """Applique les conventions de script : accroche immédiate, un seul appel à l'abonnement, placé au milieu.

    - une introduction générique en première phrase est supprimée (si le paragraphe
      contient d'autres phrases) ;
    - les appels à l'abonnement écrits par le modèle sont retirés, puis **un** appel
      canonique (``cta_text``, ou celui de la langue ; chaîne vide = aucun) est
      inséré à la fin du paragraphe désigné par :func:`cta_paragraph_index`, ou
      avant la dernière phrase si ce paragraphe est le dernier du script.
    """
    result: list[ParagraphDraft] = [p.model_copy() for p in paragraphs]
    if not result:
        return result

    intro = generic_intro(result[0].text)
    if intro is not None:
        sentences = split_sentences(result[0].text)
        if len(sentences) > 1:
            logger.warning("Introduction generique supprimee : %r", intro[:80])
            result[0] = result[0].model_copy(update={"text": " ".join(sentences[1:])})
        else:
            logger.warning("Introduction generique conservee (seule phrase du paragraphe) : %r", intro[:80])

    if cta_text is None and _model_cta_is_well_placed(result):
        logger.info("Appel a l'abonnement du modele conserve (place et formule correctement)")
        return result

    removed = 0
    for i, paragraph in enumerate(result):
        sentences = split_sentences(paragraph.text)
        kept = [s for s in sentences if not CTA_PATTERN.search(s)]
        if len(kept) != len(sentences):
            removed += len(sentences) - len(kept)
            result[i] = paragraph.model_copy(update={"text": " ".join(kept) if kept else paragraph.text})
    if removed:
        logger.info("%d appel(s) a l'abonnement du modele retire(s), remplace(s) par l'appel canonique", removed)

    cta = cta_text_for(language, cta_text)
    if cta:
        k = cta_paragraph_index(len(result))
        sentences = split_sentences(result[k].text)
        if k == len(result) - 1 and len(sentences) > 1:
            sentences.insert(len(sentences) - 1, cta)
        else:
            sentences.append(cta)
        result[k] = result[k].model_copy(update={"text": " ".join(sentences)})
        logger.info("Appel a l'abonnement place dans le paragraphe %d/%d", k, len(result))
    return result


def normalize_script(paragraphs: Sequence[ParagraphDraft], beats: Sequence[Beat]) -> list[ParagraphDraft]:
    """Rend le script cohérent : chaque beat narratif appartient à exactement un paragraphe.

    Les beats de remplissage ne sont attribués à aucun paragraphe.

    Raises:
        InvalidResponseError: aucun paragraphe valide.
    """
    story_ids = [beat.index for beat in beats if not beat.is_filler]
    if not story_ids:
        story_ids = [beat.index for beat in beats]
    return normalize_partition(paragraphs, story_ids, ids_field="beat_ids", text_field="text")  # type: ignore[return-value]


def normalize_keyframes(
    choices: Sequence[KeyframeChoice],
    group: Sequence[tuple[int, str, list[int]]],
    heights: dict[int, int],
    used: set[int],
    *,
    max_key: int = MAX_KEY_PANELS_PER_PARAGRAPH,
) -> dict[int, list[int]]:
    """Cases clés finales par paragraphe : candidates seulement, sans doublon, jamais vide.

    Args:
        choices: réponse du modèle.
        group: ``(index du paragraphe, texte, cases candidates)`` du groupe.
        heights: hauteur de chaque case (repli sur la plus grande).
        used: cases déjà retenues par des paragraphes précédents (mis à jour).
        max_key: nombre maximal de cases clés par paragraphe.
    """
    # Seul le critere "trop de numeros inconnus" s'applique ici : un groupe sans aucune
    # case clé est un repli legitime et frequent a l'etape 2, pas une erreur de reponse.
    _reject_if_wrong_numbering([c.key_panel_ids for c in choices], heights, what="groupe de cases cles")
    by_paragraph: dict[int, list[int]] = {}
    for choice in choices:
        by_paragraph.setdefault(choice.paragraph_index, []).extend(choice.key_panel_ids)
    result: dict[int, list[int]] = {}
    for index, _, candidates in group:
        wanted = by_paragraph.get(index, [])
        kept: list[int] = []
        for pid in wanted:
            if pid in candidates and pid not in used and pid not in kept:
                kept.append(pid)
            else:
                logger.warning("Paragraphe %d : case %d non candidate ou deja utilisee, ignoree", index, pid)
        if len(kept) > max_key:
            kept = sorted(sorted(kept, key=lambda p: -heights.get(p, 0))[:max_key])
        if not kept:
            fallback = [pid for pid in sorted(candidates, key=lambda p: -heights.get(p, 0)) if pid not in used]
            kept = fallback[:1] or sorted(candidates)[:1]
            logger.warning("Paragraphe %d : aucune case cle retenue par le modele, repli sur %s", index, kept)
        kept = sorted(kept)
        used.update(kept)
        result[index] = kept
    return result


def normalize_action_heavy(
    choices: Sequence[KeyframeChoice], keyframes: Mapping[int, Sequence[int]]
) -> dict[int, list[int]]:
    """Cases « action_heavy » par paragraphe, restreintes aux cases clés retenues (ordre de lecture)."""
    wanted: dict[int, set[int]] = {}
    for choice in choices:
        wanted.setdefault(choice.paragraph_index, set()).update(choice.action_heavy_ids)
    return {
        index: [pid for pid in kept if pid in wanted.get(index, set())]
        for index, kept in keyframes.items()
    }


def _check_panels(panels: Sequence[Panel]) -> list[Panel]:
    """Valide la liste de cases commune aux deux modes d'analyse.

    Raises:
        ValueError: aucune case, ou numéros de cases dupliqués.
    """
    panels = list(panels)
    if not panels:
        raise ValueError("Aucune case a analyser")
    indexes = [panel.index for panel in panels]
    if len(set(indexes)) != len(indexes):
        raise ValueError("Numeros de cases dupliques : chaque Panel.index doit etre unique")
    return panels


def _capture(func: Callable[[int], Any]) -> Callable[[int], Any]:
    """Enveloppe une fonction pour qu'elle renvoie son exception au lieu de la lever.

    Nécessaire avec ``ThreadPoolExecutor.map``, qui relèverait la première exception et
    ferait perdre les groupes déjà obtenus (donc le point de reprise).
    """

    def wrapped(value: int) -> Any:
        try:
            return func(value)
        except Exception as exc:  # noqa: BLE001 - transmise a l'appelant
            return exc

    return wrapped


def _usage_of(response: Any) -> tuple[int, int, int]:
    usage = getattr(response, "usage_metadata", None)
    if usage is None:
        return 0, 0, 0

    def _count(name: str) -> int:
        return int(getattr(usage, name, 0) or 0)

    return _count("prompt_token_count"), _count("candidates_token_count"), _count("thoughts_token_count")


def _is_retryable(exc: Exception) -> bool:
    if isinstance(exc, InvalidResponseError):
        return True
    if isinstance(exc, AnalyzerError):
        return False
    if isinstance(exc, errors.ServerError):
        return True
    if is_daily_quota_error(exc):
        return False
    if isinstance(exc, errors.ClientError):
        return exc.code in (408, 429)
    return isinstance(exc, _NETWORK_ERRORS)


def resolve_model(model: str | None = None) -> str:
    """Modèle effectif : argument, sinon ``$GEMINI_MODEL``, sinon :data:`DEFAULT_MODEL`."""
    return model or os.environ.get(MODEL_ENV_VAR) or DEFAULT_MODEL


def build_client(api_key: str | None = None) -> Any:
    """Construit un ``genai.Client`` avec la clé résolue par :func:`load_gemini_api_key`.

    Raises:
        AnalyzerError: aucune clé disponible ou client impossible à créer.
    """
    from google import genai

    key = load_gemini_api_key(api_key)
    if not key:
        raise AnalyzerError(f"Aucune cle Gemini trouvee : {gemini_key_hint()}")
    try:
        return genai.Client(api_key=key)
    except Exception as exc:  # noqa: BLE001
        raise AnalyzerError(f"Impossible de creer le client Gemini : {exc}") from exc


# --- Analyzer ------------------------------------------------------------------------
class GeminiAnalyzer:
    """Analyse un chapitre en deux étapes (script global, puis cases clés) avec Gemini.

    Args:
        client: client ``genai.Client`` (ou compatible) ; créé depuis la clé locale si ``None``.
        model: nom du modèle (voir :func:`resolve_model`).
        batch_size: images par lot (1 à :data:`MAX_BATCH_SIZE`).
        language: code de langue du script.
        temperature: température de génération.
        max_image_width: largeur maximale des images envoyées.
        max_slice_height: hauteur maximale d'une tranche d'image (cases géantes).
        max_key_panels: cases clés maximales par paragraphe.
        max_retries: nouvelles tentatives par appel sur erreur transitoire / réponse invalide.
        backoff: base (secondes) du backoff exponentiel.
        timeout_ms: timeout HTTP par requête.
        delay_between_batches: pause forcée (secondes) entre deux appels d'un même chapitre
            (:data:`BATCH_DELAY_S` par défaut, étalement des jetons par minute).
        thinking_budget: budget de tokens de réflexion (``None`` = défaut du modèle).
        api_key: clé(s) explicite(s) si ni ``client`` ni ``manager`` ne sont fournis.
        manager: :class:`~src.utils.gemini_manager.GeminiManager` partagé (rotation de clés,
            cascade de modèles, limite RPM) ; construit depuis l'environnement si ``client``
            et ``manager`` sont absents.
    """

    def __init__(
        self,
        client: Any | None = None,
        *,
        manager: GeminiManager | None = None,
        model: str | None = None,
        batch_size: int = DEFAULT_BATCH_SIZE,
        language: str = DEFAULT_LANGUAGE,
        temperature: float = DEFAULT_TEMPERATURE,
        max_image_width: int = DEFAULT_MAX_IMAGE_WIDTH,
        jpeg_quality: int = JPEG_QUALITY,
        max_slice_height: int = DEFAULT_MAX_SLICE_HEIGHT,
        max_key_panels: int = MAX_KEY_PANELS_PER_PARAGRAPH,
        max_retries: int = DEFAULT_MAX_RETRIES,
        backoff: float = DEFAULT_BACKOFF,
        timeout_ms: int = DEFAULT_TIMEOUT_MS,
        max_output_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS,
        delay_between_batches: float = BATCH_DELAY_S,
        keyframe_workers: int = DEFAULT_KEYFRAME_WORKERS,
        thinking_budget: int | None = None,
        cta_text: str | None = None,
        known_characters: Sequence[CharacterCard] = (),
        previous_tail: str = "",
        api_key: str | None = None,
    ) -> None:
        if not 1 <= batch_size <= MAX_BATCH_SIZE:
            raise ValueError(f"batch_size doit etre entre 1 et {MAX_BATCH_SIZE}, recu {batch_size}")
        if max_retries < 0:
            raise ValueError("max_retries doit etre >= 0")
        if max_image_width < 1 or max_slice_height < 1:
            raise ValueError("max_image_width et max_slice_height doivent etre >= 1")
        if not 1 <= jpeg_quality <= 95:
            raise ValueError(f"jpeg_quality doit etre entre 1 et 95, recu {jpeg_quality}")
        if timeout_ms < 1:
            raise ValueError("timeout_ms doit etre >= 1")
        if max_output_tokens < 1:
            raise ValueError("max_output_tokens doit etre >= 1")
        if max_key_panels < 1:
            raise ValueError("max_key_panels doit etre >= 1")
        self.model = resolve_model(model)
        self.batch_size = batch_size
        self.language = language
        self.temperature = temperature
        self.max_image_width = max_image_width
        self.jpeg_quality = jpeg_quality
        self.max_slice_height = max_slice_height
        self.max_key_panels = max_key_panels
        self.max_retries = max_retries
        self.backoff = backoff
        self.timeout_ms = timeout_ms
        self.max_output_tokens = max_output_tokens
        self.delay_between_batches = delay_between_batches
        self.keyframe_workers = max(1, keyframe_workers)
        self.thinking_budget = thinking_budget
        self.cta_text = cta_text
        #: Mémoire de série : personnages déjà nommés et fin du chapitre précédent, injectés
        #: dans le prompt maître pour que le héros garde son nom d'un épisode à l'autre.
        self.known_characters = list(known_characters)
        self.previous_tail = previous_tail
        #: Fiche rendue par le modèle pour le chapitre en cours, recueillie au parsing et
        #: reportée dans :class:`ChapterAnalysis` : c'est elle qui nourrira le chapitre suivant.
        self._last_characters: list[CharacterCard] = []
        self._client: Any | None = None
        self._manager: GeminiManager | None = None
        if client is not None:
            self._client = client
        elif manager is not None:
            self._manager = manager
        else:
            try:
                self._manager = GeminiManager(keys=api_key, preferred_model=self.model)
            except gm.GeminiManagerError as exc:
                raise AnalyzerError(str(exc)) from exc
        if self._manager is not None:
            self.model = self._manager.current_model
        self.n_calls = 0
        self.usage = [0, 0, 0]
        #: Secondes passées dans les appels Gemini (réponses, réessais, attente de quota) :
        #: le reste du temps d'analyse est du calcul local (encodage des images, parsing).
        self.api_seconds = 0.0
        #: Protège les compteurs partagés quand l'étape 2 appelle Gemini en parallèle.
        self._counters = threading.Lock()

    # -- configuration ------------------------------------------------------------------
    def config_for(self, system_instruction: str, schema: type[BaseModel]) -> types.GenerateContentConfig:
        """Configuration de génération pour une étape (schéma de réponse Pydantic)."""
        return types.GenerateContentConfig(
            system_instruction=system_instruction,
            response_mime_type="application/json",
            response_schema=schema,
            temperature=self.temperature,
            max_output_tokens=self.max_output_tokens,
            http_options=types.HttpOptions(timeout=self.timeout_ms),
            thinking_config=(
                types.ThinkingConfig(thinking_budget=self.thinking_budget) if self.thinking_budget is not None else None
            ),
            automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
        )

    @property
    def beats_config(self) -> types.GenerateContentConfig:
        return self.config_for(BEATS_SYSTEM_INSTRUCTION, BeatBatch)

    def _master_instruction(self, mode: str, ids: Sequence[int], meta: ChapterMeta | None) -> str:
        """Prompt maître pour ce mode, borné par les identifiants réellement envoyés."""
        return master_instruction(
            mode=mode,
            language=self.language,
            n_ids=len(ids),
            first_id=ids[0] if ids else 0,
            last_id=ids[-1] if ids else 0,
            meta=meta,
            max_key=self.max_key_panels,
            characters=self.known_characters,
            previous_tail=self.previous_tail,
        )

    def script_config(
        self, beats: Sequence[Beat], meta: ChapterMeta | None = None
    ) -> types.GenerateContentConfig:
        ids = [beat.index for beat in beats]
        return self.config_for(self._master_instruction("beats", ids, meta), ScriptDraft)

    def single_call_config(
        self, panels: Sequence[Panel], meta: ChapterMeta | None = None
    ) -> types.GenerateContentConfig:
        ids = [panel.index for panel in panels]
        return self.config_for(self._master_instruction("images", ids, meta), RecapDraft)

    @property
    def keyframes_config(self) -> types.GenerateContentConfig:
        return self.config_for(KEYFRAMES_SYSTEM_INSTRUCTION_TEMPLATE.format(max_key=self.max_key_panels), KeyframeBatch)

    # -- construction des prompts ---------------------------------------------------------
    def _image_parts(self, panel: Panel, *, thumbnail: bool = False) -> list[types.Part]:
        if thumbnail:
            slices = encode_panel_image(
                panel, max_width=min(self.max_image_width, KEYFRAME_IMAGE_WIDTH),
                max_slice_height=min(self.max_slice_height, KEYFRAME_SLICE_HEIGHT), max_slices=KEYFRAME_MAX_SLICES,
                quality=self.jpeg_quality,
            )
        else:
            slices = encode_panel_image(
                panel, max_width=self.max_image_width, max_slice_height=self.max_slice_height,
                quality=self.jpeg_quality,
            )
        parts = [types.Part.from_text(text=panel_caption(panel, len(slices)))]
        for k, (data, mime) in enumerate(slices, start=1):
            if len(slices) > 1:
                parts.append(types.Part.from_text(text=slice_caption(panel, k, len(slices))))
            parts.append(types.Part.from_bytes(data=data, mime_type=mime))
        return parts

    def build_beats_contents(
        self, batch: Sequence[Panel], context: Sequence[BeatDraft] = (), meta: ChapterMeta | None = None
    ) -> list[types.Part]:
        """``Part`` (texte + images) d'un lot pour l'extraction des beats (étape 1a)."""
        parts = [types.Part.from_text(text=build_beats_header(batch, context, meta))]
        for panel in batch:
            parts.extend(self._image_parts(panel))
        parts.append(types.Part.from_text(text=build_beats_footer(batch)))
        return parts

    def build_single_call_contents(self, panels: Sequence[Panel], meta: ChapterMeta | None = None) -> list[types.Part]:
        """``Part`` (texte + toutes les images du chapitre) du mode « une requête »."""
        parts = [types.Part.from_text(text=build_single_call_header(panels, meta, self.language))]
        for panel in panels:
            parts.extend(self._image_parts(panel))
        parts.append(types.Part.from_text(text=build_single_call_footer(panels, self.max_key_panels)))
        return parts

    def payload_bytes(self, panels: Sequence[Panel]) -> int:
        """Poids des images encodées d'un chapitre (octets), pour vérifier la limite inline."""
        return sum(
            len(data)
            for panel in panels
            for data, _ in encode_panel_image(
                panel, max_width=self.max_image_width, max_slice_height=self.max_slice_height,
                quality=self.jpeg_quality,
            )
        )

    def fit_for_single_call(
        self, panels: Sequence[Panel], *, limit: int = MAX_INLINE_PAYLOAD_BYTES
    ) -> bool:
        """Comprime les images jusqu'à ce que le chapitre tienne en une requête.

        Descend :data:`INLINE_PAYLOAD_LADDER` et s'arrête au premier palier qui passe sous
        ``limit``, en fixant ``max_image_width`` et ``jpeg_quality`` en conséquence.

        L'enjeu n'est pas le quota mais la **qualité du script** : en mode deux étapes, le
        modèle qui écrit ne voit aucune image, il ne fait que reformuler des résumés déjà
        aplatis par l'étape beats. Mieux vaut un JPEG plus compressé - les bulles de webtoon
        restent lisibles bien en dessous - qu'un récit écrit à l'aveugle.

        Returns:
            ``True`` si un palier convient (réglages appliqués), ``False`` s'il faut
            se rabattre sur le mode en deux étapes.
        """
        if not panels:
            return True
        for width, quality in INLINE_PAYLOAD_LADDER:
            self.max_image_width, self.jpeg_quality = width, quality
            payload = self.payload_bytes(panels)
            if payload <= limit:
                logger.info(
                    "Requete unique : %.1f Mo a %d px / qualite %d (plafond %.0f Mo)",
                    payload / 1_048_576, width, quality, limit / 1_048_576,
                )
                return True
            logger.info(
                "Palier %d px / qualite %d trop lourd (%.1f Mo) : compression suivante",
                width, quality, payload / 1_048_576,
            )
        logger.warning(
            "Chapitre trop lourd meme au palier le plus compresse (%d px / qualite %d) : "
            "retour au mode en deux etapes",
            *INLINE_PAYLOAD_LADDER[-1],
        )
        return False

    def build_script_contents(self, beats: Sequence[Beat], meta: ChapterMeta | None = None) -> list[types.Part]:
        """``Part`` (texte seul) de l'étape 1b."""
        return [types.Part.from_text(text=build_script_prompt(beats, meta, self.language))]

    def build_keyframes_contents(
        self,
        group: Sequence[tuple[int, str, list[int]]],
        panels_by_id: dict[int, Panel],
        meta: ChapterMeta | None = None,
    ) -> list[types.Part]:
        """``Part`` (paragraphes + images candidates) d'un groupe pour l'étape 2."""
        parts = [types.Part.from_text(text=build_keyframes_header(group, meta))]
        for index, text, candidates in group:
            parts.append(types.Part.from_text(text=paragraph_line(index, text, candidates)))
        parts.append(types.Part.from_text(text="Candidate panels:"))
        sent: set[int] = set()
        for _, _, candidates in group:
            for pid in candidates:
                if pid in sent or pid not in panels_by_id:
                    continue
                sent.add(pid)
                parts.extend(self._image_parts(panels_by_id[pid], thumbnail=True))
        parts.append(types.Part.from_text(text=build_keyframes_footer(group, self.max_key_panels)))
        return parts

    # -- appel API avec retries --------------------------------------------------------
    def _retry_delay(self, exc: Exception, attempt: int) -> float:
        delay = self.backoff * (2 ** (attempt - 1))
        if isinstance(exc, errors.ClientError) and exc.code == 429:
            delay = max(delay, RATE_LIMIT_MIN_DELAY, suggested_retry_delay(exc))
        return min(delay, MAX_BACKOFF)

    @property
    def manager(self) -> GeminiManager | None:
        """Gestionnaire Gemini partagé (``None`` avec un client injecté)."""
        return self._manager

    def _add_api_seconds(self, seconds: float) -> None:
        with self._counters:
            self.api_seconds += seconds

    def generate(self, contents: list[types.Part], config: types.GenerateContentConfig, parse: Callable[[Any], Any], label: str) -> Any:
        """Appelle le modèle et applique ``parse`` (retries sur erreurs transitoires / réponses invalides).

        Avec un :class:`GeminiManager`, le transport (429, 5xx, clés, cascade) est délégué au
        gestionnaire ; seules les réponses invalides sont réessayées ici.
        """
        if self._manager is not None:
            return self._generate_via_manager(contents, config, parse, label)
        attempts = self.max_retries + 1
        last_error: Exception | None = None
        truncations = 0
        for attempt in range(1, attempts + 1):
            try:
                if self.n_calls and self.delay_between_batches > 0:
                    _sleep(self.delay_between_batches)
                self.n_calls += 1
                call_started = time.perf_counter()
                response = self._client.models.generate_content(model=self.model, contents=contents, config=config)
                self.api_seconds += time.perf_counter() - call_started
                for i, value in enumerate(_usage_of(response)):
                    self.usage[i] += value
                return parse(response)
            except Exception as exc:  # noqa: BLE001 - tri retryable / definitif ci-dessous
                if not _is_retryable(exc):
                    if isinstance(exc, AnalyzerError):
                        raise
                    if is_daily_quota_error(exc):
                        raise QuotaExhaustedError(f"{label} : {QUOTA_HINT} ({self.model})") from exc
                    raise AnalyzerError(f"Erreur definitive Gemini ({label}) : {type(exc).__name__}: {exc}") from exc
                if isinstance(exc, TruncatedResponseError):
                    # Reessayer une troncature a l'identique la reproduit presque toujours :
                    # insister brulerait du quota pour rien.
                    truncations += 1
                    if truncations > TRUNCATION_RETRIES:
                        raise
                last_error = exc
                logger.warning("%s : tentative %d/%d echouee (%s: %s)", label, attempt, attempts, type(exc).__name__, exc)
                if attempt < attempts:
                    _sleep(self._retry_delay(exc, attempt))
        raise AnalyzerError(f"Abandon ({label}) apres {attempts} tentative(s) : {type(last_error).__name__}: {last_error}") from last_error

    def _generate_via_manager(self, contents: list[types.Part], config: types.GenerateContentConfig, parse: Callable[[Any], Any], label: str) -> Any:
        assert self._manager is not None
        attempts = self.max_retries + 1
        last_error: Exception | None = None
        truncations = 0
        for attempt in range(1, attempts + 1):
            with self._counters:
                first_call = self.n_calls == 0
                self.n_calls += 1
            if not first_call and self.delay_between_batches > 0:
                _sleep(self.delay_between_batches)  # delai force entre deux envois (TPM)
            call_started = time.perf_counter()
            try:
                response = self._manager.generate(contents, config, label=label)
            except gm.QuotaExhaustedError as exc:
                self._add_api_seconds(time.perf_counter() - call_started)
                raise QuotaExhaustedError(str(exc)) from exc  # le gestionnaire inclut deja le libelle
            except gm.GeminiManagerError as exc:
                self._add_api_seconds(time.perf_counter() - call_started)
                raise AnalyzerError(f"Erreur definitive Gemini : {exc}") from exc
            self._add_api_seconds(time.perf_counter() - call_started)
            self.model = self._manager.current_model
            with self._counters:
                for i, value in enumerate(_usage_of(response)):
                    self.usage[i] += value
            try:
                return parse(response)
            except InvalidResponseError as exc:
                if isinstance(exc, TruncatedResponseError):
                    # Meme regle que la boucle a client injecte : une troncature ne se
                    # rejoue qu'une fois, sinon on paie plusieurs unites de quota pour
                    # obtenir exactement la meme coupure.
                    truncations += 1
                    if truncations > TRUNCATION_RETRIES:
                        raise
                last_error = exc
                logger.warning("%s : reponse invalide (tentative %d/%d) : %s", label, attempt, attempts, exc)
        raise AnalyzerError(f"Abandon ({label}) apres {attempts} tentative(s) : {type(last_error).__name__}: {last_error}") from last_error

    # -- etapes --------------------------------------------------------------------------
    def extract_beats(
        self, panels: Sequence[Panel], meta: ChapterMeta | None = None, checkpoint: AnalysisCheckpoint | None = None
    ) -> list[Beat]:
        """Étape 1a : beats de tout le chapitre, lot par lot (lots déjà obtenus relus du point de reprise)."""
        beats: list[Beat] = []
        context: list[BeatDraft] = []
        batches = make_batches(panels, self.batch_size)
        for b_index, batch in enumerate(batches):
            batch_ids = [panel.index for panel in batch]
            drafts = checkpoint.beats_for(b_index, batch_ids) if checkpoint is not None else None
            if drafts is None:
                logger.info("Etape 1a - lot %d/%d : cases %s", b_index + 1, len(batches), batch_ids)
                contents = self.build_beats_contents(batch, context[-CONTEXT_BEATS:], meta)
                drafts = self.generate(
                    contents, self.beats_config, lambda r, ids=batch_ids: normalize_beats(parse_beats(r), ids), f"beats lot {b_index + 1}",
                )
                if checkpoint is not None:
                    checkpoint.store_beats(b_index, batch_ids, drafts)
            else:
                logger.info("Etape 1a - lot %d/%d : cases %s (repris)", b_index + 1, len(batches), batch_ids)
            for draft in drafts:
                beats.append(Beat(index=len(beats), **draft.model_dump()))
            context.extend(drafts)
        logger.info("Etape 1a : %d beat(s) dont %d de remplissage", len(beats), sum(1 for b in beats if b.is_filler))
        return beats

    def write_script(self, beats: Sequence[Beat], meta: ChapterMeta | None = None) -> list[ParagraphDraft]:
        """Étape 1b : script global, régénéré une fois si des formules visuelles subsistent."""
        contents = self.build_script_contents(beats, meta)
        config = self.script_config(beats, meta)
        def parse(response: Any) -> list[ParagraphDraft]:
            draft = parse_master_script(response)
            self._remember_characters(draft.characters)
            return normalize_script(draft.paragraphs, beats)

        paragraphs = self.generate(contents, config, parse, "script")
        violations = [hit for p in paragraphs for hit in find_forbidden(p.text)]
        intro = generic_intro(paragraphs[0].text) if paragraphs else None
        if violations or intro:
            logger.warning(
                "Script : %d formule(s) visuelle(s) %s%s, regeneration",
                len(violations), violations[:6], f" ; intro generique {intro[:60]!r}" if intro else "",
            )
            problems = [f'"{v}"' for v in violations[:10]]
            if intro:
                problems.append(f'generic opening "{intro[:80]}"')
            reminder = (
                "Your previous script broke the rules: " + "; ".join(problems)
                + ". Rewrite the whole script as pure narrated storytelling without any reference to "
                "panels, images, scenes shown or the act of seeing, and open directly with a strong hook "
                "from the story (no greeting, no 'in this chapter'). Same structure, same beats."
            )
            retry_contents = [*contents, types.Part.from_text(text=reminder)]
            try:
                paragraphs = self.generate(retry_contents, config, lambda r: normalize_script(parse_script(r), beats), "script (retry)")
            except AnalyzerError as exc:
                logger.warning("Regeneration du script impossible (%s) : nettoyage local", exc)
            violations = [hit for p in paragraphs for hit in find_forbidden(p.text)]
            if violations:
                logger.warning("Script : formules encore presentes %s, nettoyage local", violations[:6])
                paragraphs = [p.model_copy(update={"text": scrub_forbidden(p.text)}) for p in paragraphs]
        paragraphs = enforce_script_conventions(paragraphs, language=self.language, cta_text=self.cta_text)
        words = sum(len(p.text.split()) for p in paragraphs)
        logger.info("Etape 1b : %d paragraphe(s), %d mots", len(paragraphs), words)
        return paragraphs

    def candidate_panels(self, paragraph: ParagraphDraft, beats: Sequence[Beat], panels_by_id: dict[int, Panel]) -> list[int]:
        """Cases candidates d'un paragraphe : celles de ses beats (les plus grandes si trop nombreuses)."""
        beat_by_index = {beat.index: beat for beat in beats}
        candidates = [
            pid
            for bid in paragraph.beat_ids
            for pid in beat_by_index[bid].panel_ids
            if bid in beat_by_index and pid in panels_by_id
        ]
        if len(candidates) > MAX_CANDIDATES_PER_PARAGRAPH:
            tallest = sorted(candidates, key=lambda p: -panels_by_id[p].height)[:MAX_CANDIDATES_PER_PARAGRAPH]
            candidates = sorted(tallest)
        return candidates

    def select_keyframes(
        self,
        paragraphs: Sequence[ParagraphDraft],
        beats: Sequence[Beat],
        panels: Sequence[Panel],
        meta: ChapterMeta | None = None,
        checkpoint: AnalysisCheckpoint | None = None,
    ) -> list[Scene]:
        """Étape 2 : cases clés par paragraphe (groupes de paragraphes ≤ :data:`MAX_BATCH_SIZE` images)."""
        panels_by_id = {panel.index: panel for panel in panels}
        heights = {panel.index: panel.height for panel in panels}
        entries = [(i, p.text, self.candidate_panels(p, beats, panels_by_id)) for i, p in enumerate(paragraphs)]
        groups: list[list[tuple[int, str, list[int]]]] = []
        current: list[tuple[int, str, list[int]]] = []
        count = 0
        for entry in entries:
            if current and count + len(entry[2]) > MAX_BATCH_SIZE:
                groups.append(current)
                current, count = [], 0
            current.append(entry)
            count += len(entry[2])
        if current:
            groups.append(current)

        # Les groupes sont independants : leurs appels partent en parallele. La normalisation
        # (qui empeche deux paragraphes de reutiliser la meme case) reste sequentielle, donc
        # le resultat est identique a une execution serie.
        todo: list[int] = []
        choices_by_group: dict[int, list[KeyframeChoice]] = {}
        for g_index, group in enumerate(groups):
            paragraph_indexes = [i for i, _, _ in group]
            if all(not candidates for _, _, candidates in group):
                continue
            stored = checkpoint.choices_for(g_index, paragraph_indexes) if checkpoint is not None else None
            if stored is None:
                todo.append(g_index)
            else:
                logger.info("Etape 2 - groupe %d/%d : paragraphes %s (repris)", g_index + 1, len(groups), paragraph_indexes)
                choices_by_group[g_index] = stored

        def fetch(g_index: int) -> list[KeyframeChoice]:
            group = groups[g_index]
            logger.info("Etape 2 - groupe %d/%d : paragraphes %s", g_index + 1, len(groups), [i for i, _, _ in group])
            contents = self.build_keyframes_contents(group, panels_by_id, meta)
            return self.generate(contents, self.keyframes_config, parse_keyframes, f"cases cles groupe {g_index + 1}")

        failure: Exception | None = None
        if todo:
            workers = min(self.keyframe_workers, len(todo))
            if workers > 1:
                logger.info("Etape 2 : %d groupe(s) a demander, %d en parallele", len(todo), workers)
                with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="keyframes") as pool:
                    for g_index, result in zip(todo, pool.map(_capture(fetch), todo)):
                        if isinstance(result, Exception):
                            failure = failure or result
                        else:
                            choices_by_group[g_index] = result
            else:
                for g_index in todo:
                    try:
                        choices_by_group[g_index] = fetch(g_index)
                    except Exception as exc:  # noqa: BLE001 - consigne puis releve apres sauvegarde
                        failure = exc
                        break

        # Chaque groupe obtenu est consigne, meme si un autre a echoue : aucun appel perdu.
        if checkpoint is not None:
            for g_index in sorted(set(todo) & set(choices_by_group)):
                checkpoint.store_choices(g_index, [i for i, _, _ in groups[g_index]], choices_by_group[g_index])
        if failure is not None:
            raise failure

        used: set[int] = set()
        keyframes: dict[int, list[int]] = {}
        action_heavy: dict[int, list[int]] = {}
        for g_index, group in enumerate(groups):
            choices = choices_by_group.get(g_index)
            if choices is None:
                continue
            chosen = normalize_keyframes(choices, group, heights, used, max_key=self.max_key_panels)
            keyframes.update(chosen)
            action_heavy.update(normalize_action_heavy(choices, chosen))

        scenes: list[Scene] = []
        for i, paragraph in enumerate(paragraphs):
            key = keyframes.get(i, [])
            if not key:
                logger.warning("Paragraphe %d sans case candidate : ignore", i)
                continue
            scenes.append(
                Scene(
                    index=len(scenes), panel_ids=key, narration=paragraph.text, emotion=paragraph.emotion,
                    is_filler=False, beat_ids=list(paragraph.beat_ids), action_heavy_ids=action_heavy.get(i, []),
                )
            )
        n_heavy = sum(len(s.action_heavy_ids) for s in scenes)
        logger.info(
            "Etape 2 : %d scene(s), %d case(s) cle(s) sur %d dont %d action_heavy", len(scenes), len(used), len(panels), n_heavy,
        )
        return scenes

    def write_recap(self, panels: Sequence[Panel], meta: ChapterMeta | None = None) -> list[Scene]:
        """Mode « une requête » : script **et** cases clés en un seul appel.

        Les garde-fous du mode en deux étapes sont conservés : formules visuelles interdites
        (régénération une fois puis nettoyage local), accroche obligatoire, appel à
        l'abonnement unique et bien placé, cases clés valides et jamais réutilisées.
        """
        contents = self.build_single_call_contents(panels, meta)
        config = self.single_call_config(panels, meta)
        def parse(response: Any) -> list[tuple[RecapParagraph, list[int], list[int]]]:
            draft = parse_master_recap(response)
            self._remember_characters(draft.characters)
            return normalize_recap(draft.paragraphs, panels, max_key=self.max_key_panels)

        entries = self.generate(contents, config, parse, "recap complet")

        paragraphs = [draft for draft, _, _ in entries]
        violations = [hit for p in paragraphs for hit in find_forbidden(p.text)]
        intro = generic_intro(paragraphs[0].text) if paragraphs else None
        # Seule la *qualité* déclenche une reprise : la longueur du script n'est plus un critère,
        # cinq essais mesurés ayant montré que Gemini n'obéit pas à une cible de mots.
        if violations or intro:
            logger.warning(
                "Recap : %d formule(s) visuelle(s) %s%s, regeneration",
                len(violations), violations[:6], f" ; intro generique {intro[:60]!r}" if intro else "",
            )
            problems = [f'"{v}"' for v in violations[:10]]
            if intro:
                problems.append(f'generic opening "{intro[:80]}"')
            reminder = (
                "Your previous recap broke the rules: " + "; ".join(problems)
                + ". Rewrite the whole recap as pure narrated storytelling without any reference to "
                "panels, images, scenes shown or the act of seeing, and open inside the action with a "
                "gesture or a quoted line (no greeting, no 'in this chapter', no pitch of the series). "
                "Keep the same plan, the same weights and the same key panel numbers."
            )
            try:
                entries = self.generate(
                    [*contents, types.Part.from_text(text=reminder)], config, parse, "recap complet (retry)",
                )
                paragraphs = [draft for draft, _, _ in entries]
            except AnalyzerError as exc:
                logger.warning("Regeneration du recap impossible (%s) : nettoyage local", exc)
            if any(find_forbidden(p.text) for p in paragraphs):
                paragraphs = [p.model_copy(update={"text": scrub_forbidden(p.text)}) for p in paragraphs]

        paragraphs = enforce_script_conventions(paragraphs, language=self.language, cta_text=self.cta_text)
        scenes = [
            Scene(
                index=i, panel_ids=keys, narration=paragraph.text, emotion=paragraph.emotion,
                is_filler=False, beat_ids=[], action_heavy_ids=heavy,
            )
            for i, (paragraph, (_, keys, heavy)) in enumerate(zip(paragraphs, entries))
        ]
        words = sum(len(s.narration.split()) for s in scenes)
        n_keys = sum(len(s.panel_ids) for s in scenes)
        logger.info(
            "Mode une requete : %d paragraphe(s), %d mots, %d case(s) cle(s) sur %d, %d action_heavy",
            len(scenes), words, n_keys, len(panels), sum(len(s.action_heavy_ids) for s in scenes),
        )
        return scenes

    def analyze_panels_single_call(self, panels: Sequence[Panel], meta: ChapterMeta | None = None) -> ChapterAnalysis:
        """Analyse complète d'un chapitre en **un seul appel** Gemini.

        Raises:
            ValueError: aucune case ou numéros dupliqués.
            AnalyzerError: échec définitif de l'appel.
        """
        panels = _check_panels(panels)
        self.n_calls = 0
        self.usage = [0, 0, 0]
        self.api_seconds = 0.0
        logger.info("Analyse de %d case(s) avec %s (une seule requete)", len(panels), self.model)
        scenes = self.write_recap(panels, meta)
        return self._chapter_analysis(scenes, [], panels, meta)

    def analyze_panels(
        self, panels: Sequence[Panel], meta: ChapterMeta | None = None, *, checkpoint: str | Path | None = None
    ) -> ChapterAnalysis:
        """Enchaîne les deux étapes et renvoie le script illustré du chapitre.

        Args:
            panels: cases du chapitre (numéros uniques).
            meta: métadonnées du chapitre (titres, URL).
            checkpoint: fichier de reprise (:class:`AnalysisCheckpoint`) : chaque appel réussi y
                est consigné ; une analyse interrompue (quota) repart de là, même avec un autre
                modèle. Le fichier est supprimé quand l'analyse aboutit.

        Raises:
            ValueError: aucune case ou numéros de cases dupliqués.
            AnalyzerError: échec définitif d'un appel.
        """
        panels = _check_panels(panels)
        indexes = [panel.index for panel in panels]
        self.n_calls = 0
        self.usage = [0, 0, 0]
        self.api_seconds = 0.0
        logger.info("Analyse de %d case(s) avec %s (generation en deux etapes)", len(panels), self.model)
        ckpt = AnalysisCheckpoint(checkpoint, panel_ids=indexes, batch_size=self.batch_size, language=self.language)

        beats = self.extract_beats(panels, meta, ckpt)
        paragraphs = ckpt.paragraphs()
        if paragraphs is None:
            paragraphs = self.write_script(beats, meta)
            ckpt.store_paragraphs(paragraphs)
        else:
            logger.info("Etape 1b : script repris du point de reprise (%d paragraphes)", len(paragraphs))
        scenes = self.select_keyframes(paragraphs, beats, panels, meta, ckpt)
        if not scenes:
            raise AnalyzerError("Aucune scene produite : le script n'a pu etre associe a aucune case")
        if ckpt.reused:
            logger.info("Analyse terminee : %d appel(s) repris du point de reprise, %d nouveau(x)", ckpt.reused, self.n_calls)
        ckpt.clear()

        return self._chapter_analysis(scenes, beats, panels, meta)

    def _remember_characters(self, cards: Sequence[CharacterCard]) -> None:
        """Retient la fiche rendue par le modèle, sans doublon de nom (dernière vue gagne)."""
        merged: dict[str, CharacterCard] = {}
        for card in cards:
            name = card.name.strip()
            if name:
                merged[name.casefold()] = card
        self._last_characters = list(merged.values())

    def _chapter_analysis(
        self, scenes: list[Scene], beats: list[Beat], panels: Sequence[Panel], meta: ChapterMeta | None
    ) -> ChapterAnalysis:
        """Assemble le résultat commun aux deux modes d'analyse."""
        return ChapterAnalysis(
            series_title=meta.series_title if meta else "",
            episode_title=meta.episode_title if meta else "",
            source_url=meta.url if meta else "",
            model=self.model,
            language=self.language,
            n_panels=len(panels),
            n_batches=self.n_calls,
            scenes=scenes,
            beats=beats,
            characters=self._last_characters,
            prompt_tokens=self.usage[0],
            output_tokens=self.usage[1],
            thinking_tokens=self.usage[2],
        )


# --- Sorties disque -----------------------------------------------------------------
def save_analysis(analysis: ChapterAnalysis, path: str | Path) -> Path:
    """Écrit l'analyse en JSON (UTF-8, indenté) et renvoie le chemin."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(analysis.model_dump_json(indent=2), encoding="utf-8")
    logger.info("Analyse enregistree : %s (%d scene(s))", path, analysis.n_scenes)
    return path


def load_analysis(path: str | Path) -> ChapterAnalysis:
    """Relit une analyse écrite par :func:`save_analysis`."""
    return ChapterAnalysis.model_validate_json(Path(path).read_text(encoding="utf-8"))


def format_scenes(analysis: ChapterAnalysis) -> str:
    """Rendu ASCII multi-lignes du script illustré (console Windows cp1252)."""
    lines = [
        "=" * 72,
        f"Serie      : {analysis.series_title or '-'}",
        f"Episode    : {analysis.episode_title or '-'}",
        f"Modele     : {analysis.model}  (langue {analysis.language})",
        f"Cases      : {analysis.n_panels} soumises, {analysis.n_key_panels} cles montees, "
        f"{len(analysis.beats)} beats, {analysis.n_batches} appel(s)",
        f"Scenes     : {analysis.n_scenes} dont {analysis.n_filler} de remplissage (filler), "
        f"{analysis.script_words} mots",
        f"Tokens     : {analysis.prompt_tokens} in / {analysis.output_tokens} out / "
        f"{analysis.thinking_tokens} reflexion",
        "-" * 72,
    ]
    for scene in analysis.scenes:
        ids = ",".join(str(pid) for pid in scene.panel_ids)
        tag = " [filler]" if scene.is_filler else ""
        lines.append(f"[{scene.index:>3}] cases {ids:<12} ({scene.emotion}){tag}")
        lines.append(f"      {scene.narration}")
    lines.append("=" * 72)
    return "\n".join(lines).encode("ascii", "replace").decode("ascii")


__all__ = [
    "DEFAULT_MODEL",
    "MODEL_ENV_VAR",
    "DEFAULT_BATCH_SIZE",
    "MAX_BATCH_SIZE",
    "DEFAULT_LANGUAGE",
    "DEFAULT_MAX_IMAGE_WIDTH",
    "DEFAULT_MAX_SLICE_HEIGHT",
    "MAX_SLICES_PER_PANEL",
    "CONTEXT_SCENES",
    "CONTEXT_BEATS",
    "MAX_KEY_PANELS_PER_PARAGRAPH",
    "MAX_CANDIDATES_PER_PARAGRAPH",
    "KEYFRAME_IMAGE_WIDTH",
    "KEYFRAME_SLICE_HEIGHT",
    "KEYFRAME_MAX_SLICES",
    "QuotaExhaustedError",
    "QUOTA_HINT",
    "BATCH_DELAY_S",
    "DEFAULT_KEYFRAME_WORKERS",
    "DEFAULT_MAX_OUTPUT_TOKENS",
    "TRUNCATION_RETRIES",
    "MAX_INT_DIGITS",
    "RUNAWAY_INT",
    "MAX_INLINE_PAYLOAD_BYTES",
    "build_single_call_header",
    "build_single_call_footer",
    "parse_recap",
    "parse_master_recap",
    "parse_master_script",
    "normalize_recap",
    "AnalysisCheckpoint",
    "is_daily_quota_error",
    "suggested_retry_delay",
    "FORBIDDEN_PATTERNS",
    "GENERIC_INTRO_PATTERNS",
    "CTA_PATTERN",
    "CTA_TEXTS",
    "CTA_POSITION",
    "split_sentences",
    "generic_intro",
    "cta_text_for",
    "cta_paragraph_index",
    "enforce_script_conventions",
    "BEATS_SYSTEM_INSTRUCTION",
    "MASTER_PROMPT_TEMPLATE",
    "INPUT_BLOCK_IMAGES",
    "INPUT_BLOCK_BEATS",
    "INPUT_BLOCKS",
    "master_instruction",
    "character_sheet",
    "PEAK_MAX_SPAN",
    "KEYFRAMES_SYSTEM_INSTRUCTION_TEMPLATE",
    "AnalyzerError",
    "InvalidResponseError",
    "TruncatedResponseError",
    "GeminiAnalyzer",
    "make_batches",
    "encode_panel_image",
    "build_beats_header",
    "build_beats_footer",
    "build_script_prompt",
    "build_keyframes_header",
    "build_keyframes_footer",
    "paragraph_line",
    "panel_caption",
    "slice_caption",
    "coerce_emotion",
    "coerce_bool",
    "response_data",
    "parse_scene_json",
    "parse_response",
    "parse_beats",
    "parse_script",
    "parse_keyframes",
    "normalize_partition",
    "normalize_scenes",
    "normalize_beats",
    "normalize_script",
    "normalize_keyframes",
    "normalize_action_heavy",
    "find_forbidden",
    "scrub_forbidden",
    "resolve_model",
    "suggested_retry_delay",
    "build_client",
    "save_analysis",
    "load_analysis",
    "format_scenes",
]
