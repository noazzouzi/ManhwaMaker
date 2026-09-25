"""Mémoire de série : ce qu'un chapitre transmet au suivant.

Sans elle, chaque chapitre est analysé à l'aveugle : Gemini ne voit que les cases de
l'épisode en cours, invente un nom pour le héros, et en change à l'épisode suivant. Sur
une série de cinquante chapitres, le spectateur suit trois personnages qui portent huit
noms. C'est le seul défaut que le prompt maître ne peut pas corriger tout seul.

Deux informations voyagent, écrites dans ``characters.json`` à côté de ``scenes.json`` :

- la **fiche des personnages** (:class:`~src.models.scene.CharacterCard`) : nom canonique,
  autres appellations rencontrées, et qui il est ;
- la **fin du chapitre** (``tail``) : les dernières phrases narrées, pour que l'épisode
  suivant enchaîne au lieu de repartir de zéro.

Aucun appel Gemini ici : tout se lit sur le disque, dans le dossier de sortie partagé.
La fusion se fait **à la lecture**, jamais à l'écriture : un chapitre n'écrit que ce qu'il
a lui-même produit, donc le relancer (``--redo``) ne corrompt pas la mémoire des autres, et
deux chapitres traités en parallèle n'écrivent jamais le même fichier.
"""

from __future__ import annotations

import json
import logging
import os
import re
from collections import Counter
from pathlib import Path
from typing import Iterable, Sequence

from src.models.chapter import ChapterMeta
from src.models.scene import ChapterAnalysis, CharacterCard

logger = logging.getLogger(__name__)

#: Nom du fichier écrit dans le dossier de chaque chapitre, à côté de ``scenes.json``.
CHARACTERS_FILE: str = "characters.json"
#: Nombre de fiches maximum envoyées au modèle. Au-delà, le prompt se dilue et les
#: figurants d'un seul chapitre noient les personnages principaux.
MAX_SHEET_CARDS: int = 20
#: Longueur maximale du « qui est-ce » d'une fiche (caractères).
MAX_WHO_CHARS: int = 160
#: Longueur maximale du rappel de fin de chapitre (caractères).
MAX_TAIL_CHARS: int = 320
#: Nombre maximal d'autres appellations conservées par personnage.
MAX_ALIASES: int = 6

_NON_ALNUM = re.compile(r"[^a-z0-9]+")


def series_key(meta: ChapterMeta | None) -> str:
    """Clé stable identifiant la série, pour ne pas mélanger deux séries du même dossier.

    ``title_no`` est l'identifiant Webtoons : il ne bouge jamais. Le titre n'est utilisé
    qu'en repli (cases venant d'un dossier local, sans URL).
    """
    if meta is None:
        return ""
    if meta.title_no is not None:
        return f"t{meta.title_no}"
    slug = _NON_ALNUM.sub("-", (meta.series_title or "").lower()).strip("-")
    return f"s-{slug}" if slug else ""


def _trim(text: str, limit: int) -> str:
    """Texte réduit à ``limit`` caractères, coupé sur un mot et suffixé de « ... »."""
    cleaned = " ".join(str(text).split())
    if len(cleaned) <= limit:
        return cleaned
    cut = cleaned[:limit].rsplit(" ", 1)[0]
    return f"{cut or cleaned[:limit]}..."


def chapter_tail(analysis: ChapterAnalysis) -> str:
    """Fin du chapitre en clair : le dernier paragraphe narré, remplissage exclu.

    Dérivée des scènes plutôt que stockée : il n'y a ainsi qu'une seule vérité, et les
    ``scenes.json`` déjà sur le disque donnent leur fin sans migration.
    """
    for scene in reversed(analysis.story_scenes()):
        if scene.narration.strip():
            return _trim(scene.narration, MAX_TAIL_CHARS)
    return ""


def save_chapter_sheet(
    analysis: ChapterAnalysis, out_dir: str | Path, meta: ChapterMeta | None = None
) -> Path | None:
    """Écrit ``characters.json`` dans le dossier du chapitre ; ``None`` si rien à retenir.

    L'écriture est atomique (fichier temporaire puis :func:`os.replace`, comme
    ``batch_status.json``) : un lot interrompu ne laisse pas de fiche tronquée derrière lui.
    """
    key = series_key(meta)
    episode_no = meta.episode_no if meta is not None else None
    if not key or episode_no is None:
        logger.debug("Fiche de serie non ecrite : serie ou episode inconnu (%s, %s)", key, episode_no)
        return None

    cards = [
        {
            "name": card.name.strip(),
            "also_called": [alias.strip() for alias in card.also_called if alias.strip()][:MAX_ALIASES],
            "who": _trim(card.who, MAX_WHO_CHARS),
        }
        for card in analysis.characters
        if card.name.strip()
    ]
    payload = {
        "series": key,
        "series_title": analysis.series_title,
        "episode_no": episode_no,
        "episode_title": analysis.episode_title,
        "characters": cards,
        "tail": chapter_tail(analysis),
    }

    path = Path(out_dir) / CHARACTERS_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, path)
    logger.info("Fiche de serie enregistree : %s (%d personnage(s))", path, len(cards))
    return path


def _read_sheet(path: Path) -> dict | None:
    """Une fiche relue, ou ``None`` si le fichier est illisible ou mal formé.

    Un fichier corrompu (lot tué en plein vol, disque plein) ne doit jamais faire échouer
    un chapitre : on l'ignore et on continue avec les autres.
    """
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        logger.warning("Fiche de serie illisible, ignoree : %s (%s)", path, exc)
        return None
    if not isinstance(data, dict) or not isinstance(data.get("characters"), list):
        logger.warning("Fiche de serie mal formee, ignoree : %s", path)
        return None
    if not isinstance(data.get("episode_no"), int):
        return None
    return data


def merge_cards(sheets: Iterable[dict]) -> list[CharacterCard]:
    """Fusionne les fiches de plusieurs chapitres, du plus ancien au plus récent.

    Règles :

    - le **premier** chapitre où un personnage apparaît fixe son nom canonique et son
      « qui est-ce » ; c'est ce nom que le modèle devra réemployer ;
    - les autres appellations s'additionnent d'un chapitre à l'autre ;
    - un personnage qui réapparaît sous une appellation déjà connue rejoint sa fiche au
      lieu d'en créer une deuxième.
    """
    by_key: dict[str, CharacterCard] = {}
    alias_to_key: dict[str, str] = {}

    for sheet in sheets:
        for raw in sheet.get("characters", []):
            if not isinstance(raw, dict):
                continue
            name = str(raw.get("name", "")).strip()
            if not name:
                continue
            aliases = [str(a).strip() for a in raw.get("also_called", []) if str(a).strip()]
            key = alias_to_key.get(name.casefold(), name.casefold())
            card = by_key.get(key)
            if card is None:
                card = CharacterCard(name=name, also_called=[], who=_trim(raw.get("who", ""), MAX_WHO_CHARS))
                by_key[key] = card
                alias_to_key[name.casefold()] = key
            known = {card.name.casefold(), *(a.casefold() for a in card.also_called)}
            for alias in aliases:
                if alias.casefold() not in known and len(card.also_called) < MAX_ALIASES:
                    card.also_called.append(alias)
                    known.add(alias.casefold())
                alias_to_key.setdefault(alias.casefold(), key)
    return list(by_key.values())


def _rank_cards(cards: Sequence[CharacterCard], sheets: Sequence[dict]) -> list[CharacterCard]:
    """Trie les fiches par nombre de chapitres où le personnage apparaît (décroissant).

    Au-delà de :data:`MAX_SHEET_CARDS`, il faut choisir. Un personnage présent dans dix
    chapitres compte plus qu'un figurant vu une fois : le récurrent est justement celui
    dont le nom doit rester stable. À égalité, l'ordre d'apparition tranche.
    """
    seen: Counter[str] = Counter()
    for sheet in sheets:
        names = {
            str(raw.get("name", "")).strip().casefold()
            for raw in sheet.get("characters", [])
            if isinstance(raw, dict) and str(raw.get("name", "")).strip()
        }
        seen.update(names)
    order = {id(card): i for i, card in enumerate(cards)}

    def score(card: CharacterCard) -> tuple[int, int]:
        hits = max(seen[card.name.casefold()], *(seen[a.casefold()] for a in card.also_called), 0)
        return (-hits, order[id(card)])

    return sorted(cards, key=score)


def load_series_context(
    series_root: str | Path, *, key: str, before_episode: int | None
) -> tuple[list[CharacterCard], str]:
    """``(fiches, fin du chapitre précédent)`` des épisodes **antérieurs** de cette série.

    Args:
        series_root: dossier contenant un sous-dossier par chapitre (typiquement ``output/``).
        key: clé de série (:func:`series_key`) ; les autres séries sont ignorées.
        before_episode: numéro du chapitre en cours. Les épisodes de numéro supérieur ou
            égal sont exclus - sans quoi un ``--redo`` ferait lire au chapitre sa propre
            fiche, et le modèle se citerait lui-même.

    Renvoie deux listes vides si rien n'est disponible : c'est le cas normal du premier
    épisode, pas une erreur.
    """
    if not key or before_episode is None:
        return [], ""
    root = Path(series_root)
    if not root.is_dir():
        return [], ""

    sheets: list[dict] = []
    for path in sorted(root.glob(f"*/{CHARACTERS_FILE}")):
        data = _read_sheet(path)
        if data is None or data.get("series") != key:
            continue
        if data["episode_no"] >= before_episode:
            continue
        sheets.append(data)
    if not sheets:
        return [], ""

    # Les fiches peuvent avoir été écrites dans le désordre (lot parallèle, reprise) :
    # c'est le numéro d'épisode qui fait foi, pas la date du fichier.
    sheets.sort(key=lambda d: d["episode_no"])
    cards = _rank_cards(merge_cards(sheets), sheets)[:MAX_SHEET_CARDS]
    tail = _trim(str(sheets[-1].get("tail", "")), MAX_TAIL_CHARS)
    logger.info(
        "Memoire de serie %s : %d personnage(s) repris de %d chapitre(s) anterieur(s)",
        key, len(cards), len(sheets),
    )
    return cards, tail


__all__ = [
    "CHARACTERS_FILE",
    "MAX_ALIASES",
    "MAX_SHEET_CARDS",
    "MAX_TAIL_CHARS",
    "MAX_WHO_CHARS",
    "chapter_tail",
    "load_series_context",
    "merge_cards",
    "save_chapter_sheet",
    "series_key",
]
