"""Configuration transverse : racine du projet, langues par défaut, clés API Gemini.

Convention du projet : **l'anglais est la langue par défaut** partout (chapitres
Webtoons ``/en/``, narration générée, voix off), surchargeable par paramètre.

Les clés Gemini sont cherchées dans cet ordre (:func:`gemini_api_keys`) :

1. valeur(s) explicite(s) passée(s) par l'appelant ;
2. variable d'environnement ``GEMINI_API_KEYS`` (plusieurs clés séparées par des
   virgules, points-virgules ou retours à la ligne), lue aussi depuis le fichier
   ``.env`` à la racine du projet (ignoré par git) ;
3. variables d'environnement ``GEMINI_API_KEY`` puis ``GOOGLE_API_KEY`` (une clé) ;
4. fichier ``.gemini_key`` à la racine du projet (ignoré par git), une ou
   plusieurs clés (une par ligne ou séparées par des virgules).

Les clés ne sont jamais journalisées en clair (voir ``mask_key`` du gestionnaire).
"""

from __future__ import annotations

import os
import re
from collections.abc import Iterable
from pathlib import Path

#: Racine du projet (dossier contenant ``src/``, ``tests/``, ``output/``).
PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]

#: Langue par défaut des chapitres Webtoons (segment ``/en/`` des URL).
DEFAULT_SITE_LANGUAGE: str = "en"
#: Langue par défaut de la narration générée et de la voix off.
DEFAULT_NARRATION_LANGUAGE: str = "en"

#: Variable d'environnement listant plusieurs clés Gemini (rotation).
GEMINI_KEYS_ENV_VAR: str = "GEMINI_API_KEYS"
#: Variables d'environnement acceptées pour une clé Gemini unique, par ordre de priorité.
GEMINI_KEY_ENV_VARS: tuple[str, ...] = ("GEMINI_API_KEY", "GOOGLE_API_KEY")
#: Fichier local (non versionné) contenant la ou les clés Gemini.
GEMINI_KEY_FILE: Path = PROJECT_ROOT / ".gemini_key"
#: Fichier ``.env`` local (non versionné) : ``GEMINI_API_KEYS=cle1,cle2``.
DOTENV_FILE: Path = PROJECT_ROOT / ".env"

_KEY_SEPARATORS = re.compile(r"[,;\s]+")
_DOTENV_LINE = re.compile(r"^\s*(?:export\s+)?([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*?)\s*$")


def load_dotenv(path: str | Path | None = None, *, override: bool = False) -> dict[str, str]:
    """Charge un fichier ``.env`` minimal (``CLE=valeur``, guillemets et ``#`` commentaires).

    Les variables déjà présentes dans l'environnement sont conservées sauf ``override``.
    Un fichier absent ou illisible est ignoré (dictionnaire vide).
    """
    path = Path(path) if path is not None else DOTENV_FILE
    try:
        lines = path.read_text(encoding="utf-8-sig").splitlines()
    except OSError:
        return {}
    parsed: dict[str, str] = {}
    for raw in lines:
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        match = _DOTENV_LINE.match(line)
        if not match:
            continue
        name, value = match.group(1), match.group(2)
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        else:
            value = value.split(" #", 1)[0].strip()
        parsed[name] = value
        if override or not os.environ.get(name):
            os.environ[name] = value
    return parsed


def _split_keys(text: str) -> list[str]:
    return [part.strip() for part in _KEY_SEPARATORS.split(text) if part.strip()]


def gemini_api_keys(
    explicit: str | Iterable[str] | None = None, *, key_file: Path | None = None, dotenv: Path | None = None
) -> list[str]:
    """Liste ordonnée et dédoublonnée des clés Gemini disponibles (vide si aucune).

    Args:
        explicit: une clé, ou plusieurs (liste, ou chaîne séparée par des virgules) ; prioritaire.
        key_file: fichier à lire à défaut d'environnement (``GEMINI_KEY_FILE`` sinon).
        dotenv: fichier ``.env`` à charger d'abord (``DOTENV_FILE`` sinon).
    """
    load_dotenv(dotenv)
    keys: list[str] = []
    if explicit:
        keys = _split_keys(explicit) if isinstance(explicit, str) else [k.strip() for k in explicit if k and k.strip()]
    if not keys:
        keys = _split_keys(os.environ.get(GEMINI_KEYS_ENV_VAR, ""))
    if not keys:
        for var in GEMINI_KEY_ENV_VARS:
            value = os.environ.get(var, "").strip()
            if value:
                keys = [value]
                break
    if not keys:
        path = key_file if key_file is not None else GEMINI_KEY_FILE
        try:
            keys = _split_keys(path.read_text(encoding="utf-8-sig"))
        except OSError:
            keys = []
    return list(dict.fromkeys(keys))


def load_gemini_api_key(explicit: str | None = None, *, key_file: Path | None = None) -> str | None:
    """Première clé Gemini disponible (voir :func:`gemini_api_keys`), ou ``None``."""
    keys = gemini_api_keys(explicit, key_file=key_file)
    return keys[0] if keys else None


def gemini_key_hint() -> str:
    """Message d'aide (ASCII) indiquant où fournir la clé Gemini."""
    return (
        f"definir {GEMINI_KEYS_ENV_VAR} (plusieurs cles separees par des virgules, dans .env ou l'environnement), "
        f"{GEMINI_KEY_ENV_VARS[0]}, ou ecrire la cle dans {GEMINI_KEY_FILE.name} a la racine du projet"
    )


__all__ = [
    "PROJECT_ROOT",
    "DEFAULT_SITE_LANGUAGE",
    "DEFAULT_NARRATION_LANGUAGE",
    "GEMINI_KEYS_ENV_VAR",
    "GEMINI_KEY_ENV_VARS",
    "GEMINI_KEY_FILE",
    "DOTENV_FILE",
    "load_dotenv",
    "gemini_api_keys",
    "load_gemini_api_key",
    "gemini_key_hint",
]
