"""Helpers HTTP transverses : session ``requests`` pré-configurée et GET avec retries.

Garde-fou n°1 du PRD : Webtoons renvoie **HTTP 403** si les en-têtes
``User-Agent`` (navigateur desktop) et ``Referer: https://www.webtoons.com/``
sont absents, aussi bien sur la page du chapitre que sur chaque image.
Toutes les requêtes du projet doivent donc passer par :class:`requests.Session`
construite via :func:`build_session` (ou au minimum par
:func:`ensure_mandatory_headers`).
"""

from __future__ import annotations

import logging
import time
from collections.abc import Mapping
from typing import Any

import requests

logger = logging.getLogger(__name__)

# --- Constantes publiques -------------------------------------------------------------

DEFAULT_USER_AGENT: str = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/128.0.0.0 Safari/537.36"
)
WEBTOONS_REFERER: str = "https://www.webtoons.com/"

DEFAULT_HEADERS: dict[str, str] = {
    "User-Agent": DEFAULT_USER_AGENT,
    "Referer": WEBTOONS_REFERER,
    "Accept": (
        "text/html,application/xhtml+xml,application/xml;q=0.9,"
        "image/avif,image/webp,image/apng,*/*;q=0.8"
    ),
    "Accept-Language": "en-US,en;q=0.9",
}

#: En-têtes strictement obligatoires (anti-403) ; sous-ensemble de ``DEFAULT_HEADERS``.
MANDATORY_HEADERS: dict[str, str] = {
    "User-Agent": DEFAULT_USER_AGENT,
    "Referer": WEBTOONS_REFERER,
}

#: Préfixe du ``User-Agent`` que ``requests`` installe par défaut sur toute session ;
#: il est traité comme *absent* car il ne correspond pas à un navigateur desktop.
_LIBRARY_USER_AGENT_PREFIX: str = "python-requests"

#: Codes HTTP qui justifient une nouvelle tentative : rate-limit (429) et **tous**
#: les 5xx (y compris 501/505/507 et la famille 520-527 des CDN), conformément au
#: contrat « retries on 5xx/429 ».
RETRY_STATUS_CODES: frozenset[int] = frozenset({429, *range(500, 600)})

#: Exceptions réseau réessayables : connexion/timeout, mais aussi coupure en cours de
#: lecture du corps (``ChunkedEncodingError``) et corps compressé corrompu
#: (``ContentDecodingError``), qui ne dérivent PAS de ``requests.ConnectionError``.
RETRY_EXCEPTIONS: tuple[type[requests.RequestException], ...] = (
    requests.ConnectionError,
    requests.Timeout,
    requests.exceptions.ChunkedEncodingError,
    requests.exceptions.ContentDecodingError,
)

#: Plafond (secondes) appliqué à un en-tête ``Retry-After`` pour ne pas bloquer trop longtemps.
MAX_RETRY_AFTER: float = 30.0

# Indirection vers ``time.sleep`` pour pouvoir neutraliser l'attente dans les tests.
_sleep = time.sleep


# --- Session ------------------------------------------------------------------------


def build_session(headers: Mapping[str, str] | None = None) -> requests.Session:
    """Construit une session ``requests`` munie des en-têtes obligatoires.

    Args:
        headers: En-têtes supplémentaires ou de remplacement, fusionnés
            par-dessus :data:`DEFAULT_HEADERS`.

    Returns:
        Session prête à l'emploi (``User-Agent`` + ``Referer`` garantis).
    """
    session = requests.Session()
    session.headers.update(DEFAULT_HEADERS | dict(headers or {}))
    return session


def _is_missing_header(name: str, value: object) -> bool:
    """Indique si un en-tête obligatoire doit être (ré)écrit.

    Un en-tête vide est manquant. Le ``User-Agent`` par défaut de ``requests``
    (``python-requests/x.y``, présent sur **toute** ``requests.Session()`` nue)
    est aussi considéré comme manquant : ce n'est pas un navigateur desktop.
    """
    if not value:
        return True
    return name == "User-Agent" and str(value).startswith(_LIBRARY_USER_AGENT_PREFIX)


def ensure_mandatory_headers(session: requests.Session | None) -> requests.Session:
    """Garantit la présence des en-têtes anti-403 sur une session.

    - ``None`` : une nouvelle session est construite via :func:`build_session`.
    - Session existante : les en-têtes manquants parmi :data:`MANDATORY_HEADERS`
      sont ajoutés **sans écraser** ceux définis par l'appelant. Le ``User-Agent``
      ``python-requests/…`` installé d'office par ``requests`` n'est pas
      considéré comme défini par l'appelant : il est remplacé par
      :data:`DEFAULT_USER_AGENT` (navigateur desktop, exigé par le PRD).

    Args:
        session: Session fournie par l'appelant, ou ``None``.

    Returns:
        La session (créée ou complétée), portant ``User-Agent`` desktop + ``Referer``.
    """
    if session is None:
        return build_session()
    session_headers = getattr(session, "headers", None)
    if session_headers is not None:
        for name, value in MANDATORY_HEADERS.items():
            if _is_missing_header(name, session_headers.get(name)):
                logger.debug("Ajout de l'en-tete obligatoire manquant: %s", name)
                session_headers[name] = value
    return session


# --- GET avec retries ---------------------------------------------------------------


def _retry_delay(response: requests.Response | None, attempt: int, backoff: float) -> float:
    """Calcule le délai d'attente avant la tentative suivante.

    Backoff exponentiel ``backoff * 2 ** (attempt - 1)``, ou l'en-tête
    ``Retry-After`` (en secondes) s'il est présent et numérique, plafonné à
    :data:`MAX_RETRY_AFTER`.
    """
    delay = backoff * (2 ** (attempt - 1))
    if response is not None:
        retry_after = response.headers.get("Retry-After") if response.headers else None
        if retry_after:
            try:
                delay = max(delay, float(retry_after))
            except ValueError:
                pass
    return min(delay, MAX_RETRY_AFTER)


def get_with_retry(
    session: requests.Session,
    url: str,
    *,
    max_retries: int = 3,
    backoff: float = 0.5,
    timeout: float = 20,
    **kwargs: Any,
) -> requests.Response:
    """Effectue un GET avec nouvelles tentatives sur erreurs transitoires.

    Sont réessayés : les erreurs réseau de :data:`RETRY_EXCEPTIONS` (connexion,
    timeout, coupure en cours de lecture du corps) et les statuts
    :data:`RETRY_STATUS_CODES` (429 et tout 5xx). Les statuts définitifs
    (403, 404, autres 4xx) lèvent immédiatement :class:`requests.HTTPError`.

    Args:
        session: Session (idéalement issue de :func:`build_session`).
        url: URL à télécharger.
        max_retries: Nombre de **nouvelles** tentatives après la première
            (soit ``max_retries + 1`` requêtes au maximum).
        backoff: Délai de base (secondes) du backoff exponentiel.
        timeout: Timeout ``requests`` (secondes) de chaque requête.
        **kwargs: Arguments transmis tels quels à ``session.get``.

    Returns:
        La réponse HTTP (statut 2xx/3xx).

    Raises:
        requests.HTTPError: Statut d'erreur définitif, ou statut réessayable
            persistant après épuisement des tentatives. Une erreur réseau
            persistante est également remontée sous forme de
            :class:`requests.HTTPError` (contrat du module), l'exception réseau
            d'origine étant chaînée dans ``__cause__``.
    """
    attempts = max(1, int(max_retries) + 1)
    last_error: Exception | None = None

    for attempt in range(1, attempts + 1):
        response: requests.Response | None = None
        try:
            response = session.get(url, timeout=timeout, **kwargs)
        except RETRY_EXCEPTIONS as exc:
            last_error = exc
            logger.warning(
                "Erreur reseau (%s/%s) sur %s: %s", attempt, attempts, url, exc
            )
        else:
            status = response.status_code
            if status in RETRY_STATUS_CODES:
                last_error = requests.HTTPError(
                    f"HTTP {status} sur {url}", response=response
                )
                logger.warning(
                    "HTTP %s reessayable (%s/%s) sur %s", status, attempt, attempts, url
                )
            else:
                if status == 403:
                    logger.error(
                        "HTTP 403 sur %s : verifier les en-tetes User-Agent / Referer", url
                    )
                response.raise_for_status()
                return response

        if attempt < attempts:
            delay = _retry_delay(response, attempt, backoff)
            logger.debug("Nouvelle tentative dans %.2fs", delay)
            _sleep(delay)

    assert last_error is not None  # garanti par la boucle
    logger.error("Abandon apres %s tentative(s) sur %s: %s", attempts, url, last_error)
    if isinstance(last_error, requests.HTTPError):
        raise last_error
    # Normalisation au type du contrat : l'appelant n'attrape que HTTPError.
    raise requests.HTTPError(
        f"Abandon apres {attempts} tentative(s) sur {url}: "
        f"{type(last_error).__name__}: {last_error}"
    ) from last_error
