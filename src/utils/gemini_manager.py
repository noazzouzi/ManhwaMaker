"""Gestionnaire Gemini de production : rotation de clés, cascade de modèles, limitation RPM.

Un seul :class:`GeminiManager` est partagé par tous les chapitres d'un lot (il est
thread-safe : les appels partent des threads de travail de l'orchestrateur).

Règles :

- **Clés** : ``GEMINI_API_KEYS`` (virgules) dans ``.env`` ou l'environnement, sinon
  les sources habituelles (:func:`src.utils.config.gemini_api_keys`). La clé active
  est conservée d'un appel à l'autre ; on passe à la suivante quand elle est
  invalide / expirée (400 ``API_KEY_INVALID``, 401, 403) ou quand son quota
  journalier (RPD) est atteint pour le modèle courant.
- **Rate limit** : sur 429 « par minute » (``RESOURCE_EXHAUSTED``), 5xx ou erreur
  réseau, réessai avec backoff exponentiel 2 s, 4 s, 8 s, 16 s (le délai suggéré
  par l'API est honoré s'il est plus long) ; **4 essais au plus par clé**, puis
  rotation vers la clé suivante.
- **Cascade de modèles** : quand toutes les clés ont échoué sur le modèle courant,
  le gestionnaire bascule sur le modèle suivant de la cascade
  (:data:`DEFAULT_MODEL_CASCADE`, surchargeable par ``GEMINI_MODEL_CASCADE`` ou
  ``models=``). Un modèle inconnu du compte (404) est sauté. Toutes clés et tous
  modèles épuisés → :class:`QuotaExhaustedError`.
- **RPM global** : :class:`RateLimiter` (fenêtre glissante de 60 s) borne le nombre
  de requêtes démarrées par minute, toutes clés et tous chapitres confondus
  (``--max-gemini-rpm``).

Les clés ne sont jamais écrites dans les journaux (:func:`mask_key`).
"""

from __future__ import annotations

import logging
import os
import re
import threading
import time
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any

import httpx
from google.genai import errors

from src.utils.config import gemini_api_keys, gemini_key_hint

logger = logging.getLogger(__name__)

#: Cascade de secours par défaut, de la meilleure qualité au dernier recours. Sur ce compte,
#: ``gemini-2.0-flash``, ``gemini-1.5-flash`` et ``gemini-1.5-pro`` ne sont plus servis
#: (retirés, 404) : la cascade enchaîne les modèles **Flash** disponibles, puis les
#: **Flash-Lite** (quotas gratuits journaliers nettement plus larges, qualité moindre sur
#: l'analyse d'images), puis ``gemini-pro-latest``. Chaque modèle a son propre quota
#: journalier : plus la cascade est longue, plus on traite de chapitres par jour et par clé.
#: Surcharge complète : ``GEMINI_MODEL_CASCADE``.
DEFAULT_MODEL_CASCADE: tuple[str, ...] = (
    "gemini-3.5-flash",
    "gemini-3.6-flash",
    "gemini-3.7-flash",
    "gemini-3.8-flash",
    "gemini-2.5-flash",
    "gemini-3.5-flash-lite",
    "gemini-3.1-flash-lite",
    "gemini-2.5-flash-lite",
    "gemini-pro-latest",
)
MODEL_CASCADE_ENV_VAR: str = "GEMINI_MODEL_CASCADE"
#: Backoff exponentiel (secondes) entre les essais d'une même clé.
BACKOFF_SCHEDULE: tuple[float, ...] = (2.0, 4.0, 8.0, 16.0)
#: Essais au plus par clé avant rotation.
MAX_ATTEMPTS_PER_KEY: int = 4
#: Quand **toutes** les clés sont momentanément limitées (429 par minute) sur le modèle
#: courant, on patiente puis on réessaie le même modèle plutôt que de gâcher la cascade :
#: le quota journalier du modèle est encore disponible.
RATE_LIMIT_COOLDOWN_S: float = 30.0
MAX_RATE_LIMIT_ROUNDS: int = 2
#: Requêtes Gemini par minute (toutes clés confondues) par défaut.
DEFAULT_MAX_RPM: int = 10
#: Plafond d'attente entre deux essais (le délai suggéré par l'API est borné à cette valeur).
MAX_RETRY_DELAY_S: float = 65.0

_NETWORK_ERRORS: tuple[type[BaseException], ...] = (
    httpx.TimeoutException, httpx.NetworkError, httpx.RemoteProtocolError, ConnectionError, TimeoutError,
)
_RETRY_HINTS = (
    re.compile(r"retry in ([\d.]+)\s*s", re.IGNORECASE),
    re.compile(r"retryDelay['\"]?\s*[:=]\s*['\"]?([\d.]+)s", re.IGNORECASE),
)


class GeminiManagerError(RuntimeError):
    """Erreur définitive d'un appel Gemini (requête invalide, prompt bloqué...)."""


class QuotaExhaustedError(GeminiManagerError):
    """Toutes les clés sont épuisées ou invalides sur tous les modèles de la cascade."""


class ModelUnavailableError(GeminiManagerError):
    """Le modèle courant n'est pas servi sur ce compte (404) : passer au suivant."""


# --- Classification des erreurs ------------------------------------------------------------
def mask_key(key: str) -> str:
    """Forme journalisable d'une clé : ``...`` + 4 derniers caractères."""
    return f"...{key[-4:]}" if len(key) >= 8 else "...****"


def _message_of(exc: BaseException) -> str:
    return str(exc)


def is_daily_quota_error(exc: BaseException) -> bool:
    """Vrai pour un 429 dont le message indique un quota par jour (``PerDay``)."""
    return isinstance(exc, errors.ClientError) and exc.code == 429 and "PerDay" in _message_of(exc)


def is_rate_limit_error(exc: BaseException) -> bool:
    """Vrai pour un 429 « par minute » (RPM / TPM), à réessayer avec backoff."""
    return isinstance(exc, errors.ClientError) and exc.code == 429 and not is_daily_quota_error(exc)


def is_invalid_key_error(exc: BaseException) -> bool:
    """Vrai pour une clé invalide, expirée ou sans permission (400 ``API_KEY_INVALID``, 401, 403)."""
    if not isinstance(exc, errors.ClientError):
        return False
    if exc.code in (401, 403):
        return True
    text = _message_of(exc).lower()
    return exc.code == 400 and ("api key" in text or "api_key_invalid" in text)


def is_model_unavailable_error(exc: BaseException) -> bool:
    """Vrai pour un modèle inconnu / retiré (404 ``NOT_FOUND``)."""
    return isinstance(exc, errors.ClientError) and exc.code == 404


def is_transient_error(exc: BaseException) -> bool:
    """Vrai pour une erreur à réessayer sur la même clé : 429 par minute, 5xx, 408, réseau."""
    if is_rate_limit_error(exc):
        return True
    if isinstance(exc, errors.ServerError):
        return True
    if isinstance(exc, errors.ClientError):
        return exc.code == 408
    return isinstance(exc, _NETWORK_ERRORS)


def suggested_retry_delay(exc: BaseException) -> float:
    """Délai de réessai (secondes, +1 s de marge) suggéré par un message 429 de Gemini, sinon 0."""
    text = _message_of(exc)
    best = 0.0
    for pattern in _RETRY_HINTS:
        for match in pattern.finditer(text):
            try:
                best = max(best, float(match.group(1)))
            except ValueError:
                continue
    return best + 1.0 if best > 0 else 0.0


def resolve_model_cascade(preferred: str | None = None, cascade: Iterable[str] | None = None) -> list[str]:
    """Cascade effective : modèle préféré en tête, puis la cascade (argument, ``$GEMINI_MODEL_CASCADE``, défaut)."""
    if cascade is None:
        env = os.environ.get(MODEL_CASCADE_ENV_VAR, "")
        cascade = [m.strip() for m in env.split(",") if m.strip()] or list(DEFAULT_MODEL_CASCADE)
    models = ([preferred] if preferred else []) + [m for m in cascade if m]
    return list(dict.fromkeys(models))


# --- Limiteur RPM -------------------------------------------------------------------------
class RateLimiter:
    """Fenêtre glissante : au plus ``max_per_minute`` acquisitions par ``window_s`` secondes (thread-safe)."""

    def __init__(
        self,
        max_per_minute: int,
        *,
        window_s: float = 60.0,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        if max_per_minute < 1:
            raise ValueError("max_per_minute doit etre >= 1")
        self.max_per_minute = max_per_minute
        self.window_s = window_s
        self._clock = clock
        self._sleep = sleep
        self._lock = threading.Lock()
        self._starts: list[float] = []
        self.total_wait_s = 0.0

    def acquire(self) -> float:
        """Bloque jusqu'à ce qu'un créneau soit libre ; renvoie le temps attendu (s)."""
        waited = 0.0
        while True:
            with self._lock:
                now = self._clock()
                self._starts = [t for t in self._starts if now - t < self.window_s]
                if len(self._starts) < self.max_per_minute:
                    self._starts.append(now)
                    self.total_wait_s += waited
                    return waited
                wait = self._starts[0] + self.window_s - now
            wait = max(wait, 0.05)
            self._sleep(wait)
            waited += wait


# --- État des clés -----------------------------------------------------------------------
@dataclass
class KeyState:
    """État d'une clé : invalide, modèles dont le quota journalier est épuisé, compteurs."""

    key: str
    index: int
    invalid: bool = False
    exhausted: dict[str, str] = field(default_factory=dict)
    calls: int = 0
    failures: int = 0

    @property
    def label(self) -> str:
        return f"cle #{self.index + 1} ({mask_key(self.key)})"

    def usable(self, model: str) -> bool:
        return not self.invalid and model not in self.exhausted

    def as_dict(self) -> dict[str, Any]:
        return {
            "label": self.label, "invalid": self.invalid, "exhausted_models": sorted(self.exhausted),
            "calls": self.calls, "failures": self.failures,
        }


def percentile(values: Sequence[float], q: float) -> float:
    """Centile ``q`` (0-1) d'une série (interpolation au plus proche) ; 0 si la série est vide."""
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int(round(q * (len(ordered) - 1)))))
    return ordered[index]


#: Rotation de clé : cette clé ne peut plus servir pour ce modèle (quota journalier, clé invalide).
_ROTATE = object()
#: Clé momentanément limitée (429 par minute) : essayer une autre clé, sinon patienter.
_RATE_LIMITED = object()


class GeminiManager:
    """Point d'entrée unique des appels Gemini : clés, modèles, backoff et RPM.

    Args:
        keys: clé(s) explicite(s) ; sinon :func:`gemini_api_keys` (``.env`` compris).
        models: cascade de modèles (sinon ``$GEMINI_MODEL_CASCADE`` ou :data:`DEFAULT_MODEL_CASCADE`).
        preferred_model: modèle à essayer en premier (``--model``), la cascade suit.
        max_rpm: requêtes par minute, toutes clés confondues.
        max_attempts_per_key: essais par clé avant rotation.
        backoff: délais du backoff exponentiel.
        client_factory: ``key -> client`` (tests) ; défaut ``genai.Client(api_key=key)``.
        sleep, clock: injectables pour les tests.
    """

    def __init__(
        self,
        keys: str | Iterable[str] | None = None,
        *,
        models: Sequence[str] | None = None,
        preferred_model: str | None = None,
        max_rpm: int = DEFAULT_MAX_RPM,
        max_attempts_per_key: int = MAX_ATTEMPTS_PER_KEY,
        backoff: Sequence[float] = BACKOFF_SCHEDULE,
        client_factory: Callable[[str], Any] | None = None,
        sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        resolved = gemini_api_keys(keys)
        if not resolved:
            raise GeminiManagerError(f"Aucune cle Gemini trouvee : {gemini_key_hint()}")
        if max_attempts_per_key < 1:
            raise ValueError("max_attempts_per_key doit etre >= 1")
        self.keys = [KeyState(key=k, index=i) for i, k in enumerate(resolved)]
        self.models = resolve_model_cascade(preferred_model, models)
        self.max_attempts_per_key = max_attempts_per_key
        self.backoff = tuple(backoff) or BACKOFF_SCHEDULE
        self.limiter = RateLimiter(max_rpm, clock=clock, sleep=sleep)
        self._sleep = sleep
        self._client_factory = client_factory or self._default_client
        self._clients: dict[int, Any] = {}
        self._lock = threading.RLock()
        self._active_key = 0
        self._model_index = 0
        self.n_calls = 0
        self.n_rotations = 0
        self.n_cascades = 0
        #: Temps de réponse (secondes) de chaque appel, succès comme échec.
        self.latencies: list[float] = []
        #: Par modèle : appels réussis, échoués, temps cumulé.
        self.calls_by_model: dict[str, dict[str, float]] = {}
        logger.info(
            "Gestionnaire Gemini : %d cle(s), cascade %s, %d req/min max", len(self.keys), " > ".join(self.models), max_rpm,
        )

    # -- accès -----------------------------------------------------------------------------
    @staticmethod
    def _default_client(key: str) -> Any:
        from google import genai

        return genai.Client(api_key=key)

    @property
    def current_model(self) -> str:
        return self.models[self._model_index]

    @property
    def current_key(self) -> KeyState:
        return self.keys[self._active_key]

    def client_for(self, key: KeyState) -> Any:
        with self._lock:
            client = self._clients.get(key.index)
            if client is None:
                try:
                    client = self._client_factory(key.key)
                except Exception as exc:  # noqa: BLE001
                    raise GeminiManagerError(f"Impossible de creer le client Gemini pour la {key.label} : {exc}") from exc
                self._clients[key.index] = client
            return client

    def _record_latency(self, model: str, elapsed: float, *, ok: bool) -> None:
        with self._lock:
            self.latencies.append(elapsed)
            stats = self.calls_by_model.setdefault(model, {"ok": 0, "failed": 0, "seconds": 0.0})
            stats["ok" if ok else "failed"] += 1
            stats["seconds"] += elapsed

    def status(self) -> dict[str, Any]:
        """État courant (clé active masquée, modèle, temps de réponse) pour les journaux et ``batch_status.json``."""
        with self._lock:
            latencies = list(self.latencies)
            return {
                "model": self.current_model,
                "models": list(self.models),
                "active_key": self.current_key.label,
                "keys": [k.as_dict() for k in self.keys],
                "calls": self.n_calls,
                "rotations": self.n_rotations,
                "cascades": self.n_cascades,
                "rpm_wait_s": round(self.limiter.total_wait_s, 1),
                "latency_s": {
                    "n": len(latencies),
                    "mean": round(sum(latencies) / len(latencies), 2) if latencies else 0.0,
                    "p50": round(percentile(latencies, 0.5), 2),
                    "p95": round(percentile(latencies, 0.95), 2),
                    "max": round(max(latencies), 2) if latencies else 0.0,
                    "total": round(sum(latencies), 1),
                },
                "calls_by_model": {
                    model: {"ok": int(s["ok"]), "failed": int(s["failed"]), "seconds": round(s["seconds"], 1)}
                    for model, s in self.calls_by_model.items()
                },
            }

    # -- sélection --------------------------------------------------------------------------
    def _pick_key(self, model: str, tried: set[int]) -> KeyState | None:
        with self._lock:
            n = len(self.keys)
            for offset in range(n):
                key = self.keys[(self._active_key + offset) % n]
                if key.index not in tried and key.usable(model):
                    if key.index != self._active_key:
                        logger.warning("Rotation de cle Gemini : %s -> %s", self.current_key.label, key.label)
                        self._active_key = key.index
                        self.n_rotations += 1
                    return key
            return None

    def _advance_model(self, reason: str) -> bool:
        with self._lock:
            if self._model_index + 1 >= len(self.models):
                return False
            previous = self.current_model
            self._model_index += 1
            self.n_cascades += 1
            logger.warning("Cascade de modele Gemini : %s -> %s (%s)", previous, self.current_model, reason)
            return True

    # -- appel ------------------------------------------------------------------------------
    def generate(self, contents: Any, config: Any, *, label: str = "") -> Any:
        """Appelle ``generate_content`` avec rotation de clés, backoff et cascade ; renvoie la réponse.

        Raises:
            QuotaExhaustedError: toutes les clés sont épuisées ou invalides sur toute la cascade.
            GeminiManagerError: erreur définitive (requête invalide, contenu bloqué...).
        """
        cooldowns = 0
        while True:
            model = self.current_model
            tried: set[int] = set()
            rate_limited = False
            model_gone: ModelUnavailableError | None = None
            while True:
                key = self._pick_key(model, tried)
                if key is None:
                    break
                tried.add(key.index)
                try:
                    outcome = self._attempt_with_key(key, model, contents, config, label)
                except ModelUnavailableError as exc:
                    model_gone = exc
                    break
                if outcome is _RATE_LIMITED:
                    rate_limited = True
                    continue
                if outcome is not _ROTATE:
                    return outcome
            if model_gone is not None:
                if not self._advance_model(str(model_gone)):
                    raise QuotaExhaustedError(f"{label} : aucun modele disponible ({model_gone})") from model_gone
                continue
            if rate_limited and cooldowns < MAX_RATE_LIMIT_ROUNDS:
                # Limite par minute sur toutes les cles : le quota journalier du modele reste
                # disponible, on patiente plutot que de gaspiller la cascade.
                cooldowns += 1
                logger.warning(
                    "%s : toutes les cles limitees par minute sur %s, pause de %.0fs (%d/%d)",
                    label, model, RATE_LIMIT_COOLDOWN_S, cooldowns, MAX_RATE_LIMIT_ROUNDS,
                )
                self._sleep(RATE_LIMIT_COOLDOWN_S)
                continue
            if not self._advance_model("toutes les cles ont echoue"):
                raise QuotaExhaustedError(
                    f"{label} : toutes les cles Gemini ({len(self.keys)}) sont epuisees ou invalides sur "
                    f"{', '.join(self.models)} : ajouter des cles dans GEMINI_API_KEYS, activer la facturation, "
                    "ou attendre le lendemain"
                )

    def _attempt_with_key(self, key: KeyState, model: str, contents: Any, config: Any, label: str) -> Any:
        client = self.client_for(key)
        for attempt in range(1, self.max_attempts_per_key + 1):
            self.limiter.acquire()
            with self._lock:
                self.n_calls += 1
                key.calls += 1
            started = time.perf_counter()
            try:
                response = client.models.generate_content(model=model, contents=contents, config=config)
            except Exception as exc:  # noqa: BLE001 - tri ci-dessous
                self._record_latency(model, time.perf_counter() - started, ok=False)
                with self._lock:
                    key.failures += 1
                if is_daily_quota_error(exc):
                    key.exhausted[model] = "quota journalier"
                    logger.warning("%s : quota journalier atteint pour %s sur %s : rotation", label, key.label, model)
                    return _ROTATE
                if is_invalid_key_error(exc):
                    key.invalid = True
                    logger.error("%s : %s invalide ou expiree (%s) : rotation", label, key.label, type(exc).__name__)
                    return _ROTATE
                if is_model_unavailable_error(exc):
                    raise ModelUnavailableError(f"{model} indisponible sur ce compte : {exc}") from exc
                if is_transient_error(exc):
                    if attempt < self.max_attempts_per_key:
                        delay = min(max(self.backoff[min(attempt, len(self.backoff)) - 1], suggested_retry_delay(exc)), MAX_RETRY_DELAY_S)
                        logger.warning(
                            "%s : essai %d/%d echoue sur %s (%s: %s) ; nouvel essai dans %.0fs",
                            label, attempt, self.max_attempts_per_key, key.label, type(exc).__name__, str(exc)[:160], delay,
                        )
                        self._sleep(delay)
                        continue
                    logger.warning("%s : %d essais echoues sur %s : rotation de cle", label, self.max_attempts_per_key, key.label)
                    return _RATE_LIMITED
                raise GeminiManagerError(f"{label} : {type(exc).__name__}: {exc}") from exc
            else:
                self._record_latency(model, time.perf_counter() - started, ok=True)
                return response
        return _RATE_LIMITED


__all__ = [
    "DEFAULT_MODEL_CASCADE",
    "MODEL_CASCADE_ENV_VAR",
    "BACKOFF_SCHEDULE",
    "MAX_ATTEMPTS_PER_KEY",
    "RATE_LIMIT_COOLDOWN_S",
    "MAX_RATE_LIMIT_ROUNDS",
    "DEFAULT_MAX_RPM",
    "percentile",
    "GeminiManagerError",
    "QuotaExhaustedError",
    "ModelUnavailableError",
    "mask_key",
    "is_daily_quota_error",
    "is_rate_limit_error",
    "is_invalid_key_error",
    "is_model_unavailable_error",
    "is_transient_error",
    "suggested_retry_delay",
    "resolve_model_cascade",
    "RateLimiter",
    "KeyState",
    "GeminiManager",
]
