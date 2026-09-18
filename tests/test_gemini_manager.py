"""Tests du gestionnaire Gemini (``src.utils.gemini_manager``) : rotation de clés, backoff, cascade, RPM."""

from __future__ import annotations

import threading

import httpx
import pytest
from google.genai import errors

from src.utils import gemini_manager as gm
from src.utils.gemini_manager import (
    BACKOFF_SCHEDULE,
    DEFAULT_MODEL_CASCADE,
    GeminiManager,
    GeminiManagerError,
    QuotaExhaustedError,
    RateLimiter,
    mask_key,
    resolve_model_cascade,
)

KEYS = ["AIzaSyKEY-ONE-0001", "AIzaSyKEY-TWO-0002", "AIzaSyKEY-THREE-03"]


def daily_quota(model: str = "m") -> errors.ClientError:
    return errors.ClientError(429, {"error": {"message": (
        f"Quota exceeded for metric: generate_content_free_tier_requests, limit: 20, model: {model}."),
        "details": [{"violations": [{"quotaId": "GenerateRequestsPerDayPerProjectPerModel-FreeTier"}]}]}})


def per_minute(retry_s: float | None = None) -> errors.ClientError:
    message = "Quota exceeded for metric: GenerateRequestsPerMinutePerProjectPerModel-FreeTier, limit: 5."
    if retry_s is not None:
        message += f" Please retry in {retry_s}s."
    return errors.ClientError(429, {"error": {"message": message}})


class FakeModels:
    """``client.models.generate_content`` factice piloté par une fonction ``(key, model, n) -> réponse | exception``."""

    def __init__(self, key: str, behaviour, log: list):
        self.key, self.behaviour, self.log = key, behaviour, log

    def generate_content(self, *, model, contents, config):
        self.log.append((self.key, model))
        outcome = self.behaviour(self.key, model, len(self.log))
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome


class FakeClient:
    def __init__(self, key: str, behaviour, log: list):
        self.models = FakeModels(key, behaviour, log)


def make_manager(behaviour, *, keys=KEYS, models=("m1", "m2"), max_rpm=100, sleeps: list | None = None, **kwargs):
    log: list = []
    created: list[str] = []

    def factory(key: str):
        created.append(key)
        return FakeClient(key, behaviour, log)

    manager = GeminiManager(
        keys, models=list(models), max_rpm=max_rpm, client_factory=factory,
        sleep=(sleeps.append if sleeps is not None else lambda s: None), **kwargs,
    )
    return manager, log, created


def test_helpers_and_cascade_resolution(monkeypatch) -> None:
    assert mask_key("AIzaSyKEY-ONE-0001") == "...0001" and mask_key("short") == "...****"
    assert resolve_model_cascade() == list(DEFAULT_MODEL_CASCADE)
    assert resolve_model_cascade("gemini-3.7-flash")[0] == "gemini-3.7-flash"
    assert resolve_model_cascade("gemini-3.7-flash").count("gemini-3.7-flash") == 1
    monkeypatch.setenv("GEMINI_MODEL_CASCADE", "a, b ,a,c")
    assert resolve_model_cascade() == ["a", "b", "c"] and resolve_model_cascade("z", ["x"]) == ["z", "x"]
    assert gm.is_daily_quota_error(daily_quota()) and not gm.is_rate_limit_error(daily_quota())
    assert gm.is_rate_limit_error(per_minute()) and gm.is_transient_error(per_minute())
    assert gm.is_transient_error(errors.ServerError(503, {"error": {"message": "busy"}}))
    assert gm.is_transient_error(httpx.ReadTimeout("t"))
    assert gm.is_invalid_key_error(errors.ClientError(400, {"error": {"message": "API key not valid. Please pass a valid API key."}}))
    assert gm.is_invalid_key_error(errors.ClientError(403, {"error": {"message": "PERMISSION_DENIED"}}))
    assert not gm.is_invalid_key_error(errors.ClientError(400, {"error": {"message": "bad request"}}))
    assert gm.is_model_unavailable_error(errors.ClientError(404, {"error": {"message": "models/x is not found"}}))
    assert gm.suggested_retry_delay(per_minute(12.5)) == pytest.approx(13.5) and gm.suggested_retry_delay(per_minute()) == 0.0
    from src.utils import config as config_mod

    monkeypatch.setattr(config_mod, "GEMINI_KEY_FILE", config_mod.PROJECT_ROOT / "nope.key")
    for var in ("GEMINI_API_KEY", "GOOGLE_API_KEY"):
        monkeypatch.delenv(var, raising=False)
    with pytest.raises(GeminiManagerError, match="Aucune cle"):
        GeminiManager([], models=["m"])


def test_rate_limiter_sliding_window() -> None:
    now = [0.0]
    sleeps: list[float] = []

    def sleep(s: float) -> None:
        sleeps.append(s)
        now[0] += s

    limiter = RateLimiter(3, window_s=60.0, clock=lambda: now[0], sleep=sleep)
    for _ in range(3):
        assert limiter.acquire() == 0.0
    now[0] = 10.0
    waited = limiter.acquire()  # 4e requete : attend la liberation du premier creneau (t = 60 s)
    assert waited == pytest.approx(50.0) and sleeps == [pytest.approx(50.0)] and now[0] == pytest.approx(60.0)
    # A t = 60 les trois premiers creneaux sont sortis de la fenetre : deux requetes passent, la suivante attend.
    assert limiter.acquire() == 0.0 and limiter.acquire() == 0.0
    assert limiter.acquire() == pytest.approx(60.0) and now[0] == pytest.approx(120.0)
    assert limiter.total_wait_s == pytest.approx(110.0)
    with pytest.raises(ValueError):
        RateLimiter(0)


def test_generate_keeps_active_key_and_returns_response() -> None:
    manager, log, created = make_manager(lambda key, model, n: {"ok": key})
    assert manager.generate("c", "cfg", label="x") == {"ok": KEYS[0]}
    assert manager.generate("c", "cfg", label="y") == {"ok": KEYS[0]}
    assert created == [KEYS[0]] and manager.current_key.label.endswith("(...0001)")
    status = manager.status()
    assert status["calls"] == 2 and status["rotations"] == 0 and status["model"] == "m1"
    assert KEYS[0] not in str(status)  # jamais de cle en clair


def test_daily_quota_rotates_key_without_retry() -> None:
    def behaviour(key, model, n):
        return daily_quota(model) if key == KEYS[0] else {"ok": key}

    sleeps: list[float] = []
    manager, log, _ = make_manager(behaviour, sleeps=sleeps)
    assert manager.generate("c", "cfg", label="beats") == {"ok": KEYS[1]}
    assert log == [(KEYS[0], "m1"), (KEYS[1], "m1")] and sleeps == []
    assert manager.keys[0].exhausted == {"m1": "quota journalier"} and manager.current_key.index == 1
    # La cle active reste la deuxieme pour les appels suivants.
    manager.generate("c", "cfg", label="next")
    assert log[-1] == (KEYS[1], "m1") and manager.status()["rotations"] == 1


def test_single_key_rate_limited_waits_instead_of_burning_the_cascade() -> None:
    """Une seule cle limitee par minute : pause puis nouvel essai sur le meme modele."""
    calls = {"n": 0}

    def behaviour(key, model, n):
        calls["n"] += 1
        return per_minute() if calls["n"] <= 4 else {"ok": model}

    sleeps: list[float] = []
    manager, log, _ = make_manager(behaviour, keys=KEYS[:1], sleeps=sleeps)
    assert manager.generate("c", "cfg", label="beats") == {"ok": "m1"}  # toujours m1 : pas de cascade
    assert manager.current_model == "m1" and manager.status()["cascades"] == 0
    assert sleeps == [2.0, 4.0, 8.0, gm.RATE_LIMIT_COOLDOWN_S]  # backoff puis pause globale
    assert len(log) == 5 and not manager.keys[0].invalid and not manager.keys[0].exhausted

    # Limitation permanente : apres MAX_RATE_LIMIT_ROUNDS pauses (budget global, 60 s au plus),
    # la cascade reprend ses droits et l'appel echoue au bout de la liste des modeles.
    pauses: list[float] = []
    always, log2, _ = make_manager(lambda key, model, n: per_minute(), keys=KEYS[:1], models=["m1", "m2"], sleeps=pauses)
    with pytest.raises(QuotaExhaustedError):
        always.generate("c", "cfg")
    assert always.status()["cascades"] == 1
    assert [m for _, m in log2] == ["m1"] * 12 + ["m2"] * 4  # 3 series de 4 essais sur m1, puis m2
    assert pauses.count(gm.RATE_LIMIT_COOLDOWN_S) == gm.MAX_RATE_LIMIT_ROUNDS


def test_status_reports_latency_and_calls_by_model() -> None:
    now = [0.0]

    def behaviour(key, model, n):
        now[0] += 0.5 * n  # 0,5 s, 1 s, 1,5 s...
        return {"ok": n}

    manager, log, _ = make_manager(behaviour, keys=KEYS[:1], models=["m1"])
    import src.utils.gemini_manager as module

    original = module.time.perf_counter
    module.time.perf_counter = lambda: now[0]  # type: ignore[assignment]
    try:
        for _ in range(4):
            manager.generate("c", "cfg")
    finally:
        module.time.perf_counter = original  # type: ignore[assignment]
    status = manager.status()
    latency = status["latency_s"]
    assert latency["n"] == 4 and latency["max"] == pytest.approx(2.0) and latency["p50"] > 0
    assert latency["total"] == pytest.approx(5.0) and latency["mean"] == pytest.approx(1.25)
    assert status["calls_by_model"] == {"m1": {"ok": 4, "failed": 0, "seconds": 5.0}}
    from src.utils.gemini_manager import percentile

    assert percentile([], 0.5) == 0.0 and percentile([1.0, 2.0, 3.0], 0.5) == 2.0
    assert percentile([1.0, 2.0, 3.0, 4.0], 0.95) == 4.0


def test_rate_limit_backoff_then_key_rotation() -> None:
    attempts = {}

    def behaviour(key, model, n):
        attempts[key] = attempts.get(key, 0) + 1
        if key == KEYS[0]:
            return per_minute()  # toujours limitee
        return {"ok": key}

    sleeps: list[float] = []
    manager, log, _ = make_manager(behaviour, sleeps=sleeps)
    assert manager.generate("c", "cfg", label="x") == {"ok": KEYS[1]}
    assert attempts[KEYS[0]] == 4 and sleeps == list(BACKOFF_SCHEDULE[:3])  # 2, 4, 8 entre les 4 essais
    assert manager.current_key.index == 1 and not manager.keys[0].invalid and not manager.keys[0].exhausted


def test_backoff_honours_longer_server_delay_and_caps() -> None:
    calls = {"n": 0}

    def behaviour(key, model, n):
        calls["n"] += 1
        if calls["n"] == 1:
            return per_minute(30.0)
        if calls["n"] == 2:
            return errors.ServerError(503, {"error": {"message": "high demand"}})
        if calls["n"] == 3:
            return httpx.ReadTimeout("slow")
        return {"ok": key}

    sleeps: list[float] = []
    manager, log, _ = make_manager(behaviour, sleeps=sleeps)
    assert manager.generate("c", "cfg") == {"ok": KEYS[0]}
    assert sleeps == [pytest.approx(31.0), 4.0, 8.0]  # delai suggere (30 + 1) > 2 s, puis planning normal


def test_invalid_key_is_skipped_for_good() -> None:
    def behaviour(key, model, n):
        if key == KEYS[0]:
            return errors.ClientError(400, {"error": {"message": "API key not valid. Please pass a valid API key."}})
        return {"ok": key}

    manager, log, _ = make_manager(behaviour)
    assert manager.generate("c", "cfg") == {"ok": KEYS[1]}
    assert manager.keys[0].invalid and log == [(KEYS[0], "m1"), (KEYS[1], "m1")]
    manager.generate("c", "cfg")
    assert log[-1] == (KEYS[1], "m1")  # la cle invalide n'est plus jamais essayee


def test_cascade_to_next_model_when_all_keys_fail_then_quota_exhausted() -> None:
    def behaviour(key, model, n):
        if model == "m1":
            return daily_quota(model)
        if key == KEYS[2]:
            return {"ok": (key, model)}
        return daily_quota(model)

    manager, log, _ = make_manager(behaviour)
    assert manager.generate("c", "cfg", label="x") == {"ok": (KEYS[2], "m2")}
    # Les 3 cles echouent sur m1 ; la cle active (la 3e, derniere rotation) est reessayee en premier sur m2.
    assert [m for _, m in log] == ["m1", "m1", "m1", "m2"]
    assert manager.current_model == "m2" and manager.status()["cascades"] == 1
    assert all("m1" in k.exhausted for k in manager.keys) and manager.current_key.index == 2
    # Nouvel appel : la cle 3 reste active sur m2, aucune rotation supplementaire.
    manager.generate("c", "cfg")
    assert log[-1] == (KEYS[2], "m2") and manager.status()["rotations"] == 2

    everything_dead, log2, _ = make_manager(lambda key, model, n: daily_quota(model))
    with pytest.raises(QuotaExhaustedError, match="toutes les cles"):
        everything_dead.generate("c", "cfg", label="beats lot 1")
    assert len(log2) == 6  # 3 cles x 2 modeles, un seul essai chacun (quota journalier)
    # Etat persistant : le prochain appel echoue sans nouvel appel reseau.
    with pytest.raises(QuotaExhaustedError):
        everything_dead.generate("c", "cfg")
    assert len(log2) == 6


def test_unknown_model_is_skipped_and_definitive_errors_raise() -> None:
    def behaviour(key, model, n):
        if model == "m1":
            return errors.ClientError(404, {"error": {"message": "models/m1 is not found"}})
        return {"ok": model}

    manager, log, _ = make_manager(behaviour)
    assert manager.generate("c", "cfg") == {"ok": "m2"} and log == [(KEYS[0], "m1"), (KEYS[0], "m2")]

    bad, log, _ = make_manager(lambda key, model, n: errors.ClientError(400, {"error": {"message": "bad request"}}))
    with pytest.raises(GeminiManagerError, match="bad request"):
        bad.generate("c", "cfg", label="script")
    assert len(log) == 1  # definitif : aucune rotation

    only_404, _, _ = make_manager(lambda key, model, n: errors.ClientError(404, {"error": {"message": "nope"}}), models=["m1"])
    with pytest.raises(QuotaExhaustedError, match="aucun modele"):
        only_404.generate("c", "cfg")


def test_manager_is_thread_safe_under_concurrent_calls() -> None:
    manager, log, _ = make_manager(lambda key, model, n: {"ok": n}, max_rpm=1000)
    errors_seen: list[BaseException] = []

    def worker():
        try:
            for _ in range(20):
                manager.generate("c", "cfg")
        except BaseException as exc:  # noqa: BLE001
            errors_seen.append(exc)

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors_seen and len(log) == 160 and manager.status()["calls"] == 160


def test_rpm_limiter_paces_calls() -> None:
    now = [0.0]
    sleeps: list[float] = []

    def sleep(s: float) -> None:
        sleeps.append(s)
        now[0] += s

    manager = GeminiManager(
        KEYS[:1], models=["m"], max_rpm=2, client_factory=lambda k: FakeClient(k, lambda *a: {"ok": 1}, []),
        sleep=sleep, clock=lambda: now[0],
    )
    for _ in range(3):
        manager.generate("c", "cfg")
    assert len(sleeps) == 1 and sleeps[0] == pytest.approx(60.0) and manager.status()["rpm_wait_s"] == 60.0
