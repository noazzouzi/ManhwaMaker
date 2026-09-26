"""Progression d'un traitement, lisible par ManhwaMaker Studio pendant qu'il tourne.

Inactif tant que la variable d'environnement ``MM_PROGRESS_FILE`` n'est pas définie : la
ligne de commande seule n'écrit rien de plus et chaque appel ne coûte qu'un test.

Le fichier JSON décrit le traitement (``kind``, ``phase``) et ses éléments (``items``, un
par chapitre) : étape (``prep`` / ``script`` / ``voice`` / ``montage``), état (``waiting``
avec la raison de l'attente, ou ``running``), sous-étape en cours (``step``) avec son
avancement ``done``/``total``, et l'horodatage de début et de fin de chaque étape. Les
heures sont des secondes epoch (``time.time()``).

Un élément est désigné par une clé liée au contexte (:func:`bind`) : une étape lancée dans
un thread par ``asyncio.to_thread`` hérite de la clé de son chapitre, si bien que le code
du pipeline appelle :func:`step` sans savoir quel chapitre il traite. Hors de tout
chapitre (compilation, rendu Kdenlive), les sous-étapes vont à l'élément du traitement
lui-même (:data:`RUN_KEY`).
"""

from __future__ import annotations

import json
import os
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path
from typing import Any

ENV_VAR = "MM_PROGRESS_FILE"
#: Élément du traitement lui-même (compilation, rendu), quand aucun chapitre n'est lié.
RUN_KEY = "__run__"
#: Écart minimal entre deux écritures d'une simple avancée (``step``) ; un changement
#: d'étape ou d'état est écrit tout de suite.
WRITE_INTERVAL_S = 0.5

_key: ContextVar[str] = ContextVar("mm_progress_key", default=RUN_KEY)


class _Tracker:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.lock = threading.Lock()
        self.last_write = 0.0
        self.data: dict[str, Any] = {"pid": os.getpid(), "started": time.time(), "phase": "starting", "items": {}}

    def item(self, key: str) -> dict[str, Any]:
        return self.data["items"].setdefault(key, {})

    def write(self, force: bool = False) -> None:
        now = time.time()
        if not force and now - self.last_write < WRITE_INTERVAL_S:
            return
        self.data["updated"] = now
        text = json.dumps(self.data, ensure_ascii=False)
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            tmp.write_text(text, encoding="utf-8")
            # Windows refuse de remplacer un fichier qu'un lecteur tient ouvert : le Studio
            # le lit en quelques millisecondes, on réessaie donc brièvement.
            for attempt in range(5):
                try:
                    os.replace(tmp, self.path)
                    break
                except PermissionError:
                    if attempt == 4:
                        return
                    time.sleep(0.02)
        except OSError:
            return  # la progression est un confort : elle ne doit jamais faire échouer un traitement
        self.last_write = now


_tracker: _Tracker | None = None
_tracker_path: str | None = None
_init_lock = threading.Lock()


def _get() -> _Tracker | None:
    global _tracker, _tracker_path
    path = os.environ.get(ENV_VAR)
    if not path:
        return None
    if _tracker is None or _tracker_path != path:
        with _init_lock:
            if _tracker is None or _tracker_path != path:
                _tracker, _tracker_path = _Tracker(Path(path)), path
    return _tracker


def enabled() -> bool:
    """Vrai si un fichier de progression est demandé."""
    return _get() is not None


def reset() -> None:
    """Oublie l'état en mémoire (tests)."""
    global _tracker, _tracker_path
    _tracker, _tracker_path = None, None


def start(kind: str, **fields: Any) -> None:
    """Décrit le traitement (``batch``, ``render``...) ; les appels suivants complètent."""
    tracker = _get()
    if tracker is None:
        return
    with tracker.lock:
        tracker.data.update(kind=kind, **fields)
        tracker.write(force=True)


def phase(name: str, **fields: Any) -> None:
    """Phase du traitement : ``chapters``, ``compile``, puis ``done`` ou ``failed``."""
    tracker = _get()
    if tracker is None:
        return
    now = time.time()
    with tracker.lock:
        tracker.data.update(phase=name, phase_since=now, **fields)
        tracker.data.setdefault("phases", {})[name] = now
        run_item = tracker.data["items"].get(RUN_KEY)
        if run_item is not None:
            run_item["step"] = None  # la sous-étape d'une phase ne vaut plus pour la suivante
        if name in ("done", "failed"):
            tracker.data["finished"] = now
        tracker.write(force=True)


def bind(key: str) -> None:
    """Lie le contexte courant (tâche asyncio, thread) à l'élément ``key``."""
    _key.set(key)


@contextmanager
def bound(key: str) -> Iterator[None]:
    """Comme :func:`bind`, le temps d'un bloc."""
    token = _key.set(key)
    try:
        yield
    finally:
        _key.reset(token)


def item(key: str | None = None, **fields: Any) -> None:
    """Complète un élément (titre, numéro d'épisode, dossier...) sans changer son étape."""
    tracker = _get()
    if tracker is None:
        return
    with tracker.lock:
        tracker.item(key or _key.get()).update(fields)
        tracker.write(force=True)


def stage(name: str, state: str = "running", *, key: str | None = None, reason: str | None = None, **fields: Any) -> None:
    """Étape d'un élément et son état : ``waiting`` (``reason`` dit pourquoi) ou ``running``.

    L'heure de début d'une étape est celle où elle passe en ``running`` : le temps passé à
    attendre son tour n'est pas compté comme du travail.
    """
    tracker = _get()
    if tracker is None:
        return
    now = time.time()
    with tracker.lock:
        entry = tracker.item(key or _key.get())
        stages = entry.setdefault("stages", {})
        previous = entry.get("stage")
        if previous and previous != name and previous in stages and "end" not in stages[previous]:
            stages[previous]["end"] = now
        if previous != name or entry.get("state") != state:
            entry["since"] = now
            entry["step"] = None
        entry.update(stage=name, state=state, reason=reason if state == "waiting" else None, status="processing", **fields)
        if state == "running":
            stages.setdefault(name, {})["start"] = now
        tracker.write(force=True)


def step(step_id: str, label: str, done: int | None = None, total: int | None = None, *, key: str | None = None) -> None:
    """Sous-étape en cours (``Voix``, scène 11 sur 24...) ; écrite au plus toutes les 0,5 s."""
    tracker = _get()
    if tracker is None:
        return
    now = time.time()
    with tracker.lock:
        entry = tracker.item(key or _key.get())
        current = entry.get("step") or {}
        since = current.get("since", now) if current.get("id") == step_id else now
        entry["step"] = {"id": step_id, "label": label, "done": done, "total": total, "since": since}
        finished = total is not None and done is not None and done >= total
        tracker.write(force=current.get("id") != step_id or finished)


def finish(status: str, *, key: str | None = None, **fields: Any) -> None:
    """Fin d'un élément : ``done``, ``failed`` ou ``skipped``."""
    tracker = _get()
    if tracker is None:
        return
    now = time.time()
    with tracker.lock:
        entry = tracker.item(key or _key.get())
        current = entry.get("stage")
        if current and "end" not in entry.get("stages", {}).get(current, {"end": 0}):
            entry["stages"][current]["end"] = now
        entry.update(status=status, state=None, reason=None, step=None, finished=now, **fields)
        tracker.write(force=True)


def fail(message: str) -> None:
    """Échec expliqué : le message est gardé quand le processus sort ensuite en erreur."""
    phase("failed", error=message[:600])


@contextmanager
def outcome() -> Iterator[None]:
    """Termine le traitement en ``done`` ou ``failed`` selon la sortie du bloc.

    Une sortie ``typer.Exit(code=0)`` (liste affichée, rien à faire) compte comme réussie.
    """
    try:
        yield
    except BaseException as exc:
        code = getattr(exc, "exit_code", getattr(exc, "code", 1))
        if code in (0, None):
            phase("done")
        elif str(exc) and not isinstance(exc, SystemExit):
            phase("failed", error=f"{type(exc).__name__}: {exc}"[:600])
        else:
            phase("failed")  # garde le message déjà donné par :func:`fail`
        raise
    else:
        phase("done")


__all__ = [
    "ENV_VAR", "RUN_KEY", "enabled", "reset", "start", "phase", "bind", "bound", "item", "stage", "step", "finish", "fail", "outcome",
]
