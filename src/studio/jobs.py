"""File d'attente du Studio : chaque traitement tourne dans son propre processus.

- Le serveur lance la ligne de commande habituelle (``python -m src.main batch ...`` ou le
  rendu Kdenlive) avec ``MM_PROGRESS_FILE`` : le processus écrit sa progression
  (:mod:`src.utils.progress`), le serveur la relit. Sa sortie va dans un fichier journal,
  pas dans un tube : le processus ne dépend jamais du serveur pour écrire.
- Fermer le navigateur ne change rien. Arrêter le serveur non plus : le processus continue,
  et au redémarrage le serveur le retrouve par son numéro (``pid``) et reprend son suivi.
- Deux files indépendantes (:data:`LANES`) : les lots (un à la fois, ils se partagent le
  quota et le processeur) et les rendus Kdenlive (un à la fois).
- ``jobs.json`` garde l'historique ; ``history.json`` les durées mesurées (compilation,
  rendus) qui affinent les estimations suivantes.
"""

from __future__ import annotations

import json
import logging
import os
import re
import subprocess
import sys
import threading
import time
import uuid
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from src.studio import eta
from src.utils.config import PROJECT_ROOT
from src.utils.progress import ENV_VAR

logger = logging.getLogger(__name__)

LANES = {"batch": "pipeline", "render": "render"}
ACTIVE = ("queued", "running")
STAGE_LABELS = {"prep": "Préparation", "script": "Script", "voice": "Voix", "montage": "Montage"}
#: Réglages imposés du lot (voir CLAUDE.md : au-delà de 5 requêtes/min, Gemini boucle en 429).
BATCH_DEFAULTS = ["--max-chapters", "3", "--max-gemini-rpm", "5"]
URL_PATTERN = re.compile(r"^https://(www\.|m\.)?(webtoons\.com|asurascans\.com)/\S+$")
VOICES = ("am_fenrir,am_michael", "am_fenrir", "am_michael", "am_puck")
REDO_STAGES = ("analyze", "tts", "montage")
MAX_FINISHED = 50


class JobError(ValueError):
    """Demande de traitement invalide."""


@dataclass
class Job:
    id: str
    kind: str
    title: str
    subtitle: str
    params: dict[str, Any]
    cmd: list[str]
    status: str = "queued"
    created: float = field(default_factory=time.time)
    started: float | None = None
    finished: float | None = None
    pid: int | None = None
    returncode: int | None = None
    error: str | None = None
    target: str | None = None
    planned_s: float | None = None

    @property
    def lane(self) -> str:
        return LANES[self.kind]


def _pid_alive(pid: int) -> bool:
    if os.name == "nt":
        import ctypes

        kernel32 = ctypes.windll.kernel32
        handle = kernel32.OpenProcess(0x1000, False, int(pid))  # PROCESS_QUERY_LIMITED_INFORMATION
        if not handle:
            return False
        try:
            code = ctypes.c_ulong()
            return bool(kernel32.GetExitCodeProcess(handle, ctypes.byref(code))) and code.value == 259  # STILL_ACTIVE
        finally:
            kernel32.CloseHandle(handle)
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def _kill_tree(pid: int) -> None:
    if os.name == "nt":
        subprocess.run(["taskkill", "/PID", str(pid), "/T", "/F"], capture_output=True,
                       creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    else:  # pragma: no cover - le Studio tourne sous Windows
        os.killpg(os.getpgid(pid), 9)


def pretty_series(url: str) -> str:
    """Nom lisible d'une série à partir de son URL (``bad-born-blood-05c7df14`` -> ``Bad Born Blood``)."""
    parts = [p for p in re.split(r"[/?#]", url.split("://", 1)[-1]) if p]
    slug = ""
    if "asurascans.com" in url and "comics" in parts:
        slug = parts[parts.index("comics") + 1] if parts.index("comics") + 1 < len(parts) else ""
        slug = re.sub(r"-[0-9a-f]{6,}$", "", slug)
    elif len(parts) >= 4:
        slug = parts[3] if parts[1] in ("en", "fr", "es", "de") else parts[2]
    return " ".join(w.capitalize() for w in slug.split("-") if w) or "Série"


def _chapter_range(start: Any, end: Any) -> str:
    if start is None and end is None:
        return "tous les chapitres"
    if start == end or end is None:
        return f"chapitre {start}"
    return f"chapitres {start} à {end}"


class JobManager:
    """File d'attente persistante, processus séparés, suivi et estimations."""

    def __init__(
        self, data_dir: str | Path, *, status_file: str | Path | None = None, python: str = sys.executable,
        cwd: str | Path = PROJECT_ROOT, out_root: str | Path = PROJECT_ROOT / "output",
        popen: Callable[..., Any] = subprocess.Popen, poll_interval: float = 0.5,
    ) -> None:
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        (self.data_dir / "jobs").mkdir(exist_ok=True)
        self.status_file = Path(status_file) if status_file else None
        self.python, self.cwd, self.out_root = python, Path(cwd), Path(out_root)
        self.popen = popen
        self.poll_interval = poll_interval
        self.jobs: dict[str, Job] = {}
        self.procs: dict[str, Any] = {}
        self.lock = threading.RLock()
        self._progress_cache: dict[str, tuple[float, dict[str, Any]]] = {}
        self._model: eta.StageModel | None = None
        self._model_at = 0.0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._load()

    # --- persistance -------------------------------------------------------------------
    @property
    def jobs_file(self) -> Path:
        return self.data_dir / "jobs.json"

    @property
    def history_file(self) -> Path:
        return self.data_dir / "history.json"

    def job_dir(self, job_id: str) -> Path:
        return self.data_dir / "jobs" / job_id

    def log_path(self, job_id: str) -> Path:
        return self.job_dir(job_id) / "output.log"

    def progress_path(self, job_id: str) -> Path:
        return self.job_dir(job_id) / "progress.json"

    def _load(self) -> None:
        try:
            raw = json.loads(self.jobs_file.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            raw = []
        fields = set(Job.__dataclass_fields__)
        for data in raw:
            job = Job(**{k: v for k, v in data.items() if k in fields})
            self.jobs[job.id] = job
        # Traitements lancés avant un redémarrage du serveur : on reprend leur suivi.
        for job in self.jobs.values():
            if job.status == "running" and not (job.pid and _pid_alive(job.pid)):
                self._conclude(job, None)
        self._save()

    def _save(self) -> None:
        with self.lock:
            finished = sorted((j for j in self.jobs.values() if j.status not in ACTIVE), key=lambda j: j.finished or j.created)
            for old in finished[:-MAX_FINISHED]:
                del self.jobs[old.id]
            data = [asdict(j) for j in sorted(self.jobs.values(), key=lambda j: j.created)]
        tmp = self.jobs_file.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
        os.replace(tmp, self.jobs_file)

    def history(self) -> dict[str, Any]:
        try:
            return json.loads(self.history_file.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {}

    def _record(self, key: str, sample: Any) -> None:
        history = self.history()
        history.setdefault(key, []).append(sample)
        history[key] = history[key][-50:]
        self.history_file.write_text(json.dumps(history, indent=1), encoding="utf-8")
        self._model = None

    @property
    def model(self) -> eta.StageModel:
        if self._model is None or time.time() - self._model_at > 60:
            self._model = eta.StageModel.from_history(self.status_file, self.history())
            self._model_at = time.time()
        return self._model

    # --- demandes ----------------------------------------------------------------------
    def build_batch(self, params: dict[str, Any]) -> tuple[list[str], str, str]:
        url = str(params.get("url") or "").strip()
        if not URL_PATTERN.match(url):
            raise JobError("Lien attendu : une page webtoons.com ou asurascans.com (https).")
        cmd = [self.python, "-m", "src.main", "batch", url, *BATCH_DEFAULTS]
        start, end = params.get("start"), params.get("end")
        for flag, value in (("--start-chapter", start), ("--end-chapter", end)):
            if value is not None:
                if not isinstance(value, int) or value < 0:
                    raise JobError("Numéros de chapitres invalides.")
                cmd += [flag, str(value)]
        if start is not None and end is not None and end < start:
            raise JobError("Le dernier chapitre doit suivre le premier.")
        fmt = str(params.get("format") or "LONG").upper()
        if fmt not in ("LONG", "SHORT"):
            raise JobError("Format inconnu.")
        cmd += ["--format", fmt]
        if params.get("compile"):
            cmd.append("--compile")
            if params.get("keep_chapters"):
                cmd.append("--keep-chapters")
        voice = params.get("voice")
        if voice:
            if voice not in VOICES:
                raise JobError("Voix inconnue.")
            if voice != VOICES[0]:
                cmd += ["--voice", voice]
        speed = params.get("speed")
        if speed is not None:
            speed = float(speed)
            if not 0.8 <= speed <= 1.6:
                raise JobError("Vitesse de voix hors limites (0,8 à 1,6).")
            if abs(speed - 1.2) > 1e-6:  # 1,2 = vitesse par défaut des deux formats
                cmd += ["--speed", f"{speed:g}"]
        if params.get("cta") is False:
            cmd.append("--no-cta")
        if params.get("bgm") is False:
            cmd.append("--no-bgm")
        if params.get("sfx") is False:
            cmd.append("--no-sfx")
        if params.get("script_ai") == "gemini":
            cmd += ["--script-ai", "gemini"]
        if params.get("language") == "fr":
            cmd += ["--language", "fr"]
        redo = params.get("redo")
        if redo:
            if redo not in REDO_STAGES:
                raise JobError("Étape à refaire inconnue.")
            cmd += ["--redo", redo]
        if params.get("force"):
            cmd.append("--force")
        series = params.get("series") or pretty_series(url)
        if redo:
            title = f"{series} · refaire {dict(analyze='le script', tts='la voix', montage='le montage')[redo]}"
        else:
            title = series
        subtitle = str(params.get("subtitle") or _chapter_range(start, end).capitalize())
        if params.get("compile"):
            subtitle += " · une seule vidéo"
        return cmd, title, subtitle

    def build_render(self, params: dict[str, Any], folder: Path, title: str) -> tuple[list[str], str, str]:
        fps = int(params.get("fps") or 30)
        codec = str(params.get("codec") or "h264_amf")
        if fps not in (30, 60) or codec not in ("h264_amf", "libx264"):
            raise JobError("Réglages de rendu invalides.")
        cmd = [self.python, "-m", "src.modules.kdenlive_builder", str(folder), "--fps", str(fps), "--codec", codec]
        seconds = params.get("seconds")
        if seconds:
            cmd += ["--seconds", str(float(seconds))]
        subtitle = f"Vidéo finale · {fps} i/s · {'carte graphique' if codec == 'h264_amf' else 'processeur'}"
        if seconds:
            subtitle += f" · extrait de {float(seconds):g} s"
        return cmd, title, subtitle

    def submit(self, kind: str, params: dict[str, Any], *, folder: Path | None = None, title: str = "", video_s: float = 0.0) -> Job:
        if kind == "batch":
            cmd, title, subtitle = self.build_batch(params)
            n = 1 if params.get("redo") or params.get("start") is None else int(params.get("end", params["start"])) - int(params["start"]) + 1
            planned = eta.batch_plan(self.model, n, compile_video=bool(params.get("compile")))
            target = params.get("target")
        elif kind == "render":
            if folder is None:
                raise JobError("Vidéo introuvable.")
            cmd, title, subtitle = self.build_render(params, folder, title)
            params = {**params, "video_s": video_s}
            planned = eta.render_plan(self.model, float(params.get("seconds") or video_s), int(params.get("fps") or 30),
                                      str(params.get("codec") or "h264_amf"))
            target = folder.name
        else:
            raise JobError("Type de traitement inconnu.")
        job = Job(id=uuid.uuid4().hex[:10], kind=kind, title=title, subtitle=subtitle, params=params, cmd=cmd,
                  target=target, planned_s=round(planned, 1))
        with self.lock:
            self.jobs[job.id] = job
        self._save()
        if self._thread is not None:  # sans boucle (tests), la file reste en attente
            self.tick()
        return job

    def cancel(self, job_id: str) -> Job:
        with self.lock:
            job = self.jobs[job_id]
            if job.status == "queued":
                job.status, job.finished = "cancelled", time.time()
            elif job.status == "running":
                if job.pid:
                    _kill_tree(job.pid)
                proc = self.procs.pop(job.id, None)
                if proc is not None:
                    try:
                        proc.wait(timeout=10)
                    except Exception:  # noqa: BLE001
                        pass
                job.status, job.finished, job.error = "cancelled", time.time(), "Arrêté à la demande."
        self._save()
        return job

    def remove(self, job_id: str) -> None:
        with self.lock:
            job = self.jobs[job_id]
            if job.status in ACTIVE:
                raise JobError("Un traitement en cours ne peut pas être retiré : l'arrêter d'abord.")
            del self.jobs[job_id]
        self._save()

    # --- boucle -------------------------------------------------------------------------
    def start(self) -> None:
        if self._thread is None:
            self._thread = threading.Thread(target=self._loop, name="studio-jobs", daemon=True)
            self._thread.start()

    def stop(self) -> None:
        self._stop.set()

    def _loop(self) -> None:
        while not self._stop.wait(self.poll_interval):
            try:
                self.tick()
            except Exception:  # noqa: BLE001 - la boucle ne doit jamais mourir
                logger.exception("Boucle des traitements")

    def tick(self) -> None:
        changed = False
        with self.lock:
            for job in [j for j in self.jobs.values() if j.status == "running"]:
                proc = self.procs.get(job.id)
                code = proc.poll() if proc is not None else (None if job.pid and _pid_alive(job.pid) else -1)
                if code is not None:
                    self.procs.pop(job.id, None)
                    self._conclude(job, None if proc is None else code)
                    changed = True
            busy = {j.lane for j in self.jobs.values() if j.status == "running"}
            for job in sorted((j for j in self.jobs.values() if j.status == "queued"), key=lambda j: j.created):
                if job.lane in busy:
                    continue
                self._launch(job)
                busy.add(job.lane)
                changed = True
        if changed:
            self._save()

    def _launch(self, job: Job) -> None:
        folder = self.job_dir(job.id)
        folder.mkdir(parents=True, exist_ok=True)
        env = {**os.environ, ENV_VAR: str(self.progress_path(job.id)), "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8",
               "PYTHONUNBUFFERED": "1"}
        flags = 0
        if os.name == "nt":
            # Groupe à part : un Ctrl+C dans la console du serveur n'arrête pas le traitement.
            flags = subprocess.CREATE_NEW_PROCESS_GROUP | getattr(subprocess, "CREATE_NO_WINDOW", 0)
        with open(self.log_path(job.id), "ab") as log:
            try:
                proc = self.popen(job.cmd, cwd=str(self.cwd), stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                                  env=env, creationflags=flags)
            except OSError as exc:
                job.status, job.finished, job.error = "failed", time.time(), f"Lancement impossible : {exc}"
                return
        self.procs[job.id] = proc
        job.status, job.started, job.pid = "running", time.time(), getattr(proc, "pid", None)
        logger.info("Traitement %s lancé (pid %s) : %s", job.id, job.pid, " ".join(job.cmd[2:]))

    def _conclude(self, job: Job, code: int | None) -> None:
        """Fin d'un processus : issue lue dans le code de sortie, ou dans la progression s'il a été perdu de vue."""
        prog = self.progress(job.id, fresh=True)
        phase = prog.get("phase")
        job.finished = prog.get("finished") or time.time()
        job.returncode = code
        if code is None:
            job.status = {"done": "done", "failed": "failed"}.get(phase, "interrupted")
        else:
            job.status = "done" if code == 0 and phase != "failed" else "failed"
        if job.status == "failed":
            job.error = prog.get("error") or self._last_error(job.id) or f"Code de sortie {code}"
        elif job.status == "interrupted":
            job.error = "Suivi perdu : le processus s'est arrêté sans dire comment."
        self._learn(job, prog)

    def _learn(self, job: Job, prog: dict[str, Any]) -> None:
        phases = prog.get("phases") or {}
        if job.status != "done":
            return
        try:
            if job.kind == "batch" and "compile" in phases and "done" in phases:
                self._record("compile_s", round(phases["done"] - phases["compile"], 1))
            if job.kind == "render" and "render" in phases and "done" in phases and prog.get("video_s"):
                self._record("renders", {
                    "codec": prog.get("codec"), "fps": prog.get("fps"), "video_s": prog.get("video_s"),
                    "project_s": round(phases["render"] - phases.get("project", prog.get("started", phases["render"])), 1),
                    "melt_s": round(phases["done"] - phases["render"], 1),
                })
            if job.started and job.finished:
                self._record("jobs", {"kind": job.kind, "planned_s": job.planned_s, "actual_s": round(job.finished - job.started, 1)})
        except (OSError, TypeError, ValueError):
            logger.warning("Historique des durées non mis à jour", exc_info=True)

    def _last_error(self, job_id: str) -> str | None:
        lines = self.log_tail(job_id, 80)
        for line in reversed(lines):
            if "FAILED:" in line or " ERROR " in line or "Error" in line:
                return line.split("FAILED:", 1)[-1].strip()[:400]
        return lines[-1][:400] if lines else None

    # --- lecture -------------------------------------------------------------------------
    def progress(self, job_id: str, fresh: bool = False) -> dict[str, Any]:
        path = self.progress_path(job_id)
        try:
            mtime = path.stat().st_mtime
        except OSError:
            return {}
        cached = self._progress_cache.get(job_id)
        if cached and cached[0] == mtime and not fresh:
            return cached[1]
        for _ in range(3):
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                self._progress_cache[job_id] = (mtime, data)
                return data
            except (OSError, ValueError):
                time.sleep(0.02)
        return cached[1] if cached else {}

    def log_tail(self, job_id: str, lines: int = 200) -> list[str]:
        path = self.log_path(job_id)
        try:
            with open(path, "rb") as handle:
                handle.seek(0, os.SEEK_END)
                size = handle.tell()
                handle.seek(max(0, size - 96_000))
                text = handle.read().decode("utf-8", errors="replace")
        except OSError:
            return []
        rows = [row.rstrip() for row in text.replace("\r", "\n").split("\n") if row.strip()]
        return rows[-lines:]

    def get(self, job_id: str) -> Job:
        return self.jobs[job_id]

    def list(self) -> list[Job]:
        with self.lock:
            return sorted(self.jobs.values(), key=lambda j: (j.status not in ACTIVE, -(j.started or j.created) if j.status not in ACTIVE else j.created))

    # --- vues ---------------------------------------------------------------------------
    def summary(self, job: Job, now: float | None = None) -> dict[str, Any]:
        """État lisible d'un traitement : étape en cours, temps restant, heure de fin."""
        now = time.time() if now is None else now
        prog = self.progress(job.id) if job.status != "queued" else {}
        view: dict[str, Any] = {
            "id": job.id, "kind": job.kind, "lane": job.lane, "title": job.title, "subtitle": job.subtitle, "status": job.status,
            "created": job.created, "started": job.started, "finished": job.finished, "error": job.error, "target": job.target,
            "params": job.params, "phase": prog.get("phase"), "current": [], "remaining_s": None, "eta_at": None,
            "percent": 100.0 if job.status == "done" else None, "overrun": False, "counts": {}, "planned_s": job.planned_s,
            "basis": self._basis(job),
        }
        if job.status == "queued":
            wait = self._queue_wait(job, now)
            view.update(wait_s=wait, remaining_s=wait + (job.planned_s or 0.0), eta_at=now + wait + (job.planned_s or 0.0), percent=0.0)
            view["current"] = [f"En attente · démarre dans ≈ {_short(wait)}" if wait > 1 else "Démarrage…"]
            return view
        if job.status != "running":
            if job.kind == "batch":
                view["counts"] = self._counts(prog)
            return view
        if job.kind == "batch":
            est = eta.batch_estimate(self.model, prog, now, compile_video=bool(job.params.get("compile")),
                                     planned_chapters=self._planned_chapters(job))
            view["counts"] = self._counts(prog)
            view["current"] = self._batch_lines(prog, now)
        else:
            video_s = float(job.params.get("video_s") or 0.0)
            if job.params.get("seconds"):
                video_s = min(video_s or float(job.params["seconds"]), float(job.params["seconds"]))
            est = eta.render_estimate(self.model, prog, now, video_s=video_s,
                                      fps=int(job.params.get("fps") or 30), codec=str(job.params.get("codec") or "h264_amf"))
            view["current"] = self._render_lines(prog)
        remaining = float(est.get("remaining_s") or 0.0)
        elapsed = now - (job.started or now)
        view.update(remaining_s=remaining, eta_at=now + remaining, overrun=bool(est.get("overrun")),
                    percent=round(100.0 * elapsed / max(1.0, elapsed + remaining), 1), chapter_eta=est.get("chapters") or {})
        return view

    def detail(self, job: Job, now: float | None = None) -> dict[str, Any]:
        now = time.time() if now is None else now
        view = self.summary(job, now)
        prog = self.progress(job.id) if job.status != "queued" else {}
        chapter_eta = view.pop("chapter_eta", {}) or {}
        chapters = []
        items = prog.get("items") or {}
        for key, entry in sorted(((k, v) for k, v in items.items() if k != "__run__"), key=lambda kv: kv[1].get("order", 0)):
            after = entry.get("after")
            chapters.append({
                "key": key, "order": entry.get("order"), "episode_no": entry.get("episode_no"), "title": entry.get("title"),
                "status": entry.get("status"), "stage": entry.get("stage"), "state": entry.get("state"),
                "reason": _reason(entry, items.get(after) if after else None), "step": entry.get("step"),
                "stages": entry.get("stages") or {}, "since": entry.get("since"), "error": entry.get("error"),
                "out_dir": Path(entry["out_dir"]).name if entry.get("out_dir") else None,
                "remaining_s": chapter_eta.get(key), "eta_at": now + chapter_eta[key] if key in chapter_eta else None,
                "video_s": entry.get("video_s"), "finished": entry.get("finished"),
            })
        run_item = items.get("__run__") or {}
        view.update(chapters=chapters, run_step=run_item.get("step"), phases=prog.get("phases") or {},
                    cmd=" ".join(job.cmd[2:]), log=self.log_tail(job.id, 250), output=prog.get("output"))
        return view

    # --- aides ---------------------------------------------------------------------------
    def _planned_chapters(self, job: Job) -> int:
        start, end = job.params.get("start"), job.params.get("end")
        if job.params.get("redo") or start is None:
            return 1
        return int(end if end is not None else start) - int(start) + 1

    def _queue_wait(self, job: Job, now: float) -> float:
        wait = 0.0
        for other in self.list():
            if other.lane != job.lane or other.id == job.id:
                continue
            if other.status == "running":
                wait += float(self.summary(other, now).get("remaining_s") or 0.0)
            elif other.status == "queued" and other.created < job.created:
                wait += other.planned_s or 0.0
        return wait

    def _basis(self, job: Job) -> str:
        model = self.model
        if job.kind == "render":
            return "d'après les rendus déjà faits" if self.history().get("renders") else "d'après le rendu d'essai du 26/09"
        if model.samples:
            return f"d'après {model.samples} chapitres déjà traités, recalculé en direct"
        return "d'après des durées par défaut, recalculé en direct"

    @staticmethod
    def _counts(prog: dict[str, Any]) -> dict[str, int]:
        counts = {"total": 0, "done": 0, "running": 0, "waiting": 0, "pending": 0, "failed": 0, "skipped": 0}
        for key, entry in (prog.get("items") or {}).items():
            if key == "__run__":
                continue
            counts["total"] += 1
            status = entry.get("status")
            if status == "processing":
                counts["running" if entry.get("state") == "running" else "waiting"] += 1
            elif status in counts:
                counts[status] += 1
        return counts

    def _batch_lines(self, prog: dict[str, Any], now: float) -> list[str]:
        phase = prog.get("phase")
        if phase in (None, "starting", "resolve"):
            return ["Recherche des chapitres…"]
        if phase == "compile":
            step = ((prog.get("items") or {}).get("__run__") or {}).get("step") or {}
            return ["Compilation · " + _step_text(step) if step else "Compilation des chapitres"]
        lines = []
        running = [(k, v) for k, v in (prog.get("items") or {}).items() if k != "__run__" and v.get("status") == "processing" and v.get("state") == "running"]
        for _, entry in sorted(running, key=lambda kv: kv[1].get("order", 0)):
            label = f"Ch. {_episode(entry)} · {STAGE_LABELS.get(entry.get('stage'), entry.get('stage'))}"
            step = entry.get("step")
            lines.append(label + (" · " + _step_text(step) if step else ""))
        return lines or ["Chapitres en attente de leur tour"]

    @staticmethod
    def _render_lines(prog: dict[str, Any]) -> list[str]:
        step = ((prog.get("items") or {}).get("__run__") or {}).get("step") or {}
        if prog.get("phase") == "render":
            return ["Rendu · " + _step_text(step) if step else "Rendu de la vidéo"]
        return ["Préparation du projet · " + _step_text(step) if step else "Préparation du projet Kdenlive"]


def _episode(entry: dict[str, Any]) -> str:
    number = entry.get("episode_no")
    if number is None:
        return "?"
    return f"{number:g}" if isinstance(number, float) else str(number)


def _step_text(step: dict[str, Any]) -> str:
    label = step.get("label") or ""
    done, total = step.get("done"), step.get("total")
    if total:
        if step.get("id") in ("melt", "preview"):
            return f"{label} {round(100 * (done or 0) / total)} %"
        return f"{label} {min(done or 0, total)}/{total}"
    return label


def _reason(entry: dict[str, Any], previous: dict[str, Any] | None) -> str | None:
    if entry.get("status") != "processing" or entry.get("state") != "waiting":
        return None
    if entry.get("reason") == "previous":
        return f"Attend le script du chapitre {_episode(previous)}" if previous else "Attend le chapitre précédent"
    return f"Attend une place libre ({STAGE_LABELS.get(entry.get('stage'), '')})"


def _short(seconds: float) -> str:
    seconds = max(0, int(round(seconds)))
    if seconds < 60:
        return f"{seconds} s"
    minutes = round(seconds / 60)
    if minutes < 60:
        return f"{minutes} min"
    return f"{minutes // 60} h {minutes % 60:02d}"


__all__ = ["Job", "JobError", "JobManager", "pretty_series", "LANES", "VOICES"]
