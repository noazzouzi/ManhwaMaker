"""Serveur local de ManhwaMaker Studio (FastAPI) : API, flux en direct et interface web.

Le serveur ne fait aucun calcul lourd lui-même : il lance les traitements dans des
processus séparés (:mod:`src.studio.jobs`) et lit ce qu'ils produisent. L'interface
reçoit l'état des traitements chaque seconde par ``/api/stream`` (Server-Sent Events) ;
une page fermée puis rouverte retrouve tout, puisque rien ne vit dans le navigateur.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any

from fastapi import Body, FastAPI, HTTPException, Query, Request
from fastapi.responses import FileResponse, PlainTextResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from starlette.concurrency import run_in_threadpool

from src.studio import eta
from src.studio.jobs import VOICES, JobError, JobManager, pretty_series
from src.studio.library import Library
from src.utils.config import PROJECT_ROOT

logger = logging.getLogger(__name__)

STATIC_DIR = Path(__file__).parent / "static"
DEFAULT_DATA_DIR = PROJECT_ROOT / "studio_data"
STREAM_INTERVAL_S = 1.0
#: Traitements terminés renvoyés dans le flux (les plus récents).
RECENT_FINISHED = 6


def create_app(
    *, data_dir: str | Path = DEFAULT_DATA_DIR, out_root: str | Path = PROJECT_ROOT / "output",
    status_file: str | Path | None = PROJECT_ROOT / "batch_status.json", manager: JobManager | None = None,
    start_worker: bool = True,
) -> FastAPI:
    data_dir = Path(data_dir)
    manager = manager or JobManager(data_dir, status_file=status_file, out_root=out_root)
    library = Library(out_root, status_file, data_dir / "cache")
    app = FastAPI(title="ManhwaMaker Studio", docs_url=None, redoc_url=None)
    app.state.manager, app.state.library = manager, library
    if start_worker:
        manager.start()

    def video_or_404(name: str) -> dict[str, Any]:
        try:
            return library.get(name)
        except KeyError:
            raise HTTPException(404, "Vidéo introuvable") from None

    def job_or_404(job_id: str):
        try:
            return manager.get(job_id)
        except KeyError:
            raise HTTPException(404, "Traitement introuvable") from None

    # --- pages ----------------------------------------------------------------------------
    @app.get("/", include_in_schema=False)
    def index() -> FileResponse:
        return FileResponse(STATIC_DIR / "index.html", headers={"Cache-Control": "no-cache"})

    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

    # --- système ------------------------------------------------------------------------
    @app.get("/api/system")
    def system() -> dict[str, Any]:
        return _system_info()

    # --- vidéos -------------------------------------------------------------------------
    @app.get("/api/videos")
    def videos() -> list[dict[str, Any]]:
        return library.list()

    @app.get("/api/videos/{name}")
    def video(name: str) -> dict[str, Any]:
        try:
            return library.detail(name)
        except KeyError:
            raise HTTPException(404, "Vidéo introuvable") from None

    @app.get("/api/videos/{name}/thumb")
    def thumb(name: str) -> FileResponse:
        try:
            return FileResponse(library.thumbnail(name), media_type="image/jpeg", headers={"Cache-Control": "max-age=300"})
        except (KeyError, OSError):
            raise HTTPException(404, "Pas de vignette") from None

    @app.get("/api/videos/{name}/image")
    def image(name: str, path: str = Query(...), w: int = Query(320)) -> FileResponse:
        try:
            return FileResponse(library.image(name, path, w), media_type="image/jpeg", headers={"Cache-Control": "max-age=3600"})
        except (KeyError, OSError):
            raise HTTPException(404, "Image introuvable") from None

    @app.get("/api/videos/{name}/file")
    def media(name: str, path: str = Query(...)) -> FileResponse:
        try:
            return FileResponse(library.file(name, path))
        except KeyError:
            raise HTTPException(404, "Fichier introuvable") from None

    @app.post("/api/videos/{name}/open")
    def open_in(name: str, body: dict[str, Any] = Body(default={})) -> dict[str, Any]:
        item = video_or_404(name)
        folder = library.folder(name)
        what = body.get("what", "folder")
        if what == "folder":
            _open_path(folder)
        elif what == "kdenlive":
            if not item.get("kdenlive_project"):
                raise HTTPException(400, "Aucun projet Kdenlive : rendre d'abord la vidéo finale.")
            from src.modules.kdenlive_builder import KDENLIVE_BIN

            exe = KDENLIVE_BIN / "kdenlive.exe"
            if not exe.is_file():
                raise HTTPException(400, "Kdenlive n'est pas installé.")
            subprocess.Popen([str(exe), str(folder / item["kdenlive_project"])], cwd=str(folder))
        elif what == "file":
            _open_path(library.file(name, str(body.get("path") or "")))
        else:
            raise HTTPException(400, "Action inconnue")
        return {"ok": True}

    @app.get("/api/videos/{name}/render-estimate")
    def render_estimate(name: str, fps: int = 30, codec: str = "h264_amf") -> dict[str, Any]:
        item = video_or_404(name)
        seconds = eta.render_plan(manager.model, item["duration_s"], fps, codec)
        wait = _lane_wait(manager, "render")
        return {"seconds": seconds, "wait_s": wait, "video_s": item["duration_s"],
                "file": f"melt_{codec}_{fps}fps.mp4", "basis": "d'après les rendus déjà faits" if manager.history().get("renders")
                else "d'après le rendu d'essai du 26/09 (5 min 51 en 2 min 45)"}

    @app.post("/api/videos/{name}/render")
    def render(name: str, body: dict[str, Any] = Body(default={})) -> dict[str, Any]:
        item = video_or_404(name)
        if not item["has_timeline"]:
            raise HTTPException(400, "Cette vidéo n'a pas encore de montage.")
        title = f"{item['series']} · {item['episode']}".strip(" ·")
        try:
            job = manager.submit("render", body, folder=library.folder(name), title=title, video_s=item["duration_s"])
        except JobError as exc:
            raise HTTPException(400, str(exc)) from None
        return manager.summary(job)

    @app.post("/api/videos/{name}/redo")
    def redo(name: str, body: dict[str, Any] = Body(default={})) -> dict[str, Any]:
        from src.pipeline import slug_from_url

        item = video_or_404(name)
        url = item.get("url")
        if item["kind"] != "chapter" or not url:
            raise HTTPException(400, "Seul un chapitre peut être refait (une compilation se refait depuis ses chapitres).")
        if slug_from_url(url) != name:
            raise HTTPException(400, f"Ce dossier a un nom personnalisé : le pipeline écrirait dans « {slug_from_url(url)} ». "
                                     "Refaire depuis la ligne de commande avec --out.")
        stage = body.get("stage")
        failed = item["status"] in ("failed", "incomplete")
        params = {"url": url, "series": item["series"], "target": name,
                  "subtitle": f"Chapitre {item['episode_no']}" if item.get("episode_no") is not None else item["episode"],
                  **({"redo": stage} if stage and not failed else {})}
        try:
            job = manager.submit("batch", params)
        except JobError as exc:
            raise HTTPException(400, str(exc)) from None
        return manager.summary(job)

    # --- nouvelle vidéo --------------------------------------------------------------------
    @app.get("/api/inspect")
    async def inspect(url: str = Query(...)) -> dict[str, Any]:
        return await run_in_threadpool(_inspect, url, status_file, library)

    @app.post("/api/estimate")
    def estimate(body: dict[str, Any] = Body(default={})) -> dict[str, Any]:
        n = max(0, int(body.get("count") or 0))
        compile_video = bool(body.get("compile"))
        seconds = eta.batch_plan(manager.model, n, compile_video=compile_video)
        model = manager.model
        return {"seconds": seconds, "wait_s": _lane_wait(manager, "pipeline"),
                "per_chapter_s": seconds / n if n else 0.0,
                "basis": f"d'après {model.samples} chapitres déjà traités" if model.samples else "d'après des durées par défaut",
                "usd": round(1.04 * n, 2)}

    # --- traitements -----------------------------------------------------------------------
    @app.get("/api/jobs")
    def jobs() -> list[dict[str, Any]]:
        now = time.time()
        return [manager.summary(job, now) for job in manager.list()]

    @app.post("/api/jobs")
    def create_job(body: dict[str, Any] = Body(...)) -> dict[str, Any]:
        try:
            job = manager.submit("batch", body)
        except JobError as exc:
            raise HTTPException(400, str(exc)) from None
        return manager.summary(job)

    @app.get("/api/jobs/{job_id}")
    def job_detail(job_id: str) -> dict[str, Any]:
        return manager.detail(job_or_404(job_id))

    @app.get("/api/jobs/{job_id}/log", response_class=PlainTextResponse)
    def job_log(job_id: str) -> str:
        job_or_404(job_id)
        try:
            return manager.log_path(job_id).read_text(encoding="utf-8", errors="replace")
        except OSError:
            return ""

    @app.post("/api/jobs/{job_id}/cancel")
    def cancel(job_id: str) -> dict[str, Any]:
        job_or_404(job_id)
        return manager.summary(manager.cancel(job_id))

    @app.post("/api/jobs/{job_id}/retry")
    def retry(job_id: str) -> dict[str, Any]:
        old = job_or_404(job_id)
        try:
            if old.kind == "render":
                folder = library.folder(old.target or "")
                job = manager.submit("render", {k: v for k, v in old.params.items() if k != "video_s"}, folder=folder,
                                     title=old.title, video_s=float(old.params.get("video_s") or 0.0))
            else:
                job = manager.submit("batch", dict(old.params))
        except (JobError, KeyError) as exc:
            raise HTTPException(400, str(exc)) from None
        return manager.summary(job)

    @app.delete("/api/jobs/{job_id}")
    def remove(job_id: str) -> dict[str, Any]:
        job_or_404(job_id)
        try:
            manager.remove(job_id)
        except JobError as exc:
            raise HTTPException(400, str(exc)) from None
        return {"ok": True}

    @app.get("/api/stream")
    async def stream(request: Request) -> StreamingResponse:
        async def events():
            yield "retry: 2000\n\n"
            while not await request.is_disconnected():
                payload = await run_in_threadpool(_snapshot, manager)
                yield f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"
                await asyncio.sleep(STREAM_INTERVAL_S)

        return StreamingResponse(events(), media_type="text/event-stream", headers={"Cache-Control": "no-cache"})

    return app


# --- aides ----------------------------------------------------------------------------------
def _snapshot(manager: JobManager) -> dict[str, Any]:
    now = time.time()
    jobs = manager.list()
    active = [j for j in jobs if j.status in ("queued", "running")]
    finished = [j for j in jobs if j.status not in ("queued", "running")][:RECENT_FINISHED]
    return {"now": now, "jobs": [manager.summary(job, now) for job in active + finished]}


def _lane_wait(manager: JobManager, lane: str) -> float:
    now = time.time()
    wait = 0.0
    for job in manager.list():
        if job.lane != lane:
            continue
        if job.status == "running":
            wait += float(manager.summary(job, now).get("remaining_s") or 0.0)
        elif job.status == "queued":
            wait += job.planned_s or 0.0
    return wait


def _inspect(url: str, status_file: str | Path | None, library: Library) -> dict[str, Any]:
    """Série derrière un lien : chapitres disponibles et chapitres déjà faits."""
    from src.modules.scraper import chapter_ids, discover_episodes, is_series_url
    from src.pipeline import slug_from_url
    from src.studio.jobs import URL_PATTERN

    url = url.strip()
    if not URL_PATTERN.match(url):
        raise HTTPException(400, "Lien attendu : une page webtoons.com ou asurascans.com (https).")
    site = "Asura Scans" if "asurascans.com" in url else "Webtoons"
    try:
        episodes = discover_episodes(url)
    except Exception as exc:  # noqa: BLE001 - réseau, page inattendue
        raise HTTPException(502, f"Impossible de lire la liste des chapitres : {exc}") from None
    statuses = {}
    if status_file and Path(status_file).is_file():
        try:
            statuses = json.loads(Path(status_file).read_text(encoding="utf-8")).get("chapters") or {}
        except (OSError, ValueError):
            statuses = {}
    names = {item["name"]: item for item in library.list()}
    numbers = sorted(episodes)
    done, compiled, failed = [], [], []
    for number in numbers:
        ep_url = episodes[number]
        entry = statuses.get(ep_url) or {}
        folder = names.get(slug_from_url(ep_url))
        if folder and folder["has_timeline"]:
            done.append(number)
        elif entry.get("status") == "done":
            compiled.append(number)  # fait, puis compilé : dossier supprimé
        elif entry.get("status") == "failed":
            failed.append(number)
    series = next((it["series"] for it in names.values() if it.get("url") and chapter_ids(it["url"])[0] == chapter_ids(url)[0]), None)
    _, current = chapter_ids(url)
    return {
        "url": url, "site": site, "series": series or pretty_series(url), "is_series": is_series_url(url),
        "episodes": [n if not float(n).is_integer() else int(n) for n in numbers], "done": done, "compiled": compiled, "failed": failed,
        "current": current,
    }


def _system_info() -> dict[str, Any]:
    from src.modules.kdenlive_builder import KDENLIVE_BIN
    from src.utils.config import gemini_api_keys

    claude = shutil.which("claude") or shutil.which("claude.exe")
    try:
        keys = len(gemini_api_keys())
    except Exception:  # noqa: BLE001
        keys = 0
    gpu = None
    try:
        import onnxruntime

        gpu = "DirectML" if "DmlExecutionProvider" in onnxruntime.get_available_providers() else None
    except Exception:  # noqa: BLE001
        gpu = None
    return {
        "claude": bool(claude), "gemini_keys": keys, "kdenlive": (KDENLIVE_BIN / "melt.exe").is_file(), "gpu": gpu,
        "voices": list(VOICES),
    }


def _open_path(path: Path) -> None:
    if os.name == "nt":
        os.startfile(str(path))  # noqa: S606 - application locale
    else:  # pragma: no cover
        subprocess.Popen(["xdg-open", str(path)])


__all__ = ["create_app", "DEFAULT_DATA_DIR"]
