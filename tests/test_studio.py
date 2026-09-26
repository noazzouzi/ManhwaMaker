"""Tests de ManhwaMaker Studio : progression, temps restant, file des traitements, bibliothèque, API."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import pytest

from src.studio import eta
from src.studio.jobs import Job, JobError, JobManager, pretty_series
from src.studio.library import Library
from src.utils import progress

FAKE_BATCH = Path(__file__).with_name("studio_fake_batch.py")


# --- Progression -------------------------------------------------------------------------
@pytest.fixture
def progress_file(tmp_path, monkeypatch):
    path = tmp_path / "progress.json"
    monkeypatch.setenv(progress.ENV_VAR, str(path))
    progress.reset()
    yield path
    progress.reset()


def test_progress_is_silent_without_the_environment_variable(tmp_path, monkeypatch) -> None:
    monkeypatch.delenv(progress.ENV_VAR, raising=False)
    progress.reset()
    progress.start("batch")
    progress.stage("prep")
    progress.step("download", "x", 1, 2)
    assert not progress.enabled() and list(tmp_path.iterdir()) == []


def test_progress_records_stages_waits_steps_and_outcome(progress_file) -> None:
    progress.start("batch", compile=True)
    progress.phase("chapters")
    with progress.bound("ch1"):
        progress.stage("prep", "waiting", reason="slot")
        progress.stage("prep")
        progress.step("download", "Téléchargement", 3, 10)
        progress.stage("script", "waiting", reason="previous", after="ch0")
        progress.stage("script")
        progress.finish("done", video_s=12.0)
    progress.step("preview", "Aperçu", 1, 4)  # hors chapitre : élément du traitement
    data = json.loads(progress_file.read_text(encoding="utf-8"))
    item = data["items"]["ch1"]
    assert data["kind"] == "batch" and data["compile"] is True and data["phase"] == "chapters"
    assert item["status"] == "done" and item["video_s"] == 12.0 and item["after"] == "ch0"
    assert item["stages"]["prep"]["start"] <= item["stages"]["prep"]["end"] <= item["stages"]["script"]["start"]
    assert "end" in item["stages"]["script"] and item["step"] is None
    with pytest.raises(RuntimeError):
        with progress.outcome():
            raise RuntimeError("boom")
    data = json.loads(progress_file.read_text(encoding="utf-8"))
    assert data["phase"] == "failed" and "boom" in data["error"] and "failed" in data["phases"]


def test_progress_step_is_bound_to_the_calling_context(progress_file) -> None:
    import asyncio

    async def chapter(key: str) -> None:
        progress.bind(key)
        await asyncio.to_thread(progress.step, "voice", "Voix", 1, 2)

    async def run() -> None:
        await asyncio.gather(chapter("a"), chapter("b"))

    asyncio.run(run())
    progress.step("x", "y", 1, 1, key="a")  # écriture forcée finale
    items = json.loads(progress_file.read_text(encoding="utf-8"))["items"]
    assert items["b"]["step"]["id"] == "voice" and "__run__" not in items


# --- Temps restant ----------------------------------------------------------------------
def test_simulation_respects_capacities_and_series_order() -> None:
    chapters = [eta.SimChapter(str(i), [("prep", 10), ("script", 10), ("voice", 10), ("montage", 1)], after=str(i - 1) if i else None)
                for i in range(3)]
    finish = eta.simulate(chapters, {"prep": 3, "script": 3, "voice": 2, "montage": 1})
    # Préparations en parallèle, scripts à la file (ordre de la série), voix sur 2 places.
    assert finish["0"] == pytest.approx(31) and finish["1"] == pytest.approx(41) and finish["2"] == pytest.approx(51)
    free = eta.simulate([eta.SimChapter(c.key, c.remaining) for c in chapters], {"prep": 3, "script": 3, "voice": 2, "montage": 1})
    assert max(free.values()) < max(finish.values())


def test_running_stage_uses_real_progress_and_never_reaches_zero() -> None:
    model = eta.StageModel(timings={**eta.DEFAULT_TIMINGS, "tts": 100.0})
    now = 1000.0
    entry = {"since": now - 30, "step": {"id": "voice", "done": 6, "total": 24, "since": now - 30}}
    left, overrun = eta.running_remaining(model, "voice", entry, {}, now)
    assert 60 < left < 100 and not overrun  # 1/4 fait en 30 s : environ 90 s, mêlé au plan
    late = {"since": now - 400, "step": {"id": "script", "since": now - 400}}
    left, overrun = eta.running_remaining(model, "script", late, {}, now)
    assert overrun and left >= eta.MIN_REMAINING_S


def test_batch_estimate_from_a_progress_snapshot() -> None:
    model = eta.StageModel()
    now = 5000.0
    prog = {"phase": "chapters", "compile": True, "parallel": {"prep": 3, "script": 3, "voice": 2, "montage": 1},
            "predecessors": {"b": "a", "c": "b"}, "montage_steps": ["timeline"],
            "items": {"a": {"order": 0, "status": "done"},
                      "b": {"order": 1, "status": "processing", "stage": "voice", "state": "running", "since": now - 10,
                            "step": {"id": "voice", "done": 5, "total": 10, "since": now - 10}},
                      "c": {"order": 2, "status": "processing", "stage": "script", "state": "waiting", "since": now - 5}}}
    est = eta.batch_estimate(model, prog, now)
    assert set(est["chapters"]) == {"b", "c"} and est["chapters"]["b"] < est["chapters"]["c"]
    assert est["remaining_s"] == pytest.approx(max(est["chapters"].values()) + model.compile_s)
    assert eta.batch_estimate(model, {**prog, "phase": "done"}, now)["remaining_s"] == 0


def test_estimate_learns_from_the_chapters_already_done_in_this_batch() -> None:
    model = eta.StageModel()
    now = 9000.0
    fast = {"prep": {"start": 0.0, "end": 10.0}, "script": {"start": 10.0, "end": 20.0}}
    pending = {"order": 2, "status": "pending"}
    prog = {"phase": "chapters", "items": {"a": {"order": 0, "status": "done", "stages": fast},
                                            "b": {"order": 1, "status": "done", "stages": fast}, "c": pending}}
    scale = eta.job_scale(model, prog["items"], {})
    assert scale["prep"] < 0.5 and scale["script"] < 0.6 and "voice" not in scale
    learned = eta.batch_estimate(model, prog, now)["remaining_s"]
    naive = eta.batch_estimate(model, {"phase": "chapters", "items": {"c": pending}}, now)["remaining_s"]
    assert learned < naive


def test_stage_model_reads_history_and_calibrates(tmp_path) -> None:
    status = {"chapters": {f"u{i}": {"status": "done", "finished": f"2026-09-2{i}", "timings": {"analyze": 60.0 + i, "tts": 100.0}}
                           for i in range(5)},
              "run": {"done": 5, "elapsed_s": 5000.0, "timings": {"steps": {"analyze": {"median_s": 60}, "tts": {"median_s": 100}}},
                      "parallelism": {"max_chapters": 3, "max_tts_workers": 2}}}
    path = tmp_path / "batch_status.json"
    path.write_text(json.dumps(status), encoding="utf-8")
    model = eta.StageModel.from_history(path, {"compile_s": [80.0, 100.0], "renders": [{"codec": "h264_amf", "fps": 30, "video_s": 300, "melt_s": 150, "project_s": 15}]})
    assert model.samples == 5 and model.timings["analyze"] == 62.0 and model.calibration == 2.0  # borné
    assert model.compile_s == 90.0 and model.render_ratio[("h264_amf", 30)] == 0.5
    assert eta.batch_plan(model, 0, compile_video=True) == 0.0
    assert eta.batch_plan(model, 10, compile_video=True) > eta.batch_plan(model, 5, compile_video=True)


# --- File des traitements ------------------------------------------------------------------
def fake_job(manager: JobManager, root: Path, *args: str, kind: str = "batch", params: dict | None = None) -> Job:
    job = Job(id=f"t{len(manager.jobs)}{int(time.time() * 1000) % 100000}", kind=kind, title="Serie Test", subtitle="test",
              params=params or {}, cmd=[sys.executable, str(FAKE_BATCH), str(root), *args])
    manager.jobs[job.id] = job
    return job


def wait_for(manager: JobManager, job: Job, statuses: tuple[str, ...], timeout: float = 60.0) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        manager.tick()
        if job.status in statuses:
            return
        time.sleep(0.2)
    raise AssertionError(f"statut {job.status}, journal : {manager.log_tail(job.id, 20)}")


def test_job_runs_in_its_own_process_and_reports_progress(tmp_path) -> None:
    manager = JobManager(tmp_path / "studio", status_file=tmp_path / "batch_status.json", poll_interval=0.1)
    job = fake_job(manager, tmp_path, "3", "0.3", "--fail", "2", params={"start": 1, "end": 3})
    manager.tick()
    assert job.status == "running" and job.pid
    seen_running = False
    deadline = time.time() + 60
    while job.status == "running" and time.time() < deadline:
        view = manager.summary(job)
        if view["phase"] == "chapters" and view["current"] and view["remaining_s"]:
            seen_running = True
            assert view["remaining_s"] > 0 and 0 <= view["percent"] < 100 and view["eta_at"] > time.time()
        time.sleep(0.2)
        manager.tick()
    assert seen_running and job.status == "done"
    detail = manager.detail(job)
    statuses = {c["episode_no"]: c["status"] for c in detail["chapters"]}
    assert statuses == {1: "done", 2: "failed", 3: "done"}
    assert "echec simule" in next(c for c in detail["chapters"] if c["episode_no"] == 2)["error"]
    assert detail["counts"]["failed"] == 1 and any("termine" in line for line in detail["log"])
    assert json.loads((tmp_path / "studio" / "jobs.json").read_text(encoding="utf-8"))[0]["status"] == "done"


def test_cancel_kills_the_process_and_queue_runs_one_batch_at_a_time(tmp_path) -> None:
    manager = JobManager(tmp_path / "studio", status_file=tmp_path / "batch_status.json")
    first = fake_job(manager, tmp_path, "2", "20")
    second = fake_job(manager, tmp_path, "1", "0.1")
    second.created = first.created + 1
    manager.tick()
    assert first.status == "running" and second.status == "queued"
    assert manager.summary(second)["wait_s"] > 0  # attend la fin du premier
    manager.cancel(first.id)
    assert first.status == "cancelled"
    wait_for(manager, second, ("done", "failed"))
    assert second.status == "done"
    from src.studio.jobs import _pid_alive

    assert not _pid_alive(first.pid)


def test_server_restart_reattaches_to_a_running_process(tmp_path) -> None:
    manager = JobManager(tmp_path / "studio", status_file=tmp_path / "batch_status.json")
    job = fake_job(manager, tmp_path, "1", "1.5")
    manager.tick()
    manager._save()
    reborn = JobManager(tmp_path / "studio", status_file=tmp_path / "batch_status.json")  # « redémarrage » du serveur
    again = reborn.get(job.id)
    assert again.status == "running"
    wait_for(reborn, again, ("done", "failed", "interrupted"))
    assert again.status == "done"  # issue lue dans la progression, sans code de sortie
    manager.procs[job.id].wait(timeout=10)


def test_batch_command_is_built_and_validated(tmp_path) -> None:
    manager = JobManager(tmp_path / "studio")
    cmd, title, subtitle = manager.build_batch({
        "url": "https://asurascans.com/comics/bad-born-blood-05c7df14", "start": 2, "end": 21, "compile": True,
        "voice": "am_fenrir", "speed": 1.3, "cta": False, "format": "long",
    })
    joined = " ".join(cmd)
    assert "--max-gemini-rpm 5" in joined and "--start-chapter 2 --end-chapter 21" in joined and "--compile" in cmd
    assert "--voice am_fenrir" in joined and "--speed 1.3" in joined and "--no-cta" in cmd and "--format LONG" in joined
    assert title == "Bad Born Blood" and subtitle.startswith("Chapitres 2 à 21")
    cmd, _, _ = manager.build_batch({"url": "https://www.webtoons.com/en/action/x/ep-1/viewer?title_no=1&episode_no=1", "speed": 1.2})
    assert "--speed" not in cmd  # vitesse par défaut : laissée au profil
    for bad in ({"url": "https://example.com/x"}, {"url": "https://asurascans.com/comics/x", "start": 5, "end": 2},
                {"url": "https://asurascans.com/comics/x", "voice": "rm -rf"}, {"url": "https://asurascans.com/comics/x", "redo": "all"}):
        with pytest.raises(JobError):
            manager.build_batch(bad)
    assert pretty_series("https://www.webtoons.com/en/action/im-the-max-level-newbie/list?title_no=3915") == "Im The Max Level Newbie"


# --- Bibliothèque ---------------------------------------------------------------------------
def make_output(root: Path) -> Path:
    out = root / "output"
    chapter = out / "serie-test_ep1"
    (chapter / "figures").mkdir(parents=True)
    (chapter / "kdenlive").mkdir()
    (chapter / "chapter.json").write_text(json.dumps({"url": "https://www.webtoons.com/en/a/serie-test/ep-1/viewer?title_no=1&episode_no=1",
                                                      "series_title": "Serie Test", "episode_title": "Chapter 1", "episode_no": 1}), encoding="utf-8")
    from PIL import Image

    Image.new("RGB", (400, 600), (200, 40, 40)).save(chapter / "figures" / "panel_000.png")
    (chapter / "timeline.json").write_text(json.dumps({
        "width": 1920, "height": 1080, "series_title": "Serie Test", "episode_title": "Chapter 1", "total_duration_s": 20.0,
        "panels_dir": str(chapter / "figures"),
        "clips": [{"scene_index": 0, "file": "panel_000.png", "emotion": "action"}],
        "audio": [{"scene_index": 0, "file": "audio/scene_000.wav", "start_s": 0.0, "duration_s": 20.0}],
        "subtitles": [{"start_s": 0.5, "end_s": 1.0, "text": "Hello"}]}), encoding="utf-8")
    (chapter / "scenes.json").write_text(json.dumps({"model": "claude", "n_panels": 1,
                                                     "scenes": [{"index": 0, "narration": "Hello there.", "emotion": "action"}]}), encoding="utf-8")
    playable = b"\x00\x00\x00\x18ftypisom" + b"\x00" * 16 + b"\x00\x00\x00\x08moov" + b"\x00" * 32
    (chapter / "preview_full.mp4").write_bytes(playable)
    (chapter / "kdenlive" / "melt_h264_amf_30fps.mp4").write_bytes(playable)
    (chapter / "kdenlive" / "melt_h264_amf_60fps.mp4").write_bytes(b"\x00\x00\x00\x18ftypisom" + b"\x00" * 64)  # rendu coupé
    (chapter / "kdenlive" / "melt_h264_amf_60fps.part.mp4").write_bytes(b"\x00" * 64)  # rendu en cours
    failed = out / "serie-test_ep2"
    failed.mkdir()
    (failed / "chapter.json").write_text(json.dumps({"series_title": "Serie Test", "episode_title": "Chapter 2", "episode_no": 2}), encoding="utf-8")
    (out / "bench").mkdir()  # dossier d'essai : ignoré
    (root / "batch_status.json").write_text(json.dumps({"chapters": {"https://x/2": {"status": "failed", "error": "QuotaExhaustedError: x",
                                                                                     "out_dir": str(failed)}}}), encoding="utf-8")
    return out


def test_library_lists_videos_and_guards_paths(tmp_path) -> None:
    out = make_output(tmp_path)
    library = Library(out, tmp_path / "batch_status.json", tmp_path / "cache")
    items = {item["name"]: item for item in library.list()}
    assert set(items) == {"serie-test_ep1", "serie-test_ep2"}
    ok, failed = items["serie-test_ep1"], items["serie-test_ep2"]
    assert ok["status"] == "final" and ok["duration_s"] == 20.0 and ok["media"][0]["label"] == "Vidéo finale · 30 i/s"
    assert [m["path"] for m in ok["media"]] == ["kdenlive/melt_h264_amf_30fps.mp4", "preview_full.mp4"]
    assert [b["path"] for b in ok["broken"]] == ["kdenlive/melt_h264_amf_60fps.mp4"]  # illisible : jamais proposé
    assert failed["status"] == "failed" and "Quota" in failed["error"] and failed["url"] == "https://x/2"
    detail = library.detail("serie-test_ep1")
    scene = detail["scenes"][0]
    assert scene["narration"] == "Hello there." and scene["emotion"] == "action" and scene["images"] == ["figures/panel_000.png"]
    assert library.image("serie-test_ep1", "figures/panel_000.png", 100).is_file()
    for bad in ("../batch_status.json", "..\\..\\x", "C:/Windows/win.ini"):
        with pytest.raises(KeyError):
            library.file("serie-test_ep1", bad)
    for name in ("..", "../output", "bench", "nope"):
        with pytest.raises(KeyError):
            library.get(name)


# --- API ----------------------------------------------------------------------------------------
def test_api_serves_videos_estimates_and_rejects_bad_requests(tmp_path) -> None:
    from fastapi.testclient import TestClient

    from src.studio.server import create_app

    out = make_output(tmp_path)
    app = create_app(data_dir=tmp_path / "studio", out_root=out, status_file=tmp_path / "batch_status.json", start_worker=False)
    client = TestClient(app)
    assert client.get("/").status_code == 200 and "importmap" in client.get("/").text
    assert client.get("/static/app.js").status_code == 200
    videos = client.get("/api/videos").json()
    assert {v["name"] for v in videos} == {"serie-test_ep1", "serie-test_ep2"}
    assert client.get("/api/videos/serie-test_ep1").json()["scenes"][0]["narration"] == "Hello there."
    assert client.get("/api/videos/serie-test_ep1/file", params={"path": "../../batch_status.json"}).status_code == 404
    assert client.get("/api/videos/nope").status_code == 404
    est = client.post("/api/estimate", json={"count": 3, "compile": True}).json()
    assert est["seconds"] > 0 and est["wait_s"] == 0
    assert client.post("/api/jobs", json={"url": "https://example.com/x"}).status_code == 400
    created = client.post("/api/jobs", json={"url": "https://asurascans.com/comics/serie-test-0a1b2c3d", "start": 1, "end": 2})
    assert created.status_code == 200 and created.json()["status"] == "queued" and created.json()["remaining_s"] > 0
    job_id = created.json()["id"]
    assert client.get(f"/api/jobs/{job_id}").json()["cmd"].startswith("src.main batch")
    assert client.post(f"/api/jobs/{job_id}/cancel").json()["status"] == "cancelled"
    assert client.delete(f"/api/jobs/{job_id}").json() == {"ok": True}
    redo = client.post("/api/videos/serie-test_ep1/redo", json={"stage": "tts"}).json()
    assert redo["status"] == "queued" and redo["target"] == "serie-test_ep1" and redo["subtitle"] == "Chapitre 1"
    assert "--redo tts" in client.get(f"/api/jobs/{redo['id']}").json()["cmd"]
    client.post(f"/api/jobs/{redo['id']}/cancel")
    render = client.post("/api/videos/serie-test_ep1/render", json={"fps": 24})
    assert render.status_code == 400
