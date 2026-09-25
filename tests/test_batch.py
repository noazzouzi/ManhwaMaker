"""Tests de l'orchestrateur de lots (``src.modules.batch_processor``) : sélection, sémaphores, suivi, reprise."""

from __future__ import annotations

import asyncio
import json
import threading
import time
from pathlib import Path

import pytest

from src.models.audio import VoiceoverManifest
from src.models.chapter import ChapterMeta
from src.models.scene import ChapterAnalysis, Scene
from src.modules import batch_processor as bp
from src.modules.batch_processor import (
    STATUS_DONE,
    STATUS_FAILED,
    STATUS_PENDING,
    BatchError,
    BatchOptions,
    BatchStatus,
    Stages,
    format_report,
    process_batch,
    read_url_list,
    resolve_chapter_urls,
)
from src.pipeline import PipelineOptions, PipelineResult
from src.utils.gemini_manager import GeminiManager, QuotaExhaustedError

SERIES = "https://www.webtoons.com/en/action/im-the-max-level-newbie/list?title_no=3915"
VIEWER = "https://www.webtoons.com/en/action/im-the-max-level-newbie/ep-0/viewer?title_no=3915&episode_no=1"


def url_for(n: int) -> str:
    return f"https://www.webtoons.com/en/action/serie/ep-{n}/viewer?title_no=1&episode_no={n}"


# --- Selection des chapitres ------------------------------------------------------------
def test_read_url_list_and_resolve_from_list(tmp_path) -> None:
    listing = tmp_path / "urls.txt"
    listing.write_text("﻿# commentaire\n" + url_for(1) + "\n\n" + url_for(2) + "\n" + url_for(1) + "\n", encoding="utf-8")
    assert read_url_list(listing) == [url_for(1), url_for(2)]
    assert resolve_chapter_urls(None, url_list=listing) == [url_for(1), url_for(2)]
    (tmp_path / "empty.txt").write_text("# rien\n", encoding="utf-8")
    with pytest.raises(BatchError):
        resolve_chapter_urls(None, url_list=tmp_path / "empty.txt")
    with pytest.raises(BatchError):
        resolve_chapter_urls(None)


def test_resolve_range_uses_discovery_then_falls_back_to_templates() -> None:
    discovered = {1: url_for(1), 2: url_for(2), 4: url_for(4)}
    urls = resolve_chapter_urls(SERIES, start_chapter=1, end_chapter=4, discover=lambda u: discovered)
    assert urls[:2] == [url_for(1), url_for(2)] and urls[3] == url_for(4)
    assert "episode_no=3" in urls[2] and urls[2].endswith("/viewer?title_no=3915&episode_no=3")  # derive
    # Serie entiere sans plage : tous les episodes decouverts.
    assert resolve_chapter_urls(SERIES, discover=lambda u: discovered) == [url_for(1), url_for(2), url_for(4)]
    # Decouverte en echec : URLs derivees de episode_no.
    def boom(u):
        raise RuntimeError("offline")

    urls = resolve_chapter_urls(VIEWER, start_chapter=2, end_chapter=3, discover=boom)
    assert [u.rsplit("=", 1)[1] for u in urls] == ["2", "3"] and all("/viewer?" in u for u in urls)
    assert resolve_chapter_urls(VIEWER, discover=boom) == [VIEWER]  # un seul viewer, pas de plage
    assert len(resolve_chapter_urls(SERIES, start_chapter=5, end_chapter=7, discover=None)) == 3
    with pytest.raises(BatchError):
        resolve_chapter_urls(SERIES, start_chapter=5, end_chapter=2, discover=None)


# --- Suivi ---------------------------------------------------------------------------------
def test_batch_status_persists_and_decides_what_to_process(tmp_path) -> None:
    path = tmp_path / "batch_status.json"
    status = BatchStatus(path)
    status.update(url_for(1), STATUS_DONE, out_dir="x")
    status.update(url_for(2), STATUS_FAILED, error="boom")
    status.get(url_for(3))
    status.save()
    reloaded = BatchStatus(path)
    assert reloaded.chapters[url_for(1)]["status"] == STATUS_DONE and reloaded.chapters[url_for(2)]["error"] == "boom"
    assert reloaded.chapters[url_for(3)]["status"] == STATUS_PENDING
    assert reloaded.counts() == {"pending": 1, "processing": 0, "done": 1, "failed": 1}
    assert not reloaded.should_process(url_for(1)) and reloaded.should_process(url_for(1), force=True)
    assert reloaded.should_process(url_for(2)) and not reloaded.should_process(url_for(2), retry_failed=False)
    assert reloaded.should_process(url_for(3))
    with pytest.raises(ValueError):
        status.update(url_for(1), "weird")
    path.write_text("{not json", encoding="utf-8")
    assert BatchStatus(path).chapters == {}  # fichier corrompu : reprise a zero, sans exception
    with pytest.raises(ValueError):
        BatchOptions(max_chapters=0)


# --- Orchestration avec etapes factices ------------------------------------------------------
class Tracker:
    """Compte les etapes en cours pour verifier les semaphores."""

    def __init__(self):
        self.lock = threading.Lock()
        self.active: dict[str, int] = {}
        self.peak: dict[str, int] = {}
        self.calls: list[tuple[str, str]] = []

    def enter(self, stage: str, url: str, dwell: float = 0.05):
        with self.lock:
            self.active[stage] = self.active.get(stage, 0) + 1
            self.peak[stage] = max(self.peak.get(stage, 0), self.active[stage])
            self.calls.append((stage, url))
        time.sleep(dwell)
        with self.lock:
            self.active[stage] -= 1


def fake_stages(tracker: Tracker, *, fail_analyze: set[str] = frozenset(), quota_on: set[str] = frozenset()) -> Stages:
    def scrape(url, out_dir, options, result=None):
        tracker.enter("scrape", url)
        result.n_panels = 10
        result.timings["scrape+slice"] = 20.0
        return ChapterMeta(url=url, final_url=url, series_title="S", episode_title=f"Ep {url[-1]}", title_no=1,
                           episode_no=int(url.rsplit("=", 1)[1]), image_urls=["a"])

    def analyze(meta, out_dir, options, result=None, *, manager=None):
        tracker.enter("analyze", meta.url)
        if meta.url in quota_on:
            raise QuotaExhaustedError("toutes les cles sont epuisees")
        if meta.url in fail_analyze:
            raise RuntimeError("gemini down")
        assert manager is not None
        result.timings["analyze"] = 100.0
        return ChapterAnalysis(model=manager.current_model, language="en", n_panels=10,
                               scenes=[Scene(index=0, panel_ids=[0], narration="Hi.", emotion="calm")])

    def tts(analysis, out_dir, options, result=None):
        tracker.enter("tts", str(out_dir), dwell=0.08)
        result.timings["tts"] = 50.0
        return VoiceoverManifest(language="en", lang_code="a", voice="v", speed=1.0, padding_s=0.18, sample_rate=24000,
                                 items=[], total_duration_s=12.5)

    def montage(analysis, manifest, meta, out_dir, options, result=None):
        tracker.enter("montage", str(out_dir))
        result.timings["preview"] = 30.0
        result.total_duration_s = 400.0
        result.preview_mp4 = Path(out_dir) / "preview_60s.mp4"
        result.capcut_draft = Path(out_dir) / "capcut" / "x"
        return result

    return Stages(scrape=scrape, analyze=analyze, tts=tts, montage=montage)


def make_manager() -> GeminiManager:
    class Client:
        class models:  # noqa: N801
            @staticmethod
            def generate_content(**kwargs):
                return {}

    return GeminiManager(["AIzaSyFAKEKEY000001"], models=["m1"], max_rpm=100, client_factory=lambda k: Client())


def test_process_batch_respects_semaphores_and_tracks_status(tmp_path) -> None:
    tracker = Tracker()
    urls = [url_for(n) for n in range(1, 7)]
    batch = BatchOptions(max_chapters=2, max_tts_workers=1, max_render_workers=3, status_file=tmp_path / "status.json", out_root=tmp_path / "out")
    report = asyncio.run(process_batch(urls, PipelineOptions(), batch, manager=make_manager(), stages=fake_stages(tracker)))
    assert len(report.results) == 6 and not report.errors and report.skipped == []
    assert tracker.peak["scrape"] <= 2 and tracker.peak["analyze"] <= 2 and tracker.peak["tts"] == 1
    # Le scraping a son propre semaphore : il prend de l'avance sur l'analyse.
    wide = BatchOptions(max_chapters=1, max_scrape_workers=4, status_file=tmp_path / "wide.json", out_root=tmp_path / "out2")
    tracker_wide = Tracker()
    asyncio.run(process_batch(urls, PipelineOptions(), wide, manager=make_manager(), stages=fake_stages(tracker_wide)))
    assert tracker_wide.peak["scrape"] == 4 and tracker_wide.peak["analyze"] == 1
    assert tracker.peak["montage"] >= 1
    data = json.loads((tmp_path / "status.json").read_text(encoding="utf-8"))
    entry = data["chapters"][url_for(3)]
    assert entry["status"] == STATUS_DONE and entry["stage"] == "done" and entry["episode_no"] == 3
    assert entry["out_dir"].endswith("serie_ep3") and entry["model"] == "m1" and entry["preview"].endswith("preview_60s.mp4")
    assert entry["attempts"] == 1 and entry["error"] is None and entry["duration_s"] >= 0
    assert data["gemini"]["model"] == "m1" and "calls" in data["gemini"]
    # Durees par etape conservees par chapitre et agregees pour le lot.
    assert entry["timings"] == {"scrape+slice": 20.0, "analyze": 100.0, "tts": 50.0, "preview": 30.0}
    run = data["run"]
    assert run["chapters"] == 6 and run["done"] == 6 and run["elapsed_s"] > 0
    assert run["parallelism"]["max_chapters"] == 2 and run["parallelism"]["max_tts_workers"] == 1
    t = report.timings
    assert t["n_measured"] == 6 and t["n_reused"] == 0
    assert t["steps"]["analyze"]["median_s"] == 100.0 and t["steps"]["analyze"]["n"] == 6
    assert t["chapter_median_s"] == 200.0 and t["video_median_s"] == 400.0
    assert t["compute_per_video_second"] == pytest.approx(0.5)
    text = format_report(report)
    assert "6 termine(s), 0 en echec" in text and "OK" in text and "Gemini" in text
    assert "Par etape" in text and "analyze 100s" in text and "Par chapitre" in text
    assert "0.50s de calcul par seconde de video" in text


def test_process_batch_marks_failures_and_resumes_only_unfinished(tmp_path) -> None:
    tracker = Tracker()
    urls = [url_for(1), url_for(2), url_for(3)]
    batch = BatchOptions(status_file=tmp_path / "status.json", out_root=tmp_path / "out")
    report = asyncio.run(process_batch(urls, PipelineOptions(), batch, manager=make_manager(), stages=fake_stages(tracker, fail_analyze={url_for(2)})))
    assert set(report.results) == {url_for(1), url_for(3)} and list(report.errors) == [url_for(2)]
    assert "RuntimeError: gemini down" in report.errors[url_for(2)]
    status = BatchStatus(tmp_path / "status.json")
    assert status.chapters[url_for(2)]["status"] == STATUS_FAILED and status.chapters[url_for(2)]["stage"] == "analyze"

    # Reprise : les chapitres termines sont ignores, le chapitre en echec est retente (et reussit).
    tracker2 = Tracker()
    report2 = asyncio.run(process_batch(urls, PipelineOptions(), batch, manager=make_manager(), stages=fake_stages(tracker2)))
    assert sorted(report2.skipped) == [url_for(1), url_for(3)] and list(report2.results) == [url_for(2)]
    assert [u for s, u in tracker2.calls if s == "scrape"] == [url_for(2)]
    assert BatchStatus(tmp_path / "status.json").chapters[url_for(2)]["attempts"] == 2
    assert "2 ignore(s)" in format_report(report2)
    # Le decompte du rapport ne porte que sur les URL du lot courant, pas sur tout le fichier.
    other = BatchOptions(status_file=tmp_path / "status.json", out_root=tmp_path / "out")
    report_one = asyncio.run(process_batch([url_for(4)], PipelineOptions(), other, manager=make_manager(), stages=fake_stages(Tracker())))
    assert report_one.status.counts(report_one.urls) == {"pending": 0, "processing": 0, "done": 1, "failed": 0}
    assert report_one.status.counts()["done"] == 4  # le fichier contient les chapitres precedents

    # --no-retry-failed : un echec reste en l'etat ; --force retraite tout.
    tracker3 = Tracker()
    batch_no_retry = BatchOptions(status_file=tmp_path / "status2.json", out_root=tmp_path / "out")
    asyncio.run(process_batch([url_for(9)], PipelineOptions(), batch_no_retry, manager=make_manager(), stages=fake_stages(tracker3, fail_analyze={url_for(9)})))
    batch_no_retry.retry_failed = False
    report3 = asyncio.run(process_batch([url_for(9)], PipelineOptions(), batch_no_retry, manager=make_manager(), stages=fake_stages(Tracker())))
    assert report3.skipped == [url_for(9)]
    report4 = asyncio.run(process_batch(urls, PipelineOptions(force=True), batch, manager=make_manager(), stages=fake_stages(Tracker())))
    assert len(report4.results) == 3 and report4.skipped == []


def test_quota_exhaustion_short_circuits_remaining_chapters(tmp_path) -> None:
    tracker = Tracker()
    urls = [url_for(1), url_for(2), url_for(3)]
    batch = BatchOptions(max_chapters=1, status_file=tmp_path / "status.json", out_root=tmp_path / "out")
    gemini = PipelineOptions(script_ai="gemini")
    report = asyncio.run(process_batch(urls, gemini, batch, manager=make_manager(), stages=fake_stages(tracker, quota_on={url_for(1)})))
    assert len(report.errors) == 3 and not report.results
    assert all("QuotaExhausted" in e for e in report.errors.values())
    # Apres le premier epuisement, les chapitres suivants n'appellent plus Gemini (scrape seul).
    assert [u for s, u in tracker.calls if s == "analyze"] == [url_for(1)]
    with pytest.raises(BatchError):
        asyncio.run(process_batch([], gemini, batch, manager=make_manager(), stages=fake_stages(tracker)))


def test_gemini_quota_does_not_stop_a_claude_batch(tmp_path) -> None:
    """Claude ecrit les scripts : un quota Gemini epuise (repli d'un chapitre) n'arrete pas les autres."""
    tracker = Tracker()
    urls = [url_for(1), url_for(2), url_for(3)]
    batch = BatchOptions(max_chapters=1, status_file=tmp_path / "status.json", out_root=tmp_path / "out")
    report = asyncio.run(process_batch(urls, PipelineOptions(), batch, manager=make_manager(), stages=fake_stages(tracker, quota_on={url_for(1)})))
    assert list(report.errors) == [url_for(1)] and len(report.results) == 2


def test_redo_reprocesses_done_chapters_from_a_given_stage(tmp_path) -> None:
    from src.pipeline import STAGE_RANKS

    assert STAGE_RANKS == {"analyze": 1, "tts": 2, "montage": 3}
    # --redo tts : l'analyse est reutilisee (aucun quota Gemini), la voix et le montage refaits.
    options = PipelineOptions(redo="tts")
    assert not options.recompute("analyze") and options.recompute("tts") and options.recompute("montage")
    assert PipelineOptions(redo="analyze").recompute("analyze")
    assert not PipelineOptions().recompute("analyze") and not PipelineOptions().recompute("tts")
    assert PipelineOptions(force=True).recompute("analyze")  # --force recalcule tout
    assert PipelineOptions(redo="montage").recompute("montage") and not PipelineOptions(redo="montage").recompute("tts")

    # Un chapitre deja "done" est retraite quand --redo est demande.
    tracker = Tracker()
    batch = BatchOptions(status_file=tmp_path / "status.json", out_root=tmp_path / "out")
    asyncio.run(process_batch([url_for(1)], PipelineOptions(), batch, manager=make_manager(), stages=fake_stages(tracker)))
    again = Tracker()
    skipped = asyncio.run(process_batch([url_for(1)], PipelineOptions(), batch, manager=make_manager(), stages=fake_stages(again)))
    assert skipped.skipped == [url_for(1)] and not again.calls
    redone = Tracker()
    report = asyncio.run(process_batch([url_for(1)], options, batch, manager=make_manager(), stages=fake_stages(redone)))
    assert list(report.results) == [url_for(1)] and report.skipped == []
    assert [stage for stage, _ in redone.calls] == ["scrape", "analyze", "tts", "montage"]


def test_status_timings_and_format_status_read_the_ledger(tmp_path) -> None:
    """La commande ``stats`` relit les durees du fichier de suivi, sans relancer le pipeline."""
    from src.modules.batch_processor import format_status, status_timings

    tracker = Tracker()
    urls = [url_for(1), url_for(2)]
    batch = BatchOptions(status_file=tmp_path / "status.json", out_root=tmp_path / "out")
    asyncio.run(process_batch(urls, PipelineOptions(), batch, manager=make_manager(), stages=fake_stages(tracker)))
    status = BatchStatus(tmp_path / "status.json")
    summary = status_timings(status)
    assert summary["n_measured"] == 2 and summary["chapter_median_s"] == 200.0
    assert summary["steps"]["tts"]["median_s"] == 50.0 and summary["video_median_s"] == 400.0
    text = format_status(status)
    assert "Par etape" in text and "tts 50s" in text and "Dernier lot" in text
    # Chaque ligne nomme le dossier de sortie (deux series peuvent partager un numero d'episode).
    assert "OK      serie_ep1" in text and "OK      serie_ep2" in text
    assert "0.50s de calcul par seconde de video" in text
    # Un chapitre en echec n'entre pas dans les medianes mais reste visible.
    status.update(url_for(3), STATUS_FAILED, error="quota", episode_no=3, duration_s=12.0, out_dir=str(tmp_path / "out" / "serie_ep3"))
    assert status_timings(BatchStatus(tmp_path / "status.json"))["n_measured"] == 2
    assert "ECHEC   serie_ep3" in format_status(BatchStatus(tmp_path / "status.json"))


def test_run_batch_builds_manager_from_env(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("GEMINI_API_KEYS", "AIzaSyFAKEKEY000001, AIzaSyFAKEKEY000002")
    created: list[str] = []
    monkeypatch.setattr(GeminiManager, "_default_client", staticmethod(lambda key: created.append(key) or object()))
    tracker = Tracker()
    batch = BatchOptions(max_gemini_rpm=7, status_file=tmp_path / "s.json", out_root=tmp_path / "o")
    report = asyncio.run(process_batch([url_for(1)], PipelineOptions(model="gemini-3.7-flash"), batch, stages=fake_stages(tracker)))
    assert list(report.results) == [url_for(1)]
    assert report.gemini["models"][0] == "gemini-3.7-flash" and report.gemini["model"] == "gemini-3.7-flash"
    assert report.gemini["keys"][0]["label"].endswith("(...0001)") and len(report.gemini["keys"]) == 2


# --- Ordre de serie (memoire des personnages) -------------------------------------------------
def url_for_series(title_no: int, n: int) -> str:
    return f"https://www.webtoons.com/en/action/s{title_no}/ep-{n}/viewer?title_no={title_no}&episode_no={n}"


def analyzed_order(tracker: Tracker) -> list[str]:
    """Episodes analyses, dans l'ordre reel des appels."""
    return [url.rsplit("=", 1)[1] for stage, url in tracker.calls if stage == "analyze"]


def test_series_predecessors_chains_each_series_separately() -> None:
    urls = [url_for_series(1, 3), url_for_series(2, 5), url_for_series(1, 1), "https://x/sans-ids"]
    assert bp.series_predecessors(urls) == {url_for_series(1, 3): url_for_series(1, 1)}
    assert bp.series_predecessors([]) == {}


def test_series_order_analyses_episodes_in_ascending_order(tmp_path) -> None:
    """La memoire de serie ne marche que dans ce sens : c'est l'episode N-1 qui ecrit la
    fiche que le N va relire. Analyses dans le desordre, les noms ne se propagent pas."""
    tracker = Tracker()
    urls = [url_for(3), url_for(1), url_for(5), url_for(2), url_for(4)]
    batch = BatchOptions(max_chapters=3, status_file=tmp_path / "s.json", out_root=tmp_path / "o")
    report = asyncio.run(process_batch(urls, PipelineOptions(), batch, manager=make_manager(), stages=fake_stages(tracker)))
    assert len(report.results) == 5 and not report.errors
    assert analyzed_order(tracker) == ["1", "2", "3", "4", "5"]
    assert tracker.peak["analyze"] == 1        # une serie s'analyse desormais en file
    assert tracker.peak["scrape"] > 1          # mais le scraping garde son parallelisme


def test_two_different_series_never_wait_on_each_other(tmp_path) -> None:
    tracker = Tracker()
    urls = [url_for_series(1, 1), url_for_series(2, 1), url_for_series(1, 2), url_for_series(2, 2)]
    batch = BatchOptions(max_chapters=4, status_file=tmp_path / "s.json", out_root=tmp_path / "o")
    asyncio.run(process_batch(urls, PipelineOptions(), batch, manager=make_manager(), stages=fake_stages(tracker)))
    assert tracker.peak["analyze"] == 2        # deux series avancent en parallele


def test_series_order_off_keeps_the_historical_parallelism(tmp_path) -> None:
    tracker = Tracker()
    urls = [url_for(n) for n in range(1, 5)]
    batch = BatchOptions(max_chapters=3, series_order=False, status_file=tmp_path / "s.json", out_root=tmp_path / "o")
    asyncio.run(process_batch(urls, PipelineOptions(), batch, manager=make_manager(), stages=fake_stages(tracker)))
    assert tracker.peak["analyze"] == 3


def test_series_order_never_deadlocks_on_failed_or_skipped_chapters(tmp_path) -> None:
    """Le point d'interblocage : un chapitre qui n'analyse jamais doit quand meme liberer
    son successeur, sinon un seul echec fige toute la serie jusqu'au delai de garde."""
    status_file = tmp_path / "s.json"
    status = BatchStatus(status_file)
    status.update(url_for(2), STATUS_DONE)     # deja fait : sera ignore, sans analyse
    tracker = Tracker()
    urls = [url_for(n) for n in range(1, 6)]
    batch = BatchOptions(
        max_chapters=2, series_order_timeout_s=5.0, status_file=status_file, out_root=tmp_path / "o",
    )
    started = time.perf_counter()
    report = asyncio.run(process_batch(
        urls, PipelineOptions(), batch, manager=make_manager(),
        stages=fake_stages(tracker, fail_analyze={url_for(3)}),
    ))
    elapsed = time.perf_counter() - started

    assert report.skipped == [url_for(2)]
    assert list(report.errors) == [url_for(3)]
    # Le chapitre ignore et celui en echec n'ont bloque personne.
    assert analyzed_order(tracker) == ["1", "3", "4", "5"]
    assert elapsed < batch.series_order_timeout_s   # aucune attente arrivee a son terme


def test_a_predecessor_that_never_finishes_does_not_freeze_the_batch(tmp_path) -> None:
    """Delai de garde : passe ce point, le chapitre part sans memoire plutot que d'attendre.
    Le prealable est de ne jamais attendre en tenant le semaphore d'analyse."""
    tracker = Tracker()
    stages = fake_stages(tracker)
    slow_analyze = stages.analyze

    def analyze(meta, out_dir, options, result=None, *, manager=None):
        if meta.episode_no == 1:
            time.sleep(0.35)                   # bien au-dela du delai de garde ci-dessous
        return slow_analyze(meta, out_dir, options, result, manager=manager)

    batch = BatchOptions(
        max_chapters=2, series_order_timeout_s=0.1, status_file=tmp_path / "s.json", out_root=tmp_path / "o",
    )
    report = asyncio.run(process_batch(
        [url_for(1), url_for(2)], PipelineOptions(), batch, manager=make_manager(),
        stages=Stages(scrape=stages.scrape, analyze=analyze, tts=stages.tts, montage=stages.montage),
    ))
    assert len(report.results) == 2 and not report.errors
    assert analyzed_order(tracker) == ["2", "1"]   # le 2 a renonce a attendre
