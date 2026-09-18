"""Orchestrateur de lots : plusieurs chapitres en parallèle, charge réseau et charge locale isolées.

Chaque chapitre traverse les étapes du pipeline (:mod:`src.pipeline`) sous trois
sémaphores ``asyncio`` :

- ``max_chapters`` (défaut 5) : chapitres **analysés** simultanément par Gemini ;
- ``max_scrape_workers`` (défaut : ``max_chapters``) : téléchargements + découpes
  simultanés ; séparé de l'analyse pour que le scraping prenne de l'avance pendant
  que le quota Gemini s'écoule ;
- ``max_gemini_rpm`` (défaut 10) : requêtes Gemini par minute, toutes clés et tous
  chapitres confondus (limiteur global du :class:`~src.utils.gemini_manager.GeminiManager`) ;
- ``max_tts_workers`` (défaut 2, optimum mesuré) : synthèses Kokoro simultanées (CPU) ;
- ``max_render_workers`` (défaut 1) : montages + rendus ffmpeg simultanés (CPU).

Les étapes sont synchrones : elles tournent dans des threads (``asyncio.to_thread``)
et ``asyncio.gather`` orchestre l'ensemble. Le délai forcé entre lots d'images de
l'analyzer (:data:`src.modules.analyzer.BATCH_DELAY_S`) s'applique dans le thread du
chapitre sans bloquer les autres.

Suivi : ``batch_status.json`` (racine du projet par défaut) enregistre pour chaque
URL l'état ``pending`` / ``processing`` / ``done`` / ``failed``, l'étape courante, le
dossier de sortie, le modèle utilisé et l'erreur éventuelle. Une exécution
suivante **saute les chapitres déjà ``done``** et reprend les autres (les étapes
déjà calculées sur disque sont réutilisées par le pipeline).
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import statistics
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.modules.scraper import discover_episodes, episode_url, parse_ids_from_url
from src.pipeline import (
    PipelineOptions,
    PipelineResult,
    slug_from_url,
    stage_analyze,
    stage_montage,
    stage_scrape_slice,
    stage_tts,
)
from src.utils.config import PROJECT_ROOT
from src.utils.gemini_manager import GeminiManager, QuotaExhaustedError

logger = logging.getLogger(__name__)

STATUS_PENDING = "pending"
STATUS_PROCESSING = "processing"
STATUS_DONE = "done"
STATUS_FAILED = "failed"
STATUSES: tuple[str, ...] = (STATUS_PENDING, STATUS_PROCESSING, STATUS_DONE, STATUS_FAILED)
DEFAULT_STATUS_FILE: Path = PROJECT_ROOT / "batch_status.json"
DEFAULT_OUT_ROOT: Path = PROJECT_ROOT / "output"


class BatchError(RuntimeError):
    """Erreur de configuration du lot (aucune URL, plage invalide...)."""


@dataclass
class BatchOptions:
    """Réglages de l'orchestrateur."""

    max_chapters: int = 5
    max_gemini_rpm: int = 10
    #: Synthèses Kokoro simultanées. Mesuré sur 16 cœurs (torch utilise déjà 8 threads par
    #: flux) : 1 worker = 3,2x temps réel, **2 workers = 4,7x (optimum)**, 4 workers = 4,4x
    #: (la sur-souscription des cœurs fait régresser). Au-delà de 2, c'est du gâchis.
    max_tts_workers: int = 2
    max_render_workers: int = 1
    #: Téléchargements + découpes simultanés (réseau pur) ; ``None`` = ``max_chapters``.
    #: Séparé de l'analyse pour que le scraping prenne de l'avance pendant les appels Gemini.
    max_scrape_workers: int | None = None
    status_file: Path = DEFAULT_STATUS_FILE
    out_root: Path = DEFAULT_OUT_ROOT
    retry_failed: bool = True

    def __post_init__(self) -> None:
        if self.max_scrape_workers is None:
            self.max_scrape_workers = self.max_chapters
        for name in ("max_chapters", "max_gemini_rpm", "max_tts_workers", "max_render_workers", "max_scrape_workers"):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} doit etre >= 1")
        self.status_file = Path(self.status_file)
        self.out_root = Path(self.out_root)


@dataclass
class Stages:
    """Étapes injectables (les tests remplacent les vraies étapes par des fonctions factices)."""

    scrape: Callable[..., Any] = stage_scrape_slice
    analyze: Callable[..., Any] = stage_analyze
    tts: Callable[..., Any] = stage_tts
    montage: Callable[..., Any] = stage_montage


def _now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


class BatchStatus:
    """Fichier de suivi ``batch_status.json`` : un enregistrement par URL, sauvegardé à chaque changement."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.data: dict[str, Any] = {"created": _now(), "updated": None, "chapters": {}}
        if self.path.is_file():
            try:
                loaded = json.loads(self.path.read_text(encoding="utf-8"))
            except (OSError, ValueError) as exc:
                logger.warning("Fichier de suivi %s illisible (%s) : reprise a zero", self.path, exc)
                loaded = None
            if isinstance(loaded, dict) and isinstance(loaded.get("chapters"), dict):
                self.data = loaded

    @property
    def chapters(self) -> dict[str, dict[str, Any]]:
        return self.data["chapters"]

    def get(self, url: str) -> dict[str, Any]:
        return self.chapters.setdefault(url, {"status": STATUS_PENDING, "attempts": 0})

    def update(self, url: str, status: str | None = None, **fields: Any) -> dict[str, Any]:
        entry = self.get(url)
        if status is not None:
            if status not in STATUSES:
                raise ValueError(f"statut inconnu : {status}")
            entry["status"] = status
        entry.update(fields)
        entry["updated"] = _now()
        self.save()
        return entry

    def counts(self, urls: Sequence[str] | None = None) -> dict[str, int]:
        """Nombre de chapitres par statut (tout le fichier, ou seulement ``urls``)."""
        selected = self.chapters if urls is None else {u: self.chapters[u] for u in urls if u in self.chapters}
        result = {status: 0 for status in STATUSES}
        for entry in selected.values():
            status = entry.get("status", STATUS_PENDING)
            result[status] = result.get(status, 0) + 1
        return result

    def should_process(self, url: str, *, retry_failed: bool = True, force: bool = False) -> bool:
        """Faux pour un chapitre déjà ``done`` (sauf ``force``) ou ``failed`` sans ``retry_failed``."""
        status = self.get(url).get("status", STATUS_PENDING)
        if force:
            return True
        if status == STATUS_DONE:
            return False
        if status == STATUS_FAILED and not retry_failed:
            return False
        return True

    def save(self) -> None:
        self.data["updated"] = _now()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        tmp.write_text(json.dumps(self.data, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(tmp, self.path)


# --- Sélection des chapitres --------------------------------------------------------------
def read_url_list(path: str | Path) -> list[str]:
    """URLs d'un fichier texte (une par ligne, ``#`` commentaires, doublons retirés)."""
    urls: list[str] = []
    for raw in Path(path).read_text(encoding="utf-8-sig").splitlines():
        line = raw.strip()
        if line and not line.startswith("#"):
            urls.append(line)
    return list(dict.fromkeys(urls))


def resolve_chapter_urls(
    url: str | None,
    *,
    start_chapter: int | None = None,
    end_chapter: int | None = None,
    url_list: str | Path | None = None,
    discover: Callable[[str], dict[int, str]] | None = discover_episodes,
) -> list[str]:
    """URLs des chapitres à traiter.

    - ``url_list`` : fichier d'URLs ;
    - ``url`` + plage ``start_chapter``..``end_chapter`` : les épisodes sont découverts sur
      la page de liste de la série (URL canoniques) ; à défaut, l'URL est dérivée de
      ``episode_no`` (Webtoons redirige vers le slug canonique) ;
    - ``url`` seule : une URL de liste = toute la série, une URL de viewer = ce chapitre.
    """
    if url_list is not None:
        urls = read_url_list(url_list)
        if not urls:
            raise BatchError(f"Aucune URL dans {url_list}")
        return urls
    if not url:
        raise BatchError("Indiquer une URL de serie / d'episode ou --url-list")
    if start_chapter is not None and end_chapter is not None and end_chapter < start_chapter:
        raise BatchError("--end-chapter doit etre >= --start-chapter")
    is_list = urlparse_path_tail(url) == "list"
    _, episode_no = parse_ids_from_url(url)
    if start_chapter is None and end_chapter is None and not is_list:
        return [url]
    episodes: dict[int, str] = {}
    if discover is not None:
        try:
            episodes = discover(url)
        except Exception as exc:  # noqa: BLE001 - la decouverte est un confort
            logger.warning("Decouverte des episodes impossible (%s) : URLs derivees de episode_no", exc)
    if episodes:
        if start_chapter is None and end_chapter is None:
            return [episodes[n] for n in sorted(episodes)]  # toute la serie, telle que listee
        lo = start_chapter if start_chapter is not None else min(episodes)
        hi = end_chapter if end_chapter is not None else max(episodes)
        missing = [n for n in range(lo, hi + 1) if n not in episodes]
        if missing:
            logger.warning("Episodes absents de la liste Webtoons, URL derivee : %s", missing)
        return [episodes.get(n) or episode_url(url, n) for n in range(lo, hi + 1)]
    lo = start_chapter if start_chapter is not None else (episode_no or 1)
    hi = end_chapter if end_chapter is not None else (episode_no or lo)
    return [episode_url(url, n) for n in range(lo, hi + 1)]


def urlparse_path_tail(url: str) -> str:
    from urllib.parse import urlparse

    segments = [s for s in urlparse(url).path.split("/") if s]
    return segments[-1] if segments else ""


# --- Orchestrateur -----------------------------------------------------------------------------
@dataclass
class BatchReport:
    """Résumé d'une exécution."""

    status: BatchStatus
    results: dict[str, PipelineResult] = field(default_factory=dict)
    errors: dict[str, str] = field(default_factory=dict)
    skipped: list[str] = field(default_factory=list)
    gemini: dict[str, Any] = field(default_factory=dict)
    urls: list[str] = field(default_factory=list)
    timings: dict[str, Any] = field(default_factory=dict)
    elapsed_s: float = 0.0


#: Un échantillon de mesure : ``(durées par étape, durée de la vidéo montée, étapes réutilisées)``.
TimingSample = tuple[Mapping[str, float], float, Sequence[str]]


def timing_summary(samples: Iterable[TimingSample]) -> dict[str, Any]:
    """Durées par étape et par chapitre (médianes) + débit, sur les chapitres réellement produits.

    Les chapitres dont toutes les étapes ont été réutilisées sont exclus des médianes
    (leur temps ne mesure rien) mais restent comptés dans ``n_reused``.
    """
    steps: dict[str, list[float]] = {}
    chapter_totals: list[float] = []
    video_seconds: list[float] = []
    render_ratio: list[float] = []
    n_reused = 0
    n_chapters = 0
    for timings, video_s, reused in samples:
        n_chapters += 1
        computed = {step: s for step, s in timings.items() if s > 0.05}
        if not computed or len(reused) >= 3:
            n_reused += 1
            continue
        for step, seconds in computed.items():
            steps.setdefault(step, []).append(seconds)
        chapter_totals.append(sum(timings.values()))
        if video_s > 0:
            video_seconds.append(video_s)
            render_ratio.append(sum(timings.values()) / video_s)
    return {
        "n_chapters": n_chapters,
        "n_measured": len(chapter_totals),
        "n_reused": n_reused,
        "steps": {
            step: {
                "median_s": round(statistics.median(values), 1),
                "total_s": round(sum(values), 1),
                "share_pct": 0.0,
                "n": len(values),
            }
            for step, values in steps.items()
        },
        "chapter_median_s": round(statistics.median(chapter_totals), 1) if chapter_totals else 0.0,
        "chapter_min_s": round(min(chapter_totals), 1) if chapter_totals else 0.0,
        "chapter_max_s": round(max(chapter_totals), 1) if chapter_totals else 0.0,
        "video_median_s": round(statistics.median(video_seconds), 1) if video_seconds else 0.0,
        "compute_per_video_second": round(statistics.median(render_ratio), 2) if render_ratio else 0.0,
    }


async def process_batch(
    urls: Sequence[str],
    options: PipelineOptions | None = None,
    batch: BatchOptions | None = None,
    *,
    manager: GeminiManager | None = None,
    stages: Stages | None = None,
) -> BatchReport:
    """Traite une liste de chapitres en parallèle sous sémaphores et met à jour ``batch_status.json``.

    Args:
        urls: URLs des chapitres (une par chapitre).
        options: réglages du pipeline (communs à tous les chapitres).
        batch: réglages de parallélisme et fichier de suivi.
        manager: gestionnaire Gemini partagé ; construit ici sinon (clés de l'environnement),
            avec ``max_rpm = batch.max_gemini_rpm``.
        stages: étapes injectables (tests).
    """
    options = options or PipelineOptions()
    batch = batch or BatchOptions()
    stages = stages or Stages()
    status = BatchStatus(batch.status_file)
    report = BatchReport(status=status)
    started = time.perf_counter()
    urls = list(dict.fromkeys(urls))
    if not urls:
        raise BatchError("Aucun chapitre a traiter")
    if manager is None:
        manager = GeminiManager(preferred_model=options.model, max_rpm=batch.max_gemini_rpm)

    sem_scrape = asyncio.Semaphore(batch.max_scrape_workers or batch.max_chapters)
    sem_chapters = asyncio.Semaphore(batch.max_chapters)
    sem_tts = asyncio.Semaphore(batch.max_tts_workers)
    sem_render = asyncio.Semaphore(batch.max_render_workers)
    quota_hit = asyncio.Event()

    async def one(url: str) -> None:
        out_dir = batch.out_root / slug_from_url(url)
        entry = status.get(url)
        if not status.should_process(url, retry_failed=batch.retry_failed, force=options.force or bool(options.redo)):
            report.skipped.append(url)
            logger.info("Chapitre deja %s, ignore : %s", entry["status"], url)
            return
        result = PipelineResult(out_dir=out_dir)
        status.update(
            url, STATUS_PROCESSING, stage="scrape", out_dir=str(out_dir), started=_now(), finished=None, error=None,
            attempts=int(entry.get("attempts", 0)) + 1,
        )
        t0 = time.perf_counter()
        try:
            if quota_hit.is_set():
                raise QuotaExhaustedError("quota Gemini epuise sur toutes les cles et tous les modeles (chapitre precedent)")
            # Le scraping (reseau) avance pendant que d'autres chapitres consomment le quota Gemini.
            async with sem_scrape:
                meta = await asyncio.to_thread(stages.scrape, url, out_dir, options, result)
            status.update(
                url, stage="analyze", episode_no=meta.episode_no, title=f"{meta.series_title} - {meta.episode_title}".strip(" -"),
                n_panels=result.n_panels,
            )
            async with sem_chapters:
                if quota_hit.is_set():
                    raise QuotaExhaustedError("quota Gemini epuise sur toutes les cles et tous les modeles (chapitre precedent)")
                analysis = await asyncio.to_thread(stages.analyze, meta, out_dir, options, result, manager=manager)
            status.update(url, stage="tts", model=analysis.model, n_scenes=analysis.n_scenes)
            async with sem_tts:
                manifest = await asyncio.to_thread(stages.tts, analysis, out_dir, options, result)
            status.update(url, stage="montage", voice_s=round(manifest.total_duration_s, 1))
            async with sem_render:
                result = await asyncio.to_thread(stages.montage, analysis, manifest, meta, out_dir, options, result)
            report.results[url] = result
            status.update(
                url, STATUS_DONE, stage="done", finished=_now(), duration_s=round(time.perf_counter() - t0, 1),
                preview=str(result.preview_mp4) if result.preview_mp4 else None,
                capcut=str(result.capcut_copy or result.capcut_draft) if (result.capcut_copy or result.capcut_draft) else None,
                total_duration_s=round(result.total_duration_s, 1),
                timings={step: round(seconds, 1) for step, seconds in result.timings.items()},
                gemini_s=round(result.gemini_seconds, 1),
                reused=list(result.reused),
            )
            logger.info(
                "Chapitre termine : %s (%.0fs : %s)", url, time.perf_counter() - t0,
                ", ".join(f"{step} {seconds:.0f}s" for step, seconds in result.timings.items()),
            )
        except Exception as exc:  # noqa: BLE001 - un chapitre en echec n'arrete pas le lot
            message = f"{type(exc).__name__}: {exc}"[:600]
            report.errors[url] = message
            if isinstance(exc, QuotaExhaustedError) or "QuotaExhausted" in type(exc).__name__:
                quota_hit.set()
            status.update(
                url, STATUS_FAILED, finished=_now(), error=message, duration_s=round(time.perf_counter() - t0, 1),
                timings={step: round(seconds, 1) for step, seconds in result.timings.items()},
                gemini_s=round(result.gemini_seconds, 1),
            )
            logger.error("Chapitre en echec : %s -> %s", url, message)

    for url in urls:
        status.get(url)
    status.save()
    await asyncio.gather(*(one(url) for url in urls))
    report.urls = list(urls)
    report.gemini = manager.status()
    report.timings = timing_summary(
        (r.timings, r.total_duration_s, r.reused) for r in report.results.values()
    )
    report.elapsed_s = time.perf_counter() - started
    status.data["gemini"] = report.gemini
    status.data["run"] = {
        "finished": _now(),
        "elapsed_s": round(report.elapsed_s, 1),
        "chapters": len(urls),
        "done": len(report.results),
        "failed": len(report.errors),
        "skipped": len(report.skipped),
        "timings": report.timings,
        "parallelism": {
            "max_chapters": batch.max_chapters, "max_gemini_rpm": batch.max_gemini_rpm,
            "max_tts_workers": batch.max_tts_workers, "max_render_workers": batch.max_render_workers,
            "max_scrape_workers": batch.max_scrape_workers,
        },
    }
    status.save()
    return report


def run_batch(
    urls: Sequence[str],
    options: PipelineOptions | None = None,
    batch: BatchOptions | None = None,
    *,
    manager: GeminiManager | None = None,
) -> BatchReport:
    """Version synchrone de :func:`process_batch` (``asyncio.run``)."""
    return asyncio.run(process_batch(urls, options, batch, manager=manager))


def status_timings(status: BatchStatus, urls: Sequence[str] | None = None) -> dict[str, Any]:
    """Agrège les durées enregistrées dans un fichier de suivi (chapitres ``done`` seulement)."""
    entries = status.chapters if urls is None else {u: status.chapters[u] for u in urls if u in status.chapters}
    return timing_summary(
        (entry.get("timings") or {}, float(entry.get("total_duration_s") or 0.0), entry.get("reused") or [])
        for entry in entries.values()
        if entry.get("status") == STATUS_DONE
    )


def format_status(status: BatchStatus, urls: Sequence[str] | None = None) -> str:
    """Résumé ASCII d'un fichier de suivi : statuts, durées par étape, dernier lot exécuté."""
    counts = status.counts(urls)
    timings = status_timings(status, urls)
    lines = [
        "=" * 72,
        f"Suivi      : {status.path}",
        "Chapitres  : " + ", ".join(f"{k} {v}" for k, v in counts.items()),
    ]
    steps = timings.get("steps") or {}
    if steps:
        order = ["scrape+slice", "analyze", "tts", "timeline", "capcut", "preview"]
        ordered = [s for s in order if s in steps] + [s for s in steps if s not in order]
        lines.append("Par etape  : " + ", ".join(f"{s} {steps[s]['median_s']:.0f}s" for s in ordered) + "  (mediane)")
        lines.append(
            f"Par chapitre: {timings['chapter_median_s']:.0f}s de traitement (min {timings['chapter_min_s']:.0f}s, "
            f"max {timings['chapter_max_s']:.0f}s) pour {timings['video_median_s']:.0f}s de video montee "
            f"-> {timings['compute_per_video_second']:.2f}s de calcul par seconde de video "
            f"({timings['n_measured']} chapitre(s) mesure(s))"
        )
    run = status.data.get("run") or {}
    if run:
        lines.append(
            f"Dernier lot: {run.get('done', 0)}/{run.get('chapters', 0)} termine(s) en {run.get('elapsed_s', 0):.0f}s "
            f"le {run.get('finished', '?')} (parallelisme {run.get('parallelism', {})})"
        )
    gemini = status.data.get("gemini") or {}
    latency = gemini.get("latency_s") or {}
    if latency.get("n"):
        lines.append(
            f"Gemini     : {gemini.get('calls', 0)} appel(s), reponse mediane {latency['p50']:.1f}s, "
            f"p95 {latency['p95']:.1f}s, max {latency['max']:.1f}s"
        )
    selected = status.chapters if urls is None else {u: status.chapters[u] for u in urls if u in status.chapters}
    for entry in sorted(selected.values(), key=lambda e: (str(e.get("out_dir") or ""), e.get("episode_no") or 0)):
        mark = {STATUS_DONE: "OK     ", STATUS_FAILED: "ECHEC  "}.get(entry.get("status", ""), "ATTENTE")
        name = Path(str(entry.get("out_dir") or "?")).name
        detail = f"{entry.get('duration_s', 0):.0f}s" if entry.get("duration_s") else ""
        if entry.get("total_duration_s"):
            detail += f" -> {entry['total_duration_s']:.0f}s de video"
        if entry.get("gemini_s"):
            detail += f" (dont {entry['gemini_s']:.0f}s Gemini)"
        lines.append(f"  {mark} {name:<38} {detail} {(entry.get('error') or '')[:60]}".rstrip())
    lines.append("=" * 72)
    return "\n".join(lines).encode("ascii", "replace").decode("ascii")


def format_report(report: BatchReport) -> str:
    """Résumé ASCII d'une exécution de lot : résultats, temps par étape, appels Gemini."""
    counts = report.status.counts(report.urls or None)
    minutes = report.elapsed_s / 60
    lines = [
        "=" * 72,
        f"Chapitres  : {len(report.results)} termine(s), {len(report.errors)} en echec, {len(report.skipped)} ignore(s) "
        f"(deja faits) en {report.elapsed_s:.0f}s ({minutes:.1f} min)",
        f"Suivi      : {report.status.path}  ->  " + ", ".join(f"{k} {v}" for k, v in counts.items()),
    ]
    t = report.timings
    if t and t.get("n_measured"):
        steps = t["steps"]
        order = ["scrape+slice", "analyze", "tts", "timeline", "capcut", "preview"]
        ordered = [s for s in order if s in steps] + [s for s in steps if s not in order]
        lines.append("Par etape  : " + ", ".join(f"{s} {steps[s]['median_s']:.0f}s" for s in ordered) + "  (mediane par chapitre)")
        lines.append(
            f"Par chapitre: {t['chapter_median_s']:.0f}s de traitement (min {t['chapter_min_s']:.0f}s, "
            f"max {t['chapter_max_s']:.0f}s) pour {t['video_median_s']:.0f}s de video montee "
            f"-> {t['compute_per_video_second']:.2f}s de calcul par seconde de video"
        )
        gemini_s = sum(r.gemini_seconds for r in report.results.values())
        if gemini_s > 0:
            local_s = sum(s["total_s"] for s in steps.values()) - gemini_s
            lines.append(
                f"  dont     : {gemini_s:.0f}s d'attente Gemini (appels, reessais, quota) et {local_s:.0f}s "
                f"de calcul local (scraping, decoupe, voix, montage, rendu)"
            )
        if len(report.results) > 1:
            lines.append(
                f"Debit      : {report.elapsed_s / max(1, len(report.results)):.0f}s par chapitre en parallele "
                f"(somme des traitements {sum(s['total_s'] for s in steps.values()):.0f}s)"
            )
    if report.gemini:
        g = report.gemini
        latency = g.get("latency_s", {})
        lines.append(
            f"Gemini     : {g.get('calls', 0)} appel(s), modele {g.get('model')}, cle active {g.get('active_key')}, "
            f"{g.get('rotations', 0)} rotation(s), {g.get('cascades', 0)} cascade(s), attente RPM {g.get('rpm_wait_s', 0)}s"
        )
        if latency.get("n"):
            lines.append(
                f"  reponse  : mediane {latency['p50']:.1f}s, p95 {latency['p95']:.1f}s, max {latency['max']:.1f}s, "
                f"cumul {latency['total']:.0f}s"
            )
        for model, stats in g.get("calls_by_model", {}).items():
            lines.append(f"  {model:<22} {stats['ok']} ok / {stats['failed']} echec en {stats['seconds']:.0f}s")
    for url, result in report.results.items():
        total = sum(result.timings.values())
        lines.append(f"  OK      {url} -> {result.out_dir} ({result.total_duration_s:.0f}s de video, {total:.0f}s de traitement)")
    for url, error in report.errors.items():
        lines.append(f"  ECHEC   {url} -> {error[:120]}")
    for url in report.skipped:
        lines.append(f"  IGNORE  {url}")
    lines.append("=" * 72)
    return "\n".join(lines).encode("ascii", "replace").decode("ascii")


__all__ = [
    "STATUSES",
    "STATUS_PENDING",
    "STATUS_PROCESSING",
    "STATUS_DONE",
    "STATUS_FAILED",
    "DEFAULT_STATUS_FILE",
    "DEFAULT_OUT_ROOT",
    "BatchError",
    "BatchOptions",
    "Stages",
    "BatchStatus",
    "BatchReport",
    "read_url_list",
    "resolve_chapter_urls",
    "timing_summary",
    "status_timings",
    "format_status",
    "process_batch",
    "run_batch",
    "format_report",
]
