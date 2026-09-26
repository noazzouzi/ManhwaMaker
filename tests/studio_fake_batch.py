"""Faux lot pour les tests du Studio : le vrai orchestrateur, des étapes factices qui rapportent leur progression.

Lancé comme un processus à part par :class:`src.studio.jobs.JobManager`, exactement comme la
vraie commande ``batch`` : ``python tests/studio_fake_batch.py <dossier> <chapitres> <pause> [--fail N] [--compile]``.
Aucun réseau, aucune IA.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.models.audio import VoiceoverManifest  # noqa: E402
from src.models.chapter import ChapterMeta  # noqa: E402
from src.models.scene import ChapterAnalysis, Scene  # noqa: E402
from src.modules.batch_processor import BatchOptions, Stages, process_batch  # noqa: E402
from src.pipeline import PipelineOptions  # noqa: E402
from src.utils import progress  # noqa: E402
from src.utils.gemini_manager import GeminiManager  # noqa: E402


def url_for(n: int) -> str:
    return f"https://www.webtoons.com/en/action/serie/ep-{n}/viewer?title_no=1&episode_no={n}"


def fake_stages(dwell: float, fail: set[int]) -> Stages:
    def work(step_id: str, label: str, total: int) -> None:
        for i in range(total):
            progress.step(step_id, label, i, total)
            time.sleep(dwell / total)
        progress.step(step_id, label, total, total)

    def scrape(url, out_dir, options, result=None):
        work("download", "Téléchargement des images", 4)
        work("figures", "Détection des personnages", 4)
        result.n_panels = 12
        result.timings["scrape+slice"] = dwell
        number = int(url.rsplit("=", 1)[1])
        return ChapterMeta(url=url, final_url=url, series_title="Serie Test", episode_title=f"Chapter {number}", title_no=1,
                           episode_no=number, image_urls=["a"])

    def analyze(meta, out_dir, options, result=None, *, manager=None):
        progress.step("script", "Claude écrit le script")
        time.sleep(dwell)
        if meta.episode_no in fail:
            raise RuntimeError("script impossible (echec simule)")
        result.timings["analyze"] = dwell
        return ChapterAnalysis(model="fake", language="en", n_panels=12,
                               scenes=[Scene(index=0, panel_ids=[0], narration="Hi.", emotion="calm")])

    def tts(analysis, out_dir, options, result=None):
        work("voice", "Synthèse de la voix", 6)
        result.timings["tts"] = dwell
        return VoiceoverManifest(language="en", lang_code="a", voice="v", speed=1.0, padding_s=0.0, sample_rate=24000,
                                 items=[], total_duration_s=300.0)

    def montage(analysis, manifest, meta, out_dir, options, result=None):
        work("timeline", "Plan de montage", 2)
        result.total_duration_s = 300.0
        result.timings["timeline"] = dwell / 2
        return result

    return Stages(scrape=scrape, analyze=analyze, tts=tts, montage=montage)


class _Client:
    class models:  # noqa: N801
        @staticmethod
        def generate_content(**kwargs):
            return {}


@progress.outcome()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("chapters", type=int)
    parser.add_argument("dwell", type=float)
    parser.add_argument("--fail", type=int, action="append", default=[])
    parser.add_argument("--compile", action="store_true")
    args = parser.parse_args()
    progress.start("batch", compile=args.compile)
    progress.phase("resolve")
    urls = [url_for(n) for n in range(1, args.chapters + 1)]
    print(f"{len(urls)} chapitre(s) cible(s)", flush=True)
    manager = GeminiManager(["AIzaSyFAKEKEY000001"], models=["m1"], max_rpm=100, client_factory=lambda k: _Client())
    batch = BatchOptions(max_chapters=3, status_file=args.root / "batch_status.json", out_root=args.root / "output")
    options = PipelineOptions(make_preview=not args.compile, make_capcut=not args.compile)
    import asyncio

    report = asyncio.run(process_batch(urls, options, batch, manager=manager, stages=fake_stages(args.dwell, set(args.fail))))
    print(f"{len(report.results)} termine(s), {len(report.errors)} en echec", flush=True)
    if args.compile and not report.errors:
        progress.phase("compile")
        for i in range(5):
            progress.step("preview", "Rendu de l'aperçu", i, 5)
            time.sleep(args.dwell / 5)
    if report.errors and not report.results:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
