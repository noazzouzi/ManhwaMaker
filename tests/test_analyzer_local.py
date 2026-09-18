"""Script de validation locale de l'Analyzer Gemini (Module 3).

Usage (depuis la racine du projet) :

    python tests/test_analyzer_local.py --panels-dir output/verify_real
    python tests/test_analyzer_local.py --url "https://www.webtoons.com/.../viewer?title_no=95&episode_no=1"
    python tests/test_analyzer_local.py --panels-dir output/verify_real --dry-run

Options : ``--out FICHIER`` (defaut ``<panels-dir>/scenes.json``), ``--model``,
``--batch-size`` (1-15), ``--language`` (defaut ``fr``), ``--max-panels N``
(limite le cout d'un essai), ``--dry-run`` (aucun appel API : client factice qui
renvoie une scene par case, pour valider le pipeline hors ligne), ``--verbose``.

Une cle Gemini est requise sauf en ``--dry-run`` : variable ``GEMINI_API_KEY`` /
``GOOGLE_API_KEY`` ou fichier ``.gemini_key`` a la racine du projet (ignore par git).
Le script ecrit ``scenes.json``, affiche les scenes et termine par une ligne
``SCENES=<n> PANELS=<m> BATCHES=<b> TOKENS=<in>+<out>+<reflexion>``.
Codes de sortie : 0 succes, 1 erreur, 2 cle API absente.

Note : ce fichier est collecte par pytest (prefixe ``test_``) mais ne definit
aucune fonction ``test_*`` ; son import est sans effet de bord.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from google.genai import types  # noqa: E402

from src.models.chapter import ChapterMeta  # noqa: E402
from src.models.panel import Panel  # noqa: E402
from src.modules.analyzer import (  # noqa: E402
    DEFAULT_BATCH_SIZE,
    DEFAULT_LANGUAGE,
    MAX_BATCH_SIZE,
    GeminiAnalyzer,
    format_scenes,
    resolve_model,
    save_analysis,
)
from src.modules.slicer import load_panels, render_debug_overlay, save_panels, slice_panels  # noqa: E402
from src.utils.config import gemini_key_hint, load_gemini_api_key  # noqa: E402
from src.utils.report import build_html_report  # noqa: E402

_CAPTION = re.compile(r"^Panel (\d+) \(")


def ascii_safe(text: object) -> str:
    """Rend une chaîne imprimable sur une console cp1252."""
    return str(text).encode("ascii", "replace").decode("ascii")


def make_console_tolerant() -> None:
    """Empêche ``print``/logging de lever ``UnicodeEncodeError`` sur une console non UTF-8."""
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            try:
                reconfigure(errors="replace")
            except (ValueError, OSError):
                pass


_BEAT_LINE = re.compile(r"^Beat (\d+) \(panels [\d, ]+\)(\s*\[FILLER[^\]]*\])?:", re.MULTILINE)
_PARAGRAPH_LINE = re.compile(r"^Paragraph (\d+) \(candidate panels: ([\d, ]*)\):", re.MULTILINE)


class DryRunClient:
    """Client factice pour les trois etapes (beats, script, cases cles), sans appel reseau."""

    class _Models:
        def __init__(self) -> None:
            self.calls = 0
            self.bytes_sent = 0

        def generate_content(self, *, model: str, contents: list[types.Part], config):
            from src.models.scene import BeatBatch, KeyframeBatch, ScriptDraft

            self.calls += 1
            text = "\n".join(part.text for part in contents if part.text)
            ids: list[int] = []
            for part in contents:
                if part.inline_data is not None and part.inline_data.data:
                    self.bytes_sent += len(part.inline_data.data)
                match = _CAPTION.match(part.text or "")
                if match:
                    ids.append(int(match.group(1)))
            schema = config.response_schema
            if schema is BeatBatch:
                payload = {"beats": [
                    {"panel_ids": [pid], "summary": f"[dry-run] Beat of panel {pid}.", "characters": [], "dialogue": [], "is_filler": False}
                    for pid in ids
                ]}
            elif schema is ScriptDraft:
                story = [int(m.group(1)) for m in _BEAT_LINE.finditer(text) if not m.group(2)]
                payload = {"paragraphs": [
                    {"text": f"[dry-run] The story moves through beats {story[i:i + 3]}.", "beat_ids": story[i:i + 3], "emotion": "neutral"}
                    for i in range(0, len(story), 3)
                ]}
            elif schema is KeyframeBatch:
                payload = {"choices": [
                    {"paragraph_index": int(m.group(1)), "key_panel_ids": [int(x) for x in m.group(2).split(",") if x.strip()][:1]}
                    for m in _PARAGRAPH_LINE.finditer(text)
                ]}
            else:
                raise AssertionError(f"schema inattendu {schema}")
            return types.GenerateContentResponse(
                candidates=[types.Candidate(content=types.Content(role="model", parts=[types.Part(text=json.dumps(payload))]))]
            )

    def __init__(self) -> None:
        self.models = DryRunClient._Models()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validation locale de l'Analyzer Gemini (cases -> scenes narrees).",
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--panels-dir", type=Path, help="Dossier contenant panels.json + PNG (sortie du slicer).")
    source.add_argument("--url", type=str, help="URL d'un chapitre Webtoons (scrape + slice + analyse).")
    parser.add_argument("--out", type=Path, default=None, help="Fichier scenes.json de sortie.")
    parser.add_argument("--work-dir", type=Path, default=PROJECT_ROOT / "output" / "analyzer_test",
                        help="Dossier de travail pour --url (cases + overlay).")
    parser.add_argument("--model", type=str, default=None, help=f"Modele Gemini (defaut {resolve_model()}).")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE, metavar="N",
                        choices=range(1, MAX_BATCH_SIZE + 1),
                        help=f"Images par lot, 1-{MAX_BATCH_SIZE} (defaut {DEFAULT_BATCH_SIZE}).")
    parser.add_argument("--language", type=str, default=DEFAULT_LANGUAGE, help="Langue de la narration.")
    parser.add_argument("--thinking-budget", type=int, default=None, metavar="N",
                        help="Budget de tokens de reflexion (0 = desactive ; defaut : celui du modele).")
    parser.add_argument("--max-panels", type=int, default=None, help="N'analyser que les N premieres cases.")
    parser.add_argument("--dry-run", action="store_true", help="Client factice, aucun appel API.")
    parser.add_argument("--no-report", action="store_true",
                        help="Ne pas ecrire le rapport HTML (report.html a cote de scenes.json).")
    parser.add_argument("--verbose", "-v", action="store_true", help="Logs DEBUG.")
    return parser


def load_source(args: argparse.Namespace) -> tuple[list[Panel], ChapterMeta | None, Path]:
    """Charge les cases depuis un dossier ou un chapitre en ligne ; renvoie (cases, meta, dossier)."""
    if args.panels_dir is not None:
        print(f"Source : dossier {ascii_safe(args.panels_dir)}")
        return load_panels(args.panels_dir), None, args.panels_dir

    from src.modules.scraper import scrape_chapter

    print(f"Source : URL {ascii_safe(args.url)}")
    strip, meta = scrape_chapter(args.url)
    print(f"Serie   : {ascii_safe(meta.series_title)} - Episode : {ascii_safe(meta.episode_title)}")
    work_dir: Path = args.work_dir
    work_dir.mkdir(parents=True, exist_ok=True)
    panels = slice_panels(strip)
    save_panels(panels, work_dir)
    render_debug_overlay(strip, panels, work_dir / "debug_overlay.png")
    print(f"Cases   : {len(panels)} ecrites dans {ascii_safe(work_dir)}")
    return panels, meta, work_dir


def run(args: argparse.Namespace) -> int:
    if not args.dry_run and load_gemini_api_key() is None:
        print(f"ECHEC : aucune cle Gemini ({gemini_key_hint()}, ou lancer avec --dry-run).")
        return 2

    panels, meta, source_dir = load_source(args)
    if args.max_panels is not None:
        panels = panels[: max(0, args.max_panels)]
        print(f"Limite  : {len(panels)} case(s) analysee(s) (--max-panels)")
    if not panels:
        print("ECHEC : aucune case a analyser.")
        return 1

    client = DryRunClient() if args.dry_run else None
    analyzer = GeminiAnalyzer(
        client=client, model=args.model, batch_size=args.batch_size, language=args.language,
        thinking_budget=args.thinking_budget,
    )
    print(f"Modele  : {analyzer.model}  lots de {args.batch_size}  langue {args.language}"
          + ("  [DRY-RUN]" if args.dry_run else ""))

    started = time.perf_counter()
    analysis = analyzer.analyze_panels(panels, meta)
    elapsed = time.perf_counter() - started

    out_path: Path = args.out if args.out is not None else source_dir / "scenes.json"
    save_analysis(analysis, out_path)
    if not args.no_report:
        report_path = build_html_report(panels, analysis, out_path.with_name(out_path.stem + "_report.html"))
        print(f"Rapport : {ascii_safe(report_path)}")
    print()
    print(format_scenes(analysis))
    if args.dry_run:
        sent_kb = client.models.bytes_sent / 1024
        print(f"Dry-run : {client.models.calls} appel(s) simule(s), {sent_kb:.0f} Ko d'images encodees")
    print(f"Duree   : {elapsed:.1f}s   Fichier : {ascii_safe(out_path)}")
    skipped = sorted(set(p.index for p in panels) - set(analysis.covered_panel_ids()))
    print(f"Cases cles : {analysis.n_key_panels}/{analysis.n_panels} montees ; non retenues : {skipped}")
    print(
        f"SCENES={analysis.n_scenes} KEY_PANELS={analysis.n_key_panels} PANELS={analysis.n_panels} "
        f"BEATS={len(analysis.beats)} CALLS={analysis.n_batches} WORDS={analysis.script_words} "
        f"TOKENS={analysis.prompt_tokens}+{analysis.output_tokens}+{analysis.thinking_tokens}"
    )
    return 0 if analysis.n_scenes else 1


def main(argv: list[str] | None = None) -> int:
    make_console_tolerant()
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )
    try:
        return run(args)
    except Exception as exc:  # noqa: BLE001 - script : toute erreur => code 1
        logging.getLogger(__name__).exception("Analyzer test failed")
        print(f"ECHEC : {ascii_safe(type(exc).__name__)}: {ascii_safe(exc)}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
