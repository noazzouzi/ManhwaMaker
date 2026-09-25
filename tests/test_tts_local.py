"""Script de validation locale du moteur Kokoro TTS (Module 4).

Usage (depuis la racine du projet) :

    python tests/test_tts_local.py --scenes output/max_level_newbie_ep1/scenes.json
    python tests/test_tts_local.py --scenes output/.../scenes.json --voice am_michael --speed 1.05
    python tests/test_tts_local.py --scenes output/.../scenes.json --dry-run

Options : ``--out-dir DIR`` (defaut ``<dossier des scenes>/audio``), ``--voice``,
``--speed``, ``--padding`` (defaut 0.3 s), ``--pronunciations FICHIER``,
``--include-filler``, ``--max-scenes N``, ``--no-full`` (pas de WAV concatene),
``--dry-run`` (pipeline factice : un bip par mot, aucun modele charge), ``--verbose``.

Le script ecrit un WAV par scene + ``voiceover.json`` + ``voiceover_full.wav``,
regenere le rapport HTML (``scenes_report.html``) avec un lecteur audio par scene si
``panels.json`` est disponible a cote des scenes, affiche un tableau ASCII et termine
par ``SEGMENTS=<n> TOTAL=<secondes>s``. Codes de sortie : 0 succes, 1 erreur.

Note : ce fichier est collecte par pytest (prefixe ``test_``) mais ne definit
aucune fonction ``test_*`` ; son import est sans effet de bord.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.modules.analyzer import load_analysis  # noqa: E402
from src.modules.slicer import load_panels  # noqa: E402
from src.modules.tts_engine import (  # noqa: E402
    DEFAULT_PADDING_S,
    DEFAULT_SENTENCE_GAP_S,
    DEFAULT_SPEED,
    SAMPLE_RATE,
    KokoroTTS,
    TTSError,
    export_mp3,
    format_manifest,
)
from src.utils.config import DEFAULT_NARRATION_LANGUAGE  # noqa: E402
from src.utils.report import build_html_report  # noqa: E402


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


class DryRunPipeline:
    """Pipeline factice : un bip de 0,25 s par mot (440 Hz), aucun modèle chargé."""

    def __call__(self, text: str, *, voice: str, speed: float, split_pattern: str):
        for word in text.split():
            n = int(0.25 * SAMPLE_RATE / speed)
            t = np.arange(n, dtype=np.float32) / SAMPLE_RATE
            tone = 0.2 * np.sin(2 * np.pi * 440.0 * t).astype(np.float32)
            tone[-n // 5 :] = 0.0  # petite pause entre les mots
            yield (word, "", tone)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validation locale du moteur Kokoro TTS (scenes -> WAV + manifeste).",
    )
    parser.add_argument("--scenes", type=Path, required=True, help="Fichier scenes.json produit par l'analyzer.")
    parser.add_argument("--out-dir", type=Path, default=None, help="Dossier des WAV (defaut <scenes>/audio).")
    parser.add_argument("--language", type=str, default=None,
                        help=f"Langue de narration (defaut : celle du scenes.json, sinon {DEFAULT_NARRATION_LANGUAGE}).")
    parser.add_argument("--voice", type=str, default=None, help="Voix Kokoro (defaut : am_fenrir,am_michael pour l'anglais).")
    parser.add_argument("--speed", type=float, default=DEFAULT_SPEED, help=f"Vitesse (defaut {DEFAULT_SPEED}).")
    parser.add_argument("--padding", type=float, default=DEFAULT_PADDING_S,
                        help=f"Silence de fin en secondes (defaut {DEFAULT_PADDING_S}).")
    parser.add_argument("--sentence-gap", type=float, default=DEFAULT_SENTENCE_GAP_S,
                        help=f"Pause entre les phrases en secondes (defaut {DEFAULT_SENTENCE_GAP_S}, 0 = aucune).")
    parser.add_argument("--pronunciations", type=Path, default=None,
                        help="Dictionnaire phonetique JSON (defaut config/pronunciations.json).")
    parser.add_argument("--include-filler", action="store_true", help="Synthetiser aussi les scenes filler.")
    parser.add_argument("--max-scenes", type=int, default=None, help="Ne synthetiser que les N premieres scenes.")
    parser.add_argument("--no-full", action="store_true", help="Ne pas ecrire voiceover_full.wav.")
    parser.add_argument("--no-mp3", action="store_true", help="Ne pas exporter voiceover_full.mp3 (apercu).")
    parser.add_argument("--no-report", action="store_true", help="Ne pas regenerer le rapport HTML.")
    parser.add_argument("--dry-run", action="store_true", help="Pipeline factice, aucun modele charge.")
    parser.add_argument("--verbose", "-v", action="store_true", help="Logs DEBUG.")
    return parser


def run(args: argparse.Namespace) -> int:
    if not args.scenes.is_file():
        print(f"ECHEC : fichier introuvable {ascii_safe(args.scenes)}")
        return 1
    analysis = load_analysis(args.scenes)
    print(f"Scenes  : {ascii_safe(args.scenes)} ({analysis.n_scenes} scenes dont {analysis.n_filler} filler)")
    if args.max_scenes is not None:
        analysis.scenes = analysis.scenes[: max(0, args.max_scenes)]
        print(f"Limite  : {len(analysis.scenes)} scene(s) (--max-scenes)")

    language = args.language or analysis.language or DEFAULT_NARRATION_LANGUAGE
    out_dir: Path = args.out_dir if args.out_dir is not None else args.scenes.parent / "audio"
    tts = KokoroTTS(
        language=language, voice=args.voice, speed=args.speed, padding_s=args.padding,
        sentence_gap_s=args.sentence_gap,
        pronunciations=args.pronunciations, pipeline=DryRunPipeline() if args.dry_run else None,
    )
    print(f"Voix    : {tts.voice} (pipeline {tts.lang_code}, langue {language})  vitesse {args.speed}"
          + ("  [DRY-RUN]" if args.dry_run else ""))
    print(f"Phonet. : {len(tts.pronunciations)} remplacement(s)")

    started = time.perf_counter()
    manifest = tts.synthesize_analysis(
        analysis, out_dir, include_filler=args.include_filler,
        full_file=None if args.no_full else "voiceover_full.wav",
    )
    elapsed = time.perf_counter() - started

    print()
    print(format_manifest(manifest))
    print(f"Duree   : {elapsed:.1f}s de calcul pour {manifest.total_duration_s:.1f}s d'audio "
          f"(x{manifest.total_duration_s / max(elapsed, 1e-6):.1f} temps reel)")
    print(f"Dossier : {ascii_safe(out_dir)}")

    if manifest.full_file and not args.no_mp3:
        try:
            mp3 = export_mp3(out_dir / manifest.full_file, out_dir / "voiceover_full.mp3")
            print(f"MP3     : {ascii_safe(mp3)} ({mp3.stat().st_size // 1024} Ko)")
        except TTSError as exc:
            print(f"MP3     : non exporte ({ascii_safe(exc)})")

    if not args.no_report:
        panels_dir = args.scenes.parent
        if (panels_dir / "panels.json").is_file():
            report_path = args.scenes.with_name(args.scenes.stem + "_report.html")
            build_html_report(load_panels(panels_dir), analysis, report_path, audio=manifest, audio_dir=out_dir)
            print(f"Rapport : {ascii_safe(report_path)} (avec lecteurs audio)")
        else:
            print("Rapport : panels.json absent a cote des scenes, rapport HTML non regenere")

    print(f"SEGMENTS={manifest.n_items} TOTAL={manifest.total_duration_s:.1f}s")
    return 0 if manifest.n_items else 1


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
        logging.getLogger(__name__).exception("TTS test failed")
        print(f"ECHEC : {ascii_safe(type(exc).__name__)}: {ascii_safe(exc)}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
