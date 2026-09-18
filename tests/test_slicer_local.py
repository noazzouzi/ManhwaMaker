"""Script de validation locale du Smart Slicer (Module 2).

Usage (depuis la racine du projet) :

    python tests/test_slicer_local.py --synthetic
    python tests/test_slicer_local.py --image output/scraper_test/strip.png
    python tests/test_slicer_local.py --url "https://www.webtoons.com/en/.../viewer?title_no=95&episode_no=1"

Options : ``--out DIR`` (defaut ``output/slicer_test``), ``--threshold``,
``--min-gap``, ``--padding``, ``--giant``, ``--min-height``, ``--seed``, ``--verbose``.

Le script découpe la bande, écrit les cases PNG + ``panels.json`` + un overlay
de debug dans ``--out``, affiche un tableau ASCII et une ligne de résumé
``PANELS=<n> SCROLL_VERTICAL=<m>``. Code de sortie : 0 si au moins une case,
1 sinon (ou en cas d'erreur).

Note : ce fichier est aussi collecté par pytest (préfixe ``test_``) mais ne
définit aucune fonction ``test_*`` ; son import est sans effet de bord.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

# Rend les imports absolus (``src.…``, ``tests.…``) possibles avec
# ``python tests/test_slicer_local.py`` lancé depuis n'importe où.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from PIL import Image  # noqa: E402

from src.models.panel import Panel  # noqa: E402
from src.modules.slicer import (  # noqa: E402
    DEFAULT_MARGIN_PADDING,
    DEFAULT_MIN_GAP,
    DEFAULT_MIN_PANEL_HEIGHT,
    DEFAULT_VARIANCE_THRESHOLD,
    GIANT_PANEL_HEIGHT,
    render_debug_overlay,
    save_panels,
    slice_panels,
)
from tests.synthetic_strip import DEFAULT_SPEC, make_synthetic_strip  # noqa: E402

DEFAULT_OUT_DIR = PROJECT_ROOT / "output" / "slicer_test"


def ascii_safe(text: object) -> str:
    """Rend une chaîne imprimable sur une console cp1252 (remplace les non-ASCII par '?')."""
    return str(text).encode("ascii", "replace").decode("ascii")


def make_console_tolerant() -> None:
    """Empêche ``print``/logging de lever ``UnicodeEncodeError`` sur une console non UTF-8.

    Ceinture et bretelles en plus de ``ascii_safe`` : les chemins et messages
    d'exception passent par ``ascii_safe``, mais les logs INFO du module
    (``Saved ... in <chemin>``) partent sur ``stderr`` tels quels. Sur un flux
    cp1252 strict (sortie redirigée, ``PYTHONIOENCODING=cp1252``) un caractère
    hors cp1252 y ferait échouer le script ou polluerait la sortie avec des
    ``--- Logging error ---``.
    """
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            try:
                reconfigure(errors="replace")
            except (ValueError, OSError):  # flux fermé ou non reconfigurable
                pass


def build_parser() -> argparse.ArgumentParser:
    """Construit le parseur d'arguments de la ligne de commande."""
    parser = argparse.ArgumentParser(
        description="Validation locale du Smart Slicer (decoupe d'une bande webtoon en cases).",
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--image", type=Path, help="Chemin d'une bande (PNG/JPEG) a decouper.")
    source.add_argument("--url", type=str, help="URL d'un chapitre Webtoons (scrape + stitch + slice).")
    source.add_argument("--synthetic", action="store_true", help="Utilise une bande synthetique generee.")

    parser.add_argument("--out", type=Path, default=DEFAULT_OUT_DIR, help="Dossier de sortie.")
    parser.add_argument("--threshold", type=float, default=DEFAULT_VARIANCE_THRESHOLD,
                        help=f"Seuil de variance des gouttieres (defaut {DEFAULT_VARIANCE_THRESHOLD}).")
    parser.add_argument("--min-gap", type=int, default=DEFAULT_MIN_GAP,
                        help=f"Hauteur minimale d'une gouttiere en px (defaut {DEFAULT_MIN_GAP}).")
    parser.add_argument("--padding", type=int, default=DEFAULT_MARGIN_PADDING,
                        help=f"Marge ajoutee autour de chaque case en px (defaut {DEFAULT_MARGIN_PADDING}).")
    parser.add_argument("--giant", type=int, default=GIANT_PANEL_HEIGHT,
                        help=f"Hauteur au-dela de laquelle une case est scroll_vertical (defaut {GIANT_PANEL_HEIGHT}).")
    parser.add_argument("--min-height", type=int, default=DEFAULT_MIN_PANEL_HEIGHT,
                        help=f"Hauteur minimale d'une case en px (defaut {DEFAULT_MIN_PANEL_HEIGHT}).")
    parser.add_argument("--seed", type=int, default=0, help="Graine du generateur synthetique.")
    parser.add_argument("--verbose", "-v", action="store_true", help="Logs DEBUG.")
    return parser


def load_strip(args: argparse.Namespace, out_dir: Path) -> Image.Image:
    """Charge la bande selon la source choisie (--image, --url ou --synthetic)."""
    if args.synthetic:
        print("Source : bande synthetique (DEFAULT_SPEC)")
        for i, (panel_h, gutter_h, color) in enumerate(DEFAULT_SPEC):
            print(f"  case {i}: {panel_h}px, gouttiere {gutter_h}px ({ascii_safe(color)})")
        strip = make_synthetic_strip(DEFAULT_SPEC, seed=args.seed)
        strip_path = out_dir / "strip.png"
        strip.save(strip_path)
        print(f"Bande synthetique ecrite : {ascii_safe(strip_path)}")
        return strip

    if args.image is not None:
        print(f"Source : image {ascii_safe(args.image)}")
        if not args.image.is_file():
            raise FileNotFoundError(f"Image introuvable : {args.image}")
        with Image.open(args.image) as loaded:
            return loaded.convert("RGB")

    # --url : import tardif du scraper (Module 1, fichier distinct).
    from src.modules.scraper import scrape_chapter

    print(f"Source : URL {ascii_safe(args.url)}")
    strip, meta = scrape_chapter(args.url)
    print(f"Serie   : {ascii_safe(meta.series_title)}")
    print(f"Episode : {ascii_safe(meta.episode_title)} "
          f"(title_no={meta.title_no}, episode_no={meta.episode_no})")
    print(f"Chunks  : {len(meta.image_urls)}")
    strip_path = out_dir / "strip.png"
    strip.save(strip_path)
    print(f"Bande stitchee ecrite : {ascii_safe(strip_path)}")
    return strip


def print_table(panels: list[Panel]) -> None:
    """Affiche un tableau ASCII des cases détectées."""
    header = f"{'INDEX':>5}  {'Y_START':>8}  {'Y_END':>8}  {'HEIGHT':>7}  {'WIDTH':>6}  TYPE"
    print(header)
    print("-" * len(header))
    for p in panels:
        print(f"{p.index:>5}  {p.y_start:>8}  {p.y_end:>8}  {p.height:>7}  {p.width:>6}  {p.type}")


def run(args: argparse.Namespace) -> int:
    """Exécute la validation ; retourne le code de sortie."""
    out_dir: Path = args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    strip = load_strip(args, out_dir)
    width, height = strip.size
    print(f"STRIP={width}x{height}")
    print(f"Parametres : threshold={args.threshold} min_gap={args.min_gap} "
          f"padding={args.padding} giant={args.giant} min_height={args.min_height}")

    t0 = time.perf_counter()
    panels = slice_panels(
        strip,
        variance_threshold=args.threshold,
        min_gap=args.min_gap,
        margin_padding=args.padding,
        giant_panel_height=args.giant,
        min_panel_height=args.min_height,
    )
    elapsed = time.perf_counter() - t0
    print(f"Decoupe en {elapsed:.3f}s")

    # save_panels(clean=True) supprime les panel_NNN.png d'une execution
    # precedente : panels.json et le dossier restent coherents entre deux runs.
    paths = save_panels(panels, out_dir, clean=True)
    overlay_path = render_debug_overlay(strip, panels, out_dir / "debug_overlay.png")
    print(f"{len(paths)} PNG + panels.json ecrits dans : {ascii_safe(out_dir)}")
    print(f"Overlay de debug : {ascii_safe(overlay_path)}")
    print()
    print_table(panels)
    print()

    n_scroll = sum(1 for p in panels if p.type == "scroll_vertical")
    print(f"PANELS={len(panels)} SCROLL_VERTICAL={n_scroll}")
    if not panels:
        print("ECHEC : aucune case detectee.")
        return 1
    return 0


def main(argv: list[str] | None = None) -> int:
    """Point d'entrée : parse les arguments, configure les logs, lance la validation."""
    make_console_tolerant()
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )
    # Les bandes de chapitres complets depassent facilement la limite anti
    # "decompression bomb" de Pillow (~89 Mpx) : on la desactive pour ce script.
    Image.MAX_IMAGE_PIXELS = None
    try:
        return run(args)
    except Exception as exc:  # noqa: BLE001 - script : toute erreur => code 1
        logging.getLogger(__name__).exception("Slicer test failed")
        print(f"ECHEC : {ascii_safe(type(exc).__name__)}: {ascii_safe(exc)}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
