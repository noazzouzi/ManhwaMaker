"""Script de validation RÉSEAU du Module 1 (scraper / stitcher).

Télécharge un vrai chapitre Webtoons (Tower of God ep. 1 par défaut), assemble
la bande continue, l'enregistre dans ``output/scraper_test/strip.png`` et
affiche un résumé ASCII. Code de sortie 0 si tout va bien, 1 sinon.

Usage (depuis la racine du projet) :
    .\\.venv\\Scripts\\python.exe tests/test_scraper_local.py [--url URL] [--out FICHIER]
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

# Rend ``src`` importable même si le script est lancé depuis un autre dossier.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.modules.scraper import (  # noqa: E402
    ascii_safe,
    format_summary,
    save_strip,
    scrape_chapter,
)

DEFAULT_URL = (
    "https://www.webtoons.com/en/fantasy/tower-of-god/season-1-ep-1/viewer"
    "?title_no=95&episode_no=1"
)
DEFAULT_OUT = PROJECT_ROOT / "output" / "scraper_test" / "strip.png"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Analyse les arguments de la ligne de commande."""
    parser = argparse.ArgumentParser(description="Test reseau du scraper Webtoons.")
    parser.add_argument("--url", default=DEFAULT_URL, help="URL du viewer Webtoons.")
    parser.add_argument(
        "--out", type=Path, default=DEFAULT_OUT, help="Fichier PNG de sortie."
    )
    parser.add_argument(
        "--delay", type=float, default=0.15, help="Pause (s) entre deux images."
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Logs DEBUG.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Exécute le test réseau et retourne le code de sortie."""
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    print(ascii_safe(f"URL: {args.url}"))
    started = time.perf_counter()
    try:
        strip, meta = scrape_chapter(args.url, delay=args.delay)
        saved = save_strip(strip, args.out)
    except Exception as exc:  # noqa: BLE001 - un script de test doit tout rapporter
        logging.getLogger(__name__).exception("Echec du scraping")
        print(ascii_safe(f"FAILED: {type(exc).__name__}: {exc}"))
        return 1
    elapsed = time.perf_counter() - started

    # Bloc recapitulatif partage avec la CLI (``python -m src.modules.scraper``).
    print(format_summary(meta, strip, saved, elapsed))
    for index, (url, size) in enumerate(zip(meta.image_urls, meta.chunk_sizes)):
        print(ascii_safe(f"  [{index:02d}] {size[0]}x{size[1]}  {url}"))

    # Verifications de coherence minimales.
    problems: list[str] = []
    if meta.n_chunks == 0:
        problems.append("aucun morceau telecharge")
    if len(meta.chunk_sizes) != meta.n_chunks:
        problems.append("chunk_sizes incoherent avec image_urls")
    if strip.mode != "RGB":
        problems.append(f"mode inattendu: {strip.mode}")
    if strip.height < sum(h for _w, h in meta.chunk_sizes) * 0.5:
        problems.append("hauteur de bande anormalement faible")
    if not saved.exists() or saved.stat().st_size == 0:
        problems.append("fichier de sortie absent ou vide")
    if not meta.series_title:
        problems.append("titre de serie vide")

    print(f"CHUNKS={meta.n_chunks} STRIP={strip.width}x{strip.height}")
    if problems:
        for problem in problems:
            print(f"PROBLEM: {problem}")
        print("RESULT=FAIL")
        return 1
    print("RESULT=OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
