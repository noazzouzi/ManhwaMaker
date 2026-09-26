"""Lance ManhwaMaker Studio : ``python -m src.studio`` puis http://127.0.0.1:8777.

Le serveur n'écoute que sur la machine locale. Les traitements tournent dans leurs propres
processus : fermer le navigateur ne les arrête pas, et un traitement lancé avant un
redémarrage du serveur est retrouvé au démarrage suivant.
"""

from __future__ import annotations

import argparse
import logging
import threading
import webbrowser

import uvicorn

from src.studio.server import DEFAULT_DATA_DIR, create_app


def main() -> None:
    parser = argparse.ArgumentParser(description="ManhwaMaker Studio (interface web locale)")
    parser.add_argument("--port", type=int, default=8777)
    parser.add_argument("--host", default="127.0.0.1", help="adresse d'écoute (défaut : machine locale seulement)")
    parser.add_argument("--no-browser", action="store_true", help="ne pas ouvrir le navigateur")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    app = create_app(data_dir=DEFAULT_DATA_DIR)
    url = f"http://{'127.0.0.1' if args.host in ('0.0.0.0', '') else args.host}:{args.port}"
    print(f"ManhwaMaker Studio : {url}  (Ctrl+C pour arreter le serveur ; les traitements en cours continuent)")
    if not args.no_browser:
        threading.Timer(1.2, webbrowser.open, args=(url,)).start()
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
