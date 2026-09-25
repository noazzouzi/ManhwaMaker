"""Ligne de commande toonsplit.

    .\\.venv\\Scripts\\python.exe -m src.modules.toonsplit split eval/toonsplit/001/strip.webp
    .\\.venv\\Scripts\\python.exe -m src.modules.toonsplit split STRIP --spec none            # sans IA
    .\\.venv\\Scripts\\python.exe -m src.modules.toonsplit eval eval/toonsplit --spec manual
    .\\.venv\\Scripts\\python.exe -m src.modules.toonsplit eval eval/toonsplit --ai gemini --judge always
    .\\.venv\\Scripts\\python.exe -m src.modules.toonsplit locate STRIP DOSSIER_CROPS
    .\\.venv\\Scripts\\python.exe -m src.modules.toonsplit figures eval/toonsplit/cheon_1/strip.webp  # personnages seuls

``--spec`` : ``ai`` (défaut : le fournisseur de ``--ai``), ``manual`` (``spec_manual.json`` du
cas), ``none`` (spec déduite des détections) ou le chemin d'un JSON de specs. ``--ai`` :
``claude`` (défaut, CLI Claude Code de la machine : ``--claude-model``, ``--claude-effort``)
ou ``gemini``. ``--param cle=valeur`` surcharge un
paramètre de recherche (ex. ``--param w_ratio=0`` pour l'ablation de la préférence 2:3).
Les sorties console restent en ASCII (console Windows cp1252).
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from dataclasses import fields, replace
from pathlib import Path
from typing import Any

import cv2

from src.modules.toonsplit import detectors
from src.modules.toonsplit.ai import DEFAULT_RPM, AiJudge, AiSpecProvider, JsonClient, ManualSpecProvider, make_client
from src.modules.toonsplit.evaluate import evaluate, find_cases, locate_crops, render_block, summarize, write_html
from src.modules.toonsplit.figures import extract_figures, figures_sheet, save_figures
from src.modules.toonsplit.pipeline import analyze_strip, load_image, save_shots, write_json
from src.modules.toonsplit.search import SearchParams
from src.utils.config import PROJECT_ROOT


def _say(text: str) -> None:
    print(text.encode("ascii", "replace").decode("ascii"))


def _params(overrides: list[str]) -> SearchParams:
    params = SearchParams()
    types = {f.name: type(getattr(params, f.name)) for f in fields(SearchParams)}
    changes: dict[str, Any] = {}
    for item in overrides:
        key, _, value = item.partition("=")
        if key not in types:
            raise SystemExit(f"parametre inconnu : {key} (connus : {', '.join(types)})")
        kind = types[key]
        changes[key] = tuple(float(v) for v in value.split(",")) if kind is tuple else kind(value)
    return replace(params, **changes)


def _client(args: argparse.Namespace) -> JsonClient:
    return make_client(args.ai, model=args.claude_model, effort=args.claude_effort, max_rpm=args.max_gemini_rpm)


def _providers(args: argparse.Namespace, manual: Path | None) -> tuple[Any, Any]:
    spec = args.spec
    if spec in ("claude", "gemini"):  # raccourci : --spec claude == --spec ai --ai claude
        args.ai, spec = spec, "ai"
    client = _client(args)
    if spec == "ai":
        provider: Any = AiSpecProvider(client)
    elif spec == "none":
        provider = None
    elif spec == "manual":
        if manual is None:
            raise SystemExit("--spec manual : pas de spec_manual.json pour ce cas")
        provider = ManualSpecProvider.from_file(manual)
    else:
        provider = ManualSpecProvider.from_file(spec)
    judge_fn = AiJudge(client) if args.judge != "never" else None
    return provider, judge_fn


def cmd_split(args: argparse.Namespace) -> int:
    image = Path(args.image)
    out = Path(args.out) if args.out else PROJECT_ROOT / "output" / "toonsplit" / image.stem
    provider, judge = _providers(args, image.parent / "spec_manual.json")
    result = analyze_strip(image, spec_provider=provider, judge=judge, judge_mode=args.judge, params=_params(args.param))
    img = load_image(image)
    save_shots(img, result.shots, out / "shots")
    write_json(out / "shots.json", [s.as_dict() for s in result.shots])
    write_json(out / "analysis.json", result.as_dict())
    debug = out / "debug"
    debug.mkdir(parents=True, exist_ok=True)
    for b in result.blocks:
        ok, data = cv2.imencode(".jpg", render_block(img, b), [cv2.IMWRITE_JPEG_QUALITY, 85])
        if ok:
            (debug / f"block_{b.block.index:02d}.jpg").write_bytes(data.tobytes())
    for s in result.shots:
        kind = "pan" if s.pan else ("texte" if not s.has_image else "crop")
        _say(f"bloc {s.source_block:2d} {s.role:<10} {kind:<5} {s.y0:6d}-{s.y1:<6d} tts={len(s.tts)}")
    _say(f"PLANS={len(result.shots)} APPELS_IA={result.ai_calls} COUT_EQUIV_USD={result.ai_cost_usd} SORTIE={out}")
    return 0


def cmd_eval(args: argparse.Namespace) -> int:
    root = Path(args.root)
    cases = find_cases(root)
    if not cases:
        raise SystemExit(f"aucun cas d'evaluation sous {root}")
    out = Path(args.out) if args.out else PROJECT_ROOT / "output" / "toonsplit_eval"
    predictions = json.loads(Path(args.predictions).read_text(encoding="utf-8")) if args.predictions else None
    params = _params(args.param)
    runs = []
    for case in cases:
        provider, judge = _providers(args, case.spec_manual)
        result = analyze_strip(case.strip, spec_provider=provider, judge=judge, judge_mode=args.judge, params=params)
        windows = None
        if predictions is not None:
            shots = predictions.get(case.name, []) if isinstance(predictions, dict) else predictions
            windows = [(int(s["source_block"]), int(s["y0"]), int(s["y1"]), bool(s.get("pan", False))) for s in shots]
        ev = evaluate(case, result, windows)
        runs.append((case, result, ev))
        _say(f"{case.name}: {json.dumps(ev.summary())}")
    summary = summarize([ev for _, _, ev in runs])
    summary["config"] = {"spec": args.spec, "ai": args.ai, "claude_model": args.claude_model, "judge": args.judge,
                         "predictions": args.predictions, "params": args.param,
                         "bubble_yolo": bool(detectors.bubble_yolo_path())}
    write_json(out / "metrics.json", {"summary": summary, "cases": {
        ev.case: {"refs": [r.__dict__ for r in ev.refs], "shots": [s.__dict__ for s in ev.shots]} for _, _, ev in runs}})
    write_json(out / "analysis.json", {case.name: result.as_dict() for case, result, _ in runs})
    page = write_html(out, runs, args.title or f"toonsplit - spec {args.spec} ({args.ai}), juge {args.judge}", summary)
    _say(f"SYNTHESE {json.dumps(summary)}")
    _say(f"PLANCHE {page}")
    return 0


def cmd_figures(args: argparse.Namespace) -> int:
    """Personnages seuls (cadres jaunes) : détecteurs uniquement, aucun appel IA."""
    root = Path(args.out) if args.out else PROJECT_ROOT / "output" / "toonsplit_figures"
    total = 0
    for image in args.images:
        path = Path(image)
        name = path.parent.name if path.stem == "strip" else path.stem
        out = root / name
        img = load_image(path)
        result = analyze_strip(img, spec_provider=None, judge=None)
        figures = extract_figures(result, margin=args.margin, require_head=not args.keep_headless, bubbles=args.bubbles,
                                  group=not args.separate)
        save_figures(img, figures, out / "figures")
        write_json(out / "figures.json", [f.as_dict() for f in figures])
        ok, data = cv2.imencode(".jpg", figures_sheet(img, figures), [cv2.IMWRITE_JPEG_QUALITY, 85])
        if ok:
            (out / "figures_sheet.jpg").write_bytes(data.tobytes())
        total += len(figures)
        _say(f"{name}: {len(figures)} personnage(s) -> {out / 'figures'}")
    _say(f"PERSONNAGES={total} SORTIE={root}")
    return 0


def cmd_locate(args: argparse.Namespace) -> int:
    strip = load_image(args.strip)
    paths = sorted(p for p in Path(args.crops).iterdir() if p.suffix.lower() in (".png", ".jpg", ".jpeg", ".webp"))
    refs = locate_crops(strip, [(p.stem, load_image(p)) for p in paths])
    data = {"strip": Path(args.strip).name, "crops": [r.__dict__ for r in refs]}
    if args.out:
        write_json(Path(args.out), data)
    _say(json.dumps(data, indent=1))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m src.modules.toonsplit", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("-v", "--verbose", action="store_true")
    sub = parser.add_subparsers(dest="command", required=True)

    def common(p: argparse.ArgumentParser, spec_default: str) -> None:
        p.add_argument("--spec", default=spec_default, help="ai | manual | none | chemin JSON (claude, gemini : raccourcis)")
        p.add_argument("--ai", default="claude", choices=("claude", "gemini"), help="fournisseur de la spec IA et du juge")
        p.add_argument("--claude-model", help="modele du CLI Claude (ex. opus, sonnet) ; defaut : celui du CLI")
        p.add_argument("--claude-effort", help="effort du CLI Claude (low, medium, high...) ; defaut : celui du CLI")
        p.add_argument("--judge", default="auto", choices=("auto", "always", "never"))
        p.add_argument("--max-gemini-rpm", type=int, default=DEFAULT_RPM)
        p.add_argument("--param", action="append", default=[], metavar="CLE=VALEUR")
        p.add_argument("--bubble-yolo", action="store_true", help="ajoute le YOLOv8 bulles (ONNX dans models/)")
        p.add_argument("--out")

    p = sub.add_parser("split", help="decoupe un strip en plans")
    p.add_argument("image")
    common(p, "ai")
    p.set_defaults(func=cmd_split)
    p = sub.add_parser("eval", help="compare aux crops de reference")
    p.add_argument("root")
    p.add_argument("--predictions", help="plans externes a evaluer (JSON) au lieu de ceux du pipeline")
    p.add_argument("--title")
    common(p, "ai")
    p.set_defaults(func=cmd_eval)
    p = sub.add_parser("figures", help="extrait les personnages (cadres jaunes) en images, sans IA")
    p.add_argument("images", nargs="+")
    p.add_argument("--margin", type=float, default=0.0, help="marge autour du personnage (part de sa taille)")
    p.add_argument("--bubbles", choices=("cut", "whole"), default="cut",
                   help="cut : zone jaune telle quelle ; whole : agrandie aux bulles qu'elle touche")
    p.add_argument("--keep-headless", action="store_true", help="garde les zones sans tete detectee")
    p.add_argument("--separate", action="store_true", help="une image par personnage (defaut : une par case)")
    p.add_argument("--out")
    p.set_defaults(func=cmd_figures)
    p = sub.add_parser("locate", help="retrouve des crops faits a la main dans le strip")
    p.add_argument("strip")
    p.add_argument("crops")
    p.add_argument("--out")
    p.set_defaults(func=cmd_locate)

    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    if getattr(args, "bubble_yolo", False):
        os.environ[detectors.BUBBLE_YOLO_ENV] = "1"
    return int(args.func(args) or 0)


if __name__ == "__main__":
    sys.exit(main())
