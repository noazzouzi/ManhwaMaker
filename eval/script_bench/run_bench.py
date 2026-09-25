"""Banc d'essai du script : même chapitre, même prompt, mêmes images, Gemini contre Claude CLI.

Pour chaque chapitre (dossier de ``output/`` déjà découpé) :

- ``A_production`` : le script actuel (``scenes.json``), phrase d'abonnement retirée ;
- ``B_gemini`` / ``C_claude`` (Opus) / ``D_sonnet`` / ``E_haiku`` : le prompt et les planches de
  production (:mod:`src.modules.script_writer`), mêmes images pour tous ;
- ``judge.json`` (``--judge``) : un juge Opus lit le chapitre et les scripts anonymisés, dans
  un ordre mélangé, et les note (accroche, rythme, fidélité, clarté, voix, note globale).

Les images : toutes les cases de lecture, empilées dans l'ordre en planches d'au plus
:data:`MAX_IMAGE_PX` pixels (plafond de Claude avant réduction), chaque case surmontée d'un
bandeau « Panel N ». Moins de 100 images par chapitre (plafond de Claude par requête).

Sorties dans ``output/script_bench/<chapitre>/`` et une page de lecture à l'aveugle
``output/script_bench/index.html`` (scripts mélangés, clé et mesures repliées en bas).

    .venv\\Scripts\\python.exe eval\\script_bench\\run_bench.py <chapitre> [<chapitre>...] [--models opus,sonnet,haiku] [--judge]

``modele@effort`` compare les efforts d'un même modèle (``--models opus@low,opus@medium``) :
variantes ``C_claude_low``, ``C_claude_medium``...
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import logging
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Literal

import cv2
from pydantic import BaseModel

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.models.chapter import ChapterMeta  # noqa: E402
from src.modules.analyzer import _chapter_line  # noqa: E402
from src.modules.series_memory import load_series_context, series_key  # noqa: E402
from src.modules.slicer import load_panels  # noqa: E402
from src.modules.script_writer import JPEG_QUALITY, ChapterScript, composites, system_prompt  # noqa: E402
from src.modules.toonsplit.ai import ClaudeCliJson, GeminiJson  # noqa: E402

logger = logging.getLogger("script_bench")
OUT = ROOT / "output" / "script_bench"
GEMINI_MODELS = ("gemini-3.5-flash", "gemini-2.5-flash")
#: Modèles Claude (alias du CLI) et nom de leur variante ; ``C_claude`` reste le nom d'Opus.
CLAUDE_VARIANTS = {"opus": "C_claude", "sonnet": "D_sonnet", "haiku": "E_haiku"}
VARIANTS = {"gemini": "B_gemini", **CLAUDE_VARIANTS}
JUDGE_MODEL = "opus"


BenchScript = ChapterScript


def previous_tail(chapter_dir: Path, episode_no: int | None) -> str:
    """Fin du chapitre précédent : sa fiche de série, sinon le dernier paragraphe de son script."""
    if not episode_no or episode_no <= 1:
        return ""
    prev = chapter_dir.parent / re.sub(r"_ep\d+$", f"_ep{episode_no - 1}", chapter_dir.name)
    scenes = prev / "scenes.json"
    if not scenes.is_file():
        return ""
    data = json.loads(scenes.read_text(encoding="utf-8"))
    story = [s["narration"] for s in data["scenes"] if not s.get("is_filler")]
    return strip_cta(story[-1]) if story else ""


def strip_cta(text: str) -> str:
    return " ".join(s for s in re.split(r"(?<=[.!?])\s+", text) if not re.search(r"subscri", s, re.I)).strip()


def build_request(chapter_dir: Path) -> tuple[str, str, list[tuple[str, bytes]], dict]:
    meta = ChapterMeta.model_validate_json((chapter_dir / "chapter.json").read_text(encoding="utf-8"))
    panels = load_panels(chapter_dir)
    cards, tail = load_series_context(chapter_dir.parent, key=series_key(meta), before_episode=meta.episode_no)
    tail = tail or previous_tail(chapter_dir, meta.episode_no)
    first = not tail and not cards and (meta.episode_no or 1) <= 1
    system = system_prompt(meta, known_characters=cards, previous_tail=tail)
    sheets = composites(panels)
    input_dir = OUT / chapter_dir.name / "input"
    input_dir.mkdir(parents=True, exist_ok=True)
    images = []
    for k, (ids, img) in enumerate(sheets):
        ok, data = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
        (input_dir / f"img_{k:03d}.jpg").write_bytes(data.tobytes())
        images.append((f"Image {k + 1}/{len(sheets)}: panels {ids[0]} to {ids[-1]}", data.tobytes()))
    text = (f"{_chapter_line(meta)}\nThe chapter has {len(panels)} panels, numbered {panels[0].index} to "
            f"{panels[-1].index}, packed into {len(sheets)} images in reading order. Write the script.")
    info = {"panels": len(panels), "images": len(sheets), "image_width": int(sheets[0][1].shape[1]),
            "payload_mb": round(sum(len(d) for _, d in images) / 1e6, 1), "first_chapter": first,
            "characters_in_memory": len(cards), "previous_tail": tail}
    (OUT / chapter_dir.name / "system_prompt.txt").write_text(system, encoding="utf-8")
    return system, text, images, info


# --- Appels ------------------------------------------------------------------------------------
def variant_name(spec: str) -> str:
    """``opus`` -> ``C_claude`` ; ``opus@low`` -> ``C_claude_low`` (même modèle, autre effort)."""
    model, _, effort = spec.partition("@")
    return VARIANTS[model] + (f"_{effort}" if effort else "")


def run_model(name: str, system: str, text: str, images, out_dir: Path) -> dict:
    model, _, effort = name.partition("@")
    client = (GeminiJson(models=GEMINI_MODELS, max_rpm=5, temperature=0.4, retries=1, max_output_tokens=32768)
              if model == "gemini" else ClaudeCliJson(model=model, effort=effort or None, retries=1, timeout_s=1500))
    started = time.perf_counter()
    script = client.ask(system, text, images, BenchScript, label=f"script {name}")
    elapsed = time.perf_counter() - started
    record = {
        "variant": variant_name(name), "seconds": round(elapsed), "calls": client.n_calls,
        "cost_usd": round(client.cost_usd, 3),
        "model": (getattr(client.manager, "current_model", "") if model == "gemini" else ", ".join(sorted(client.models_used))),
        "script": script.model_dump(),
    }
    (out_dir / f"{record['variant']}.json").write_text(json.dumps(record, indent=1, ensure_ascii=False), encoding="utf-8")
    return record


def production_record(chapter_dir: Path, out_dir: Path) -> dict:
    data = json.loads((chapter_dir / "scenes.json").read_text(encoding="utf-8"))
    paragraphs = [{"text": strip_cta(s["narration"]), "emotion": s["emotion"], "key_panel_ids": s["panel_ids"],
                   "action_heavy_ids": s.get("action_heavy_ids", [])} for s in data["scenes"] if not s.get("is_filler")]
    record = {"variant": "A_production", "model": data.get("model", ""), "calls": None, "cost_usd": None, "seconds": None,
              "mode": "deux etapes (sans images)" if (data.get("n_batches") or 0) > 1 else "requete unique (images)",
              "script": {"paragraphs": paragraphs, "characters": []}}
    (out_dir / "A_production.json").write_text(json.dumps(record, indent=1, ensure_ascii=False), encoding="utf-8")
    return record


# --- Juge --------------------------------------------------------------------------------------
class ScriptScore(BaseModel):
    script: str
    hook: int
    rhythm: int
    faithfulness: int
    clarity: int
    voice: int
    overall: int
    factual_errors: list[str]
    best_line: str
    main_weakness: str


class JudgeResult(BaseModel):
    scores: list[ScriptScore]
    ranking: list[str]
    verdict: str


JUDGE_SYSTEM = """You are the head writer of a YouTube channel that retells manhwa chapters as audio stories.
You receive every panel of one chapter in reading order, packed into images (a black bar above each panel
shows "Panel N"), then several candidate scripts written from the same brief. The scripts are anonymous
and in random order. Read the whole chapter first, then every script.

Score each script from 1 (unusable) to 10 (publish as is), strictly, on:
- hook: would a newcomer keep listening after the first thirty seconds?
- rhythm: tension carried from paragraph to paragraph, quiet parts fast and big moments slow, varied
  sentence lengths, paragraph endings that pull forward;
- faithfulness: events, order, names and quoted lines match the panels. List every factual error,
  invented event, wrong name or misattributed line in "factual_errors" (empty list if none);
- clarity: can a newcomer follow who is who and how this world works?
- voice: a storyteller inside the story, plain strong words, no fourth wall, no purple prose;
- overall: how close to publishable, all things considered (not an average).

"best_line": the single best sentence of the script, quoted. "main_weakness": one sentence.
"ranking": the scripts from best to worst. "verdict": two or three sentences on what separates them.
Judge only what is written; length is not a quality by itself.

The brief the writers received:
=== BRIEF START ===
{brief}
=== BRIEF END ==="""


def judge(chapter: str, system: str, text: str, images, records: list[dict], out_dir: Path) -> dict:
    """Notes à l'aveugle des scripts ``records`` par un juge Opus qui voit le chapitre."""
    order = sorted(records, key=lambda r: hashlib.sha256(("judge" + chapter + r["variant"]).encode()).hexdigest())
    labels = {f"Script {chr(65 + k)}": rec["variant"] for k, rec in enumerate(order)}
    scripts = "\n\n".join(
        f"=== {label} ===\n" + "\n\n".join(p["text"] for p in rec["script"]["paragraphs"])
        for label, rec in zip(labels, order))
    client = ClaudeCliJson(model=JUDGE_MODEL, retries=1, timeout_s=1500)
    started = time.perf_counter()
    result = client.ask(JUDGE_SYSTEM.format(brief=system),
                        text + "\n\nThe candidate scripts come first, the chapter's panels follow as images. "
                        "Read the panels, then score the scripts.\n\n" + scripts,
                        images, JudgeResult, label=f"juge {chapter}")
    record = {"labels": labels, "seconds": round(time.perf_counter() - started), "cost_usd": round(client.cost_usd, 3),
              "model": ", ".join(sorted(client.models_used)), "result": result.model_dump()}
    (out_dir / "judge.json").write_text(json.dumps(record, indent=1, ensure_ascii=False), encoding="utf-8")
    return record


# --- Mesures -----------------------------------------------------------------------------------
FOURTH_WALL = re.compile(r"\b(panels?|pages?|chapters?|episodes?|manhwa|manga|webtoons?|comics?|videos?|recaps?|viewers?|"
                         r"readers?|audience|subscrib\w*|we see|is shown|is depicted|in this scene|the camera|close-up|"
                         r"the narrator|the protagonist)\b", re.I)
BANNED = re.compile(r"\b(massive|powerful|overwhelming|incredible|sheer|utter|colossal|epic|little did)\b", re.I)
PERCEPTION = re.compile(r"\b(suddenly|reali[sz](?:e|es|ed)|notic(?:e|es|ed)|observ(?:e|es|ed))\b", re.I)
QUOTE = re.compile(r"[\"“]([^\"”]+)[\"”]")
OPENING = re.compile(r"^\s*(our journey begins|our story begins|our hero|when we last left our hero)", re.I)


def metrics(record: dict, n_panels: int) -> dict:
    paragraphs = record["script"]["paragraphs"]
    full = " ".join(p["text"] for p in paragraphs)
    narration = QUOTE.sub("", full)
    sentences = [s for s in re.split(r"(?<=[.!?])\s+", full) if s.strip()]
    lengths = [len(s.split()) for s in sentences]
    ids = [i for p in paragraphs for i in p["key_panel_ids"]]
    return {
        "mots": len(full.split()), "paragraphes": len(paragraphs),
        "ouverture_ok": bool(OPENING.match(paragraphs[0]["text"])) if paragraphs else False,
        "our_hero": len(re.findall(r"\bour hero\b", full, re.I)),
        "4e_mur": sorted({m.group(0).lower() for m in FOURTH_WALL.finditer(narration)}),
        "mots_bannis": len(BANNED.findall(narration)), "realizes_notices_suddenly": len(PERCEPTION.findall(narration)),
        "citations": len(QUOTE.findall(full)), "questions": narration.count("?"),
        "mots_par_phrase": round(sum(lengths) / max(1, len(lengths)), 1),
        "phrases_courtes_pct": round(100 * sum(1 for n in lengths if n <= 5) / max(1, len(lengths))),
        "cases_cles": len(ids), "cases_invalides": sum(1 for i in ids if not 0 <= i < n_panels),
        "cases_en_double": len(ids) - len(set(ids)),
    }


# --- Page de lecture à l'aveugle ---------------------------------------------------------------
def render(text: str) -> str:
    """Paragraphe HTML, répliques mises en valeur."""
    out, last = [], 0
    for m in QUOTE.finditer(text):
        out += [html.escape(text[last:m.start()]), f"<q>{html.escape(m.group(1))}</q>"]
        last = m.end()
    return "".join(out) + html.escape(text[last:])


def variant_of(labels: dict[str, str], name: str) -> str:
    """Variante d'une étiquette du juge (« Script A », parfois abrégée en « A »)."""
    return labels.get(name) or labels.get(f"Script {name.strip()}") or name


def judge_table(record: dict | None) -> str:
    """Notes du juge (lettres du juge traduites en variantes)."""
    if not record:
        return ""
    labels, result = record["labels"], record["result"]
    rows = "".join(
        f"<tr><td>{html.escape(variant_of(labels, s['script']))}</td>"
        + "".join(f"<td>{s[k]}</td>" for k in ("hook", "rhythm", "faithfulness", "clarity", "voice", "overall"))
        + f"<td>{'<br>'.join(html.escape(e) for e in s['factual_errors']) or '-'}</td>"
        f"<td>{html.escape(s['best_line'])}</td><td>{html.escape(s['main_weakness'])}</td></tr>"
        for s in result["scores"])
    ranking = " &gt; ".join(html.escape(variant_of(labels, r)) for r in result["ranking"])
    return (f"<h3>Juge ({html.escape(record['model'])}, {record['seconds']} s, {record['cost_usd']} $)</h3>"
            f"<p>Classement : {ranking}</p><p>{html.escape(result['verdict'])}</p>"
            "<table><tr><th>variante</th><th>accroche</th><th>rythme</th><th>fidélité</th><th>clarté</th><th>voix</th>"
            f"<th>globale</th><th>erreurs</th><th>meilleure phrase</th><th>faiblesse</th></tr>{rows}</table>")


def write_page(results: dict[str, dict]) -> Path:
    blocks = []
    for chapter, res in results.items():
        records = sorted(res["records"], key=lambda r: hashlib.sha256((chapter + r["variant"]).encode()).hexdigest())
        cols = []
        for k, rec in enumerate(records, 1):
            paras = "".join(f"<p>{render(p['text'])}</p>" for p in rec["script"]["paragraphs"])
            cols.append(f"<section><h3>Script {k}</h3>{paras}</section>")
        key_rows = "".join(
            f"<tr><td>Script {k}</td><td>{html.escape(r['variant'])}</td><td>{html.escape(str(r.get('model') or ''))}"
            f"{' - ' + html.escape(r['mode']) if r.get('mode') else ''}</td><td>{r.get('seconds') or '-'}</td>"
            f"<td>{r.get('cost_usd') if r.get('cost_usd') is not None else '-'}</td>"
            f"<td><code>{html.escape(json.dumps(r['metrics'], ensure_ascii=False))}</code></td></tr>"
            for k, r in enumerate(records, 1))
        blocks.append(
            f"<h2>{html.escape(chapter)}</h2><p class=info>{html.escape(json.dumps(res['info'], ensure_ascii=False))}</p>"
            f"<div class=cols>{''.join(cols)}</div><details><summary>Révéler la clé et les mesures</summary>"
            f"<table><tr><th>script</th><th>variante</th><th>modèle</th><th>s</th><th>$ (tarif API)</th><th>mesures</th></tr>"
            f"{key_rows}</table>{judge_table(res.get('judge'))}</details>")
    page = f"""<!doctype html><html lang="fr"><head><meta charset="utf-8"><title>Banc d'essai du script</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>body{{font-family:Georgia,serif;margin:16px;background:#faf8f4;color:#222}}h1,h2,h3,summary,.info,table{{font-family:system-ui,sans-serif}}
.cols{{display:grid;grid-template-columns:repeat(auto-fit,minmax(320px,1fr));gap:18px}}section{{background:#fff;border:1px solid #ddd;border-radius:8px;padding:4px 16px}}
p{{line-height:1.55}}q{{color:#8a3b12}}.info{{font-size:12px;color:#666}}table{{border-collapse:collapse;font-size:12px;margin:8px 0 28px}}
td,th{{border:1px solid #ccc;padding:4px 6px;vertical-align:top}}code{{white-space:pre-wrap}}</style></head><body>
<h1>Banc d'essai du script - lecture à l'aveugle</h1>
<p>Lis les scripts de chaque chapitre, note ton préféré, puis ouvre la clé. L'ordre des scripts est mélangé.</p>
{''.join(blocks)}</body></html>"""
    path = OUT / "index.html"
    path.write_text(page, encoding="utf-8")
    return path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("chapters", nargs="+", help="dossiers de output/ (ex. the-lazy-lord-masters-the-sword_ep1)")
    parser.add_argument("--models", default="gemini,opus",
                        help="modèles à appeler, séparés par des virgules : gemini, opus, sonnet, haiku ; none : aucun appel")
    parser.add_argument("--judge", action="store_true", help="faire noter les scripts par un juge Opus (un appel par chapitre)")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    results: dict[str, dict] = {}
    for name in args.chapters:
        chapter_dir = ROOT / "output" / name
        out_dir = OUT / name
        out_dir.mkdir(parents=True, exist_ok=True)
        system, text, images, info = build_request(chapter_dir)
        logger.info("%s : %s", name, info)
        # Chapitre jamais passé par le pipeline complet : pas de script de production à comparer.
        records = [production_record(chapter_dir, out_dir)] if (chapter_dir / "scenes.json").is_file() else []
        wanted = [m.strip() for m in args.models.split(",") if m.strip() and m.strip() != "none"]
        wanted = ["opus" if m == "claude" else m for m in wanted]
        unknown = [m for m in wanted if m.partition("@")[0] not in VARIANTS]
        if unknown:
            parser.error(f"modeles inconnus : {unknown}")

        def attempt(model: str) -> dict | None:
            try:
                return run_model(model, system, text, images, out_dir)
            except Exception as exc:  # noqa: BLE001 - un modèle en échec ne bloque pas les autres
                logger.error("%s / %s en echec : %s", name, model, exc)
                return None

        # Les modèles tournent en parallèle : chaque durée reste celle de son propre appel.
        with ThreadPoolExecutor(max_workers=max(1, len(wanted))) as pool:
            fresh = {rec["variant"]: rec for rec in pool.map(attempt, wanted) if rec}
        for model, variant in VARIANTS.items():
            path = out_dir / f"{variant}.json"
            if variant in fresh:
                records.append(fresh[variant])
            elif path.is_file():
                records.append(json.loads(path.read_text(encoding="utf-8")))
        records += [rec for variant, rec in fresh.items() if variant not in VARIANTS.values()]  # modele@effort
        for rec in records:
            rec["metrics"] = metrics(rec, info["panels"])
            logger.info("%s %s %s", name, rec["variant"], rec["metrics"])
        results[name] = {"info": info, "records": records}
        judge_path = out_dir / "judge.json"
        candidates = [r for r in records if r["variant"] != "A_production"]
        if args.judge and len(candidates) > 1:
            try:
                results[name]["judge"] = judge(name, system, text, images, candidates, out_dir)
            except Exception as exc:  # noqa: BLE001 - le juge est un confort
                logger.error("%s / juge en echec : %s", name, exc)
        elif judge_path.is_file():
            results[name]["judge"] = json.loads(judge_path.read_text(encoding="utf-8"))
        if results[name].get("judge"):
            rec = results[name]["judge"]
            for score in rec["result"]["scores"]:
                logger.info("%s juge %s : %s", name, variant_of(rec["labels"], score["script"]), {
                    k: score[k] for k in ("hook", "rhythm", "faithfulness", "clarity", "voice", "overall")})
    logger.info("Page : %s", write_page(results))


if __name__ == "__main__":
    main()
