"""Temps restant d'un traitement : durées mesurées + simulation de l'ordonnancement du lot.

Un lot fait passer chaque chapitre par quatre étapes (:data:`STAGES`), chacune limitée en
parallèle comme dans :mod:`src.modules.batch_processor` (3 préparations, 3 scripts, 2 voix,
1 montage) ; le script d'un chapitre attend en plus celui du chapitre précédent de la série.
Le temps restant n'est donc pas une somme : il se calcule en **rejouant** l'ordonnancement
(:func:`simulate`) avec, pour chaque étape, sa durée attendue.

Durées attendues : médianes des chapitres déjà faits (``timings`` de ``batch_status.json``),
multipliées par un facteur de calage (:attr:`StageModel.calibration`) qui corrige ce que la
simulation ignore (processeur partagé entre étapes). Pour l'étape en cours, l'avancement
réel prend le relais (:func:`running_remaining`) : 11 scènes sur 24 en 3 min, c'est encore
environ 3 min 30.
"""

from __future__ import annotations

import heapq
import json
import statistics
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

STAGES: tuple[str, ...] = ("prep", "script", "voice", "montage")

#: Sous-étapes de chaque étape et clés de mesure correspondantes (``PipelineResult.timings``).
SUBSTEPS: dict[str, tuple[tuple[str, str], ...]] = {
    "prep": (("download", "scrape+slice"), ("figures", "figures"), ("upscale", "upscale")),
    "script": (("script", "analyze"),),
    "voice": (("voice", "tts"),),
    "montage": (("timeline", "timeline"), ("capcut", "capcut"), ("preview", "preview")),
}
#: Identifiant de sous-étape rapporté par le pipeline -> sous-étape de :data:`SUBSTEPS`.
STEP_ALIASES = {"slice": "download"}

#: Durées par défaut (s), mesurées sur The Genius Professor ch. 1-20 et Bad Born Blood ch. 1
#: (Claude Opus 5.5, cases personnages agrandies, 3 chapitres en parallèle).
DEFAULT_TIMINGS: dict[str, float] = {
    "scrape+slice": 23.0, "figures": 142.0, "upscale": 3.0, "analyze": 76.0, "tts": 139.0,
    "timeline": 0.5, "capcut": 6.0, "preview": 45.0,
}
DEFAULT_PARALLEL = {"prep": 3, "script": 3, "voice": 2, "montage": 1}
#: Compilation (assemblage, brouillon CapCut, extrait de 2 min, suppression des dossiers).
DEFAULT_COMPILE_S = 90.0
#: Rendu Kdenlive : secondes de calcul par seconde de vidéo (mesuré le 26/09 sur 5 min 51,
#: style dynamique). Le processeur (x264) est estimé depuis le rendu de Kdenlive lui-même.
DEFAULT_RENDER_RATIO = {("h264_amf", 30): 0.47, ("h264_amf", 60): 0.97, ("libx264", 30): 0.73, ("libx264", 60): 1.5}
#: Préparation du projet Kdenlive (images des cases, calques) par seconde de vidéo.
DEFAULT_PROJECT_RATIO = 0.05
#: Démarrage de melt (chargement du projet) avant la première image, en secondes.
RENDER_OVERHEAD_S = 5.0
#: Rendu par parties (au-delà de 12 min) : mixage du son et assemblage final, par seconde de vidéo.
AUDIO_MIX_RATIO, MUX_RATIO = 0.025, 0.01  # mesuré : ~28 s pour 13 min de vidéo
#: Même seuil que :data:`src.modules.kdenlive_builder.PARTS_FROM_S` (non importé : il tire numpy, PIL...).
PARTS_FROM_S = 720.0
#: Un rendu plus court ne sert pas à caler les estimations : le démarrage y pèse trop.
MIN_LEARN_VIDEO_S = 60.0
#: Temps restant minimal affiché pour une étape en cours.
MIN_REMAINING_S = 5.0
#: Au-delà de la durée attendue, on annonce encore ce fraction du temps déjà passé.
OVERRUN_SHARE = 0.2
#: Nombre de chapitres récents retenus pour les médianes.
HISTORY_WINDOW = 60


@dataclass
class StageModel:
    """Durées attendues des sous-étapes et facteur de calage."""

    timings: dict[str, float] = field(default_factory=lambda: dict(DEFAULT_TIMINGS))
    samples: int = 0
    calibration: float = 1.0
    compile_s: float = DEFAULT_COMPILE_S
    render_ratio: dict[tuple[str, int], float] = field(default_factory=lambda: dict(DEFAULT_RENDER_RATIO))
    project_ratio: float = DEFAULT_PROJECT_RATIO

    @classmethod
    def from_history(cls, status_file: str | Path | None, studio_history: Mapping[str, Any] | None = None) -> StageModel:
        """Modèle calé sur ``batch_status.json`` et l'historique du Studio (compilations, rendus)."""
        model = cls()
        data: dict[str, Any] = {}
        if status_file is not None and Path(status_file).is_file():
            try:
                data = json.loads(Path(status_file).read_text(encoding="utf-8"))
            except (OSError, ValueError):
                data = {}
        entries = [e for e in (data.get("chapters") or {}).values() if e.get("status") == "done" and e.get("timings")]
        entries = [e for e in entries if len(e.get("reused") or []) < 2]
        entries.sort(key=lambda e: str(e.get("finished") or ""))
        entries = entries[-HISTORY_WINDOW:]
        values: dict[str, list[float]] = {}
        for entry in entries:
            for key, seconds in entry["timings"].items():
                if seconds > 0.05:
                    values.setdefault(key, []).append(float(seconds))
        for key, series in values.items():
            model.timings[key] = statistics.median(series)
        model.samples = len(entries)
        model.calibration = _calibration(data.get("run") or {}, model)
        history = studio_history or {}
        compiles = [float(s) for s in history.get("compile_s", []) if s > 0]
        if compiles:
            model.compile_s = statistics.median(compiles[-10:])
        for sample in history.get("renders", [])[-20:]:
            if float(sample.get("video_s") or 0.0) < MIN_LEARN_VIDEO_S:
                continue
            key = (sample.get("codec"), int(sample.get("fps") or 0))
            if sample.get("video_s") and sample.get("melt_s"):
                model.render_ratio[key] = float(sample["melt_s"]) / float(sample["video_s"])
            if sample.get("video_s") and sample.get("project_s"):
                model.project_ratio = float(sample["project_s"]) / float(sample["video_s"])
        return model

    def expected(self, timing_key: str) -> float:
        return self.timings.get(timing_key, DEFAULT_TIMINGS.get(timing_key, 0.0)) * self.calibration

    def substeps(self, stage: str, config: Mapping[str, Any]) -> list[tuple[str, float]]:
        """Sous-étapes actives de ``stage`` (selon les réglages du lot) et leur durée attendue.

        ``config["scale"]`` corrige une étape d'après ce que le lot en cours a déjà mesuré
        (:func:`job_scale`).
        """
        montage_steps = set(config.get("montage_steps") or ("timeline",))
        factor = float((config.get("scale") or {}).get(stage, 1.0))
        out = []
        for step_id, key in SUBSTEPS[stage]:
            if step_id in ("figures", "upscale") and not config.get(step_id, True):
                continue
            if stage == "montage" and step_id not in montage_steps:
                continue
            out.append((step_id, self.expected(key) * factor))
        return out

    def stage_expected(self, stage: str, config: Mapping[str, Any]) -> float:
        return sum(seconds for _, seconds in self.substeps(stage, config))


def _calibration(run: Mapping[str, Any], model: StageModel) -> float:
    """Rapport durée réelle / durée simulée du dernier lot mesuré (borné entre 0,8 et 2)."""
    n = int(run.get("done") or 0)
    elapsed = float(run.get("elapsed_s") or 0.0)
    steps = (run.get("timings") or {}).get("steps") or {}
    if n < 3 or elapsed <= 0 or not steps:
        return 1.0
    medians = {key: float(v.get("median_s") or 0.0) for key, v in steps.items()}
    par = run.get("parallelism") or {}
    capacity = {
        "prep": int(par.get("max_scrape_workers") or par.get("max_chapters") or 3), "script": int(par.get("max_chapters") or 3),
        "voice": int(par.get("max_tts_workers") or 2), "montage": int(par.get("max_render_workers") or 1),
    }
    per_stage = {
        "prep": sum(medians.get(k, 0.0) for k in ("scrape+slice", "figures", "upscale")),
        "script": medians.get("analyze", 0.0), "voice": medians.get("tts", 0.0),
        "montage": sum(medians.get(k, 0.0) for k in ("timeline", "capcut", "preview")),
    }
    chapters = [
        SimChapter(str(i), [(s, per_stage[s]) for s in STAGES], after=str(i - 1) if i else None) for i in range(n)
    ]
    predicted = max(simulate(chapters, capacity).values(), default=0.0)
    if predicted <= 0:
        return 1.0
    return min(2.0, max(0.8, elapsed / predicted))


# --- Simulation --------------------------------------------------------------------------
@dataclass
class SimChapter:
    """Chapitre à rejouer : étapes restantes (la première peut être déjà en cours)."""

    key: str
    remaining: list[tuple[str, float]]
    running: bool = False
    #: Chapitre dont le script doit être fini avant celui-ci (ordre de la série).
    after: str | None = None


def simulate(chapters: Sequence[SimChapter], capacity: Mapping[str, int]) -> dict[str, float]:
    """Heure de fin (s, depuis maintenant) de chaque chapitre, selon les places de chaque étape.

    Files d'attente dans l'ordre des chapitres, comme les sémaphores du lot ; le script d'un
    chapitre ne part qu'une fois celui de son prédécesseur terminé (ou si le prédécesseur
    n'est pas à rejouer : déjà fait, en échec, ignoré).
    """
    order = {c.key: i for i, c in enumerate(chapters)}
    by_key = {c.key: c for c in chapters}
    free = {stage: max(1, int(capacity.get(stage, 1))) for stage in STAGES}
    queues: dict[str, list[str]] = {stage: [] for stage in STAGES}
    position: dict[str, int] = {}
    script_done: dict[str, float] = {}
    finish: dict[str, float] = {}
    events: list[tuple[float, int, str]] = []

    for chapter in chapters:
        position[chapter.key] = 0
        if not chapter.remaining:
            finish[chapter.key] = 0.0
            script_done[chapter.key] = 0.0
            continue
        if "script" not in [stage for stage, _ in chapter.remaining]:
            script_done[chapter.key] = 0.0
        stage, seconds = chapter.remaining[0]
        if chapter.running:
            free[stage] -= 1
            heapq.heappush(events, (max(0.0, seconds), order[chapter.key], chapter.key))
        else:
            queues[stage].append(chapter.key)

    def blocked(key: str, now: float) -> bool:
        before = by_key[key].after
        if before is None or before not in by_key:
            return False
        done_at = script_done.get(before)
        return done_at is None or done_at > now

    def start_ready(now: float) -> None:
        for stage in STAGES:
            queue = queues[stage]
            queue.sort(key=order.__getitem__)
            i = 0
            while free[stage] > 0 and i < len(queue):
                key = queue[i]
                if stage == "script" and blocked(key, now):
                    i += 1
                    continue
                queue.pop(i)
                free[stage] -= 1
                seconds = by_key[key].remaining[position[key]][1]
                heapq.heappush(events, (now + max(0.0, seconds), order[key], key))

    now = 0.0
    while True:
        start_ready(now)
        if not events:
            stuck = [key for queue in queues.values() for key in queue]
            if not stuck:
                break
            # Prédécesseur jamais fini (ne devrait pas arriver) : on lève la barrière.
            for key in stuck:
                by_key[key].after = None
            continue
        now, _, key = heapq.heappop(events)
        chapter = by_key[key]
        stage = chapter.remaining[position[key]][0]
        free[stage] += 1
        if stage == "script":
            script_done[key] = now
        position[key] += 1
        if position[key] < len(chapter.remaining):
            queues[chapter.remaining[position[key]][0]].append(key)
        else:
            finish[key] = now
    return finish


# --- Étape en cours ---------------------------------------------------------------------
def substep_remaining(expected: float, step: Mapping[str, Any] | None, now: float) -> tuple[float, bool]:
    """Reste d'une sous-étape : avancement réel s'il est connu, durée attendue sinon.

    Renvoie ``(secondes, dépassement)`` ; ``dépassement`` est vrai quand la sous-étape a
    déjà duré plus que prévu (le reste annoncé devient une part du temps déjà passé).
    """
    step = step or {}
    elapsed = max(0.0, now - float(step.get("since") or now))
    done, total = step.get("done"), step.get("total")
    by_plan = expected - elapsed
    overrun = by_plan < 0
    by_plan = max(by_plan, elapsed * OVERRUN_SHARE)
    if total and done and total > 0 and done > 0:
        fraction = min(1.0, done / total)
        by_rate = elapsed * (1.0 - fraction) / fraction
        weight = min(1.0, fraction * 4.0)  # la vitesse mesurée l’emporte dès 25 % de fait
        return weight * by_rate + (1.0 - weight) * by_plan, False
    return by_plan, overrun


def running_remaining(model: StageModel, stage: str, entry: Mapping[str, Any], config: Mapping[str, Any], now: float) -> tuple[float, bool]:
    """Reste de l'étape en cours d'un chapitre (sous-étape courante + suivantes)."""
    substeps = model.substeps(stage, config)
    step = entry.get("step") or {}
    step_id = STEP_ALIASES.get(str(step.get("id")), step.get("id"))
    ids = [sid for sid, _ in substeps]
    if step_id not in ids:
        started = float(entry.get("since") or now)
        return substep_remaining(sum(s for _, s in substeps), {"since": started}, now)
    index = ids.index(step_id)
    current, overrun = substep_remaining(substeps[index][1], step, now)
    later = sum(seconds for _, seconds in substeps[index + 1:])
    return max(MIN_REMAINING_S, current + later), overrun


# --- Lot ---------------------------------------------------------------------------------
#: Poids de l'historique face aux mesures du lot en cours (en nombre de chapitres).
PRIOR_WEIGHT = 1.5


def job_scale(model: StageModel, items: Mapping[str, Any], config: Mapping[str, Any]) -> dict[str, float]:
    """Correction de chaque étape d'après les chapitres du lot en cours qui l'ont finie.

    Un lot dont les chapitres sont déjà téléchargés, ou dont l'IA répond lentement ce jour-là,
    s'écarte de l'historique : chaque étape terminée rapproche l'estimation de la réalité
    (moyenne pondérée : 1 chapitre mesuré pèse 40 %, 3 en pèsent 67 %).
    """
    scale: dict[str, float] = {}
    for stage in STAGES:
        durations = []
        for entry in items.values():
            span = (entry.get("stages") or {}).get(stage) or {}
            failed_here = entry.get("status") == "failed" and entry.get("stage") == stage
            if span.get("start") is not None and span.get("end") is not None and not failed_here:
                durations.append(max(0.0, float(span["end"]) - float(span["start"])))
        expected = model.stage_expected(stage, {**config, "scale": {}})
        if not durations or expected <= 0:
            continue
        observed = statistics.median(durations)
        blended = (len(durations) * observed + PRIOR_WEIGHT * expected) / (len(durations) + PRIOR_WEIGHT)
        scale[stage] = max(0.05, blended / expected)
    return scale


def batch_estimate(
    model: StageModel, prog: Mapping[str, Any], now: float | None = None, *, compile_video: bool | None = None,
    planned_chapters: int = 0,
) -> dict[str, Any]:
    """Temps restant d'un lot en cours, et de chacun de ses chapitres.

    Args:
        prog: contenu du fichier de progression (:mod:`src.utils.progress`).
        compile_video: le lot se termine par une compilation (sinon lu dans ``prog``).
        planned_chapters: nombre de chapitres prévus, tant que la liste n'est pas connue.
    """
    now = time.time() if now is None else now
    compile_video = bool(prog.get("compile")) if compile_video is None else compile_video
    phase = prog.get("phase") or "starting"
    config = {
        "montage_steps": prog.get("montage_steps") or (["timeline"] if compile_video else ["timeline", "capcut", "preview"]),
        "figures": prog.get("figures", True), "upscale": prog.get("upscale", True),
    }
    items: dict[str, Any] = {k: v for k, v in (prog.get("items") or {}).items() if k != "__run__"}
    capacity = {**DEFAULT_PARALLEL, **(prog.get("parallel") or {})}
    series_order = prog.get("series_order", True)
    predecessors = prog.get("predecessors") or {}
    result: dict[str, Any] = {"phase": phase, "chapters": {}, "overrun": False}
    config["scale"] = job_scale(model, items, config)
    result["scale"] = config["scale"]

    compile_left = model.compile_s if compile_video else 0.0
    if phase in ("done", "failed"):
        result.update(remaining_s=0.0)
        return result
    if phase == "compile":
        run_item = (prog.get("items") or {}).get("__run__") or {}
        step = run_item.get("step") or {}
        since = float(prog.get("phase_since") or now)
        if step.get("id") == "preview" and step.get("total"):
            left, _ = substep_remaining(model.compile_s * 0.7, step, now)
            left += model.compile_s * 0.1
        else:
            left = max(model.compile_s - (now - since), (now - since) * OVERRUN_SHARE)
        result.update(remaining_s=max(MIN_REMAINING_S, left))
        return result
    if not items:
        n = max(1, planned_chapters)
        chapters = [SimChapter(str(i), [(s, model.stage_expected(s, config)) for s in STAGES],
                               after=str(i - 1) if i and series_order else None) for i in range(n)]
        finish = simulate(chapters, capacity)
        result.update(remaining_s=max(finish.values(), default=0.0) + compile_left + (15.0 if phase in ("starting", "resolve") else 0.0))
        return result

    sims: list[SimChapter] = []
    ordered = sorted(items.items(), key=lambda kv: kv[1].get("order", 0))
    for key, entry in ordered:
        status = entry.get("status")
        if status in ("done", "failed", "skipped"):
            continue
        current = entry.get("stage")
        state = entry.get("state")
        remaining: list[tuple[str, float]] = []
        running = False
        start_index = STAGES.index(current) if current in STAGES else 0
        for stage in STAGES[start_index:]:
            if stage == current and state == "running":
                seconds, overrun = running_remaining(model, stage, entry, config, now)
                result["overrun"] = result["overrun"] or overrun
                remaining.append((stage, seconds))
                running = True
            else:
                remaining.append((stage, model.stage_expected(stage, config)))
        after = predecessors.get(key) if series_order else None
        sims.append(SimChapter(key, remaining, running=running, after=after))
    finish = simulate(sims, capacity)
    for key, seconds in finish.items():
        result["chapters"][key] = seconds
    result.update(remaining_s=max(finish.values(), default=0.0) + compile_left)
    return result


def batch_plan(model: StageModel, n_chapters: int, *, compile_video: bool, make_preview: bool = True,
               series_order: bool = True, capacity: Mapping[str, int] | None = None) -> float:
    """Durée prévue d'un lot pas encore lancé (écran « Nouvelle vidéo »)."""
    montage_steps = ["timeline"] if compile_video else ["timeline", "capcut"] + (["preview"] if make_preview else [])
    config = {"montage_steps": montage_steps, "figures": True, "upscale": True}
    chapters = [SimChapter(str(i), [(s, model.stage_expected(s, config)) for s in STAGES],
                           after=str(i - 1) if i and series_order else None) for i in range(max(0, n_chapters))]
    finish = simulate(chapters, {**DEFAULT_PARALLEL, **(capacity or {})})
    return max(finish.values(), default=0.0) + (model.compile_s if compile_video and n_chapters else 0.0)


# --- Rendu Kdenlive ------------------------------------------------------------------------
def render_plan(model: StageModel, video_s: float, fps: int, codec: str) -> float:
    """Durée prévue d'un rendu Kdenlive (projet + melt)."""
    ratio = model.render_ratio.get((codec, fps), DEFAULT_RENDER_RATIO.get((codec, fps), 1.0))
    finish = video_s * (AUDIO_MIX_RATIO + MUX_RATIO) if video_s > PARTS_FROM_S else 0.0
    return video_s * (model.project_ratio + ratio) + RENDER_OVERHEAD_S + finish


def render_estimate(model: StageModel, prog: Mapping[str, Any], now: float | None = None, *, video_s: float = 0.0,
                    fps: int = 30, codec: str = "h264_amf") -> dict[str, Any]:
    """Temps restant d'un rendu en cours."""
    now = time.time() if now is None else now
    phase = prog.get("phase") or "starting"
    video_s = float(prog.get("video_s") or video_s)
    fps, codec = int(prog.get("fps") or fps), str(prog.get("codec") or codec)
    ratio = model.render_ratio.get((codec, fps), DEFAULT_RENDER_RATIO.get((codec, fps), 1.0))
    melt_expected = video_s * ratio + RENDER_OVERHEAD_S
    if phase in ("done", "failed"):
        return {"phase": phase, "remaining_s": 0.0, "overrun": False}
    # Rendu par parties : le son est mixé puis tout est assemblé après la dernière partie.
    finish_s = video_s * (AUDIO_MIX_RATIO + MUX_RATIO) if int(prog.get("parts") or 1) > 1 else 0.0
    if phase == "render":
        step = ((prog.get("items") or {}).get("__run__") or {}).get("step") or {}
        if step.get("id") == "melt":
            left, overrun = substep_remaining(melt_expected, step, now)
            left += finish_s
        elif step.get("id") == "audio":
            left, overrun = substep_remaining(video_s * AUDIO_MIX_RATIO, step, now)
            left += video_s * MUX_RATIO
        elif step.get("id") == "mux":
            left, overrun = substep_remaining(video_s * MUX_RATIO, step, now)
        else:
            left, overrun = melt_expected + finish_s, False
        return {"phase": phase, "remaining_s": max(MIN_REMAINING_S, left), "overrun": overrun}
    project_expected = video_s * model.project_ratio
    since = float(prog.get("phase_since") or prog.get("started") or now)
    left, overrun = substep_remaining(project_expected, {"since": since}, now)
    return {"phase": phase, "remaining_s": max(MIN_REMAINING_S, left + melt_expected), "overrun": overrun}


__all__ = [
    "STAGES", "SUBSTEPS", "DEFAULT_TIMINGS", "DEFAULT_PARALLEL", "StageModel", "SimChapter", "simulate",
    "substep_remaining", "running_remaining", "batch_estimate", "batch_plan", "render_plan", "render_estimate",
]
