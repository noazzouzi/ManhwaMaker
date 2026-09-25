"""toonsplit : chaîne complète (split_strip), évaluation et ligne de commande, sans modèle ni réseau."""

from __future__ import annotations

import json

import cv2
import numpy as np
import pytest

from src.modules.toonsplit import __main__ as cli
from src.modules.toonsplit import detectors
from src.modules.toonsplit.ai import BlockSpec, JudgeVerdict, ManualSpecProvider
from src.modules.toonsplit.evaluate import Case, RefCrop, evaluate, find_cases, locate_crops, summarize, write_html
from src.modules.toonsplit.geometry import BUBBLE, HEAD, PERSON, Box
from src.modules.toonsplit.pipeline import Shot, analyze_strip, load_image, save_shots, split_strip
from tests.toonsplit_helpers import art, stack, white

W = 400


def synthetic_strip() -> np.ndarray:
    """Trois blocs : un personnage (0), une ligne de narration seule (1), un bloc avec bulles détachées (2)."""
    caption = white(60)
    cv2.putText(caption, "SOMETIME LATER", (60, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2)
    return stack(white(60), art(700, seed=1), white(100), caption, white(100), art(600, seed=2), white(60))


def fake_detect(block: np.ndarray) -> list[Box]:
    """Détections selon la hauteur du bloc (les blocs synthétiques sont reconnaissables)."""
    h = block.shape[0]
    if h == 700:
        return [Box(150, 100, 250, 180, HEAD, 0.9, "fake"), Box(100, 90, 300, 690, PERSON, 0.8, "fake")]
    if h == 600:
        return [Box(60, 380, 200, 560, BUBBLE, 0.9, "fake"), Box(220, 390, 360, 570, BUBBLE, 0.9, "fake")]
    return [Box(50, 10, 350, 50, BUBBLE, 0.9, "fake")]


SPECS = [
    {"block": 0, "role": "key", "keep": [{"what": "hero", "y": [0.1, 1.0]}], "tts": ["I am the hero."]},
    {"block": 1, "role": "text_only", "tts": ["Sometime later."]},
    {"block": 2, "role": "key", "keep": [{"what": "creatures", "y": [0.0, 0.7]}],
     "drop": [{"what": "bubbles", "y": [0.6, 1.0], "kind": "detached_bubble"}], "tts": ["Ha!", "Ggo!"]},
]


class RecordingJudge:
    def __init__(self, verdict: JudgeVerdict | Exception) -> None:
        self.verdict = verdict
        self.calls: list[tuple] = []

    def __call__(self, block, windows, spec, index, notes=()):
        self.calls.append((index, list(windows), list(notes)))
        if isinstance(self.verdict, Exception):
            raise self.verdict
        return self.verdict


def run(judge=None, mode="auto", provider=None):
    return analyze_strip(
        synthetic_strip(), spec_provider=provider or ManualSpecProvider(SPECS), judge=judge, judge_mode=mode,
        detect=fake_detect,
    )


def test_split_strip_returns_shots_with_the_mission_schema() -> None:
    shots = split_strip(synthetic_strip(), spec_provider=ManualSpecProvider(SPECS), judge=None, detect=fake_detect)
    assert all(isinstance(s, Shot) for s in shots)
    assert set(shots[0].as_dict()) == {"y0", "y1", "role", "tts", "pan", "source_block"}
    roles = [(s.source_block, s.role, s.has_image) for s in shots]
    assert roles == [(0, "key", True), (1, "text_only", False), (2, "key", True)]
    assert shots[1].tts == ("Sometime later.",) and shots[0].tts == ("I am the hero.",)
    hero = shots[0]
    assert 60 <= hero.y0 and hero.y1 <= 760  # dans le bloc 0 (60-760), coordonnées du strip


def test_shots_never_cut_heads_or_bubbles() -> None:
    result = run()
    for b in result.blocks:
        cons = b.plan.constraints
        for s in b.shots:
            if not s.has_image or cons is None:
                continue
            y0, y1 = s.y0 - b.block.y0, s.y1 - b.block.y0
            assert not any(h.y0 < y < h.y1 for h in cons.hard for y in (y0, y1))


def test_auto_judge_is_called_on_conflicts_and_its_choice_applied() -> None:
    judge = RecordingJudge(JudgeVerdict(best=2, reason="bubbles whole", confident=True))
    result = run(judge)
    assert [c[0] for c in judge.calls] == [2]  # seul le bloc en conflit (bulles détachées sur le sujet)
    block = result.blocks[-1]
    assert block.plan.conflict and block.chosen == 1
    assert (block.shots[0].y0 - block.block.y0, block.shots[0].y1 - block.block.y0) == judge.calls[0][1][1]
    assert judge.calls[0][2][0] == "follows the block description"


def test_uncertain_or_failing_judge_keeps_candidate_one() -> None:
    unsure = run(RecordingJudge(JudgeVerdict(best=2, reason="?", confident=False)))
    assert unsure.blocks[-1].chosen == 0
    broken = run(RecordingJudge(RuntimeError("quota")))
    assert broken.blocks[-1].chosen == 0 and broken.blocks[-1].verdict is None


def test_judge_modes() -> None:
    never = RecordingJudge(JudgeVerdict(best=1, reason=""))
    run(never, mode="never")
    assert never.calls == []
    always = RecordingJudge(JudgeVerdict(best=1, reason=""))
    run(always, mode="always")
    assert len(always.calls) >= 1


def test_spec_failure_falls_back_to_detections() -> None:
    def failing(block, index):
        raise RuntimeError("quota epuise")

    result = run(provider=failing)
    assert {b.spec_source for b in result.blocks} == {"fallback"}
    assert all(s.role == "key" for s in result.shots)


def test_save_shots_writes_native_resolution_png(tmp_path) -> None:
    img = synthetic_strip()
    result = run()
    paths = save_shots(img, result.shots, tmp_path)
    assert len(paths) == 2  # le bloc texte seul n'a pas d'image
    first = cv2.imread(str(paths[0]))
    shot = result.shots[0]
    assert first.shape == (shot.y1 - shot.y0, W, 3)


def test_load_image_handles_non_ascii_path_and_alpha(tmp_path) -> None:
    rgba = np.zeros((10, 20, 4), np.uint8)
    rgba[..., 3] = 0
    path = tmp_path / "planche_é.png"
    path.write_bytes(cv2.imencode(".png", rgba)[1].tobytes())
    img = load_image(path)
    assert img.shape == (10, 20, 3) and img.min() == 255


# --- Évaluation -------------------------------------------------------------------------------------
def test_locate_crops_finds_rescaled_hand_crops() -> None:
    strip = stack(art(500, seed=3), art(700, seed=4), art(400, seed=5))
    crop = cv2.resize(strip[520:1100], (300, round(580 * 300 / W)), interpolation=cv2.INTER_AREA)
    (found,) = locate_crops(strip, [("c", crop)])
    assert abs(found.y0 - 520) <= 3 and abs(found.y1 - 1100) <= 3 and found.match > 0.9


def test_evaluate_scores_iou_and_violations() -> None:
    result = run()
    hero_block = result.blocks[0].block
    ref = RefCrop(hero_block.y0 + 50, hero_block.y1, "hero")
    head_cut = RefCrop(hero_block.y0 + 140, hero_block.y1, "tete coupee")
    case = Case("synthetic", None, None, [ref, head_cut])  # type: ignore[arg-type]
    ev = evaluate(case, result)
    assert ev.refs[0].block == 0 and ev.refs[0].iou > 0.8
    assert ev.refs[1].report["head_cuts"] == 1
    summary = summarize([ev])
    assert summary["hard"] == {"head_cuts": 0, "bubble_cuts": 0}
    assert summary["reference_violations"]["head_cuts"] == 1
    external = evaluate(case, result, [(0, hero_block.y0 + 140, hero_block.y1, False)])
    assert summarize([external])["hard"]["head_cuts"] == 1


def test_find_cases_locates_crops_and_writes_reference(tmp_path) -> None:
    strip = synthetic_strip()
    case_dir = tmp_path / "serie" / "ch01"
    (case_dir / "crops").mkdir(parents=True)
    cv2.imwrite(str(case_dir / "strip.png"), strip)
    cv2.imwrite(str(case_dir / "crops" / "a.png"), strip[100:700])
    (case,) = find_cases(tmp_path)
    assert case.name == "ch01" and (case_dir / "reference.json").is_file()
    assert abs(case.refs[0].y0 - 100) <= 2


def test_html_sheet_is_written(tmp_path) -> None:
    strip_path = tmp_path / "strip.png"
    cv2.imwrite(str(strip_path), synthetic_strip())
    result = analyze_strip(strip_path, spec_provider=ManualSpecProvider(SPECS), judge=None, detect=fake_detect)
    case = Case("c", tmp_path, strip_path, [RefCrop(120, 700, "hero")])
    ev = evaluate(case, result)
    page = write_html(tmp_path / "out", [(case, result, ev)], "test", summarize([ev]))
    text = page.read_text(encoding="utf-8")
    assert "IoU" in text and "img/c_b00_view.jpg" in text
    assert (tmp_path / "out" / "img" / "c_b00_view.jpg").is_file()


def test_cli_split_and_eval_offline(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(detectors, "detect_all", fake_detect)
    case_dir = tmp_path / "001"
    case_dir.mkdir()
    cv2.imwrite(str(case_dir / "strip.png"), synthetic_strip())
    (case_dir / "spec_manual.json").write_text(json.dumps(SPECS), encoding="utf-8")
    (case_dir / "reference.json").write_text(json.dumps({"crops": [{"y0": 120, "y1": 700}]}), encoding="utf-8")
    out = tmp_path / "split"
    assert cli.main(["split", str(case_dir / "strip.png"), "--spec", "manual", "--judge", "never", "--out", str(out)]) == 0
    shots = json.loads((out / "shots.json").read_text(encoding="utf-8"))
    assert [s["role"] for s in shots] == ["key", "text_only", "key"]
    assert len(list((out / "shots").glob("*.png"))) == 2 and (out / "debug" / "block_00.jpg").is_file()
    assert cli.main(["eval", str(tmp_path), "--spec", "manual", "--judge", "never", "--out", str(tmp_path / "ev"),
                     "--param", "w_ratio=0"]) == 0
    metrics = json.loads((tmp_path / "ev" / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["summary"]["references"] == 1 and metrics["summary"]["config"]["params"] == ["w_ratio=0"]
    with pytest.raises(SystemExit):
        cli.main(["eval", str(tmp_path), "--spec", "manual", "--param", "nope=1"])
