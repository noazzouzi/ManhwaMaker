"""toonsplit, étape 4 : contraintes, recherche des fenêtres, cas limites."""

from __future__ import annotations

import math

import numpy as np
import pytest

from src.modules.toonsplit.geometry import BUBBLE, FREE_TEXT, HEAD, NARRATION, PERSON, SFX, Box
from src.modules.toonsplit.search import (
    SearchParams, build_constraints, height_bounds, plan_block, search_crops, split_tall, window_report,
)
from tests.toonsplit_helpers import box, spec

W = 400
P = SearchParams()
HMIN, HMAX = math.ceil(W / 1.35), math.floor(W / 0.55)  # 297, 727


def flat_energy(h: int) -> np.ndarray:
    return np.full(h, 0.1)


def crosses(b: Box, y: int) -> bool:
    return b.y0 < y < b.y1


# --- Contraintes -------------------------------------------------------------------------------
def test_keep_zone_snaps_to_touching_heads_and_attached_bubbles() -> None:
    boxes = [box(180, 260, HEAD), box(90, 205, BUBBLE), box(700, 760, BUBBLE)]
    cons = build_constraints(spec(keep=[(0.2, 0.6)]), boxes, 1000, W)
    # 200-600 → tête (élargie de 8 %) et bulle attachée qui débordent au-dessus ; la bulle du bas ne touche pas
    assert cons.keep == (90, 600)
    head = cons.heads[0]
    assert (head.y0, head.y1) == (174, 266)
    assert cons.heads_raw[0].y0 == 180


def test_bubbles_in_drop_zones_are_not_snapped_and_raise_a_conflict() -> None:
    boxes = [box(560, 700, BUBBLE)]
    s = spec(keep=[(0.0, 0.6)], drop=[(0.6, 1.0, "detached_bubble")])
    cons = build_constraints(s, boxes, 1000, W)
    assert cons.keep == (0, 600)
    assert [(b.y0, b.y1) for b in cons.conflicts] == [(560, 700)]


def test_free_text_becomes_narration_or_sfx_from_the_spec() -> None:
    boxes = [box(10, 90, FREE_TEXT), box(300, 420, FREE_TEXT)]
    s = spec(keep=[(0.2, 1.0)], drop=[(0.0, 0.1, "narration"), (0.25, 0.45, "sfx")])
    cons = build_constraints(s, boxes, 1000, W)
    assert [(b.kind, b.y0) for b in cons.narration] == [(NARRATION, 10)]
    assert [(b.kind, b.y0) for b in cons.sfx] == [(SFX, 300)]
    assert any(b.kind == NARRATION for b in cons.hard)


def test_sfx_zone_missed_by_detectors_becomes_all_or_nothing() -> None:
    s = spec(keep=[(0.2, 1.0)], drop=[(0.25, 0.45, "sfx"), (0.0, 0.1, "sfx")])
    cons = build_constraints(s, [box(0, 90, FREE_TEXT)], 1000, W)
    zones = [(b.y0, b.y1, b.source) for b in cons.sfx]
    assert (250, 450, "spec-zone") in zones  # zone IA sans détection : boîte synthétique
    assert sum(src == "spec-zone" for *_, src in zones) == 1  # l'autre zone a sa détection


def test_cut_subject_policy_never_trims_above_a_face() -> None:
    boxes = [box(100, 180, HEAD), box(520, 700, BUBBLE)]
    s = spec(keep=[(0.1, 0.6)], drop=[(0.55, 0.75, "detached_bubble")])
    cut = build_constraints(s, boxes, 1000, W, policy="cut_subject")
    assert cut.keep == (94, 520)  # tête 100-180 élargie de 8 % : 94
    face_low = [box(480, 560, HEAD), box(520, 700, BUBBLE)]
    assert build_constraints(s, face_low, 1000, W, policy="cut_subject") is None


def test_unknown_policy_is_rejected() -> None:
    with pytest.raises(ValueError):
        build_constraints(spec(keep=[(0, 1)]), [], 500, W, policy="nope")


# --- Recherche ---------------------------------------------------------------------------------------
def test_search_never_cuts_hard_boxes_and_covers_keep() -> None:
    rng = np.random.default_rng(0)
    for trial in range(25):
        H = int(rng.integers(400, 1400))
        boxes = []
        for _ in range(int(rng.integers(1, 6))):
            y0 = int(rng.integers(0, H - 60))
            boxes.append(box(y0, min(H, y0 + int(rng.integers(30, 160))), str(rng.choice([HEAD, BUBBLE]))))
        a = float(rng.uniform(0.0, 0.5))
        s = spec(keep=[(round(a, 2), round(min(1.0, a + rng.uniform(0.2, 0.5)), 2))])
        cons = build_constraints(s, boxes, H, W)
        for c in search_crops(cons, flat_energy(H), P):
            assert not any(crosses(b, c.y0) or crosses(b, c.y1) for b in cons.hard), (trial, c)
            assert c.cover >= 0.98 and 0 <= c.y0 < c.y1 <= H
            lo, hi = height_bounds(W, H, P)
            assert lo <= c.h <= hi


def test_search_is_deterministic_and_diverse() -> None:
    boxes = [box(300, 380, HEAD), box(120, 220, BUBBLE)]
    cons = build_constraints(spec(keep=[(0.25, 0.7)]), boxes, 1200, W)
    energy = np.random.default_rng(1).random(1200)
    first = search_crops(cons, energy, P)
    assert first == search_crops(cons, energy, P)
    assert 1 <= len(first) <= 3
    gap = round(40 * W / 575)
    for i, a in enumerate(first):
        for b in first[i + 1:]:
            assert abs(a.y0 - b.y0) > gap or abs(a.y1 - b.y1) > gap


def test_search_can_reach_the_block_bottom_exactly() -> None:
    cons = build_constraints(spec(keep=[(0.8, 1.0)]), [], 1003, W)
    best = search_crops(cons, flat_energy(1003), P)[0]
    assert best.y1 == 1003


def test_narration_drop_zone_is_left_out_when_possible() -> None:
    boxes = [box(20, 150, BUBBLE)]
    s = spec(keep=[(0.2, 0.9)], drop=[(0.0, 0.17, "narration")])
    cons = build_constraints(s, boxes, 1000, W)
    best = search_crops(cons, flat_energy(1000), P)[0]
    assert best.y0 >= 150
    assert window_report(best.y0, best.y1, cons)["narration_included"] == 0


def test_sfx_is_kept_whole_rather_than_sliced_when_it_cannot_be_excluded() -> None:
    # Cas du drapeau : onomatopée 0-400 qui déborde sur le sujet (250-1460), bloc de 800 px de large.
    boxes = [box(0, 400, FREE_TEXT)]
    s = spec(role="insert", keep=[(0.17, 1.0)], drop=[(0.0, 0.35, "sfx"), (0.0, 0.17, "background")])
    plan = plan_block((1460, 800), flat_energy(1460), boxes, s, P)
    policies = [c.policy for c in plan.candidates]
    if plan.candidates[0].terms["sfx_cut"] > 0:
        assert "sfx_whole" in policies and plan.conflict
    whole = [c for c in plan.candidates if c.terms["sfx_cut"] == 0]
    assert whole, "au moins un candidat garde l'onomatopée entière"


def test_short_block_gives_whole_block() -> None:
    cons = build_constraints(spec(keep=[(0.1, 0.9)]), [], 200, W)
    (best,) = search_crops(cons, flat_energy(200), P)
    assert (best.y0, best.y1) == (0, 200)


def test_prefers_two_thirds_ratio_when_there_is_room() -> None:
    cons = build_constraints(spec(keep=[(0.7, 1.0)]), [], 1500, W)
    with_pref = search_crops(cons, flat_energy(1500), P)[0]
    without = search_crops(cons, flat_energy(1500), SearchParams(w_ratio=0.0))[0]
    assert abs(W / with_pref.h - 2 / 3) < abs(W / without.h - 2 / 3)


# --- Cas limites -------------------------------------------------------------------------------------
def test_conflict_offers_keep_bubbles_and_cut_subject_candidates() -> None:
    # Cas des créatures (bloc de 800 px de large) : sujet 0-644, bulles détachées 600-900 qui débordent sur lui.
    boxes = [box(600, 900, BUBBLE, x0=40, x1=180), box(610, 890, BUBBLE, x0=220, x1=380), box(100, 180, HEAD)]
    s = spec(keep=[(0.0, 0.7)], drop=[(0.7, 1.0, "detached_bubble")])
    plan = plan_block((920, 800), flat_energy(920), boxes, s, P)
    assert plan.conflict and plan.mode == "single"
    ends = {c.policy: c.y1 for c in plan.candidates}
    assert ends.get("cut_subject", 10**6) <= 600 or ends.get("default", 10**6) <= 600  # sujet rogné, bulles dehors
    assert max(ends.values()) >= 900  # bulles gardées entières
    assert not any(p.endswith(":relaxed") for p in ends)
    for c in plan.candidates:
        cons = plan.constraints
        assert not any(crosses(b, c.y0) or crosses(b, c.y1) for b in cons.hard)


def test_tall_block_is_split_into_consecutive_crops_at_clean_rows() -> None:
    # Deux personnages l'un sous l'autre, séparés par 200 px calmes : deux plans.
    H = 1300
    boxes = [box(60, 140, HEAD), box(760, 840, HEAD), box(50, 550, PERSON), box(750, 1250, PERSON)]
    plan = plan_block((H, W), flat_energy(H), boxes, spec(keep=[(0.0, 0.45), (0.55, 1.0)]), P)
    assert plan.mode == "multi"
    assert len(plan.windows) >= 2
    for (a0, a1, pan), (b0, _, _) in zip(plan.windows, plan.windows[1:]):
        assert a1 == b0 and not pan
    for y0, y1, _ in plan.windows:
        assert HMIN <= y1 - y0 <= HMAX
        assert not any(crosses(b, y0) or crosses(b, y1) for b in plan.constraints.hard + plan.constraints.persons)


def test_tall_continuous_landscape_becomes_a_pan() -> None:
    # Paysage d'un seul tenant plus haut que le cadre, sans personnage : aucune coupe entre deux sujets.
    H = 1650
    plan = plan_block((H, 800), flat_energy(H), [], spec(role="insert", keep=[(0.08, 0.2), (0.2, 0.75), (0.75, 1.0)]), P)
    assert plan.mode == "pan" and plan.windows[0][2]


def test_tall_subject_without_clean_cut_becomes_a_pan() -> None:
    H = 2000
    boxes = [box(100, 1950, PERSON), box(120, 220, HEAD)]
    plan = plan_block((H, W), flat_energy(H), boxes, spec(keep=[(0.05, 0.98)]), P)
    assert plan.mode == "pan"
    ((y0, y1, pan),) = plan.windows
    assert pan and y1 - y0 > HMAX
    assert split_tall(plan.constraints, flat_energy(H), P) is None


def test_impossible_constraints_fall_back_to_relaxed_search() -> None:
    # Bulle détachée 100-950 à cheval sur le sujet 300-600 : toute fenêtre propre ferait plus de 727 px.
    boxes = [box(100, 950, BUBBLE)]
    s = spec(keep=[(0.3, 0.6)], drop=[(0.1, 0.95, "detached_bubble")])
    plan = plan_block((1000, W), flat_energy(1000), boxes, s, P)
    assert plan.relaxed and plan.candidates
    assert plan.candidates[0].policy == "default:relaxed"


def test_text_only_block_has_no_image() -> None:
    plan = plan_block((300, W), flat_energy(300), [], spec(role="text_only", tts=["Hi."]), P)
    assert plan.mode == "none" and plan.windows == []


def test_window_report_counts_violations() -> None:
    boxes = [box(100, 200, HEAD), box(300, 400, BUBBLE), box(500, 560, FREE_TEXT), box(0, 80, BUBBLE)]
    s = spec(keep=[(0.1, 0.9)], drop=[(0.0, 0.1, "narration"), (0.45, 0.6, "sfx")])
    cons = build_constraints(s, boxes, 1000, W)
    report = window_report(150, 530, cons)
    assert report["head_cuts"] == 1 and report["sfx_cuts"] == 1 and report["bubble_cuts"] == 0
    assert window_report(0, 900, cons)["narration_included"] == 1
    assert report["ratio"] == round(W / 380, 4)
