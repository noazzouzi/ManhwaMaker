"""Style dynamique du rendu Kdenlive : transitions, mouvements, étalonnage."""

from __future__ import annotations

import numpy as np

from src.models.timeline import PanelClip
from src.modules.kdenlive_style import MOTION_PATTERNS, grade, motion_pattern, plan_cuts


def clips(scenes: list[int], duration: float = 3.0) -> list[PanelClip]:
    return [PanelClip(scene_index=s, panel_index=i, file=f"p{i}.png", width=100, height=100, start_s=i * duration,
                      duration_s=duration, motion="ken_burns") for i, s in enumerate(scenes)]


def test_every_cut_gets_a_transition_that_never_repeats_twice_in_a_row() -> None:
    cuts = plan_cuts(clips([0, 0, 0, 0, 0]), ["action"] * 5)
    assert [(c.kind, c.option) for c in cuts] == [("push", "left"), ("whip", "right"), ("push", "up"), ("whip", "left")]
    assert all(c.duration_s == 0.25 and not c.scene_change for c in cuts)


def test_scene_changes_are_stronger_and_follow_the_incoming_emotion() -> None:
    cuts = plan_cuts(clips([0, 1, 2]), ["calm", "tension", "sad"])
    assert [(c.kind, c.scene_change, c.duration_s) for c in cuts] == [("glitch", True, 0.4), ("dip_black", True, 0.5)]


def test_short_clips_cap_the_transition_or_keep_a_hard_cut() -> None:
    short = clips([0, 1], duration=1.0)
    assert plan_cuts(short, ["", "action"])[0].duration_s == 0.3  # 30 % de la case la plus courte
    assert plan_cuts(clips([0, 1], duration=0.3), ["", "action"]) == [None]


def test_chapter_start_fades_through_black() -> None:
    cuts = plan_cuts(clips([0, 1, 2]), ["action"] * 3, chapter_starts=frozenset({2}))
    assert (cuts[1].kind, cuts[1].duration_s) == ("dip_black", 0.9)  # 1 s, bornée à 30 % des cases de 3 s


def test_motion_cycles_through_patterns_within_the_margin() -> None:
    assert motion_pattern(0) == motion_pattern(len(MOTION_PATTERNS)) == ("in", 0.5, 0.5)
    assert all(0.0 <= ax <= 1.0 and 0.0 <= ay <= 1.0 for _, ax, ay in MOTION_PATTERNS)


def test_grade_keeps_the_centre_and_darkens_the_corners() -> None:
    image = np.full((90, 160, 3), 128, np.uint8)
    neutral = grade(image, "neutral")
    assert neutral.shape == image.shape and abs(int(neutral[45, 80, 0]) - 128) <= 1
    assert neutral[0, 0, 0] < 100  # vignettage
    sad = grade(image, "sad").astype(int)
    assert sad[45, 80, 2] > sad[45, 80, 0]  # triste : plus froid
