"""Tests du bi-format : profils, fabrique, cadrage et rythme."""

from __future__ import annotations

import numpy as np
import pytest

from src.models.format_profile import CropWindow, FormatProfile
from src.modules.format_factory import (
    LONG_SPEED,
    SHORT_MAX_CLIP_S,
    SHORT_MAX_UPSCALE,
    VideoConfigFactory,
    describe,
)
from src.modules.framing import (
    ContainFraming,
    SalientCropFraming,
    allowed_tighten,
    base_window,
    best_offset,
    make_framing,
    saliency_profile,
    spread_offsets,
    tighten_window,
)
from src.modules.pacing import LongPacing, ShortPacing, ShotPlan, make_pacing

#: Cases typiques du corpus : 800 px de large, hauteurs variees.
META = {i: {"index": i, "file": f"panel_{i:03d}.png", "width": 800, "height": h}
        for i, h in enumerate((775, 648, 203, 1464))}


def _drawing(width: int, height: int, *, ink_at: int, ink_width: int = 120) -> np.ndarray:
    """Case synthetique : fond blanc, zone dessinee (bruit contraste) a une position donnee."""
    image = np.full((height, width, 3), 250, dtype=np.uint8)
    rng = np.random.default_rng(0)
    image[:, ink_at : ink_at + ink_width] = rng.integers(0, 255, (height, ink_width, 3), dtype=np.uint8)
    return image


# --- Profils et fabrique -----------------------------------------------------------------------
def test_factory_builds_both_formats_and_freezes_them() -> None:
    assert set(VideoConfigFactory.available()) == {"LONG", "SHORT"}
    long_p, short_p = VideoConfigFactory.create("LONG"), VideoConfigFactory.create("short")
    assert isinstance(long_p, FormatProfile) and short_p.is_short and not long_p.is_short
    # Le mode long garde exactement les reglages historiques.
    assert (long_p.framing.width, long_p.framing.height) == (1920, 1080)
    assert long_p.framing.fit == "contain" and long_p.framing.max_upscale == 1.05
    assert long_p.pacing.min_clip_s == 2.5 and long_p.pacing.max_clip_s is None
    assert long_p.subtitles.max_words == 4 and long_p.subtitles.stroke_px == 2
    # Seule derogation aux reglages historiques : la voix est acceleree de 15 %, a la
    # demande. Kokoro parle plus vite a la synthese, la hauteur de voix ne monte pas.
    assert long_p.audio.speed == LONG_SPEED == 1.15 and not long_p.outro.enabled
    # Le format court applique la specification.
    assert (short_p.framing.width, short_p.framing.height) == (1080, 1920)
    assert short_p.framing.fit == "cover_crop" and short_p.framing.saliency_crop
    assert short_p.pacing.max_clip_s == SHORT_MAX_CLIP_S and short_p.pacing.min_clip_s is None
    assert (short_p.subtitles.min_words, short_p.subtitles.max_words) == (1, 3)
    assert short_p.subtitles.stroke_px == 8 and short_p.subtitles.highlight_color == (255, 255, 0)
    assert short_p.subtitles.highlight_current_word
    assert 1.18 <= short_p.audio.speed <= 1.22 and short_p.audio.max_internal_silence_s == 0.1
    assert short_p.outro.enabled and short_p.outro.duration_s == 5.0
    # Le profil est immuable : on ne peut pas le modifier par megarde en aval.
    with pytest.raises(Exception):
        short_p.name = "LONG"


def test_factory_overrides_are_revalidated() -> None:
    profile = VideoConfigFactory.create("SHORT", framing={"fps": 30}, audio={"speed": 1.18})
    assert profile.framing.fps == 30 and profile.audio.speed == 1.18
    assert profile.framing.width == 1080  # le reste du profil est intact
    assert "SHORT" in describe(profile) and "1080x1920" in describe(profile)
    with pytest.raises(ValueError, match="Format inconnu"):
        VideoConfigFactory.create("VERTICAL")
    with pytest.raises(ValueError, match="Section de profil inconnue"):
        VideoConfigFactory.create("SHORT", couleurs={})
    with pytest.raises(ValueError, match="dictionnaire"):
        VideoConfigFactory.create("SHORT", audio=1.2)
    # Les bornes des sections sont revalidees, pas contournees.
    with pytest.raises(ValueError):
        VideoConfigFactory.create("SHORT", audio={"speed": 0})


def test_crop_window_knows_how_it_is_placed() -> None:
    window = CropWindow(x=10, y=20, width=400, height=800)
    assert window.right == 410 and window.bottom == 820 and window.center == (210.0, 420.0)
    # Remplir un cadre 9:16 demande plus d'agrandissement que d'y entrer entier.
    assert window.cover_scale(1080, 1920) == pytest.approx(2.7)
    assert window.contain_scale(1080, 1920) == pytest.approx(2.4)
    assert window.effective_scale(1080, 1920) == window.cover_scale(1080, 1920)  # fit="cover" par defaut
    contained = window.model_copy(update={"fit": "contain"})
    assert contained.effective_scale(1080, 1920) == contained.contain_scale(1080, 1920)


# --- Cadrage -----------------------------------------------------------------------------------
def test_base_window_matches_the_frame_ratio() -> None:
    # Case plus large que le 9:16 : la fenetre prend toute la hauteur.
    assert base_window(800, 775, 9 / 16) == (436, 775)
    # Case plus haute : la fenetre prend toute la largeur.
    assert base_window(800, 1600, 9 / 16) == (800, 1422)
    # Cadre 16:9 sur une case presque carree : la hauteur est reduite.
    assert base_window(800, 775, 16 / 9) == (800, 450)


def test_saliency_finds_the_drawn_area() -> None:
    image = _drawing(800, 400, ink_at=600)
    profile = saliency_profile(image, axis=0)
    assert len(profile) == 800 and profile.max() == pytest.approx(1.0)
    # Le pic tombe sur la zone dessinee, pas sur le fond blanc.
    assert 560 < int(np.argmax(profile)) < 740
    # Une fenetre de 200 px se pose sur cette zone.
    assert 500 < best_offset(profile, 200) < 700
    # Image vide : profil nul, aucune erreur.
    assert saliency_profile(np.zeros((0, 0, 3), dtype=np.uint8)).size == 1


def test_spread_offsets_are_distinct_and_ordered_by_saliency() -> None:
    image = _drawing(800, 400, ink_at=600)
    profile = saliency_profile(image, axis=0)
    offsets = spread_offsets(profile, 200, 4)
    assert len(offsets) == 4 and len(set(offsets)) == 4  # jamais deux fois la meme position
    assert all(0 <= o <= 600 for o in offsets)
    # Le premier plan tombe sur la bande la plus dessinee.
    assert abs(offsets[0] - best_offset(profile, 200)) < 200
    # Sans marge, toutes les positions valent 0 plutot que de lever.
    assert spread_offsets(profile, 800, 3) == [0, 0, 0]


def test_tighten_is_capped_by_the_upscale_ceiling() -> None:
    window = CropWindow(x=100, y=0, width=400, height=711)
    base = window.cover_scale(1080, 1920)
    assert base == pytest.approx(max(1080 / 400, 1920 / 711))
    # Plafond 3,2x : il reste 18 % de resserrement, pas les 28 % demandes.
    room = allowed_tighten(window, 1080, 1920, 3.2)
    assert room == pytest.approx(3.2 / base - 1.0)
    assert 0.15 < room < 0.20
    # Case deja au-dela du plafond : plus aucun resserrement autorise.
    assert allowed_tighten(CropWindow(x=0, y=0, width=100, height=178), 1080, 1920, 3.2) == 0.0
    # Le resserrement conserve le centre, le mode de pose, et ne sort pas de la case.
    tightened = tighten_window(window, 0.25, (800, 711))
    assert tightened.width < window.width and tightened.fit == window.fit
    assert tightened.x >= 0 and tightened.right <= 800
    assert abs(tightened.center[0] - window.center[0]) <= 1


def test_contain_framing_never_crops() -> None:
    framing = ContainFraming()
    windows = framing.windows(META[0], 3)
    assert len(windows) == 3
    assert all(w.x == 0 and w.y == 0 and w.width == 800 and w.height == 775 for w in windows)
    assert all(w.fit == "contain" for w in windows)


def test_salient_crop_uses_the_drawn_area_and_respects_the_ceiling() -> None:
    images = {0: _drawing(800, 775, ink_at=620)}
    framing = SalientCropFraming(1080, 1920, max_upscale=3.2, loader=lambda p: images[p["index"]])
    windows = framing.windows(META[0], 4)
    assert len(windows) == 4 and len({(w.x, w.y, w.width) for w in windows}) == 4
    assert all(w.fit == "cover" for w in windows)
    # La premiere fenetre couvre la zone dessinee.
    assert windows[0].x <= 620 <= windows[0].right
    # Aucune fenetre ne depasse le plafond d'agrandissement.
    assert max(w.effective_scale(1080, 1920) for w in windows) <= 3.2 + 1e-6


def test_salient_crop_falls_back_to_contain_on_flat_panels() -> None:
    """Une case de 203 px de haut demanderait 9,5x : on l'affiche entiere a la place."""
    framing = SalientCropFraming(1080, 1920, max_upscale=3.2, loader=None)
    windows = framing.windows(META[2], 4)
    assert all(w.fit == "contain" for w in windows)
    assert windows[0].width == 800 and windows[0].height == 203  # case entiere
    # Les plans restent distincts : sans marge de recadrage, la variation vient du zoom.
    assert len({(w.x, w.y, w.width) for w in windows}) == 4
    assert max(w.effective_scale(1080, 1920) for w in windows) <= 3.2 + 1e-6
    # Soupape desactivee : le recadrage est impose, au prix de la nettete.
    strict = SalientCropFraming(1080, 1920, max_upscale=3.2, loader=None, fallback_contain=False)
    assert strict.windows(META[2], 1)[0].fit == "cover"


def test_make_framing_follows_the_profile() -> None:
    assert isinstance(make_framing(VideoConfigFactory.create("LONG")), ContainFraming)
    assert isinstance(make_framing(VideoConfigFactory.create("SHORT")), SalientCropFraming)


# --- Rythme ------------------------------------------------------------------------------------
def _plans(profile, scene_duration: float, panel_ids, heavy=()) -> list[ShotPlan]:
    pacing = make_pacing(profile, make_framing(profile))
    return pacing.plan(0, list(panel_ids), scene_duration, META, heavy)


def test_long_pacing_keeps_one_clip_per_panel() -> None:
    profile = VideoConfigFactory.create("LONG")
    assert isinstance(make_pacing(profile, make_framing(profile)), LongPacing)
    shots = _plans(profile, 8.0, [0, 1], heavy=[0])
    assert len(shots) == 2 and {s.panel_index for s in shots} == {0, 1}
    assert sum(s.duration_s for s in shots) == pytest.approx(8.0)
    assert shots[0].motion == "punch_in" and shots[1].motion == "ken_burns"
    assert all(s.subshot == 0 for s in shots)
    # Scene trop courte pour deux cases : la plus petite est ecartee.
    assert len(_plans(profile, 4.0, [0, 1])) == 1


def test_short_pacing_holds_the_cadence_and_the_exact_duration() -> None:
    profile = VideoConfigFactory.create("SHORT")
    assert isinstance(make_pacing(profile, make_framing(profile)), ShortPacing)
    # Une phrase de 4 s illustree par 4 plans, meme avec une seule case.
    shots = _plans(profile, 4.0, [0])
    assert len(shots) == 4 and max(s.duration_s for s in shots) <= SHORT_MAX_CLIP_S + 1e-6
    assert sum(s.duration_s for s in shots) == pytest.approx(4.0)
    assert [s.subshot for s in shots] == [0, 1, 2, 3]
    # Les mouvements alternent punch-in et balayage.
    assert {s.motion for s in shots} == {"punch_in", "fast_pan"}
    # Chaque plan porte une fenetre de cadrage.
    assert all(s.crop is not None for s in shots)
    # Scene plus courte que le plafond : un seul plan.
    assert len(_plans(profile, 1.0, [0, 1])) == 1


def test_short_pacing_is_capped_by_available_material() -> None:
    """Faute de cases, les plans s'allongent plutot que de se multiplier a l'infini."""
    profile = VideoConfigFactory.create("SHORT")
    shots = _plans(profile, 30.0, [0])  # 30 s, une seule case
    assert len(shots) == profile.pacing.max_subshots_per_panel
    assert sum(s.duration_s for s in shots) == pytest.approx(30.0)
    # Avec plus de cases, la cadence redevient tenable.
    many = _plans(profile, 30.0, [0, 1, 2, 3])
    assert len(many) > len(shots)
    assert max(s.duration_s for s in many) < max(s.duration_s for s in shots)


def test_expand_panel_ids_gives_the_short_format_more_material() -> None:
    from src.modules.timeline_builder import expand_panel_ids

    # Trois scenes aux cases cles espacees, et 10 cases disponibles.
    widened = expand_panel_ids([[0, 2], [5], [8]], list(range(10)))
    assert widened == [[0, 1, 2, 3, 4], [5, 6, 7], [8, 9]]
    # Les plages sont disjointes et couvrent tout.
    flat = [pid for ids in widened for pid in ids]
    assert flat == sorted(flat) and len(set(flat)) == len(flat)
    # Une scene sans case reste vide, et rien n'est invente hors des cases existantes.
    assert expand_panel_ids([[], [3]], [3, 4]) == [[], [3, 4]]
    assert expand_panel_ids([[0]], []) == [[0]]


def test_both_profiles_expand_to_unused_panels() -> None:
    """Le garde-fou : oublier le drapeau sur un profil ferait regresser ce format en silence."""
    for name in VideoConfigFactory.available():
        assert VideoConfigFactory.create(name).pacing.expand_to_unused_panels is True, name


def test_long_expands_without_inheriting_short_pacing() -> None:
    """L'elargissement et la cadence sont deux reglages distincts : le long garde son plancher."""
    long_p = VideoConfigFactory.create("LONG")
    assert long_p.pacing.expand_to_unused_panels is True
    assert long_p.pacing.max_clip_s is None          # pas de subdivision facon format court
    assert long_p.pacing.min_clip_s == 2.5           # plancher historique intact
    assert long_p.pacing.max_subshots_per_panel == 1  # aucun re-cadrage


def test_expansion_can_be_switched_off_by_profile() -> None:
    assert VideoConfigFactory.create("LONG", pacing={"expand_to_unused_panels": False}).pacing.expand_to_unused_panels is False
