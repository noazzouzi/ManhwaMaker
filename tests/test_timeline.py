"""Tests du constructeur de timeline (``src.modules.timeline_builder``)."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.models.audio import SceneAudio, VoiceoverManifest
from src.models.format_profile import PacingRules
from src.models.scene import ChapterAnalysis, Scene
from src.models.timeline import PanelClip
from src.modules.format_factory import VideoConfigFactory
from src.modules.timeline_builder import (
    DEFAULT_BGM_CROSSFADE_S,
    DEFAULT_BGM_GAIN_DB,
    DEFAULT_MAX_SUBTITLE_WORDS,
    DEFAULT_MIN_CLIP_S,
    DEFAULT_MIN_PANEL_WEIGHT,
    KEN_BURNS_ZOOM,
    MAX_ZOOM,
    MIN_CUE_DURATION_S,
    PUNCH_IN_S,
    PUNCH_IN_ZOOM,
    SFX_CYCLE,
    TimelineError,
    animation_for_cue,
    attach_transitions,
    build_bgm_clips,
    build_subtitle_cues,
    build_timeline,
    build_vfx_clips,
    concat_timelines,
    load_panels_meta,
    load_timeline,
    mood_of,
    motion_for,
    save_timeline,
    split_subtitle_text,
    transition_for,
)

PANELS_META = [
    {"index": 0, "file": "panel_000.png", "width": 720, "height": 1000, "type": "static", "y_start": 0, "y_end": 1000},
    {"index": 1, "file": "panel_001.png", "width": 720, "height": 60, "type": "static", "y_start": 1000, "y_end": 1060},
    {"index": 2, "file": "panel_002.png", "width": 720, "height": 400, "type": "static", "y_start": 1060, "y_end": 1460},
    {"index": 3, "file": "panel_003.png", "width": 720, "height": 3000, "type": "scroll_vertical", "y_start": 1460, "y_end": 4460},
]
SFX = {"impact": Path("/sfx/impact.wav"), "swoosh": Path("/sfx/swoosh.wav"), "roar": Path("/sfx/roar.wav")}
BGM = {"calm": Path("/bgm/calm.mp3"), "action": Path("/bgm/action.mp3")}


def _analysis() -> ChapterAnalysis:
    return ChapterAnalysis(
        series_title="S", episode_title="E", model="fake", language="en", n_panels=4,
        scenes=[
            Scene(index=0, panel_ids=[0, 1], emotion="calm",
                  narration="The hero wakes up. He looks around the ruined city, searching for survivors."),
            Scene(index=1, panel_ids=[2], narration="Title card.", emotion="neutral", is_filler=True),
            Scene(index=2, panel_ids=[3], narration="Fight!", emotion="action"),
        ],
    )


def _manifest() -> VoiceoverManifest:
    return VoiceoverManifest(
        language="en", lang_code="a", voice="am_fenrir", speed=1.0, padding_s=0.3, sample_rate=24000,
        items=[
            SceneAudio(scene_index=0, file="scene_000.wav", duration_s=5.3, speech_s=5.0, sample_rate=24000, text="..."),
            SceneAudio(scene_index=2, file="scene_002.wav", duration_s=3.3, speech_s=3.0, sample_rate=24000, text="Fight!"),
        ],
        total_duration_s=8.6,
    )


def test_build_timeline_durations_follow_audio_and_panel_weights(tmp_path) -> None:
    timeline = build_timeline(_analysis(), _manifest(), PANELS_META, panels_dir=tmp_path, audio_dir=tmp_path / "audio")
    assert timeline.width == 1920 and timeline.height == 1080 and timeline.fps == 60
    assert timeline.series_title == "S" and timeline.total_duration_s == pytest.approx(8.6)
    assert [c.panel_index for c in timeline.clips] == [0, 1, 3]  # la scene filler (case 2) n'est pas montee
    first, second, third = timeline.clips
    # 2 cases pour 5.3 s, partagees au prorata des poids et non a parts egales : la grande
    # case (1000 px) reste bien plus longtemps que la petite (60 px, ramenee au plancher de
    # poids). Un partage uniforme donnerait 2.65 s chacune, c'est-a-dire le metronome.
    assert first.duration_s == pytest.approx(3.70)
    assert second.duration_s == pytest.approx(1.60)
    assert first.duration_s > 2 * second.duration_s
    assert second.duration_s >= PacingRules().emphasis_floor_s - 1e-9
    assert second.start_s == pytest.approx(first.end_s)
    assert second.end_s == pytest.approx(5.3)  # la derniere case absorbe l'arrondi
    assert third.start_s == pytest.approx(5.3) and third.duration_s == pytest.approx(3.3)
    # Plus aucun defilement ni recadrage : Ken Burns doux sur toutes les cases, geante comprise.
    assert [c.motion for c in timeline.clips] == ["ken_burns", "ken_burns", "ken_burns"]
    assert [(a.scene_index, a.start_s, a.duration_s) for a in timeline.audio] == [(0, 0.0, 5.3), (2, 5.3, 3.3)]
    # Sous-titres : blocs de 2-4 mots repartis sur 5.0 s de parole, puis "Fight!".
    cues = [c for c in timeline.subtitles if c.scene_index == 0]
    assert [c.text for c in cues] == [
        "The hero wakes up.", "He looks around", "the ruined city,", "searching for survivors.",
    ]
    assert cues[0].start_s == 0.0 and cues[0].end_s == pytest.approx(5.0 * 18 / 73)
    assert all(cues[i].end_s == cues[i + 1].start_s for i in range(len(cues) - 1))
    assert cues[-1].end_s == pytest.approx(5.0)
    assert timeline.subtitles[-1].text == "Fight!" and timeline.subtitles[-1].start_s == pytest.approx(5.3)
    assert timeline.n_scenes == 2
    assert timeline.sfx == [] and timeline.bgm == []  # aucun fichier audio fourni


def test_the_impact_panel_gets_more_screen_time(tmp_path) -> None:
    """Le defaut que ce test protege : un coup decisif et un plan de dialogue avaient
    exactement le meme temps d'ecran. Mesure sur un vrai chapitre avant correction : les
    78 cases tenaient toutes entre 2.57 s et 2.89 s, soit un metronome.

    ``action_heavy_ids`` est le seul signal disant *ce moment compte*, et il ne servait
    qu'a choisir le mouvement de camera, jamais le partage du temps.
    """
    meta = [
        {"index": i, "file": f"panel_{i:03d}.png", "width": 720, "height": 800,
         "type": "static", "y_start": 800 * i, "y_end": 800 * (i + 1)}
        for i in range(4)
    ]

    def timeline_for(heavy: list[int]):
        analysis = ChapterAnalysis(
            model="fake", language="en", n_panels=4,
            scenes=[Scene(index=0, panel_ids=[0, 1, 2, 3], narration="Quatre cases.",
                          emotion="action", action_heavy_ids=heavy)],
        )
        manifest = VoiceoverManifest(
            language="en", lang_code="a", voice="v", speed=1.0, padding_s=0.3, sample_rate=24000,
            items=[SceneAudio(scene_index=0, file="s.wav", duration_s=12.0, speech_s=11.5,
                              sample_rate=24000, text="t")],
            total_duration_s=12.0,
        )
        return build_timeline(analysis, manifest, meta, panels_dir=tmp_path, audio_dir=tmp_path)

    marque = timeline_for([2])
    durees = [c.duration_s for c in marque.clips]
    assert durees == pytest.approx([2.4, 2.4, 4.8, 2.4])
    assert durees[2] == pytest.approx(2 * durees[0])          # exactement heavy_emphasis
    assert marque.clips[2].motion == "punch_in"
    # L'image ne doit jamais glisser par rapport a la voix.
    assert sum(durees) == pytest.approx(12.0)
    assert marque.total_duration_s == pytest.approx(12.0)

    # Rien de marque et des cases de meme hauteur : aucune variation inventee.
    neutre = timeline_for([])
    assert [c.duration_s for c in neutre.clips] == pytest.approx([3.0] * 4)
    assert sum(c.duration_s for c in neutre.clips) == pytest.approx(12.0)


def test_a_panel_taller_than_the_frame_is_held_longer(tmp_path) -> None:
    """Une case plus haute que le cadre est affichee en entier, donc plus petite a l'ecran :
    il faut plus de temps pour la parcourir."""
    meta = [
        {"index": 0, "file": "panel_000.png", "width": 720, "height": 900,
         "type": "static", "y_start": 0, "y_end": 900},
        {"index": 1, "file": "panel_001.png", "width": 720, "height": 1600,
         "type": "static", "y_start": 900, "y_end": 2500},
    ]
    analysis = ChapterAnalysis(
        model="fake", language="en", n_panels=2,
        scenes=[Scene(index=0, panel_ids=[0, 1], narration="Deux cases.", emotion="calm")],
    )
    manifest = VoiceoverManifest(
        language="en", lang_code="a", voice="v", speed=1.0, padding_s=0.3, sample_rate=24000,
        items=[SceneAudio(scene_index=0, file="s.wav", duration_s=10.0, speech_s=9.5,
                          sample_rate=24000, text="t")],
        total_duration_s=10.0,
    )
    clips = build_timeline(analysis, manifest, meta, panels_dir=tmp_path, audio_dir=tmp_path).clips
    assert clips[1].duration_s > clips[0].duration_s
    assert sum(c.duration_s for c in clips) == pytest.approx(10.0)


def test_build_timeline_truncates_to_max_duration(tmp_path) -> None:
    timeline = build_timeline(
        _analysis(), _manifest(), PANELS_META, panels_dir=tmp_path, audio_dir=tmp_path, max_duration_s=6.0,
        sfx_files=SFX, bgm_files=BGM,
    )
    assert timeline.total_duration_s == pytest.approx(6.0)
    assert timeline.clips[-1].panel_index == 3 and timeline.clips[-1].duration_s == pytest.approx(0.7)
    assert timeline.audio[-1].duration_s == pytest.approx(0.7)
    assert all(c.end_s <= 6.0 + 1e-9 for c in timeline.subtitles)
    assert all(c.start_s < 6.0 for c in timeline.sfx) and all(c.end_s <= 6.0 + 1e-9 for c in timeline.bgm)
    # Troncature avant la deuxieme scene : elle n'est pas montee du tout.
    short = build_timeline(_analysis(), _manifest(), PANELS_META, panels_dir=tmp_path, audio_dir=tmp_path, max_duration_s=5.3)
    assert [c.panel_index for c in short.clips] == [0, 1] and short.total_duration_s == pytest.approx(5.3)
    with pytest.raises(ValueError):
        build_timeline(_analysis(), _manifest(), PANELS_META, panels_dir=tmp_path, audio_dir=tmp_path, max_duration_s=0)


def test_build_timeline_errors(tmp_path) -> None:
    with pytest.raises(TimelineError, match="absentes"):
        build_timeline(_analysis(), _manifest(), PANELS_META[:3], panels_dir=tmp_path, audio_dir=tmp_path)
    empty = VoiceoverManifest(
        language="en", lang_code="a", voice="v", speed=1.0, padding_s=0.3, sample_rate=24000, items=[], total_duration_s=0,
    )
    with pytest.raises(TimelineError, match="Aucune scene"):
        build_timeline(_analysis(), empty, PANELS_META, panels_dir=tmp_path, audio_dir=tmp_path)
    with pytest.raises(TimelineError):
        load_panels_meta(tmp_path / "nowhere")


def test_split_subtitle_text_short_balanced_blocks() -> None:
    assert DEFAULT_MAX_SUBTITLE_WORDS == 4
    long = (
        "Humanity has made 79 attempts to conquer the Tower of Trials, resulting in over 315,000 casualties, "
        "and these are the grim statistics of their struggle against the formidable tower that never falls."
    )
    pieces = split_subtitle_text(long)
    assert all(2 <= len(p.split()) <= 4 for p in pieces)
    assert " ".join(pieces) == long
    assert split_subtitle_text("Short. Ok.") == ["Short.", "Ok."]  # jamais a cheval sur deux phrases
    assert split_subtitle_text("One two three four.") == ["One two three four."]
    assert split_subtitle_text("One two three four five.") == ["One two three", "four five."]
    assert split_subtitle_text("One two three four five six seven.") == ["One two three four", "five six seven."]
    assert split_subtitle_text("A b c d e f g h i.") == ["A b c", "d e f", "g h i."]
    assert split_subtitle_text("A b c d e f", max_words=2) == ["A b", "c d", "e f."]  # ponctuation finale ajoutee
    assert split_subtitle_text("   ") == []


def test_build_subtitle_cues_timing() -> None:
    cues = build_subtitle_cues(4, "One. Two words here.", 10.0, 2.0)
    assert [c.text for c in cues] == ["One.", "Two words here."]
    assert cues[0].start_s == 10.0 and cues[0].end_s == pytest.approx(10.0 + 2.0 * 4 / 19)
    assert cues[1].start_s == cues[0].end_s and cues[1].end_s == pytest.approx(12.0)
    # Bloc minuscule : au moins MIN_CUE_DURATION_S, le dernier bloc absorbe le reste.
    cues = build_subtitle_cues(0, "A. Bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb.", 0.0, 3.0)
    assert cues[0].end_s == pytest.approx(MIN_CUE_DURATION_S) and cues[-1].end_s == pytest.approx(3.0)
    assert build_subtitle_cues(0, "x", 0.0, 0.0) == []


def test_motion_for_rules_and_zoom_cap() -> None:
    assert MAX_ZOOM == 1.05 and KEN_BURNS_ZOOM <= 0.05 and PUNCH_IN_ZOOM <= 0.05 and PUNCH_IN_S == 0.2
    normal = {"type": "static", "width": 720, "height": 1000}
    giant = {"type": "scroll_vertical", "width": 720, "height": 3000}
    # Plus de defilement : toute case est fixe, native, avec un Ken Burns doux...
    assert motion_for(normal) == "ken_burns" and motion_for(giant) == "ken_burns"
    # ... sauf les impacts marques par Gemini (punch-in).
    assert motion_for(normal, action_heavy=True) == "punch_in" and motion_for(giant, action_heavy=True) == "punch_in"


def test_build_timeline_punch_in_sfx_and_bgm(tmp_path) -> None:
    analysis = ChapterAnalysis(
        model="fake", language="en", n_panels=4,
        scenes=[
            Scene(index=0, panel_ids=[0, 1], narration="Calm start.", emotion="calm", action_heavy_ids=[1]),
            Scene(index=1, panel_ids=[2, 3], narration="The blow lands.", emotion="action", action_heavy_ids=[3]),
        ],
    )
    manifest = VoiceoverManifest(
        language="en", lang_code="a", voice="v", speed=1.0, padding_s=0.4, sample_rate=24000,
        items=[
            SceneAudio(scene_index=0, file="a.wav", duration_s=6.0, speech_s=5.6, sample_rate=24000, text="t"),
            SceneAudio(scene_index=1, file="b.wav", duration_s=7.0, speech_s=6.6, sample_rate=24000, text="t"),
        ],
        total_duration_s=13.0,
    )
    timeline = build_timeline(
        analysis, manifest, PANELS_META, panels_dir=tmp_path, audio_dir=tmp_path,
        sfx_files=SFX, bgm_files=BGM, sfx_gain_db=-10.0, bgm_gain_db=-20.0,
    )
    assert [c.motion for c in timeline.clips] == ["ken_burns", "punch_in", "ken_burns", "punch_in"]
    # Bruitages : un impact sur la case punch-in de la scene calme, puis un par case de la scene d'action (cycle).
    assert [(s.scene_index, s.panel_index, s.kind, s.gain_db) for s in timeline.sfx] == [
        (0, 1, "impact", -10.0), (1, 2, SFX_CYCLE[0], -10.0), (1, 3, SFX_CYCLE[1], -10.0),
    ]
    assert [s.start_s for s in timeline.sfx] == pytest.approx([timeline.clips[i].start_s for i in (1, 2, 3)])
    assert timeline.sfx[0].file.endswith("impact.wav")
    # Musique : calme puis action, avec chevauchement pour le fondu enchaine.
    calm, action = timeline.bgm
    assert (calm.mood, action.mood) == ("calm", "action")
    assert calm.start_s == 0.0 and calm.end_s == pytest.approx(6.0 + DEFAULT_BGM_CROSSFADE_S / 2)
    assert action.start_s == pytest.approx(6.0 - DEFAULT_BGM_CROSSFADE_S / 2) and action.end_s == pytest.approx(13.0)
    assert (calm.fade_in_s, calm.fade_out_s) == (0.5, DEFAULT_BGM_CROSSFADE_S)
    assert (action.fade_in_s, action.fade_out_s) == (DEFAULT_BGM_CROSSFADE_S, 1.0)
    assert calm.gain_db == -20.0 and calm.file.endswith("calm.mp3") and action.file.endswith("action.mp3")


def test_build_bgm_clips_merges_moods_and_falls_back() -> None:
    spans = [(0, 0.0, 4.0, "calm"), (1, 4.0, 9.0, "happy"), (2, 9.0, 12.0, "fear"), (3, 12.0, 15.0, "epic")]
    assert [mood_of(e) for _, _, _, e in spans] == ["calm", "calm", "tense", "action"]
    clips = build_bgm_clips(spans, BGM, crossfade_s=2.0)
    # calm + happy fusionnes ; "tense" absent -> repli sur la premiere musique disponible.
    assert [c.mood for c in clips] == ["calm", "tense", "action"]
    assert clips[0].end_s == pytest.approx(10.0) and clips[1].start_s == pytest.approx(8.0)
    assert clips[1].file.endswith("calm.mp3")
    assert build_bgm_clips(spans, {"default": Path("/x.wav")}, single_mood=True)[0].mood == "all"
    assert build_bgm_clips(spans, {}) == [] and build_bgm_clips([], BGM) == []


def test_build_timeline_single_bgm_file_compat_and_default_gain(tmp_path) -> None:
    assert DEFAULT_BGM_GAIN_DB == -22.0
    timeline = build_timeline(
        _analysis(), _manifest(), PANELS_META, panels_dir=tmp_path, audio_dir=tmp_path, bgm_file=tmp_path / "music.mp3",
    )
    assert timeline.bgm_file is not None and timeline.bgm_file.endswith("music.mp3")
    assert len(timeline.bgm) == 1 and timeline.bgm[0].mood == "all"
    assert timeline.bgm[0].start_s == 0.0 and timeline.bgm[0].end_s == pytest.approx(8.6)
    assert timeline.bgm_gain_db == -22.0 and timeline.bgm[0].gain_db == -22.0


def test_limit_panels_for_duration_keeps_largest_in_reading_order() -> None:
    from src.modules.timeline_builder import limit_panels_for_duration

    meta = {i: {"height": h} for i, h in enumerate([300, 900, 120, 700, 500])}
    # 6 s / 2.5 s -> 2 cases max : les deux plus grandes (1 et 3), ordre de lecture conserve.
    assert limit_panels_for_duration([0, 1, 2, 3, 4], 6.0, meta, 2.5) == [1, 3]
    assert limit_panels_for_duration([0, 1, 2, 3, 4], 20.0, meta, 2.5) == [0, 1, 2, 3, 4]
    assert limit_panels_for_duration([0, 1], 1.0, meta, 2.5) == [1]  # au moins une case
    assert limit_panels_for_duration([2], 0.5, meta, 2.5) == [2]
    assert limit_panels_for_duration([0, 1, 2], 1.0, meta, 0) == [0, 1, 2]  # desactive


def test_build_timeline_applies_min_clip_duration(tmp_path) -> None:
    analysis = ChapterAnalysis(
        model="fake", language="en", n_panels=4,
        scenes=[Scene(index=0, panel_ids=[0, 1, 2, 3], narration="Four panels, short audio.", emotion="calm")],
    )
    manifest = VoiceoverManifest(
        language="en", lang_code="a", voice="v", speed=1.0, padding_s=0.3, sample_rate=24000,
        items=[SceneAudio(scene_index=0, file="s.wav", duration_s=5.3, speech_s=5.0, sample_rate=24000, text="t")],
        total_duration_s=5.3,
    )
    timeline = build_timeline(analysis, manifest, PANELS_META, panels_dir=tmp_path, audio_dir=tmp_path)
    # 5.3 s / 2.5 s -> 2 cases : les plus grandes (3 = 3000 px, 0 = 1000 px), dans l'ordre de lecture.
    assert [c.panel_index for c in timeline.clips] == [0, 3]
    # min_clip_s fixe le NOMBRE de cases (2 ici) ; la duree de chacune vient ensuite du
    # partage pondere, dont le seul plancher est emphasis_floor_s.
    assert all(c.duration_s >= PacingRules().emphasis_floor_s - 1e-9 for c in timeline.clips)
    assert timeline.total_duration_s == pytest.approx(5.3)
    with pytest.raises(ValueError):
        build_timeline(analysis, manifest, PANELS_META, panels_dir=tmp_path, audio_dir=tmp_path, min_clip_s=-1)


def test_concat_timelines_merges_chapters_with_absolute_paths(tmp_path) -> None:
    from src.modules.timeline_builder import DEFAULT_CHAPTER_GAP_S, concat_timelines

    assert DEFAULT_CHAPTER_GAP_S == 0.6
    first = build_timeline(_analysis(), _manifest(), PANELS_META, panels_dir=tmp_path / "ep1", audio_dir=tmp_path / "ep1" / "audio",
                           sfx_files=SFX, bgm_files=BGM)
    second = build_timeline(_analysis(), _manifest(), PANELS_META, panels_dir=tmp_path / "ep2", audio_dir=tmp_path / "ep2" / "audio",
                            sfx_files=SFX, bgm_files=BGM)
    merged = concat_timelines([first, second], gap_s=1.0, series_title="Ma serie", panels_dir=tmp_path / "compil", audio_dir=tmp_path / "compil")
    # Duree : deux chapitres plus un seul intervalle.
    assert merged.total_duration_s == pytest.approx(first.total_duration_s + 1.0 + second.total_duration_s)
    assert merged.series_title == "Ma serie" and merged.episode_title == "2 chapitres"
    assert len(merged.clips) == len(first.clips) + len(second.clips)
    assert len(merged.subtitles) == len(first.subtitles) + len(second.subtitles)
    # Le second chapitre est decale, le premier ne bouge pas.
    offset = first.total_duration_s + 1.0
    assert merged.clips[0].start_s == first.clips[0].start_s
    assert merged.clips[len(first.clips)].start_s == pytest.approx(offset)
    assert merged.audio[-1].start_s == pytest.approx(offset + second.audio[-1].start_s)
    assert merged.subtitles[-1].end_s == pytest.approx(offset + second.subtitles[-1].end_s)
    assert merged.sfx[-1].start_s == pytest.approx(offset + second.sfx[-1].start_s)
    assert merged.bgm[-1].start_s == pytest.approx(offset + second.bgm[-1].start_s)
    # Les numeros de scene restent uniques d'un chapitre a l'autre.
    first_scenes = {c.scene_index for c in merged.clips[: len(first.clips)]}
    second_scenes = {c.scene_index for c in merged.clips[len(first.clips) :]}
    assert not (first_scenes & second_scenes) and merged.n_scenes == first.n_scenes + second.n_scenes
    # Les medias sont references en absolu, chacun dans son chapitre.
    from pathlib import Path as _Path

    assert _Path(merged.clips[0].file).is_absolute() and "ep1" in merged.clips[0].file
    assert "ep2" in merged.clips[-1].file and "ep2" in merged.audio[-1].file
    # Un chemin absolu reste valide meme combine au panels_dir de la compilation.
    assert _Path(merged.panels_dir) / merged.clips[0].file == _Path(merged.clips[0].file)

    with pytest.raises(TimelineError, match="Aucune timeline"):
        concat_timelines([])
    other = first.model_copy(update={"fps": 30})
    with pytest.raises(TimelineError, match="incompatibles"):
        concat_timelines([first, other])
    with pytest.raises(ValueError):
        concat_timelines([first], gap_s=-1)
    # Un seul chapitre : timeline identique en duree, sans intervalle ajoute.
    alone = concat_timelines([first], gap_s=5.0)
    assert alone.total_duration_s == pytest.approx(first.total_duration_s)


def test_save_and_load_timeline(tmp_path) -> None:
    timeline = build_timeline(
        _analysis(), _manifest(), PANELS_META, panels_dir=tmp_path, audio_dir=tmp_path, sfx_files=SFX, bgm_files=BGM,
    )
    path = save_timeline(timeline, tmp_path / "t" / "timeline.json")
    assert load_timeline(path) == timeline


# --- Dynamisme du montage ---------------------------------------------------------------------
def test_animation_for_cue_is_a_fraction_of_the_block() -> None:
    """Un bloc de 2-4 mots dure souvent moins d'une demi-seconde : l'animation doit tenir dedans."""
    # 35 % de 3 s depasse le plafond : on s'arrete a MAX_CUE_ANIMATION_S.
    action = animation_for_cue(3.0, "action")
    assert action is not None and action.intro == "弹入" and action.duration_s == pytest.approx(0.40)
    assert action.loop == "心跳"  # boucle reservee aux scenes intenses
    # Bloc court : l'animation vaut 35 % du bloc, pas les 0,5 s par defaut de CapCut.
    short = animation_for_cue(0.4, "calm")
    assert short is not None and short.duration_s == pytest.approx(0.14) and short.loop == ""
    # Trop court pour etre anime proprement : le bloc reste fixe.
    assert animation_for_cue(0.3, "action") is None
    assert animation_for_cue(0.0, "action") is None
    # Les niveaux de dynamisme.
    assert animation_for_cue(3.0, "action", dynamics="none") is None
    subtle = animation_for_cue(3.0, "action", dynamics="subtle")
    assert subtle is not None and subtle.intro == "弹入" and subtle.loop == ""  # pas de boucle en mode sobre
    # Emotion inconnue : animation de repli.
    unknown = animation_for_cue(2.0, "sarcastic")
    assert unknown is not None and unknown.intro == "逐字"


def test_transition_only_between_intense_scenes_and_stays_short() -> None:
    # Une des deux scenes est action -> transition, choisie cycliquement.
    first = transition_for(0, "calm", "action", 3.0, 4.0)
    assert first is not None and first.kind == "快速挥动" and first.duration_s == pytest.approx(0.30)
    assert first.overlap is True  # presque toutes les transitions CapCut empietent
    assert transition_for(1, "action", "calm", 3.0, 4.0).kind == "甩鞭转场"  # l'effet varie
    assert transition_for(0, "tension", "calm", 3.0, 4.0).kind == "故障"  # liste propre a tension
    # Deux scenes calmes : coupe franche.
    assert transition_for(0, "calm", "sad", 3.0, 4.0) is None
    # La duree est plafonnee par la plus courte des deux cases (15 %).
    capped = transition_for(0, "calm", "action", 1.6, 9.0)
    assert capped is not None and capped.duration_s == pytest.approx(0.24)
    # Cases trop breves : mieux vaut couper net que d'ecraser les deux cases.
    assert transition_for(0, "action", "action", 1.0, 1.0) is None
    assert transition_for(0, "action", "action", 3.0, 3.0, dynamics="subtle") is None


def test_attach_transitions_only_at_scene_boundaries() -> None:
    def clip(scene: int, panel: int, start: float, duration: float):
        return PanelClip(scene_index=scene, panel_index=panel, file=f"p{panel}.png", width=720, height=1000,
                         start_s=start, duration_s=duration, motion="ken_burns")

    clips = [clip(0, 0, 0.0, 4.0), clip(0, 1, 4.0, 4.0), clip(1, 2, 8.0, 4.0), clip(1, 3, 12.0, 4.0)]
    result = attach_transitions(clips, {0: "calm", 1: "action"})
    # Seule la derniere case de la scene 0 porte la transition (convention pycapcut).
    assert [c.transition is not None for c in result] == [False, True, False, False]
    assert result[1].transition.kind == "快速挥动"
    # Mode sobre : aucune transition.
    assert all(c.transition is None for c in attach_transitions(clips, {0: "calm", 1: "action"}, dynamics="none"))


def test_build_vfx_clips_targets_spectacular_scenes_only() -> None:
    spans = [(0, 0.0, 5.0, "action"), (1, 5.0, 6.0, "epic"), (2, 6.0, 12.0, "mystery"), (3, 12.0, 20.0, "calm")]
    clips = build_vfx_clips(spans)
    # epic dure 1 s (< MIN_VFX_SCENE_S) et calm n'est pas spectaculaire : 2 effets seulement.
    assert [(c.scene_index, c.kind, c.effect) for c in clips] == [
        (0, "speed_lines", "冲刺"), (2, "rain", "下雨"),
    ]
    assert all(c.file == "" for c in clips)  # effets integres : aucun asset requis
    assert clips[0].duration_s == pytest.approx(5.0)
    # Un asset a canal alpha est prefere a l'effet integre quand il existe.
    with_asset = build_vfx_clips(spans, assets={"speed_lines": Path("/vfx/lines.webm")})
    assert with_asset[0].file.endswith("lines.webm") and with_asset[0].effect == ""
    assert with_asset[1].effect == "下雨"  # pas d'asset pour la pluie : effet integre
    assert build_vfx_clips(spans, dynamics="none") == []


def test_build_timeline_wires_dynamics_and_reports_possible_drift(tmp_path) -> None:
    timeline = build_timeline(_analysis(), _manifest(), PANELS_META, panels_dir=tmp_path, audio_dir=tmp_path)
    # Une seule frontiere de scene (calm -> action), donc une seule transition.
    assert timeline.n_transitions == 1
    assert [c.transition is not None for c in timeline.clips] == [False, True, False]
    # La transition est plafonnee a 15 % de la plus courte des deux cases qu'elle relie :
    # elle suit donc le nouveau partage du temps.
    assert timeline.transition_drift_s == pytest.approx(0.24)
    # La scene action recoit un effet superpose, la scene calme non.
    assert [(c.scene_index, c.kind) for c in timeline.vfx] == [(2, "speed_lines")]
    # Les sous-titres sont animes selon l'emotion de leur scene.
    animated = [c for c in timeline.subtitles if c.animation is not None]
    assert animated and all(c.animation.intro for c in animated)
    assert {c.animation.intro for c in timeline.subtitles if c.scene_index == 0} <= {"渐显"}
    # Les durees restent calees sur l'audio : le dynamisme ne deplace rien.
    assert timeline.total_duration_s == pytest.approx(8.6)

    sober = build_timeline(_analysis(), _manifest(), PANELS_META, panels_dir=tmp_path, audio_dir=tmp_path, dynamics="none")
    assert sober.n_transitions == 0 and sober.vfx == [] and sober.transition_drift_s == 0.0
    assert all(c.animation is None for c in sober.subtitles)
    assert [c.duration_s for c in sober.clips] == [c.duration_s for c in timeline.clips]


def test_concat_drops_the_last_transition_and_carries_vfx(tmp_path) -> None:
    first = build_timeline(_analysis(), _manifest(), PANELS_META, panels_dir=tmp_path / "ep1", audio_dir=tmp_path / "ep1")
    # On force une transition sur la derniere case : elle enjamberait le silence inter-chapitre.
    clips = list(first.clips)
    clips[-1] = clips[-1].model_copy(update={"transition": clips[1].transition})
    first = first.model_copy(update={"clips": clips})
    second = build_timeline(_analysis(), _manifest(), PANELS_META, panels_dir=tmp_path / "ep2", audio_dir=tmp_path / "ep2")

    merged = concat_timelines([first, second], gap_s=0.6)
    assert merged.clips[len(first.clips) - 1].transition is None  # coupee a la jointure
    assert merged.clips[1].transition is not None  # les transitions internes survivent
    assert len(merged.vfx) == len(first.vfx) + len(second.vfx)
    offset = first.total_duration_s + 0.6
    assert merged.vfx[-1].start_s == pytest.approx(offset + second.vfx[-1].start_s)
    # Les numeros de scene des effets suivent le meme decalage que les clips.
    assert merged.vfx[-1].scene_index != second.vfx[-1].scene_index
    assert merged.subtitles[0].animation == first.subtitles[0].animation


def _analysis_with_unused_panels() -> ChapterAnalysis:
    """Deux scenes, cases cles 0 et 3 : les cases 1 et 2 sont libres et non filler."""
    return ChapterAnalysis(
        series_title="S", episode_title="E", model="fake", language="en", n_panels=4,
        scenes=[
            Scene(index=0, panel_ids=[0], emotion="calm",
                  narration="The hero wakes up alone in the ruined city and starts to walk north."),
            Scene(index=1, panel_ids=[3], emotion="action", narration="Fight!"),
        ],
    )


def _manifest_with_room() -> VoiceoverManifest:
    """9 s sur la premiere scene : de quoi tenir trois cases au plancher de 2,5 s."""
    return VoiceoverManifest(
        language="en", lang_code="a", voice="am_puck", speed=1.0, padding_s=0.3, sample_rate=24000,
        items=[
            SceneAudio(scene_index=0, file="scene_000.wav", duration_s=9.0, speech_s=8.7, sample_rate=24000, text="..."),
            SceneAudio(scene_index=1, file="scene_001.wav", duration_s=3.3, speech_s=3.0, sample_rate=24000, text="Fight!"),
        ],
        total_duration_s=12.3,
    )


def test_long_timeline_shows_the_panels_left_out_by_the_analysis(tmp_path) -> None:
    """En format long, une scene s'etend aux cases non retenues qui la suivent."""
    timeline = build_timeline(
        _analysis_with_unused_panels(), _manifest_with_room(), PANELS_META,
        panels_dir=tmp_path, audio_dir=tmp_path / "audio",
    )
    assert [c.panel_index for c in timeline.clips] == [0, 1, 2, 3]
    # Sans elargissement, seules les deux cases cles seraient montees.
    sober = build_timeline(
        _analysis_with_unused_panels(), _manifest_with_room(), PANELS_META,
        panels_dir=tmp_path, audio_dir=tmp_path / "audio",
        profile=VideoConfigFactory.create("LONG", pacing={"expand_to_unused_panels": False}),
    )
    assert [c.panel_index for c in sober.clips] == [0, 3]
    # La voix reste le maitre du temps : elargir ne change pas la duree de la video.
    assert timeline.total_duration_s == pytest.approx(sober.total_duration_s)


def test_long_widening_never_shows_a_filler_scene_panel(tmp_path) -> None:
    """La case 2 appartient a la scene filler : l'elargissement ne doit pas la repecher."""
    timeline = build_timeline(
        _analysis(), _manifest(), PANELS_META, panels_dir=tmp_path, audio_dir=tmp_path / "audio",
    )
    assert 2 not in [c.panel_index for c in timeline.clips]


def test_long_widening_keeps_native_resolution_and_capped_zoom(tmp_path) -> None:
    """Les regles dures tiennent sur les cases repechees : pas de rognage, zoom <= 1,05."""
    timeline = build_timeline(
        _analysis_with_unused_panels(), _manifest_with_room(), PANELS_META,
        panels_dir=tmp_path, audio_dir=tmp_path / "audio",
    )
    by_index = {m["index"]: m for m in PANELS_META}
    for clip in timeline.clips:
        meta = by_index[clip.panel_index]
        # La fenetre couvre la case entiere : aucun pixel rogne, ni en largeur ni en hauteur.
        assert clip.crop is not None and clip.crop.fit == "contain"
        assert (clip.crop.x, clip.crop.y) == (0, 0)
        assert (clip.crop.width, clip.crop.height) == (meta["width"], meta["height"])
    # Le zoom n'est pas porte par le clip : il decoule du mouvement, plafonne a MAX_ZOOM.
    # En long, seul ken_burns est autorise, et son amplitude tient dans le plafond.
    assert {c.motion for c in timeline.clips} == {"ken_burns"}
    assert 1.0 + KEN_BURNS_ZOOM <= MAX_ZOOM + 1e-9
    assert all(c.duration_s >= PacingRules().emphasis_floor_s - 1e-9 for c in timeline.clips)
