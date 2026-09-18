"""Tests du CapCut Draft Builder (``src.modules.capcut_builder``) : brouillon pycapcut valide."""

from __future__ import annotations

import json
import os

import numpy as np
import pytest
from PIL import Image

from src.models.audio import SceneAudio, VoiceoverManifest
from src.models.scene import ChapterAnalysis, Scene
from src.models.timeline import BgmClip, PanelClip
from src.modules.capcut_builder import (
    BACKGROUNDS_DIRNAME,
    KEN_BURNS_END,
    KEN_BURNS_START,
    PUNCH_IN_END,
    TRACK_BGM,
    TRACK_SFX,
    TRACK_VFX,
    CapCutError,
    build_capcut_draft,
    compensated_ranges,
    contiguous_ranges,
    copy_draft,
    loop_ranges,
    panel_placement,
)
from src.modules.timeline_builder import build_timeline
from src.modules.tts_engine import write_wav
from src.utils.audio_assets import ensure_default_sfx

SR = 24000


def _project(tmp_path, *, sound: bool = False, action_heavy: bool = False):
    panels_dir, audio_dir = tmp_path / "panels", tmp_path / "audio"
    panels_dir.mkdir()
    rng = np.random.default_rng(0)
    Image.fromarray(rng.integers(0, 256, size=(500, 720, 3), dtype=np.uint8)).save(panels_dir / "panel_000.png")
    Image.fromarray(rng.integers(0, 256, size=(3000, 720, 3), dtype=np.uint8)).save(panels_dir / "panel_001.png")
    Image.fromarray(rng.integers(0, 256, size=(80, 720, 3), dtype=np.uint8)).save(panels_dir / "panel_002.png")
    tone = (0.3 * np.sin(2 * np.pi * 440 * np.arange(SR) / SR)).astype(np.float32)
    write_wav(audio_dir / "scene_000.wav", np.concatenate([tone, np.zeros(SR * 3 // 10, dtype=np.float32)]))
    write_wav(audio_dir / "scene_001.wav", np.concatenate([tone, tone]))
    meta = [
        {"index": 0, "file": "panel_000.png", "width": 720, "height": 500, "type": "static"},
        {"index": 1, "file": "panel_001.png", "width": 720, "height": 3000, "type": "scroll_vertical"},
        {"index": 2, "file": "panel_002.png", "width": 720, "height": 80, "type": "static"},
    ]
    analysis = ChapterAnalysis(
        series_title="Serie", episode_title="Ep. 0", model="fake", language="en", n_panels=3,
        scenes=[
            Scene(index=0, panel_ids=[0], narration="The hero wakes up. He looks around.", emotion="calm"),
            Scene(index=1, panel_ids=[1, 2], narration="He climbs the tower and wins.", emotion="action",
                  action_heavy_ids=[2] if action_heavy else []),
        ],
    )
    manifest = VoiceoverManifest(
        language="en", lang_code="a", voice="am_fenrir", speed=1.0, padding_s=0.3, sample_rate=SR,
        items=[
            SceneAudio(scene_index=0, file="scene_000.wav", duration_s=1.3, speech_s=1.0, sample_rate=SR, text="a"),
            SceneAudio(scene_index=1, file="scene_001.wav", duration_s=2.0, speech_s=1.7, sample_rate=SR, text="b"),
        ],
        total_duration_s=3.3,
    )
    extra = {}
    if sound:
        import soundfile as sf

        bgm_dir = tmp_path / "bgm"
        bgm_dir.mkdir()
        for mood in ("calm", "action"):
            sf.write(str(bgm_dir / f"{mood}.wav"), np.full(SR, 0.2, dtype=np.float32), SR)  # 1 s : sera bouclee
        extra = {"sfx_files": ensure_default_sfx(tmp_path / "sfx"), "bgm_files": {m: bgm_dir / f"{m}.wav" for m in ("calm", "action")}}
    # min_clip_s=0 : on garde toutes les cases (le test porte sur le placement CapCut, pas le rythme).
    return build_timeline(analysis, manifest, meta, panels_dir=panels_dir, audio_dir=audio_dir, min_clip_s=0, **extra)


def test_panel_placement_native_resolution(tmp_path) -> None:
    timeline = _project(tmp_path)
    static, giant, tiny = timeline.clips
    # 720x500 : contain CapCut = 1080/500 = 2.16 ; affichage natif -> echelle CapCut 1/2.16.
    p = panel_placement(static, timeline)
    assert p["mode"] == "native" and p["target"] == 1.0 and p["scale"] == pytest.approx(1 / 2.16)
    # 720x3000 : plus haute que le cadre -> reduite a 1080/3000 = contain, echelle CapCut 1.0, jamais de defilement.
    assert giant.motion == "ken_burns"
    g = panel_placement(giant, timeline)
    assert g["mode"] == "contain" and g["target"] == pytest.approx(0.36) and g["scale"] == pytest.approx(1.0)
    # 720x80 : contain CapCut limite par la largeur = 1920/720 = 2.667 ; natif -> 0.375 (jamais agrandie).
    t = panel_placement(tiny, timeline)
    assert t["mode"] == "native" and t["scale"] == pytest.approx(720 / 1920)
    # Case plus large que le cadre : reduite a la largeur.
    wide = PanelClip(scene_index=0, panel_index=9, file="x.png", width=3840, height=1000, start_s=0, duration_s=5, motion="ken_burns")
    w = panel_placement(wide, timeline)
    assert w["mode"] == "contain" and w["target"] == pytest.approx(0.5) and w["scale"] == pytest.approx(1.0)
    # Zoom borne a 105 % pour le Ken Burns comme pour le punch-in.
    assert KEN_BURNS_START == 1.0 and KEN_BURNS_END == pytest.approx(1.05) and PUNCH_IN_END == pytest.approx(1.05)


def test_loop_ranges_repeats_short_music() -> None:
    clip = BgmClip(mood="calm", file="x.wav", start_s=0.2, duration_s=2.05)
    assert loop_ranges(clip, 1_000_000) == [(200_000, 1_000_000), (1_200_000, 1_000_000), (2_200_000, 50_000)]
    assert loop_ranges(clip, 10_000_000) == [(200_000, 2_050_000)]  # musique plus longue : un seul segment
    assert loop_ranges(clip, 0) == []


def test_build_capcut_draft_writes_valid_project(tmp_path) -> None:
    timeline = _project(tmp_path)
    out_root = tmp_path / "capcut"
    draft_dir = build_capcut_draft(timeline, out_root, "Serie ep0")
    content_path = draft_dir / "draft_content.json"
    assert draft_dir == out_root / "Serie ep0" and content_path.is_file()
    assert (draft_dir / "draft_meta_info.json").is_file()

    content = json.loads(content_path.read_text(encoding="utf-8"))
    assert content["canvas_config"]["width"] == 1920 and content["canvas_config"]["height"] == 1080
    assert content["duration"] == round(3.3 * 1_000_000)
    tracks = {t["name"]: t for t in content["tracks"]}
    assert {"V1 background", "V2 panels", "A1 voice", "T1 subtitles"} <= set(tracks)
    assert TRACK_SFX not in tracks and TRACK_BGM[0] not in tracks  # aucun son fourni
    assert len(tracks["V1 background"]["segments"]) == 3
    assert len(tracks["V2 panels"]["segments"]) == 3
    assert len(tracks["A1 voice"]["segments"]) == 2
    assert len(tracks["T1 subtitles"]["segments"]) == 4  # blocs de 2-4 mots
    assert tracks["V1 background"]["type"] == "video" and tracks["A1 voice"]["type"] == "audio"
    assert tracks["T1 subtitles"]["type"] == "text"

    # Segments V2 : uniquement des images cles d'echelle (plus de position / defilement), centres.
    v2 = tracks["V2 panels"]["segments"]
    for seg in v2:
        assert sorted(kf["property_type"] for kf in seg["common_keyframes"]) == ["KFTypeScaleX", "KFTypeScaleY"]
        assert seg["clip"]["transform"]["x"] == 0 and seg["clip"]["transform"]["y"] == 0
    # Ken Burns : de 100 % a 105 % de l'echelle native, sur toute la duree du clip.
    zoom_kf = next(kf for kf in v2[0]["common_keyframes"] if kf["property_type"] == "KFTypeScaleX")
    zoom_values = [pt["values"][0] for pt in zoom_kf["keyframe_list"]]
    assert len(zoom_values) == 2 and zoom_values[0] == pytest.approx(1 / 2.16) and zoom_values[1] == pytest.approx(zoom_values[0] * 1.05)
    # Segments V2 contigus, sans chevauchement ni trou, fin = duree totale.
    starts = [seg["target_timerange"]["start"] for seg in v2]
    durations = [seg["target_timerange"]["duration"] for seg in v2]
    assert starts == [round(c.start_s * 1e6) for c in timeline.clips]
    assert all(starts[i] + durations[i] == starts[i + 1] for i in range(len(v2) - 1))
    assert starts[-1] + durations[-1] == round(3.3 * 1e6)
    a1 = tracks["A1 voice"]["segments"]
    assert a1[0]["target_timerange"]["start"] + a1[0]["target_timerange"]["duration"] == a1[1]["target_timerange"]["start"]

    # Materiaux : fonds pre-rendus + cases + WAV, chemins absolus existants.
    videos = content["materials"]["videos"]
    paths = [v["path"] for v in videos]
    assert len(videos) == 6 and all(os.path.isabs(p) and os.path.isfile(p) for p in paths)
    backgrounds = sorted((tmp_path / "panels" / BACKGROUNDS_DIRNAME).glob("bg_*.jpg"))
    assert [b.name for b in backgrounds] == ["bg_000.jpg", "bg_001.jpg", "bg_002.jpg"]
    with Image.open(backgrounds[1]) as bg:
        assert bg.size == (1920, 1080)
    audios = content["materials"]["audios"]
    assert len(audios) == 2 and all(a["path"].endswith(".wav") for a in audios)
    texts = content["materials"]["texts"]
    assert len(texts) == len(tracks["T1 subtitles"]["segments"])
    assert any("hero" in json.dumps(t) for t in texts)
    # Style : gras, blanc, contour noir fin, centre, positionne en bas.
    style = json.loads(texts[0]["content"])["styles"][0]
    assert style["bold"] is True and style["fill"]["content"]["solid"]["color"] == [1.0, 1.0, 1.0]
    assert style["strokes"][0]["content"]["solid"]["color"] == [0.0, 0.0, 0.0]
    assert 0 < style["strokes"][0]["width"] <= 0.05
    t1 = tracks["T1 subtitles"]["segments"][0]
    assert t1["clip"]["transform"]["y"] == pytest.approx(-0.78) and t1["clip"]["transform"]["x"] == 0

    # Regeneration : brouillon remplace, fonds reutilises (meme mtime).
    mtime = backgrounds[0].stat().st_mtime_ns
    build_capcut_draft(timeline, out_root, "Serie ep0")
    assert backgrounds[0].stat().st_mtime_ns == mtime

    copied = copy_draft(draft_dir, tmp_path / "CapCutProjects")
    assert (copied / "draft_content.json").is_file()


def test_build_capcut_draft_sound_design_loops_bgm_and_punch_in(tmp_path) -> None:
    timeline = _project(tmp_path, sound=True, action_heavy=True)
    assert [c.motion for c in timeline.clips] == ["ken_burns", "ken_burns", "punch_in"]
    assert len(timeline.sfx) == 2 and [b.mood for b in timeline.bgm] == ["calm", "action"]
    draft_dir = build_capcut_draft(timeline, tmp_path / "capcut", "Sound")
    content = json.loads((draft_dir / "draft_content.json").read_text(encoding="utf-8"))
    tracks = {t["name"]: t for t in content["tracks"]}
    assert {TRACK_SFX, TRACK_BGM[0], TRACK_BGM[1]} <= set(tracks)
    # Punch-in : 3 images cles d'echelle (depart, +5 % a 0,2 s, maintien jusqu'a la fin).
    punch = tracks["V2 panels"]["segments"][2]
    kf = next(k for k in punch["common_keyframes"] if k["property_type"] == "KFTypeScaleX")
    times = [pt["time_offset"] for pt in kf["keyframe_list"]]
    values = [pt["values"][0] for pt in kf["keyframe_list"]]
    assert times[0] == 0 and times[1] == 200_000 and times[2] == punch["target_timerange"]["duration"]
    assert values[1] == pytest.approx(values[0] * 1.05) and values[2] == values[1]
    # Bruitages : un segment par transition de la scene d'action, volume < 1 (gain -12 dB).
    sfx_segments = tracks[TRACK_SFX]["segments"]
    assert [s["target_timerange"]["start"] for s in sfx_segments] == [round(c.start_s * 1e6) for c in timeline.sfx]
    assert all(0 < s["volume"] < 1 for s in sfx_segments)
    # Musique de 1 s bouclee : segments contigus couvrant chaque ambiance, sur des pistes alternees, a -22 dB.
    calm, action = timeline.bgm
    for track_name, clip in zip(TRACK_BGM, (calm, action)):
        segments = tracks[track_name]["segments"]
        assert len(segments) == 3
        assert segments[0]["target_timerange"]["start"] == round(clip.start_s * 1e6)
        for a, b in zip(segments, segments[1:]):
            assert a["target_timerange"]["start"] + a["target_timerange"]["duration"] == b["target_timerange"]["start"]
        last = segments[-1]
        assert last["target_timerange"]["start"] + last["target_timerange"]["duration"] == round(clip.end_s * 1e6)
        assert all(s["volume"] == pytest.approx(10 ** (-22 / 20)) for s in segments)
        assert all(s["source_timerange"]["start"] == 0 for s in segments)
    # Fondus : entree sur le premier segment de chaque ambiance, sortie sur le dernier.
    fades = content["materials"]["audio_fades"]
    assert len(fades) == 4
    assert sum(1 for f in fades if f["fade_in_duration"] > 0) == 2 and sum(1 for f in fades if f["fade_out_duration"] > 0) == 2
    audio_paths = [a["path"] for a in content["materials"]["audios"]]
    assert any(p.endswith("impact.wav") for p in audio_paths)
    assert any(p.endswith("calm.wav") for p in audio_paths) and any(p.endswith("action.wav") for p in audio_paths)
    assert all(os.path.isabs(p) and os.path.isfile(p) for p in audio_paths)


def test_copy_draft_falls_back_when_project_is_locked(tmp_path, monkeypatch) -> None:
    import shutil

    draft_dir = tmp_path / "src" / "Serie ep0"
    draft_dir.mkdir(parents=True)
    (draft_dir / "draft_content.json").write_text("{}", encoding="utf-8")
    projects = tmp_path / "CapCutProjects"
    (projects / "Serie ep0").mkdir(parents=True)
    real_rmtree = shutil.rmtree

    def locked_rmtree(path, *args, **kwargs):
        if str(path).endswith("Serie ep0"):  # projet ouvert dans CapCut : fichiers verrouilles
            raise PermissionError(32, "The process cannot access the file because it is being used by another process")
        return real_rmtree(path, *args, **kwargs)

    monkeypatch.setattr(shutil, "rmtree", locked_rmtree)
    copied = copy_draft(draft_dir, projects)
    assert copied == projects / "Serie ep0 (2)" and (copied / "draft_content.json").is_file()
    # Tout est verrouille : erreur explicite (le pipeline la transforme en avertissement).
    monkeypatch.setattr(shutil, "copytree", lambda *a, **k: (_ for _ in ()).throw(PermissionError(32, "locked")))
    with pytest.raises(CapCutError, match="fermer CapCut"):
        copy_draft(draft_dir, projects)


def test_contiguous_ranges_never_overlap() -> None:
    from src.modules.capcut_builder import contiguous_ranges

    # Bornes flottantes dont l'arrondi independant creerait un chevauchement d'1 us.
    clips = []
    t = 0.0
    for i, d in enumerate([8.313311, 38.411689 - 8.313311, 46.725 - 38.411689, 0.1 + 1e-7]):
        clips.append(PanelClip(scene_index=0, panel_index=i, file="p.png", width=10, height=10,
                               start_s=t, duration_s=d, motion="ken_burns"))
        t += d
    ranges = contiguous_ranges(clips)
    for (s1, d1), (s2, _) in zip(ranges, ranges[1:]):
        assert s1 + d1 == s2
    assert all(d > 0 for _, d in ranges)
    assert ranges[-1][0] + ranges[-1][1] == round(t * 1e6)


def test_build_capcut_draft_errors(tmp_path) -> None:
    timeline = _project(tmp_path)
    broken = timeline.model_copy(update={"clips": []})
    with pytest.raises(CapCutError, match="vide"):
        build_capcut_draft(broken, tmp_path / "c", "x")
    missing = timeline.model_copy(update={"panels_dir": str(tmp_path / "nope")})
    with pytest.raises(CapCutError, match="introuvable"):
        build_capcut_draft(missing, tmp_path / "c", "x")


# --- Dynamisme ---------------------------------------------------------------------------------
def _dynamic_project(tmp_path):
    """Deux scenes de 3 s (calm puis action) : assez longues pour une transition et un effet."""
    panels_dir, audio_dir = tmp_path / "panels", tmp_path / "audio"
    panels_dir.mkdir()
    rng = np.random.default_rng(1)
    for i in range(2):
        Image.fromarray(rng.integers(0, 256, size=(600, 720, 3), dtype=np.uint8)).save(panels_dir / f"panel_00{i}.png")
    tone = (0.3 * np.sin(2 * np.pi * 440 * np.arange(3 * SR) / SR)).astype(np.float32)
    write_wav(audio_dir / "scene_000.wav", tone)
    write_wav(audio_dir / "scene_001.wav", tone * 0.5)
    meta = [{"index": i, "file": f"panel_00{i}.png", "width": 720, "height": 600, "type": "static"} for i in range(2)]
    analysis = ChapterAnalysis(
        series_title="S", episode_title="E", model="fake", language="en", n_panels=2,
        scenes=[
            Scene(index=0, panel_ids=[0], narration="The hero wakes up slowly.", emotion="calm"),
            Scene(index=1, panel_ids=[1], narration="He strikes the beast hard.", emotion="action"),
        ],
    )
    manifest = VoiceoverManifest(
        language="en", lang_code="a", voice="am_puck", speed=1.0, padding_s=0.18, sample_rate=SR,
        items=[
            SceneAudio(scene_index=0, file="scene_000.wav", duration_s=3.0, speech_s=2.8, sample_rate=SR, text="a"),
            SceneAudio(scene_index=1, file="scene_001.wav", duration_s=3.0, speech_s=2.8, sample_rate=SR, text="b"),
        ],
        total_duration_s=6.0,
    )
    return build_timeline(analysis, manifest, meta, panels_dir=panels_dir, audio_dir=audio_dir)


def test_draft_carries_transitions_animations_and_effects(tmp_path) -> None:
    """Non-regression : la transition doit etre attachee AVANT ``add_segment``.

    C'est l'insertion dans le script qui collecte ``segment.transition`` dans les
    materials ; l'attacher apres laissait un brouillon sans aucune transition, sans
    lever la moindre erreur.
    """
    timeline = _dynamic_project(tmp_path)
    assert timeline.n_transitions == 1 and timeline.vfx  # le projet produit bien du dynamisme
    draft = build_capcut_draft(timeline, tmp_path / "c", "dyn")
    content = json.loads((draft / "draft_content.json").read_text(encoding="utf-8"))
    materials, tracks = content["materials"], content["tracks"]

    # Une transition par piste video (fond + case) : sinon le fond couperait net.
    transitions = materials["transitions"]
    assert len(transitions) == 2
    assert {t["name"] for t in transitions} == {"快速挥动"}
    assert {t["duration"] for t in transitions} == {300_000} and all(t["is_overlap"] for t in transitions)
    ids = {t["id"] for t in transitions}
    carriers = [s for tr in tracks for s in tr["segments"] if ids & set(s.get("extra_material_refs", []))]
    assert len(carriers) == 2

    # Animations de sous-titres : une entree par bloc, la boucle en plus sur la scene action.
    kinds = [a["type"] for group in materials["material_animations"] for a in group["animations"]]
    assert kinds.count("in") == len(timeline.subtitles) and kinds.count("loop") >= 1

    # Effets superposes sur une piste d'effets dediee.
    assert [e["name"] for e in materials["video_effects"]] == ["冲刺"]
    effect_tracks = [tr for tr in tracks if tr["type"] == "effect"]
    assert len(effect_tracks) == 1 and len(effect_tracks[0]["segments"]) == 1

    # Le montage sobre n'ecrit rien de tout cela.
    plain = build_capcut_draft(timeline, tmp_path / "c2", "plain", dynamics=False)
    plain_content = json.loads((plain / "draft_content.json").read_text(encoding="utf-8"))
    assert plain_content["materials"]["transitions"] == []
    assert plain_content["materials"]["video_effects"] == []
    assert not any(g["animations"] for g in plain_content["materials"]["material_animations"])
    assert not [tr for tr in plain_content["tracks"] if tr["type"] == "effect"]


def test_compensated_ranges_absorbs_transition_overlap(tmp_path) -> None:
    timeline = _dynamic_project(tmp_path)
    clips = timeline.clips
    plain = compensated_ranges(clips, "none")
    shifted = compensated_ranges(clips, "shift")
    # Sans compensation, les bornes sont contigues et inchangees.
    assert plain == contiguous_ranges(clips)
    # Avec compensation, les cases qui suivent une transition sont declarees plus tard de
    # sa duree, pour que le recouvrement applique par CapCut les ramene a leur place.
    assert shifted[0] == plain[0]
    assert shifted[1][0] == plain[1][0] + 300_000
    assert [d for _, d in shifted] == [d for _, d in plain]  # les durees ne changent pas
    with pytest.raises(CapCutError, match="Compensation inconnue"):
        compensated_ranges(clips, "n'importe quoi")


def test_concurrent_drafts_do_not_corrupt_each_other(tmp_path) -> None:
    """Non-regression : deux brouillons construits en parallele doivent aboutir.

    ``pycapcut`` sonde chaque media avec ``pymediainfo``, qui n'est pas sur en
    concurrence : sans serialisation, deux constructions simultanees echouaient en
    ``ParseError: syntax error: line 1, column 0`` ou en « fichier sans piste video ni
    image » sur des PNG pourtant valides.
    """
    import threading

    timeline = _dynamic_project(tmp_path)
    results: list[object] = []

    def build(index: int) -> None:
        try:
            results.append(build_capcut_draft(timeline, tmp_path / f"draft{index}", f"p{index}"))
        except Exception as exc:  # noqa: BLE001 - on veut l'erreur, pas l'arret du thread
            results.append(exc)

    threads = [threading.Thread(target=build, args=(i,)) for i in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=120)
    assert len(results) == 4
    failures = [r for r in results if isinstance(r, Exception)]
    assert not failures, f"Echec(s) en concurrence : {failures}"
    assert all((tmp_path / f"draft{i}" / f"p{i}" / "draft_content.json").is_file() for i in range(4))
