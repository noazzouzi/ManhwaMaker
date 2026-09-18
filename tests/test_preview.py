"""Tests du rendu de prévisualisation (``src.modules.preview_renderer``) : composition et encodage ffmpeg."""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from src.models.audio import SceneAudio, VoiceoverManifest
from src.models.scene import ChapterAnalysis, Scene
from src.models.timeline import BgmClip, CueAnimation, SfxClip, SubtitleCue
from src.modules.preview_renderer import (
    PreviewError,
    PreviewRenderer,
    apply_cue_state,
    apply_vfx,
    back_out,
    blend_rgba,
    blend_transition,
    cue_animation_state,
    ease_in_out,
    ease_out,
    load_font,
    make_background,
    mix_background_music,
    mix_bgm_clips,
    mix_sfx_clips,
    native_scale,
    paste,
    render_subtitle_image,
    transition_family,
    zoom_factor,
)
from src.modules.timeline_builder import MAX_ZOOM, PUNCH_IN_S, build_timeline
from src.modules.tts_engine import write_wav

SR = 24000


def _gradient(width: int, height: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    base = np.linspace(0, 255, width, dtype=np.float32)[None, :, None]
    img = np.repeat(np.repeat(base, height, axis=0), 3, axis=2)
    img[..., seed % 3] = rng.integers(0, 256, size=(height, width))
    return img.astype(np.uint8)


def _project(tmp_path):
    """Deux cases (une petite, une geante) + deux segments audio + timeline 640x360 @ 10 fps."""
    panels_dir, audio_dir = tmp_path / "panels", tmp_path / "audio"
    panels_dir.mkdir()
    Image.fromarray(_gradient(400, 200, 1)).save(panels_dir / "panel_000.png")
    Image.fromarray(_gradient(300, 1600, 2)).save(panels_dir / "panel_001.png")
    t = np.arange(int(1.0 * SR)) / SR
    tone = (0.3 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    silence = np.zeros(int(0.3 * SR), dtype=np.float32)
    write_wav(audio_dir / "scene_000.wav", np.concatenate([tone, silence]))
    write_wav(audio_dir / "scene_001.wav", np.concatenate([tone * 0.5, silence]))
    meta = [
        {"index": 0, "file": "panel_000.png", "width": 400, "height": 200, "type": "static"},
        {"index": 1, "file": "panel_001.png", "width": 300, "height": 1600, "type": "scroll_vertical"},
    ]
    analysis = ChapterAnalysis(
        model="fake", language="en", n_panels=2,
        scenes=[
            Scene(index=0, panel_ids=[0], narration="The hero wakes up in a ruined city.", emotion="calm"),
            Scene(index=1, panel_ids=[1], narration="He climbs the tower.", emotion="epic"),
        ],
    )
    manifest = VoiceoverManifest(
        language="en", lang_code="a", voice="am_fenrir", speed=1.0, padding_s=0.3, sample_rate=SR,
        items=[
            SceneAudio(scene_index=0, file="scene_000.wav", duration_s=1.3, speech_s=1.0, sample_rate=SR, text="a"),
            SceneAudio(scene_index=1, file="scene_001.wav", duration_s=1.3, speech_s=1.0, sample_rate=SR, text="b"),
        ],
        total_duration_s=2.6,
    )
    return build_timeline(analysis, manifest, meta, panels_dir=panels_dir, audio_dir=audio_dir, width=640, height=360, fps=10)


def test_composition_helpers() -> None:
    assert ease_in_out(0) == 0 and ease_in_out(1) == 1 and 0.49 < ease_in_out(0.5) < 0.51
    assert ease_in_out(-1) == 0 and ease_in_out(2) == 1
    assert ease_out(0) == 0 and ease_out(1) == 1 and ease_out(0.5) > 0.5
    # Echelle native : jamais agrandie, reduite seulement si la case depasse le cadre.
    assert native_scale(720, 500, 1920, 1080) == 1.0
    assert native_scale(100, 50, 1920, 1080) == 1.0
    assert native_scale(720, 1200, 1920, 1080) == pytest.approx(0.9)
    assert native_scale(3000, 1000, 1920, 1080) == pytest.approx(1920 / 3000)
    assert native_scale(4000, 3000, 1920, 1080) == pytest.approx(0.36)
    # Zoom : Ken Burns 100 -> 105 %, punch-in 105 % en 0,2 s puis maintien ; jamais au-dela de 105 %.
    assert MAX_ZOOM == 1.05
    assert zoom_factor("ken_burns", 0.0, 0.0) == 1.0 and zoom_factor("ken_burns", 1.0, 9.0) == pytest.approx(1.05)
    assert zoom_factor("ken_burns", 1.0, 9.0, zoom=0.5) == 1.05  # amplitude excessive bornee
    assert zoom_factor("punch_in", 0.0, 0.0) == 1.0
    assert 1.0 < zoom_factor("punch_in", 0.01, PUNCH_IN_S / 2) < 1.05
    assert zoom_factor("punch_in", 0.02, PUNCH_IN_S) == pytest.approx(1.05)
    assert zoom_factor("punch_in", 0.9, 4.0) == pytest.approx(1.05)
    assert zoom_factor("scroll_vertical", 1.0, 5.0) == pytest.approx(1.05)  # ancien mouvement : Ken Burns

    rgb = np.full((500, 720, 3), 200, dtype=np.uint8)
    bg = make_background(rgb, 640, 360)
    assert bg.shape == (360, 640, 3) and bg.dtype == np.uint8
    assert abs(float(bg.mean()) - 200 * 0.7) < 3  # assombri de 30 %

    frame = np.zeros((10, 10, 3), dtype=np.uint8)
    paste(frame, np.full((4, 4, 3), 255, dtype=np.uint8), -2, 8)  # deborde en haut-gauche et en bas
    assert frame[8:, :2].min() == 255 and frame[:8].max() == 0 and frame[8:, 2:].max() == 0

    font = load_font(24)
    image = render_subtitle_image("Hello brave new world of webtoons", 300, font)
    assert image.ndim == 3 and image.shape[2] == 4 and image.shape[1] <= 300
    assert image[..., 3].max() == 255 and image[..., 3].min() == 0  # texte opaque, fond transparent
    opaque = image[..., 3] == 255
    assert (image[..., :3][opaque].max(axis=1) == 255).any()  # blanc (texte)
    assert (image[..., :3][opaque].max(axis=1) == 0).any()  # noir (contour net de 2 px)
    assert image[0, 0, 3] == 0 and image[-1, -1, 3] == 0  # pas de bandeau
    frame = np.zeros((200, 320, 3), dtype=np.uint8)
    blend_rgba(frame, image, 10, 100)
    assert frame.max() == 255

    voice = np.zeros(SR, dtype=np.float32)
    music = 0.5 * np.ones(SR // 2, dtype=np.float32)
    import soundfile as sf

    sf.write("_bgm_test.wav", music, 48000)
    try:
        mixed = mix_background_music(voice, SR, "_bgm_test.wav", -22.0)
    finally:
        import os

        os.remove("_bgm_test.wav")
    assert mixed.shape == voice.shape and abs(float(mixed.max()) - 0.5 * 10 ** (-22 / 20)) < 0.01


def test_mix_sfx_and_bgm_clips(tmp_path) -> None:
    import soundfile as sf

    sfx_path, bgm_path = tmp_path / "impact.wav", tmp_path / "calm.wav"
    sf.write(str(sfx_path), np.full(SR // 10, 0.5, dtype=np.float32), SR)  # 0,1 s a 0,5
    sf.write(str(bgm_path), np.full(SR, 0.4, dtype=np.float32), SR)  # 1 s a 0,4
    voice = np.zeros(2 * SR, dtype=np.float32)
    mixed = mix_sfx_clips(voice, SR, [SfxClip(scene_index=0, panel_index=0, start_s=1.0, kind="impact", file=str(sfx_path), gain_db=-6.0)])
    assert mixed[: SR - 1].max() == 0 and mixed[SR + 10] == pytest.approx(0.5 * 10 ** (-6 / 20), abs=1e-3)
    assert mixed[SR + SR // 10 + 5] == 0
    # Musique bouclee (1 s -> 1,6 s), fondu d'entree 0,4 s et de sortie 0,4 s, gain -20 dB.
    clip = BgmClip(mood="calm", file=str(bgm_path), start_s=0.2, duration_s=1.6, gain_db=-20.0, fade_in_s=0.4, fade_out_s=0.4)
    mixed = mix_bgm_clips(voice, SR, [clip])
    level = 0.4 * 10 ** (-20 / 20)
    assert mixed[int(0.1 * SR)] == 0 and mixed[int(0.2 * SR) + 1] < level / 10
    assert mixed[int(1.0 * SR)] == pytest.approx(level, abs=1e-3)  # plateau
    assert mixed[int(1.3 * SR)] == pytest.approx(level, abs=1e-3)  # apres le point de boucle (1,2 s)
    assert mixed[int(1.6 * SR)] == pytest.approx(level / 2, abs=2e-3)  # milieu du fondu de sortie (1,4 -> 1,8 s)
    assert mixed[int(1.8 * SR) - 2] < level / 10 and mixed[int(1.9 * SR)] == 0


def test_renderer_frames_native_and_contain(tmp_path) -> None:
    timeline = _project(tmp_path)
    renderer = PreviewRenderer(timeline)
    small = renderer.prepare_clip(timeline.clips[0])
    giant = renderer.prepare_clip(timeline.clips[1])
    # 400x200 dans 640x360 : resolution native, pixels intacts ; 300x1600 : reduite a la hauteur du cadre.
    assert small["mode"] == "native" and small["scale"] == 1.0
    with Image.open(tmp_path / "panels" / "panel_000.png") as img:
        assert np.array_equal(small["fg"], np.asarray(img.convert("RGB")))
    assert giant["mode"] == "contain" and giant["scale"] == pytest.approx(360 / 1600)
    assert giant["fg"].shape[0] == 360 and giant["fg"].shape[1] == round(300 * 360 / 1600)
    first = renderer.compose(small, 0.0)
    last = renderer.compose(small, 1.0)
    assert first.shape == (360, 640, 3) and not np.array_equal(first, last)  # Ken Burns
    # Case centree, fond floute visible de part et d'autre.
    fg_rows = np.any(first[:, :, :] != renderer.compose(small, 0.0, None)[:, :, :], axis=2)
    assert not fg_rows.any()
    frames = list(renderer.frames())
    assert len(frames) == 26  # 2.6 s a 10 fps
    assert all(f.shape == (360, 640, 3) and f.dtype == np.uint8 for f in frames)
    assert len(list(renderer.frames(max_duration_s=1.0))) == 10

    # Punch-in : 105 % des 0,2 s, puis image stable.
    punch = renderer.prepare_clip(timeline.clips[0].model_copy(update={"motion": "punch_in"}))
    start = renderer.compose(punch, 0.0, elapsed_s=0.0)
    hit = renderer.compose(punch, 0.2, elapsed_s=PUNCH_IN_S)
    hold = renderer.compose(punch, 0.8, elapsed_s=1.0)
    assert not np.array_equal(start, hit) and np.array_equal(hit, hold)
    # Sous-titre : bloc dessine en bas au centre.
    cue = SubtitleCue(scene_index=0, start_s=0.0, end_s=1.0, text="Hero wakes")
    with_cue = renderer.compose(small, 0.5, cue)
    without = renderer.compose(small, 0.5)
    diff = np.argwhere(np.any(with_cue != without, axis=2))
    assert len(diff) and diff[:, 0].min() > 180 and 200 < diff[:, 1].mean() < 440


def test_render_mp4_with_audio(tmp_path) -> None:
    import imageio_ffmpeg

    timeline = _project(tmp_path)
    out = PreviewRenderer(timeline).render(tmp_path / "out" / "preview.mp4", max_duration_s=2.0)
    assert out.exists() and out.stat().st_size > 1000
    assert not (tmp_path / "out" / "preview_audio.wav").exists()  # WAV temporaire nettoye
    reader = imageio_ffmpeg.read_frames(str(out))
    meta = next(reader)
    assert meta["size"] == (640, 360) and abs(meta["fps"] - 10) < 0.01
    n = sum(1 for _ in reader)
    assert n == 20
    assert "audio" in str(meta.get("audio_codec", "aac")).lower() or meta.get("audio_codec") in (None, "aac")

    with pytest.raises(PreviewError):
        PreviewRenderer(timeline).render(tmp_path / "x.mp4", max_duration_s=0.0)


# --- Dynamisme : transitions, effets superposes, sous-titres animes ----------------------------
def _action_project(tmp_path):
    """Meme projet que ``_project``, mais scene 1 en ``action`` et scenes assez longues
    pour qu'une transition et un effet superpose soient generes."""
    panels_dir, audio_dir = tmp_path / "panels", tmp_path / "audio"
    panels_dir.mkdir()
    Image.fromarray(_gradient(400, 200, 1)).save(panels_dir / "panel_000.png")
    Image.fromarray(_gradient(380, 240, 2)).save(panels_dir / "panel_001.png")
    tone = (0.3 * np.sin(2 * np.pi * 440 * np.arange(int(2.6 * SR)) / SR)).astype(np.float32)
    write_wav(audio_dir / "scene_000.wav", tone)
    write_wav(audio_dir / "scene_001.wav", tone * 0.5)
    meta = [
        {"index": 0, "file": "panel_000.png", "width": 400, "height": 200, "type": "static"},
        {"index": 1, "file": "panel_001.png", "width": 380, "height": 240, "type": "static"},
    ]
    analysis = ChapterAnalysis(
        model="fake", language="en", n_panels=2,
        scenes=[
            Scene(index=0, panel_ids=[0], narration="The hero wakes up.", emotion="calm"),
            Scene(index=1, panel_ids=[1], narration="He strikes the beast.", emotion="action"),
        ],
    )
    manifest = VoiceoverManifest(
        language="en", lang_code="a", voice="am_puck", speed=1.0, padding_s=0.18, sample_rate=SR,
        items=[
            SceneAudio(scene_index=0, file="scene_000.wav", duration_s=2.6, speech_s=2.4, sample_rate=SR, text="a"),
            SceneAudio(scene_index=1, file="scene_001.wav", duration_s=2.6, speech_s=2.4, sample_rate=SR, text="b"),
        ],
        total_duration_s=5.2,
    )
    return build_timeline(analysis, manifest, meta, panels_dir=panels_dir, audio_dir=audio_dir, width=640, height=360, fps=10)


def test_blend_transition_families() -> None:
    before = np.zeros((40, 60, 3), dtype=np.uint8)
    after = np.full((40, 60, 3), 200, dtype=np.uint8)
    assert transition_family("快速挥动") == "whip" and transition_family("故障") == "glitch"
    assert transition_family("inconnue") == "dissolve"  # repli
    # Fondu : les extremites rendent les images d'origine, le milieu un melange.
    assert np.array_equal(blend_transition("dissolve", before, after, 0.0), before)
    assert np.array_equal(blend_transition("dissolve", before, after, 1.0), after)
    assert 80 < blend_transition("dissolve", before, after, 0.5).mean() < 120
    # Flash noir : le milieu est plus sombre que les deux images.
    assert blend_transition("flash_black", before, after, 0.5).mean() < 10
    assert blend_transition("flash_white", before, after, 0.5).mean() > 240
    # Balayage : a mi-course, la gauche vient de l'image sortante et la droite de l'entrante.
    whip = blend_transition("whip", before, after, 0.5)
    assert whip[:, -1].mean() > whip[:, 0].mean()
    # Glitch : l'image est deformee, donc differente des deux sources. Il faut une image
    # texturee : decaler des canaux ou des bandes d'une image uniforme la laisse identique.
    textured_a, textured_b = _gradient(60, 40, 1), _gradient(60, 40, 2)
    glitch = blend_transition("glitch", textured_a, textured_b, 0.5)
    assert not np.array_equal(glitch, textured_a) and not np.array_equal(glitch, textured_b)
    assert blend_transition("glitch", textured_a, textured_b, 0.0).shape == textured_a.shape
    # Bornes : un avancement hors [0, 1] est ramene dans l'intervalle.
    assert np.array_equal(blend_transition("dissolve", before, after, -1.0), before)


def test_apply_vfx_lightens_in_place() -> None:
    for family in ("glow", "speed_lines", "rain", "sparks"):
        frame = np.full((180, 320, 3), 60, dtype=np.uint8)
        apply_vfx(frame, family, 0.5)
        assert frame.mean() > 60.0, family  # melange additif : l'image s'eclaircit
        assert frame.max() <= 255  # addition saturante, jamais de debordement
    # Intensite nulle ou famille inconnue : image intacte.
    frame = np.full((180, 320, 3), 60, dtype=np.uint8)
    apply_vfx(frame, "speed_lines", 0.5, strength=0.0)
    apply_vfx(frame, "inconnue", 0.5)
    assert frame.mean() == 60.0


def test_cue_animation_state_runs_intro_then_loop() -> None:
    assert back_out(0.0) == pytest.approx(0.0) and back_out(1.0) == pytest.approx(1.0)
    assert max(back_out(p / 100) for p in range(101)) > 1.0  # depassement (rebond)
    # Fondu : l'opacite monte pendant l'entree puis reste pleine.
    assert cue_animation_state("渐显", "", 0.4, 0.0)["alpha"] == pytest.approx(0.0)
    assert cue_animation_state("渐显", "", 0.4, 0.2)["alpha"] == pytest.approx(0.5)
    assert cue_animation_state("渐显", "", 0.4, 1.0)["alpha"] == 1.0
    # Rebond : part plus petit que la taille finale.
    assert cue_animation_state("弹入", "", 0.4, 0.0)["scale"] == pytest.approx(0.6)
    assert cue_animation_state("弹入", "", 0.4, 0.4)["scale"] == pytest.approx(1.0)
    # Mot a mot : la fraction revelee suit l'avancement.
    assert cue_animation_state("逐字", "", 0.4, 0.1)["reveal"] == pytest.approx(0.25)
    assert cue_animation_state("逐字", "", 0.4, 0.9)["reveal"] == 1.0
    # Glitch : deformation decroissante, nulle une fois l'entree terminee.
    assert cue_animation_state("故障", "", 0.4, 0.0)["glitch"] == pytest.approx(1.0)
    assert cue_animation_state("故障", "", 0.4, 0.5)["glitch"] == 0.0
    # Boucle : elle ne demarre qu'apres l'entree, puis fait osciller l'echelle.
    # (Pendant l'entree, le rebond depasse brievement 1 : c'est l'effet recherche.)
    assert cue_animation_state("弹入", "心跳", 0.4, 0.4)["scale"] == pytest.approx(1.0)  # jonction
    peak = cue_animation_state("弹入", "心跳", 0.4, 0.4 + 0.9 / 4)["scale"]
    trough = cue_animation_state("弹入", "心跳", 0.4, 0.4 + 3 * 0.9 / 4)["scale"]
    assert peak > 1.0 > trough and peak == pytest.approx(1.035) and trough == pytest.approx(0.965)
    shake = cue_animation_state("渐显", "颤抖", 0.2, 0.2 + 0.9 / 12, ratio=2.0)
    assert shake["dx"] != 0
    # Aucune animation declaree : etat neutre.
    assert cue_animation_state("", "", 0.0, 1.0) == {
        "scale": 1.0, "alpha": 1.0, "dx": 0, "dy": 0, "reveal": 1.0, "glitch": 0.0,
    }


def test_apply_cue_state_never_mutates_the_cached_image() -> None:
    rgba = np.full((20, 40, 4), 200, dtype=np.uint8)
    original = rgba.copy()
    faded = apply_cue_state(rgba, {"alpha": 0.5})
    assert np.array_equal(rgba, original)  # l'image en cache reste intacte
    assert faded[:, :, 3].mean() == pytest.approx(100, abs=1)
    # Mot a mot : la partie droite est rendue transparente, le bloc ne bouge pas.
    revealed = apply_cue_state(rgba, {"reveal": 0.5})
    assert revealed.shape == rgba.shape
    assert revealed[:, :20, 3].mean() == 200 and revealed[:, 20:, 3].mean() == 0
    # Echelle : l'image est redimensionnee.
    assert apply_cue_state(rgba, {"scale": 0.5}).shape[1] == 20
    assert np.array_equal(apply_cue_state(rgba, {}), rgba)


def test_transitions_do_not_shift_the_timeline(tmp_path) -> None:
    """Garde-fou central : une transition se joue DANS le clip sortant, sans decaler la voix."""
    timeline = _action_project(tmp_path)
    assert timeline.n_transitions == 1 and timeline.vfx  # scene action : transition + effet
    transition = next(c.transition for c in timeline.clips if c.transition)

    animated = PreviewRenderer(timeline)
    sober = PreviewRenderer(timeline, dynamics=False)
    frames_animated = list(animated.frames())
    frames_sober = list(sober.frames())
    # Meme nombre d'images, donc meme duree : aucun glissement image / voix.
    assert len(frames_animated) == len(frames_sober) == int(round(timeline.total_duration_s * timeline.fps))

    # Hors fenetre de transition et hors effet, les deux rendus coincident.
    cut = next(c for c in timeline.clips if c.transition)
    begin = cut.end_s - transition.duration_s
    early = int(0.2 * timeline.fps)
    assert np.array_equal(frames_animated[early], frames_sober[early]) or timeline.subtitles
    # Pendant la transition, l'image differe de la composition simple.
    mid = int((begin + transition.duration_s / 2) * timeline.fps)
    assert not np.array_equal(frames_animated[mid], frames_sober[mid])
    # Apres la transition, la case suivante est bien a l'ecran dans les deux cas.
    after = int((cut.end_s + 0.3) * timeline.fps)
    assert np.array_equal(frames_animated[after], frames_sober[after]) or timeline.vfx


def test_renderer_draws_animated_subtitles(tmp_path) -> None:
    timeline = _action_project(tmp_path)
    renderer = PreviewRenderer(timeline)
    prepared = renderer.prepare_clip(timeline.clips[0])
    cue = SubtitleCue(
        scene_index=0, start_s=0.0, end_s=1.0, text="Hero wakes",
        animation=CueAnimation(intro="渐显", loop="", duration_s=0.4),
    )
    # Au debut de l'animation le bloc est encore transparent, a la fin il est pleinement dessine.
    start = renderer.compose(prepared, 0.2, cue, cue_elapsed_s=0.0)
    settled = renderer.compose(prepared, 0.2, cue, cue_elapsed_s=1.0)
    plain = renderer.compose(prepared, 0.2)
    assert np.array_equal(start, plain)  # opacite nulle : rien n'est dessine
    assert not np.array_equal(settled, plain)
    # Sans ``cue_elapsed_s``, le bloc est dessine dans son etat stabilise (comportement d'origine).
    assert np.array_equal(renderer.compose(prepared, 0.2, cue), settled)
    # Dynamisme coupe : l'animation est ignoree, le bloc est plein des la premiere image.
    assert np.array_equal(
        PreviewRenderer(timeline, dynamics=False).compose(prepared, 0.2, cue, cue_elapsed_s=0.0), settled
    )
