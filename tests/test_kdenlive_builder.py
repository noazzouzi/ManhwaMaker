"""Projet Kdenlive / rendu melt tiré de la timeline (essai)."""

from __future__ import annotations

import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
from PIL import Image

from src.models.timeline import AudioClip, CueAnimation, PanelClip, SubtitleCue, Timeline, VfxClip
from src.modules import kdenlive_builder as kb
from src.modules.kdenlive_style import Cut


def timeline(tmp_path: Path) -> Timeline:
    """Trois cases de 2 s (scènes 0, 0, 1 ; la dernière en punch-in), voix, sous-titres et une lueur."""
    panels = tmp_path / "panels"
    panels.mkdir()
    for i, color in enumerate([(200, 40, 40), (40, 200, 40), (40, 40, 200)]):
        Image.new("RGB", (120, 90), color).save(panels / f"panel_{i:03d}.png")
    audio = tmp_path / "audio"
    audio.mkdir()
    sf.write(audio / "scene_000.wav", np.zeros(24000 * 6, dtype=np.float32), 24000)
    clips = [PanelClip(scene_index=0 if i < 2 else 1, panel_index=i, file=f"panel_{i:03d}.png", width=120, height=90,
                       start_s=2.0 * i, duration_s=2.0, motion="punch_in" if i == 2 else "ken_burns")
             for i in range(3)]
    cues = [SubtitleCue(scene_index=0, start_s=0.0, end_s=1.0, text="Our journey begins",
                        animation=CueAnimation(intro="弹入", duration_s=0.3)),
            SubtitleCue(scene_index=0, start_s=1.0, end_s=2.0, text="in blood")]
    return Timeline(width=320, height=180, fps=10, panels_dir=str(panels), audio_dir=str(audio), clips=clips,
                    audio=[AudioClip(scene_index=0, file="scene_000.wav", start_s=0.0, duration_s=6.0)],
                    subtitles=cues, total_duration_s=6.0,
                    vfx=[VfxClip(scene_index=0, kind="glow", start_s=1.0, duration_s=3.0, opacity=0.85)])


def services(tree: ET.Element, tag: str) -> list[str]:
    return [e.find("property[@name='mlt_service']").text for e in tree.iter(tag)]


def prop(tree: ET.Element, tag: str, service: str, name: str) -> list[str]:
    return [e.find(f"property[@name='{name}']").text for e in tree.iter(tag)
            if e.find("property[@name='mlt_service']").text == service]


def test_layout_overlaps_a_mix_and_keeps_a_dip_on_the_cut(tmp_path) -> None:
    tl = timeline(tmp_path)
    cuts = [Cut("dissolve", 0.4), Cut("dip_black", 0.4, scene_change=True)]
    placed, mixes = kb._layout(kb._Doc(tl), tl, [Path("a.png")] * 3, cuts)
    # Fondu : la 2e case passe sur l'autre sous-piste et chevauche la 1re de 4 images, centrées sur la coupe.
    # Après le fondu au noir (coupe franche), retour sur la 1re sous-piste : Kdenlive l'exige.
    assert [p.sub for p in placed] == [0, 1, 0]
    assert (placed[0].end, placed[1].start) == (22, 18)
    assert mixes == [{"in": 18, "out": 21, "reverse": 0, "mixcut": 2, "cut": cuts[0]}]
    # Fondu au noir : coupe franche, effet de part et d'autre, aucune image décalée.
    assert (placed[1].end, placed[2].start) == (40, 40)
    assert placed[1].cut_out is placed[2].cut_in is cuts[1]


def test_project_opens_in_kdenlive_order_and_render_copy_plays_the_timeline(tmp_path) -> None:
    project = kb.build_project(timeline(tmp_path), tmp_path / "kd", name="essai", emotions=["calm", "calm", "tension"])
    root = ET.parse(project.project).getroot()
    assert root.get("producer") == "main_bin" and "producer" not in ET.parse(project.render_mlt).getroot().attrib
    defined: set[str] = set()
    for element in root:  # chaque producteur, sous-piste ou piste est défini avant d'être référencé
        for ref in [e.get("producer") for e in element.iter() if e.tag in ("entry", "track")]:
            assert ref in defined, ref
        defined.add(element.get("id"))
    # Une entrée de chutier par fichier : 3 images, voix, lueur, impact ajouté au punch-in, flash blanc.
    assert len(root.find("playlist[@id='main_bin']").findall("entry")) == 7
    # Coupe dans une scène calme : fondu ; changement de scène vers la tension : glitch des deux côtés.
    assert services(root, "transition").count("luma") == 1
    assert services(root, "filter").count("frei0r.glitch0r") == 2
    # Mouvements variés, 105 % au plus : zoom vers le centre, puis vers le bord gauche, puis punch-in qui tremble.
    rects = prop(root, "filter", "qtblend", "rect")
    assert rects[0].startswith("0i=0 0 320 180 1;") and rects[0].endswith("=-8 -4 336 189 1")
    assert any(r.startswith("0i=0 0 320 180 1;") and r.endswith("=0 -4 336 189 1") for r in rects)
    assert any(r.startswith("0h=0 0 320 180 1;2=-8 -4 336 189 1;") and r.count(";") >= 3 for r in rects)
    # Impact : flash blanc qui s'éteint (piste V3), lueur qui bat (piste V2).
    assert "0h=0 0 320 180 0.7;1=0 0 320 180 0" in rects
    assert any(r.startswith("0~=0 0 320 180 ") and ";25~=0 0 320 180 " in r for r in rects)
    # Sous-titres : dans le rendu melt seulement (Kdenlive plante sur un filtre écrit à la main).
    assert "avfilter.subtitles" not in services(root, "filter")
    assert "avfilter.subtitles" in services(ET.parse(project.render_mlt).getroot(), "filter")
    ass = project.subtitles.read_text(encoding="utf-8-sig")
    assert "Dialogue: 0,0:00:00.00,0:00:01.00,Default,,0,0,0,,{\\fscx60\\fscy60" in ass and ",in blood" in ass


def test_push_mix_runs_backwards_when_the_incoming_clip_is_underneath(tmp_path) -> None:
    track, doc = ET.Element("tractor"), kb._Doc(timeline(tmp_path))
    kb._mix(track, doc, {"in": 18, "out": 21, "reverse": 1, "mixcut": 2, "cut": Cut("push", 0.4, "left")})
    transition = track.find("transition")
    assert transition.find("property[@name='mlt_service']").text == "frei0r.sleid0r_push-right"
    assert transition.find("property[@name='position']").text == "0=1;3=0"


@pytest.mark.skipif(not (kb.KDENLIVE_BIN / "melt.exe").is_file(), reason="Kdenlive (melt) non installe")
def test_melt_renders_the_project(tmp_path) -> None:
    project = kb.build_project(timeline(tmp_path), tmp_path / "kd", name="essai", emotions=["action"] * 3)
    out = tmp_path / "out.mp4"
    kb.render(project, out, vcodec="libx264", threads=2, fps=10)
    probe = subprocess.run([str(kb.KDENLIVE_BIN / "ffprobe.exe"), "-v", "error", "-show_entries", "format=duration",
                            "-of", "csv=p=0", str(out)], capture_output=True, text=True)
    assert abs(float(probe.stdout) - 6.0) < 0.2


def test_no_clip_sits_on_the_second_subtrack_without_a_mix_in(tmp_path) -> None:
    tl = timeline(tmp_path)
    tl = tl.model_copy(update={"clips": tl.clips + [tl.clips[2].model_copy(update={"start_s": 6.0, "panel_index": 3})],
                               "total_duration_s": 8.0})
    cuts = [Cut("dissolve", 0.4), None, Cut("push", 0.4, "left")]
    placed, mixes = kb._layout(kb._Doc(tl), tl, [Path("a.png")] * 4, cuts)
    # Kdenlive « corrige » à l'ouverture toute case de la 2e sous-piste qui n'entre pas par un fondu.
    assert [p.sub for p in placed] == [0, 1, 0, 1]
    assert all(p.sub == 0 or (p.cut_in is not None and p.cut_in.kind in kb.MIX_KINDS) for p in placed)


def compilation(tmp_path: Path, chapters: int = 3, clips_per_chapter: int = 2, clip_s: float = 2.0) -> Timeline:
    """Compilation factice : fichiers ``serie_epN__panel_k.png``, une scène par case."""
    panels = tmp_path / "media"
    panels.mkdir()
    clips = []
    for c in range(chapters):
        for k in range(clips_per_chapter):
            name = f"serie_ep{c + 1}__panel_{k:03d}.png"
            Image.new("RGB", (120, 90), (60 * c, 100, 40 * k)).save(panels / name)
            i = len(clips)
            clips.append(PanelClip(scene_index=i, panel_index=i, file=name, width=120, height=90, start_s=clip_s * i,
                                   duration_s=clip_s, motion="ken_burns"))
    audio = tmp_path / "audio"
    audio.mkdir()
    total = clip_s * len(clips)
    t = np.arange(int(24000 * total)) / 24000
    sf.write(audio / "voice.wav", (0.3 * np.sin(2 * np.pi * 220 * t)).astype(np.float32), 24000)
    cues = [SubtitleCue(scene_index=0, start_s=clip_s * i + 0.2, end_s=clip_s * i + 1.5, text=f"cue {i}") for i in range(len(clips))]
    return Timeline(width=320, height=180, fps=10, panels_dir=str(panels), audio_dir=str(audio), clips=clips,
                    audio=[AudioClip(scene_index=0, file="voice.wav", start_s=0.0, duration_s=total)], subtitles=cues,
                    total_duration_s=total)


def test_parts_are_cut_at_chapters_frame_exact_and_never_through_a_mix(tmp_path) -> None:
    tl = compilation(tmp_path)
    chapters = kb.chapter_starts(tl)
    assert chapters == frozenset({2, 4})
    assert kb.plan_parts(tl) == [0]  # 12 s : bien en dessous du seuil
    starts = kb.plan_parts(tl, chapters=chapters, target_s=3.0, from_s=5.0)
    assert starts == [0, 2, 4]
    cuts = kb.part_boundary_cuts([Cut("dissolve", 0.4)] * 5, starts)
    assert [c.kind for c in cuts] == ["dissolve", "dip_black", "dissolve", "dip_black", "dissolve"]
    parts = [kb.slice_timeline(tl, i0, i1) for i0, i1 in zip(starts, starts[1:] + [len(tl.clips)])]
    assert [round(p.total_duration_s * tl.fps) for p in parts] == [40, 40, 40]
    assert parts[1].clips[0].start_s == 0.0 and parts[1].clips[-1].end_s == pytest.approx(4.0)
    assert [c.text for c in parts[2].subtitles] == ["cue 4", "cue 5"] and parts[2].subtitles[0].start_s == pytest.approx(0.2)
    assert parts[0].audio == [] and parts[0].bgm == []  # le son est mixé à part, pour toute la vidéo
    excerpt = kb.slice_timeline(tl, 0, 3, end_s=5.0, keep_audio=True)
    assert excerpt.total_duration_s == 5.0 and excerpt.clips[-1].end_s == pytest.approx(5.0)
    assert excerpt.audio[0].duration_s == pytest.approx(5.0)


def test_audio_mix_is_sample_exact_with_looped_faded_music(tmp_path) -> None:
    from src.models.timeline import BgmClip

    tl = compilation(tmp_path)
    loop = tmp_path / "loop.wav"
    sf.write(loop, np.full(48000, 0.5, dtype=np.float32), 48000)  # 1 s, bouclé
    tl = tl.model_copy(update={"bgm": [BgmClip(mood="calm", file=str(loop), start_s=1.0, duration_s=6.0, gain_db=0.0,
                                               fade_in_s=1.0, fade_out_s=1.0)]})
    wav = kb.mix_audio(tl, [None] * 5, tmp_path / "mix.wav", window_s=0.7)
    data, rate = sf.read(wav)
    assert rate == 48000 and len(data) == 12 * 48000  # exactement la durée de la vidéo
    voice = 0.3 * np.sin(2 * np.pi * 220 * np.arange(len(data)) / 48000)
    music = data - voice
    assert abs(music[int(0.5 * 48000)]) < 0.01  # avant la musique
    assert music[int(1.5 * 48000)] == pytest.approx(0.25, abs=0.02)  # mi-fondu d'entrée
    assert music[int(4.0 * 48000)] == pytest.approx(0.5, abs=0.02)  # bouclée, plein volume
    assert abs(music[int(7.5 * 48000)]) < 0.01  # après la musique


def test_an_interrupted_render_resumes_at_the_first_missing_part(tmp_path, monkeypatch) -> None:
    tl = compilation(tmp_path)
    starts = [0, 2, 4]
    cuts = kb.part_boundary_cuts([Cut("dissolve", 0.4)] * 5, starts)
    rendered: list[str] = []

    def fake_melt(mlt, out, **kwargs):
        rendered.append(Path(mlt).parent.name)
        if len(rendered) == 2 and not hasattr(fake_melt, "crashed"):
            fake_melt.crashed = True
            raise RuntimeError("AllocSurface() failed with error 6")
        Path(out).write_bytes(b"video")

    monkeypatch.setattr(kb, "_run_melt", fake_melt)
    monkeypatch.setattr(kb, "_mux", lambda parts, wav, out: Path(out).write_bytes(b"".join(p.read_bytes() for p in parts)))
    out = tmp_path / "kd" / "final.mp4"
    with pytest.raises(RuntimeError):
        kb.render_in_parts(tl, tmp_path / "kd", out, cuts=cuts, emotions=[""] * 6, starts=starts)
    assert not out.exists()  # jamais de vidéo à moitié écrite sous son nom
    rendered.clear()
    kb.render_in_parts(tl, tmp_path / "kd", out, cuts=cuts, emotions=[""] * 6, starts=starts)
    assert rendered == ["part_001", "part_002"]  # la partie 1 n'est pas refaite
    assert out.read_bytes() == b"video" * 3 and not (tmp_path / "kd" / "render_parts").exists() or \
        not any((tmp_path / "kd" / "render_parts").iterdir())


@pytest.mark.skipif(not (kb.KDENLIVE_BIN / "melt.exe").is_file(), reason="Kdenlive (melt) non installe")
def test_melt_renders_a_long_video_in_parts_with_one_soundtrack(tmp_path) -> None:
    tl = compilation(tmp_path)
    starts = kb.plan_parts(tl, chapters=kb.chapter_starts(tl), target_s=3.0, from_s=5.0)
    emotions = ["calm", "action"] * 3
    cuts = kb.part_boundary_cuts(kb.plan_cuts(tl.clips, emotions, chapter_starts=kb.chapter_starts(tl)), starts)
    out = tmp_path / "kd" / "long.mp4"
    kb.render_in_parts(tl, tmp_path / "kd", out, cuts=cuts, emotions=emotions, starts=starts, vcodec="libx264", threads=2)
    probe = subprocess.run([str(kb.KDENLIVE_BIN / "ffprobe.exe"), "-v", "error", "-show_entries", "stream=codec_type,nb_frames",
                            "-of", "csv=p=0", str(out)], capture_output=True, text=True)
    streams = dict(line.split(",")[:2] for line in probe.stdout.split())
    assert streams.get("video") == "120" and "audio" in streams  # 3 parties de 40 images, un seul son
    assert not list((tmp_path / "kd").glob("**/*.part.*"))


def test_media_paths_are_absolute_even_from_a_relative_folder(tmp_path, monkeypatch) -> None:
    tl = timeline(tmp_path)
    monkeypatch.chdir(tmp_path)
    project = kb.build_project(tl, Path("kd"), name="essai", emotions=["calm"] * 3)  # Kdenlive plante sinon
    resources = [p.text for p in ET.parse(project.project).getroot().iter("property") if p.get("name") == "resource"]
    assert all(Path(r).is_absolute() for r in resources if r and r.endswith((".png", ".wav", ".mov")))
