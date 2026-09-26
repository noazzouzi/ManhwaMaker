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
    assert [p.sub for p in placed] == [0, 1, 1]
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
    assert rects[1].endswith("=0 -4 336 189 1")
    assert rects[2].startswith("0h=0 0 320 180 1;2=-8 -4 336 189 1;") and rects[2].count(";") >= 3
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


def test_media_paths_are_absolute_even_from_a_relative_folder(tmp_path, monkeypatch) -> None:
    tl = timeline(tmp_path)
    monkeypatch.chdir(tmp_path)
    project = kb.build_project(tl, Path("kd"), name="essai", emotions=["calm"] * 3)  # Kdenlive plante sinon
    resources = [p.text for p in ET.parse(project.project).getroot().iter("property") if p.get("name") == "resource"]
    assert all(Path(r).is_absolute() for r in resources if r and r.endswith((".png", ".wav", ".mov")))
