"""Tests du rapport HTML (``src.utils.report``) et de l'indicateur ``is_filler``."""

from __future__ import annotations

import numpy as np

from src.models.panel import Panel
from src.models.scene import ChapterAnalysis, Scene
from src.utils.report import build_html_report


def _panel(index: int, height: int = 60, width: int = 900, kind: str = "static") -> Panel:
    rng = np.random.default_rng(index)
    return Panel(
        index=index, y_start=index * 100, y_end=index * 100 + height, height=height, width=width,
        type=kind,  # type: ignore[arg-type]
        image=rng.integers(0, 256, size=(height, width, 3), dtype=np.uint8),
    )


def test_build_html_report_groups_panels_by_scene(tmp_path) -> None:
    panels = [_panel(0), _panel(1), _panel(2, height=1800, kind="scroll_vertical"), _panel(3)]
    analysis = ChapterAnalysis(
        series_title="Serie <X>", episode_title="Ep. 1", model="fake", language="en", n_panels=4,
        n_batches=1,
        scenes=[
            Scene(index=0, panel_ids=[0, 1], narration="Un <b>debut</b>.", emotion="calm"),
            Scene(index=1, panel_ids=[2], narration="La chute.", emotion="action", is_filler=False),
            # La case 3 n'est dans aucune scene : listee a part.
        ],
        prompt_tokens=10, output_tokens=5, thinking_tokens=2,
    )
    analysis.scenes[0].is_filler = True
    assert analysis.n_filler == 1 and [s.index for s in analysis.story_scenes()] == [1]

    path = build_html_report(panels, analysis, tmp_path / "r" / "report.html", thumb_width=300)
    text = path.read_text(encoding="utf-8")
    assert text.count("<img ") == 4
    assert text.count('<section class="scene') == 3
    assert 'class="scene filler"' in text and text.count('class="badge filler"') == 1
    assert "Un &lt;b&gt;debut&lt;/b&gt;." in text and "Serie &lt;X&gt; - Ep. 1" in text
    assert "Cases non retenues" in text
    assert "scroll_vertical" in text
    assert 'width="300"' in text  # miniatures reduites a la largeur demandee
    assert "http" not in text.split("<body>")[1]  # aucune ressource externe

    # Sans analyse : rapport de decoupe seule.
    alone = build_html_report(panels, None, tmp_path / "alone.html", title="Decoupe")
    alone_text = alone.read_text(encoding="utf-8")
    assert alone_text.count("<img ") == 4 and "<title>Decoupe</title>" in alone_text


def test_build_html_report_embeds_audio_players_with_relative_paths(tmp_path) -> None:
    from src.models.audio import SceneAudio, VoiceoverManifest

    panels = [_panel(0), _panel(1)]
    analysis = ChapterAnalysis(
        model="fake", language="en", n_panels=2,
        scenes=[
            Scene(index=0, panel_ids=[0], narration="Intro.", emotion="calm"),
            Scene(index=1, panel_ids=[1], narration="Credits.", emotion="neutral", is_filler=True),
        ],
    )
    manifest = VoiceoverManifest(
        language="en", lang_code="a", voice="af_heart", speed=1.0, padding_s=0.3, sample_rate=24000,
        items=[SceneAudio(scene_index=0, file="scene_000.wav", duration_s=2.5, speech_s=2.2,
                          sample_rate=24000, text="Intro.", emotion="calm")],
        total_duration_s=2.5, full_file="voiceover_full.wav",
    )
    report = build_html_report(
        panels, analysis, tmp_path / "chapter" / "report.html",
        audio=manifest, audio_dir=tmp_path / "chapter" / "audio",
    )
    text = report.read_text(encoding="utf-8")
    assert text.count("<audio ") == 2  # voix off complete + scene 0 (la scene filler n'a pas d'audio)
    assert 'src="audio/scene_000.wav"' in text and 'src="audio/voiceover_full.wav"' in text
    assert '<span class="badge audio">2.5s</span>' in text
    assert "voix af_heart - 1 segments - 2.5s" in text
    assert "Voix off complete" in text
