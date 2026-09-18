"""Rapport HTML local de contrôle : cases découpées regroupées par scène narrée.

Produit un fichier HTML autonome (images intégrées en JPEG base64) à ouvrir dans
un navigateur pour vérifier d'un coup d'œil la découpe du slicer et la narration
de l'analyzer. Aucun serveur ni ressource externe : le fichier est portable.
"""

from __future__ import annotations

import base64
import html
import logging
import os
from collections.abc import Sequence
from io import BytesIO
from pathlib import Path

from PIL import Image

from src.models.audio import VoiceoverManifest
from src.models.panel import Panel
from src.models.scene import ChapterAnalysis
from src.utils.image_utils import to_pil

logger = logging.getLogger(__name__)

DEFAULT_THUMB_WIDTH: int = 420
DEFAULT_JPEG_QUALITY: int = 78

_CSS = """
:root { color-scheme: dark; }
body { margin: 0; padding: 24px; background: #14161a; color: #e6e6e6;
       font: 15px/1.5 system-ui, Segoe UI, Roboto, sans-serif; }
h1 { font-size: 22px; margin: 0 0 4px; }
.meta { color: #9aa0a6; margin-bottom: 24px; }
.scene { border: 1px solid #2a2e35; border-radius: 10px; padding: 16px; margin-bottom: 20px;
         background: #1b1e24; }
.scene.filler { opacity: 0.6; border-style: dashed; }
.scene-head { display: flex; gap: 12px; align-items: baseline; flex-wrap: wrap; margin-bottom: 8px; }
.scene-head .idx { font-weight: 700; font-size: 17px; }
.badge { font-size: 12px; padding: 2px 8px; border-radius: 999px; background: #2f3541; }
.badge.filler { background: #5a3d1a; }
.badge.emotion { background: #24405a; }
.narration { font-size: 16px; margin: 6px 0 14px; }
.panels { display: flex; gap: 12px; flex-wrap: wrap; align-items: flex-start; }
.panel { display: flex; flex-direction: column; gap: 4px; }
.panel img { display: block; border-radius: 6px; background: #000; }
.panel .cap { font-size: 12px; color: #9aa0a6; }
.panel.giant .cap { color: #f0b35a; }
audio { display: block; width: 100%; max-width: 640px; margin: 4px 0 12px; }
.full-audio { margin: 0 0 24px; padding: 12px 16px; border: 1px solid #2a2e35; border-radius: 10px;
              background: #1b1e24; }
.badge.audio { background: #2f4a2f; }
"""


def _relative_href(target: Path, from_dir: Path) -> str:
    """Chemin relatif (séparateurs ``/``) de ``target`` depuis le dossier du rapport."""
    try:
        rel = os.path.relpath(target, from_dir)
    except ValueError:  # lecteurs Windows differents : chemin absolu en repli
        rel = str(target)
    return html.escape(rel.replace(os.sep, "/"))


def _thumbnail_data_uri(panel: Panel, width: int, quality: int) -> tuple[str, int, int]:
    """Miniature JPEG (data URI) d'une case, largeur bornée à ``width``."""
    img = to_pil(panel.image)
    if img.width > width:
        new_height = max(1, round(img.height * width / img.width))
        img = img.resize((width, new_height), Image.Resampling.LANCZOS)
    buffer = BytesIO()
    img.save(buffer, format="JPEG", quality=quality, optimize=True)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}", img.width, img.height


def _panel_html(panel: Panel, width: int, quality: int) -> str:
    uri, w, h = _thumbnail_data_uri(panel, width, quality)
    css_class = "panel giant" if panel.type == "scroll_vertical" else "panel"
    caption = f"#{panel.index} - {panel.width}x{panel.height} px - y {panel.y_start}-{panel.y_end}"
    if panel.part:
        caption += f" - {panel.part} (split #{panel.source_index})"
    if panel.type == "scroll_vertical":
        caption += " - scroll_vertical"
    return (
        f'<div class="{css_class}"><img src="{uri}" width="{w}" height="{h}" '
        f'alt="panel {panel.index}"><span class="cap">{html.escape(caption)}</span></div>'
    )


def build_html_report(
    panels: Sequence[Panel],
    analysis: ChapterAnalysis | None,
    path: str | Path,
    *,
    title: str = "",
    thumb_width: int = DEFAULT_THUMB_WIDTH,
    jpeg_quality: int = DEFAULT_JPEG_QUALITY,
    audio: VoiceoverManifest | None = None,
    audio_dir: str | Path | None = None,
) -> Path:
    """Écrit le rapport HTML et renvoie son chemin.

    Args:
        panels: cases découpées (ordre de lecture).
        analysis: scènes narrées ; ``None`` pour un rapport de découpe seule.
        path: fichier HTML de sortie.
        title: titre affiché (série / épisode).
        thumb_width: largeur maximale des miniatures (px).
        jpeg_quality: qualité JPEG des miniatures.
        audio: manifeste de la voix off ; ajoute un lecteur audio par scène
            (fichiers référencés par chemin relatif, non intégrés).
        audio_dir: dossier des WAV du manifeste (défaut : dossier du rapport).
    """
    path = Path(path)
    by_index = {panel.index: panel for panel in panels}
    n_giant = sum(1 for panel in panels if panel.type == "scroll_vertical")
    heading = html.escape(title or (analysis.series_title if analysis else "") or "Rapport de decoupe")
    meta_bits = [f"{len(panels)} cases", f"{n_giant} scroll_vertical"]
    if analysis is not None:
        meta_bits.append(f"{analysis.n_scenes} scenes dont {analysis.n_filler} filler")
        meta_bits.append(f"modele {html.escape(analysis.model)} ({html.escape(analysis.language)})")
        meta_bits.append(
            f"tokens {analysis.prompt_tokens} in / {analysis.output_tokens} out / "
            f"{analysis.thinking_tokens} reflexion"
        )
        if analysis.episode_title:
            heading += f" - {html.escape(analysis.episode_title)}"

    audio_items = audio.by_scene() if audio is not None else {}
    audio_root = Path(audio_dir) if audio_dir is not None else path.parent
    full_audio_html = ""
    if audio is not None:
        meta_bits.append(
            f"voix {html.escape(audio.voice)} - {audio.n_items} segments - {audio.total_duration_s:.1f}s"
        )
        if audio.full_file:
            href = _relative_href(audio_root / audio.full_file, path.parent)
            full_audio_html = (
                f'<div class="full-audio"><strong>Voix off complete</strong> '
                f"({audio.total_duration_s:.1f}s, {html.escape(audio.voice)})"
                f'<audio controls preload="none" src="{href}"></audio></div>'
            )

    sections: list[str] = []
    covered: set[int] = set()
    if analysis is not None:
        for scene in analysis.scenes:
            covered.update(scene.panel_ids)
            css_class = "scene filler" if scene.is_filler else "scene"
            badges = f'<span class="badge emotion">{html.escape(scene.emotion)}</span>'
            if scene.is_filler:
                badges += '<span class="badge filler">filler</span>'
            ids = ", ".join(str(pid) for pid in scene.panel_ids)
            player = ""
            item = audio_items.get(scene.index)
            if item is not None:
                badges += f'<span class="badge audio">{item.duration_s:.1f}s</span>'
                href = _relative_href(audio_root / item.file, path.parent)
                player = f'<audio controls preload="none" src="{href}"></audio>'
            thumbs = "".join(
                _panel_html(by_index[pid], thumb_width, jpeg_quality)
                for pid in scene.panel_ids
                if pid in by_index
            )
            sections.append(
                f'<section class="{css_class}"><div class="scene-head">'
                f'<span class="idx">Scene {scene.index}</span>{badges}'
                f'<span class="badge">cases {html.escape(ids)}</span></div>'
                f'<p class="narration">{html.escape(scene.narration)}</p>{player}'
                f'<div class="panels">{thumbs}</div></section>'
            )
    leftovers = [panel for panel in panels if panel.index not in covered]
    if leftovers:
        label = "Cases" if analysis is None else "Cases non retenues (transition, non montees)"
        thumbs = "".join(_panel_html(panel, thumb_width, jpeg_quality) for panel in leftovers)
        sections.append(
            f'<section class="scene"><div class="scene-head"><span class="idx">{label}</span>'
            f'<span class="badge">{len(leftovers)}</span></div><div class="panels">{thumbs}</div></section>'
        )

    document = (
        "<!doctype html><html lang=\"fr\"><head><meta charset=\"utf-8\">"
        f"<title>{heading}</title><style>{_CSS}</style></head><body>"
        f"<h1>{heading}</h1><div class=\"meta\">{' - '.join(meta_bits)}</div>"
        f"{full_audio_html}{''.join(sections)}</body></html>"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(document, encoding="utf-8")
    logger.info("Rapport HTML ecrit : %s (%d Ko)", path, path.stat().st_size // 1024)
    return path


__all__ = ["build_html_report", "DEFAULT_THUMB_WIDTH", "DEFAULT_JPEG_QUALITY"]
