"""Bibliothèque des vidéos : dossiers de ``output/`` lus tels quels (chapitres et compilations).

Un dossier de chapitre a un ``chapter.json`` ; une compilation n'a que ``timeline.json``
(et ``media/``). Un dossier sans ``timeline.json`` est un chapitre inachevé : en échec
d'après ``batch_status.json``, ou interrompu. Les dossiers d'essais sans ces fichiers
(bancs, tests de découpe) sont ignorés.

Rien n'est copié ni déplacé : le Studio lit les fichiers du pipeline, et sert les médias
(vidéos, images) en vérifiant qu'ils restent dans le dossier de la vidéo.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import subprocess
import threading
from pathlib import Path
from typing import Any

from PIL import Image

logger = logging.getLogger(__name__)

#: Vidéo finale rendue par Kdenlive / melt (dossier ``kdenlive/`` du chapitre).
FINAL_DIRNAME = "kdenlive"
THUMB_WIDTH = 480


def _mtime(path: Path) -> float:
    try:
        return path.stat().st_mtime
    except OSError:
        return 0.0


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def fix_mojibake(text: str | None) -> str | None:
    """Répare un titre UTF-8 lu en Latin-1 par un ancien scraping (« Archerâs » -> « Archer’s »)."""
    if not text or not any(0x80 <= ord(ch) <= 0xFF for ch in text):
        return text
    try:
        return text.encode("latin-1").decode("utf-8")
    except (UnicodeEncodeError, UnicodeDecodeError):
        return text


def _norm(path: str | Path) -> str:
    return str(Path(path)).replace("/", "\\").rstrip("\\").lower()


class Library:
    """Inventaire des vidéos d'un dossier de sortie, mis en cache par date de modification."""

    def __init__(self, out_root: str | Path, status_file: str | Path | None, cache_dir: str | Path) -> None:
        self.out_root = Path(out_root).resolve()
        self.status_file = Path(status_file) if status_file else None
        self.cache_dir = Path(cache_dir)
        self._cache: dict[str, tuple[tuple[float, ...], dict[str, Any]]] = {}
        self._lock = threading.Lock()

    # --- inventaire ---------------------------------------------------------------------
    def _status_by_dir(self) -> dict[str, dict[str, Any]]:
        data = _read_json(self.status_file) if self.status_file and self.status_file.is_file() else None
        out: dict[str, dict[str, Any]] = {}
        for url, entry in ((data or {}).get("chapters") or {}).items():
            if entry.get("out_dir"):
                out[_norm(entry["out_dir"])] = {**entry, "url": url}
        return out

    def folder(self, name: str) -> Path:
        """Dossier d'une vidéo ; ``KeyError`` si le nom sort de ``output/`` ou n'existe pas."""
        if not name or name in (".", "..") or "/" in name or "\\" in name:
            raise KeyError(name)
        path = (self.out_root / name).resolve()
        if path.parent != self.out_root.resolve() or not path.is_dir():
            raise KeyError(name)
        return path

    def list(self) -> list[dict[str, Any]]:
        statuses = self._status_by_dir()
        items = []
        if not self.out_root.is_dir():
            return items
        for folder in self.out_root.iterdir():
            if not folder.is_dir():
                continue
            item = self._item(folder, statuses)
            if item is not None:
                items.append(item)
        items.sort(key=lambda it: it["updated"], reverse=True)
        return items

    def get(self, name: str) -> dict[str, Any]:
        item = self._item(self.folder(name), self._status_by_dir())
        if item is None:
            raise KeyError(name)
        return item

    def _item(self, folder: Path, statuses: dict[str, dict[str, Any]]) -> dict[str, Any] | None:
        chapter_json, timeline_json = folder / "chapter.json", folder / "timeline.json"
        if not chapter_json.is_file() and not timeline_json.is_file():
            return None
        final_dir = folder / FINAL_DIRNAME
        key = (_mtime(chapter_json), _mtime(timeline_json), _mtime(folder), _mtime(final_dir))
        with self._lock:
            cached = self._cache.get(folder.name)
        if cached is not None and cached[0] == key:
            item = dict(cached[1])
        else:
            item = self._build(folder)
            with self._lock:
                self._cache[folder.name] = (key, dict(item))
        entry = statuses.get(_norm(folder.resolve()))
        item["batch_status"] = entry.get("status") if entry else None
        if not item["has_timeline"]:
            if entry and entry.get("status") == "failed":
                item.update(status="failed", error=entry.get("error"), failed_stage=entry.get("stage"))
            else:
                item["status"] = "incomplete"
            if entry and not item.get("url"):
                item["url"] = entry.get("url")
        return item

    def _build(self, folder: Path) -> dict[str, Any]:
        chapter = _read_json(folder / "chapter.json") if (folder / "chapter.json").is_file() else None
        timeline = _read_json(folder / "timeline.json") if (folder / "timeline.json").is_file() else None
        scenes = _read_json(folder / "scenes.json") if (folder / "scenes.json").is_file() else None
        kind = "chapter" if chapter else "compilation"
        series = fix_mojibake((chapter or {}).get("series_title") or (timeline or {}).get("series_title")) or folder.name
        episode = fix_mojibake((chapter or {}).get("episode_title") or (timeline or {}).get("episode_title")) or ""
        videos = [p for p in folder.glob("preview_*.mp4") if not p.name.endswith(".part.mp4")]
        if (folder / FINAL_DIRNAME).is_dir():
            videos += [p for p in (folder / FINAL_DIRNAME).glob("*.mp4") if not p.name.endswith(".part.mp4")]
        broken = [p for p in videos if not _playable(p)]
        previews = sorted((p for p in videos if p.parent == folder and p not in broken),
                          key=lambda p: (p.name != "preview_full.mp4", -_mtime(p)))
        finals = sorted((p for p in videos if p.parent != folder and p not in broken), key=_mtime, reverse=True)
        projects = sorted((folder / FINAL_DIRNAME).glob("*.kdenlive")) if (folder / FINAL_DIRNAME).is_dir() else []
        media = [{"path": p.relative_to(folder).as_posix(), "label": _media_label(p), "kind": "final", "mtime": _mtime(p)} for p in finals]
        media += [{"path": p.relative_to(folder).as_posix(), "label": _media_label(p), "kind": "preview", "mtime": _mtime(p)} for p in previews]
        status = "final" if finals else "preview" if previews else "montage" if timeline else "incomplete"
        updated = max([_mtime(folder / n) for n in ("timeline.json", "chapter.json")] + [m["mtime"] for m in media] + [0.0])
        return {
            "name": folder.name, "kind": kind, "series": series, "episode": episode,
            "episode_no": (chapter or {}).get("episode_no"), "url": (chapter or {}).get("url"),
            "duration_s": round(float((timeline or {}).get("total_duration_s") or 0.0), 1),
            "n_scenes": len((scenes or {}).get("scenes") or []) or len((timeline or {}).get("audio") or []),
            "n_panels": (scenes or {}).get("n_panels"), "model": (scenes or {}).get("model"),
            "format": "SHORT" if (timeline or {}).get("height", 0) > (timeline or {}).get("width", 1) else "LONG",
            "status": status, "has_timeline": timeline is not None, "media": media,
            "kdenlive_project": projects[0].relative_to(folder).as_posix() if projects else None,
            "broken": [{"path": p.relative_to(folder).as_posix(), "label": _media_label(p), "mtime": _mtime(p),
                        "size": p.stat().st_size} for p in broken],
            "updated": updated, "error": None,
        }

    # --- détail -------------------------------------------------------------------------
    def detail(self, name: str) -> dict[str, Any]:
        """La vidéo et ses scènes : texte, émotion, début, durée, images montées."""
        folder = self.folder(name)
        item = self.get(name)
        timeline = _read_json(folder / "timeline.json") or {}
        scenes_doc = _read_json(folder / "scenes.json") or {}
        by_index = {s.get("index"): s for s in scenes_doc.get("scenes") or []}
        panels_dir = Path(timeline.get("panels_dir") or folder)
        clips_by_scene: dict[int, list[str]] = {}
        emotion_by_scene: dict[int, str] = {}
        for clip in timeline.get("clips") or []:
            index = clip.get("scene_index")
            path = self._relative(folder, panels_dir / clip.get("file", ""))
            if path and path not in clips_by_scene.setdefault(index, []):
                clips_by_scene[index].append(path)
            if clip.get("emotion") and index not in emotion_by_scene:
                emotion_by_scene[index] = clip["emotion"]
        cues = timeline.get("subtitles") or []
        scenes = []
        for number, audio in enumerate(timeline.get("audio") or []):
            index = audio.get("scene_index", number)
            start, duration = float(audio.get("start_s") or 0.0), float(audio.get("duration_s") or 0.0)
            source = by_index.get(index) or {}
            narration = source.get("narration")
            if not narration:  # compilation : texte des sous-titres de la scène
                narration = " ".join(c.get("text", "") for c in cues if start <= float(c.get("start_s", -1)) < start + duration)
            scenes.append({
                "number": number + 1, "index": index, "start_s": round(start, 2), "duration_s": round(duration, 2),
                "emotion": source.get("emotion") or emotion_by_scene.get(index, ""), "narration": narration,
                "images": clips_by_scene.get(index, [])[:8],
            })
        return {**item, "scenes": scenes, "chapters": _chapter_marks(timeline)}

    @staticmethod
    def _relative(folder: Path, path: Path) -> str | None:
        try:
            return path.resolve().relative_to(folder.resolve()).as_posix()
        except (ValueError, OSError):
            return None

    def file(self, name: str, relative: str) -> Path:
        """Fichier d'une vidéo, en refusant tout chemin qui sortirait de son dossier."""
        folder = self.folder(name)
        path = (folder / relative).resolve()
        if folder not in path.parents or not path.is_file():
            raise KeyError(relative)
        return path

    # --- vignettes ------------------------------------------------------------------------
    def thumbnail(self, name: str) -> Path:
        """Vignette 16:9 : une image de l'aperçu (ou de la vidéo finale), sinon la première case."""
        item = self.get(name)
        folder = self.folder(name)
        source = next((folder / m["path"] for m in item["media"]), None)
        stamp = hashlib.sha1(f"{name}|{source}|{_mtime(source) if source else 0}".encode()).hexdigest()[:12]
        target = self.cache_dir / "thumbs" / f"{stamp}.jpg"
        if target.is_file():
            return target
        target.parent.mkdir(parents=True, exist_ok=True)
        if source is not None and _extract_frame(source, target, at=min(30.0, max(1.0, item["duration_s"] * 0.25))):
            return target
        image = self._first_image(folder)
        if image is None:
            raise KeyError(name)
        return self.image(name, image.relative_to(folder).as_posix(), THUMB_WIDTH)

    def _first_image(self, folder: Path) -> Path | None:
        timeline = _read_json(folder / "timeline.json") or {}
        clips = timeline.get("clips") or []
        if clips:
            candidate = Path(timeline.get("panels_dir") or folder) / clips[0].get("file", "")
            if candidate.is_file() and self._relative(folder, candidate):
                return candidate
        return next(iter(sorted(folder.glob("panel_*.png"))), None)

    def image(self, name: str, relative: str, width: int) -> Path:
        """Image d'une vidéo réduite à ``width`` px de large (cache JPEG)."""
        source = self.file(name, relative)
        width = max(64, min(int(width), 1280))
        stamp = hashlib.sha1(f"{source}|{_mtime(source)}|{width}".encode()).hexdigest()[:16]
        target = self.cache_dir / "images" / f"{stamp}.jpg"
        if not target.is_file():
            target.parent.mkdir(parents=True, exist_ok=True)
            with Image.open(source) as img:
                img = img.convert("RGB")
                if img.width > width:
                    img = img.resize((width, max(1, round(img.height * width / img.width))), Image.LANCZOS)
                img.save(target, quality=82)
        return target


def _playable(path: Path) -> bool:
    """Vrai si le MP4 a son index (« moov ») : un rendu interrompu n'en a pas et ne se lit pas."""
    try:
        size = path.stat().st_size
        with open(path, "rb") as handle:
            if b"moov" in handle.read(1 << 20):
                return True
            handle.seek(max(0, size - (8 << 20)))
            return b"moov" in handle.read()
    except OSError:
        return False


def _media_label(path: Path) -> str:
    stem = path.stem
    if path.parent.name == FINAL_DIRNAME:
        fps = "".join(ch for ch in stem.rsplit("_", 1)[-1] if ch.isdigit())
        excerpt = re.search(r"_(\d+)s_", stem)
        label = f"Extrait final {excerpt.group(1)} s" if excerpt else "Vidéo finale"
        return f"{label}{f' · {fps} i/s' if stem.endswith('fps') and fps else ''}"
    label = stem.replace("preview_", "")
    return "Aperçu complet" if label == "full" else f"Aperçu {label}"


def _chapter_marks(timeline: dict[str, Any]) -> list[dict[str, Any]]:
    """Début de chaque chapitre d'une compilation (d'après les noms des fichiers de voix)."""
    marks: list[dict[str, Any]] = []
    last = None
    for audio in timeline.get("audio") or []:
        prefix = str(audio.get("file", "")).split("__", 1)[0] if "__" in str(audio.get("file", "")) else None
        if prefix and prefix != last:
            marks.append({"label": prefix.rsplit("_ep", 1)[-1] if "_ep" in prefix else prefix, "start_s": audio.get("start_s", 0.0)})
            last = prefix
    return marks if len(marks) > 1 else []


def _extract_frame(video: Path, target: Path, *, at: float) -> bool:
    try:
        import imageio_ffmpeg

        ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:  # noqa: BLE001 - ffmpeg absent : on retombe sur une case
        return False
    cmd = [ffmpeg, "-v", "error", "-y", "-ss", f"{at:.2f}", "-i", str(video), "-frames:v", "1",
           "-vf", f"scale={THUMB_WIDTH}:-2", "-q:v", "4", str(target)]
    try:
        subprocess.run(cmd, capture_output=True, timeout=30, check=False,
                       creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    except (OSError, subprocess.TimeoutExpired):
        return False
    return target.is_file() and target.stat().st_size > 0


__all__ = ["Library"]
