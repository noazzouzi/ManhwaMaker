"""Faux client Gemini partage par les tests.

``FakeClient`` imite ``genai.Client().models.generate_content`` et repond selon le
schema demande (beats, script, cases cles) avec de vraies ``GenerateContentResponse``.
Aucun appel reseau. Extrait de ``tests/test_analyzer.py`` pour que les autres suites
puissent s'en servir sans executer ce fichier, sur le modele de ``tests/synthetic_strip.py``.
"""

from __future__ import annotations

import io
import json
import re

import numpy as np
from google.genai import types
from PIL import Image

from src.models.panel import Panel
from src.models.scene import EMOTIONS, BeatBatch, KeyframeBatch, ScriptDraft

__all__ = [
    "make_response", "all_text", "batch_ids_of", "beats_payload", "script_payload",
    "keyframes_payload", "default_responder", "FakeModels", "FakeClient",
    "panel", "panels", "jpeg_size",
]


_CAPTION = re.compile(r"^Panel (\d+) \(")
_BEAT_LINE = re.compile(r"^Beat (\d+) \(panels ([\d, ]+)\)(\s*\[FILLER[^\]]*\])?:", re.MULTILINE)
_PARAGRAPH_LINE = re.compile(r"^Paragraph (\d+) \(candidate panels: ([\d, ]*)\):", re.MULTILINE)


# --- Outils ------------------------------------------------------------------------
def panel(index: int, height: int = 80, width: int = 120, kind: str = "static") -> Panel:
    rng = np.random.default_rng(index)
    return Panel(
        index=index, y_start=index * 100, y_end=index * 100 + height, height=height, width=width,
        type=kind,  # type: ignore[arg-type]
        image=rng.integers(0, 256, size=(height, width, 3), dtype=np.uint8),
    )


def panels(n: int) -> list[Panel]:
    return [panel(i, height=80 + 10 * (i % 4)) for i in range(n)]


def jpeg_size(data: bytes) -> tuple[int, int]:
    with Image.open(io.BytesIO(data)) as img:
        assert img.format == "JPEG"
        return img.size


def make_response(payload: str | dict | list, *, prompt_tokens: int = 100, output_tokens: int = 40, thinking_tokens: int = 0):
    """Construit une vraie ``GenerateContentResponse`` contenant ``payload`` en texte."""
    text = payload if isinstance(payload, str) else json.dumps(payload)
    return types.GenerateContentResponse(
        candidates=[types.Candidate(content=types.Content(role="model", parts=[types.Part(text=text)]))],
        usage_metadata=types.GenerateContentResponseUsageMetadata(
            prompt_token_count=prompt_tokens, candidates_token_count=output_tokens, thoughts_token_count=thinking_tokens,
        ),
    )


def all_text(contents: list[types.Part]) -> str:
    return "\n".join(part.text for part in contents if part.text)


def batch_ids_of(contents: list[types.Part]) -> list[int]:
    """Numéros de cases annoncés par les libellés ``Panel N (...)`` d'un prompt."""
    return [int(m.group(1)) for part in contents if part.text for m in [_CAPTION.match(part.text)] if m]


def beats_payload(ids: list[int]) -> dict:
    """Un beat par paire de cases."""
    beats = []
    for i in range(0, len(ids), 2):
        group = ids[i : i + 2]
        beats.append({"panel_ids": group, "summary": f"Beat with panels {group}.", "characters": ["Hero"], "dialogue": []})
    return {"beats": beats}


def script_payload(contents: list[types.Part], *, dirty: bool = False) -> dict:
    """Un paragraphe pour deux beats narratifs, dans l'ordre."""
    story = [int(m.group(1)) for m in _BEAT_LINE.finditer(all_text(contents)) if not m.group(3)]
    paragraphs = []
    for i in range(0, len(story), 2):
        beat_ids = story[i : i + 2]
        text = f"The hero pushes forward through beats {beat_ids}. He refuses to give up."
        if dirty:
            text = f"In this panel, we see the hero pushing through beats {beat_ids}. Here, he refuses to give up."
        paragraphs.append({"text": text, "beat_ids": beat_ids, "emotion": EMOTIONS[i % len(EMOTIONS)]})
    return {"paragraphs": paragraphs}


def keyframes_payload(contents: list[types.Part]) -> dict:
    """La premiere case candidate de chaque paragraphe."""
    choices = []
    for m in _PARAGRAPH_LINE.finditer(all_text(contents)):
        candidates = [int(x) for x in m.group(2).split(",") if x.strip()]
        index = int(m.group(1))
        # Paragraphes impairs : la case cle est un impact (punch-in) ; 99 = numero invente, ignore.
        heavy = candidates[:1] + [99] if index % 2 == 1 else []
        choices.append({"paragraph_index": index, "key_panel_ids": candidates[:1], "action_heavy_ids": heavy})
    return {"choices": choices}


def default_responder(contents, call_no, config):
    schema = config.response_schema
    if schema is BeatBatch:
        return make_response(beats_payload(batch_ids_of(contents)))
    if schema is ScriptDraft:
        return make_response(script_payload(contents), thinking_tokens=7)
    if schema is KeyframeBatch:
        return make_response(keyframes_payload(contents))
    raise AssertionError(f"schema inattendu {schema}")


class FakeModels:
    def __init__(self, responder):
        self.responder = responder
        self.calls: list[dict] = []

    def generate_content(self, *, model, contents, config):
        self.calls.append({"model": model, "contents": contents, "config": config})
        return self.responder(contents, len(self.calls), config)


class FakeClient:
    def __init__(self, responder=default_responder):
        self.models = FakeModels(responder)

    def calls_for(self, schema) -> list[dict]:
        return [c for c in self.models.calls if c["config"].response_schema is schema]
