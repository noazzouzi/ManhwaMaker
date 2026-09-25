"""Outils partagés par les tests toonsplit : strips synthétiques, specs, boîtes, faux Gemini."""

from __future__ import annotations

import json
from typing import Any

import numpy as np
from google.genai import types

from src.modules.toonsplit.ai import BlockSpec, DropZone, KeepZone
from src.modules.toonsplit.geometry import Box


def art(height: int, width: int = 400, seed: int = 0) -> np.ndarray:
    """Dessin texturé (jamais pris pour du fond) : dégradé coloré + bruit."""
    rng = np.random.default_rng(seed)
    y = np.linspace(60, 190, height)[:, None, None]
    base = np.broadcast_to(y, (height, width, 3)).astype(np.float64)
    noise = rng.normal(0, 25, size=(height, width, 3))
    return np.clip(base + noise, 0, 255).astype(np.uint8)


def white(height: int, width: int = 400, value: int = 255) -> np.ndarray:
    return np.full((height, width, 3), value, np.uint8)


def stack(*parts: np.ndarray) -> np.ndarray:
    return np.vstack(parts)


def spec(role: str = "key", keep: list[tuple[float, float]] | None = None,
         drop: list[tuple[float, float, str]] | None = None, tts: list[str] | None = None) -> BlockSpec:
    return BlockSpec(
        role=role,  # type: ignore[arg-type]
        keep=[KeepZone(what="subject", y=[a, b]) for a, b in (keep or [])],
        drop=[DropZone(what=kind, y=[a, b], kind=kind) for a, b, kind in (drop or [])],  # type: ignore[arg-type]
        tts=tts or [],
    )


def box(y0: int, y1: int, kind: str, x0: int = 50, x1: int = 250, score: float = 0.9) -> Box:
    return Box(x0, y0, x1, y1, kind, score, "test")


def response(payload: Any) -> types.GenerateContentResponse:
    text = payload if isinstance(payload, str) else json.dumps(payload)
    return types.GenerateContentResponse(
        candidates=[types.Candidate(content=types.Content(role="model", parts=[types.Part(text=text)]))],
    )


class FakeManager:
    """Imite ``GeminiManager.generate`` : rejoue une liste de réponses et garde les requêtes."""

    def __init__(self, *payloads: Any) -> None:
        self.payloads = list(payloads)
        self.calls: list[tuple[list[Any], Any, str]] = []

    def generate(self, contents: list[Any], config: Any, *, label: str = "") -> Any:
        self.calls.append((contents, config, label))
        if not self.payloads:
            raise AssertionError("appel Gemini inattendu")
        return response(self.payloads.pop(0))
