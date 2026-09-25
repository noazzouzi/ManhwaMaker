"""toonsplit avec les vrais modèles ONNX sur le strip de test ``eval/toonsplit/001``.

Lent (~15 s) et dépendant des modèles Hugging Face : lancé seulement avec ``TOONSPLIT_SLOW=1``.
Aucun appel Gemini (spec écrite à la main).
"""

from __future__ import annotations

import os

import pytest

from src.modules.toonsplit import detectors
from src.modules.toonsplit.ai import ManualSpecProvider
from src.modules.toonsplit.blocks import segment_blocks
from src.modules.toonsplit.evaluate import evaluate, load_case, summarize
from src.modules.toonsplit.geometry import BUBBLE, FREE_TEXT, HEAD
from src.modules.toonsplit.pipeline import analyze_strip, load_image
from src.utils.config import PROJECT_ROOT

CASE = PROJECT_ROOT / "eval" / "toonsplit" / "001"
pytestmark = pytest.mark.skipif(
    os.environ.get("TOONSPLIT_SLOW") != "1" or not (CASE / "strip.webp").is_file(),
    reason="TOONSPLIT_SLOW=1 et eval/toonsplit/001 requis",
)


def test_real_detectors_on_reference_strip() -> None:
    img = load_image(CASE / "strip.webp")
    blocks = segment_blocks(img)
    assert len(blocks) == 6
    found = [detectors.detect_all(img[b.y0:b.y1]) for b in blocks]
    heads = [sum(x.kind == HEAD for x in boxes) for boxes in found]
    assert heads[0] >= 2  # la vallée : personnages vus de dos
    assert sum(x.kind == BUBBLE for boxes in found for x in boxes) == 7
    assert any(x.kind == FREE_TEXT for x in found[1])  # FLAP


def test_manual_spec_has_no_hard_violation() -> None:
    case = load_case(CASE)
    result = analyze_strip(case.strip, spec_provider=ManualSpecProvider.from_file(case.spec_manual), judge=None)
    summary = summarize([evaluate(case, result)])
    assert summary["hard"] == {"head_cuts": 0, "bubble_cuts": 0}
    assert summary["iou_mean"] >= 0.85
