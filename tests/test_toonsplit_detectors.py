"""toonsplit, étape 2 : détecteurs (classique, ONNX simulés, fusion, points d'extension)."""

from __future__ import annotations

import logging
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from src.modules.toonsplit import detectors as d
from src.modules.toonsplit.geometry import BUBBLE, FREE_TEXT, HEAD, PERSON, Box, fuse_boxes
from tests.toonsplit_helpers import art, box, stack, white


# --- Détecteur classique et énergie -----------------------------------------------------------
def bubble_block() -> np.ndarray:
    img = art(500, seed=1)
    cv2.ellipse(img, (200, 250), (150, 70), 0, 0, 360, (255, 255, 255), -1)
    cv2.ellipse(img, (200, 250), (150, 70), 0, 0, 360, (0, 0, 0), 3)
    cv2.putText(img, "HELLO THERE", (100, 245), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
    cv2.putText(img, "MY FRIEND", (115, 280), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
    return img


def test_classic_detector_grows_text_to_its_bubble() -> None:
    boxes = d.detect_text_boxes(bubble_block())
    assert len(boxes) == 1
    b = boxes[0]
    assert b.kind == BUBBLE and b.source == "classic"
    assert b.y0 <= 180 + 5 and b.y1 >= 320 - 5  # l'ellipse (250 ± 70) entière
    assert b.x0 <= 55 and b.x1 >= 345


def test_classic_detector_ignores_art_without_text() -> None:
    assert d.detect_text_boxes(art(500, seed=2)) == []


def test_classic_detector_scales_with_width() -> None:
    small = bubble_block()
    big = cv2.resize(small, (800, 1000), interpolation=cv2.INTER_CUBIC)
    (b_small,), (b_big,) = d.detect_text_boxes(small), d.detect_text_boxes(big)
    assert abs(b_big.y0 - 2 * b_small.y0) <= 12 and abs(b_big.y1 - 2 * b_small.y1) <= 12


def test_cut_energy_is_normalized_and_low_on_flat_rows() -> None:
    img = stack(art(100, seed=3), white(50), art(100, seed=4))
    e = d.cut_energy(img)
    assert e.shape == (250,) and 0 <= e.min() and e.max() <= 1
    assert e[120] < 0.01 < e[50]


# --- Tuiles, NMS, fusion ---------------------------------------------------------------------------
def test_square_tiles_cover_the_block_with_half_overlap() -> None:
    assert d.square_tiles(300, 400) == [0]
    tiles = d.square_tiles(1500, 400)
    assert tiles[0] == 0 and tiles[-1] == 1100
    assert all(b - a <= 200 for a, b in zip(tiles, tiles[1:]))


def test_nms_keeps_best_of_overlapping_boxes() -> None:
    boxes = np.array([[0, 0, 10, 10], [1, 1, 11, 11], [20, 20, 30, 30]], dtype=float)
    assert d.nms(boxes, np.array([0.9, 0.8, 0.7]), 0.5) == [0, 2]


def test_merge_tile_hits_dedups_and_joins_truncated_pieces() -> None:
    whole = box(100, 300, HEAD, score=0.8)
    duplicate = box(102, 301, HEAD, score=0.9)
    inside = box(100, 200, HEAD, score=0.95)  # morceau tronqué contenu dans la détection entière
    top = box(500, 700, BUBBLE)  # un objet haut coupé par deux tuiles
    bottom = box(650, 900, BUBBLE)
    other_column = Box(300, 650, 390, 900, BUBBLE, 0.9, "t")  # autre colonne : pas le même objet
    hits = [d.TileHit(whole, False), d.TileHit(duplicate, False), d.TileHit(inside, True),
            d.TileHit(top, True), d.TileHit(bottom, True), d.TileHit(other_column, True)]
    merged = d.merge_tile_hits(hits)
    heads = [b for b in merged if b.kind == HEAD]
    bubbles = [b for b in merged if b.kind == BUBBLE]
    assert len(heads) == 1 and heads[0].score == 0.9
    assert (500, 900) in [(b.y0, b.y1) for b in bubbles]
    assert len(bubbles) == 2


def test_fuse_boxes_is_union_of_overlapping_boxes() -> None:
    a = Box(10, 10, 100, 100, BUBBLE, 0.5, "classic")
    b = Box(20, 5, 110, 95, BUBBLE, 0.9, "detr")
    c = Box(300, 300, 350, 350, BUBBLE, 0.7, "detr")
    fused = fuse_boxes([a, b, c])
    assert fused[0] == Box(10, 5, 110, 100, BUBBLE, 0.9, "classic+detr")
    assert fused[1] == c


# --- Détecteurs ONNX avec sessions simulées ----------------------------------------------------------
class FakeSession:
    def __init__(self, outputs, names: str = "{0: 'head'}", input_name: str = "images") -> None:
        self.outputs = outputs
        self.feeds: list[dict] = []
        self._meta = SimpleNamespace(custom_metadata_map={"names": names})
        self._input = SimpleNamespace(name=input_name)

    def get_inputs(self):
        return [self._input]

    def get_modelmeta(self):
        return self._meta

    def run(self, _names, feed):
        self.feeds.append(feed)
        out = self.outputs(len(self.feeds) - 1, feed) if callable(self.outputs) else self.outputs
        return out


def yolo_output(dets: list[tuple[float, float, float, float, float]], transpose: bool = False) -> list[np.ndarray]:
    """Sortie YOLO [1, 4+1, N] pour des boîtes (cx, cy, w, h, score) en pixels d'entrée 640."""
    arr = np.array(dets, dtype=np.float32).T[None]  # (1, 5, N)
    return [arr.transpose(0, 2, 1) if transpose else arr]


@pytest.mark.parametrize("transpose", [False, True])
def test_yolo_detector_maps_tiles_back_to_block(monkeypatch: pytest.MonkeyPatch, transpose: bool) -> None:
    # Bloc 400 x 600 : tuiles 0 et 200 ; une tête au centre de chaque tuile (entrée 640 → facteur 400/640).
    session = FakeSession(lambda i, feed: yolo_output([(320, 320, 64, 64, 0.9), (100, 100, 10, 10, 0.1)], transpose))
    monkeypatch.setattr(d, "_session", lambda path: session)
    det = d.YoloOnnxDetector("fake", lambda: "x.onnx", {"head": HEAD}, threshold=0.4)
    boxes = det(art(600, seed=1))
    assert [feed["images"].shape for feed in session.feeds] == [(1, 3, 640, 640)] * 2
    assert [(b.y0, b.y1) for b in boxes] == [(180, 220), (380, 420)]
    assert all(b.kind == HEAD and b.x0 == 180 and b.x1 == 220 for b in boxes)


def test_yolo_detector_pads_short_blocks(monkeypatch: pytest.MonkeyPatch) -> None:
    session = FakeSession(yolo_output([(320, 600, 40, 40, 0.9)]))
    monkeypatch.setattr(d, "_session", lambda path: session)
    det = d.YoloOnnxDetector("fake", lambda: "x.onnx", HEAD, threshold=0.4)
    boxes = det(art(200, seed=1))
    assert session.feeds[0]["images"].shape == (1, 3, 640, 640)
    # la détection tombe dans la zone de remplissage : bornée au bloc (hauteur 200), donc vide
    assert boxes == []


def test_comic_detr_kinds_and_thresholds(monkeypatch: pytest.MonkeyPatch) -> None:
    labels = np.array([[0, 1, 2, 2]])
    boxes = np.array([[[10, 10, 200, 100], [20, 20, 150, 80], [30, 300, 300, 380], [30, 150, 300, 200]]], dtype=np.float32)
    scores = np.array([[0.9, 0.8, 0.3, 0.2]], dtype=np.float32)
    session = FakeSession([labels, boxes, scores])
    monkeypatch.setattr(d, "_session", lambda path: session)
    monkeypatch.setattr(d, "hf_file", lambda repo, name: "detector.onnx")
    found = d.ComicDetrDetector()(art(400, seed=1))
    assert session.feeds[0]["orig_target_sizes"].tolist() == [[400, 400]]
    kinds = sorted((b.kind, b.y0) for b in found)
    # text_free à 0,3 gardé (seuil 0,25), à 0,2 écarté ; text_bubble = bulle
    assert kinds == [(BUBBLE, 10), (BUBBLE, 20), (FREE_TEXT, 300)]


def test_detect_bubbles_fuses_sources_and_drops_text_inside_bubbles(monkeypatch: pytest.MonkeyPatch) -> None:
    classic = [Box(50, 100, 300, 200, BUBBLE, 0.5, "classic")]
    detr = [Box(45, 95, 310, 210, BUBBLE, 0.9, "detr"), Box(60, 110, 200, 190, FREE_TEXT, 0.4, "detr"),
            Box(10, 400, 390, 480, FREE_TEXT, 0.6, "detr")]
    monkeypatch.setattr(d, "detect_text_boxes", lambda img: classic)
    monkeypatch.setattr(d, "COMIC_DETR", lambda img: detr)
    monkeypatch.delenv(d.BUBBLE_YOLO_ENV, raising=False)
    found = d.detect_bubbles(art(500))
    assert [(b.kind, b.y0, b.y1) for b in found] == [(BUBBLE, 95, 210), (FREE_TEXT, 400, 480)]
    assert found[0].source == "classic+detr"


def test_extension_points_are_read_at_call_time(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(d, "SUBJECT_DETECTOR", lambda img: [box(10, 20, HEAD)])
    monkeypatch.setattr(d, "BUBBLE_DETECTOR", lambda img: [box(5, 8, BUBBLE), box(30, 40, PERSON)])
    assert [(b.kind, b.y0) for b in d.detect_all(art(100))] == [(BUBBLE, 5), (HEAD, 10), (PERSON, 30)]


def test_missing_model_degrades_with_single_warning(monkeypatch: pytest.MonkeyPatch, caplog) -> None:
    def broken(img):
        raise FileNotFoundError("model.onnx")

    monkeypatch.setattr(d, "_WARNED", set())
    with caplog.at_level(logging.WARNING, logger=d.__name__):
        assert d._safe(broken, art(10), "tetes") == []
        assert d._safe(broken, art(10), "tetes") == []
    assert sum("tetes" in r.message for r in caplog.records) == 1


def test_bubble_yolo_is_opt_in(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    onnx = tmp_path / "bubbles.onnx"
    onnx.write_bytes(b"x")
    monkeypatch.delenv(d.BUBBLE_YOLO_ENV, raising=False)
    assert d.bubble_yolo_path() is None
    monkeypatch.setenv(d.BUBBLE_YOLO_ENV, str(onnx))
    assert d.bubble_yolo_path() == onnx
    monkeypatch.setenv(d.BUBBLE_YOLO_ENV, str(tmp_path / "absent.onnx"))
    assert d.bubble_yolo_path() is None
