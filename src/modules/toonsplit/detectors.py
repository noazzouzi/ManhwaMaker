"""Étape 2 : détecteurs au pixel près (têtes, personnes, bulles, texte, énergie de coupe).

Deux points d'extension, lus à chaque appel par :func:`detect_all` :

- :data:`SUBJECT_DETECTOR` : ``image BGR -> [Box]`` de type ``head`` (contrainte dure) et
  ``person`` (zone à préférer). Défaut : détecteurs deepghs (licence MIT).
- :data:`BUBBLE_DETECTOR` : ``image BGR -> [Box]`` de type ``bubble`` (contrainte dure) et
  ``free_text`` (texte hors bulle : onomatopée ou narration). Défaut : fusion du détecteur
  classique du prototype, du RT-DETR ``ogkalu/comic-text-and-bubble-detector``
  (Apache-2.0) et, s'il a été exporté en ONNX, du YOLOv8
  ``ogkalu/comic-speech-bubble-detector-yolov8m`` (Apache-2.0).

Les modèles tournent en ONNX avec ``onnxruntime`` : ni ``dghs-imgutils`` (qui impose
``numpy<2`` et casserait l'environnement), ni ``ultralytics`` (AGPL-3.0) à l'exécution.
Le pré- et post-traitement YOLO reproduit celui d'``imgutils.generic.yolo``.

Les blocs sont analysés en **résolution native** par tuiles carrées de côté la largeur
du bloc, avec 50 % de recouvrement : tout objet de moins d'une demi-largeur de haut est
entier dans au moins une tuile.
"""

from __future__ import annotations

import logging
import os
import threading
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from src.modules.toonsplit.blocks import scaled
from src.modules.toonsplit.geometry import (
    BUBBLE, FREE_TEXT, HEAD, PERSON, Box, containment, enclosing, fuse_boxes, intersection, iou,
)
from src.utils.config import PROJECT_ROOT

logger = logging.getLogger(__name__)

Detector = Callable[[np.ndarray], list[Box]]

#: Modèles par défaut (dépôt Hugging Face, fichier, seuil recommandé par la fiche du modèle).
HEAD_MODEL = ("deepghs/anime_head_detection", "head_detect_v2.0_s/model.onnx", 0.413)
PERSON_MODEL = ("deepghs/anime_person_detection", "person_detect_v1.3_s/model.onnx", 0.324)
COMIC_DETR_MODEL = ("ogkalu/comic-text-and-bubble-detector", "detector.onnx")
#: YOLOv8 bulles exporté en ONNX, **en option** : ``TOONSPLIT_BUBBLE_YOLO=1`` (fichier par
#: défaut dans ``models/``) ou ``=chemin.onnx``. Classes : ``text_bubble``, ``text_free``.
#: La fiche du modèle dit Apache-2.0 mais il a été entraîné avec Ultralytics, qui revendique
#: l'AGPL-3.0 sur les modèles produits : option désactivée par défaut.
BUBBLE_YOLO_ENV = "TOONSPLIT_BUBBLE_YOLO"
BUBBLE_YOLO_DEFAULT = PROJECT_ROOT / "models" / "comic-speech-bubble-detector.onnx"
BUBBLE_YOLO_KINDS = {"text_bubble": BUBBLE, "text_free": FREE_TEXT}
#: Seuils du RT-DETR. Les classes qui deviennent des contraintes dures (bulles) demandent de
#: la précision ; le texte libre ne sert qu'à une pénalité souple, un faux positif y coûte peu.
DETR_THRESHOLDS = {"bubble": 0.5, "text_bubble": 0.5, "text_free": 0.25}
DETR_LABELS = ("bubble", "text_bubble", "text_free")


# --- Détecteur classique (prototype, mis à l'échelle) ----------------------------------
def detect_text_boxes(block: np.ndarray) -> list[Box]:
    """Lignes de glyphes sombres sur fond blanc, étendues à leur bulle (flood fill).

    Algorithme du prototype ; ses longueurs, calibrées à 575 px de large, sont mises à
    l'échelle de la largeur réelle.
    """
    g = cv2.cvtColor(block, cv2.COLOR_BGR2GRAY)
    H, W = g.shape
    s = W / 575.0
    px = lambda v: scaled(v, W)  # noqa: E731
    dark = (g < 110).astype(np.uint8)
    lines = cv2.morphologyEx(dark, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (px(15), px(3))))
    n, _, st, _ = cv2.connectedComponentsWithStats(lines)
    cand = []
    for i in range(1, n):
        x, y, w, h, a = (int(v) for v in st[i])
        if not (px(7) <= h <= px(40) and w >= px(12) and w < 0.8 * W):
            continue
        r = px(6)
        ring = g[max(0, y - r):y + h + r, max(0, x - r):x + w + r]
        if np.percentile(ring, 60) < 215:
            continue
        if a / (w * h) < 0.25 or w < 1.8 * h:
            continue
        sub = dark[y:y + h, x:x + w]
        _, _, gs, _ = cv2.connectedComponentsWithStats(sub)
        gl = [q for q in gs[1:] if q[3] >= 0.5 * h and q[4] >= 6 * s * s]
        if len(gl) < 3:
            continue
        hs = np.array([q[3] for q in gl], dtype=float)
        bots = np.array([q[1] + q[3] for q in gl], dtype=float)
        if hs.std() / hs.mean() > 0.35 or bots.std() > 0.25 * h:
            continue
        cand.append([x, y, x + w, y + h])
    cand.sort(key=lambda b: (b[1], b[0]))
    groups: list[list[int]] = []
    for b in cand:
        for t in groups:
            if b[1] - t[3] < px(22) and b[0] < t[2] + px(40) and b[2] > t[0] - px(40):
                t[:] = [min(t[0], b[0]), min(t[1], b[1]), max(t[2], b[2]), max(t[3], b[3])]
                break
        else:
            groups.append(list(b))
    white = ((g > 200) * 255).astype(np.uint8)
    out: list[Box] = []
    for x0, y0, x1, y1 in groups:
        mask = np.zeros((H + 2, W + 2), np.uint8)
        seed = (max(0, x0 - px(3)), (y0 + y1) // 2)
        bx = None
        if white[seed[1], seed[0]]:
            cv2.floodFill(white.copy(), mask, seed, 128, flags=4 | (255 << 8) | cv2.FLOODFILL_MASK_ONLY)
            ys, xs = np.where(mask[1:-1, 1:-1])
            if len(xs):
                bx = (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))
        leaked = bx is None or (bx[2] - bx[0]) > 0.9 * W or (bx[3] - bx[1]) > 6 * (y1 - y0 + px(40))
        if leaked:  # texte flottant : large marge (contours en pointes)
            bx = (x0 - px(30), y0 - px(45), x1 + px(30), y1 + px(45))
        else:  # inclure le contour de la bulle
            bx = (bx[0] - px(12), bx[1] - px(12), bx[2] + px(12), bx[3] + px(12))
        out.append(Box(max(0, bx[0]), max(0, bx[1]), min(W, bx[2]), min(H, bx[3]), BUBBLE, 0.5, "classic"))
    return fuse_boxes(out, iou_threshold=0.5, contain_threshold=0.9)


def cut_energy(block: np.ndarray) -> np.ndarray:
    """Énergie de gradient vertical par ligne, normalisée à [0, 1] : couper au calme se voit moins."""
    g = cv2.cvtColor(block, cv2.COLOR_BGR2GRAY).astype(np.float32)
    e = np.abs(cv2.Sobel(g, cv2.CV_32F, 0, 1)).mean(1)
    return e / (e.max() + 1e-6)


# --- Inférence ONNX ------------------------------------------------------------------------
_SESSIONS: dict[str, Any] = {}
_SESSION_LOCK = threading.Lock()
#: Avertissements déjà émis (un seul par détecteur indisponible).
_WARNED: set[str] = set()


def _session(path: str | Path) -> Any:
    import onnxruntime as ort

    key = str(path)
    with _SESSION_LOCK:
        if key not in _SESSIONS:
            options = ort.SessionOptions()
            options.log_severity_level = 3
            _SESSIONS[key] = ort.InferenceSession(key, sess_options=options, providers=["CPUExecutionProvider"])
        return _SESSIONS[key]


@lru_cache(maxsize=None)
def hf_file(repo: str, filename: str) -> str:
    """Chemin local d'un fichier Hugging Face (téléchargé une fois dans le cache HF)."""
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
    from huggingface_hub import hf_hub_download

    try:  # déjà en cache : aucun accès réseau (déterminisme, hors ligne)
        return hf_hub_download(repo, filename, local_files_only=True)
    except Exception:  # noqa: BLE001 - premier usage : téléchargement
        return hf_hub_download(repo, filename)


def square_tiles(height: int, width: int) -> list[int]:
    """Ordonnées des tuiles carrées (côté ``width``, pas d'une demi-largeur, dernière calée en bas)."""
    if height <= width:
        return [0]
    step = max(1, width // 2)
    return sorted(set(list(range(0, height - width, step)) + [height - width]))


def _tile_input(tile: np.ndarray, side: int, size: int, pad_value: int) -> np.ndarray:
    if tile.shape[0] < side:
        pad = np.full((side - tile.shape[0], tile.shape[1], 3), pad_value, np.uint8)
        tile = np.vstack([tile, pad])
    rgb = cv2.cvtColor(cv2.resize(tile, (size, size), interpolation=cv2.INTER_LINEAR), cv2.COLOR_BGR2RGB)
    return (rgb.astype(np.float32) / 255.0).transpose(2, 0, 1)[None]


def nms(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float) -> list[int]:
    """NMS glouton (même formule qu'``imgutils``) ; ordre déterministe (score puis index)."""
    if not len(boxes):
        return []
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = np.lexsort((np.arange(len(scores)), -scores))
    keep: list[int] = []
    while order.size:
        i = int(order[0])
        keep.append(i)
        xx1, yy1 = np.maximum(x1[i], x1[order[1:]]), np.maximum(y1[i], y1[order[1:]])
        xx2, yy2 = np.minimum(x2[i], x2[order[1:]]), np.minimum(y2[i], y2[order[1:]])
        inter = np.maximum(0.0, xx2 - xx1 + 1) * np.maximum(0.0, yy2 - yy1 + 1)
        overlap = inter / (areas[i] + areas[order[1:]] - inter)
        order = order[1:][overlap <= iou_threshold]
    return keep


@dataclass
class TileHit:
    """Détection dans une tuile ; ``truncated`` si elle touche un bord intérieur de la tuile."""

    box: Box
    truncated: bool


def merge_tile_hits(hits: Sequence[TileHit], *, iou_threshold: float = 0.5) -> list[Box]:
    """Fusionne les détections de tuiles qui se recouvrent.

    Les détections entières passent d'abord (NMS par type). Une détection tronquée par un
    bord de tuile est jetée si une détection gardée la contient, sinon fusionnée (boîte
    englobante) avec celle qu'elle chevauche : un objet plus haut que le recouvrement
    reste couvert en entier.
    """
    ordered = sorted(hits, key=lambda h: (h.truncated, -h.box.score, h.box.y0, h.box.x0))
    kept: list[Box] = []
    for hit in ordered:
        box = hit.box
        same = [i for i, k in enumerate(kept) if k.kind == box.kind]
        if any(iou(kept[i], box) >= iou_threshold for i in same):
            continue
        if hit.truncated:
            if any(containment(kept[i], box) >= 0.7 and kept[i].area >= box.area for i in same):
                continue
            touching = [i for i in same if intersection(kept[i], box) > 0 and _same_column(kept[i], box)]
            if touching:
                i = touching[0]
                kept[i] = enclosing(kept[i], box)
                continue
        kept.append(box)
    return sorted(kept, key=lambda b: (b.y0, b.x0, b.kind))


@dataclass
class YoloOnnxDetector:
    """Détecteur YOLO (v8 / v11, sortie ``[4 + classes, ancres]``) exporté en ONNX.

    ``kinds`` associe chaque nom de classe du modèle à un type de :class:`Box` ; une simple
    chaîne donne ce type à toutes les classes.
    """

    name: str
    path_fn: Callable[[], str]
    kinds: dict[str, str] | str
    threshold: float
    input_size: int = 640
    iou_threshold: float = 0.7
    pad_value: int = 114
    _labels: list[str] | None = field(default=None, init=False, repr=False)

    def _load(self) -> tuple[Any, list[str]]:
        sess = _session(self.path_fn())
        if self._labels is None:
            names = sess.get_modelmeta().custom_metadata_map.get("names", "")
            parsed = _parse_names(names)
            self._labels = [parsed[i] for i in range(len(parsed))] if parsed else list(self.kinds or [])
        return sess, self._labels

    def __call__(self, img: np.ndarray) -> list[Box]:
        sess, labels = self._load()
        height, width = img.shape[:2]
        n_cls = len(labels)
        input_name = sess.get_inputs()[0].name
        hits: list[TileHit] = []
        for ty in square_tiles(height, width):
            tile = img[ty:ty + width]
            bottom = ty + tile.shape[0]
            out = sess.run(None, {input_name: _tile_input(tile, width, self.input_size, self.pad_value)})[0][0]
            if out.shape[0] != 4 + n_cls and out.shape[1] == 4 + n_cls:
                out = out.T
            scores = out[4:].max(axis=0)
            mask = scores > self.threshold
            if not mask.any():
                continue
            xywh, cls_scores = out[:4, mask].T, out[4:, mask].T
            best = scores[mask]
            xyxy = np.stack([xywh[:, 0] - xywh[:, 2] / 2, xywh[:, 1] - xywh[:, 3] / 2,
                             xywh[:, 0] + xywh[:, 2] / 2, xywh[:, 1] + xywh[:, 3] / 2], axis=1)
            f = width / self.input_size
            for i in nms(xyxy, best, self.iou_threshold):
                label = labels[int(cls_scores[i].argmax())]
                kind = self.kinds if isinstance(self.kinds, str) else self.kinds.get(label)
                if kind is None:
                    continue
                x0, y0, x1, y1 = (float(v) * f for v in xyxy[i])
                box = Box(
                    int(np.clip(round(x0), 0, width)), int(np.clip(round(y0 + ty), ty, bottom)),
                    int(np.clip(round(x1), 0, width)), int(np.clip(round(y1 + ty), ty, bottom)),
                    kind, float(best[i]), self.name,
                )
                if box.area:
                    hits.append(TileHit(box, _truncated(box, ty, bottom, height)))
        return merge_tile_hits(hits)


@dataclass
class ComicDetrDetector:
    """RT-DETR-v2 ``ogkalu/comic-text-and-bubble-detector`` : bulles, texte en bulle, texte libre."""

    name: str = "comic-detr"
    thresholds: dict[str, float] = field(default_factory=lambda: dict(DETR_THRESHOLDS))
    input_size: int = 640
    pad_value: int = 255

    def __call__(self, img: np.ndarray) -> list[Box]:
        sess = _session(hf_file(*COMIC_DETR_MODEL))
        height, width = img.shape[:2]
        hits: list[TileHit] = []
        for ty in square_tiles(height, width):
            tile = img[ty:ty + width]
            bottom = ty + tile.shape[0]
            labels, boxes, scores = sess.run(None, {
                "images": _tile_input(tile, width, self.input_size, self.pad_value),
                "orig_target_sizes": np.array([[width, width]], dtype=np.int64),
            })
            for label, bx, score in zip(labels[0], boxes[0], scores[0]):
                name = DETR_LABELS[int(label)] if 0 <= int(label) < len(DETR_LABELS) else ""
                if not name or score < self.thresholds.get(name, 1.1):
                    continue
                kind = FREE_TEXT if name == "text_free" else BUBBLE
                box = Box(
                    int(np.clip(round(bx[0]), 0, width)), int(np.clip(round(bx[1]) + ty, ty, bottom)),
                    int(np.clip(round(bx[2]), 0, width)), int(np.clip(round(bx[3]) + ty, ty, bottom)),
                    kind, float(score), f"{self.name}:{name}",
                )
                if box.area:
                    hits.append(TileHit(box, _truncated(box, ty, bottom, height)))
        return merge_tile_hits(hits)


def _same_column(a: Box, b: Box) -> bool:
    """Deux morceaux du même objet : ils occupent à peu près la même bande horizontale."""
    overlap = min(a.x1, b.x1) - max(a.x0, b.x0)
    return overlap >= 0.6 * max(1, min(a.w, b.w))


def _truncated(box: Box, top: int, bottom: int, height: int, margin: int = 2) -> bool:
    return (top > 0 and box.y0 <= top + margin) or (bottom < height and box.y1 >= bottom - margin)


def _parse_names(text: str) -> dict[int, str]:
    import ast

    try:
        value = ast.literal_eval(text) if text else {}
    except (ValueError, SyntaxError):
        return {}
    return {int(k): str(v) for k, v in value.items()} if isinstance(value, dict) else {}


# --- Détecteurs par défaut et points d'extension ---------------------------------------
HEAD_DETECTOR = YoloOnnxDetector("deepghs-head", lambda: hf_file(*HEAD_MODEL[:2]), {"head": HEAD}, HEAD_MODEL[2])
PERSON_DETECTOR = YoloOnnxDetector(
    "deepghs-person", lambda: hf_file(*PERSON_MODEL[:2]), {"person": PERSON}, PERSON_MODEL[2],
)
COMIC_DETR = ComicDetrDetector()


def bubble_yolo_path() -> Path | None:
    """ONNX du YOLOv8 bulles si l'option est activée et le fichier présent, sinon ``None``."""
    env = os.environ.get(BUBBLE_YOLO_ENV, "").strip()
    if env.lower() in ("", "0", "off", "false"):
        return None
    path = BUBBLE_YOLO_DEFAULT if env.lower() in ("1", "on", "true") else Path(env)
    if not path.is_file():
        if "yolo-missing" not in _WARNED:
            _WARNED.add("yolo-missing")
            logger.warning("%s=%s : fichier ONNX introuvable (%s)", BUBBLE_YOLO_ENV, env, path)
        return None
    return path


@lru_cache(maxsize=4)
def _bubble_yolo_for(path: str) -> YoloOnnxDetector:
    return YoloOnnxDetector("bubble-yolov8m", lambda: path, BUBBLE_YOLO_KINDS, 0.5, input_size=1024)


def _bubble_yolo() -> YoloOnnxDetector | None:
    path = bubble_yolo_path()
    return None if path is None else _bubble_yolo_for(str(path))


def _safe(detector: Callable[[np.ndarray], list[Box]], img: np.ndarray, name: str) -> list[Box]:
    """Appelle un détecteur ML ; modèle absent ou hors ligne → liste vide et avertissement unique."""
    try:
        return detector(img)
    except Exception as exc:  # noqa: BLE001 - un modèle manquant ne doit pas bloquer la découpe
        if name not in _WARNED:
            _WARNED.add(name)
            logger.warning("Detecteur %s indisponible (%s: %s) : ignore", name, type(exc).__name__, str(exc)[:200])
        return []


def detect_subjects(block: np.ndarray) -> list[Box]:
    """Têtes (contrainte dure) et personnes (zone à préférer)."""
    return _safe(HEAD_DETECTOR, block, "tetes") + _safe(PERSON_DETECTOR, block, "personnes")


def detect_bubbles(block: np.ndarray) -> list[Box]:
    """Bulles (union des détecteurs classique, RT-DETR et YOLOv8 éventuel) et texte libre."""
    boxes = detect_text_boxes(block) + _safe(COMIC_DETR, block, "bulles RT-DETR")
    yolo = _bubble_yolo()
    if yolo is not None:
        boxes += _safe(yolo, block, "bulles YOLOv8")
    bubbles = fuse_boxes([b for b in boxes if b.kind == BUBBLE])
    free = [
        b for b in fuse_boxes([b for b in boxes if b.kind == FREE_TEXT])
        if not any(containment(b, bubble) >= 0.6 and bubble.area >= b.area for bubble in bubbles)
    ]
    return sorted(bubbles + free, key=lambda b: (b.y0, b.x0, b.kind))


#: Points d'extension (remplaçables) : ``image BGR -> [Box]``.
SUBJECT_DETECTOR: Detector = detect_subjects
BUBBLE_DETECTOR: Detector = detect_bubbles


def detect_all(block: np.ndarray) -> list[Box]:
    """Toutes les boîtes d'un bloc via les points d'extension courants."""
    return sorted(SUBJECT_DETECTOR(block) + BUBBLE_DETECTOR(block), key=lambda b: (b.y0, b.x0, b.kind))


__all__ = [
    "Detector", "HEAD_MODEL", "PERSON_MODEL", "COMIC_DETR_MODEL", "BUBBLE_YOLO_ENV", "BUBBLE_YOLO_DEFAULT",
    "BUBBLE_YOLO_KINDS",
    "DETR_THRESHOLDS", "detect_text_boxes", "cut_energy", "hf_file", "square_tiles", "nms", "TileHit",
    "merge_tile_hits", "YoloOnnxDetector", "ComicDetrDetector", "HEAD_DETECTOR", "PERSON_DETECTOR", "COMIC_DETR",
    "bubble_yolo_path", "detect_subjects", "detect_bubbles", "SUBJECT_DETECTOR", "BUBBLE_DETECTOR", "detect_all",
]
