"""Agrandissement IA des cases personnages (Real-ESRGAN anime, ONNX).

Les personnages détectés (:mod:`src.modules.figure_panels`) sont souvent petits : un
personnage de 500 px de haut n'occupe que la moitié d'un cadre 1080p. Chacun est donc
agrandi par Real-ESRGAN ``realesr-animevideov3`` (x4, entraîné sur de l'animation : traits
nets, aplats propres, artefacts JPEG gommés), puis réduit à sa taille cible : il remplit
:data:`DEFAULT_FILL` du cadre (90 %), sans dépasser :data:`MAX_FACTOR` (x4). Un personnage
déjà assez grand (facteur < :data:`MIN_FACTOR`) est gardé tel quel.

L'image agrandie devient la résolution native de la case pour le montage : CapCut et
l'aperçu ffmpeg l'affichent sans autre agrandissement, et le zoom reste plafonné à 105 %
de cette taille (``MAX_ZOOM``). C'est la seule exception à la règle « jamais d'upscale ».

Vitesse mesurée (chapitre de test, 65 personnages, 21 Mpx en entrée) : ~7 s sur GPU via
DirectML (paquet ``onnxruntime-directml``), ~85 s en CPU ; sorties identiques à 1/255
près. Le GPU est pris quand il est disponible, sinon le CPU.

Écartés à la mesure : waifu2x ``cunet`` / ``swin_unet`` (images fausses sous DirectML,
1,5 à 3 fois plus lents en CPU) et Lanczos seul (flou, garde les artefacts JPEG) : ce
dernier ne sert que de secours quand le modèle est indisponible.

Cache : ``figures/hd_<largeur>x<hauteur>/`` (``panels.json`` + PNG, un dossier par cadre),
refait seulement quand les cases personnages, le cadre ou les réglages changent. Dans ce
``panels.json``, ``width`` / ``height`` sont les dimensions agrandies et
``native_width`` / ``native_height`` celles d'origine (le rythme du montage se règle sur
ces dernières) ; ``y_start`` / ``y_end`` restent les positions dans le strip.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import shutil
import threading
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from src.modules.figure_panels import FIGURES_DIRNAME

logger = logging.getLogger(__name__)

#: Real-ESRGAN ``realesr-animevideov3`` (BSD-3-Clause, Xintao Wang) converti en ONNX.
#: Révision et empreinte figées : sortie vérifiée identique (écart 2e-6) à celle des poids
#: officiels ``realesr-animevideov3.pth`` (release GitHub v0.2.5.0 de xinntao/Real-ESRGAN).
MODEL_REPO = "skillsafe-ai/realesr-animevideov3"
MODEL_FILE = "model.onnx"
MODEL_REVISION = "185e9142d439d17e3fb99395600fb7d08af09de5"
MODEL_SHA256 = "78baa685a1a92cac6e14ab2af2a8ad0ef56fa48629563b1414e3a7acc54d86e2"
MODEL_NAME = "realesr-animevideov3"
MODEL_SCALE = 4
#: Part du cadre que vise un personnage agrandi (0,9 = 972 px de haut en 1080p).
DEFAULT_FILL = 0.9
#: Agrandissement maximal (celui du modèle) : au-delà, le dessin n'a plus assez de détail.
MAX_FACTOR = 4.0
#: En dessous, le personnage est gardé tel quel : le gain ne vaut pas le calcul.
MIN_FACTOR = 1.1
#: Version de l'agrandissement : l'incrémenter invalide les dossiers ``hd_*`` existants.
UPSCALE_VERSION = 1
PARAMS_FILE = "upscale_params.json"
#: Écritures PNG menées en parallèle de l'inférence (le modèle, lui, est sérialisé).
IO_WORKERS = 4

#: ``(image RGB, (largeur, hauteur)) -> image RGB`` à cette taille.
UpscaleFn = Callable[[np.ndarray, tuple[int, int]], np.ndarray]


class UpscaleError(RuntimeError):
    """Modèle d'agrandissement introuvable ou corrompu."""


def target_factor(
    width: int, height: int, frame_width: int, frame_height: int, *,
    fill: float = DEFAULT_FILL, max_factor: float = MAX_FACTOR,
) -> float:
    """Agrandissement qui fait remplir ``fill`` du cadre à la case, borné à ``max_factor``."""
    return min(fill * frame_width / width, fill * frame_height / height, max_factor)


def target_size(
    width: int, height: int, frame: tuple[int, int], *,
    fill: float = DEFAULT_FILL, max_factor: float = MAX_FACTOR, min_factor: float = MIN_FACTOR,
) -> tuple[int, int] | None:
    """``(largeur, hauteur)`` agrandies, ou ``None`` si la case est déjà assez grande."""
    factor = target_factor(width, height, frame[0], frame[1], fill=fill, max_factor=max_factor)
    if factor < min_factor:
        return None
    return max(1, round(width * factor)), max(1, round(height * factor))


@lru_cache(maxsize=1)
def model_path() -> str:
    """Chemin local du modèle ONNX (téléchargé une fois, empreinte vérifiée)."""
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
    from huggingface_hub import hf_hub_download

    try:  # déjà en cache : aucun accès réseau
        path = hf_hub_download(MODEL_REPO, MODEL_FILE, revision=MODEL_REVISION, local_files_only=True)
    except Exception:  # noqa: BLE001 - premier usage : téléchargement
        path = hf_hub_download(MODEL_REPO, MODEL_FILE, revision=MODEL_REVISION)
    digest = hashlib.sha256(Path(path).read_bytes()).hexdigest()
    if digest != MODEL_SHA256:
        raise UpscaleError(f"Empreinte inattendue pour {path} : {digest}")
    return path


class Upscaler:
    """Real-ESRGAN anime x4 : GPU (DirectML) quand il est disponible, sinon CPU.

    Appelable comme :data:`UpscaleFn`. Les inférences sont sérialisées (DirectML
    n'accepte pas deux ``run`` simultanés sur une même session) : plusieurs chapitres
    traités en parallèle partagent donc le GPU sans conflit.
    """

    method = MODEL_NAME

    def __init__(self, path: str | Path | None = None) -> None:
        self._path = str(path) if path else None
        self._session: Any = None
        self._lock = threading.Lock()
        #: ``"gpu"`` ou ``"cpu"`` une fois la session ouverte.
        self.device = ""

    def _open(self, gpu: bool) -> tuple[Any, str]:
        import onnxruntime as ort

        options = ort.SessionOptions()
        options.log_severity_level = 3
        providers = ["CPUExecutionProvider"]
        if gpu:
            # Exigences du fournisseur DirectML : pas de motifs mémoire, exécution séquentielle.
            options.enable_mem_pattern = False
            options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
            providers = ["DmlExecutionProvider", *providers]
        session = ort.InferenceSession(self._path or model_path(), sess_options=options, providers=providers)
        return session, "gpu" if session.get_providers()[0] == "DmlExecutionProvider" else "cpu"

    def _ensure_session(self) -> Any:
        if self._session is None:
            import onnxruntime as ort

            gpu = "DmlExecutionProvider" in ort.get_available_providers()
            try:
                self._session, self.device = self._open(gpu)
            except Exception as exc:  # noqa: BLE001 - pilote GPU capricieux : le CPU prend le relais
                if not gpu:
                    raise
                logger.warning("GPU indisponible pour l'agrandissement (%s) : CPU", str(exc)[:200])
                self._session, self.device = self._open(False)
        return self._session

    def load(self) -> str:
        """Ouvre la session maintenant (erreur immédiate si le modèle manque) ; renvoie l'appareil."""
        with self._lock:
            self._ensure_session()
        return self.device

    def super_resolve(self, rgb: np.ndarray) -> np.ndarray:
        """Image RGB ``uint8`` agrandie x4 par le modèle."""
        tensor = np.ascontiguousarray((rgb.astype(np.float32) / 255.0).transpose(2, 0, 1)[None])
        with self._lock:
            session = self._ensure_session()
            feed = {session.get_inputs()[0].name: tensor}
            try:
                out = session.run(None, feed)[0]
            except Exception as exc:  # noqa: BLE001 - échec GPU en cours de route : on refait en CPU
                if self.device != "gpu":
                    raise
                logger.warning("Agrandissement GPU en echec (%s) : bascule sur CPU", str(exc)[:200])
                self._session, self.device = self._open(False)
                out = self._session.run(None, feed)[0]
        return (np.clip(out[0].transpose(1, 2, 0), 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)

    def __call__(self, rgb: np.ndarray, size: tuple[int, int]) -> np.ndarray:
        big = self.super_resolve(rgb)
        if (big.shape[1], big.shape[0]) == tuple(size):
            return big
        return cv2.resize(big, size, interpolation=cv2.INTER_AREA)


def lanczos(rgb: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """Secours sans modèle : agrandissement Lanczos."""
    return cv2.resize(rgb, size, interpolation=cv2.INTER_LANCZOS4)


lanczos.method = "lanczos"  # type: ignore[attr-defined]

_DEFAULT: Upscaler | None = None
_DEFAULT_LOCK = threading.Lock()


def default_upscaler() -> UpscaleFn:
    """Le modèle partagé par tous les chapitres, ou Lanczos s'il ne peut pas être chargé."""
    global _DEFAULT
    with _DEFAULT_LOCK:
        if _DEFAULT is None:
            _DEFAULT = Upscaler()
    try:
        _DEFAULT.load()
    except Exception as exc:  # noqa: BLE001 - hors ligne ou modèle corrompu : la vidéo se fait quand même
        logger.warning("Modele d'agrandissement indisponible (%s) : Lanczos", str(exc)[:200])
        return lanczos
    return _DEFAULT


def upscaled_dirname(frame: tuple[int, int]) -> str:
    return f"hd_{frame[0]}x{frame[1]}"


def _read_rgb(path: Path) -> np.ndarray:
    image = cv2.imdecode(np.fromfile(str(path), np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise OSError(f"Image illisible : {path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def _write_png(path: Path, rgb: np.ndarray) -> None:
    ok, buffer = cv2.imencode(".png", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    if not ok:
        raise OSError(f"Echec de l'encodage PNG pour {path}")
    path.write_bytes(buffer.tobytes())


def _is_cached(dst: Path, params: dict[str, Any]) -> bool:
    params_path, meta_path = dst / PARAMS_FILE, dst / "panels.json"
    if not (params_path.is_file() and meta_path.is_file()):
        return False
    try:
        if json.loads(params_path.read_text(encoding="utf-8")) != params:
            return False
        entries = json.loads(meta_path.read_text(encoding="utf-8"))
    except ValueError:
        return False
    return all((dst / str(entry["file"])).is_file() for entry in entries)


def ensure_upscaled(
    out_dir: str | Path, frame: tuple[int, int], *,
    fill: float = DEFAULT_FILL, max_factor: float = MAX_FACTOR, min_factor: float = MIN_FACTOR,
    force: bool = False, upscale: UpscaleFn | None = None,
) -> Path:
    """Cases personnages agrandies pour ce cadre ; renvoie leur dossier (``panels.json`` + PNG).

    Args:
        out_dir: dossier du chapitre (contenant ``figures/``).
        frame: ``(largeur, hauteur)`` du cadre vidéo.
        fill, max_factor, min_factor: voir :func:`target_size`.
        force: recalculer même si le cache est à jour.
        upscale: fonction d'agrandissement (tests) ; défaut : :func:`default_upscaler`.

    Raises:
        FileNotFoundError: ``figures/panels.json`` absent (cases personnages non calculées).
    """
    src_dir = Path(out_dir) / FIGURES_DIRNAME
    src_json = src_dir / "panels.json"
    if not src_json.is_file():
        raise FileNotFoundError(f"panels.json introuvable dans {src_dir}")
    frame = (int(frame[0]), int(frame[1]))
    dst = src_dir / upscaled_dirname(frame)
    base = {
        "version": UPSCALE_VERSION, "frame": list(frame), "fill": fill, "max_factor": max_factor,
        "min_factor": min_factor, "figures": hashlib.sha256(src_json.read_bytes()).hexdigest(),
    }
    if upscale is None:
        if not force and _is_cached(dst, {**base, "method": MODEL_NAME}):
            return dst
        upscale = default_upscaler()
    params = {**base, "method": getattr(upscale, "method", "custom")}
    if not force and _is_cached(dst, params):
        return dst

    entries = json.loads(src_json.read_text(encoding="utf-8"))
    dst.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()

    def process(entry: dict[str, Any]) -> dict[str, Any]:
        name, width, height = str(entry["file"]), int(entry["width"]), int(entry["height"])
        out = {**entry, "native_width": width, "native_height": height, "upscale": 1.0}
        size = target_size(width, height, frame, fill=fill, max_factor=max_factor, min_factor=min_factor)
        if size is None:
            shutil.copyfile(src_dir / name, dst / name)
            return out
        image = upscale(_read_rgb(src_dir / name), size)
        _write_png(dst / name, image)
        out.update(width=int(image.shape[1]), height=int(image.shape[0]), upscale=round(image.shape[0] / height, 3))
        return out

    with ThreadPoolExecutor(max_workers=IO_WORKERS) as pool:
        written = list(pool.map(process, entries))
    keep = {str(entry["file"]) for entry in written}
    stale = re.compile(r"^panel_\d{3,}\.png$")
    for path in dst.iterdir():
        if path.is_file() and stale.match(path.name) and path.name not in keep:
            path.unlink()
    (dst / "panels.json").write_text(json.dumps(written, indent=1), encoding="utf-8")
    (dst / PARAMS_FILE).write_text(json.dumps(params, indent=1), encoding="utf-8")
    device = getattr(upscale, "device", "")
    logger.info(
        "Personnages agrandis : %d sur %d (%s%s) en %.1fs -> %s",
        sum(1 for entry in written if entry["upscale"] > 1.0), len(written), params["method"],
        f", {device}" if device else "", time.perf_counter() - started, dst,
    )
    return dst


__all__ = [
    "DEFAULT_FILL", "MAX_FACTOR", "MIN_FACTOR", "MODEL_NAME", "UPSCALE_VERSION", "UpscaleError", "UpscaleFn",
    "Upscaler", "default_upscaler", "ensure_upscaled", "lanczos", "model_path", "target_factor", "target_size",
    "upscaled_dirname",
]
