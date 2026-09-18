"""Helpers de conversion d'images entre Pillow, NumPy et OpenCV.

Convention du projet : en mémoire, une image est un ``np.ndarray`` RGB de forme
(H, W, 3) et de dtype ``uint8``. OpenCV travaille en BGR : la conversion n'est
faite qu'au moment d'écrire un fichier avec ``cv2``.

Tout tableau d'un autre dtype est ramené en ``uint8`` par ``_to_uint8`` (voir sa
docstring pour les conventions : flottants normalisés dans [0, 1], booléens...).
Une telle conversion est signalée par un log WARNING : l'appelant est censé
fournir de l'``uint8`` et une conversion implicite est le signe d'un pipeline
inattendu.
"""

from __future__ import annotations

import logging

import cv2
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


def _to_uint8(arr: np.ndarray) -> np.ndarray:
    """Ramène un tableau de n'importe quel dtype en ``uint8`` [0, 255].

    Conventions :

    - ``uint8`` : retourné tel quel (aucune copie) ;
    - ``bool`` : ``False`` -> 0, ``True`` -> 255 ;
    - flottant dont toutes les valeurs sont dans [0, 1] : considéré comme
      normalisé et multiplié par 255 (arrondi) ;
    - flottant hors [0, 1] : ``NaN`` -> 0, borné à [0, 255] puis arrondi ;
    - entier : borné à [0, 255].

    Un flottant en échelle 0-255 dont toutes les valeurs seraient <= 1 (image
    quasi noire) est donc interprété comme normalisé : cas limite documenté.
    Toute conversion est loguée en WARNING.
    """
    if arr.dtype == np.uint8:
        return arr

    if arr.dtype == np.bool_:
        out = arr.astype(np.uint8) * np.uint8(255)
        detail = "bool -> {0, 255}"
    elif np.issubdtype(arr.dtype, np.floating):
        values = np.nan_to_num(arr, nan=0.0, posinf=255.0, neginf=0.0)
        if values.size and float(values.min()) >= 0.0 and float(values.max()) <= 1.0:
            out = np.rint(values * 255.0).astype(np.uint8)
            detail = f"{arr.dtype} normalise [0, 1] -> x255"
        else:
            out = np.rint(np.clip(values, 0, 255)).astype(np.uint8)
            detail = f"{arr.dtype} borne a [0, 255]"
    else:
        out = np.clip(arr, 0, 255).astype(np.uint8)
        detail = f"{arr.dtype} borne a [0, 255]"

    logger.warning(
        "Conversion implicite d'un tableau %s en uint8 (%s) : fournir de l'uint8 "
        "pour eviter toute ambiguite", arr.dtype, detail,
    )
    return out


def to_numpy_rgb(img: Image.Image | np.ndarray) -> np.ndarray:
    """Convertit une image PIL ou un tableau NumPy en tableau RGB uint8 (H, W, 3).

    - Une image PIL est convertie en mode ``RGB`` (gestion de ``L``, ``RGBA``, ``P``...).
    - Un tableau 2D (niveaux de gris) est dupliqué sur 3 canaux.
    - Un tableau à 4 canaux (RGBA) perd son canal alpha.
    - Un dtype différent de ``uint8`` est converti par ``_to_uint8`` (flottants
      normalisés dans [0, 1] multipliés par 255, booléens -> 0/255, sinon borné à
      [0, 255]) avec un log WARNING.

    Le tableau retourné est C-contigu. Les tableaux vides (une dimension nulle)
    sont acceptés et retournés avec la forme (H, W, 3) correspondante.

    Aucune copie superflue n'est faite : un tableau RGB uint8 est retourné tel
    quel, et l'export d'une image PIL (une seule copie, adossée aux ``bytes``
    produits par Pillow) est **en lecture seule**. Copier (``.copy()``) avant
    d'écrire dans le résultat.

    Args:
        img: image PIL ou ``np.ndarray`` (H, W), (H, W, 1), (H, W, 3) ou (H, W, 4).

    Returns:
        ``np.ndarray`` RGB uint8 de forme (H, W, 3).

    Raises:
        TypeError: si ``img`` n'est ni une image PIL ni un ``np.ndarray``.
        ValueError: si la forme du tableau n'est pas exploitable.
    """
    if isinstance(img, Image.Image):
        if img.mode != "RGB":
            img = img.convert("RGB")
        return np.ascontiguousarray(np.asarray(img, dtype=np.uint8))

    if not isinstance(img, np.ndarray):
        raise TypeError(
            f"to_numpy_rgb attend une image PIL ou un np.ndarray, recu {type(img)!r}"
        )

    arr = _to_uint8(img)

    # Opérations NumPy (et non cv2.cvtColor) : équivalentes pour GRAY->RGB et
    # RGBA->RGB, et sans assertion OpenCV sur les tableaux vides.
    if arr.ndim == 2:
        arr = np.repeat(arr[:, :, np.newaxis], 3, axis=2)
    elif arr.ndim == 3 and arr.shape[2] == 4:
        arr = arr[:, :, :3]
    elif arr.ndim == 3 and arr.shape[2] == 1:
        arr = np.repeat(arr, 3, axis=2)
    elif arr.ndim != 3 or arr.shape[2] != 3:
        raise ValueError(
            f"Forme de tableau non supportee : {arr.shape} (attendu (H, W[, 1|3|4]))"
        )

    return np.ascontiguousarray(arr)


def to_pil(arr: np.ndarray) -> Image.Image:
    """Convertit un tableau RGB uint8 (H, W, 3) ou gris (H, W) en image PIL.

    Args:
        arr: tableau NumPy (RGB ou niveaux de gris) ; un dtype non ``uint8`` est
            converti par ``_to_uint8`` (log WARNING).

    Returns:
        Image PIL en mode ``RGB`` (ou ``L`` pour un tableau 2D).
    """
    if not isinstance(arr, np.ndarray):
        raise TypeError(f"to_pil attend un np.ndarray, recu {type(arr)!r}")
    arr = _to_uint8(arr)
    if arr.ndim == 2:
        return Image.fromarray(np.ascontiguousarray(arr), mode="L")
    return Image.fromarray(to_numpy_rgb(arr), mode="RGB")


def rgb_to_gray(arr: np.ndarray) -> np.ndarray:
    """Convertit un tableau RGB (H, W, 3) en niveaux de gris (H, W) via OpenCV.

    Un tableau déjà 2D est retourné tel quel (converti en uint8 si besoin).

    Args:
        arr: tableau RGB uint8 (H, W, 3) ou gris (H, W).

    Returns:
        ``np.ndarray`` uint8 de forme (H, W).

    Raises:
        TypeError: si ``arr`` n'est pas un ``np.ndarray``.
        ValueError: si le tableau est vide (``cv2.cvtColor`` refuse une image vide).
    """
    if not isinstance(arr, np.ndarray):
        raise TypeError(f"rgb_to_gray attend un np.ndarray, recu {type(arr)!r}")
    if arr.ndim == 2:
        return _to_uint8(arr)
    rgb = to_numpy_rgb(arr)
    if rgb.size == 0:
        raise ValueError(f"Image vide : forme {rgb.shape}")
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)


def rgb_to_bgr(arr: np.ndarray) -> np.ndarray:
    """Convertit un tableau RGB en BGR (ordre attendu par ``cv2.imwrite``/``imencode``)."""
    return cv2.cvtColor(to_numpy_rgb(arr), cv2.COLOR_RGB2BGR)
