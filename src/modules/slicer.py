"""Module 2 — Smart Slicer : découpe de la bande continue en cases.

Principe
--------
Un webtoon est une longue bande verticale dans laquelle les cases sont
séparées par des « gouttières » : des zones de fond uniforme (blanc, noir ou
gris) sans dessin. Sur une ligne de pixels uniforme, la variance des niveaux de
gris est proche de 0 ; sur une ligne dessinée, elle est nettement plus élevée.

Algorithme :

1. ``gray = cv2.cvtColor(rgb, COLOR_RGB2GRAY)`` puis variance par ligne
   (``compute_row_variance``) ;
2. ``gutter_mask = row_variance < variance_threshold`` ; les runs de ``True``
   d'au moins ``min_gap`` lignes consécutives sont des gouttières
   (``find_gutters``, vectorisé avec ``np.diff`` sur le masque paddé) ;
3. les segments de contenu sont le complément des gouttières (les lignes avant
   la première gouttière et après la dernière sont incluses) ;
4. chaque segment est étendu de ``margin_padding`` de chaque côté (borné à
   l'image), les segments plus courts que ``min_panel_height`` sont ignorés ;
5. une case plus haute que ``SPLIT_MAX_HEIGHT`` (1200 px) est sous-découpée
   horizontalement en 2 ou 3 blocs de pleine largeur (``part`` = ``top`` /
   ``middle`` / ``bottom``), la coupe étant placée sur la ligne la plus calme
   autour de la coupe idéale ; aucune case n'est jamais rognée en largeur ;
6. chaque case est recadrée (copie) dans un ``Panel`` ; une case plus haute que
   ``giant_panel_height`` reçoit le type ``"scroll_vertical"``, sinon ``"static"``.

Les cases voisines peuvent se chevaucher légèrement à cause du padding : c'est
voulu (aucune ligne de contenu n'est perdue). Avec les valeurs par défaut
(``min_gap`` >= ``margin_padding``) le chevauchement reste dans la gouttière ;
si ``margin_padding > min_gap`` une case peut mordre sur le contenu de sa
voisine, ce que ``slice_panels`` signale par un WARNING.
"""

from __future__ import annotations

import json
import logging
import math
import re
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from src.models.panel import Panel, PanelType
from src.utils.image_utils import rgb_to_bgr, rgb_to_gray, to_numpy_rgb

logger = logging.getLogger(__name__)

# --- Paramètres par défaut (cf. PRD, Module 2) ---------------------------------
#: Variance (niveaux de gris) en dessous de laquelle une ligne est une gouttière.
DEFAULT_VARIANCE_THRESHOLD: float = 15.0
#: Nombre minimal de lignes consécutives de faible variance pour former une gouttière.
DEFAULT_MIN_GAP: int = 20
#: Marge ajoutée au-dessus et en dessous de chaque case (px).
DEFAULT_MARGIN_PADDING: int = 15
#: Au-delà de cette hauteur (px, strictement), une case est ``scroll_vertical``.
GIANT_PANEL_HEIGHT: int = 1500
#: Hauteur minimale (px, après padding) pour conserver une case : élimine les
#: micro-cases (onomatopées, bruitages, « tick », objets isolés).
MIN_PANEL_HEIGHT: int = 180
DEFAULT_MIN_PANEL_HEIGHT: int = MIN_PANEL_HEIGHT
#: Deux cases voisines dont chacune mesure moins de cette hauteur (px, contenu
#: brut) sont fusionnées en une seule (suites de bulles de chat, lignes de texte).
MERGE_SMALL_BELOW: int = 250
#: Écart maximal (px) entre deux petites cases pour qu'elles soient fusionnées.
MERGE_MAX_GAP: int = 150
#: Au-delà de cette hauteur (px, padding inclus, strictement), une case est sous-découpée
#: **horizontalement** en 2 ou 3 blocs (``top`` / ``middle`` / ``bottom``) qui gardent
#: 100 % de la largeur d'origine ; le montage les enchaîne par un cut, sans défilement.
SPLIT_MAX_HEIGHT: int = 1200
#: Nombre maximal de blocs d'une case coupée.
SPLIT_MAX_PIECES: int = 3
#: Fenêtre de recherche de la ligne de coupe autour de la coupe idéale (fraction de la
#: hauteur d'un bloc) : la ligne de plus faible variance y est choisie pour ne pas
#: trancher un visage ou une bulle.
SPLIT_SEARCH_RATIO: float = 0.15

#: Hauteur maximale (px) d'une colonne de l'overlay de debug : au-delà, la
#: bande est réduite puis, si besoin, découpée en colonnes côte à côte.
DEBUG_OVERLAY_MAX_HEIGHT: int = 8000
#: Largeur (px) en dessous de laquelle l'overlay n'est plus réduit (les
#: libellés resteraient illisibles) : on passe alors en colonnes.
DEBUG_OVERLAY_MIN_WIDTH: int = 320
#: Espace (px) entre deux colonnes de l'overlay.
DEBUG_OVERLAY_TILE_GAP: int = 12
#: Couleur (RGB) des séparateurs de colonnes et du remplissage de la dernière colonne.
_TILE_GAP_COLOR: tuple[int, int, int] = (96, 96, 96)
#: Taille des blocs de lignes traités d'un coup par ``compute_row_variance``
#: (limite la mémoire : 4096 lignes x 800 px x 8 octets = 26 Mo en float64).
_VARIANCE_BLOCK_ROWS: int = 4096

#: Palette (RGB) des rectangles de l'overlay de debug.
_OVERLAY_PALETTE: tuple[tuple[int, int, int], ...] = (
    (230, 40, 40),  # rouge
    (40, 170, 60),  # vert
    (40, 90, 230),  # bleu
    (240, 150, 20),  # orange
    (190, 40, 190),  # magenta
    (20, 170, 190),  # cyan
)


# --- Étape 1 : variance par ligne -----------------------------------------------
def compute_row_variance(strip: np.ndarray | Image.Image) -> np.ndarray:
    """Calcule la variance des niveaux de gris de chaque ligne de pixels.

    L'image est convertie en niveaux de gris (``cv2.COLOR_RGB2GRAY``) puis la
    variance est calculée avec ``np.var(axis=1)`` en ``float64``. Le calcul est
    vectorisé par blocs de ``_VARIANCE_BLOCK_ROWS`` lignes (aucune boucle Python
    sur les lignes) pour ne pas matérialiser toute la bande en float64.

    Args:
        strip: bande RGB (H, W, 3) uint8, gris (H, W) ou image PIL.

    Returns:
        ``np.ndarray`` float64 de forme (H,) : variance de chaque ligne. Une
        bande de hauteur nulle donne un tableau vide.

    Raises:
        TypeError: si ``strip`` n'est ni une image PIL ni un ``np.ndarray``.
        ValueError: si la bande a une largeur nulle (variance indéfinie) ou une
            forme non exploitable.
    """
    if isinstance(strip, Image.Image):
        strip = to_numpy_rgb(strip)
    if not isinstance(strip, np.ndarray):
        raise TypeError(
            f"compute_row_variance attend une image PIL ou un np.ndarray, recu {type(strip)!r}"
        )
    if strip.ndim < 2:
        raise ValueError(f"Image (H, W[, C]) attendue, recu la forme {strip.shape}")

    height, width = strip.shape[:2]
    if width == 0:
        raise ValueError(f"Bande de largeur nulle : forme {strip.shape}")
    if height == 0:
        return np.zeros(0, dtype=np.float64)

    gray = rgb_to_gray(strip)
    if gray.ndim != 2:
        raise ValueError(f"Image grise attendue (H, W), recu {gray.shape}")

    row_variance = np.zeros(height, dtype=np.float64)
    for start in range(0, height, _VARIANCE_BLOCK_ROWS):
        stop = min(start + _VARIANCE_BLOCK_ROWS, height)
        row_variance[start:stop] = gray[start:stop].astype(np.float64).var(axis=1)
    return row_variance


# --- Étape 2 : détection des gouttières -----------------------------------------
def find_gutters(
    row_variance: np.ndarray,
    *,
    variance_threshold: float = DEFAULT_VARIANCE_THRESHOLD,
    min_gap: int = DEFAULT_MIN_GAP,
) -> list[tuple[int, int]]:
    """Trouve les gouttières : runs de lignes de variance < seuil, longs d'au moins ``min_gap``.

    Implémentation vectorisée : le masque booléen est paddé d'un ``False`` de
    chaque côté, ``np.diff`` donne +1 au début d'un run et -1 à sa fin.

    Args:
        row_variance: variance par ligne, tableau 1D de forme (H,) (une séquence
            Python 1D est acceptée).
        variance_threshold: seuil strict (``variance < seuil`` = ligne de gouttière).
        min_gap: longueur minimale (>= 1) d'un run pour être considéré comme gouttière.

    Returns:
        Liste triée d'intervalles ``[start, end)`` (indices de lignes).

    Raises:
        ValueError: si ``row_variance`` n'est pas 1D (par exemple l'image grise
            passée par erreur) ou si ``min_gap < 1``.
    """
    row_variance = np.asarray(row_variance, dtype=np.float64)
    if row_variance.ndim != 1:
        raise ValueError(
            f"row_variance doit etre 1D (H,), recu la forme {row_variance.shape}"
        )
    if min_gap < 1:
        raise ValueError(f"min_gap doit etre >= 1, recu {min_gap}")
    if row_variance.size == 0:
        return []

    mask = row_variance < variance_threshold
    padded = np.concatenate(([False], mask, [False])).astype(np.int8)
    edges = np.diff(padded)
    starts = np.flatnonzero(edges == 1)
    ends = np.flatnonzero(edges == -1)

    keep = (ends - starts) >= int(min_gap)
    return [(int(s), int(e)) for s, e in zip(starts[keep], ends[keep])]


def _content_segments(
    gutters: list[tuple[int, int]], height: int
) -> list[tuple[int, int]]:
    """Retourne le complément des gouttières sur ``[0, height)`` : les segments de contenu."""
    segments: list[tuple[int, int]] = []
    cursor = 0
    for g_start, g_end in gutters:
        if g_start > cursor:
            segments.append((cursor, g_start))
        cursor = max(cursor, g_end)
    if cursor < height:
        segments.append((cursor, height))
    return segments


def split_tall_range(
    y_start: int,
    y_end: int,
    row_variance: np.ndarray | None = None,
    *,
    max_height: int = SPLIT_MAX_HEIGHT,
    max_pieces: int = SPLIT_MAX_PIECES,
    search_ratio: float = SPLIT_SEARCH_RATIO,
) -> list[tuple[int, int, str | None]]:
    """Coupe une plage de lignes trop haute en 2 ou 3 blocs horizontaux de pleine largeur.

    Le nombre de blocs est ``ceil(hauteur / max_height)`` borné à ``max_pieces`` ;
    chaque coupe idéale (blocs égaux) est déplacée vers la ligne de plus faible
    variance dans une fenêtre de ``± search_ratio × hauteur de bloc`` quand
    ``row_variance`` est fourni (coupe dans une zone calme plutôt que dans un dessin).

    Args:
        y_start, y_end: bornes (padding inclus) de la case.
        row_variance: variance par ligne de la bande entière (indices absolus).
        max_height: hauteur au-delà de laquelle on coupe (0 = jamais).
        max_pieces: nombre maximal de blocs (< 2 = jamais).

    Returns:
        ``[(y0, y1, part)]`` : une seule entrée ``part=None`` si la case est
        conservée entière, sinon les blocs jointifs ``top`` / (``middle``) / ``bottom``.
    """
    height = y_end - y_start
    if max_height <= 0 or max_pieces < 2 or height <= max_height or height < 2:
        return [(y_start, y_end, None)]
    n_pieces = min(max_pieces, math.ceil(height / max_height))
    block = height / n_pieces
    cuts: list[int] = []
    for k in range(1, n_pieces):
        ideal = y_start + round(block * k)
        if row_variance is not None:
            radius = int(block * search_ratio)
            lo = max(y_start + 1, ideal - radius, cuts[-1] + 1 if cuts else 0)
            hi = min(y_end - 1, ideal + radius + 1)
            if hi > lo:
                window = np.asarray(row_variance[lo:hi], dtype=np.float64)
                # Parmi les lignes les plus calmes (a 5 % de la plus calme), la plus proche de l'ideal.
                tolerance = window.min() + 0.05 * max(1e-9, window.max() - window.min())
                candidates = np.flatnonzero(window <= tolerance) + lo
                ideal = int(candidates[np.argmin(np.abs(candidates - ideal))])
        cuts.append(ideal)
    bounds = [y_start, *cuts, y_end]
    labels = ["top", "bottom"] if n_pieces == 2 else ["top", *(["middle"] * (n_pieces - 2)), "bottom"]
    return [(bounds[i], bounds[i + 1], labels[i]) for i in range(n_pieces)]


def merge_small_segments(
    segments: list[tuple[int, int]],
    *,
    max_height: int = MERGE_SMALL_BELOW,
    max_gap: int = MERGE_MAX_GAP,
) -> list[tuple[int, int]]:
    """Fusionne les suites de petites cases voisines en un seul segment.

    Une case de contenu plus basse que ``max_height`` est « petite ». Une suite
    de petites cases consécutives, séparées par des gouttières d'au plus
    ``max_gap`` px, devient un seul segment couvrant de la première à la
    dernière (gouttières incluses). Une case normale n'est jamais fusionnée.

    Args:
        segments: segments de contenu ``[start, end)`` triés.
        max_height: hauteur (px) en dessous de laquelle une case est petite.
        max_gap: écart maximal (px) entre deux petites cases à fusionner.

    Returns:
        Segments fusionnés, triés.
    """
    merged: list[tuple[int, int]] = []
    block_is_small = False
    for start, end in segments:
        small = (end - start) < max_height
        if merged and block_is_small and small and start - merged[-1][1] <= max_gap:
            merged[-1] = (merged[-1][0], end)
            continue
        merged.append((start, end))
        block_is_small = small
    return merged


# --- Étape 3 : découpe ------------------------------------------------------------
def slice_panels(
    strip: Image.Image | np.ndarray,
    *,
    variance_threshold: float = DEFAULT_VARIANCE_THRESHOLD,
    min_gap: int = DEFAULT_MIN_GAP,
    margin_padding: int = DEFAULT_MARGIN_PADDING,
    giant_panel_height: int = GIANT_PANEL_HEIGHT,
    min_panel_height: int = MIN_PANEL_HEIGHT,
    merge_small_below: int = MERGE_SMALL_BELOW,
    merge_max_gap: int = MERGE_MAX_GAP,
    split_max_height: int = SPLIT_MAX_HEIGHT,
    split_max_pieces: int = SPLIT_MAX_PIECES,
) -> list[Panel]:
    """Découpe la bande continue en cases (``Panel``) à partir de la variance des lignes.

    Args:
        strip: bande continue (image PIL ou ``np.ndarray`` RGB/gris).
        variance_threshold: seuil de variance (>= 0) sous lequel une ligne est une gouttière.
        min_gap: hauteur minimale (px, >= 1) d'une gouttière.
        margin_padding: marge (px, >= 0) ajoutée de chaque côté de chaque case,
            bornée à l'image. Si elle dépasse ``min_gap``, deux cases voisines
            peuvent se chevaucher au-delà de la gouttière (WARNING).
        giant_panel_height: hauteur (px, >= 0) au-delà de laquelle une case est ``scroll_vertical``.
        min_panel_height: hauteur minimale (px, >= 0, padding inclus) pour conserver
            une case ; 180 px par défaut pour éliminer les micro-cases.
        merge_small_below: les cases voisines plus basses que cette hauteur (px,
            contenu brut) sont fusionnées entre elles (0 = jamais).
        merge_max_gap: écart maximal (px) entre deux petites cases fusionnées.
        split_max_height: une case (padding inclus) plus haute que cette valeur est
            coupée horizontalement en 2 ou 3 blocs de pleine largeur ``top`` /
            ``middle`` / ``bottom`` (0 = jamais) ; le montage les enchaîne par un cut.
        split_max_pieces: nombre maximal de blocs par case coupée.

    Returns:
        Liste de ``Panel`` ordonnée de haut en bas, indices contigus à partir de 0.
        Une bande sans gouttière donne une seule case ; une bande entièrement
        uniforme donne une liste vide.

    Raises:
        ValueError: image vide, ``min_gap < 1`` ou autre paramètre négatif.
    """
    if variance_threshold < 0:
        raise ValueError("variance_threshold doit etre >= 0")
    if min_gap < 1:
        raise ValueError("min_gap doit etre >= 1")
    if margin_padding < 0:
        raise ValueError("margin_padding doit etre >= 0")
    if giant_panel_height < 0:
        raise ValueError("giant_panel_height doit etre >= 0")
    if min_panel_height < 0:
        raise ValueError("min_panel_height doit etre >= 0")
    if merge_small_below < 0 or merge_max_gap < 0:
        raise ValueError("merge_small_below et merge_max_gap doivent etre >= 0")
    if split_max_height < 0:
        raise ValueError("split_max_height doit etre >= 0")
    if split_max_pieces < 1:
        raise ValueError("split_max_pieces doit etre >= 1")
    if margin_padding > min_gap:
        logger.warning(
            "margin_padding (%d) > min_gap (%d): neighbouring panels may overlap "
            "beyond the gutter and include rows of each other's content",
            margin_padding, min_gap,
        )

    rgb = to_numpy_rgb(strip)
    height, width = rgb.shape[:2]
    if height == 0 or width == 0:
        raise ValueError(f"Bande vide : forme {rgb.shape}")

    row_variance = compute_row_variance(rgb)
    gutters = find_gutters(
        row_variance, variance_threshold=variance_threshold, min_gap=min_gap
    )
    segments = _content_segments(gutters, height)
    n_raw_segments = len(segments)
    if merge_small_below > 0:
        segments = merge_small_segments(segments, max_height=merge_small_below, max_gap=merge_max_gap)
    n_merged = n_raw_segments - len(segments)
    if n_merged:
        logger.debug("%d petite(s) case(s) fusionnee(s) avec leur voisine (< %d px)", n_merged, merge_small_below)

    panels: list[Panel] = []
    dropped = 0
    n_split = 0
    kept_segments = 0
    for seg_start, seg_end in segments:
        y_start = max(0, seg_start - margin_padding)
        y_end = min(height, seg_end + margin_padding)
        panel_height = y_end - y_start
        if panel_height < min_panel_height:
            dropped += 1
            logger.debug(
                "Dropping segment [%d, %d) -> padded [%d, %d): height %d < min_panel_height %d",
                seg_start, seg_end, y_start, y_end, panel_height, min_panel_height,
            )
            continue

        pieces = split_tall_range(y_start, y_end, row_variance, max_height=split_max_height, max_pieces=split_max_pieces)
        if len(pieces) > 1:
            n_split += 1
        for y0, y1, part in pieces:
            piece_height = y1 - y0
            panel_type: PanelType = (
                "scroll_vertical" if piece_height > giant_panel_height else "static"
            )
            panels.append(
                Panel(
                    index=len(panels),
                    y_start=y0,
                    y_end=y1,
                    height=piece_height,
                    width=width,
                    type=panel_type,
                    part=part,
                    source_index=kept_segments if part is not None else None,
                    # ``rgb`` est C-contigu : la copie de la tranche l'est aussi.
                    image=rgb[y0:y1].copy(),
                )
            )
        kept_segments += 1
    if n_split:
        logger.debug("%d case(s) haute(s) (> %d px) coupee(s) en 2 ou 3 blocs", n_split, split_max_height)

    if dropped:
        logger.debug("Dropped %d panel(s) shorter than %d px", dropped, min_panel_height)
    if not panels:
        if dropped:
            logger.warning(
                "No panel kept: %d content segment(s) found in the strip (%dx%d) but all "
                "shorter than min_panel_height=%d px after padding=%d "
                "(threshold=%.1f, min_gap=%d)",
                dropped, width, height, min_panel_height, margin_padding,
                variance_threshold, min_gap,
            )
        else:
            logger.warning(
                "No panel found: the strip (%dx%d) seems entirely uniform "
                "(threshold=%.1f, min_gap=%d)",
                width, height, variance_threshold, min_gap,
            )

    n_scroll = sum(1 for p in panels if p.type == "scroll_vertical")
    logger.info(
        "slice_panels: strip %dx%d, %d gutter(s), %d panel(s) (%d scroll_vertical, %d merged, %d split, %d dropped)",
        width, height, len(gutters), len(panels), n_scroll, n_merged, n_split, dropped,
    )
    return panels


# --- Sorties disque -----------------------------------------------------------------
def _write_png(path: Path, rgb: np.ndarray) -> None:
    """Encode un tableau RGB en PNG via ``cv2.imencode`` (robuste aux chemins non ASCII)."""
    ok, buffer = cv2.imencode(".png", rgb_to_bgr(rgb))
    if not ok:
        raise OSError(f"Echec de l'encodage PNG pour {path}")
    path.write_bytes(buffer.tobytes())


def _remove_stale_panel_files(out_dir: Path, prefix: str, keep: set[str]) -> int:
    """Supprime les ``<prefix>_NNN.png`` de ``out_dir`` absents de ``keep`` ; retourne leur nombre."""
    pattern = re.compile(rf"^{re.escape(prefix)}_\d{{3,}}\.png$")
    removed = 0
    for entry in out_dir.iterdir():
        if entry.is_file() and entry.name not in keep and pattern.match(entry.name):
            entry.unlink()
            removed += 1
    return removed


def save_panels(
    panels: list[Panel],
    out_dir: str | Path,
    prefix: str = "panel",
    *,
    clean: bool = True,
) -> list[Path]:
    """Écrit chaque case en PNG dans ``out_dir`` ainsi qu'un ``panels.json`` de métadonnées.

    Les fichiers sont nommés ``<prefix>_<index:03d>.png``. Le JSON contient une
    liste de dicts ``{index, y_start, y_end, height, width, type, file}`` (sans
    les pixels), ``file`` étant le nom du fichier PNG relatif à ``out_dir``.
    ``panels.json`` est la source de vérité : par défaut (``clean=True``) les
    ``<prefix>_NNN.png`` d'une exécution précédente qui ne correspondent à aucune
    case actuelle sont supprimés, afin qu'un ``glob`` du dossier ne retourne pas
    de cases fantômes.

    Args:
        panels: cases à sauvegarder.
        out_dir: dossier de sortie (créé si besoin).
        prefix: préfixe des fichiers PNG.
        clean: supprimer les PNG ``<prefix>_NNN.png`` obsolètes du dossier.

    Returns:
        Liste des chemins des PNG écrits, dans l'ordre des cases.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    filenames = [f"{prefix}_{panel.index:03d}.png" for panel in panels]
    if clean:
        removed = _remove_stale_panel_files(out_dir, prefix, keep=set(filenames))
        if removed:
            logger.info("Removed %d stale %s_*.png file(s) from %s", removed, prefix, out_dir)

    written: list[Path] = []
    metadata: list[dict] = []
    for panel, filename in zip(panels, filenames):
        path = out_dir / filename
        _write_png(path, panel.image)
        written.append(path)
        entry = panel.to_dict()
        entry["file"] = filename
        metadata.append(entry)

    json_path = out_dir / "panels.json"
    json_path.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=True), encoding="utf-8"
    )
    logger.info("Saved %d panel(s) + %s in %s", len(written), json_path.name, out_dir)
    return written


def load_panels(out_dir: str | Path) -> list[Panel]:
    """Recharge les cases écrites par :func:`save_panels` (``panels.json`` + PNG).

    Permet aux modules aval (analyzer, TTS, builder) de repartir d'un dossier de
    cases sans refaire le scraping ni la découpe.

    Args:
        out_dir: dossier contenant ``panels.json`` et les PNG qu'il référence.

    Returns:
        Liste de ``Panel`` dans l'ordre de ``panels.json``.

    Raises:
        FileNotFoundError: ``panels.json`` ou un PNG référencé est absent.
        ValueError: ``panels.json`` mal formé (clé manquante, index dupliqué) ou
            métadonnées incohérentes avec le PNG (validation ``Panel``).
    """
    out_dir = Path(out_dir)
    json_path = out_dir / "panels.json"
    if not json_path.is_file():
        raise FileNotFoundError(f"panels.json introuvable dans {out_dir}")
    try:
        entries = json.loads(json_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"{json_path} n'est pas un JSON valide : {exc}") from exc
    if not isinstance(entries, list):
        raise ValueError(f"{json_path} doit contenir une liste de cases")

    panels: list[Panel] = []
    seen_indexes: set[int] = set()
    for position, entry in enumerate(entries):
        try:
            index = int(entry["index"])
            png_path = out_dir / str(entry["file"])
            fields = {
                "y_start": entry["y_start"],
                "y_end": entry["y_end"],
                "height": entry["height"],
                "width": entry["width"],
                "type": entry["type"],
                "part": entry.get("part"),
                "source_index": entry.get("source_index"),
            }
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"Entree #{position} de {json_path} invalide : {exc!r}") from exc
        if index in seen_indexes:
            raise ValueError(f"Index de case duplique dans {json_path} : {index}")
        seen_indexes.add(index)
        if not png_path.is_file():
            raise FileNotFoundError(f"Case manquante : {png_path}")
        with Image.open(png_path) as img:
            rgb = to_numpy_rgb(img).copy()
        panels.append(Panel(index=index, image=rgb, **fields))
    logger.info("%d case(s) rechargee(s) depuis %s", len(panels), json_path)
    return panels


def _draw_label(
    canvas: np.ndarray,
    text: str,
    x: int,
    y: int,
    color: tuple[int, int, int],
    font_scale: float,
    thickness: int,
) -> None:
    """Dessine ``text`` en blanc sur un cartouche de couleur ``color`` au point (x, y)."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    canvas_h, canvas_w = canvas.shape[:2]
    (text_w, text_h), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    tx, ty = x + 6, min(y + text_h + 8, canvas_h - 1)
    cv2.rectangle(
        canvas,
        (tx - 3, ty - text_h - 4),
        (min(tx + text_w + 3, canvas_w - 1), ty + baseline + 2),
        color,
        cv2.FILLED,
    )
    cv2.putText(canvas, text, (tx, ty), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)


def render_debug_overlay(
    strip: Image.Image | np.ndarray,
    panels: list[Panel],
    path: str | Path,
    *,
    max_height: int = DEBUG_OVERLAY_MAX_HEIGHT,
    min_width: int = DEBUG_OVERLAY_MIN_WIDTH,
) -> Path:
    """Dessine les cases détectées (rectangles colorés + libellés) sur une copie de la bande.

    Mise à l'échelle :

    - si la bande tient dans ``max_height`` lignes, elle est dessinée telle quelle ;
    - sinon elle est réduite (ratio conservé) pour tenir dans ``max_height``,
      mais jamais en dessous de ``min(min_width, largeur)`` px de large : un
      chapitre complet (60-100 morceaux de 1280 px) resterait sinon une
      colonne de quelques dizaines de pixels aux libellés illisibles ;
    - si, à cette échelle plancher, la bande dépasse encore ``max_height``, elle
      est découpée en colonnes de ``max_height`` lignes posées côte à côte
      (séparées par ``DEBUG_OVERLAY_TILE_GAP`` px), de gauche à droite dans
      l'ordre de lecture. Une case qui continue dans la colonne suivante y reçoit
      un libellé ``#i (suite)``.

    Les coordonnées des cases sont mises à l'échelle en conséquence ; la bande
    d'origine n'est jamais modifiée.

    Args:
        strip: bande d'origine (PIL ou ``np.ndarray``).
        panels: cases issues de ``slice_panels``.
        path: chemin du PNG de sortie.
        max_height: hauteur maximale d'une colonne de l'overlay (>= 1).
        min_width: largeur plancher de la bande réduite (>= 1).

    Returns:
        Chemin du fichier écrit.

    Raises:
        ValueError: bande vide ou ``max_height``/``min_width`` < 1.
    """
    if max_height < 1 or min_width < 1:
        raise ValueError("max_height et min_width doivent etre >= 1")
    rgb = to_numpy_rgb(strip)
    height, width = rgb.shape[:2]
    if height == 0 or width == 0:
        raise ValueError(f"Bande vide : forme {rgb.shape}")

    # Échelle : réduction pour tenir dans max_height, bornée par l'échelle
    # plancher qui garantit min_width px de large (on n'agrandit jamais).
    scale = 1.0
    target_h = height
    if height > max_height:
        fit_scale = max_height / height
        floor_scale = min(1.0, min_width / width)
        if fit_scale >= floor_scale:
            scale, target_h = fit_scale, max_height  # tient dans une colonne
        else:
            scale, target_h = floor_scale, max(1, int(round(height * floor_scale)))

    if scale < 1.0:
        scaled_w = max(1, int(round(width * scale)))
        scaled = cv2.resize(rgb, (scaled_w, target_h), interpolation=cv2.INTER_AREA)
    elif not rgb.flags.writeable or (
        isinstance(strip, np.ndarray) and np.may_share_memory(rgb, strip)
    ):
        # Copie obligatoire : tableau en lecture seule (export Pillow, adossé à
        # des ``bytes``) ou tableau de l'appelant, qu'on ne modifie jamais.
        scaled = rgb.copy()
    else:
        scaled = rgb  # ``to_numpy_rgb`` a déjà produit un tableau neuf et modifiable
    scaled_h, scaled_w = scaled.shape[:2]

    thickness = 2
    if scaled_w >= 600:
        font_scale, text_thickness = 0.8, 2
    elif scaled_w >= min_width:
        font_scale, text_thickness = 0.6, 2
    else:
        font_scale, text_thickness = 0.45, 1

    for panel in panels:
        color = _OVERLAY_PALETTE[panel.index % len(_OVERLAY_PALETTE)]
        y0 = int(round(panel.y_start * scale))
        y1 = int(round(panel.y_end * scale)) - 1
        y0 = min(max(y0, 0), scaled_h - 1)
        y1 = min(max(y1, y0), scaled_h - 1)
        # Léger décalage horizontal alterné : les bords de deux cases qui se
        # chevauchent (padding) restent tous deux visibles.
        inset = 3 * (panel.index % 3)
        x0, x1 = inset, max(inset, scaled_w - 1 - inset)

        # Teinte translucide de la bande de la case (les chevauchements ressortent).
        band = scaled[y0 : y1 + 1]
        tint = np.empty_like(band)
        tint[:] = color
        band[:] = cv2.addWeighted(band, 0.88, tint, 0.12, 0.0)

        cv2.rectangle(scaled, (x0, y0), (x1, y1), color, thickness)

        part = f" {panel.part}" if panel.part else ""
        label = f"#{panel.index}{part} {panel.type} {panel.height}px"
        _draw_label(scaled, label, x0, y0, color, font_scale, text_thickness)
        # Rappel du libellé en tête de chaque colonne où la case se poursuit.
        for boundary in range(max_height, y1 + 1, max_height):
            if boundary > y0:
                _draw_label(
                    scaled, f"#{panel.index} (suite)", x0, boundary, color,
                    font_scale, text_thickness,
                )

    n_tiles = max(1, math.ceil(scaled_h / max_height))
    if n_tiles > 1:
        gap = DEBUG_OVERLAY_TILE_GAP
        canvas = np.empty(
            (max_height, n_tiles * scaled_w + (n_tiles - 1) * gap, 3), dtype=np.uint8
        )
        canvas[:] = _TILE_GAP_COLOR
        for tile in range(n_tiles):
            chunk = scaled[tile * max_height : (tile + 1) * max_height]
            x = tile * (scaled_w + gap)
            canvas[: chunk.shape[0], x : x + scaled_w] = chunk
    else:
        canvas = scaled
    canvas_h, canvas_w = canvas.shape[:2]

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    _write_png(path, canvas)
    logger.info(
        "Debug overlay written to %s (%dx%d, scale=%.3f, %d column(s))",
        path, canvas_w, canvas_h, scale, n_tiles,
    )
    return path


__all__ = [
    "DEFAULT_VARIANCE_THRESHOLD",
    "DEFAULT_MIN_GAP",
    "DEFAULT_MARGIN_PADDING",
    "GIANT_PANEL_HEIGHT",
    "MIN_PANEL_HEIGHT",
    "DEFAULT_MIN_PANEL_HEIGHT",
    "MERGE_SMALL_BELOW",
    "MERGE_MAX_GAP",
    "SPLIT_MAX_HEIGHT",
    "SPLIT_MAX_PIECES",
    "SPLIT_SEARCH_RATIO",
    "merge_small_segments",
    "split_tall_range",
    "DEBUG_OVERLAY_MAX_HEIGHT",
    "DEBUG_OVERLAY_MIN_WIDTH",
    "DEBUG_OVERLAY_TILE_GAP",
    "Panel",
    "PanelType",
    "compute_row_variance",
    "find_gutters",
    "slice_panels",
    "save_panels",
    "load_panels",
    "render_debug_overlay",
]
