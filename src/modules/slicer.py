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
5. une case plus haute que le cadre vidéo est sous-découpée horizontalement en
   blocs de pleine largeur (``part`` = ``top`` / ``middle``… / ``bottom``) dont
   la hauteur approche celle du cadre, pour qu'ils s'affichent en résolution
   native plutôt que réduits (``split_segment_to_frame``). Le nombre de blocs
   n'est pas plafonné : il est choisi pour maximiser la part d'écran occupée en
   moyenne. Les coupes sont attirées par les frontières réelles détectées
   (``compute_row_change`` / ``find_borders``), car dans les styles où les cases
   sont collées bord à bord il n'existe aucune ligne calme où couper ; aucune
   case n'est jamais rognée en largeur ;
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
from collections.abc import Sequence
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
#: Hauteur du cadre vidéo visée par la sous-découpe. Une case affichée « contain » couvre
#: au mieux l'écran quand sa hauteur vaut exactement celle du cadre : au-dessus elle est
#: réduite, en dessous elle laisse du fond flouté. Mesuré sur un chapitre réel, viser le
#: cadre au lieu d'un seuil arbitraire fait passer la part d'écran occupée par le dessin
#: de 35,8 % à 41,6 %.
DEFAULT_FRAME_HEIGHT: int = 1080
DEFAULT_FRAME_WIDTH: int = 1920
#: Hauteur maximale d'un bloc, en multiple du cadre. Au-delà, la case est forcément
#: recoupée ; en dessous, la découpe n'est faite que si elle améliore la couverture.
SPLIT_MAX_PIECE_RATIO: float = 1.6
#: Pas (px) de la grille des coupes candidates : 24 px est invisible à l'écran et divise
#: par autant le coût de la recherche.
SPLIT_GRID_PX: int = 24
#: Écart vertical (niveaux de gris) au-delà duquel une colonne est dite « changer » d'une
#: ligne à la suivante.
SPLIT_CHANGE_THRESHOLD: int = 24
#: Fraction de la largeur devant changer d'un coup pour qu'une ligne soit une bordure de
#: case. Une vraie bordure traverse toute la largeur ; le haut d'une bulle, non.
SPLIT_BORDER_COVER: float = 0.50
#: Bonus accordé à une coupe tombant sur une bordure réelle, pondéré par sa force. Gardé
#: **strictement sous** la couverture moyenne d'un bloc : une bordure doit pouvoir
#: *déplacer* une coupe de quelques dizaines de pixels, jamais en *créer* une.
SPLIT_BORDER_BONUS: float = 0.04
#: Pénalité d'un bloc plus court que le cadre, quadratique en l'écart relatif. La
#: couverture seule est aveugle à l'équilibre : découper 1 400 px en 1 080 + 320 occupe
#: exactement autant d'écran au total que 700 + 700, alors qu'un ruban de 320 px est bien
#: pire à regarder. Ce terme départage, et lui seul.
SPLIT_BALANCE_COST: float = 0.05
#: Distance (px) au-delà de laquelle une coupe n'est plus considérée « sur » une bordure.
SPLIT_BORDER_SNAP_PX: int = 25

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


def compute_row_change(
    strip: np.ndarray | Image.Image, *, threshold: int = SPLIT_CHANGE_THRESHOLD
) -> np.ndarray:
    """Fraction des colonnes qui changent brutalement entre une ligne et la précédente.

    Complément de :func:`compute_row_variance`, qui ne sait voir qu'une gouttière (zone
    uniforme). Dans les styles où les cases sont **collées bord à bord**, il n'y a aucune
    zone calme à trouver : la frontière est une discontinuité. Mesuré sur un chapitre
    réel, 100 % des coupes placées par la recherche de « ligne calme » tombaient au-dessus
    du seuil de gouttière du module lui-même — autrement dit, en plein dessin.

    Une vraie bordure fait changer *toute* la largeur d'un coup ; le haut d'une bulle de
    dialogue, seulement une partie. La fraction de colonnes qui changent sépare les deux.

    Args:
        strip: bande RGB (H, W, 3) uint8, gris (H, W) ou image PIL.
        threshold: écart de niveau de gris à partir duquel une colonne « change ».

    Returns:
        ``np.ndarray`` float64 de forme (H,), valeurs dans [0, 1]. La première ligne vaut
        toujours 0 (aucune ligne précédente).
    """
    if isinstance(strip, Image.Image):
        strip = to_numpy_rgb(strip)
    if not isinstance(strip, np.ndarray):
        raise TypeError(f"compute_row_change attend une image PIL ou un np.ndarray, recu {type(strip)!r}")
    if strip.ndim < 2:
        raise ValueError(f"Image (H, W[, C]) attendue, recu la forme {strip.shape}")
    height, width = strip.shape[:2]
    if width == 0:
        raise ValueError(f"Bande de largeur nulle : changement indefini (forme {strip.shape})")
    if height < 2:
        return np.zeros(height, dtype=np.float64)

    gray = rgb_to_gray(strip)
    row_change = np.zeros(height, dtype=np.float64)
    # Blocs avec recouvrement d'UNE ligne : sans lui, la ligne de jointure n'aurait pas de
    # predecesseur et une bordure tombant pile sur un bord de bloc serait invisible.
    for start in range(1, height, _VARIANCE_BLOCK_ROWS):
        stop = min(start + _VARIANCE_BLOCK_ROWS, height)
        block = gray[start - 1 : stop].astype(np.int16)
        delta = np.abs(np.diff(block, axis=0))
        row_change[start:stop] = (delta > threshold).mean(axis=1)
    return row_change


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


def frame_coverage(height: int, width: int, frame_width: int, frame_height: int) -> float:
    """Part du cadre vidéo réellement occupée par une case affichée en « contain ».

    L'affichage ne fait jamais d'agrandissement : ``echelle = min(1, cadre/case)``. La
    couverture est donc maximale quand la case remplit exactement la hauteur du cadre, et
    décroît **des deux côtés** : une case trop haute est réduite (et rétrécit aussi en
    largeur), une case trop courte laisse du fond flouté au-dessus et en dessous.
    """
    if height <= 0 or width <= 0 or frame_width <= 0 or frame_height <= 0:
        return 0.0
    scale = min(1.0, frame_width / width, frame_height / height)
    return (width * scale) * (height * scale) / float(frame_width * frame_height)


def find_borders(
    row_change: np.ndarray | None, y_start: int, y_end: int, *, cover: float = SPLIT_BORDER_COVER
) -> list[tuple[int, float]]:
    """Frontières de cases repérées dans une plage : lignes où *toute* la largeur change.

    Args:
        row_change: signal de :func:`compute_row_change` (indices absolus), ou ``None``.
        y_start, y_end: bornes de la plage examinée.
        cover: fraction de largeur minimale pour retenir une ligne.

    Returns:
        ``[(ligne, force)]`` triées, la force valant 0,5 / 0,75 / 1,0 selon l'ampleur du
        changement. Les lignes voisines sont regroupées : une bordure épaisse de quelques
        pixels ne donne qu'un seul candidat, sa ligne la plus marquée.
    """
    if row_change is None or y_end - y_start < 2:
        return []
    window = np.asarray(row_change[y_start + 1 : y_end], dtype=np.float64)
    if window.size == 0:
        return []
    hits = np.flatnonzero(window >= cover)
    if hits.size == 0:
        return []
    borders: list[tuple[int, float]] = []
    group = [int(hits[0])]
    for index in hits[1:]:
        if int(index) - group[-1] <= 20:
            group.append(int(index))
        else:
            borders.append(_border_of(group, window, y_start))
            group = [int(index)]
    borders.append(_border_of(group, window, y_start))
    return borders


def _border_of(group: list[int], window: np.ndarray, y_start: int) -> tuple[int, float]:
    peak = max(group, key=lambda i: window[i])
    value = float(window[peak])
    strength = 1.0 if value >= 0.70 else (0.75 if value >= 0.60 else 0.5)
    return y_start + 1 + peak, strength


def split_segment_to_frame(
    y_start: int,
    y_end: int,
    *,
    width: int,
    frame_width: int = DEFAULT_FRAME_WIDTH,
    frame_height: int = DEFAULT_FRAME_HEIGHT,
    borders: Sequence[tuple[int, float]] = (),
    min_piece: int = MIN_PANEL_HEIGHT,
) -> list[tuple[int, int, str | None]]:
    """Coupe une plage en blocs pleine largeur qui **remplissent** au mieux le cadre vidéo.

    Contrairement à l'ancienne règle (« au-delà de 1200 px, 2 ou 3 blocs égaux »), le
    nombre de blocs n'est pas plafonné et découle de la seule question qui compte : quelle
    découpe montre le plus de dessin à l'écran. Un segment plus court que le cadre reste
    donc entier — le couper ne ferait que rétrécir ses deux moitiés.

    Les coupes candidates sont une grille de :data:`SPLIT_GRID_PX` px, enrichie des
    frontières réelles détectées ; une coupe qui tombe sur une frontière reçoit un bonus,
    ce qui la déplace de quelques dizaines de pixels pour éviter de trancher un dessin.

    Args:
        y_start, y_end: bornes (padding inclus) de la plage.
        width: largeur de la bande (identique pour tous les blocs).
        frame_width, frame_height: cadre visé. ``frame_height <= 0`` désactive la découpe.
        borders: frontières de :func:`find_borders`.
        min_piece: hauteur minimale d'un bloc.

    Returns:
        ``[(y0, y1, part)]`` : une entrée ``part=None`` si la plage reste entière, sinon
        les blocs jointifs ``top`` / (``middle``…) / ``bottom``.
    """
    height = y_end - y_start
    whole = [(y_start, y_end, None)]
    if frame_height <= 0 or height < 2 * min_piece:
        return whole
    # En dessous du cadre, l'echelle d'affichage vaut deja 1 : couper ne montre pas un
    # pixel de plus et multiplie les plans. On ne coupe donc jamais sous cette hauteur,
    # meme si une frontiere passe par la.
    if height <= frame_height:
        return whole
    max_piece = max(min_piece, int(frame_height * SPLIT_MAX_PIECE_RATIO))

    snap = {}
    for row, strength in borders:
        if y_start < row < y_end:
            snap[row] = max(snap.get(row, 0.0), strength)
    nodes = sorted({y_start, y_end, *range(y_start, y_end, SPLIT_GRID_PX), *snap})
    bonus = [SPLIT_BORDER_BONUS * _nearest_border_strength(node, snap) for node in nodes]

    # La duree de la video est fixee par la voix : plus il y a de blocs, moins chacun reste
    # a l'ecran. Le critere est donc la couverture MOYENNE par bloc, pas la somme. Comme
    # une moyenne ne se calcule pas par programmation dynamique, on resout a nombre de
    # blocs FIXE (la somme, elle, se decompose bien) puis on compare les moyennes.
    max_pieces = max(1, height // max(min_piece, 1))
    best = [[-math.inf] * (max_pieces + 1) for _ in nodes]
    prev = [[-1] * (max_pieces + 1) for _ in nodes]
    best[0][0] = 0.0
    for j in range(1, len(nodes)):
        for i in range(j - 1, -1, -1):
            piece = nodes[j] - nodes[i]
            if piece < min_piece:
                continue
            if piece > max_piece:
                break
            shortfall = max(0.0, 1.0 - piece / frame_height)
            gain = frame_coverage(piece, width, frame_width, frame_height)
            gain -= SPLIT_BALANCE_COST * shortfall * shortfall
            # Le bonus recompense la coupe qui OUVRE ce bloc : la borne 0 n'est pas une
            # coupe, c'est le debut du segment.
            gain += bonus[i] if i > 0 else 0.0
            for count in range(max_pieces):
                if best[i][count] == -math.inf:
                    continue
                score = best[i][count] + gain
                if score > best[j][count + 1]:
                    best[j][count + 1], prev[j][count + 1] = score, i

    last = len(nodes) - 1
    chosen = max(
        (n for n in range(1, max_pieces + 1) if best[last][n] > -math.inf),
        key=lambda n: best[last][n] / n,
        default=0,
    )
    if chosen <= 1:
        return whole

    bounds: list[int] = []
    node, count = last, chosen
    while node != -1:
        bounds.append(nodes[node])
        node, count = prev[node][count], count - 1
    bounds.reverse()
    if len(bounds) <= 2:
        return whole
    labels = ["top", *(["middle"] * (len(bounds) - 3)), "bottom"]
    return [(bounds[i], bounds[i + 1], labels[i]) for i in range(len(bounds) - 1)]


def _nearest_border_strength(node: int, snap: dict[int, float]) -> float:
    """Attraction de la frontière la plus proche de ``node``, nulle au-delà de la portée.

    L'attraction **décroît avec la distance** : sans cela, tous les nœuds situés dans la
    portée recevraient le même bonus et la coupe n'aurait aucune raison de tomber sur la
    frontière plutôt qu'à vingt pixels de là.
    """
    if not snap:
        return 0.0
    row = min(snap, key=lambda r: abs(r - node))
    distance = abs(row - node)
    if distance > SPLIT_BORDER_SNAP_PX:
        return 0.0
    return snap[row] * (1.0 - distance / SPLIT_BORDER_SNAP_PX)


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
    frame_height: int = DEFAULT_FRAME_HEIGHT,
    frame_width: int = DEFAULT_FRAME_WIDTH,
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
        frame_height: hauteur (px, >= 0) du cadre vidéo visé. Une case trop haute est
            coupée en blocs de pleine largeur ``top`` / ``middle``… / ``bottom`` dont la
            hauteur approche celle du cadre, pour qu'ils s'affichent en résolution native
            plutôt que réduits. ``0`` désactive complètement la sous-découpe (profils dont
            l'affichage n'est pas un « contain », comme le format court qui recadre).
        frame_width: largeur (px, >= 1) du cadre vidéo visé.

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
    if frame_height < 0:
        raise ValueError("frame_height doit etre >= 0")
    if frame_width < 1:
        raise ValueError("frame_width doit etre >= 1")
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
    # Une passe complete de plus sur la bande : inutile quand la sous-decoupe est coupee.
    row_change = compute_row_change(rgb) if frame_height > 0 else None
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

        pieces = split_segment_to_frame(
            y_start, y_end, width=width, frame_width=frame_width, frame_height=frame_height,
            borders=find_borders(row_change, y_start, y_end), min_piece=min_panel_height,
        )
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
        logger.debug("%d case(s) sous-decoupee(s) pour tenir dans un cadre de %d px", n_split, frame_height)

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
    "split_segment_to_frame",
    "find_borders",
    "compute_row_change",
    "frame_coverage",
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
