"""Étape 4 : recherche des fenêtres pleine largeur ``(y0, y1)`` sous contraintes.

Le problème est purement vertical. Pour chaque bloc :

1. **Contraintes** (:func:`build_constraints`) : zones ``keep`` / ``drop`` de la spec IA
   converties en pixels, puis « accrochées » : toute tête ou bulle attachée qui touche la
   zone à garder y est intégrée. Contraintes dures : ne couper aucune tête (élargie de
   8 %), aucune bulle, aucun texte de narration ; couvrir au moins 98 % de la zone à garder.
   Les onomatopées sont « tout ou rien » en souple ; les personnes sont des zones à préférer.
2. **Recherche** (:func:`search_crops`) : toutes les fenêtres de hauteur permise (bande de
   ratio largeur/hauteur dure 0,55-1,35) au pas de 4 px, vectorisées, puis affinées au pixel
   autour des meilleures. Score du prototype, plus trois termes nouveaux :

   score = couverture − 3·drop − 0,35·marge − 0,15·énergie − 1,5·hors bande 0,62-1,05
           − 0,3·écart au 2:3 − 0,5·onomatopée tranchée − 0,3·personne rognée

   (les onomatopées incluses ne pèsent que 1 au lieu de 3 : priorité faible). Couper au
   bord du bloc, dans la gouttière, ne coûte aucune énergie.
3. **Cas limites** (:func:`plan_block`) : zone à garder plus haute que le crop maximal →
   plusieurs crops consécutifs si on peut couper sans trancher une tête, une bulle ni une
   personne, sinon un panoramique ``pan`` ; bulle détachée (ou narration) qui chevauche le
   sujet → candidats concurrents « garder les bulles » et « couper le sujet sous les
   visages » ; onomatopée tranchée par le meilleur candidat → candidat « onomatopée
   entière » ; le juge départage.

Tous les poids sont des valeurs a priori, pas ajustées sur les crops de référence.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np

from src.modules.toonsplit.blocks import scaled
from src.modules.toonsplit.geometry import BUBBLE, FREE_TEXT, HEAD, NARRATION, PERSON, SFX, Box, iou1d, overlap1d


@dataclass(frozen=True)
class SearchParams:
    """Paramètres de la recherche (longueurs en pixels du prototype, calibré à 575 px)."""

    ar_soft: tuple[float, float] = (0.62, 1.05)
    ar_hard: tuple[float, float] = (0.55, 1.35)
    #: Règle 3 : préférence autour du 2:3 (nouveau terme, poids a priori).
    ar_target: float = 2 / 3
    step: int = 4
    top: int = 3
    min_cover: float = 0.98
    w_drop: float = 3.0
    w_drop_sfx: float = 1.0
    w_slack: float = 0.35
    w_energy: float = 0.15
    w_off: float = 1.5
    w_ratio: float = 0.3
    w_sfx_cut: float = 0.5
    w_person: float = 0.3
    #: Mode dégradé (aucune fenêtre ne respecte les contraintes dures) : coût par coupe.
    w_relaxed_bubble: float = 1.0
    w_relaxed_head: float = 3.0
    #: Deux candidats sont distincts si l'un de leurs bords diffère de plus de 40 px (à 575 px).
    diversity: int = 40
    #: Marge ajoutée au-dessus et au-dessous de chaque tête (part de sa hauteur : mèches, crêtes).
    head_pad: float = 0.08
    #: Nombre de meilleures fenêtres grossières affinées au pixel.
    refine_seeds: int = 30


@dataclass(frozen=True)
class Zone:
    y0: int
    y1: int
    kind: str
    what: str = ""
    priority: int = 1


@dataclass
class Constraints:
    """Contraintes d'un bloc pour une politique donnée (coordonnées du bloc)."""

    height: int
    width: int
    policy: str
    keep: tuple[int, int] | None
    keep_zones: list[Zone]
    drops: list[Zone]
    hard: list[Box]
    heads: list[Box]
    heads_raw: list[Box]
    bubbles: list[Box]
    narration: list[Box]
    sfx: list[Box]
    persons: list[Box]
    #: Bulles détachées ou narration qui chevauchent le sujet : exclure l'une oblige à rogner l'autre.
    conflicts: list[Box] = field(default_factory=list)
    #: Boîtes souples rendues dures par la politique (``sfx_whole`` : onomatopées entières).
    whole: list[Box] = field(default_factory=list)

    @property
    def keep_h(self) -> int:
        return 0 if self.keep is None else self.keep[1] - self.keep[0]


@dataclass(frozen=True)
class Candidate:
    y0: int
    y1: int
    score: float
    policy: str
    cover: float
    terms: dict[str, float] = field(default_factory=dict, compare=False)

    @property
    def h(self) -> int:
        return self.y1 - self.y0

    def as_dict(self) -> dict[str, Any]:
        return {
            "y0": self.y0, "y1": self.y1, "score": round(self.score, 4), "policy": self.policy,
            "cover": round(self.cover, 4), "terms": {k: round(v, 4) for k, v in self.terms.items()},
        }


@dataclass
class BlockPlan:
    """Résultat de la recherche pour un bloc : candidats à juger et plans retenus."""

    mode: Literal["none", "single", "multi", "pan"]
    candidates: list[Candidate]
    #: Plans retenus par défaut (candidat n° 1) : ``(y0, y1, pan)`` en coordonnées du bloc.
    windows: list[tuple[int, int, bool]]
    constraints: Constraints | None
    conflict: bool = False
    relaxed: bool = False


# --- Contraintes ---------------------------------------------------------------------------
#: Politiques de contraintes et leur description (reprise dans la question posée au juge).
POLICIES: dict[str, str] = {
    "default": "follows the block description",
    "keep_text": "keeps the detached bubbles / narration that overlap the subject whole",
    "cut_subject": "leaves the detached bubbles / narration out, trimming the subject below the faces",
    "sfx_whole": "keeps every sound effect whole or fully out instead of slicing it",
}
def _inside(box: Box, zone: Zone, share: float = 0.5) -> bool:
    """La boîte est-elle majoritairement dans la zone (tolère la dérive des fractions IA) ?"""
    return overlap1d(box.y0, box.y1, zone.y0, zone.y1) >= share * max(1, box.h)


def _zone(y: Sequence[float], height: int, kind: str, what: str, priority: int = 1) -> Zone:
    return Zone(int(round(y[0] * height)), int(round(y[1] * height)), kind, what, priority)


def _snap(k0: int, k1: int, boxes: Sequence[Box]) -> tuple[int, int]:
    """Étend ``[k0, k1)`` à toute boîte qui le touche, jusqu'à stabilité."""
    changed = True
    while changed:
        changed = False
        for b in boxes:
            if overlap1d(b.y0, b.y1, k0, k1) > 0 and (b.y0 < k0 or b.y1 > k1):
                k0, k1 = min(k0, b.y0), max(k1, b.y1)
                changed = True
    return k0, k1


def build_constraints(
    spec: Any, boxes: Sequence[Box], height: int, width: int, params: SearchParams = SearchParams(),
    *, policy: str = "default",
) -> Constraints | None:
    """Contraintes d'un bloc ; ``None`` si la politique est impossible (elle couperait une tête).

    Politiques : ``default`` (la spec telle quelle), ``keep_text`` (les bulles détachées ou la
    narration qui chevauchent le sujet sont gardées entières), ``cut_subject`` (on les exclut
    en rognant le sujet, jamais au-dessus des visages), ``sfx_whole`` (aucune onomatopée
    tranchée : incluse entière ou laissée dehors).
    """
    keep_zones = [_zone(k.y, height, "keep", k.what, getattr(k, "priority", 1)) for k in spec.keep]
    drops = [_zone(d.y, height, d.kind, d.what) for d in spec.drop]
    narration_zones = [z for z in drops if z.kind == "narration"]
    excluding_zones = [z for z in drops if z.kind in ("narration", "detached_bubble")]

    heads_raw = [b for b in boxes if b.kind == HEAD]
    heads = []
    for b in heads_raw:
        pad = int(round(params.head_pad * b.h))
        heads.append(Box(b.x0, max(0, b.y0 - pad), b.x1, min(height, b.y1 + pad), HEAD, b.score, b.source))
    bubbles = [b for b in boxes if b.kind == BUBBLE]
    free = [b for b in boxes if b.kind in (FREE_TEXT, SFX, NARRATION)]
    narration = [b.with_kind(NARRATION) for b in free
                 if b.kind == NARRATION or any(_inside(b, z) for z in narration_zones)]
    sfx = [b.with_kind(SFX) for b in free if not (b.kind == NARRATION or any(_inside(b, z) for z in narration_zones))]
    # Onomatopée signalée par l'IA mais ratée par les détecteurs (lettrage très stylisé) : la
    # zone, grossière, devient une boîte « tout ou rien » pour ne pas la trancher à l'aveugle.
    for z in drops:
        if z.kind == "sfx" and not any(overlap1d(b.y0, b.y1, z.y0, z.y1) > 0 for b in sfx):
            sfx.append(Box(0, max(0, z.y0), width, min(height, z.y1), SFX, 0.5, "spec-zone"))
    persons = [b for b in boxes if b.kind == PERSON]
    excluded = [b for b in bubbles if any(_inside(b, z) for z in excluding_zones)]
    attached = [b for b in bubbles if b not in excluded]

    keep: tuple[int, int] | None = None
    conflicts: list[Box] = []
    if keep_zones:
        k0, k1 = _snap(min(z.y0 for z in keep_zones), max(z.y1 for z in keep_zones), heads + attached)
        conflicts = [b for b in excluded + narration if overlap1d(b.y0, b.y1, k0, k1) > 0]
        if policy == "keep_text" and conflicts:
            drops = [z for z in drops if not any(_inside(b, z) for b in conflicts)]
            k0, k1 = _snap(min(k0, *(b.y0 for b in conflicts)), max(k1, *(b.y1 for b in conflicts)), heads + attached)
        elif policy == "cut_subject" and conflicts:
            subject_heads = [h for h in heads if overlap1d(h.y0, h.y1, k0, k1) > 0]
            centre = (k0 + k1) / 2
            for b in conflicts:
                if (b.y0 + b.y1) / 2 >= centre:
                    k1 = min(k1, b.y0)
                else:
                    k0 = max(k0, b.y1)
            if k1 - k0 < 0.2 * height or any(h.y0 < k0 or h.y1 > k1 for h in subject_heads):
                return None
        elif policy not in POLICIES:
            raise ValueError(f"politique inconnue : {policy}")
        keep = (max(0, k0), min(height, k1))
    whole: list[Box] = []
    if policy == "sfx_whole":
        whole, sfx = sfx, []
    return Constraints(
        height=height, width=width, policy=policy, keep=keep, keep_zones=keep_zones, drops=drops,
        hard=heads + bubbles + narration + whole, heads=heads, heads_raw=heads_raw, bubbles=bubbles,
        narration=narration, sfx=sfx, persons=persons, conflicts=conflicts, whole=whole,
    )


# --- Recherche -----------------------------------------------------------------------------
def boundary_counts(boxes: Sequence[Box], height: int) -> np.ndarray:
    """(H+1,) : nombre de boîtes traversées par une coupe à chaque ligne ``y`` (``y0 < y < y1``)."""
    diff = np.zeros(height + 2, dtype=np.int32)
    for b in boxes:
        lo, hi = max(0, b.y0 + 1), min(height + 1, b.y1)
        if lo < hi:
            diff[lo] += 1
            diff[hi] -= 1
    return np.cumsum(diff)[: height + 1]


def height_bounds(width: int, height: int, params: SearchParams) -> tuple[int, int]:
    """Hauteurs permises (bande de ratio dure), bornées par la hauteur du bloc."""
    hmin = math.ceil(width / params.ar_hard[1])
    hmax = math.floor(width / params.ar_hard[0])
    return min(hmin, height), min(hmax, height)


@dataclass
class _Scorer:
    cons: Constraints
    energy: np.ndarray
    params: SearchParams
    relaxed: bool = False

    def __post_init__(self) -> None:
        c, p = self.cons, self.params
        H = c.height
        self.hard_bubbles = boundary_counts(c.bubbles + c.narration + c.whole, H)
        self.hard_heads = boundary_counts(c.heads, H)
        self.sfx_cuts = boundary_counts(c.sfx, H)
        weights = np.zeros(H, dtype=np.float64)
        for z in c.drops:
            w = p.w_drop_sfx if z.kind == "sfx" else p.w_drop
            weights[max(0, z.y0):max(0, min(H, z.y1))] = np.maximum(weights[max(0, z.y0):max(0, min(H, z.y1))], w)
        self.drop_cum = np.concatenate([[0.0], np.cumsum(weights)])
        edge_energy = np.asarray(self.energy, dtype=np.float64)
        self.cut_energy = np.concatenate([[0.0], edge_energy[1:H], [0.0]]) if H > 1 else np.zeros(H + 1)
        if c.keep is not None:
            k0, k1 = c.keep
            self.persons = [b for b in c.persons if overlap1d(b.y0, b.y1, k0, k1) > 0]
        else:
            self.persons = list(c.persons)

    def __call__(self, y0: np.ndarray, y1: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, np.ndarray]]:
        c, p = self.cons, self.params
        H, W = c.height, c.width
        h = (y1 - y0).astype(np.float64)
        bubble_cuts = self.hard_bubbles[y0] + self.hard_bubbles[y1]
        head_cuts = self.hard_heads[y0] + self.hard_heads[y1]
        if c.keep is not None and c.keep_h > 0:
            k0, k1 = c.keep
            cover = np.clip(np.minimum(y1, k1) - np.maximum(y0, k0), 0, None) / (k1 - k0)
            slack = (h - (k1 - k0)) / H
        else:
            cover = np.ones_like(h)
            slack = h / H
        dropped = (self.drop_cum[y1] - self.drop_cum[y0]) / H
        energy = self.cut_energy[y0] + self.cut_energy[y1]
        soft_lo, soft_hi = W / p.ar_soft[1], W / p.ar_soft[0]
        off = np.maximum(0.0, np.maximum(soft_lo - h, h - soft_hi)) / W
        ratio = np.abs(W / h - p.ar_target)
        sfx_cut = (self.sfx_cuts[y0] + self.sfx_cuts[y1]).astype(np.float64)
        person = np.zeros_like(h)
        for b in self.persons:
            person += (np.clip(y0 - b.y0, 0, b.h) + np.clip(b.y1 - y1, 0, b.h)) / max(1, b.h)
        score = (cover - dropped - p.w_slack * slack - p.w_energy * energy - p.w_off * off - p.w_ratio * ratio
                 - p.w_sfx_cut * sfx_cut - p.w_person * person)
        if self.relaxed:
            score = score - p.w_relaxed_bubble * bubble_cuts - p.w_relaxed_head * head_cuts
            ok = np.ones_like(h, dtype=bool)
        else:
            ok = (bubble_cuts == 0) & (head_cuts == 0) & (cover >= p.min_cover - 1e-9)
        terms = {
            "drop": dropped, "slack": slack, "energy": energy, "off_band": off, "ratio_dev": ratio,
            "sfx_cut": sfx_cut, "person_out": person, "bubble_cuts": bubble_cuts.astype(np.float64),
            "head_cuts": head_cuts.astype(np.float64),
        }
        return score, ok, cover, terms


def _windows(height: int, lo: int, hi: int, step: int) -> tuple[np.ndarray, np.ndarray]:
    heights = np.unique(np.concatenate([np.arange(lo, hi + 1, step), [hi]]))
    y0s, y1s = [], []
    for h in heights:
        starts = np.unique(np.concatenate([np.arange(0, height - h + 1, step), [height - h]]))
        y0s.append(starts)
        y1s.append(starts + h)
    return np.concatenate(y0s).astype(np.int64), np.concatenate(y1s).astype(np.int64)


def search_crops(
    cons: Constraints, energy: np.ndarray, params: SearchParams = SearchParams(), *, top: int | None = None,
    relaxed: bool = False,
) -> list[Candidate]:
    """Meilleures fenêtres ``(y0, y1)`` du bloc, distinctes entre elles, meilleure en tête.

    Liste vide si aucune fenêtre ne respecte les contraintes dures (``relaxed=True`` les
    transforme alors en pénalités lourdes : on rend toujours quelque chose).
    """
    H, W = cons.height, cons.width
    top = params.top if top is None else top
    lo, hi = height_bounds(W, H, params)
    if H <= 0 or hi <= 0:
        return []
    scorer = _Scorer(cons, energy, params, relaxed)
    y0, y1 = _windows(H, lo, hi, max(1, params.step))
    score, ok, _, _ = scorer(y0, y1)
    idx = np.flatnonzero(ok)
    if not idx.size:
        return []
    seeds = idx[np.lexsort((y1[idx], y0[idx], -score[idx]))][: params.refine_seeds]
    d = np.arange(-(params.step - 1), params.step)
    ry0 = (y0[seeds][:, None, None] + d[None, :, None]).repeat(len(d), axis=2).ravel()
    ry1 = (y1[seeds][:, None, None] + d[None, None, :]).repeat(len(d), axis=1).ravel()
    valid = (ry0 >= 0) & (ry1 <= H) & (ry1 - ry0 >= lo) & (ry1 - ry0 <= hi)
    pairs = np.unique(np.stack([np.concatenate([y0[idx], ry0[valid]]), np.concatenate([y1[idx], ry1[valid]])], 1), axis=0)
    y0, y1 = pairs[:, 0], pairs[:, 1]
    score, ok, cover, terms = scorer(y0, y1)
    idx = np.flatnonzero(ok)
    order = idx[np.lexsort((y1[idx], y0[idx], -score[idx]))]
    gap = scaled(params.diversity, W)
    policy = cons.policy + (":relaxed" if relaxed else "")
    picked: list[Candidate] = []
    for i in order:
        a, b = int(y0[i]), int(y1[i])
        if all(abs(a - c.y0) > gap or abs(b - c.y1) > gap for c in picked):
            picked.append(Candidate(a, b, float(score[i]), policy, float(cover[i]),
                                    {k: float(v[i]) for k, v in terms.items()}))
            if len(picked) == top:
                break
    return picked


# --- Blocs trop hauts ------------------------------------------------------------------------
def split_tall(cons: Constraints, energy: np.ndarray, params: SearchParams = SearchParams()) -> list[tuple[int, int]] | None:
    """Découpe la zone à garder en crops consécutifs de hauteur permise (programmation dynamique).

    Une coupe intérieure ne doit traverser ni tête, ni bulle, ni narration, ni personne, ni
    zone à garder essentielle (priorité 1) : on ne coupe qu'entre deux sujets distincts.
    Trancher un personnage ou un paysage continu entre deux plans est pire qu'un
    panoramique. ``None`` si impossible.
    """
    if cons.keep is None:
        return None
    k0, k1 = cons.keep
    H, W = cons.height, cons.width
    hmin = math.ceil(W / params.ar_hard[1])
    hmax = math.floor(W / params.ar_hard[0])
    if k1 - k0 < 2 * hmin:
        return None
    essential = [Box(0, z.y0, W, z.y1, "keep") for z in cons.keep_zones if z.priority <= 1]
    blocked = (boundary_counts(cons.hard + cons.persons + essential, H) > 0)
    sfx = boundary_counts(cons.sfx, H)
    step = max(1, params.step)
    inner = [y for y in range(k0 + hmin, k1 - hmin + 1, step) if not blocked[y]]
    positions = [k0] + inner + [k1]
    pos = np.array(positions)
    cost_cut = np.array([params.w_energy * float(energy[min(y, H - 1)]) + params.w_sfx_cut * float(sfx[y]) for y in positions])
    cost_cut[0] = cost_cut[-1] = 0.0
    soft_lo, soft_hi = W / params.ar_soft[1], W / params.ar_soft[0]
    best = np.full(len(positions), np.inf)
    prev = np.full(len(positions), -1)
    best[0] = 0.0
    for i in range(1, len(positions)):
        length = pos[i] - pos[:i]
        allowed = (length >= hmin) & (length <= hmax) & np.isfinite(best[:i])
        if not allowed.any():
            continue
        L = length.astype(np.float64)
        seg = (params.w_off * np.maximum(0.0, np.maximum(soft_lo - L, L - soft_hi)) / W
               + params.w_ratio * np.abs(W / np.maximum(L, 1) - params.ar_target) + 0.1)
        total = np.where(allowed, best[:i] + seg, np.inf)
        j = int(np.argmin(total))
        best[i], prev[i] = total[j] + cost_cut[i], j
    if not np.isfinite(best[-1]):
        return None
    cuts, i = [], len(positions) - 1
    while i > 0:
        cuts.append((int(pos[prev[i]]), int(pos[i])))
        i = int(prev[i])
    return cuts[::-1]


# --- Plan d'un bloc --------------------------------------------------------------------------
def plan_block(
    shape: tuple[int, ...], energy: np.ndarray, boxes: Sequence[Box], spec: Any, params: SearchParams = SearchParams(),
) -> BlockPlan:
    """Candidats et plans d'un bloc selon son rôle, sa hauteur et ses conflits."""
    H, W = int(shape[0]), int(shape[1])
    if spec.role == "text_only" or (spec.role == "transition" and not spec.keep):
        return BlockPlan("none", [], [], None)
    cons = build_constraints(spec, boxes, H, W, params)
    assert cons is not None
    hmax = math.floor(W / params.ar_hard[0])
    if cons.keep is not None and cons.keep_h > hmax and H > hmax:
        segments = split_tall(cons, energy, params)
        if segments:
            cands = [Candidate(a, b, 0.0, "multi", 1.0) for a, b in segments]
            return BlockPlan("multi", cands, [(a, b, False) for a, b in segments], cons, bool(cons.conflicts))
        k0, k1 = cons.keep
        return BlockPlan("pan", [Candidate(k0, k1, 0.0, "pan", 1.0)], [(k0, k1, True)], cons, bool(cons.conflicts))
    cands = search_crops(cons, energy, params)
    relaxed = False
    if not cands:
        cands = search_crops(cons, energy, params, relaxed=True)
        relaxed = True
    if not cands:
        cands = [Candidate(0, H, 0.0, "whole", 1.0)]
    pool = cands[:1]
    policies = ["keep_text", "cut_subject"] if cons.conflicts else []
    sliced_sfx = cands[0].terms.get("sfx_cut", 0) > 0
    if sliced_sfx:
        policies.append("sfx_whole")
    for policy in policies:
        alt = build_constraints(spec, boxes, H, W, params, policy=policy)
        if alt is None:
            continue
        # Jamais de recherche dégradée pour une alternative : le juge ne doit voir que des
        # candidats qui respectent les contraintes dures.
        pool += search_crops(alt, energy, params, top=1)
    pool += cands[1:]
    chosen: list[Candidate] = []
    for c in pool:
        if all(iou1d((c.y0, c.y1), (o.y0, o.y1)) < 0.9 for o in chosen):
            chosen.append(c)
        if len(chosen) == params.top:
            break
    best = chosen[0]
    return BlockPlan("single", chosen, [(best.y0, best.y1, False)], cons, bool(cons.conflicts) or sliced_sfx, relaxed)


# --- Bilan d'une fenêtre ---------------------------------------------------------------------
def window_report(y0: int, y1: int, cons: Constraints) -> dict[str, float]:
    """Violations d'une fenêtre : dures (tête, bulle coupée) et souples (narration, décentrage...)."""
    def cut(box: Box) -> bool:
        return box.y0 < y0 < box.y1 or box.y0 < y1 < box.y1

    def mostly_in(box: Box) -> bool:
        return overlap1d(box.y0, box.y1, y0, y1) >= 0.5 * max(1, box.h)

    narration_boxes = cons.narration + [
        b for b in cons.bubbles if any(z.kind == "narration" and _inside(b, z) for z in cons.drops)
    ]
    offcenter = 0.0
    if cons.keep is not None and y1 > y0:
        k0, k1 = max(cons.keep[0], y0), min(cons.keep[1], y1)
        if k1 > k0:
            offcenter = abs((k0 + k1) / 2 - (y0 + y1) / 2) / (y1 - y0)
    return {
        "head_cuts": float(sum(cut(b) for b in cons.heads_raw)),
        "bubble_cuts": float(sum(cut(b) for b in cons.bubbles + cons.narration)),
        "narration_included": float(sum(mostly_in(b) for b in narration_boxes)),
        "sfx_cuts": float(sum(cut(b) for b in cons.sfx)),
        "subject_offcenter": round(offcenter, 4),
        "ratio": round(cons.width / max(1, y1 - y0), 4),
    }


__all__ = [
    "POLICIES", "SearchParams", "Zone", "Constraints", "Candidate", "BlockPlan", "build_constraints", "boundary_counts",
    "height_bounds", "search_crops", "split_tall", "plan_block", "window_report",
]
