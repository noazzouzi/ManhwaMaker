"""toonsplit prototype — AI decides WHAT matters, code decides WHERE to cut.

Stage 1  segment_blocks   : split strip at background gutters (code)
Stage 2  detect_*         : faces + text/bubbles, pixel-exact (code)
Stage 3  AI block spec    : role + coarse keep/drop zones (VLM, JSON)
Stage 4  search_crops     : full-width windows, hard no-cut constraints (code)
Stage 5  AI judge         : pick best of top-3 (VLM) -- here: top-1 by score
"""
import json, sys
import cv2
import numpy as np

HERE = __import__("os").path.dirname(__import__("os").path.abspath(__file__))  # (Windows : os.path au lieu de rsplit("/"))
# face model: curl -O https://raw.githubusercontent.com/nagadomi/lbpcascade_animeface/master/lbpcascade_animeface.xml
FACE = cv2.CascadeClassifier(f"{HERE}/lbpcascade_animeface.xml")

# ---------------- Stage 1: blocks ----------------
def segment_blocks(img, min_gap=40, min_block=120):
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    std, mean = g.std(1), g.mean(1)
    bg = (std < 5) & ((mean > 235) | (mean < 20))          # flat white/black row
    blocks, y, H = [], 0, len(bg)
    while y < H:
        if bg[y]:
            y += 1; continue
        s = y
        while y < H:
            if bg[y]:
                e = y
                while e < H and bg[e]: e += 1
                if e - y >= min_gap or e == H: break
                y = e
            else:
                y += 1
        if y - s >= min_block: blocks.append((s, y))
    return blocks

# ---------------- Stage 2: detectors ----------------
def detect_faces(block):
    g = cv2.equalizeHist(cv2.cvtColor(block, cv2.COLOR_BGR2GRAY))
    f = FACE.detectMultiScale(g, scaleFactor=1.05, minNeighbors=4, minSize=(24, 24))
    return [(int(x), int(y), int(x + w), int(y + h)) for x, y, w, h in f]

def detect_text_boxes(block):
    """Dark glyph lines on locally white background -> grown to their bubble."""
    g = cv2.cvtColor(block, cv2.COLOR_BGR2GRAY)
    H, W = g.shape
    dark = (g < 110).astype(np.uint8)
    lines = cv2.morphologyEx(dark, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (15, 3)))
    n, lab, st, _ = cv2.connectedComponentsWithStats(lines)
    cand = []
    for i in range(1, n):
        x, y, w, h, a = st[i]
        if not (7 <= h <= 40 and w >= 12 and w < 0.8 * W): continue
        ring = g[max(0, y - 6):y + h + 6, max(0, x - 6):x + w + 6]
        if np.percentile(ring, 60) < 215: continue          # background must be white-ish
        if a / (w * h) < 0.25 or w < 1.8 * h: continue
        # a text line = several glyphs of similar height, sitting on a flat baseline
        sub = dark[y:y + h, x:x + w]
        m, _, gs, _ = cv2.connectedComponentsWithStats(sub)
        gl = [q for q in gs[1:] if q[3] >= 0.5 * h and q[4] >= 6]
        if len(gl) < 3: continue
        hs = np.array([q[3] for q in gl]); bots = np.array([q[1] + q[3] for q in gl])
        if hs.std() / hs.mean() > 0.35 or bots.std() > 0.25 * h: continue
        cand.append([x, y, x + w, y + h])
    # merge lines into text blocks
    cand.sort(key=lambda b: b[1]); blocks = []
    for b in cand:
        for t in blocks:
            if b[1] - t[3] < 22 and b[0] < t[2] + 40 and b[2] > t[0] - 40:
                t[:] = [min(t[0], b[0]), min(t[1], b[1]), max(t[2], b[2]), max(t[3], b[3])]; break
        else:
            blocks.append(list(b))
    # grow each text block to its bubble via flood fill on the white mask
    white = ((g > 200) * 255).astype(np.uint8)
    out = []
    for x0, y0, x1, y1 in blocks:
        m = np.zeros((H + 2, W + 2), np.uint8)
        seed = (max(0, x0 - 3), (y0 + y1) // 2)
        if white[seed[1], seed[0]]:
            cv2.floodFill(white.copy(), m, seed, 128, flags=4 | (255 << 8) | cv2.FLOODFILL_MASK_ONLY)
            ys, xs = np.where(m[1:-1, 1:-1])
            bx = (xs.min(), ys.min(), xs.max(), ys.max()) if len(xs) else None
        else:
            bx = None
        leaked = bx is None or (bx[2] - bx[0]) > 0.9 * W or (bx[3] - bx[1]) > 6 * (y1 - y0 + 40)
        if leaked:  # floating text: pad generously (spiky outlines)
            bx = (x0 - 30, y0 - 45, x1 + 30, y1 + 45)
        else:       # include the bubble outline
            bx = (bx[0] - 12, bx[1] - 12, bx[2] + 12, bx[3] + 12)
        out.append(tuple(int(v) for v in (max(0, bx[0]), max(0, bx[1]), min(W, bx[2]), min(H, bx[3]))))
    return out

def cut_energy(block):
    """Row-wise edge energy: cutting through calm rows looks deliberate."""
    g = cv2.cvtColor(block, cv2.COLOR_BGR2GRAY).astype(np.float32)
    e = np.abs(cv2.Sobel(g, cv2.CV_32F, 0, 1)).mean(1)
    return e / (e.max() + 1e-6)

# ---------------- Stage 4: crop search ----------------
def cuts(box, y0, y1):
    return box[1] < y0 < box[3] or box[1] < y1 < box[3]

def search_crops(block, spec, faces, texts, ar=(0.62, 1.05), hard_ar=(0.55, 1.35), step=4, top=3):
    H, W = block.shape[:2]
    hmin, hmax = int(W / hard_ar[1]), int(W / hard_ar[0])
    soft_lo, soft_hi = int(W / ar[1]), int(W / ar[0])
    keep = [(int(k["y"][0] * H), int(k["y"][1] * H)) for k in spec.get("keep", [])]
    drop = [(int(d["y"][0] * H), int(d["y"][1] * H)) for d in spec.get("drop", [])]
    if not keep: return []
    # snap: any detected face/bubble touching a keep zone becomes part of it (unless in a drop zone)
    k0, k1 = min(a for a, _ in keep), max(b for _, b in keep)
    for b in faces + texts:
        in_drop = any(b[1] >= d0 - 20 and b[3] <= d1 + 20 for d0, d1 in drop)
        if b[3] > k0 and b[1] < k1 and not in_drop:
            k0, k1 = min(k0, b[1]), max(k1, b[3])
    energy = cut_energy(block)
    hard = faces + texts
    res = []
    for h in range(min(hmin, H), min(hmax, H) + 1, step):
        for y0 in range(0, H - h + 1, step):
            y1 = y0 + h
            if any(cuts(b, y0, y1) for b in hard): continue
            cover = max(0, min(y1, k1) - max(y0, k0)) / (k1 - k0)
            if cover < 0.98: continue
            dropped = sum(max(0, min(y1, d1) - max(y0, d0)) for d0, d1 in drop) / H
            slack = (h - (k1 - k0)) / H
            e = energy[min(y0, H - 1)] + energy[min(y1, H - 1)]
            off = max(0, soft_lo - h, h - soft_hi) / W        # soft aspect band
            score = cover - 3.0 * dropped - 0.35 * slack - 0.15 * e - 1.5 * off
            res.append((score, y0, y1))
    res.sort(reverse=True)
    picked = []
    for r in res:                                # diverse top-k
        if all(abs(r[1] - p[1]) > 40 or abs(r[2] - p[2]) > 40 for p in picked):
            picked.append(r)
        if len(picked) == top: break
    return picked

def iou1d(a, b):
    i = max(0, min(a[1], b[1]) - max(a[0], b[0]))
    return i / (max(a[1], b[1]) - min(a[0], b[0]))
