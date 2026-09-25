"""Mesure « avant » : le prototype d'origine, tel quel, sur le strip réduit à 575 px.

Adapté de ``run.py`` (chemins locaux au lieu de ``/mnt/user-data``) ; l'algorithme et les
paramètres de ``toonsplit_proto.py`` ne sont pas modifiés. Écrit ``baseline_predictions.json``
(plans ramenés au strip natif) pour ``python -m src.modules.toonsplit eval ... --predictions``.

    .\\.venv\\Scripts\\python.exe eval/toonsplit/prototype/run_proto.py
"""
import json
import os
import sys

import cv2

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import toonsplit_proto as t  # noqa: E402

native = cv2.imread(os.path.join(HERE, "..", "001", "strip.webp"))
S = cv2.resize(native, (575, 8000), interpolation=cv2.INTER_AREA)  # le strip de test du prototype
scale = native.shape[0] / S.shape[0]
spec = json.load(open(os.path.join(HERE, "ai_spec.json")))
human = {0: (784, 1523), 2: (3589, 4466), 3: (5059, 5616), 5: (6769, 7621)}
shots = []
for (a, b), sp in zip(t.segment_blocks(S), spec):
    blk = S[a:b]
    faces, texts = t.detect_faces(blk), t.detect_text_boxes(blk)
    top = t.search_crops(blk, sp, faces, texts)
    line = f"block {sp['block']} [{sp['role']}] {a}-{b} faces={len(faces)}"
    if not top:
        print(line, "-> no image (TTS only)")
        continue
    s, y0, y1 = top[0]
    got = (a + y0, a + y1)
    hm = human.get(sp["block"])
    print(line, f"-> crop {got}", f"| human {hm} IoU={t.iou1d(got, hm):.2f}" if hm else "")
    shots.append({"source_block": sp["block"], "y0": round(got[0] * scale), "y1": round(got[1] * scale), "pan": False})
with open(os.path.join(HERE, "baseline_predictions.json"), "w") as f:
    json.dump({"001": shots}, f, indent=1)
print("ecrit", os.path.join(HERE, "baseline_predictions.json"))
