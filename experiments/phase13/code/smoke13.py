"""Phase 13 smoke test (CPU): the layout assumptions of the new scripts, on the real data, before the GPU runs."""
import json
from pathlib import Path

import numpy as np

import p13common as P

D = P.load_p3("openood")
paths = P.image_paths("openood")
# 1. shots layout: support rows [12c, 12c+12) and calibration rows [12000+4c, 12000+4c+4) share one wnid; class index = sorted wnid
from in200_13 import class_layout
wn = class_layout(paths)
print("class_layout ok; first wnids", wn[:3], "last", wn[-1])
# 2. D['row'] = 4000 + position in D['ids']
assert all(D["row"][x] == 4000 + i for i, x in enumerate(D["ids"][:1000]))
# 3. labels of the ID test images (openood part) and of the fourood validation images
labels, pp = P.id_labels("openood", D)
print("openood: labelled ID images", int((labels >= 0).sum()), "classes", len(np.unique(labels[labels >= 0])), "per class", np.bincount(labels[labels >= 0]).min(), np.bincount(labels[labels >= 0]).max())
assert int((labels >= 0).sum()) == 45000 and len(np.unique(labels[labels >= 0])) == 1000
# 3b. the labels agree with the nearest prototype (DINOv2 L/14): top-1 accuracy should be high (sanity check of the class index)
sup = P.l2n(D["shots"]["L14"][:12000]).reshape(1000, 12, -1).mean(1)
sup /= np.linalg.norm(sup, axis=1, keepdims=True)
idx = np.flatnonzero(labels >= 0)[:5000]
ev = P.l2n(D["ev"]["L14"][[D["row"][D["ids"][i]] - 4000 for i in idx]])
acc = float((np.argmax(ev @ sup.T, 1) == labels[idx]).mean())
print("prototype top-1 accuracy on 5,000 ID test images (DINOv2 L/14):", round(acc, 4))
assert acc > 0.6
D4 = P.load_p3("fourood")
labels4, _ = P.id_labels("fourood", D4)
print("fourood: labelled ID images", int((labels4 >= 0).sum()), "per class min/max", np.bincount(labels4[labels4 >= 0]).min(), np.bincount(labels4[labels4 >= 0]).max())
assert int((labels4 >= 0).sum()) == 50000
# 4. ImageNet-200: the 200 classes, the 9,000 ID test images in the feature space, the OOD lists
IN200 = P.IMGLIST / "imagenet200"
lab2w = {}
for line in (IN200 / "train_imagenet200.txt").read_text().splitlines():
    if line.strip():
        p, y = line.split()
        lab2w.setdefault(int(y), set()).add(Path(p).parts[-2])
print("in200 classes", len(lab2w), "single wnid each", all(len(v) == 1 for v in lab2w.values()))
name2id = {Path(p).name: s for s, p in zip(D["ids"], paths[16000:]) if "imagenet_1k/val/" in p}
names = [l.split()[0] for l in (IN200 / "test_imagenet200.txt").read_text().splitlines() if l.strip()]
found = sum(Path(n).name in name2id for n in names)
print("in200 ID test images", len(names), "found in feature space", found)
for ds in ("ssb_hard", "ninco", "inaturalist", "textures", "openimage_o"):
    a = {l.split()[0] for l in (IN200 / f"test_{ds}.txt").read_text().splitlines() if l.strip()}
    b = {l.split()[0] for l in (P.IMGLIST / "imagenet" / f"test_{ds}.txt").read_text().splitlines() if l.strip()}
    print("ood list", ds, len(a), len(b), "identical" if a == b else f"DIFFER (common {len(a & b)})")
print("val_imagenet.txt lines", len([l for l in (P.IMGLIST / "imagenet" / "val_imagenet.txt").read_text().splitlines() if l.strip()]))
# 5. the Phase 12 draws npz carry the per-image terms the product script needs
z = np.load(P.RESULTS12 / "draws" / "openood" / "p3" / "ninco.npz", allow_pickle=True)
print("draws npz keys", [k for k in z.files if k.startswith("s::L14")][:8])
print("SMOKE13_OK")
