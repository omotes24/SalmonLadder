"""Phase 3 / M2b: calibration images from the OpenOOD ImageNet-1K ID-val list (val_imagenet.txt), 4 per class,
seeded choice (numpy rng [20260927, 71, class]); the 12 support shots stay the TINS train shots (16 per class in all).
Features (DINOv2 B/14, L/14) and CLIP candidates (TINS positive text features, top 5/10/20) are taken from the
already extracted ImageNet-1K val features (extra_feats/in_val.*). The ID test set of the OpenOOD streams
(test_imagenet) is disjoint from val_imagenet. Requires --unseal <sha256 of the frozen pre-registration>.
Output: <R5>/phase3/m2b_calibration.pt
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from vins import config as C  # noqa: E402
from vins import r5  # noqa: E402

P3 = r5.R5 / "phase3"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--unseal", required=True)
    opts = parser.parse_args()
    digest = hashlib.sha256((P3 / "prereg_phase3.json").read_bytes()).hexdigest()
    if opts.unseal != digest:
        raise SystemExit("the unseal token does not match the frozen Phase 3 pre-registration")
    lines = [ln.split() for ln in (C.OPENOOD_IMGLIST_DIR / "val_imagenet.txt").read_text().splitlines() if ln.strip()]
    by_cls = {}
    for rel, lab in lines:
        by_cls.setdefault(int(lab), []).append(rel)
    blobs = {k: torch.load(C.WORK / "extra_feats" / f"in_val.{k}.pt", map_location="cpu") for k in ("dino", "dinol14", "clipb16")}
    paths = [str(p) for p in blobs["dino"]["paths"]]
    assert paths == [str(p) for p in blobs["dinol14"]["paths"]] == [str(p) for p in blobs["clipb16"]["paths"]]
    index = {}
    for i, p in enumerate(paths):
        index[str(Path(p).relative_to(C.OPENOOD_IMAGES))] = i
    rows, chosen = [], []
    for c in range(1000):
        rels = sorted(by_cls[c])
        pick = np.random.default_rng([20260927, 71, c]).permutation(len(rels))[:4]
        for j in pick:
            rows.append(index[rels[j]])
            chosen.append(rels[j])
    rows = np.array(rows)
    pos = torch.load(C.WORK / "analysis" / "baselines" / "pos_tins.pt")["pos"].float()
    clip = blobs["clipb16"]["features"][rows].float()
    sims = clip @ pos.T
    out = {"ids": chosen, "B14": blobs["dino"]["features"][rows], "L14": blobs["dinol14"]["features"][rows],
           "cand": {str(k): sims.topk(k, dim=1).indices.numpy() for k in (5, 10, 20)}, "prereg_sha256": digest}
    torch.save(out, P3 / "m2b_calibration.pt")
    print(json.dumps({"n": len(chosen), "per_class": 4}))


if __name__ == "__main__":
    main()
