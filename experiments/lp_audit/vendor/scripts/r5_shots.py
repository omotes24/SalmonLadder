"""R5 step 1: five disjoint 16-shot draws (12 support + 4 calibration) for every ImageNet-1K class.

Pool per class: ImageNet-1K train images of the class minus
  - every image of dev1 and dev2 (stream images and the original shots; by file name and by sha256),
  - the TINS 16-shot list (all 1000 classes) and Codex's excluded duplicates,
  - sealed images (path or sha256; OpenOOD test lists and val_imagenet), and images already drawn (sha256).
Draw: seeded permutation per class (numpy rng [20260927, 5, idx_1k]); the first 80 eligible images form draws
0..4 in order (16 each); inside a draw a second seeded permutation assigns 12 support / 4 calibration.
Images drawn for two classes (byte-identical files under two class folders) are dropped from both; 12 spare picks
per class fill the gap in the same seeded order.
Outputs (global, shared by dev1 and dev2): <R5>/shots/draws.parquet, draws_info.json,
          lists/draw{k}_train16.txt (TINS --train-imglist format: "train/<wnid>/<file> <label>", 1000 classes)
No OOD label or stream order is involved.
"""
import json
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from vins import config as C  # noqa: E402
from vins.r5 import R5  # noqa: E402
from vins.sealing import load_sealed  # noqa: E402
from vins.splits import list_class_files, load_imagenet_classes, sha256_file  # noqa: E402

N_DRAWS, PER_DRAW = 5, 16
N_SPARE = 12                     # extra picks per class, used when a pick is also drawn for another class
BASE = Path.home() / "vins_gonogo_20260925"
G = {}


def pick_class(task):
    idx, wnid = task
    excluded_names, excluded_hashes = G["names"].get(wnid, set()), G["hashes"]
    cand = [f for f in list_class_files(wnid) if f not in excluded_names]
    rng = np.random.default_rng([20260927, 5, idx])
    picked, rejected = [], {"sealed_path": 0, "sealed_hash": 0, "excluded_hash": 0, "duplicate": 0}
    seen = set()
    for j in rng.permutation(len(cand)):
        name = cand[j]
        real = os.path.realpath(C.IMAGENET_ROOT / "train" / wnid / name)
        if real in G["sealed_paths"]:
            rejected["sealed_path"] += 1
            continue
        digest = sha256_file(real)
        if digest in G["sealed_hashes"]:
            rejected["sealed_hash"] += 1
            continue
        if digest in excluded_hashes:
            rejected["excluded_hash"] += 1
            continue
        if digest in seen:
            rejected["duplicate"] += 1
            continue
        seen.add(digest)
        picked.append((name, real, digest))
        if len(picked) == N_DRAWS * PER_DRAW + N_SPARE:
            break
    if len(picked) < N_DRAWS * PER_DRAW:
        raise RuntimeError(f"{wnid}: only {len(picked)} eligible images")
    return idx, picked, rejected, len(cand)


def assign(idx, wnid, picked):
    rows = []
    for k in range(N_DRAWS):
        chunk = picked[k * PER_DRAW:(k + 1) * PER_DRAW]
        perm = np.random.default_rng([20260927, 6, idx, k]).permutation(PER_DRAW)
        for pos, j in enumerate(perm):
            name, real, digest = chunk[j]
            rows.append({"draw": k, "idx_1k": idx, "wnid": wnid, "pos": pos,
                         "role": "support" if pos < C.N_SUPPORT else "calib",
                         "sample_id": f"in1k/train/{wnid}/{name}", "rel_path": f"train/{wnid}/{name}",
                         "path": real, "sha256": digest})
    return rows


def main():
    start = time.time()
    out = R5 / "shots"
    (out / "lists").mkdir(parents=True, exist_ok=True)
    if (out / "draws.parquet").exists():
        raise SystemExit("draws already built; refusing to overwrite")
    wnids, _, _ = load_imagenet_classes()
    names, hashes = {}, set()
    for work in (BASE, BASE / "dev2"):
        s = pd.read_parquet(work / "splits" / "samples.parquet")
        hashes |= set(s.sha256)
        for w, r in zip(s.wnid, s.rel_path):
            if w:
                names.setdefault(w, set()).add(Path(r).name)
    proto = BASE / "inputs" / "prototype_train16.txt"
    for line in proto.read_text().splitlines():
        if line.strip():
            rel = line.split()[0]
            names.setdefault(Path(rel).parts[1], set()).add(Path(rel).name)
    codex_excluded = json.loads(C.PROTO_RESOLUTION_SRC.read_text())["excluded"]
    for e in codex_excluded:
        pth = Path(e["path"])
        names.setdefault(pth.parent.name, set()).add(pth.name)
    sealed_paths, sealed_hashes, _ = load_sealed()
    G.update(names=names, hashes=hashes, sealed_paths=sealed_paths, sealed_hashes=sealed_hashes)
    with mp.get_context("fork").Pool(20) as pool:
        res = pool.map(pick_class, list(enumerate(wnids)), chunksize=4)
    # ImageNet files a few byte-identical images under two classes: such images are dropped from every class
    from collections import Counter

    count = Counter(dg for _, picked, _, _ in res for _, _, dg in picked)
    cross = {dg for dg, n in count.items() if n > 1}
    rows, rejections, pool_sizes = [], {}, {}
    for idx, picked, rej, n in sorted(res, key=lambda x: x[0]):
        keep = [x for x in picked if x[2] not in cross]
        if len(keep) < N_DRAWS * PER_DRAW:
            raise RuntimeError(f"{wnids[idx]}: {len(keep)} images after removing cross-class duplicates")
        rej["cross_class_duplicate"] = len(picked) - len(keep)
        rows += assign(idx, wnids[idx], keep[:N_DRAWS * PER_DRAW])
        pool_sizes[wnids[idx]] = n
        if any(rej.values()):
            rejections[wnids[idx]] = rej
    frame = pd.DataFrame(rows)
    assert frame.sha256.is_unique, "an image was drawn twice"
    assert not (set(frame.sha256) & hashes), "drawn image overlaps dev1/dev2"
    assert not (set(frame.sha256) & sealed_hashes)
    frame.to_parquet(out / "draws.parquet", index=False)
    for k in range(N_DRAWS):
        sub = frame[frame.draw == k].sort_values(["idx_1k", "rel_path"])
        with open(out / "lists" / f"draw{k}_train16.txt", "w") as handle:
            for rel, lab in zip(sub.rel_path, sub.idx_1k):
                handle.write(f"{rel} {lab}\n")
    info = {"utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "n_draws": N_DRAWS, "per_draw": PER_DRAW,
            "n_images": len(frame), "rejections": rejections,
            "n_rejected": {k: sum(r.get(k, 0) for r in rejections.values()) for k in
                           ("sealed_path", "sealed_hash", "excluded_hash", "duplicate", "cross_class_duplicate")},
            "n_cross_class_duplicates": len(cross),
            "min_pool": int(min(pool_sizes.values())), "seconds": round(time.time() - start, 1),
            "rule": __doc__}
    (out / "draws_info.json").write_text(json.dumps(info, indent=1) + "\n")
    print(json.dumps({k: info[k] for k in ("n_images", "n_rejected", "min_pool", "seconds")}))


if __name__ == "__main__":
    main()
