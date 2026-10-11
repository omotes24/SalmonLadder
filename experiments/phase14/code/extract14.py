"""Phase 14 (f1): the OpenOOD v1.5 CIFAR-100 benchmark - image table and frozen features.
ID = CIFAR-100 (OpenOOD lists: train 50,000; test 9,000); near-OOD = CIFAR-10, Tiny ImageNet; far-OOD = MNIST, SVHN, Textures,
Places365 (the OpenOOD test lists). Labelled images: six disjoint draws of 16 training images per class (12 support + 4
calibration), drawn once at random (rng [14, 100]); only those 9,600 training images are embedded. Features: DINOv2 L/14 (CLS,
224) and DINOv3 L/16 (CLS, 256) through the Phase 7 loader (extract7.load; the 32 x 32 images are resized like any other image).
The Textures images are the sealed OpenOOD set shared with the ImageNet-1K benchmark: they are not opened; their features are
copied from the ImageNet-1K feature space at merge time (same models and preprocessing).
Usage: --table | --shard i --nshards n [--models L14,D3L] | --merge.  Output: P14/cifar/table.parquet, P14/cifar/features/<model>.npy."""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

import p14common as P

sys.path.insert(0, str(P.P7 / "code"))
import extract7 as X7  # noqa: E402
from vins.sealing import Guard, load_sealed  # noqa: E402

ROOT = Path("/home/omote/openood_simple_sota/data/images_classic")
LISTS = P.IMGLIST / "cifar100"
OOD = {"cifar10": "near", "tin": "near", "mnist": "far", "svhn": "far", "texture": "far", "places365": "far"}
CIFAR = P.P14 / "cifar"
N_DRAWS, PER_CLASS, N_SUP = 6, 16, 12


def read_list(name):
    rows = []
    for line in (LISTS / name).read_text().splitlines():
        if line.strip():
            p, y = line.split()
            rows.append((p, int(y)))
    return rows


def build_table():
    train = read_list("train_cifar100.txt")
    labels = np.array([y for _, y in train])
    assert len(train) == 50000 and labels.min() == 0 and labels.max() == 99
    rng = np.random.default_rng([14, 100])
    rows = []
    for c in range(100):
        idx = rng.permutation(np.flatnonzero(labels == c))
        for d in range(N_DRAWS):
            for j, i in enumerate(idx[PER_CLASS * d:PER_CLASS * (d + 1)]):
                rows.append({"path": train[i][0], "label": c, "split": "train", "dataset": "cifar100", "draw": d, "role": "sup" if j < N_SUP else "cal", "pos": j})
    for p, y in read_list("test_cifar100.txt"):
        rows.append({"path": p, "label": y, "split": "test", "dataset": "cifar100", "draw": -1, "role": "test", "pos": -1})
    for ds in OOD:
        for p, y in read_list(f"test_{ds}.txt"):
            rows.append({"path": p, "label": -1, "split": "ood", "dataset": ds, "draw": -1, "role": "test", "pos": -1})
    t = pd.DataFrame(rows)
    assert t.path.is_unique, "duplicate image paths across the lists"
    t.insert(0, "sample_id", np.arange(len(t)))
    first = t.path.iloc[0]
    base = next((b for b in (ROOT, ROOT.parent) if (b / first).exists()), None)
    assert base is not None, f"image root not found for {first}"
    t["path"] = [str(base / p) for p in t.path]
    sample = t.groupby("dataset").head(20).path.tolist() + t.groupby("dataset").tail(20).path.tolist()
    missing = [p for p in sample if not Path(p).exists()]
    assert not missing, missing[:3]
    CIFAR.mkdir(parents=True, exist_ok=True)
    t.to_parquet(CIFAR / "table.parquet", index=False)
    counts = t.groupby(["split", "dataset"]).size().to_dict()
    print(json.dumps({"table": str(CIFAR / "table.parquet"), "n": len(t), "counts": {f"{a}/{b}": int(n) for (a, b), n in counts.items()}}), flush=True)


@torch.no_grad()
def run_shard(models, shard, nshards, batch=128, workers=4):
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    t = pd.read_parquet(CIFAR / "table.parquet")
    out_dir = CIFAR / "features"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{'-'.join(models)}.shard{shard}of{nshards}.npz"
    if out.exists():
        print(json.dumps({"exists": str(out)}))
        return
    rows_all = np.arange(len(t))[shard::nshards]
    sealed = load_sealed()[0]
    is_sealed = np.array([os.path.realpath(p) in sealed for p in t.path.values[rows_all]])
    rows, sealed_rows = rows_all[~is_sealed], rows_all[is_sealed]            # sealed images (the OpenOOD Textures set shared with the
    print(json.dumps({"shard": shard, "rows": int(len(rows_all)), "sealed_rows": int(len(sealed_rows))}), flush=True)   # ImageNet benchmark) are never opened:
    dev = "cuda"                                                                                                           # their features are copied at merge time
    enc, tkey, tfs = {}, {}, {}
    for n in models:
        enc[n], tkey[n], tf, _ = X7.load(n, dev, out_dir)
        tfs.setdefault(tkey[n], tf)
    keys = list(tfs)
    ds = X7.Images(t.path.values[rows], [tfs[k] for k in keys], Guard(sealed))
    dl = DataLoader(ds, batch_size=batch, shuffle=False, num_workers=workers, pin_memory=True)
    buf, order, t0 = {n: [] for n in models}, [], time.time()
    for it, b in enumerate(dl):
        x = {k: b[j].to(dev, non_blocking=True) for j, k in enumerate(keys)}
        for n in models:
            f = X7.pooled(enc[n](x[tkey[n]])).float()
            buf[n].append((f / f.norm(dim=-1, keepdim=True)).cpu().numpy().astype(np.float32))
        order.append(b[-1].numpy())
        if it % 50 == 0:
            done = sum(len(o) for o in order)
            print(json.dumps({"shard": shard, "done": done, "total": len(rows), "img_per_s": round(done / max(time.time() - t0, 1e-9), 1), "utc": P.utc()}), flush=True)
    order = np.concatenate(order)
    assert np.array_equal(order, np.arange(len(rows)))
    arrays = {n: np.concatenate(v) for n, v in buf.items()}
    assert all(np.isfinite(f).all() for f in arrays.values())
    tmp = str(out) + f".{os.getpid()}.tmp.npz"
    np.savez(tmp, rows=rows, sealed_rows=sealed_rows, **arrays)
    os.replace(tmp, out)
    print(json.dumps({"shard": shard, "complete": str(out), "n": len(rows), "sealed": int(len(sealed_rows)), "seconds": round(time.time() - t0, 1), "dims": {n: int(f.shape[1]) for n, f in arrays.items()}}), flush=True)


def merge(models):
    t = pd.read_parquet(CIFAR / "table.parquet")
    out_dir = CIFAR / "features"
    tag = "-".join(models)
    files = sorted(f for f in out_dir.glob(f"{tag}.shard*of*.npz") if ".tmp" not in f.name)
    parts = [np.load(f) for f in files]
    sealed_rows = np.concatenate([z["sealed_rows"] for z in parts]) if all("sealed_rows" in z.files for z in parts) else np.zeros(0, int)
    ref = {}
    if len(sealed_rows):
        # the sealed images are the OpenOOD Textures set, already embedded (same models, same preprocessing) in the ImageNet-1K
        # feature space of Phases 3/10: copy those rows instead of opening the images
        D = P.load_p3("openood")
        P.add_views(D, "openood", [m for m in models if m in ("D3B", "D3L")])
        paths = P.image_paths("openood")[16000:]
        pos = {os.path.realpath(p): i for i, p in enumerate(paths)}
        for r in sealed_rows:
            key = os.path.realpath(t.path.values[r])
            assert key in pos, key
            ref[int(r)] = pos[key]
    for n in models:
        full = np.zeros((len(t), parts[0][n].shape[1]), np.float32)
        filled = np.zeros(len(t), int)
        for z in parts:
            full[z["rows"]] = z[n]
            filled[z["rows"]] += 1
        for r, i in ref.items():
            full[r] = np.asarray(D["ev"][n][i], np.float32)
            filled[r] += 1
        assert (filled == 1).all(), (n, int((filled == 0).sum()), int((filled > 1).sum()))
        assert np.allclose(np.linalg.norm(full, axis=1), 1.0, atol=1e-3)
        np.save(out_dir / f"{n}.npy", full)
    P.write_json(out_dir / "info.json", {"utc": P.utc(), "n": len(t), "models": models, "files": [f.name for f in files], "sealed_rows_copied": int(len(sealed_rows))})
    print(json.dumps({"merged": models, "n": len(t), "sealed_rows_copied": int(len(sealed_rows))}), flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--table", action="store_true")
    ap.add_argument("--merge", action="store_true")
    ap.add_argument("--models", default="L14,D3L")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    models = a.models.split(",")
    if a.table:
        build_table()
    elif a.merge:
        merge(models)
    else:
        run_shard(models, a.shard, a.nshards, a.batch, a.workers)
