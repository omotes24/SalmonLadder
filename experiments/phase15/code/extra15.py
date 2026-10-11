"""Phase 15 (a): image tables and frozen features for the baselines with many labels.
  in200_train : all training images of the 200 ImageNet-200 classes (OpenOOD list train_imagenet200.txt, 258,951), DINOv2 L/14 +
                DINOv3 L/16 (the paper's two views, the Phase 7 loader as in extract14.py).
  cifar_train : all 50,000 CIFAR-100 training images, same two views; the 9,600 images already embedded in Phase 14 are copied.
  cifar_clip  : CLIP ViT-B/16 image embeddings (the TINS path, the features MCM / NegLabel / TINS receive) of the CIFAR-100 test
                and OOD images of the Phase 14 table (160,971); the sealed Textures images are copied from the ImageNet-1K space.
Usage: --table NAME | --extract NAME --models L14,D3L --shard i --nshards n | --merge NAME --models L14,D3L
Output: P15/tables/<name>.parquet, P15/features/<name>/<model>.npy (L2-normalised float32)."""
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

import p15common as P

sys.path.insert(0, str(P.P7 / "code"))
import extract7 as X7  # noqa: E402
from vins.sealing import Guard, load_sealed  # noqa: E402
from extract14 import CIFAR, read_list  # noqa: E402

TABLES = P.P15 / "tables"
FEATS = P.P15 / "features"
LARGE = Path("/home/omote/openood_simple_sota/data/images_largescale")
IN200 = P.IMGLIST / "imagenet200"


def build_table(name):
    TABLES.mkdir(parents=True, exist_ok=True)
    if name == "in200_train":
        rows = []
        for line in (IN200 / "train_imagenet200.txt").read_text().splitlines():
            if line.strip():
                p, y = line.split()
                rows.append({"path": str(LARGE / p), "label": int(y), "wnid": Path(p).parts[-2], "p14_row": -1})
        t = pd.DataFrame(rows)
        assert t.label.min() == 0 and t.label.max() == 199 and t.path.is_unique
        assert os.path.exists(t.path.iloc[0]), t.path.iloc[0]
    elif name == "cifar_train":
        t14 = pd.read_parquet(CIFAR / "table.parquet")
        have = {p: i for i, p in zip(t14.index.values, t14.path.values) if True}
        tr = {p: i for p, i in have.items() if t14.split.iloc[i] == "train"}
        rows = []
        for p, y in read_list("train_cifar100.txt"):
            rows.append({"path": p, "label": y, "wnid": "", "p14_row": int(tr.get(p, -1))})
        t = pd.DataFrame(rows)
        assert len(t) == 50000 and (t.p14_row >= 0).sum() == 9600, int((t.p14_row >= 0).sum())
        assert os.path.exists(t.path.iloc[0]), t.path.iloc[0]
    elif name == "cifar_clip":
        t14 = pd.read_parquet(CIFAR / "table.parquet")
        t = t14[t14.split != "train"][["path", "label", "dataset", "split"]].copy()
        t["p14_row"] = t.index.values
        t = t.reset_index(drop=True)
        assert len(t) == 160971, len(t)
    else:
        raise ValueError(name)
    t.to_parquet(TABLES / f"{name}.parquet")
    print(json.dumps({"table": name, "rows": int(len(t)), "utc": P.utc()}), flush=True)


def run_shard(name, models, shard, nshards, batch=128, workers=4):
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    t = pd.read_parquet(TABLES / f"{name}.parquet")
    out_dir = FEATS / name
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{'-'.join(models)}.shard{shard}of{nshards}.npz"
    if out.exists():
        print(json.dumps({"exists": str(out)}))
        return
    todo = np.flatnonzero(t.p14_row.values < 0) if name == "cifar_train" else np.arange(len(t))     # embedded rows are copied at merge
    rows_all = todo[shard::nshards]
    sealed = load_sealed()[0]
    is_sealed = np.array([os.path.realpath(p) in sealed for p in t.path.values[rows_all]])
    rows, sealed_rows = rows_all[~is_sealed], rows_all[is_sealed]
    print(json.dumps({"table": name, "shard": shard, "rows": int(len(rows_all)), "sealed_rows": int(len(sealed_rows))}), flush=True)
    dev = "cuda"
    enc, tkey, tfs = {}, {}, {}
    for n in models:
        enc[n], tkey[n], tf, _ = X7.load(n, dev, out_dir)
        tfs.setdefault(tkey[n], tf)
    keys = list(tfs)
    ds = X7.Images(t.path.values[rows], [tfs[k] for k in keys], Guard(sealed))
    dl = DataLoader(ds, batch_size=batch, shuffle=False, num_workers=workers, pin_memory=True)
    buf, order, t0 = {n: [] for n in models}, [], time.time()
    with torch.no_grad():
        for it, b in enumerate(dl):
            x = {k: b[j].to(dev, non_blocking=True) for j, k in enumerate(keys)}
            for n in models:
                f = X7.pooled(enc[n](x[tkey[n]])).float()
                buf[n].append((f / f.norm(dim=-1, keepdim=True)).cpu().numpy().astype(np.float32))
            order.append(b[-1].numpy())
            if it % 50 == 0:
                done = sum(len(o) for o in order)
                print(json.dumps({"table": name, "shard": shard, "done": done, "total": len(rows), "img_per_s": round(done / max(time.time() - t0, 1e-9), 1), "utc": P.utc()}), flush=True)
    order = np.concatenate(order)
    assert np.array_equal(order, np.arange(len(rows)))
    arrays = {n: np.concatenate(v) for n, v in buf.items()}
    assert all(np.isfinite(f).all() for f in arrays.values())
    tmp = str(out) + f".{os.getpid()}.tmp.npz"
    np.savez(tmp, rows=rows, sealed_rows=sealed_rows, **arrays)
    os.replace(tmp, out)
    print(json.dumps({"table": name, "shard": shard, "complete": str(out), "n": len(rows), "sealed": int(len(sealed_rows)), "seconds": round(time.time() - t0, 1), "dims": {n: int(f.shape[1]) for n, f in arrays.items()}}), flush=True)


def merge(name, models):
    t = pd.read_parquet(TABLES / f"{name}.parquet")
    out_dir = FEATS / name
    tag = "-".join(models)
    files = sorted(f for f in out_dir.glob(f"{tag}.shard*of*.npz") if ".tmp" not in f.name)
    parts = [np.load(f) for f in files]
    sealed_rows = np.concatenate([z["sealed_rows"] for z in parts]) if parts and all("sealed_rows" in z.files for z in parts) else np.zeros(0, int)
    ref = {}
    if len(sealed_rows):
        D = P.load_p3("openood")                         # the sealed images are the OpenOOD Textures set of the ImageNet-1K benchmark
        P.add_views(D, "openood", [m for m in models if m in ("D3B", "D3L")])
        paths = P.image_paths("openood")[16000:]
        pos = {os.path.realpath(p): i for i, p in enumerate(paths)}
        for r in sealed_rows:
            key = os.path.realpath(t.path.values[r])
            assert key in pos, key
            ref[int(r)] = pos[key]
    copied = {}
    if name == "cifar_train":
        F14 = {m: np.load(CIFAR / "features" / f"{m}.npy", mmap_mode="r") for m in models}
        copied = {int(i): int(r) for i, r in zip(np.flatnonzero(t.p14_row.values >= 0), t.p14_row.values[t.p14_row.values >= 0])}
    for n in models:
        dim = parts[0][n].shape[1] if parts else F14[n].shape[1]
        full = np.zeros((len(t), dim), np.float32)
        filled = np.zeros(len(t), int)
        for z in parts:
            full[z["rows"]] = z[n]
            filled[z["rows"]] += 1
        for r, i in ref.items():
            full[r] = np.asarray(D["ev"][n][i], np.float32)
            filled[r] += 1
        for r, i in copied.items():
            full[r] = np.asarray(F14[n][i], np.float32)
            filled[r] += 1
        assert (filled == 1).all(), (n, int((filled == 0).sum()), int((filled > 1).sum()))
        assert np.allclose(np.linalg.norm(full, axis=1), 1.0, atol=1e-3)
        np.save(out_dir / f"{n}.npy", full)
    P.write_json(out_dir / f"info_{tag}.json", {"utc": P.utc(), "table": name, "n": len(t), "models": models, "files": [f.name for f in files],
                                                "sealed_rows_copied": int(len(sealed_rows)), "phase14_rows_copied": len(copied)})
    print(json.dumps({"merged": name, "models": models, "n": len(t), "sealed_rows_copied": int(len(sealed_rows)), "phase14_rows_copied": len(copied)}), flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--table", default=None)
    ap.add_argument("--extract", default=None)
    ap.add_argument("--merge", default=None)
    ap.add_argument("--models", default="L14,D3L")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    a = ap.parse_args()
    models = a.models.split(",")
    if a.table:
        build_table(a.table)
    if a.extract:
        run_shard(a.extract, models, a.shard, a.nshards)
    if a.merge:
        merge(a.merge, models)
