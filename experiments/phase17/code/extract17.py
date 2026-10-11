"""Phase 17 (preparatory, no scoring): image tables and frozen features of the covariate-shifted ID images of the OpenOOD v1.5
full-spectrum benchmark that are on the server: ImageNet-V2 (matched frequency, 10,000; test_imagenet_v2.txt) and ImageNet-R
(30,000; test_imagenet_r.txt), with the paper's two views (DINOv2 L/14, DINOv3 L/16) and the Phase 7 loader (the preprocessing of
the benchmark features). ImageNet-C is not on the server and is not used. None of these images is in the sealed manifest
(checked per path with the guard, as in Phases 14-15).
Usage: --table NAME | --extract NAME --models L14,D3L --shard i --nshards n | --merge NAME --models L14,D3L
Output: P17/tables/<name>.parquet, P17/features/<name>/<model>.npy (L2-normalised float32)."""
import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

import p17common as P

sys.path.insert(0, str(P.P7 / "code"))
import extract7 as X7  # noqa: E402
from vins.sealing import Guard, load_sealed  # noqa: E402


def build_table(name):
    P.TABLES17.mkdir(parents=True, exist_ok=True)
    root = P.CSID_ROOTS[name]
    rows = []
    for line in (P.IMGLIST / "imagenet" / P.CSID_LISTS[name]).read_text().splitlines():
        if line.strip():
            p, y = line.split()
            top, rest = p.split("/", 1)
            assert top == name, p
            rows.append({"path": str(root / rest), "label": int(y), "list_path": p})
    t = pd.DataFrame(rows)
    assert t.path.is_unique and all(os.path.exists(x) for x in t.path.values)
    sealed = load_sealed()[0]
    n_sealed = int(sum(os.path.realpath(x) in sealed for x in t.path.values))
    assert n_sealed == 0, n_sealed
    t.to_parquet(P.TABLES17 / f"{name}.parquet")
    print(json.dumps({"table": name, "rows": int(len(t)), "labels": int(t.label.nunique()), "sealed": n_sealed, "utc": P.utc()}), flush=True)


def run_shard(name, models, shard, nshards, batch=128, workers=4):
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    t = pd.read_parquet(P.TABLES17 / f"{name}.parquet")
    out_dir = P.FEATS17 / name
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{'-'.join(models)}.shard{shard}of{nshards}.npz"
    if out.exists():
        print(json.dumps({"exists": str(out)}))
        return
    rows = np.arange(len(t))[shard::nshards]
    sealed = load_sealed()[0]
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
    np.savez(tmp, rows=rows, **arrays)
    os.replace(tmp, out)
    print(json.dumps({"table": name, "shard": shard, "complete": str(out), "n": len(rows), "seconds": round(time.time() - t0, 1), "dims": {n: int(f.shape[1]) for n, f in arrays.items()}}), flush=True)


def merge(name, models):
    t = pd.read_parquet(P.TABLES17 / f"{name}.parquet")
    out_dir = P.FEATS17 / name
    tag = "-".join(models)
    files = sorted(f for f in out_dir.glob(f"{tag}.shard*of*.npz") if ".tmp" not in f.name)
    parts = [np.load(f) for f in files]
    for n in models:
        dim = parts[0][n].shape[1]
        full = np.zeros((len(t), dim), np.float32)
        filled = np.zeros(len(t), int)
        for z in parts:
            full[z["rows"]] = z[n]
            filled[z["rows"]] += 1
        assert (filled == 1).all(), (n, int((filled == 0).sum()), int((filled > 1).sum()))
        assert np.allclose(np.linalg.norm(full, axis=1), 1.0, atol=1e-3)
        np.save(out_dir / f"{n}.npy", full)
    P.write_json(out_dir / f"info_{tag}.json", {"utc": P.utc(), "table": name, "n": len(t), "models": models, "files": [f.name for f in files]})
    print(json.dumps({"merged": name, "models": models, "n": len(t)}), flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--table")
    ap.add_argument("--extract")
    ap.add_argument("--merge")
    ap.add_argument("--models", default="L14,D3L")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    a = ap.parse_args()
    ms = a.models.split(",")
    if a.table:
        build_table(a.table)
    if a.extract:
        run_shard(a.extract, ms, a.shard, a.nshards)
    if a.merge:
        merge(a.merge, ms)
