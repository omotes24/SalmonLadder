"""R5 (3): audit of the conformal p-values (dev1 audit streams, 50 calibration resamples; no test data).

Audit streams (seeded, 6,000 images = 24 batches of 256):
  train : ID = 5 images per dev1 ID class from id_dev (ImageNet train, same distribution as the calibration shots),
          OOD = 15 images per held-out class from near_dev
  v2    : ID = 5 images per dev1 ID class from ImageNet-V2 matched-frequency (a val-like distribution), same OOD
Support: draw 0 (12 per class, fixed). Calibration pool per class: the other 68 drawn images of the class.
Resample r = 0..49: 8 pool images per class (seeded); the first 4 are the calibration shots, the next 4 a held-out set.
Per resample and view (B/14, L/14), the rate of p <= alpha over ID arrivals (alpha in 0.01, 0.05, 0.10, 0.20) for
  static p, p_all, p_A1, p_A2, p_t (memory) under three uses of the calibration shots:
     shared4 : the 4 shots at every stage (v4)
     split1  : shot 0 for p_all, shot 1 for p_A1, shot 2 for p_A2, shot 3 for p_t (stages use disjoint shots)
     reuse1  : shot 0 at every stage (same count as split1, reused)
  held-out audit: memory p of the held-out images (never streamed) at fixed histories (batches 6, 12, 18, 23), shared4
  LP p: warm (v4), cold (all nodes from 0, 15 sweeps per batch), converge (from 0 to relative residual 1e-6, at
        most 2,000 sweeps), aligned (calibration shots join the graph together with the batch; diagnostic)
Also by stream half and per ID class. Output: <R5>/audit/<stream>_r<r>.npz; --phase summary -> audit/summary.json
"""
import argparse
import json
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

from vins import config as C  # noqa: E402
from vins import r5  # noqa: E402
from vins.features import load_features  # noqa: E402

OUT = r5.R5 / "audit"
ALPHAS = (0.01, 0.05, 0.10, 0.20)
SNAPS = (6, 12, 18, 23)
N_RES = 50
G = {}


def load_ev():
    import hashlib
    import importlib.util

    code = ROOT / "scripts" / "iter3_eval.py"
    assert hashlib.sha256(code.read_bytes()).hexdigest() == "40a9bf336c7ea02cb062b2b78fc48492445add34168c309857f12df670c161ad"
    spec = importlib.util.spec_from_file_location("iter3_eval", code)
    ev = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ev)
    return ev


def zeroshot(clip_feats, pos_text):
    return (clip_feats.float() @ pos_text.T).topk(C.K_TOP, dim=1).indices.numpy()


def build_streams():
    """IDs, features and candidate sets of the pool and the two audit streams (dev1)."""
    samples = pd.read_parquet(C.SPLITS_DIR / "samples.parquet")
    dview = pd.read_parquet(C.RUNS_DIR / "dview" / "dview.parquet").set_index("sample_id")
    id_classes = json.loads((C.SPLITS_DIR / "id_classes.json").read_text())
    id_of = {c["idx_1k"]: c["id_idx"] for c in id_classes}
    n_id = len(id_classes)
    pos_text = torch.load(C.RUNS_DIR / "setup" / "tins_setup.pt", map_location="cpu")["positive_features"].float()
    draws = pd.read_parquet(r5.R5 / "shots" / "draws.parquet")
    draws = draws[draws.idx_1k.isin(id_of)].assign(id_idx=lambda x: x.idx_1k.map(id_of))
    sup = draws[(draws.draw == 0) & (draws.role == "support")].sort_values(["id_idx", "pos"])
    pool = draws[~((draws.draw == 0) & (draws.role == "support"))].sort_values(["id_idx", "draw", "pos"])
    assert (pool.groupby("id_idx").size() == 68).all()
    rng = np.random.default_rng(20260927)
    id_dev = samples[samples.split == "id_dev"]
    id_train = id_dev.groupby("class_idx_id", group_keys=False).apply(
        lambda g: g.sample(5, random_state=int(rng.integers(1 << 30)))).sample_id.values
    near = samples[samples.split == "near_dev"]
    ood = near.groupby("wnid", group_keys=False).apply(
        lambda g: g.sample(15, random_state=int(rng.integers(1 << 30)))).sample_id.values
    v2b = torch.load(C.WORK / "extra_feats" / "in_v2.dino.pt", map_location="cpu")
    v2l = torch.load(C.WORK / "extra_feats" / "in_v2.dinol14.pt", map_location="cpu")
    v2c = torch.load(C.WORK / "extra_feats" / "in_v2.clipb16.pt", map_location="cpu")
    assert v2b["ids"] == v2l["ids"] == v2c["ids"]
    lab = np.asarray(v2b["labels"])
    v2_rows = []
    for c in id_classes:
        idx = np.flatnonzero(lab == c["idx_1k"])
        v2_rows += sorted(rng.choice(idx, 5, replace=False).tolist())
    v2_rows = np.array(v2_rows)
    n_stream = len(id_train) + len(ood)
    order = rng.permutation(n_stream)                          # positions: first 4,500 ids = ID, rest = OOD
    is_ood = np.r_[np.zeros(len(id_train), bool), np.ones(len(ood), bool)][order]
    bidx = np.arange(n_stream) // C.BATCH
    shot_clip, sc_blob = load_features(r5.R5 / "features" / "shots.clip.pt")
    spos = {s: i for i, s in enumerate(sc_blob["sample_id"])}
    k_pool = zeroshot(shot_clip[[spos[s] for s in pool.sample_id]], pos_text)
    k_train = np.array(dview.loc[np.r_[id_train, ood]].K_id.tolist())[order]
    k_v2_id = zeroshot(v2c["features"][v2_rows], pos_text)
    k_ood = np.array(dview.loc[ood].K_id.tolist())
    k_v2 = np.concatenate([k_v2_id, k_ood])[order]
    cls_train = np.r_[samples.set_index("sample_id").loc[id_train].class_idx_id.values.astype(int),
                      np.full(len(ood), -1)][order]
    cls_v2 = np.r_[np.array([id_of[int(lab[r_])] for r_ in v2_rows]), np.full(len(ood), -1)][order]
    feats = {}
    for name, dev_file, shot_file, v2 in (("B14", "dino.pt", "dino", v2b), ("L14", "dino_vitl14.pt", "dino_vitl14", v2l)):
        dev_f, blob = load_features(C.FEATURES_DIR / dev_file)
        dpos = {s: i for i, s in enumerate(blob["sample_id"])}
        sh_f, sblob = load_features(r5.R5 / "features" / f"shots.{shot_file}.pt")
        sp2 = {s: i for i, s in enumerate(sblob["sample_id"])}
        feats[name] = {
            "support": sh_f[[sp2[s] for s in sup.sample_id]].numpy().reshape(n_id, 12, -1),
            "pool": sh_f[[sp2[s] for s in pool.sample_id]].numpy(),
            "train": dev_f[[dpos[s] for s in np.r_[id_train, ood]]].numpy()[order],
            "v2": np.concatenate([v2["features"][v2_rows].numpy(), dev_f[[dpos[s] for s in ood]].numpy()])[order],
        }
    return {"feats": feats, "k_pool": k_pool, "k": {"train": k_train, "v2": k_v2}, "is_ood": is_ood, "bidx": bidx,
            "cls": {"train": cls_train, "v2": cls_v2}, "pool_cls": pool.id_idx.values, "n_id": n_id}


def rates(p, is_ood, bidx, cls, n_id):
    idm = ~is_ood
    half = bidx < (bidx.max() + 1) // 2
    out = {}
    for a in ALPHAS:
        hit = (p <= a) & idm
        out[f"all@{a:g}"] = float(hit.sum() / idm.sum())
        out[f"first@{a:g}"] = float(hit[half].sum() / idm[half].sum())
        out[f"second@{a:g}"] = float(hit[~half].sum() / idm[~half].sum())
        out[f"cls@{a:g}"] = np.bincount(cls[hit], minlength=n_id).astype(np.int16)
    return out


def task(job):
    stream, r = job
    D, ev = G["D"], G["ev"]
    is_ood, bidx, cls, n_id = D["is_ood"], D["bidx"], D["cls"][stream], D["n_id"]
    rng = np.random.default_rng([r, 11])
    pool_cls = D["pool_cls"]
    pick = np.stack([rng.choice(np.flatnonzero(pool_cls == c), 8, replace=False) for c in range(n_id)])  # (n_id, 8)
    res, t0 = {}, time.time()
    for name in ("B14", "L14"):
        V = G["views"][(stream, name)]
        n_pool = len(pool_cls)
        d, d_all, stats = V["d"], V["d_all"], V["stats"]
        med, mad = stats["med_all"], stats["mad_all"]
        pf = G["D"]["feats"][name]["pool"]
        sf = G["D"]["feats"][name][stream]
        ds, das = d[n_pool:], d_all[n_pool:]
        cal = lambda cols: (pf[pick[:, cols].ravel()], d[pick[:, cols].ravel()], d_all[pick[:, cols].ravel()])  # noqa: E731
        c4 = cal([0, 1, 2, 3])
        res[f"{name}|static_p"] = rates(r5.pval_high(c4[1], ds), is_ood, bidx, cls, n_id)
        variants = {"shared4": [c4] * 4, "split1": [cal([0]), cal([1]), cal([2]), cal([3])], "reuse1": [cal([0])] * 4}
        for vname, sets in variants.items():
            masks, ps, pt = r5.entrance_cal(sf, ds, das, bidx, sets, med, mad)
            for stage, p in zip(("p_all", "p_A1", "p_A2"), ps):
                res[f"{name}|{vname}|{stage}"] = rates(p, is_ood, bidx, cls, n_id)
            res[f"{name}|{vname}|p_t"] = rates(pt, is_ood, bidx, cls, n_id)
            res[f"{name}|{vname}|admit_M"] = {"id_rate": float(masks[-1][~is_ood].mean()),
                                              "ood_rate": float(masks[-1][is_ood].mean())}
            if vname == "shared4":
                ho = pick[:, 4:].ravel()
                snap = r5.memory_p_snapshot(sf, bidx, masks[-1], c4[0], c4[1], pf[ho], d[ho], med, mad, 2, SNAPS)
                for b, p in snap.items():
                    res[f"{name}|heldout|b{b}"] = {f"all@{a:g}": float((p <= a).mean()) for a in ALPHAS}
        sup = G["D"]["feats"][name]["support"]
        for mode in ("warm", "cold", "converge"):
            lp = r5.lp_run(sup, c4[0], sf, bidx, **r5.V4_LP, init=mode, tol=1e-6, max_iter=2000)
            res[f"{name}|lp_{mode}"] = rates(lp["p"], is_ood, bidx, cls, n_id)
            res[f"{name}|lp_{mode}"]["mean_sweeps"] = float(lp["sweeps"].mean())
        res[f"{name}|lp_aligned"] = rates(r5.lp_aligned(sup, c4[0], sf, bidx, **{k: r5.V4_LP[k] for k in
                                                                               ("k", "alpha", "gamma", "iters")}),
                                          is_ood, bidx, cls, n_id)
    flat = {}
    for key, val in res.items():
        for k2, x in val.items():
            flat[f"{key}|{k2}"] = x
    np.savez_compressed(OUT / f"{stream}_r{r}.npz", **{k: np.asarray(v) for k, v in flat.items()})
    return stream, r, round(time.time() - t0, 1)


def summary():
    out = {}
    for stream in ("train", "v2"):
        files = [np.load(OUT / f"{stream}_r{r}.npz") for r in range(N_RES) if (OUT / f"{stream}_r{r}.npz").exists()]
        if not files:
            continue
        keys = files[0].files
        out[stream] = {"n_resamples": len(files)}
        for k in keys:
            vals = np.stack([f[k] for f in files])
            if vals.ndim == 1:
                out[stream][k] = {"mean": float(vals.mean()), "sd": float(vals.std(ddof=1)) if len(files) > 1 else 0.0,
                                  "min": float(vals.min()), "max": float(vals.max())}
            else:                                  # per-class counts: pooled class rates over resamples
                per_cls = vals.sum(axis=0) / (5.0 * len(files))
                out[stream][k] = {"q05": float(np.quantile(per_cls, 0.05)), "q50": float(np.quantile(per_cls, 0.5)),
                                  "q95": float(np.quantile(per_cls, 0.95)), "max": float(per_cls.max()),
                                  "frac_gt_2alpha": float((per_cls > 2 * float(k.split("@")[-1])).mean())}
    (OUT / "summary.json").write_text(json.dumps(out, indent=1) + "\n")
    L = ["| stream | p-value | alpha | mean Pr(p<=alpha) | sd over 50 resamples | first half | second half |", "|---|---|---|---|---|---|---|"]
    for stream, d in out.items():
        for k in sorted(d):
            if k.endswith("|all@0.1") and "cls" not in k:
                base = k[:-len("|all@0.1")]
                f1, f2 = d.get(f"{base}|first@0.1"), d.get(f"{base}|second@0.1")
                L.append(f"| {stream} | {base} | 0.10 | {100 * d[k]['mean']:.2f} | {100 * d[k]['sd']:.2f} | "
                         f"{100 * f1['mean']:.2f} | {100 * f2['mean']:.2f} |" if f1 else
                         f"| {stream} | {base} | 0.10 | {100 * d[k]['mean']:.2f} | {100 * d[k]['sd']:.2f} | - | - |")
    (OUT / "summary.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=["run", "summary"], default="run")
    parser.add_argument("--workers", type=int, default=20)
    parser.add_argument("--streams", nargs="+", default=["train", "v2"])
    opts = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if opts.phase == "summary":
        summary()
        return
    start = time.time()
    ev = load_ev()
    D = build_streams()
    views = {}
    n_pool = len(D["pool_cls"])
    for stream in opts.streams:
        cand = np.concatenate([D["k_pool"], D["k"][stream]])
        for name in ("B14", "L14"):
            F = D["feats"][name]
            q = np.concatenate([F["pool"], F[stream]]).astype(np.float32)
            is_cal = np.zeros(len(q), bool)
            is_cal[:4 * D["n_id"]] = True                           # placeholder; p values are recomputed per resample
            v = ev.custom_view(F["support"].astype(np.float32), q, cand, is_cal, np.repeat(np.arange(D["n_id"]), 4),
                               m=2, proto=True)
            views[(stream, name)] = {"d": v["d"], "d_all": v["d_all"], "stats": v["stats"]}
            assert len(v["d"]) == n_pool + len(D["is_ood"])
    G.update(D=D, ev=ev, views=views)
    jobs = [(s, r) for s in opts.streams for r in range(N_RES) if not (OUT / f"{s}_r{r}.npz").exists()]
    with mp.get_context("fork").Pool(opts.workers) as pool:
        for s, r, sec in pool.imap_unordered(task, jobs):
            print(json.dumps({"done": f"{s}_r{r}", "seconds": sec}), flush=True)
    summary()
    print(json.dumps({"seconds": round(time.time() - start, 1)}))


if __name__ == "__main__":
    main()
