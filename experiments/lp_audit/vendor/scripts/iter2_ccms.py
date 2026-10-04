"""Iteration 2 (dev only): class-conditional memory standardization (CCMS) for the visual memory.

g_glob(x) = d(x) - (r_M(x) - med_all) / mad_all                     (CLAVIS-M c1)
g_ccms(x) = d(x) - (r_M(x) - mu_M[c(x)]) / mad_all
  c(x)    = the candidate class attaining d(x) (argmin over K(x) of z_c)
  mu_M[c] = shrunk median of r_M over the 16 shots of class c (12 support + 4 calibration; never admitted),
            (16 * med_c + n0 * med_pool) / (16 + n0), recomputed under the memory of every batch.
An admitted ID image that sits close to its own class moves that class's shots as much as the class's
test images, so the offset cancels the class-specific boost; a novel class clustered in the memory does not.
Admission rules: pall<E> | cand<EC>_<E> (candidate memory p_all <= EC; admit if p_C <= E, p_C from the same g kind).
Output: <WORK>/iter2/eval/<name>.json
"""
import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from vins import config as C  # noqa: E402
from vins.clavism import z_all_classes  # noqa: E402
from vins.dview import loo_stats  # noqa: E402
from vins.features import load_features  # noqa: E402
from vins.memory import EMPTY, TopM, pvalues_sorted, top_sims  # noqa: E402
from vins.metrics import aggregate, measures  # noqa: E402
from vins.tins_dev import import_tins  # noqa: E402


def build_view(feat_file="dino.pt", m=2, n0=C.N0):
    samples = pd.read_parquet(C.SPLITS_DIR / "samples.parquet")
    dview = pd.read_parquet(C.RUNS_DIR / "dview" / "dview.parquet")
    feats, blob = load_features(C.FEATURES_DIR / feat_file)
    assert blob["sample_id"] == samples.sample_id.tolist()
    feats = feats.numpy().astype(np.float32)
    row_of = pd.Series(np.arange(len(samples)), index=samples.sample_id.values)
    n_id = 1000 - C.N_HELDOUT
    sup = np.flatnonzero((samples.split == "support").values)
    sup = sup[np.argsort(samples.class_idx_id.values[sup], kind="stable")]
    support = feats[sup].reshape(n_id, C.N_SUPPORT, -1).astype(np.float64)
    stats = loo_stats(support, m, n0, C.MAD_SCALE)
    q = feats[row_of.loc[dview.sample_id].values].astype(np.float64)
    z = z_all_classes(q, support, stats, m)
    cand = np.array(dview.K_id.tolist())
    zc = np.take_along_axis(z, cand, axis=1)
    j = zc.argmin(axis=1)
    rows = np.arange(len(q))
    d, cls = zc[rows, j], cand[rows, j]
    d_all = z.min(axis=1)
    is_cal = (dview.split == "calib").values
    cal_d, cal_all = np.sort(d[is_cal]), np.sort(d_all[is_cal])
    frame = pd.DataFrame({"d": d, "p": pvalues_sorted(cal_d, d), "p_all": pvalues_sorted(cal_all, d_all), "cls": cls},
                         index=dview.sample_id.values)
    cal_cls = samples.class_idx_id.values[row_of.loc[dview.sample_id.values[is_cal]].values]
    assert (cal_cls == cls[is_cal]).mean() > 0.5          # sanity: most shots attain d at their own class
    # all 16 shots per class (support + calibration), with the calibration rows marked
    shot_feats = np.concatenate([support.reshape(-1, support.shape[-1]).astype(np.float32), q[is_cal].astype(np.float32)])
    shot_cls = np.concatenate([np.repeat(np.arange(n_id), C.N_SUPPORT), cal_cls])
    n_sup = n_id * C.N_SUPPORT
    return {"frame": frame, "feats": feats, "row_of": row_of, "stats": stats, "d_cal": d[is_cal], "cls_cal": cls[is_cal],
            "shot_feats": shot_feats, "shot_cls": shot_cls, "cal_slice": slice(n_sup, len(shot_feats)), "n_id": n_id}


def run_memory(v, sf, bidx, d, cls, admit, kind, m_mem=2, n0=C.N0):
    """Static admission mask; returns p_t, g for the stream (batch b scored against admits of batches < b)."""
    med_all, mad_all = v["stats"]["med_all"], v["stats"]["mad_all"]
    sf = np.ascontiguousarray(sf, dtype=np.float32)
    shots = TopM(v["shot_feats"], m_mem)
    cal_sl, n_id, shot_cls = v["cal_slice"], v["n_id"], v["shot_cls"]
    order = np.argsort(shot_cls, kind="stable")
    counts = np.bincount(shot_cls, minlength=n_id)
    assert (counts == counts[0]).all()
    mem = np.zeros((0, sf.shape[1]), dtype=np.float32)
    batches = np.unique(bidx)
    p_t, g = np.empty(len(d)), np.empty(len(d))
    for b in batches:
        rows = np.flatnonzero(bidx == b)
        top = top_sims(sf[rows], mem, m_mem)
        r = 1.0 - top[:, m_mem - 1].astype(np.float64) if top.shape[1] >= m_mem else np.full(len(rows), EMPTY)
        r_shots = shots.dist(m_mem)
        if kind == "glob":
            gb = d[rows] - (r - med_all) / mad_all
            gc = v["d_cal"] - (r_shots[cal_sl] - med_all) / mad_all
        else:
            per = r_shots[order].reshape(n_id, counts[0])
            mu = (counts[0] * np.median(per, axis=1) + n0 * np.median(r_shots)) / (counts[0] + n0)
            gb = d[rows] - (r - mu[cls[rows]]) / mad_all
            gc = v["d_cal"] - (r_shots[cal_sl] - mu[v["cls_cal"]]) / mad_all
        p_t[rows] = pvalues_sorted(np.sort(gc), gb)
        g[rows] = gb
        new = sf[rows[admit[rows]]]
        if len(new):
            mem = np.concatenate([mem, new], axis=0)
            shots.add(new)
    return p_t, g


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dino-file", default="dino.pt")
    parser.add_argument("--admits", nargs="+", default=["pall10", "cand30_10"])
    parser.add_argument("--kinds", nargs="+", default=["glob", "ccms"])
    parser.add_argument("--tag", default="default")
    parser.add_argument("--name", required=True)
    opts = parser.parse_args()
    tick = time.time()
    import_tins()
    v = build_view(opts.dino_file)
    per_seed, admits_log = [], {}
    for seed in C.ORDER_SEEDS:
        entry = {}
        for stream in ("near", "far"):
            z = np.load(C.WORK / "iter2" / "tins" / opts.tag / f"{stream}_seed{seed}.npz", allow_pickle=True)
            sid, is_id, bidx = z["sample_id"], z["is_ood"] == 0, z["batch_index"]
            S = z["S_final"].astype(np.float64)
            fr = v["frame"].loc[sid]
            d, p, p_all, cls = fr.d.values, fr.p.values, fr.p_all.values, fr.cls.values.astype(int)
            sf = v["feats"][v["row_of"].loc[sid].values]
            res = {"tins": measures(S[is_id], S[~is_id]), "S*p": measures((S * p)[is_id], (S * p)[~is_id]),
                   "d": measures(-d[is_id], -d[~is_id])}
            for kind in opts.kinds:
                for a in opts.admits:
                    if a.startswith("cand"):
                        ec, e = [float(x) / 100 for x in a[4:].split("_")]
                        p_c, _ = run_memory(v, sf, bidx, d, cls, p_all <= ec, kind)
                        admit = p_c <= e
                    else:
                        admit = p_all <= float(a[4:]) / 100
                    p_t, g = run_memory(v, sf, bidx, d, cls, admit, kind)
                    res[f"{kind}|{a}|S*pt"] = measures((S * p_t)[is_id], (S * p_t)[~is_id])
                    res[f"{kind}|{a}|g"] = measures(-g[is_id], -g[~is_id])
                    admits_log[f"{kind}|{a}|{stream}_seed{seed}"] = [float(admit[is_id].mean()), float(admit[~is_id].mean())]
            entry[stream] = res
        per_seed.append(entry)
    agg = aggregate(per_seed)
    print(f"== CCMS {opts.dino_file} tag={opts.tag}  (AUROC near / far, FPR95 near / far)")
    for k in sorted(agg["near"]):
        a_n, a_f = agg["near"][k]["AUROC"]["mean"] * 100, agg["far"][k]["AUROC"]["mean"] * 100
        f_n, f_f = agg["near"][k]["FPR95"]["mean"] * 100, agg["far"][k]["FPR95"]["mean"] * 100
        print(f"  {k:28s} {a_n:6.2f} {a_f:6.2f} | {f_n:6.2f} {f_f:6.2f}")
    for kind in opts.kinds:
        for a in opts.admits:
            vals = np.array([admits_log[f"{kind}|{a}|{s}_seed{sd}"] for s in ("near", "far") for sd in C.ORDER_SEEDS])
            print(f"  admit {kind}|{a:12s} ID {100 * vals[:, 0].mean():5.2f}%  OOD near {100 * vals[:3, 1].mean():5.1f}% far {100 * vals[3:, 1].mean():5.1f}%")
    out = C.WORK / "iter2" / "eval"
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{opts.name}.json").write_text(json.dumps({"aggregate": agg, "admits": admits_log,
                                                       "seconds": round(time.time() - tick, 1)}) + "\n")
    print(json.dumps({"done": opts.name, "seconds": round(time.time() - tick, 1)}))


if __name__ == "__main__":
    main()
