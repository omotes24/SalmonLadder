"""Iteration 2 (dev only): TINS seed-rule variants x memory admission rules x fusion rules.

Inputs: <WORK>/iter2/tins/<tag>/<stream>_seed<seed>.npz (iter2_tins_dev.py) and the frozen visual view.
p_T(x)     : conformal p-value of S_final(x) against the calibration shots' TINS scores of the same batch
             (same negatives), small = OOD.
p_joint(x) : conformal p-value of T = p_T * p_all against the calibration shots' T (leave-one-out ranks).
Admission rules (memory M of the visual view):
  pall<E>  : p_all <= E%             (CLAVIS-M c1 uses pall10)
  pT<E>    : p_T <= E%
  joint<E> : p_joint <= E%
  union<E> : p_all <= E/2 % or p_T <= E/2 %
  ...|wb   : batch b's admitted samples also join M before batch b is scored (self excluded)
Fusion: S*pt = S_final * p_t (c1), pT*pt = p_T * p_t (Fisher order).
Output: <WORK>/iter2/eval/<name>.json
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from vins import config as C  # noqa: E402
from vins.clavism import dev_views, views_from_arrays  # noqa: E402
from vins.features import load_features  # noqa: E402
from vins.memory import EMPTY, TopM, memory_distances, online_pvalues, pvalues_sorted, top_sims  # noqa: E402
from vins.metrics import aggregate, measures  # noqa: E402
from vins.tins_dev import import_tins  # noqa: E402


def loo_le(values):
    """(#{k: v_k <= v_j}) / n for every j (includes j itself) = leave-one-out p-value with ties counted."""
    s = np.sort(values)
    return np.searchsorted(s, values, side="right") / len(values)


def p_text(S, cal, bidx):
    """p_T per sample and leave-one-out p_T of the calibration shots per batch."""
    batches = np.unique(bidx)
    pT = np.empty(len(S))
    pT_cal = np.empty((len(batches), cal.shape[1]))
    for bi, b in enumerate(batches):
        rows = np.flatnonzero(bidx == b)
        c = np.sort(cal[bi].astype(np.float64))
        pT[rows] = (1.0 + np.searchsorted(c, S[rows], side="right")) / (len(c) + 1.0)
        pT_cal[bi] = loo_le(cal[bi].astype(np.float64))
    return pT, pT_cal


def p_joint(pT, p_all, pT_cal, pall_cal, bidx):
    batches = np.unique(bidx)
    out = np.empty(len(pT))
    for bi, b in enumerate(batches):
        rows = np.flatnonzero(bidx == b)
        tc = np.sort(pT_cal[bi] * pall_cal)
        out[rows] = (1.0 + np.searchsorted(tc, pT[rows] * p_all[rows], side="right")) / (len(tc) + 1.0)
    return out


def memory_distances_wb(stream_feats, batch_index, admit, cal_feats, m_max=2):
    """As memory_distances, but batch b's admitted samples join the memory before batch b is scored
    (a sample never matches itself)."""
    stream_feats = np.ascontiguousarray(stream_feats, dtype=np.float32)
    n = len(stream_feats)
    batches = np.unique(batch_index)
    r_stream = np.empty((n, m_max))
    r_cal = np.empty((len(batches), len(cal_feats), m_max))
    cal = TopM(cal_feats, m_max)
    mem = np.zeros((0, stream_feats.shape[1]), dtype=np.float32)
    for bi, b in enumerate(batches):
        rows = np.flatnonzero(batch_index == b)
        new_rows = rows[admit[rows]]
        q = stream_feats[rows]
        old = top_sims(q, mem, m_max)
        if len(new_rows):
            sims = q @ stream_feats[new_rows].T
            self_pos = np.flatnonzero(admit[rows])
            sims[self_pos, np.arange(len(new_rows))] = -np.inf
            both = np.concatenate([old, sims], axis=1)
        else:
            both = old
        both = -np.sort(-both, axis=1)[:, :m_max] if both.shape[1] else both
        cal.add(stream_feats[new_rows])
        for m in range(1, m_max + 1):
            if both.shape[1] >= m:
                col = both[:, m - 1]
                r_stream[rows, m - 1] = np.where(np.isfinite(col), 1.0 - col, EMPTY)
            else:
                r_stream[rows, m - 1] = EMPTY
            r_cal[bi, :, m - 1] = cal.dist(m)
        if len(new_rows):
            mem = np.concatenate([mem, stream_feats[new_rows]], axis=0)
    return r_stream, r_cal, batches


ONLINE = ("self", "pself", "pselfu", "jself")


def online_memory(feats, bidx, d, cal_feats, d_cal, med_all, mad_all, kind, e, p_all, pT, pT_cal, m_mem=2):
    """Memory whose admission uses the memory-dependent p_t itself (decided after batch b is scored):
    self: p_t <= e | pself: p_t <= e/2 or p_all <= e/2 | pselfu: p_t <= e or p_all <= e |
    jself: conformal p of p_T * p_t (calibration: leave-one-out ranks) <= e."""
    feats = np.ascontiguousarray(feats, dtype=np.float32)
    batches = np.unique(bidx)
    cal = TopM(cal_feats, m_mem)
    mem = np.zeros((0, feats.shape[1]), dtype=np.float32)
    n = len(d)
    p_t, g, admit = np.empty(n), np.empty(n), np.zeros(n, dtype=bool)
    for bi, b in enumerate(batches):
        rows = np.flatnonzero(bidx == b)
        top = top_sims(feats[rows], mem, m_mem)
        r = 1.0 - top[:, m_mem - 1].astype(np.float64) if top.shape[1] >= m_mem else np.full(len(rows), EMPTY)
        gb = d[rows] - (r - med_all) / mad_all
        gc = d_cal - (cal.dist(m_mem) - med_all) / mad_all
        gcs = np.sort(gc)
        pb = pvalues_sorted(gcs, gb)
        p_t[rows], g[rows] = pb, gb
        if kind == "self":
            a = pb <= e
        elif kind == "pself":
            a = (pb <= e / 2) | (p_all[rows] <= e / 2)
        elif kind == "pselfu":
            a = (pb <= e) | (p_all[rows] <= e)
        elif kind == "jself":
            pt_cal = (len(gcs) - np.searchsorted(gcs, gc, side="left")) / len(gcs)
            tc = np.sort(pT_cal[bi] * pt_cal)
            a = (1.0 + np.searchsorted(tc, pT[rows] * pb, side="right")) / (len(tc) + 1.0) <= e
        else:
            raise ValueError(kind)
        admit[rows] = a
        new = feats[rows[a]]
        if len(new):
            mem = np.concatenate([mem, new], axis=0)
            cal.add(new)
    return p_t, g, admit


def views_generic(feat_file, m=2):
    """dev_views for another visual backbone (same support/calibration split and candidate sets)."""
    samples = pd.read_parquet(C.SPLITS_DIR / "samples.parquet")
    dview = pd.read_parquet(C.RUNS_DIR / "dview" / "dview.parquet")
    feats, blob = load_features(C.FEATURES_DIR / feat_file)
    assert blob["sample_id"] == samples.sample_id.tolist()
    feats = feats.numpy().astype(np.float32)
    row_of = pd.Series(np.arange(len(samples)), index=samples.sample_id.values)
    sup = np.flatnonzero((samples.split == "support").values)
    sup = sup[np.argsort(samples.class_idx_id.values[sup], kind="stable")]
    support = feats[sup].reshape(1000 - C.N_HELDOUT, C.N_SUPPORT, -1)
    queries = feats[row_of.loc[dview.sample_id].values]
    is_cal = (dview.split == "calib").values
    v = views_from_arrays(support, queries, np.array(dview.K_id.tolist()), is_cal, m)
    v["frame"] = pd.DataFrame({"d": v["d"], "p": v["p"], "p_all": v["p_all"]}, index=dview.sample_id.values)
    v["feats"], v["row_of"] = feats, row_of
    return v


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tags", nargs="+", required=True)
    parser.add_argument("--admits", nargs="+", default=["pall10", "pT10", "joint10", "union10"])
    parser.add_argument("--wb", action="store_true", help="also evaluate within-batch admission for each rule")
    parser.add_argument("--name", required=True)
    parser.add_argument("--dino-file", default=None, help="features/<file> of another backbone (default: frozen B/14 view)")
    parser.add_argument("--save-scores", action="store_true", help="save per-sample p-values to iter2/scores/<name>/")
    opts = parser.parse_args()
    tick = time.time()
    import_tins()
    v = dev_views(m=2) if opts.dino_file is None else views_generic(opts.dino_file, m=2)
    dview = pd.read_parquet(C.RUNS_DIR / "dview" / "dview.parquet")
    is_cal = (dview.split == "calib").values
    cal_ids = dview.sample_id.values[is_cal]
    d_all_cal = v["d_all"][is_cal]
    pall_cal = np.searchsorted(np.sort(-d_all_cal), -d_all_cal, side="right") / len(d_all_cal)   # #{d_all_k >= d_all_j}/n
    med_all, mad_all = v["stats"]["med_all"], v["stats"]["mad_all"]
    results = {}
    for tag in opts.tags:
        per_seed, admits_log = [], {}
        for seed in C.ORDER_SEEDS:
            entry = {}
            for stream in ("near", "far"):
                z = np.load(C.WORK / "iter2" / "tins" / tag / f"{stream}_seed{seed}.npz", allow_pickle=True)
                assert (z["cal_ids"] == cal_ids).all()
                sid, is_id, bidx = z["sample_id"], z["is_ood"] == 0, z["batch_index"]
                S = z["S_final"].astype(np.float64)
                fr = v["frame"].loc[sid]
                d, p, p_all = fr.d.values, fr.p.values, fr.p_all.values
                sf = v["feats"][v["row_of"].loc[sid].values]
                pT, pT_cal = p_text(S, z["cal_scores"], bidx)
                pj = p_joint(pT, p_all, pT_cal, pall_cal, bidx)
                res = {"tins": measures(S[is_id], S[~is_id]), "pT": measures(pT[is_id], pT[~is_id]),
                       "S*p": measures((S * p)[is_id], (S * p)[~is_id])}
                rules, online = {}, {}
                for a in opts.admits:
                    if a.startswith("cand"):          # two-stage: candidate memory p_all <= EC, admit p_C <= E [or p_all <= EU]
                        parts = [float(x) / 100 for x in a[4:].split("_")]
                        ec, e = parts[0], parts[1]
                        r_s, r_c, batches = memory_distances(sf, bidx, p_all <= ec, v["cal_feats"], m_max=2)
                        p_c, _ = online_pvalues("diff", 2, d, v["d_cal"], r_s, r_c, bidx, batches, med_all, mad_all)
                        rules[a] = (p_c <= e) | (p_all <= parts[2]) if len(parts) > 2 else p_c <= e
                        continue
                    kind, e = "".join(ch for ch in a if ch.isalpha()), float("".join(ch for ch in a if not ch.isalpha())) / 100
                    if kind in ONLINE:
                        online[a] = (kind, e)
                        continue
                    rules[a] = {"pall": p_all <= e, "pT": pT <= e, "joint": pj <= e,
                                "union": (p_all <= e / 2) | (pT <= e / 2),
                                # analysis only (uses labels): memory without ID contamination / with every OOD
                                "oraclepall": (p_all <= e) & ~is_id, "oracleood": ~is_id & (p_all <= 1.0),
                                "oraclepallid": (p_all <= e) | ~is_id}[kind]
                evals = []
                for a, admit in rules.items():
                    for wb in ([False, True] if opts.wb else [False]):
                        fn = memory_distances_wb if wb else memory_distances
                        r_s, r_c, batches = fn(sf, bidx, admit, v["cal_feats"], m_max=2)
                        p_t, g = online_pvalues("diff", 2, d, v["d_cal"], r_s, r_c, bidx, batches, med_all, mad_all)
                        evals.append((a + ("|wb" if wb else ""), p_t, g, admit))
                for a, (kind, e) in online.items():
                    p_t, g, admit = online_memory(sf, bidx, d, v["cal_feats"], v["d_cal"], med_all, mad_all, kind, e,
                                                  p_all, pT, pT_cal)
                    evals.append((a, p_t, g, admit))
                for name, p_t, g, admit in evals:
                    res[f"{name}|S*pt"] = measures((S * p_t)[is_id], (S * p_t)[~is_id])
                    res[f"{name}|pT*pt"] = measures((pT * p_t)[is_id], (pT * p_t)[~is_id])
                    res[f"{name}|g"] = measures(-g[is_id], -g[~is_id])
                    admits_log[f"{name}|{stream}_seed{seed}"] = [float(admit[is_id].mean()), float(admit[~is_id].mean())]
                names = [x[0] for x in evals]
                if opts.save_scores:
                    sdir = C.WORK / "iter2" / "scores" / opts.name
                    sdir.mkdir(parents=True, exist_ok=True)
                    extra = {}
                    for name, p_t, g, admit in evals:
                        key = name.replace("|", "_")
                        extra[f"pt__{key}"], extra[f"adm__{key}"] = p_t.astype(np.float32), admit
                    np.savez_compressed(sdir / f"{tag}_{stream}_seed{seed}.npz", sample_id=sid, is_id=is_id, S=S,
                                        pT=pT.astype(np.float32), p=p.astype(np.float32),
                                        p_all=p_all.astype(np.float32), batch_index=bidx, **extra)
                entry[stream] = res
            per_seed.append(entry)
        agg = aggregate(per_seed)
        results[tag] = {"aggregate": agg, "admits": admits_log}
        rows = sorted(agg["near"].keys())
        print(f"== {tag}  (AUROC near / far, FPR95 near / far)")
        for k in rows:
            a_n, a_f = agg["near"][k]["AUROC"]["mean"] * 100, agg["far"][k]["AUROC"]["mean"] * 100
            f_n, f_f = agg["near"][k]["FPR95"]["mean"] * 100, agg["far"][k]["FPR95"]["mean"] * 100
            print(f"  {k:24s} {a_n:6.2f} {a_f:6.2f} | {f_n:6.2f} {f_f:6.2f}")
        for name in names:
            vals = np.array([admits_log[f"{name}|{s}_seed{sd}"] for s in ("near", "far") for sd in C.ORDER_SEEDS])
            print(f"  admit {name:12s} ID {100 * vals[:, 0].mean():5.2f}%  OOD near {100 * vals[:3, 1].mean():5.1f}% far {100 * vals[3:, 1].mean():5.1f}%")
    out = C.WORK / "iter2" / "eval"
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{opts.name}.json").write_text(json.dumps({"results": results, "seconds": round(time.time() - tick, 1)}) + "\n")
    print(json.dumps({"done": opts.name, "seconds": round(time.time() - tick, 1)}))


if __name__ == "__main__":
    main()
