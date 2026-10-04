"""Iteration 3 (dev only): multi-view CLAVIS-M2-style evaluation with feature-space transforms.

View spec  : <feature file>[:whiten<A>]            e.g. dino.pt, dino_vitl14.pt:whiten30, clip.pt
  whiten<A>: within-class covariance S_w of the support shots (class means removed), shrunk
             (1-a) S_w + a tr(S_w)/D I (a = A/100); u = l2norm((f - c) S^{-1/2}), c = mean of the class means.
             The whole CLAVIS view (m-th NN robust z, K(x), conformal p, memories) then runs on u.
Rule       : pall<E> | cand<EC>_<E> (two-stage admission, vins.iter2) — the same rule for every view
Combos     : "0,1" = S_final * p_t(view0) * p_t(view1); per-view S*p (static), S*p_t, d are also reported
TINS side  : runs/main (unchanged upstream, 3 order seeds)
Output     : <WORK>/iter3/eval/<name>.json (+ per-sample p_t with --save)
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
from vins.clavism import views_from_arrays  # noqa: E402
from vins.candidates import class_r_all  # noqa: E402
from vins.dview import d_scores, loo_stats  # noqa: E402
from vins.memory import EMPTY, TopM, pvalues_sorted, top_sims  # noqa: E402
from vins.features import load_features  # noqa: E402
from vins.iter2 import p_memory  # noqa: E402
from vins.memory import g_score, memory_distances  # noqa: E402
from vins.metrics import aggregate, measures  # noqa: E402
from vins.tins_dev import import_tins  # noqa: E402


def whiten(feats, sup_rows, sup_cls, alpha):
    S = feats[sup_rows].astype(np.float64)
    n_cls = int(sup_cls.max()) + 1
    mu = np.zeros((n_cls, S.shape[1]))
    np.add.at(mu, sup_cls, S)
    mu /= np.bincount(sup_cls, minlength=n_cls)[:, None]
    R = S - mu[sup_cls]
    cov = R.T @ R / (len(S) - n_cls)
    dim = cov.shape[0]
    cov = (1 - alpha) * cov + alpha * np.trace(cov) / dim * np.eye(dim)
    evals, evecs = np.linalg.eigh(cov)
    W = evecs @ np.diag(evals ** -0.5) @ evecs.T
    U = (feats.astype(np.float64) - mu.mean(axis=0)) @ W
    return (U / np.linalg.norm(U, axis=1, keepdims=True)).astype(np.float32)


def shrink_stats(loo, n0=C.N0, mad_scale=C.MAD_SCALE):
    n = loo.shape[1]
    med_c = np.median(loo, axis=1)
    mad_c = mad_scale * np.median(np.abs(loo - med_c[:, None]), axis=1)
    med_all = float(np.median(loo))
    mad_all = float(mad_scale * np.median(np.abs(loo - med_all)))
    return {"med_t": (n * med_c + n0 * med_all) / (n + n0), "mad_t": (n * mad_c + n0 * mad_all) / (n + n0),
            "med_all": med_all, "mad_all": mad_all}


def custom_view(support12, queries, cand, is_cal, cal_cls, m=2, proto=False, sup16=False):
    """sup16: the 4 calibration shots join the support (16 per class); a calibration shot never matches itself
    (own class: (m+1)-th NN, or the prototype without it). proto: r_c = 1 - cos(x, normalised class mean),
    leave-one-out prototype distances for the class statistics. Memory standardisation keeps the kNN statistics."""
    q64 = queries.astype(np.float64)
    n_id, _, dim = support12.shape
    cal_idx = np.flatnonzero(is_cal)
    if sup16:
        order = np.argsort(cal_cls, kind="stable")
        extra = q64[cal_idx[order]].reshape(n_id, -1, dim)
        support = np.concatenate([support12.astype(np.float64), extra], axis=1)
    else:
        support = support12.astype(np.float64)
    knn = loo_stats(support, m, C.N0, C.MAD_SCALE)
    if proto:
        tot = support.sum(axis=1)
        mu = tot / np.linalg.norm(tot, axis=1, keepdims=True)
        loo_mu = tot[:, None, :] - support
        loo_mu /= np.linalg.norm(loo_mu, axis=2, keepdims=True)
        st = shrink_stats(1.0 - np.einsum("cnd,cnd->cn", support, loo_mu))
        r = 1.0 - q64 @ mu.T
        if sup16:
            own = tot[cal_cls] - q64[cal_idx]
            own /= np.linalg.norm(own, axis=1, keepdims=True)
            r[cal_idx, cal_cls] = 1.0 - np.einsum("nd,nd->n", q64[cal_idx], own)
    else:
        st = knn
        r = class_r_all(q64, support, m)
        if sup16:
            dist = 1.0 - np.einsum("nd,nkd->nk", q64[cal_idx], support[cal_cls])
            r[cal_idx, cal_cls] = np.sort(dist, axis=1)[:, m]          # self at distance 0 is skipped
    z = (r - st["med_t"][None, :]) / st["mad_t"][None, :]
    zc = np.take_along_axis(z, cand, axis=1)
    j = zc.argmin(axis=1)
    rows = np.arange(len(z))
    d, cls, d_all = zc[rows, j], cand[rows, j], z.min(axis=1)
    cal_d, cal_all = np.sort(d[is_cal]), np.sort(d_all[is_cal])
    return {"stats": knn, "d": d, "d_all": d_all, "cls": cls, "p": pvalues_sorted(cal_d, d),
            "p_all": pvalues_sorted(cal_all, d_all), "cal_feats": queries[is_cal].astype(np.float32),
            "d_cal": d[is_cal], "support": support.astype(np.float32)}


def build_view(spec, m=2):
    parts = spec.split(":")
    samples = pd.read_parquet(C.SPLITS_DIR / "samples.parquet")
    dview = pd.read_parquet(C.RUNS_DIR / "dview" / "dview.parquet")
    feats, blob = load_features(C.FEATURES_DIR / parts[0])
    assert blob["sample_id"] == samples.sample_id.tolist()
    feats = feats.numpy().astype(np.float32)
    sup = np.flatnonzero((samples.split == "support").values)
    sup = sup[np.argsort(samples.class_idx_id.values[sup], kind="stable")]
    opts = set()
    for t in parts[1:]:
        if t.startswith("whiten"):
            feats = whiten(feats, sup, samples.class_idx_id.values[sup].astype(int), float(t[6:]) / 100)
        elif t in ("proto", "sup16"):
            opts.add(t)
        else:
            raise ValueError(t)
    row_of = pd.Series(np.arange(len(samples)), index=samples.sample_id.values)
    support = feats[sup].reshape(1000 - C.N_HELDOUT, C.N_SUPPORT, -1)
    queries = feats[row_of.loc[dview.sample_id].values]
    is_cal = (dview.split == "calib").values
    cand = np.array(dview.K_id.tolist())
    cal_cls = samples.class_idx_id.values[row_of.loc[dview.sample_id.values[is_cal]].values].astype(int)
    if opts:
        v = custom_view(support, queries, cand, is_cal, cal_cls, m, proto="proto" in opts, sup16="sup16" in opts)
        cls = v["cls"]
        support = v["support"]
    else:
        v = views_from_arrays(support, queries, cand, is_cal, m)
        d2, _, _, cls = d_scores(queries, cand, support, v["stats"], m)
        assert np.allclose(d2, v["d"], atol=1e-6)
    n_id = support.shape[0]
    v["shot_feats"] = np.concatenate([support.reshape(-1, support.shape[-1]), queries[is_cal]]).astype(np.float32)
    v["shot_cls"] = np.concatenate([np.repeat(np.arange(n_id), support.shape[1]), cal_cls])
    v["cal_slice"], v["cls_cal"] = slice(n_id * support.shape[1], len(v["shot_feats"])), cls[is_cal]
    v["d_all_cal"] = v["d_all"][is_cal]
    v["support_arr"] = support
    v["frame"] = pd.DataFrame({"d": v["d"], "p": v["p"], "p_all": v["p_all"], "cls": cls}, index=dview.sample_id.values)
    v["feats"], v["row_of"] = feats, row_of
    return v


def _topk_merge(sim_a, idx_a, sim_b, idx_b, k):
    s = np.concatenate([sim_a, sim_b], axis=1)
    i = np.concatenate([idx_a, idx_b], axis=1)
    part = np.argpartition(-s, k - 1, axis=1)[:, :k]
    rows = np.arange(len(s))[:, None]
    return s[rows, part], i[rows, part]


def lp_pvalues(v, sf, bidx, k=10, alpha=0.9, gamma=3.0, iters=15):
    """Conformal label propagation: kNN graph over support shots (label 1), calibration shots and the stream so far
    (unlabelled; batch b joins before it is scored). f = propagated ID mass; p = rank of f among the calibration shots."""
    import scipy.sparse as sp
    sup = v["support_arr"].reshape(-1, v["support_arr"].shape[-1]).astype(np.float32)
    cal = v["cal_feats"].astype(np.float32)
    sf = np.ascontiguousarray(sf, dtype=np.float32)
    n_s, n_c, n_t = len(sup), len(cal), len(sf)
    X = np.concatenate([sup, cal, sf])
    N = len(X)
    top_s = np.full((N, k), -np.inf, dtype=np.float32)
    top_i = np.zeros((N, k), dtype=np.int64)
    n_fix = n_s + n_c
    for lo in range(0, n_fix, 2048):
        hi = min(lo + 2048, n_fix)
        sims = X[lo:hi] @ X[:n_fix].T
        sims[np.arange(hi - lo), np.arange(lo, hi)] = -np.inf
        part = np.argpartition(-sims, k - 1, axis=1)[:, :k]
        top_s[lo:hi] = np.take_along_axis(sims, part, axis=1)
        top_i[lo:hi] = part
    y = np.zeros(N)
    y[:n_s] = 1.0
    f = y.copy()
    p = np.empty(n_t)
    cur = n_fix
    for b in np.unique(bidx):
        rows = np.flatnonzero(bidx == b)
        assert rows[0] == cur - n_fix
        new = np.arange(cur, cur + len(rows))
        sims = X[new] @ X[:cur + len(rows)].T                      # new vs existing + new
        sims[np.arange(len(rows)), new] = -np.inf
        part = np.argpartition(-sims, k - 1, axis=1)[:, :k]
        top_s[new] = np.take_along_axis(sims, part, axis=1)
        top_i[new] = part
        old = sims[:, :cur].T                                          # existing nodes vs new nodes
        top_s[:cur], top_i[:cur] = _topk_merge(top_s[:cur], top_i[:cur], old,
                                               np.broadcast_to(new, old.shape), k)
        cur += len(rows)
        n = cur
        src = np.repeat(np.arange(n), k)
        w = np.maximum(top_s[:n].ravel(), 0.0) ** gamma
        A = sp.csr_matrix((w, (src, top_i[:n].ravel())), shape=(n, n))
        W = A.maximum(A.T)
        deg = np.asarray(W.sum(axis=1)).ravel()
        dinv = 1.0 / np.sqrt(np.maximum(deg, 1e-12))
        Wn = sp.diags(dinv) @ W @ sp.diags(dinv)
        ff = f[:n]
        for _ in range(iters):
            ff = alpha * (Wn @ ff) + (1 - alpha) * y[:n]
        f[:n] = ff
        fc = np.sort(ff[n_s:n_fix])
        p[rows] = (1.0 + np.searchsorted(fc, ff[new], side="right")) / (n_c + 1.0)
    return p


def p_memory_ccms(v, sf, d, cls, bidx, admit, m_mem=2, n0=12):
    """Memory p-value with class-conditional offsets: z_M = (r_M(x) - mu_M[c(x)]) / mad_all, mu_M[c] = shrunk median
    of r_M over the 16 shots of class c under the same memory (shots are never admitted)."""
    med_all, mad_all = v["stats"]["med_all"], v["stats"]["mad_all"]
    sf = np.ascontiguousarray(sf, dtype=np.float32)
    shots = TopM(v["shot_feats"], m_mem)
    order = np.argsort(v["shot_cls"], kind="stable")
    counts = np.bincount(v["shot_cls"])
    k = counts[0]
    mem = np.zeros((0, sf.shape[1]), dtype=np.float32)
    p = np.empty(len(d))
    for b in np.unique(bidx):
        rows = np.flatnonzero(bidx == b)
        top = top_sims(sf[rows], mem, m_mem)
        r = 1.0 - top[:, m_mem - 1].astype(np.float64) if top.shape[1] >= m_mem else np.full(len(rows), EMPTY)
        rs = shots.dist(m_mem)
        mu = (k * np.median(rs[order].reshape(-1, k), axis=1) + n0 * np.median(rs)) / (k + n0)
        gb = d[rows] - (r - mu[cls[rows]]) / mad_all
        gc = v["d_cal"] - (rs[v["cal_slice"]] - mu[v["cls_cal"]]) / mad_all
        p[rows] = pvalues_sorted(np.sort(gc), gb)
        new = sf[rows[admit[rows]]]
        if len(new):
            mem = np.concatenate([mem, new], axis=0)
            shots.add(new)
    return p


def memory_p_cal(v, sf, d, bidx, admit, m_mem=2):
    """Stream p-values (as p_memory) and leave-one-out p-values of the calibration shots per batch."""
    r_s, r_c, batches = memory_distances(sf, bidx, admit, v["cal_feats"], m_max=m_mem)
    med, mad = v["stats"]["med_all"], v["stats"]["mad_all"]
    g = g_score("diff", d, r_s[:, m_mem - 1], med, mad)
    p = np.empty(len(d))
    p_cal = np.empty((len(batches), len(v["d_cal"])))
    for bi, b in enumerate(batches):
        rows = np.flatnonzero(bidx == b)
        gc = g_score("diff", v["d_cal"], r_c[bi, :, m_mem - 1], med, mad)
        gs = np.sort(gc)
        p[rows] = pvalues_sorted(gs, g[rows])
        p_cal[bi] = (len(gs) - np.searchsorted(gs, gc, side="left")) / len(gs)
    return p, p_cal


def joint_admission(views, sfs, frs, bidx, ec=0.30, e=0.10):
    """Shared two-stage admission across views: candidates by the conformal p of prod_v p_all^v,
    admission by the conformal p of prod_v p_C^v (per batch, leave-one-out calibration products)."""
    t_all = np.prod([fr.p_all.values for fr in frs], axis=0)
    t_cal = np.ones(len(views[0]["d_cal"]))
    for v in views:                      # leave-one-out p_all of the calibration shots per view (static)
        da = v["d_all_cal"]
        t_cal = t_cal * (np.searchsorted(np.sort(-da), -da, side="right") / len(da))
    p_jall = (1.0 + np.searchsorted(np.sort(t_cal), t_all, side="right")) / (len(t_cal) + 1.0)
    cand = p_jall <= ec
    ps, pcs = [], []
    for v, sf, fr in zip(views, sfs, frs):
        p_c, p_c_cal = memory_p_cal(v, sf, fr.d.values, bidx, cand)
        ps.append(p_c)
        pcs.append(p_c_cal)
    t_s, t_c = np.prod(ps, axis=0), np.prod(pcs, axis=0)
    p_j = np.empty(len(t_s))
    for bi, b in enumerate(np.unique(bidx)):
        rows = np.flatnonzero(bidx == b)
        p_j[rows] = (1.0 + np.searchsorted(np.sort(t_c[bi]), t_s[rows], side="right")) / (t_c.shape[1] + 1.0)
    return p_j <= e, cand


def admit_mask(v, sf, d, p_all, bidx, rule, is_id=None, cls=None):
    """cand<E1>_<E2>[_<E3>...]: iterated candidate stages (C1 = p_all <= E1, C_k = p_{C_{k-1}} <= E_k),
    admission = p_{C_last} <= last threshold. or<kind>: label oracles on top of cand30_10 (analysis only).
    sdens<E>: stream-density admission, p_S (every earlier stream sample in the memory, class-conditional offsets) <= E
    sdensg<E>: the same with the global offset. A+B: union of two rules."""
    if "+" in rule:
        out = np.zeros(len(d), dtype=bool)
        for r_ in rule.split("+"):
            out |= admit_mask(v, sf, d, p_all, bidx, r_, is_id, cls)
        return out
    if rule.startswith("sdensg"):
        p_s, _ = p_memory(v, sf, d, bidx, np.ones(len(d), dtype=bool))
        return p_s <= float(rule[6:]) / 100
    if rule.startswith("sdens"):
        p_s = p_memory_ccms(v, sf, d, cls, bidx, np.ones(len(d), dtype=bool))
        return p_s <= float(rule[5:]) / 100
    if rule.startswith("cand"):
        th = [float(x) / 100 for x in rule[4:].split("_")]
        cand = p_all <= th[0]
        for t in th[1:-1]:
            p_c, _ = p_memory(v, sf, d, bidx, cand)
            cand = p_c <= t
        p_c, _ = p_memory(v, sf, d, bidx, cand)
        return p_c <= th[-1]
    if rule.startswith("or"):
        base = admit_mask(v, sf, d, p_all, bidx, "cand30_10")
        return {"orpure": base & ~is_id, "orrecall": base | ~is_id, "orall": ~is_id & np.ones_like(base)}[rule]
    return p_all <= float(rule[4:]) / 100


def view_pt(v, sid, bidx, rule, is_id=None):
    fr = v["frame"].loc[sid]
    sf = v["feats"][v["row_of"].loc[sid].values]
    d, p_all = fr.d.values, fr.p_all.values
    admit = admit_mask(v, sf, d, p_all, bidx, rule, is_id, fr.cls.values.astype(int))
    p_t, g = p_memory(v, sf, d, bidx, admit)
    return p_t, admit, fr


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--views", nargs="+", required=True)
    parser.add_argument("--rule", default="cand30_10")
    parser.add_argument("--combos", nargs="+", default=[])
    parser.add_argument("--name", required=True)
    parser.add_argument("--save", action="store_true")
    parser.add_argument("--lp", default=None, help="k,alpha,gamma: add conformal label-propagation p-values per view "
                                                   "as extra combo indices n_views + i")
    opts = parser.parse_args()
    tick = time.time()
    import_tins()
    views = [build_view(s) for s in opts.views]
    print(json.dumps({"views_built": opts.views, "t": round(time.time() - tick, 1)}), flush=True)
    per_seed, adm = [], {}
    for seed in C.ORDER_SEEDS:
        entry = {}
        for stream in ("near", "far"):
            f = pd.read_parquet(C.RUNS_DIR / "main" / f"stream_{stream}_seed{seed}.parquet").sort_values("position")
            sid, is_id, bidx = f.sample_id.values, (f.group == "ID").values, f.batch_index.values
            S = f.S_final.values.astype(np.float64)
            res, pts = {"tins": measures(S[is_id], S[~is_id])}, []
            shared = None
            if opts.rule.startswith("joint"):
                ec, e = [float(x) / 100 for x in opts.rule[5:].split("_")]
                frs = [v["frame"].loc[sid] for v in views]
                sfs = [v["feats"][v["row_of"].loc[sid].values] for v in views]
                shared, _ = joint_admission(views, sfs, frs, bidx, ec, e)
            for i, v in enumerate(views):
                if shared is not None:
                    fr = v["frame"].loc[sid]
                    p_t, _ = p_memory(v, v["feats"][v["row_of"].loc[sid].values], fr.d.values, bidx, shared)
                    admit = shared
                else:
                    p_t, admit, fr = view_pt(v, sid, bidx, opts.rule, is_id)
                pts.append(p_t)
                res[f"v{i}|d"] = measures(-fr.d.values[is_id], -fr.d.values[~is_id])
                sp = S * fr.p.values
                res[f"v{i}|S*p"] = measures(sp[is_id], sp[~is_id])
                spt = S * p_t
                res[f"v{i}|S*pt"] = measures(spt[is_id], spt[~is_id])
                adm[f"v{i}|{stream}_seed{seed}"] = [float(admit[is_id].mean()), float(admit[~is_id].mean())]
            if opts.lp:
                k_, a_, g_ = opts.lp.split(",")
                for i, v in enumerate(views):
                    p_lp = lp_pvalues(v, v["feats"][v["row_of"].loc[sid].values], bidx, int(k_), float(a_), float(g_))
                    pts.append(p_lp)
                    sl = S * p_lp
                    res[f"v{i}|S*pLP"] = measures(sl[is_id], sl[~is_id])
                    res[f"v{i}|pLP"] = measures(p_lp[is_id], p_lp[~is_id])
            for combo in opts.combos:
                idx = [int(x) for x in combo.split(",")]
                prod = np.prod([pts[j] for j in idx], axis=0)
                sc = S * prod
                res[f"c[{combo}]|S*prod"] = measures(sc[is_id], sc[~is_id])
            if opts.save:
                sdir = C.WORK / "iter3" / "scores" / opts.name
                sdir.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(sdir / f"{stream}_seed{seed}.npz", sample_id=sid, is_id=is_id, S=S,
                                    **{f"pt{i}": p.astype(np.float32) for i, p in enumerate(pts)})
            entry[stream] = res
        per_seed.append(entry)
    agg = aggregate(per_seed)
    print(f"== {opts.name}  rule={opts.rule}  views={opts.views}  (AUROC near / far | FPR95 near / far)")
    for k in sorted(agg["near"]):
        a_n, a_f = agg["near"][k]["AUROC"]["mean"] * 100, agg["far"][k]["AUROC"]["mean"] * 100
        f_n, f_f = agg["near"][k]["FPR95"]["mean"] * 100, agg["far"][k]["FPR95"]["mean"] * 100
        print(f"  {k:22s} {a_n:6.2f} {a_f:6.2f} | {f_n:6.2f} {f_f:6.2f}")
    for i in range(len(views)):
        vals = np.array([adm[f"v{i}|{s}_seed{sd}"] for s in ("near", "far") for sd in C.ORDER_SEEDS])
        print(f"  admit v{i} ID {100 * vals[:, 0].mean():5.2f}%  near OOD {100 * vals[:3, 1].mean():5.1f}%  far OOD {100 * vals[3:, 1].mean():5.1f}%")
    out = C.WORK / "iter3" / "eval"
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{opts.name}.json").write_text(json.dumps({"views": opts.views, "rule": opts.rule, "aggregate": agg,
                                                       "admits": adm, "seconds": round(time.time() - tick, 1)}) + "\n")
    print(json.dumps({"done": opts.name, "seconds": round(time.time() - tick, 1)}))


if __name__ == "__main__":
    main()
