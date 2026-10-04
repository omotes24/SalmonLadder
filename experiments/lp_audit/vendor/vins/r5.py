"""Round 5 (R5): matched-condition experiments on the frozen v4 (REPRISE / CLAVIS-M3) components.

Everything here is pure numpy / scipy. Decision functions (entrance, memory, label propagation, baselines)
receive features, frozen view statistics, calibration shots and batch indices only -- never OOD labels.
Labels enter only the metric helpers at the bottom of the file.

Conventions
  sf        : stream features (N, D), L2-normalised, in stream order
  bidx      : batch index per stream sample (non-decreasing)
  d, p_all  : frozen static view (OOD direction) and its all-class conformal p-value
  cal_feats : calibration shot features (n_cal, D); d_cal: their d
  med, mad  : scale of the memory distance (support LOO m-th NN statistics, as v4 uses for m = 2)
A p-value is small for OOD-looking inputs; a score used for AUROC is large for ID-looking inputs.
"""
import math
import os
from pathlib import Path

import numpy as np
import scipy.sparse as sp

from .memory import EMPTY, TopM, memory_distances, online_pvalues, pvalues_sorted, top_sims

R5 = Path(os.environ.get("VINS_R5", Path.home() / "vins_gonogo_20260925" / "r5"))
V4_ENTRANCE = (0.40, 0.30, 0.10)
V4_LP = {"k": 10, "alpha": 0.9, "gamma": 3.0, "iters": 15}


# ----------------------------------------------------------------------------------------------------------
# conformal helpers
# ----------------------------------------------------------------------------------------------------------
def pval_high(cal, x):
    """Conformal p-value of an OOD-direction score x (large = OOD): (1 + #{cal >= x}) / (n + 1)."""
    return pvalues_sorted(np.sort(np.asarray(cal, dtype=np.float64)), np.asarray(x, dtype=np.float64))


def pval_low(cal, x):
    """Conformal p-value of an ID-direction score x (small = OOD): (1 + #{cal <= x}) / (n + 1)."""
    cal = np.sort(np.asarray(cal, dtype=np.float64))
    return (1.0 + np.searchsorted(cal, np.asarray(x, dtype=np.float64), side="right")) / (len(cal) + 1.0)


# ----------------------------------------------------------------------------------------------------------
# memory and entrance (v4 semantics, generalised neighbour rank and thresholds)
# ----------------------------------------------------------------------------------------------------------
def memory_p(sf, d, bidx, admit, cal_feats, d_cal, med, mad, m=2):
    """v4 memory p-value (vins.iter2.p_memory, kind='diff') for neighbour rank m.
    Batch b is scored against the samples admitted in batches < b; calibration shots are never admitted."""
    r_s, r_c, batches = memory_distances(sf, bidx, admit, cal_feats, m_max=m)
    return online_pvalues("diff", m, d, d_cal, r_s, r_c, bidx, batches, med, mad)


def entrance(sf, d, p_all, bidx, cal_feats, d_cal, med, mad, thresholds=V4_ENTRANCE, m=2):
    """Stage masks of an entrance; the last mask is the memory M.
      (e1,)          static one-stage       M  = {p_all <= e1}
      (e1, e2)       A1 -> M                A1 = {p_all <= e1}, M = {p_A1 <= e2}
      (e1, e2, e3)   A1 -> A2 -> M (v4)     A2 = {p_A1 <= e2},  M = {p_A2 <= e3}
    Every set is chosen among all stream samples (no nesting is required); membership is decided after the
    sample's batch has been scored and is never revoked."""
    masks = [np.asarray(p_all) <= thresholds[0]]
    for t in thresholds[1:]:
        p_c, _ = memory_p(sf, d, bidx, masks[-1], cal_feats, d_cal, med, mad, m)
        masks.append(p_c <= t)
    return masks


def memory_p_rows(sf, d, bidx, admit, cal_feats, d_cal, med, mad, m, rows_eval):
    """memory_p evaluated only at the stream rows `rows_eval` (same values; the memory uses every admitted row)."""
    sf = np.ascontiguousarray(sf, dtype=np.float32)
    want = np.zeros(len(sf), dtype=bool)
    want[rows_eval] = True
    cal = TopM(cal_feats, m)
    mem = np.zeros((0, sf.shape[1]), dtype=np.float32)
    out = {}
    for b in np.unique(bidx):
        rows = np.flatnonzero(bidx == b)
        ev_rows = rows[want[rows]]
        if len(ev_rows):
            top = top_sims(sf[ev_rows], mem, m)
            r = 1.0 - top[:, m - 1].astype(np.float64) if top.shape[1] >= m else np.full(len(ev_rows), EMPTY)
            g = d[ev_rows] - (r - med) / mad
            gc = np.sort(d_cal - (cal.dist(m) - med) / mad)
            out.update(zip(ev_rows.tolist(), pvalues_sorted(gc, g)))
        new = sf[rows[admit[rows]]]
        if len(new):
            mem = np.concatenate([mem, new], axis=0)
            cal.add(new)
    return np.array([out[r] for r in np.asarray(rows_eval).tolist()])


def bh_mask(p, q):
    """Benjamini-Hochberg rejections at level q among the given p-values."""
    p = np.asarray(p)
    n = len(p)
    if n == 0:
        return np.zeros(0, dtype=bool)
    order = np.argsort(p)
    ok = p[order] <= q * np.arange(1, n + 1) / n
    k = np.flatnonzero(ok).max() + 1 if ok.any() else 0
    out = np.zeros(n, dtype=bool)
    out[order[:k]] = True
    return out


def entrance_bh(sf, d, p_all, bidx, cal_feats, d_cal, med, mad, qs=(0.1, 0.1, 0.1), fixed=(None, None, None), m=2):
    """Entrance whose stage i admits the Benjamini-Hochberg rejections at level qs[i] within each batch (M1).
    fixed[i] (a threshold) replaces BH at stage i when not None. p-values of a batch use the sets built from earlier
    batches only, so the per-batch BH is applied to p-values that do not depend on the batch's own admissions."""
    def stage_mask(p, i):
        if fixed[i] is not None:
            return p <= fixed[i]
        out = np.zeros(len(p), dtype=bool)
        for b in np.unique(bidx):
            rows = np.flatnonzero(bidx == b)
            out[rows] = bh_mask(p[rows], qs[i])
        return out
    masks = [stage_mask(np.asarray(p_all), 0)]
    for i in range(1, len(qs)):
        p_c, _ = memory_p(sf, d, bidx, masks[-1], cal_feats, d_cal, med, mad, m)
        masks.append(stage_mask(p_c, i))
    return masks


def combine_p(ps, rule):
    """Combine p-values (list of arrays, small = OOD) into an ID-direction score (large = ID).
    product (Fisher order) | cauchy (Cauchy combination test) | hmp (harmonic mean p) | min."""
    P = np.clip(np.stack([np.asarray(x, dtype=np.float64) for x in ps]), 1e-300, 1.0)
    if rule == "product":
        return np.exp(np.log(P).sum(axis=0))
    if rule == "cauchy":
        t = np.tan((0.5 - P) * np.pi).mean(axis=0)
        return 0.5 - np.arctan(t) / np.pi
    if rule == "hmp":
        return len(P) / (1.0 / P).sum(axis=0)
    if rule == "min":
        return P.min(axis=0)
    raise ValueError(rule)


# ----------------------------------------------------------------------------------------------------------
# conformal label propagation
# ----------------------------------------------------------------------------------------------------------
def _topk_merge(sim_a, idx_a, sim_b, idx_b, k):
    s = np.concatenate([sim_a, sim_b], axis=1)
    i = np.concatenate([idx_a, idx_b], axis=1)
    part = np.argpartition(-s, k - 1, axis=1)[:, :k]
    rows = np.arange(len(s))[:, None]
    return s[rows, part], i[rows, part]


def _normalised_graph(top_s, top_i, n, k, gamma):
    src = np.repeat(np.arange(n), k)
    w = np.maximum(top_s[:n].ravel(), 0.0) ** gamma
    A = sp.csr_matrix((w, (src, top_i[:n].ravel())), shape=(n, n))
    W = A.maximum(A.T)
    deg = np.asarray(W.sum(axis=1)).ravel()
    dinv = 1.0 / np.sqrt(np.maximum(deg, 1e-12))
    return sp.diags(dinv) @ W @ sp.diags(dinv), deg


def lp_run(support, cal, sf, bidx, k=10, alpha=0.9, gamma=3.0, iters=15, init="warm", tol=1e-6, max_iter=1000,
           within_batch=True):
    """Conformal label propagation over support (label 1), calibration and stream nodes (label 0).

    init="warm"      : v4. u keeps its values across batches, new nodes start at 0 (= y), `iters` sweeps.
    init="cold"      : every batch, all nodes restart at 0 and `iters` sweeps are run.
    init="converge"  : every batch, all nodes restart at 0; sweeps until ||u' - u|| <= tol ||u'|| (max_iter).
    within_batch=True  : batch b joins the graph before it is scored (images of the batch refer to each other).
    within_batch=False : each image of batch b is scored alone against the graph of batches < b:
                         u(x) = alpha * sum_j w_xj / sqrt(deg_x deg_j) u_j over its k nearest old nodes, with the
                         old nodes' u and degrees fixed; the batch then joins the graph as in v4.
    Returns p (conformal, small = OOD), u (ID mass at scoring time), n_sweeps per batch.
    """
    dim = sf.shape[1]
    sup = np.ascontiguousarray(np.asarray(support).reshape(-1, dim), dtype=np.float32)
    cal = np.ascontiguousarray(cal, dtype=np.float32)
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
    u = y.copy()
    p = np.empty(n_t)
    u_out = np.empty(n_t)
    sweeps = []
    cur = n_fix
    state_ready = False            # within_batch=False needs u on the old graph before the first batch
    for b in np.unique(bidx):
        rows = np.flatnonzero(bidx == b)
        assert rows[0] == cur - n_fix, "stream must be in batch order"
        new = np.arange(cur, cur + len(rows))
        sims = X[new] @ X[:cur + len(rows)].T
        sims[np.arange(len(rows)), new] = -np.inf
        if not within_batch:
            if not state_ready:    # graph of support + calibration only (v4 never propagates before batch 0)
                Wn0, _ = _normalised_graph(top_s, top_i, cur, k, gamma)
                uu = u[:cur]
                for _ in range(iters):
                    uu = alpha * (Wn0 @ uu) + (1 - alpha) * y[:cur]
                u[:cur] = uu
                state_ready = True
            _, deg_old = _normalised_graph(top_s, top_i, cur, k, gamma)
            old_s = sims[:, :cur]
            part = np.argpartition(-old_s, k - 1, axis=1)[:, :k]
            ws = np.maximum(np.take_along_axis(old_s, part, axis=1), 0.0).astype(np.float64) ** gamma
            deg_x = np.maximum(ws.sum(axis=1), 1e-12)
            ux = alpha * (ws / np.sqrt(deg_x[:, None] * np.maximum(deg_old[part], 1e-12)) * u[part]).sum(axis=1)
            fc = np.sort(u[n_s:n_fix])
            p[rows] = (1.0 + np.searchsorted(fc, ux, side="right")) / (n_c + 1.0)
            u_out[rows] = ux
        part = np.argpartition(-sims, k - 1, axis=1)[:, :k]
        top_s[new] = np.take_along_axis(sims, part, axis=1)
        top_i[new] = part
        old = sims[:, :cur].T
        top_s[:cur], top_i[:cur] = _topk_merge(top_s[:cur], top_i[:cur], old, np.broadcast_to(new, old.shape), k)
        cur += len(rows)
        n = cur
        Wn, _ = _normalised_graph(top_s, top_i, n, k, gamma)
        if init == "warm":
            uu = u[:n]
            for _ in range(iters):
                uu = alpha * (Wn @ uu) + (1 - alpha) * y[:n]
            n_sw = iters
        elif init == "cold":
            uu = np.zeros(n)
            for _ in range(iters):
                uu = alpha * (Wn @ uu) + (1 - alpha) * y[:n]
            n_sw = iters
        elif init == "converge":
            uu = np.zeros(n)
            n_sw = 0
            while n_sw < max_iter:
                nxt = alpha * (Wn @ uu) + (1 - alpha) * y[:n]
                n_sw += 1
                done = np.linalg.norm(nxt - uu) <= tol * max(np.linalg.norm(nxt), 1e-30)
                uu = nxt
                if done:
                    break
        else:
            raise ValueError(init)
        u[:n] = uu
        sweeps.append(n_sw)
        if within_batch:
            fc = np.sort(uu[n_s:n_fix])
            p[rows] = (1.0 + np.searchsorted(fc, uu[new], side="right")) / (n_c + 1.0)
            u_out[rows] = uu[new]
    return {"p": p, "u": u_out, "sweeps": np.asarray(sweeps)}


def memory_p_snapshot(sf, bidx, admit, cal_feats, d_cal, extra_feats, d_extra, med, mad, m, snap_batches):
    """Memory p-values of extra (never streamed, never admitted) images at fixed histories: at each batch b in
    snap_batches, against the memory available to batch b (admitted samples of batches < b) and the calibration
    shots' scores under the same memory. Returns {b: p_extra}."""
    sf = np.ascontiguousarray(sf, dtype=np.float32)
    cal = TopM(cal_feats, m)
    ext = TopM(extra_feats, m)
    snaps = set(int(b) for b in snap_batches)
    out = {}
    for b in np.unique(bidx):
        rows = np.flatnonzero(bidx == b)
        if int(b) in snaps:
            ge = d_extra - (ext.dist(m) - med) / mad
            gc = np.sort(d_cal - (cal.dist(m) - med) / mad)
            out[int(b)] = pvalues_sorted(gc, ge)
        new = sf[rows[admit[rows]]]
        if len(new):
            cal.add(new)
            ext.add(new)
    return out


def entrance_cal(sf, d, d_all, bidx, cal_sets, med, mad, thresholds=V4_ENTRANCE, m=2):
    """entrance() with stage-specific calibration: cal_sets[i] = (feats, d, d_all) of the calibration images of
    stage i (stage 0: p_all, stage 1..: p of the previous stage's set, last: the memory score p_t).
    Returns (stage masks, stage p-values, p_t)."""
    p_all = pval_high(cal_sets[0][2], d_all)
    masks, ps = [p_all <= thresholds[0]], [p_all]
    for i, t in enumerate(thresholds[1:], start=1):
        cf, dc, _ = cal_sets[i]
        p_c, _ = memory_p(sf, d, bidx, masks[-1], cf, dc, med, mad, m)
        ps.append(p_c)
        masks.append(p_c <= t)
    cf, dc, _ = cal_sets[len(thresholds)]
    p_t, _ = memory_p(sf, d, bidx, masks[-1], cf, dc, med, mad, m)
    return masks, ps, p_t


def lp_aligned(support, cal, sf, bidx, k=10, alpha=0.9, gamma=3.0, iters=15):
    """Diagnostic LP in which the calibration shots join the graph only at the batch they are compared with.
    Persistent graph: support + stream batches < b (warm state, new nodes at 0, as v4 without calibration nodes).
    At batch b the batch and the calibration shots are inserted together (both start at 0), kNN lists are
    recomputed for the union, `iters` sweeps are run, and p = rank of u(batch) among u(calibration). The persistent
    state is then updated with the batch alone (iters sweeps), and the calibration nodes are dropped."""
    dim = sf.shape[1]
    sup = np.ascontiguousarray(np.asarray(support).reshape(-1, dim), dtype=np.float32)
    cal = np.ascontiguousarray(cal, dtype=np.float32)
    sf = np.ascontiguousarray(sf, dtype=np.float32)
    n_s, n_c = len(sup), len(cal)
    P = sup.copy()                                   # persistent node features
    ps = P @ P.T
    np.fill_diagonal(ps, -np.inf)
    part = np.argpartition(-ps, k - 1, axis=1)[:, :k]
    top_s, top_i = np.take_along_axis(ps, part, axis=1), part
    y_p = np.zeros(len(P))
    y_p[:n_s] = 1.0
    u_p = y_p.copy()
    p = np.empty(len(sf))
    for b in np.unique(bidx):
        rows = np.flatnonzero(bidx == b)
        B = sf[rows]
        n_p, n_b = len(P), len(B)
        for with_cal in (True, False):
            Nw = np.concatenate([B, cal]) if with_cal else B
            n_w = len(Nw)
            s_pw = P @ Nw.T                                       # persistent x new
            s_ww = Nw @ Nw.T
            np.fill_diagonal(s_ww, -np.inf)
            ts, ti = _topk_merge(top_s, top_i, s_pw, np.broadcast_to(np.arange(n_p, n_p + n_w), s_pw.shape), k)
            s_new = np.concatenate([s_pw.T, s_ww], axis=1)
            part = np.argpartition(-s_new, k - 1, axis=1)[:, :k]
            ts = np.concatenate([ts, np.take_along_axis(s_new, part, axis=1)])
            ti = np.concatenate([ti, part])
            n = n_p + n_w
            Wn, _ = _normalised_graph(ts, ti, n, k, gamma)
            y = np.concatenate([y_p, np.zeros(n_w)])
            uu = np.concatenate([u_p, np.zeros(n_w)])
            for _ in range(iters):
                uu = alpha * (Wn @ uu) + (1 - alpha) * y
            if with_cal:
                fc = np.sort(uu[n_p + n_b:])
                p[rows] = (1.0 + np.searchsorted(fc, uu[n_p:n_p + n_b], side="right")) / (n_c + 1.0)
            else:
                P = np.concatenate([P, B])
                top_s, top_i, u_p = ts, ti, uu
                y_p = y
    return p


def proto_view(support, q, cand, is_cal, n0=12, m=2, mad_scale=1.4826):
    """iter3_eval.custom_view(proto=True) with a free shrinkage n0 (and candidate sets of any width).
    Returns d, d_all, p, p_all, cal_feats, d_cal and the memory scale (support LOO m-th NN statistics)."""
    from .dview import loo_stats

    q64 = np.asarray(q, dtype=np.float64)
    sup = np.asarray(support, dtype=np.float64)
    knn = loo_stats(sup, m, n0, mad_scale)
    tot = sup.sum(axis=1)
    mu = tot / np.linalg.norm(tot, axis=1, keepdims=True)
    loo_mu = tot[:, None, :] - sup
    loo_mu /= np.linalg.norm(loo_mu, axis=2, keepdims=True)
    loo = 1.0 - np.einsum("cnd,cnd->cn", sup, loo_mu)
    n = loo.shape[1]
    med_c = np.median(loo, axis=1)
    mad_c = mad_scale * np.median(np.abs(loo - med_c[:, None]), axis=1)
    med_all = float(np.median(loo))
    mad_all = float(mad_scale * np.median(np.abs(loo - med_all)))
    med_t = (n * med_c + n0 * med_all) / (n + n0)
    mad_t = (n * mad_c + n0 * mad_all) / (n + n0)
    z = ((1.0 - q64 @ mu.T) - med_t[None, :]) / mad_t[None, :]
    d = np.take_along_axis(z, np.asarray(cand), axis=1).min(axis=1)
    d_all = z.min(axis=1)
    return {"d": d, "d_all": d_all, "p": pval_high(d[is_cal], d), "p_all": pval_high(d_all[is_cal], d_all),
            "cal_feats": np.asarray(q)[is_cal].astype(np.float32), "d_cal": d[is_cal], "d_all_cal": d_all[is_cal],
            "stats": {"med_all": knn["med_all"], "mad_all": knn["mad_all"]}}


def lp_oracle(support, cal, sf, bidx, persist, k=10, alpha=0.9, gamma=3.0, iters=15):
    """v4 LP in which only the stream rows with persist=True join the graph (transductively, as in v4); the other rows
    are inserted only when they are scored (per-image, against the graph after the batch's persisted rows joined).
    With persist all True this equals lp_run(init='warm', within_batch=True). Diagnostic (persist may use labels)."""
    dim = sf.shape[1]
    sup = np.ascontiguousarray(np.asarray(support).reshape(-1, dim), dtype=np.float32)
    cal = np.ascontiguousarray(cal, dtype=np.float32)
    sf = np.ascontiguousarray(sf, dtype=np.float32)
    n_s, n_c = len(sup), len(cal)
    keep_rows = np.flatnonzero(persist)
    X = np.concatenate([sup, cal, sf[keep_rows]])
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
    u = y.copy()
    p = np.empty(len(sf))
    cur = n_fix
    kept_before = 0
    for b in np.unique(bidx):
        rows = np.flatnonzero(bidx == b)
        krows = rows[persist[rows]]
        new = np.arange(cur, cur + len(krows))
        if len(krows):
            sims = X[new] @ X[:cur + len(krows)].T
            sims[np.arange(len(krows)), new] = -np.inf
            part = np.argpartition(-sims, k - 1, axis=1)[:, :k]
            top_s[new] = np.take_along_axis(sims, part, axis=1)
            top_i[new] = part
            old = sims[:, :cur].T
            top_s[:cur], top_i[:cur] = _topk_merge(top_s[:cur], top_i[:cur], old, np.broadcast_to(new, old.shape), k)
            cur += len(krows)
        n = cur
        Wn, deg = _normalised_graph(top_s, top_i, n, k, gamma)
        uu = u[:n]
        for _ in range(iters):
            uu = alpha * (Wn @ uu) + (1 - alpha) * y[:n]
        u[:n] = uu
        fc = np.sort(uu[n_s:n_fix])
        if len(krows):
            p[krows] = (1.0 + np.searchsorted(fc, uu[new], side="right")) / (n_c + 1.0)
        other = rows[~persist[rows]]
        if len(other):
            s_o = sf[other] @ X[:n].T
            part = np.argpartition(-s_o, k - 1, axis=1)[:, :k]
            ws = np.maximum(np.take_along_axis(s_o, part, axis=1), 0.0).astype(np.float64) ** gamma
            deg_x = np.maximum(ws.sum(axis=1), 1e-12)
            ux = alpha * (ws / np.sqrt(deg_x[:, None] * np.maximum(deg[part], 1e-12)) * uu[part]).sum(axis=1)
            p[other] = (1.0 + np.searchsorted(fc, ux, side="right")) / (n_c + 1.0)
        kept_before += len(krows)
    return p


# ----------------------------------------------------------------------------------------------------------
# same-feature static baselines
# ----------------------------------------------------------------------------------------------------------
def knn_distance(support_flat, queries, ks, chunk=4096):
    """k-th nearest cosine distance to the pooled support (Sun et al. 2022 with normalised features).
    Returns {k: (N,) distances} for every k in ks (OOD direction)."""
    S = np.ascontiguousarray(support_flat, dtype=np.float32)
    Q = np.ascontiguousarray(queries, dtype=np.float32)
    kmax = max(ks)
    out = {k: np.empty(len(Q)) for k in ks}
    for lo in range(0, len(Q), chunk):
        sims = Q[lo:lo + chunk] @ S.T
        top = -np.partition(-sims, kmax - 1, axis=1)[:, :kmax]
        top = -np.sort(-top, axis=1)
        for k in ks:
            out[k][lo:lo + chunk] = 1.0 - top[:, k - 1]
    return out


def maha_pp(support, queries, lams, chunk=4096):
    """Mahalanobis++ (Mueller & Hein 2025): L2-normalised features, class means of the support and one shared
    covariance of the support residuals, shrunk as (1 - lam) S + lam tr(S)/D I. Score = min_c Mahalanobis distance
    (OOD direction). support: (C, n, D). Returns {lam: (N,)}."""
    Sp = np.asarray(support, dtype=np.float64)
    Sp = Sp / np.linalg.norm(Sp, axis=2, keepdims=True)
    n_cls, n, dim = Sp.shape
    mu = Sp.mean(axis=1)
    R = (Sp - mu[:, None, :]).reshape(-1, dim)
    cov = R.T @ R / (len(R) - n_cls)
    Q = np.asarray(queries, dtype=np.float64)
    Q = Q / np.linalg.norm(Q, axis=1, keepdims=True)
    out = {}
    for lam in lams:
        c = (1 - lam) * cov + lam * np.trace(cov) / dim * np.eye(dim)
        evals, evecs = np.linalg.eigh(c)
        evals = np.maximum(evals, 1e-10 * evals.max())
        Wm = evecs / np.sqrt(evals)                       # x -> x W gives whitened coordinates
        muw = mu @ Wm
        mun = (muw ** 2).sum(axis=1)
        res = np.empty(len(Q))
        for lo in range(0, len(Q), chunk):
            qw = Q[lo:lo + chunk] @ Wm
            dist = (qw ** 2).sum(axis=1)[:, None] - 2 * qw @ muw.T + mun[None, :]
            res[lo:lo + chunk] = dist.min(axis=1)
        out[lam] = res
    return out


# ----------------------------------------------------------------------------------------------------------
# metrics (labels are used here only)
# ----------------------------------------------------------------------------------------------------------
def auroc(id_s, ood_s):
    """P(score_ID > score_OOD) + 0.5 P(tie) (ID positive, larger = more ID)."""
    from scipy.stats import rankdata

    id_s, ood_s = np.asarray(id_s, dtype=np.float64), np.asarray(ood_s, dtype=np.float64)
    r = rankdata(np.concatenate([id_s, ood_s]))
    m, n = len(id_s), len(ood_s)
    return float((r[:m].sum() - m * (m + 1) / 2) / (m * n))


def fpr_at_tpr(id_s, ood_s, tpr=0.95):
    """Fraction of OOD with score >= the threshold that keeps `tpr` of ID (ties counted as accepted)."""
    id_s = np.sort(np.asarray(id_s, dtype=np.float64))
    thr = id_s[int(math.floor((1 - tpr) * len(id_s)))]
    return float((np.asarray(ood_s) >= thr).mean())


def within_batch_auroc(score, is_ood, bidx):
    """Pooled ID-vs-OOD concordance restricted to pairs from the same batch (ties count 1/2)."""
    from scipy.stats import rankdata

    num = den = 0.0
    for b in np.unique(bidx):
        rows = np.flatnonzero(bidx == b)
        o = is_ood[rows]
        m, n = int((~o).sum()), int(o.sum())
        if m == 0 or n == 0:
            continue
        r = rankdata(score[rows])
        num += r[~o].sum() - m * (m + 1) / 2
        den += m * n
    return float(num / den) if den else float("nan")


def t_interval(values, level=0.95):
    """Mean, sd (ddof=1) and the t confidence interval of the mean over independent units."""
    from scipy.stats import t as student_t

    v = np.asarray(values, dtype=np.float64)
    n = len(v)
    mean = float(v.mean())
    sd = float(v.std(ddof=1)) if n > 1 else float("nan")
    half = float(student_t.ppf(0.5 + level / 2, n - 1) * sd / math.sqrt(n)) if n > 1 else float("nan")
    return {"mean": mean, "sd": sd, "lo": mean - half, "hi": mean + half, "n": n,
            "n_positive": int((v > 0).sum()), "n_negative": int((v < 0).sum())}
