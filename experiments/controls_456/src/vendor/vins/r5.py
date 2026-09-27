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

from .memory import memory_distances, online_pvalues, pvalues_sorted

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
