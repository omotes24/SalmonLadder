"""Step 2: frozen class-conditional visual view d(x) and split-conformal calibration (pure numpy).

r_c(x)  = m-th smallest cosine distance (1 - cos) between x and the support set S_c
LOO     = for each s_j in S_c, m-th smallest distance to S_c minus s_j
med_c, MAD_c (x1.4826) from the LOO values of class c; med_all, MAD_all from all classes pooled
med~_c  = (n*med_c + n0*med_all)/(n+n0), MAD~_c likewise
z_c(x)  = (r_c(x) - med~_c)/MAD~_c,  d(x) = min_{c in K(x)} z_c(x)   (OOD direction)
q_eps   = ceil((n+1)(1-eps))-th smallest calibration d;  nominated = d > q_eps
"""
import math
from fractions import Fraction

import numpy as np


def mth_smallest(values, m):
    return np.partition(values, m - 1, axis=-1)[..., m - 1]


def loo_stats(support, m, n0, mad_scale=1.4826):
    """support: (C, n, D) L2-normalised features."""
    support = np.asarray(support, dtype=np.float64)
    n_cls, n, _ = support.shape
    dist = 1.0 - np.einsum("cid,cjd->cij", support, support)
    diag = np.arange(n)
    dist[:, diag, diag] = np.inf
    loo = mth_smallest(dist, m)                                  # (C, n)
    med_c = np.median(loo, axis=1)
    mad_c = mad_scale * np.median(np.abs(loo - med_c[:, None]), axis=1)
    med_all = float(np.median(loo))
    mad_all = float(mad_scale * np.median(np.abs(loo - med_all)))
    med_t = (n * med_c + n0 * med_all) / (n + n0)
    mad_t = (n * mad_c + n0 * mad_all) / (n + n0)
    return {"loo": loo, "med_c": med_c, "mad_c": mad_c, "med_all": med_all, "mad_all": mad_all,
            "med_t": med_t, "mad_t": mad_t, "n": n, "n0": n0, "m": m}


def d_scores(queries, cand, support, stats, m, chunk=512):
    """queries: (N, D); cand: (N, k) class indices; support: (C, n, D). Returns d, z, r, argmin class."""
    queries = np.asarray(queries, dtype=np.float64)
    support = np.asarray(support, dtype=np.float64)
    cand = np.asarray(cand)
    n_q, k = cand.shape
    r = np.empty((n_q, k))
    for lo in range(0, n_q, chunk):
        hi = min(lo + chunk, n_q)
        sims = np.einsum("bd,bknd->bkn", queries[lo:hi], support[cand[lo:hi]])
        r[lo:hi] = mth_smallest(1.0 - sims, m)
    z = (r - stats["med_t"][cand]) / stats["mad_t"][cand]
    arg = z.argmin(axis=1)
    rows = np.arange(n_q)
    return z[rows, arg], z, r, cand[rows, arg]


def conformal_rank(n, eps):
    """1-based rank ceil((n+1)(1-eps)) computed in exact rational arithmetic."""
    return math.ceil((n + 1) * (1 - Fraction(str(eps))))


def conformal_threshold(cal_scores, eps):
    cal = np.sort(np.asarray(cal_scores, dtype=np.float64))
    rank = conformal_rank(len(cal), eps)
    return float("inf") if rank > len(cal) else float(cal[rank - 1])


def nominate(d, threshold):
    return np.asarray(d) > threshold
