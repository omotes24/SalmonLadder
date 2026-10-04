"""AUROC, paired DeLong test and class-level bootstrap (numpy / scipy only). ID is the positive class."""
import numpy as np
from scipy.stats import norm, rankdata


def auroc(id_s, ood_s):
    id_s, ood_s = np.asarray(id_s, float), np.asarray(ood_s, float)
    ranks = rankdata(np.concatenate([id_s, ood_s]))
    m, n = len(id_s), len(ood_s)
    return float((ranks[:m].sum() - m * (m + 1) / 2) / (m * n))


def fpr95(id_s, ood_s):
    """FPR of OOD at the threshold that keeps 95% of ID (ID scores >= threshold are accepted)."""
    thr = np.percentile(np.asarray(id_s, float), 5)
    return float(np.mean(np.asarray(ood_s, float) >= thr))


def _fast_delong(preds, m):
    """preds: (k, N) with the first m columns positives. Returns aucs (k,) and covariance (k, k). Sun & Xu (2014)."""
    k, n_all = preds.shape
    n = n_all - m
    tx = np.vstack([rankdata(p[:m]) for p in preds])
    ty = np.vstack([rankdata(p[m:]) for p in preds])
    tz = np.vstack([rankdata(p) for p in preds])
    aucs = tz[:, :m].sum(axis=1) / m / n - (m + 1.0) / 2.0 / n
    v01 = (tz[:, :m] - tx) / n
    v10 = 1.0 - (tz[:, m:] - ty) / m
    cov = np.atleast_2d(np.cov(v01)) / m + np.atleast_2d(np.cov(v10)) / n
    return aucs, cov


def delong_paired(id_a, ood_a, id_b, ood_b):
    """Two scores on the same ID / OOD samples. Returns auc_a, auc_b, z, two-sided p."""
    m = len(id_a)
    preds = np.vstack([np.concatenate([id_a, ood_a]), np.concatenate([id_b, ood_b])]).astype(float)
    aucs, cov = _fast_delong(preds, m)
    var = cov[0, 0] + cov[1, 1] - 2 * cov[0, 1]
    z = (aucs[0] - aucs[1]) / np.sqrt(max(var, 1e-300))
    return float(aucs[0]), float(aucs[1]), float(z), float(2 * norm.sf(abs(z)))


def cluster_resample(groups, rng):
    """Indices of a cluster bootstrap: draw clusters with replacement, keep all members of each draw."""
    uniq, inv = np.unique(groups, return_inverse=True)
    members = [np.flatnonzero(inv == g) for g in range(len(uniq))]
    draw = rng.integers(0, len(uniq), len(uniq))
    return np.concatenate([members[g] for g in draw])


def precompute_members(groups):
    uniq, inv = np.unique(groups, return_inverse=True)
    order = np.argsort(inv, kind="stable")
    counts = np.bincount(inv, minlength=len(uniq))
    starts = np.concatenate([[0], np.cumsum(counts)[:-1]])
    return order, starts, counts


def draw_members(pre, rng):
    order, starts, counts = pre
    draw = rng.integers(0, len(counts), len(counts))
    return np.concatenate([order[starts[g]:starts[g] + counts[g]] for g in draw])
