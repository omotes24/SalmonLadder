import numpy as np
from scipy.stats import rankdata, t
from sklearn.metrics import roc_auc_score


def metrics(score, is_ood):
    """ID-high log-score; FPR95 at the empirical threshold accepting >= 95% of ID (ties accepted), as the audit."""
    score = np.asarray(score, np.float64)
    is_ood = np.asarray(is_ood, bool)
    if np.isnan(score).any() or np.isposinf(score).any():
        raise ValueError("invalid score")
    id_s, ood_s = score[~is_ood], score[is_ood]
    naccept = int(np.ceil(0.95 * len(id_s)))
    thr = np.sort(id_s)[len(id_s) - naccept]
    return {"AUROC": 100 * roc_auc_score(~is_ood, rankdata(score)), "FPR95": 100 * np.mean(ood_s >= thr)}


def tci(values):
    x = np.asarray(values, float)
    n = len(x)
    m = float(np.mean(x))
    h = float(t.ppf(0.975, n - 1) * np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else float("nan")
    return {"mean": m, "lo": m - h, "hi": m + h, "n": n}
