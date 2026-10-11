"""Salmon Ladder Phase 12 (paper v14): selection on the development splits, draws of the labelled images, recurrence and
mixing-rate grid, same-feature baselines, and the arrival-time (online) mode. Shared paths and helpers."""
import json
import sys
from pathlib import Path

sys.path.insert(0, "/home/omote/reprise_p11_20261010/code")
import p11common as P  # noqa: E402  (Phase 10/11 paths, Phase 5 code on sys.path, vendored vins package)
from p11common import *  # noqa: E402,F401,F403

P12 = Path("/home/omote/reprise_p12_20261011")
RESULTS12 = P12 / "results"
LOGS12 = P12 / "logs"
P4 = Path("/home/omote/reprise_p4_20260928")
P7 = Path("/home/omote/reprise_p7_20261003")
KG, LAM, Q_MAIN = 10, 0.9, 0.1
QS = (0.05, 0.1, 0.2)


def storey_lambda(n_cal):
    """Storey-AdaDetect: lambda = K / (l + 1) with K = floor(l / 2) (Marandon et al. 2024, Cor. 3.7); 2000/4001 for l = 4,000."""
    return (n_cal // 2) / (n_cal + 1)


def storey_adadetect(p, q, n_cal):
    """Storey-AdaDetect (Marandon et al. 2024): BH at level q / pi0_hat with pi0_hat = (1 + #{p >= lam}) / (m (1 - lam)).
    Returns the boolean mask of the rejected (seed) p-values and pi0_hat (not capped at 1: the corollary's estimator)."""
    import numpy as np
    p = np.asarray(p, np.float64)
    m = len(p)
    lam = storey_lambda(n_cal)
    pi0 = (1.0 + np.sum(p >= lam)) / (m * (1.0 - lam))
    order = np.argsort(p)
    ps = p[order]
    thr = (q / pi0) * np.arange(1, m + 1) / m
    ok = np.flatnonzero(ps <= thr)
    mask = np.zeros(m, bool)
    if len(ok):
        mask[order[: ok[-1] + 1]] = True
    return mask, float(pi0)


def plain_bh(p, q):
    import numpy as np
    p = np.asarray(p, np.float64)
    m = len(p)
    order = np.argsort(p)
    ps = p[order]
    ok = np.flatnonzero(ps <= q * np.arange(1, m + 1) / m)
    mask = np.zeros(m, bool)
    if len(ok):
        mask[order[: ok[-1] + 1]] = True
    return mask


def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=1, default=str) + "\n")
