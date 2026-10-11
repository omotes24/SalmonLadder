"""Salmon Ladder Phase 15 (paper v18): (1) baselines that use pre-trained features with far more labels on ImageNet-200 and
CIFAR-100 (kNN and class means over ALL training images of the same two backbones; zero-shot MCM and NegLabel with CLIP ViT-B/16);
(2) the recurrence grid with the graph's neighbour count k_g varied; (3) a seed rule whose threshold is capped at a fixed level
tau (the expected number of ID images among the seeds is at most tau m0) evaluated once, without selection, on the development
splits, the benchmarks and the grid; (4) centred static / kNN read-outs for the six single views. Shared paths and the capped rules."""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "/home/omote/reprise_p14_20261011/code")
import p14common as _M14  # noqa: E402  (Phase 14 -> 13 -> 12 -> 11/10 paths, Phase 5 code, vendored vins package)
from p14common import *  # noqa: E402,F401,F403  (also brings P14 = the Phase 14 directory and P = Phase 13's namespace)
M14 = _M14                # the Phase 14 module itself (the star import must not shadow it)
P = _M14.P                # Phase 13's namespace (storey_adadetect, plain_bh, load_p3, add_views, l2n, utc, ...)
_SELECT14 = _M14.select_seeds

P15 = Path("/home/omote/reprise_p15_20261011")
RESULTS15 = P15 / "results"
LOGS15 = P15 / "logs"
TAU = 0.05                                   # the cap: fixed in advance, never selected (prereg_p15.json)
CAP_RULES = ("storeyc", "bhc")               # Storey / plain BH step-up with critical values capped at tau
KGS = (3, 5, 10, 20, 40)                     # the k_g grid (10 = the paper's value)


def capped_stepup(p, level, tau):
    """Step-up with critical values c_i = min(level * i / m, tau) (sorted p-values): rejects the first i* order statistics,
    i* = max{i : p_(i) <= c_i}. Every rejected p satisfies p <= level * i*/m (self-consistency w.r.t. the linear threshold
    collection) and p <= tau. Returns the boolean mask."""
    p = np.asarray(p, np.float64)
    m = len(p)
    order = np.argsort(p)
    ps = p[order]
    thr = np.minimum(level * np.arange(1, m + 1) / m, tau)
    ok = np.flatnonzero(ps <= thr)
    mask = np.zeros(m, bool)
    if len(ok):
        mask[order[: ok[-1] + 1]] = True
    return mask


def select_seeds(p, q, n_cal, rule):
    """Phase 14 rules ('storey', 'bh') plus the capped rules: 'bhc' = plain BH level q with the cap tau (FDR <= q m0/m by
    self-consistency under PRDS, Blanchard & Roquain 2008, and E[#ID seeds] <= tau m0); 'storeyc' = the Storey level q / pi0_hat
    with the same cap (E[#ID seeds] <= tau m0; the FDR statement of the uncapped Storey rule is not claimed for the capped set)."""
    if rule in ("storey", "bh"):
        return _SELECT14(p, q, n_cal, rule)
    if rule == "bhc":
        return capped_stepup(p, q, TAU), 1.0
    if rule == "storeyc":
        p = np.asarray(p, np.float64)
        lam = P.storey_lambda(n_cal)
        pi0 = (1.0 + np.sum(p >= lam)) / (len(p) * (1.0 - lam))
        return capped_stepup(p, q / pi0, TAU), float(pi0)
    raise ValueError(rule)


def install():
    """Routes core14's seed selection through the Phase 15 rules (core14 calls P.select_seeds on the p14common module)."""
    _M14.select_seeds = select_seeds


def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=1, default=str) + "\n")
