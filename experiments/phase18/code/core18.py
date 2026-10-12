"""Phase 18 core (descriptive): the read-outs of one view's graph, all read as ranks among the reference images R.
Statistics are returned for the pool U = C u T (calibration images first, then the test images), as in core17.sigma_ref, so
that p_fin = rank among C can be formed for any of them; AUROC / FPR95 are computed on the test positions only."""
import time

import numpy as np
import torch

import p18common as P
import core14 as K
import core16 as H
import core17 as X
from engine import _log, p_high, p_low
from an11 import measures


def sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


class Timer:
    """Accumulates wall-clock seconds per key (CUDA synchronised at both ends)."""

    def __init__(self):
        self.t = {}

    def __call__(self, key):
        tm = self

        class _T:
            def __enter__(self_):
                sync()
                self_.t0 = time.time()

            def __exit__(self_, *a):
                sync()
                tm.t[key] = tm.t.get(key, 0.0) + time.time() - self_.t0
        return _T()


def nodes(g):
    """Reference R, calibration C, test T and the pool U = C u T (node indices)."""
    R, C = H.split_cal(g.nc, g.ns, P.N_REF, P.N_CAL_PER_CLASS)
    T = g.nfix + np.arange(g.nt)
    return R, C, T, np.concatenate([C, T])


def neg_mass(g, seeds, k, lam):
    """u~- = lambda S u- of the seed mass (a node's own injected mass removed), as core17.sigma_ref; zeros without seeds."""
    if len(seeds) == 0:
        return np.zeros(g.N)
    un = g.mass(seeds, k=k, lam=lam).copy()
    un[seeds] -= (1.0 - lam)
    return np.maximum(un, 0.0)


def seeds_of(u, R, pool, q=P.Q18):
    """Storey-BH seeds (uncapped pi0_hat) on the rank of u among R over the pool, exactly as core17.sigma_ref."""
    pR = p_low(u[R], u[pool])
    mask, pi0 = P.M15.select_seeds(pR, q, len(R), "storey")
    return pR, pool[mask], float(pi0)


@torch.no_grad()
def one_step(g, k=P.KG18):
    """(D^-1 W y+)_i = sum_{j in S} W_ij / delta_i (W = max(A, A^T) of the top-k lists, the arithmetic of the propagation matrix)."""
    W, deg = H.sym_W(g.graph, k)
    y = torch.zeros((g.N, 1), dtype=torch.float64, device=g.dev)
    y[:g.ns, 0] = 1.0
    out = (torch.sparse.mm(W, y)[:, 0] / deg.clamp_min(1e-300)).cpu().numpy()
    del W, deg
    return out


@torch.no_grad()
def kth_sim(sup, Xq, k=P.KNN_K, dev="cuda"):
    """Cosine similarity to the k-th nearest support image (as core12.knn_rank, without the rank)."""
    S = torch.as_tensor(np.asarray(sup, np.float32), device=dev)
    out = []
    for lo in range(0, len(Xq), 4096):
        x = torch.as_tensor(np.asarray(Xq[lo:lo + 4096], np.float32), device=dev)
        out.append((x @ S.T).topk(k, dim=1).values[:, -1])
    return torch.cat(out).double().cpu().numpy()


def seed_info(seeds, C, nfix, flag):
    """Number of seeds, of which calibration images, and the ID fraction (calibration images count as ID)."""
    if len(seeds) == 0:
        return {"seeds": 0, "cal_seeds": 0, "idfrac": None}
    is_id = np.array([(s < nfix) or (not flag[s - nfix]) for s in seeds])
    return {"seeds": int(len(seeds)), "cal_seeds": int(np.isin(seeds, C).sum()), "idfrac": float(is_id.mean())}


def lp_sref(g, R, C, pool, flag, k, lam, timer=None, tag=""):
    """log p+ and log p- (pool positions) of one view for (k_g, lambda), with the seed statistics."""
    tm = timer or Timer()
    with tm(f"u_pos{tag}"):
        u = g.mass(np.arange(g.ns), k=k, lam=lam)
    with tm(f"seeds{tag}"):
        pR, seeds, pi0 = seeds_of(u, R, pool)
    with tm(f"u_neg{tag}"):
        un = neg_mass(g, seeds, k, lam)
    with tm(f"read{tag}"):
        lp = _log(pR)
        ln = _log(p_high(un[R], un[pool])) if len(seeds) else np.zeros(len(pool))
    return lp, ln, {**seed_info(seeds, C, g.nfix, flag), "pi0_hat": pi0}, u


def baselines(g, sup, cal, sf, R, pool, labels_sup, n_classes, dev, per_class, timer=None, with_lgc=True):
    """Same-feature read-outs of one view, read as log ranks among R over the pool."""
    tm = timer or Timer()
    out = {}
    ns = g.ns
    with tm("onestep"):
        u1 = one_step(g)
    out["onestep"] = _log(p_low(u1[R], u1[pool]))
    with tm("static"):
        sc, st = K.proto_scores(sup, cal, sf, dev, per_class=per_class)
    a = np.concatenate([sc, st])                      # positions: node index - ns
    out["static"] = _log(p_low(a[R - ns], a[pool - ns]))
    with tm("knn"):
        a = np.concatenate([kth_sim(sup, cal, dev=dev), kth_sim(sup, sf, dev=dev)])
    out["knn"] = _log(p_low(a[R - ns], a[pool - ns]))
    if with_lgc:
        with tm("lgc"):
            sl = K.lgc_max(g, labels_sup, n_classes)
        out["lgc"] = _log(p_low(sl[R], sl[pool]))
    return out


def eval_rows(tag, name, x_pool, nC, flag, bases, extra=None):
    """Rows of one statistic over the pool: standalone and x every base (log base added), on the test positions."""
    x = np.asarray(x_pool, np.float64)[nC:]
    rows = []
    au, fp = measures(x, flag)
    rows.append({**tag, "cfg": name, "base": "none", "seed": -1, "AUROC": au, "FPR95": fp, **(extra or {})})
    for (b, seed), lb in bases.items():
        au, fp = measures(x + lb, flag)
        rows.append({**tag, "cfg": name, "base": b, "seed": int(seed), "AUROC": au, "FPR95": fp})
    return rows


def pfin_rows(tag, name, s_pool, nC, flag, bases):
    """p_fin = rank of the statistic among C (no folds); its validity and its log as a score (standalone and x bases)."""
    s = np.asarray(s_pool, np.float64)
    pfin = p_low(s[:nC], s[nC:])
    return eval_rows(tag, name, np.r_[np.zeros(nC), _log(pfin)], nC, flag, bases, X.validity(pfin, flag)), pfin
