"""Theory checks on dev1 (already used; diagnostics only):
(T1) mass identity of symmetric-normalised LP: with v = D^{1/2} 1, S v = v, so the converged solution satisfies
     sum_i sqrt(d_i) u_i = sum_{j in support} sqrt(d_j) exactly; we track both sides, the mean u and N_s / N_t per batch
     for the converged (L2) and the 15-sweep solvers (L0 warm, L1 zero-start).
(T2) common-monotone-transform test: Spearman correlation between the calibration images' u at batch t and at the last
     batch (a purely common monotone drift keeps it at 1), from the audit's saved calibration traces."""
import json

import numpy as np
import torch
from scipy.stats import spearmanr

from common import AUDIT, RESULTS, V5, dump, utc
from engine import PrefixTopK, propagate, propagate_converged
from pilot import load_development, prepare


def t1(draw=0, stream="near", seed=123, view="L14"):
    R, D = load_development()
    su, ca, sf, bi, vv, sid, flag, S = prepare(R, D, draw, view, stream, seed)
    sup = su.reshape(-1, su.shape[-1])
    ns, nc = len(sup), len(ca)
    g = PrefixTopK(np.concatenate([sup, ca]), device="cuda")
    warm = None
    rows = []
    for b in np.unique(bi):
        r = np.flatnonzero(bi == b)
        g.append(sf[r])
        n = len(g)
        M = g.matrix(10, 1.0, "sym")
        # degrees of the symmetrised graph (same construction as matrix())
        k = 10
        rr = torch.arange(n, device="cuda").repeat_interleave(k)
        cc = g.I[:, :k].reshape(-1)
        vals = g.S[:, :k].clamp_min(0).reshape(-1)
        keys = torch.cat([rr * n + cc, cc * n + rr])
        uniq, inv = torch.unique(keys, return_inverse=True)
        w = torch.zeros(len(uniq), device="cuda").scatter_reduce(0, inv, torch.cat([vals, vals]), reduce="amax", include_self=False)
        deg = torch.zeros(n, device="cuda").index_add_(0, uniq // n, w).double()
        sq = torch.sqrt(deg)
        y = torch.zeros(n, dtype=torch.float64, device="cuda")
        y[:ns] = 1
        lam = torch.tensor([0.9], dtype=torch.float64, device="cuda")
        u2, it, res = propagate_converged(M, y[:, None], lam)
        w0 = y.clone() if warm is None else torch.cat([warm, torch.zeros(n - len(warm), dtype=torch.float64, device="cuda")])
        u0 = propagate(M, y[:, None], lam, w0[:, None])
        warm = u0[:, 0].clone()
        u1 = propagate(M, y[:, None], lam, torch.zeros((n, 1), dtype=torch.float64, device="cuda"))
        rhs = float((sq[:ns]).sum())
        rows.append({"batch": int(b), "N_t": n, "Ns_over_Nt": ns / n,
                     "mass_L2": float((sq * u2[:, 0]).sum()) / rhs, "mass_L0": float((sq * u0[:, 0]).sum()) / rhs,
                     "mass_L1": float((sq * u1[:, 0]).sum()) / rhs,
                     "mean_u_L2": float(u2[:, 0].mean()), "mean_u_unlabelled_L2": float(u2[ns:, 0].mean()),
                     "mean_sqrtdeg_support": float(sq[:ns].mean()), "mean_sqrtdeg_all": float(sq.mean()),
                     "cal_median_L2": float(u2[ns:ns + nc, 0].median()), "iters_L2": it, "res_L2": res})
    return rows


def t2():
    out = {}
    for stream in ("near", "far"):
        for view in ("B14", "L14"):
            for solver in ("L0", "L1", "L2"):
                rhos_last, rhos_next = [], []
                for draw in range(5):
                    for seed in (123, 124, 125):
                        z = np.load(AUDIT / f"results/primary/draw{draw}_{stream}_seed{seed}_calibration.npz")
                        U = z[f"{view}_{solver}"]
                        last = U[-1]
                        rl = [spearmanr(U[t], last).correlation for t in range(len(U))]
                        rn = [spearmanr(U[t], U[t + 1]).correlation for t in range(len(U) - 1)]
                        rhos_last.append(rl)
                        rhos_next.append(rn)
                L = np.array([r[:78] for r in rhos_last])
                out[f"{stream}|{view}|{solver}"] = {"rho_first_vs_last_min": float(L[:, 0].min()), "rho_first_vs_last_mean": float(L[:, 0].mean()),
                                                     "rho_consecutive_min": float(np.min([min(r) for r in rhos_next])),
                                                     "rho_consecutive_mean": float(np.mean([np.mean(r) for r in rhos_next]))}
    return out


def t3():
    """(T3) rank-error bound: for calibration image c, leave-one-out rank p_t(c) among the other calibration images.
    |p_t(c) - p_t'(c)| <= D_c / (n - 1) with D_c the calibration images whose order with c flips, hence
    mean_c |dp| <= fraction of discordant pairs = (1 - tau_a) / 2. Reported for consecutive batches and first vs last."""
    from scipy.stats import kendalltau
    out = {}
    for stream in ("near", "far"):
        for view in ("B14", "L14"):
            for solver in ("L0", "L1", "L2"):
                cons, fl = [], []
                for draw in range(5):
                    for seed in (123, 124, 125):
                        z = np.load(AUDIT / f"results/primary/draw{draw}_{stream}_seed{seed}_calibration.npz")
                        U = z[f"{view}_{solver}"]
                        n = U.shape[1]
                        R = np.argsort(np.argsort(U, axis=1, kind="stable"), axis=1) / (n - 1)
                        for t in range(len(U) - 1):
                            tau = kendalltau(U[t], U[t + 1]).correlation
                            cons.append((np.abs(R[t] - R[t + 1]).mean(), (1 - tau) / 2))
                        tau = kendalltau(U[0], U[-1]).correlation
                        fl.append((np.abs(R[0] - R[-1]).mean(), (1 - tau) / 2))
                cons, fl = np.array(cons), np.array(fl)
                out[f"{stream}|{view}|{solver}"] = {"consecutive_mean_abs_dp": float(cons[:, 0].mean()), "consecutive_bound": float(cons[:, 1].mean()),
                                                     "consecutive_max_abs_dp": float(cons[:, 0].max()),
                                                     "first_last_mean_abs_dp": float(fl[:, 0].mean()), "first_last_bound": float(fl[:, 1].mean())}
    return out


if __name__ == "__main__":
    import sys
    if "--t3" in sys.argv:
        r3 = {"utc": utc(), "T3": t3()}
        dump(RESULTS / "theory_t3.json", r3)
        print(json.dumps(r3, indent=1))
    else:
        res = {"utc": utc(), "T1": t1(), "T2": t2()}
        dump(RESULTS / "theory_checks.json", res)
        T1 = res["T1"]
        print(json.dumps({"T1_first": T1[0], "T1_last": T1[-1], "T2": res["T2"]}, indent=1))
