"""Phase 16 core (registered configuration, prereg_p16.json): the one-quantity read-out h.

Graph: per view the symmetric k_g-NN graph of Phases 12-15 (W = max(A, A^T), weights = clamped cosine, k_g = 10); each view's
weights are divided by that view's median degree and the views are summed (one graph, one random walk). The continuous-time
random walk with jump rates W_ij is killed with probability 1 - lambda at every jump and at rate gamma per unit time,
gamma = GREL16 * median degree of the summed graph. Support images S (all 12 per class) are absorbing with value 1.

Labelled split (fixed by position, data independent): of the 4 calibration images of each class (stored class-major order) the
first 2 are the reference R, the last 2 the calibration set C.
  stage 1: h0 = P(reach S before vanishing)                        (no seeds)
  seeds  : Storey-BH (level q, uncapped pi0_hat, lambda_S = floor(|R|/2)/(|R|+1)) on p_R(u) = rank of h0(u) among h0(R), over the
           pool U = C u T (calibration AND test images, so that the seeds are a symmetric function of the unordered pool)
  stage 2: h = P(reach S before a seed and before vanishing)          (seeds absorbing with value 0)
  score  : s(x) = lambda (W h)(x) / (delta_x + gamma) for every unlabelled node (= h off the seeds; one-step look-ahead on seeds)
  p(x)   = (1 + #{c in C : s(c) <= s(x)}) / (|C| + 1)                 (conformal p-value: AdaDetect Thm 3.3, no folds)
The linear systems (D + gamma I - lambda W)_FF h_F = lambda W_F,S 1 are solved by Jacobi-preconditioned conjugate gradients."""
import numpy as np
import torch

import p16common as P
from engine import p_low
from an11 import measures

ALPHAS = (0.01, 0.05, 0.1, 0.2)


def sym_W(graph, k=P.KG):
    """W = max(A, A^T) of the top-k lists (the same arithmetic as PrefixTopK.matrix 'sym', not normalised) and the degrees."""
    n = len(graph.X)
    dev = graph.device
    rows = torch.arange(n, device=dev).repeat_interleave(k)
    cols = graph.I[:, :k].reshape(-1)
    vals = graph.S[:, :k].clamp_min(0.0).reshape(-1)
    keys = torch.cat([rows * n + cols, cols * n + rows])
    uniq, inv = torch.unique(keys, return_inverse=True)
    w = torch.zeros(len(uniq), dtype=torch.float32, device=dev).scatter_reduce(0, inv, torch.cat([vals, vals]), reduce="amax", include_self=False)
    keep = w > 0
    r, c, w = uniq[keep] // n, uniq[keep] % n, w[keep].double()
    W = torch.sparse_coo_tensor(torch.stack([r, c]), w, (n, n)).coalesce().to_sparse_csr()
    deg = torch.zeros(n, dtype=torch.float64, device=dev).index_add_(0, r, w)
    return W, deg


@torch.no_grad()
def solve_h(W, deg, one, zero, lam, gamma, tol=1e-11, maxit=20000):
    """h = E_x[lam^N exp(-gamma T); tau_one < tau_zero] for the continuous-time walk with jump rates W_ij:
    (D + gamma I - lam W)_FF h_F = lam W_F,one 1, h = 1 on `one`, 0 on `zero` (bool vectors). Returns (h, iterations, residual)."""
    one, zero = one[:, None], zero[:, None]
    free = (~(one | zero)).double()
    onef = one.double()
    b = lam * torch.sparse.mm(W, onef) * free
    dg = deg + gamma
    dinv = (1.0 / dg.clamp_min(1e-30))[:, None]

    def A(x):
        return (dg[:, None] * x - lam * torch.sparse.mm(W, x)) * free
    x = torch.zeros_like(b)
    r = b.clone()
    z = r * dinv
    p = z.clone()
    rz = (r * z).sum(0)
    bn = torch.linalg.vector_norm(b, dim=0).clamp_min(1e-300)
    res, it = 1.0, 0
    for it in range(maxit):
        Ap = A(p)
        alpha = rz / (p * Ap).sum(0).clamp_min(1e-300)
        x = x + alpha * p
        r = r - alpha * Ap
        res = float((torch.linalg.vector_norm(r, dim=0) / bn).max())
        if res < tol:
            break
        z = r * dinv
        rz_new = (r * z).sum(0)
        p = z + (rz_new / rz.clamp_min(1e-300)) * p
        rz = rz_new
    return (x + onef)[:, 0], it + 1, res


def union_graph(graphs):
    """Degree-normalised union of the views' graphs (same node set): sum_v W_v / median(delta_v)."""
    Wn, dn, med = None, None, {}
    for v, g in graphs.items():
        W, d = sym_W(g.graph)
        med[v] = float(d.median())
        Wv, dv = W * (1.0 / med[v]), d / med[v]
        Wn = Wv if Wn is None else Wn + Wv
        dn = dv if dn is None else dn + dv
    return Wn, dn, med


def split_cal(nc, ns, n_ref=P.N_REF, per_class=P.N_CAL_PER_CLASS):
    """Node indices of the reference R and the calibration set C (positions within each block of `per_class`)."""
    cal_nodes = ns + np.arange(nc)
    pos = np.arange(nc) % per_class
    return cal_nodes[pos < n_ref], cal_nodes[pos >= n_ref]


def h16(graphs, lam=P.LAM16, grel=P.GREL16, q=P.Q16):
    """The registered read-out on one task. graphs: {view: K.Graph} with the same nodes (support, calibration, test).
    Returns dict with s (test), p (test), s_cal (C), seeds info and solver info."""
    g0 = next(iter(graphs.values()))
    ns, nc, nt, nfix, N = g0.ns, g0.nc, g0.nt, g0.nfix, g0.N
    dev = g0.dev
    W, d, med = union_graph(graphs)
    gamma = grel * float(d.median())
    R, C = split_cal(nc, ns)
    test_nodes = nfix + np.arange(nt)
    pool = np.concatenate([C, test_nodes])
    S = torch.zeros(N, dtype=torch.bool, device=dev)
    S[:ns] = True
    none = torch.zeros(N, dtype=torch.bool, device=dev)
    h0, it0, r0 = solve_h(W, d, S, none, lam, gamma)
    h0 = h0.cpu().numpy()
    mask, pi0 = P.M14.select_seeds(p_low(h0[R], h0[pool]), q, len(R), "storey")
    seeds = pool[mask]
    Z = torch.zeros(N, dtype=torch.bool, device=dev)
    if len(seeds):
        Z[torch.as_tensor(seeds, device=dev)] = True
    h, it1, r1 = solve_h(W, d, S, Z, lam, gamma)
    s = (lam * torch.sparse.mm(W, h[:, None])[:, 0] / (d + gamma)).cpu().numpy()
    p = p_low(s[C], s[test_nodes])
    info = {"seeds": int(len(seeds)), "cal_seeds": int(np.isin(seeds, C).sum()), "pi0_hat": float(pi0), "gamma": gamma, "nfix": int(nfix),
            "median_degree_views": med, "median_degree_union": float(d.median()), "n_ref": int(len(R)), "n_cal": int(len(C)),
            "solver": {"stage1": [it0, r0], "stage2": [it1, r1]}}
    return {"s": s[test_nodes], "p": p, "s_cal": s[C], "h0_test": h0[test_nodes], "seed_nodes": seeds, "info": info}


def validity(p, flag, q=P.Q16):
    """Realised ID false-alarm rate / OOD recall of p <= alpha and the FDP of BH(q) on p over the whole test set (no folds)."""
    res = {}
    for a in ALPHAS:
        res[f"fpr@{a:g}"] = float((p[~flag] <= a).mean())
        res[f"tpr@{a:g}"] = float((p[flag] <= a).mean())
    mask = P.P.plain_bh(p, q)
    V, Rn = int((mask & ~flag).sum()), int(mask.sum())
    res["bh_fdp"] = float(V / max(Rn, 1))
    res["bh_rejections"] = Rn
    res["bh_recall"] = float((Rn - V) / max(int(flag.sum()), 1))
    return res


def logc(x):
    """log with the same floor as the base scores (log(max(x, 1e-300))): exact zeros of s tie at the bottom."""
    return np.log(np.maximum(np.asarray(x, np.float64), 1e-300))


def h16_rows(out, flag, bases, tag):
    """Rows of the registered read-out: 'h16|s' (ranking score s, the registered standalone score) and 'h16|p' (its conformal p),
    standalone and multiplied by every base ({(name, seed): log base score}); both base combinations s0*s and s0*p are reported
    (the one used in the paper is chosen on dev1 by the registered rule)."""
    rows = []
    sd, nfix = out["seed_nodes"], out["info"]["nfix"]
    idf = float(np.mean([(i < nfix) or (not flag[i - nfix]) for i in sd])) if len(sd) else None
    seed_tag = {"seeds_h16": out["info"]["seeds"], "cal_seeds_h16": out["info"]["cal_seeds"], "seed_idfrac_h16": idf}
    for name, x in (("h16|s", logc(out["s"])), ("h16|p", logc(out["p"]))):
        au, fp = measures(x, flag)
        extra = validity(out["p"], flag) if name == "h16|p" else {}
        rows.append({**tag, "cfg": name, "base": "none", "seed": -1, "AUROC": au, "FPR95": fp, **extra, **seed_tag})
        for (b, seed), lb in bases.items():
            au, fp = measures(x + lb, flag)
            rows.append({**tag, "cfg": name, "base": b, "seed": int(seed), "AUROC": au, "FPR95": fp})
    return rows
