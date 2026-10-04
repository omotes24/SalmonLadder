"""Phase 4 streaming engine. Labels never enter this module.

One exact prefix kNN graph (top-20 cosine lists, GPU, float32 similarities) serves every LP read-out. For each
(k, gamma) the symmetrised graph max(A, A^T) is normalised (D^-1/2 W D^-1/2, or D^-1 W for the random-walk
variant) and propagated in float64. Per batch: memory scores use only earlier batches; the graph includes the
current batch (as REPRISE); issued scores are copied out and never revised. All read-outs are log-scores.
"""
import math
import time

import numpy as np
import torch
from scipy.special import log_ndtr

KMAX = 20


def p_low(cal, x):
    return (1 + np.searchsorted(np.sort(cal), x, side="right")) / (len(cal) + 1)


def p_high(cal, x):
    return (1 + len(cal) - np.searchsorted(np.sort(cal), x, side="left")) / (len(cal) + 1)


def _log(x):
    with np.errstate(divide="ignore"):
        return np.log(np.asarray(x, np.float64))


class PrefixTopK:
    """Exact prefix kNN lists: every node keeps its KMAX most similar earlier-or-current nodes (self excluded).
    Lists are sorted by similarity, so the top-k graph for k <= KMAX is the first k columns."""

    def __init__(self, fixed, device="cuda", kmax=KMAX):
        torch.backends.cuda.matmul.allow_tf32 = False
        self.device, self.kmax = device, kmax
        self.X = torch.as_tensor(np.ascontiguousarray(fixed, np.float32), device=device)
        n = len(self.X)
        S, I = [], []
        for lo in range(0, n, 1024):
            hi = min(lo + 1024, n)
            sims = self.X[lo:hi] @ self.X.T
            sims[torch.arange(hi - lo, device=device), torch.arange(lo, hi, device=device)] = -math.inf
            v, i = sims.topk(kmax, dim=1)
            S.append(v)
            I.append(i)
        self.S, self.I = torch.cat(S), torch.cat(I)

    def __len__(self):
        return len(self.X)

    def append(self, batch):
        b = torch.as_tensor(np.ascontiguousarray(batch, np.float32), device=self.device)
        oldn, nb = len(self.X), len(b)
        self.X = torch.cat([self.X, b])
        sims = b @ self.X.T
        ar = torch.arange(nb, device=self.device)
        sims[ar, oldn + ar] = -math.inf
        v, i = sims.topk(self.kmax, dim=1)
        both_s = torch.cat([self.S, sims[:, :oldn].T], dim=1)
        both_i = torch.cat([self.I, (oldn + ar).expand(oldn, nb)], dim=1)
        vv, pos = both_s.topk(self.kmax, dim=1)
        self.S = torch.cat([vv, v])
        self.I = torch.cat([both_i.gather(1, pos), i])
        return sims, oldn

    def matrix(self, k, gamma, kind="sym"):
        n = len(self.X)
        dev = self.device
        rows = torch.arange(n, device=dev).repeat_interleave(k)
        cols = self.I[:, :k].reshape(-1)
        vals = self.S[:, :k].clamp_min(0.0).pow(gamma).reshape(-1)          # float32, as the frozen graph
        keys = torch.cat([rows * n + cols, cols * n + rows])
        uniq, inv = torch.unique(keys, return_inverse=True)
        w = torch.zeros(len(uniq), dtype=torch.float32, device=dev).scatter_reduce(
            0, inv, torch.cat([vals, vals]), reduce="amax", include_self=False)
        r, c = uniq // n, uniq % n
        deg = torch.zeros(n, dtype=torch.float32, device=dev).index_add_(0, r, w)
        if kind == "sym":
            di = 1.0 / torch.sqrt(deg.clamp_min(1e-12))
            wn = di[r] * w * di[c]
        else:
            wn = w / deg.clamp_min(1e-12)[r]
        return torch.sparse_coo_tensor(torch.stack([r, c]), wn.double(), (n, n)).coalesce().to_sparse_csr()


def propagate(M, Y, lam, U0, iters=15):
    U = U0
    for _ in range(iters):
        U = lam * torch.sparse.mm(M, U) + (1.0 - lam) * Y
    return U


def propagate_converged(M, Y, lam, tol=1e-8, max_iter=1000):
    U = torch.zeros_like(Y)
    rhs = (1.0 - lam) * Y
    den = torch.linalg.vector_norm(rhs, dim=0).clamp_min(1e-300)
    for it in range(max_iter):
        U = lam * torch.sparse.mm(M, U) + rhs
        res = torch.linalg.vector_norm(U - lam * torch.sparse.mm(M, U) - rhs, dim=0) / den
        if float(res.max()) <= tol:
            break
    return U, it + 1, float(res.max())


class Memory:
    """REPRISE m=1 entrance (A1 -> A2 -> M) and read-out p_t; decisions use only earlier batches.
    Port of the audited core.Memory (same float32 similarity / float64 subtraction rule)."""

    def __init__(self, dc, dac, nc, med, mad, thresholds, split=False, capacity=None):
        self.dc, self.dac, self.nc = np.asarray(dc, np.float64), np.asarray(dac, np.float64), nc
        self.med, self.mad, self.th = med, mad, thresholds
        self.roles = [np.arange(nc)] * 4 if not split else [np.arange(j, nc, 4) for j in range(4)]
        self.mem = [[] for _ in range(3)]
        self.best_cal = [np.full(nc, -np.inf, np.float32) for _ in range(3)]
        self.capacity = capacity

    def score(self, d, da, sims, ns, nfix, start):
        ps = [p_high(self.dac[self.roles[0]], da)]
        for j in range(3):
            ids = self.mem[j]
            if ids:
                idx = torch.as_tensor(np.asarray(ids) + nfix, device=sims.device)
                rs = (1.0 - sims.index_select(1, idx).max(1).values).cpu().numpy().astype(np.float64)
            else:
                rs = np.full(len(d), 2.0)
            rc = np.where(np.isfinite(self.best_cal[j]), 1.0 - self.best_cal[j].astype(np.float64), 2.0)
            g = d - (rs - self.med) / self.mad
            gc = self.dc - (rc - self.med) / self.mad
            ps.append(p_high(gc[self.roles[j + 1]], g))
        masks = [ps[j] <= self.th[j] for j in range(3)]
        for j, mask in enumerate(masks):
            sel = np.flatnonzero(mask)
            if len(sel):
                self.mem[j].extend((start + sel).tolist())
                bc = sims[torch.as_tensor(sel, device=sims.device), ns:nfix].max(0).values.cpu().numpy()
                self.best_cal[j] = np.maximum(self.best_cal[j], bc)
        return ps, masks


def run_view(support, cal, sf, bidx, static, thresholds, graphs=((10, 1.0),), lams=(0.9,), device="cuda",
             families=("raw", "cdf", "z", "mass", "cdf_L0", "rw", "sprop"), converged=(), memory=True, trace=False):
    """support: C x 12 x d, cal: 4C x d, sf: stream features in arrival order, bidx: batch of each row (contiguous).
    static: dict with d, d_all, p (cal rows first, then stream), d_cal, d_all_cal, d_all_sup, med, mad.
    Returns {key: log-score array over stream rows}, keys 'static', 'Mpt', and '<family>|k<k>g<gamma>l<lam>'."""
    sup = np.asarray(support, np.float32).reshape(-1, support.shape[-1])
    cal = np.asarray(cal, np.float32)
    ns, nc, nt = len(sup), len(cal), len(sf)
    nfix = ns + nc
    graph = PrefixTopK(np.concatenate([sup, cal]), device=device)
    out = {"static": _log(static["p"][nc:])}
    lam_t = torch.tensor(lams, dtype=torch.float64, device=device)
    L = len(lams)
    keys = [(k, g) for k, g in graphs]
    tag = {(k, g, l): f"k{k}g{g:g}l{l:g}" for k, g in keys for l in lams}
    for (k, g) in keys:
        for l in lams:
            for fam in families:
                out[f"{fam}|{tag[(k, g, l)]}"] = np.empty(nt)
            for fam in converged:
                out[f"{fam}|{tag[(k, g, l)]}"] = np.empty(nt)
    if memory:
        mem = Memory(static["d_cal"], static["d_all_cal"], nc, static["med"], static["mad"], thresholds)
        out["Mpt"] = np.empty(nt)
        admit = np.zeros((nt, 3), bool)
    e_fixed = np.r_[static["d_all_sup"], static["d_all"][:nc]]
    e_stream = static["d_all"][nc:]
    prev = {kg: None for kg in keys}           # warm state (L0) per graph, one column per lambda
    diag = []
    batches = np.unique(bidx)
    for b in batches:
        rows = np.flatnonzero(bidx == b)
        start = int(rows[0])
        assert np.array_equal(rows, np.arange(start, start + len(rows)))
        t0 = time.perf_counter()
        sims, oldn = graph.append(sf[rows])
        n = len(graph)
        new = np.arange(oldn, n)
        t_graph = time.perf_counter() - t0
        if memory:
            t0 = time.perf_counter()
            ps, masks = mem.score(static["d"][nc + rows], static["d_all"][nc + rows], sims, ns, nfix, start)
            out["Mpt"][rows] = _log(ps[-1])
            for j in range(3):
                admit[rows, j] = masks[j]
            t_mem = time.perf_counter() - t0
        y = torch.zeros(n, dtype=torch.float64, device=device)
        y[:ns] = 1.0
        e0 = torch.as_tensor(np.r_[e_fixed, e_stream[:start + len(rows)]], dtype=torch.float64, device=device)
        t0 = time.perf_counter()
        for (k, g) in keys:
            want_sym = any(f in families for f in ("raw", "cdf", "z", "mass", "cdf_L0", "sprop")) or converged
            if want_sym:
                M = graph.matrix(k, g, "sym")
                cols_y = [y] * L
                U0 = [torch.zeros(n, dtype=torch.float64, device=device)] * L
                blocks = ["L1"] * L
                lamcols = list(lams)
                if "cdf_L0" in families:
                    if prev[(k, g)] is None:
                        w0 = [y.clone() for _ in range(L)]
                    else:
                        w0 = []
                        for j in range(L):
                            z = torch.zeros(n, dtype=torch.float64, device=device)
                            z[:oldn] = prev[(k, g)][:, j]
                            w0.append(z)
                    cols_y += [y] * L
                    U0 = U0 + w0
                    blocks += ["L0"] * L
                    lamcols += list(lams)
                if "sprop" in families:
                    cols_y += [e0] * L
                    U0 = U0 + [torch.zeros(n, dtype=torch.float64, device=device)] * L
                    blocks += ["SP"] * L
                    lamcols += list(lams)
                U = propagate(M, torch.stack(cols_y, 1), torch.tensor(lamcols, dtype=torch.float64, device=device),
                              torch.stack(U0, 1)).cpu().numpy()
                for j, l in enumerate(lams):
                    t = tag[(k, g, l)]
                    u = U[:, j]
                    uc, un = u[ns:nfix], u[new]
                    if "raw" in families:
                        out[f"raw|{t}"][rows] = _log(un)
                    if "cdf" in families:
                        out[f"cdf|{t}"][rows] = _log(p_low(uc, un))
                    if "z" in families:
                        med = np.median(uc)
                        mad = 1.4826 * np.median(np.abs(uc - med))
                        out[f"z|{t}"][rows] = log_ndtr((un - med) / max(mad, 1e-300))
                    if "mass" in families:
                        out[f"mass|{t}"][rows] = _log(un) + math.log(n / ns)
                    if "cdf_L0" in families:
                        u0 = U[:, L + j]
                        out[f"cdf_L0|{t}"][rows] = _log(p_low(u0[ns:nfix], u0[new]))
                    if "sprop" in families:
                        off = L * (2 if "cdf_L0" in families else 1)
                        e = U[:, off + j]
                        out[f"sprop|{t}"][rows] = _log(p_high(e[ns:nfix], e[new]))
                if "cdf_L0" in families:
                    prev[(k, g)] = torch.as_tensor(U[:, L:2 * L], device=device)
                if converged:
                    Uc, it, res = propagate_converged(M, torch.stack([y] * L, 1), lam_t)
                    Uc = Uc.cpu().numpy()
                    for j, l in enumerate(lams):
                        t = tag[(k, g, l)]
                        if "raw_L2" in converged:
                            out[f"raw_L2|{t}"][rows] = _log(Uc[new, j])
                        if "cdf_L2" in converged:
                            out[f"cdf_L2|{t}"][rows] = _log(p_low(Uc[ns:nfix, j], Uc[new, j]))
            if "rw" in families:
                Mr = graph.matrix(k, g, "rw")
                Ur = propagate(Mr, torch.stack([y] * L, 1), lam_t, torch.zeros((n, L), dtype=torch.float64, device=device)).cpu().numpy()
                for j, l in enumerate(lams):
                    out[f"rw|{tag[(k, g, l)]}"][rows] = _log(Ur[new, j])
        t_lp = time.perf_counter() - t0
        if trace:
            diag.append({"batch": int(b), "n_nodes": n, "graph_s": t_graph, "memory_s": t_mem if memory else 0.0,
                         "lp_s": t_lp})
    if memory:
        out["_admit"] = admit
    return out, diag
