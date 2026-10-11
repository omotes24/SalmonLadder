"""Phase 12 core: the post-stream two-sided read-out of one view (as final11.view_terms, with the Storey-AdaDetect seed rule),
same-feature baselines (kNN rank, static prototype rank), and the arrival-time (online) mode."""
import math

import numpy as np
import torch

import p12common as P
import engine5 as E5
from engine import PrefixTopK, _log, p_high, p_low, propagate, propagate_converged


class Graph:
    """One k-NN graph over [support; calibration; test] (exact top-20 lists; W = max(A, A^T); S = D^-1/2 W D^-1/2)."""

    def __init__(self, sup, cal, sf, dev):
        self.ns, self.nc, self.nt = len(sup), len(cal), len(sf)
        self.nfix = self.ns + self.nc
        self.X = np.concatenate([sup, cal, sf]).astype(np.float32)
        self.N = len(self.X)
        self.dev = dev
        self.graph = E5.Graph5(self.X, device=dev)
        self.M = {}

    def matrix(self, k=P.KG):
        if k not in self.M:
            self.M[k] = self.graph.matrix(k, 1.0, "sym")
        return self.M[k]

    @torch.no_grad()
    def mass(self, idx, k=P.KG, lam=P.LAM):
        y = torch.zeros((self.N, 1), dtype=torch.float64, device=self.dev)
        y[torch.as_tensor(np.asarray(idx), device=self.dev), 0] = 1.0
        U, it, res = propagate_converged(self.matrix(k), y, lam, max_iter=3000)
        return U[:, 0].cpu().numpy()

    def cut(self, idx):
        """Weight of the edges leaving the node set idx (W symmetrised), and the number of those edges that the set's own
        nodes list themselves vs that outside nodes list (graph lists = top-k_g neighbours)."""
        I = self.graph.I[:, :P.KG].cpu().numpy()
        S = self.graph.S[:, :P.KG].clamp_min(0).cpu().numpy()
        inset = np.zeros(self.N, bool)
        inset[idx] = True
        own = (~inset[I[idx]]).sum()                                       # edges listed by members pointing outside
        outside = np.flatnonzero(~inset)
        incoming = inset[I[outside]].sum()                                 # edges listed by outsiders pointing inside
        w_own = float(np.where(~inset[I[idx]], S[idx], 0.0).sum())
        w_in = float(np.where(inset[I[outside]], S[outside], 0.0).sum())
        return {"own_edges": int(own), "incoming_edges": int(incoming), "w_own": w_own, "w_incoming": w_in}

    def free(self):
        del self.graph, self.M, self.X
        torch.cuda.empty_cache()


def read_out(g, flag=None, qs=P.QS, lam=P.LAM, rule="storey"):
    """log p+ of the test images, and per q: seeds, log p-, pi0_hat and the realised ID fraction (flag given)."""
    u = g.mass(np.arange(g.ns), lam=lam)
    uc, ut = u[g.ns:g.nfix], u[g.nfix:]
    p_pos = p_low(uc, ut)
    out = {"lp": _log(p_pos), "u_pos": u}
    meta = {}
    for q in qs:
        if rule == "storey":
            mask, pi0 = P.storey_adadetect(p_pos, q, g.nc)
        else:
            mask, pi0 = P.plain_bh(p_pos, q), 1.0
        idx = np.flatnonzero(mask)
        meta[f"q{q:g}"] = {"seeds": int(len(idx)), "pi0_hat": pi0,
                           "id_frac": (float(1.0 - flag[idx].mean()) if (flag is not None and len(idx)) else None)}
        if len(idx) == 0:
            out[f"ln|q{q:g}"] = np.zeros_like(out["lp"])
            out[f"seeds|q{q:g}"] = idx
            continue
        un = g.mass(g.nfix + idx, lam=lam)
        out[f"ln|q{q:g}"] = _log(p_high(un[g.ns:g.nfix], un[g.nfix:]))
        out[f"seeds|q{q:g}"] = idx
        out[f"u_neg|q{q:g}"] = un
    return out, meta


@torch.no_grad()
def knn_rank(sup, cal, sf, k=10, dev="cuda"):
    """Same-feature baseline: cosine similarity to the k-th nearest support image (Sun et al. 2022), read as the calibrated
    rank among the calibration images (small = unknown). Returns log p for the test images."""
    S = torch.as_tensor(sup, device=dev)

    def kth(X):
        X = torch.as_tensor(X, device=dev)
        vals = []
        for lo in range(0, len(X), 4096):
            s = X[lo:lo + 4096] @ S.T
            vals.append(s.topk(k, dim=1).values[:, -1])
        return torch.cat(vals).cpu().numpy().astype(np.float64)

    return _log(p_low(kth(cal), kth(sf)))


@torch.no_grad()
def online_view(sup, cal, sf, bidx, q=P.Q_MAIN, k=P.KG, lam=P.LAM, iters=15, dev="cuda", flag=None):
    """Arrival-time mode of one view: the graph grows batch by batch (exact k-NN lists), u+ and u- are iterated 15 sweeps from the
    previous values (new nodes start at 0; the first batch from y), the seeds are re-selected at every batch from the current
    p+ of every image so far, and the score of a batch is emitted once. Returns log p+, log p- (emitted values) and seed stats."""
    ns, nc = len(sup), len(cal)
    nfix = ns + nc
    graph = PrefixTopK(np.concatenate([sup, cal]).astype(np.float32), device=dev)
    nt = len(sf)
    lp, ln = np.empty(nt), np.empty(nt)
    prev_pos = prev_neg = None
    y = torch.zeros(nfix, dtype=torch.float64, device=dev)
    y[:ns] = 1.0
    stats = []
    for b in np.unique(bidx):
        rows = np.flatnonzero(bidx == b)
        sims, oldn = graph.append(sf[rows])
        n = len(graph)
        new = np.arange(oldn, n)
        M = graph.matrix(k, 1.0, "sym")
        ypos = torch.zeros((n, 1), dtype=torch.float64, device=dev)
        ypos[:ns, 0] = 1.0
        if prev_pos is None:
            u0 = ypos.clone()
        else:
            u0 = torch.zeros((n, 1), dtype=torch.float64, device=dev)
            u0[:oldn, 0] = prev_pos
        U = propagate(M, ypos, lam, u0, iters)
        u = U[:, 0]
        prev_pos = u
        un = u.cpu().numpy()
        p_all = p_low(un[ns:nfix], un[nfix:])                                   # every stream image so far, current graph
        lp[rows] = _log(p_all[new - nfix])
        mask, pi0 = P.storey_adadetect(p_all, q, nc)
        idx = np.flatnonzero(mask)
        if len(idx) == 0:
            ln[rows] = 0.0
            prev_neg = torch.zeros(n, dtype=torch.float64, device=dev)
            stats.append((int(b), 0, None))
            continue
        yneg = torch.zeros((n, 1), dtype=torch.float64, device=dev)
        yneg[torch.as_tensor(nfix + idx, device=dev), 0] = 1.0
        if prev_neg is None:
            v0 = yneg.clone()
        else:
            v0 = torch.zeros((n, 1), dtype=torch.float64, device=dev)
            v0[:oldn, 0] = prev_neg
        V = propagate(M, yneg, lam, v0, iters)
        v = V[:, 0]
        prev_neg = v
        vn = v.cpu().numpy()
        ln[rows] = _log(p_high(vn[ns:nfix], vn[nfix:][new - nfix]))
        stats.append((int(b), int(len(idx)), (float(1.0 - flag[idx].mean()) if flag is not None else None)))
    del graph
    torch.cuda.empty_cache()
    return lp, ln, stats
