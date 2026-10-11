"""Phase 15 graph with a variable neighbour count k_g (exact top-k lists up to kmax; the propagation and LGC max use g.k)."""
import numpy as np
import torch

import p15common as P
import core14 as K


class Graph15(K.Graph):
    """K.Graph whose default neighbour count is the attribute k (set before computing the terms); lists kept up to kmax."""

    def __init__(self, sup, cal, sf, dev, kmax=max(P.KGS), k=P.KG):
        self.ns, self.nc, self.nt = len(sup), len(cal), len(sf)
        self.nfix = self.ns + self.nc
        self.X = np.concatenate([sup, cal, sf]).astype(np.float32)
        self.N = len(self.X)
        self.dev = dev
        import engine5 as E5
        self.graph = E5.Graph5(self.X, device=dev, kmax=kmax)
        self.M = {}
        self.k = k

    def matrix(self, k=None):
        k = self.k if k is None else k
        if k not in self.M:
            self.M[k] = self.graph.matrix(k, 1.0, "sym")
        return self.M[k]

    @torch.no_grad()
    def mass(self, idx, k=None, lam=P.LAM):
        k = self.k if k is None else k
        y = torch.zeros((self.N, 1), dtype=torch.float64, device=self.dev)
        y[torch.as_tensor(np.asarray(idx), device=self.dev), 0] = 1.0
        U, it, res = K.propagate_converged(self.matrix(k), y, lam, max_iter=3000)
        return U[:, 0].cpu().numpy()

    def cut(self, idx):
        raise NotImplementedError


def centred(sup, cal, sf):
    """Subtracts the mean of the support features and re-normalises (the centred static / kNN read-outs of views15)."""
    mu = np.asarray(sup, np.float32).mean(0, keepdims=True)
    return tuple(P.l2n(np.asarray(x, np.float32) - mu) for x in (sup, cal, sf))
