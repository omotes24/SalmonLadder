"""Stateful REPRISE stream (for probes and operating studies): history batches mutate the state; a probe scores one
image on a shallow clone and discards it. Optional bounded retention of stream nodes and memory members."""
import copy

import numpy as np
import torch

from engine import Memory, PrefixTopK, _log, p_high, p_low, propagate
from scipy.special import log_ndtr


def score_only(mem, d, da, sims, ns, nfix):
    """Memory read-out without admission (the probe never enters A1/A2/M)."""
    ps = [p_high(mem.dac[mem.roles[0]], da)]
    for j in range(3):
        ids = mem.mem[j]
        if ids:
            idx = torch.as_tensor(np.asarray(ids) + nfix, device=sims.device)
            rs = (1.0 - sims.index_select(1, idx).max(1).values).cpu().numpy().astype(np.float64)
        else:
            rs = np.full(len(d), 2.0)
        rc = np.where(np.isfinite(mem.best_cal[j]), 1.0 - mem.best_cal[j].astype(np.float64), 2.0)
        g = d - (rs - mem.med) / mem.mad
        gc = mem.dc - (rc - mem.med) / mem.mad
        ps.append(p_high(gc[mem.roles[j + 1]], g))
    return ps


class StreamState:
    """cfgs: list of (family, k, gamma, lam) read-outs; families: cdf_L0 (warm, state kept), cdf, raw, z, mass,
    rw, sprop (all nodes from 0). static dict as engine.run_view (fixed rows only needed: d_cal, d_all_cal, med,
    mad, d_all_sup)."""

    def __init__(self, sup, cal, static, thresholds, cfgs, device="cuda"):
        sup = np.asarray(sup, np.float32).reshape(-1, sup.shape[-1])
        self.ns, self.nc = len(sup), len(cal)
        self.nfix = self.ns + self.nc
        self.device = device
        self.graph = PrefixTopK(np.concatenate([sup, np.asarray(cal, np.float32)]), device=device)
        self.mem = Memory(static["d_cal"], static["d_all_cal"], self.nc, static["med"], static["mad"], thresholds)
        self.cfgs = list(cfgs)
        self.warm = {c: None for c in self.cfgs if c[0] == "cdf_L0"}
        self.e = list(np.r_[static["d_all_sup"], static["d_all_cal"]])
        self.t = 0

    def _y(self, n):
        y = torch.zeros(n, dtype=torch.float64, device=self.device)
        y[:self.ns] = 1.0
        return y

    def step(self, feats, d, d_all):
        sims, oldn = self.graph.append(feats)
        self.mem.score(np.asarray(d), np.asarray(d_all), sims, self.ns, self.nfix, self.t)
        n = len(self.graph)
        y = self._y(n)
        mats = {}
        for c in self.warm:
            _, k, g, lam = c
            M = mats.setdefault((k, g), self.graph.matrix(k, g, "sym"))
            if self.warm[c] is None:
                w0 = y.clone()
            else:
                w0 = torch.zeros(n, dtype=torch.float64, device=self.device)
                w0[:oldn] = self.warm[c]
            self.warm[c] = propagate(M, y[:, None], torch.tensor([lam], dtype=torch.float64, device=self.device), w0[:, None])[:, 0]
        self.e.extend(np.asarray(d_all).tolist())
        self.t += len(feats)

    def probe(self, f, d, d_all, p_static):
        g2 = copy.copy(self.graph)
        sims, oldn = g2.append(np.asarray(f, np.float32)[None])
        n = len(g2)
        out = {"static": float(np.log(p_static)),
               "Mpt": float(_log(score_only(self.mem, np.array([d]), np.array([d_all]), sims, self.ns, self.nfix)[-1])[0])}
        y = self._y(n)
        mats = {}
        for c in self.cfgs:
            fam, k, g, lam = c
            kind = "rw" if fam == "rw" else "sym"
            M = mats.setdefault((k, g, kind), g2.matrix(k, g, kind))
            lt = torch.tensor([lam], dtype=torch.float64, device=self.device)
            if fam == "cdf_L0":
                w0 = torch.zeros(n, dtype=torch.float64, device=self.device)
                w0[:oldn] = self.warm[c]
                u = propagate(M, y[:, None], lt, w0[:, None])[:, 0].cpu().numpy()
            elif fam == "sprop":
                e0 = torch.as_tensor(np.r_[self.e, d_all], dtype=torch.float64, device=self.device)
                e = propagate(M, e0[:, None], lt, torch.zeros((n, 1), dtype=torch.float64, device=self.device))[:, 0].cpu().numpy()
                out[c] = float(np.log(p_high(e[self.ns:self.nfix], e[-1:])[0]))
                continue
            else:
                u = propagate(M, y[:, None], lt, torch.zeros((n, 1), dtype=torch.float64, device=self.device))[:, 0].cpu().numpy()
            uc, uq = u[self.ns:self.nfix], u[-1:]
            if fam in ("cdf_L0", "cdf"):
                out[c] = float(np.log(p_low(uc, uq)[0]))
            elif fam == "raw" or fam == "rw":
                out[c] = float(_log(uq)[0])
            elif fam == "mass":
                out[c] = float(_log(uq)[0] + np.log(n / self.ns))
            elif fam == "z":
                med = np.median(uc)
                mad = 1.4826 * np.median(np.abs(uc - med))
                out[c] = float(log_ndtr((uq[0] - med) / max(mad, 1e-300)))
        return out
