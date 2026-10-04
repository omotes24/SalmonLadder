"""Phase 7 streaming engine: the frozen REPRISE v5 components (Phase 4 classes, unchanged) with extra read-outs.

Labels never enter this module. Per view and stream row (all are log p, small = OOD, ranks against the calibration
images under the same state):
  static        static p of d (candidate classes)
  M, Ma1, Ma2   frozen memory p_M and the two earlier entrance stages
  Mh<c>         one-sided (hinge) memory: g = d + max(0, c - (rho - med) / mad); same formula for the calibration images
  Mrho          memory proximity alone: rank of the nearest-member distance rho (no d)
  lp0           frozen propagation (warm start);  lp, z  zero start: rank and robust z (z = zeta of Phase 4)
  lpB           (optional) propagation on support + calibration + the current batch only
  lpA           (optional) propagation on support + calibration + the image alone (no test-time information)
Raw (float32): rho (nearest-member distance at scoring time, 2 = empty memory). Admissions: (rows, 3) bool.
"""
import copy
import math
import time

import numpy as np
import torch
from scipy.special import log_ndtr

from engine import Memory, PrefixTopK, _log, p_high, p_low, propagate

MAD = 1.4826
HINGES = (0, 1, 2, 3, 4)


def mem_parts(mem, j, sims, nfix, nb):
    """Nearest-member distance of the batch rows and of the calibration images for stage set j (before admission)."""
    ids = mem.mem[j]
    if ids:
        idx = torch.as_tensor(np.asarray(ids) + nfix, device=sims.device)
        rs = (1.0 - sims.index_select(1, idx).max(1).values).cpu().numpy().astype(np.float64)
    else:
        rs = np.full(nb, 2.0)
    rc = np.where(np.isfinite(mem.best_cal[j]), 1.0 - mem.best_cal[j].astype(np.float64), 2.0)
    return rs, rc


def memory_readouts(d, dc, rs, rc, med, mad, hinges=HINGES):
    """{key: p} for the frozen read-out, the hinge read-outs and the proximity-only read-out."""
    x, xc = (rs - med) / mad, (rc - med) / mad
    out = {"M": p_high(dc - xc, d - x), "Mrho": p_low(rc, rs)}
    for c in hinges:
        out[f"Mh{c:g}"] = p_high(dc + np.maximum(0.0, c - xc), d + np.maximum(0.0, c - x))
    return out


def lp_readouts(u, ns, nfix, new):
    uc, un = u[ns:nfix], u[new]
    med = np.median(uc)
    mad = MAD * np.median(np.abs(uc - med))
    return _log(p_low(uc, un)), log_ndtr((un - med) / max(mad, 1e-300))


def lp_fresh(fixed_graph, feats, ns, nfix, k, gamma, lam, iters=15):
    """Zero-start propagation on a copy of the fixed graph (support + calibration) plus `feats`; returns u."""
    g2 = copy.copy(fixed_graph)
    _, oldn = g2.append(feats)
    n = len(g2)
    dev = g2.device
    y = torch.zeros(n, dtype=torch.float64, device=dev)
    y[:ns] = 1.0
    M = g2.matrix(k, gamma, "sym")
    U = propagate(M, y[:, None], torch.tensor([lam], dtype=torch.float64, device=dev),
                  torch.zeros((n, 1), dtype=torch.float64, device=dev), iters)
    return U[:, 0].cpu().numpy(), oldn, n


def run_view7(support, cal, sf, bidx, static, thresholds, k=10, gamma=1.0, lam=0.9, device="cuda", hinges=HINGES,
              lp_batch=False, lp_alone=False, iters=15, check=True, retro=()):
    """support: C x n_sup x D, cal: nc x D (class-major), sf: stream features in arrival order, bidx: batch per row.
    static: dict of static.static_view (d, d_all, p with the calibration rows first; d_cal, d_all_cal, med, mad).
    retro: delays (in batches). '<key>@<delay>' is the read-out of an image re-scored after `delay` further batches have
    been processed (graph and memory of that time, the image itself never counted as its own memory neighbour);
    '<key>@end' is the read-out at the end of the stream. Rows whose delay runs past the stream get the end value.
    The issued scores (no suffix) are never revised."""
    sup = np.asarray(support, np.float32).reshape(-1, support.shape[-1])
    cal = np.asarray(cal, np.float32)
    sf = np.asarray(sf, np.float32)
    ns, nc, nt = len(sup), len(cal), len(sf)
    nfix = ns + nc
    graph = PrefixTopK(np.concatenate([sup, cal]), device=device)
    fixed = copy.copy(graph) if (lp_batch or lp_alone) else None       # append() rebinds, the fixed lists stay intact
    mem = Memory(static["d_cal"], static["d_all_cal"], nc, static["med"], static["mad"], thresholds)
    dc = np.asarray(static["d_cal"], np.float64)
    med, mad = static["med"], static["mad"]
    keys = ["M", "Ma1", "Ma2", "Mrho", "lp0", "lp", "z"] + [f"Mh{c:g}" for c in hinges]
    keys += ["lpB", "zB"] if lp_batch else []
    keys += ["lpA", "zA"] if lp_alone else []
    out = {key: np.full(nt, np.nan) for key in keys}
    out["static"] = _log(static["p"][nc:])
    admit = np.zeros((nt, 3), bool)
    rho = np.full(nt, np.nan, np.float32)
    lam_t = torch.tensor([lam, lam], dtype=torch.float64, device=device)
    prev = None
    t_mem = t_lp = t_extra = 0.0
    batches = np.unique(bidx)
    retro = tuple(retro)
    if retro:
        mkeys = ["M"] + [f"Mh{c:g}" for c in hinges] + ["Mrho"]
        rkeys = mkeys + ["lp0", "lp", "z"]
        R = {f"{key}@{dl}": np.full(nt, np.nan) for key in rkeys for dl in list(retro) + ["end"]}
        best = torch.full((nt,), -math.inf, device=device)        # nearest-member similarity of every stream node
        rows_of = []
    for bi, b in enumerate(batches):
        rows = np.flatnonzero(bidx == b)
        start = int(rows[0])
        assert np.array_equal(rows, np.arange(start, start + len(rows))), "stream must be in batch order"
        t0 = time.perf_counter()
        sims, oldn = graph.append(sf[rows])
        n = len(graph)
        new = np.arange(oldn, n)
        d, da = static["d"][nc + rows], static["d_all"][nc + rows]
        rs, rc = mem_parts(mem, 2, sims, nfix, len(rows))               # state of the earlier batches
        ps, masks = mem.score(d, da, sims, ns, nfix, start)             # frozen read-out + admission (Phase 4 code)
        ro = memory_readouts(d, dc, rs, rc, med, mad, hinges)
        if check:
            assert np.array_equal(ro["M"], ps[3]), "memory read-out differs from the frozen code"
        out["M"][rows], out["Ma1"][rows], out["Ma2"][rows] = _log(ps[3]), _log(ps[1]), _log(ps[2])
        for key, p in ro.items():
            if key != "M":
                out[key][rows] = _log(p)
        rho[rows] = rs
        for j in range(3):
            admit[rows, j] = masks[j]
        t_mem += time.perf_counter() - t0
        if bi % 8 == 7 and str(device).startswith("cuda"):
            torch.cuda.empty_cache()                 # the graph grows every batch: keep the allocator from fragmenting
        t0 = time.perf_counter()
        y = torch.zeros(n, dtype=torch.float64, device=device)
        y[:ns] = 1.0
        if prev is None:
            w0 = y.clone()
        else:
            w0 = torch.zeros(n, dtype=torch.float64, device=device)
            w0[:oldn] = prev
        M = graph.matrix(k, gamma, "sym")
        U = propagate(M, torch.stack([y, y], 1), lam_t, torch.stack([torch.zeros_like(y), w0], 1), iters).cpu().numpy()
        out["lp"][rows], out["z"][rows] = lp_readouts(U[:, 0], ns, nfix, new)
        u0 = U[:, 1]
        out["lp0"][rows] = _log(p_low(u0[ns:nfix], u0[new]))
        prev = torch.as_tensor(u0, device=device)
        t_lp += time.perf_counter() - t0
        t0 = time.perf_counter()
        if retro:
            rows_of.append(rows)
            if mem.mem[2] and len(mem.mem[2]) > int(masks[2].sum()):                  # members of earlier batches existed
                best[torch.as_tensor(rows, device=device)] = torch.as_tensor(1.0 - rs, dtype=torch.float32, device=device)
            sel = np.flatnonzero(masks[2])
            if len(sel):                                                              # members admitted from this batch
                st_ = torch.as_tensor(sel, device=device)
                ss = sims[st_, nfix:n].clone()
                ss[torch.arange(len(sel), device=device), start + st_] = -math.inf     # never its own neighbour
                best[:n - nfix] = torch.maximum(best[:n - nfix], ss.max(0).values)
            rc2 = np.where(np.isfinite(mem.best_cal[2]), 1.0 - mem.best_cal[2].astype(np.float64), 2.0)
            last = bi == len(batches) - 1
            targets = [(dl, rows_of[bi - dl]) for dl in retro if bi - dl >= 0]
            if last:
                targets.append(("end", np.arange(start + len(rows))))
            for dl, T in targets:
                bt = best[torch.as_tensor(T, device=device)]
                rsT = torch.where(torch.isfinite(bt), 1.0 - bt, torch.full_like(bt, 2.0)).cpu().numpy().astype(np.float64)
                for key, p in memory_readouts(static["d"][nc + T], dc, rsT, rc2, med, mad, hinges).items():
                    R[f"{key}@{dl}"][T] = _log(p)
                a1, z1 = lp_readouts(U[:, 0], ns, nfix, nfix + T)
                R[f"lp@{dl}"][T], R[f"z@{dl}"][T] = a1, z1
                R[f"lp0@{dl}"][T] = _log(p_low(u0[ns:nfix], u0[nfix + T]))
        if lp_batch:
            u, o2, n2 = lp_fresh(fixed, sf[rows], ns, nfix, k, gamma, lam, iters)
            out["lpB"][rows], out["zB"][rows] = lp_readouts(u, ns, nfix, np.arange(o2, n2))
        if lp_alone:
            for i in rows:
                u, o2, n2 = lp_fresh(fixed, sf[i:i + 1], ns, nfix, k, gamma, lam, iters)
                a, z = lp_readouts(u, ns, nfix, np.arange(o2, n2))
                out["lpA"][i], out["zA"][i] = a[0], z[0]
        t_extra += time.perf_counter() - t0
    if retro:
        for key in rkeys:
            for dl in retro:                                  # the stream ended before the delay: end-of-stream value
                miss = np.isnan(R[f"{key}@{dl}"])
                R[f"{key}@{dl}"][miss] = R[f"{key}@end"][miss]
        out.update(R)
    for key, v in out.items():
        assert not np.isnan(v).any(), key
    return out, {"admit": admit, "rho": rho, "seconds": {"memory": t_mem, "lp": t_lp, "extra": t_extra}}


class State7:
    """Stateful frozen stream for probes (Exp 4 design): history batches mutate the state; a probe scores one image on
    a shallow clone (never admitted, never kept in the graph)."""

    def __init__(self, sup, cal, static, thresholds, k=10, gamma=1.0, lam=0.9, device="cuda", hinges=HINGES, iters=15):
        sup = np.asarray(sup, np.float32).reshape(-1, sup.shape[-1])
        self.ns, self.nc = len(sup), len(cal)
        self.nfix = self.ns + self.nc
        self.device, self.k, self.gamma, self.lam, self.hinges, self.iters = device, k, gamma, lam, hinges, iters
        self.graph = PrefixTopK(np.concatenate([sup, np.asarray(cal, np.float32)]), device=device)
        self.mem = Memory(static["d_cal"], static["d_all_cal"], self.nc, static["med"], static["mad"], thresholds)
        self.dc = np.asarray(static["d_cal"], np.float64)
        self.med, self.mad = static["med"], static["mad"]
        self.warm = None
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
        if self.warm is None:
            w0 = y.clone()
        else:
            w0 = torch.zeros(n, dtype=torch.float64, device=self.device)
            w0[:oldn] = self.warm
        M = self.graph.matrix(self.k, self.gamma, "sym")
        self.warm = propagate(M, y[:, None], torch.tensor([self.lam], dtype=torch.float64, device=self.device), w0[:, None],
                              self.iters)[:, 0]
        self.t += len(feats)

    def probe(self, f, d, d_all, p_static):
        g2 = copy.copy(self.graph)
        sims, oldn = g2.append(np.asarray(f, np.float32)[None])
        n = len(g2)
        rs, rc = mem_parts(self.mem, 2, sims, self.nfix, 1)
        ro = memory_readouts(np.array([d]), self.dc, rs, rc, self.med, self.mad, self.hinges)
        out = {key: float(_log(p)[0]) for key, p in ro.items()}
        out["static"] = float(np.log(p_static))
        out["rho"] = float(rs[0])
        y = self._y(n)
        w0 = torch.zeros(n, dtype=torch.float64, device=self.device)
        if self.warm is None:
            w0 = y.clone()
        else:
            w0[:oldn] = self.warm
        M = g2.matrix(self.k, self.gamma, "sym")
        lam_t = torch.tensor([self.lam, self.lam], dtype=torch.float64, device=self.device)
        U = propagate(M, torch.stack([y, y], 1), lam_t, torch.stack([torch.zeros_like(y), w0], 1), self.iters).cpu().numpy()
        new = np.arange(oldn, n)
        a, z = lp_readouts(U[:, 0], self.ns, self.nfix, new)
        out["lp"], out["z"] = float(a[0]), float(z[0])
        u0 = U[:, 1]
        out["lp0"] = float(_log(p_low(u0[self.ns:self.nfix], u0[new]))[0])
        return out
