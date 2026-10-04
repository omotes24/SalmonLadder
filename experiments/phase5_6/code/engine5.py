"""Phase 5 exploration engine (development splits only). Labels (is_ood) never enter this module.

All views advance together, batch by batch, so that an admission rule may use the evidence of every view.
Every read-out is a conformal rank against the calibration images under the same state (log p, small = OOD).
The frozen components (three-stage memory M0, warm-start LP) run through the unchanged Phase 4 code.

Read-outs per view (keys of the returned dict; <cfg> = k<k>g<gamma>b<beta>l<lambda>):
  static, static_all      static p of d (candidate classes) and d_all (all classes)
  M0, M0a1, M0a2          frozen memory p_t and the stage p-values p_A1, p_A2
  lp0|<cfg>               rank of u, warm start (frozen REPRISE LP when cfg is the reference)
  lp|<cfg>, z|<cfg>       rank / robust z of u, every node from 0 (15 sweeps)
  cr|<cfg>, cz|<cfg>      class-conditional: received mass relative to the support nodes of the nearest class
  ctr1, ctr3, gall        all-stream statistics (no entrance): stream-vs-support similarity contrast (top-1, top-3
                          mean) and the memory score with every earlier stream image as memory
  <X>                     memory p_t of the admission variant X (Mlp<e>, Mfin<e>, Mjoint<e>)
  neg:<X>|<cfg>           rank of the mass propagated from the members of X (OOD seeds), reference cfg only
  tier 2 (optional)       cmax, cK, cpur (class-resolved propagation), ep, epc (propagated embedding)
"""
import math
import time

import numpy as np
import torch
from scipy.special import log_ndtr

from engine import Memory, PrefixTopK, _log, p_high, p_low, propagate

MAD = 1.4826


def loo_low(c):
    """Leave-one-out rank of every calibration value among the calibration values (ID direction: small = OOD)."""
    c = np.asarray(c, np.float64)
    return np.searchsorted(np.sort(c), c, side="right") / len(c)


def loo_high(c):
    c = np.asarray(c, np.float64)
    return (len(c) - np.searchsorted(np.sort(c), c, side="left")) / len(c)


def l2n(x):
    x = np.asarray(x, np.float64)
    return (x / np.linalg.norm(x, axis=-1, keepdims=True)).astype(np.float32)


def fit_transform(sup, kind="raw", alpha=0.5, device="cpu"):
    """Feature map fitted on the support images only. sup: C x n x D (L2-normalised).
    raw | center (subtract the support mean, renormalise) | white (pooled within-class covariance, shrunk as
    (1 - alpha) S + alpha tr(S)/D I, then centre, whiten, renormalise). float64 on `device`."""
    if kind == "raw":
        return lambda x: np.asarray(x, np.float32)
    S = torch.as_tensor(np.asarray(sup), dtype=torch.float64, device=device)
    C, n, D = S.shape
    mean = S.reshape(-1, D).mean(0)
    W = None
    if kind == "white":
        R = (S - S.mean(1, keepdim=True)).reshape(-1, D)
        cov = R.T @ R / (len(R) - C)
        cov = (1 - alpha) * cov + alpha * torch.trace(cov) / D * torch.eye(D, dtype=torch.float64, device=device)
        ev, evec = torch.linalg.eigh(cov)
        ev = ev.clamp_min(1e-10 * float(ev.max()))
        W = (evec / ev.sqrt()) @ evec.T
    elif kind != "center":
        raise ValueError(kind)

    def f(x):
        x = torch.as_tensor(np.asarray(x), dtype=torch.float64, device=device)
        y = x.reshape(-1, D) - mean
        if W is not None:
            y = y @ W
        y = y / y.norm(dim=1, keepdim=True)
        return y.reshape(x.shape).float().cpu().numpy()

    return f


def proto_stats(sup, n0=48):
    """Prototype statistics of vins.r5.proto_view (same arithmetic): prototypes, shrunken LOO median / MAD."""
    sup = np.asarray(sup, np.float64)
    tot = sup.sum(axis=1)
    mu = tot / np.linalg.norm(tot, axis=1, keepdims=True)
    loo_mu = tot[:, None, :] - sup
    loo_mu /= np.linalg.norm(loo_mu, axis=2, keepdims=True)
    loo = 1.0 - np.einsum("cnd,cnd->cn", sup, loo_mu)
    n = loo.shape[1]
    med_c = np.median(loo, axis=1)
    mad_c = MAD * np.median(np.abs(loo - med_c[:, None]), axis=1)
    med_all = float(np.median(loo))
    mad_all = float(MAD * np.median(np.abs(loo - med_all)))
    return {"mu": mu, "med_t": (n * med_c + n0 * med_all) / (n + n0), "mad_t": (n * mad_c + n0 * mad_all) / (n + n0)}


def nearest_class(q, ps, chunk=8192):
    """argmin_c z_c(x) over all classes, the minimum (= d_all of proto_view) and max_c cos(x, mu_c)."""
    q = np.asarray(q, np.float64)
    cls = np.empty(len(q), np.int64)
    dmin = np.empty(len(q))
    pcos = np.empty(len(q))
    for lo in range(0, len(q), chunk):
        c = q[lo:lo + chunk] @ ps["mu"].T
        z = ((1.0 - c) - ps["med_t"][None, :]) / ps["mad_t"][None, :]
        cls[lo:lo + chunk] = z.argmin(1)
        dmin[lo:lo + chunk] = z.min(1)
        pcos[lo:lo + chunk] = c.max(1)
    return cls, dmin, pcos


def static5(proto_view, support_dall, sup, cal, sf, cand_cal, cand_sf, n0=48, m=1):
    """Static view of the frozen method (vins.r5.proto_view) plus the nearest class of every row."""
    q = np.concatenate([cal, sf])
    is_cal = np.zeros(len(q), bool)
    is_cal[:len(cal)] = True
    v = proto_view(sup, q, np.concatenate([cand_cal, cand_sf]), is_cal, n0=n0, m=m)
    ps = proto_stats(sup, n0)
    cls, dmin, pcos = nearest_class(q, ps)
    assert np.allclose(dmin, v["d_all"], rtol=0, atol=1e-9), "nearest_class disagrees with proto_view"
    return {"d": v["d"], "d_all": v["d_all"], "p": v["p"], "p_all": v["p_all"], "d_cal": v["d_cal"],
            "d_all_cal": v["d_all_cal"], "med": v["stats"]["med_all"], "mad": v["stats"]["mad_all"],
            "d_all_sup": support_dall(sup, n0), "cls": cls, "pcos": pcos, "proto": ps}


class Graph5(PrefixTopK):
    def matrix2(self, k, gamma, beta=1.0):
        """Symmetrised kNN graph; an edge listed by only one end point is weighted by beta (1 = the frozen graph,
        0 = mutual kNN graph). D^-1/2 W D^-1/2 in float64."""
        if beta == 1.0:
            return self.matrix(k, gamma, "sym")
        n = len(self.X)
        dev = self.device
        rows = torch.arange(n, device=dev).repeat_interleave(k)
        cols = self.I[:, :k].reshape(-1)
        vals = self.S[:, :k].clamp_min(0.0).pow(gamma).reshape(-1)
        keys = torch.cat([rows * n + cols, cols * n + rows])
        uniq, inv, cnt = torch.unique(keys, return_inverse=True, return_counts=True)
        w = torch.zeros(len(uniq), dtype=torch.float32, device=dev).scatter_reduce(
            0, inv, torch.cat([vals, vals]), reduce="amax", include_self=False)
        w = torch.where(cnt >= 2, w, beta * w)
        keep = w > 0
        uniq, w = uniq[keep], w[keep]
        r, c = uniq // n, uniq % n
        deg = torch.zeros(n, dtype=torch.float32, device=dev).index_add_(0, r, w)
        di = 1.0 / torch.sqrt(deg.clamp_min(1e-12))
        wn = di[r] * w * di[c]
        return torch.sparse_coo_tensor(torch.stack([r, c]), wn.double(), (n, n)).coalesce().to_sparse_csr()


def prefix_cos(top_s, top_i, X):
    """max_j cos(x, centroid of the j most similar members), j = 1..k. top_s: q x k similarities to the members
    (descending, -inf where there is no member), top_i: their node indices. j = 1 is the nearest-member similarity."""
    valid = torch.isfinite(top_s)
    F = X[top_i]
    G = torch.bmm(F, F.transpose(1, 2)) * (valid[:, :, None] & valid[:, None, :])
    cs = torch.cumsum(torch.where(valid, top_s, torch.zeros_like(top_s)), dim=1)
    cg = torch.cumsum(torch.cumsum(G, dim=1), dim=2)
    den = torch.sqrt(torch.diagonal(cg, dim1=1, dim2=2).clamp_min(1e-12))
    cos = torch.where(valid, cs / den, torch.full_like(cs, -math.inf))
    return cos.max(1).values


class MemX:
    """One admitted set with the frozen read-out g = d - (rho - med) / mad (rho: distance to the nearest member).
    cen_k > 0 adds the centroid read-out: rho = 1 - max_j cos(x, centroid of the j nearest members), j <= cen_k."""

    def __init__(self, dc, nc, med, mad, cen_k=0, device="cpu"):
        self.dc, self.nc, self.med, self.mad = np.asarray(dc, np.float64), nc, med, mad
        self.ids = []
        self.best_cal = np.full(nc, -np.inf, np.float32)
        self.cen_k = cen_k
        if cen_k:
            self.cal_s = torch.full((nc, cen_k), -math.inf, device=device)
            self.cal_i = torch.zeros((nc, cen_k), dtype=torch.long, device=device)

    def score(self, d, sims, nfix):
        if self.ids:
            idx = torch.as_tensor(np.asarray(self.ids) + nfix, device=sims.device)
            rs = (1.0 - sims.index_select(1, idx).max(1).values).cpu().numpy().astype(np.float64)
        else:
            rs = np.full(len(d), 2.0)
        rc = np.where(np.isfinite(self.best_cal), 1.0 - self.best_cal.astype(np.float64), 2.0)
        g = d - (rs - self.med) / self.mad
        gc = self.dc - (rc - self.med) / self.mad
        return p_high(gc, g), gc

    def score_cen(self, d, sims, nfix, X):
        """Returns p of g with the centroid distance, the calibration g, and the raw centroid cosines (batch, cal)."""
        k = self.cen_k
        if self.ids:
            idx = torch.as_tensor(np.asarray(self.ids) + nfix, device=sims.device)
            kk = min(k, len(self.ids))
            ts, pos = sims.index_select(1, idx).topk(kk, dim=1)
            ti = idx[pos]
            if kk < k:
                ts = torch.cat([ts, torch.full((len(d), k - kk), -math.inf, device=sims.device)], 1)
                ti = torch.cat([ti, torch.zeros((len(d), k - kk), dtype=torch.long, device=sims.device)], 1)
            cs = prefix_cos(ts, ti, X).cpu().numpy().astype(np.float64)
            cc = prefix_cos(self.cal_s, self.cal_i, X).cpu().numpy().astype(np.float64)
            cc = np.where(np.isfinite(cc), cc, -1.0)
        else:
            cs, cc = np.full(len(d), -1.0), np.full(self.nc, -1.0)
        g = d - ((1.0 - cs) - self.med) / self.mad
        gc = self.dc - ((1.0 - cc) - self.med) / self.mad
        return p_high(gc, g), gc, cs, cc

    def admit(self, mask, start, sims, ns, nfix):
        sel = np.flatnonzero(mask)
        if len(sel):
            self.ids.extend((start + sel).tolist())
            st = torch.as_tensor(sel, device=sims.device)
            sc = sims[st, ns:nfix]                                           # members x calibration
            self.best_cal = np.maximum(self.best_cal, sc.max(0).values.cpu().numpy())
            if self.cen_k:
                both_s = torch.cat([self.cal_s, sc.T], dim=1)
                node = (nfix + start + st)[None, :].expand(self.nc, len(sel))
                both_i = torch.cat([self.cal_i, node], dim=1)
                self.cal_s, pos = both_s.topk(self.cen_k, dim=1)
                self.cal_i = both_i.gather(1, pos)


def cfg_tag(k, g, b, l):
    return f"k{k}g{g:g}b{b:g}l{l:g}"


class View:
    def __init__(self, name, sup, cal, sf, static, spec, device):
        self.name, self.spec, self.device = name, spec, device
        self.C = sup.shape[0]
        self.sup_flat = np.asarray(sup, np.float32).reshape(-1, sup.shape[-1])
        self.cal = np.asarray(cal, np.float32)
        self.sf = np.asarray(sf, np.float32)
        self.st = static
        self.ns, self.nc, self.nt = len(self.sup_flat), len(self.cal), len(self.sf)
        self.nfix = self.ns + self.nc
        self.graph = Graph5(np.concatenate([self.sup_flat, self.cal]), device=device)
        self.mem0 = Memory(static["d_cal"], static["d_all_cal"], self.nc, static["med"], static["mad"], spec["thresholds"])
        self.memx = {x: MemX(static["d_cal"], self.nc, static["med"], static["mad"],
                             spec["cen_k"] if x in spec["cen"] else 0, device) for x in spec["variants"]}
        self.out = {"static": _log(static["p"][self.nc:]), "static_all": _log(static["p_all"][self.nc:])}
        self.adm = {x: np.zeros(self.nt, bool) for x in list(spec["variants"]) + ["M0"]}
        self.adm["M0a1"] = np.zeros(self.nt, bool)
        self.adm["M0a2"] = np.zeros(self.nt, bool)
        self.warm = {}
        # all-stream statistics: running top-3 similarities of every calibration image to the stream
        X = self.graph.X
        s_cs = X[self.ns:self.nfix] @ X[:self.ns].T
        self.cal_ref = s_cs.topk(3, dim=1).values                                  # nc x 3 (support only)
        self.cal_str = torch.full((self.nc, 3), -1.0, device=device)
        if spec["cen"]:
            ts, ti = s_cs.topk(spec["cen_k"], dim=1)
            self.cal_supcos = prefix_cos(ts, ti, X).double().cpu().numpy()
        self.sup_cls = np.repeat(np.arange(self.C), self.ns // self.C)
        self.cal_cls = np.repeat(np.arange(self.C), self.nc // self.C)              # class-major calibration rows
        self.cls_cal, self.cls_sf = static["cls"][:self.nc], static["cls"][self.nc:]
        self.timing = []
        self.CS = None                                   # calibration x stream similarities (dynamic sets)
        self.dyn_last = {}

    def put(self, key, rows, val):
        if key not in self.out:
            self.out[key] = np.full(self.nt, np.nan)
        self.out[key][rows] = val

    # ------------------------------------------------------------------ batch: scores
    def score(self, rows):
        sp, st = self.spec, self.st
        ns, nc, nfix = self.ns, self.nc, self.nfix
        start = int(rows[0])
        t0 = time.perf_counter()
        sims, oldn = self.graph.append(self.sf[rows])
        self.sims, self.oldn, self.start = sims, oldn, start
        n = len(self.graph)
        new = np.arange(oldn, n)
        d, da = st["d"][nc + rows], st["d_all"][nc + rows]
        res = {}
        # --- frozen memory (admits inside, as Phase 4)
        rc0 = np.where(np.isfinite(self.mem0.best_cal[2]), 1.0 - self.mem0.best_cal[2].astype(np.float64), 2.0)
        res["pm0_cal"] = loo_high(self.mem0.dc - (rc0 - self.mem0.med) / self.mem0.mad)     # before this batch is admitted
        ps, masks = self.mem0.score(d, da, sims, ns, nfix, start)
        res["pm0"] = ps[3]
        self.put("M0", rows, _log(ps[3]))
        self.put("M0a1", rows, _log(ps[1]))
        self.put("M0a2", rows, _log(ps[2]))
        for key, m in zip(("M0a1", "M0a2", "M0"), masks):
            self.adm[key][rows] = m
        self.m0_mask = masks[2]
        # --- variants: read-out only (admission after the gates are known)
        res["pm"], res["pm_cal"] = {}, {}
        for x, mem in self.memx.items():
            p, gc = mem.score(d, sims, nfix)
            self.put(x, rows, _log(p))
            res["pm"][x], res["pm_cal"][x] = p, loo_high(gc)
            if mem.cen_k:
                pc, gcc, cs_, cc_ = mem.score_cen(d, sims, nfix, self.graph.X)
                self.put(x + "c", rows, _log(pc))
                # margins of raw cosines: local memory centroid against the nearest prototype / the local support centroid
                self.put(x + "cm", rows, _log(p_high(cc_ - st["pcos"][:nc], cs_ - st["pcos"][nc + rows])))
                ks = self.spec["cen_k"]
                ts, ti = sims[:, :ns].topk(ks, dim=1)
                sup_b = prefix_cos(ts, ti, self.graph.X).double().cpu().numpy()
                self.put(x + "cs", rows, _log(p_high(cc_ - self.cal_supcos, cs_ - sup_b)))
        # --- all-stream statistics
        k3 = min(3, oldn - nfix)
        s_ref = sims[:, :ns].topk(3, dim=1).values
        if k3 > 0:
            s_str = sims[:, nfix:oldn].topk(k3, dim=1).values
            if k3 < 3:
                s_str = torch.cat([s_str, torch.full((len(rows), 3 - k3), -1.0, device=sims.device)], 1)
        else:
            s_str = torch.full((len(rows), 3), -1.0, device=sims.device)
        cs, cr = self.cal_str, self.cal_ref
        f64 = lambda t: t.double().cpu().numpy()
        self.put("ctr1", rows, _log(p_high(f64(cs[:, 0] - cr[:, 0]), f64(s_str[:, 0] - s_ref[:, 0]))))
        self.put("ctr3", rows, _log(p_high(f64(cs.mean(1) - cr.mean(1)), f64(s_str.mean(1) - s_ref.mean(1)))))
        g = d - ((1.0 - f64(s_str[:, 0])) - st["med"]) / st["mad"]
        gc = st["d_cal"] - ((1.0 - f64(cs[:, 0])) - st["med"]) / st["mad"]
        self.put("gall", rows, _log(p_high(gc, g)))
        both = torch.cat([cs, sims[:, ns:nfix].T], dim=1)
        self.cal_str = both.topk(3, dim=1).values
        if sp["dyn"]:
            self.CS = sims[:, ns:nfix].T.clone() if self.CS is None else torch.cat([self.CS, sims[:, ns:nfix].T], dim=1)
        t_mem = time.perf_counter() - t0
        # --- label propagation
        t0 = time.perf_counter()
        dev = self.device
        y = torch.zeros(n, dtype=torch.float64, device=dev)
        y[:ns] = 1.0
        ref = sp["ref"]
        res["pneg"], res["pneg_cal"] = {}, {}
        seeds = {}
        for x in sp["neg"]:
            ids = self.mem0.mem[2] if x == "M0" else self.memx[x].ids
            # members admitted in earlier batches only (M0 has already taken this batch: drop rows >= start)
            ids = [i for i in ids if i < start]
            yn = torch.zeros(n, dtype=torch.float64, device=dev)
            if ids:
                yn[torch.as_tensor(np.asarray(ids) + nfix, device=dev)] = 1.0
            seeds[x] = yn
        for (k, g_, b) in sp["graphs"]:
            M = self.graph.matrix2(k, g_, b)
            lams = [l for (kk, gg, bb, l) in sp["cfgs"] if (kk, gg, bb) == (k, g_, b)]
            cols, u0, lam, names = [], [], [], []
            for l in lams:
                cols.append(y); u0.append(torch.zeros_like(y)); lam.append(l); names.append(("lp", l))
                if (k, g_, b, l) in sp["warm"]:
                    key = (k, g_, b, l)
                    if key not in self.warm:
                        w0 = y.clone()
                    else:
                        w0 = torch.zeros(n, dtype=torch.float64, device=dev)
                        w0[:oldn] = self.warm[key]
                    cols.append(y); u0.append(w0); lam.append(l); names.append(("warm", l))
            for l in [l for (kk, gg, bb, l) in sp["negcfgs"] if (kk, gg, bb) == (k, g_, b)]:
                for x in sp["neg"]:
                    cols.append(seeds[x]); u0.append(torch.zeros_like(y)); lam.append(l); names.append(("neg", (x, l)))
            if not cols:
                continue
            U = propagate(M, torch.stack(cols, 1), torch.tensor(lam, dtype=torch.float64, device=dev), torch.stack(u0, 1))
            Un = U.cpu().numpy()
            for j, (kind, a) in enumerate(names):
                u = Un[:, j]
                uc, un = u[ns:nfix], u[new]
                if kind == "lp":
                    t = cfg_tag(k, g_, b, a)
                    p = p_low(uc, un)
                    self.put(f"lp|{t}", rows, _log(p))
                    med = np.median(uc)
                    mad = MAD * np.median(np.abs(uc - med))
                    self.put(f"z|{t}", rows, log_ndtr((un - med) / max(mad, 1e-300)))
                    if (k, g_, b, a) == ref:
                        res["plp"], res["plp_cal"] = p, loo_low(uc)
                        res["plp_old"] = p_low(uc, u[nfix:oldn])              # current rank of every earlier stream node
                        self._Mref = M
                    if sp.get("classcond", True):
                        self._classcond(t, rows, u, a, new)
                elif kind == "warm":
                    t = cfg_tag(k, g_, b, a)
                    self.put(f"lp0|{t}", rows, _log(p_low(uc, un)))
                    self.warm[(k, g_, b, a)] = U[:, j].clone()
                else:
                    x, l = a
                    p = p_high(uc, un)
                    self.put(f"neg:{x}|{cfg_tag(k, g_, b, l)}", rows, _log(p))
                    if (k, g_, b, l) == ref:
                        res["pneg"][x], res["pneg_cal"][x] = p, loo_high(uc)
            if sp.get("tier2") and (k, g_, b) == ref[:3]:
                self._tier2(M, rows, new, n, ref)
        self.timing.append((len(self.graph), t_mem, time.perf_counter() - t0))
        self._res = res
        return res

    def _classcond(self, t, rows, u, lam, new):
        """Received mass r = S u (u / lam for unlabelled nodes) relative to the support nodes of the nearest class."""
        ns, nc, nfix = self.ns, self.nc, self.nfix
        r_sup = (u[:ns] - (1.0 - lam)) / lam
        r_cal, r_new = u[ns:nfix] / lam, u[new] / lam
        ref = np.median(r_sup.reshape(self.C, -1), axis=1)                         # per class
        ref = np.maximum(ref, 1e-300)
        cc, cn = self.cls_cal, self.cls_sf[rows]
        with np.errstate(divide="ignore"):
            lr_cal, lr_new = np.log(r_cal) - np.log(ref[cc]), np.log(r_new) - np.log(ref[cn])
        self.put(f"cr|{t}", rows, _log(p_low(lr_cal, lr_new)))
        dc_, dn_ = r_cal - ref[cc], r_new - ref[cn]
        self.put(f"cz|{t}", rows, _log(p_low(dc_, dn_)))

    def _tier2(self, M, rows, new, n, ref):
        """Class-resolved propagation and propagated embedding on the reference graph (float32)."""
        ns, nc, nfix, dev = self.ns, self.nc, self.nfix, self.device
        lam = ref[3]
        M32 = M.float()
        t = cfg_tag(*ref)
        idx = torch.as_tensor(np.r_[np.arange(ns, nfix), new], device=dev)
        Y = torch.zeros((n, self.C), dtype=torch.float32, device=dev)
        Y[torch.arange(ns, device=dev), torch.as_tensor(self.sup_cls, device=dev)] = 1.0
        U = torch.zeros_like(Y)
        for _ in range(15):
            U = lam * torch.sparse.mm(M32, U) + (1.0 - lam) * Y
        Ur = U[idx]
        del U, Y
        tot = Ur.sum(1)
        cmax = Ur.max(1).values
        cls = torch.as_tensor(np.r_[self.cls_cal, self.cls_sf[rows]], device=dev)
        cown = Ur.gather(1, cls[:, None])[:, 0]
        f = lambda v: v.double().cpu().numpy()
        for key, v in (("cmax", cmax), ("cown", cown), ("cpur", cmax / tot.clamp_min(1e-30)), ("ctot", tot)):
            v = f(v)
            self.put(f"{key}|{t}", rows, _log(p_low(v[:nc], v[nc:])))
        del Ur
        X = self.graph.X
        E = torch.zeros_like(X)
        for _ in range(sp_ep_sweeps(self.spec)):
            E = lam * torch.sparse.mm(M32, E) + (1.0 - lam) * X
        Er = E[idx]
        del E
        Er = Er / Er.norm(dim=1, keepdim=True).clamp_min(1e-30)
        ps = self.st["proto"]
        mu = torch.as_tensor(ps["mu"], dtype=torch.float32, device=dev)
        dist = 1.0 - Er @ mu.T
        z = (dist - torch.as_tensor(ps["med_t"], dtype=torch.float32, device=dev)[None]) / \
            torch.as_tensor(ps["mad_t"], dtype=torch.float32, device=dev)[None]
        zmin, dmin = f(z.min(1).values), f(dist.min(1).values)
        self.put(f"ep|{t}", rows, _log(p_high(zmin[:nc], zmin[nc:])))
        self.put(f"epc|{t}", rows, _log(p_high(dmin[:nc], dmin[nc:])))

    # ------------------------------------------------------------------ batch: dynamic sets (second pass)
    def refine(self, raw, m, k=10):
        """raw: bool over earlier stream rows. Keeps the rows with at least m raw members among their k nearest nodes."""
        if m <= 0 or not raw.any():
            return raw
        dev = self.device
        n = len(self.graph)
        is_seed = torch.zeros(n, dtype=torch.bool, device=dev)
        is_seed[self.nfix + torch.as_tensor(np.flatnonzero(raw), device=dev)] = True
        cnt = is_seed[self.graph.I[self.nfix:self.oldn, :k]].sum(1).cpu().numpy()
        return raw & (cnt >= m)

    def second(self, rows, sets, weights=None, want=()):
        """sets: {name: bool over earlier stream rows}; weights: {name: seed weights over the same rows} (default 1).
        Memory read-out against the set and the mass propagated from it (reference graph), both ranked against the
        calibration images under the same set. Returns {name: (p of the received mass of every earlier stream node,
        leave-one-out p of the calibration images)} for the names in `want`."""
        if not sets:
            return {}
        weights = weights or {}
        st, ns, nc, nfix, oldn, dev = self.st, self.ns, self.nc, self.nfix, self.oldn, self.device
        n = len(self.graph)
        new = np.arange(oldn, n)
        d = st["d"][nc + rows]
        names = list(sets)
        pmem = {}
        Y = torch.zeros((n, len(names)), dtype=torch.float64, device=dev)
        for j, name in enumerate(names):
            mask = sets[name]
            self.dyn_last[name] = mask
            if mask.any():
                idx = torch.as_tensor(np.flatnonzero(mask), device=dev)
                w = weights.get(name)
                Y[nfix + idx, j] = 1.0 if w is None else torch.as_tensor(w[mask], dtype=torch.float64, device=dev)
                rs = (1.0 - self.sims[:, nfix:oldn].index_select(1, idx).max(1).values).cpu().numpy().astype(np.float64)
                rc = (1.0 - self.CS[:, :oldn - nfix].index_select(1, idx).max(1).values).cpu().numpy().astype(np.float64)
            else:
                rs, rc = np.full(len(rows), 2.0), np.full(nc, 2.0)
            g = d - (rs - st["med"]) / st["mad"]
            gc = st["d_cal"] - (rc - st["med"]) / st["mad"]
            pmem[name] = (p_high(gc, g), loo_high(gc))
            self.put(name, rows, _log(pmem[name][0]))
        lamv = self.spec["ref"][3]
        lam = torch.full((len(names),), lamv, dtype=torch.float64, device=dev)
        Ut = propagate(self._Mref, Y, lam, torch.zeros_like(Y))
        out = {}
        if want:
            recv = ((Ut - (1.0 - lamv) * Y) / lamv).cpu().numpy()            # mass received from the neighbours
        U = Ut.cpu().numpy()
        r0 = self._res
        for j, name in enumerate(names):
            pn, pn_c = p_high(U[ns:nfix, j], U[new, j]), loo_high(U[ns:nfix, j])
            self.put(f"neg:{name}", rows, _log(pn))
            if self.spec.get("recal"):
                # partial re-calibration: one conformal rank for a group of factors (calibration side leave-one-out)
                self.put(f"ln:{name}", rows, _log(comb_p([r0["plp"], pn], [r0["plp_cal"], pn_c])))
                self.put(f"m0ln:{name}", rows, _log(comb_p([r0["pm0"], r0["plp"], pn], [r0["pm0_cal"], r0["plp_cal"], pn_c])))
                self.put(f"all:{name}", rows, _log(comb_p([r0["pm0"], pmem[name][0], r0["plp"], pn],
                                                           [r0["pm0_cal"], pmem[name][1], r0["plp_cal"], pn_c])))
            if name in want:
                rc_, ro_ = recv[ns:nfix, j], recv[nfix:oldn, j]
                out[name] = (p_high(rc_, ro_), loo_high(rc_))
        return out

    def count_near(self, mask, k=10):
        """Number of members of `mask` (bool over earlier stream rows) among the k nearest nodes of each earlier row."""
        dev = self.device
        is_m = torch.zeros(len(self.graph), dtype=torch.bool, device=dev)
        if mask.any():
            is_m[self.nfix + torch.as_tensor(np.flatnonzero(mask), device=dev)] = True
        return is_m[self.graph.I[self.nfix:self.oldn, :k]].sum(1).cpu().numpy()

    # ------------------------------------------------------------------ batch: admission
    def admit(self, rows, gates):
        for x, mask in gates.items():
            self.memx[x].admit(mask, self.start, self.sims, self.ns, self.nfix)
            self.adm[x][rows] = mask
        self.sims = None


def sp_ep_sweeps(spec):
    return int(spec.get("ep_sweeps", 15))


def bh_reject(p, q, storey=False, lam=0.5):
    """Benjamini-Hochberg rejections at level q (conformal p-values: Bates et al. 2023); storey=True divides the
    level by the Storey estimate of the null proportion, pi0 = (1 + #{p > lam}) / (n (1 - lam))."""
    p = np.asarray(p, np.float64)
    n = len(p)
    if n == 0:
        return np.zeros(0, bool)
    pi0 = min(1.0, (1.0 + float((p > lam).sum())) / (n * (1.0 - lam))) if storey else 1.0
    ps = np.sort(p)
    ok = ps <= q * np.arange(1, n + 1) / (n * pi0)
    if not ok.any():
        return np.zeros(n, bool)
    return p <= ps[np.flatnonzero(ok).max()]


def dyn_tag(stat, e, m, union):
    return f"D{stat}{e:g}m{m}" + ("u" if union else "")


def dyn_entry(x):
    """Dynamic seed sets, re-decided at every batch from the current evidence of the earlier stream images.
      (stat, eps, m, union)                       -> kind 'joint' (as round 3)
      dict(kind='hyst', lo, hi, m)                 strong (p <= lo) or weak (p <= hi) with >= m strong among 10 neighbours
      dict(kind='soft', eps[, w0])                 seed weight max(log(eps / p), 0) + w0
      dict(kind='own' | 'any' | 'all', eps)        per-view p_LP: the view's own / any view / all views
      dict(kind='bh', q[, storey])                 Benjamini-Hochberg rejections at level q among the earlier images
      dict(kind='iter', base=<set name>, eps | q)  second round: p(prod_views p_LP x p of the mass received from base),
                                                   thresholded at eps or by Benjamini-Hochberg at level q"""
    if isinstance(x, dict):
        d = dict(x)
        if "name" not in d:
            k = d["kind"]
            if k == "hyst":
                d["name"] = f"H{d['lo']:g}-{d['hi']:g}m{d['m']}"
            elif k == "soft":
                d["name"] = f"W{d['eps']:g}" + (f"+{d['w0']:g}" if d.get("w0") else "")
            elif k == "iter":
                d["name"] = (f"IB{d['q']:g}" + ("s" if d.get("storey") else "") if "q" in d else f"I{d['eps']:g}") + f"<{d['base']}"
            elif k == "bh":
                d["name"] = f"B{d['q']:g}" + ("s" if d.get("storey") else "")
            else:
                d["name"] = f"D{k}{d['eps']:g}"
        return d
    stat, e, m, union = x
    return {"kind": "joint", "stat": stat, "eps": e, "m": m, "union": union, "name": dyn_tag(stat, e, m, union)}


def comb_p(parts, parts_cal):
    """Conformal p of a sum of log p-values: calibration side uses the leave-one-out ranks."""
    t = sum(_log(p) for p in parts)
    tc = sum(_log(p) for p in parts_cal)
    return p_low(tc, t)


def make_spec(cfgs=((10, 1.0, 1.0, 0.9),), ref=(10, 1.0, 1.0, 0.9), warm=None, eps_lp=(), eps_fin=(), eps_joint=(),
              eps_full=(), frz=False, oracle=False, neg=("M0",), negcfgs=None, cen=(), cen_k=10, dyn=(), recal=False,
              tier2=False,
              thresholds=(0.3, 0.2, 0.10191613435745239), classcond=True, ep_sweeps=15):
    """cfgs: (k, gamma, beta, lambda) read-outs of the support mass; ref: the configuration whose rank feeds the gates;
    negcfgs: configurations of the OOD-seed mass (default: ref). Admission variants:
      Mlp<e>    frozen third stage or p_LP <= e                 Mfin<e>   the view's own p(p_M x p_LP) <= e
      Mjoint<e> p(prod_views p_M x p_LP) <= e                  Mfull<e>  p(prod_views p_M x p_LP x p_neg) <= e
      Mfrz      the frozen third-stage set (for extra read-outs of the frozen memory)
    dyn: dynamic sets (stat, eps, m, union): at every batch, the earlier stream images whose current evidence
      (stat 'lp': prod_views current p_LP; 'lpst': x static p) has a calibration rank <= eps, kept when at least m of
      their 10 nearest nodes are also in the raw set; union=True adds the frozen memory M0. Keys D<stat><eps>m<m>[u]."""
    cfgs = [tuple(c) for c in cfgs]
    negcfgs = [tuple(c) for c in (negcfgs if negcfgs is not None else [ref])]
    assert ref in cfgs
    graphs = []
    for (k, g, b, l) in cfgs + negcfgs:
        if (k, g, b) not in graphs:
            graphs.append((k, g, b))
    variants = [f"Mlp{e:g}" for e in eps_lp] + [f"Mfin{e:g}" for e in eps_fin] + [f"Mjoint{e:g}" for e in eps_joint] + \
               [f"Mfull{e:g}" for e in eps_full] + (["Mfrz"] if frz else []) + (["Mora"] if oracle else [])
    neg = [x for x in neg if x == "M0" or x in variants]
    for e in eps_full:
        if f"Mfull{e:g}" not in neg:
            neg.append(f"Mfull{e:g}")
        assert ref in negcfgs
    return {"graphs": graphs, "cfgs": cfgs, "ref": ref, "warm": set(warm if warm is not None else [ref]),
            "eps_lp": tuple(eps_lp), "eps_fin": tuple(eps_fin), "eps_joint": tuple(eps_joint), "eps_full": tuple(eps_full),
            "frz": frz, "variants": variants, "neg": neg, "negcfgs": negcfgs, "cen": [x for x in cen if x in variants],
            "cen_k": cen_k, "dyn": [dyn_entry(x) for x in dyn], "recal": recal, "tier2": tier2, "thresholds": thresholds, "classcond": classcond, "ep_sweeps": ep_sweeps}


def run5(views, bidx, spec, device="cuda", diag_oracle=None):
    """views: {name: (sup C x n x D, cal nc x D, sf N x D, static5 dict)}; bidx: batch of every stream row.
    Returns {name: {key: log p over stream rows}}, {name: {variant: admission mask}}, timing.
    diag_oracle (bool per stream row, DIAGNOSTIC ONLY): admission mask of the variant 'Mora' (the true OOD flag);
    its read-outs measure head-room and are never part of a method."""
    V = {name: View(name, *v, spec, device) for name, v in views.items()}
    names = list(V)
    for b in np.unique(bidx):
        rows = np.flatnonzero(bidx == b)
        assert np.array_equal(rows, np.arange(rows[0], rows[0] + len(rows)))
        R = {name: V[name].score(rows) for name in names}
        gates = {name: {} for name in names}
        for e in spec["eps_lp"]:
            x = f"Mlp{e:g}"
            for name in names:
                gates[name][x] = V[name].m0_mask | (R[name]["plp"] <= e)
        for e in spec["eps_fin"]:
            x = f"Mfin{e:g}"
            for name in names:
                r = R[name]
                gates[name][x] = comb_p([r["pm"][x], r["plp"]], [r["pm_cal"][x], r["plp_cal"]]) <= e
        for e in spec["eps_joint"]:
            x = f"Mjoint{e:g}"
            parts = [R[name]["pm"][x] for name in names] + [R[name]["plp"] for name in names]
            parts_c = [R[name]["pm_cal"][x] for name in names] + [R[name]["plp_cal"] for name in names]
            mask = comb_p(parts, parts_c) <= e
            for name in names:
                gates[name][x] = mask
        for e in spec["eps_full"]:
            x = f"Mfull{e:g}"
            parts = [R[name][key][x] for name in names for key in ("pm", "pneg")] + [R[name]["plp"] for name in names]
            parts_c = [R[name][key][x] for name in names for key in ("pm_cal", "pneg_cal")] + [R[name]["plp_cal"] for name in names]
            mask = comb_p(parts, parts_c) <= e
            for name in names:
                gates[name][x] = mask
        if spec["frz"]:
            for name in names:
                gates[name]["Mfrz"] = V[name].m0_mask
        if "Mora" in spec["variants"]:
            for name in names:
                gates[name]["Mora"] = np.asarray(diag_oracle, bool)[rows]
        if spec["dyn"]:
            start = int(rows[0])
            if start > 0:
                lp_o = {name: _log(R[name]["plp_old"]) for name in names}
                lp_c = {name: _log(R[name]["plp_cal"]) for name in names}
                lp_old, lp_cal = sum(lp_o.values()), sum(lp_c.values())
                st_old = sum(_log(V[name].st["p"][V[name].nc:V[name].nc + start]) for name in names)
                st_cal = sum(_log(loo_high(V[name].st["d_cal"])) for name in names)
                p_joint = p_low(lp_cal, lp_old)
            pending, done = list(spec["dyn"]), set()
            got = {name: {} for name in names}
            while pending:                                   # rounds: a set that builds on another one comes later
                ready = [dd for dd in pending if dd["kind"] != "iter" or dd["base"] in done]
                assert ready, "dynamic sets: unknown or circular base"
                pending = [dd for dd in pending if not any(dd is r_ for r_ in ready)]
                sets = {name: {} for name in names}
                wts = {name: {} for name in names}
                for dd in ready:
                    key, kind = dd["name"], dd["kind"]
                    for name in names:
                        if start == 0:
                            sets[name][key] = np.zeros(0, bool)
                            continue
                        if kind == "joint":
                            raw = (p_joint if dd["stat"] == "lp" else p_low(lp_cal + st_cal, lp_old + st_old)) <= dd["eps"]
                            r = V[name].refine(raw, dd["m"])
                            if dd.get("union"):
                                r = r.copy()
                                r[[i for i in V[name].mem0.mem[2] if i < start]] = True
                        elif kind == "hyst":
                            lo, hi = p_joint <= dd["lo"], p_joint <= dd["hi"]
                            r = lo | (hi & (V[name].count_near(lo) >= dd["m"]))
                        elif kind == "soft":
                            r = p_joint <= dd["eps"]
                            wts[name][key] = np.maximum(np.log(dd["eps"]) - _log(p_joint), 0.0) + dd.get("w0", 0.0)
                        elif kind == "own":
                            r = R[name]["plp_old"] <= dd["eps"]
                        elif kind == "bh":
                            r = bh_reject(p_joint, dd["q"], dd.get("storey", False))
                        elif kind in ("any", "all"):
                            ms = np.stack([R[v_]["plp_old"] <= dd["eps"] for v_ in names])
                            r = ms.any(0) if kind == "any" else ms.all(0)
                        elif kind == "iter":                 # current support mass x mass received from the base set
                            b = dd["base"]
                            p2 = p_low(lp_cal + sum(_log(got[v_][b][1]) for v_ in names),
                                       lp_old + sum(_log(got[v_][b][0]) for v_ in names))
                            r = bh_reject(p2, dd["q"], dd.get("storey", False)) if "q" in dd else p2 <= dd["eps"]
                        else:
                            raise ValueError(kind)
                        sets[name][key] = r
                want = {dd["base"] for dd in pending if dd["kind"] == "iter"}
                for name in names:
                    got[name].update(V[name].second(rows, sets[name], wts[name], want))
                done |= {dd["name"] for dd in ready}
        for name in names:
            V[name].admit(rows, gates[name])
    out = {name: V[name].out for name in names}
    adm = {name: V[name].adm for name in names}
    for name in names:                                   # dynamic sets: membership at the last batch (diagnostics)
        for key, mask in V[name].dyn_last.items():
            a = np.zeros(V[name].nt, bool)
            a[:len(mask)] = mask
            adm[name][key] = a
    timing = {name: np.asarray(V[name].timing) for name in names}
    return out, adm, timing
