"""Phase 11d: Salmon Ladder-T candidate = joint multi-view transductive two-sided read-out, metrics computed in place.
Per stream and per view set:
  1. per view v: full graph (k_g = 10), converged support mass u_v, calibrated rank p_v (test), LOO rank r_v (calibration)
  2. joint score S1 = sum_v log p_v; calibrated rank of S1 among the calibration images; seeds = Storey-BH(q)
  3. per view: mass from the seeds on the graph with k_neg (5 or 10) -> rank NEG_v (test) and LOO rank (calibration);
     optional NN_v = rank of the closeness to the nearest seed (engine's g)
  4. S2 = sum_v (log p_v + log NEG_v [+ log NN_v]); optional smoothing over the k_g neighbours of each image:
     S2' = S2 + alpha * mean_{j in N(i)} S2_j (neighbours among the stream images; calibration smoothed likewise)
Scores x base (none / TINS) are evaluated with the upstream measures; rows go to P11/results/sweep_d/<part>/<stream>.csv"""
import argparse
import glob
import json
import math
import time

import numpy as np
import pandas as pd
import torch

import p11common as P
import engine5 as E5
from engine import _log, p_high, p_low, propagate_converged
from sweep11 import l2n, storey_bh
from sweep11c import loo_rank_high, loo_rank_low
from an11 import measures

KG = 10
KNEGS, QS, ALPHAS = (5, 10), (0.05, 0.1), (0.0, 0.5, 1.0)
VIEWSETS = [("L14", "D3L"), ("L14xD3L", "D3L"), ("L14", "D3B", "D3L"), ("B14", "L14", "D3L")]


class ViewGraph:
    def __init__(self, sup2, cal, sf, st, rows, dev):
        self.ns, self.nc, self.nt = len(sup2), len(cal), len(sf)
        self.nfix = self.ns + self.nc
        self.X = np.concatenate([sup2, cal, sf]).astype(np.float32)
        self.N = len(self.X)
        self.graph = E5.Graph5(self.X, device=dev)
        self.dev = dev
        self.M = {}
        self.d_test, self.d_cal, self.med, self.mad = st["d"][rows], st["d_cal"], st["med"], st["mad"]
        self._lp = None

    def matrix(self, k):
        if k not in self.M:
            self.M[k] = self.graph.matrix(k, 1.0, "sym")
        return self.M[k]

    @torch.no_grad()
    def mass(self, k, idx, lam=0.9):
        y = torch.zeros((self.N, 1), dtype=torch.float64, device=self.dev)
        y[torch.as_tensor(idx, device=self.dev), 0] = 1.0
        U, it, res = propagate_converged(self.matrix(k), y, lam, max_iter=3000)
        return U[:, 0].cpu().numpy()

    def ranks_low(self, u):
        return _log(p_low(u[self.ns:self.nfix], u[self.nfix:])), np.log(np.maximum(loo_rank_low(u[self.ns:self.nfix]), 1e-300))

    def ranks_high(self, u):
        return _log(p_high(u[self.ns:self.nfix], u[self.nfix:])), np.log(np.maximum(loo_rank_high(u[self.ns:self.nfix]), 1e-300))

    def lp(self):
        if self._lp is None:
            self._lp = self.ranks_low(self.mass(KG, np.arange(self.ns)))
        return self._lp

    @torch.no_grad()
    def nn_seed(self, seed_idx):
        """Closeness g to the nearest seed (self excluded) for the stream images and the calibration images; ranks."""
        Xt = torch.as_tensor(self.X, device=self.dev)
        S = Xt[torch.as_tensor(self.nfix + seed_idx, device=self.dev)]
        pos = -np.ones(self.nt, np.int64)
        pos[seed_idx] = np.arange(len(seed_idx))
        rs = np.empty(self.nt, np.float64)
        T = Xt[self.nfix:]
        for lo in range(0, self.nt, 4096):
            hi = min(lo + 4096, self.nt)
            s = T[lo:hi] @ S.T
            p = pos[lo:hi]
            own = np.flatnonzero(p >= 0)
            if len(own):
                s[torch.as_tensor(own, device=self.dev), torch.as_tensor(p[own], device=self.dev)] = -math.inf
            rs[lo:hi] = (1.0 - s.max(1).values).cpu().numpy().astype(np.float64)
        rc = (1.0 - (Xt[self.ns:self.nfix] @ S.T).max(1).values).cpu().numpy().astype(np.float64)
        g = self.d_test - (rs - self.med) / self.mad
        gc = self.d_cal - (rc - self.med) / self.mad
        return _log(p_high(gc, g)), np.log(np.maximum(loo_rank_high(gc), 1e-300))

    def neighbours(self):
        """k_g nearest stream neighbours (indices into the stream) of every stream image and of every calibration image."""
        I = self.graph.I[:, :KG].cpu().numpy()
        return I[self.nfix:], I[self.ns:self.nfix]

    def free(self):
        del self.graph, self.M, self.X
        torch.cuda.empty_cache()


def smooth(s_t, s_c, nb_t, nb_c, nfix, alpha):
    """Adds alpha x mean score of the graph neighbours that are stream images (others ignored)."""
    if alpha == 0.0:
        return s_t, s_c
    def mean_nb(nb):
        m = nb >= nfix
        idx = np.where(m, nb - nfix, 0)
        vals = s_t[idx] * m
        cnt = m.sum(1)
        return np.where(cnt > 0, vals.sum(1) / np.maximum(cnt, 1), 0.0)
    return s_t + alpha * mean_nb(nb_t), s_c + alpha * mean_nb(nb_c)


def run_stream(G, views, flag, logS, rows_out, tag):
    lp = {v: G[v].lp() for v in views}
    s1_t = sum(lp[v][0] for v in views)
    s1_c = sum(lp[v][1] for v in views)
    nb = {v: G[v].neighbours() for v in views}
    nfix = G[views[0]].nfix
    def emit(s_t, s_c, extra):
        for alpha in ALPHAS:
            st, sc = s_t, s_c
            if alpha > 0:
                # smooth with the neighbours of the first view of the set
                st, sc = smooth(s_t, s_c, nb[views[0]][0], nb[views[0]][1], nfix, alpha)
            for base, add in (("none", 0.0), ("TINS", logS)):
                au, fp = measures(st + add, flag)
                rows_out.append({**tag, **extra, "alpha": alpha, "base": base, "AUROC": au, "FPR95": fp})
    emit(s1_t, s1_c, {"q": 0.0, "kneg": 0, "nn": 0, "seeds": 0, "seed_ood_frac": float("nan")})
    for q in QS:
        p = p_low(s1_c, s1_t)
        seeds = np.flatnonzero(storey_bh(p, q))
        if len(seeds) == 0:
            continue
        ex = {"q": q, "seeds": int(len(seeds)), "seed_ood_frac": float(flag[seeds].mean())}
        nn = {v: G[v].nn_seed(seeds) for v in views}
        for kneg in KNEGS:
            neg = {v: G[v].ranks_high(G[v].mass(kneg, G[v].nfix + seeds)) for v in views}
            s2_t = s1_t + sum(neg[v][0] for v in views)
            s2_c = s1_c + sum(neg[v][1] for v in views)
            emit(s2_t, s2_c, {**ex, "kneg": kneg, "nn": 0})
            emit(s2_t + sum(nn[v][0] for v in views), s2_c + sum(nn[v][1] for v in views), {**ex, "kneg": kneg, "nn": 1})
        emit(s1_t + sum(nn[v][0] for v in views), s1_c + sum(nn[v][1] for v in views), {**ex, "kneg": 0, "nn": 1})


def run_part(part, worker, nworkers, dev="cuda"):
    out_dir = P.RESULTS11 / "sweep_d" / part
    out_dir.mkdir(parents=True, exist_ok=True)
    tasks = P.stream_tasks(part)
    todo = [(ds, s, p) for (ds, s, p) in tasks[worker::nworkers] if not (out_dir / f"{ds}_seed{s}.csv").exists()]
    if not todo:
        return
    D = P.load_p3(part)
    P.add_views(D, part, ["D3B", "D3L"])
    cand = D["cands"][P.V5["K"]]
    from static import support_dall
    from vins import r5
    feats, stat = {}, {}
    for v in ["B14", "L14", "D3B", "D3L"]:
        sup = D["shots"][v][:12000].reshape(1000, 12, -1)
        cal, ev = D["shots"][v][12000:16000], D["ev"][v]
        stat[v] = E5.static5(r5.proto_view, support_dall, sup, cal, ev, cand[:4000], cand[4000:], P.V5["n0"], P.V5["m"])
        feats[v] = (D["shots"][v][:12000], cal, ev)
    feats["L14xD3L"] = tuple(l2n(np.concatenate([l2n(feats["L14"][i]), l2n(feats["D3L"][i])], 1)) for i in range(3))
    stat["L14xD3L"] = stat["D3L"]
    for (ds, seed, path) in todo:
        name = f"{ds}_seed{seed}"
        t0 = time.time()
        run = np.load(path, allow_pickle=True)
        ids, flag = run["sample_id"], run["is_ood"].astype(bool)
        logS = np.log(run["S_final"].astype(np.float64))
        rows = np.array([D["row"][x] for x in ids])
        G = {v: ViewGraph(sup, cal, ev[rows - 4000], stat[v], rows, dev) for v, (sup, cal, ev) in feats.items()}
        out_rows = []
        for vs in VIEWSETS:
            run_stream(G, vs, flag, logS, out_rows, {"part": part, "stream": name, "views": "+".join(vs)})
        for g in G.values():
            g.free()
        pd.DataFrame(out_rows).to_csv(out_dir / f"{name}.csv", index=False)
        print(json.dumps({"part": part, "stream": name, "rows": len(out_rows), "seconds": round(time.time() - t0, 1), "utc": P.utc()}), flush=True)


def aggregate():
    rows = pd.concat([pd.read_csv(f) for part in ("openood", "fourood") for f in sorted(glob.glob(str(P.RESULTS11 / "sweep_d" / part / "*.csv")))])
    rows["ds"] = rows.stream.str.replace(r"_seed\d+$", "", regex=True)
    rows["seed"] = rows.stream.str.extract(r"seed(\d+)$", expand=False).astype(int)
    rows["cfg"] = (rows.views + "|q" + rows.q.astype(str) + "|kn" + rows.kneg.astype(str) + "|nn" + rows.nn.astype(str)
                   + "|a" + rows.alpha.astype(str) + "|" + rows.base)
    lines, out = [], {}
    for part, d in rows.groupby("part"):
        groups = {"near": P.OO_NEAR, "far": P.OO_FAR} if part == "openood" else {"fourood": P.FOUR}
        res = {}
        for g, sets in groups.items():
            dd = d[d.ds.isin(sets)]
            per = dd.groupby(["cfg", "seed", "ds"])[["AUROC", "FPR95"]].mean().groupby(["cfg", "seed"]).mean().groupby("cfg").mean()
            res[g] = {c: {"AUROC": float(r.AUROC), "FPR95": float(r.FPR95)} for c, r in per.iterrows()}
            lines.append(f"== {part} {g}: top 40 by FPR95 ==")
            for c, r in per.sort_values("FPR95").head(40).iterrows():
                lines.append(f"{c:48s} {r.AUROC:6.2f} / {r.FPR95:6.2f}")
            lines.append("")
        per_ds = d.groupby(["cfg", "ds"])[["AUROC", "FPR95"]].mean()
        res["per_dataset"] = {f"{c}|{ds}": {k: float(v) for k, v in r.items()} for (c, ds), r in per_ds.iterrows()}
        out[part] = res
    (P.RESULTS11 / "sweep_d" / "p11d_summary.json").write_text(json.dumps(out, indent=1) + "\n")
    t = "\n".join(lines)
    (P.RESULTS11 / "sweep_d" / "p11d_table.txt").write_text(t + "\n")
    print(t)
    print("AN11D_DONE", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", choices=["openood", "fourood"])
    ap.add_argument("--worker", type=int, default=0)
    ap.add_argument("--nworkers", type=int, default=1)
    ap.add_argument("--aggregate", action="store_true")
    a = ap.parse_args()
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    if a.aggregate:
        aggregate()
    else:
        run_part(a.part, a.worker, a.nworkers)
