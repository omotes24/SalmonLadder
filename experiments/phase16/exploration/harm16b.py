"""Phase 16 (exploration, dev1 only; no dev2, no test data): the absorbing random walk read-out.
h(x) = P_x(walk reaches a support image before a seed and before it is killed), killing 1 - lambda per step; lambda -> 1 without
seeds = ranking by the expected hitting time of the support set. Compared on the same tasks with the Phase 14 statistic sigma.
Configurations (standalone):
  cur_sigma     Phase 14 cross-fitted sigma (Storey q = 0.2, p+ p-, pooled) and its p_final        [reference]
  cur_lp        sum over views of log rank(u+) against all calibration images                         [reference]
  abs1_prod     sum over views of log rank(h+) (S absorbing, no seeds), lambda = 0.9
  abs1_union    h+ on the union graph W = W_L14 + W_D3L (one walk), lambda = 0.9 / 0.99
  hit_union     - E[tau_S] on the union graph (the lambda -> 1 limit of the ranking)
  abs2_xfit_*   seeds selected per fold as in Phase 14 (Storey q = 0.2 on the fold's calibrated ranks of h+), the nodes of fold b
                scored by h with S = 1 and the OTHER fold's seeds = 0; prod = per view + sum of log ranks, union = one walk
  abs2_pool_*   no folds: the 12th support image of each class is a labelled reference R (not absorbing); seeds = Storey q = 0.2
                over ALL unlabelled nodes (calibration + test) on their ranks against R; h with S(11/class) = 1, seeds = 0; the seeds
                themselves scored by the one-step look-ahead lambda (P h)(x); final p = rank among all calibration images
  dir*          the same on the directed k-NN walk (each node follows its own top-k_g list), union of the two views' walks
Output: P16/results/dev1/<task>.csv (one row per configuration) + meta.json."""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, "/home/omote/reprise_p14_20261011/code")
import p14common as P  # noqa: E402
import core14 as K
from engine import _log, p_low
from an11 import measures

sys.path.insert(0, "/home/omote/reprise_p5_20261002/code")
import dev5  # noqa: E402

VIEWS = ["L14", "D3L"]
P16 = Path("/home/omote/reprise_p16_20261011")
LAM1 = 1.0 - 1e-9


def sym_W(graph, k=P.KG):
    """W = max(A, A^T) of the top-k lists (same arithmetic as PrefixTopK.matrix 'sym', not normalised) and the degrees."""
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


def dir_P(graph, k=P.KG):
    """Row-stochastic directed walk on the node's own top-k list (weights = clamped cosine)."""
    n = len(graph.X)
    dev = graph.device
    rows = torch.arange(n, device=dev).repeat_interleave(k)
    cols = graph.I[:, :k].reshape(-1)
    vals = graph.S[:, :k].clamp_min(1e-12).double()
    vals = (vals / vals.sum(1, keepdim=True)).reshape(-1)
    return torch.sparse_coo_tensor(torch.stack([rows, cols]), vals, (n, n)).coalesce().to_sparse_csr()


@torch.no_grad()
def absorb_sym(W, deg, one, zero, lam, tol=1e-11, maxit=20000):
    """h = P_x(hit `one` before `zero` and before killing) on a symmetric graph, by Jacobi-preconditioned CG on
    (D - lam W)_FF h_F = lam W_F,one 1. one, zero: bool (N, B). Returns (h (N, B) float64, iterations, residual)."""
    free = (~(one | zero)).double()
    onef = one.double()
    b = lam * torch.sparse.mm(W, onef) * free
    dinv = (1.0 / deg.clamp_min(1e-30))[:, None]

    def A(x):
        return (deg[:, None] * x - lam * torch.sparse.mm(W, x)) * free
    x = torch.zeros_like(b)
    r = b.clone()
    z = r * dinv
    p = z.clone()
    rz = (r * z).sum(0)
    bn = torch.linalg.vector_norm(b, dim=0).clamp_min(1e-300)
    res = 1.0
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
    return x + onef, it + 1, res


@torch.no_grad()
def hit_time_sym(W, deg, target, tol=1e-11, maxit=20000):
    """E_x[tau_target] (expected hitting time of the target set) on a symmetric graph: (D - W)_FF t_F = D_F 1."""
    free = (~target).double()[:, None]
    b = deg[:, None] * free
    dinv = (1.0 / deg.clamp_min(1e-30))[:, None]

    def A(x):
        return (deg[:, None] * x - torch.sparse.mm(W, x)) * free
    x = torch.zeros_like(b)
    r = b.clone()
    z = r * dinv
    p = z.clone()
    rz = (r * z).sum(0)
    bn = torch.linalg.vector_norm(b, dim=0).clamp_min(1e-300)
    res = 1.0
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
    return x[:, 0], it + 1, res


@torch.no_grad()
def absorb_dir(Pm, one, zero, lam, tol=1e-11, maxit=20000):
    """Minimal non-negative solution of h = lam P h on the free nodes (h = 1 on one, 0 on zero), by monotone value iteration."""
    free = ~(one | zero)
    h = one.double().clone()
    it, d = 0, 1.0
    for it in range(maxit):
        hn = torch.where(free, lam * torch.sparse.mm(Pm, h), h)
        d = float((hn - h).abs().max())
        h = hn
        if d < tol:
            break
    return h, it + 1, d


@torch.no_grad()
def ground_sym(W, deg, one, zero, gamma, tol=1e-11, maxit=20000):
    """Grounded Dirichlet problem: every node leaks to ground through conductance gamma (the continuous-time walk with jump rates
    W_ij is killed at rate gamma). h = P_x(reach `one` before `zero` and before the leak) solves (L + gamma I)_FF h_F = W_F,one 1."""
    free = (~(one | zero)).double()
    onef = one.double()
    b = torch.sparse.mm(W, onef) * free
    dg = deg + gamma
    dinv = (1.0 / dg.clamp_min(1e-30))[:, None]

    def A(x):
        return (dg[:, None] * x - torch.sparse.mm(W, x)) * free
    x = torch.zeros_like(b)
    r = b.clone()
    z = r * dinv
    p = z.clone()
    rz = (r * z).sum(0)
    bn = torch.linalg.vector_norm(b, dim=0).clamp_min(1e-300)
    res = 1.0
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
    return x + onef, it + 1, res


@torch.no_grad()
def ct_hit_sym(W, deg, target, tol=1e-11, maxit=20000):
    """Expected hitting time of `target` for the continuous-time walk with jump rates W_ij (generator -L): L_FF T_F = 1_F."""
    free = (~target).double()[:, None]
    b = free.clone()
    dinv = (1.0 / deg.clamp_min(1e-30))[:, None]

    def A(x):
        return (deg[:, None] * x - torch.sparse.mm(W, x)) * free
    x = torch.zeros_like(b)
    r = b.clone()
    z = r * dinv
    p = z.clone()
    rz = (r * z).sum(0)
    bn = torch.linalg.vector_norm(b, dim=0).clamp_min(1e-300)
    res = 1.0
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
    return x[:, 0], it + 1, res


def run(dev, worker, nworkers, device="cuda"):
    out_dir = P16 / "results" / (dev + "_b")
    out_dir.mkdir(parents=True, exist_ok=True)
    D = dev5.load(dev)
    keys = sorted(D["tins"].keys())
    seen, tasks = set(), []
    for k in keys:                                          # standalone scores do not depend on the TINS seed: one per (draw, stream)
        if (k[0], k[1]) not in seen:
            seen.add((k[0], k[1]))
            tasks.append(k)
    tasks = tasks[worker::nworkers]
    print(json.dumps({"dev": dev, "tasks": [list(map(str, t)) for t in tasks], "utc": P.utc()}), flush=True)
    for (draw, stream, seed) in tasks:
        name = f"{dev}_draw{draw}_{stream}"
        if (out_dir / f"{name}.csv").exists():
            continue
        t0 = time.time()
        sid, flag, S_, bidx = D["tins"][(draw, stream, seed)]
        flag = np.asarray(flag, bool)
        rows = np.array([D["stream_row"][s] for s in sid])
        graphs, terms = {}, {}
        for v in VIEWS:
            sup, cal, sf = dev5.raw_view(D, draw, v, rows)
            sup, cal, sf = P.l2n(sup.reshape(-1, sup.shape[-1])), P.l2n(cal), P.l2n(sf)
            g = K.Graph(sup, cal, sf, device)
            graphs[v], terms[v] = g, K.view_terms(g)
        g0 = graphs[VIEWS[0]]
        ns, nc, nt, nfix, N = g0.ns, g0.nc, g0.nt, g0.nfix, g0.N
        n_cls = ns // 12
        cal_nodes, test_nodes = ns + np.arange(nc), nfix + np.arange(nt)
        res, meta = [], {"task": name, "n_id": int((~flag).sum()), "n_ood": int(flag.sum()), "ns": ns, "nc": nc, "nt": nt, "solver": {}}

        def add(cfg, score, pfin=None, extra=None):
            au, fp = measures(score, flag)
            row = {"cfg": cfg, "AUROC": au, "FPR95": fp}
            if pfin is not None:
                row.update({"fpr@0.05": float((pfin[~flag] <= 0.05).mean()), "tpr@0.05": float((pfin[flag] <= 0.05).mean()),
                            "fpr@0.2": float((pfin[~flag] <= 0.2).mean())})
            row.update(extra or {})
            res.append(row)
        # ---------------- references (Phase 14 machinery)
        out, m14 = K.xfit(graphs, terms, flag, folds=2, seed=0, pairs=[("storey", 0.2)], modes=("pp",))
        e = out[("storey", 0.2, "pp")]
        add("cur_sigma", e["s"], e["pfinal"])
        lp = sum(_log(p_low(terms[v]["u_pos"][cal_nodes], terms[v]["u_pos"][test_nodes])) for v in VIEWS)
        add("cur_lp", lp)
        fc, ft = m14["fold_of_cal"], m14["fold_of_test"]
        # ---------------- graphs for the absorbing walk
        Ws = {v: sym_W(graphs[v].graph) for v in VIEWS}
        Wu = Ws[VIEWS[0]][0] + Ws[VIEWS[1]][0]
        du = Ws[VIEWS[0]][1] + Ws[VIEWS[1]][1]
        Pd = {v: dir_P(graphs[v].graph) for v in VIEWS}
        Pu = (Pd[VIEWS[0]] + Pd[VIEWS[1]]) * 0.5
        dev_ = g0.dev
        S_all = torch.zeros(N, dtype=torch.bool, device=dev_)
        S_all[:ns] = True
        none = torch.zeros(N, dtype=torch.bool, device=dev_)
        # ---------------- uniform killing, stage 1 (reference from harm16 round 1)
        h, it, rr = absorb_sym(Wu, du, S_all[:, None], none[:, None], 0.9)
        h = h[:, 0].cpu().numpy()
        add("abs1_union_l0.9", h[test_nodes], p_low(h[cal_nodes], h[test_nodes]))
        # ---------------- continuous-time walk (jump rates W): expected hitting time of S
        T, it, rr = ct_hit_sym(Wu, du, S_all)
        T = T.cpu().numpy()
        meta["solver"]["ct_hit_union"] = [it, rr]
        add("ct_hit_union", -T[test_nodes], p_low(-T[cal_nodes], -T[test_nodes]))
        Tv = {}
        for v in VIEWS:
            t_, it, rr = ct_hit_sym(Ws[v][0], Ws[v][1], S_all)
            Tv[v] = t_.cpu().numpy()
        add("ct_hit_prod", sum(_log(p_low(-Tv[v][cal_nodes], -Tv[v][test_nodes])) for v in VIEWS))
        meta["deg_median"] = {"union": float(du.median()), **{v: float(Ws[v][1].median()) for v in VIEWS}}

        def xfit_seeds(hstage1):
            seeds = []
            for b in range(2):
                Cb, Tb = cal_nodes[fc == b], test_nodes[ft == b]
                mask, pi0 = P.select_seeds(p_low(hstage1[Cb], hstage1[Tb]), 0.2, len(Cb), "storey")
                seeds.append(Tb[mask])
            return seeds

        def zero_cols(seeds):                        # column b: zero boundary = seeds of the OTHER fold
            Z = torch.zeros((N, 2), dtype=torch.bool, device=dev_)
            for b in range(2):
                if len(seeds[1 - b]):
                    Z[torch.as_tensor(seeds[1 - b], device=dev_), b] = True
            return Z
        fold_node = np.empty(N, int)
        fold_node[cal_nodes], fold_node[test_nodes] = fc, ft
        R = np.arange(n_cls) * 12 + 11
        S11 = torch.zeros(N, dtype=torch.bool, device=dev_)
        S11[:ns] = True
        S11[torch.as_tensor(R, device=dev_)] = False
        pool = np.concatenate([cal_nodes, test_nodes])
        for gm in (0.3, 1.0, 3.0):
            # stage 1 grounded
            h, it, rr = ground_sym(Wu, du, S_all[:, None], none[:, None], gm)
            g1 = h[:, 0].cpu().numpy()
            meta["solver"][f"gnd1_union_g{gm}"] = [it, rr]
            add(f"gnd1_union_g{gm}", g1[test_nodes], p_low(g1[cal_nodes], g1[test_nodes]))
            gv = {}
            for v in VIEWS:
                hv, it, rr = ground_sym(Ws[v][0], Ws[v][1], S_all[:, None], none[:, None], gm / 2)
                gv[v] = hv[:, 0].cpu().numpy()
            add(f"gnd1_prod_g{gm}", sum(_log(p_low(gv[v][cal_nodes], gv[v][test_nodes])) for v in VIEWS))
            # stage 2 grounded, cross-fitted seeds (union walk)
            sd = xfit_seeds(g1)
            h, it, rr = ground_sym(Wu, du, S_all[:, None].expand(N, 2), zero_cols(sd), gm)
            h = h.cpu().numpy()
            s = h[np.arange(N), np.clip(fold_node, 0, 1)]
            pf = np.empty(nt)
            for b in range(2):
                tp = np.flatnonzero(ft == b)
                pf[tp] = p_low(s[cal_nodes[fc == b]], s[test_nodes[tp]])
            ns_ = int(sum(len(x) for x in sd))
            add(f"gnd2_xfit_union_g{gm}", s[test_nodes], pf, {"seeds": ns_, "seed_idfrac": float(np.mean([not flag[i - nfix] for x in sd for i in x])) if ns_ else None})
            # stage 2 grounded, per view + sum of log ranks against the other fold (the Phase 14 combination)
            sv = {}
            for v in VIEWS:
                sdv = xfit_seeds(gv[v])
                hv, it, rr = ground_sym(Ws[v][0], Ws[v][1], S_all[:, None].expand(N, 2), zero_cols(sdv), gm / 2)
                sv[v] = hv.cpu().numpy()
            stat = np.zeros(N)
            for b2 in range(2):
                nodes = np.concatenate([cal_nodes[fc == b2], test_nodes[ft == b2]])
                ref = cal_nodes[fc != b2]
                stat[nodes] = sum(_log(p_low(sv[v][ref, b2], sv[v][nodes, b2])) for v in VIEWS)
            pf = np.empty(nt)
            for b in range(2):
                tp = np.flatnonzero(ft == b)
                pf[tp] = p_low(stat[cal_nodes[fc == b]], stat[test_nodes[tp]])
            add(f"gnd2_xfit_prod_g{gm}", stat[test_nodes], pf)
            # stage 2 grounded, pool seeds against R (no folds)
            h, it, rr = ground_sym(Wu, du, S11[:, None], none[:, None], gm)
            g1p = h[:, 0].cpu().numpy()
            mask, pi0 = P.select_seeds(p_low(g1p[R], g1p[pool]), 0.2, len(R), "storey")
            seeds = pool[mask]
            Zp = torch.zeros(N, dtype=torch.bool, device=dev_)
            if len(seeds):
                Zp[torch.as_tensor(seeds, device=dev_)] = True
            h, it, rr = ground_sym(Wu, du, S11[:, None], Zp[:, None], gm)
            hh = h[:, 0]
            s = hh.clone()
            if len(seeds):
                look = torch.sparse.mm(Wu, hh[:, None])[:, 0] / (du + gm)
                s[Zp] = look[Zp]
            s = s.cpu().numpy()
            add(f"gnd2_pool_union_g{gm}", s[test_nodes], p_low(s[cal_nodes], s[test_nodes]),
                {"seeds": int(len(seeds)), "seed_idfrac": float(np.mean([(i < nfix) or (not flag[i - nfix]) for i in seeds])) if len(seeds) else None,
                 "cal_seeds": int(np.isin(seeds, cal_nodes).sum())})
        for g in graphs.values():
            g.free()
        torch.cuda.empty_cache()
        import pandas as pd
        pd.DataFrame(res).to_csv(out_dir / f"{name}.csv", index=False)
        meta["seconds"] = round(time.time() - t0, 1)
        (out_dir / f"{name}.meta.json").write_text(json.dumps(meta, indent=1, default=str) + "\n")
        print(json.dumps({"task": name, "FPR95": {r["cfg"]: round(r["FPR95"], 2) for r in res}, "seconds": meta["seconds"], "utc": P.utc()}), flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dev", default="dev1")
    ap.add_argument("--worker", type=int, default=0)
    ap.add_argument("--nworkers", type=int, default=1)
    a = ap.parse_args()
    assert a.dev == "dev1", "Phase 16 exploration touches dev1 only"
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    run(a.dev, a.worker, a.nworkers)
    print("HARM16B_DONE", flush=True)
