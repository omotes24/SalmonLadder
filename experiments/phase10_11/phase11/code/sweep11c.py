"""Phase 11c: refinement of the transductive two-sided read-out (support mass x self-labelled OOD-seed mass).
Per view (L14, D3L, D3B, B14, fused L14xD3L), graph k in {10, 20} (beta 1, lambda 0.9 for the support mass):
  LP          calibrated rank of the support mass
  round 1     seeds_1 = Storey-BH(q) on the LP p-values of the stream images, q in {0.05, 0.1, 0.2}
              NEG1(q,lamn) = rank (p_high) of the mass propagated from seeds_1 with lambda_neg in {0.9, 0.99}
              NN1(q)       = rank (p_high) of the engine's closeness g to the seed set (nearest seed, self excluded)
  round 2     p_1 = calibrated rank of log LP + log NEG1(q, 0.9) among the calibration images; seeds_2 = Storey-BH(q)
              NEG2(q) = rank of the mass propagated from seeds_2
Output: P11/results/sweep_c/<part>/<stream>.npz with s::<view>::<key> (float32), is_ood, logS, sample_id."""
import argparse
import json
import math
import os
import time

import numpy as np
import torch

import p11common as P
import engine5 as E5
from engine import _log, p_high, p_low, propagate_converged
from sweep11 import l2n, storey_bh

KS, QS, LAMNS = (10, 20), (0.05, 0.1, 0.2), (0.9, 0.99)
LAM = 0.9


@torch.no_grad()
def nn_neg(X, ns, nfix, seed_idx, d_test, d_cal, med, mad, dev):
    """Closeness to the seed set (engine's g = d - (rho - med)/mad), seeds excluded from themselves; calibration rank."""
    Xt = torch.as_tensor(X, device=dev)
    S = Xt[torch.as_tensor(nfix + seed_idx, device=dev)]
    n = len(X) - nfix
    pos = -np.ones(n, np.int64)
    pos[seed_idx] = np.arange(len(seed_idx))
    rs = np.empty(n, np.float64)
    T = Xt[nfix:]
    for lo in range(0, n, 4096):
        hi = min(lo + 4096, n)
        s = T[lo:hi] @ S.T
        p = pos[lo:hi]
        own = np.flatnonzero(p >= 0)
        if len(own):
            s[torch.as_tensor(own, device=dev), torch.as_tensor(p[own], device=dev)] = -math.inf
        rs[lo:hi] = (1.0 - s.max(1).values).cpu().numpy().astype(np.float64)
    rc = (1.0 - (Xt[ns:nfix] @ S.T).max(1).values).cpu().numpy().astype(np.float64)
    g = d_test - (rs - med) / mad
    gc = d_cal - (rc - med) / mad
    return _log(p_high(gc, g))


@torch.no_grad()
def view_sweep(sup2, cal, sf, d_test, d_cal, med, mad, dev):
    ns, nc, nt = len(sup2), len(cal), len(sf)
    nfix = ns + nc
    X = np.concatenate([sup2, cal, sf]).astype(np.float32)
    graph = E5.Graph5(X, device=dev)
    out, meta = {}, {}
    ypos = torch.zeros((len(X), 1), dtype=torch.float64, device=dev)
    ypos[:ns, 0] = 1.0
    for k in KS:
        M = graph.matrix(k, 1.0, "sym")
        U, it, res = propagate_converged(M, ypos, LAM, max_iter=3000)
        u = U[:, 0].cpu().numpy()
        uc, ut = u[ns:nfix], u[nfix:]
        p = p_low(uc, ut)
        lp = _log(p)
        out[f"LP|k{k}"] = lp
        meta[f"k{k}"] = {"iters": int(it)}
        for q in QS:
            seeds = storey_bh(p, q)
            idx = np.flatnonzero(seeds)
            meta[f"k{k}"][f"seeds1_q{q:g}"] = int(len(idx))
            if len(idx) == 0:
                continue
            yneg = torch.zeros((len(X), 1), dtype=torch.float64, device=dev)
            yneg[torch.as_tensor(nfix + idx, device=dev), 0] = 1.0
            neg, un_all = {}, {}
            for lamn in LAMNS:
                Un, it2, _ = propagate_converged(M, yneg, lamn, max_iter=3000)
                un_all[lamn] = Un[:, 0].cpu().numpy()
                neg[lamn] = _log(p_high(un_all[lamn][ns:nfix], un_all[lamn][nfix:]))
                out[f"NEG1|k{k}|q{q:g}|l{lamn:g}"] = neg[lamn]
            if k == 10:
                out[f"NN1|q{q:g}"] = nn_neg(X, ns, nfix, idx, d_test, d_cal, med, mad, dev)
            # round 2: calibrated rank of the product among the calibration images, then new seeds
            un = un_all[0.9]
            s_cal = np.log(np.maximum(loo_rank_low(uc), 1e-300)) + np.log(np.maximum(loo_rank_high(un[ns:nfix]), 1e-300))
            s_test = lp + neg[0.9]
            p1 = p_low(s_cal, s_test)
            seeds2 = storey_bh(p1, q)
            idx2 = np.flatnonzero(seeds2)
            meta[f"k{k}"][f"seeds2_q{q:g}"] = int(len(idx2))
            if len(idx2):
                yneg2 = torch.zeros((len(X), 1), dtype=torch.float64, device=dev)
                yneg2[torch.as_tensor(nfix + idx2, device=dev), 0] = 1.0
                Un2, _, _ = propagate_converged(M, yneg2, 0.9, max_iter=3000)
                un2 = Un2[:, 0].cpu().numpy()
                out[f"NEG2|k{k}|q{q:g}"] = _log(p_high(un2[ns:nfix], un2[nfix:]))
        del M
    del graph
    torch.cuda.empty_cache()
    return out, meta


def loo_rank_low(c):
    c = np.asarray(c, np.float64)
    return np.searchsorted(np.sort(c), c, side="right") / len(c)


def loo_rank_high(c):
    c = np.asarray(c, np.float64)
    return (len(c) - np.searchsorted(np.sort(c), c, side="left")) / len(c)


def run_part(part, worker, nworkers, dev="cuda"):
    out_dir = P.RESULTS11 / "sweep_c" / part
    out_dir.mkdir(parents=True, exist_ok=True)
    tasks = P.stream_tasks(part)
    todo = [(ds, s, p) for (ds, s, p) in tasks[worker::nworkers] if not (out_dir / f"{ds}_seed{s}.npz").exists()]
    if not todo:
        return
    views = ["B14", "L14", "D3B", "D3L"]
    D = P.load_p3(part)
    P.add_views(D, part, ["D3B", "D3L"])
    cand = D["cands"][P.V5["K"]]
    from static import support_dall
    from vins import r5
    feats, stat = {}, {}
    for v in views:
        sup = D["shots"][v][:12000].reshape(1000, 12, -1)
        cal, ev = D["shots"][v][12000:16000], D["ev"][v]
        st = E5.static5(r5.proto_view, support_dall, sup, cal, ev, cand[:4000], cand[4000:], P.V5["n0"], P.V5["m"])
        feats[v] = (D["shots"][v][:12000], cal, ev)
        stat[v] = st
        print(json.dumps({"static": v, "utc": P.utc()}), flush=True)
    feats["L14xD3L"] = tuple(l2n(np.concatenate([l2n(feats["L14"][i]), l2n(feats["D3L"][i])], 1)) for i in range(3))
    stat["L14xD3L"] = stat["D3L"]   # the engine's closeness statistics of the D3L view (used by NN1 only)
    for (ds, seed, path) in todo:
        name = f"{ds}_seed{seed}"
        t0 = time.time()
        run = np.load(path, allow_pickle=True)
        ids, flag = run["sample_id"], run["is_ood"].astype(bool)
        rows = np.array([D["row"][x] for x in ids])
        arrays = {"sample_id": np.array(ids), "is_ood": flag, "logS": np.log(run["S_final"].astype(np.float64))}
        meta = {"part": part, "stream": name, "views": {}}
        for v, (sup, cal, ev) in feats.items():
            st = stat[v]
            out, m = view_sweep(sup, cal, ev[rows - 4000], st["d"][rows], st["d_cal"], st["med"], st["mad"], dev)
            for k, x in out.items():
                arrays[f"s::{v}::{k}"] = np.asarray(x, np.float32)
            meta["views"][v] = m
        tmp = out_dir / f"{name}.{os.getpid()}.tmp.npz"
        np.savez_compressed(tmp, meta=json.dumps(meta), **arrays)
        os.replace(tmp, out_dir / f"{name}.npz")
        print(json.dumps({"part": part, "stream": name, "seconds": round(time.time() - t0, 1),
                          "seeds": {v: m["k10"] for v, m in meta["views"].items()}, "utc": P.utc()}), flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", required=True, choices=["openood", "fourood"])
    ap.add_argument("--worker", type=int, default=0)
    ap.add_argument("--nworkers", type=int, default=1)
    a = ap.parse_args()
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    run_part(a.part, a.worker, a.nworkers)
