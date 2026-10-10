"""Phase 11b: sweep of transductive propagation variants on the full graph of every public-benchmark stream.
Per view (B14, L14, D3B, D3L) and per fused view (L14 (+) D3L, concatenated L2-normalised features):
  graph: k in {10, 20}, mutual-kNN weight beta in {1 (frozen), 0 (mutual)}; lambda in {0.9, 0.95, 0.99}
  read-outs: LP = calibrated rank (p_low) of the converged mass from the support; z = robust z (log Phi)
  negative seeds (base graph k10 beta1 lambda0.9): Storey-BH on the LP p-values of the stream images at q in {0.1, 0.2}
     -> mass propagated from the seeds u_neg: NEG = rank (p_high) of u_neg; RAT = rank (p_low) of log u - log u_neg
Output: P11/results/sweep/<part>/<stream>.npz with s::<view>::<key>; is_ood, logS (TINS), sample_id."""
import argparse
import json
import os
import time

import numpy as np
import torch
from scipy.special import log_ndtr

import p11common as P
import engine5 as E5
from engine import _log, p_high, p_low, propagate_converged
from static import support_dall
from vins import r5

MAD = 1.4826
KS, BETAS, LAMS = (10, 20), (1.0, 0.0), (0.9, 0.95, 0.99)
QS = (0.1, 0.2)
BASE = (10, 1.0, 0.9)


def l2n(x):
    x = np.asarray(x, np.float32)
    return x / np.linalg.norm(x, axis=1, keepdims=True)


def storey_bh(p, q):
    """Storey-BH selection (pi0 estimated at lambda = 0.5); returns the boolean mask of the rejected (OOD-like) p."""
    p = np.asarray(p, np.float64)
    n = len(p)
    pi0 = min(1.0, (np.sum(p > 0.5) + 1) / (n * 0.5))
    order = np.argsort(p)
    ps = p[order]
    thr = q * np.arange(1, n + 1) / (n * pi0)
    ok = np.flatnonzero(ps <= thr)
    mask = np.zeros(n, bool)
    if len(ok):
        mask[order[:ok[-1] + 1]] = True
    return mask


@torch.no_grad()
def view_sweep(sup2, cal, sf, dev):
    ns, nc, nt = len(sup2), len(cal), len(sf)
    nfix = ns + nc
    X = np.concatenate([sup2, cal, sf]).astype(np.float32)
    graph = E5.Graph5(X, device=dev)
    out, meta = {}, {}
    ypos = torch.zeros((len(X), 1), dtype=torch.float64, device=dev)
    ypos[:ns, 0] = 1.0
    base_p = None
    base_M = None
    for k in KS:
        for beta in BETAS:
            M = graph.matrix2(k, 1.0, beta)
            for lam in LAMS:
                U, it, res = propagate_converged(M, ypos, lam, max_iter=3000)
                u = U[:, 0].cpu().numpy()
                uc, ut = u[ns:nfix], u[nfix:]
                tag = f"k{k}b{beta:g}l{lam:g}"
                p = p_low(uc, ut)
                out[f"LP|{tag}"] = _log(p)
                med = np.median(uc)
                mad = MAD * np.median(np.abs(uc - med))
                out[f"z|{tag}"] = log_ndtr((ut - med) / max(mad, 1e-300))
                meta[tag] = {"iters": int(it), "res": float(res)}
                if (k, beta, lam) == BASE:
                    base_p, base_M, base_u = p, M, u
            if (k, beta) != BASE[:2]:
                del M
    # negative seeds on the base graph
    for q in QS:
        seeds = storey_bh(base_p, q)
        idx = np.flatnonzero(seeds)
        yneg = torch.zeros((len(X), 1), dtype=torch.float64, device=dev)
        if len(idx):
            yneg[torch.as_tensor(nfix + idx, device=dev), 0] = 1.0
            Un, it, res = propagate_converged(base_M, yneg, BASE[2], max_iter=3000)
            un = Un[:, 0].cpu().numpy()
            out[f"NEG|q{q:g}"] = _log(p_high(un[ns:nfix], un[nfix:]))
            r = np.log(np.maximum(base_u, 1e-300)) - np.log(np.maximum(un, 1e-300))
            out[f"RAT|q{q:g}"] = _log(p_low(r[ns:nfix], r[nfix:]))
        else:
            out[f"NEG|q{q:g}"] = np.zeros(nt)
            out[f"RAT|q{q:g}"] = out[f"LP|k10b1l0.9"]
        meta[f"seeds_q{q:g}"] = int(len(idx))
    del graph, base_M
    torch.cuda.empty_cache()
    return out, meta


def run_part(part, worker, nworkers, dev="cuda"):
    out_dir = P.RESULTS11 / "sweep" / part
    out_dir.mkdir(parents=True, exist_ok=True)
    tasks = P.stream_tasks(part)
    todo = [(ds, s, p) for (ds, s, p) in tasks[worker::nworkers] if not (out_dir / f"{ds}_seed{s}.npz").exists()]
    if not todo:
        return
    views = ["B14", "L14", "D3B", "D3L"]
    D = P.load_p3(part)
    P.add_views(D, part, ["D3B", "D3L"])
    feats = {v: (D["shots"][v][:12000], D["shots"][v][12000:16000], D["ev"][v]) for v in views}
    # fused views: concatenation of L2-normalised features, re-normalised
    feats["L14xD3L"] = tuple(l2n(np.concatenate([l2n(feats["L14"][i]), l2n(feats["D3L"][i])], 1)) for i in range(3))
    feats["B14xL14xD3L"] = tuple(l2n(np.concatenate([l2n(feats["B14"][i]), l2n(feats["L14"][i]), l2n(feats["D3L"][i])], 1)) for i in range(3))
    for (ds, seed, path) in todo:
        name = f"{ds}_seed{seed}"
        t0 = time.time()
        run = np.load(path, allow_pickle=True)
        ids, flag = run["sample_id"], run["is_ood"].astype(bool)
        rows = np.array([D["row"][x] for x in ids]) - 4000
        arrays = {"sample_id": np.array(ids), "is_ood": flag, "logS": np.log(run["S_final"].astype(np.float64))}
        meta = {"part": part, "stream": name, "views": {}}
        for v, (sup, cal, ev) in feats.items():
            out, m = view_sweep(sup, cal, ev[rows], dev)
            for k, x in out.items():
                arrays[f"s::{v}::{k}"] = np.asarray(x, np.float32)
            meta["views"][v] = m
        tmp = out_dir / f"{name}.{os.getpid()}.tmp.npz"
        np.savez_compressed(tmp, meta=json.dumps(meta), **arrays)
        os.replace(tmp, out_dir / f"{name}.npz")
        print(json.dumps({"part": part, "stream": name, "n": int(len(ids)), "seconds": round(time.time() - t0, 1),
                          "seeds": {v: (m["seeds_q0.1"], m["seeds_q0.2"]) for v, m in meta["views"].items()}, "utc": P.utc()}), flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", required=True, choices=["openood", "fourood"])
    ap.add_argument("--worker", type=int, default=0)
    ap.add_argument("--nworkers", type=int, default=1)
    a = ap.parse_args()
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    run_part(a.part, a.worker, a.nworkers)
