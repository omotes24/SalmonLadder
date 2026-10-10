"""Phase 11: transductive read-outs of Salmon Ladder on the public-benchmark streams. The streaming run (final10 /
Phase 5) fixes the memory M_T (final admissions a::<view>::M0); here every image of the stream is re-read at the end:
  Mfull   calibrated rank of the nearest-neighbour closeness to the FINAL memory (self excluded), same g as the engine
  LPfull  calibrated rank of the label-propagation mass on the FULL graph (support + calibration + all stream images),
          converged solution of u = lam S u + (1 - lam) y, k_g 10, gamma 1, lambda 0.9 (the frozen graph)
  zfull   robust z of the same u against the calibration images (log Phi)
  NEGfull calibrated rank (high = OOD) of the mass propagated from the final memory on the same graph
The streaming read-outs (M0, lp0, z) of the same stream are copied into the output so the analysis mixes both.
  --part openood|fourood  --views B14,L14,D3B,D3L  --worker w --nworkers n
Output: P11/results/<part>/<stream>.npz"""
import argparse
import json
import math
import os
import time

import numpy as np
import torch
from scipy.special import log_ndtr

import p11common as P
import engine5 as E5
from engine import PrefixTopK, _log, p_high, p_low, propagate_converged
from static import support_dall
from vins import r5

MAD = 1.4826
LAM, KG, GAMMA = 0.9, 10, 1.0


def statics(D, views, cand):
    full = {}
    for v in views:
        sup = D["shots"][v][:12000].reshape(1000, 12, -1)
        cal, ev = D["shots"][v][12000:16000], D["ev"][v]
        t0 = time.time()
        st = E5.static5(r5.proto_view, support_dall, sup, cal, ev, cand[:4000], cand[4000:], P.V5["n0"], P.V5["m"])
        full[v] = (sup, cal, ev, st)
        print(json.dumps({"static": v, "rows": int(len(ev)), "seconds": round(time.time() - t0, 1), "utc": P.utc()}), flush=True)
    return full


@torch.no_grad()
def memory_full(sf, cal, mem_idx, d_test, d_cal, med, mad, dev):
    """Transductive memory read-out: closeness of every stream image (self excluded) and of every calibration image
    to the final memory, standardised as the engine does, read as the calibration rank (p_high)."""
    n = len(sf)
    if len(mem_idx) == 0:
        return np.ones(n), np.full(n, 2.0)
    X = torch.as_tensor(np.ascontiguousarray(sf, np.float32), device=dev)
    XM = X[torch.as_tensor(mem_idx, device=dev)]
    pos = -np.ones(n, np.int64)
    pos[mem_idx] = np.arange(len(mem_idx))
    rs = np.empty(n, np.float64)
    for lo in range(0, n, 4096):
        hi = min(lo + 4096, n)
        s = X[lo:hi] @ XM.T
        p = pos[lo:hi]
        own = np.flatnonzero(p >= 0)
        if len(own):
            s[torch.as_tensor(own, device=dev), torch.as_tensor(p[own], device=dev)] = -math.inf
        rs[lo:hi] = (1.0 - s.max(1).values).cpu().numpy().astype(np.float64)
    C = torch.as_tensor(np.ascontiguousarray(cal, np.float32), device=dev)
    rc = (1.0 - (C @ XM.T).max(1).values).cpu().numpy().astype(np.float64)
    g = d_test - (rs - med) / mad
    gc = d_cal - (rc - med) / mad
    return p_high(gc, g), rs


@torch.no_grad()
def lp_full(sup2, cal, sf, mem_idx, dev):
    """Converged propagation on the full graph from the support (column 0) and from the final memory (column 1)."""
    ns, nc, nt = len(sup2), len(cal), len(sf)
    nfix = ns + nc
    X = np.concatenate([sup2, cal, sf]).astype(np.float32)
    graph = PrefixTopK(X, device=dev)
    M = graph.matrix(KG, GAMMA, "sym")
    Y = torch.zeros((len(X), 2), dtype=torch.float64, device=dev)
    Y[:ns, 0] = 1.0
    if len(mem_idx):
        Y[torch.as_tensor(nfix + np.asarray(mem_idx), device=dev), 1] = 1.0
    U, it, res = propagate_converged(M, Y, LAM)
    U = U.cpu().numpy()
    del graph, M
    torch.cuda.empty_cache()
    u, un = U[:, 0], U[:, 1]
    uc, ut = u[ns:nfix], u[nfix:]
    med = np.median(uc)
    mad = MAD * np.median(np.abs(uc - med))
    out = {"LPfull": _log(p_low(uc, ut)), "zfull": log_ndtr((ut - med) / max(mad, 1e-300)),
           "NEGfull": _log(p_high(un[ns:nfix], un[nfix:])) if len(mem_idx) else np.zeros(nt)}
    return out, it, res


def run_part(part, views, worker, nworkers, dev="cuda"):
    out_dir = P.RESULTS11 / part
    out_dir.mkdir(parents=True, exist_ok=True)
    tasks = P.stream_tasks(part)
    todo = [(ds, s, p) for (ds, s, p) in tasks[worker::nworkers] if not (out_dir / f"{ds}_seed{s}.npz").exists()]
    if not todo:
        return
    D = P.load_p3(part)
    new = [v for v in views if v not in D["shots"]]
    if new:
        P.add_views(D, part, new)
    from vins import config as C
    pos = torch.load(C.WORK / "analysis" / "baselines" / "pos_tins.pt")["pos"].float().numpy()
    cand = D["cands"][P.V5["K"]]
    full = statics(D, views, cand)
    for (ds, seed, path) in todo:
        name = f"{ds}_seed{seed}"
        t0 = time.time()
        run = np.load(path, allow_pickle=True)
        ids, flag = run["sample_id"], run["is_ood"].astype(bool)
        S = run["S_final"].astype(np.float64)
        rows = np.array([D["row"][x] for x in ids])
        sims = D["ev"]["CLIP"][rows - 4000] @ pos.T
        mcm = torch.softmax(torch.as_tensor(sims, dtype=torch.float64), dim=1).max(1).values.numpy()
        arrays = {"sample_id": np.array(ids), "is_ood": flag, "logS": np.log(S), "logM": np.log(mcm)}
        meta = {"part": part, "stream": name, "views": {}}
        for v in views:
            sup, cal, ev, st = full[v]
            sf = ev[rows - 4000]
            z = np.load(P.stream_result_file(part, name, v), allow_pickle=True)
            assert np.array_equal(z["sample_id"], ids), (name, v)
            adm = z[f"a::{v}::M0"].astype(bool)
            mem_idx = np.flatnonzero(adm)
            for k in ("M0", f"lp0|{P.REFT}", f"z|{P.REFT}", "static"):
                arrays[f"s::{v}::{k}"] = z[f"s::{v}::{k}"]
            d_test = st["d"][rows]
            pM, rs = memory_full(sf, cal, mem_idx, d_test, st["d_cal"], st["med"], st["mad"], dev)
            arrays[f"s::{v}::Mfull"] = _log(pM)
            lp, it, res = lp_full(sup.reshape(-1, sup.shape[-1]), cal, sf, mem_idx, dev)
            for k, x in lp.items():
                assert np.isfinite(x).all() or k == "NEGfull", (v, k)
                arrays[f"s::{v}::{k}"] = x
            meta["views"][v] = {"memory": int(len(mem_idx)), "memory_ood_frac": float(flag[mem_idx].mean()) if len(mem_idx) else None,
                                "lp_iters": int(it), "lp_res": float(res)}
        tmp = out_dir / f"{name}.{os.getpid()}.tmp.npz"
        np.savez_compressed(tmp, meta=json.dumps(meta), **arrays)
        os.replace(tmp, out_dir / f"{name}.npz")
        print(json.dumps({"part": part, "stream": name, "n": int(len(ids)), "seconds": round(time.time() - t0, 1),
                          "mem": {v: m["memory"] for v, m in meta["views"].items()}, "utc": P.utc()}), flush=True)
        torch.cuda.empty_cache()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", required=True, choices=["openood", "fourood"])
    ap.add_argument("--views", default="B14,L14,D3B,D3L")
    ap.add_argument("--worker", type=int, default=0)
    ap.add_argument("--nworkers", type=int, default=1)
    a = ap.parse_args()
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    run_part(a.part, a.views.split(","), a.worker, a.nworkers)
