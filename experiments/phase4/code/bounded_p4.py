"""Exp 7 (descriptive): bounded history on a long U1 stream (split 1, draw 0; 72,000 ID + 5,000 OOD images).
Caps {1000, 8192, 32768} x retention {fifo, random, diversity} and the unbounded run, for the frozen REPRISE,
the best minimal configuration and static x p_LP (all at k10 g1 l0.9). Retention applies to the graph's stream
nodes (exact deletion update of the kNN lists) and to each of A1/A2/M. Per-batch time by component and peak GPU
memory (allocated and reserved) are recorded."""
import argparse
import json
import math
import os
import time

import numpy as np
import pandas as pd
import torch
from scipy.special import log_ndtr

from bank_data import Features, pos_text, shots_of, spec
from common import BANKS, RESULTS, V5, utc
from engine import PrefixTopK, _log, p_high, p_low, propagate
from metrics_p4 import metrics
from static import static_view

OUT = RESULTS / "bounded"
K, G, LAM = 10, 1.0, 0.9


def prune(graph, keep):
    """Keep rows `keep` (sorted); rows that lose a neighbour get an exact new top-k over the retained nodes."""
    dev = graph.device
    keep_t = torch.as_tensor(keep, device=dev)
    n_old = len(graph.X)
    mapping = torch.full((n_old,), -1, dtype=torch.int64, device=dev)
    mapping[keep_t] = torch.arange(len(keep), device=dev)
    graph.X = graph.X[keep_t]
    I = mapping[graph.I[keep_t]]
    S = graph.S[keep_t]
    bad = torch.nonzero((I < 0).any(1)).flatten()
    for lo in range(0, len(bad), 1024):
        rows = bad[lo:lo + 1024]
        sims = graph.X[rows] @ graph.X.T
        sims[torch.arange(len(rows), device=dev), rows] = -math.inf
        v, i = sims.topk(graph.kmax, dim=1)
        S[rows], I[rows] = v, i
    graph.S, graph.I = S, I
    return graph


def choose_keep(policy, n_stream, cap, rng, redundancy=None):
    if n_stream <= cap:
        return np.arange(n_stream)
    if policy == "fifo":
        return np.arange(n_stream - cap, n_stream)
    if policy == "random":
        return np.sort(rng.choice(n_stream, size=cap, replace=False))
    order = np.argsort(redundancy, kind="stable")            # diversity: drop the most redundant
    return np.sort(order[:cap])


def run(inp, view, cap, policy, seed=0):
    dev = "cuda"
    torch.cuda.reset_peak_memory_stats()
    sup = inp[f"sup_{view}"].reshape(-1, inp[f"sup_{view}"].shape[-1])
    cal, sf, st = inp[f"cal_{view}"], inp[f"sf_{view}"], inp[f"static_{view}"]
    ns, nc = len(sup), len(cal)
    nfix = ns + nc
    graph = PrefixTopK(np.concatenate([sup, cal]), device=dev)
    tx = torch.as_tensor(sf, device=dev)
    tc = torch.as_tensor(cal, device=dev)
    rng = np.random.default_rng([20260928, 70, seed])
    ids_graph = np.zeros(0, np.int64)                        # stream positions of retained graph nodes
    mem = [np.zeros(0, np.int64) for _ in range(3)]
    warm = None
    d, da = st["d"][nc:], st["d_all"][nc:]
    th = V5["thresholds"]
    n = len(sf)
    out = {k: np.empty(n) for k in ("pLP", "Mpt", "z")}
    trace = []
    bidx = np.arange(n) // 256
    for b in np.unique(bidx):
        rows = np.flatnonzero(bidx == b)
        t = {}
        t0 = time.perf_counter()
        # memory (earlier batches only)
        ps = [p_high(st["d_all_cal"], da[rows])]
        for j in range(3):
            if len(mem[j]):
                mt = tx[torch.as_tensor(mem[j], device=dev)]
                rs = (1 - (tx[rows] @ mt.T).max(1).values).cpu().numpy().astype(np.float64)
                rc = (1 - (tc @ mt.T).max(1).values).cpu().numpy().astype(np.float64)
            else:
                rs, rc = np.full(len(rows), 2.0), np.full(nc, 2.0)
            g = d[rows] - (rs - st["med"]) / st["mad"]
            gc = st["d_cal"] - (rc - st["med"]) / st["mad"]
            ps.append(p_high(gc, g))
        out["Mpt"][rows] = _log(ps[-1])
        for j in range(3):
            mem[j] = np.r_[mem[j], rows[ps[j] <= th[j]]]
            if cap and len(mem[j]) > cap:
                red = None
                if policy == "diversity":
                    mt = tx[torch.as_tensor(mem[j], device=dev)]
                    red = np.empty(len(mt))
                    for lo in range(0, len(mt), 4096):
                        s = mt[lo:lo + 4096] @ mt.T
                        s[torch.arange(len(s), device=dev), torch.arange(lo, lo + len(s), device=dev)] = -math.inf
                        red[lo:lo + len(s)] = s.max(1).values.cpu().numpy()
                mem[j] = mem[j][choose_keep(policy, len(mem[j]), cap, rng, red)]
        torch.cuda.synchronize()
        t["memory_s"] = time.perf_counter() - t0
        # graph: evict before appending, then append the batch
        t0 = time.perf_counter()
        if cap and len(ids_graph) + len(rows) > cap:
            keep_n = cap - len(rows)
            red = graph.S[nfix:, 0].cpu().numpy() if policy == "diversity" else None
            keep_stream = choose_keep(policy, len(ids_graph), keep_n, rng, red)
            keep = np.r_[np.arange(nfix), nfix + keep_stream]
            graph = prune(graph, keep)
            ids_graph = ids_graph[keep_stream]
            if warm is not None:
                warm = warm[torch.as_tensor(keep, device=dev)]
        torch.cuda.synchronize()
        t["evict_s"] = time.perf_counter() - t0
        t0 = time.perf_counter()
        graph.append(sf[rows])
        ids_graph = np.r_[ids_graph, rows]
        torch.cuda.synchronize()
        t["graph_s"] = time.perf_counter() - t0
        t0 = time.perf_counter()
        nn = len(graph)
        M = graph.matrix(K, G, "sym")
        y = torch.zeros(nn, dtype=torch.float64, device=dev)
        y[:ns] = 1.0
        w0 = y.clone() if warm is None else torch.cat([warm, torch.zeros(nn - len(warm), dtype=torch.float64, device=dev)])
        lam = torch.tensor([LAM, LAM], dtype=torch.float64, device=dev)
        U = propagate(M, torch.stack([y, y], 1), lam, torch.stack([w0, torch.zeros_like(y)], 1))
        warm = U[:, 0].clone()
        U = U.cpu().numpy()
        new = np.arange(nn - len(rows), nn)
        u0, u1 = U[:, 0], U[:, 1]
        out["pLP"][rows] = _log(p_low(u0[ns:nfix], u0[new]))
        uc = u1[ns:nfix]
        med = np.median(uc)
        mad = 1.4826 * np.median(np.abs(uc - med))
        out["z"][rows] = log_ndtr((u1[new] - med) / max(mad, 1e-300))
        torch.cuda.synchronize()
        t["lp_s"] = time.perf_counter() - t0
        trace.append({"batch": int(b), "n_graph": nn, "mem_sizes": [int(len(m)) for m in mem], **t})
    peak = {"allocated_GiB": torch.cuda.max_memory_allocated() / 2**30, "reserved_GiB": torch.cuda.max_memory_reserved() / 2**30}
    return out, trace, peak


def long_inputs(F, split=1, draw=0, seed=123):
    names, shots, _, ood = spec("U1", split)
    pool = pd.read_parquet(BANKS / "imagenet_pool.parquet")
    sp = json.loads((BANKS / "U1" / f"split{split}.json").read_text())
    idw = set(sp["id_wnids"])
    ide = pool[pool.role.isin(["ideval", "long"]) & pool.wnid.isin(idw)].sort_values(["idx_1k", "slot"]).sample_id.tolist()
    items = [(s, 0) for s in ide] + [(s, 1) for s in ood]
    rng = np.random.default_rng([20260928, 71, seed])
    items = [items[i] for i in rng.permutation(len(items))]
    ids = [s for s, _ in items]
    flag = np.array([f for _, f in items], bool)
    C = len(names)
    sup_ids, cal_ids = shots_of(shots, draw, C)
    pt = pos_text("U1", split)
    clip_s, clip_c = F.get("CLIP", ids), F.get("CLIP", cal_ids)
    cand_s = np.argsort(-(clip_s @ pt.T), axis=1)[:, :20]
    cand_c = np.argsort(-(clip_c @ pt.T), axis=1)[:, :20]
    inp = {"ids": ids, "flag": flag}
    for v in ("B14", "L14"):
        sup, cal, X = F.get(v, sup_ids).reshape(C, 12, -1), F.get(v, cal_ids), F.get(v, ids)
        inp[f"sup_{v}"], inp[f"cal_{v}"], inp[f"sf_{v}"] = sup, cal, X
        inp[f"static_{v}"] = static_view(sup, cal, X, cand_c, cand_s, V5["n0"], V5["m"])
    return inp


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--settings", required=True, help="comma list cap:policy, cap 0 = unbounded")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    F = Features()
    inp = long_inputs(F)
    flag = inp["flag"]
    for s in a.settings.split(","):
        cap, policy = s.split(":")
        cap = int(cap)
        f_out = OUT / f"cap{cap}_{policy}.json"
        if f_out.exists():
            continue
        res, traces, peaks = {}, {}, {}
        t0 = time.time()
        for v in ("B14", "L14"):
            res[v], traces[v], peaks[v] = run(inp, v, cap if cap else None, policy)
        sc = {"REPRISE": res["B14"]["Mpt"] + res["B14"]["pLP"] + res["L14"]["Mpt"] + res["L14"]["pLP"],
              "static_pLP": sum(_log(inp[f"static_{v}"]["p"][len(inp[f'cal_{v}']):]) + res[v]["pLP"] for v in ("B14", "L14")),
              "best_z": res["B14"]["z"] + res["L14"]["z"]}
        # the same metrics on the last 5,000 stream positions (long-run behaviour)
        tail = np.arange(len(flag)) >= len(flag) - 20000
        m = {k: {"all": metrics(x, flag), "last20k": metrics(x[tail], flag[tail])} for k, x in sc.items()}
        tr = pd.DataFrame(traces["B14"])
        summary = {"cap": cap, "policy": policy, "metrics": m, "peaks": peaks, "seconds": time.time() - t0,
                   "time_per_batch_ms": {c: float(1000 * tr[c].mean()) for c in ("memory_s", "evict_s", "graph_s", "lp_s")},
                   "time_last50_batches_ms": {c: float(1000 * tr[c].iloc[-50:].mean()) for c in ("memory_s", "evict_s", "graph_s", "lp_s")},
                   "time_p95_batch_ms": float(1000 * tr[["memory_s", "evict_s", "graph_s", "lp_s"]].sum(1).quantile(0.95)),
                   "final_graph_nodes": int(tr.n_graph.iloc[-1]), "final_mem_sizes": traces["B14"][-1]["mem_sizes"], "utc": utc()}
        f_out.write_text(json.dumps(summary, indent=1))
        print(json.dumps({"bounded": f_out.name, "seconds": round(summary["seconds"], 1)}), flush=True)
