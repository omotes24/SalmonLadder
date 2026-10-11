"""Phase 13 (a): the arrival-time mode with the propagation run to convergence at every batch (the Phase 12 version iterated
15 sweeps from the previous values, so nodes of different age had different iteration counts and the permutation invariance
behind Proposition 3 did not hold at a given time). Batches of 256 as the stream files fix them; the graph grows batch by
batch (exact prefix k-NN lists); u+ and u- are the fixed points of the propagation on the current graph (warm-started from
the previous values, iterated until the relative residual is <= 1e-8); the seeds are re-selected at every batch from the
current p+ of every image so far (Storey-AdaDetect, q = 0.2); the score of a batch is emitted once and never revised.
Views DINOv2 L/14 and DINOv3 L/16; base none / TINS. Output: P13/results/online/<part>/<stream>.{csv,meta.json,npz}."""
import argparse
import json
import os
import time

import numpy as np
import pandas as pd
import torch

import p13common as P
from engine import PrefixTopK, _log, p_high, p_low
from an11 import measures

VIEWS = ["L14", "D3L"]
VIEWSETS = [("D3L",), ("L14", "D3L")]


def propagate_fixed_point(M, Y, lam, U0, tol=1e-8, max_iter=3000):
    """Fixed point of U = lam M U + (1 - lam) Y from the start U0 (the fixed point is unique; U0 only saves iterations)."""
    U = U0
    rhs = (1.0 - lam) * Y
    den = torch.linalg.vector_norm(rhs, dim=0).clamp_min(1e-300)
    for it in range(max_iter):
        U = lam * torch.sparse.mm(M, U) + rhs
        if it % 5 == 4:
            res = torch.linalg.vector_norm(U - lam * torch.sparse.mm(M, U) - rhs, dim=0) / den
            if float(res.max()) <= tol:
                return U, it + 1, float(res.max())
    res = torch.linalg.vector_norm(U - lam * torch.sparse.mm(M, U) - rhs, dim=0) / den
    return U, max_iter, float(res.max())


@torch.no_grad()
def online_view(sup, cal, sf, bidx, q=P.Q_PAPER, k=P.KG, lam=P.LAM, dev="cuda", flag=None):
    ns, nc = len(sup), len(cal)
    nfix = ns + nc
    graph = PrefixTopK(np.concatenate([sup, cal]).astype(np.float32), device=dev)
    nt = len(sf)
    lp, ln = np.empty(nt), np.empty(nt)
    prev_pos = prev_neg = None
    stats, iters = [], []
    for b in np.unique(bidx):
        rows = np.flatnonzero(bidx == b)
        sims, oldn = graph.append(sf[rows])
        n = len(graph)
        new = np.arange(oldn, n)
        M = graph.matrix(k, 1.0, "sym")
        ypos = torch.zeros((n, 1), dtype=torch.float64, device=dev)
        ypos[:ns, 0] = 1.0
        u0 = ypos.clone()
        if prev_pos is not None:
            u0 = torch.zeros((n, 1), dtype=torch.float64, device=dev)
            u0[:oldn, 0] = prev_pos
        U, it_pos, res_pos = propagate_fixed_point(M, ypos, lam, u0)
        u = U[:, 0]
        prev_pos = u
        un = u.cpu().numpy()
        p_all = p_low(un[ns:nfix], un[nfix:])                                   # every stream image so far, current graph
        lp[rows] = _log(p_all[new - nfix])
        mask, pi0 = P.storey_adadetect(p_all, q, nc)
        idx = np.flatnonzero(mask)
        if len(idx) == 0:
            ln[rows] = 0.0
            prev_neg = None
            stats.append((int(b), 0, None, float(pi0)))
            iters.append((it_pos, 0))
            continue
        yneg = torch.zeros((n, 1), dtype=torch.float64, device=dev)
        yneg[torch.as_tensor(nfix + idx, device=dev), 0] = 1.0
        v0 = yneg.clone()
        if prev_neg is not None:
            v0 = torch.zeros((n, 1), dtype=torch.float64, device=dev)
            v0[:oldn, 0] = prev_neg
        V, it_neg, res_neg = propagate_fixed_point(M, yneg, lam, v0)
        v = V[:, 0]
        prev_neg = v
        vn = v.cpu().numpy()
        ln[rows] = _log(p_high(vn[ns:nfix], vn[nfix:][new - nfix]))
        stats.append((int(b), int(len(idx)), (float(1.0 - flag[idx].mean()) if flag is not None else None), float(pi0)))
        iters.append((it_pos, it_neg))
    del graph
    torch.cuda.empty_cache()
    return lp, ln, stats, iters


def run_part(part, worker, nworkers, q=P.Q_PAPER, device="cuda"):
    out_dir = P.RESULTS13 / "online" / part
    out_dir.mkdir(parents=True, exist_ok=True)
    tasks = P.stream_tasks(part)
    todo = [(ds, s, p) for (ds, s, p) in tasks[worker::nworkers] if not (out_dir / f"{ds}_seed{s}.csv").exists()]
    if not todo:
        return
    D = P.load_p3(part)
    P.add_views(D, part, ["D3L"])
    shots = {v: (P.l2n(D["shots"][v][:12000]), P.l2n(D["shots"][v][12000:16000])) for v in VIEWS}
    for ds, seed, path in todo:
        name = f"{ds}_seed{seed}"
        t0 = time.time()
        z = np.load(path, allow_pickle=True)
        ids, flag, bidx = z["sample_id"], z["is_ood"].astype(bool), z["batch_index"]
        logS = np.log(np.maximum(z["S_final"].astype(np.float64), 1e-300))
        rows = np.array([D["row"][x] for x in ids])
        feats = {v: (shots[v][0], shots[v][1], P.l2n(D["ev"][v][rows - 4000])) for v in VIEWS}
        terms, meta = {}, {"part": part, "stream": name, "q": q, "n": int(len(ids)), "n_ood": int(flag.sum()), "batches": int(bidx.max()) + 1,
                           "propagation": "fixed point at every batch (relative residual <= 1e-8, warm start)", "views": {}}
        for v, (sup, cal, sf) in feats.items():
            lp, ln, stats, iters = online_view(sup, cal, sf, bidx, q=q, dev=device, flag=flag)
            terms[(v, "lp")], terms[(v, "ln")] = lp, ln
            last = stats[-1]
            it = np.array(iters)
            meta["views"][v] = {"seeds_last_batch": last[1], "id_frac_last_batch": last[2], "pi0_last_batch": last[3],
                                "seeds_by_batch": [s[1] for s in stats[::max(1, len(stats) // 20)]],
                                "iters_pos_mean": float(it[:, 0].mean()), "iters_pos_max": int(it[:, 0].max()),
                                "iters_neg_mean": float(it[:, 1].mean()), "iters_neg_max": int(it[:, 1].max())}
        arrays = {"sample_id": ids, "is_ood": flag, "bidx": bidx}
        for (v, k), x in terms.items():
            arrays[f"s::{v}::{k}"] = np.asarray(x, np.float32)
        rows_out = []
        for vs in VIEWSETS:
            for ro, ks in (("T", ["lp", "ln"]), ("Tlp", ["lp"])):
                x = sum(terms[(v, k)] for v in vs for k in ks)
                for base, add in (("none", 0.0), ("TINS", logS)):
                    au, fp = measures(x + add, flag)
                    rows_out.append({"part": part, "stream": name, "views": "+".join(vs), "readout": ro, "base": base, "AUROC": au, "FPR95": fp})
        meta["seconds"] = round(time.time() - t0, 1)
        tmp = out_dir / f"{name}.{os.getpid()}.tmp.npz"
        np.savez_compressed(tmp, meta=json.dumps(meta), **arrays)
        os.replace(tmp, out_dir / f"{name}.npz")
        (out_dir / f"{name}.meta.json").write_text(json.dumps(meta, indent=1) + "\n")
        pd.DataFrame(rows_out).to_csv(out_dir / f"{name}.csv", index=False)
        main = [r for r in rows_out if r["views"] == "L14+D3L" and r["readout"] == "T" and r["base"] == "TINS"][0]
        print(json.dumps({"part": part, "stream": name, "L14+D3L|T|TINS": [round(main["AUROC"], 2), round(main["FPR95"], 2)],
                          "iters": {v: round(meta["views"][v]["iters_pos_mean"], 1) for v in VIEWS}, "seconds": meta["seconds"], "utc": P.utc()}), flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", required=True, choices=["openood", "fourood"])
    ap.add_argument("--worker", type=int, default=0)
    ap.add_argument("--nworkers", type=int, default=1)
    ap.add_argument("--q", type=float, default=P.Q_PAPER)
    a = ap.parse_args()
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    run_part(a.part, a.worker, a.nworkers, q=a.q)
