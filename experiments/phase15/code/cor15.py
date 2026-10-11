"""Phase 15 (f), descriptive: the escape-probability bound (Corollary 1 of paper v18) measured on the recurrence grid.
For NINCO c = 64 and SSB-hard c = 980 (repetition 0 of the Phase 12-14 test sets: ID validation images + c unknown classes x r
images), every k_g of the grid and every r, per view (L14, D3L) and per unknown class Q (the r images of that class):
  e_i      = sum_{j not in Q} W_ij / delta_i   (escape probability of node i of Q),  ebar_Q = max_i e_i,
  psi(e)   = lambda e / (1 - lambda + lambda e),
  B_Q      = max_{j in dQ} u_j / sqrt(delta_j)   (dQ = outside nodes adjacent to Q; u = converged support mass),
  bound_i  = psi(ebar_Q) sqrt(delta_i) B_Q       (Corollary 1, closed form: u_i <= bound_i),
  pi_i     = lambda (I - lambda P_QQ)^-1 e        (exact escape probability of the stopped walk from i; pi_i <= psi(ebar_Q)),
  bound_pi_i = pi_i sqrt(delta_i) B_Q            (Corollary 1, exact form),
  p+_bound(i) = (1 + #{c in C : u_c <= bound_i}) / (n + 1)   vs   the actual p+(i) = (1 + #{c : u_c <= u_i}) / (n + 1),
with C = all 4,000 calibration images (no cross-fitting; the setting of Proposition 2 / Corollary 1). One graph per test set
(exact top-40 lists), all k_g on it. No selection, nothing registered: a descriptive check of how tight the corollary is.
Output: P15/results/cor/<ds>/r<r>_rep0_kg<k>.csv (one row per cluster and view) + meta.json."""
import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch

import p15common as P
import core15 as K15
from rec14 import GRID, VIEWS

CMAX = {"ninco": 64, "ssb_hard": 980}
LAM = P.LAM


def psi(e, lam=LAM):
    return lam * e / (1.0 - lam + lam * e)


def sym_graph(graph, k):
    """W = max(A, A^T) of the top-k lists (the same arithmetic as PrefixTopK.matrix, kind 'sym') as a scipy CSR matrix + degrees."""
    n = len(graph.X)
    dev = graph.device
    rows = torch.arange(n, device=dev).repeat_interleave(k)
    cols = graph.I[:, :k].reshape(-1)
    vals = graph.S[:, :k].clamp_min(0.0).reshape(-1)
    keys = torch.cat([rows * n + cols, cols * n + rows])
    uniq, inv = torch.unique(keys, return_inverse=True)
    w = torch.zeros(len(uniq), dtype=torch.float32, device=dev).scatter_reduce(0, inv, torch.cat([vals, vals]), reduce="amax", include_self=False)
    r, c = (uniq // n).cpu().numpy(), (uniq % n).cpu().numpy()
    w = w.double().cpu().numpy()
    W = sp.csr_matrix((w, (r, c)), shape=(n, n))
    deg = np.asarray(W.sum(1)).ravel()
    return W, deg


def rank_p(u_cal_sorted, x):
    """p = (1 + #{c : u_c <= x}) / (n + 1) for an array x."""
    return (1.0 + np.searchsorted(u_cal_sorted, x, side="right")) / (len(u_cal_sorted) + 1.0)


def run(worker, nworkers, device="cuda"):
    P.install()
    D = P.load_p3("openood")
    P.add_views(D, "openood", ["D3L"])
    paths = P.image_paths("openood")[16000:]
    ids = list(D["ids"])
    cls = {s: Path(pth).parts[-2] for s, pth in zip(ids, paths)}
    sets = {}
    for ds, seed, path in P.stream_tasks("openood"):
        if ds not in ("ninco", "ssb_hard") or seed != 123:
            continue
        z = np.load(path, allow_pickle=True)
        sid, flag = z["sample_id"], z["is_ood"].astype(bool)
        sets[ds] = {"id": sid[~flag], "ood": sid[flag]}
    id_ids = sets["ninco"]["id"]
    shots = {v: (P.l2n(D["shots"][v][:12000]), P.l2n(D["shots"][v][12000:16000])) for v in VIEWS}
    rep = 0
    tasks = [(ds, c, r) for ds in GRID for (c, r) in GRID[ds] if c == CMAX[ds]][worker::nworkers]
    for ds, c, r in tasks:
        out_dir = P.RESULTS15 / "cor" / ds
        out_dir.mkdir(parents=True, exist_ok=True)
        names = {k: f"r{r}_rep{rep}_kg{k}" for k in P.KGS}
        if all((out_dir / f"{n}.csv").exists() for n in names.values()):
            continue
        t0 = time.time()
        rng = np.random.default_rng([12, int(c), int(r), rep, 0 if ds == "ninco" else 1])          # the Phase 12/13/14/15 test sets
        by_class = defaultdict(list)
        for s in sets[ds]["ood"]:
            by_class[cls[s]].append(s)
        eligible = sorted(k for k, v in by_class.items() if len(v) >= r)
        chosen = rng.choice(eligible, size=c, replace=False)
        ood_ids, ood_cls = [], []
        for k in chosen:
            pick = rng.choice(np.array(by_class[k]), size=r, replace=False).tolist()
            ood_ids.extend(pick)
            ood_cls.extend([k] * len(pick))
        test_ids = np.concatenate([id_ids, np.array(ood_ids)])
        rows = np.array([D["row"][x] for x in test_ids])
        graphs = {}
        for v in VIEWS:
            sup, cal = shots[v]
            sf = P.l2n(D["ev"][v][rows - 4000])
            graphs[v] = K15.Graph15(sup, cal, sf, device, kmax=max(P.KGS))
        ns, nc = graphs[VIEWS[0]].ns, graphs[VIEWS[0]].nc
        nfix = ns + nc
        n_id = len(id_ids)
        clusters = defaultdict(list)                                     # class -> node indices of its r images
        for t, k in enumerate(ood_cls):
            clusters[k].append(nfix + n_id + t)
        for kg in P.KGS:
            if (out_dir / f"{names[kg]}.csv").exists():
                continue
            t1 = time.time()
            out = []
            for v in VIEWS:
                g = graphs[v]
                u = g.mass(np.arange(ns), k=kg)                           # converged support mass, all nodes (float64)
                W, deg = sym_graph(g.graph, kg)
                sq = np.sqrt(deg)
                un = u / np.maximum(sq, 1e-12)                            # u_j / sqrt(delta_j)
                u_cal_sorted = np.sort(u[ns:nfix])
                top_in = g.graph.I[:, :kg].cpu().numpy()
                for k, Q in clusters.items():
                    Q = np.asarray(Q)
                    inQ = np.zeros(len(u), bool)
                    inQ[Q] = True
                    WQ = W[Q]                                             # rows of Q
                    inside = np.asarray(WQ[:, Q].sum(1)).ravel()
                    e = (deg[Q] - inside) / np.maximum(deg[Q], 1e-12)      # escape probabilities
                    ebar = float(e.max())
                    cols = WQ.indices
                    boundary = np.unique(cols[~inQ[cols]])
                    B = float(un[boundary].max()) if len(boundary) else 0.0
                    bound = psi(ebar) * sq[Q] * B                            # closed-form relaxation psi(ebar_Q)
                    # exact escape probabilities pi = lambda (I - lambda P_QQ)^-1 e  (P = D^-1 W restricted to Q)
                    PQQ = np.asarray(WQ[:, Q].todense()) / np.maximum(deg[Q], 1e-12)[:, None]
                    pi = LAM * np.linalg.solve(np.eye(len(Q)) - LAM * PQQ, e)
                    bound_pi = pi * sq[Q] * B
                    p_act = rank_p(u_cal_sorted, u[Q])
                    p_bnd = rank_p(u_cal_sorted, bound)
                    p_bnd_pi = rank_p(u_cal_sorted, bound_pi)
                    floor = 1.0 / (nc + 1)
                    self_inside = float(inQ[top_in[Q]].mean())             # fraction of Q's own top-k_g entries that stay inside Q
                    out.append({"ds": ds, "c": c, "r": r, "kg": kg, "view": v, "cls": k, "size": len(Q), "ebar": ebar, "psi": psi(ebar),
                                "e_med": float(np.median(e)), "pi_med": float(np.median(pi)), "pi_min": float(pi.min()), "pi_max": float(pi.max()),
                                "B": B, "n_boundary": int(len(boundary)), "self_inside_frac": self_inside,
                                "u_med": float(np.median(u[Q])), "bound_med": float(np.median(bound)), "bound_pi_med": float(np.median(bound_pi)),
                                "p_act_med": float(np.median(p_act)), "p_bound_med": float(np.median(p_bnd)), "p_bound_pi_med": float(np.median(p_bnd_pi)),
                                "frac_act_le005": float((p_act <= 0.05).mean()), "frac_bound_le005": float((p_bnd <= 0.05).mean()), "frac_bound_pi_le005": float((p_bnd_pi <= 0.05).mean()),
                                "frac_act_le02": float((p_act <= 0.2).mean()), "frac_bound_le02": float((p_bnd <= 0.2).mean()), "frac_bound_pi_le02": float((p_bnd_pi <= 0.2).mean()),
                                "frac_act_floor": float((p_act <= floor + 1e-12).mean()), "frac_bound_floor": float((p_bnd <= floor + 1e-12).mean()), "frac_bound_pi_floor": float((p_bnd_pi <= floor + 1e-12).mean()),
                                "max_ratio_psi": float((u[Q] / np.maximum(bound, 1e-300)).max()), "max_ratio_pi": float((u[Q] / np.maximum(bound_pi, 1e-300)).max()),
                                "bound_valid": bool(np.all(u[Q] <= bound * (1 + 1e-6) + 1e-15)), "bound_pi_valid": bool(np.all(u[Q] <= bound_pi * (1 + 1e-6) + 1e-15)),
                                "pi_le_psi": bool(np.all(pi <= psi(ebar) + 1e-12))})
            df = pd.DataFrame(out)
            df.to_csv(out_dir / f"{names[kg]}.csv", index=False)
            meta = {"ds": ds, "c": int(c), "r": int(r), "rep": rep, "kg": int(kg), "n_id": int(n_id), "n_ood": int(len(ood_ids)), "n_cal": int(nc),
                    "lambda": LAM, "views": list(VIEWS), "seconds": round(time.time() - t1, 1), "utc": P.utc(),
                    "all_bounds_valid": bool(df.bound_valid.all()), "all_pi_bounds_valid": bool(df.bound_pi_valid.all()), "all_pi_le_psi": bool(df.pi_le_psi.all())}
            (out_dir / f"{names[kg]}.meta.json").write_text(json.dumps(meta, indent=1) + "\n")
            s = df[df.view == "D3L"]
            print(json.dumps({"ds": ds, "r": r, "kg": kg, "ebar_med": round(float(s.ebar.median()), 4), "pi_med": round(float(s.pi_med.median()), 4),
                              "p_act_med": round(float(s.p_act_med.median()), 4), "p_bound_med": round(float(s.p_bound_med.median()), 4), "p_bound_pi_med": round(float(s.p_bound_pi_med.median()), 4),
                              "frac_act_le005": round(float(s.frac_act_le005.mean()), 3), "frac_bound_le005": round(float(s.frac_bound_le005.mean()), 3),
                              "frac_bound_pi_le005": round(float(s.frac_bound_pi_le005.mean()), 3), "valid": bool(df.bound_valid.all() and df.bound_pi_valid.all() and df.pi_le_psi.all()), "seconds": meta["seconds"]}), flush=True)
        for g in graphs.values():
            g.free()
        print(json.dumps({"ds": ds, "r": r, "all_kg_seconds": round(time.time() - t0, 1)}), flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--worker", type=int, default=0)
    ap.add_argument("--nworkers", type=int, default=1)
    a = ap.parse_args()
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    run(a.worker, a.nworkers)
    print("COR15_DONE", flush=True)
