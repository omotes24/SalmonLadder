"""Phase 14 (g): the arrival-time mode of the cross-fitted ladder. Batches of 256 as the stream files fix them; the graph grows
batch by batch (exact prefix k-NN lists); the fold of a stream image is the parity of its arrival index, the calibration images are
split once at random (rng [14, 0]). At every batch: u+ is the fixed point of the propagation on the current graph; in each fold
the seeds are re-selected from the p+ of the fold's images so far against the fold's own calibration images (the dev-selected
rule and q), u- of each fold's seeds is the fixed point; a new image of fold b is scored against the other fold's calibration
images and seeds (statistic s of the selected mode; pp, pps and min are all emitted), and p_final is the rank of s among the
own fold's calibration images, scored at the same time. Scores are emitted once and never revised.
Views L14 + D3L; base none / TINS. Output: P14/results/online/<part>/<stream>.{csv,meta.json,npz}."""
import argparse
import json
import os
import time

import numpy as np
import pandas as pd
import torch

import p14common as P
import core14 as K
from engine import PrefixTopK, _log, p_high, p_low
from online13 import propagate_fixed_point
from an11 import measures

VIEWS = ["L14", "D3L"]
MODES = ("pp", "pps", "min")


@torch.no_grad()
def online_xfit_view(sup, cal, sf, bidx, rule, q, static_cal, static_sf, k=P.KG, lam=P.LAM, dev="cuda", flag=None):
    ns, nc, nt = len(sup), len(cal), len(sf)
    nfix = ns + nc
    fc = np.random.default_rng([14, 0]).permutation(np.arange(nc) % 2)
    ft = np.arange(nt) % 2
    cal_nodes = ns + np.arange(nc)
    graph = PrefixTopK(np.concatenate([sup, cal]).astype(np.float32), device=dev)
    st_all = np.concatenate([np.asarray(static_cal, np.float64), np.asarray(static_sf, np.float64)])
    out = {m: np.empty(nt) for m in MODES}
    scal = {m: {} for m in MODES}                                                  # scal[m][(batch, fold)] = s of the fold's calibration images
    prev_pos, prev_neg = None, {0: None, 1: None}
    stats, iters = [], []
    for b in np.unique(bidx):
        rows = np.flatnonzero(bidx == b)
        sims, oldn = graph.append(sf[rows])
        n = len(graph)
        new = np.arange(oldn, n)                                                   # node indices of the new images
        M = graph.matrix(k, 1.0, "sym")
        ypos = torch.zeros((n, 1), dtype=torch.float64, device=dev)
        ypos[:ns, 0] = 1.0
        u0 = ypos.clone()
        if prev_pos is not None:
            u0 = torch.zeros((n, 1), dtype=torch.float64, device=dev)
            u0[:oldn, 0] = prev_pos
        U, it_pos, _ = propagate_fixed_point(M, ypos, lam, u0)
        prev_pos = U[:, 0]
        u = prev_pos.cpu().numpy()
        arrived = np.arange(nfix, n)                                               # every stream image so far (arrival order)
        fold_arr = ft[arrived - nfix]
        uneg, nseeds, idfrac, pi0s, it_negs = {}, {}, {}, {}, []
        for f in (0, 1):
            Cf, Tf = cal_nodes[fc == f], arrived[fold_arr == f]
            if len(Tf) == 0:
                uneg[f], nseeds[f], idfrac[f], pi0s[f] = np.zeros(n), 0, None, None
                continue
            mask, pi0 = P.select_seeds(p_low(u[Cf], u[Tf]), q, len(Cf), rule)
            idx = Tf[mask]
            nseeds[f], pi0s[f] = int(len(idx)), float(pi0)
            idfrac[f] = (float(1.0 - flag[idx - nfix].mean()) if (flag is not None and len(idx)) else None)
            if len(idx) == 0:
                uneg[f], prev_neg[f] = np.zeros(n), None
                continue
            yneg = torch.zeros((n, 1), dtype=torch.float64, device=dev)
            yneg[torch.as_tensor(idx, device=dev), 0] = 1.0
            v0 = yneg.clone()
            if prev_neg[f] is not None:
                v0 = torch.zeros((n, 1), dtype=torch.float64, device=dev)
                v0[:oldn, 0] = prev_neg[f]
            V, it_neg, _ = propagate_fixed_point(M, yneg, lam, v0)
            prev_neg[f] = V[:, 0]
            uneg[f] = prev_neg[f].cpu().numpy()
            it_negs.append(it_neg)
        # scores of the new images and of their own-fold calibration images, against the other fold
        for f in (0, 1):
            newf = new[ft[new - nfix] == f]
            if len(newf) == 0:
                continue
            own, ref = cal_nodes[fc == f], cal_nodes[fc != f]
            nodes = np.concatenate([own, newf])
            s_pp = _log(p_low(u[ref], u[nodes]))
            un = uneg[1 - f]
            if np.any(un):
                s_pp = s_pp + _log(p_high(un[ref], un[nodes]))
            st_nodes = np.concatenate([st_all[own - ns], st_all[nc + (newf - nfix)]])
            s_st = _log(p_low(st_all[ref - ns], st_nodes))
            S = {"pp": s_pp, "pps": s_pp + s_st, "min": np.minimum(s_pp, s_st)}
            for m, s in S.items():
                out[m][newf - nfix] = s[len(own):]
                scal[m][(int(b), f)] = s[:len(own)]
        stats.append((int(b), nseeds, idfrac, pi0s))
        iters.append((it_pos, max(it_negs) if it_negs else 0))
    del graph
    torch.cuda.empty_cache()
    return out, scal, (fc, ft), stats, iters


def pfinal_of(terms, scals, views, mode, bidx, fc, ft):
    """p_final of the multi-view statistic: rank of the summed s of a new image among the summed s of its own fold's calibration
    images, both scored at the image's arrival batch."""
    nt = len(ft)
    s = sum(terms[(v, mode)] for v in views)
    pf = np.empty(nt)
    for b in np.unique(bidx):
        for f in (0, 1):
            pos = np.flatnonzero((bidx == b) & (ft == f))
            if len(pos) == 0:
                continue
            sc = sum(scals[v][mode][(int(b), f)] for v in views)
            pf[pos] = p_low(sc, s[pos])
    return pf


def run_part(part, worker, nworkers, device="cuda"):
    sel = P.selected()
    rule, q, _ = sel
    out_dir = P.RESULTS14 / "online" / part
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
        terms, scals, meta = {}, {}, {"part": part, "stream": name, "selected": list(sel), "n": int(len(ids)), "n_ood": int(flag.sum()), "batches": int(bidx.max()) + 1,
                                     "propagation": "fixed point at every batch (relative residual <= 1e-8, warm start); folds = arrival parity", "views": {}}
        fc = ft = None
        for v in VIEWS:
            sup, cal = shots[v]
            sf = P.l2n(D["ev"][v][rows - 4000])
            sc, st = K.proto_scores(sup, cal, sf, device)
            out, scal, (fc, ft), stats, iters = online_xfit_view(sup, cal, sf, bidx, rule, q, sc, st, dev=device, flag=flag)
            scals[v] = scal
            for m in MODES:
                terms[(v, m)] = out[m]
            it = np.array(iters)
            last = stats[-1]
            meta["views"][v] = {"seeds_last_batch": last[1], "id_frac_last_batch": last[2], "pi0_last_batch": last[3],
                                "iters_pos_mean": float(it[:, 0].mean()), "iters_pos_max": int(it[:, 0].max()), "iters_neg_mean": float(it[:, 1].mean()), "iters_neg_max": int(it[:, 1].max())}
        arrays = {"sample_id": ids, "is_ood": flag, "bidx": bidx, "fold": ft.astype(np.int8)}
        rows_out = []
        for m in MODES:
            x = sum(terms[(v, m)] for v in VIEWS)
            pf = pfinal_of(terms, scals, VIEWS, m, bidx, fc, ft)
            val = K.validity(pf, flag, ft, q=q)
            cfg = f"xonl|{rule}|q{q:g}|{m}"
            for base, add in (("none", 0.0), ("TINS", logS)):
                au, fp = measures(_log(pf) + add, flag)
                rows_out.append({"part": part, "stream": name, "views": "L14+D3L", "cfg": cfg, "base": base, "AUROC": au, "FPR95": fp, **(val if base == "none" else {})})
                au, fp = measures(x + add, flag)
                rows_out.append({"part": part, "stream": name, "views": "L14+D3L", "cfg": cfg + "|pooled", "base": base, "AUROC": au, "FPR95": fp})
            arrays[f"s::{m}"], arrays[f"pfinal::{m}"] = np.asarray(x, np.float32), np.asarray(pf, np.float32)
        meta["seconds"] = round(time.time() - t0, 1)
        tmp = out_dir / f"{name}.{os.getpid()}.tmp.npz"
        np.savez_compressed(tmp, meta=json.dumps(meta, default=str), **arrays)
        os.replace(tmp, out_dir / f"{name}.npz")
        (out_dir / f"{name}.meta.json").write_text(json.dumps(meta, indent=1, default=str) + "\n")
        pd.DataFrame(rows_out).to_csv(out_dir / f"{name}.csv", index=False)
        main = [r for r in rows_out if r["cfg"] == f"xonl|{rule}|q{q:g}|pp" and r["base"] == "TINS"][0]
        alone = [r for r in rows_out if r["cfg"] == f"xonl|{rule}|q{q:g}|pp" and r["base"] == "none"][0]
        print(json.dumps({"part": part, "stream": name, "xonl|pp|TINS": [round(main["AUROC"], 2), round(main["FPR95"], 2)], "fpr@0.05": round(alone["fpr@0.05"], 3), "bh_fdp": round(alone["bh_fdp"], 3),
                          "iters": {v: round(meta["views"][v]["iters_pos_mean"], 1) for v in VIEWS}, "seconds": meta["seconds"], "utc": P.utc()}), flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", required=True, choices=["openood", "fourood"])
    ap.add_argument("--worker", type=int, default=0)
    ap.add_argument("--nworkers", type=int, default=1)
    a = ap.parse_args()
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    run_part(a.part, a.worker, a.nworkers)
