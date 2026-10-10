"""Phase 11e: weights of the terms of the transductive two-sided read-out (fixed graph k_g 10, seeds Storey-BH q,
negative graph k_neg) and of the base detectors (TINS, TANL, both). Metrics in place -> P11/results/sweep_e/<part>/<stream>.csv"""
import argparse
import glob
import itertools
import json
import time

import numpy as np
import pandas as pd
import torch

import p11common as P
import engine5 as E5
from engine import p_low
from sweep11 import l2n, storey_bh
from sweep11d import ViewGraph
from an11 import measures

VIEWSETS = [("L14", "D3L"), ("L14xD3L", "D3L"), ("L14", "D3B", "D3L")]
QS, KNEG = (0.1, 0.05), (5, 10)
W_LP, W_NEG, W_NN, W_BASE = (1.0, 1.5, 2.0), (0.5, 1.0, 1.5, 2.0), (0.0, 1.0), (0.0, 0.5, 1.0, 2.0)
BASES = ("TINS", "TANL", "TINS+TANL")


def run_stream(G, views, flag, bases, rows_out, tag):
    lp = {v: G[v].lp() for v in views}
    s1_t = sum(lp[v][0] for v in views)
    s1_c = sum(lp[v][1] for v in views)
    for q in QS:
        p = p_low(s1_c, s1_t)
        seeds = np.flatnonzero(storey_bh(p, q))
        if len(seeds) == 0:
            continue
        nn = {v: G[v].nn_seed(seeds) for v in views}
        for kneg in KNEG:
            neg = {v: G[v].ranks_high(G[v].mass(kneg, G[v].nfix + seeds)) for v in views}
            LPt = sum(lp[v][0] for v in views)
            NEGt = sum(neg[v][0] for v in views)
            NNt = sum(nn[v][0] for v in views)
            for wl, wn, wnn in itertools.product(W_LP, W_NEG, W_NN):
                core = wl * LPt + wn * NEGt + wnn * NNt
                for bname, bval in bases.items():
                    for wb in W_BASE:
                        if wb == 0.0 and bname != "TINS":
                            continue
                        au, fp = measures(core + wb * bval, flag)
                        rows_out.append({**tag, "q": q, "kneg": kneg, "w_lp": wl, "w_neg": wn, "w_nn": wnn, "base": bname if wb > 0 else "none",
                                         "w_base": wb, "AUROC": au, "FPR95": fp})


def run_part(part, worker, nworkers, dev="cuda"):
    out_dir = P.RESULTS11 / "sweep_e" / part
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
    for v in ["L14", "D3B", "D3L"]:
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
        zv = np.load(P.RESULTS / "vlm" / part / f"{name}.npz", allow_pickle=True)
        assert np.array_equal(zv["sample_id"], ids)
        logT = np.log(np.maximum(zv["TANL"].astype(np.float64), 1e-300))
        bases = {"TINS": logS, "TANL": logT, "TINS+TANL": logS + logT}
        rows = np.array([D["row"][x] for x in ids])
        G = {v: ViewGraph(sup, cal, ev[rows - 4000], stat[v], rows, dev) for v, (sup, cal, ev) in feats.items()}
        out_rows = []
        for vs in VIEWSETS:
            run_stream(G, vs, flag, bases, out_rows, {"part": part, "stream": name, "views": "+".join(vs)})
        for g in G.values():
            g.free()
        pd.DataFrame(out_rows).to_csv(out_dir / f"{name}.csv", index=False)
        print(json.dumps({"part": part, "stream": name, "rows": len(out_rows), "seconds": round(time.time() - t0, 1), "utc": P.utc()}), flush=True)


def aggregate():
    rows = pd.concat([pd.read_csv(f) for part in ("openood", "fourood") for f in sorted(glob.glob(str(P.RESULTS11 / "sweep_e" / part / "*.csv")))])
    rows["ds"] = rows.stream.str.replace(r"_seed\d+$", "", regex=True)
    rows["seed"] = rows.stream.str.extract(r"seed(\d+)$", expand=False).astype(int)
    rows["cfg"] = (rows.views + "|q" + rows.q.astype(str) + "|kn" + rows.kneg.astype(str) + "|wl" + rows.w_lp.astype(str) + "|wn" + rows.w_neg.astype(str)
                   + "|wnn" + rows.w_nn.astype(str) + "|" + rows.base + "|wb" + rows.w_base.astype(str))
    lines, out = [], {}
    for part, d in rows.groupby("part"):
        groups = {"near": P.OO_NEAR, "far": P.OO_FAR} if part == "openood" else {"fourood": P.FOUR}
        res = {}
        for g, sets in groups.items():
            dd = d[d.ds.isin(sets)]
            per = dd.groupby(["cfg", "seed", "ds"])[["AUROC", "FPR95"]].mean().groupby(["cfg", "seed"]).mean().groupby("cfg").mean()
            res[g] = {c: {"AUROC": float(r.AUROC), "FPR95": float(r.FPR95)} for c, r in per.iterrows()}
            lines.append(f"== {part} {g}: top 30 by FPR95 ==")
            for c, r in per.sort_values("FPR95").head(30).iterrows():
                lines.append(f"{c:60s} {r.AUROC:6.2f} / {r.FPR95:6.2f}")
            lines.append("")
        per_ds = d.groupby(["cfg", "ds"])[["AUROC", "FPR95"]].mean()
        res["per_dataset"] = {f"{c}|{ds}": {k: float(v) for k, v in r.items()} for (c, ds), r in per_ds.iterrows()}
        out[part] = res
    (P.RESULTS11 / "sweep_e" / "p11e_summary.json").write_text(json.dumps(out, indent=1) + "\n")
    t = "\n".join(lines)
    (P.RESULTS11 / "sweep_e" / "p11e_table.txt").write_text(t + "\n")
    print(t)
    print("AN11E_DONE", flush=True)


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
