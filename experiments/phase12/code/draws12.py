"""Phase 12 (b): the post-stream read-out on the public benchmarks for six draws of the 16 labelled images per class
("p3" = the draw of Phases 3-11, i.e. the paper's; "0".."4" = the five draws of the Phase 4 pool, disjoint from it), with the
same-feature baselines (kNN rank; static prototype rank for the p3 draw). The transductive scores of a data set do not depend
on the arrival order, so every (part, draw, data set) is computed once; the base detector TINS (fixed to the p3 draw) is added
per stream order at evaluation.
Output: P12/results/draws/<part>/<draw>/<ds>.npz (per-view arrays), <ds>.meta.json (seeds), <ds>.csv (metrics per configuration
and per stream order)."""
import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import p12common as P
import core12 as C
import engine5 as E5
from engine import _log
from an11 import measures

VIEWS = ["L14", "D3L"]
DRAWS = ["p3", "0", "1", "2", "3", "4"]
VIEWSETS = [("D3L",), ("L14",), ("L14xD3L",), ("L14", "D3L"), ("L14xD3L", "D3L")]
READOUTS = {"T": ["lp", "ln|q0.1"], "Tq0.05": ["lp", "ln|q0.05"], "Tq0.2": ["lp", "ln|q0.2"], "Tlp": ["lp"], "knn": ["knn"], "static": ["static"]}


def l2n(x):
    x = np.asarray(x, np.float32)
    return x / np.linalg.norm(x, axis=1, keepdims=True)


def pool_shots(draw):
    """Support (12,000 x D) and calibration (4,000 x D) features of one Phase 4 pool draw, per view, in (class, pos) order."""
    pool = pd.read_parquet(P.P4 / "banks" / "imagenet_pool.parquet")
    s = pool[(pool.role == "shot") & (pool.draw == int(draw))].sort_values(["idx_1k", "pos"])
    assert len(s) == 16000 and (s.groupby("idx_1k").size() == 16).all()
    sup_ids, cal_ids = s[s.pos < 12].sample_id.tolist(), s[s.pos >= 12].sample_id.tolist()
    out = {}
    ft = pd.read_parquet(P.P4 / "banks" / "feature_table.parquet")
    pos = {x: i for i, x in enumerate(ft.sample_id)}
    L = np.load(P.P4 / "features" / "L14.npy", mmap_mode="r")
    out["L14"] = (l2n(np.array(L[[pos[x] for x in sup_ids]])), l2n(np.array(L[[pos[x] for x in cal_ids]])))
    t = pd.read_parquet(P.P7 / "features" / "u1x" / "table.parquet")
    pos7 = {x: i for i, x in enumerate(t.sample_id)}
    D3 = np.load(P.P7 / "features" / "u1x" / "D3L.npy", mmap_mode="r")
    out["D3L"] = (l2n(np.array(D3[[pos7[x] for x in sup_ids]])), l2n(np.array(D3[[pos7[x] for x in cal_ids]])))
    return out


def datasets(part):
    """{ds: (sample ids, is_ood, {seed: logS aligned to the ids})} from the stream files of the part."""
    out = {}
    for ds, seed, path in P.stream_tasks(part):
        z = np.load(path, allow_pickle=True)
        ids, flag, S = z["sample_id"], z["is_ood"].astype(bool), np.log(np.maximum(z["S_final"].astype(np.float64), 1e-300))
        if ds not in out:
            order = np.argsort(ids)
            out[ds] = [ids[order], flag[order], {}]
        pos = {x: i for i, x in enumerate(ids)}
        out[ds][2][seed] = S[[pos[x] for x in out[ds][0]]]
    return out


def run_part(part, draws, worker, nworkers, device="cuda"):
    out_dir = P.RESULTS12 / "draws" / part
    D = P.load_p3(part)
    P.add_views(D, part, ["D3L"])
    cand = D["cands"][P.V5["K"]]
    sets = datasets(part)
    tasks = [(d, ds) for d in draws for ds in sorted(sets)][worker::nworkers]
    todo = [(d, ds) for d, ds in tasks if not (out_dir / d / f"{ds}.csv").exists()]
    if not todo:
        return
    from static import support_dall
    from vins import r5
    shots = {}
    for d in sorted(set(d for d, _ in todo)):
        if d == "p3":
            shots[d] = {v: (l2n(D["shots"][v][:12000]), l2n(D["shots"][v][12000:16000])) for v in VIEWS}
        else:
            shots[d] = pool_shots(d)
    for d, ds in todo:
        (out_dir / d).mkdir(parents=True, exist_ok=True)
        t0 = time.time()
        ids, flag, logS = sets[ds]
        rows = np.array([D["row"][x] for x in ids])
        terms, meta = {}, {"part": part, "draw": d, "ds": ds, "n": int(len(ids)), "n_ood": int(flag.sum()), "views": {}}
        feats = {v: (shots[d][v][0], shots[d][v][1], l2n(D["ev"][v][rows - 4000])) for v in VIEWS}
        feats["L14xD3L"] = tuple(l2n(np.concatenate([feats["L14"][i], feats["D3L"][i]], 1)) for i in range(3))
        for v, (sup, cal, sf) in feats.items():
            g = C.Graph(sup, cal, sf, device)
            out, m = C.read_out(g, flag)
            g.free()
            terms[(v, "lp")] = out["lp"]
            for q in P.QS:
                terms[(v, f"ln|q{q:g}")] = out[f"ln|q{q:g}"]
            terms[(v, "knn")] = C.knn_rank(sup, cal, sf, dev=device)
            if d == "p3" and v in VIEWS:
                st = E5.static5(r5.proto_view, support_dall, sup.reshape(1000, 12, -1), cal, sf, cand[:4000], cand[rows], P.V5["n0"], P.V5["m"])
                terms[(v, "static")] = _log(np.asarray(st["p"])[len(cal):])
            meta["views"][v] = m
        arrays = {"sample_id": ids, "is_ood": flag}
        for (v, k), x in terms.items():
            arrays[f"s::{v}::{k}"] = np.asarray(x, np.float32)
        rows_out = []
        for vs in VIEWSETS:
            for ro, ks in READOUTS.items():
                if any((v, k) not in terms for v in vs for k in ks):
                    continue
                x = sum(terms[(v, k)] for v in vs for k in ks)
                au, fp = measures(x, flag)
                rows_out.append({"part": part, "draw": d, "ds": ds, "seed": -1, "views": "+".join(vs), "readout": ro, "base": "none", "AUROC": au, "FPR95": fp})
                for seed, S in logS.items():
                    au, fp = measures(x + S, flag)
                    rows_out.append({"part": part, "draw": d, "ds": ds, "seed": int(seed), "views": "+".join(vs), "readout": ro, "base": "TINS", "AUROC": au, "FPR95": fp})
        tmp = out_dir / d / f"{ds}.{os.getpid()}.tmp.npz"
        np.savez_compressed(tmp, meta=json.dumps(meta), **arrays)
        os.replace(tmp, out_dir / d / f"{ds}.npz")
        (out_dir / d / f"{ds}.meta.json").write_text(json.dumps(meta, indent=1) + "\n")
        pd.DataFrame(rows_out).to_csv(out_dir / d / f"{ds}.csv", index=False)
        main = [r for r in rows_out if r["views"] == "D3L" and r["readout"] == "T" and r["base"] == "none"][0]
        print(json.dumps({"part": part, "draw": d, "ds": ds, "rows": len(rows_out), "D3L|T|none": [round(main["AUROC"], 2), round(main["FPR95"], 2)],
                          "seeds": {v: meta["views"][v]["q0.1"] for v in ("D3L",)}, "seconds": round(time.time() - t0, 1), "utc": P.utc()}), flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", required=True, choices=["openood", "fourood"])
    ap.add_argument("--draws", default=",".join(DRAWS))
    ap.add_argument("--worker", type=int, default=0)
    ap.add_argument("--nworkers", type=int, default=1)
    a = ap.parse_args()
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    run_part(a.part, a.draws.split(","), a.worker, a.nworkers)
