"""Phase 13 (c1): the product with a static rank on the public benchmarks, for the six draws of the labelled images. Reads the
per-image terms of the Phase 12 draws runs (log p+, log p- at q = 0.2, the kNN rank, the Phase 5 static rank for the paper's
draw), adds the plain prototype rank (cosine to the nearest class prototype of the 12 support images, calibrated rank) and
evaluates the products. Output: P13/results/product/<part>/<draw>/<ds>.csv (metrics per configuration and stream order)."""
import argparse
import json
import time

import numpy as np
import pandas as pd
import torch

import p13common as P
from draws12 import pool_shots, datasets, DRAWS
from an11 import measures

VIEWS = ["L14", "D3L"]
VIEWSETS = [("D3L",), ("L14", "D3L")]
READOUTS = {"Tq0.2": ["lp", "ln|q0.2"], "Tq0.2xknn": ["lp", "ln|q0.2", "knn"], "Tq0.2xproto": ["lp", "ln|q0.2", "proto"],
            "Tlpxproto": ["lp", "proto"], "Tlp": ["lp"], "proto": ["proto"], "knn": ["knn"], "static": ["static"], "Tq0.2xstatic": ["lp", "ln|q0.2", "static"]}


def run_part(part, worker, nworkers, device="cuda"):
    D = P.load_p3(part)
    P.add_views(D, part, ["D3L"])
    sets = datasets(part)
    tasks = [(d, ds) for d in DRAWS for ds in sorted(sets)][worker::nworkers]
    out_root = P.RESULTS13 / "product" / part
    todo = [(d, ds) for d, ds in tasks if not (out_root / d / f"{ds}.csv").exists()]
    if not todo:
        return
    shots = {}
    for d in sorted(set(d for d, _ in todo)):
        shots[d] = {v: (P.l2n(D["shots"][v][:12000]), P.l2n(D["shots"][v][12000:16000])) for v in VIEWS} if d == "p3" else pool_shots(d)
    for d, ds in todo:
        (out_root / d).mkdir(parents=True, exist_ok=True)
        t0 = time.time()
        z = np.load(P.RESULTS12 / "draws" / part / d / f"{ds}.npz", allow_pickle=True)
        ids, flag = z["sample_id"], z["is_ood"].astype(bool)
        ids_ref, flag_ref, logS = sets[ds]
        assert np.array_equal(ids, ids_ref) and np.array_equal(flag, flag_ref)
        rows = np.array([D["row"][x] for x in ids])
        terms = {}
        for v in VIEWS:
            for k in ("lp", "ln|q0.2", "knn", "static"):
                key = f"s::{v}::{k}"
                if key in z.files:
                    terms[(v, k)] = z[key].astype(np.float64)
            sup, cal = shots[d][v]
            sf = P.l2n(D["ev"][v][rows - 4000])
            terms[(v, "proto")] = P.proto_rank(sup, cal, sf, dev=device)
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
        pd.DataFrame(rows_out).to_csv(out_root / d / f"{ds}.csv", index=False)
        pick = {r["readout"]: round(r["FPR95"], 2) for r in rows_out if r["views"] == "L14+D3L" and r["base"] == "none"}
        print(json.dumps({"part": part, "draw": d, "ds": ds, "FPR95 (L14+D3L, standalone)": pick, "seconds": round(time.time() - t0, 1), "utc": P.utc()}), flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", required=True, choices=["openood", "fourood"])
    ap.add_argument("--worker", type=int, default=0)
    ap.add_argument("--nworkers", type=int, default=1)
    a = ap.parse_args()
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    run_part(a.part, a.worker, a.nworkers)
