"""Phase 14 (e): backbone families. The cross-fitted ladder (fixed references and the dev-selected pair, all modes), p+ only,
static, the Phase 12 read-out, kNN and LGC max on every single frozen view available for the OpenOOD ImageNet-1K part (the paper's
draw of the labelled images): DINOv2 B/14, L/14 (Phase 3), DINOv3 B/16, L/16 (Phase 10), CLIP ViT-B/16 image embedding and DINO
(v1) ViT-B/16 (Phase 3 feature space), standalone and x TINS. Output: P14/results/views/<view>/<ds>.{csv,meta.json}."""
import argparse
import json
import time

import numpy as np
import pandas as pd
import torch

import p14common as P
import core14 as K
from draws12 import datasets

CANDIDATES = ["B14", "L14", "D3B", "D3L", "CLIP", "DINO1"]


def run(worker, nworkers, device="cuda"):
    sel = P.selected()
    pairs = P.pairs_for(sel)
    D = P.load_p3("openood")
    P.add_views(D, "openood", ["D3B", "D3L"])
    views = [v for v in CANDIDATES if v in D["shots"] and v in D["ev"]]
    print(json.dumps({"views": views, "shots_keys": sorted(D["shots"].keys()), "selected": list(sel)}), flush=True)
    sets = datasets("openood")
    tasks = [(v, ds) for v in views for ds in sorted(sets)][worker::nworkers]
    out_root = P.RESULTS14 / "views"
    todo = [(v, ds) for v, ds in tasks if not (out_root / v / f"{ds}.csv").exists()]
    labels_sup = np.repeat(np.arange(1000), 12)
    for v, ds in todo:
        (out_root / v).mkdir(parents=True, exist_ok=True)
        t0 = time.time()
        ids, flag, logS = sets[ds]
        rows = np.array([D["row"][x] for x in ids])
        sup, cal, sf = P.l2n(D["shots"][v][:12000]), P.l2n(D["shots"][v][12000:16000]), P.l2n(D["ev"][v][rows - 4000])
        g = K.Graph(sup, cal, sf, device)
        sc, st = K.proto_scores(sup, cal, sf, device)
        graphs, terms, feats = {v: g}, {v: K.view_terms(g, sc, st)}, {v: (sup, cal, sf)}
        bases = {("TINS", int(seed)): S for seed, S in logS.items()}
        tag = {"part": "openood", "view": v, "dim": int(sup.shape[1]), "ds": ds, "n": int(len(ids)), "n_ood": int(flag.sum())}
        rows_out, meta, _ = K.score_all(graphs, terms, feats, flag, bases, labels_sup, 1000, tag, device, pairs=pairs, v1_qs=(P.Q_PAPER,))
        g.free()
        meta["selected"] = list(sel)
        meta["seconds"] = round(time.time() - t0, 1)
        pd.DataFrame(rows_out).to_csv(out_root / v / f"{ds}.csv", index=False)
        (out_root / v / f"{ds}.meta.json").write_text(json.dumps(meta, indent=1, default=str) + "\n")
        pick = {r["cfg"]: round(r["FPR95"], 2) for r in rows_out if r["base"] == "none" and r["cfg"] in (K.cfg_name(*sel), "v1|storey|q0.2|pp", "xfit|lp", "xfit|static", "knn", "lgcmax")}
        print(json.dumps({"view": v, "ds": ds, "FPR95": pick, "seconds": meta["seconds"], "utc": P.utc()}), flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--worker", type=int, default=0)
    ap.add_argument("--nworkers", type=int, default=1)
    a = ap.parse_args()
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    run(a.worker, a.nworkers)
