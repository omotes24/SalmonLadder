"""Phase 15 (e): the static read-outs of the six single views with and without centring (subtracting the mean of the support
features before the cosine), on OpenOOD ImageNet-1K (the paper's draw of the labelled images): the prototype rank (static p) and
the kNN rank (k = 10), standalone and x TINS. No graph. Output: P15/results/views_c/<view>/<ds>.csv."""
import argparse
import json
import time

import numpy as np
import pandas as pd
import torch

import p15common as P
import core14 as K
import core15 as K15
from draws12 import datasets
from engine import _log, p_low
from an11 import measures

CANDIDATES = ["B14", "L14", "D3B", "D3L", "CLIP", "DINO1"]


def run(worker, nworkers, device="cuda"):
    D = P.load_p3("openood")
    P.add_views(D, "openood", ["D3B", "D3L"])
    views = [v for v in CANDIDATES if v in D["shots"] and v in D["ev"]]
    sets = datasets("openood")
    tasks = [(v, ds) for v in views for ds in sorted(sets)][worker::nworkers]
    out_root = P.RESULTS15 / "views_c"
    for v, ds in tasks:
        out_dir = out_root / v
        out_dir.mkdir(parents=True, exist_ok=True)
        if (out_dir / f"{ds}.csv").exists():
            continue
        t0 = time.time()
        ids, flag, logS = sets[ds]
        rows = np.array([D["row"][x] for x in ids])
        sup, cal, sf = P.l2n(D["shots"][v][:12000]), P.l2n(D["shots"][v][12000:16000]), P.l2n(D["ev"][v][rows - 4000])
        out = []
        for tag, (s_, c_, f_) in (("raw", (sup, cal, sf)), ("centred", K15.centred(sup, cal, sf))):
            sc, st = K.proto_scores(s_, c_, f_, device)
            scores = {"static": _log(p_low(sc, st)), "knn": K.knn_rank(s_, c_, f_, dev=device)}
            for name, x in scores.items():
                au, fp = measures(x, flag)
                out.append({"view": v, "ds": ds, "cfg": f"{name}|{tag}", "base": "none", "seed": -1, "AUROC": au, "FPR95": fp})
                for seed, S in logS.items():
                    au, fp = measures(x + S, flag)
                    out.append({"view": v, "ds": ds, "cfg": f"{name}|{tag}", "base": "TINS", "seed": int(seed), "AUROC": au, "FPR95": fp})
        pd.DataFrame(out).to_csv(out_dir / f"{ds}.csv", index=False)
        mu = np.asarray(sup, np.float32).mean(0)
        (out_dir / f"{ds}.meta.json").write_text(json.dumps({"view": v, "ds": ds, "dim": int(sup.shape[1]), "support_mean_norm": float(np.linalg.norm(mu)),
                                                            "seconds": round(time.time() - t0, 1), "utc": P.utc()}, indent=1) + "\n")
        print(json.dumps({"view": v, "ds": ds, "FPR95": {r["cfg"]: round(r["FPR95"], 2) for r in out if r["base"] == "none"}, "seconds": round(time.time() - t0, 1)}), flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--worker", type=int, default=0)
    ap.add_argument("--nworkers", type=int, default=1)
    a = ap.parse_args()
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    run(a.worker, a.nworkers)
    print("VIEWS15_DONE", flush=True)
