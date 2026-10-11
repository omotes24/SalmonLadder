"""Phase 14 (f2): the OpenOOD v1.5 CIFAR-100 benchmark with the frozen features of extract14.py. Per draw (six) and OOD set
(CIFAR-10, TIN | MNIST, SVHN, Textures, Places365): graph over support (1,200) + calibration (400) + ID test (9,000) + OOD;
every cross-fit configuration (rule x q x mode), p+ only, static, the Phase 12 read-out, kNN, LGC max; standalone (no base
detector exists for this benchmark in our pipeline); validity of p_final (the calibration and the ID test images are both
CIFAR-100 images here). Views L14 + D3L (and each alone). Output: P14/results/cifar/<draw>/<ds>.{csv,meta.json}."""
import argparse
import json
import time

import numpy as np
import pandas as pd
import torch

import p14common as P
import core14 as K
from extract14 import CIFAR, OOD

VIEWS = ["L14", "D3L"]
VIEWSETS = [("L14", "D3L"), ("D3L",), ("L14",)]


def run(worker, nworkers, device="cuda"):
    t = pd.read_parquet(CIFAR / "table.parquet")
    F = {v: np.load(CIFAR / "features" / f"{v}.npy", mmap_mode="r") for v in VIEWS}
    id_test = t.index[(t.split == "test")].values
    tasks = [(d, ds) for d in range(6) for ds in OOD][worker::nworkers]
    out_root = P.RESULTS14 / "cifar"
    labels_sup = np.repeat(np.arange(100), 12)
    for d, ds in tasks:
        out_dir = out_root / f"draw{d}"
        out_dir.mkdir(parents=True, exist_ok=True)
        if (out_dir / f"{ds}.csv").exists():
            continue
        t0 = time.time()
        tr = t[(t.split == "train") & (t.draw == d)].sort_values(["label", "pos"])
        sup_rows, cal_rows = tr.index[tr.role == "sup"].values, tr.index[tr.role == "cal"].values
        assert len(sup_rows) == 1200 and len(cal_rows) == 400
        ood_rows = t.index[t.dataset == ds].values
        test_rows = np.concatenate([id_test, ood_rows])
        flag = np.r_[np.zeros(len(id_test), bool), np.ones(len(ood_rows), bool)]
        rows_all = []
        meta_all = {"benchmark": "cifar100", "draw": d, "ds": ds, "group": OOD[ds], "n_id": int(len(id_test)), "n_ood": int(len(ood_rows)), "views": {}}
        graphs, terms, feats = {}, {}, {}
        for v in VIEWS:
            sup, cal, sf = (P.l2n(np.asarray(F[v][r])) for r in (sup_rows, cal_rows, test_rows))
            g = K.Graph(sup, cal, sf, device)
            sc, st = K.proto_scores(sup, cal, sf, device)
            graphs[v], terms[v], feats[v] = g, K.view_terms(g, sc, st), (sup, cal, sf)
        for vs in VIEWSETS:
            tag = {"benchmark": "cifar100", "draw": d, "ds": ds, "group": OOD[ds], "views": "+".join(vs), "n": int(len(test_rows)), "n_ood": int(len(ood_rows))}
            rows_out, meta, _ = K.score_all({v: graphs[v] for v in vs}, {v: terms[v] for v in vs}, {v: feats[v] for v in vs}, flag, {}, labels_sup, 100, tag, device)
            rows_all.extend(rows_out)
            meta_all["views"]["+".join(vs)] = meta
        for g in graphs.values():
            g.free()
        meta_all["seconds"] = round(time.time() - t0, 1)
        pd.DataFrame(rows_all).to_csv(out_dir / f"{ds}.csv", index=False)
        (out_dir / f"{ds}.meta.json").write_text(json.dumps(meta_all, indent=1, default=str) + "\n")
        pick = {r["cfg"]: round(r["FPR95"], 2) for r in rows_all if r["views"] == "L14+D3L" and r["cfg"] in ("xfit|storey|q0.2|pp", "xfit|bh|q0.2|pp", "xfit|bh|q0.2|min", "v1|storey|q0.2|pp", "xfit|lp", "xfit|static", "knn", "lgcmax")}
        print(json.dumps({"draw": d, "ds": ds, "FPR95 (L14+D3L)": pick, "seconds": meta_all["seconds"], "utc": P.utc()}), flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--worker", type=int, default=0)
    ap.add_argument("--nworkers", type=int, default=1)
    a = ap.parse_args()
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    run(a.worker, a.nworkers)
