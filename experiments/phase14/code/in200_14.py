"""Phase 14 (d): the OpenOOD v1.5 ImageNet-200 benchmark (ID = 200 ImageNet classes, 9,000 validation images; the OOD lists are
the ImageNet-1K ones) with the cross-fitted ladder: every (rule, q, mode), p+ only, static, the Phase 12 read-out, kNN, LGC max;
labelled images restricted to the 200 classes (12 support + 4 calibration per class), six draws, standalone.
Output: P14/results/in200/<draw>/<ds>.{csv,meta.json}."""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import p14common as P
import core14 as K
from draws12 import pool_shots, datasets, DRAWS
from in200_13 import class_layout

VIEWS = ["L14", "D3L"]
IN200 = P.IMGLIST / "imagenet200"


def layout(D, paths):
    wnids = class_layout(paths)
    widx = {w: i for i, w in enumerate(wnids)}
    lab2w = {}
    for line in (IN200 / "train_imagenet200.txt").read_text().splitlines():
        if line.strip():
            p, y = line.split()
            lab2w.setdefault(int(y), set()).add(Path(p).parts[-2])
    assert len(lab2w) == 200 and all(len(v) == 1 for v in lab2w.values())
    cls200 = [widx[next(iter(lab2w[y]))] for y in range(200)]
    name2id = {Path(p).name: s for s, p in zip(D["ids"], paths[16000:]) if "imagenet_1k/val/" in p}
    id200 = [Path(l.split()[0]).name for l in (IN200 / "test_imagenet200.txt").read_text().splitlines() if l.strip()]
    assert all(n in name2id for n in id200)
    id_ids = np.array(sorted(name2id[n] for n in id200))
    sup_rows = np.concatenate([np.arange(12 * c, 12 * c + 12) for c in cls200])
    cal_rows = np.concatenate([np.arange(12000 + 4 * c, 12000 + 4 * c + 4) for c in cls200])
    return id_ids, sup_rows, cal_rows


def run(worker, nworkers, device="cuda"):
    D = P.load_p3("openood")
    P.add_views(D, "openood", ["D3L"])
    paths = P.image_paths("openood")
    id_ids, sup_rows, cal_rows = layout(D, paths)
    sets = datasets("openood")
    tasks = [(d, ds) for d in DRAWS for ds in sorted(sets)][worker::nworkers]
    out_root = P.RESULTS14 / "in200"
    todo = [(d, ds) for d, ds in tasks if not (out_root / d / f"{ds}.csv").exists()]
    if not todo:
        return
    shots = {}
    for d in sorted(set(d for d, _ in todo)):
        full = {v: (P.l2n(D["shots"][v][:12000]), P.l2n(D["shots"][v][12000:16000])) for v in VIEWS} if d == "p3" else pool_shots(d)
        shots[d] = {v: (full[v][0][sup_rows], full[v][1][cal_rows - 12000]) for v in VIEWS}
    labels_sup = np.repeat(np.arange(200), 12)
    for d, ds in todo:
        (out_root / d).mkdir(parents=True, exist_ok=True)
        t0 = time.time()
        ids_all, flag_all, _ = sets[ds]
        ood_ids = ids_all[flag_all]
        test_ids = np.concatenate([id_ids, ood_ids])
        flag = np.r_[np.zeros(len(id_ids), bool), np.ones(len(ood_ids), bool)]
        rows = np.array([D["row"][x] for x in test_ids])
        graphs, terms, feats = {}, {}, {}
        for v in VIEWS:
            sup, cal = shots[d][v]
            sf = P.l2n(D["ev"][v][rows - 4000])
            g = K.Graph(sup, cal, sf, device)
            sc, st = K.proto_scores(sup, cal, sf, device)
            graphs[v], terms[v], feats[v] = g, K.view_terms(g, sc, st), (sup, cal, sf)
        tag = {"benchmark": "imagenet200", "draw": d, "ds": ds, "n": int(len(test_ids)), "n_id": int(len(id_ids)), "n_ood": int(len(ood_ids)), "n_cal": int(len(cal_rows))}
        rows_out, meta, _ = K.score_all(graphs, terms, feats, flag, {}, labels_sup, 200, tag, device)
        for g in graphs.values():
            g.free()
        meta["seconds"] = round(time.time() - t0, 1)
        pd.DataFrame(rows_out).to_csv(out_root / d / f"{ds}.csv", index=False)
        (out_root / d / f"{ds}.meta.json").write_text(json.dumps(meta, indent=1, default=str) + "\n")
        pick = {r["cfg"]: round(r["FPR95"], 2) for r in rows_out if r["cfg"] in ("xfit|storey|q0.2|pp", "xfit|bh|q0.2|pp", "xfit|bh|q0.2|min", "v1|storey|q0.2|pp", "xfit|lp", "xfit|static", "lgcmax")}
        print(json.dumps({"draw": d, "ds": ds, "FPR95": pick, "seconds": meta["seconds"], "utc": P.utc()}), flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--worker", type=int, default=0)
    ap.add_argument("--nworkers", type=int, default=1)
    a = ap.parse_args()
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    run(a.worker, a.nworkers)
