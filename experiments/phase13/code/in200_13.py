"""Phase 13 (d): the OpenOOD v1.5 ImageNet-200 benchmark (ID = 200 ImageNet classes, 9,000 validation images; near-OOD =
SSB-hard, NINCO; far-OOD = iNaturalist, Textures, OpenImage-O; the OOD image lists are the ImageNet-1K ones) with the same
frozen features and the same read-out, the labelled images restricted to the 200 classes (12 support + 4 calibration per class,
n = 800), six draws, standalone (no base detector). Same-feature baselines: kNN rank, prototype rank.
Output: P13/results/in200/<draw>/<ds>.{csv,meta.json}."""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import p13common as P
import core12 as C
from draws12 import pool_shots, datasets, DRAWS
from an11 import measures

VIEWS = ["L14", "D3L"]
VIEWSETS = [("D3L",), ("L14", "D3L")]
READOUTS = {"Tlp": ["lp"], "Tq0.05": ["lp", "ln|q0.05"], "T": ["lp", "ln|q0.1"], "Tq0.2": ["lp", "ln|q0.2"], "knn": ["knn"], "proto": ["proto"],
            "Tq0.2xproto": ["lp", "ln|q0.2", "proto"], "Tlpxproto": ["lp", "proto"]}
IN200 = P.IMGLIST / "imagenet200"


def class_layout(paths):
    """wnid of every class index of the shots (support rows [12c, 12c+12), calibration rows [12000+4c, 12000+4c+4))."""
    sp, cp = paths[:12000], paths[12000:16000]
    wn = []
    for c in range(1000):
        ws = {Path(p).parts[-2] for p in sp[12 * c:12 * c + 12]}
        wc = {Path(p).parts[-2] for p in cp[4 * c:4 * c + 4]}
        assert len(ws) == 1 and ws == wc, (c, ws, wc)
        wn.append(ws.pop())
    assert wn == sorted(wn), "class index is not the sorted-wnid index"
    return wn


def run(worker, nworkers, device="cuda"):
    D = P.load_p3("openood")
    P.add_views(D, "openood", ["D3L"])
    paths = P.image_paths("openood")
    wnids = class_layout(paths)
    widx = {w: i for i, w in enumerate(wnids)}
    # the 200 classes (label of the imagenet200 lists -> wnid of the training folder -> ImageNet-1K index)
    lab2w = {}
    for line in (IN200 / "train_imagenet200.txt").read_text().splitlines():
        if line.strip():
            p, y = line.split()
            lab2w.setdefault(int(y), set()).add(Path(p).parts[-2])
    assert len(lab2w) == 200 and all(len(v) == 1 for v in lab2w.values())
    cls200 = [widx[next(iter(lab2w[y]))] for y in range(200)]
    # ID test images of ImageNet-200 -> sample ids (by file name) of the ImageNet-1K feature space
    name2id = {Path(p).name: s for s, p in zip(D["ids"], paths[16000:]) if "imagenet_1k/val/" in p}
    id200, missing = [], []
    for line in (IN200 / "test_imagenet200.txt").read_text().splitlines():
        if line.strip():
            p, y = line.split()
            (id200 if Path(p).name in name2id else missing).append(Path(p).name)
    print(json.dumps({"in200_id_test": len(id200), "missing_in_feature_space": len(missing), "classes": cls200[:5]}), flush=True)
    assert not missing, f"{len(missing)} ImageNet-200 test images are not in the ImageNet-1K feature space"
    id_ids = np.array(sorted(name2id[n] for n in id200))
    # OOD lists identical to the ImageNet-1K ones?
    for ds in ("ssb_hard", "ninco", "inaturalist", "textures", "openimage_o"):
        a = {l.split()[0] for l in (IN200 / f"test_{ds}.txt").read_text().splitlines() if l.strip()}
        b = {l.split()[0] for l in (P.IMGLIST / "imagenet" / f"test_{ds}.txt").read_text().splitlines() if l.strip()}
        assert a == b, (ds, len(a), len(b), len(a & b))
    sets = datasets("openood")
    sup_rows = np.concatenate([np.arange(12 * c, 12 * c + 12) for c in cls200])
    cal_rows = np.concatenate([np.arange(12000 + 4 * c, 12000 + 4 * c + 4) for c in cls200])
    tasks = [(d, ds) for d in DRAWS for ds in sorted(sets)][worker::nworkers]
    out_root = P.RESULTS13 / "in200"
    todo = [(d, ds) for d, ds in tasks if not (out_root / d / f"{ds}.csv").exists()]
    if not todo:
        return
    shots = {}
    for d in sorted(set(d for d, _ in todo)):
        full = {v: (P.l2n(D["shots"][v][:12000]), P.l2n(D["shots"][v][12000:16000])) for v in VIEWS} if d == "p3" else pool_shots(d)
        shots[d] = {v: (full[v][0][sup_rows - 0], full[v][1][cal_rows - 12000]) for v in VIEWS}
    for d, ds in todo:
        (out_root / d).mkdir(parents=True, exist_ok=True)
        t0 = time.time()
        ids_all, flag_all, _ = sets[ds]
        ood_ids = ids_all[flag_all]
        test_ids = np.concatenate([id_ids, ood_ids])
        flag = np.r_[np.zeros(len(id_ids), bool), np.ones(len(ood_ids), bool)]
        rows = np.array([D["row"][x] for x in test_ids])
        terms, meta = {}, {"benchmark": "imagenet200", "draw": d, "ds": ds, "n_id": int(len(id_ids)), "n_ood": int(len(ood_ids)), "n_cal": int(len(cal_rows)), "views": {}}
        for v in VIEWS:
            sup, cal = shots[d][v]
            sf = P.l2n(D["ev"][v][rows - 4000])
            g = C.Graph(sup, cal, sf, device)
            out, m = C.read_out(g, flag)
            g.free()
            terms[(v, "lp")] = out["lp"]
            for q in P.QS:
                terms[(v, f"ln|q{q:g}")] = out[f"ln|q{q:g}"]
            terms[(v, "knn")] = C.knn_rank(sup, cal, sf, dev=device)
            terms[(v, "proto")] = P.proto_rank(sup, cal, sf, dev=device)
            p_id = np.exp(out["lp"][~flag])
            m["uniformity"] = {f"le{a:g}": float((p_id <= a).mean()) for a in (0.01, 0.05, 0.1, 0.2, 0.5)}
            meta["views"][v] = m
        rows_out = []
        for vs in VIEWSETS:
            for ro, ks in READOUTS.items():
                x = sum(terms[(v, k)] for v in vs for k in ks)
                au, fp = measures(x, flag)
                rows_out.append({"benchmark": "imagenet200", "draw": d, "ds": ds, "views": "+".join(vs), "readout": ro, "base": "none", "AUROC": au, "FPR95": fp})
        meta["seconds"] = round(time.time() - t0, 1)
        pd.DataFrame(rows_out).to_csv(out_root / d / f"{ds}.csv", index=False)
        (out_root / d / f"{ds}.meta.json").write_text(json.dumps(meta, indent=1) + "\n")
        pick = {r["readout"]: round(r["FPR95"], 2) for r in rows_out if r["views"] == "L14+D3L"}
        print(json.dumps({"draw": d, "ds": ds, "FPR95 (L14+D3L)": pick, "seeds q0.2 D3L": meta["views"]["D3L"]["q0.2"], "seconds": meta["seconds"], "utc": P.utc()}), flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--worker", type=int, default=0)
    ap.add_argument("--nworkers", type=int, default=1)
    a = ap.parse_args()
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    run(a.worker, a.nworkers)
