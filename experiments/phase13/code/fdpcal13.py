"""Phase 13 (b): the realised false-discovery proportion of the seeds under exchangeable calibration. The calibration images
are drawn from the ID validation images (4 per class, the rest of the ID images stay as the ID test set), the support images
stay the 12 training images per class of the paper's draw. Three random splits. For every data set and view: the uniformity of
p+ on the ID test images, the seeds at q in {0.05, 0.1, 0.2} with their realised ID fraction and pi0_hat, and the metrics of
p+ and p+ p- (standalone and x TINS, the TINS scores of the stream files restricted to the test images).
Output: P13/results/fdpcal/<part>/split<s>/<ds>.{csv,meta.json}."""
import argparse
import json
import time

import numpy as np
import pandas as pd
import torch

import p13common as P
import core12 as C
from engine import _log
from an11 import measures

VIEWS = ["L14", "D3L"]
VIEWSETS = [("D3L",), ("L14", "D3L")]
READOUTS = {"Tlp": ["lp"], "Tq0.05": ["lp", "ln|q0.05"], "T": ["lp", "ln|q0.1"], "Tq0.2": ["lp", "ln|q0.2"], "knn": ["knn"], "proto": ["proto"]}
SPLITS = 3
N_CAL = 4
ALPHAS = (0.01, 0.05, 0.1, 0.2, 0.5)


def datasets(part):
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


def run_part(part, worker, nworkers, device="cuda"):
    D = P.load_p3(part)
    P.add_views(D, part, ["D3L"])
    labels, _ = P.id_labels(part, D)
    sets = datasets(part)
    tasks = [(s, ds) for s in range(SPLITS) for ds in sorted(sets)][worker::nworkers]
    out_root = P.RESULTS13 / "fdpcal" / part
    todo = [(s, ds) for s, ds in tasks if not (out_root / f"split{s}" / f"{ds}.csv").exists()]
    if not todo:
        return
    sup = {v: P.l2n(D["shots"][v][:12000]) for v in VIEWS}
    id_pos = {x: i for i, x in enumerate(D["ids"])}
    for s, ds in todo:
        (out_root / f"split{s}").mkdir(parents=True, exist_ok=True)
        t0 = time.time()
        ids, flag, logS = sets[ds]
        # the ID images of this data set (all parts share the same ID images), their labels, and the calibration draw of the split
        id_ids = ids[~flag]
        lab = np.array([labels[id_pos[x]] for x in id_ids])
        assert (lab >= 0).all() and len(np.unique(lab)) == 1000
        rng = np.random.default_rng([13, s])
        cal_mask = np.zeros(len(id_ids), bool)
        for c in range(1000):
            members = np.flatnonzero(lab == c)
            cal_mask[rng.choice(members, size=N_CAL, replace=False)] = True
        cal_ids = id_ids[cal_mask]
        order_cal = np.argsort(lab[cal_mask], kind="stable")                      # class order, as the training calibration
        cal_ids = cal_ids[order_cal]
        keep = np.ones(len(ids), bool)
        keep[np.isin(ids, cal_ids)] = False
        test_ids, test_flag = ids[keep], flag[keep]
        logS_t = {seed: S[keep] for seed, S in logS.items()}
        rows_t = np.array([D["row"][x] for x in test_ids])
        rows_c = np.array([D["row"][x] for x in cal_ids])
        terms, meta = {}, {"part": part, "split": s, "ds": ds, "n_test": int(len(test_ids)), "n_id_test": int((~test_flag).sum()),
                           "n_ood": int(test_flag.sum()), "n_cal": int(len(cal_ids)), "calibration": "4 ID validation images per class",
                           "support": "12 training images per class (paper draw)", "views": {}}
        for v in VIEWS:
            cal = P.l2n(D["ev"][v][rows_c - 4000])
            sf = P.l2n(D["ev"][v][rows_t - 4000])
            g = C.Graph(sup[v], cal, sf, device)
            out, m = C.read_out(g, test_flag)
            g.free()
            p_id = np.exp(out["lp"][~test_flag])
            m["uniformity"] = {f"le{a:g}": float((p_id <= a).mean()) for a in ALPHAS}
            terms[(v, "lp")] = out["lp"]
            for q in P.QS:
                terms[(v, f"ln|q{q:g}")] = out[f"ln|q{q:g}"]
            terms[(v, "knn")] = C.knn_rank(sup[v], cal, sf, dev=device)
            terms[(v, "proto")] = P.proto_rank(sup[v], cal, sf, dev=device)
            meta["views"][v] = m
        rows_out = []
        for vs in VIEWSETS:
            for ro, ks in READOUTS.items():
                x = sum(terms[(v, k)] for v in vs for k in ks)
                au, fp = measures(x, test_flag)
                rows_out.append({"part": part, "split": s, "ds": ds, "seed": -1, "views": "+".join(vs), "readout": ro, "base": "none", "AUROC": au, "FPR95": fp})
                for seed, S in logS_t.items():
                    au, fp = measures(x + S, test_flag)
                    rows_out.append({"part": part, "split": s, "ds": ds, "seed": int(seed), "views": "+".join(vs), "readout": ro, "base": "TINS", "AUROC": au, "FPR95": fp})
        meta["seconds"] = round(time.time() - t0, 1)
        pd.DataFrame(rows_out).to_csv(out_root / f"split{s}" / f"{ds}.csv", index=False)
        (out_root / f"split{s}" / f"{ds}.meta.json").write_text(json.dumps(meta, indent=1) + "\n")
        d3 = meta["views"]["D3L"]
        print(json.dumps({"part": part, "split": s, "ds": ds, "n_id_test": meta["n_id_test"], "D3L q0.2": d3["q0.2"], "uniform": d3["uniformity"],
                          "seconds": meta["seconds"], "utc": P.utc()}), flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", required=True, choices=["openood", "fourood"])
    ap.add_argument("--worker", type=int, default=0)
    ap.add_argument("--nworkers", type=int, default=1)
    a = ap.parse_args()
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    run_part(a.part, a.worker, a.nworkers)
