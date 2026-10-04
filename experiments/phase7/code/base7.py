"""Static same-feature references for the dataset study (D): 16-shot kNN and Mahalanobis++ (Phase 3 registered
parameters), each converted to a calibrated rank per view; the two-view score is the product (as Phase 3).
Output: <P7>/results/ds_base/<stream>.npz with log p sums over the views B14 + L14 (knn, maha)."""
import json
import os
import sys

import numpy as np

import an7 as A
import data7
from p7common import GONOGO, RESULTS
from vins import r5

VIEWS = ("B14", "L14")
PR = json.loads((GONOGO / "r5" / "phase3" / "prereg_phase3.json").read_text())["baselines"]


def problem(task):
    if task.startswith("cubssb_"):
        return data7.cub_ssb(task.split("_")[1])
    if task.startswith("U1_"):
        return data7.u1(int(task.split("_")[1][1:]), 0)
    name, s = task.split("_")[0], int(task.split("_")[1][1:])
    return data7.ds(name, s)


def main():
    out_dir = RESULTS / "ds_base"
    out_dir.mkdir(parents=True, exist_ok=True)
    src = "dsw" if A.tasks("dsw", "B14") else "ds"
    ts = A.tasks(src, "B14")
    w, nw = (int(sys.argv[1]), int(sys.argv[2])) if len(sys.argv) > 2 else (0, 1)
    for t in ts[w::nw]:
        if (out_dir / f"{t}.npz").exists():
            continue
        z = np.load(A.RESULTS / src / "B14" / f"{t}.npz", allow_pickle=True)
        ids = z["sample_id"].tolist()
        P = problem(t)
        knn, maha = 0.0, 0.0
        for v in VIEWS:
            sup = P.feat(v, P.sup_ids).reshape(P.C, 12, -1)
            cal, sf = P.feat(v, P.cal_ids), P.feat(v, ids)
            q = np.concatenate([cal, sf])
            nc = len(cal)
            kd = r5.knn_distance(sup.reshape(-1, sup.shape[-1]), q, (PR["knn_k"],))[PR["knn_k"]]
            md = r5.maha_pp(sup, q, (PR["maha_lam"],))[PR["maha_lam"]]
            knn = knn + np.log(r5.pval_high(kd[:nc], kd[nc:]))
            maha = maha + np.log(r5.pval_high(md[:nc], md[nc:]))
        tmp = out_dir / f"{t}.{os.getpid()}.tmp.npz"
        np.savez_compressed(tmp, sample_id=z["sample_id"], is_ood=z["is_ood"], knn=knn, maha=maha)
        os.replace(tmp, out_dir / f"{t}.npz")
        print(t, flush=True)
    print("BASE7_OK", PR)


if __name__ == "__main__":
    main()
