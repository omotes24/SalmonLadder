"""Exp 8 (descriptive): label budget and view dependence on U1 (splits 1-5, draw 0 [and 1 for 32 shots], order 123).
Budgets per ID class (support + calibration): 3+1, 6+2, 12+4 (registered setting), 24+8, and 8+8 / 14+2 at 16.
Single-view condition: L/14 only, no CLIP candidate restriction (d = d_all), 12+4.
Families: frozen REPRISE, best minimal (z), static x p_LP, all at k10 g1 l0.9; s0 = 1."""
import json
import os
import time

import numpy as np
import pandas as pd

from bank_data import Features, pos_text, spec
from common import RESULTS, V5, utc
from engine import run_view
from metrics_p4 import metrics
from static import static_view
from vins.tins_dev import build_order

OUT = RESULTS / "exp8"
CONDS = {"b4": (3, 1), "b8": (6, 2), "b16": (12, 4), "b32": (24, 8), "b16_8+8": (8, 8), "b16_14+2": (14, 2)}


def shots(shots_f, C, n_sup, n_cal):
    """Draw 0 provides 16 per class, draw 1 the next 16 (for the 32-shot budget); within a draw the registered
    12/4 permutation positions are reused so that 12+4 equals the registered split."""
    s = shots_f[shots_f.draw.isin([0, 1])].sort_values(["class_rank", "draw", "pos"])
    sup, cal = [], []
    for c in range(C):
        rows = s[s.class_rank == c]
        d0 = rows[rows.draw == 0]
        d1 = rows[rows.draw == 1]
        order = list(d0[d0.pos < 12].sample_id) + list(d1[d1.pos < 12].sample_id)
        corder = list(d0[d0.pos >= 12].sample_id) + list(d1[d1.pos >= 12].sample_id)
        pool = order + corder
        if n_sup <= 12 and n_cal <= 4:
            sup += order[:n_sup]
            cal += corder[:n_cal]
        elif n_sup + n_cal == 16:
            sup += (order[:12] + corder[:4])[:n_sup]
            cal += (order[:12] + corder[:4])[n_sup:16]
        else:
            sup += order[:n_sup]
            cal += corder[:n_cal]
    return sup, cal


def run_cond(F, k, cond):
    f_out = OUT / f"s{k}_{cond}.parquet"
    if f_out.exists():
        return
    t0 = time.time()
    names, shots_f, ide, ood = spec("U1", k)
    C = len(names)
    single = cond == "L14_noK"
    n_sup, n_cal = (12, 4) if single else CONDS[cond]
    sup_ids, cal_ids = shots(shots_f, C, n_sup, n_cal)
    order = build_order(len(ide), len(ood), 123)
    ids = [ide[i] if o == 0 else ood[i] for o, i in order]
    flag = np.array([o for o, _ in order], bool)
    pt = pos_text("U1", k)
    views = ("L14",) if single else ("B14", "L14")
    cand_s = np.argsort(-(F.get("CLIP", ids) @ pt.T), axis=1)[:, :20]
    cand_c = np.argsort(-(F.get("CLIP", cal_ids) @ pt.T), axis=1)[:, :20]
    if single:
        cand_s = np.tile(np.arange(C), (len(ids), 1))
        cand_c = np.tile(np.arange(C), (len(cal_ids), 1))
    S = {}
    for v in views:
        sup = F.get(v, sup_ids).reshape(C, n_sup, -1)
        cal = F.get(v, cal_ids)
        sf = F.get(v, ids)
        st = static_view(sup, cal, sf, cand_c, cand_s, V5["n0"], V5["m"])
        out, _ = run_view(sup, cal, sf, np.arange(len(ids)) // 256, st, V5["thresholds"], ((10, 1.0),), (0.9,), "cuda",
                          ("cdf_L0", "z"))
        for key in ("static", "Mpt", "cdf_L0|k10g1l0.9", "z|k10g1l0.9"):
            S[key] = S.get(key, 0.0) + out[key]
    fam = {"REPRISE": S["Mpt"] + S["cdf_L0|k10g1l0.9"], "best_z": S["z|k10g1l0.9"],
           "static_pLP": S["static"] + S["cdf_L0|k10g1l0.9"], "static": S["static"]}
    rows = [{"family": f, "cond": cond, "split": k, "n_sup": n_sup, "n_cal": n_cal, **metrics(x, flag)} for f, x in fam.items()]
    OUT.mkdir(parents=True, exist_ok=True)
    tmp = OUT / f"s{k}_{cond}.{os.getpid()}.parquet"
    pd.DataFrame(rows).to_parquet(tmp, index=False)
    os.replace(tmp, f_out)
    print(json.dumps({"exp8": f_out.name, "seconds": round(time.time() - t0, 1), "utc": utc()}), flush=True)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--worker", type=int, default=0)
    ap.add_argument("--nworkers", type=int, default=1)
    a = ap.parse_args()
    F = Features()
    tasks = [(k, c) for k in range(1, 6) for c in list(CONDS) + ["L14_noK"]][a.worker::a.nworkers]
    for k, c in tasks:
        run_cond(F, k, c)
