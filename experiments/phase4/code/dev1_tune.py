"""Phase 4 step T: equal-budget tuning of every configuration family on dev1 (the already used selection split).

Grid for every family: k_g in {5,10,20} x gamma in {0,1,3} x lambda in {0.8,0.9,0.99} (27 configurations; the
frozen REPRISE graph k10 g1 l0.9 is one of them). Families (log-scores summed over B/14 and L/14):
  raw     standard LP u (all nodes from 0, 15 sweeps)          cdf      current-calibration rank of u (0-start)
  cdf_L0  current-calibration rank, warm start (REPRISE LP)    z        Phi((u - median_cal) / MAD_cal)
  mass    u * N_t / N_s (propagated-mass correction)          rw       random-walk normalised LP u
  sprop   static prototype distance propagated on the graph, current-calibration rank
  stat_cdf(_L0) static p x cdf(_L0)     rep_L1 / rep_L0  REPRISE memory p_t x cdf / cdf_L0
Bases: none (s0 = 1), TINS, MCM; weighted control: log TINS + a x (visual log-score), a in {0.5,1.5,2,3}, every family.
Output: results/dev1_tune/<task>.parquet (metrics) and <task>_ref.npz (reference-configuration scores).
"""
import argparse
import json
import os
import time

import numpy as np
import pandas as pd
import torch

from common import RESULTS, V5, dump, utc
from engine import run_view
from metrics_p4 import metrics
from pilot import load_development, prepare
from static import from_proto

GRAPHS = tuple((k, g) for k in (5, 10, 20) for g in (0.0, 1.0, 3.0))
LAMS = (0.8, 0.9, 0.99)
FAMILIES = ("raw", "cdf", "z", "mass", "cdf_L0", "rw", "sprop")
A_LIST = (0.5, 1.5, 2.0, 3.0)
REF = "k10g1l0.9"
OUT = RESULTS / "dev1_tune"


def family_scores(S):
    fam = {("static", "-"): S["static"], ("Mpt", "-"): S["Mpt"]}
    for key, x in S.items():
        if "|" not in key:
            continue
        f, cfg = key.split("|")
        fam[(f, cfg)] = x
        if f == "cdf":
            fam[("stat_cdf", cfg)] = S["static"] + x
            fam[("rep_L1", cfg)] = S["Mpt"] + x
        if f == "cdf_L0":
            fam[("stat_cdf_L0", cfg)] = S["static"] + x
            fam[("rep_L0", cfg)] = S["Mpt"] + x
    return fam


def mcm_table(D):
    s = torch.as_tensor(D["st_sims"], dtype=torch.float64)
    return torch.softmax(s, dim=1).max(1).values.numpy()


def run_task(R, D, mcm_all, draw, stream, seed):
    name = f"draw{draw}_{stream}_seed{seed}"
    if (OUT / f"{name}.parquet").exists():
        return
    t0 = time.time()
    views = {}
    for v in ("B14", "L14"):
        su, ca, sf, bi, vv, sid, flag, S = prepare(R, D, draw, v, stream, seed)
        views[v], _ = run_view(su, ca, sf, bi, from_proto(vv, su, V5["n0"]), V5["thresholds"], GRAPHS, LAMS, "cuda", FAMILIES)
    keys = [k for k in views["B14"] if not k.startswith("_")]
    Sc = {k: views["B14"][k] + views["L14"][k] for k in keys}
    fam = family_scores(Sc)
    logS = np.log(S.astype(np.float64))
    logM = np.log(mcm_all[np.array([D["stream_row"][s] for s in sid])])
    rows = []
    for (f, cfg), x in fam.items():
        for base, b in (("none", 0.0), ("TINS", logS), ("MCM", logM)):
            rows.append({"family": f, "config": cfg, "base": base, "a": 1.0, **metrics(x + b, flag)})
        if f not in ("static", "Mpt"):
            for a in A_LIST:
                rows.append({"family": f, "config": cfg, "base": "TINS", "a": a, **metrics(logS + a * x, flag)})
    df = pd.DataFrame(rows).assign(draw=draw, stream=stream, seed=seed)
    OUT.mkdir(parents=True, exist_ok=True)
    ref = {f"{f}": x for (f, cfg), x in fam.items() if cfg in (REF, "-")}
    np.savez_compressed(OUT / f"{name}_ref.npz", sample_id=np.array(sid), is_ood=flag, logS=logS, logM=logM,
                        admit_B14=views["B14"]["_admit"], admit_L14=views["L14"]["_admit"], **ref)
    tmp = OUT / f"{name}.{os.getpid()}.parquet"
    df.to_parquet(tmp, index=False)
    os.replace(tmp, OUT / f"{name}.parquet")
    print(json.dumps({"task": name, "seconds": round(time.time() - t0, 1), "utc": utc()}), flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--worker", type=int, default=0)
    ap.add_argument("--nworkers", type=int, default=1)
    ap.add_argument("--only", default="")
    a = ap.parse_args()
    torch.set_num_threads(2)
    tasks = [(d, s, sd) for d in range(5) for s in ("near", "far") for sd in (123, 124, 125)]
    if a.only:
        tasks = [t for t in tasks if f"draw{t[0]}_{t[1]}_seed{t[2]}" in a.only.split(",")]
    tasks = tasks[a.worker::a.nworkers]
    R, D = load_development()
    mcm_all = mcm_table(D)
    for t in tasks:
        run_task(R, D, mcm_all, *t)
