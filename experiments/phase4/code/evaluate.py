"""Phase 4 step E: confirmatory evaluation of the locked selections on the unused banks U1-U3.

Per stream (bank, split, draw, order): the full adaptive pipeline is re-run for every family at its dev1-selected
configuration, the frozen REPRISE (rep_L0 at k10 g1 l0.9), the reference configuration of every family, and the
converged standard LP (raw_L2) at the raw family's selection. Bases: none, TINS, MCM; weighted control with the
locked a*. Output: results/eval/<stream>.parquet (+ _scores.npz with the key log-scores).
"""
import argparse
import json
import os
import time

import numpy as np
import pandas as pd

from bank_data import Features, stream_inputs
from common import RESULTS, ROOT, V5, sha_file, utc
from engine import run_view
from metrics_p4 import metrics
from static import static_view

REF = "k10g1l0.9"
OUT = RESULTS / "eval"
FAMS = ("raw", "cdf", "z", "mass", "cdf_L0", "rw", "sprop")


def parse(cfg):
    k, rest = cfg[1:].split("g")
    g, l = rest.split("l")
    return int(k), float(g), float(l)


def load_lock():
    lock_b = (ROOT / "selection_lock.json").read_bytes()
    assert sha_file(ROOT / "selection_lock.json") == (ROOT / "selection_lock.sha256").read_text().strip()
    return json.loads(lock_b)


def base_family(f):
    return {"stat_cdf": "cdf", "rep_L1": "cdf", "stat_cdf_L0": "cdf_L0", "rep_L0": "cdf_L0"}.get(f, f)


def assemble(S, fam, cfg):
    b = base_family(fam)
    x = S[f"{b}|{cfg}"]
    if fam.startswith("stat_"):
        x = S["static"] + x
    if fam.startswith("rep_"):
        x = S["Mpt"] + x
    return x


def run_stream(F, lock, bank, split, draw, seed):
    name = f"{bank}_s{split}_d{draw}_seed{seed}"
    if (OUT / f"{name}.parquet").exists():
        return
    t0 = time.time()
    inp = stream_inputs(F, bank, split, draw, seed)
    sel = lock["selected"]
    want = {(f, sel[f]["config"]) for f in sel} | {(f, REF) for f in sel}
    cfgs = sorted({c for _, c in want})
    graphs = sorted({parse(c)[:2] for c in cfgs})
    lams = sorted({parse(c)[2] for c in cfgs})
    rk, rg, rl = parse(sel["raw"]["config"])
    views = {}
    for v in ("B14", "L14"):
        st = static_view(inp[f"sup_{v}"], inp[f"cal_{v}"], inp[f"sf_{v}"], inp["cand_c"], inp["cand_s"], V5["n0"], V5["m"])
        out, _ = run_view(inp[f"sup_{v}"], inp[f"cal_{v}"], inp[f"sf_{v}"], inp["bidx"], st, V5["thresholds"],
                          tuple(graphs), tuple(lams), "cuda", FAMS)
        conv, _ = run_view(inp[f"sup_{v}"], inp[f"cal_{v}"], inp[f"sf_{v}"], inp["bidx"], st, V5["thresholds"],
                           ((rk, rg),), (rl,), "cuda", (), converged=("raw_L2",), memory=False)
        out[f"raw_L2|{sel['raw']['config']}"] = conv[f"raw_L2|{sel['raw']['config']}"]
        views[v] = out
    keys = [k for k in views["B14"] if not k.startswith("_")]
    S = {k: views["B14"][k] + views["L14"][k] for k in keys}
    flag = inp["flag"]
    logS = np.log(inp["S"])
    logM = np.log(inp["mcm"])
    scores = {}
    for f, c in want:
        scores[f"{f}|{c}"] = assemble(S, f, c)
    scores[f"raw_L2|{sel['raw']['config']}"] = S[f"raw_L2|{sel['raw']['config']}"]
    scores["static|-"], scores["Mpt|-"] = S["static"], S["Mpt"]
    rows = []
    for key, x in scores.items():
        f, c = key.split("|")
        role = "frozen" if (f == "rep_L0" and c == REF) else ("selected" if (f in sel and sel[f]["config"] == c) else "reference")
        if f == "raw_L2":
            role = "selected"
        for base, b in (("none", 0.0), ("TINS", logS), ("MCM", logM)):
            rows.append({"family": f, "config": c, "role": role, "base": base, "a": 1.0, **metrics(x + b, flag)})
    for f, w in lock["weighted"].items():
        c = sel[f]["config"]
        x = scores[f"{f}|{c}"]
        rows.append({"family": f, "config": c, "role": "weighted", "base": "TINS", "a": w["a"], **metrics(logS + w["a"] * x, flag)})
    df = pd.DataFrame(rows).assign(bank=bank, split=split, draw=draw, seed=seed, n_id=int((~flag).sum()), n_ood=int(flag.sum()))
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT / f"{name}_scores.npz", sample_id=np.array(inp["ids"]), is_ood=flag, logS=logS, logM=logM,
                        bidx=inp["bidx"], admit_B14=views["B14"]["_admit"], admit_L14=views["L14"]["_admit"],
                        **{k.replace("|", "__"): v for k, v in scores.items()})
    tmp = OUT / f"{name}.{os.getpid()}.parquet"
    df.to_parquet(tmp, index=False)
    os.replace(tmp, OUT / f"{name}.parquet")
    print(json.dumps({"stream": name, "seconds": round(time.time() - t0, 1), "utc": utc()}), flush=True)


def tasks():
    t = [("U1", s, d, sd) for s in range(1, 6) for d in range(3) for sd in (123, 124, 125)]
    t += [("U2", 0, d, sd) for d in range(5) for sd in (123, 124, 125)]
    t += [("U3", 0, d, sd) for d in range(5) for sd in (123, 124, 125)]
    return t


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--worker", type=int, default=0)
    ap.add_argument("--nworkers", type=int, default=1)
    ap.add_argument("--banks", default="U1,U2,U3")
    a = ap.parse_args()
    lock = load_lock()
    F = Features()
    todo = [t for t in tasks() if t[0] in a.banks.split(",")][a.worker::a.nworkers]
    for t in todo:
        run_stream(F, lock, *t)
