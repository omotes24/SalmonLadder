"""Phase 6: DINOv3 views in place of the DINOv2 views (no tuning) -- paired comparison on the development splits.

Reads results/lock (DINOv2 B/14 + L/14), results/d3 (DINOv3 B/16 + L/16 at 256 px, pre-registered primary) and
results/d3s (the same encoders on the exact DINOv2 input, 224 px). The two methods are frozen REPRISE v5 and the locked
two-sided propagation; every hyper-parameter is the one of the DINOv2 pipeline. Units: support draws (the three stream
orders averaged); paired t interval (df = 4). Descriptive: nothing is selected here.
"""
import json
import os
import sys
from pathlib import Path

import numpy as np

import analyze5 as A
from metrics_p4 import tci

LOCK = Path(os.environ.get("P5_LOCK", "/home/omote/reprise_p5_20261002/selection_lock_p5.json"))
ENC = {"DINOv2": ("lock", ["B14", "L14"]), "DINOv3@256": ("d3", ["D3B", "D3L"]), "DINOv3@224": ("d3s", ["D3Bs", "D3Ls"])}
S2 = "IB0.2s<B0.5s"


def methods():
    L = json.loads(LOCK.read_text())
    return {"v5": A.FROZEN, "two-sided": [tuple(c) for c in L["score"]]}


def check_alignment(T0, T1, what):
    assert set(T0) == set(T1), (what, sorted(set(T0) ^ set(T1)))
    for t in T0:
        for k in ("sample_id", "is_ood", "bidx", "logS", "logM"):
            assert np.array_equal(T0[t][k], T1[t][k]), (what, t, k)


def fmt(s):
    return f"{s['AUROC']:6.2f} /{s['FPR95']:6.2f}"


def dfmt(d):
    return f"{d['mean']:+6.2f} [{d['lo']:+6.2f},{d['hi']:+6.2f}]"


def seed_stats(T, view):
    a, b, p = [], [], []
    for z in T.values():
        last = int(np.flatnonzero(z["bidx"] == z["bidx"].max())[0])
        m, f = z[f"a::{view}::{S2}"][:last], z["is_ood"][:last]
        a.append(m[~f].mean())
        b.append(m[f].mean())
        p.append(m[f].sum() / max(m.sum(), 1))
    return [100 * float(np.mean(x)) for x in (a, b, p)]


def main(dev):
    M = methods()
    out = {"dev": dev, "cells": {}, "single_view": {}, "gain_two_sided_over_v5": {}, "seed_sets": {}, "n_tasks": {}}
    avail = {}
    for enc, (cfg, views) in ENC.items():
        T = {s: A.load(cfg, dev, s) for s in ("near", "far")}
        if all(len(T[s]) == 15 for s in T):
            avail[enc] = (T, views)
        out["n_tasks"][enc] = {s: len(T[s]) for s in T}
    print(f"##### {dev}: tasks {out['n_tasks']}")
    if "DINOv2" not in avail:
        print("DINOv2 results incomplete")
        return
    for enc in avail:
        if enc != "DINOv2":
            for s in ("near", "far"):
                check_alignment(avail["DINOv2"][0][s], avail[enc][0][s], (dev, enc, s))
    print("streams, labels, batches, TINS and MCM scores are identical across encoders (checked)")
    for stream in ("near", "far"):
        for base in ("none", "TINS", "MCM"):
            print(f"== {dev} {stream}, base = {base}: AUROC / FPR95; difference to DINOv2 (FPR95, then AUROC) [95% CI]")
            for name, combo in M.items():
                ref = A.evaluate(avail["DINOv2"][0][stream], combo, avail["DINOv2"][1], base)
                for enc, (T, views) in avail.items():
                    met = ref if enc == "DINOv2" else A.evaluate(T[stream], combo, views, base)
                    s = A.summarize(met)
                    cell = {"AUROC": s["AUROC"], "FPR95": s["FPR95"]}
                    line = f"  {name:10s} {enc:11s} {fmt(s)}"
                    if enc != "DINOv2":
                        d, da = A.diff(met, ref, "FPR95"), A.diff(met, ref, "AUROC")
                        cell.update(dFPR95=d, dAUROC=da,
                                    per_draw_dFPR95=(A.draw_means(met, "FPR95") - A.draw_means(ref, "FPR95")).tolist())
                        line += f" | dFPR95 {dfmt(d)} | dAUROC {dfmt(da)}"
                    out["cells"][f"{stream}|{base}|{name}|{enc}"] = cell
                    print(line)
    print(f"== {dev}: gain of the two-sided propagation over v5 within each encoder (FPR95 difference, standalone and x TINS)")
    for enc, (T, views) in avail.items():
        for stream in ("near", "far"):
            for base in ("none", "TINS"):
                d = A.diff(A.evaluate(T[stream], M["two-sided"], views, base), A.evaluate(T[stream], M["v5"], views, base), "FPR95")
                out["gain_two_sided_over_v5"][f"{enc}|{stream}|{base}"] = d
                print(f"  {enc:11s} {stream:4s} {base:4s} {dfmt(d)}")
    print(f"== {dev}: single views, standalone (AUROC / FPR95); the seed set of the two-sided method stays the joint one")
    for enc, (T, views) in avail.items():
        for stream in ("near", "far"):
            for name, combo in M.items():
                row = []
                for v in views:
                    s = A.summarize(A.evaluate(T[stream], combo, [v], "none"))
                    out["single_view"][f"{enc}|{stream}|{name}|{v}"] = s
                    row.append(f"{v} {fmt(s)}")
                print(f"  {enc:11s} {stream:4s} {name:10s} | " + " | ".join(row))
    print(f"== {dev}: seed set S2 at the last batch (ID% in set / OOD% in set / purity%), per view")
    for enc, (T, views) in avail.items():
        for stream in ("near", "far"):
            for v in views:
                st = seed_stats(T[stream], v)
                out["seed_sets"][f"{enc}|{stream}|{v}"] = st
                print(f"  {enc:11s} {stream:4s} {v:5s} {st[0]:5.1f} {st[1]:5.1f} {st[2]:5.1f}")
    path = A.OUT / f"d3_summary_{dev}.json"
    path.write_text(json.dumps(out, indent=1, default=float) + "\n")
    print("written", path)


if __name__ == "__main__":
    for dev in sys.argv[1:] or ("dev1", "dev2"):
        main(dev)
