"""Phase 6, post hoc (after the pre-registered dev comparison was seen): which DINOv3 view carries the change?

1. Per-view read-outs of the pre-registered runs (static score, memory M0, zero-start propagation, v5).
2. Variants chosen after seeing the scores: DINOv3 L/16 alone, DINOv2 B/14 + DINOv3 L/16, DINOv2 L/14 + DINOv3 L/16,
   and DINOv2 L/14 alone as the single-view reference. Same locked specification, no tuning.
Descriptive only: these choices were made on dev1/dev2 scores and carry no claim for unused data."""
import json
import sys

import numpy as np

import analyze5 as A
from d3_analysis import dfmt, fmt, methods, seed_stats

ENC = {"DINOv2 B+L (ref)": ("lock", ["B14", "L14"]), "DINOv3 B+L": ("d3", ["D3B", "D3L"]),
       "DINOv2 L": ("l14", ["L14"]), "DINOv3 L": ("d3L", ["D3L"]),
       "DINOv2 B + DINOv3 L": ("d3mixB", ["B14", "D3L"]), "DINOv2 L + DINOv3 L": ("d3mixL", ["L14", "D3L"])}
PARTS = {"static": [("static", 1.0)], "M0": [("M0", 1.0)], "lp (zero start)": [("lp|k10g1b1l0.9", 1.0)], "v5": A.FROZEN}


def main(dev):
    M = methods()
    out = {"dev": dev, "parts": {}, "cells": {}, "gain": {}, "seed_sets": {}, "consistency": {}}
    print(f"##### {dev}: per-view parts of the pre-registered runs, standalone (AUROC / FPR95)")
    for stream in ("near", "far"):
        for cfg, views in (("lock", ["B14", "L14"]), ("d3", ["D3B", "D3L"])):
            T = A.load(cfg, dev, stream)
            for v in views:
                row = []
                for name, combo in PARTS.items():
                    s = A.summarize(A.evaluate(T, combo, [v], "none"))
                    out["parts"][f"{stream}|{v}|{name}"] = s
                    row.append(f"{name} {fmt(s)}")
                print(f"  {stream:4s} {v:4s} | " + " | ".join(row))
    avail = {}
    for enc, (cfg, views) in ENC.items():
        T = {s: A.load(cfg, dev, s) for s in ("near", "far")}
        if all(len(T[s]) == 15 for s in T):
            avail[enc] = (T, views)
        else:
            print("incomplete:", enc, {s: len(T[s]) for s in T})
    ref_enc = "DINOv2 B+L (ref)"
    for stream in ("near", "far"):
        for base in ("none", "TINS"):
            print(f"== {dev} {stream}, base = {base}: AUROC / FPR95; difference to DINOv2 B+L (FPR95, then AUROC) [95% CI]")
            for name, combo in M.items():
                ref = A.evaluate(avail[ref_enc][0][stream], combo, avail[ref_enc][1], base)
                for enc, (T, views) in avail.items():
                    met = ref if enc == ref_enc else A.evaluate(T[stream], combo, views, base)
                    s = A.summarize(met)
                    cell = {"AUROC": s["AUROC"], "FPR95": s["FPR95"]}
                    line = f"  {name:10s} {enc:20s} {fmt(s)}"
                    if enc != ref_enc:
                        d, da = A.diff(met, ref, "FPR95"), A.diff(met, ref, "AUROC")
                        cell.update(dFPR95=d, dAUROC=da)
                        line += f" | dFPR95 {dfmt(d)} | dAUROC {dfmt(da)}"
                    out["cells"][f"{stream}|{base}|{name}|{enc}"] = cell
                    print(line)
    print(f"== {dev}: DINOv3 L alone against DINOv2 L alone (true single-view runs), FPR95 difference [95% CI]")
    if "DINOv3 L" in avail and "DINOv2 L" in avail:
        for stream in ("near", "far"):
            for base in ("none", "TINS"):
                for name, combo in M.items():
                    d = A.diff(A.evaluate(avail["DINOv3 L"][0][stream], combo, ["D3L"], base),
                               A.evaluate(avail["DINOv2 L"][0][stream], combo, ["L14"], base), "FPR95")
                    out["cells"][f"LvsL|{stream}|{base}|{name}"] = d
                    print(f"  {stream:4s} {base:4s} {name:10s} {dfmt(d)}")
    print(f"== {dev}: gain of the two-sided propagation over v5 within each configuration (FPR95 difference)")
    for enc, (T, views) in avail.items():
        row = []
        for stream in ("near", "far"):
            for base in ("none", "TINS"):
                d = A.diff(A.evaluate(T[stream], M["two-sided"], views, base), A.evaluate(T[stream], M["v5"], views, base), "FPR95")
                out["gain"][f"{enc}|{stream}|{base}"] = d
                row.append(f"{stream} {base} {dfmt(d)}")
        print(f"  {enc:20s} " + " | ".join(row))
    print(f"== {dev}: seed set S2 at the last batch (ID% / OOD% / purity%)")
    for enc, (T, views) in avail.items():
        row = []
        for stream in ("near", "far"):
            st = seed_stats(T[stream], views[0])
            out["seed_sets"][f"{enc}|{stream}"] = st
            row.append(f"{stream} {st[0]:5.1f} {st[1]:5.1f} {st[2]:5.1f}")
        print(f"  {enc:20s} " + " | ".join(row))
    # consistency: the v5 read-out of one view does not depend on the other view of the run
    if "DINOv3 L" in avail:
        a = A.summarize(A.evaluate(avail["DINOv3 L"][0]["near"], A.FROZEN, ["D3L"], "none"))
        b = A.summarize(A.evaluate(avail["DINOv3 B+L"][0]["near"], A.FROZEN, ["D3L"], "none"))
        out["consistency"]["v5 D3L alone vs read-out"] = [a, b]
        print(f"consistency (v5, near): D3L single-view run {fmt(a)} vs D3L read-out of the two-view run {fmt(b)}")
    path = A.OUT / f"d3_posthoc_{dev}.json"
    path.write_text(json.dumps(out, indent=1, default=float) + "\n")
    print("written", path)


if __name__ == "__main__":
    for dev in sys.argv[1:] or ("dev2", "dev1"):
        main(dev)
