"""Phase 5: evaluation of read-out combinations on the saved development scores (labels are used here only).

A combination is a list of (key, weight) pairs; its score is sum over views of weight x log p (ID-high), plus the
base detector (none / TINS / MCM). Units: support draws (the three orders averaged), paired t interval (df = 4).
"""
import argparse
import glob
import json
import os
import re
from pathlib import Path

import numpy as np

from metrics_p4 import metrics, tci

OUT = Path(os.environ.get("P5_RESULTS", "/home/omote/reprise_p5_20261002/results"))
REFT = "k10g1b1l0.9"
FROZEN = [("M0", 1.0), (f"lp0|{REFT}", 1.0)]


class Z:
    """npz with an in-memory cache (NpzFile decompresses an array on every access)."""

    def __init__(self, path):
        self.z = np.load(path, allow_pickle=True)
        self.files = self.z.files
        self.c = {}

    def __getitem__(self, k):
        if k not in self.c:
            self.c[k] = self.z[k]
        return self.c[k]


_CACHE = {}


def load(cfg, dev="dev1", stream="near"):
    key = (cfg, dev, stream)
    if key not in _CACHE:
        tasks = {}
        for f in sorted(glob.glob(str(OUT / cfg / f"{dev}_draw*_{stream}_seed*.npz"))):
            m = re.search(r"draw(\d+)_(\w+)_seed(\d+)\.npz$", f)
            tasks[(int(m.group(1)), int(m.group(3)))] = Z(f)
        _CACHE[key] = tasks
    return _CACHE[key]


def views_of(z):
    return sorted({k.split("::")[1] for k in z.files if k.startswith("s::")})


def keys_of(z, view=None):
    v = view or views_of(z)[0]
    return sorted(k.split("::", 2)[2] for k in z.files if k.startswith(f"s::{v}::"))


def visual(z, combo, views=None):
    x = 0.0
    for v in (views or views_of(z)):
        for key, w in combo:
            x = x + w * z[f"s::{v}::{key}"]
    return x


def evaluate(tasks, combo, views=None, base="none", a=1.0):
    """{(draw, seed): {AUROC, FPR95}}"""
    out = {}
    for t, z in tasks.items():
        x = a * visual(z, combo, views)
        if base == "TINS":
            x = x + z["logS"]
        elif base == "MCM":
            x = x + z["logM"]
        out[t] = metrics(x, z["is_ood"])
    return out


def draw_means(met, key):
    draws = sorted({d for d, _ in met})
    return np.array([np.mean([met[(d, s)][key] for (dd, s) in met if dd == d]) for d in draws])


def summarize(met):
    return {k: float(draw_means(met, k).mean()) for k in ("AUROC", "FPR95")}


def diff(met, ref, key="FPR95"):
    return tci(draw_means(met, key) - draw_means(ref, key))


def table(cfg, combos, dev="dev1", views=None, bases=("none", "TINS"), ref=None):
    """combos: {name: combo}. Prints near / far FPR95 and AUROC with the paired difference to the reference."""
    res = {}
    T = {s: load(cfg, dev, s) for s in ("near", "far")}
    ref = ref or FROZEN
    for base in bases:
        R = {s: evaluate(T[s], ref, views, base) for s in T if T[s]}
        for name, combo in combos.items():
            row = {}
            for s in R:
                met = evaluate(T[s], combo, views, base)
                d = diff(met, R[s])
                da = diff(met, R[s], "AUROC")
                row[s] = {**summarize(met), "dFPR": d["mean"], "lo": d["lo"], "hi": d["hi"], "dAUROC": da["mean"]}
            res[(base, name)] = row
    return res


def show(res, title=""):
    print(f"== {title}")
    print(f"{'base':5s} {'combination':34s} | near AUROC  FPR95   dFPR [95% CI]            | far AUROC  FPR95   dFPR")
    for (base, name), row in res.items():
        n, f = row.get("near"), row.get("far")
        s = f"{base:5s} {name:34s} |"
        if n:
            s += f" {n['AUROC']:6.2f} {n['FPR95']:6.2f} {n['dFPR']:+6.2f} [{n['lo']:+6.2f},{n['hi']:+6.2f}] |"
        if f:
            s += f" {f['AUROC']:6.2f} {f['FPR95']:6.2f} {f['dFPR']:+6.2f}"
        print(s)


def greedy(cfg, pool, dev="dev1", views=None, base="none", start=None, steps=4, far_tol=0.5, weights=(1.0,)):
    """Forward selection on dev1 near FPR95 (far FPR95 may not exceed the frozen value by more than far_tol)."""
    T = {s: load(cfg, dev, s) for s in ("near", "far")}
    ref = {s: summarize(evaluate(T[s], FROZEN, views, base)) for s in T}
    cur = list(start or [])
    hist = []
    for _ in range(steps):
        best = None
        for key in pool:
            if any(key == k for k, _ in cur):
                continue
            for w in weights:
                c = cur + [(key, w)]
                n = summarize(evaluate(T["near"], c, views, base))
                f = summarize(evaluate(T["far"], c, views, base))
                if f["FPR95"] > ref["far"]["FPR95"] + far_tol:
                    continue
                if best is None or n["FPR95"] < best[1]["FPR95"]:
                    best = (c, n, f)
        if best is None:
            break
        cur = best[0]
        hist.append({"combo": cur, "near": best[1], "far": best[2]})
        print(json.dumps({"step": len(hist), "added": cur[-1], "near_FPR95": round(best[1]["FPR95"], 2),
                          "near_AUROC": round(best[1]["AUROC"], 2), "far_FPR95": round(best[2]["FPR95"], 2)}), flush=True)
    return hist, ref


def standard_combos(z):
    ks = keys_of(z)
    lp = lambda t: f"lp|{t}"
    C = {"frozen (M0 x lp0)": FROZEN, "static": [("static", 1.0)], "M0": [("M0", 1.0)],
         "lp0 (warm)": [(f"lp0|{REFT}", 1.0)], "lp (zero)": [(lp(REFT), 1.0)],
         "M0 x lp": [("M0", 1.0), (lp(REFT), 1.0)], "M0 x z": [("M0", 1.0), (f"z|{REFT}", 1.0)]}
    for k in ks:
        if k.startswith("lp|") and k != lp(REFT):
            C[f"M0 x {k}"] = [("M0", 1.0), (k, 1.0)]
        if k.startswith(("cr|", "cz|")):
            C[f"M0 x {k}"] = [("M0", 1.0), (k, 1.0)]
        if k.startswith(("Mlp", "Mfin", "Mjoint")):
            C[f"{k} x lp"] = [(k, 1.0), (lp(REFT), 1.0)]
        if k.startswith("neg:"):
            C[f"M0 x lp x {k.split('|')[0]}"] = [("M0", 1.0), (lp(REFT), 1.0), (k, 1.0)]
        if k in ("ctr1", "ctr3", "gall", "M0a1", "M0a2", "static"):
            C[f"M0 x lp x {k}"] = [("M0", 1.0), (lp(REFT), 1.0), (k, 1.0)]
        if k.startswith(("cmax|", "cown|", "cpur|", "ep|", "epc|")):
            C[f"M0 x lp x {k.split('|')[0]}"] = [("M0", 1.0), (lp(REFT), 1.0), (k, 1.0)]
            C[f"M0 x {k.split('|')[0]}"] = [("M0", 1.0), (k, 1.0)]
    return C


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", default="base")
    ap.add_argument("--dev", default="dev1")
    ap.add_argument("--views", default="")
    ap.add_argument("--greedy", type=int, default=0)
    a = ap.parse_args()
    T = load(a.cfg, a.dev, "near")
    z = next(iter(T.values()))
    views = a.views.split(",") if a.views else None
    print("tasks near:", len(T), "views:", views_of(z), "keys:", len(keys_of(z)))
    res = table(a.cfg, standard_combos(z), a.dev, views)
    show(res, f"{a.cfg} / {a.dev} / views {views or 'all'}")
    if a.greedy:
        pool = [k for k in keys_of(z)]
        greedy(a.cfg, pool, a.dev, views, steps=a.greedy)
