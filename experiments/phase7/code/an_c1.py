"""C1: the recurrence made visible to the memory only / the graph only / both / neither (2 x 2), from the per-query
scores of the registered intervention experiment (Phase 4 Exp 4; no new run).

score(a, b) = log p_M[H_a] + log p_LP[H_b], a, b in {0, r}: the frozen memory and propagation do not read each other,
so mixing the factors of the two histories is exactly the channel-visibility intervention.
AUROC per (split, class) pair: 8 class queries (OOD) vs 64 common ID queries. Percentile bootstrap over the 250 pairs
(10,000 resamples) and a split-clustered bootstrap."""
import json
import sys

import numpy as np
import pandas as pd

import an7 as A
from p7common import P4, SEED

SRC = P4 / "results" / "exp4"
NB = 10000


def auroc(id_s, ood_s):
    a = np.asarray(id_s)[:, None]
    b = np.asarray(ood_s)[None, :]
    return 100 * (np.mean(a > b) + 0.5 * np.mean(a == b))


def boot(x, splits, rng, clustered=False):
    x = np.asarray(x, float)
    if clustered:
        us = np.unique(splits)
        vals = []
        for _ in range(NB):
            ss = rng.choice(us, size=len(us), replace=True)
            vals.append(np.concatenate([rng.choice(x[splits == s], size=int((splits == s).sum()), replace=True) for s in ss]).mean())
    else:
        idx = rng.integers(0, len(x), size=(NB, len(x)))
        vals = x[idx].mean(1)
    return {"mean": float(x.mean()), "lo": float(np.quantile(vals, 0.025)), "hi": float(np.quantile(vals, 0.975)), "n": int(len(x)),
            "frac_pos": float((x > 0).mean())}


def main():
    base = pd.concat([pd.read_parquet(f) for f in sorted(SRC.glob("s*_r0.parquet"))], ignore_index=True)
    rows = []
    for k in range(1, 6):
        dz = json.loads((P4 / "banks" / "exp4" / f"split{k}.json").read_text())
        idq = list(dz["id_queries"])
        b = base[base.split == k].pivot(index="query", columns="family", values="score")
        for w in dz["probe"]:
            qs = list(dz["probes"][w]["queries"])
            p = pd.read_parquet(SRC / f"s{k}_{w}.parquet")
            M0i, G0i, M0o, G0o = b.loc[idq, "Mpt"].values, b.loc[idq, "pLP"].values, b.loc[qs, "Mpt"].values, b.loc[qs, "pLP"].values
            S0i, S0o = b.loc[idq, "static"].values, b.loc[qs, "static"].values
            for (cond, r), g in p.groupby(["cond", "r"]):
                t = g.pivot(index="query", columns="family", values="score")
                Mri, Gri, Mro, Gro = t.loc[idq, "Mpt"].values, t.loc[idq, "pLP"].values, t.loc[qs, "Mpt"].values, t.loc[qs, "pLP"].values
                c00 = auroc(M0i + G0i, M0o + G0o)
                cr0 = auroc(Mri + G0i, Mro + G0o)
                c0r = auroc(M0i + Gri, M0o + Gro)
                crr = auroc(Mri + Gri, Mro + Gro)
                chk = auroc(t.loc[idq, "REPRISE"].values, t.loc[qs, "REPRISE"].values)
                assert abs(chk - crr) < 1e-9, (k, w, cond, r)
                rows.append({"split": k, "wnid": w, "cond": cond, "r": int(r), "c00": c00, "cr0": cr0, "c0r": c0r, "crr": crr,
                             "E_M": cr0 - c00, "E_G": c0r - c00, "E_MG": crr - c00, "I": crr - cr0 - c0r + c00,
                             "M_alone_0": auroc(M0i, M0o), "M_alone_r": auroc(Mri, Mro), "G_alone_0": auroc(G0i, G0o), "G_alone_r": auroc(Gri, Gro),
                             "static": auroc(S0i, S0o),
                             "static_G_r": auroc(S0i + Gri, S0o + Gro), "static_G_0": auroc(S0i + G0i, S0o + G0o)})
    T = pd.DataFrame(rows)
    T.to_parquet(A.RESULTS / "c1_cells.parquet", index=False)
    rng = np.random.default_rng([SEED, 31])
    out = {"n_pairs": int(T[["split", "wnid"]].drop_duplicates().shape[0]), "table": []}
    for cond in ("same", "dup", "near", "far"):
        for r in (1, 2, 5, 10, 20):
            g = T[(T.cond == cond) & (T.r == r)].sort_values(["split", "wnid"])
            sp = g.split.values
            rec = {"cond": cond, "r": r, "cells": {c: float(g[c].mean()) for c in ("c00", "cr0", "c0r", "crr")},
                   "M_alone": [float(g.M_alone_0.mean()), float(g.M_alone_r.mean())], "G_alone": [float(g.G_alone_0.mean()), float(g.G_alone_r.mean())],
                   "static": float(g.static.mean()), "static_G": [float(g.static_G_0.mean()), float(g.static_G_r.mean())]}
            for e in ("E_M", "E_G", "E_MG", "I"):
                rec[e] = boot(g[e].values, sp, rng)
                rec[e + "_cluster"] = boot(g[e].values, sp, rng, True) if cond == "same" else None
            out["table"].append(rec)
    A.dumpj(A.RESULTS / "c1_summary.json", out)
    f = lambda d: f"{d['mean']:+6.2f} [{d['lo']:+5.2f},{d['hi']:+5.2f}]"
    print(f"pairs: {out['n_pairs']}; AUROC of the cells (memory sees, graph sees): none / memory / graph / both")
    for rec in out["table"]:
        c = rec["cells"]
        print(f"{rec['cond']:4s} r={rec['r']:2d} | {c['c00']:6.2f} {c['cr0']:6.2f} {c['c0r']:6.2f} {c['crr']:6.2f} | E_M {f(rec['E_M'])} E_G {f(rec['E_G'])} "
              f"E_MG {f(rec['E_MG'])} I {f(rec['I'])} | p_M alone {rec['M_alone'][0]:.2f}->{rec['M_alone'][1]:.2f} p_LP alone {rec['G_alone'][0]:.2f}->{rec['G_alone'][1]:.2f} "
              f"static {rec['static']:.2f} static x p_LP {rec['static_G'][0]:.2f}->{rec['static_G'][1]:.2f}")
    print("split-clustered intervals (same):")
    for rec in out["table"]:
        if rec["cond"] == "same":
            print(f"  r={rec['r']:2d} E_M {f(rec['E_M_cluster'])} E_G {f(rec['E_G_cluster'])} E_MG {f(rec['E_MG_cluster'])} I {f(rec['I_cluster'])}")


if __name__ == "__main__":
    main()
