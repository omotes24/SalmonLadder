"""Derived descriptive numbers of Phase 17 that the registered summary does not aggregate (no selection, no new metric):
the seeds of sigma_ref on the recurrence grid (per cell and view: number of seeds, of which calibration images, ID fraction;
mean over the 3 repetitions) from the per-task CSVs experiments/phase17/results/p17_rec/<ds>/c<c>_r<r>_rep<k>.csv.
Writes data/p17_rec_seeds.json."""
import glob
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
SRC = Path(sys.argv[1]) if len(sys.argv) > 1 else (HERE.parent.parent / "experiments" / "phase17" / "results" if (HERE.parent.parent / "experiments" / "phase17" / "results").exists() else Path("/home/claude/SalmonLadder/experiments/phase17/results"))
rows = []
for f in sorted(glob.glob(str(SRC / "p17_rec" / "*" / "*.csv"))):
    d = pd.read_csv(f)
    x = d[(d.cfg == "ref|storey|q0.2|pp") & (d.base == "none")]
    assert len(x) == 1, f
    x = x.iloc[0]
    for v in ("L14", "D3L"):
        rows.append({"ds": x.ds, "c": int(x.c), "r": int(x.r), "rep": int(x.rep), "mixing_rate": float(x.mixing_rate), "n_ood": int(x.n_ood),
                     "view": v, "seeds": int(x[f"seeds_ref_{v}"]), "cal_seeds": int(x[f"cal_seeds_ref_{v}"]),
                     "idfrac": (None if pd.isna(x[f"seed_idfrac_ref_{v}"]) else float(x[f"seed_idfrac_ref_{v}"]))})
t = pd.DataFrame(rows)
out = {"cells": {}}
for (ds, c, r, v), g in t.groupby(["ds", "c", "r", "view"]):
    idf = g.idfrac.dropna()
    out["cells"][f"{ds}|c{c}|r{r}|{v}"] = {"mixing_rate": float(g.mixing_rate.mean()), "seeds": float(g.seeds.mean()), "cal_seeds": float(g.cal_seeds.mean()),
                                           "idfrac": (float(idf.mean()) if len(idf) else None), "reps_with_seeds": int((g.seeds > 0).sum()), "n_reps": int(len(g))}
# buckets of the mixing rate (cell-views; a cell-view has seeds if any repetition selected one)
buckets = {"lt1": (0, 0.01), "1to10": (0.01, 0.10), "ge10": (0.10, 1.01)}
for name, (lo, hi) in buckets.items():
    cv = [e for e in out["cells"].values() if lo <= e["mixing_rate"] < hi]
    with_seeds = [e for e in cv if e["idfrac"] is not None]
    out[name] = {"cells_views": len(cv), "with_seeds": len(with_seeds),
                 "idfrac_mean": (float(np.mean([e["idfrac"] for e in with_seeds])) if with_seeds else None),
                 "idfrac_min": (float(np.min([e["idfrac"] for e in with_seeds])) if with_seeds else None),
                 "idfrac_max": (float(np.max([e["idfrac"] for e in with_seeds])) if with_seeds else None),
                 "seeds_median": (float(np.median([e["seeds"] for e in with_seeds])) if with_seeds else None),
                 "cal_seeds_share": (float(np.sum([e["cal_seeds"] for e in with_seeds]) / max(np.sum([e["seeds"] for e in with_seeds]), 1)) if with_seeds else None)}
    # the false discovery proportion with FDP = 0 when no seed is selected (the quantity whose expectation Proposition 3 bounds), mean over
    # cell x view x repetition
    tt = t[(t.mixing_rate >= lo) & (t.mixing_rate < hi)]
    out[name]["fdp_uncond_mean"] = float(np.mean([(x if x == x and s > 0 else 0.0) for x, s in zip(tt.idfrac.fillna(0.0), tt.seeds)]))
    out[name]["share_reps_with_seeds"] = float(np.mean(tt.seeds > 0))
(HERE / "data" / "p17_rec_seeds.json").write_text(json.dumps(out, indent=1) + "\n")
print(json.dumps({k: v for k, v in out.items() if k != "cells"}, indent=1))


# ---------------------------------------------------------------- paired comparisons that the registered summary does not form
# (sigma_ref against p+ only read with the reference images, against p+ only cross-fitted, against the read-out without folds or
# references, and the decomposition rows against sigma_ref), per draw: mean over the base orders, then over the data sets of a group
from scipy import stats  # noqa: E402


def tci(x):
    x = np.asarray(x, np.float64)
    n = len(x)
    m = float(x.mean())
    h = float(stats.t.ppf(0.975, n - 1) * x.std(ddof=1) / np.sqrt(n)) if n >= 2 else float("nan")
    return {"mean": m, "lo": m - h, "hi": m + h, "n": n}


GROUPS = {"test_openood": {"near": ["ssb_hard", "ninco"], "far": ["inaturalist", "textures", "openimageo"]},
          "test_fourood": {"fourood": ["inat", "sun", "places", "dtd"]},
          "in200": {"near": ["ssb_hard", "ninco"], "far": ["inaturalist", "textures", "openimageo"]},
          "cifar": {"near": ["cifar10", "tin"], "far": ["mnist", "svhn", "texture", "places365"]}}
PATTERNS = {"test_openood": "p17_test/openood/*/*.csv", "test_fourood": "p17_test/fourood/*/*.csv", "in200": "p17_in200/*/*.csv", "cifar": "p17_cifar/*/*.csv"}
REF = "ref|storey|q0.2|pp|stat"
OTHERS = ["ref|lp|stat", "xfit|lp|pooled", "v1|storey|q0.2|pp", "xfit|storey|q0.2|pp|pooled", "ref|storey|q0.2|pp", "dec|deg", "dec|q", "dec|u", "dec|sigma_q", "knn"]
pairs = {}
for sec, pat in PATTERNS.items():
    d = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(str(SRC / pat)))], ignore_index=True)
    if "views" in d:
        d = d[d.views == "L14+D3L"] if (d.views == "L14+D3L").any() else d
    d["draw"] = d["draw"].astype(str)
    m = d.groupby(["draw", "ds", "cfg", "base"], as_index=False)[["AUROC", "FPR95"]].mean()
    out_sec = {}
    for g, sets in GROUPS[sec].items():
        x = m[m.ds.isin(sets)].groupby(["cfg", "base", "draw"])[["AUROC", "FPR95"]].mean()
        for base in sorted(m.base.unique()):
            if (REF, base) not in x.index.droplevel(2):
                continue
            a = x.loc[(REF, base)]
            for o in OTHERS:
                if (o, base) not in x.index.droplevel(2):
                    continue
                b = x.loc[(o, base)]
                common = a.index.intersection(b.index)
                if len(common) < 2:
                    continue
                out_sec[f"{g}|{o}|{base}"] = {"dFPR95_ref_minus": tci((a.loc[common].FPR95 - b.loc[common].FPR95).values),
                                              "dAUROC_ref_minus": tci((a.loc[common].AUROC - b.loc[common].AUROC).values),
                                              "FPR95": float(b.loc[common].FPR95.mean()), "AUROC": float(b.loc[common].AUROC.mean()),
                                              "ref_FPR95": float(a.loc[common].FPR95.mean()), "ref_AUROC": float(a.loc[common].AUROC.mean())}
    pairs[sec] = out_sec
(HERE / "data" / "p17_pairs.json").write_text(json.dumps(pairs, indent=1) + "\n")
for sec in pairs:
    for k in ("near|ref|lp|stat|none", "near|xfit|lp|pooled|none", "near|ref|lp|stat|TINS", "far|ref|lp|stat|none", "fourood|ref|lp|stat|none", "near|xfit|storey|q0.2|pp|pooled|none"):
        if k in pairs[sec]:
            e = pairs[sec][k]["dFPR95_ref_minus"]
            print(sec, k, f"{e['mean']:+.2f} [{e['lo']:+.2f}, {e['hi']:+.2f}]", round(pairs[sec][k]["FPR95"], 2), round(pairs[sec][k]["ref_FPR95"], 2))


# ---------------------------------------------------------------- seeds of sigma_ref on the benchmarks (registered per-task columns): per data set and
# view, mean over the six draws of the number of seeds, the calibration seeds and the ID fraction of the seeds
bench = {}
for part in ("openood", "fourood"):
    d = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(str(SRC / "p17_test" / part / "*" / "*.csv")))], ignore_index=True)
    x = d[(d.cfg == "ref|storey|q0.2|pp") & (d.base == "none")]
    for ds, g in x.groupby("ds"):
        for v in ("L14", "D3L"):
            bench[f"{part}|{ds}|{v}"] = {"seeds": float(g[f"seeds_ref_{v}"].mean()), "cal_seeds": float(g[f"cal_seeds_ref_{v}"].mean()),
                                         "idfrac": float(g[f"seed_idfrac_ref_{v}"].mean()), "pool": float(g.n.mean() + 2000), "n_draws": int(g.draw.nunique())}
sd = json.loads((HERE / "data" / "p17_rec_seeds.json").read_text())
sd["bench"] = bench
(HERE / "data" / "p17_rec_seeds.json").write_text(json.dumps(sd, indent=1) + "\n")
print("bench seeds: idfrac", min(e["idfrac"] for e in bench.values()), max(e["idfrac"] for e in bench.values()))
