"""Summary of the Corollary-1 check (cor15.py): per (data set, r, k_g, view) medians over the unknown classes and the global
validity statistics -> P15/results/cor/p15_cor_summary.json (also usable from a local copy: pass the results/cor directory)."""
import glob
import json
import sys
from pathlib import Path

import pandas as pd

root = Path(sys.argv[1]) if len(sys.argv) > 1 else None
if root is None:
    import p15common as P
    root = P.RESULTS15 / "cor"
df = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(str(root / "*" / "*.csv")))], ignore_index=True)
cols = {"ebar_med": ("ebar", "median"), "e_med": ("e_med", "median"), "pi_med": ("pi_med", "median"), "pi_min_med": ("pi_min", "median"),
        "n_boundary_med": ("n_boundary", "median"), "self_inside_med": ("self_inside_frac", "median"),
        "p_act_med": ("p_act_med", "median"), "p_bound_psi_med": ("p_bound_med", "median"), "p_bound_pi_med": ("p_bound_pi_med", "median"),
        "frac_act_le005": ("frac_act_le005", "mean"), "frac_bound_psi_le005": ("frac_bound_le005", "mean"), "frac_bound_pi_le005": ("frac_bound_pi_le005", "mean"),
        "frac_act_le02": ("frac_act_le02", "mean"), "frac_bound_psi_le02": ("frac_bound_le02", "mean"), "frac_bound_pi_le02": ("frac_bound_pi_le02", "mean"),
        "frac_act_floor": ("frac_act_floor", "mean"), "frac_bound_pi_floor": ("frac_bound_pi_floor", "mean"),
        "max_ratio_pi": ("max_ratio_pi", "max"), "max_ratio_psi": ("max_ratio_psi", "max"), "n_clusters": ("cls", "count")}
out = {"cells": {}, "views": sorted(df.view.unique()), "n_rows": int(len(df)),
       "max_ratio_pi": float(df.max_ratio_pi.max()), "max_ratio_psi": float(df.max_ratio_psi.max()),
       "all_valid_psi": bool(df.bound_valid.all()), "all_valid_pi": bool(df.bound_pi_valid.all()), "all_pi_le_psi": bool(df.pi_le_psi.all()),
       "n_ratio_pi_above_1": int((df.max_ratio_pi > 1).sum()), "max_excess_pi": float(max(df.max_ratio_pi.max() - 1.0, 0.0))}
for (ds, r, kg, v), g in df.groupby(["ds", "r", "kg", "view"]):
    key = f"{ds}|r{r}|kg{kg}|{v}"
    out["cells"][key] = {k: float(getattr(g[c], f)()) for k, (c, f) in cols.items()}
    out["cells"][key]["n_clusters"] = int(len(g))
    out["cells"][key]["c"] = int(g.c.iloc[0])
(root / "p15_cor_summary.json").write_text(json.dumps(out, indent=1) + "\n")
print(json.dumps({k: v for k, v in out.items() if k != "cells"}))
print("ANA_COR15_DONE")
