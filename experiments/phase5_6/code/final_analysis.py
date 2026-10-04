"""Phase 5 final analysis (pre-registered). Paired differences of the locked method to frozen REPRISE v5.
  u4:      differences averaged within split (9 draw x order conditions) -> 5 split values -> t interval (df 4)
  openood: near = mean(SSB-hard, NINCO), far = mean(iNaturalist, Textures, OpenImage-O) per order -> 5 orders (df 4)
  fourood: mean of the four sets per order -> 3 orders (df 2)"""
import argparse
import glob
import json
import re
from pathlib import Path

import numpy as np

from analyze5 import FROZEN, REFT, Z
from metrics_p4 import metrics, tci
from u4_common import P5, require_lock

OUT = P5 / "results_final"
OO_NEAR, OO_FAR = ["ssb_hard", "ninco"], ["inaturalist", "textures", "openimageo"]
FOUR = ["inat", "sun", "places", "dtd"]


_UP = {}


def upstream(x, f):
    """Upstream TINS get_measures (as Phase 3). The upstream package is put on the path once (import_tins)."""
    if "ok" not in _UP:
        from vins.tins_dev import import_tins
        import_tins()
        _UP["ok"] = True
    from vins.metrics import measures
    m = measures(x[~f], x[f])
    return {"AUROC": 100 * m["AUROC"], "FPR95": 100 * m["FPR95"]}


def score(z, combo, views, base):
    x = 0.0
    for v in views:
        for key, w in combo:
            x = x + w * z[f"s::{v}::{key}"]
    if base == "TINS":
        x = x + z["logS"]
    elif base == "MCM":
        x = x + z["logM"]
    return x


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", required=True)
    a = ap.parse_args()
    L = require_lock()
    lviews, combo = [v[0] for v in L["views"]], [tuple(c) for c in L["score"]]
    methods = {"frozen_v5": (FROZEN, ["B14", "L14"]), "locked": (combo, lviews), "zeta": ([(f"z|{REFT}", 1.0)], ["B14", "L14"]),
               "M0xlp_zero": ([("M0", 1.0), (f"lp|{REFT}", 1.0)], ["B14", "L14"])}
    files = sorted(glob.glob(str(OUT / a.part / "*.npz")))
    met = metrics if a.part == "u4" else upstream
    rows = []
    for f in files:
        z = Z(f)
        name = Path(f).stem
        for base in ("none", "TINS", "MCM"):
            for m, (c, vs) in methods.items():
                r = met(score(z, c, vs, base), z["is_ood"])
                rows.append({"stream": name, "base": base, "method": m, **r})
            if base != "none":
                r = met(z["logS"] if base == "TINS" else z["logM"], z["is_ood"])
                rows.append({"stream": name, "base": base, "method": "base_only", **r})
    import pandas as pd
    df = pd.DataFrame(rows)
    res = {"part": a.part, "n_streams": len(files), "lock": L["name"]}
    if a.part == "u4":
        df["split"] = df.stream.str.extract(r"_s(\d)_", expand=False).astype(int)
        df["group"] = "near"
        unit = "split"
    else:
        df["ds"] = df.stream.str.replace(r"_seed\d+$", "", regex=True)
        df["seed"] = df.stream.str.extract(r"seed(\d+)$", expand=False).astype(int)
        if a.part == "openood":
            df["group"] = np.where(df.ds.isin(OO_NEAR), "near", "far")
        else:
            df["group"] = "fourood"
        unit = "seed"
    def unit_means(d):
        if a.part == "u4":
            return d.groupby("split")[["AUROC", "FPR95"]].mean()
        return d.groupby(["seed", "ds"])[["AUROC", "FPR95"]].mean().groupby("seed").mean()
    for grp in sorted(df.group.unique()):
        for base in ("none", "TINS", "MCM"):
            d = df[(df.group == grp) & (df.base == base)]
            um = {m: unit_means(d[d.method == m]) for m in d.method.unique()}
            key = f"{grp}_{base}"
            res[key] = {m: {k: float(v[k].mean()) for k in ("AUROC", "FPR95")} for m, v in um.items()}
            for ref in ("frozen_v5", "zeta"):
                for k in ("FPR95", "AUROC"):
                    res[key][f"locked_minus_{ref}_{k}"] = tci((um["locked"][k] - um[ref][k]).values)
            print(f"{key:14s} " + "  ".join(f"{m} {v['AUROC']:.2f}/{v['FPR95']:.2f}" for m, v in res[key].items() if isinstance(v, dict) and "AUROC" in v))
            dd = res[key]["locked_minus_frozen_v5_FPR95"]
            da = res[key]["locked_minus_frozen_v5_AUROC"]
            print(f"{'':14s} locked - frozen: FPR95 {dd['mean']:+.2f} [{dd['lo']:+.2f}, {dd['hi']:+.2f}]  AUROC {da['mean']:+.2f} [{da['lo']:+.2f}, {da['hi']:+.2f}] (n={dd['n']})")
    if a.part != "u4":
        per = df[df.method.isin(["frozen_v5", "locked", "base_only"])].groupby(["ds", "base", "method"])[["AUROC", "FPR95"]].mean().round(2)
        print(per.to_string())
        res["per_dataset"] = {f"{i[0]}|{i[1]}|{i[2]}": r.to_dict() for i, r in per.iterrows()}
    if a.part == "u4":
        f1 = res["near_none"]["locked_minus_frozen_v5_FPR95"]
        res["confirmed_large_improvement"] = bool(f1["mean"] <= -4.0 and f1["hi"] < 0)
        print(json.dumps({"F1": f1, "F2": res["near_TINS"]["locked_minus_frozen_v5_FPR95"], "confirmed": res["confirmed_large_improvement"]}))
    (OUT / f"summary_{a.part}.json").write_text(json.dumps(res, indent=1, default=float) + "\n")


if __name__ == "__main__":
    main()
