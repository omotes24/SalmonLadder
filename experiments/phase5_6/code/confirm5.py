"""Phase 5: evaluation of the locked method against frozen REPRISE v5 on a development split (dev2 = confirmation).

Reads results/lock/<dev>_*.npz (dev5.py --cfgs lock). Units: support draws (orders averaged), paired t interval, df 4.
Checks the pre-registered definition of a large improvement and writes results/confirm_<dev>.json."""
import argparse
import json
from pathlib import Path

import numpy as np

from analyze5 import FROZEN, REFT, OUT, diff, draw_means, load, summarize, tci
from metrics_p4 import metrics

LOCK = Path("/home/omote/reprise_p5_20261002/selection_lock_p5.json")


def ev(tasks, combo, views, base, a=1.0):
    out = {}
    for t, z in tasks.items():
        x = 0.0
        for v in views:
            for key, w in combo:
                x = x + w * z[f"s::{v}::{key}"]
        x = a * x
        if base == "TINS":
            x = x + z["logS"]
        elif base == "MCM":
            x = x + z["logM"]
        out[t] = metrics(x, z["is_ood"])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dev", default="dev2")
    a = ap.parse_args()
    L = json.loads(LOCK.read_text())
    views = [v[0] for v in L["views"]]
    fviews = ["B14", "L14"]
    combo = [tuple(c) for c in L["score"]]
    zeta = [(f"z|{REFT}", 1.0)]
    res = {"dev": a.dev, "lock_name": L["name"], "n_tasks": {}}
    for stream in ("near", "far"):
        T = load("lock", a.dev, stream)
        res["n_tasks"][stream] = len(T)
        for base in ("none", "TINS", "MCM"):
            F = ev(T, FROZEN, fviews, base)
            N = ev(T, combo, views, base)
            Z = ev(T, zeta, fviews, base)
            key = f"{stream}_{base}"
            res[key] = {"frozen": summarize(F), "locked": summarize(N), "zeta": summarize(Z),
                        "dFPR95": diff(N, F, "FPR95"), "dAUROC": diff(N, F, "AUROC"),
                        "dFPR95_vs_zeta": diff(N, Z, "FPR95"),
                        "per_draw_dFPR95": (draw_means(N, "FPR95") - draw_means(F, "FPR95")).tolist()}
    n, t, f = res["near_none"]["dFPR95"], res["near_TINS"]["dFPR95"], res["far_none"]["dFPR95"]
    res["criteria"] = {"standalone_near_dFPR95_le_-5.0": n["mean"] <= -5.0, "upper_CI_lt_-3.0": n["hi"] < -3.0,
                       "xTINS_near_dFPR95_le_-4.0": t["mean"] <= -4.0, "standalone_far_dFPR95_le_+0.5": f["mean"] <= 0.5}
    res["large_improvement"] = bool(all(res["criteria"].values()))
    out = OUT / f"confirm_{a.dev}.json"
    out.write_text(json.dumps(res, indent=1, default=float) + "\n")
    for key in ("near_none", "near_TINS", "near_MCM", "far_none", "far_TINS"):
        r = res[key]
        print(f"{key:10s} frozen {r['frozen']['AUROC']:6.2f}/{r['frozen']['FPR95']:6.2f}  locked {r['locked']['AUROC']:6.2f}/{r['locked']['FPR95']:6.2f}"
              f"  dFPR95 {r['dFPR95']['mean']:+6.2f} [{r['dFPR95']['lo']:+6.2f},{r['dFPR95']['hi']:+6.2f}]  dAUROC {r['dAUROC']['mean']:+5.2f}"
              f"  vs zeta {r['dFPR95_vs_zeta']['mean']:+6.2f}  per draw {[round(x, 2) for x in r['per_draw_dFPR95']]}")
    print(json.dumps({"criteria": res["criteria"], "large_improvement": res["large_improvement"], "tasks": res["n_tasks"]}))


if __name__ == "__main__":
    main()
