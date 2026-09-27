"""Iteration 2 (dev only): fuse per-sample p-values saved by iter2_eval.py --save-scores from two visual views.

Usage: python scripts/iter2_fuse.py --a eB --b eG --tag default --pairs pall10,pall10 self10,pall10 ... --name fuse1
For each pair (rule of view A, rule of view B): S*ptA, S*ptB, S*ptA*ptB, pT*ptA*ptB.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from vins import config as C  # noqa: E402
from vins.metrics import aggregate, measures  # noqa: E402
from vins.tins_dev import import_tins  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--a", required=True)
    parser.add_argument("--b", required=True)
    parser.add_argument("--tag", default="default")
    parser.add_argument("--pairs", nargs="+", required=True)
    parser.add_argument("--name", required=True)
    opts = parser.parse_args()
    import_tins()
    per_seed = []
    for seed in C.ORDER_SEEDS:
        entry = {}
        for stream in ("near", "far"):
            A = np.load(C.WORK / "iter2" / "scores" / opts.a / f"{opts.tag}_{stream}_seed{seed}.npz", allow_pickle=True)
            B = np.load(C.WORK / "iter2" / "scores" / opts.b / f"{opts.tag}_{stream}_seed{seed}.npz", allow_pickle=True)
            assert (A["sample_id"] == B["sample_id"]).all()
            is_id, S, pT = A["is_id"], A["S"].astype(np.float64), A["pT"].astype(np.float64)
            res = {}
            for pair in opts.pairs:
                ra, rb = pair.split(",")
                pa, pb = A[f"pt__{ra}"].astype(np.float64), B[f"pt__{rb}"].astype(np.float64)
                for key, sc in {"S*ptA": S * pa, "S*ptB": S * pb, "S*ptA*ptB": S * pa * pb,
                                "pT*ptA*ptB": pT * pa * pb, "ptA*ptB": pa * pb}.items():
                    res[f"{pair}|{key}"] = measures(sc[is_id], sc[~is_id])
            entry[stream] = res
        per_seed.append(entry)
    agg = aggregate(per_seed)
    print(f"== fuse A={opts.a} B={opts.b} tag={opts.tag}  (AUROC near / far, FPR95 near / far)")
    for k in sorted(agg["near"]):
        a_n, a_f = agg["near"][k]["AUROC"]["mean"] * 100, agg["far"][k]["AUROC"]["mean"] * 100
        f_n, f_f = agg["near"][k]["FPR95"]["mean"] * 100, agg["far"][k]["FPR95"]["mean"] * 100
        print(f"  {k:34s} {a_n:6.2f} {a_f:6.2f} | {f_n:6.2f} {f_f:6.2f}")
    out = C.WORK / "iter2" / "eval"
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{opts.name}.json").write_text(json.dumps({"aggregate": agg}) + "\n")


if __name__ == "__main__":
    main()
