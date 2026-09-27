"""AUROC / FPR95 of TINS with OpenOOD-style preprocessing (seed 123) vs the default run."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from vins import config as C  # noqa: E402
from vins.tins_dev import import_tins  # noqa: E402

OO = ["ssb_hard", "ninco", "inaturalist", "textures", "openimageo"]


def main():
    t = import_tins()
    out = {}
    for tag in ("preproc_oodpre", "default"):
        out[tag] = {}
        for ds in OO:
            run = np.load(C.WORK / "test_runs" / tag / f"{ds}_seed123.npz")
            s, is_id = run["S_final"].astype(np.float64), ~run["is_ood"].astype(bool)
            a, _, f = t.get_measures(s[is_id], s[~is_id])
            out[tag][ds] = {"AUROC": 100 * float(a), "FPR95": 100 * float(f)}
    (C.WORK / "analysis" / "test" / "preproc.json").write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps({k: {d: round(v[d]["AUROC"], 2) for d in OO} for k, v in out.items()}))


if __name__ == "__main__":
    main()
