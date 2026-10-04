"""Recurrence analysis, part 2 (post-hoc, frozen REPRISE, OpenOOD test, 5 orders).

Same bins as reprise_recurrence.py (earlier same-class OOD images in earlier batches: 0, 1-4, 5-19, >=20) for SSB-hard
and NINCO. Adds, per bin: the static variant (S * p_B * p_L), memory-only and LP-only variants, and the mean relative
stream position of the OOD images in the bin (batch_index / last batch index), to show the position confound.
AUROC is always ID (all ID images of the stream) vs the OOD images of the bin.
Output: analysis/reprise/recurrence2.json (also printed).
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from vins import config as C  # noqa: E402
from vins.stats import auroc  # noqa: E402

A = C.WORK / "analysis" / "reprise"
BINS = [(0, 0), (1, 4), (5, 19), (20, 10 ** 9)]


def main():
    manifest = {}
    with open(C.SEAL_MANIFEST) as handle:
        for line in handle:
            row = json.loads(line)
            manifest[row["sample_id"]] = row

    def cls_of(sid):
        parts = manifest[sid]["relative_path"].split("/")
        return parts[1] if len(parts) > 2 and not parts[1] == "images" else sid

    out = {}
    for ds in ("ssb_hard", "ninco"):
        names = ("tins", "static", "memory_only", "lp_only", "reprise")
        acc = {f"{lo}-{hi}": {k: [] for k in names + ("n", "pos")} for lo, hi in BINS}
        for seed in (123, 124, 125, 126, 127):
            z = np.load(A / "openood" / f"{ds}_seed{seed}.npz", allow_pickle=True)
            ids, is_ood, bidx = z["sample_id"], z["is_ood"].astype(bool), z["batch_index"]
            g = lambda k: z[k].astype(np.float64)  # noqa: E731
            S = g("S")
            sc = {"tins": S, "static": S * g("p_B14") * g("p_L14"), "memory_only": S * g("pt3_B14") * g("pt3_L14"),
                  "lp_only": S * g("plp_B14") * g("plp_L14"),
                  "reprise": S * g("pt3_B14") * g("pt3_L14") * g("plp_B14") * g("plp_L14")}
            prior = np.zeros(len(ids), dtype=int)
            seen = {}
            for b in np.unique(bidx):
                rows = np.flatnonzero(bidx == b)
                cls = [cls_of(ids[i]) if is_ood[i] else None for i in rows]
                for i, c in zip(rows, cls):
                    if c is not None:
                        prior[i] = seen.get(c, 0)
                for c in cls:
                    if c is not None:
                        seen[c] = seen.get(c, 0) + 1
            id_idx = np.flatnonzero(~is_ood)
            rel = bidx / max(1, bidx.max())
            for lo, hi in BINS:
                sel = np.flatnonzero(is_ood & (prior >= lo) & (prior <= hi))
                key = f"{lo}-{hi}"
                acc[key]["n"].append(int(len(sel)))
                acc[key]["pos"].append(float(rel[sel].mean()) if len(sel) else float("nan"))
                for k in names:
                    acc[key][k].append(100 * auroc(sc[k][id_idx], sc[k][sel]) if len(sel) else float("nan"))
        out[ds] = {key: {k: float(np.nanmean(v)) for k, v in d.items()} for key, d in acc.items()}
        print(ds, json.dumps({key: {k: round(v, 2) for k, v in d.items()} for key, d in out[ds].items()}), flush=True)
    (A / "recurrence2.json").write_text(json.dumps(out, indent=1) + "\n")


if __name__ == "__main__":
    main()
