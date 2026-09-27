"""How REPRISE's gain depends on recurrence (post-hoc, OpenOOD test, all 5 orders).

For SSB-hard and NINCO, OOD images are grouped by the number of earlier images of the same OOD class in the stream
(earlier batches only): 0, 1-4, 5-19, >= 20. AUROC is computed for ID vs each group (TINS and TINS * REPRISE).
Output: analysis/reprise/recurrence.json
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
        acc = {f"{lo}-{hi}": {"n": [], "tins": [], "reprise": []} for lo, hi in BINS}
        for seed in (123, 124, 125, 126, 127):
            z = np.load(A / "openood" / f"{ds}_seed{seed}.npz", allow_pickle=True)
            ids, is_ood, bidx = z["sample_id"], z["is_ood"].astype(bool), z["batch_index"]
            S = z["S"].astype(np.float64)
            rep = S * np.prod([z[k].astype(np.float64) for k in ("pt3_B14", "pt3_L14", "plp_B14", "plp_L14")], axis=0)
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
            for lo, hi in BINS:
                sel = np.flatnonzero(is_ood & (prior >= lo) & (prior <= hi))
                key = f"{lo}-{hi}"
                acc[key]["n"].append(int(len(sel)))
                if len(sel):
                    acc[key]["tins"].append(100 * auroc(S[id_idx], S[sel]))
                    acc[key]["reprise"].append(100 * auroc(rep[id_idx], rep[sel]))
        out[ds] = {k: {"n_mean": float(np.mean(v["n"])), "tins": float(np.mean(v["tins"])) if v["tins"] else None,
                       "reprise": float(np.mean(v["reprise"])) if v["reprise"] else None} for k, v in acc.items()}
        print(ds, json.dumps({k: {kk: (round(vv, 2) if isinstance(vv, float) else vv) for kk, vv in v.items()}
                              for k, v in out[ds].items()}))
    (A / "recurrence.json").write_text(json.dumps(out, indent=1) + "\n")


if __name__ == "__main__":
    main()
