"""Small post-test analyses: (a) d over ALL classes (standardised) vs K = top-5, (b) combined streams per dataset."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from vins import config as C  # noqa: E402
from vins.clavism import views_from_arrays  # noqa: E402
from vins.tins_dev import import_tins  # noqa: E402

OO = ["ssb_hard", "ninco", "inaturalist", "textures", "openimageo"]
GROUP = {"ssb_hard": "nearood", "ninco": "nearood", "inaturalist": "farood", "textures": "farood", "openimageo": "farood"}


def main():
    t = import_tins()
    gm = t.get_measures
    blob = torch.load(C.WORK / "test_eval" / "results" / "features.pt", map_location="cpu")
    dino = blob["dino"].numpy().astype(np.float32)
    support, queries = dino[:12000].reshape(1000, 12, -1), dino[12000:]
    is_cal = np.zeros(len(queries), dtype=bool)
    is_cal[:4000] = True
    v = views_from_arrays(support, queries, np.asarray(blob["cand"]), is_cal, m=2)
    ids_of, ood_ids = {}, {}
    for ds in OO:
        z = np.load(C.CODEX_TINS / f"scores_openood_{GROUP[ds]}_{ds}.npz", allow_pickle=True)
        sid, is_ood = z["sample_id"].astype(str), z["is_ood"].astype(bool)
        ids_of[ds] = (sid, is_ood)
        ood_ids[ds] = set(sid[is_ood].tolist())
    test_ids = sorted({s for sid, _ in ids_of.values() for s in sid})
    row = {s: 4000 + i for i, s in enumerate(test_ids)}
    out = {"d_topk": {}, "d_all": {}}
    for ds in OO:
        sid, is_ood = ids_of[ds]
        q = np.array([row[s] for s in sid])
        for key, arr in (("d_topk", v["d"]), ("d_all", v["d_all"])):
            a, _, f = gm(-arr[q][~is_ood], -arr[q][is_ood])
            out[key][ds] = {"AUROC": 100 * float(a), "FPR95": 100 * float(f)}
    for key in ("d_topk", "d_all"):
        out[key]["near"] = float(np.mean([out[key][d]["AUROC"] for d in ("ssb_hard", "ninco")]))
        out[key]["far"] = float(np.mean([out[key][d]["AUROC"] for d in ("inaturalist", "textures", "openimageo")]))
    comb = {}
    for name, members in (("near_all", ["ssb_hard", "ninco"]), ("far_all", ["inaturalist", "textures", "openimageo"])):
        for seed in (123, 124, 125):
            run = np.load(C.WORK / "test_runs" / "default" / f"{name}_seed{seed}.npz")
            sid, s = run["sample_id"], run["S_final"].astype(np.float64)
            is_id = ~run["is_ood"].astype(bool)
            for ds in members:
                sel = np.array([x in ood_ids[ds] for x in sid])
                a, _, f = gm(s[is_id], s[sel])
                comb.setdefault(ds, []).append({"AUROC": 100 * float(a), "FPR95": 100 * float(f)})
    out["combined"] = {ds: {"AUROC_mean": float(np.mean([x["AUROC"] for x in v_])), "per_seed": v_} for ds, v_ in comb.items()}
    dest = C.WORK / "analysis" / "test" / "misc.json"
    dest.write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
