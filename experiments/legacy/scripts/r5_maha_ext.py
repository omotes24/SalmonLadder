"""R5 (1) follow-up: Mahalanobis++ shrinkage grid extended beyond 0.5 (the dev1 choice sat on the grid edge).

Same features, shots, calibration and combination as r5_eval.py: per view (DINOv2 B/14, L/14) Mahalanobis++ of the
12 support shots per class (vins.r5.maha_pp), static conformal p against the 4 calibration shots per class, product
over the two views; alone (v_) and x TINS S_final (s_). lam in LAMS_EXT (0.5 repeated as a check against r5_eval).
Output: <R5>/<dev>/eval_ext/draw<k>/<stream>_seed<s>.json
"""
import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "4")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from vins import config as C  # noqa: E402
from vins import r5  # noqa: E402
from vins.features import load_features  # noqa: E402

LAMS_EXT = (0.3, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--draws", nargs="+", default=["0", "1", "2", "3", "4"])
    opts = parser.parse_args()
    from vins.tins_dev import import_tins
    from vins.metrics import measures as upstream

    import_tins()
    spec = importlib.util.spec_from_file_location("r5_eval", ROOT / "scripts" / "r5_eval.py")
    E = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(E)
    dev = E.dev_name()
    samples = pd.read_parquet(C.SPLITS_DIR / "samples.parquet")
    id_classes = json.loads((C.SPLITS_DIR / "id_classes.json").read_text())
    n_id = len(id_classes)
    stream_ids = samples.sample_id.values[samples.split.isin(["id_dev", "near_dev", "far_dev"]).values]
    row = {s: i for i, s in enumerate(stream_ids)}
    for draw in opts.draws:
        sup_ids, cal_ids, src = E.shot_table(draw, id_classes, samples)
        pv = {}
        for name, dev_file, shot_file in E.VIEWS:
            dev_f, blob = load_features(C.FEATURES_DIR / dev_file)
            dpos = {s: i for i, s in enumerate(blob["sample_id"])}
            if src == "features":
                sup_f, cal_f = dev_f[[dpos[s] for s in sup_ids]], dev_f[[dpos[s] for s in cal_ids]]
            else:
                sh_f, sblob = load_features(r5.R5 / "features" / f"shots.{shot_file}.pt")
                spos = {s: i for i, s in enumerate(sblob["sample_id"])}
                sup_f, cal_f = sh_f[[spos[s] for s in sup_ids]], sh_f[[spos[s] for s in cal_ids]]
            st_f = dev_f[[dpos[s] for s in stream_ids]]
            support = sup_f.numpy().astype(np.float32).reshape(n_id, 12, -1)
            q = np.concatenate([cal_f.numpy(), st_f.numpy()]).astype(np.float32)
            md = r5.maha_pp(support, q, LAMS_EXT)
            nc = len(cal_ids)
            pv[name] = {lam: r5.pval_high(md[lam][:nc], md[lam][nc:]) for lam in LAMS_EXT}
        out = r5.R5 / dev / "eval_ext" / f"draw{draw}"
        out.mkdir(parents=True, exist_ok=True)
        for stream in ("near", "far"):
            for seed in C.ORDER_SEEDS:
                z = np.load(r5.R5 / dev / "tins" / f"draw{draw}" / f"{stream}_seed{seed}.npz", allow_pickle=True)
                sid, is_ood, S = z["sample_id"], z["is_ood"].astype(bool), z["S_final"].astype(np.float64)
                qi = np.array([row[s] for s in sid])
                metrics = {}
                for lam in LAMS_EXT:
                    v = pv["B14"][lam][qi] * pv["L14"][lam][qi]
                    vl = pv["L14"][lam][qi]
                    for key, s in ((f"v_maha{lam:g}", v), (f"s_maha{lam:g}", S * v),
                                   (f"vL14_maha{lam:g}", vl), (f"sL14_maha{lam:g}", S * vl)):
                        m = upstream(s[~is_ood], s[is_ood])
                        metrics[key] = {"AUROC": 100 * m["AUROC"], "FPR95": 100 * m["FPR95"]}
                (out / f"{stream}_seed{seed}.json").write_text(json.dumps({"metrics": metrics}) + "\n")
        print(json.dumps({"dev": dev, "draw": draw}), flush=True)


if __name__ == "__main__":
    main()
