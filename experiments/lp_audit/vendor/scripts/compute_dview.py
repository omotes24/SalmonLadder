"""Step 2: d(x) for CLIP / DINOv2 features and m in {1, 2}; split-conformal thresholds from calib.

d(x) is computed once and never updated by any stream.
Outputs: runs/dview/dview.parquet, runs/dview/thresholds.json
"""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from vins import config as C  # noqa: E402
from vins.dview import conformal_rank, conformal_threshold, d_scores, loo_stats, nominate  # noqa: E402
from vins.features import load_features  # noqa: E402


def eps_tag(eps):
    return f"{eps * 100:g}".replace(".", "p")          # 0.005 -> "0p5", 0.01 -> "1", 0.02 -> "2"


def nom_col(feat, m, eps):
    return f"nom_{feat}_m{m}_eps{eps_tag(eps)}"


def main():
    start = time.time()
    out = C.RUNS_DIR / "dview"
    out.mkdir(parents=True, exist_ok=True)
    samples = pd.read_parquet(C.SPLITS_DIR / "samples.parquet")
    zeroshot = pd.read_parquet(C.FEATURES_DIR / "zeroshot.parquet")
    assert (zeroshot.sample_id.values == samples.sample_id.values).all()
    cand_all = np.array(zeroshot.K_id.tolist(), dtype=np.int64)
    n_id = 1000 - C.N_HELDOUT

    sup_idx = np.flatnonzero((samples.split == "support").values)
    sup_idx = sup_idx[np.argsort(samples.class_idx_id.values[sup_idx], kind="stable")]
    assert (samples.class_idx_id.values[sup_idx] == np.repeat(np.arange(n_id), C.N_SUPPORT)).all()
    q_idx = np.flatnonzero(samples.split.isin(["calib", "id_dev", "near_dev", "far_dev"]).values)

    frame = samples.iloc[q_idx][["sample_id", "split", "group", "wnid", "class_idx_id", "smoke"]].reset_index(drop=True)
    cand = cand_all[q_idx]
    is_id = (frame.group == "ID").values
    frame["zs_top1_id"] = cand[:, 0]
    frame["K_id"] = [list(map(int, r)) for r in cand]
    frame["zs_top1_correct"] = is_id & (cand[:, 0] == frame.class_idx_id.values)
    frame["true_in_K"] = is_id & (cand == frame.class_idx_id.values[:, None]).any(axis=1)
    is_cal = (frame.split == "calib").values

    thresholds, diagnostics = {}, {}
    for feat in C.FEATURES:
        feats, blob = load_features(C.FEATURES_DIR / f"{feat}.pt")
        assert blob["sample_id"] == samples.sample_id.tolist()
        feats = feats.numpy().astype(np.float64)
        support = feats[sup_idx].reshape(n_id, C.N_SUPPORT, -1)
        queries = feats[q_idx]
        for m in C.M_LIST:
            key = f"{feat}_m{m}"
            stats = loo_stats(support, m, C.N0, C.MAD_SCALE)
            d, _, _, cstar = d_scores(queries, cand, support, stats, m)
            frame[f"d_{key}"] = d
            frame[f"cstar_{key}"] = cstar
            thresholds[key] = {}
            for eps in C.EPS_LIST:
                q = conformal_threshold(d[is_cal], eps)
                thresholds[key][str(eps)] = {"q": q, "rank": conformal_rank(int(is_cal.sum()), eps),
                                             "n_cal": int(is_cal.sum())}
                frame[nom_col(feat, m, eps)] = nominate(d, q)
            # LOO (11 neighbours) vs query (12 neighbours) bias: z of the true class for ID queries
            true_cls = frame.class_idx_id.values[is_id][:, None]
            z_true, _, r_true, _ = d_scores(queries[is_id], true_cls, support, stats, m)
            cal_in_id = is_cal[is_id]
            diagnostics[key] = {
                "med_all": stats["med_all"], "mad_all": stats["mad_all"],
                "median_med_c": float(np.median(stats["med_c"])), "median_mad_c": float(np.median(stats["mad_c"])),
                "n_classes_mad_c_zero": int((stats["mad_c"] == 0).sum()), "min_mad_tilde": float(stats["mad_t"].min()),
                "median_loo": float(np.median(stats["loo"])),
                "median_r_true_calib": float(np.median(r_true[cal_in_id])),
                "median_r_true_iddev": float(np.median(r_true[~cal_in_id])),
                "median_z_true_calib": float(np.median(z_true[cal_in_id])),
                "median_z_true_iddev": float(np.median(z_true[~cal_in_id])),
            }
    frame.to_parquet(out / "dview.parquet", index=False)
    (out / "thresholds.json").write_text(json.dumps({"thresholds": thresholds, "diagnostics": diagnostics,
                                                      "seconds": round(time.time() - start, 1)}, indent=1) + "\n")
    print(json.dumps({"thresholds": thresholds, "seconds": round(time.time() - start, 1)}, indent=1))


if __name__ == "__main__":
    main()
