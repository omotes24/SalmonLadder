"""Exploratory: does combining TINS with the frozen visual view d(x) improve detection (AUROC / FPR95)?

Uses the main run's TINS logs (3 order seeds) and the d(x) variants from explore_candidates. No TINS re-run.
Fusion rules are parameter-free and label-free at test time (nothing is tuned on dev):
  rank_avg : mean of the within-stream ranks of S_final (ID-ness) and -d (ID-ness)
  p_prod   : S_final * p_d(x), p_d = (1 + #{calib d >= d(x)}) / (n_calib + 1)   (conformal p-value of d)
"""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import rankdata  # noqa: E402

from vins import config as C  # noqa: E402
from vins.metrics import aggregate, measures  # noqa: E402
from vins.tins_dev import import_tins  # noqa: E402

VARIANTS = ["zs5", "vp5", "zs5+vp5"]
KEYS = ["dino_m1", "dino_m2", "clip_m1", "clip_m2"]


def pvalue(cal, d):
    cal = np.sort(cal)
    ge = len(cal) - np.searchsorted(cal, d, side="left")        # #{cal >= d}
    return (1.0 + ge) / (len(cal) + 1.0)


def main():
    start = time.time()
    import_tins()
    rep = C.REPORTS_DIR / "explore_candidates"
    per_seed = []
    for seed in C.ORDER_SEEDS:
        entry = {}
        for stream in ("near", "far"):
            frame = pd.read_parquet(C.RUNS_DIR / "main" / f"stream_{stream}_seed{seed}.parquet")
            is_id = (frame.group == "ID").values
            s = frame.S_final.values.astype(np.float64)
            entry[stream] = {"tins": measures(s[is_id], s[~is_id])}
            for variant in VARIANTS:
                dv = pd.read_parquet(C.RUNS_DIR / "explore_candidates" / f"dview_{variant.replace('+', '_')}.parquet")
                cal = dv[dv.split == "calib"]
                dv = dv.set_index("sample_id").loc[frame.sample_id]
                for key in KEYS:
                    d = dv[f"d_{key}"].values
                    fused_rank = (rankdata(s) + rankdata(-d)) / 2.0
                    fused_p = s * pvalue(cal[f"d_{key}"].values, d)
                    entry[stream][f"{variant}|{key}"] = {
                        "d_only": measures(-d[is_id], -d[~is_id]),
                        "rank_avg": measures(fused_rank[is_id], fused_rank[~is_id]),
                        "p_prod": measures(fused_p[is_id], fused_p[~is_id]),
                    }
        per_seed.append(entry)
    agg = aggregate(per_seed)
    out = {"note": "exploratory, post hoc; fusion rules are parameter-free", "aggregate": agg,
           "seconds": round(time.time() - start, 1)}
    (rep / "fusion.json").write_text(json.dumps(out, indent=1) + "\n")

    def cell(x):
        return f"{100 * x['AUROC']['mean']:.1f} / {100 * x['FPR95']['mean']:.1f}"

    lines = ["\n## TINS と d(x) のスコア融合（事後解析、パラメータなし）\n",
             "AUROC / FPR95（%、3 シード平均）。rank_avg = ストリーム内順位の平均、p_prod = S_final × d の較正 p 値。\n",
             "| スコア | ID vs near | ID vs far |", "|---|---|---|",
             f"| TINS S_final | {cell(agg['near']['tins'])} | {cell(agg['far']['tins'])} |"]
    for variant in VARIANTS:
        for key in KEYS:
            k = f"{variant}|{key}"
            for rule in ("d_only", "rank_avg", "p_prod"):
                lines.append(f"| {rule}（{variant}, {key}） | {cell(agg['near'][k][rule])} | {cell(agg['far'][k][rule])} |")
    report = rep / "report.md"
    text = report.read_text()
    marker = "\n## TINS と d(x) のスコア融合"
    if marker in text:
        text = text[:text.index(marker)]
    report.write_text(text.rstrip("\n") + "\n" + "\n".join(lines) + "\n")
    summary = {k: cell(agg["near"][k]["rank_avg"]) for k in agg["near"] if k != "tins"}
    print(json.dumps({"tins_near": cell(agg["near"]["tins"]), "rank_avg_near": summary, "seconds": out["seconds"]}, indent=1))


if __name__ == "__main__":
    main()
