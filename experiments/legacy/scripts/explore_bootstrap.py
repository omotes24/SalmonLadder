"""Class-level bootstrap for the candidate-set exploration (post hoc, dev reused).

The order-seed SD only reflects stream order; the dev sample itself (100 held-out classes, 900 ID classes) is fixed.
Here we resample classes with replacement, B times:
  ID classes  -> their 4 calibration images (threshold q_eps is re-estimated) and their 20 ID dev images
  held-out    -> their 50 near images
Novelty uses the per-sample seeded rate over the 3 order seeds of the main run (near stream), so the statistic is
the seed-averaged (c) and (d)[1:1]. Output: CI and the fraction of replicates meeting the reference X/Y rule.
"""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from vins import config as C  # noqa: E402
from vins.dview import conformal_rank  # noqa: E402

VARIANTS = ["zs5", "zs20", "vp5", "zs5+vp5", "all"]
CELLS = [(feat, m, eps) for feat in ("dino", "clip") for m in (1, 2) for eps in (0.005, 0.01)]
B = 2000


def main():
    start = time.time()
    criteria = json.loads(C.CRITERIA_PATH.read_text())
    rng = np.random.default_rng(20260926)
    near_frames = [pd.read_parquet(C.RUNS_DIR / "main" / f"stream_near_seed{s}.parquet") for s in C.ORDER_SEEDS]
    seeded = pd.concat([f[["sample_id", "seeded"]] for f in near_frames]).groupby("sample_id").seeded.mean()

    base = pd.read_parquet(C.RUNS_DIR / "explore_candidates" / "dview_zs5.parquet")
    split, group = base.split.values, base.group.values
    cls_id = base.class_idx_id.values
    wnid = pd.read_parquet(C.RUNS_DIR / "dview" / "dview.parquet").set_index("sample_id").loc[base.sample_id].wnid.values
    cal_by_cls = [np.flatnonzero((split == "calib") & (cls_id == c)) for c in range(900)]
    iddev_by_cls = [np.flatnonzero((split == "id_dev") & (cls_id == c)) for c in range(900)]
    held = sorted(set(wnid[group == "near"]))
    near_by_cls = [np.flatnonzero((group == "near") & (wnid == w)) for w in held]
    unseeded = np.zeros(len(base))
    in_stream = split != "far_dev"
    unseeded[in_stream & (split != "calib")] = 1.0 - seeded.loc[base.sample_id[in_stream & (split != "calib")]].values

    draws = []
    for _ in range(B):
        ci = rng.integers(0, 900, 900)
        hi = rng.integers(0, len(held), len(held))
        draws.append((np.concatenate([cal_by_cls[c] for c in ci]), np.concatenate([iddev_by_cls[c] for c in ci]),
                      np.concatenate([near_by_cls[h] for h in hi])))

    out = {"B": B, "note": "post hoc; class-level bootstrap over ID classes (calib + ID dev) and held-out classes",
           "reference_rule": {k: criteria[k] for k in ("X", "Y")}, "results": {}}
    for variant in VARIANTS:
        dv = pd.read_parquet(C.RUNS_DIR / "explore_candidates" / f"dview_{variant.replace('+', '_')}.parquet")
        assert (dv.sample_id.values == base.sample_id.values).all()
        res = {}
        for feat, m, eps in CELLS:
            d = dv[f"d_{feat}_m{m}"].values
            cs, ds = np.empty(B), np.empty(B)
            for b, (cal, iddev, near) in enumerate(draws):
                cal_d = np.sort(d[cal])
                rank = conformal_rank(len(cal_d), eps)
                q = np.inf if rank > len(cal_d) else cal_d[rank - 1]
                c = float(((d[near] > q) * unseeded[near]).mean())
                idr = float(((d[iddev] > q) * unseeded[iddev]).mean())
                cs[b], ds[b] = c, (c / (c + idr) if c + idr > 0 else np.nan)
            ok = (cs >= criteria["X"]) & (ds >= criteria["Y"])
            res[f"{feat}_m{m}_eps{eps}"] = {
                "c_median": float(np.median(cs)), "c_ci95": [float(np.quantile(cs, 0.025)), float(np.quantile(cs, 0.975))],
                "d11_median": float(np.nanmedian(ds)),
                "d11_ci95": [float(np.nanquantile(ds, 0.025)), float(np.nanquantile(ds, 0.975))],
                "frac_meeting_rule": float(ok.mean()),
            }
        out["results"][variant] = res
        print(variant, {k: round(v["frac_meeting_rule"], 3) for k, v in res.items() if k.startswith("dino")}, flush=True)
    out["seconds"] = round(time.time() - start, 1)
    rep = C.REPORTS_DIR / "explore_candidates"
    (rep / "bootstrap.json").write_text(json.dumps(out, indent=1) + "\n")

    lines = ["\n## クラス単位ブートストラップ（B = 2000、事後解析）\n",
             "順序シードの SD は並び順の揺れだけを表す。ここでは ID クラス（較正 4 枚＋ID dev 20 枚、q_ε も再推定）と held-out クラス"
             "（near 50 枚）を復元抽出し、別の dev を引いた場合のばらつきを見る。\n",
             "| 候補集合 | セル | (c) 中央値 [95%] | (d) 1:1 中央値 [95%] | 参考基準を満たす割合 |", "|---|---|---|---|---|"]
    for variant in VARIANTS:
        for key, r in out["results"][variant].items():
            if not key.startswith("dino"):
                continue
            lines.append(f"| {variant} | {key.replace('_eps', ', ε=')} | {100 * r['c_median']:.2f}% "
                         f"[{100 * r['c_ci95'][0]:.2f}, {100 * r['c_ci95'][1]:.2f}] | {r['d11_median']:.3f} "
                         f"[{r['d11_ci95'][0]:.3f}, {r['d11_ci95'][1]:.3f}] | {100 * r['frac_meeting_rule']:.1f}% |")
    report = rep / "report.md"
    text = report.read_text()
    marker = "\n## クラス単位ブートストラップ"
    if marker in text:
        text = text[:text.index(marker)]
    report.write_text(text.rstrip("\n") + "\n" + "\n".join(lines) + "\n")
    print(json.dumps({"seconds": out["seconds"]}))


if __name__ == "__main__":
    main()
