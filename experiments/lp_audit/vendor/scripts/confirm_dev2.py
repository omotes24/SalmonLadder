"""Pre-registered confirmation on dev2. Reads prereg_confirm.json (written before dev2 existed); never edits it.

Primary  : fused = S_final * p_d(x), d = DINOv2 view, m=2, K = CLIP zero-shot top-5 (the dview.parquet definition)
           PASS iff  mean AUROC_near(fused) - mean AUROC_near(TINS) >= 0.05
                and  mean AUROC_far(fused)  >= max(mean AUROC_far(TINS), AUROC_far(d only))
Secondary: VINS seed-level rule with K = vp5 (DINOv2 prototypes), m=2, eps=1%: (c) >= 0.05 and (d)[1:1] >= 0.85
"""
import hashlib
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from vins import config as C  # noqa: E402
from vins.candidates import class_r_all, d_over_mask, proto_topk_mask  # noqa: E402
from vins.dview import conformal_threshold, loo_stats  # noqa: E402
from vins.features import load_features  # noqa: E402
from vins.metrics import aggregate, measures  # noqa: E402
from vins.tins_dev import import_tins  # noqa: E402


def pvalue(cal, d):
    cal = np.sort(cal)
    return (1.0 + len(cal) - np.searchsorted(cal, d, side="left")) / (len(cal) + 1.0)


def main():
    start = time.time()
    prereg_bytes = (C.WORK / "prereg_confirm.json").read_bytes()
    prereg = json.loads(prereg_bytes)
    prereg_sha = hashlib.sha256(prereg_bytes).hexdigest()
    import_tins()
    out = C.REPORTS_DIR / "confirm"
    out.mkdir(parents=True, exist_ok=True)
    samples = pd.read_parquet(C.SPLITS_DIR / "samples.parquet")
    dview = pd.read_parquet(C.RUNS_DIR / "dview" / "dview.parquet")
    is_cal = (dview.split == "calib").values
    key = "dino_m2"
    frames = {s: {st: pd.read_parquet(C.RUNS_DIR / "main" / f"stream_{st}_seed{s}.parquet") for st in ("near", "far")}
              for s in C.ORDER_SEEDS}

    # ---------------- primary ----------------
    cal_d = dview[f"d_{key}"].values[is_cal]
    per_seed = []
    for seed in C.ORDER_SEEDS:
        entry = {}
        for stream in ("near", "far"):
            f = frames[seed][stream]
            is_id = (f.group == "ID").values
            s = f.S_final.values.astype(np.float64)
            d = f[f"d_{key}"].values
            fused = s * pvalue(cal_d, d)
            entry[stream] = {"tins": measures(s[is_id], s[~is_id]), "d_only": measures(-d[is_id], -d[~is_id]),
                             "fused": measures(fused[is_id], fused[~is_id])}
        per_seed.append(entry)
    agg = aggregate(per_seed)
    near_gain = agg["near"]["fused"]["AUROC"]["mean"] - agg["near"]["tins"]["AUROC"]["mean"]
    far_ref = max(agg["far"]["tins"]["AUROC"]["mean"], agg["far"]["d_only"]["AUROC"]["mean"])
    far_ok = agg["far"]["fused"]["AUROC"]["mean"] >= far_ref
    primary = {"near_gain": near_gain, "near_ok": near_gain >= 0.05, "far_fused": agg["far"]["fused"]["AUROC"]["mean"],
               "far_reference": far_ref, "far_ok": bool(far_ok), "pass": bool(near_gain >= 0.05 and far_ok)}

    # ---------------- secondary (seed level, vp5) ----------------
    feats, blob = load_features(C.FEATURES_DIR / "dino.pt")
    assert blob["sample_id"] == samples.sample_id.tolist()
    feats = feats.numpy().astype(np.float64)
    sup = np.flatnonzero((samples.split == "support").values)
    sup = sup[np.argsort(samples.class_idx_id.values[sup], kind="stable")]
    support = feats[sup].reshape(1000 - C.N_HELDOUT, C.N_SUPPORT, -1)
    row_of = {s: i for i, s in enumerate(samples.sample_id)}
    queries = feats[[row_of[s] for s in dview.sample_id]]
    stats = loo_stats(support, 2, C.N0, C.MAD_SCALE)
    z = (class_r_all(queries, support, 2) - stats["med_t"][None, :]) / stats["mad_t"][None, :]
    d_vp5 = d_over_mask(z, proto_topk_mask(queries, support, 5))
    q = conformal_threshold(d_vp5[is_cal], 0.01)
    nominated = pd.Series(d_vp5 > q, index=dview.sample_id)
    cs, ds = [], []
    for seed in C.ORDER_SEEDS:
        f = frames[seed]["near"]
        nom = nominated.loc[f.sample_id].values
        seeded = f.seeded.values.astype(bool)
        near, ident = (f.group == "near").values, (f.group == "ID").values
        c = float((nom & ~seeded & near).sum() / near.sum())
        idr = float((nom & ~seeded & ident).sum() / ident.sum())
        cs.append(c)
        ds.append(c / (c + idr) if c + idr > 0 else float("nan"))
    secondary = {"c_per_seed": cs, "d11_per_seed": ds, "c_mean": float(np.mean(cs)), "d11_mean": float(np.mean(ds)),
                 "P_nom_iddev": float(nominated.loc[dview.sample_id[(dview.split == "id_dev").values]].mean()),
                 "threshold": q}
    secondary["pass"] = bool(secondary["c_mean"] >= 0.05 and secondary["d11_mean"] >= 0.85)

    info = json.loads((C.SPLITS_DIR / "build_info.json").read_text())
    result = {"prereg_sha256": prereg_sha, "prereg": prereg, "primary": primary, "aggregate": agg,
              "secondary": secondary, "split_counts": info["counts"],
              "previous_dev_exclusion": info.get("previous_dev_exclusion"), "seconds": round(time.time() - start, 1)}
    (out / "metrics.json").write_text(json.dumps(result, indent=1, ensure_ascii=False, default=float) + "\n")

    def cell(x):
        return (f"{100 * x['AUROC']['mean']:.2f} ± {100 * x['AUROC']['sd']:.2f} / "
                f"{100 * x['FPR95']['mean']:.2f} ± {100 * x['FPR95']['sd']:.2f}")

    L = ["# dev2 確認実験（事前登録）\n",
         f"- 事前登録: `prereg_confirm.json`（sha256 `{prereg_sha[:16]}…`、凍結 {prereg['frozen_utc']}、dev2 構築前）",
         "- dev2: held-out 100 クラスは dev1 と重複なし、ID dev / near dev の画像も dev1 と重複なし。far は OpenImage-O val を再利用。",
         f"- 順序シード: {list(C.ORDER_SEEDS)}（平均 ± SD、ddof=1）\n",
         f"## 主要評価項目: **{'PASS' if primary['pass'] else 'FAIL'}**\n",
         "| スコア | ID vs near AUROC / FPR95（%） | ID vs far AUROC / FPR95（%） |", "|---|---|---|"]
    for name, label in (("tins", "TINS S_final"), ("d_only", "d(x) 単体（DINOv2, m=2, zs5）"), ("fused", "融合 S_final × p_d")):
        L.append(f"| {label} | {cell(agg['near'][name])} | {cell(agg['far'][name])} |")
    L.append("")
    L.append(f"- near: 融合 − TINS = {100 * near_gain:+.2f} pt（基準 ≥ +5 pt） → {'満たす' if primary['near_ok'] else '満たさない'}")
    L.append(f"- far: 融合 {100 * primary['far_fused']:.2f}% vs max(TINS, d 単体) {100 * far_ref:.2f}% → "
             f"{'満たす' if primary['far_ok'] else '満たさない'}\n")
    L.append(f"## 副次評価項目（種推薦、vp5・DINOv2・m=2・ε=1%）: **{'PASS' if secondary['pass'] else 'FAIL'}**\n")
    L.append(f"- (c) near 新規種率: {100 * secondary['c_mean']:.2f}%（シード別 {', '.join(f'{100 * x:.2f}' for x in cs)}、基準 ≥ 5%）")
    L.append(f"- (d) 1:1 精度: {secondary['d11_mean']:.3f}（シード別 {', '.join(f'{x:.3f}' for x in ds)}、基準 ≥ 0.85）")
    L.append(f"- 較正確認 P(推薦 | ID dev) = {100 * secondary['P_nom_iddev']:.2f}%（ε = 1%）\n")
    L.append("## 分割\n")
    L.append(", ".join(f"{k}: {v:,}" for k, v in info["counts"].items()))
    (out / "report.md").write_text("\n".join(L) + "\n")
    print(json.dumps({"primary": primary, "secondary": {k: secondary[k] for k in ("c_mean", "d11_mean", "pass")}}, indent=1))


if __name__ == "__main__":
    main()
