"""Exploratory (post hoc, dev data reused): candidate-set generation K(x) for d(x).

TINS is NOT re-run: d(x) never feeds TINS, so (c)/(d) are recomputed from the main run's per-sample TINS logs
(seeded, S_arrival, S_final) joined with the new nominations. The frozen verdict of the main run is unchanged;
the frozen X/Y rule is applied here only as a reference line.

Variants (fixed before computing, written to reports/explore_candidates/plan.json):
  zsK      CLIP zero-shot top-K over the 900 ID labels ("The nice {}."), K in {5, 10, 20, 50}; zs5 = main run
  vpK      top-K classes by cosine to visual class prototypes (mean of the 12 support features) in the SAME
           feature space as d (CLIP image space for the clip view, DINOv2 space for the dino view), K in {5, 10, 20}
  zs5+vp5  union of the two
  all      every ID class (K = 900)
"""
import hashlib
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

from vins import config as C  # noqa: E402
from vins.dview import conformal_threshold, loo_stats  # noqa: E402
from vins.features import load_features  # noqa: E402
from vins.metrics import aggregate, decide, eps_tag, nom_col, per_seed, stream_independent  # noqa: E402
from vins.tins_dev import import_tins  # noqa: E402

VARIANTS = ["zs5", "zs10", "zs20", "zs50", "vp5", "vp10", "vp20", "zs5+vp5", "all"]
DESCRIPTION = {
    "zsK": "CLIP zero-shot top-K over the 900 ID labels (TINS template); zs5 is the main-run definition",
    "vpK": "top-K classes by cosine to visual prototypes (mean of 12 support features) in the view's own space",
    "zs5+vp5": "union of zs5 and vp5",
    "all": "all 900 ID classes",
}


def class_r_all(queries, support, m, chunk=1024):
    """m-th smallest cosine distance from every query to every class's support set -> (N, n_cls)."""
    n_cls, n_sup, dim = support.shape
    flat = support.reshape(n_cls * n_sup, dim)
    out = np.empty((len(queries), n_cls))
    for lo in range(0, len(queries), chunk):
        dist = 1.0 - (queries[lo:lo + chunk] @ flat.T).reshape(-1, n_cls, n_sup)
        out[lo:lo + chunk] = np.partition(dist, m - 1, axis=-1)[..., m - 1]
    return out


def mask_from_idx(idx, n_cls):
    mask = np.zeros((idx.shape[0], n_cls), dtype=bool)
    np.put_along_axis(mask, idx, True, axis=1)
    return mask


def fmt_cell(c, d, ok):
    return f"{100 * c:.2f}% / {d:.3f}{' ✓' if ok else ''}"


def main():
    start = time.time()
    rep = C.REPORTS_DIR / "explore_candidates"
    rep.mkdir(parents=True, exist_ok=True)
    criteria = json.loads(C.CRITERIA_PATH.read_text())
    plan = {
        "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "status": "exploratory, post hoc on the main-run dev data; does not change the frozen NO-GO verdict",
        "variants": VARIANTS, "descriptions": DESCRIPTION, "views": list(C.FEATURES), "m": list(C.M_LIST),
        "eps": list(C.EPS_LIST), "reference_rule": {k: criteria[k] for k in ("X", "Y", "eps_for_decision")},
        "tins_logs": "runs/main/stream_*_seed*.parquet (unchanged)",
    }
    plan_path = rep / "plan.json"
    if plan_path.exists():            # keep the plan written before the first computation
        plan_bytes = plan_path.read_bytes()
        assert json.loads(plan_bytes)["variants"] == VARIANTS
    else:
        plan_bytes = (json.dumps(plan, indent=1) + "\n").encode()
        plan_path.write_bytes(plan_bytes)
    plan_sha = hashlib.sha256(plan_bytes).hexdigest()

    import_tins()   # upstream utils on sys.path for get_measures
    samples = pd.read_parquet(C.SPLITS_DIR / "samples.parquet")
    dview = pd.read_parquet(C.RUNS_DIR / "dview" / "dview.parquet")
    row_of = {s: i for i, s in enumerate(samples.sample_id)}
    q_idx = np.array([row_of[s] for s in dview.sample_id])
    n_id = 1000 - C.N_HELDOUT
    sup_idx = np.flatnonzero((samples.split == "support").values)
    sup_idx = sup_idx[np.argsort(samples.class_idx_id.values[sup_idx], kind="stable")]
    is_cal = (dview.split == "calib").values
    is_id = (dview.group == "ID").values
    true_cls = dview.class_idx_id.values
    rows = np.arange(len(dview))

    # CLIP zero-shot ranking, computed exactly as in prepare_tins_clip (GPU float32 matmul + topk)
    setup = torch.load(C.RUNS_DIR / "setup" / "tins_setup.pt", map_location="cpu")
    clip_all, _ = load_features(C.FEATURES_DIR / "clip.pt")
    with torch.no_grad():
        sims = clip_all.cuda() @ setup["positive_features"].cuda().T
        top50 = sims.topk(50, dim=1).indices.cpu().numpy()[q_idx]
    saved_top5 = np.array(dview.K_id.tolist())
    top5_mismatch = int(sum(set(a) != set(b) for a, b in zip(top50[:, :5], saved_top5)))   # as sets
    top5_order_only = int((top50[:, :5] != saved_top5).any(axis=1).sum()) - top5_mismatch
    zs5 = mask_from_idx(saved_top5, n_id)
    masks = {"zs5": {f: zs5 for f in C.FEATURES}}
    for k in (10, 20, 50):
        mk = zs5 | mask_from_idx(top50[:, :k], n_id)
        masks[f"zs{k}"] = {f: mk for f in C.FEATURES}
    masks["all"] = {f: np.ones_like(zs5) for f in C.FEATURES}

    z_all, check = {}, {}
    for feat in C.FEATURES:
        feats, blob = load_features(C.FEATURES_DIR / f"{feat}.pt")
        assert blob["sample_id"] == samples.sample_id.tolist()
        feats = feats.numpy().astype(np.float64)
        support = feats[sup_idx].reshape(n_id, C.N_SUPPORT, -1)
        queries = feats[q_idx]
        protos = support.mean(axis=1)
        protos /= np.linalg.norm(protos, axis=1, keepdims=True)
        order = np.argsort(-(queries @ protos.T), axis=1)
        for k in (5, 10, 20):
            masks.setdefault(f"vp{k}", {})[feat] = mask_from_idx(order[:, :k], n_id)
        masks.setdefault("zs5+vp5", {})[feat] = zs5 | masks["vp5"][feat]
        for m in C.M_LIST:
            stats = loo_stats(support, m, C.N0, C.MAD_SCALE)
            z = (class_r_all(queries, support, m) - stats["med_t"][None, :]) / stats["mad_t"][None, :]
            z_all[(feat, m)] = z
            d0 = np.where(zs5, z, np.inf).min(axis=1)
            check[f"{feat}_m{m}"] = float(np.abs(d0 - dview[f"d_{feat}_m{m}"].values).max())

    frames = {}
    for seed in C.ORDER_SEEDS:
        frames[seed] = (pd.read_parquet(C.RUNS_DIR / "main" / f"stream_near_seed{seed}.parquet"),
                        pd.read_parquet(C.RUNS_DIR / "main" / f"stream_far_seed{seed}.parquet"))
    view_cols = [c for c in dview.columns if c.startswith(("d_", "nom_", "cstar_"))]

    results = {}
    for variant in VARIANTS:
        dv = dview[["sample_id", "split", "group", "class_idx_id", "zs_top1_correct"]].copy()
        info = {"mean_K": {}, "true_in_K_iddev": {}, "true_in_K_calib": {}, "thresholds": {}}
        for feat in C.FEATURES:
            mask = masks[variant][feat]
            in_k = np.zeros(len(dv), dtype=bool)
            in_k[is_id] = mask[rows[is_id], true_cls[is_id]]
            dv[f"true_in_K_{feat}"] = in_k
            info["mean_K"][feat] = float(mask.sum(axis=1).mean())
            info["true_in_K_iddev"][feat] = float(in_k[(dv.split == "id_dev").values].mean())
            info["true_in_K_calib"][feat] = float(in_k[is_cal].mean())
            for m in C.M_LIST:
                d = np.where(mask, z_all[(feat, m)], np.inf).min(axis=1)
                dv[f"d_{feat}_m{m}"] = d
                for eps in C.EPS_LIST:
                    q = conformal_threshold(d[is_cal], eps)
                    info["thresholds"][f"{feat}_m{m}_{eps}"] = q
                    dv[nom_col(feat, m, eps)] = d > q
        indep = stream_independent(dv)
        out_dir = C.RUNS_DIR / "explore_candidates"
        out_dir.mkdir(parents=True, exist_ok=True)
        dv.to_parquet(out_dir / f"dview_{variant.replace('+', '_')}.parquet", index=False)
        seed_metrics = []
        cols = ["sample_id"] + [c for c in dv.columns if c.startswith(("d_", "nom_"))]
        for seed in C.ORDER_SEEDS:
            fn, ff = frames[seed]
            fn2 = fn.drop(columns=view_cols).merge(dv[cols], on="sample_id", how="left", validate="one_to_one")
            ff2 = ff.drop(columns=view_cols).merge(dv[cols], on="sample_id", how="left", validate="one_to_one")
            assert fn2.sample_id.equals(fn.sample_id) and not fn2[cols].isna().any().any()
            seed_metrics.append(per_seed(fn2, ff2))
        agg = aggregate(seed_metrics)
        verdict, rows_dec = decide(agg, criteria)
        results[variant] = {"info": info, "stream_independent": indep, "aggregate": agg,
                            "reference_rule": verdict, "decision": rows_dec}
        print(variant, verdict, flush=True)

    def clean(obj):
        if isinstance(obj, dict):
            return {str(k): clean(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [clean(v) for v in obj]
        if isinstance(obj, (float, np.floating)):
            return None if not np.isfinite(obj) else float(obj)
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, np.bool_):
            return bool(obj)
        return obj

    summary = {"plan_sha256": plan_sha, "zs5_reproduces_main_max_abs_diff": check,
               "zeroshot_top5_set_mismatch_vs_main": top5_mismatch, "zeroshot_top5_order_only_differences": top5_order_only, "results": results,
               "seconds": round(time.time() - start, 1)}
    (rep / "metrics.json").write_text(json.dumps(clean(summary), indent=1, ensure_ascii=False) + "\n")

    # ---------- report ----------
    L = ["# 候補集合 K(x) の生成方法の探索（探索的・事後解析）\n",
         f"- 生成: {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}、計画 `plan.json` sha256 `{plan_sha[:16]}…`（計算前に保存）",
         "- 本実行と同じ dev データ・同じ TINS ログ（runs/main、3 シード）を使い、d(x) の候補集合だけを変えた。TINS は再実行していない。",
         "- 凍結した判定（NO-GO）は変わらない。X/Y の基準は参考として当てはめているだけで、dev を見て選んだ結果は確認用の別 split で検証が必要。",
         f"- 整合性: zs5 は本実行の d(x) を再現（最大差 {max(check.values()):.1e}）、zero-shot 上位 5 の集合の不一致 {top5_mismatch} 件（並び順だけの違い {top5_order_only} 件）\n"]
    L.append("## 候補集合の大きさと、真のクラスを含む割合（ID dev）\n")
    L.append("| 候補集合 | 平均 \\|K\\|（CLIP 空間 / DINOv2 空間） | 真のクラス ∈ K（CLIP / DINOv2） |")
    L.append("|---|---|---|")
    for v in VARIANTS:
        inf = results[v]["info"]
        L.append(f"| {v} | {inf['mean_K']['clip']:.1f} / {inf['mean_K']['dino']:.1f} | "
                 f"{100 * inf['true_in_K_iddev']['clip']:.1f}% / {100 * inf['true_in_K_iddev']['dino']:.1f}% |")
    L.append("")
    for feat, title in (("dino", "DINOv2"), ("clip", "CLIP")):
        L.append(f"## 判定セル（{title}）: (c) near 新規種率 / (d) 1:1 精度（✓ = 参考基準 X={100 * criteria['X']:.0f}%, Y={criteria['Y']} を満たす）\n")
        head = [f"m={m}, ε={100 * e:g}%" for m in C.M_LIST for e in criteria["eps_for_decision"]]
        L.append("| 候補集合 | " + " | ".join(head) + " | ε=2%（判定外）m=1 |")
        L.append("|---|" + "---|" * (len(head) + 1))
        for v in VARIANTS:
            cells = []
            for m in C.M_LIST:
                for e in criteria["eps_for_decision"]:
                    r = next(x for x in results[v]["decision"] if x["feature"] == feat and x["m"] == m and x["eps"] == e)
                    cells.append(fmt_cell(r["c_mean"], r["d11_mean"], r["pass"]))
            a2 = results[v]["aggregate"]["views"][f"{feat}_m1"][eps_tag(0.02)]
            cells.append(fmt_cell(a2["c"]["rate"]["mean"], a2["d"]["precision_1to1"]["mean"], False))
            L.append(f"| {v} | " + " | ".join(cells) + " |")
        L.append("")
    L.append("## d(x) 単体の AUROC（ID dev vs near / ID dev vs far）\n")
    L.append("| 候補集合 | clip m=1 | clip m=2 | dino m=1 | dino m=2 |")
    L.append("|---|---|---|---|---|")
    for v in VARIANTS:
        cells = []
        for key in ("clip_m1", "clip_m2", "dino_m1", "dino_m2"):
            g = results[v]["stream_independent"][key]["g_d"]
            cells.append(f"{100 * g['ID_vs_near']['AUROC']:.1f} / {100 * g['ID_vs_far']['AUROC']:.1f}")
        L.append(f"| {v} | " + " | ".join(cells) + " |")
    L.append("")
    L.append("## ID が推薦された場合の内訳（DINOv2, m=1, ε=1%）と較正確認\n")
    L.append("| 候補集合 | P(推薦 \\| ID dev) | 推薦された ID | 真のクラス ∉ K | top-1 誤り（∈ K） | その他 |")
    L.append("|---|---|---|---|---|---|")
    for v in VARIANTS:
        cell = results[v]["stream_independent"]["dino_m1"][eps_tag(0.01)]
        e, b = cell["e"], cell["b"]
        L.append(f"| {v} | {100 * b['P_nom_ID']:.2f}% | {e['n_ID_nominated']} | {100 * (e['true_not_in_K']['frac'] or 0):.1f}% | "
                 f"{100 * (e['top1_wrong_true_in_K']['frac'] or 0):.1f}% | {100 * (e['other_top1_correct']['frac'] or 0):.1f}% |")
    L.append("")
    (rep / "report.md").write_text("\n".join(L) + "\n")
    print(json.dumps({"check": check, "top5_mismatch": top5_mismatch, "seconds": round(time.time() - start, 1)}))


if __name__ == "__main__":
    main()
