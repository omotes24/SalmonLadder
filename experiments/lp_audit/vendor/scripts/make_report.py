"""Step 4: metrics.json + report.md (the go/no-go rule is read from the frozen criteria.json)."""
import argparse
import glob
import hashlib
import json
import math
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from vins import config as C  # noqa: E402
from vins.metrics import aggregate, decide, eps_tag, per_seed, stream_independent  # noqa: E402
from vins.tins_dev import import_tins  # noqa: E402


def clean(obj):
    if isinstance(obj, dict):
        return {str(k): clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [clean(v) for v in obj]
    if isinstance(obj, (np.floating, float)):
        return None if math.isnan(float(obj)) or math.isinf(float(obj)) else float(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    return obj


def load_json(path, default=None):
    path = Path(path)
    return json.loads(path.read_text()) if path.exists() else default


def pct(x, nd=2):
    return "–" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{100 * x:.{nd}f}%"


def ms(cell, scale=100, nd=2, unit="%"):
    mean, sd = cell["mean"], cell["sd"]
    if mean is None or math.isnan(mean):
        return "–"
    text = f"{scale * mean:.{nd}f}"
    if sd is not None and not math.isnan(sd):
        text += f" ± {scale * sd:.{nd}f}"
    return text + unit


def num(x, nd=3):
    return "–" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.{nd}f}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", required=True)
    opts = parser.parse_args()
    run = C.RUNS_DIR / opts.tag
    out = C.REPORTS_DIR / ("gonogo" if opts.tag == "main" else f"gonogo_{opts.tag}")
    out.mkdir(parents=True, exist_ok=True)
    import_tins()  # puts upstream utils on sys.path (metrics use utils.detection_util.get_measures)

    criteria_bytes = C.CRITERIA_PATH.read_bytes()
    criteria = json.loads(criteria_bytes)
    criteria_sha = hashlib.sha256(criteria_bytes).hexdigest()
    samples = pd.read_parquet(C.SPLITS_DIR / "samples.parquet")
    build_info = load_json(C.SPLITS_DIR / "build_info.json")
    heldout = load_json(C.SPLITS_DIR / "heldout.json")
    leak = load_json(C.RUNS_DIR / "setup" / "static_negative_leak_check.json")
    dview_meta = load_json(C.RUNS_DIR / "dview" / "thresholds.json")
    dv = pd.read_parquet(C.RUNS_DIR / "dview" / "dview.parquet")
    if opts.tag.startswith("smoke"):
        dv = dv[dv.smoke.values | (dv.split.values == "calib")]

    seeds = sorted(int(Path(p).stem.split("seed")[1]) for p in glob.glob(str(run / "stream_near_seed*.parquet")))
    seed_metrics = []
    frames = {}
    for seed in seeds:
        fn = pd.read_parquet(run / f"stream_near_seed{seed}.parquet")
        ff = pd.read_parquet(run / f"stream_far_seed{seed}.parquet")
        frames[seed] = (fn, ff)
        seed_metrics.append(per_seed(fn, ff))
    agg = aggregate(seed_metrics)
    indep = stream_independent(dv)
    verdict, rows = decide(agg, criteria)

    tins_commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=C.TINS_DIR, capture_output=True,
                                 text=True).stdout.strip()
    timings = {"setup": load_json(C.RUNS_DIR / "setup" / "timings.json"),
               "dino": load_json(C.RUNS_DIR / "setup" / "timings_dino.json"),
               "dview_s": (dview_meta or {}).get("seconds"),
               "splits_s": (build_info or {}).get("seconds"),
               "tins": load_json(run / "timings_tins.json"),
               "stages": load_json(run / "stages.json")}
    shadow = load_json(run / "shadow.json")
    regression = {Path(p).stem: load_json(p) for p in sorted(glob.glob(str(C.RUNS_DIR / "regression" / "result_*.json")))}
    fn0, ff0 = frames[seeds[0]]
    sizes = {
        "support": int((samples.split == "support").sum()), "calib": int((samples.split == "calib").sum()),
        "id_dev": int(fn0.group.eq("ID").sum()), "near_dev": int(fn0.group.eq("near").sum()),
        "far_dev": int(ff0.group.eq("far").sum()),
        "stream_near": int(len(fn0)), "stream_far": int(len(ff0)),
    }
    metrics = clean({
        "tag": opts.tag, "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "tins_commit": tins_commit, "upstream_commit": C.TINS_COMMIT, "criteria": criteria,
        "criteria_sha256": criteria_sha, "verdict": verdict, "decision": rows, "order_seeds": seeds,
        "sizes": sizes, "split_build": {k: build_info.get(k) for k in ("counts", "n_rejections", "split_seed")},
        "static_negative_leak": {k: v for k, v in leak.items() if k != "rows"},
        "per_seed": dict(zip(map(str, seeds), seed_metrics)), "aggregate": agg,
        "stream_independent": indep, "thresholds": dview_meta, "shadow": shadow,
        "regression": regression, "timings": timings,
    })
    (out / "metrics.json").write_text(json.dumps(metrics, indent=1, ensure_ascii=False) + "\n")

    L = []
    L.append(f"# VINS go/no-go 予備実験レポート（{opts.tag}）\n")
    L.append(f"- 生成: {metrics['generated_utc']}")
    L.append(f"- TINS: upstream {C.TINS_COMMIT[:7]} ＋ フック 1 コミット（HEAD {tins_commit[:7]}）")
    L.append(f"- 判定基準: `criteria.json`（sha256 `{criteria_sha[:16]}…`、凍結 {criteria.get('frozen_utc')}）")
    L.append(f"- 順序シード: {seeds}（平均 ± SD は ddof=1）\n")
    if opts.tag.startswith("smoke"):
        L.append("> スモークテストです（ストリームを約 1/10 に縮小、シード 1 個）。パイプライン確認用で、判定には使いません。\n")
    L.append("## 判定\n")
    L.append(f"**{verdict}**\n")
    L.append(f"条件: ε ∈ {{{', '.join(pct(e, 1) for e in criteria['eps_for_decision'])}}} × 特徴 ∈ {{CLIP, DINOv2}} × m ∈ {{1, 2}} "
             f"のいずれかで、(c) ≥ {pct(criteria['X'], 0)} かつ (d)[1:1] ≥ {criteria['Y']}（シード平均）。\n")
    L.append("| 特徴 | m | ε | (c) near 新規種率 | (d) 1:1 精度 | (c) 基準 | (d) 基準 | 結果 |")
    L.append("|---|---|---|---|---|---|---|---|")
    for r in rows:
        c_cell = {"mean": r["c_mean"], "sd": r["c_sd"]}
        d_cell = {"mean": r["d11_mean"], "sd": r["d11_sd"]}
        L.append(f"| {r['feature']} | {r['m']} | {pct(r['eps'], 1)} | {ms(c_cell)} | {ms(d_cell, 1, 3, '')} | "
                 f"{'満たす' if r['pass_c'] else '満たさない'} | {'満たす' if r['pass_d'] else '満たさない'} | "
                 f"{'○' if r['pass'] else '×'} |")
    met = [f"{r['feature']}/m={r['m']}/ε={pct(r['eps'], 1)}" for r in rows if r["pass"]]
    L.append(f"\n満たした組み合わせ: {', '.join(met) if met else 'なし'}\n")

    L.append("## 分割サイズ\n")
    L.append("| 分割 | 枚数 |")
    L.append("|---|---|")
    for k in ("support", "calib", "id_dev", "near_dev", "far_dev", "stream_near", "stream_far"):
        L.append(f"| {k} | {sizes[k]:,} |")
    nrej = build_info.get("n_rejections", {})
    L.append(f"\nテスト画像とのバイト一致・重複で除外して補充した候補: ID dev {nrej.get('id_dev', 0)}、near dev "
             f"{nrej.get('near_dev', 0)}、far dev {nrej.get('far_dev', 0)}（far は補充なし）。")
    sdup = build_info.get("support_cross_class_duplicates", [])
    L.append(f"TINS の 16-shot 自体に、別クラスで同一の画像が {len(sdup)} 組あります（すべて支持集合内）。"
             "upstream のプロトタイプと同じく残しています: "
             + "; ".join(" = ".join(d["copies"]) for d in sdup) + "\n")

    L.append("## 静的負例への held-out 名の混入（記録のみ、除外なし）\n")
    L.append(f"- 選ばれた静的負例 {leak['n_selected_static_negatives']} 語のうち、held-out クラス名・WordNet 同義語と一致したクラス: "
             f"{leak['n_heldout_with_name_in_selected_static_negatives']} / {leak['n_heldout']}")
    L.append(f"- 候補語彙（選択前）に含まれていたクラス: {leak['n_heldout_with_name_in_candidate_pool']} / {leak['n_heldout']}")
    hits = [r for r in leak["rows"] if r["in_selected_static_negatives"]]
    for r in hits:
        L.append(f"  - {r['wnid']} {r['clean_name']}: {', '.join(r['in_selected_static_negatives'])}")
    L.append("")

    A = agg
    L.append("## 指標\n")
    L.append("### (a) TINS の種カバレッジ P(seeded)\n")
    L.append("| ID（near ストリーム） | near | ID（far ストリーム） | far |")
    L.append("|---|---|---|---|")
    a = A["a"]
    L.append(f"| {ms(a['P_seeded_ID_nearstream'])} | {ms(a['P_seeded_near'])} | {ms(a['P_seeded_ID_farstream'])} | {ms(a['P_seeded_far'])} |\n")

    L.append("### (b) 推薦率 P(nominated)（d はストリーム非依存）\n")
    L.append("| 特徴/m | ε | ID dev（較正確認） | 較正集合 | near | far |")
    L.append("|---|---|---|---|---|---|")
    for key, view in indep.items():
        for eps in C.EPS_LIST:
            b = view[eps_tag(eps)]["b"]
            L.append(f"| {key} | {pct(eps, 1)} | {pct(b['P_nom_ID'])} | {pct(b['P_nom_calib'])} | {pct(b['P_nom_near'])} | {pct(b['P_nom_far'])} |")
    L.append("")

    L.append("### (c) 新規種 P(nominated ∧ ¬seeded | near)（主指標）と (d) 精度\n")
    L.append("| 特徴/m | ε | (c) 率 | (c) 件数 | ID 新規種率 | (d) dev 比率 | (d) 1:1 | far 新規種率 | far (d) 1:1 |")
    L.append("|---|---|---|---|---|---|---|---|---|")
    for key in indep:
        for eps in C.EPS_LIST:
            v = A["views"][key][eps_tag(eps)]
            L.append(f"| {key} | {pct(eps, 1)} | {ms(v['c']['rate'])} | {ms(v['c']['count'], 1, 1, '')} | "
                     f"{ms(v['ID_novel_nearstream']['rate'])} | {ms(v['d']['precision_dev'], 1, 3, '')} | "
                     f"{ms(v['d']['precision_1to1'], 1, 3, '')} | {ms(v['far_novel']['rate'])} | "
                     f"{ms(v['far_novel']['precision_1to1'], 1, 3, '')} |")
    L.append("")

    L.append("### (e) ID が推薦された場合の内訳（排他的: 真のクラス ∉ K(x) → top-1 誤り → その他）\n")
    L.append("| 特徴/m | ε | 推薦された ID | ∉ K(x) | top-1 誤り（∈ K） | その他 |")
    L.append("|---|---|---|---|---|---|")
    for key, view in indep.items():
        for eps in C.EPS_LIST:
            e = view[eps_tag(eps)]["e"]
            L.append(f"| {key} | {pct(eps, 1)} | {e['n_ID_nominated']} | {pct(e['true_not_in_K']['frac'], 1)} | "
                     f"{pct(e['top1_wrong_true_in_K']['frac'], 1)} | {pct(e['other_top1_correct']['frac'], 1)} |")
    L.append("")

    L.append("### (f) Spearman(d, S_arrival)（d は OOD 方向、S は ID 方向）\n")
    L.append("| 特徴/m | near | ID（near ストリーム） | far | ID（far ストリーム） |")
    L.append("|---|---|---|---|---|")
    for key in indep:
        f = A["views"][key]["f"]
        L.append(f"| {key} | {ms(f['near'], 1, 3, '')} | {ms(f['ID_nearstream'], 1, 3, '')} | {ms(f['far'], 1, 3, '')} | {ms(f['ID_farstream'], 1, 3, '')} |")
    L.append("")

    L.append("### (g) AUROC / FPR95\n")
    L.append("| スコア | ID vs near AUROC | FPR95 | ID vs far AUROC | FPR95 |")
    L.append("|---|---|---|---|---|")
    g = A["g_S_final"]
    L.append(f"| TINS S_final | {ms(g['ID_vs_near']['AUROC'])} | {ms(g['ID_vs_near']['FPR95'])} | {ms(g['ID_vs_far']['AUROC'])} | {ms(g['ID_vs_far']['FPR95'])} |")
    for key, view in indep.items():
        gd = view["g_d"]
        L.append(f"| d(x) {key} | {pct(gd['ID_vs_near']['AUROC'])} | {pct(gd['ID_vs_near']['FPR95'])} | {pct(gd['ID_vs_far']['AUROC'])} | {pct(gd['ID_vs_far']['FPR95'])} |")
    L.append("")

    L.append("## 較正と d(x) の診断\n")
    L.append("| 特徴/m | q_0.5% | q_1% | q_2% | med_all | MAD_all | 較正 z(真のクラス) 中央値 | ID dev z(真のクラス) 中央値 |")
    L.append("|---|---|---|---|---|---|---|---|")
    for key, th in dview_meta["thresholds"].items():
        dg = dview_meta["diagnostics"][key]
        qs = [num(th[str(e)]["q"]) for e in C.EPS_LIST]
        L.append(f"| {key} | {' | '.join(qs)} | {num(dg['med_all'], 4)} | {num(dg['mad_all'], 4)} | {num(dg['median_z_true_calib'])} | {num(dg['median_z_true_iddev'])} |")
    L.append("\nLOO は 11 近傍、クエリは 12 近傍で r_c を計算するため、真のクラスの z はやや負に寄ります（上の中央値がその大きさです）。\n")

    if shadow:
        L.append("## シャドーモード（任意）\n")
        own = shadow["tins_own_seeds_logged"]
        L.append(f"- 設定: {shadow['config']}、順序シード {shadow['order_seed']}、上限 {shadow['limit']} サンプル")
        L.append(f"- TINS 自身の種（ログ）: 全体 {pct(own['all']['pass_rate'], 1)}（n={own['all']['n']}）、near {pct(own['near']['pass_rate'], 1)}（n={own['near']['n']}）、ID {pct(own['ID']['pass_rate'], 1)}（n={own['ID']['n']}）")
        so = shadow["shadow_on_tins_own_seeds"]
        L.append(f"- 対照（TINS 自身の種をシャドーで再実行）: near {pct(so['near']['pass_rate'], 1)}（n={so['near']['n']}）、ID {pct(so['ID']['pass_rate'], 1)}（n={so['ID']['n']}）")
        for feat, r in shadow["shadow_on_near_novel"].items():
            L.append(f"- near 新規種（{feat}）: 通過率 {pct(r['pass_rate'], 1)}（n={r['n']} / 全 {r['n_novel_total']}）")
        L.append("")

    if regression:
        L.append("## 回帰テスト（フック OFF / ON と pristine upstream の比較）\n")
        for name, res in regression.items():
            for cfg in res["configs"]:
                L.append(f"- {res['mode']} / {cfg['name']}: pristine 再現 {cfg['pristine_repeatable']}、"
                         f"hook OFF = pristine {cfg['off_equals_pristine']}、hook ON = OFF {cfg['on_equals_off']}"
                         f"（n={cfg['n']}、種 {cfg['n_seeded']}、更新バッチ {cfg['n_batches_updated']}、Flash {cfg['n_flash']}）")
        L.append("")

    equiv = load_json(C.WORK / "equiv" / "result.json")
    if equiv:
        L.append("## upstream main() との等価性チェック（1000 ラベル、dev 画像のみ）\n")
        L.append(f"GitHub 版そのまま（pristine worktree @{C.TINS_COMMIT[:7]}）の `main()` と、本実験のドライバを同じ設定で比較。"
                 f"静的負例 2000 語の一致: {equiv['selected_static_negatives_identical']}\n")
        L.append("| ストリーム | n | 順序一致 | main の特徴でスコアがビット一致 | 自前特徴との特徴最大差 | 自前特徴でのスコア最大差 | AUROC（main / ドライバ） |")
        L.append("|---|---|---|---|---|---|---|")
        for name, r in equiv["streams"].items():
            L.append(f"| {name} | {r['n']} | {r['order_matches_upstream']} | {r['a_scores_bitwise_equal_upstream']} | "
                     f"{r['features_max_abs_diff']:.2e} | {r['b_max_abs_diff']:.2e} | "
                     f"{100 * r['auroc_upstream_main']:.2f}% / {100 * r['auroc_driver_a']:.2f}% |")
        L.append("")

    L.append("## 実行時間と GPU\n")
    tins_t = timings["tins"] or {}
    for k, v in tins_t.items():
        enc = f"、特徴エンコード {v['encode_seconds']} 秒" if "encode_seconds" in v else ""
        L.append(f"- TINS {k}: スコア計算 {v['seconds']} 秒（{v['batches']} バッチ、{v['sec_per_batch']} 秒/バッチ、"
                 f"ピーク {v['cuda_peak_MiB']} MiB{enc}）")
    if timings["setup"]:
        L.append(f"- TINS 準備＋CLIP 特徴: {timings['setup'].get('total_s')} 秒（CLIP 特徴 {timings['setup'].get('clip_features_s')} 秒、ピーク {timings['setup'].get('cuda_peak_MiB')} MiB）")
    if timings["dino"]:
        L.append(f"- DINOv2 特徴: {timings['dino'].get('dino_features_s')} 秒（ピーク {timings['dino'].get('cuda_peak_MiB')} MiB）")
    L.append(f"- 分割構築: {timings['splits_s']} 秒、d(x): {timings['dview_s']} 秒")
    if timings["stages"]:
        for k, v in timings["stages"].items():
            L.append(f"- 工程 {k}: {v} 秒")
    L.append("")

    L.append("## 仕様からの逸脱・解釈\n")
    tins_t = timings["tins"] or {}
    paths = sorted({v.get("feature_path", "ours") for v in tins_t.values()})
    max_diff = max((v.get("feature_max_abs_diff_vs_dview", 0.0) for v in tins_t.values()), default=0.0)
    if paths == ["upstream"]:
        feature_line = ("TINS に渡す特徴は upstream の経路（ImageListDataset → build_mixed_stream_loader → "
                        "load_or_cache_stream_features_and_gt）でストリーム順にエンコードした。d(x) はサンプル単位で 1 回だけ"
                        f"計算した特徴を使う（両者の最大差 {max_diff:.2e}、fp16 のバッチ構成による）。")
    else:
        feature_line = ("CLIP 特徴はサンプル単位で 1 回だけ計算し、ストリーム順に並べ替えて TINS に渡した"
                        "（upstream はストリーム順にエンコードする。差は fp16 の丸め程度）。")
    deviations = [
        "held-out の候補から同名ラベルの 4 クラス（missile ×2、sunglasses ×2）を除外した（承認済みの解釈 3）。",
        "dev 候補のうち OpenOOD のテスト画像・val_imagenet とバイト一致するものを除外し、ID/near は同じクラスから補充した（far は補充なし）。",
        feature_line,
        "TINS のプロトタイプは upstream のコードで 1000 クラス分を計算し、ID 900 クラスの行だけを使った（クラスごとの平均なので 900 クラスで計算した場合と同じ）。",
        "静的負例は 900 クラスの正例で upstream の関数を使って作り直した（held-out 名は除外していない）。",
        "TINS のハイパーパラメータは Codex の再現と同じ（G=5、β=0.3、30 ステップ、λ=0.3、M=2000、バッファあり）。`--eval-protocol` は名前だけで、評価セットは読んでいない。",
        "DINOv2 の前処理は標準の評価用（Resize 256 → CenterCrop 224 → ImageNet 正規化）。",
        "判定には m=2 も含めた（ユーザー決定）。",
    ]
    for dline in deviations:
        L.append(f"- {dline}")
    L.append("")

    L.append("## held-out 100 クラス\n")
    L.append("| wnid | クラス名 | 直上の親 | ID 側の兄弟 |")
    L.append("|---|---|---|---|")
    for h in heldout:
        sib = h["id_siblings"]
        sib_text = ", ".join(s["clean_name"] for s in sib[:3]) + (f" ほか {len(sib) - 3}" if len(sib) > 3 else "")
        L.append(f"| {h['wnid']} | {h['clean_name']} | {', '.join(h['parents'])} | {sib_text} |")
    L.append("")
    (out / "report.md").write_text("\n".join(L) + "\n")
    print(json.dumps({"verdict": verdict, "report": str(out / "report.md")}))


if __name__ == "__main__":
    main()
