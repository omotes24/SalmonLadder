"""R5 summaries of r5_eval.py outputs (dev1 and dev2; 5 shot draws x 3 order seeds = 15 conditions per split).

Unit of uncertainty: the shot draw. Each draw's value is the mean over its 3 order seeds; the 95% interval is a
t interval over the 5 draw means (4 d.o.f.). Differences are computed within a condition, then averaged the same way.
Draw "orig" (the shots used during development) is reported separately as a reference and a regression check.
Baseline hyper-parameters (k of kNN, shrinkage of Mahalanobis++) are chosen on dev1 by mean near FPR95 and then
frozen for dev2.
Output: <R5>/summary/stageA.json, stageA.md
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from vins import r5  # noqa: E402

DRAWS = ["0", "1", "2", "3", "4"]
SEEDS = [123, 124, 125]
STREAMS = ["near", "far"]
METRICS = ["FPR95", "AUROC", "AUROC_within_batch"]
KS = (1, 2, 3, 5, 10, 20, 50)
LAMS = ("0", "0.001", "0.01", "0.05", "0.1", "0.2", "0.5")


def load(dev, draw):
    out = {}
    for s in STREAMS:
        for sd in SEEDS:
            f = r5.R5 / dev / "eval" / f"draw{draw}" / f"{s}_seed{sd}.json"
            if f.exists():
                out[(s, sd)] = json.loads(f.read_text())
    return out


def draw_means(runs, key, stream, metric):
    """{draw: mean over seeds}"""
    res = {}
    for draw, per in runs.items():
        vals = [per[(stream, sd)]["metrics"][key][metric] for sd in SEEDS if (stream, sd) in per]
        if len(vals) == len(SEEDS):
            res[draw] = float(np.mean(vals))
    return res


def compare(runs, a, b, stream, metric):
    """a - b within condition, averaged per draw, t interval over draws."""
    per_draw = []
    for draw, per in runs.items():
        vals = [per[(stream, sd)]["metrics"][a][metric] - per[(stream, sd)]["metrics"][b][metric] for sd in SEEDS]
        per_draw.append(float(np.mean(vals)))
    out = r5.t_interval(per_draw)
    out["per_draw"] = per_draw
    return out


def summarise(runs, keys):
    tab = {}
    for key in keys:
        tab[key] = {}
        for s in STREAMS:
            for m in METRICS:
                dm = draw_means(runs, key, s, m)
                v = np.array(list(dm.values()))
                tab[key][f"{s}_{m}"] = {"mean": float(v.mean()), "sd_draw": float(v.std(ddof=1)) if len(v) > 1 else 0.0,
                                        "per_draw": dm}
    return tab


def pick(runs, family, prefix, grid):
    """Best hyper-parameter of a baseline family by mean near FPR95 (lower is better)."""
    score = {g: np.mean(list(draw_means(runs, f"{prefix}_{family}{g}", "near", "FPR95").values())) for g in grid}
    best = min(score, key=score.get)
    return best, {str(k): float(v) for k, v in score.items()}


def main():
    res = {}
    all_runs = {dev: {d: load(dev, d) for d in DRAWS} for dev in ("dev1", "dev2")}
    orig = {dev: load(dev, "orig") for dev in ("dev1", "dev2")}
    comps = [("s_full", "s_lp"), ("s_full", "s_single_lp"), ("s_mem", "s_static"), ("s_lp", "s_static"),
             ("s_full", "s_mem"), ("s_full_i", "s_lpi"), ("s_lp", "s_lpi"), ("s_full", "s_full_i"),
             ("v_full", "v_lp"), ("v_full", "v_single_lp"), ("s_full", "tins"), ("s_lp", "tins")]
    # baseline selection on dev1 (both for x TINS and visual-only)
    sel = {}
    for prefix in ("s", "v"):
        kb, ks = pick(all_runs["dev1"], "knn", prefix, KS)
        lb, ls = pick(all_runs["dev1"], "maha", prefix, LAMS)
        sel[prefix] = {"knn": kb, "knn_scores": ks, "maha": lb, "maha_scores": ls}
    res["baseline_selection_dev1"] = sel
    keys = ["tins", "s_static", "s_mem", "s_mem1", "s_lp", "s_lpi", "s_single_lp", "s_full", "s_full_i",
            "v_static", "v_mem", "v_lp", "v_lpi", "v_single_lp", "v_full", "v_full_i",
            f"s_knn{sel['s']['knn']}", f"s_maha{sel['s']['maha']}", f"v_knn{sel['v']['knn']}", f"v_maha{sel['v']['maha']}",
            "B14_d", "B14_p", "B14_g3", "B14_pt3", "B14_u", "B14_plp", "L14_d", "L14_p", "L14_g3", "L14_pt3", "L14_u",
            "L14_plp", "raw_u_prod", "raw_g_sum"]
    comps += [("s_full", f"s_knn{sel['s']['knn']}"), ("s_full", f"s_maha{sel['s']['maha']}"),
              ("v_full", f"v_knn{sel['v']['knn']}"), ("v_full", f"v_maha{sel['v']['maha']}"),
              ("s_lp", f"s_maha{sel['s']['maha']}"), ("s_lp", f"s_knn{sel['s']['knn']}")]
    for dev, runs in all_runs.items():
        runs = {d: r for d, r in runs.items() if len(r) == 6}
        res[dev] = {"n_draws": len(runs), "table": summarise(runs, keys),
                    "compare": {f"{a} - {b}": {f"{s}_{m}": compare(runs, a, b, s, m) for s in STREAMS for m in METRICS}
                                for a, b in comps}}
        ent = {}
        for name in ("B14", "L14"):
            for stage in ("v4_stage1", "v4_stage2", "v4_M", "single_M"):
                for s in STREAMS:
                    vals = np.array([[r[(s, sd)]["entrance"][name][stage][q] for sd in SEEDS]
                                     for r in runs.values() for q in ("id_rate", "ood_rate", "purity")]).reshape(
                        len(runs), 3, len(SEEDS)).mean(axis=2)
                    ent[f"{name}|{stage}|{s}"] = {"id_rate": float(vals[:, 0].mean()), "ood_rate": float(vals[:, 1].mean()),
                                                  "purity": float(vals[:, 2].mean())}
        res[dev]["entrance"] = ent
        if len(orig[dev]) == 6:
            res[dev]["orig"] = {k: {f"{s}_{m}": float(np.mean([orig[dev][(s, sd)]["metrics"][k][m] for sd in SEEDS]))
                                    for s in STREAMS for m in ("FPR95", "AUROC")} for k in keys if k in
                                orig[dev][("near", 123)]["metrics"]}
    out = r5.R5 / "summary"
    out.mkdir(parents=True, exist_ok=True)
    (out / "stageA.json").write_text(json.dumps(res, indent=1) + "\n")
    # compact markdown
    L = []
    for dev in ("dev1", "dev2"):
        if dev not in res:
            continue
        L += [f"## {dev} (draws: {res[dev]['n_draws']})", "", "| variant | near FPR95 | near AUROC | far FPR95 | far AUROC |",
              "|---|---|---|---|---|"]
        for k in keys[:20]:
            t = res[dev]["table"][k]
            L.append(f"| {k} | {t['near_FPR95']['mean']:.2f} ± {t['near_FPR95']['sd_draw']:.2f} | "
                     f"{t['near_AUROC']['mean']:.2f} | {t['far_FPR95']['mean']:.2f} | {t['far_AUROC']['mean']:.2f} |")
        L += ["", "| comparison (a - b) | near FPR95 [95% CI] (+/-) | near AUROC [95% CI] | far FPR95 [95% CI] | far AUROC [95% CI] |",
              "|---|---|---|---|---|"]
        for c, v in res[dev]["compare"].items():
            f = lambda x: f"{x['mean']:+.2f} [{x['lo']:+.2f}, {x['hi']:+.2f}] ({x['n_positive']}/{x['n_negative']})"  # noqa: E731
            L.append(f"| {c} | {f(v['near_FPR95'])} | {f(v['near_AUROC'])} | {f(v['far_FPR95'])} | {f(v['far_AUROC'])} |")
        L.append("")
    (out / "stageA.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
