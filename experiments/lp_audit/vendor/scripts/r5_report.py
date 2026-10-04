"""R5 reports (dev only): summaries of (2) entrance, injection, (1) extended Mahalanobis++ grid, E1, streams (E2-E5,
batch size), M1/M3/E5 modifications, sensitivity, and E4 (recurrence vs position on the matched dev streams).

Unit of uncertainty = shot draw (mean over the order seeds of that draw first); 95% t interval over draws (4 d.o.f.).
Differences are taken within a condition. Usage: python scripts/r5_report.py <what> [...]; writes
<R5>/summary/<what>.json and .md and prints the markdown.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from vins import r5  # noqa: E402

DRAWS = ["0", "1", "2", "3", "4"]
SEEDS = [123, 124, 125]
STREAMS = ["near", "far"]
BASE = Path.home() / "vins_gonogo_20260925"
OUT = r5.R5 / "summary"


def ci(vals):
    return r5.t_interval(vals)


def fmt(x, nd=2):
    return f"{x['mean']:+.{nd}f} [{x['lo']:+.{nd}f}, {x['hi']:+.{nd}f}] ({x['n_positive']}+/{x['n_negative']}-)"


def per_draw(runs, key, stream, metric, draws=DRAWS, seeds=SEEDS):
    """runs[(draw, stream, seed)] -> metrics dict; draw means over seeds (only complete draws)."""
    res = []
    for d in draws:
        vals = [runs[(d, stream, sd)][key][metric] for sd in seeds if (d, stream, sd) in runs and key in runs[(d, stream, sd)]]
        if len(vals) == len(seeds):
            res.append(float(np.mean(vals)))
    return res


def per_draw_diff(runs, a, b, stream, metric, draws=DRAWS, seeds=SEEDS, runs_b=None):
    runs_b = runs_b or runs
    res = []
    for d in draws:
        vals = []
        for sd in seeds:
            k = (d, stream, sd)
            if k in runs and k in runs_b and a in runs[k] and b in runs_b[k]:
                vals.append(runs[k][a][metric] - runs_b[k][b][metric])
        if len(vals) == len(seeds):
            res.append(float(np.mean(vals)))
    return res


def load_eval(dev, sub="eval"):
    runs = {}
    for d in DRAWS:
        for s in STREAMS:
            for sd in SEEDS:
                f = r5.R5 / dev / sub / f"draw{d}" / f"{s}_seed{sd}.json"
                if f.exists():
                    runs[(d, s, sd)] = json.loads(f.read_text())["metrics"]
    return runs


def write(name, res, lines):
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{name}.json").write_text(json.dumps(res, indent=1) + "\n")
    (OUT / f"{name}.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


# ---------------------------------------------------------------------------------------------------------------
def rep_maha():
    """Pick lam over the union grid on dev1 (mean near FPR95, x TINS and alone separately); compare with REPRISE."""
    res, L = {}, ["# (1) Mahalanobis++ with the extended shrinkage grid", ""]
    ev = {dev: load_eval(dev) for dev in ("dev1", "dev2")}
    ext = {dev: load_eval(dev, "eval_ext") for dev in ("dev1", "dev2")}
    both = {dev: {k: {**ev[dev].get(k, {}), **ext[dev].get(k, {})} for k in ev[dev]} for dev in ev}
    lams = sorted({k.split("maha")[1] for r in both["dev1"].values() for k in r if k.startswith("s_maha")}, key=float)
    for prefix, ref in (("s", "s_full"), ("v", "v_full"), ("sL14", "s_full"), ("vL14", "v_full")):
        scores = {}
        for lam in lams:
            v = per_draw(both["dev1"], f"{prefix}_maha{lam}", "near", "FPR95")
            if len(v) == 5:
                scores[lam] = float(np.mean(v))
        if not scores:
            continue
        best = min(scores, key=scores.get)
        res[prefix] = {"dev1_near_fpr_by_lam": scores, "best": best}
        L += [f"## {prefix}_maha: dev1 near FPR95 by lam: " + ", ".join(f"{k}: {v:.2f}" for k, v in scores.items()),
              f"best lam = {best}", "", "| dev | metric | maha (best) | REPRISE | REPRISE - maha [95% CI] |", "|---|---|---|---|---|"]
        for dev in ("dev1", "dev2"):
            for s in STREAMS:
                for m in ("FPR95", "AUROC"):
                    a = per_draw(both[dev], ref, s, m)
                    b = per_draw(both[dev], f"{prefix}_maha{best}", s, m)
                    d = ci(per_draw_diff(both[dev], ref, f"{prefix}_maha{best}", s, m))
                    res[prefix][f"{dev}_{s}_{m}"] = {"reprise": float(np.mean(a)), "maha": float(np.mean(b)), "diff": d}
                    L.append(f"| {dev} | {s} {m} | {np.mean(b):.2f} | {np.mean(a):.2f} | {fmt(d)} |")
        L.append("")
    write("maha_ext", res, L)


# ---------------------------------------------------------------------------------------------------------------
def rep_entrance():
    E = r5.R5 / "entrance"
    frozen = json.loads((E / "frozen.json").read_text())
    res, L = {"frozen": frozen}, ["# (2) Entrances at matched ID admission (tuned on dev1, frozen, evaluated on dev2)", ""]
    for dev in ("dev2", "dev1"):
        runs, stats = {}, {}
        for d in DRAWS:
            for s in STREAMS:
                for sd in SEEDS:
                    f = E / "eval" / f"{dev}_draw{d}_{s}_seed{sd}.json"
                    if not f.exists():
                        continue
                    blob = json.loads(f.read_text())
                    runs[(d, s, sd)] = {f"{cfg}|{k}": v for cfg, r in blob.items() for k, v in r["metrics"].items()}
                    stats[(d, s, sd)] = {cfg: r["stats"] for cfg, r in blob.items()}
        if not runs:
            continue
        L += [f"## {dev}", "", "| entrance | thresholds | ID adm. % | near OOD adm. % | far OOD adm. % | purity near | "
              "s_mem near FPR95 | s_mem near AUROC | s_full near FPR95 | s_full near AUROC | s_full far FPR95 |",
              "|---|---|---|---|---|---|---|---|---|---|---|"]
        res[dev] = {}
        for cfg in sorted(frozen, key=lambda c: (c == "v4", c)):
            def st(q, stream):
                v = [np.mean([stats[k][cfg][n][q] for n in ("B14", "L14")]) for k in stats if k[1] == stream]
                return 100 * float(np.nanmean(v))
            row = {"id_rate": (st("id_rate", "near") + st("id_rate", "far")) / 2, "near_ood_rate": st("ood_rate", "near"),
                   "far_ood_rate": st("ood_rate", "far"), "purity_near": st("purity", "near") / 100}
            for key in ("s_mem", "s_full", "v_mem", "v_full"):
                for s in STREAMS:
                    for m in ("FPR95", "AUROC"):
                        row[f"{key}_{s}_{m}"] = float(np.mean(per_draw(runs, f"{cfg}|{key}", s, m)))
            res[dev][cfg] = row
            L.append(f"| {cfg} | {frozen[cfg]['thresholds']} | {row['id_rate']:.2f} | {row['near_ood_rate']:.1f} | "
                     f"{row['far_ood_rate']:.1f} | {row['purity_near']:.3f} | {row['s_mem_near_FPR95']:.2f} | "
                     f"{row['s_mem_near_AUROC']:.2f} | {row['s_full_near_FPR95']:.2f} | {row['s_full_near_AUROC']:.2f} | "
                     f"{row['s_full_far_FPR95']:.2f} |")
        L += ["", "| comparison (same target) | s_mem near FPR95 | s_full near FPR95 | s_full near AUROC | s_full far FPR95 |",
              "|---|---|---|---|---|"]
        comps = []
        for tau in ("0.01", "0.05", "0.1"):
            comps += [(f"E3m2@{tau}", f"E1@{tau}"), (f"E3m2@{tau}", f"E2@{tau}"), (f"E2@{tau}", f"E1@{tau}"),
                      (f"E3m1@{tau}", f"E3m2@{tau}"), (f"E3m5@{tau}", f"E3m2@{tau}")]
        comps.append(("v4", "E1@0.1"))
        res[dev]["compare"] = {}
        for a, b in comps:
            if a not in frozen or b not in frozen:
                continue
            c = {f"{key}_{s}_{m}": ci(per_draw_diff(runs, f"{a}|{key}", f"{b}|{key}", s, m))
                 for key in ("s_mem", "s_full") for s in STREAMS for m in ("FPR95", "AUROC")}
            res[dev]["compare"][f"{a} - {b}"] = c
            L.append(f"| {a} - {b} | {fmt(c['s_mem_near_FPR95'])} | {fmt(c['s_full_near_FPR95'])} | "
                     f"{fmt(c['s_full_near_AUROC'])} | {fmt(c['s_full_far_FPR95'])} |")
        L.append("")
    write("entrance", res, L)


def rep_inject():
    E = r5.R5 / "entrance" / "inject"
    agg = {}
    for f in sorted(E.glob("dev2_draw*_near_seed123.json")):
        for key, rec in json.loads(f.read_text()).items():
            a = agg.setdefault(key, {k: 0.0 for k in rec})
            for k, v in rec.items():
                a[k] += v
    res, L = {}, ["# (2) Diagnostic injection of n same-class ID images into the memory (dev2, near, v4 entrance)", "",
                  "| view | m | n | classes | later arrivals | Pr(p<=0.05) base -> inj | Pr(p<=0.10) base -> inj | mean p base -> inj |",
                  "|---|---|---|---|---|---|---|---|"]
    for key in sorted(agg, key=lambda k: (k.split("|")[0], int(k.split("|")[1][1:]), int(k.split("|")[2][1:]))):
        a = agg[key]
        n = max(a["n_eval"], 1)
        row = {"n_classes": a["n_classes"], "n_eval": a["n_eval"], "base_le05": a["base_le05"] / n,
               "inj_le05": a["inj_le05"] / n, "base_le10": a["base_le10"] / n, "inj_le10": a["inj_le10"] / n,
               "base_mean_p": a["base_mean_p"] / n, "inj_mean_p": a["inj_mean_p"] / n}
        res[key] = row
        v, m, nn = key.split("|")
        L.append(f"| {v} | {m[1:]} | {nn[1:]} | {int(a['n_classes'])} | {int(a['n_eval'])} | {row['base_le05']:.3f} -> "
                 f"{row['inj_le05']:.3f} | {row['base_le10']:.3f} -> {row['inj_le10']:.3f} | {row['base_mean_p']:.3f} -> "
                 f"{row['inj_mean_p']:.3f} |")
    write("inject", res, L)


# ---------------------------------------------------------------------------------------------------------------
def l14_single_from_npz(dev, draws=DRAWS):
    """16-shot single-view (L/14) kNN / Mahalanobis++ x TINS and alone, from the r5_eval per-sample arrays."""
    from vins.tins_dev import import_tins

    import_tins()
    from vins.metrics import measures as upstream

    runs = {}
    for d in draws:
        for s in STREAMS:
            for sd in SEEDS:
                e = np.load(r5.R5 / dev / "eval" / f"draw{d}" / f"{s}_seed{sd}.npz", allow_pickle=True)
                z = np.load(r5.R5 / dev / "tins" / f"draw{d}" / f"{s}_seed{sd}.npz", allow_pickle=True)
                is_ood, S = e["is_ood"].astype(bool), z["S_final"].astype(np.float64)
                met = {}
                for k in e.files:
                    if k.endswith("_L14") and (k.startswith("knn") or k.startswith("maha")):
                        base = k[:-4]
                        p = e[k].astype(np.float64)
                        for name, sc in ((f"vL14_{base}", p), (f"sL14_{base}", S * p)):
                            m = upstream(sc[~is_ood], sc[is_ood])
                            met[name] = {"AUROC": 100 * m["AUROC"], "FPR95": 100 * m["FPR95"]}
                runs[(d, s, sd)] = met
    return runs


def rep_e1():
    res, L = {}, ["# E1: is the gain due to the extra DINOv2 features?", ""]
    for dev in ("dev1", "dev2"):
        ev = load_eval(dev)
        e1 = load_eval(dev, "e1")
        full = load_eval(dev, "e1full")
        ext = load_eval(dev, "eval_ext")
        single = l14_single_from_npz(dev)
        runs = {k: {**ev.get(k, {}), **e1.get(k, {}), **full.get(k, {}), **single.get(k, {}), **ext.get(k, {})}
                for k in ev}
        keys = sorted({k for r in runs.values() for k in r})
        # hyper-parameters of the E1 baselines chosen on dev1 (x TINS and alone separately), frozen for dev2
        if dev == "dev1":
            sel = {}
            for fam in ("adaneg", "oodd", "knnF", "mahaF", "protoF", "knn", "maha"):
                for pre in ("s", "v", "sL14", "vL14"):
                    cands = [k for k in keys if k.startswith(f"{pre}_{fam}") and
                             (fam not in ("knn", "maha") or pre in ("sL14", "vL14"))]
                    if fam == "knn":
                        cands = [k for k in cands if not k.startswith(f"{pre}_knnF")]
                    if fam == "maha":
                        cands = [k for k in cands if not k.startswith(f"{pre}_mahaF")]
                    sc = {k: np.mean(per_draw(runs, k, "near", "FPR95")) for k in cands
                          if len(per_draw(runs, k, "near", "FPR95")) == 5}
                    if sc:
                        sel[f"{pre}_{fam}"] = min(sc, key=sc.get)
            res["selection_dev1"] = sel
        sel = res["selection_dev1"]
        show = ["tins", "s_full", "v_full", "s_full_CLIP", "v_full_CLIP", "s_static_CLIP", "s_mem_CLIP", "s_lp_CLIP",
                "s_full_L14", "v_full_L14"] + list(sel.values())
        L += [f"## {dev}", "", "| variant | near FPR95 | near AUROC | far FPR95 | far AUROC |", "|---|---|---|---|---|"]
        res[dev] = {"table": {}}
        for k in show:
            if not per_draw(runs, k, "near", "FPR95"):
                continue
            row = {f"{s}_{m}": float(np.mean(per_draw(runs, k, s, m))) for s in STREAMS for m in ("FPR95", "AUROC")}
            res[dev]["table"][k] = row
            L.append(f"| {k} | {row['near_FPR95']:.2f} | {row['near_AUROC']:.2f} | {row['far_FPR95']:.2f} | {row['far_AUROC']:.2f} |")
        g2 = np.array(per_draw_diff(runs, "tins", "s_full", "near", "FPR95"))
        gc = np.array(per_draw_diff(runs, "tins", "s_full_CLIP", "near", "FPR95"))
        ratio = gc / g2
        res[dev]["gain_over_tins_2view"] = ci(g2.tolist())
        res[dev]["gain_over_tins_clip"] = ci(gc.tolist())
        res[dev]["retained_ratio"] = ci(ratio.tolist())
        L += ["", f"near FPR95 gain over TINS: 2-view REPRISE {fmt(ci(g2.tolist()))}; REPRISE-CLIP {fmt(ci(gc.tolist()))}; "
              f"retained ratio {fmt(ci(ratio.tolist()), 3)} (rule: >= 0.5 keeps the method paper)", ""]
        L += ["| comparison | near FPR95 | near AUROC | far FPR95 | far AUROC |", "|---|---|---|---|---|"]
        res[dev]["compare"] = {}
        for a, b in [("s_full", "s_full_CLIP"), ("s_full_CLIP", "tins"), ("s_full_L14", "s_full")] + \
                    [("s_full", v) for v in sel.values() if v.startswith("s")] + \
                    [("v_full", v) for v in sel.values() if v.startswith("v")]:
            c = {f"{s}_{m}": ci(per_draw_diff(runs, a, b, s, m)) for s in STREAMS for m in ("FPR95", "AUROC")}
            if c["near_FPR95"]["n"] < 2:
                continue
            res[dev]["compare"][f"{a} - {b}"] = c
            L.append(f"| {a} - {b} | {fmt(c['near_FPR95'])} | {fmt(c['near_AUROC'])} | {fmt(c['far_FPR95'])} | {fmt(c['far_AUROC'])} |")
        L.append("")
    write("e1", res, L)


# ---------------------------------------------------------------------------------------------------------------
def rep_streams():
    specs = json.loads((r5.R5 / "stream_specs.json").read_text())
    res, L = {}, ["# Streams (E2 ratio / class x count, E3 temporal, E4 delayed, E5-like, batch size); one order per draw", ""]
    for dev in ("dev1", "dev2"):
        L += [f"## {dev}", "", "| stream | OOD | s_lp near-type FPR95 | s_full FPR95 | full - lp FPR95 [95% CI] | full - lp AUROC | "
              "full - single FPR95 | ID adm. % | OOD adm. % | Pr(pt3<=0.1 \\| ID) | final memory ID share |",
              "|---|---|---|---|---|---|---|---|---|---|---|"]
        res[dev] = {}
        for sp in [s for s in specs if s["dev"] == dev]:
            d_ = r5.R5 / dev / "streams" / sp["name"]
            files = {d: d_ / f"eval_draw{d}.json" for d in DRAWS}
            files = {d: f for d, f in files.items() if f.exists()}
            if len(files) < 2:
                continue
            blobs = {d: json.loads(f.read_text()) for d, f in files.items()}
            runs = {(d, "all", 0): b["metrics"] for d, b in blobs.items()}
            dr = list(blobs)

            def diff(a, b, m):
                return ci(per_draw_diff(runs, a, b, "all", m, draws=dr, seeds=[0]))

            row = {"n_draws": len(dr), "n": blobs[dr[0]]["n"], "n_ood": blobs[dr[0]]["n_ood"]}
            for k in ("tins", "s_static", "s_mem", "s_lp", "s_single_lp", "s_full", "v_full", "v_lp"):
                for m in ("FPR95", "AUROC"):
                    row[f"{k}_{m}"] = float(np.mean(per_draw(runs, k, "all", m, draws=dr, seeds=[0])))
            row["full_minus_lp"] = {m: diff("s_full", "s_lp", m) for m in ("FPR95", "AUROC")}
            row["full_minus_single"] = {m: diff("s_full", "s_single_lp", m) for m in ("FPR95", "AUROC")}
            lg = [b["logs"] for b in blobs.values()]
            row["id_admission"] = 100 * float(np.mean([np.mean([x[v]["id_admission"] for v in ("B14", "L14")]) for x in lg]))
            row["ood_admission"] = 100 * float(np.mean([np.mean([x[v]["ood_admission"] for v in ("B14", "L14")]) for x in lg]))
            row["pr_pt3_le_0.1_id"] = float(np.mean([np.mean([x[v]["pr_le_0.1"]["pt3"] for v in ("B14", "L14")]) for x in lg]))
            row["pr_le_0.1_id"] = {k: float(np.mean([np.mean([x[v]["pr_le_0.1"][k] for v in ("B14", "L14")]) for x in lg]))
                                   for k in ("p", "pt3", "pt1", "plp")}
            row["id_adm_per_class_q"] = np.mean([[np.mean([x[v]["id_admission_per_class_q"][i] for v in ("B14", "L14")])
                                                  for i in range(4)] for x in lg], axis=0).tolist()
            share = [np.mean([x[v]["memory_id_share_by_batch"][-1] for v in ("B14", "L14")]) for x in lg]
            row["final_memory_id_share"] = float(np.mean(share))
            curves = [np.mean([x[v]["memory_id_share_by_batch"] for v in ("B14", "L14")], axis=0) for x in lg]
            nb = min(len(c) for c in curves)
            row["memory_id_share_curve_deciles"] = [float(np.mean([c[min(nb - 1, int(q * nb))] for c in curves]))
                                                    for q in np.linspace(0.1, 1.0, 10)]
            res[dev][sp["name"]] = row
            L.append(f"| {sp['name']} | {row['n_ood']}/{row['n']} | {row['s_lp_FPR95']:.2f} | {row['s_full_FPR95']:.2f} | "
                     f"{fmt(row['full_minus_lp']['FPR95'])} | {fmt(row['full_minus_lp']['AUROC'])} | "
                     f"{fmt(row['full_minus_single']['FPR95'])} | {row['id_admission']:.1f} | {row['ood_admission']:.1f} | "
                     f"{row['pr_pt3_le_0.1_id']:.3f} | {row['final_memory_id_share']:.3f} |")
        L.append("")
    write("streams", res, L)


# ---------------------------------------------------------------------------------------------------------------
def rep_mods(what="m1m3e5"):
    res, L = {}, [f"# {what}: variants vs v4 (s_full) under the same 15 conditions", ""]
    devs = ("dev1", "dev2") if what == "m1m3e5" else ("dev1",)
    for dev in devs:
        runs = load_eval(dev, f"mods/{what}")
        base = load_eval(dev)
        if not runs:
            continue
        keys = sorted({k for r in runs.values() for k in r})
        ref = "s_full"
        L += [f"## {dev}", "", "| variant | near FPR95 | near AUROC | far FPR95 | far AUROC | "
              "vs v4 near FPR95 [95% CI] | vs v4 near AUROC | vs v4 far FPR95 | vs v4 far AUROC |", "|---|---|---|---|---|---|---|---|---|"]
        res[dev] = {}
        for k in keys:
            if not per_draw(runs, k, "near", "FPR95"):
                continue
            row = {f"{s}_{m}": float(np.mean(per_draw(runs, k, s, m))) for s in STREAMS for m in ("FPR95", "AUROC")}
            refkey = "v_full" if (k.startswith("v_") or "_vis_" in k) else ref
            c = {f"{s}_{m}": ci(per_draw_diff(runs, k, refkey, s, m, runs_b=base)) for s in STREAMS for m in ("FPR95", "AUROC")}
            row["vs_v4"] = c
            res[dev][k] = row
            if c["near_FPR95"]["n"] < 2:
                continue
            L.append(f"| {k} | {row['near_FPR95']:.2f} | {row['near_AUROC']:.2f} | {row['far_FPR95']:.2f} | {row['far_AUROC']:.2f} | "
                     f"{fmt(c['near_FPR95'])} | {fmt(c['near_AUROC'])} | {fmt(c['far_FPR95'])} | {fmt(c['far_AUROC'])} |")
        L.append("")
    write(what, res, L)


# ---------------------------------------------------------------------------------------------------------------
def rep_e4():
    """Recurrence vs position on the matched dev near streams: OOD arrivals stratified by the occurrence index of their
    class in the stream and by position decile; detection = share of the stratum below the ID 95% acceptance threshold
    of the same run (TNR at TPR 95), for TINS, static, LP only, memory only and REPRISE (x TINS)."""
    from vins import config as C  # noqa: F401

    res, L = {}, ["# E4: recurrence vs position (dev near streams, 15 conditions per split)", ""]
    occ_bins = [(1, 1), (2, 5), (6, 20), (21, 50)]
    variants = {"tins": None, "s_static": "p", "s_lp": "plp", "s_mem": "pt3", "s_full": ("pt3", "plp")}
    for dev in ("dev1", "dev2"):
        work = BASE if dev == "dev1" else BASE / "dev2"
        samples = pd.read_parquet(work / "splits" / "samples.parquet").set_index("sample_id")
        acc = {}
        for d in DRAWS:
            for sd in SEEDS:
                e = np.load(r5.R5 / dev / "eval" / f"draw{d}" / f"near_seed{sd}.npz", allow_pickle=True)
                z = np.load(r5.R5 / dev / "tins" / f"draw{d}" / f"near_seed{sd}.npz", allow_pickle=True)
                sid, is_ood, S = e["sample_id"], e["is_ood"].astype(bool), z["S_final"].astype(np.float64)
                wn = samples.loc[sid].wnid.values
                occ = np.zeros(len(sid), dtype=int)
                seen = {}
                for i in np.flatnonzero(is_ood):
                    seen[wn[i]] = seen.get(wn[i], 0) + 1
                    occ[i] = seen[wn[i]]
                pos_dec = np.minimum(9, (np.arange(len(sid)) * 10) // len(sid))
                for name, comp in variants.items():
                    if comp is None:
                        sc = S
                    elif isinstance(comp, tuple):
                        sc = S * np.prod([e[f"{c}_{v}"].astype(np.float64) for c in comp for v in ("B14", "L14")], axis=0)
                    else:
                        sc = S * e[f"{comp}_B14"].astype(np.float64) * e[f"{comp}_L14"].astype(np.float64)
                    thr = np.quantile(sc[~is_ood], 0.05)                # accept 95% of ID
                    det = sc < thr
                    for lo, hi in occ_bins:
                        mk = is_ood & (occ >= lo) & (occ <= hi)
                        acc.setdefault((name, f"occ{lo}-{hi}"), {}).setdefault(d, []).append(float(det[mk].mean()))
                    for q in range(10):
                        mk = is_ood & (pos_dec == q)
                        acc.setdefault((name, f"pos{q}"), {}).setdefault(d, []).append(float(det[mk].mean()))
                    for q in range(10):                                  # first occurrences only, by position
                        mk = is_ood & (pos_dec == q) & (occ == 1)
                        if mk.any():
                            acc.setdefault((name, f"first_pos{q}"), {}).setdefault(d, []).append(float(det[mk].mean()))
        tab = {}
        for (name, stratum), per in acc.items():
            vals = [float(np.mean(v)) for v in per.values() if len(v) == len(SEEDS)]
            tab.setdefault(name, {})[stratum] = 100 * float(np.mean(vals)) if vals else float("nan")
        res[dev] = tab
        strata = [f"occ{lo}-{hi}" for lo, hi in occ_bins]
        L += [f"## {dev}: OOD detected at ID TPR 95% (%), by occurrence index of the class", "",
              "| variant | " + " | ".join(strata) + " |", "|---" * (len(strata) + 1) + "|"]
        for name in variants:
            L.append(f"| {name} | " + " | ".join(f"{tab[name][s]:.1f}" for s in strata) + " |")
        L += ["", f"## {dev}: by stream position decile (all OOD / first occurrences only)", "",
              "| variant | " + " | ".join(f"d{q + 1}" for q in range(10)) + " |", "|---" * 11 + "|"]
        for name in variants:
            L.append(f"| {name} | " + " | ".join(f"{tab[name][f'pos{q}']:.1f}" for q in range(10)) + " |")
            L.append(f"| {name} (1st) | " + " | ".join(f"{tab[name].get(f'first_pos{q}', float('nan')):.1f}" for q in range(10)) + " |")
        L.append("")
    write("e4", res, L)


# ---------------------------------------------------------------------------------------------------------------
def rep_decide():
    """Apply the pre-registered Phase 2 rule (r5/prereg_phase2.json) to the dev results."""
    rule = json.loads((r5.R5 / "prereg_phase2.json").read_text())
    cands = [f"m1_bh_{w}_{q}_full" for w in ("all", "last") for q in ("0.05", "0.1", "0.2")]
    cands += ["m2a_split1_full", "m3_product", "m3_cauchy", "m3_hmp", "m3_withinview_product", "m3_withinview_cauchy",
              "m4_full_x_maha", "E3m1@0.1"]
    res, L = {"rule_written_utc": rule["written_utc"]}, ["# Phase 2 decision (pre-registered rule r5/prereg_phase2.json)", "",
                                                         "| candidate | dev | near FPR95 vs v4 [95% CI] | far FPR95 vs v4 (mean) | near AUROC vs v4 |",
                                                         "|---|---|---|---|---|"]
    table = {}
    for dev in ("dev1", "dev2"):
        base = load_eval(dev)
        mods = load_eval(dev, "mods/m1m3e5")
        ent = {}
        for d in DRAWS:
            for s in STREAMS:
                for sd in SEEDS:
                    f = r5.R5 / "entrance" / "eval" / f"{dev}_draw{d}_{s}_seed{sd}.json"
                    if f.exists():
                        ent[(d, s, sd)] = {"E3m1@0.1": json.loads(f.read_text())["E3m1@0.1"]["metrics"]["s_full"]}
        for c in cands:
            src = ent if c.startswith("E3") else mods
            dn = ci(per_draw_diff(src, c, "s_full", "near", "FPR95", runs_b=base))
            df = ci(per_draw_diff(src, c, "s_full", "far", "FPR95", runs_b=base))
            da = ci(per_draw_diff(src, c, "s_full", "near", "AUROC", runs_b=base))
            if dn["n"] < 5:
                continue
            table[(c, dev)] = {"near_fpr": dn, "far_fpr": df, "near_auroc": da,
                               "near_fpr_level": float(np.mean(per_draw(src, c, "near", "FPR95")))}
            L.append(f"| {c} | {dev} | {fmt(dn)} | {df['mean']:+.2f} | {fmt(da)} |")
    ok1 = [c for c in cands if (c, "dev1") in table and table[(c, "dev1")]["far_fpr"]["mean"] <= 0.5]
    chosen = min(ok1, key=lambda c: table[(c, "dev1")]["near_fpr_level"]) if ok1 else None
    verdict = "v5 not defined (Phase 3 evaluates v4)"
    if chosen is not None and table[(chosen, "dev1")]["near_fpr"]["hi"] < 0:
        t2 = table.get((chosen, "dev2"))
        if t2 and t2["near_fpr"]["hi"] < 0 and t2["far_fpr"]["mean"] <= 0.5:
            verdict = f"v5 = v4 + {chosen}"
        else:
            verdict = f"{chosen} selected on dev1 but not confirmed on dev2 -> v5 not defined"
    elif chosen is not None:
        verdict = f"best candidate {chosen} does not beat v4 on dev1 -> v5 not defined"
    res.update({"table": {f"{c}|{d}": v for (c, d), v in table.items()}, "chosen_dev1": chosen, "verdict": verdict})
    L += ["", f"dev1 selection: {chosen}", f"verdict: {verdict}"]
    write("decide", res, L)


if __name__ == "__main__":
    what = sys.argv[1]
    {"maha": rep_maha, "entrance": rep_entrance, "inject": rep_inject, "e1": rep_e1, "streams": rep_streams,
     "mods": lambda: rep_mods("m1m3e5"), "sens": lambda: rep_mods("sens"), "e4": rep_e4,
     "decide": rep_decide}[what]()
