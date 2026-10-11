"""Phase 14 analysis -> P14/results/p14_summary.json and p14_table.txt.
test_train: benchmark protocol, means over the 6 draws per data set and per group (bases: over the stream orders first), 95% t
            intervals over the draws, paired intervals against the Phase 12 read-out (same base), against p+ only (same base) and
            against the standalone row of the same configuration; seeds / realised ID fraction per fold.
test_val:   exchangeable calibration (3 splits): validity of p_final (false-alarm rate at alpha, FDP and recall of BH), metrics.
rec:        the grid: FPR95 per configuration per (data set, c, r), means over the replicates.
in200:      ImageNet-200: means over the draws per group, intervals, validity.
views:      backbone families: per view and configuration, means over the data sets of a group.
cifar:      CIFAR-100: per view set and configuration, means over the draws per data set and group, intervals, validity.
online:     arrival-time cross-fit: means over the stream orders per group, validity, iteration counts."""
import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd

import p14common as P
from an11 import tci

OO_NEAR, OO_FAR, FOUR = ["ssb_hard", "ninco"], ["inaturalist", "textures", "openimageo"], ["inat", "sun", "places", "dtd"]
GROUPS = {"openood": {"near": OO_NEAR, "far": OO_FAR}, "fourood": {"fourood": FOUR}}
CIFAR_GROUPS = {"near": ["cifar10", "tin"], "far": ["mnist", "svhn", "texture", "places365"]}
VAL_KEYS = ("fpr@0.01", "fpr@0.05", "fpr@0.1", "fpr@0.2", "tpr@0.05", "bh_fdp", "bh_recall", "bh_rejections")


def load_rows(pattern):
    files = sorted(glob.glob(pattern))
    return pd.concat([pd.read_csv(f) for f in files], ignore_index=True) if files else None


def ci_or_none(x):
    x = np.asarray(x, float)
    return tci(x) if len(x) >= 2 else {"mean": float(x.mean()), "lo": None, "hi": None, "n": int(len(x))}


def group_stats(d, unit, groups, with_pairs=True):
    """d: rows with columns key (cfg|base), ds, <unit>, AUROC, FPR95 (bases already averaged over seeds by the caller).
    Returns {"per_dataset": {key|ds: ...}, group: {key: {AUROC, FPR95, FPR95_ci, n, dFPR95_vs_v1, dFPR95_vs_lp, dFPR95_vs_alone}}}."""
    per = d.groupby(["key", unit, "ds"])[["AUROC", "FPR95"]].mean()
    res = {"per_dataset": {f"{k}|{ds}": {"AUROC": float(v.AUROC), "FPR95": float(v.FPR95)} for (k, ds), v in per.groupby(level=[0, 2]).mean().iterrows()},
           "per_dataset_ci": {f"{k}|{ds}": {"AUROC_ci": ci_or_none(v.AUROC.values), "FPR95_ci": ci_or_none(v.FPR95.values)}
                              for (k, ds), v in per.groupby(level=[0, 2]) if len(v) >= 2}}
    for g, sets in groups.items():
        pu = per[per.index.get_level_values(2).isin(sets)].groupby(level=[0, 1]).mean()
        gres = {}
        keys = pu.index.get_level_values(0).unique()
        for k in keys:
            v = pu.loc[k]
            e = {"AUROC": float(v.AUROC.mean()), "FPR95": float(v.FPR95.mean()), "n": int(len(v)), "FPR95_ci": ci_or_none(v.FPR95.values), "AUROC_ci": ci_or_none(v.AUROC.values)}
            if with_pairs:
                cfg, base = k.rsplit("|", 1)
                refs = {"v1": f"v1|storey|q0.2|pp|{base}", "lp": (f"xfit|lp|pooled|{base}" if "|pooled" in cfg else f"xfit|lp|{base}"),
                        "alone": f"{cfg}|none" if base != "none" else None}
                for tag, rk in refs.items():
                    if rk and rk in keys and rk != k:
                        r = pu.loc[rk]
                        common = v.index.intersection(r.index)
                        if len(common) >= 2:
                            e[f"dFPR95_vs_{tag}"] = tci((v.loc[common].FPR95 - r.loc[common].FPR95).values)
                            e[f"dAUROC_vs_{tag}"] = tci((v.loc[common].AUROC - r.loc[common].AUROC).values)
            gres[k] = e
        res[g] = gres
    return res


def validity_stats(d, groups, by="ds"):
    """Means of the validity numbers of the p_final rows per configuration, per data set and per group (over the units)."""
    pf = d[d["fpr@0.05"].notna()] if "fpr@0.05" in d else d.iloc[0:0]
    out = {"per_dataset": {}, **{g: {} for g in groups}}
    if len(pf) == 0:
        return out
    cols = [k for k in VAL_KEYS if k in pf]
    per_ds = pf.groupby(["cfg", by])[cols].mean()
    for (c, ds), v in per_ds.iterrows():
        out["per_dataset"][f"{c}|{ds}"] = {k: float(v[k]) for k in cols}
    for g, sets in groups.items():
        sub = pf[pf[by].isin(sets)]
        for c, v in sub.groupby("cfg")[cols].mean().iterrows():
            out[g][c] = {k: float(v[k]) for k in cols}
            out[g][c]["fpr@0.05_max"] = float(sub[sub.cfg == c]["fpr@0.05"].max())
            out[g][c]["bh_fdp_max"] = float(sub[sub.cfg == c]["bh_fdp"].max())
    return out


def seeds_from_meta(pattern, unit_key):
    acc = {}
    for f in sorted(glob.glob(pattern)):
        m = json.loads(Path(f).read_text())
        ds = m.get("ds") or Path(f).name.replace(".meta.json", "")        # the data set is the file name (score_all's meta has no tag)
        for v, mv in m["views"].items():
            for fk, e in mv.items():
                if not fk.startswith("fold"):
                    continue
                _, rule, qk = fk.split("|")
                acc.setdefault(f"{ds}|{v}|{rule}|{qk}", []).append((e["seeds"], e["pi0_hat"], np.nan if e["id_frac"] is None else e["id_frac"], e["n_test"]))
    out = {}
    for k, L in acc.items():
        a = np.array(L, float)
        out[k] = {"seeds": float(a[:, 0].mean()), "pi0_hat": float(a[:, 1].mean()), "id_frac": (float(np.nanmean(a[:, 2])) if np.isfinite(a[:, 2]).any() else None),
                  "id_frac_max": (float(np.nanmax(a[:, 2])) if np.isfinite(a[:, 2]).any() else None), "seeds_over_n_test": float((a[:, 0] / a[:, 3]).mean()), "n": int(len(L))}
    return out


def test_train_summary():
    d = load_rows(str(P.RESULTS14 / "test_train" / "*" / "*" / "*.csv"))
    if d is None or len(d) == 0:
        return None
    d["draw"] = d.draw.astype(str)
    d["key"] = d.cfg + "|" + d.base
    out = {"n_draws": int(d.draw.nunique()), "bases": sorted(d.base.unique())}
    for part, groups in GROUPS.items():
        dd = d[d.part == part]
        if len(dd) == 0:
            continue
        dd = dd.groupby(["key", "cfg", "base", "draw", "ds"], as_index=False)[["AUROC", "FPR95"]].mean()      # bases: over the orders first
        out[part] = group_stats(dd, "draw", groups)
        out[part]["validity"] = validity_stats(d[d.part == part], groups)
        out[part]["seeds"] = seeds_from_meta(str(P.RESULTS14 / "test_train" / part / "*" / "*.meta.json"), "draw")
    return out


def test_val_summary():
    d = load_rows(str(P.RESULTS14 / "test_val" / "*" / "*" / "*.csv"))
    if d is None or len(d) == 0:
        return None
    d["key"] = d.cfg + "|" + d.base
    out = {"n_splits": int(d.split.nunique())}
    for part, groups in GROUPS.items():
        dd = d[d.part == part]
        if len(dd) == 0:
            continue
        m = dd.groupby(["key", "cfg", "base", "split", "ds"], as_index=False)[["AUROC", "FPR95"]].mean()
        out[part] = {"metrics": group_stats(m, "split", groups), "validity": validity_stats(dd, groups),
                     "seeds": seeds_from_meta(str(P.RESULTS14 / "test_val" / part / "*" / "*.meta.json"), "split")}
    return out


def rec_summary():
    d = load_rows(str(P.RESULTS14 / "rec" / "*" / "*.csv"))
    if d is None or len(d) == 0:
        return None
    d["key"] = d.cfg + "|" + d.base
    out = {}
    for (ds, c, r), g in d.groupby(["ds", "c", "r"]):
        e = {"n_reps": int(g.rep.nunique()), "n_ood": int(g.n_ood.iloc[0]), "mixing_rate": float(g.mixing_rate.iloc[0]), "cfg": {}}
        for k, gg in g.groupby("key"):
            e["cfg"][k] = {"AUROC": float(gg.AUROC.mean()), "FPR95": float(gg.FPR95.mean())}
            if k.endswith("|none") and "fpr@0.05" in gg and gg["fpr@0.05"].notna().any():
                e["cfg"][k].update({x: float(gg[x].mean()) for x in ("fpr@0.05", "bh_fdp", "bh_recall") if x in gg})
        out[f"{ds}|c{c}|r{r}"] = e
    # seeds per view (summed over the two folds, mean over the replicates): seeds, unknown images among them, ID fraction
    for f in sorted(glob.glob(str(P.RESULTS14 / "rec" / "*" / "*.meta.json"))):
        m = json.loads(Path(f).read_text())
        stem = Path(f).name.replace(".meta.json", "")                      # c<c>_r<r>_rep<i>
        c_, r_ = stem.split("_")[0][1:], stem.split("_")[1][1:]
        key = f"{Path(f).parent.name}|c{c_}|r{r_}"
        if key not in out:
            continue
        for v, mv in m["views"].items():
            for (rule, qk) in (("storey", "q0.2"), ("bh", "q0.2")):
                tot = sum(mv[fk]["seeds"] for fk in mv if fk.startswith("fold") and fk.endswith(f"|{rule}|{qk}"))
                ood = sum(mv[fk]["seeds"] * (1.0 - (mv[fk]["id_frac"] if mv[fk]["id_frac"] is not None else 1.0)) for fk in mv if fk.startswith("fold") and fk.endswith(f"|{rule}|{qk}"))
                acc = out[key].setdefault(f"view_{v}", {}).setdefault(f"{rule}_{qk}", {"seeds": [], "ood_seeds": [], "n_with_seeds": 0})
                acc["seeds"].append(tot)
                acc["ood_seeds"].append(ood)
                acc["n_with_seeds"] += int(tot > 0)
    for key, e in out.items():
        for vk in [k for k in e if k.startswith("view_")]:
            for rk, acc in e[vk].items():
                e[vk][rk] = {"seeds": float(np.mean(acc["seeds"])), "ood_seeds": float(np.mean(acc["ood_seeds"])),
                             "id_frac": (float(1.0 - np.sum(acc["ood_seeds"]) / np.sum(acc["seeds"])) if np.sum(acc["seeds"]) > 0 else None),
                             "n_reps_with_seeds": int(acc["n_with_seeds"]), "n_reps": len(acc["seeds"])}
    return out


def in200_summary():
    d = load_rows(str(P.RESULTS14 / "in200" / "*" / "*.csv"))
    if d is None or len(d) == 0:
        return None
    d["draw"] = d.draw.astype(str)
    d["key"] = d.cfg + "|" + d.base
    groups = {"near": OO_NEAR, "far": OO_FAR}
    out = {"n_draws": int(d.draw.nunique()), **group_stats(d, "draw", groups),
           "validity": validity_stats(d, groups), "seeds": seeds_from_meta(str(P.RESULTS14 / "in200" / "*" / "*.meta.json"), "draw")}
    out["sizes"] = {ds: {"n_id": int(g.n_id.iloc[0]), "n_ood": int(g.n_ood.iloc[0]), "n_cal": int(g.n_cal.iloc[0])} for ds, g in d.groupby("ds")}
    return out


def views_summary():
    d = load_rows(str(P.RESULTS14 / "views" / "*" / "*.csv"))
    if d is None or len(d) == 0:
        return None
    d["key"] = d.cfg + "|" + d.base
    out = {"views": sorted(d.view.unique()), "dim": {v: int(x) for v, x in d.groupby("view").dim.first().items()}}
    for v, dv in d.groupby("view"):
        dd = dv.groupby(["key", "cfg", "base", "ds"], as_index=False)[["AUROC", "FPR95"]].mean()
        dd["unit"] = 0
        out[v] = {g: {k: {"AUROC": float(e.AUROC), "FPR95": float(e.FPR95)} for k, e in dd[dd.ds.isin(sets)].groupby("key")[["AUROC", "FPR95"]].mean().iterrows()}
                  for g, sets in GROUPS["openood"].items()}
        out[v]["per_dataset"] = {f"{k}|{ds}": {"AUROC": float(e.AUROC), "FPR95": float(e.FPR95)} for (k, ds), e in dd.groupby(["key", "ds"])[["AUROC", "FPR95"]].mean().iterrows()}
        out[v]["validity"] = validity_stats(dv, GROUPS["openood"])
    return out


def cifar_summary():
    d = load_rows(str(P.RESULTS14 / "cifar" / "*" / "*.csv"))
    if d is None or len(d) == 0:
        return None
    d["draw"] = d.draw.astype(str)
    d["key"] = d.views + "|" + d.cfg + "|" + d.base
    m = d
    out = {"n_draws": int(d.draw.nunique()), **group_stats(m, "draw", CIFAR_GROUPS, with_pairs=False)}
    # paired intervals (over the draws) for the two-view rows against the Phase 12 read-out and p+ only
    per = m.groupby(["key", "draw", "ds"])[["AUROC", "FPR95"]].mean()
    for g, sets in CIFAR_GROUPS.items():
        pu = per[per.index.get_level_values(2).isin(sets)].groupby(level=[0, 1]).mean()
        keys = pu.index.get_level_values(0).unique()
        for k in keys:
            vs, rest = k.split("|", 1)
            for tag, rk in (("v1", f"{vs}|v1|storey|q0.2|pp|none"), ("lp", f"{vs}|xfit|lp|none")):
                if rk in keys and rk != k:
                    v, r = pu.loc[k], pu.loc[rk]
                    common = v.index.intersection(r.index)
                    if len(common) >= 2:
                        out[g][k][f"dFPR95_vs_{tag}"] = tci((v.loc[common].FPR95 - r.loc[common].FPR95).values)
    dv = d.copy()
    dv["cfg"] = dv.views + "|" + dv.cfg
    out["validity"] = validity_stats(dv, CIFAR_GROUPS)
    # seeds per data set, view and rule (two-view set, both folds, all draws) and the sizes of the test sets
    acc, sizes = {}, {}
    for f in sorted(glob.glob(str(P.RESULTS14 / "cifar" / "*" / "*.meta.json"))):
        m = json.loads(Path(f).read_text())
        ds = m["ds"]
        sizes[ds] = {"n_id": int(m["n_id"]), "n_ood": int(m["n_ood"]), "mixing": m["n_ood"] / (m["n_ood"] + m["n_id"])}
        for view, dd in m["views"]["L14+D3L"]["views"].items():
            for fk, e in dd.items():
                if fk.startswith("fold"):
                    _, rule, qk = fk.split("|")
                    acc.setdefault(f"{ds}|{view}|{rule}|{qk}", []).append((e["seeds"], e["pi0_hat"], np.nan if e["id_frac"] is None else e["id_frac"], e["n_test"]))
    out["seeds"] = {k: {"seeds": float(a[:, 0].mean()), "pi0_hat": float(a[:, 1].mean()), "id_frac": (float(np.nanmean(a[:, 2])) if np.isfinite(a[:, 2]).any() else None),
                        "seeds_over_n_test": float((a[:, 0] / a[:, 3]).mean()), "n": int(len(a))} for k, a in ((k, np.array(L, float)) for k, L in acc.items())}
    out["sizes"] = sizes
    return out


def online_summary():
    d = load_rows(str(P.RESULTS14 / "online" / "*" / "*.csv"))
    if d is None or len(d) == 0:
        return None
    d["ds"] = d.stream.str.replace(r"_seed\d+$", "", regex=True)
    d["seed"] = d.stream.str.extract(r"seed(\d+)$", expand=False).astype(int)
    d["key"] = d.cfg + "|" + d.base
    out = {}
    for part, groups in GROUPS.items():
        dd = d[d.part == part]
        if len(dd) == 0:
            continue
        res = group_stats(dd, "seed", groups, with_pairs=False)
        res["validity"] = validity_stats(dd, groups)
        out[part] = res
    metas = [json.loads(Path(f).read_text()) for f in sorted(glob.glob(str(P.RESULTS14 / "online" / "*" / "*.meta.json")))]
    if metas:
        out["iters_mean"] = float(np.mean([m["views"][v][k] for m in metas for v in m["views"] for k in ("iters_pos_mean", "iters_neg_mean")]))
        out["iters_max"] = int(max(m["views"][v][k] for m in metas for v in m["views"] for k in ("iters_pos_max", "iters_neg_max")))
        out["seconds_per_stream_mean"] = float(np.mean([m["seconds"] for m in metas]))
        out["selected"] = metas[0].get("selected")
    return out


def fmt_ci(e):
    c = e.get("FPR95_ci") or {}
    return f"[{c['lo']:.2f}, {c['hi']:.2f}]" if c.get("lo") is not None else ""


def fmt_d(e, tag):
    dd = e.get(f"dFPR95_vs_{tag}")
    return f"d{tag} {dd['mean']:+.2f} [{dd['lo']:+.2f},{dd['hi']:+.2f}]" if dd else ""


def main():
    sel_file = P.RESULTS14 / "sel" / "sel14_summary.json"
    out = {"selection": (json.loads(sel_file.read_text())["_selection"] if sel_file.exists() else None),
           "test_train": test_train_summary(), "test_val": test_val_summary(), "rec": rec_summary(), "in200": in200_summary(),
           "views": views_summary(), "cifar": cifar_summary(), "online": online_summary(), "_meta": {"utc": P.utc()}}
    P.write_json(P.RESULTS14 / "p14_summary.json", out)
    L = []
    if out["selection"]:
        L.append(f"== selection: {out['selection']['chosen']} (feasible {out['selection']['n_feasible']} of {out['selection']['n_candidates']}) ==")
        L.append("")
    if out["test_train"]:
        L.append(f"== benchmarks, benchmark calibration (means over {out['test_train']['n_draws']} draws; FPR95 95% t; paired d vs v1 / p+ only / standalone) ==")
        for part in ("openood", "fourood"):
            if part in out["test_train"]:
                for g in GROUPS[part]:
                    for k, e in sorted(out["test_train"][part][g].items(), key=lambda ke: ke[1]["FPR95"]):
                        L.append(f"{part:8s} {g:8s} {k:34s} {e['AUROC']:6.2f} / {e['FPR95']:6.2f} {fmt_ci(e):16s} {fmt_d(e, 'v1'):28s} {fmt_d(e, 'lp'):28s} {fmt_d(e, 'alone')}")
        L.append("")
    if out["test_val"]:
        L.append("== exchangeable calibration: validity of p_final (means over splits and data sets; max over data sets) ==")
        for part in ("openood", "fourood"):
            if part in out["test_val"]:
                for g in GROUPS[part]:
                    for c, e in sorted(out["test_val"][part]["validity"][g].items()):
                        L.append(f"{part:8s} {g:8s} {c:34s} " + " ".join(f"{k}={e[k]:.3f}" for k in ("fpr@0.01", "fpr@0.05", "fpr@0.1", "fpr@0.2", "fpr@0.05_max", "bh_fdp", "bh_fdp_max", "bh_recall") if k in e))
                for g in GROUPS[part]:
                    for k, e in sorted(out["test_val"][part]["metrics"][g].items(), key=lambda ke: ke[1]["FPR95"])[:12]:
                        L.append(f"val {part:8s} {g:8s} {k:34s} {e['AUROC']:6.2f} / {e['FPR95']:6.2f}")
        L.append("")
    if out["rec"]:
        sel = out["selection"]["chosen"] if out["selection"] else None
        L.append(f"== grid: FPR95 (L14+D3L standalone) static | p+ | v1 p+p- | xfit storey pp | xfit bh pp | selected ({sel}) | selected pfinal fpr@.05 ==")
        for key, e in out["rec"].items():
            cf = e["cfg"]
            f = lambda k: (f"{cf[k]['FPR95']:5.1f}" if k in cf else "  -  ")
            selk = sel if sel in cf else None
            pf = cf.get(sel or "", {})
            L.append(f"{key:22s} {e['n_ood']:6d} {100 * e['mixing_rate']:5.2f}% | {f('xfit|static|none')} | {f('xfit|lp|none')} | {f('v1|storey|q0.2|pp|none')} | {f('xfit|storey|q0.2|pp|none')} | {f('xfit|bh|q0.2|pp|none')} | {f(selk) if selk else '  -  '} | {pf.get('fpr@0.05', float('nan')):.3f}")
        L.append("")
    if out["in200"]:
        L.append("== ImageNet-200 (means over draws) ==")
        for g in ("near", "far"):
            for k, e in sorted(out["in200"][g].items(), key=lambda ke: ke[1]["FPR95"]):
                L.append(f"in200 {g:5s} {k:34s} {e['AUROC']:6.2f} / {e['FPR95']:6.2f} {fmt_ci(e)}")
            for c, e in sorted(out["in200"]["validity"][g].items()):
                L.append(f"in200 {g:5s} {c:34s} fpr@0.05={e['fpr@0.05']:.3f} (max {e['fpr@0.05_max']:.3f}) bh_fdp={e['bh_fdp']:.3f} (max {e['bh_fdp_max']:.3f}) recall={e['bh_recall']:.3f}")
        L.append("")
    if out["views"]:
        L.append("== backbone families (OpenOOD ImageNet-1K, p3 draw; means over the data sets of the group) ==")
        for v in out["views"]["views"]:
            for g in ("near", "far"):
                for k, e in sorted(out["views"][v][g].items(), key=lambda ke: ke[1]["FPR95"])[:14]:
                    L.append(f"{v:6s} {g:5s} {k:34s} {e['AUROC']:6.2f} / {e['FPR95']:6.2f}")
        L.append("")
    if out["cifar"]:
        L.append(f"== CIFAR-100 (means over {out['cifar']['n_draws']} draws) ==")
        for g in ("near", "far"):
            for k, e in sorted(out["cifar"][g].items(), key=lambda ke: ke[1]["FPR95"])[:30]:
                L.append(f"cifar {g:5s} {k:42s} {e['AUROC']:6.2f} / {e['FPR95']:6.2f} {fmt_ci(e):16s} {fmt_d(e, 'v1'):28s} {fmt_d(e, 'lp')}")
            for c, e in sorted(out["cifar"]["validity"][g].items()):
                if c.startswith("L14+D3L"):
                    L.append(f"cifar {g:5s} {c:42s} fpr@0.05={e['fpr@0.05']:.3f} (max {e['fpr@0.05_max']:.3f}) bh_fdp={e['bh_fdp']:.3f} (max {e['bh_fdp_max']:.3f}) recall={e['bh_recall']:.3f}")
        L.append("")
    if out["online"]:
        L.append(f"== arrival-time cross-fit: iters mean {out['online'].get('iters_mean', float('nan')):.1f} max {out['online'].get('iters_max')}, {out['online'].get('seconds_per_stream_mean', float('nan')):.0f} s/stream ==")
        for part in ("openood", "fourood"):
            if part in out["online"]:
                for g in GROUPS[part]:
                    for k, e in sorted(out["online"][part][g].items()):
                        L.append(f"onl {part:8s} {g:8s} {k:34s} {e['AUROC']:6.2f} / {e['FPR95']:6.2f}")
                    for c, e in sorted(out["online"][part]["validity"][g].items()):
                        L.append(f"onl {part:8s} {g:8s} {c:34s} fpr@0.05={e['fpr@0.05']:.3f} (max {e['fpr@0.05_max']:.3f}) bh_fdp={e['bh_fdp']:.3f}")
    t = "\n".join(L)
    (P.RESULTS14 / "p14_table.txt").write_text(t + "\n")
    print(t)
    print("ANA14_DONE", flush=True)


if __name__ == "__main__":
    main()
