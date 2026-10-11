"""Phase 15 summary -> P15/results/p15_summary.json (+ p15_table.txt).
base:    the many-label / zero-shot baselines on ImageNet-200 and CIFAR-100 (per data set and near / far means).
kg:      the k_g grid: per (ds, r, k_g) means over the repetitions, every configuration (standalone and x TINS).
cap_*:   the capped seed rules next to the registered ones: dev (per dev / stream / draw), test (as ana14 test_train), in200,
         cifar (per draw, paired intervals vs p+ only and vs the registered rule), rec (per cell).
views_c: static / kNN read-outs raw vs centred per single view (near / far means, standalone and x TINS)."""
import glob
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import p15common as P

sys.path.insert(0, "/home/omote/reprise_p14_20261011/code")
import ana14 as A14  # noqa: E402  (group_stats, validity_stats, seeds_from_meta, load_rows, tci, GROUPS)

GROUPS = A14.GROUPS
OO_NEAR, OO_FAR = A14.OO_NEAR, A14.OO_FAR
CIFAR_GROUPS = A14.CIFAR_GROUPS
R15 = P.RESULTS15


def base_summary():
    d = A14.load_rows(str(R15 / "base" / "*" / "*.csv"))
    if d is None or len(d) == 0:
        return None
    out = {}
    for bench, g in d.groupby("benchmark"):
        groups = {"near": OO_NEAR, "far": OO_FAR} if bench == "imagenet200" else CIFAR_GROUPS
        e = {"per_dataset": {f"{m}|{ds}": {"AUROC": float(x.AUROC), "FPR95": float(x.FPR95)} for (m, ds), x in g.groupby(["method", "ds"])[["AUROC", "FPR95"]].mean().iterrows()}}
        for grp, sets in groups.items():
            sub = g[g.ds.isin(sets)]
            e[grp] = {m: {"AUROC": float(x.AUROC), "FPR95": float(x.FPR95), "n_sets": int(len(sub[sub.method == m]))} for m, x in sub.groupby("method")[["AUROC", "FPR95"]].mean().iterrows()}
        metas = [json.loads(Path(f).read_text()) for f in sorted(glob.glob(str(R15 / "base" / bench / "*.meta.json")))]
        e["n_train"] = int(metas[0]["n_train"]) if metas else None
        out[bench] = e
    return out


def kg_summary():
    d = A14.load_rows(str(R15 / "kg" / "*" / "*.csv"))
    if d is None or len(d) == 0:
        return None
    d["key"] = d.cfg + "|" + d.base
    out = {}
    for (ds, c, r, kg), g in d.groupby(["ds", "c", "r", "kg"]):
        e = {"n_reps": int(g.rep.nunique()), "n_ood": int(g.n_ood.iloc[0]), "mixing_rate": float(g.mixing_rate.iloc[0]), "cfg": {}}
        for k, gg in g.groupby("key"):
            e["cfg"][k] = {"AUROC": float(gg.AUROC.mean()), "FPR95": float(gg.FPR95.mean())}
        out[f"{ds}|c{c}|r{r}|kg{kg}"] = e
    return out


def cap_test_summary():
    d = A14.load_rows(str(R15 / "cap_test" / "*" / "*" / "*.csv"))
    if d is None or len(d) == 0:
        return None
    d["draw"] = d.draw.astype(str)
    d["key"] = d.cfg + "|" + d.base
    out = {"n_draws": int(d.draw.nunique()), "bases": sorted(d.base.unique())}
    for part, groups in GROUPS.items():
        dd = d[d.part == part]
        if len(dd) == 0:
            continue
        dd = dd.groupby(["key", "cfg", "base", "draw", "ds"], as_index=False)[["AUROC", "FPR95"]].mean()
        out[part] = A14.group_stats(dd, "draw", groups)
        out[part]["validity"] = A14.validity_stats(d[d.part == part], groups)
        out[part]["seeds"] = A14.seeds_from_meta(str(R15 / "cap_test" / part / "*" / "*.meta.json"), "draw")
        out[part].update(paired(dd, "draw", groups))
    return out


def paired(dd, unit, groups, refs=(("storey", "xfit|storey|q0.2|pp|pooled|{base}"), ("lp", "xfit|lp|pooled|{base}"))):
    """Paired differences (over the units) of every pooled configuration against the registered rule and against p+ only."""
    per = dd.groupby(["key", unit, "ds"])[["AUROC", "FPR95"]].mean()
    res = {}
    for g, sets in groups.items():
        pu = per[per.index.get_level_values(2).isin(sets)].groupby(level=[0, 1]).mean()
        keys = pu.index.get_level_values(0).unique()
        res[f"paired_{g}"] = {}
        for k in keys:
            if "|pooled|" not in k:
                continue
            base = k.rsplit("|", 1)[1]
            e = {}
            for tag, rk in refs:
                rk = rk.format(base=base)
                if rk in keys and rk != k:
                    v, r = pu.loc[k], pu.loc[rk]
                    common = v.index.intersection(r.index)
                    if len(common) >= 2:
                        e[f"dFPR95_vs_{tag}"] = A14.tci((v.loc[common].FPR95 - r.loc[common].FPR95).values)
                        e[f"dAUROC_vs_{tag}"] = A14.tci((v.loc[common].AUROC - r.loc[common].AUROC).values)
            if e:
                res[f"paired_{g}"][k] = e
    return res


def cap_dev_summary():
    d = A14.load_rows(str(R15 / "cap_dev" / "*" / "*.csv"))
    if d is None or len(d) == 0:
        return None
    d["key"] = d.cfg + "|" + d.base
    out = {}
    for dev, g in d.groupby("dev"):
        res = {}
        for stream, gg in g.groupby("stream"):
            per_draw = gg.groupby(["key", "draw"])[["AUROC", "FPR95"]].mean()
            for c in per_draw.index.get_level_values(0).unique():
                v = per_draw.loc[c]
                res.setdefault(c, {})[stream] = {"AUROC": float(v.AUROC.mean()), "FPR95": float(v.FPR95.mean()), "n_draws": int(len(v))}
        for c, gg in g[g["fpr@0.05"].notna()].groupby("key"):
            res.setdefault(c, {})["validity"] = {k: float(gg[k].mean()) for k in ("fpr@0.01", "fpr@0.05", "fpr@0.1", "fpr@0.2", "bh_fdp", "bh_recall") if k in gg}
        out[dev] = res
        out[dev]["_seeds"] = A14.seeds_from_meta(str(R15 / "cap_dev" / dev / "*.meta.json"), "draw")
    return out


def cap_in200_summary():
    d = A14.load_rows(str(R15 / "cap_in200" / "*" / "*.csv"))
    if d is None or len(d) == 0:
        return None
    d["draw"] = d.draw.astype(str)
    d["key"] = d.cfg + "|" + d.base
    groups = {"near": OO_NEAR, "far": OO_FAR}
    out = {"n_draws": int(d.draw.nunique()), **A14.group_stats(d, "draw", groups), "validity": A14.validity_stats(d, groups),
           "seeds": A14.seeds_from_meta(str(R15 / "cap_in200" / "*" / "*.meta.json"), "draw")}
    out.update(paired(d, "draw", groups))
    return out


def cap_cifar_summary():
    d = A14.load_rows(str(R15 / "cap_cifar" / "*" / "*.csv"))
    if d is None or len(d) == 0:
        return None
    d["draw"] = d.draw.astype(str)
    d["key"] = d.cfg + "|" + d.base
    out = {"n_draws": int(d.draw.nunique()), **A14.group_stats(d, "draw", CIFAR_GROUPS), "validity": A14.validity_stats(d, CIFAR_GROUPS)}
    out.update(paired(d, "draw", CIFAR_GROUPS))
    acc, sizes = {}, {}
    for f in sorted(glob.glob(str(R15 / "cap_cifar" / "*" / "*.meta.json"))):
        m = json.loads(Path(f).read_text())
        ds = m.get("ds") or Path(f).name.replace(".meta.json", "")
        sizes[ds] = {"n_id": int(m["n_id"]), "n_ood": int(m["n_ood"]), "mixing": m["n_ood"] / (m["n_ood"] + m["n_id"])}
        for view, dd in m["views"].items():
            for fk, e in dd.items():
                if fk.startswith("fold"):
                    _, rule, qk = fk.split("|")
                    acc.setdefault(f"{ds}|{view}|{rule}|{qk}", []).append((e["seeds"], e["pi0_hat"], np.nan if e["id_frac"] is None else e["id_frac"], e["n_test"]))
    out["seeds"] = {k: {"seeds": float(a[:, 0].mean()), "pi0_hat": float(a[:, 1].mean()), "id_frac": (float(np.nanmean(a[:, 2])) if np.isfinite(a[:, 2]).any() else None),
                        "seeds_over_n_test": float((a[:, 0] / a[:, 3]).mean()), "n": int(len(a))} for k, a in ((k, np.array(L, float)) for k, L in acc.items())}
    out["sizes"] = sizes
    return out


def cap_rec_summary():
    d = A14.load_rows(str(R15 / "cap_rec" / "*" / "*.csv"))
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
    for f in sorted(glob.glob(str(R15 / "cap_rec" / "*" / "*.meta.json"))):
        m = json.loads(Path(f).read_text())
        stem = Path(f).name.replace(".meta.json", "")
        c_, r_ = stem.split("_")[0][1:], stem.split("_")[1][1:]
        key = f"{Path(f).parent.name}|c{c_}|r{r_}"
        if key not in out:
            continue
        for v, mv in m["views"].items():
            for rule in ("storey", "bh", "storeyc", "bhc"):
                fks = [fk for fk in mv if fk.startswith("fold") and fk.endswith(f"|{rule}|q0.2")]
                tot = sum(mv[fk]["seeds"] for fk in fks)
                ood = sum(mv[fk]["seeds"] * (1.0 - (mv[fk]["id_frac"] if mv[fk]["id_frac"] is not None else 1.0)) for fk in fks)
                acc = out[key].setdefault(f"view_{v}", {}).setdefault(rule, {"seeds": [], "ood_seeds": []})
                acc["seeds"].append(tot)
                acc["ood_seeds"].append(ood)
    for key, e in out.items():
        for vk in [k for k in e if k.startswith("view_")]:
            for rk, acc in e[vk].items():
                e[vk][rk] = {"seeds": float(np.mean(acc["seeds"])), "ood_seeds": float(np.mean(acc["ood_seeds"])),
                             "id_frac": (float(1.0 - np.sum(acc["ood_seeds"]) / np.sum(acc["seeds"])) if np.sum(acc["seeds"]) > 0 else None), "n_reps": len(acc["seeds"])}
    return out


def views_c_summary():
    d = A14.load_rows(str(R15 / "views_c" / "*" / "*.csv"))
    if d is None or len(d) == 0:
        return None
    d["key"] = d.cfg + "|" + d.base
    out = {"views": sorted(d.view.unique())}
    for v, dv in d.groupby("view"):
        dd = dv.groupby(["key", "ds"], as_index=False)[["AUROC", "FPR95"]].mean()
        out[v] = {"per_dataset": {f"{k}|{ds}": {"AUROC": float(e.AUROC), "FPR95": float(e.FPR95)} for (k, ds), e in dd.groupby(["key", "ds"])[["AUROC", "FPR95"]].mean().iterrows()}}
        for g, sets in GROUPS["openood"].items():
            out[v][g] = {k: {"AUROC": float(e.AUROC), "FPR95": float(e.FPR95)} for k, e in dd[dd.ds.isin(sets)].groupby("key")[["AUROC", "FPR95"]].mean().iterrows()}
    metas = {}
    for f in sorted(glob.glob(str(R15 / "views_c" / "*" / "*.meta.json"))):
        m = json.loads(Path(f).read_text())
        metas.setdefault(m["view"], []).append(m["support_mean_norm"])
    out["support_mean_norm"] = {v: float(np.mean(x)) for v, x in metas.items()}
    return out


def main():
    out = {"tau": P.TAU, "kgs": list(P.KGS), "base": base_summary(), "kg": kg_summary(), "cap_dev": cap_dev_summary(), "cap_test": cap_test_summary(),
           "cap_in200": cap_in200_summary(), "cap_cifar": cap_cifar_summary(), "cap_rec": cap_rec_summary(), "views_c": views_c_summary(), "_meta": {"utc": P.utc()}}
    P.write_json(R15 / "p15_summary.json", out)
    L = []
    if out["base"]:
        for bench, e in out["base"].items():
            L.append(f"== baselines {bench} (training images: {e['n_train']}) ==")
            for g in ("near", "far"):
                for m, x in sorted(e[g].items(), key=lambda kv: kv[1]["FPR95"]):
                    L.append(f"{bench:12s} {g:5s} {m:20s} {x['AUROC']:6.2f} / {x['FPR95']:6.2f}")
            L.append("")
    if out["kg"]:
        L.append("== k_g grid: FPR95 standalone (sigma | p+ only | static | v1) per (ds, r, k_g) ==")
        for key, e in out["kg"].items():
            cf = e["cfg"]
            def f(k):
                return f"{cf[k]['FPR95']:5.1f}" if k in cf else "  -  "
            L.append(f"{key:28s} {f('xfit|storey|q0.2|pp|pooled|none')} | {f('xfit|lp|pooled|none')} | {f('xfit|static|pooled|none')} | {f('v1|storey|q0.2|pp|none')}")
        L.append("")
    for step in ("cap_test", "cap_in200", "cap_cifar"):
        e = out[step]
        if not e:
            continue
        L.append(f"== {step}: FPR95 (standalone / x TINS where present) ==")
        parts = [k for k in ("openood", "fourood") if k in e] if step == "cap_test" else [None]
        for part in parts:
            ee = e[part] if part else e
            for g in [k for k in ("near", "far", "fourood") if k in ee]:
                for k, x in sorted(ee[g].items(), key=lambda kv: kv[1]["FPR95"]):
                    if "|pooled|" in k or k.startswith("v1|") or k.startswith("knn"):
                        L.append(f"{step:10s} {part or '':8s} {g:7s} {k:40s} {x['AUROC']:6.2f} / {x['FPR95']:6.2f}  {A14.fmt_ci(x)}")
            L.append("")
    if out["cap_rec"]:
        L.append("== cap_rec: FPR95 standalone (storey | storeyc | bh | bhc | p+ only | static) ==")
        for key, e in out["cap_rec"].items():
            cf = e["cfg"]
            def f(k):
                return f"{cf[k]['FPR95']:5.1f}" if k in cf else "  -  "
            L.append(f"{key:24s} {f('xfit|storey|q0.2|pp|pooled|none')} | {f('xfit|storeyc|q0.2|pp|pooled|none')} | {f('xfit|bh|q0.2|pp|pooled|none')} | {f('xfit|bhc|q0.2|pp|pooled|none')} | {f('xfit|lp|pooled|none')} | {f('xfit|static|pooled|none')}")
        L.append("")
    if out["views_c"]:
        L.append("== views: static / kNN raw vs centred (near FPR95 / far FPR95, standalone) ==")
        for v in out["views_c"]["views"]:
            e = out["views_c"][v]
            L.append(f"{v:6s} " + "  ".join(f"{k}: {e['near'][k]['FPR95']:5.1f}/{e['far'][k]['FPR95']:5.1f}" for k in sorted(e["near"]) if k.endswith("|none")))
    (R15 / "p15_table.txt").write_text("\n".join(L) + "\n")
    print("\n".join(L))
    print("ANA15_DONE", flush=True)


if __name__ == "__main__":
    main()
