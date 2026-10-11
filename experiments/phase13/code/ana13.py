"""Phase 13 analysis -> P13/results/p13_summary.json and p13_table.txt.
online:  arrival-time mode with converged propagation: means over the stream orders (per group and per data set), iteration counts.
fdpcal:  exchangeable calibration: seeds / realised ID fraction / pi0_hat per (data set, view, q) over the 3 splits, uniformity of
         p+, metrics per configuration (means over splits; TINS rows over the stream orders first).
product: the static-rank products on the benchmarks: means over the 6 draws per data set and per group, paired 95% t intervals
         over the draws against p+ p- (same base).
rec:     the grid with the products: means over the replicates per (data set, c, r).
in200:   ImageNet-200: means over the 6 draws per data set and per group (near / far), 95% t intervals over the draws."""
import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd

import p13common as P
from an11 import tci

OO_NEAR, OO_FAR, FOUR = ["ssb_hard", "ninco"], ["inaturalist", "textures", "openimageo"], ["inat", "sun", "places", "dtd"]
GROUPS = {"openood": {"near": OO_NEAR, "far": OO_FAR}, "fourood": {"fourood": FOUR}}


def load_rows(pattern):
    files = sorted(glob.glob(pattern))
    return pd.concat([pd.read_csv(f) for f in files]) if files else None


def online_summary():
    d = load_rows(str(P.RESULTS13 / "online" / "*" / "*.csv"))
    if d is None or len(d) == 0:
        return None
    d["ds"] = d.stream.str.replace(r"_seed\d+$", "", regex=True)
    d["seed"] = d.stream.str.extract(r"seed(\d+)$", expand=False).astype(int)
    d["cfg"] = d.views + "|" + d.readout + "|" + d.base
    out = {}
    for part, groups in GROUPS.items():
        dd = d[d.part == part]
        if len(dd) == 0:
            continue
        res = {"per_dataset": {f"{c}|{ds}": {"AUROC": float(v.AUROC), "FPR95": float(v.FPR95)}
                               for (c, ds), v in dd.groupby(["cfg", "ds"])[["AUROC", "FPR95"]].mean().iterrows()}}
        for g, sets in groups.items():
            per_seed = dd[dd.ds.isin(sets)].groupby(["cfg", "seed", "ds"])[["AUROC", "FPR95"]].mean().groupby(["cfg", "seed"]).mean()
            res[g] = {c: {"AUROC": float(per_seed.loc[c].AUROC.mean()), "FPR95": float(per_seed.loc[c].FPR95.mean()), "n_orders": int(len(per_seed.loc[c]))}
                      for c in per_seed.index.get_level_values(0).unique()}
        out[part] = res
    metas = [json.loads(Path(f).read_text()) for f in sorted(glob.glob(str(P.RESULTS13 / "online" / "*" / "*.meta.json")))]
    out["streams"] = {f"{m['part']}|{m['stream']}": {"seconds": m.get("seconds"), **{f"{v}_{k}": m["views"][v][k] for v in m["views"] for k in ("iters_pos_mean", "iters_pos_max", "iters_neg_mean", "iters_neg_max", "seeds_last_batch", "id_frac_last_batch")}} for m in metas}
    its = [m["views"][v][k] for m in metas for v in m["views"] for k in ("iters_pos_mean", "iters_neg_mean")]
    out["iters_mean"] = float(np.mean(its)) if its else None
    out["iters_max"] = int(max(m["views"][v][k] for m in metas for v in m["views"] for k in ("iters_pos_max", "iters_neg_max"))) if metas else None
    out["seconds_per_stream_mean"] = float(np.mean([m["seconds"] for m in metas])) if metas else None
    return out


def fdpcal_summary():
    d = load_rows(str(P.RESULTS13 / "fdpcal" / "*" / "*" / "*.csv"))
    if d is None or len(d) == 0:
        return None
    d["cfg"] = d.views + "|" + d.readout + "|" + d.base
    out = {"n_splits": int(d.split.nunique()), "metrics": {}, "seeds": {}, "uniformity": {}}
    for part, groups in GROUPS.items():
        dd = d[d.part == part]
        if len(dd) == 0:
            continue
        per = dd.groupby(["cfg", "split", "ds"])[["AUROC", "FPR95"]].mean()        # TINS: over the orders first
        res = {"per_dataset": {f"{c}|{ds}": {"AUROC": float(v.AUROC), "FPR95": float(v.FPR95)} for (c, ds), v in per.groupby(level=[0, 2]).mean().iterrows()}}
        for g, sets in groups.items():
            ps = per[per.index.get_level_values(2).isin(sets)].groupby(level=[0, 1]).mean()
            res[g] = {c: {"AUROC": float(ps.loc[c].AUROC.mean()), "FPR95": float(ps.loc[c].FPR95.mean()), "FPR95_ci": tci(ps.loc[c].FPR95.values), "n_splits": int(len(ps.loc[c]))}
                      for c in ps.index.get_level_values(0).unique()}
        out["metrics"][part] = res
    acc, uni = {}, {}
    for f in sorted(glob.glob(str(P.RESULTS13 / "fdpcal" / "*" / "*" / "*.meta.json"))):
        m = json.loads(Path(f).read_text())
        for v, mv in m["views"].items():
            for qk in ("q0.05", "q0.1", "q0.2"):
                e = mv[qk]
                acc.setdefault(f"{m['part']}|{m['ds']}|{v}|{qk}", []).append((e["seeds"], e["pi0_hat"], np.nan if e["id_frac"] is None else e["id_frac"], m["n_ood"], m["n_id_test"]))
            uni.setdefault(f"{m['part']}|{m['ds']}|{v}", []).append(mv["uniformity"])
    for k, L in acc.items():
        a = np.array(L, float)
        out["seeds"][k] = {"seeds": float(a[:, 0].mean()), "pi0_hat": float(a[:, 1].mean()), "id_frac": float(np.nanmean(a[:, 2])) if np.isfinite(a[:, 2]).any() else None,
                           "id_frac_max": float(np.nanmax(a[:, 2])) if np.isfinite(a[:, 2]).any() else None, "id_frac_splits": [None if not np.isfinite(x) else float(x) for x in a[:, 2]],
                           "n_ood": int(a[0, 3]), "n_id_test": int(a[0, 4]), "true_pi0": float(a[0, 4] / (a[0, 3] + a[0, 4])), "n_splits": int(len(L)),
                           "seeds_over_n_ood": float((a[:, 0] / a[:, 3]).mean())}
    for k, L in uni.items():
        out["uniformity"][k] = {a: float(np.mean([e[a] for e in L])) for a in L[0]}
        out["uniformity"][k]["n_splits"] = len(L)
    return out


def product_summary():
    d = load_rows(str(P.RESULTS13 / "product" / "*" / "*" / "*.csv"))
    if d is None or len(d) == 0:
        return None
    d["draw"] = d.draw.astype(str)
    d["cfg"] = d.views + "|" + d.readout + "|" + d.base
    out = {"n_draws": int(d.draw.nunique())}
    for part, groups in GROUPS.items():
        dd = d[d.part == part]
        if len(dd) == 0:
            continue
        per_draw_ds = dd.groupby(["cfg", "draw", "ds"])[["AUROC", "FPR95"]].mean()
        res = {"per_dataset": {f"{c}|{ds}": {"AUROC": float(v.AUROC), "FPR95": float(v.FPR95)} for (c, ds), v in per_draw_ds.groupby(level=[0, 2]).mean().iterrows()}}
        for g, sets in groups.items():
            per_draw = per_draw_ds[per_draw_ds.index.get_level_values(2).isin(sets)].groupby(level=[0, 1]).mean()
            gres = {}
            for c in per_draw.index.get_level_values(0).unique():
                v = per_draw.loc[c]
                base = c.split("|")[2]
                ref_key = f"{c.split('|')[0]}|Tq0.2|{base}"
                e = {"AUROC": float(v.AUROC.mean()), "FPR95": float(v.FPR95.mean()), "n_draws": int(len(v)), "FPR95_ci": tci(v.FPR95.values)}
                if ref_key in per_draw.index.get_level_values(0):
                    ref = per_draw.loc[ref_key]
                    common = v.index.intersection(ref.index)
                    if len(common) >= 2:
                        e["dFPR95_vs_ref"] = tci((v.loc[common].FPR95 - ref.loc[common].FPR95).values)
                        e["dAUROC_vs_ref"] = tci((v.loc[common].AUROC - ref.loc[common].AUROC).values)
                        e["ref"] = ref_key
                gres[c] = e
            res[g] = gres
        out[part] = res
    return out


def rec_summary():
    d = load_rows(str(P.RESULTS13 / "rec" / "*" / "*.csv"))
    if d is None or len(d) == 0:
        return None
    d["cfg"] = d.views + "|" + d.readout + "|" + d.base
    out = {}
    for (ds, c, r), g in d.groupby(["ds", "c", "r"]):
        e = {"n_reps": int(g.rep.nunique()), "n_ood": int(g.n_ood.iloc[0]), "mixing_rate": float(g.mixing_rate.iloc[0]), "cfg": {}}
        for cfg, gg in g.groupby("cfg"):
            e["cfg"][cfg] = {"AUROC": float(gg.AUROC.mean()), "FPR95": float(gg.FPR95.mean())}
        out[f"{ds}|c{c}|r{r}"] = e
    return out


def in200_summary():
    d = load_rows(str(P.RESULTS13 / "in200" / "*" / "*.csv"))
    if d is None or len(d) == 0:
        return None
    d["draw"] = d.draw.astype(str)
    d["cfg"] = d.views + "|" + d.readout + "|" + d.base
    out = {"n_draws": int(d.draw.nunique()), "per_dataset": {}, "per_dataset_p3": {}}
    per_draw_ds = d.groupby(["cfg", "draw", "ds"])[["AUROC", "FPR95"]].mean()
    for (c, ds), v in per_draw_ds.groupby(level=[0, 2]).mean().iterrows():
        out["per_dataset"][f"{c}|{ds}"] = {"AUROC": float(v.AUROC), "FPR95": float(v.FPR95)}
    for (c, draw, ds), v in per_draw_ds.iterrows():
        if draw == "p3":
            out["per_dataset_p3"][f"{c}|{ds}"] = {"AUROC": float(v.AUROC), "FPR95": float(v.FPR95)}
    for g, sets in (("near", OO_NEAR), ("far", OO_FAR)):
        per_draw = per_draw_ds[per_draw_ds.index.get_level_values(2).isin(sets)].groupby(level=[0, 1]).mean()
        out[g] = {}
        for c in per_draw.index.get_level_values(0).unique():
            v = per_draw.loc[c]
            out[g][c] = {"AUROC": float(v.AUROC.mean()), "FPR95": float(v.FPR95.mean()), "n_draws": int(len(v)), "AUROC_ci": tci(v.AUROC.values), "FPR95_ci": tci(v.FPR95.values)}
    acc = {}
    for f in sorted(glob.glob(str(P.RESULTS13 / "in200" / "*" / "*.meta.json"))):
        m = json.loads(Path(f).read_text())
        for v, mv in m["views"].items():
            for qk in ("q0.05", "q0.1", "q0.2"):
                e = mv[qk]
                acc.setdefault(f"{m['ds']}|{v}|{qk}", []).append((e["seeds"], e["pi0_hat"], np.nan if e["id_frac"] is None else e["id_frac"], m["n_ood"], m["n_id"]))
    out["seeds"] = {k: {"seeds": float(np.mean([x[0] for x in L])), "pi0_hat": float(np.mean([x[1] for x in L])), "id_frac": float(np.nanmean([x[2] for x in L])),
                        "n_ood": int(L[0][3]), "n_id": int(L[0][4]), "n_draws": len(L)} for k, L in acc.items()}
    return out


def main():
    out = {"online": online_summary(), "fdpcal": fdpcal_summary(), "product": product_summary(), "rec": rec_summary(), "in200": in200_summary(),
           "_meta": {"utc": P.utc()}}
    P.write_json(P.RESULTS13 / "p13_summary.json", out)
    lines = []
    if out["online"]:
        lines.append(f"== online (converged), iters mean {out['online']['iters_mean']:.1f} max {out['online']['iters_max']}, {out['online']['seconds_per_stream_mean']:.0f} s/stream ==")
        for part in ("openood", "fourood"):
            if part in out["online"]:
                for g in GROUPS[part]:
                    for c, e in sorted(out["online"][part][g].items()):
                        lines.append(f"{part:8s} {g:8s} {c:24s} {e['AUROC']:6.2f} / {e['FPR95']:6.2f}")
        lines.append("")
    if out["fdpcal"]:
        lines.append("== exchangeable calibration: seeds (mean over splits), realised ID fraction, pi0_hat / true pi0 ==")
        for k, e in sorted(out["fdpcal"]["seeds"].items()):
            lines.append(f"{k:36s} seeds {e['seeds']:9.1f}  id_frac {e['id_frac'] if e['id_frac'] is None else round(e['id_frac'], 3)}  max {e['id_frac_max'] if e['id_frac_max'] is None else round(e['id_frac_max'], 3)}  pi0 {e['pi0_hat']:.3f}/{e['true_pi0']:.3f}")
        lines.append("== uniformity of p+ (ID test, exchangeable calibration) ==")
        for k, e in sorted(out["fdpcal"]["uniformity"].items()):
            lines.append(f"{k:30s} " + " ".join(f"{a}:{e[a]:.3f}" for a in e if a != "n_splits"))
        for part in out["fdpcal"]["metrics"]:
            for g in GROUPS[part]:
                for c, e in sorted(out["fdpcal"]["metrics"][part][g].items()):
                    lines.append(f"fdpcal {part:8s} {g:8s} {c:24s} {e['AUROC']:6.2f} / {e['FPR95']:6.2f}")
        lines.append("")
    if out["product"]:
        lines.append("== products on the benchmarks (means over draws; dFPR vs p+p- same base [95% t]) ==")
        for part in ("openood", "fourood"):
            if part in out["product"]:
                for g in GROUPS[part]:
                    for c, e in sorted(out["product"][part][g].items()):
                        dd = e.get("dFPR95_vs_ref")
                        lines.append(f"{part:8s} {g:8s} {c:30s} {e['AUROC']:6.2f} / {e['FPR95']:6.2f}  n={e['n_draws']}" + (f"  dFPR {dd['mean']:+.2f} [{dd['lo']:+.2f},{dd['hi']:+.2f}]" if dd else ""))
        lines.append("")
    if out["rec"]:
        lines.append("== grid: FPR95 (L14+D3L standalone) static | proto | knn | p+ | p+p- | p+ x proto | p+p- x proto | p+p- x knn ==")
        for key, e in out["rec"].items():
            cf = e["cfg"]
            f = lambda k: (f"{cf[k]['FPR95']:5.1f}" if k in cf else "  -  ")
            lines.append(f"{key:22s} {e['n_ood']:6d} {100 * e['mixing_rate']:5.2f}% | {f('L14+D3L|static|none')} | {f('L14+D3L|proto|none')} | {f('L14+D3L|knn|none')} | {f('L14+D3L|Tlp|none')} | {f('L14+D3L|Tq0.2|none')} | {f('L14+D3L|Tlpxproto|none')} | {f('L14+D3L|Tq0.2xproto|none')} | {f('L14+D3L|Tq0.2xknn|none')}")
        lines.append("")
    if out["in200"]:
        lines.append("== ImageNet-200 (means over draws) ==")
        for g in ("near", "far"):
            for c, e in sorted(out["in200"][g].items()):
                lines.append(f"in200 {g:5s} {c:28s} {e['AUROC']:6.2f} / {e['FPR95']:6.2f}  [{e['FPR95_ci']['lo']:.2f}, {e['FPR95_ci']['hi']:.2f}]")
        for k, e in sorted(out["in200"]["seeds"].items()):
            lines.append(f"in200 seeds {k:28s} {e['seeds']:8.1f} id_frac {e['id_frac']:.3f} pi0 {e['pi0_hat']:.3f}")
    t = "\n".join(lines)
    (P.RESULTS13 / "p13_table.txt").write_text(t + "\n")
    print(t)
    print("ANA13_DONE", flush=True)


if __name__ == "__main__":
    main()
