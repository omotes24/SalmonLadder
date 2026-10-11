"""Phase 12 analysis -> P12/results/p12_summary.json and p12_table.txt.
draws:   per configuration (views|readout|base): means over the six draws of the labelled images (TINS rows: mean over the stream
         orders first), paired 95% t intervals over the draws for the difference to the main configuration, per data set and per group;
         the paper's draw (p3) separately.
online:  the arrival-time mode: means over the stream orders (as Phase 11), per group and per data set.
rec:     the recurrence x mixing-rate grid: means over the replicates per (data set, c, r).
selection: copied from results/sel/sel12_summary.json."""
import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd

import p12common as P
from an11 import tci

OO_NEAR, OO_FAR, FOUR = ["ssb_hard", "ninco"], ["inaturalist", "textures", "openimageo"], ["inat", "sun", "places", "dtd"]
GROUPS = {"openood": {"near": OO_NEAR, "far": OO_FAR}, "fourood": {"fourood": FOUR}}


def load_rows(pattern):
    files = sorted(glob.glob(pattern))
    return pd.concat([pd.read_csv(f) for f in files]) if files else None


def draws_summary(main):
    d = load_rows(str(P.RESULTS12 / "draws" / "*" / "*" / "*.csv"))
    d["draw"] = d.draw.astype(str)
    d["cfg"] = d.views + "|" + d.readout + "|" + d.base
    out = {"n_draws": int(d.draw.nunique()), "draws": sorted(d.draw.unique().tolist())}
    for part, groups in GROUPS.items():
        dd = d[d.part == part]
        res = {"per_dataset": {}, "per_dataset_p3": {}}
        # per data set: mean over seeds (TINS) within a draw, then over draws
        per_draw_ds = dd.groupby(["cfg", "draw", "ds"])[["AUROC", "FPR95"]].mean()
        for (c, ds), v in per_draw_ds.groupby(level=[0, 2]).mean().iterrows():
            res["per_dataset"][f"{c}|{ds}"] = {"AUROC": float(v.AUROC), "FPR95": float(v.FPR95)}
        for (c, draw, ds), v in per_draw_ds.iterrows():
            if draw == "p3":
                res["per_dataset_p3"][f"{c}|{ds}"] = {"AUROC": float(v.AUROC), "FPR95": float(v.FPR95)}
        res["per_dataset_ci"] = {}
        for (c, ds), v in per_draw_ds.groupby(level=[0, 2]):
            if len(v) > 1:
                res["per_dataset_ci"][f"{c}|{ds}"] = {"AUROC_ci": tci(v.AUROC.values), "FPR95_ci": tci(v.FPR95.values), "n_draws": int(len(v))}
        for g, sets in groups.items():
            per_draw = per_draw_ds[per_draw_ds.index.get_level_values(2).isin(sets)].groupby(level=[0, 1]).mean()
            alone = main.replace("|TINS", "|none")
            ref = per_draw.loc[main] if main in per_draw.index.get_level_values(0) else None
            ref_alone = per_draw.loc[alone] if alone in per_draw.index.get_level_values(0) else None
            gres = {}
            for c in per_draw.index.get_level_values(0).unique():
                v = per_draw.loc[c]
                e = {"AUROC": float(v.AUROC.mean()), "FPR95": float(v.FPR95.mean()), "n_draws": int(len(v)),
                     "AUROC_ci": tci(v.AUROC.values), "FPR95_ci": tci(v.FPR95.values),
                     "p3": {"AUROC": float(v.AUROC.get("p3", np.nan)), "FPR95": float(v.FPR95.get("p3", np.nan))}}
                if ref is not None and len(v) == len(ref):
                    e["dFPR95_vs_main"] = tci((v.FPR95 - ref.FPR95).values)          # vs the main configuration (x TINS)
                    e["dAUROC_vs_main"] = tci((v.AUROC - ref.AUROC).values)
                if ref_alone is not None and len(v) == len(ref_alone):
                    e["dFPR95_vs_alone"] = tci((v.FPR95 - ref_alone.FPR95).values)   # vs the standalone main configuration
                    e["dAUROC_vs_alone"] = tci((v.AUROC - ref_alone.AUROC).values)
                gres[c] = e
            res[g] = gres
        out[part] = res
    return out


def seeds_summary():
    """Seeds of the draws runs: per (part, data set, view, q): mean over draws of the number of seeds, pi0_hat and the realised ID fraction."""
    acc = {}
    for f in sorted(glob.glob(str(P.RESULTS12 / "draws" / "*" / "*" / "*.meta.json"))):
        m = json.loads(Path(f).read_text())
        for v, mv in m["views"].items():
            for qk, e in mv.items():
                acc.setdefault(f"{m['part']}|{m['ds']}|{v}|{qk}", []).append((e["seeds"], e["pi0_hat"], e["id_frac"] if e["id_frac"] is not None else np.nan, m["n_ood"], m["n"], m["draw"]))
    out = {}
    for k, L in acc.items():
        a = np.array([x[:5] for x in L], float)
        out[k] = {"seeds": float(a[:, 0].mean()), "pi0_hat": float(a[:, 1].mean()), "id_frac": float(np.nanmean(a[:, 2])), "id_frac_max": float(np.nanmax(a[:, 2])),
                  "n_ood": int(a[0, 3]), "n": int(a[0, 4]), "n_draws": int(len(L)), "seeds_over_n_ood": float((a[:, 0] / a[:, 3]).mean()),
                  "id_frac_p3": next((x[2] for x in L if str(x[5]) == "p3"), None)}
    return out


def online_summary(dirname="online"):
    d = load_rows(str(P.RESULTS12 / dirname / "*" / "*.csv"))
    if d is None or len(d) == 0:
        return None
    d["ds"] = d.stream.str.replace(r"_seed\d+$", "", regex=True)
    d["seed"] = d.stream.str.extract(r"seed(\d+)$", expand=False).astype(int)
    d["cfg"] = d.views + "|" + d.readout + "|" + d.base
    out = {}
    for part, groups in GROUPS.items():
        dd = d[d.part == part]
        res = {"per_dataset": {f"{c}|{ds}": {"AUROC": float(v.AUROC), "FPR95": float(v.FPR95)}
                               for (c, ds), v in dd.groupby(["cfg", "ds"])[["AUROC", "FPR95"]].mean().iterrows()}}
        for g, sets in groups.items():
            per_seed = dd[dd.ds.isin(sets)].groupby(["cfg", "seed", "ds"])[["AUROC", "FPR95"]].mean().groupby(["cfg", "seed"]).mean()
            res[g] = {c: {"AUROC": float(per_seed.loc[c].AUROC.mean()), "FPR95": float(per_seed.loc[c].FPR95.mean()), "n_orders": int(len(per_seed.loc[c]))}
                      for c in per_seed.index.get_level_values(0).unique()}
        out[part] = res
    metas = [json.loads(Path(f).read_text()) for f in sorted(glob.glob(str(P.RESULTS12 / dirname / "*" / "*.meta.json")))]
    out["seeds_last_batch"] = {f"{m['part']}|{m['stream']}|{v}": m["views"][v] for m in metas for v in m["views"]}
    return out


def rec_summary():
    d = load_rows(str(P.RESULTS12 / "rec" / "*" / "*.csv"))
    if d is None or len(d) == 0:
        return None
    d["cfg"] = d.views + "|" + d.readout + "|" + d.base
    out = {}
    for (ds, c, r), g in d.groupby(["ds", "c", "r"]):
        key = f"{ds}|c{c}|r{r}"
        e = {"n_reps": int(g.rep.nunique()), "n_ood": int(g.n_ood.iloc[0]), "mixing_rate": float(g.mixing_rate.iloc[0]), "cfg": {}}
        for cfg, gg in g.groupby("cfg"):
            e["cfg"][cfg] = {"AUROC": float(gg.AUROC.mean()), "FPR95": float(gg.FPR95.mean()), "FPR95_sd": float(gg.FPR95.std(ddof=1)) if len(gg) > 1 else None}
        metas = [json.loads(Path(f).read_text()) for f in glob.glob(str(P.RESULTS12 / "rec" / ds / f"c{c}_r{r}_rep*.meta.json"))]
        for v in ("D3L", "L14", "L14xD3L"):
            ms = [m["views"][v] for m in metas if v in m["views"]]
            if not ms:
                continue
            ev = {"pi0_hat": float(np.mean([m["q0.1"]["pi0_hat"] for m in ms])),
                  "cut": {k: float(np.mean([m["cut"][k] for m in ms if m["cut"]])) for k in ms[0]["cut"]} if ms[0].get("cut") else None,
                  "u_pos_mean": {k: float(np.mean([m["u_pos_mean"][k] for m in ms])) for k in ms[0]["u_pos_mean"]}}
            for qk in ("q0.05", "q0.1", "q0.2"):
                # seeds: mean over replicates; id_frac: mean over the replicates that selected at least one seed (nan if none did)
                ev[f"seeds_{qk}"] = float(np.mean([m[qk]["seeds"] for m in ms]))
                fr = [m[qk]["id_frac"] for m in ms if m[qk]["id_frac"] is not None]
                ev[f"id_frac_{qk}"] = float(np.mean(fr)) if fr else None
                ev[f"n_reps_with_seeds_{qk}"] = len(fr)
                # seeds that are unknown images (0 for a replicate without seeds): mean over replicates
                ev[f"ood_seeds_{qk}"] = float(np.mean([0.0 if m[qk]["id_frac"] is None else m[qk]["seeds"] * (1.0 - m[qk]["id_frac"]) for m in ms]))
            e[f"view_{v}"] = ev
        out[key] = e
    return out


def main():
    sel = json.loads((P.RESULTS12 / "sel" / "sel12_summary.json").read_text()) if (P.RESULTS12 / "sel" / "sel12_summary.json").exists() else None
    chosen = sel["_selection"]["chosen"] if sel else "D3L|T|q0.1|none"
    views, _, q, _ = chosen.split("|")
    main_cfg = f"{views}|T|TINS" if q == "q0.1" else f"{views}|T{q}|TINS"
    out = {"selection": sel, "main": main_cfg, "draws": draws_summary(main_cfg), "seeds": seeds_summary(), "online": online_summary(), "online_q0.2": online_summary("online_q0.2"), "rec": rec_summary(),
           "_meta": {"utc": P.utc(), "note": "draws: means over the draws of the labelled images (TINS: over the stream orders first); "
                                              "online: means over the stream orders; rec: means over replicates"}}
    P.write_json(P.RESULTS12 / "p12_summary.json", out)
    lines = [f"main configuration (dev selection -> test): {main_cfg}", ""]
    for part in ("openood", "fourood"):
        for g in GROUPS[part]:
            lines.append(f"== draws {part} {g}: mean over draws (p3 draw in brackets), dFPR95 vs main [95% t over draws] ==")
            items = sorted(out["draws"][part][g].items(), key=lambda kv: kv[1]["FPR95"])
            for c, e in items[:60]:
                dd = e.get("dFPR95_vs_main")
                lines.append(f"{c:30s} {e['AUROC']:6.2f} / {e['FPR95']:6.2f}  [{e['p3']['AUROC']:6.2f} / {e['p3']['FPR95']:6.2f}]  n={e['n_draws']}"
                             + (f"  dFPR {dd['mean']:+.2f} [{dd['lo']:+.2f},{dd['hi']:+.2f}]" if dd else ""))
            lines.append("")
    if out["online"]:
        for part in ("openood", "fourood"):
            for g in GROUPS[part]:
                lines.append(f"== online {part} {g} ==")
                for c, e in sorted(out["online"][part][g].items(), key=lambda kv: kv[1]["FPR95"]):
                    lines.append(f"{c:30s} {e['AUROC']:6.2f} / {e['FPR95']:6.2f}")
                lines.append("")
    if out["rec"]:
        lines.append("== recurrence x mixing (D3L, standalone): key  n_ood  rate  | static | knn | p+ | p+p- (q=0.1) | seeds id_frac | cut own/incoming ==")
        for key, e in out["rec"].items():
            cf = e["cfg"]
            v = e.get("view_D3L", {})
            f = lambda k: (f"{cf[k]['AUROC']:5.1f}/{cf[k]['FPR95']:5.1f}" if k in cf else "   -   ")
            cut = v.get("cut") or {}
            idf = v.get("id_frac_q0.1")
            lines.append(f"{key:22s} {e['n_ood']:6d} {100 * e['mixing_rate']:5.2f}% | {f('D3L|static|none')} | {f('D3L|knn|none')} | {f('D3L|Tlp|none')} | {f('D3L|T|none')} "
                         f"| {v.get('seeds_q0.1', 0):7.0f} {(float('nan') if idf is None else idf):5.2f} | {cut.get('own_edges', 0):6.1f} {cut.get('incoming_edges', 0):6.1f}")
    t = "\n".join(lines)
    (P.RESULTS12 / "p12_table.txt").write_text(t + "\n")
    print(t)
    print("ANA12_DONE", flush=True)


if __name__ == "__main__":
    main()
