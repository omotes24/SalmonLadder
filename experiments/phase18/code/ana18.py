"""Phase 18 summary (descriptive): per step, configuration and base, the mean over the draws of the group means (near = SSB-hard,
NINCO; far = iNaturalist, Textures, OpenImage-O; four = the Four-OOD mean), with the paired 95% t interval over draws; paired
differences against p+ ('lp') and sigma_ref ('sref'); validity of p_fin; the ID fraction of the seeds; Phase 17 checks; timing.
TINS products: mean over the stream orders first. Output: P18/results/p18_summary.json (and a short text table)."""
import glob
import json

import numpy as np
import pandas as pd
from scipy import stats

import p18common as P

GROUPS = {"openood": {"near": ["ssb_hard", "ninco"], "far": ["inaturalist", "textures", "openimageo"]},
          "fourood": {"four": ["inat", "sun", "places", "dtd"]}}
VCOLS = ["fpr@0.01", "fpr@0.05", "fpr@0.1", "fpr@0.2", "tpr@0.05", "bh_fdp", "bh_recall"]


def tci(x):
    x = np.asarray(x, np.float64)
    n = len(x)
    m = float(x.mean())
    if n < 2:
        return {"mean": m, "lo": None, "hi": None, "n": n}
    h = float(stats.t.ppf(0.975, n - 1) * x.std(ddof=1) / np.sqrt(n))
    return {"mean": m, "lo": m - h, "hi": m + h, "n": n}


def load(step):
    fs = sorted(glob.glob(str(P.RESULTS18 / f"p18_{step}" / "*" / "*" / "*.csv")))
    if not fs:
        return None
    d = pd.concat([pd.read_csv(f) for f in fs], ignore_index=True)
    d["draw"] = d["draw"].astype(str)
    return d


def summarise(d, key_cols=("cfg", "base")):
    m = d.groupby(["part", "draw", "ds"] + list(key_cols), as_index=False)[["AUROC", "FPR95"]].mean()
    out = {"groups": {}, "per_dataset": {}}
    for part, groups in GROUPS.items():
        mp = m[m.part == part]
        if mp.empty:
            continue
        for g, sets in groups.items():
            x = mp[mp.ds.isin(sets)].groupby(list(key_cols) + ["draw"])[["AUROC", "FPR95"]].mean()
            nds = mp[mp.ds.isin(sets)].groupby(list(key_cols) + ["draw"]).ds.nunique()
            for key, sub in x.groupby(level=list(range(len(key_cols)))):
                key = key if isinstance(key, tuple) else (key,)
                full = nds.loc[key]
                if (full != len(sets)).any():
                    continue
                out["groups"].setdefault(g, {})["|".join(key)] = {"AUROC": tci(sub.AUROC.values), "FPR95": tci(sub.FPR95.values),
                                                                    "draws": sorted(sub.index.get_level_values(-1))}
        for keys_, sub in mp.groupby(["ds"] + list(key_cols)):
            ds, key = keys_[0], keys_[1:]
            out["per_dataset"][f"{'|'.join(key)}|{ds}"] = {"AUROC": tci(sub.AUROC.values), "FPR95": tci(sub.FPR95.values)}
    return out, m


def paired(m, refs=("lp", "sref"), base="none"):
    out = {}
    for part, groups in GROUPS.items():
        mp = m[(m.part == part) & (m.base == base)]
        for g, sets in groups.items():
            x = mp[mp.ds.isin(sets)].groupby(["cfg", "draw"])[["AUROC", "FPR95"]].mean()
            cfgs = sorted(set(x.index.get_level_values(0)))
            for ref in refs:
                if ref not in cfgs:
                    continue
                a = x.loc[ref]
                for c in cfgs:
                    if c == ref:
                        continue
                    b = x.loc[c]
                    common = a.index.intersection(b.index)
                    if len(common) < 2:
                        continue
                    out.setdefault(g, {})[f"{c}-{ref}|{base}"] = {"dFPR95": tci(b.loc[common].FPR95 - a.loc[common].FPR95),
                                                                   "dAUROC": tci(b.loc[common].AUROC - a.loc[common].AUROC)}
    return out


def validity(d):
    v = d[(d.base == "none") & d.cfg.str.endswith("pfin")]
    out = {}
    for part, groups in GROUPS.items():
        vp = v[v.part == part]
        for g, sets in groups.items():
            for c, sub in vp[vp.ds.isin(sets)].groupby("cfg"):
                per_draw = sub.groupby("draw")[VCOLS].mean()
                out.setdefault(g, {})[c] = {k: tci(per_draw[k].values) for k in VCOLS}
                out[g][c]["fpr@0.05_min_task"] = float(sub["fpr@0.05"].min())
                out[g][c]["fpr@0.05_max_task"] = float(sub["fpr@0.05"].max())
    return out


def seeds(d):
    out = {}
    s = d[(d.base == "none") & (d.cfg.isin(["sref", "sref|pfin"]) | d.cfg.str.startswith("sref|"))]
    for c, sub in s.groupby("cfg"):
        cols = [x for x in sub.columns if x.startswith("idfrac_")]
        if not cols:
            continue
        vals = sub[cols].values.astype(np.float64).ravel()
        vals = vals[np.isfinite(vals)]
        if len(vals) == 0:
            continue
        nseeds = sub[[x for x in sub.columns if x.startswith("seeds_")]].values.astype(np.float64).ravel()
        out[c] = {"idfrac_mean": float(vals.mean()), "idfrac_min": float(vals.min()), "idfrac_max": float(vals.max()),
                  "n_task_views": int(len(vals)), "seeds_mean": float(np.nanmean(nseeds))}
        per_part = {}
        for part, sp in sub.groupby("part"):
            vv = sp[cols].values.astype(np.float64).ravel()
            vv = vv[np.isfinite(vv)]
            if len(vv):
                per_part[part] = {"idfrac_mean": float(vv.mean()), "idfrac_min": float(vv.min()), "idfrac_max": float(vv.max())}
        out[c]["per_part"] = per_part
    return out


def main():
    S = {"utc": P.utc(), "note": "Phase 18, descriptive (not registered); see p18common.py"}
    frames = {}
    for step in ("main", "views", "valrc"):
        d = load(step)
        if d is None:
            continue
        key = ("cfg", "base")
        summ, m = summarise(d, key)
        frames[step] = m
        S[step] = {"summary": summ, "paired": paired(m), "paired_tins": paired(m, base="TINS"), "validity": validity(d), "seeds": seeds(d),
                   "n_tasks": int(d.groupby(["part", "draw", "ds"] + (["backbone"] if "backbone" in d else [])).ngroups)}
    # valrc vs main (same draw and data set; Four-OOD test sets differ by the 4,000 R/C images)
    if "valrc" in frames and "main" in frames:
        out = {}
        for part, groups in GROUPS.items():
            for g, sets in groups.items():
                for c in ("lp", "sref", "lp|pfin", "sref|pfin"):
                    for base in ("none", "TINS"):
                        xs = []
                        for name in ("valrc", "main"):
                            mm = frames[name]
                            x = mm[(mm.part == part) & (mm.cfg == c) & (mm.base == base) & mm.ds.isin(sets)].groupby("draw")[["AUROC", "FPR95"]].mean()
                            xs.append(x)
                        common = xs[0].index.intersection(xs[1].index)
                        if len(common) >= 2:
                            out.setdefault(g, {})[f"{c}|{base}"] = {"dFPR95_valrc_minus_main": tci(xs[0].loc[common].FPR95 - xs[1].loc[common].FPR95),
                                                                     "dAUROC_valrc_minus_main": tci(xs[0].loc[common].AUROC - xs[1].loc[common].AUROC)}
        S["valrc_vs_main"] = out
    checks = []
    for f in glob.glob(str(P.RESULTS18 / "p18_main" / "*" / "*" / "*.meta.json")):
        mt = json.load(open(f))
        checks.append(mt.get("check17", np.nan))
    S["check17_max"] = float(np.nanmax(checks)) if checks else None
    S["check17_n"] = len(checks)
    accs = []
    for f in glob.glob(str(P.RESULTS18 / "p18_valrc" / "*" / "*" / "*.meta.json")):
        mt = json.load(open(f))
        accs += list(mt["proto_acc_of_val_RC"].values())
    S["valrc_proto_acc"] = {"min": float(min(accs)), "max": float(max(accs))} if accs else None
    secs = {}
    for f in glob.glob(str(P.RESULTS18 / "p18_main" / "openood" / "*" / "*.meta.json")):
        mt = json.load(open(f))
        secs[f.split("/")[-2] + "/" + f.split("/")[-1].split(".")[0]] = {"seconds": mt["seconds"], "nodes": mt["nodes"]}
    S["main_task_seconds"] = secs
    tf = P.RESULTS18 / "p18_timing" / "timing.json"
    if tf.exists():
        S["timing"] = json.load(open(tf))
    P.write_json(P.RESULTS18 / "p18_summary.json", S)
    # short table
    lines = []
    for step in ("main", "views", "valrc"):
        if step not in S:
            continue
        lines.append(f"== {step}")
        for g, e in S[step]["summary"]["groups"].items():
            for k in sorted(e):
                if not k.endswith("|none") and not k.endswith("|TINS"):
                    continue
                a, f = e[k]["AUROC"], e[k]["FPR95"]
                ci = f"[{f['lo']:.2f}, {f['hi']:.2f}]" if f["lo"] is not None else ""
                lines.append(f"{g:5s} {k:28s} AUROC {a['mean']:6.2f} FPR95 {f['mean']:6.2f} {ci} n={f['n']}")
        for g, e in S[step]["validity"].items():
            for c, vv in e.items():
                lines.append(f"{g:5s} validity {c:16s} fpr@0.05 {vv['fpr@0.05']['mean']:.4f} fpr@0.2 {vv['fpr@0.2']['mean']:.4f} bh_fdp {vv['bh_fdp']['mean']:.4f}")
        for c, e in S[step]["seeds"].items():
            lines.append(f"seeds {c:16s} idfrac mean {e['idfrac_mean']:.3f} [{e['idfrac_min']:.3f}, {e['idfrac_max']:.3f}] n={e['n_task_views']}")
    (P.RESULTS18 / "p18_table.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print("check17_max", S["check17_max"], "n", S["check17_n"])
    print("ANA18_DONE", flush=True)


if __name__ == "__main__":
    main()
