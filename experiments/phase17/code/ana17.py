"""Phase 17 analysis (prereg_p17.json).
--dev2-adopt : the registered adoption rule of sigma_ref, on dev2 (5 draws, standalone): for near and for far the paired 95% t
               interval over the draws of FPR95(sigma_ref) - FPR95(sigma) must have its upper end <= +0.5, and the mean over the 10
               dev2 tasks (draws x streams) of the false-alarm rate of p_fin(sigma_ref) must be <= 0.06 at alpha = 0.05 and
               <= 0.21 at alpha = 0.2. All three -> sigma_ref becomes the main configuration of the paper; otherwise sigma stays.
               dev1 is reported alongside (descriptive). -> results/p17_dev/adoption.json
--summary    : all steps -> results/p17_summary.json (+ p17_table.txt): benchmarks (sigma, sigma_ref, p+ only, static, kNN; standalone
               and x bases; paired differences sigma_ref - sigma), decomposition rows, the bounds on the grid, covariate shift."""
import argparse
import glob
import json
import sys

import numpy as np
import pandas as pd

import p17common as P

sys.path.insert(0, "/home/omote/reprise_p14_20261011/code")
import ana14 as A14  # noqa: E402

SIG = "xfit|storey|q0.2|pp|pooled"
PF = "xfit|storey|q0.2|pp"
REF_S = "ref|storey|q0.2|pp|stat"
REF_P = "ref|storey|q0.2|pp"
LP = "xfit|lp|pooled"
STATIC = "xfit|static|pooled"
DEC = ("dec|deg", "dec|q", "dec|u", "dec|sigma_q")
KEYS = (SIG, PF, REF_S, REF_P, "ref|lp|stat", "ref|lp", LP, STATIC, "knn", "v1|storey|q0.2|pp") + DEC
R17 = P.RESULTS17


def load(pattern):
    files = sorted(glob.glob(str(pattern)))
    return pd.concat([pd.read_csv(f) for f in files], ignore_index=True) if files else None


def seed_mean(d, unit_cols):
    return d.groupby(unit_cols + ["cfg", "base"], as_index=False)[["AUROC", "FPR95"]].mean()


def tci(x):
    return A14.tci(np.asarray(x, float)) if len(x) >= 2 else {"mean": float(np.mean(x)), "lo": None, "hi": None, "n": int(len(x))}


def cell(m, cfg, base, stream, col="FPR95"):
    x = m[(m.cfg == cfg) & (m.base == base) & (m.stream == stream)].sort_values("draw")
    return x[col].to_numpy(), x.draw.to_numpy()


def dev_compare(dev):
    d = load(R17 / "p17_dev" / dev / "*.csv")
    assert d is not None, dev
    m = seed_mean(d, ["draw", "stream"])
    assert m.draw.nunique() == 5 and set(m.stream) == {"far", "near"}
    out = {}
    for s in ("near", "far"):
        a, da = cell(m, REF_S, "none", s)
        b, db = cell(m, SIG, "none", s)
        assert np.array_equal(da, db)
        au, _ = cell(m, REF_S, "none", s, "AUROC")
        bu, _ = cell(m, SIG, "none", s, "AUROC")
        ta, _ = cell(m, REF_S, "TINS", s)
        tb, _ = cell(m, SIG, "TINS", s)
        out[s] = {"sigma_ref_FPR95": float(a.mean()), "sigma_FPR95": float(b.mean()), "diff_ci": tci(a - b),
                  "sigma_ref_AUROC": float(au.mean()), "sigma_AUROC": float(bu.mean()), "dAUROC_ci": tci(au - bu),
                  "x_tins": {"sigma_ref": float(ta.mean()), "sigma": float(tb.mean()), "diff_ci": tci(ta - tb)},
                  "per_draw_FPR95": {"sigma_ref": a.tolist(), "sigma": b.tolist()}}
    v = d[(d.cfg == REF_P) & (d.base == "none")]
    out["validity"] = {"fpr@0.05": float(v["fpr@0.05"].mean()), "fpr@0.2": float(v["fpr@0.2"].mean()),
                       "fpr@0.05_max": float(v["fpr@0.05"].max()), "fpr@0.2_max": float(v["fpr@0.2"].max()), "bh_fdp_mean": float(v["bh_fdp"].mean()),
                       "n_tasks": int(len(v))}
    vs = d[(d.cfg == PF) & (d.base == "none")]
    out["validity_sigma"] = {"fpr@0.05": float(vs["fpr@0.05"].mean()), "fpr@0.2": float(vs["fpr@0.2"].mean())}
    return out


def dev2_adopt():
    res = {"rule": "dev2, standalone: upper end of the paired 95% t interval (5 draws) of FPR95(sigma_ref) - FPR95(sigma) <= +0.5 for "
                   "near and for far, and mean false-alarm rate of p_fin(sigma_ref) over the 10 tasks <= 0.06 at alpha 0.05 and <= 0.21 at 0.2",
           "margin": P.ADOPT_MARGIN, "fpr_limits": {str(k): v for k, v in P.ADOPT_FPR.items()}}
    res["dev2"] = dev_compare("dev2")
    res["dev1_descriptive"] = dev_compare("dev1")
    c = res["dev2"]
    res["near_holds"] = bool(c["near"]["diff_ci"]["hi"] <= P.ADOPT_MARGIN)
    res["far_holds"] = bool(c["far"]["diff_ci"]["hi"] <= P.ADOPT_MARGIN)
    res["validity_holds"] = bool(c["validity"]["fpr@0.05"] <= P.ADOPT_FPR[0.05] and c["validity"]["fpr@0.2"] <= P.ADOPT_FPR[0.2])
    res["adopt_sigma_ref_as_main"] = bool(res["near_holds"] and res["far_holds"] and res["validity_holds"])
    res["utc"] = P.utc()
    P.write_json(R17 / "p17_dev" / "adoption.json", res)
    print(json.dumps(res, indent=1))
    return res


def bench_summary(d, unit, groups):
    d = d[d.cfg.isin(KEYS)].copy()
    m = d.groupby([unit, "ds", "cfg", "base"], as_index=False)[["AUROC", "FPR95"]].mean()
    m["key"] = m.cfg + "|" + m.base
    out = A14.group_stats(m, unit, groups, with_pairs=False)
    for g, sets in groups.items():
        pu = m[m.ds.isin(sets)].groupby(["key", unit])[["AUROC", "FPR95"]].mean()
        keys = set(pu.index.get_level_values(0))
        for base in sorted(m.base.unique()):
            for c in (REF_S, REF_P, "ref|lp|stat") + DEC + (LP,):
                k, r = f"{c}|{base}", f"{SIG}|{base}"
                if k in keys and r in keys:
                    a, b = pu.loc[k], pu.loc[r]
                    common = a.index.intersection(b.index)
                    if len(common) >= 2:
                        out[g].setdefault(k, {})["dFPR95_vs_sigma"] = tci((a.loc[common].FPR95 - b.loc[common].FPR95).values)
                        out[g][k]["dAUROC_vs_sigma"] = tci((a.loc[common].AUROC - b.loc[common].AUROC).values)
    out["validity"] = A14.validity_stats(d, groups)
    vr = {}
    for g, sets in groups.items():
        x = d[(d.cfg == REF_P) & (d.base == "none") & d.ds.isin(sets)]
        if len(x):
            vr[g] = {k: float(x[k].mean()) for k in ("fpr@0.01", "fpr@0.05", "fpr@0.1", "fpr@0.2", "bh_fdp") if k in x}
    out["validity_ref"] = vr
    return out


def bounds_summary():
    files = sorted(glob.glob(str(R17 / "p17_rec" / "*" / "*.npz")))
    cells = {}
    for f in files:
        z = np.load(f)
        ds = f.split("/")[-2]
        name = f.split("/")[-1][:-4]
        c, r, rep = (int(x[1:]) if x[0] in "cr" else int(x[3:]) for x in name.split("_"))
        key = f"{ds}|c{c}|r{r}"
        e = cells.setdefault(key, {"reps": []})
        rec = {}
        for tag in ("c2", "c2r"):
            if f"{tag}_B" in z.files:
                rec[tag] = {"cert_005": float(np.mean(z[f"{tag}_pbar"] <= 0.05)), "actual_005": float(np.mean(z[f"{tag}_pfin"] <= 0.05)),
                            "violations": int(np.sum(z[f"{tag}_sigma"] > z[f"{tag}_B"] + 1e-9)),
                            "slack_median": float(np.median(z[f"{tag}_B"] - z[f"{tag}_sigma"])), "n": int(len(z[f"{tag}_B"]))}
        if "a1_pi" in z.files and len(z["a1_pi"]):
            h, pi, ph = z["a1_h"], z["a1_pi"], z["a1_phibar"]
            rec["a1"] = {"cert_l_005": float(np.mean(z["a1_pbar_l"] <= 0.05)), "cert_s_005": float(np.mean(z["a1_pbar_s"] <= 0.05)),
                         "actual_005": float(np.mean(z["a1_p"] <= 0.05)), "violations": int(np.sum(h > pi + 1e-9)),
                         "phibar_violations": int(np.sum(pi > ph + 1e-9)),
                         "h_over_pi_median": float(np.median(h / np.maximum(pi, 1e-300))), "pi_over_phibar_median": float(np.median(pi / np.maximum(ph, 1e-300))),
                         "pi_median": float(np.median(pi)), "n": int(len(pi)),
                         "ell_over_s_cal_median": float(np.median(z["a1_ell_over_s_cal"])), "ell_over_s_cal_q10": float(np.quantile(z["a1_ell_over_s_cal"], 0.1))}
        e["reps"].append(rec)
    out = {}
    for key, e in cells.items():
        agg = {}
        for tag in ("c2", "c2r", "a1"):
            recs = [r[tag] for r in e["reps"] if tag in r]
            if recs:
                agg[tag] = {k: float(np.mean([r[k] for r in recs])) for k in recs[0]}
        out[key] = agg
    return out


def csid_summary():
    files = sorted(glob.glob(str(R17 / "p17_csid" / "*" / "*" / "*.meta.json")))
    rows = []
    for f in files:
        mt = json.loads(open(f).read())
        cs, d, ds = f.split("/")[-3], f.split("/")[-2], f.split("/")[-1].split(".")[0]
        sub = mt["csid_metrics"]
        for name in ("sigma", "sigma_ref", "lp", "static_all", "knn"):
            rows.append({"csid": cs, "draw": d, "ds": ds, "score": name, **sub[name]})
        rows.append({"csid": cs, "draw": d, "ds": ds, "score": "_validity", **{f"{k}|{s}": v for k, dct in sub.items() if k.startswith("fpr_") for s, v in dct.items()},
                     "seeds_sigma_all": sub["seeds_sigma"]["all"], "seeds_sigma_csid": sub["seeds_sigma"]["csid"],
                     "seeds_ref_all": sub["seeds_ref"]["all"], "seeds_ref_csid": sub["seeds_ref"]["csid"]})
    if not rows:
        return None
    t = pd.DataFrame(rows)
    out = {}
    groups = {"near": A14.OO_NEAR, "far": A14.OO_FAR}
    for cs in sorted(t.csid.unique()):
        for g, sets in groups.items():
            x = t[(t.csid == cs) & t.ds.isin(sets)]
            for name in ("sigma", "sigma_ref", "lp", "static_all", "knn"):
                y = x[x.score == name].groupby("draw").mean(numeric_only=True)
                out[f"{cs}|{g}|{name}"] = {k: float(y[k].mean()) for k in y.columns}
            y = x[x.score == "_validity"].groupby("draw").mean(numeric_only=True)
            out[f"{cs}|{g}|validity"] = {k: float(y[k].mean()) for k in y.columns}
    return out


def summary():
    res = {"utc": P.utc()}
    p = R17 / "p17_dev" / "adoption.json"
    res["adoption"] = json.loads(p.read_text()) if p.exists() else None
    for dev in ("dev1", "dev2"):
        d = load(R17 / "p17_dev" / dev / "*.csv")
        if d is not None:
            d["ds"] = d["stream"]
            res[f"dev_{dev}"] = bench_summary(d, "draw", {"near": ["near"], "far": ["far"]})
    for part in ("openood", "fourood"):
        d = load(R17 / "p17_test" / part / "*" / "*.csv")
        if d is not None:
            res[f"test_{part}"] = bench_summary(d, "draw", A14.GROUPS[part])
    d = load(R17 / "p17_in200" / "*" / "*.csv")
    if d is not None:
        res["in200"] = bench_summary(d, "draw", {"near": A14.OO_NEAR, "far": A14.OO_FAR})
    d = load(R17 / "p17_cifar" / "*" / "*.csv")
    if d is not None:
        res["cifar"] = bench_summary(d, "draw", A14.CIFAR_GROUPS)
    d = load(R17 / "p17_rec" / "*" / "*.csv")
    if d is not None:
        d = d[d.cfg.isin(KEYS)]
        cells = {}
        for (ds, c, r, cfg, base), g in d.groupby(["ds", "c", "r", "cfg", "base"]):
            cells[f"{ds}|c{c}|r{r}|{cfg}|{base}"] = {"FPR95": float(g.FPR95.mean()), "AUROC": float(g.AUROC.mean()), "n": int(g.rep.nunique()),
                                                     "mixing_rate": float(g.mixing_rate.mean())}
        res["rec"] = cells
        res["bounds"] = bounds_summary()
    res["csid"] = csid_summary()
    P.write_json(R17 / "p17_summary.json", res)
    lines = []
    for key in ("dev_dev1", "dev_dev2", "test_openood", "test_fourood", "in200", "cifar"):
        if res.get(key) is None:
            continue
        lines.append(f"== {key} ==")
        for g in [k for k in res[key] if k not in ("per_dataset", "per_dataset_ci", "validity", "validity_ref")]:
            for k in (f"{SIG}|none", f"{REF_S}|none", f"{REF_P}|none", f"{LP}|none") + tuple(f"{x}|none" for x in DEC) + (f"{SIG}|TINS", f"{REF_S}|TINS"):
                e = res[key][g].get(k)
                if e and "FPR95" in e:
                    dd = e.get("dFPR95_vs_sigma", {})
                    lines.append(f"{g:8s} {k:34s} AUROC {e['AUROC']:6.2f} FPR95 {e['FPR95']:6.2f}" + (f"  d_vs_sigma {dd['mean']:+.2f} [{dd['lo']:+.2f}, {dd['hi']:+.2f}]" if dd and dd.get("lo") is not None else ""))
    (R17 / "p17_table.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dev2-adopt", action="store_true")
    ap.add_argument("--summary", action="store_true")
    a = ap.parse_args()
    if a.dev2_adopt:
        dev2_adopt()
    if a.summary:
        summary()
    print("ANA17_DONE", flush=True)
