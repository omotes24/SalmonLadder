"""Phase 16 analysis (prereg_p16.json).
--dev1-select : the registered choice of the base combination on dev1 (s0*s vs s0*p, x TINS): the candidate with the lower mean
                dev1-near FPR95 among those whose mean dev1-far FPR95 is at most that of sigma x TINS + 0.5 (if none, the lower far;
                ties -> s0*s). -> results/h16_dev/dev1_selection.json
--dev2-confirm: H1 (near) and H2 (far): mean over the 5 draws of FPR95(h16|s) - FPR95(sigma), standalone, <= +0.5;
                H3: mean false-alarm rate of p (standalone) <= 0.06 at alpha = 0.05 and <= 0.21 at alpha = 0.2 (draws x streams);
                adoption of h16 as the main configuration iff H1 and H2 and H3. -> results/h16_dev/dev2_confirmation.json
--summary     : all steps -> results/p16_summary.json (+ p16_table.txt)."""
import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import p16common as P

sys.path.insert(0, "/home/omote/reprise_p14_20261011/code")
import ana14 as A14  # noqa: E402

REF = "xfit|storey|q0.2|pp|pooled"
PF14 = "xfit|storey|q0.2|pp"
LP = "xfit|lp|pooled"
CANDS = ("h16|s", "h16|p")
MARGIN = 0.5
R16 = P.RESULTS16
KEYS = (REF, PF14, LP, "h16|s", "h16|p", "v1|storey|q0.2|pp", "knn")


def load(pattern):
    files = sorted(glob.glob(str(pattern)))
    return pd.concat([pd.read_csv(f) for f in files], ignore_index=True) if files else None


def seed_mean(d, unit_cols):
    """Average the base rows over the base's seeds (orders) per unit; standalone rows have seed = -1."""
    return d.groupby(unit_cols + ["cfg", "base"], as_index=False)[["AUROC", "FPR95"]].mean()


def tci(x):
    return A14.tci(np.asarray(x, float)) if len(x) >= 2 else {"mean": float(np.mean(x)), "lo": None, "hi": None, "n": int(len(x))}


def dev_table(dev):
    d = load(R16 / "h16_dev" / dev / "*.csv")
    assert d is not None, f"no rows for {dev}"
    return d, seed_mean(d, ["draw", "stream"])


def cell(m, cfg, base, stream, col="FPR95"):
    x = m[(m.cfg == cfg) & (m.base == base) & (m.stream == stream)].sort_values("draw")
    return x[col].to_numpy(), x.draw.to_numpy()


def dev1_select():
    d, m = dev_table("dev1")
    assert m.draw.nunique() == 5 and set(m.stream) == {"far", "near"}
    far_ref = float(cell(m, REF, "TINS", "far")[0].mean())
    cand = {}
    for c in CANDS:
        cand[c] = {"near_x_tins": float(cell(m, c, "TINS", "near")[0].mean()), "far_x_tins": float(cell(m, c, "TINS", "far")[0].mean())}
        cand[c]["eligible"] = bool(cand[c]["far_x_tins"] <= far_ref + MARGIN)
    elig = [c for c in CANDS if cand[c]["eligible"]]
    if elig:
        chosen = min(elig, key=lambda c: (cand[c]["near_x_tins"], CANDS.index(c)))
    else:
        chosen = min(CANDS, key=lambda c: (cand[c]["far_x_tins"], CANDS.index(c)))
    out = {"rule": "min dev1-near FPR95 (x TINS) among candidates with dev1-far FPR95 (x TINS) <= sigma x TINS + 0.5; else min far; ties -> h16|s",
           "sigma_x_tins": {"near": float(cell(m, REF, "TINS", "near")[0].mean()), "far": far_ref}, "candidates": cand, "chosen": chosen,
           "standalone_dev1": {s: {"h16|s": float(cell(m, "h16|s", "none", s)[0].mean()), "sigma": float(cell(m, REF, "none", s)[0].mean())} for s in ("near", "far")},
           "n_tasks": int(len(d.groupby(["draw", "stream"]))), "utc": P.utc()}
    P.write_json(R16 / "h16_dev" / "dev1_selection.json", out)
    print(json.dumps(out, indent=1))
    return out


def dev2_confirm():
    sel = json.loads((R16 / "h16_dev" / "dev1_selection.json").read_text())
    d, m = dev_table("dev2")
    assert m.draw.nunique() == 5 and set(m.stream) == {"far", "near"}
    res = {"chosen_base_combination": sel["chosen"]}
    for s, H in (("near", "H1"), ("far", "H2")):
        a, da = cell(m, "h16|s", "none", s)
        b, db = cell(m, REF, "none", s)
        assert np.array_equal(da, db)
        au, _ = cell(m, "h16|s", "none", s, "AUROC")
        bu, _ = cell(m, REF, "none", s, "AUROC")
        diff = a - b
        res[H] = {"stream": s, "h16_FPR95": float(a.mean()), "sigma_FPR95": float(b.mean()), "diff_mean": float(diff.mean()), "diff_ci": tci(diff),
                  "margin": MARGIN, "holds": bool(diff.mean() <= MARGIN), "h16_AUROC": float(au.mean()), "sigma_AUROC": float(bu.mean()),
                  "dAUROC_ci": tci(au - bu)}
        tins = sel["chosen"]
        ta, _ = cell(m, tins, "TINS", s)
        tb, _ = cell(m, REF, "TINS", s)
        res[H]["x_tins"] = {"h16": float(ta.mean()), "sigma": float(tb.mean()), "diff_ci": tci(ta - tb)}
    v = d[(d.cfg == "h16|p") & (d.base == "none")]
    f05, f20 = float(v["fpr@0.05"].mean()), float(v["fpr@0.2"].mean())
    res["H3"] = {"fpr@0.05": f05, "fpr@0.2": f20, "fpr@0.05_max": float(v["fpr@0.05"].max()), "fpr@0.2_max": float(v["fpr@0.2"].max()),
                 "bh_fdp_mean": float(v["bh_fdp"].mean()), "holds": bool(f05 <= 0.06 and f20 <= 0.21)}
    res["adopt_h16_as_main"] = bool(res["H1"]["holds"] and res["H2"]["holds"] and res["H3"]["holds"])
    res["utc"] = P.utc()
    P.write_json(R16 / "h16_dev" / "dev2_confirmation.json", res)
    print(json.dumps(res, indent=1))
    return res


def bench_summary(d, unit, groups, bases_keep=None):
    """Group means (over the unit, after the per-dataset means) of every key cfg|base, and the paired differences h16 - sigma."""
    d = d[d.cfg.isin(KEYS)].copy()
    m = d.groupby([unit, "ds", "cfg", "base"], as_index=False)[["AUROC", "FPR95"]].mean()     # seeds of a base averaged
    m["key"] = m.cfg + "|" + m.base
    out = A14.group_stats(m, unit, groups, with_pairs=False)
    for g, sets in groups.items():
        pu = m[m.ds.isin(sets)].groupby(["key", unit])[["AUROC", "FPR95"]].mean()
        for base in sorted(m.base.unique()):
            for c in CANDS:
                k, r = f"{c}|{base}", f"{REF}|{base}"
                if k in pu.index.get_level_values(0) and r in pu.index.get_level_values(0):
                    a, b = pu.loc[k], pu.loc[r]
                    common = a.index.intersection(b.index)
                    if len(common) >= 2:
                        out[g].setdefault(k, {})["dFPR95_vs_sigma"] = tci((a.loc[common].FPR95 - b.loc[common].FPR95).values)
                        out[g][k]["dAUROC_vs_sigma"] = tci((a.loc[common].AUROC - b.loc[common].AUROC).values)
    out["validity"] = A14.validity_stats(d, groups)
    return out


def summary():
    res = {"utc": P.utc()}
    for f in ("dev1_selection.json", "dev2_confirmation.json"):
        p = R16 / "h16_dev" / f
        res[f.split(".")[0]] = json.loads(p.read_text()) if p.exists() else None
    for dev in ("dev1", "dev2"):
        d = load(R16 / "h16_dev" / dev / "*.csv")
        if d is not None:
            d["ds"] = d["stream"]
            res[f"dev_{dev}"] = bench_summary(d, "draw", {"near": ["near"], "far": ["far"]})
    for part in ("openood", "fourood"):
        d = load(R16 / "h16_test" / part / "*" / "*.csv")
        if d is not None:
            res[f"test_{part}"] = bench_summary(d, "draw", A14.GROUPS[part])
    d = load(R16 / "h16_in200" / "*" / "*.csv")
    if d is not None:
        res["in200"] = bench_summary(d, "draw", {"near": A14.OO_NEAR, "far": A14.OO_FAR})
    d = load(R16 / "h16_cifar" / "*" / "*.csv")
    if d is not None:
        res["cifar"] = bench_summary(d, "draw", A14.CIFAR_GROUPS)
    d = load(R16 / "h16_rec" / "*" / "*.csv")
    if d is not None:
        d = d[d.cfg.isin(KEYS)]
        cells = {}
        for (ds, c, r, cfg, base), g in d.groupby(["ds", "c", "r", "cfg", "base"]):
            cells[f"{ds}|c{c}|r{r}|{cfg}|{base}"] = {"FPR95": float(g.FPR95.mean()), "AUROC": float(g.AUROC.mean()), "n": int(g.rep.nunique()),
                                                     "mixing_rate": float(g.mixing_rate.mean())}
        res["rec"] = cells
    P.write_json(R16 / "p16_summary.json", res)
    lines = []
    for key in ("dev_dev1", "dev_dev2", "test_openood", "test_fourood", "in200", "cifar"):
        if res.get(key) is None:
            continue
        lines.append(f"== {key} ==")
        for g in [k for k in res[key] if k not in ("per_dataset", "per_dataset_ci", "validity")]:
            for k in (f"{REF}|none", "h16|s|none", "h16|p|none", f"{LP}|none", f"{REF}|TINS", "h16|s|TINS", "h16|p|TINS"):
                e = res[key][g].get(k)
                if e and "FPR95" in e:
                    dd = e.get("dFPR95_vs_sigma", {})
                    lines.append(f"{g:8s} {k:34s} AUROC {e['AUROC']:6.2f} FPR95 {e['FPR95']:6.2f}" + (f"  d_vs_sigma {dd['mean']:+.2f} [{dd['lo']:+.2f}, {dd['hi']:+.2f}]" if dd and dd.get("lo") is not None else ""))
    (R16 / "p16_table.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dev1-select", action="store_true")
    ap.add_argument("--dev2-confirm", action="store_true")
    ap.add_argument("--summary", action="store_true")
    a = ap.parse_args()
    if a.dev1_select:
        dev1_select()
    if a.dev2_confirm:
        dev2_confirm()
    if a.summary:
        summary()
    print("ANA16_DONE", flush=True)
