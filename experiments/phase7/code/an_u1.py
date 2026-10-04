"""Descriptive analyses of the FROZEN components on the standard U1 streams (45 = 5 splits x 3 draws x 3 orders).
Nothing is selected here and no candidate of experiment A is evaluated.

  A1  miss rate / AUROC per appearance bin (static p, memory only, p_LP only, zeta, static x p_LP, REPRISE)
  C2  three separate calibrated ranks: static (d), memory proximity alone (rho), propagation (u); products; overlap
      of the detections of p_M and p_LP at their ID-95% thresholds; rank correlations within ID and within OOD
  C3  history on / off per channel: memory {static p, p_M} x propagation {alone, batch only, full history}
  E4  time course: ID false-alarm rate and OOD miss rate per stream decile at the stream's ID-95% threshold
Unit: split (the 9 streams of a split averaged); paired 95% t intervals (df = 4)."""
import sys

import numpy as np
from scipy.stats import spearmanr

import an7 as A
from metrics_p4 import metrics, tci

VIEWS = ["B14", "L14"]
FAMS = ["static", "mem", "plp", "zeta", "static_plp", "reprise"]


def split_ci(rows, key):
    x, _ = A.unit_mean(rows, "split", key)
    return tci(x)


def diff_ci(rows, a, b):
    xa, _ = A.unit_mean(rows, "split", a)
    xb, _ = A.unit_mean(rows, "split", b)
    return tci(xa - xb)


def main(exp="u1std", views=VIEWS, tag="u1"):
    ts = A.tasks(exp, views[0])
    print(f"{exp}: {len(ts)} streams, views {views}")
    R, RB, RC, RO, RT = [], [], [], [], []
    for t in ts:
        info = A.parse_std(t)
        s = A.Stream(exp, t, views)
        k = s.appearance()
        o = s.is_ood
        row = dict(info)
        sc = {f: s.family(f) for f in FAMS}
        sc["rho"] = s.s("Mrho")
        sc["static_rho"] = s.s("static") + s.s("Mrho")
        sc["static_rho_plp"] = sc["static_rho"] + s.s("lp0")
        sc["rho_plp"] = s.s("Mrho") + s.s("lp0")
        sc["lp_zero"] = s.s("lp")
        sc["mem_lpzero"] = s.s("M") + s.s("lp")
        if "s::lpB" in s.z[views[0]].files:
            sc["lpB"] = s.s("lpB")
            sc["static_lpB"] = s.s("static") + s.s("lpB")
            sc["mem_lpB"] = s.s("M") + s.s("lpB")
            sc["static_lpzero"] = s.s("static") + s.s("lp")
        if "s::lpA" in s.z[views[0]].files:
            sc["lpA"] = s.s("lpA")
            sc["static_lpA"] = s.s("static") + s.s("lpA")
            sc["mem_lpA"] = s.s("M") + s.s("lpA")
        for nm, x in sc.items():
            for base in ("none", "TINS", "MCM"):
                m = metrics(x + s.base(base), o)
                row[f"{nm}|{base}|FPR95"], row[f"{nm}|{base}|AUROC"] = m["FPR95"], m["AUROC"]
        R.append(row)
        rb = dict(info)
        for nm in FAMS + ["rho", "static_rho_plp"] + [x for x in ("lpB", "lpA", "static_lpA", "mem_lpA") if x in sc]:
            for b, v in A.by_bin(sc[nm], o, k).items():
                rb[f"{nm}|{b}|miss"], rb[f"{nm}|{b}|auroc"], rb[f"n|{b}"] = v["miss"], v["auroc"], v["n"]
        RB.append(rb)
        # A1b: first appearances (k = 1) by arrival position (history length)
        pos = np.arange(len(o))
        for nm in ("static", "reprise", "zeta", "static_plp"):
            thr = A.threshold(sc[nm], o)
            for tag_, (lo, hi) in (("b0-3", (0, 1024)), ("b4-15", (1024, 4096)), ("b16+", (4096, 10 ** 9))):
                sel = o & (k == 1) & (pos >= lo) & (pos < hi)
                rb[f"{nm}|k1pos|{tag_}"] = 100 * float(np.mean(sc[nm][sel] >= thr)) if sel.any() else np.nan
                rb[f"n|k1pos|{tag_}"] = int(sel.sum())
        # C2: overlap of detections (p_M vs p_LP) and rank correlations
        tm, tl = A.threshold(sc["mem"], o), A.threshold(sc["plp"], o)
        dm, dl = sc["mem"] < tm, sc["plp"] < tl
        rc = dict(info)
        for b, (lo, hi) in [("all", (1, 10 ** 9))] + [(f"{lo}-{hi}", (lo, hi)) for lo, hi in A.BINS]:
            sel = o & (k >= lo) & (k <= hi)
            rc[f"both|{b}"] = 100 * float(np.mean(dm[sel] & dl[sel]))
            rc[f"mem_only|{b}"] = 100 * float(np.mean(dm[sel] & ~dl[sel]))
            rc[f"lp_only|{b}"] = 100 * float(np.mean(~dm[sel] & dl[sel]))
            rc[f"neither|{b}"] = 100 * float(np.mean(~dm[sel] & ~dl[sel]))
        for a_, b_, nm in (("mem", "plp", "mem_plp"), ("rho", "plp", "rho_plp"), ("static", "plp", "static_plp"), ("static", "rho", "static_rho"), ("static", "mem", "static_mem")):
            rc[f"rho_id|{nm}"] = float(spearmanr(sc[a_][~o], sc[b_][~o]).statistic)
            rc[f"rho_ood|{nm}"] = float(spearmanr(sc[a_][o], sc[b_][o]).statistic)
        RC.append(rc)
        # E4: time course (deciles of the stream position)
        rt = dict(info)
        dec = (np.arange(len(o)) * 10) // len(o)
        for nm in ("static", "zeta", "reprise", "mem", "plp"):
            thr = A.threshold(sc[nm], o)
            for j in range(10):
                rt[f"{nm}|fa|{j}"] = 100 * float(np.mean(sc[nm][(dec == j) & ~o] < thr))
                rt[f"{nm}|miss|{j}"] = 100 * float(np.mean(sc[nm][(dec == j) & o] >= thr))
        RT.append(rt)
    out = {"exp": exp, "views": views, "n_streams": len(ts), "overall": {}, "bins": {}, "overlap": {}, "corr": {}, "time": {}, "diffs": {}}
    print("== overall (AUROC / FPR95): standalone | x TINS | x MCM")
    names = [n for n in ("static", "rho", "mem", "plp", "lp_zero", "zeta", "static_plp", "static_rho", "rho_plp", "static_rho_plp", "reprise", "mem_lpzero",
                         "lpA", "lpB", "static_lpA", "mem_lpA", "static_lpB", "mem_lpB", "static_lpzero") if f"{n}|none|FPR95" in R[0]]
    for nm in names:
        cell = {}
        for base in ("none", "TINS", "MCM"):
            cell[base] = {"AUROC": split_ci(R, f"{nm}|{base}|AUROC"), "FPR95": split_ci(R, f"{nm}|{base}|FPR95")}
        out["overall"][nm] = cell
        print(f"  {nm:16s} " + " | ".join(f"{cell[b]['AUROC']['mean']:6.2f} / {cell[b]['FPR95']['mean']:6.2f}" for b in ("none", "TINS", "MCM")))
    pairs = [("reprise", "zeta"), ("reprise", "static"), ("reprise", "static_plp"), ("reprise", "mem"), ("static_rho_plp", "reprise"), ("mem_lpzero", "reprise")]
    pairs += [(a, b) for a, b in (("mem_lpA", "static_lpA"), ("reprise", "static_plp"), ("static_plp", "static_lpA"), ("reprise", "mem_lpA"),
                                   ("mem_lpB", "static_lpB"), ("static_lpB", "static_lpA"), ("mem_lpzero", "static_lpzero"), ("static_lpzero", "static_lpB"),
                                   ("mem_lpzero", "mem_lpB")) if f"{a}|none|FPR95" in R[0] and f"{b}|none|FPR95" in R[0]]
    print("== paired differences (standalone FPR95; AUROC), unit split")
    for a_, b_ in pairs:
        d, da = diff_ci(R, f"{a_}|none|FPR95", f"{b_}|none|FPR95"), diff_ci(R, f"{a_}|none|AUROC", f"{b_}|none|AUROC")
        out["diffs"][f"{a_}-{b_}"] = {"FPR95": d, "AUROC": da}
        print(f"  {a_:15s} - {b_:12s} FPR95 {A.fmt_ci(d)}   AUROC {A.fmt_ci(da)}")
    if "mem_lpA|none|FPR95" in R[0]:
        for met in ("FPR95", "AUROC"):
            g = lambda n: A.unit_mean(R, "split", f"{n}|none|{met}")[0]
            inter = (g("reprise") - g("static_plp")) - (g("mem_lpA") - g("static_lpA"))
            out["diffs"][f"interaction_alone_{met}"] = tci(inter)
            print(f"  interaction (memory effect with history LP - with alone LP) {met}: {A.fmt_ci(tci(inter))}")
    bins = [f"{lo}-{hi}" for lo, hi in A.BINS]
    print("== A1: miss rate (%) at the ID-95% threshold per appearance bin k = " + ", ".join(bins))
    for nm in [n for n in FAMS + ["rho", "static_rho_plp", "lpB", "lpA", "static_lpA", "mem_lpA"] if f"{n}|1-1|miss" in RB[0]]:
        out["bins"][nm] = {b: {"miss": split_ci(RB, f"{nm}|{b}|miss"), "auroc": split_ci(RB, f"{nm}|{b}|auroc")} for b in bins}
        print(f"  {nm:15s} miss  " + "  ".join(f"{out['bins'][nm][b]['miss']['mean']:6.2f}" for b in bins))
        print(f"  {'':15s} auroc " + "  ".join(f"{out['bins'][nm][b]['auroc']['mean']:6.2f}" for b in bins))
    out["bins"]["n_per_stream"] = {b: float(np.mean([r[f"n|{b}"] for r in RB])) for b in bins}
    print("  images per stream in each bin:", out["bins"]["n_per_stream"])
    for a_, b_ in (("reprise", "static"), ("reprise", "zeta"), ("mem", "static"), ("plp", "static")):
        d = {b: diff_ci(RB, f"{a_}|{b}|miss", f"{b_}|{b}|miss") for b in bins}
        da = {b: diff_ci(RB, f"{a_}|{b}|auroc", f"{b_}|{b}|auroc") for b in bins}
        out["bins"][f"{a_}-{b_}"] = {"miss": d, "auroc": da}
        print(f"  {a_} - {b_}: miss " + "  ".join(A.fmt_ci(d[b]) for b in bins[:4]))
        print(f"  {'':18s} auroc " + "  ".join(A.fmt_ci(da[b]) for b in bins[:4]))
    print("== A1b: first-appearance miss rate by arrival position (rows 0-1023 / 1024-4095 / 4096+); images per stream:",
          {t_: float(np.nanmean([r[f"n|k1pos|{t_}"] for r in RB])) for t_ in ("b0-3", "b4-15", "b16+")})
    out["k1pos"] = {}
    for nm in ("static", "reprise", "zeta", "static_plp"):
        vals = {}
        for t_ in ("b0-3", "b4-15", "b16+"):
            x = [r[f"{nm}|k1pos|{t_}"] for r in RB if not np.isnan(r[f"{nm}|k1pos|{t_}"])]
            vals[t_] = float(np.mean(x)) if x else float("nan")
        out["k1pos"][nm] = vals
        print(f"  {nm:12s} " + "  ".join(f"{t_} {v:6.2f}" for t_, v in vals.items()))
    print("== C2: share of OOD detected by p_M only / p_LP only / both / neither (each at its own ID-95% threshold)")
    for b in ["all"] + bins:
        out["overlap"][b] = {c: split_ci(RC, f"{c}|{b}") for c in ("mem_only", "lp_only", "both", "neither")}
        print(f"  k {b:6s} " + "  ".join(f"{c} {out['overlap'][b][c]['mean']:6.2f}" for c in ("mem_only", "lp_only", "both", "neither")))
    print("== C2: Spearman correlation of the log ranks within ID / within OOD")
    for nm in ("mem_plp", "rho_plp", "static_plp", "static_rho", "static_mem"):
        out["corr"][nm] = {"id": split_ci(RC, f"rho_id|{nm}"), "ood": split_ci(RC, f"rho_ood|{nm}")}
        print(f"  {nm:12s} ID {out['corr'][nm]['id']['mean']:+.3f}  OOD {out['corr'][nm]['ood']['mean']:+.3f}")
    print("== E4: ID false alarms / OOD misses per stream decile")
    for nm in ("static", "zeta", "reprise", "mem", "plp"):
        out["time"][nm] = {"fa": [split_ci(RT, f"{nm}|fa|{j}")["mean"] for j in range(10)], "miss": [split_ci(RT, f"{nm}|miss|{j}")["mean"] for j in range(10)]}
        print(f"  {nm:8s} FA   " + " ".join(f"{v:5.2f}" for v in out["time"][nm]["fa"]))
        print(f"  {'':8s} miss " + " ".join(f"{v:5.2f}" for v in out["time"][nm]["miss"]))
    A.dumpj(A.RESULTS / f"an_{tag}.json", out)
    print("written", A.RESULTS / f"an_{tag}.json")


if __name__ == "__main__":
    a = sys.argv[1:]
    main(a[0] if a else "u1std", a[1].split(",") if len(a) > 1 else VIEWS, a[2] if len(a) > 2 else "u1")
