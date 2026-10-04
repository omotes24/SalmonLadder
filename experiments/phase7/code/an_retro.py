"""A (post hoc on dev1, then descriptive on U1 / U4 per amendment 01): revised scores after a delay.
The issued score of an image is never changed; '@delay' re-scores it with the graph and memory `delay` batches later
(same frozen components, same calibration ranks). Every image (ID and OOD) is re-scored at the same delay, so the
ID-95% threshold is that of the re-scored ID images.
usage: an_retro.py <exp: devr|u1r|u4r> <unit: draw|split> [pattern]"""
import sys

import numpy as np

import an7 as A
from metrics_p4 import metrics, tci

VIEWS = ["B14", "L14"]
DELAYS = ["0", "1", "2", "5", "10", "20", "end"]
READ = {"reprise": ("M", "lp0"), "hinge2_zero": ("Mh2", "lp"), "mem": ("M", None), "plp": (None, "lp0")}


def score(s, fam, dl):
    mk, lk = READ[fam]
    suf = "" if dl == "0" else f"@{dl}"
    x = 0.0
    if mk:
        x = x + s.s(mk + suf)
    if lk:
        x = x + s.s(lk + suf)
    return x


def main(exp, unit, pattern="*"):
    ts = A.tasks(exp, VIEWS[0], pattern)
    ts = [t for t in ts if (A.RESULTS / exp / VIEWS[1] / f"{t}.npz").exists()]
    rows = []
    for t in ts:
        info = A.parse_dev(t) if exp.startswith("dev") else A.parse_std(t)
        if exp.startswith("dev") and info["stream"] != "near":
            continue
        s = A.Stream(exp, t, VIEWS)
        k = s.appearance()
        row = dict(info)
        st = s.s("static")
        m = metrics(st, s.is_ood)
        bb = A.by_bin(st, s.is_ood, k)
        row["static|FPR95"], row["static|AUROC"], row["static|F1"], row["static|A1"] = m["FPR95"], m["AUROC"], bb["1-1"]["miss"], bb["1-1"]["auroc"]
        for fam in READ:
            for dl in DELAYS:
                x = score(s, fam, dl)
                m = metrics(x, s.is_ood)
                bb = A.by_bin(x, s.is_ood, k)
                row[f"{fam}|{dl}|FPR95"], row[f"{fam}|{dl}|AUROC"] = m["FPR95"], m["AUROC"]
                for b, v in bb.items():
                    row[f"{fam}|{dl}|miss_{b}"], row[f"{fam}|{dl}|auroc_{b}"] = v["miss"], v["auroc"]
        rows.append(row)
    g = lambda key: A.unit_mean(rows, unit, key)[0]
    out = {"exp": exp, "unit": unit, "n_streams": len(rows), "static": {m: tci(g(f"static|{m}")) for m in ("FPR95", "AUROC", "F1", "A1")}, "fam": {}}
    bins = [f"{lo}-{hi}" for lo, hi in A.BINS]
    print(f"##### {exp}: {len(rows)} streams; static p: FPR95 {out['static']['FPR95']['mean']:.2f} AUROC {out['static']['AUROC']['mean']:.2f} "
          f"first-appearance miss {out['static']['F1']['mean']:.2f} AUROC(k=1) {out['static']['A1']['mean']:.2f}")
    for fam in READ:
        out["fam"][fam] = {}
        print(f"== {fam}: delay | AUROC  FPR95 | miss rate per appearance bin " + " ".join(f"{b:>6s}" for b in bins) + " | AUROC(k=1) | k=1 miss: delayed - issued")
        for dl in DELAYS:
            rec = {"AUROC": tci(g(f"{fam}|{dl}|AUROC")), "FPR95": tci(g(f"{fam}|{dl}|FPR95")),
                   "miss": {b: tci(g(f"{fam}|{dl}|miss_{b}")) for b in bins}, "auroc_bin": {b: tci(g(f"{fam}|{dl}|auroc_{b}")) for b in bins},
                   "dF1": tci(g(f"{fam}|{dl}|miss_1-1") - g(f"{fam}|0|miss_1-1")), "dFPR95": tci(g(f"{fam}|{dl}|FPR95") - g(f"{fam}|0|FPR95")),
                   "F1_vs_static": tci(g(f"{fam}|{dl}|miss_1-1") - g("static|F1"))}
            out["fam"][fam][dl] = rec
            print(f"   {dl:>4s} | {rec['AUROC']['mean']:6.2f} {rec['FPR95']['mean']:6.2f} | " + " ".join(f"{rec['miss'][b]['mean']:6.2f}" for b in bins) +
                  f" | {rec['auroc_bin']['1-1']['mean']:6.2f} | {A.fmt_ci(rec['dF1'])}")
    A.dumpj(A.RESULTS / f"an_retro_{exp}.json", out)


if __name__ == "__main__":
    main(*sys.argv[1:])
