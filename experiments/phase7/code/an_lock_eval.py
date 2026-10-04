"""Amendment 02, evaluation after the lock (once): the locked candidate against the frozen REPRISE on the standard
U1 and U4 streams. Unit: split (9 streams averaged), paired 95% t interval (df = 4)."""
import json

import numpy as np
from scipy.stats import rankdata

import an7 as A
import an_cases as C
from metrics_p4 import metrics, tci
from p7common import P7, sha_file

VIEWS = ["B14", "L14"]
lock_f = P7 / "selection_lock_p7.json"
assert sha_file(lock_f) == (P7 / "selection_lock_p7.sha256").read_text().split()[0], "lock changed"
SPEC = tuple(json.loads(lock_f.read_text())["spec"])
EXTRA = {"within_only": ("M@0", 0.0, "lp0", 1.0), "zero_only": ("M", 0.0, "lp", 1.0), "hinge2_zero": ("Mh2", 0.0, "lp", 1.0)}     # descriptive decomposition
out = {"spec": list(SPEC)}
for exp in ("u1w", "u4w"):
    ts = A.tasks(exp, VIEWS[0])
    assert len(ts) == 45 and all((A.RESULTS / exp / VIEWS[1] / f"{t}.npz").exists() for t in ts), (exp, len(ts))
    rows = []
    for t in ts:
        s = A.Stream(exp, t, VIEWS)
        o = s.is_ood
        k = s.appearance()
        case, _, _ = C.cases(s)
        row = dict(A.parse_std(t))
        sc = {"frozen": s.family("reprise"), "cand": s.cand(SPEC), "static": s.family("static"), **{n: s.cand(sp) for n, sp in EXTRA.items()}}
        for nm, x in sc.items():
            for base in ("none", "TINS", "MCM"):
                m = metrics(x + s.base(base), o)
                row[f"{nm}|{base}|AUROC"], row[f"{nm}|{base}|FPR95"] = m["AUROC"], m["FPR95"]
            for b, v in A.by_bin(x, o, k).items():
                row[f"{nm}|miss_{b}"], row[f"{nm}|auroc_{b}"] = v["miss"], v["auroc"]
            thr = A.threshold(x, o)
            r = rankdata(x)
            for c in C.CASES:
                sel = o & (case == c)
                row[f"{nm}|case_{c}|miss"] = 100 * float(np.mean(x[sel] >= thr)) if sel.any() else np.nan
                row[f"{nm}|case_{c}|AUROC"] = C.auroc_subset(r, o, sel) if sel.any() else np.nan
        rows.append(row)
    g = lambda key: np.array([np.nanmean([r[key] for r in rows if r["split"] == u]) for u in sorted({r["split"] for r in rows})])
    res = {"n_streams": len(rows)}
    print(f"##### {exp}: {len(rows)} streams; candidate {SPEC}")
    for base in ("none", "TINS", "MCM"):
        for met in ("AUROC", "FPR95"):
            res[f"{base}|{met}"] = {"frozen": tci(g(f"frozen|{base}|{met}")), "cand": tci(g(f"cand|{base}|{met}")),
                                    "diff": tci(g(f"cand|{base}|{met}") - g(f"frozen|{base}|{met}")),
                                    **{n: tci(g(f"{n}|{base}|{met}") - g(f"frozen|{base}|{met}")) for n in EXTRA}}
        a, f = res[f"{base}|AUROC"], res[f"{base}|FPR95"]
        print(f"  base {base:4s}: AUROC {a['frozen']['mean']:.2f} -> {a['cand']['mean']:.2f} ({A.fmt_ci(a['diff'])})   FPR95 {f['frozen']['mean']:.2f} -> {f['cand']['mean']:.2f} ({A.fmt_ci(f['diff'])})")
    print("  decomposition (standalone, difference to frozen): " + "; ".join(
        f"{n}: AUROC {res['none|AUROC'][n]['mean']:+.2f}, FPR95 {A.fmt_ci(res['none|FPR95'][n])}" for n in EXTRA))
    bins = [f"{lo}-{hi}" for lo, hi in A.BINS]
    res["bins"] = {b: {"frozen": tci(g(f"frozen|miss_{b}")), "cand": tci(g(f"cand|miss_{b}")), "diff": tci(g(f"cand|miss_{b}") - g(f"frozen|miss_{b}")),
                       "static": tci(g(f"static|miss_{b}")), "auroc_diff": tci(g(f"cand|auroc_{b}") - g(f"frozen|auroc_{b}"))} for b in bins}
    print("  miss rate per appearance bin  " + " ".join(f"{b:>7s}" for b in bins))
    for nm in ("frozen", "cand"):
        print(f"    {nm:8s}                    " + " ".join(f"{res['bins'][b][nm]['mean']:7.2f}" for b in bins))
    print("    difference [95% CI]: " + "  ".join(f"{b} {A.fmt_ci(res['bins'][b]['diff'])}" for b in bins[:4]))
    print("    AUROC of k=1 images: difference " + A.fmt_ci(res["bins"]["1-1"]["auroc_diff"]))
    res["cases"] = {c: {"frozen_miss": tci(g(f"frozen|case_{c}|miss")), "cand_miss": tci(g(f"cand|case_{c}|miss")),
                        "miss_diff": tci(g(f"cand|case_{c}|miss") - g(f"frozen|case_{c}|miss")),
                        "frozen_AUROC": tci(g(f"frozen|case_{c}|AUROC")), "cand_AUROC": tci(g(f"cand|case_{c}|AUROC")),
                        "AUROC_diff": tci(g(f"cand|case_{c}|AUROC") - g(f"frozen|case_{c}|AUROC"))} for c in C.CASES}
    print("  per case (R / C / N1 / N+): AUROC " + "  ".join(f"{c} {res['cases'][c]['frozen_AUROC']['mean']:.2f}->{res['cases'][c]['cand_AUROC']['mean']:.2f}" for c in C.CASES))
    print("                              miss  " + "  ".join(f"{c} {res['cases'][c]['frozen_miss']['mean']:.2f}->{res['cases'][c]['cand_miss']['mean']:.2f} ({A.fmt_ci(res['cases'][c]['miss_diff'])})" for c in C.CASES))
    if exp == "u1w":
        res["W1"] = bool(res["none|AUROC"]["diff"]["lo"] > 0)
        res["W2"] = bool(res["none|FPR95"]["diff"]["hi"] < 0)
        print(f"  criteria: W1 (AUROC lower bound > 0) {'met' if res['W1'] else 'NOT met'}; W2 (FPR95 upper bound < 0) {'met' if res['W2'] else 'NOT met'}")
    out[exp] = res
A.dumpj(A.RESULTS / "an_lock_eval.json", out)
print("written")
