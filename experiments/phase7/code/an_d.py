"""D: dataset dependence (frozen configuration; descriptive). Natural streams and the matched composition.
Unit: split (the three orders averaged); SSB CUB has one split, so its unit is the order."""
import json
import re
import sys

import numpy as np

import an7 as A
from metrics_p4 import metrics, tci

VIEWS = ["B14", "L14"]
FAMS = ["static", "zeta", "mem", "plp", "static_plp", "reprise"]
ORDER = ["U1", "insk", "inr", "places365", "cub", "cifar100", "cubssb"]


def parse(t):
    m = re.match(r"U1_s(\d+)_d0_mat_seed(\d+)$", t)
    if m:
        return {"ds": "U1", "split": int(m.group(1)), "cond": "mat", "seed": int(m.group(2))}
    m = re.match(r"cubssb_(\w+)_seed(\d+)$", t)
    if m:
        return {"ds": f"cubssb-{m.group(1)}", "split": int(m.group(2)), "cond": "nat", "seed": int(m.group(2))}
    m = re.match(r"(\w+?)_s(\d+)_(nat|mat)_seed(\d+)$", t)
    return {"ds": m.group(1), "split": int(m.group(2)), "cond": m.group(3), "seed": int(m.group(4))}


def main():
    rows = []
    cand = A.locked_spec()
    EXP = "dsw" if cand else "ds"
    for t in A.tasks(EXP, VIEWS[0]):
        if not (A.RESULTS / EXP / VIEWS[1] / f"{t}.npz").exists():
            continue
        s = A.Stream(EXP, t, VIEWS)
        row = parse(t)
        o = s.is_ood
        oc, cnt = np.unique(s.cls[o], return_counts=True)
        row.update(n=len(o), n_ood=int(o.sum()), pi=100 * float(o.mean()), U=len(oc), m=float(np.median(cnt)),
                   C=json.loads(str(s.z[VIEWS[0]]["meta"]))["C"])
        row["adm_ood"] = 100 * float((s.z["B14"]["admit"][:, 2] | s.z["L14"]["admit"][:, 2])[o].mean())
        row["adm_id"] = 100 * float(s.z["L14"]["admit"][:, 2][~o].mean())
        fams = {f: s.family(f) for f in FAMS}
        if cand:
            fams["ext"] = s.cand(cand)
        bf = A.RESULTS / "ds_base" / f"{t}.npz"
        if bf.exists():
            b = np.load(bf)
            fams["knn"], fams["maha"] = b["knn"], b["maha"]
        for f, x in fams.items():
            for base in ("none", "MCM") + (("TINS",) if s.logS is not None else ()):
                m = metrics(x + s.base(base), o)
                row[f"{f}|{base}|FPR95"], row[f"{f}|{base}|AUROC"] = m["FPR95"], m["AUROC"]
        m = metrics(s.logM, o)
        row["mcm|FPR95"], row["mcm|AUROC"] = m["FPR95"], m["AUROC"]
        k = s.appearance()
        bb = A.by_bin(fams["reprise"], o, k)
        bs = A.by_bin(fams["static"], o, k)
        row["rep_miss_k1"], row["static_miss_k1"] = bb["1-1"]["miss"], bs["1-1"]["miss"]
        rows.append(row)
    out = {"rows": []}
    for cond in ("nat", "mat"):
        print(f"##### condition: {'natural streams' if cond == 'nat' else 'matched composition (40 unknown classes x 25, 4,000 ID)'}")
        print(f"{'dataset':16s} |   C    U     m   OOD%      n | MCM           | kNN           | Maha++        | static p      | zeta          | memory        | REPRISE       | "
              f"REPRISE x MCM | ext           | REPRISE - zeta (FPR95)     | REPRISE - static (FPR95)   | ext - REPRISE (FPR95)      | units")
        names = sorted({r["ds"] for r in rows if r["cond"] == cond}, key=lambda d: (ORDER.index(d.split("-")[0]) if d.split("-")[0] in ORDER else 99, d))
        for d in names:
            rr = [r for r in rows if r["ds"] == d and r["cond"] == cond]
            g = lambda key: A.unit_mean(rr, "split", key)[0]
            rec = {"ds": d, "cond": cond, "units": len({r["split"] for r in rr}), "n_runs": len(rr)}
            for key in ("C", "U", "m", "pi", "n", "n_ood", "rep_miss_k1", "static_miss_k1", "adm_ood", "adm_id"):
                rec[key] = float(np.mean([r[key] for r in rr]))
            fams = [f for f in FAMS + ["knn", "maha", "ext"] if f"{f}|none|FPR95" in rr[0]]
            for f in fams:
                for base in ("none", "MCM", "TINS"):
                    if f"{f}|{base}|FPR95" in rr[0]:
                        rec[f"{f}|{base}"] = {m: tci(g(f"{f}|{base}|{m}")) for m in ("AUROC", "FPR95")}
            rec["mcm"] = {m: tci(g(f"mcm|{m}")) for m in ("AUROC", "FPR95")}
            for a_, b_ in (("reprise", "zeta"), ("reprise", "static"), ("reprise", "static_plp"), ("reprise", "mem"), ("zeta", "static")) + \
                    ((("ext", "reprise"), ("ext", "zeta")) if "ext|none" in rec else ()):
                rec[f"{a_}-{b_}"] = {m: tci(g(f"{a_}|none|{m}") - g(f"{b_}|none|{m}")) for m in ("AUROC", "FPR95")}
            out["rows"].append(rec)
            c = lambda key: f"{rec[key]['AUROC']['mean']:5.2f}/{rec[key]['FPR95']['mean']:6.2f}" if key in rec else " " * 12
            print(f"{d:16s} | {rec['C']:4.0f} {rec['U']:4.0f} {rec['m']:5.0f} {rec['pi']:6.2f} {rec['n']:6.0f} | {c('mcm')}  | {c('knn|none')}  | {c('maha|none')}  | "
                  f"{c('static|none')}  | {c('zeta|none')}  | {c('mem|none')}  | {c('reprise|none')}  | {c('reprise|MCM')}  | {c('ext|none')}  | "
                  f"{A.fmt_ci(rec['reprise-zeta']['FPR95']):26s} | {A.fmt_ci(rec['reprise-static']['FPR95']):26s} | "
                  f"{(A.fmt_ci(rec['ext-reprise']['FPR95']) if 'ext-reprise' in rec else ''):26s} | {rec['units']}")
    A.dumpj(A.RESULTS / "an_d.json", out)
    print("written", A.RESULTS / "an_d.json")


if __name__ == "__main__":
    main()
