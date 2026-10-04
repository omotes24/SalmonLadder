"""E: design space of stream conditions on U1 (frozen components; descriptive). Unit: split (seeds averaged).

  e1  composition: U unknown classes x m images each, n_id ID images, random order
  e2  arrival patterns at two compositions
  e3  batch size x pattern
Families: static p, zeta, memory only, static x p_LP, REPRISE and the locked extension (amendment 02; 'ext').
Reads the '<exp>w' runs, which hold every frozen read-out plus the within-batch memory read-out."""
import json
import sys
from pathlib import Path

import numpy as np

import an7 as A
from metrics_p4 import metrics, tci

VIEWS = ["B14", "L14"]
FAMS = ["static", "zeta", "mem", "plp", "static_plp", "reprise"]


def cell_rows(exp, cand=None):
    rows = []
    for t in A.tasks(exp, VIEWS[0]):
        if not (A.RESULTS / exp / VIEWS[1] / f"{t}.npz").exists():
            continue
        s = A.Stream(exp, t, VIEWS)
        c = json.loads(str(s.z[VIEWS[0]]["meta"]))["cell"]
        row = dict(c)
        row["n_ood"], row["n"] = int(s.is_ood.sum()), int(len(s.is_ood))
        row["pi"] = 100 * row["n_ood"] / row["n"]
        fams = {f: s.family(f) for f in FAMS}
        if cand is not None:
            fams["ext"] = s.cand(cand)
        for f, x in fams.items():
            for base in ("none", "MCM"):
                m = metrics(x + s.base(base), s.is_ood)
                row[f"{f}|{base}|FPR95"], row[f"{f}|{base}|AUROC"] = m["FPR95"], m["AUROC"]
        row["adm_id_L14"] = 100 * float(s.z["L14"]["admit"][:, 2][~s.is_ood].mean())
        row["adm_ood_L14"] = 100 * float(s.z["L14"]["admit"][:, 2][s.is_ood].mean())
        rows.append(row)
    return rows


def table(rows, keys, title, fams, cand=False):
    cells = sorted({tuple(r[k] for k in keys) for r in rows})
    out = []
    print(f"== {title}: AUROC / FPR95 (standalone); differences in FPR95 [95% CI over splits]")
    hdr = " ".join(f"{k:>8s}" for k in keys)
    print(f"{hdr} | OOD%   n | " + " | ".join(f"{f:13s}" for f in fams) + " | REPRISE-static             | REPRISE-zeta               | ext-REPRISE                | splits")
    for c in cells:
        rr = [r for r in rows if tuple(r[k] for k in keys) == c]
        g = lambda key: A.unit_mean(rr, "split", key)[0]
        rec = {"cell": dict(zip(keys, c)), "pi": float(np.mean([r["pi"] for r in rr])), "n": float(np.mean([r["n"] for r in rr])),
               "n_ood": float(np.mean([r["n_ood"] for r in rr])), "n_splits": len({r["split"] for r in rr}), "n_runs": len(rr),
               "adm_id_L14": float(np.mean([r["adm_id_L14"] for r in rr])), "adm_ood_L14": float(np.mean([r["adm_ood_L14"] for r in rr]))}
        for f in fams:
            for base in ("none", "MCM"):
                rec[f"{f}|{base}"] = {m: tci(g(f"{f}|{base}|{m}")) for m in ("AUROC", "FPR95")}
        for a_, b_ in (("reprise", "static"), ("reprise", "zeta"), ("reprise", "static_plp"), ("reprise", "mem"), ("zeta", "static")) + \
                ((("ext", "reprise"), ("ext", "static"), ("ext", "zeta")) if cand else ()):
            rec[f"{a_}-{b_}"] = {m: tci(g(f"{a_}|none|{m}") - g(f"{b_}|none|{m}")) for m in ("AUROC", "FPR95")}
        out.append(rec)
        print(" ".join(f"{str(v):>8s}" for v in c) + f" | {rec['pi']:5.2f} {rec['n']:6.0f} | " +
              " | ".join(f"{rec[f + '|none']['AUROC']['mean']:5.2f}/{rec[f + '|none']['FPR95']['mean']:6.2f}" for f in fams) +
              f" | {A.fmt_ci(rec['reprise-static']['FPR95']):26s} | {A.fmt_ci(rec['reprise-zeta']['FPR95']):26s} | "
              f"{(A.fmt_ci(rec['ext-reprise']['FPR95']) if cand else ''):26s} | {rec['n_splits']}")
    return out


if __name__ == "__main__":
    cand = A.locked_spec()
    sfx = "w" if cand else ""
    fams = FAMS + (["ext"] if cand else [])
    res = {"extension": list(cand) if cand else None, "source": f"e1{sfx} / e2{sfx} / e3{sfx}"}
    for exp, keys, title in (("e1", ["n_id", "U", "m"], "E1 composition (random order, batch 256)"),
                             ("e2", ["comp", "pattern"], "E2 arrival patterns"),
                             ("e3", ["pattern", "batch"], "E3 batch size")):
        rows = cell_rows(exp + sfx, cand)
        if rows:
            res[exp] = table(rows, keys, title, fams, cand is not None)
    A.dumpj(A.RESULTS / "an_e.json", res)
    print("written")
