"""Amendment 02: within-batch memory read-out ('@0': images admitted from the same batch count as memory members,
the image itself never does). Selection on dev1, one confirmation on dev2 (registered rule).

  an_within.py select    -> results/a2_dev1_selection.json
  an_within.py confirm   -> results/a2_dev2_confirm.json
Units: support draws (the three orders averaged), paired t interval (df = 4)."""
import json
import sys

import numpy as np

import an7 as A
import an_a_dev as D
from metrics_p4 import tci

MEMS = ["M", "M@0", "Mh4@0", "Mh3@0", "Mh2@0"]
LPS = ["lp0", "lp"]
CANDS = [(m, 0.0, l, 1.0) for m in MEMS for l in LPS]


def changed(spec):
    m, _, l, _ = spec
    return int(m != "M") + int(m.startswith("Mh")) + int(l != "lp0")


def select():
    res = D.evaluate("dev1", CANDS, exp="devw")
    fz = res[D.name(A.FROZEN)]
    table = []
    for sp in CANDS:
        o = res[D.name(sp)]
        d = {k: tci(o[k] - fz[k]) for k in ("near_AUROC", "near_FPR95", "far_FPR95", "far_AUROC", "nearT_FPR95", "nearT_AUROC", "F1")}
        feas = d["near_FPR95"]["mean"] <= 0.0 and d["far_FPR95"]["mean"] <= 0.5 and d["nearT_FPR95"]["mean"] <= 0.0
        table.append({"spec": list(sp), "name": D.name(sp), "changed": changed(sp), "feasible": bool(feas), **D.summary(o), "d": d})
    feas = [r for r in table if r["feasible"]]
    best = max(r["near_AUROC"] for r in feas)
    order = lambda r: (r["changed"], MEMS.index(r["spec"][0]))
    chosen = sorted([r for r in feas if r["near_AUROC"] >= best - 0.03], key=order)[0]
    gain = chosen["near_AUROC"] - float(np.mean(fz["near_AUROC"]))
    sel = chosen["name"] != D.name(A.FROZEN) and gain >= 0.05
    A.dumpj(A.RESULTS / "a2_dev1_selection.json", {"dev": "dev1", "frozen": D.summary(fz), "chosen": chosen, "AUROC_gain": gain, "selected": bool(sel), "table": table,
                                                   "references": {r: D.summary(res[r]) for r in D.REFS}})
    cols = ("near_AUROC", "near_FPR95", "nearT_AUROC", "nearT_FPR95", "far_AUROC", "far_FPR95", "F1", "miss_2-2", "miss_3-5", "miss_6-10", "miss_11-20", "miss_31-50")
    print("dev1: " + "  ".join(cols))
    for r in table:
        print(f"{r['name']:18s} " + "  ".join(f"{r[c]:6.2f}" for c in cols) + f" | feas {int(r['feasible'])} ch {r['changed']} dAUROC {A.fmt_ci(r['d']['near_AUROC'])} "
              f"dFPR95 {A.fmt_ci(r['d']['near_FPR95'])} dfar {r['d']['far_FPR95']['mean']:+.2f} dnearT {r['d']['nearT_FPR95']['mean']:+.2f} dF1 {A.fmt_ci(r['d']['F1'])}")
    print("CHOSEN", chosen["name"], "AUROC gain", round(gain, 3), "selected" if sel else "NOT selected (rule)")


def confirm():
    selj = json.loads((A.RESULTS / "a2_dev1_selection.json").read_text())
    assert selj["selected"]
    spec = tuple(selj["chosen"]["spec"])
    res = D.evaluate("dev2", [A.FROZEN, spec], exp="devw")
    fz, c = res[D.name(A.FROZEN)], res[D.name(spec)]
    d = {k: tci(c[k] - fz[k]) for k in c}
    ok = d["near_AUROC"]["lo"] > 0 and d["near_FPR95"]["hi"] < 0 and d["far_FPR95"]["mean"] <= 0.5
    A.dumpj(A.RESULTS / "a2_dev2_confirm.json", {"dev": "dev2", "spec": list(spec), "frozen": D.summary(fz), "candidate": D.summary(c), "difference": d, "confirmed": bool(ok),
                                                 "criteria": "near AUROC difference lower bound > 0; near FPR95 difference upper bound < 0; far FPR95 difference <= +0.5"})
    for k in ("near_AUROC", "near_FPR95", "nearT_AUROC", "nearT_FPR95", "far_AUROC", "far_FPR95", "farT_FPR95", "F1", "A1", "miss_2-2", "miss_3-5", "miss_6-10", "miss_11-20", "miss_21-30", "miss_31-50"):
        print(f"{k:12s} frozen {np.mean(fz[k]):6.2f}  candidate {np.mean(c[k]):6.2f}  diff {A.fmt_ci(d[k])}")
    print("CONFIRMED" if ok else "NOT CONFIRMED", D.name(spec))


if __name__ == "__main__":
    select() if sys.argv[1] == "select" else confirm()
