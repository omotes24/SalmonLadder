"""Experiment A, selection on dev1 and confirmation on dev2 (registered rule; prereg_p7.json, A2).

  an_a_dev.py select            evaluate every candidate on dev1, apply the rule, write results/a_dev1_selection.json
  an_a_dev.py confirm           evaluate ONLY the candidate chosen on dev1 on dev2, write results/a_dev2_confirm_<n>.json
Units: support draws (the three orders averaged), paired t interval (df = 4)."""
import itertools
import json
import sys

import numpy as np

import an7 as A
from metrics_p4 import metrics, tci

VIEWS = ["B14", "L14"]
MEMS = ["M", "Mh4", "Mh3", "Mh2", "Mh1", "Mh0", "Mmin"]            # order = tie-break preference (closer to frozen first)
AS = [0.0, 0.5, 1.0]
LPS = ["lp0", "lp"]
TAUS = [1.0, 0.5, 0.2]
CANDS = [(m, a, l, t) for m in MEMS for a in AS for l in LPS for t in TAUS]
REFS = ["static", "mem", "plp", "zeta", "static_plp"]


def name(spec):
    m, a, l, t = spec
    return f"{m}|a{a:g}|{l}|t{t:g}"


def changed(spec):
    m, a, l, t = spec
    return int(m != "M") + int(a != 0) + int(l != "lp0") + int(t != 1.0)


def evaluate(dev, specs, refs=REFS, exp="dev"):
    """{name: {metric: per-draw array}} for the candidates and the reference families."""
    rows = {}

    def put(nm, draw, **kw):
        rows.setdefault(nm, []).append({"draw": draw, **kw})

    near = A.tasks(exp, VIEWS[0], f"{dev}_draw*_near_seed*")
    far = A.tasks(exp, VIEWS[0], f"{dev}_draw*_far_seed*")
    assert len(near) == 15 and len(far) == 15, (len(near), len(far))
    for t in near:
        info = A.parse_dev(t)
        s = A.Stream(exp, t, VIEWS)
        k = s.appearance()
        scores = {name(sp): s.cand(sp) for sp in specs}
        scores.update({r: s.family(r) for r in refs})
        for nm, x in scores.items():
            m, mt = metrics(x, s.is_ood), metrics(x + s.logS, s.is_ood)
            bb = A.by_bin(x, s.is_ood, k)
            put(nm, info["draw"], near_FPR95=m["FPR95"], near_AUROC=m["AUROC"], nearT_FPR95=mt["FPR95"], nearT_AUROC=mt["AUROC"],
                F1=bb["1-1"]["miss"], A1=bb["1-1"]["auroc"], **{f"miss_{b}": v["miss"] for b, v in bb.items()})
    far_rows = {}
    for t in far:
        info = A.parse_dev(t)
        s = A.Stream(exp, t, VIEWS)
        scores = {name(sp): s.cand(sp) for sp in specs}
        scores.update({r: s.family(r) for r in refs})
        for nm, x in scores.items():
            m, mt = metrics(x, s.is_ood), metrics(x + s.logS, s.is_ood)
            far_rows.setdefault(nm, []).append({"draw": info["draw"], "far_FPR95": m["FPR95"], "far_AUROC": m["AUROC"], "farT_FPR95": mt["FPR95"]})
    out = {}
    for nm in rows:
        o = {}
        for key in rows[nm][0]:
            if key != "draw":
                o[key] = A.unit_mean(rows[nm], "draw", key)[0]
        for key in ("far_FPR95", "far_AUROC", "farT_FPR95"):
            o[key] = A.unit_mean(far_rows[nm], "draw", key)[0]
        out[nm] = o
    return out


def summary(o):
    return {k: float(np.mean(v)) for k, v in o.items()}


def select():
    res = evaluate("dev1", CANDS)
    fz = res[name(A.FROZEN)]
    table = []
    for sp in CANDS:
        o = res[name(sp)]
        d = {k: tci(o[k] - fz[k]) for k in ("F1", "near_FPR95", "far_FPR95", "nearT_FPR95", "near_AUROC")}
        feas = (d["near_FPR95"]["mean"] <= 0.30 and d["far_FPR95"]["mean"] <= 0.50 and d["nearT_FPR95"]["mean"] <= 0.30)
        table.append({"spec": list(sp), "name": name(sp), "changed": changed(sp), "feasible": bool(feas), **summary(o),
                      "d": {k: v for k, v in d.items()}})
    feas = [r for r in table if r["feasible"]]
    best_f1 = min(r["F1"] for r in feas)
    near_best = [r for r in feas if r["F1"] <= best_f1 + 0.5]
    order = lambda r: (r["changed"], MEMS.index(r["spec"][0]), -r["spec"][3], r["spec"][1])
    chosen = sorted(near_best, key=order)[0]
    gain = float(np.mean(fz["F1"]) - chosen["F1"])
    locked = chosen["name"] != name(A.FROZEN) and gain >= 1.0
    out = {"dev": "dev1", "n_candidates": len(CANDS), "frozen": summary(fz), "references": {r: summary(res[r]) for r in REFS},
           "best_feasible_F1": best_f1, "chosen": chosen, "F1_gain_vs_frozen": gain, "selected": bool(locked), "table": table}
    A.dumpj(A.RESULTS / "a_dev1_selection.json", out)
    cols = ("F1", "near_FPR95", "near_AUROC", "nearT_FPR95", "far_FPR95", "miss_2-2", "miss_3-5", "miss_6-10", "miss_11-20", "miss_31-50")
    print("dev1 (standalone unless T): " + "  ".join(cols))
    print(f"{'frozen':22s} " + "  ".join(f"{np.mean(fz[c]):6.2f}" for c in cols))
    for r in REFS:
        print(f"{r:22s} " + "  ".join(f"{np.mean(res[r][c]):6.2f}" for c in cols))
    print("--- candidates sorted by F1 (feasible first); d = candidate - frozen")
    for r in sorted(table, key=lambda r: (not r["feasible"], r["F1"]))[:45]:
        print(f"{r['name']:22s} " + "  ".join(f"{r[c]:6.2f}" for c in cols) + f" | feas {int(r['feasible'])} ch {r['changed']} "
              f"dF1 {A.fmt_ci(r['d']['F1'])} dnear {A.fmt_ci(r['d']['near_FPR95'])} dfar {r['d']['far_FPR95']['mean']:+.2f} dnearT {r['d']['nearT_FPR95']['mean']:+.2f}")
    print("CHOSEN", chosen["name"], "F1 gain", round(gain, 2), "selected" if locked else "NOT selected (rule)")


def confirm(attempt):
    sel = json.loads((A.RESULTS / "a_dev1_selection.json").read_text())
    assert sel["selected"], "no candidate was selected on dev1"
    spec = tuple(sel["chosen"]["spec"])
    res = evaluate("dev2", [A.FROZEN, spec])
    fz, c = res[name(A.FROZEN)], res[name(spec)]
    d = {k: tci(c[k] - fz[k]) for k in c}
    ok = d["F1"]["hi"] < 0 and d["near_FPR95"]["mean"] <= 0.30 and d["near_FPR95"]["hi"] < 1.0
    out = {"dev": "dev2", "attempt": attempt, "spec": list(spec), "frozen": summary(fz), "candidate": summary(c), "difference": d,
           "references": {r: summary(res[r]) for r in REFS}, "confirmed": bool(ok),
           "criteria": "F1 difference upper bound < 0; near FPR95 difference point <= +0.30 and upper bound < +1.0"}
    A.dumpj(A.RESULTS / f"a_dev2_confirm_{attempt}.json", out)
    for k in ("F1", "A1", "near_FPR95", "near_AUROC", "nearT_FPR95", "far_FPR95", "farT_FPR95", "miss_2-2", "miss_3-5", "miss_6-10", "miss_11-20", "miss_21-30", "miss_31-50"):
        print(f"{k:12s} frozen {np.mean(fz[k]):6.2f}  candidate {np.mean(c[k]):6.2f}  diff {A.fmt_ci(d[k])}   static {np.mean(res['static'][k]):6.2f}")
    print("CONFIRMED" if ok else "NOT CONFIRMED", name(spec))


if __name__ == "__main__":
    if sys.argv[1] == "select":
        select()
    else:
        confirm(int(sys.argv[2]) if len(sys.argv) > 2 else 1)
