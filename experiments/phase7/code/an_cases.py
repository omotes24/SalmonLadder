"""Case analysis of the OOD images by the recurrence evidence that exists when they arrive (descriptive; labels are
used to DEFINE the cases, so this is a diagnosis, not a detector).

  R   an image of the same unknown class from an EARLIER batch is in the memory M (at least one view)
  C   not R, but an image of the same class in the SAME batch is admitted to M (co-arrival)
  N1  neither, and it is the first image of its class (true first appearance)
  N+  neither, although earlier images of the class exist (none of them was admitted)
Per case: share of the OOD images, miss rate at the stream's ID-95% threshold and AUROC against all ID images, for the
issued scores and (when present) the re-scored ones. AUROC against all ID is linear in the OOD subsets, so
sum_cases share x AUROC_case = overall AUROC (checked).
usage: an_cases.py <exp> <unit> [pattern] [tag]"""
import sys

import numpy as np
from scipy.stats import rankdata

import an7 as A
from metrics_p4 import metrics, tci

VIEWS = ["B14", "L14"]
CASES = ["R", "C", "N1", "N+"]


def cases(s):
    o, cls, b = s.is_ood, s.cls, s.bidx
    n = len(o)
    prev = np.zeros(n, bool)
    co = np.zeros(n, bool)
    co_any = np.zeros(n, bool)
    for v in s.views:
        mem = s.z[v]["admit"][:, 2]
        first = {}                                   # class -> batch of its first admitted image
        cnt = {}                                     # (class, batch) -> admitted images
        for i in np.flatnonzero(o & mem):
            first.setdefault(cls[i], b[i])
            cnt[(cls[i], b[i])] = cnt.get((cls[i], b[i]), 0) + 1
        for i in np.flatnonzero(o):
            f = first.get(cls[i])
            if f is not None and f < b[i]:
                prev[i] = True
            if cnt.get((cls[i], b[i]), 0) - int(mem[i]) > 0:
                co[i] = True
    k = s.appearance()
    tot = {}
    for i in np.flatnonzero(o):
        tot[(cls[i], b[i])] = tot.get((cls[i], b[i]), 0) + 1
    for i in np.flatnonzero(o):
        co_any[i] = tot[(cls[i], b[i])] > 1
    case = np.full(n, "", dtype=object)
    case[o & prev] = "R"
    case[o & ~prev & co] = "C"
    case[o & ~prev & ~co & (k == 1)] = "N1"
    case[o & ~prev & ~co & (k > 1)] = "N+"
    return case, k, co_any


def auroc_subset(r_all, o, sel):
    """AUROC of the OOD subset `sel` against all ID images, from the ranks of all scores."""
    n_id = int((~o).sum())
    rid = np.sort(r_all[~o])
    below = np.searchsorted(rid, r_all[sel], side="left")       # ID images ranked strictly below each OOD image
    ties = np.searchsorted(rid, r_all[sel], side="right") - below
    return 100 * float(np.mean(1.0 - (below + 0.5 * ties) / n_id))


def main(exp, unit, pattern="*", tag=None):
    ts = [t for t in A.tasks(exp, VIEWS[0], pattern) if (A.RESULTS / exp / VIEWS[1] / f"{t}.npz").exists()]
    rows = []
    for t in ts:
        info = A.parse_dev(t) if exp.startswith("dev") else A.parse_std(t)
        if exp.startswith("dev") and info["stream"] != "near":
            continue
        s = A.Stream(exp, t, VIEWS)
        o = s.is_ood
        case, k, co_any = cases(s)
        row = dict(info)
        fams = {"reprise": s.family("reprise"), "static": s.family("static"), "static_plp": s.family("static_plp"), "zeta": s.family("zeta"), "mem": s.family("mem")}
        z0 = s.z[VIEWS[0]].files
        for dl in ("0", "1", "5", "end"):
            if f"s::M@{dl}" in z0:
                fams[f"reprise@{dl}"] = s.s(f"M@{dl}") + s.s(f"lp0@{dl}")
        for c in CASES:
            row[f"share|{c}"] = 100 * float(np.mean(case[o] == c))
        row["share|k1"] = 100 * float(np.mean(k[o] == 1))
        row["k1_coarrival"] = 100 * float(np.mean(co_any[o & (k == 1)]))
        for v in s.views:
            mem = s.z[v]["admit"][:, 2]
            row[f"adm_ood|{v}"], row[f"adm_id|{v}"] = 100 * float(mem[o].mean()), 100 * float(mem[~o].mean())
            row[f"adm_k1|{v}"] = 100 * float(mem[o & (k == 1)].mean())
        both = s.z[VIEWS[0]]["admit"][:, 2] | s.z[VIEWS[1]]["admit"][:, 2]
        row["adm_ood|any"], row["adm_k1|any"] = 100 * float(both[o].mean()), 100 * float(both[o & (k == 1)].mean())
        for f, x in fams.items():
            m = metrics(x, o)
            row[f"{f}|all|AUROC"], row[f"{f}|all|FPR95"] = m["AUROC"], m["FPR95"]
            thr = A.threshold(x, o)
            r = rankdata(x)
            acc = 0.0
            for c in CASES:
                sel = o & (case == c)
                if sel.any():
                    row[f"{f}|{c}|miss"] = 100 * float(np.mean(x[sel] >= thr))
                    row[f"{f}|{c}|AUROC"] = auroc_subset(r, o, sel)
                    acc += row[f"{f}|{c}|AUROC"] * sel.sum() / o.sum()
                else:
                    row[f"{f}|{c}|miss"] = row[f"{f}|{c}|AUROC"] = np.nan
            assert abs(acc - m["AUROC"]) < 1e-6, (f, acc, m["AUROC"])
        rows.append(row)

    def g(key):
        us = sorted({r[unit] for r in rows})
        return np.array([np.nanmean([r[key] for r in rows if r[unit] == u]) for u in us])

    out = {"exp": exp, "n_streams": len(rows), "share": {c: tci(g(f"share|{c}")) for c in CASES}, "fam": {}}
    print(f"##### {exp} ({pattern}): {len(rows)} streams; cases by the memory evidence at arrival")
    print("share of the OOD images (%): " + "  ".join(f"{c} {out['share'][c]['mean']:.2f}" for c in CASES) +
          f"  | k=1 images {np.mean(g('share|k1')):.2f}, of which with a same-class image in the same batch {np.mean(g('k1_coarrival')):.1f}")
    out["admission"] = {key: float(np.mean(g(key))) for key in ("adm_ood|B14", "adm_ood|L14", "adm_ood|any", "adm_id|B14", "adm_id|L14", "adm_k1|B14", "adm_k1|L14", "adm_k1|any")}
    print("admission to M (%): " + "  ".join(f"{k_} {v:.1f}" for k_, v in out["admission"].items()))
    fams = [f for f in ("static", "static_plp", "zeta", "mem", "reprise", "reprise@0", "reprise@1", "reprise@5", "reprise@end") if f"{f}|all|AUROC" in rows[0]]
    print(f"{'detector':13s} | overall AUROC / FPR95 | AUROC per case " + " ".join(f"{c:>6s}" for c in CASES) + " | miss rate per case " + " ".join(f"{c:>6s}" for c in CASES))
    for f in fams:
        rec = {"AUROC": tci(g(f"{f}|all|AUROC")), "FPR95": tci(g(f"{f}|all|FPR95")),
               "case_AUROC": {c: float(np.mean(g(f"{f}|{c}|AUROC"))) for c in CASES}, "case_miss": {c: float(np.mean(g(f"{f}|{c}|miss"))) for c in CASES}}
        out["fam"][f] = rec
        print(f"{f:13s} | {rec['AUROC']['mean']:6.2f} / {rec['FPR95']['mean']:6.2f}      | " + " ".join(f"{rec['case_AUROC'][c]:6.2f}" for c in CASES) +
              "                     | " + " ".join(f"{rec['case_miss'][c]:6.2f}" for c in CASES))
    # ceiling: every case as good as case R of the same detector (issued REPRISE)
    sh = np.array([out["share"][c]["mean"] for c in CASES]) / 100
    a = np.array([out["fam"]["reprise"]["case_AUROC"][c] for c in CASES])
    out["ceiling"] = {"overall": float((sh * a).sum()), "k1_as_R": float((sh * np.where(np.array(CASES) == "N1", a[0], a)).sum()),
                      "all_as_R": float(a[0]), "N_as_static_best": None}
    print(f"ceiling arithmetic (issued REPRISE): overall {out['ceiling']['overall']:.2f}; if N1 were detected like R: {out['ceiling']['k1_as_R']:.2f}; "
          f"if every case were like R: {out['ceiling']['all_as_R']:.2f}")
    A.dumpj(A.RESULTS / f"an_cases_{tag or exp}.json", out)


if __name__ == "__main__":
    main(*sys.argv[1:])
