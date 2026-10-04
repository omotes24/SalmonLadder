"""B: backbone dependence on U1 (unit: split) and U2 (unit: draw). Frozen hyper-parameters; only the views change.
A multi-view set is the product over its views. Families: static p, zeta, memory only, static x p_LP, REPRISE."""
import json
import sys

import numpy as np

import an7 as A
from metrics_p4 import metrics, tci

SETS = ["S14", "B14", "L14", "G14", "D3B", "D3L", "DINO1", "MAE", "CLIP", "CLIPL", "SIG2L",
        "B14+L14", "S14+B14", "S14+CLIP", "B14+CLIP", "L14+CLIP", "L14+G14", "D3B+D3L", "L14+D3L", "CLIP+CLIPL",
        "D3S", "D3SP", "RN50", "S14+D3S", "D3S+D3SP", "S14+RN50"]          # the last six: amendment 03 (added after the first results)
FAMS = ["static", "zeta", "mem", "plp", "static_plp", "reprise"]
REF = "B14+L14"
CAND = A.locked_spec()


def run(exp, unit, sets):
    have = {}
    rows = {}
    for name in sets:
        views = name.split("+")
        ts = [A.tasks(exp, v) for v in views]
        if not all(ts) or any(t != ts[0] for t in ts):
            continue
        have[name] = len(ts[0])
        rr = []
        for t in ts[0]:
            s = A.Stream(exp, t, views)
            row = dict(A.parse_std(t))
            for f in FAMS + (["ext"] if CAND else []):
                x = s.cand(CAND) if f == "ext" else s.family(f)
                for base in ("none", "TINS"):
                    m = metrics(x + s.base(base), s.is_ood)
                    row[f"{f}|{base}|FPR95"], row[f"{f}|{base}|AUROC"] = m["FPR95"], m["AUROC"]
            rr.append(row)
        rows[name] = rr
    out = {"exp": exp, "unit": unit, "streams": have, "sets": {}}
    g = lambda nm, key: A.unit_mean(rows[nm], unit, key)[0]
    print(f"##### {exp} (unit: {unit}); streams per set: {have}")
    print(f"{'views':10s} | static        | zeta          | memory        | REPRISE       | REPRISE x TINS | ext           | REPRISE-zeta FPR95        | REPRISE-static FPR95      | ext-REPRISE FPR95         | vs {REF} (REPRISE FPR95)")
    for nm in rows:
        o = {}
        for f in FAMS + (["ext"] if CAND else []):
            for base in ("none", "TINS"):
                o[f"{f}|{base}"] = {"AUROC": tci(g(nm, f"{f}|{base}|AUROC")), "FPR95": tci(g(nm, f"{f}|{base}|FPR95"))}
        if CAND:
            o["ext-reprise"] = {m: tci(g(nm, f"ext|none|{m}") - g(nm, f"reprise|none|{m}")) for m in ("FPR95", "AUROC")}
        o["reprise-zeta"] = {m: tci(g(nm, f"reprise|none|{m}") - g(nm, f"zeta|none|{m}")) for m in ("FPR95", "AUROC")}
        o["reprise-static"] = {m: tci(g(nm, f"reprise|none|{m}") - g(nm, f"static|none|{m}")) for m in ("FPR95", "AUROC")}
        o["reprise-static_plp"] = {m: tci(g(nm, f"reprise|none|{m}") - g(nm, f"static_plp|none|{m}")) for m in ("FPR95", "AUROC")}
        if REF in rows and have.get(REF) == have[nm]:
            o["vs_ref"] = {f"{f}|{b}": {m: tci(g(nm, f"{f}|{b}|{m}") - g(REF, f"{f}|{b}|{m}")) for m in ("FPR95", "AUROC")}
                           for f in ("reprise", "zeta", "static") + (("ext",) if CAND else ()) for b in ("none", "TINS")}
        out["sets"][nm] = o
        c = lambda f, b="none": f"{o[f'{f}|{b}']['AUROC']['mean']:5.2f}/{o[f'{f}|{b}']['FPR95']['mean']:5.2f}"
        vr = A.fmt_ci(o["vs_ref"]["reprise|none"]["FPR95"]) if "vs_ref" in o else ""
        print(f"{nm:10s} | {c('static')}   | {c('zeta')}   | {c('mem')}   | {c('reprise')}   | {c('reprise', 'TINS')}    | {(c('ext') if CAND else ''):13s} | "
              f"{A.fmt_ci(o['reprise-zeta']['FPR95']):25s} | {A.fmt_ci(o['reprise-static']['FPR95']):25s} | {(A.fmt_ci(o['ext-reprise']['FPR95']) if CAND else ''):25s} | {vr}")
    if "L14+D3L" in out["sets"] and "vs_ref" in out["sets"]["L14+D3L"]:
        h = out["sets"]["L14+D3L"]["vs_ref"]
        out["H_B"] = {"standalone_FPR95": h["reprise|none"]["FPR95"], "standalone_AUROC": h["reprise|none"]["AUROC"],
                      "TINS_FPR95": h["reprise|TINS"]["FPR95"], "confirmed": bool(h["reprise|none"]["FPR95"]["hi"] < 0)}
        print("H-B (L14+D3L minus B14+L14, REPRISE):", "standalone FPR95", A.fmt_ci(h["reprise|none"]["FPR95"]), "AUROC", A.fmt_ci(h["reprise|none"]["AUROC"]),
              "x TINS FPR95", A.fmt_ci(h["reprise|TINS"]["FPR95"]), "->", "confirmed" if out["H_B"]["confirmed"] else "not confirmed")
    return out


if __name__ == "__main__":
    sets = sys.argv[1].split(",") if len(sys.argv) > 1 else SETS
    sfx = ("u1w", "u2w") if CAND else ("u1std", "u2std")
    res = {"U1": run(sfx[0], "split", sets), "U2": run(sfx[1], "draw", sets), "extension": list(CAND) if CAND else None}
    A.dumpj(A.RESULTS / "an_b.json", res)
    print("written", A.RESULTS / "an_b.json")
