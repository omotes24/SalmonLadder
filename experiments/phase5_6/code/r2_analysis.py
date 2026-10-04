"""Round-2 analysis on dev1 (selection split): two-sided propagation, gates, centroid read-outs, oracle head-room."""
import sys
import numpy as np
from analyze5 import FROZEN, REFT, load, show, table
import diag5

cfg = sys.argv[1] if len(sys.argv) > 1 else "r2"
lp, lp0 = f"lp|{REFT}", f"lp0|{REFT}"
neg = lambda x, t=REFT: f"neg:{x}|{t}"
sets = ["M0", "Mjoint0.05", "Mjoint0.1", "Mjoint0.2", "Mjoint0.3", "Mfull0.05", "Mfull0.1", "Mfull0.2", "Mfull0.3", "Mora"]
C = {"frozen": FROZEN, "M0 x lp": [("M0", 1), (lp, 1)]}
for x in sets:
    C[f"{x} x lp"] = [(x, 1), (lp, 1)]
    C[f"{x} x lp x neg:{x}"] = [(x, 1), (lp, 1), (neg(x), 1)]
    C[f"lp x neg:{x}"] = [(lp, 1), (neg(x), 1)]
    if x != "M0":
        C[f"M0 x lp x neg:{x}"] = [("M0", 1), (lp, 1), (neg(x), 1)]
show(table(cfg, C, bases=("none",)), "A. memory set x support mass x seed mass (reference graph)")

C = {"frozen": FROZEN}
for x in ("Mjoint0.1", "Mjoint0.2", "Mfull0.1"):
    for t in ("k10g1b1l0.9", "k10g1b1l0.8", "k10g1b1l0.5", "k10g0b1l0.9", "k20g1b1l0.9", "k5g1b1l0.9"):
        C[f"{x} x lp x neg[{t}]"] = [(x, 1), (lp, 1), (neg(x, t), 1)]
    C[f"{x} x lp(g0) x neg(g0)"] = [(x, 1), ("lp|k10g0b1l0.9", 1), (neg(x, "k10g0b1l0.9"), 1)]
    C[f"{x} x lp x neg x neg(g0)"] = [(x, 1), (lp, 1), (neg(x), 0.5), (neg(x, "k10g0b1l0.9"), 0.5)]
show(table(cfg, C, bases=("none",)), "B. configuration of the seed mass")

C = {"frozen": FROZEN}
for x in ("Mfrz", "Mjoint0.1", "Mjoint0.2", "Mfull0.1", "Mfull0.2", "Mora"):
    nx = "M0" if x == "Mfrz" else x
    for suf in ("", "c", "cm", "cs"):
        C[f"{x}{suf} x lp"] = [(x + suf, 1), (lp, 1)]
        C[f"{x}{suf} x lp x neg"] = [(x + suf, 1), (lp, 1), (neg(nx), 1)]
    C[f"{x} x {x}c x lp x neg"] = [(x, 0.5), (x + "c", 0.5), (lp, 1), (neg(nx), 1)]
    C[f"{x} x {x}cs x lp x neg"] = [(x, 1), (x + "cs", 1), (lp, 1), (neg(nx), 1)]
show(table(cfg, C, bases=("none",)), "C. centroid read-outs (c: g with centroid distance; cm / cs: cosine margins)")

T = load(cfg, "dev1", "near")
adm = [diag5.admission(z, ["L14"]) for z in T.values()]
print("admission (L14) ID / OOD %: " + "; ".join(f"{k.split(':')[1]} {100 * np.mean([a[k][0] for a in adm]):.1f}/{100 * np.mean([a[k][1] for a in adm]):.1f}" for k in adm[0]))
D = {"frozen": FROZEN, "M0 x lp": [("M0", 1), (lp, 1)]}
for x in ("Mjoint0.1", "Mjoint0.2", "Mfull0.1", "Mfull0.2", "Mora"):
    D[f"{x} x lp x neg"] = [(x, 1), (lp, 1), (neg(x), 1)]
    D[f"{x}c x lp x neg"] = [(x + "c", 1), (lp, 1), (neg(x), 1)]
D["neg:Mora"] = [(neg("Mora"), 1)]
D["Mora"] = [("Mora", 1)]
D["Morac"] = [("Morac", 1)]
res = diag5.run(cfg, D)
print("by appearance index", diag5.BINS)
for name, r in res.items():
    print(f"{name:26s} FPR95 {100 * r['FPR95']:5.1f} | by k: " + " ".join(f"{100 * v:5.1f}" for v in r["by_k"]) +
          " | late q: " + " ".join(f"{100 * v:4.0f}" for v in r["late_q"]) + f" | IDrej top5% cls {100 * r['id_rej_top5pct_classes']:.0f}%")
