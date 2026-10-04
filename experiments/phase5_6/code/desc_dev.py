"""Descriptive analyses of the locked method on the development splits (after the lock; no selection)."""
import json
import sys
import numpy as np
from analyze5 import FROZEN, REFT, load, show, table
import diag5

dev = sys.argv[1] if len(sys.argv) > 1 else "dev2"
lp, lp0 = f"lp|{REFT}", f"lp0|{REFT}"
S1, S2 = "B0.5s", "IB0.2s<B0.5s"
C = {"frozen v5 (M0 x lp warm)": FROZEN,
     "M0 x lp (zero start)": [("M0", 1), (lp, 1)],
     "+ seed mass (S2)": [("M0", 1), (lp, 1), (f"neg:{S2}", 1)],
     "+ set memory (S2)": [("M0", 1), (S2, 1), (lp, 1)],
     "locked: M0 x S2 x lp x neg": [("M0", 1), (S2, 1), (lp, 1), (f"neg:{S2}", 1)],
     "locked with round-1 set (S1)": [("M0", 1), (S1, 1), (lp, 1), (f"neg:{S1}", 1)],
     "without M0": [(S2, 1), (lp, 1), (f"neg:{S2}", 1)],
     "lp x neg only": [(lp, 1), (f"neg:{S2}", 1)],
     "zeta": [(f"z|{REFT}", 1)], "static": [("static", 1)], "M0": [("M0", 1)], "lp": [(lp, 1)], "neg(S2)": [(f"neg:{S2}", 1)], "S2 memory": [(S2, 1)]}
res = table("lock", C, dev=dev, bases=("none", "TINS"))
show(res, f"locked method and its parts on {dev} (descriptive)")
T = load("lock", dev, "near")
TF = load("lock", dev, "far")
print("seed sets at the last batch: ID% in set / OOD% in set / purity (1 - FDR); nominal FDR bound q x pi0")
for name, TT in (("near", T), ("far", TF)):
    for x in (S1, S2, "M0"):
        a, b, p, pi0 = [], [], [], []
        for t, z in TT.items():
            last = int(np.flatnonzero(z["bidx"] == z["bidx"].max())[0])
            for v in ("B14", "L14"):
                m, f = z[f"a::{v}::{x}"][:last], z["is_ood"][:last]
                a.append(m[~f].mean()); b.append(m[f].mean()); p.append(m[f].sum() / max(m.sum(), 1)); pi0.append((~f).mean())
        print(f"  {name:5s} {x:14s} {100 * np.mean(a):5.1f} {100 * np.mean(b):5.1f} {100 * np.mean(p):5.1f}   (pi0 = {np.mean(pi0):.3f})")
D = {k: C[k] for k in ("frozen v5 (M0 x lp warm)", "M0 x lp (zero start)", "locked: M0 x S2 x lp x neg")}
r = diag5.run("lock", D, dev)
print("OOD accepted at the ID-95% threshold by appearance index within the unknown class", diag5.BINS)
for name, rr in r.items():
    print(f"{name:30s} FPR95 {100 * rr['FPR95']:5.1f} | by k: " + " ".join(f"{100 * v:5.1f}" for v in rr["by_k"]) +
          " | late per-class quantiles: " + " ".join(f"{100 * v:4.0f}" for v in rr["late_q"]))
out = {"table": {f"{b}|{n}": v for (b, n), v in res.items()}, "by_k": r}
open(f"/home/omote/reprise_p5_20261002/results/desc_{dev}.json", "w").write(json.dumps(out, indent=1, default=float) + "\n")
