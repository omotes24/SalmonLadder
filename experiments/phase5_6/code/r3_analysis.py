"""Round-3 analysis on dev1: dynamic seed sets."""
import sys
import numpy as np
from analyze5 import FROZEN, REFT, load, show, table, keys_of
import diag5

cfg = sys.argv[1] if len(sys.argv) > 1 else "r3"
lp = f"lp|{REFT}"
T = load(cfg, "dev1", "near")
z = next(iter(T.values()))
dyn = sorted(k for k in keys_of(z) if f"neg:{k}" in keys_of(z))
C = {"frozen": FROZEN, "M0 x lp": [("M0", 1), (lp, 1)]}
for x in dyn:
    C[f"{x} x lp x neg"] = [(x, 1), (lp, 1), (f"neg:{x}", 1)]
    C[f"M0 x lp x neg:{x}"] = [("M0", 1), (lp, 1), (f"neg:{x}", 1)]
    C[f"M0 x {x} x lp x neg"] = [("M0", 1), (x, 1), (lp, 1), (f"neg:{x}", 1)]
show(table(cfg, C, bases=("none",)), "dynamic sets: memory read-out x support mass x seed mass")
last = {t: int(np.flatnonzero(zz["bidx"] == zz["bidx"].max())[0]) for t, zz in T.items()}
print("membership at the last batch (L14), ID% / OOD% / purity:")
for x in dyn + [m for m in ("M0", "Mjoint0.1") if f"a::L14::{m}" in z.files]:
    a, b, p = [], [], []
    for t, zz in T.items():
        m, f = zz[f"a::L14::{x}"][:last[t]], zz["is_ood"][:last[t]]
        a.append(m[~f].mean()); b.append(m[f].mean()); p.append(m[f].sum() / max(m.sum(), 1))
    print(f"  {x:16s} {100 * np.mean(a):5.1f} {100 * np.mean(b):5.1f} {100 * np.mean(p):5.1f}")
best = sorted(((table(cfg, {n: c}, bases=("none",))[("none", n)]["near"]["FPR95"], n) for n, c in C.items()))[:6]
D = {"frozen": FROZEN}
D.update({n: C[n] for _, n in best})
res = diag5.run(cfg, D)
print("by appearance index", diag5.BINS)
for name, r in res.items():
    print(f"{name:34s} FPR95 {100 * r['FPR95']:5.1f} | by k: " + " ".join(f"{100 * v:5.1f}" for v in r["by_k"]) +
          " | late q: " + " ".join(f"{100 * v:4.0f}" for v in r["late_q"]) + f" | IDrej top5% cls {100 * r['id_rej_top5pct_classes']:.0f}%")
