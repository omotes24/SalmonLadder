"""Round-6 analysis on dev1: BH levels and partially re-calibrated factor groups; standalone and x TINS."""
import sys
import numpy as np
from analyze5 import FROZEN, REFT, load, show, table, keys_of
import diag5

cfg = sys.argv[1] if len(sys.argv) > 1 else "r6"
lp = f"lp|{REFT}"
T = load(cfg, "dev1", "near")
z = next(iter(T.values()))
dyn = sorted(k for k in keys_of(z) if f"neg:{k}" in keys_of(z))
C = {"frozen": FROZEN, "M0 x lp": [("M0", 1), (lp, 1)]}
for x in dyn:
    C[f"M0 x {x} x lp x neg"] = [("M0", 1), (x, 1), (lp, 1), (f"neg:{x}", 1)]
    C[f"M0 x lp x neg:{x}"] = [("M0", 1), (lp, 1), (f"neg:{x}", 1)]
    if f"ln:{x}" in keys_of(z):
        C[f"M0 x ln:{x}"] = [("M0", 1), (f"ln:{x}", 1)]
        C[f"M0 x {x} x ln:{x}"] = [("M0", 1), (x, 1), (f"ln:{x}", 1)]
        C[f"m0ln:{x}"] = [(f"m0ln:{x}", 1)]
        C[f"{x} x m0ln:{x}"] = [(x, 1), (f"m0ln:{x}", 1)]
        C[f"all:{x}"] = [(f"all:{x}", 1)]
res = table(cfg, C, bases=("none", "TINS"))
show(res, f"{cfg}: BH sets, re-calibrated groups")
last = {t: int(np.flatnonzero(zz["bidx"] == zz["bidx"].max())[0]) for t, zz in T.items()}
print("membership at the last batch (L14), ID% / OOD% / purity; near and far streams:")
TF = load(cfg, "dev1", "far")
lastf = {t: int(np.flatnonzero(zz["bidx"] == zz["bidx"].max())[0]) for t, zz in TF.items()}
for x in dyn + ["M0"]:
    out = []
    for TT, ll in ((T, last), (TF, lastf)):
        a, b, p = [], [], []
        for t, zz in TT.items():
            m, f = zz[f"a::L14::{x}"][:ll[t]], zz["is_ood"][:ll[t]]
            a.append(m[~f].mean()); b.append(m[f].mean()); p.append(m[f].sum() / max(m.sum(), 1))
        out.append(f"{100 * np.mean(a):5.1f} {100 * np.mean(b):5.1f} {100 * np.mean(p):5.1f}")
    print(f"  {x:16s} near {out[0]} | far {out[1]}")
ok = [(v["near"]["FPR95"], n) for (b, n), v in res.items() if b == "none" and v["far"]["dFPR"] <= 0.5]
best = [n for _, n in sorted(ok)[:6]]
D = {"frozen": FROZEN}
D.update({n: C[n] for n in best})
r = diag5.run(cfg, D)
print("far-compliant candidates by appearance index", diag5.BINS)
for name, rr in r.items():
    print(f"{name:34s} FPR95 {100 * rr['FPR95']:5.1f} | by k: " + " ".join(f"{100 * v:5.1f}" for v in rr["by_k"]) +
          " | late q: " + " ".join(f"{100 * v:4.0f}" for v in rr["late_q"]) + f" | IDrej top5% cls {100 * rr['id_rej_top5pct_classes']:.0f}%")
