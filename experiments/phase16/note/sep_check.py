"""Corollary 1 on the instance of theory16_checks.py (same seed) in the setting of check (6): the seeds are every 5th image of each
tight unknown class (no other seeds); for every non-seed image y of those classes, the actual conformal p(y) and the bound
(1 + #{c in C : rho_c omega_c <= pi_Q(y)}) / (|C| + 1). Merged into checks16.json as "separation_escape"."""
import json
from pathlib import Path

import numpy as np

import theory16_checks as T

rng = np.random.default_rng(16)
X, idx, flag, lab = T.instance(rng)
W = T.knn_graph(X, 10)
d = np.asarray(W.sum(1)).ravel()
lam, gam = 0.95, 0.0625 * float(np.median(d))
S, Cn, Tn = idx["S"], idx["C"], idx["T"]
r, nu = lab["r"], lab["n_unk"]
clusters = [idx["U"][j * r:(j + 1) * r] for j in range(nu)]
Z = np.concatenate([q[::5] for q in clusters])
h, _ = T.solve(W, S, Z, lam, gam)
s = lam * (W @ h) / (d + gam)
p = T.rank_low(s[Cn], s[Tn])
pos = {int(t): i for i, t in enumerate(Tn)}
omega = np.asarray(W[:, S].sum(1)).ravel() / d
rho = lam * d / (d + gam)
low_c = np.sort(rho[Cn] * omega[Cn])
floor = 1.0 / (len(Cn) + 1.0)
rows = []
for q in clusters:
    Q = np.setdiff1d(q, Z)
    inQ = np.zeros(len(X), bool)
    inQ[Q] = True
    one = np.flatnonzero(~inQ & ~np.isin(np.arange(len(X)), Z))
    piQ, _ = T.solve(W, one, Z, lam, gam)
    for y in Q:
        rows.append((p[pos[int(y)]], (1.0 + np.searchsorted(low_c, piQ[y], side="right")) / (len(Cn) + 1.0), piQ[y]))
rows = np.array(rows)
out = {"n": int(len(rows)), "violations": int(np.sum(rows[:, 0] > rows[:, 1] + 1e-12)),
       "frac_actual_floor": float(np.mean(rows[:, 0] <= floor + 1e-12)), "frac_bound_floor": float(np.mean(rows[:, 1] <= floor + 1e-12)),
       "frac_actual_le005": float(np.mean(rows[:, 0] <= 0.05)), "frac_bound_le005": float(np.mean(rows[:, 1] <= 0.05)),
       "median_pi": float(np.median(rows[:, 2])), "median_rho_omega_cal": float(np.median(low_c)), "min_rho_omega_cal": float(low_c.min())}
path = Path(__file__).with_name("checks16.json")
ck = json.loads(path.read_text())
ck["separation_escape"] = out
path.write_text(json.dumps(ck, indent=1) + "\n")
print(json.dumps(out, indent=1))
