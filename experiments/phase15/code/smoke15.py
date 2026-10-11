"""Phase 15 smoke test: (1) the capped step-up (self-consistency, cap, agreement with BH when the cap is slack); (2) the capped
rules through core14.xfit (p15common.install) on synthetic data: seeds and p_final finite, at a high mixing rate the capped rule
selects at most ~tau of the ID images of the fold as seeds while the uncapped Storey rule selects most of them; (3) Graph15 with
kmax = 40: k = 10 reproduces K.Graph (same propagated mass), k = 40 differs; (4) centred features; (5) layout assumptions on hades
(lists, Phase 14 features, the Phase 4 VLM package)."""
import json
import sys
from pathlib import Path

import numpy as np

import p15common as P
import core14 as K
import core15 as K15

CPU_ONLY = "--cpu" in sys.argv
dev = "cpu" if CPU_ONLY else "cuda"
rng = np.random.default_rng(0)
# ---------- (1) capped step-up ----------
p = rng.uniform(size=2000) ** 3                                   # many small p-values
for level, tau in ((0.2, 0.05), (0.2, 1.0), (0.9, 0.05)):
    mask = P.capped_stepup(p, level, tau)
    R = int(mask.sum())
    assert (p[mask] <= tau + 1e-12).all() and (p[mask] <= level * R / len(p) + 1e-12).all(), (level, tau)
    if tau >= 1.0:
        assert np.array_equal(mask, P.P.plain_bh(p, level)), "cap = 1 must reproduce BH"
    print(json.dumps({"level": level, "tau": tau, "rejections": R, "max_p_rejected": float(p[mask].max()) if R else None}))
assert P.capped_stepup(p, 0.2, 0.05).sum() <= P.P.plain_bh(p, 0.2).sum()
# ---------- (2) capped rules through xfit at a high mixing rate ----------
P.install()
C_, d = 20, 16
centres = rng.normal(size=(C_, d))
sup = np.concatenate([centres[c] + 0.3 * rng.normal(size=(12, d)) for c in range(C_)])
cal = np.concatenate([centres[c] + 0.3 * rng.normal(size=(8, d)) for c in range(C_)])
id_test = np.concatenate([centres[c] + 0.3 * rng.normal(size=(20, d)) for c in range(C_)])         # 400 ID
ood = rng.normal(size=(3200, d)) * 1.5 + 4.0                                                        # 3,200 unknown: mixing 89%
sf = np.concatenate([id_test, ood])
flag = np.r_[np.zeros(len(id_test), bool), np.ones(len(ood), bool)]
l2n = P.l2n
g = K.Graph(l2n(sup), l2n(cal), l2n(sf), dev)
sc, st = K.proto_scores(l2n(sup), l2n(cal), l2n(sf), dev)
pairs = [("storey", 0.2), ("bh", 0.2), ("storeyc", 0.2), ("bhc", 0.2)]
out, meta = K.xfit({"v": g}, {"v": K.view_terms(g, sc, st)}, flag, folds=2, seed=0, pairs=pairs, modes=("pp",))
rep = {}
for rule in ("storey", "bh", "storeyc", "bhc"):
    e = out[(rule, 0.2, "pp")]
    assert np.isfinite(e["s"]).all() and np.isfinite(e["pfinal"]).all()
    m = [meta["views"]["v"][f"fold{b}|{rule}|q0.2"] for b in (0, 1)]
    n_id_fold = [int((~flag[meta["fold_of_test"] == b]).sum()) for b in (0, 1)]
    id_seeds = [x["seeds"] * (x["id_frac"] or 0.0) for x in m]
    rep[rule] = {"seeds": [x["seeds"] for x in m], "id_frac": [round(x["id_frac"] or 0.0, 3) for x in m], "id_seeds_over_id": [round(a / b, 3) for a, b in zip(id_seeds, n_id_fold)]}
print(json.dumps({"high_mixing": rep}))
assert max(rep["storeyc"]["id_seeds_over_id"]) <= 0.12 and max(rep["bhc"]["id_seeds_over_id"]) <= 0.12, rep       # at most ~tau of the ID images (tau = 0.05, finite-sample slack)
assert max(rep["storey"]["id_seeds_over_id"]) > max(rep["storeyc"]["id_seeds_over_id"]), rep                       # the uncapped Storey rule takes more ID images
g.free()
# ---------- (3) Graph15 ----------
g0 = K.Graph(l2n(sup), l2n(cal), l2n(sf), dev)
g1 = K15.Graph15(l2n(sup), l2n(cal), l2n(sf), dev, kmax=40, k=10)
u0, u1 = g0.mass(np.arange(g0.ns)), g1.mass(np.arange(g1.ns))
assert np.allclose(u0, u1, rtol=1e-6, atol=1e-12), "Graph15 with k = 10 must reproduce K.Graph"
g1.k = 40
u40 = g1.mass(np.arange(g1.ns))
assert not np.allclose(u0, u40, rtol=1e-3), "k = 40 must change the propagated mass"
g1.k = 3
u3 = g1.mass(np.arange(g1.ns))
print(json.dumps({"graph15": {"max|u10-u10'|": float(np.abs(u0 - u1).max()), "corr(u10,u40)": float(np.corrcoef(u0, u40)[0, 1]), "corr(u10,u3)": float(np.corrcoef(u0, u3)[0, 1])}}))
g0.free(); g1.free()
# ---------- (4) centred ----------
cs, cc, cf = K15.centred(l2n(sup), l2n(cal), l2n(sf))
assert np.allclose(np.linalg.norm(cs, axis=1), 1.0, atol=1e-5) and cf.shape == sf.shape
print("SMOKE15_SYNTHETIC_OK", flush=True)
if CPU_ONLY:
    sys.exit(0)
# ---------- (5) hades layout ----------
from extract14 import CIFAR  # noqa: E402
assert (CIFAR / "table.parquet").exists() and (CIFAR / "features" / "L14.npy").exists() and (CIFAR / "features" / "D3L.npy").exists()
assert (P.IMGLIST / "imagenet200" / "train_imagenet200.txt").exists()
import extra15  # noqa: E402
assert extra15.LARGE.exists()
sys.path.insert(0, str(P.P4 / "code"))
import vlm_tta as VT  # noqa: E402
cfp, Ada, Tanl = VT.import_official()
assert len(cfp.get_class_names("imagenet200")) == 200 and len(cfp.get_class_names("cifar100")) == 100
sel = P.selected()
print(json.dumps({"selected": list(sel), "results15": str(P.RESULTS15), "tau": P.TAU, "kgs": list(P.KGS)}))
print("SMOKE15_OK", flush=True)
