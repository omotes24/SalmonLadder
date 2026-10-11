"""Phase 14 smoke test: (1) the cross-fit on synthetic data (CPU): shapes, permutation invariance of s within a fold, uniformity of
p_final for exchangeable nulls, FDR control of BH on p_final under a null-only test set; (2) the layout assumptions on the real
data (hades): Phase 3 feature space views, VLM score files, CIFAR-100 lists and images, dev caches, the Phase 7 extractor import."""
import json
import sys
from pathlib import Path

import numpy as np

import p14common as P

CPU_ONLY = "--cpu" in sys.argv

# ---------- (1) synthetic ----------
import core14 as K  # noqa: E402

rng = np.random.default_rng(0)
C_, d = 20, 16
centres = rng.normal(size=(C_, d))
sup = np.concatenate([centres[c] + 0.3 * rng.normal(size=(12, d)) for c in range(C_)])
cal = np.concatenate([centres[c] + 0.3 * rng.normal(size=(8, d)) for c in range(C_)])
id_test = np.concatenate([centres[c] + 0.3 * rng.normal(size=(40, d)) for c in range(C_)])
ood = rng.normal(size=(300, d)) * 1.5 + 4.0
dev = "cpu" if CPU_ONLY else "cuda"
l2n = P.l2n
fdps, fprs = [], []
for rep in range(3):
    perm = rng.permutation(len(id_test) + len(ood))
    sf = np.concatenate([id_test, ood])[perm]
    flag = np.r_[np.zeros(len(id_test), bool), np.ones(len(ood), bool)][perm]
    g = K.Graph(l2n(sup), l2n(cal), l2n(sf), dev)
    sc, st = K.proto_scores(l2n(sup), l2n(cal), l2n(sf), dev)
    out, meta = K.xfit({"v": g}, {"v": K.view_terms(g, sc, st)}, flag, folds=2, seed=rep)
    e = out[("bh", 0.2, "pp")]
    assert e["s"].shape == (len(sf),) and e["pfinal"].shape == (len(sf),) and np.isfinite(e["s"]).all()
    v = K.validity(e["pfinal"], flag, meta["fold_of_test"], q=0.2)
    fdps.append(v["bh_fdp"]); fprs.append(v["fpr@0.05"])
    au = float((e["s"][~flag][:, None] > e["s"][flag][None, :]).mean())
    print(json.dumps({"rep": rep, "AUROC(s)": round(au, 3), "fpr@0.05": round(v["fpr@0.05"], 3), "bh_fdp": round(v["bh_fdp"], 3), "bh_recall": round(v["bh_recall"], 3),
                      "seeds": {k: m["seeds"] for k, m in meta["views"]["v"].items() if k.endswith("bh|q0.2")}}))
    g.free()
assert np.mean(fprs) < 0.12 and np.mean(fdps) < 0.35, (fprs, fdps)
# permutation invariance: shuffling the test images does not change their scores
sf = np.concatenate([id_test, ood])
flag = np.r_[np.zeros(len(id_test), bool), np.ones(len(ood), bool)]
perm = rng.permutation(len(sf))
res = []
for order in (np.arange(len(sf)), perm):
    g = K.Graph(l2n(sup), l2n(cal), l2n(sf[order]), dev)
    sc, st = K.proto_scores(l2n(sup), l2n(cal), l2n(sf[order]), dev)
    fc = np.arange(len(cal)) % 2
    ft_base = np.arange(len(sf)) % 2
    out, _ = K.xfit({"v": g}, {"v": K.view_terms(g, sc, st)}, flag[order], fold_of=(fc, ft_base[order]))
    s = np.empty(len(sf)); s[order] = out[("storey", 0.2, "pps")]["s"]
    pf = np.empty(len(sf)); pf[order] = out[("storey", 0.2, "pps")]["pfinal"]
    res.append((s, pf))
    g.free()
assert np.allclose(res[0][0], res[1][0], atol=1e-6) and np.allclose(res[0][1], res[1][1], atol=1e-9), "cross-fit is not permutation invariant"
# null-only test set: p_final should be uniform (K-S) and BH should reject (almost) nothing
sf0 = np.concatenate([centres[c] + 0.3 * rng.normal(size=(40, d)) for c in range(C_)])
g = K.Graph(l2n(sup), l2n(cal), l2n(sf0), dev)
sc, st = K.proto_scores(l2n(sup), l2n(cal), l2n(sf0), dev)
out, meta = K.xfit({"v": g}, {"v": K.view_terms(g, sc, st)}, np.zeros(len(sf0), bool))
pf = out[("storey", 0.2, "pp")]["pfinal"]
u = np.sort(pf)
ks = float(np.max(np.abs(u - np.arange(1, len(u) + 1) / len(u))))
print(json.dumps({"null_only": {"P(pfinal<=0.05)": round(float((pf <= 0.05).mean()), 3), "P(pfinal<=0.5)": round(float((pf <= 0.5).mean()), 3), "KS": round(ks, 3),
                                "bh_rejections": int(sum(P.plain_bh(pf[meta['fold_of_test'] == b], 0.2).sum() for b in (0, 1)))}}))
assert ks < 0.08, ks
g.free()
# LGC max on the synthetic graph separates ID from OOD
g = K.Graph(l2n(sup), l2n(cal), l2n(sf), dev)
lg = K.lgc_max(g, np.repeat(np.arange(C_), 12), C_)
au = float((lg[g.nfix:][~flag][:, None] > lg[g.nfix:][flag][None, :]).mean())
print(json.dumps({"lgc_max AUROC": round(au, 3)}))
assert au > 0.9
g.free()
print("SMOKE14_SYNTHETIC_OK", flush=True)
if CPU_ONLY:
    sys.exit(0)

# ---------- (2) real data layout ----------
D = P.load_p3("openood")
P.add_views(D, "openood", ["D3B", "D3L"])
print("shots views:", sorted(D["shots"].keys()), "ev views:", sorted(D["ev"].keys()))
for v in ("B14", "L14", "D3B", "D3L", "CLIP", "DINO1"):
    if v in D["shots"]:
        print(v, "shots", np.asarray(D["shots"][v]).shape, "ev", np.asarray(D["ev"][v]).shape)
from draws12 import datasets  # noqa: E402
sets = datasets("openood")
from test14 import vlm_bases  # noqa: E402
for ds in sorted(sets):
    ids, flag, logS = sets[ds]
    vb = vlm_bases("openood", ds, ids)
    print("vlm bases", ds, sorted({m for (m, s) in vb}), "orders", len({s for (m, s) in vb}))
sets4 = datasets("fourood")
for ds in sorted(sets4):
    ids, flag, logS = sets4[ds]
    vb = vlm_bases("fourood", ds, ids)
    print("vlm bases fourood", ds, sorted({m for (m, s) in vb}))
# CIFAR-100 lists and images
LISTS = P.IMGLIST / "cifar100"
print("cifar100 lists:", sorted(p.name for p in LISTS.glob("*.txt")))
first = (LISTS / "train_cifar100.txt").read_text().splitlines()[0].split()[0]
ROOT = Path("/home/omote/openood_simple_sota/data/images_classic")
print("first train path", first, "exists under images_classic:", (ROOT / first).exists(), "under data:", (ROOT.parent / first).exists())
for name in ("train_cifar100.txt", "test_cifar100.txt", "test_cifar10.txt", "test_tin.txt", "test_mnist.txt", "test_svhn.txt", "test_texture.txt", "test_places365.txt"):
    lines = [l for l in (LISTS / name).read_text().splitlines() if l.strip()]
    labs = sorted({int(l.split()[1]) for l in lines})
    print(name, len(lines), "labels", labs[:3], "...", labs[-1] if labs else None)
sys.path.insert(0, str(P.P7 / "code"))
import extract7  # noqa: E402
print("extract7 import ok; models", extract7.ALL)
sys.path.insert(0, "/home/omote/reprise_p5_20261002/code")
import dev5  # noqa: E402
Dd = dev5.load("dev1")
k = sorted(Dd["tins"].keys())[0]
print("dev1 tasks", len(Dd["tins"]), "first", k, "feats", sorted(Dd["feats"].keys()))
print("SMOKE14_OK", flush=True)
