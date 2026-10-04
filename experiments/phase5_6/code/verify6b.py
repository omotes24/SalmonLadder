"""Phase 6: independent re-computation of the post-hoc variants (does not import analyze5 / metrics_p4)."""
import glob
import re

import numpy as np
from scipy.stats import t as tdist

R = "/home/omote/reprise_p5_20261002/results"
M = {"v5": ["M0", "lp0|k10g1b1l0.9"], "two-sided": ["M0", "IB0.2s<B0.5s", "lp|k10g1b1l0.9", "neg:IB0.2s<B0.5s"]}
ENC = {"DINOv2": ("lock", ["B14", "L14"]), "v2L": ("l14", ["L14"]), "v3L": ("d3L", ["D3L"]), "v2B+v3L": ("d3mixB", ["B14", "D3L"]),
       "v2L+v3L": ("d3mixL", ["L14", "D3L"])}


def auroc(score, ood):
    """P(ID score > OOD score) + 0.5 P(tie), by sorting (ID-high score)."""
    i, o = np.sort(score[~ood]), score[ood]
    lo, hi = np.searchsorted(i, o, "left"), np.searchsorted(i, o, "right")
    return 100 * float(((len(i) - hi) + 0.5 * (hi - lo)).sum() / (len(i) * len(o)))


def fpr95(score, ood):
    i = np.sort(score[~ood])[::-1]                      # descending: accept the top 95% of ID
    thr = i[int(np.ceil(0.95 * len(i))) - 1]
    return 100 * float((score[ood] >= thr).mean())


def run(dev, stream, enc, method, base):
    cfg, views = ENC[enc]
    per = {}
    for f in sorted(glob.glob(f"{R}/{cfg}/{dev}_draw*_{stream}_seed*.npz")):
        d = int(re.search(r"draw(\d+)_", f).group(1))
        z = np.load(f, allow_pickle=True)
        s = sum(z[f"s::{v}::{k}"].astype(np.float64) for v in views for k in M[method])
        if base == "TINS":
            s = s + z["logS"]
        ood = z["is_ood"].astype(bool)
        per.setdefault(d, []).append((auroc(s, ood), fpr95(s, ood)))
    assert sorted(per) == [0, 1, 2, 3, 4] and all(len(v) == 3 for v in per.values()), (dev, stream, enc, {k: len(v) for k, v in per.items()})
    return np.array([np.mean(per[d], axis=0) for d in sorted(per)])       # (5 draws, [AUROC, FPR95])


def ci(x):
    h = tdist.ppf(0.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x))
    return f"{x.mean():+6.2f} [{x.mean() - h:+6.2f},{x.mean() + h:+6.2f}]"


for dev in ("dev2", "dev1"):
    for stream in ("near", "far"):
        for base in ("none", "TINS"):
            for method in M:
                ref = run(dev, stream, "DINOv2", method, base)
                line = f"{dev} {stream:4s} {base:4s} {method:10s} DINOv2 {ref[:, 0].mean():6.2f}/{ref[:, 1].mean():6.2f}"
                for enc in ("v2L", "v3L", "v2B+v3L", "v2L+v3L"):
                    r = run(dev, stream, enc, method, base)
                    line += f" | {enc} {r[:, 0].mean():5.2f}/{r[:, 1].mean():5.2f} d {ci(r[:, 1] - ref[:, 1])}"
                print(line, flush=True)
z = np.load(sorted(glob.glob(f"{R}/d3mixL/dev2_draw0_near_seed123.npz"))[0], allow_pickle=True)
print("example task: n", len(z["is_ood"]), "OOD share", round(float(z["is_ood"].mean()), 3), "batches", int(z["bidx"].max()) + 1)
