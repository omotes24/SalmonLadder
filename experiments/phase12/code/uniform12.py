"""Super-uniformity check of p+ on the ID test images (draws runs, every draw and data set): the share of ID images with p+ <= alpha."""
import glob, json
import numpy as np
from pathlib import Path
import p12common as P
out = {}
for f in sorted(glob.glob(str(P.RESULTS12 / "draws" / "*" / "*" / "*.npz"))):
    z = np.load(f, allow_pickle=True)
    part, draw, ds = Path(f).parts[-3], Path(f).parts[-2], Path(f).stem
    flag = z["is_ood"].astype(bool)
    for v in ("L14", "D3L"):
        p = np.exp(z[f"s::{v}::lp"].astype(np.float64))[~flag]
        out.setdefault(f"{part}|{ds}|{v}", []).append({"draw": draw, **{f"le{a:g}": float((p <= a).mean()) for a in (0.01, 0.05, 0.1, 0.2, 0.5)}})
summary = {}
for k, L in out.items():
    summary[k] = {a: float(np.mean([e[a] for e in L])) for a in ("le0.01", "le0.05", "le0.1", "le0.2", "le0.5")}
    summary[k]["n_draws"] = len(L)
P.write_json(P.RESULTS12 / "p12_uniform.json", summary)
for k, e in summary.items():
    print(f"{k:34s} " + " ".join(f"{a}:{e[a]:.3f}" for a in ("le0.01", "le0.05", "le0.1", "le0.2", "le0.5")))
print("UNIFORM_DONE")
