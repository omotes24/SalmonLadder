"""Phase 5 diagnostics on dev1 near streams (labels are used; nothing here feeds the detector).

For a combination: the OOD acceptance rate at the ID-95% threshold as a function of the appearance index of the
image within its unknown class (1 = first image of the class in the stream), per-class miss rates, and the
composition of the rejected ID images.
"""
import argparse
import json

import numpy as np

from analyze5 import FROZEN, REFT, load, visual

BINS = [(1, 1), (2, 2), (3, 5), (6, 10), (11, 20), (21, 30), (31, 50)]


def appearance(wnid, is_ood):
    k = np.zeros(len(wnid), int)
    seen = {}
    for i, (w, o) in enumerate(zip(wnid, is_ood)):
        seen[w] = seen.get(w, 0) + 1
        k[i] = seen[w]
    return k


def one(z, combo, views=None):
    x = visual(z, combo, views)
    f = z["is_ood"]
    ids = np.sort(x[~f])
    thr = ids[len(ids) - int(np.ceil(0.95 * len(ids)))]
    if "_k" not in z.c:
        z.c["_k"] = appearance(z["wnid"], f)
        z.c["_code"] = np.unique(z["wnid"], return_inverse=True)[1]
    k, code = z.c["_k"], z.c["_code"]
    acc = (x >= thr) & f                                    # OOD accepted as ID (misses)
    by_k = [float(acc[f & (k >= lo) & (k <= hi)].mean()) for lo, hi in BINS]
    nb = code.max() + 1
    n_ood = np.bincount(code[f], minlength=nb)
    per_cls = (np.bincount(code[acc], minlength=nb) / np.maximum(n_ood, 1))[n_ood > 0]
    lt = f & (k > 20)
    n_late = np.bincount(code[lt], minlength=nb)
    late = (np.bincount(code[acc & lt], minlength=nb) / np.maximum(n_late, 1))[n_late > 0]
    rej = (x < thr) & ~f
    n_id = np.bincount(code[~f], minlength=nb)
    rej_cls = (np.bincount(code[rej], minlength=nb) / np.maximum(n_id, 1))[n_id > 0]
    top = np.sort(rej_cls)[::-1]
    return {"FPR95": float(acc[f].mean()), "by_k": by_k, "class_q": np.quantile(per_cls, [0.1, 0.25, 0.5, 0.75, 0.9]).tolist(),
            "late_q": np.quantile(late, [0.1, 0.25, 0.5, 0.75, 0.9]).tolist(),
            "share_top10_classes": float(np.sort(per_cls)[::-1][:len(per_cls) // 10].sum() / max(per_cls.sum(), 1e-9)),
            "id_rej_top5pct_classes": float(top[:len(top) // 20].sum() / max(top.sum(), 1e-9)),
            "id_classes_with_rej": float((rej_cls > 0).mean())}


def admission(z, views=None):
    f = z["is_ood"]
    out = {}
    for key in z.files:
        if key.startswith("a::"):
            _, v, name = key.split("::")
            if views and v not in views:
                continue
            a = z[key]
            out[f"{v}:{name}"] = (float(a[~f].mean()), float(a[f].mean()))
    return out


def run(cfg, combos, dev="dev1", views=None):
    T = load(cfg, dev, "near")
    out = {}
    for name, combo in combos.items():
        rs = [one(z, combo, views) for z in T.values()]
        out[name] = {k: (np.mean([r[k] for r in rs], axis=0).tolist()) for k in rs[0]}
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", default="base")
    ap.add_argument("--dev", default="dev1")
    a = ap.parse_args()
    lp = f"lp|{REFT}"
    combos = {"frozen": FROZEN, "static": [("static", 1.0)], "M0": [("M0", 1.0)], "lp": [(lp, 1.0)], "M0 x lp": [("M0", 1.0), (lp, 1.0)],
              "gall": [("gall", 1.0)], "ctr1": [("ctr1", 1.0)]}
    res = run(a.cfg, combos, a.dev)
    print("bins (appearance index):", BINS)
    for name, r in res.items():
        print(f"{name:10s} FPR95 {100 * r['FPR95']:5.1f} | by k: " + " ".join(f"{100 * v:5.1f}" for v in r["by_k"]) +
              f" | class q10-90: " + " ".join(f"{100 * v:4.0f}" for v in r["class_q"]) +
              f" | late(k>20) q: " + " ".join(f"{100 * v:4.0f}" for v in r["late_q"]) +
              f" | top10% classes share {100 * r['share_top10_classes']:.0f}% | ID rej in top5% classes {100 * r['id_rej_top5pct_classes']:.0f}%")
    T = load(a.cfg, a.dev, "near")
    adm = [admission(z) for z in T.values()]
    print("admission rates (ID, OOD) in %:")
    for key in adm[0]:
        print(f"  {key:22s} ID {100 * np.mean([x[key][0] for x in adm]):5.1f}  OOD {100 * np.mean([x[key][1] for x in adm]):5.1f}")
