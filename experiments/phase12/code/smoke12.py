"""Smoke test of core12 on random features (seconds)."""
import numpy as np
import torch

import core12 as C
import p12common as P

rng = np.random.default_rng(0)


def l2n(x):
    return (x / np.linalg.norm(x, axis=1, keepdims=True)).astype(np.float32)


dev = "cuda" if torch.cuda.is_available() else "cpu"
sup, cal, sf = l2n(rng.normal(size=(1200, 64))), l2n(rng.normal(size=(400, 64))), l2n(rng.normal(size=(3000, 64)))
flag = rng.random(3000) < 0.2
sf[flag] = l2n(sf[flag] + 0.5)
g = C.Graph(sup, cal, sf, dev)
out, meta = C.read_out(g, flag)
print("read_out", {k: v.shape for k, v in out.items() if hasattr(v, "shape")}, meta)
print("cut", g.cut(g.nfix + np.flatnonzero(flag)[:20]))
g.free()
print("knn", C.knn_rank(sup, cal, sf, dev=dev)[:5])
bidx = np.repeat(np.arange(12), 250)
lp, ln, stats = C.online_view(sup, cal, sf, bidx, dev=dev, flag=flag)
print("online", lp[:3], ln[:3], stats[-1])
mask, pi0 = P.storey_adadetect(rng.random(1000), 0.1, 400)
print("storey", mask.sum(), pi0, P.storey_lambda(4000))
print("SMOKE_OK")
