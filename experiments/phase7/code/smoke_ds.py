"""Smoke test of the dataset code paths that need no new features: SSB CUB (compared with the Phase 3 evaluation of
the same frozen v5) and the matched composition on U1."""
import json

import numpy as np

import data7
import run7
import streams7 as S7
from metrics_p4 import metrics
from p7common import GONOGO

P = data7.cub_ssb("Easy")
ids, flag = S7.standard(P, 123)
logS, bidx = run7.tins_scores(P, 123, ids)
x = 0.0
for v in ("B14", "L14"):
    arr = run7.run_stream(P, ids, flag, bidx, v, logS=logS)
    x = x + arr["s::M"] + arr["s::lp0"]
ref = json.loads((GONOGO / "r5" / "phase3" / "cub" / "eval_Easy_seed123.json").read_text())["metrics"]
print("SSB CUB Easy seed 123: REPRISE x TINS", metrics(x + logS, flag), "| Phase 3 v5:", ref["v5"], "| standalone", metrics(x, flag), "| Phase 3 v5_vis:", ref.get("v5_vis"))
print("  n", len(ids), "ood", int(flag.sum()), "C", P.C, "batches", int(bidx.max()) + 1)
P = data7.u1(1, 0)
ids, flag, info = S7.matched(P, 123)
arr = run7.run_stream(P, ids, flag, np.arange(len(ids)) // 256, "B14", extra=info)
print("U1 matched:", info, "n", len(ids), "classes", len(set(arr["cls"][flag])), "per class", np.unique(arr["cls"][flag], return_counts=True)[1][:5])
print("SMOKE_DS_OK")
