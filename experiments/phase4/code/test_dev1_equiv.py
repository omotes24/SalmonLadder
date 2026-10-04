"""Equivalence of the Phase 4 engine with the audited REPRISE implementation on dev1 (draw0, near, seed123)."""
import json
import numpy as np
from common import AUDIT, RESULTS, V5, dump, utc
from engine import run_view
from metrics_p4 import metrics
from pilot import load_development, prepare
from static import from_proto

R, D = load_development()
z = np.load(AUDIT / "results/primary/draw0_near_seed123.npz", allow_pickle=True)
views = {}
for v in ("B14", "L14"):
    su, ca, sf, bi, vv, sid, flag, S = prepare(R, D, 0, v, "near", 123)
    assert list(z["sample_id"]) == list(sid)
    views[v], _ = run_view(su, ca, sf, bi, from_proto(vv, su, 48), V5["thresholds"], ((10, 1.0),), (0.9,), "cuda",
                           ("raw", "cdf", "cdf_L0"))
t = "k10g1l0.9"
rep = sum(views[v][f"cdf_L0|{t}"] + views[v]["Mpt"] for v in ("B14", "L14"))
ref = z["log_score_REPRISE_L0_M0"]
res = {"utc": utc()}
for v in ("B14", "L14"):
    for mine, key in ((f"cdf_L0|{t}", "L0_plp"), (f"cdf|{t}", "L1_plp"), ("Mpt", "M0_pt"), (f"raw|{t}", "L1_raw")):
        a, b = np.exp(views[v][mine]), z[f"{v}_{key}"]
        res[f"{v}_{key}_maxabs"] = float(np.max(np.abs(a - b)))
        res[f"{v}_{key}_n_changed"] = int(np.sum(np.abs(a - b) > 1e-12))
m1, m2 = metrics(rep, flag), metrics(ref, flag)
res.update(mine=m1, audit=m2, rep_maxabs=float(np.max(np.abs(rep[np.isfinite(ref)] - ref[np.isfinite(ref)]))))
dump(RESULTS / "test_dev1_equiv.json", res)
print(json.dumps(res, indent=1))
