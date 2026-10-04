"""Regression tests on hades (GPU): engine7 / run7 / dev7 against the saved Phase 4 and Phase 5 scores.
No new metric of a new method is computed here: only the frozen read-outs are compared."""
import json

import numpy as np

import data7
import run7
import streams7 as S7
from metrics_p4 import metrics
from p7common import P4, P5, RESULTS

rep = {}


def cmp(tag, a, b):
    d = np.abs(np.asarray(a, np.float64) - np.asarray(b, np.float64))
    rep[tag] = {"max_abs": float(d.max()), "rows_gt_1e-9": int((d > 1e-9).sum()), "n": int(len(d))}
    print(tag, rep[tag], flush=True)


# --- U1 (Phase 4 eval scores)
P = data7.u1(1, 0)
ids, flag = S7.standard(P, 123)
logS, bidx = run7.tins_scores(P, 123, ids)
z = np.load(P4 / "results" / "eval" / "U1_s1_d0_seed123_scores.npz", allow_pickle=True)
assert list(z["sample_id"]) == ids and np.array_equal(z["is_ood"].astype(bool), flag) and np.array_equal(z["bidx"], bidx)
S = {}
for v in ("B14", "L14"):
    arr = run7.run_stream(P, ids, flag, bidx, v, lp_batch=True, logS=logS)
    assert np.array_equal(arr["admit"], z[f"admit_{v}"]), "admissions differ"
    for k in arr:
        if k.startswith("s::"):
            S[k[3:]] = S.get(k[3:], 0.0) + arr[k]
    print(v, json.loads(str(arr["meta"]))["seconds"], flush=True)
cmp("U1 logM", arr["logM"], z["logM"])
cmp("U1 logS", arr["logS"], z["logS"])
for a, b in (("static", "static__-"), ("M", "Mpt__-"), ("lp0", "cdf_L0__k10g1l0.9"), ("z", "z__k10g1l0.9"), ("lp", "cdf__k10g1l0.9")):
    cmp(f"U1 {a}", S[a], z[b])
cmp("U1 REPRISE", S["M"] + S["lp0"], z["rep_L0__k10g1l0.9"])
for name, x, ref in (("REPRISE", S["M"] + S["lp0"], z["rep_L0__k10g1l0.9"]), ("zeta", S["z"], z["z__k10g1l0.9"])):
    print(name, "new", metrics(x, flag), "phase4", metrics(ref, flag), flush=True)

# --- U4 (Phase 5 final scores)
P = data7.u4(1, 0)
ids, flag = S7.standard(P, 123)
logS, bidx = run7.tins_scores(P, 123, ids)
z = np.load(P5 / "results_final" / "u4" / "U4_s1_d0_seed123.npz", allow_pickle=True)
assert list(z["sample_id"]) == ids and np.array_equal(z["bidx"], bidx)
for v in ("B14",):
    arr = run7.run_stream(P, ids, flag, bidx, v, logS=logS)
    for a, b in (("static", "static"), ("M", "M0"), ("lp0", "lp0|k10g1b1l0.9"), ("lp", "lp|k10g1b1l0.9"), ("z", "z|k10g1b1l0.9")):
        cmp(f"U4 {v} {a}", arr[f"s::{a}"], z[f"s::{v}::{b}"])
    assert np.array_equal(arr["admit"][:, 2], z[f"a::{v}::M0"])

# --- dev1 (Phase 5 lock scores)
import dev5  # noqa: E402
import dev7  # noqa: E402

D = dev5.load("dev1")
dev7.run_task(D, 0, "near", 123, "L14")
a = np.load(RESULTS / "dev" / "L14" / "dev1_draw0_near_seed123.npz", allow_pickle=True)
z = np.load(P5 / "results" / "lock" / "dev1_draw0_near_seed123.npz", allow_pickle=True)
assert list(a["sample_id"]) == list(z["sample_id"])
for x, y in (("static", "static"), ("M", "M0"), ("lp0", "lp0|k10g1b1l0.9"), ("lp", "lp|k10g1b1l0.9"), ("z", "z|k10g1b1l0.9")):
    cmp(f"dev1 L14 {x}", a[f"s::{x}"], z[f"s::L14::{y}"])
cmp("dev1 logS", a["logS"], z["logS"])
cmp("dev1 logM", a["logM"], z["logM"])
(RESULTS / "test7_hades.json").write_text(json.dumps(rep, indent=1) + "\n")
bad = {k: v for k, v in rep.items() if isinstance(v, dict) and "max_abs" in v and v["max_abs"] > 0.05}
print("TEST7_HADES_OK" if not bad else f"TEST7_HADES_DIFF {bad}")
