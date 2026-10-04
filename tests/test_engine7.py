"""Phase 7 engine (the components of the frozen Salmon Ladder, v5, plus the additional read-outs) on synthetic streams.

  * the frozen read-outs of engine7 equal the Phase 4 engine (static p, memory p_M, warm and zero-start ranks, zeta);
  * the one-sided memory with c -> infinity equals the frozen memory; before any admission every memory read-out
    equals the static p;
  * batch-only and image-alone propagation agree with direct runs;
  * State7 probes equal the Phase 4 StreamState probes and do not change the state;
  * delayed re-scoring never changes an issued score, agrees with a truncated stream, and its end-of-stream memory
    read-out agrees with a brute-force computation.
"""
import numpy as np
import pytest

import engine
import engine7 as E7
import synth5
from state import StreamState
from static import static_view

DEV = "cpu"
TH = (0.3, 0.2, 0.10191613435745239)      # frozen entrance thresholds


@pytest.mark.parametrize("view", ["B14", "L14"])
def test_frozen_readouts_and_probes(view):
    S = synth5.make(seed=3)
    v = S["views"][view]
    st = static_view(v["sup"], v["cal"], v["sf"], S["cand_cal"], S["cand_sf"], 48, 1)
    ref, _ = engine.run_view(v["sup"], v["cal"], v["sf"], S["bidx"], st, TH, ((10, 1.0),), (0.9,), DEV, ("cdf", "z", "cdf_L0"))
    out, aux = E7.run_view7(v["sup"], v["cal"], v["sf"], S["bidx"], st, TH, device=DEV, hinges=(0, 1, 2, 3, 4, 1e9), lp_batch=True)
    for a, b in (("static", "static"), ("M", "Mpt"), ("lp0", "cdf_L0|k10g1l0.9"), ("lp", "cdf|k10g1l0.9"), ("z", "z|k10g1l0.9")):
        assert float(np.max(np.abs(out[a] - ref[b]))) < 1e-9, (a, b)
    assert np.array_equal(aux["admit"], ref["_admit"])
    assert np.array_equal(out["Mh1e+09"], out["M"]), "hinge with c -> infinity must equal the frozen read-out"
    first = S["bidx"] == 0
    assert np.array_equal(out["Mh0"][first], out["static"][first]) and np.array_equal(out["M"][first], out["static"][first])
    assert np.all(out["Mrho"][first] == 0.0)
    # batch-only graph: the first batch equals the zero-start read-out of the full graph
    assert np.max(np.abs(out["lpB"][first] - out["lp"][first])) < 1e-9
    # image-alone graph: a few rows against a batch of one on the fixed graph
    nc = len(v["cal"])
    cut = {**st, "d": st["d"][:nc + 3], "d_all": st["d_all"][:nc + 3], "p": st["p"][:nc + 3]}
    o1, _ = E7.run_view7(v["sup"], v["cal"], v["sf"][:3], np.arange(3) * 0, cut, TH, device=DEV, lp_alone=True)
    for i in range(3):
        one = {**st, "d": np.r_[st["d"][:nc], st["d"][nc + i]], "d_all": np.r_[st["d_all"][:nc], st["d_all"][nc + i]],
               "p": np.r_[st["p"][:nc], st["p"][nc + i]]}
        o2, _ = E7.run_view7(v["sup"], v["cal"], v["sf"][i:i + 1], np.zeros(1, int), one, TH, device=DEV)
        assert abs(o1["lpA"][i] - o2["lp"][0]) < 1e-9, i
    # State7 probes equal the Phase 4 StreamState probes
    cfgs = [("cdf_L0", 10, 1.0, 0.9), ("cdf", 10, 1.0, 0.9), ("z", 10, 1.0, 0.9)]
    A = StreamState(v["sup"], v["cal"], st, TH, cfgs, device=DEV)
    B = E7.State7(v["sup"], v["cal"], st, TH, device=DEV)
    d, da, p = st["d"][nc:], st["d_all"][nc:], st["p"][nc:]
    for lo in range(0, 256, 64):
        A.step(v["sf"][lo:lo + 64], d[lo:lo + 64], da[lo:lo + 64])
        B.step(v["sf"][lo:lo + 64], d[lo:lo + 64], da[lo:lo + 64])
    for i in (300, 301, 500):
        a = A.probe(v["sf"][i], d[i], da[i], p[i])
        b = B.probe(v["sf"][i], d[i], da[i], p[i])
        for ka, kb in (("static", "static"), ("Mpt", "M"), (cfgs[0], "lp0"), (cfgs[1], "lp"), (cfgs[2], "z")):
            assert abs(a[ka] - b[kb]) < 1e-9, (i, ka)
    # probing does not change the state
    assert B.probe(v["sf"][300], d[300], da[300], p[300]) == B.probe(v["sf"][300], d[300], da[300], p[300])


@pytest.mark.parametrize("view", ["B14", "L14"])
def test_delayed_rescoring(view):
    S = synth5.make(seed=5, batch=64)
    v = S["views"][view]
    st = static_view(v["sup"], v["cal"], v["sf"], S["cand_cal"], S["cand_sf"], 48, 1)
    base, _ = E7.run_view7(v["sup"], v["cal"], v["sf"], S["bidx"], st, TH, device=DEV)
    out, aux = E7.run_view7(v["sup"], v["cal"], v["sf"], S["bidx"], st, TH, device=DEV, retro=(1, 2, 5))
    for key in base:
        assert np.array_equal(base[key], out[key]), key                      # issued scores are untouched
    nc = len(v["cal"])
    bidx = S["bidx"]
    last = bidx == bidx.max()
    for key in ("lp", "lp0", "z"):
        assert np.max(np.abs(out[f"{key}@end"][last] - out[key][last])) < 1e-12, key
    # brute force: memory read-out at the end of the stream
    X = np.asarray(v["sf"], np.float32)
    mem = np.flatnonzero(aux["admit"][:, 2])
    sm = X @ X[mem].T
    sm[mem, np.arange(len(mem))] = -np.inf
    rho = (1.0 - sm.max(1).astype(np.float32)).astype(np.float64)
    calf = np.asarray(v["cal"], np.float32)
    rc = (1.0 - (calf @ X[mem].T).max(1).astype(np.float32)).astype(np.float64)
    x, xc = (rho - st["med"]) / st["mad"], (rc - st["med"]) / st["mad"]
    ref = E7._log(E7.p_high(st["d_cal"] - xc, st["d"][nc:] - x))
    assert float(np.max(np.abs(ref - out["M@end"]))) < 1e-6
    # delay 1: batch b re-scored after batch b + 1 equals the end values of a stream truncated after b + 1
    b0 = 3
    keep = bidx <= b0 + 1
    st2 = {**st, "d": st["d"][:nc + keep.sum()], "d_all": st["d_all"][:nc + keep.sum()], "p": st["p"][:nc + keep.sum()]}
    o2, _ = E7.run_view7(v["sup"], v["cal"], v["sf"][keep], bidx[keep], st2, TH, device=DEV, retro=(1,))
    rows = np.flatnonzero(bidx == b0)
    for key in ("M", "Mh2", "Mrho", "lp", "lp0"):
        assert np.max(np.abs(o2[f"{key}@end"][rows] - out[f"{key}@1"][rows])) < 1e-9, key


def test_within_batch_memory_readout():
    """'@0' (the read-out of the extension): members admitted from the same batch count, the image itself never does."""
    S = synth5.make(seed=7, batch=64)
    v = S["views"]["L14"]
    st = static_view(v["sup"], v["cal"], v["sf"], S["cand_cal"], S["cand_sf"], 48, 1)
    base, _ = E7.run_view7(v["sup"], v["cal"], v["sf"], S["bidx"], st, TH, device=DEV)
    out, aux = E7.run_view7(v["sup"], v["cal"], v["sf"], S["bidx"], st, TH, device=DEV, retro=(0,))
    for key in base:
        assert np.array_equal(base[key], out[key]), key
    # brute force for one batch: nearest member among the members admitted up to and including this batch, self excluded
    nc, bidx = len(v["cal"]), S["bidx"]
    X = np.asarray(v["sf"], np.float32)
    b = 4
    rows = np.flatnonzero(bidx == b)
    members = np.flatnonzero(aux["admit"][:, 2] & (bidx <= b))
    sm = X[rows] @ X[members].T
    for j, r in enumerate(rows):
        sm[j, members == r] = -np.inf
    rho = (1.0 - sm.max(1).astype(np.float32)).astype(np.float64)
    calf = np.asarray(v["cal"], np.float32)
    rc = (1.0 - (calf @ X[members].T).max(1).astype(np.float32)).astype(np.float64)
    x, xc = (rho - st["med"]) / st["mad"], (rc - st["med"]) / st["mad"]
    d, dc = st["d"][nc + rows], np.asarray(st["d_cal"], np.float64)
    ref = E7._log(E7.p_high(dc + np.maximum(0.0, 2 - xc), d + np.maximum(0.0, 2 - x)))
    assert float(np.max(np.abs(ref - out["Mh2@0"][rows]))) < 1e-6
