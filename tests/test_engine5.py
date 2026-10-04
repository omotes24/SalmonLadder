"""Phase 5 engine (two-sided propagation and the explored variants) on synthetic streams.

 1. frozen read-outs of engine5 equal engine.run_view (memory p_t, warm / zero-start ranks, robust z, admissions)
 2. MemX with the frozen third-stage gate equals the frozen memory
 3. matrix2(beta=1) equals the frozen graph; beta=0 keeps mutual edges only
 4. no future leakage: scores of the first batches do not change when later batches are replaced
 5. the joint gate admits the same rows in every view
 6. dynamic seed sets, centroid read-outs and the Benjamini-Hochberg helper agree with direct computations
"""
import numpy as np
import torch

from vins import r5

import engine
import engine5 as E5
import synth5
from metrics_p4 import metrics
from static import support_dall

DEV = "cpu"
REF = (10, 1.0, 1.0, 0.9)


def views_of(S, kinds=None):
    out = {}
    for name, v in S["views"].items():
        f = E5.fit_transform(v["sup"], *(kinds or ("raw",)))
        sup, cal, sf = f(v["sup"]), f(v["cal"]), f(v["sf"])
        st = E5.static5(r5.proto_view, support_dall, sup, cal, sf, S["cand_cal"], S["cand_sf"], 48, 1)
        out[name] = (sup, cal, sf, st)
    return out


def _frozen():
    S = synth5.make(seed=1)
    V = views_of(S)
    spec = E5.make_spec(cfgs=[REF, (5, 1.0, 1.0, 0.9), (10, 0.0, 1.0, 0.8)], eps_lp=(0.1,), eps_fin=(0.1,), eps_joint=(0.1,),
                        neg=("M0", "Mjoint0.1"))
    out, adm, _ = E5.run5(V, S["bidx"], spec, DEV)
    for name, (sup, cal, sf, st) in V.items():
        ref, _ = engine.run_view(sup, cal, sf, S["bidx"], st, spec["thresholds"], ((10, 1.0), (5, 1.0), (10, 0.0)),
                                 (0.9, 0.8), DEV, ("cdf", "z", "cdf_L0"))
        pairs = [("M0", "Mpt"), ("static", "static"), ("lp0|k10g1b1l0.9", "cdf_L0|k10g1l0.9"),
                 ("lp|k10g1b1l0.9", "cdf|k10g1l0.9"), ("z|k10g1b1l0.9", "z|k10g1l0.9"),
                 ("lp|k5g1b1l0.9", "cdf|k5g1l0.9"), ("lp|k10g0b1l0.8", "cdf|k10g0l0.8")]
        for a, b in pairs:
            d = float(np.max(np.abs(out[name][a] - ref[b])))
            assert d < 1e-9, (name, a, b, d)
        assert np.array_equal(adm[name]["M0"], ref["_admit"][:, 2])
        assert np.array_equal(adm[name]["M0a1"], ref["_admit"][:, 0])
    for name in out:
        for k, v in out[name].items():
            assert not np.isnan(v).any(), (name, k)
    return S, V, out, adm


def test_frozen():
    _frozen()


def test_memx():
    S = synth5.make(seed=2)
    name = "B14"
    sup, cal, sf, st = views_of(S)[name]
    ns, nc = sup.shape[0] * sup.shape[1], len(cal)
    nfix = ns + nc
    g = E5.Graph5(np.concatenate([sup.reshape(ns, -1), cal]), device=DEV)
    mem = engine.Memory(st["d_cal"], st["d_all_cal"], nc, st["med"], st["mad"], (0.3, 0.2, 0.10191613435745239))
    mx = E5.MemX(st["d_cal"], nc, st["med"], st["mad"])
    for b in np.unique(S["bidx"]):
        rows = np.flatnonzero(S["bidx"] == b)
        sims, oldn = g.append(sf[rows])
        p, _ = mx.score(st["d"][nc + rows], sims, nfix)
        ps, masks = mem.score(st["d"][nc + rows], st["d_all"][nc + rows], sims, ns, nfix, int(rows[0]))
        assert np.array_equal(p, ps[3])
        mx.admit(masks[2], int(rows[0]), sims, ns, nfix)
    assert mx.ids == mem.mem[2]


def test_graph():
    S = synth5.make(seed=3)
    sup, cal, sf, st = views_of(S)["L14"]
    g = E5.Graph5(np.concatenate([sup.reshape(-1, sup.shape[-1]), cal, sf[:300]]), device=DEV)
    a, b = g.matrix(10, 1.0, "sym").to_dense(), g.matrix2(10, 1.0, 1.0).to_dense()
    assert torch.equal(a, b)
    m0 = g.matrix2(10, 1.0, 0.0).to_dense()
    I = g.I[:, :10].cpu().numpy()
    n = len(I)
    A = np.zeros((n, n), bool)
    A[np.repeat(np.arange(n), 10), I.ravel()] = True
    mutual = A & A.T
    assert np.array_equal((m0 > 0).numpy(), mutual), "beta=0 must keep exactly the mutual edges"
    half = g.matrix2(10, 1.0, 0.5).to_dense()
    assert np.array_equal((half > 0).numpy(), A | A.T)


def test_prefix():
    S = synth5.make(seed=4)
    V = views_of(S)
    spec = E5.make_spec(cfgs=[REF], eps_lp=(0.1,), eps_fin=(0.1,), eps_joint=(0.1, 0.3), eps_full=(0.1,), frz=True,
                        neg=("M0", "Mjoint0.1"), negcfgs=[REF, (10, 0.0, 1.0, 0.5)], cen=("Mjoint0.1", "Mfull0.1", "Mfrz"), tier2=True,
                        dyn=[("lp", 0.1, 0, False), ("lp", 0.1, 3, False), ("lpst", 0.2, 3, True),
                             dict(kind="hyst", lo=0.05, hi=0.3, m=2), dict(kind="soft", eps=0.1), dict(kind="own", eps=0.1),
                             dict(kind="any", eps=0.1), dict(kind="all", eps=0.1), dict(kind="iter", base="Dlp0.1m0", eps=0.1),
                             dict(kind="bh", q=0.2), dict(kind="bh", q=0.2, storey=True), dict(kind="iter", base="B0.2", q=0.2),
                             dict(kind="iter", base="IB0.2<B0.2", q=0.2, storey=True)],
                        recal=True)
    out, adm, _ = E5.run5(V, S["bidx"], spec, DEV)
    cut = int(np.flatnonzero(S["bidx"] == 6)[0])
    rng = np.random.default_rng(0)
    V2 = {}
    for name, (sup, cal, sf, st) in V.items():
        sf2 = sf.copy()
        sf2[cut:] = synth5.l2n(rng.normal(size=sf2[cut:].shape))
        cand2 = S["cand_sf"].copy()
        st2 = E5.static5(r5.proto_view, support_dall, sup, cal, sf2, S["cand_cal"], cand2, 48, 1)
        V2[name] = (sup, cal, sf2, st2)
    out2, adm2, _ = E5.run5(V2, S["bidx"], spec, DEV)
    for name in out:
        for k in out[name]:
            d = float(np.max(np.abs(out[name][k][:cut] - out2[name][k][:cut])))
            assert d < 1e-6, (name, k, d)
        for k in adm[name]:
            if k not in {d["name"] for d in spec["dyn"]}:   # dynamic sets store the last-batch membership (diagnostic)
                assert np.array_equal(adm[name][k][:cut], adm2[name][k][:cut]), (name, k)


def test_round2():
    S = synth5.make(seed=5)
    V = views_of(S)
    spec = E5.make_spec(cfgs=[REF], frz=True, eps_full=(0.1,), eps_joint=(0.1,), neg=("M0", "Mjoint0.1"),
                        negcfgs=[REF, (10, 0.0, 1.0, 0.5)], cen=("Mfrz", "Mfull0.1"), cen_k=1)
    out, adm, _ = E5.run5(V, S["bidx"], spec, DEV)
    for name in out:
        assert np.array_equal(adm[name]["Mfrz"], adm[name]["M0"])
        assert np.max(np.abs(out[name]["Mfrz"] - out[name]["M0"])) == 0.0
        assert "Mfrzcm" in out[name] and "Mfrzcs" in out[name]
        d = np.max(np.abs(out[name]["Mfrzc"] - out[name]["M0"]))
        assert d < 1e-5, d                                   # cen_k = 1 is the nearest-member read-out (float32 cos)
        assert np.array_equal(adm["B14"]["Mfull0.1"], adm[name]["Mfull0.1"])
    # prefix_cos against a direct computation
    rng = np.random.default_rng(0)
    X = torch.as_tensor(synth5.l2n(rng.normal(size=(50, 16))))
    q = torch.as_tensor(synth5.l2n(rng.normal(size=(7, 16))))
    sims = q @ X.T
    ts, ti = sims.topk(6, dim=1)
    ts[0, 4:] = -np.inf
    got = E5.prefix_cos(ts, ti, X).numpy()
    for r in range(7):
        kk = 4 if r == 0 else 6
        best = max(float(q[r] @ (X[ti[r, :j]].sum(0) / X[ti[r, :j]].sum(0).norm())) for j in range(1, kk + 1))
        assert abs(best - got[r]) < 1e-5, (r, best, got[r])
    # dynamic sets: m = 0 keeps the raw set; the refined set is a subset; the union contains the frozen memory
    spec = E5.make_spec(cfgs=[REF], dyn=[("lp", 0.2, 0, False), ("lp", 0.2, 3, False), ("lp", 0.2, 3, True), ("lpst", 0.2, 3, False)])
    o3, a3, _ = E5.run5(V, S["bidx"], spec, DEV)
    last = int(np.flatnonzero(S["bidx"] == S["bidx"].max())[0])
    for name in o3:
        raw, ref, uni = a3[name]["Dlp0.2m0"], a3[name]["Dlp0.2m3"], a3[name]["Dlp0.2m3u"]
        assert not (ref & ~raw).any() and not (ref & ~uni).any()
        assert not (a3[name]["M0"][:last] & ~uni[:last]).any()
        assert not raw[last:].any()
        for k in ("Dlp0.2m0", "neg:Dlp0.2m3", "Dlpst0.2m3"):
            assert not np.isnan(o3[name][k]).any()
    # Benjamini-Hochberg: against a direct implementation, and monotone in q
    rng = np.random.default_rng(1)
    p = np.r_[rng.uniform(size=900), rng.uniform(size=100) * 0.01]
    for q in (0.05, 0.2):
        ps = np.sort(p)
        k = max([i + 1 for i in range(len(p)) if ps[i] <= q * (i + 1) / len(p)], default=0)
        assert E5.bh_reject(p, q).sum() == k
        assert E5.bh_reject(p, q, storey=True).sum() >= k
    assert not E5.bh_reject(rng.uniform(size=50) * 0.5 + 0.5, 0.1).any()


def test_joint():
    S, V, out, adm = _frozen()
    a, b = adm["B14"]["Mjoint0.1"], adm["L14"]["Mjoint0.1"]
    assert np.array_equal(a, b)
    # unknown classes are admitted more often than ID images by every gate of the synthetic stream
    for k in adm["B14"]:
        assert adm["B14"][k][S["is_ood"]].mean() > adm["B14"][k][~S["is_ood"]].mean(), k


def test_metrics_on_synthetic_stream():
    """The frozen product separates the synthetic unknown classes better than the static score (sanity check only)."""
    S, V, out, adm = _frozen()
    f = S["is_ood"]
    total = lambda keys: sum(out[v][k] for v in out for k in keys)
    static = metrics(total(["static"]), f)
    frozen = metrics(total(["M0", "lp0|k10g1b1l0.9"]), f)
    assert frozen["AUROC"] > static["AUROC"]
