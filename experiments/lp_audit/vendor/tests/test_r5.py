"""R5 code audits (synthetic data; run on hades and locally).

1. The R5 re-implementations equal the frozen v4 functions (iter3_eval.admit_mask / p_memory / lp_pvalues).
2. No future leakage: outputs of batches <= b do not change when features of later batches change.
3. Labels are never an input of a decision function.
4. Within-batch reference: the v4 LP lets images of the same batch influence each other; the per-image mode does not.
"""
import hashlib
import importlib.util
import inspect
import sys
import types
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
FROZEN_SHA = "40a9bf336c7ea02cb062b2b78fc48492445add34168c309857f12df670c161ad"


def _load_ev():
    """Frozen v4 code; torch-dependent modules are stubbed when torch is missing (their functions are unused)."""
    code = ROOT / "scripts" / "iter3_eval.py"
    assert hashlib.sha256(code.read_bytes()).hexdigest() == FROZEN_SHA
    try:
        import torch  # noqa: F401
    except ImportError:
        for name, attrs in (("vins.features", ("load_features",)), ("vins.tins_dev", ("import_tins",))):
            if name not in sys.modules:
                mod = types.ModuleType(name)
                for a in attrs:
                    setattr(mod, a, lambda *x, **y: None)
                sys.modules[name] = mod
    spec = importlib.util.spec_from_file_location("iter3_eval_frozen", code)
    ev = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ev)
    return ev


from vins import r5  # noqa: E402

LABEL_NAMES = {"is_ood", "is_id", "labels", "label", "group", "y_true", "flags"}


def _unit(x):
    return (x / np.linalg.norm(x, axis=1, keepdims=True)).astype(np.float32)


def _world(seed=0, n_cls=6, n_sup=12, n_cal=4, dim=24, n_batches=8, bs=40):
    """Toy stream: ID classes + two recurring OOD clusters; a view-like dict as custom_view returns it."""
    rng = np.random.default_rng(seed)
    centers = _unit(rng.normal(size=(n_cls + 2, dim)))
    support = np.stack([_unit(centers[c] + 0.35 * rng.normal(size=(n_sup, dim))) for c in range(n_cls)])
    cal = np.concatenate([_unit(centers[c] + 0.35 * rng.normal(size=(n_cal, dim))) for c in range(n_cls)])
    n = n_batches * bs
    lab = rng.integers(0, n_cls + 2, size=n)
    sf = _unit(centers[lab] + 0.35 * rng.normal(size=(n, dim)))
    bidx = np.repeat(np.arange(n_batches), bs)
    mu = _unit(support.mean(axis=1))
    d_of = lambda x: (1 - x @ mu.T).min(axis=1) * 10  # noqa: E731  (a stand-in static view, OOD direction)
    d, d_cal = d_of(sf.astype(np.float64)), d_of(cal.astype(np.float64))
    p_all = r5.pval_high(d_cal, d)
    v = {"stats": {"med_all": 0.3, "mad_all": 0.1}, "cal_feats": cal, "d_cal": d_cal,
         "support_arr": support, "d": d, "p_all": p_all}
    return v, sf, d, p_all, bidx, lab >= n_cls


def test_entrance_and_memory_equal_frozen_v4():
    ev = _load_ev()
    v, sf, d, p_all, bidx, _ = _world(0)
    ref_mask = ev.admit_mask(v, sf, d, p_all, bidx, "cand40_30_10")
    masks = r5.entrance(sf, d, p_all, bidx, v["cal_feats"], v["d_cal"], 0.3, 0.1)
    assert np.array_equal(ref_mask, masks[-1])
    p_ref, g_ref = ev.p_memory(v, sf, d, bidx, ref_mask)
    p, g = r5.memory_p(sf, d, bidx, masks[-1], v["cal_feats"], v["d_cal"], 0.3, 0.1)
    assert np.array_equal(p, p_ref) and np.array_equal(g, g_ref)


def test_lp_equals_frozen_v4():
    ev = _load_ev()
    v, sf, _, _, bidx, _ = _world(1)
    ref = ev.lp_pvalues(v, sf, bidx, k=10, alpha=0.9, gamma=3.0, iters=15)
    out = r5.lp_run(v["support_arr"], v["cal_feats"], sf, bidx, 10, 0.9, 3.0, 15, init="warm", within_batch=True)
    assert np.array_equal(out["p"], ref)


@pytest.mark.parametrize("b0", [2, 5])
def test_no_future_leakage(b0):
    v, sf, d, p_all, bidx, _ = _world(2)
    rng = np.random.default_rng(9)
    later = bidx > b0
    sf2 = sf.copy()
    sf2[later] = _unit(rng.normal(size=(later.sum(), sf.shape[1])))
    d2, p2 = d.copy(), p_all.copy()
    d2[later] = rng.normal(size=later.sum()) * 5
    p2[later] = rng.random(later.sum())
    keep = ~later
    args = (v["cal_feats"], v["d_cal"], 0.3, 0.1)
    for th in ((0.1,), (0.4, 0.1), (0.4, 0.3, 0.1)):
        a = r5.entrance(sf, d, p_all, bidx, *args, thresholds=th)
        b = r5.entrance(sf2, d2, p2, bidx, *args, thresholds=th)
        for ma, mb in zip(a, b):
            assert np.array_equal(ma[keep], mb[keep])
        pa, _ = r5.memory_p(sf, d, bidx, a[-1], *args)
        pb, _ = r5.memory_p(sf2, d2, bidx, b[-1], *args)
        assert np.array_equal(pa[keep], pb[keep])
    for init in ("warm", "cold", "converge"):
        for wb in (True, False):
            la = r5.lp_run(v["support_arr"], v["cal_feats"], sf, bidx, init=init, within_batch=wb)
            lb = r5.lp_run(v["support_arr"], v["cal_feats"], sf2, bidx, init=init, within_batch=wb)
            assert np.array_equal(la["p"][keep], lb["p"][keep]), (init, wb)


def test_within_batch_reference():
    """v4 LP (within_batch=True) is transductive inside a batch; within_batch=False scores each image alone."""
    v, sf, _, _, bidx, _ = _world(3)
    b0 = 4
    rows = np.flatnonzero(bidx == b0)
    target, others = rows[0], rows[1:]
    rng = np.random.default_rng(5)
    sf2 = sf.copy()
    sf2[others] = sf[target] + 0.01 * rng.normal(size=(len(others), sf.shape[1]))   # many near-copies of target
    sf2[others] /= np.linalg.norm(sf2[others], axis=1, keepdims=True)
    for wb, should_change in ((True, True), (False, False)):
        a = r5.lp_run(v["support_arr"], v["cal_feats"], sf, bidx, within_batch=wb)
        b = r5.lp_run(v["support_arr"], v["cal_feats"], sf2, bidx, within_batch=wb)
        changed = not np.isclose(a["u"][target], b["u"][target], rtol=0, atol=1e-12)
        assert changed == should_change, wb
    # the memory and the static view never look inside the current batch
    args = (v["cal_feats"], v["d_cal"], 0.3, 0.1)
    d = v["d"]
    m = r5.entrance(sf, d, v["p_all"], bidx, *args)[-1]
    pa, _ = r5.memory_p(sf, d, bidx, m, *args)
    pb, _ = r5.memory_p(sf2, d, bidx, m, *args)
    assert pa[target] == pb[target]


def test_labels_are_not_inputs():
    """No decision function takes or references a label-like name (docstrings are ignored: AST names only)."""
    import ast
    import textwrap

    for fn in (r5.memory_p, r5.entrance, r5.lp_run, r5._normalised_graph, r5.knn_distance, r5.maha_pp,
               r5.pval_high, r5.pval_low):
        params = set(inspect.signature(fn).parameters)
        assert not params & LABEL_NAMES, (fn.__name__, params & LABEL_NAMES)
        tree = ast.parse(textwrap.dedent(inspect.getsource(fn)))
        names = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
        names |= {n.arg for n in ast.walk(tree) if isinstance(n, ast.arg)}
        names |= {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)}
        assert not names & LABEL_NAMES, (fn.__name__, names & LABEL_NAMES)


def test_converge_and_cold_modes():
    v, sf, _, _, bidx, _ = _world(4)
    c = r5.lp_run(v["support_arr"], v["cal_feats"], sf, bidx, init="converge", tol=1e-6, max_iter=5000)
    assert (c["sweeps"] < 5000).all() and (c["sweeps"] > 15).all()
    w = r5.lp_run(v["support_arr"], v["cal_feats"], sf, bidx, init="warm")
    k = r5.lp_run(v["support_arr"], v["cal_feats"], sf, bidx, init="cold")
    assert (w["sweeps"] == 15).all() and (k["sweeps"] == 15).all()
    assert ((c["p"] > 0) & (c["p"] <= 1)).all()


def test_static_baselines():
    rng = np.random.default_rng(6)
    sup = np.stack([_unit(rng.normal(size=(12, 16)) + 3 * np.eye(16)[c]) for c in range(5)])
    d = r5.knn_distance(sup.reshape(-1, 16), sup.reshape(-1, 16), [1, 2])
    assert np.allclose(d[1], 0, atol=1e-6) and (d[2] > 0).all()
    mh = r5.maha_pp(sup, sup.mean(axis=1), [0.0, 0.1])
    far = r5.maha_pp(sup, _unit(rng.normal(size=(5, 16))), [0.1])
    assert mh[0.1].max() < far[0.1].min()


def test_metrics():
    rng = np.random.default_rng(7)
    a, b = rng.normal(1, 1, 500), rng.normal(0, 1, 400)
    from scipy.stats import mannwhitneyu

    assert np.isclose(r5.auroc(a, b), mannwhitneyu(a, b).statistic / (500 * 400))
    s = np.concatenate([a, b])
    o = np.r_[np.zeros(500, bool), np.ones(400, bool)]
    bi = np.zeros(900, int)
    assert np.isclose(r5.within_batch_auroc(s, o, bi), r5.auroc(a, b))
    ci = r5.t_interval([1.0, 2.0, 3.0, 4.0, 5.0])
    assert np.isclose(ci["mean"], 3.0) and ci["lo"] < 3.0 < ci["hi"]


def test_memory_p_rows_equals_memory_p():
    v, sf, d, p_all, bidx, _ = _world(8)
    args = (v["cal_feats"], v["d_cal"], 0.3, 0.1)
    for m in (1, 2, 5):
        adm = r5.entrance(sf, d, p_all, bidx, *args, thresholds=(0.4, 0.3, 0.1), m=m)[-1]
        full, _ = r5.memory_p(sf, d, bidx, adm, *args, m=m)
        rows = np.arange(3, len(sf), 7)
        part = r5.memory_p_rows(sf, d, bidx, adm, *args, m, rows)
        assert np.array_equal(full[rows], part)


def test_snapshot_and_stage_calibration():
    v, sf, d, p_all, bidx, _ = _world(10)
    args = (v["cal_feats"], v["d_cal"], 0.3, 0.1)
    M = r5.entrance(sf, d, p_all, bidx, *args)[-1]
    # the calibration shots themselves as extra images reproduce the leave-in p-values of the calibration
    snap = r5.memory_p_snapshot(sf, bidx, M, v["cal_feats"], v["d_cal"], sf[:5], d[:5], 0.3, 0.1, 2, [0, 3])
    assert set(snap) == {0, 3} and all(((x > 0) & (x <= 1)).all() for x in snap.values())
    # entrance_cal with the same calibration at every stage equals entrance + memory_p
    d_all = d.copy()
    cal = (v["cal_feats"], v["d_cal"], v["d_cal"])
    masks, ps, pt = r5.entrance_cal(sf, d, d_all, bidx, [cal] * 4, 0.3, 0.1)
    ref = r5.entrance(sf, d, r5.pval_high(v["d_cal"], d_all), bidx, *args)
    assert all(np.array_equal(a, b) for a, b in zip(masks, ref))
    assert np.array_equal(pt, r5.memory_p(sf, d, bidx, ref[-1], *args)[0])


def test_lp_aligned_runs_without_leakage():
    v, sf, _, _, bidx, _ = _world(11, n_batches=5)
    a = r5.lp_aligned(v["support_arr"], v["cal_feats"], sf, bidx)
    sf2 = sf.copy()
    sf2[bidx > 2] = sf2[bidx > 2][::-1]
    b = r5.lp_aligned(v["support_arr"], v["cal_feats"], sf2, bidx)
    assert ((a > 0) & (a <= 1)).all() and np.array_equal(a[bidx <= 2], b[bidx <= 2])


def test_bh_and_combination():
    p = np.array([0.001, 0.2, 0.01, 0.04, 0.5])
    assert r5.bh_mask(p, 0.1).tolist() == [True, False, True, True, False]
    v, sf, d, p_all, bidx, _ = _world(12)
    args = (v["cal_feats"], v["d_cal"], 0.3, 0.1)
    ref = r5.entrance(sf, d, p_all, bidx, *args)
    fx = r5.entrance_bh(sf, d, p_all, bidx, *args, fixed=(0.4, 0.3, 0.1))
    assert all(np.array_equal(a, b) for a, b in zip(ref, fx))
    a, b = np.array([0.5, 0.01]), np.array([0.5, 0.2])
    for rule in ("product", "cauchy", "hmp", "min"):
        c = r5.combine_p([a, b], rule)
        assert c[0] > c[1], rule


def test_proto_view_and_lp_oracle():
    ev = _load_ev()
    v, sf, d, p_all, bidx, _ = _world(13)
    sup, cal = v["support_arr"], v["cal_feats"]
    q = np.concatenate([cal, sf])
    is_cal = np.r_[np.ones(len(cal), bool), np.zeros(len(sf), bool)]
    cand = np.tile(np.arange(3), (len(q), 1))
    ref = ev.custom_view(sup, q, cand, is_cal, np.repeat(np.arange(sup.shape[0]), 4), m=2, proto=True)
    mine = r5.proto_view(sup, q, cand, is_cal, n0=12)
    assert np.allclose(ref["d"], mine["d"]) and np.array_equal(ref["p"], mine["p"])
    assert np.isclose(ref["stats"]["med_all"], mine["stats"]["med_all"])
    full = r5.lp_run(sup, cal, sf, bidx)["p"]
    orc = r5.lp_oracle(sup, cal, sf, bidx, np.ones(len(sf), bool))
    assert np.array_equal(full, orc)
    half = r5.lp_oracle(sup, cal, sf, bidx, np.arange(len(sf)) % 2 == 0)
    assert ((half > 0) & (half <= 1)).all()
