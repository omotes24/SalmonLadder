"""d(x): vectorised implementation vs a naive reference, plus a hand-computed example."""
import math
import statistics

import numpy as np
import pytest

from vins.dview import d_scores, loo_stats


def unit(x):
    x = np.asarray(x, dtype=np.float64)
    return x / np.linalg.norm(x, axis=-1, keepdims=True)


def naive(support, queries, cand, m, n0, scale=1.4826):
    n_cls, n, _ = support.shape
    loo = []
    for c in range(n_cls):
        values = []
        for j in range(n):
            dists = sorted(1.0 - float(support[c, j] @ support[c, i]) for i in range(n) if i != j)
            values.append(dists[m - 1])
        loo.append(values)
    pooled = [v for values in loo for v in values]
    med_all = statistics.median(pooled)
    mad_all = scale * statistics.median([abs(v - med_all) for v in pooled])
    med_t, mad_t = [], []
    for values in loo:
        med_c = statistics.median(values)
        mad_c = scale * statistics.median([abs(v - med_c) for v in values])
        med_t.append((n * med_c + n0 * med_all) / (n + n0))
        mad_t.append((n * mad_c + n0 * mad_all) / (n + n0))
    d = []
    for x, classes in zip(queries, cand):
        zs = []
        for c in classes:
            dists = sorted(1.0 - float(x @ support[c, i]) for i in range(n))
            zs.append((dists[m - 1] - med_t[c]) / mad_t[c])
        d.append(min(zs))
    return np.array(d), np.array(med_t), np.array(mad_t), med_all, mad_all


@pytest.mark.parametrize("m", [1, 2])
def test_matches_naive_reference(m):
    rng = np.random.default_rng(0)
    n_cls, n, dim, n_q = 9, 12, 16, 50
    support = unit(rng.normal(size=(n_cls, n, dim)))
    queries = unit(rng.normal(size=(n_q, dim)))
    cand = np.stack([rng.choice(n_cls, 5, replace=False) for _ in range(n_q)])
    stats = loo_stats(support, m, n0=12)
    d, z, r, cstar = d_scores(queries, cand, support, stats, m, chunk=7)
    d_ref, med_ref, mad_ref, med_all, mad_all = naive(support, queries, cand, m, 12)
    np.testing.assert_allclose(stats["med_t"], med_ref, rtol=0, atol=1e-12)
    np.testing.assert_allclose(stats["mad_t"], mad_ref, rtol=0, atol=1e-12)
    assert abs(stats["med_all"] - med_all) < 1e-12 and abs(stats["mad_all"] - mad_all) < 1e-12
    np.testing.assert_allclose(d, d_ref, rtol=0, atol=1e-10)
    np.testing.assert_allclose(z.min(axis=1), d, rtol=0, atol=0)
    assert all(cstar[i] in cand[i] for i in range(n_q))


def test_hand_computed_example():
    # class 0: angles 0, 60, 180 deg; class 1: 90, 120, 150 deg (n = n0 = 3, m = 1)
    def pts(degrees):
        rad = np.deg2rad(degrees)
        return np.stack([np.cos(rad), np.sin(rad)], axis=1)

    support = np.stack([pts([0, 60, 180]), pts([90, 120, 150])])
    a = 1 - math.sqrt(3) / 2                         # 1 - cos 30 deg
    # LOO (m=1): class 0 -> [0.5, 0.5, 1.5]; class 1 -> [a, a, a]
    med0, mad0, med1, mad1 = 0.5, 0.0, a, 0.0
    med_all = (a + 0.5) / 2                           # pooled median of [a, a, a, .5, .5, 1.5]
    mad_all = 1.4826 * (0.5 - med_all)               # |a - med_all| == |0.5 - med_all|
    stats = loo_stats(support, m=1, n0=3)
    assert stats["med_c"] == pytest.approx([med0, med1], abs=1e-12)
    assert stats["mad_c"] == pytest.approx([mad0, mad1], abs=1e-12)
    assert stats["med_all"] == pytest.approx(med_all, abs=1e-12)
    assert stats["mad_all"] == pytest.approx(mad_all, abs=1e-12)
    med_t = [(med0 + med_all) / 2, (med1 + med_all) / 2]
    mad_t = [(mad0 + mad_all) / 2, (mad1 + mad_all) / 2]
    # query at 30 deg: r_0 = a (neighbours at 0 and 60 deg), r_1 = 0.5 (nearest: 90 deg)
    x = pts([30])
    d, z, r, cstar = d_scores(x, np.array([[0, 1]]), support, stats, m=1)
    assert r[0] == pytest.approx([a, 0.5], abs=1e-12)
    expected = [(a - med_t[0]) / mad_t[0], (0.5 - med_t[1]) / mad_t[1]]
    assert z[0] == pytest.approx(expected, abs=1e-9)
    assert d[0] == pytest.approx(min(expected), abs=1e-9) and cstar[0] == 0
    # m = 2: second-nearest distances
    stats2 = loo_stats(support, m=2, n0=3)
    _, _, r2, _ = d_scores(x, np.array([[0, 1]]), support, stats2, m=2)
    assert r2[0] == pytest.approx([a, 1.0], abs=1e-12)


def test_candidate_restriction():
    # d uses only classes in K(x): a class outside K(x) that is closer must be ignored
    support = np.stack([unit(np.eye(3)[[0, 0, 0]] + 1e-3 * np.arange(9).reshape(3, 3)),
                        unit(np.eye(3)[[1, 1, 1]] + 1e-3 * np.arange(9).reshape(3, 3))])
    stats = loo_stats(support, m=1, n0=3)
    x = unit([[1.0, 0.05, 0.0]])
    d_in, _, _, c_in = d_scores(x, np.array([[0]]), support, stats, m=1)
    d_out, _, _, c_out = d_scores(x, np.array([[1]]), support, stats, m=1)
    assert c_in[0] == 0 and c_out[0] == 1 and d_out[0] > d_in[0]
