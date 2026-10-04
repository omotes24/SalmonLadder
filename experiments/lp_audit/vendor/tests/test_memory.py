import numpy as np

from vins.memory import EMPTY, memory_distances, online_pvalues


def _unit(x):
    return x / np.linalg.norm(x, axis=1, keepdims=True)


def _naive(stream, batch_index, admit, cal, m_max):
    r_s = np.empty((len(stream), m_max))
    batches = np.unique(batch_index)
    r_c = np.empty((len(batches), len(cal), m_max))
    for bi, b in enumerate(batches):
        mem = stream[(batch_index < b) & admit]
        rows = np.flatnonzero(batch_index == b)
        for m in range(1, m_max + 1):
            for target, q in ((("s", rows), stream[rows]), (("c", None), cal)):
                if len(mem) < m:
                    val = np.full(len(q), EMPTY)
                else:
                    val = 1.0 - np.sort(q @ mem.T, axis=1)[:, ::-1][:, m - 1]
                if target[0] == "s":
                    r_s[rows, m - 1] = val
                else:
                    r_c[bi, :, m - 1] = val
    return r_s, r_c


def test_memory_distances_match_naive():
    rng = np.random.default_rng(0)
    stream = _unit(rng.normal(size=(300, 16))).astype(np.float32)
    cal = _unit(rng.normal(size=(40, 16))).astype(np.float32)
    batch_index = np.repeat(np.arange(10), 30)
    admit = rng.random(300) < 0.1
    admit[:30] = False                     # first batches may add nothing
    r_s, r_c, batches = memory_distances(stream, batch_index, admit, cal, m_max=2)
    n_s, n_c = _naive(stream.astype(np.float64), batch_index, admit, cal.astype(np.float64), 2)
    assert np.allclose(r_s, n_s, atol=1e-5)
    assert np.allclose(r_c, n_c, atol=1e-5)
    assert (r_s[:30] == EMPTY).all()


def test_base_matches_static_pvalue():
    rng = np.random.default_rng(1)
    d_s, d_c = rng.normal(size=200), rng.normal(size=50)
    bi = np.repeat(np.arange(4), 50)
    r_s = rng.random((200, 2)) + 0.2
    r_c = rng.random((4, 50, 2)) + 0.2
    p, _ = online_pvalues("base", 1, d_s, d_c, r_s, r_c, bi, np.arange(4), 0.4, 0.2)
    cal = np.sort(d_c)
    ref = (1 + len(cal) - np.searchsorted(cal, d_s, side="left")) / (len(cal) + 1)
    assert np.allclose(p, ref)
