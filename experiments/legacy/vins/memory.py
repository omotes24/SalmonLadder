"""Test-time visual OOD memory on top of the frozen class-conditional visual view (pure numpy).

The stream is processed in TINS's batches. Batch b is scored against the memory M_b that holds the samples
admitted in batches < b; then the admitted samples of batch b are appended. Calibration images are never
admitted, and their scores are recomputed against the same M_b, so the conformal p-value of batch b is taken
against calibration scores computed under the same memory.

r_M(x) = m-th largest cosine similarity to M turned into a distance (1 - sim); 2.0 when |M| < m.
"""
import numpy as np

EMPTY = 2.0


def top_sims(queries, mem, m_max):
    """(n, min(m_max, |mem|)) largest cosine similarities, sorted descending."""
    if len(mem) == 0:
        return np.zeros((len(queries), 0), dtype=np.float32)
    sims = queries @ mem.T
    k = min(m_max, sims.shape[1])
    if sims.shape[1] > k:
        sims = -np.partition(-sims, k - 1, axis=1)[:, :k]
    return -np.sort(-sims, axis=1)


class TopM:
    """Running top-m cosine similarities of fixed query rows to a growing memory."""

    def __init__(self, queries, m_max=2):
        self.q = np.ascontiguousarray(queries, dtype=np.float32)
        self.top = np.full((len(self.q), m_max), -np.inf, dtype=np.float32)   # descending
        self.m_max = m_max

    def add(self, new):
        if len(new) == 0:
            return
        part = top_sims(self.q, np.ascontiguousarray(new, dtype=np.float32), self.m_max)
        both = np.concatenate([self.top, part], axis=1)
        self.top = -np.sort(-both, axis=1)[:, :self.m_max]

    def dist(self, m):
        d = 1.0 - self.top[:, m - 1].astype(np.float64)
        return np.where(np.isfinite(self.top[:, m - 1]), d, EMPTY)


def memory_distances(stream_feats, batch_index, admit, cal_feats, m_max=2):
    """Returns r_stream (N, m_max) against the memory before each sample's batch and
    r_cal (n_batches, n_cal, m_max) against the same memories. Memory = admitted samples of earlier batches."""
    stream_feats = np.ascontiguousarray(stream_feats, dtype=np.float32)
    n = len(stream_feats)
    batches = np.unique(batch_index)
    assert (np.diff(batch_index) >= 0).all(), "stream must be in batch order"
    r_stream = np.empty((n, m_max))
    r_cal = np.empty((len(batches), len(cal_feats), m_max))
    cal = TopM(cal_feats, m_max)
    mem = np.zeros((0, stream_feats.shape[1]), dtype=np.float32)
    for bi, b in enumerate(batches):
        rows = np.flatnonzero(batch_index == b)
        top = top_sims(stream_feats[rows], mem, m_max)
        for m in range(1, m_max + 1):
            r_stream[rows, m - 1] = 1.0 - top[:, m - 1] if top.shape[1] >= m else EMPTY
            r_cal[bi, :, m - 1] = cal.dist(m)
        new = stream_feats[rows[admit[rows]]]
        if len(new):
            mem = np.concatenate([mem, new], axis=0)
            cal.add(new)
    return r_stream, r_cal, batches


def g_score(kind, d, r_mem, med_all, mad_all):
    """OOD-direction visual score. d: frozen view (z units); r_mem: memory distance (cosine)."""
    z_mem = (r_mem - med_all) / mad_all
    if kind == "base":
        return d
    if kind == "diff":
        return d - z_mem
    if kind == "boost":
        return d + np.maximum(0.0, -z_mem)
    if kind == "lrat":
        r_id = np.maximum(med_all + mad_all * d, 1e-3)
        return np.log(r_id) - np.log(np.maximum(r_mem, 1e-3))
    raise ValueError(kind)


def pvalues_sorted(cal_sorted, g):
    ge = len(cal_sorted) - np.searchsorted(cal_sorted, g, side="left")
    return (1.0 + ge) / (len(cal_sorted) + 1.0)


def online_pvalues(kind, m_mem, d_stream, d_cal, r_stream, r_cal, batch_index, batches, med_all, mad_all):
    """Per-batch conformal p-values of g against calibration g under the same memory."""
    g = g_score(kind, d_stream, r_stream[:, m_mem - 1], med_all, mad_all)
    p = np.empty(len(d_stream))
    for bi, b in enumerate(batches):
        rows = np.flatnonzero(batch_index == b)
        g_cal = np.sort(g_score(kind, d_cal, r_cal[bi, :, m_mem - 1], med_all, mad_all))
        p[rows] = pvalues_sorted(g_cal, g[rows])
    return p, g
