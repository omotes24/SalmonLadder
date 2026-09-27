"""CLAVIS-M: CLAVIS with a test-time visual OOD memory (frozen definition shared by confirmation and test).

d(x)      : CLAVIS's frozen view (DINOv2, m-th NN, robust z with shrinkage, K = CLIP zero-shot top-5)
d_all(x)  : the same z over ALL ID classes; p_all = conformal p-value of d_all against the calibration shots
memory    : stream samples with p_all <= eps_admit, appended after their batch has been scored
g(x)      : d(x) - (r_M(x) - med_all) / mad_all, r_M = m_mem-th NN cosine distance to the memory (2.0 if |M| < m_mem)
p_t(x)    : conformal p-value of g against the calibration shots' g under the same memory (shots are never admitted)
score     : S_final(x) * p_t(x)   (higher = more ID)
"""
import numpy as np
import pandas as pd

from . import config as C
from .candidates import class_r_all
from .dview import loo_stats
from .features import load_features
from .memory import memory_distances, online_pvalues, pvalues_sorted


def z_all_classes(queries, support, stats, m, chunk=4096):
    out = np.empty((len(queries), support.shape[0]))
    for lo in range(0, len(queries), chunk):
        r = class_r_all(queries[lo:lo + chunk], support, m)
        out[lo:lo + chunk] = (r - stats["med_t"][None, :]) / stats["mad_t"][None, :]
    return out


def views_from_arrays(support, queries, cand, is_cal, m=2, n0=C.N0):
    """support (n_cls, n, D); queries (N, D) incl. calibration rows; cand (N, k) candidate classes; is_cal (N,)."""
    support = np.asarray(support, dtype=np.float64)
    queries = np.asarray(queries, dtype=np.float64)
    stats = loo_stats(support, m, n0, C.MAD_SCALE)
    z = z_all_classes(queries, support, stats, m)
    d = np.take_along_axis(z, np.asarray(cand), axis=1).min(axis=1)
    d_all = z.min(axis=1)
    cal_d, cal_all = np.sort(d[is_cal]), np.sort(d_all[is_cal])
    return {"stats": stats, "d": d, "d_all": d_all, "p": pvalues_sorted(cal_d, d), "p_all": pvalues_sorted(cal_all, d_all),
            "cal_feats": queries[is_cal].astype(np.float32), "d_cal": d[is_cal], "cal_sorted": cal_d}


def dev_views(m=2):
    """Frozen views for a dev WORK (support 12 / calib 4 from the 16-shot list)."""
    samples = pd.read_parquet(C.SPLITS_DIR / "samples.parquet")
    dview = pd.read_parquet(C.RUNS_DIR / "dview" / "dview.parquet")
    feats, blob = load_features(C.FEATURES_DIR / "dino.pt")
    assert blob["sample_id"] == samples.sample_id.tolist()
    feats = feats.numpy().astype(np.float32)
    row_of = pd.Series(np.arange(len(samples)), index=samples.sample_id.values)
    n_id = 1000 - C.N_HELDOUT
    sup = np.flatnonzero((samples.split == "support").values)
    sup = sup[np.argsort(samples.class_idx_id.values[sup], kind="stable")]
    support = feats[sup].reshape(n_id, C.N_SUPPORT, -1)
    queries = feats[row_of.loc[dview.sample_id].values]
    is_cal = (dview.split == "calib").values
    v = views_from_arrays(support, queries, np.array(dview.K_id.tolist()), is_cal, m)
    assert np.allclose(v["d"], dview[f"d_dino_m{m}"].values, atol=1e-8), "frozen view mismatch"
    v["frame"] = pd.DataFrame({"d": v["d"], "p": v["p"], "p_all": v["p_all"]}, index=dview.sample_id.values)
    v["feats"], v["row_of"] = feats, row_of
    return v


def score_stream(v, stream_feats, d, p_all, batch_index, eps_admit=0.10, m_mem=2, kind="diff"):
    """Returns p_t (N,), g (N,), admitted mask (N,) for one stream in TINS batch order."""
    admit = p_all <= eps_admit
    r_s, r_c, batches = memory_distances(stream_feats, batch_index, admit, v["cal_feats"], m_max=m_mem)
    p_t, g = online_pvalues(kind, m_mem, d, v["d_cal"], r_s, r_c, batch_index, batches,
                            v["stats"]["med_all"], v["stats"]["mad_all"])
    return p_t, g, admit
