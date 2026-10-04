"""Iteration 2 method pieces (frozen once selected): visual view for any DINOv2 feature file and the
two-stage (candidate) memory admission.

Two-stage admission (cand):
  candidate memory C : stream samples with p_all <= eps_cand (appended after their batch is scored)
  p_C(x)             : conformal p-value of g_C(x) = d(x) - (r_C(x) - med_all) / mad_all against the calibration
                       shots' g_C under the same candidate memory (shots are never admitted)
  scoring memory M   : stream samples with p_C <= eps_admit
  p_t(x)             : conformal p-value of g_M(x) (CLAVIS-M c1 with M)
Under exchangeability of calibration shots and test ID images, the ID admission rate of M is about eps_admit.
"""
import numpy as np
import pandas as pd

from . import config as C
from .clavism import dev_views, views_from_arrays
from .features import load_features
from .memory import memory_distances, online_pvalues


def views_any(feat_file="dino.pt", m=2):
    """Frozen CLAVIS view (same support/calibration split and candidate sets) for any DINOv2 feature file."""
    if feat_file == "dino.pt":
        return dev_views(m=m)
    samples = pd.read_parquet(C.SPLITS_DIR / "samples.parquet")
    dview = pd.read_parquet(C.RUNS_DIR / "dview" / "dview.parquet")
    feats, blob = load_features(C.FEATURES_DIR / feat_file)
    assert blob["sample_id"] == samples.sample_id.tolist()
    feats = feats.numpy().astype(np.float32)
    row_of = pd.Series(np.arange(len(samples)), index=samples.sample_id.values)
    sup = np.flatnonzero((samples.split == "support").values)
    sup = sup[np.argsort(samples.class_idx_id.values[sup], kind="stable")]
    support = feats[sup].reshape(1000 - C.N_HELDOUT, C.N_SUPPORT, -1)
    queries = feats[row_of.loc[dview.sample_id].values]
    is_cal = (dview.split == "calib").values
    v = views_from_arrays(support, queries, np.array(dview.K_id.tolist()), is_cal, m)
    v["frame"] = pd.DataFrame({"d": v["d"], "p": v["p"], "p_all": v["p_all"]}, index=dview.sample_id.values)
    v["feats"], v["row_of"] = feats, row_of
    return v


def p_memory(v, stream_feats, d, batch_index, admit, m_mem=2, kind="diff"):
    r_s, r_c, batches = memory_distances(stream_feats, batch_index, admit, v["cal_feats"], m_max=m_mem)
    p, g = online_pvalues(kind, m_mem, d, v["d_cal"], r_s, r_c, batch_index, batches,
                          v["stats"]["med_all"], v["stats"]["mad_all"])
    return p, g


def score_stream_cand(v, stream_feats, d, p_all, batch_index, eps_cand=0.30, eps_admit=0.10, m_mem=2, kind="diff"):
    """Returns p_t, g, admitted mask (scoring memory), candidate mask."""
    cand = p_all <= eps_cand
    p_c, _ = p_memory(v, stream_feats, d, batch_index, cand, m_mem, kind)
    admit = p_c <= eps_admit
    p_t, g = p_memory(v, stream_feats, d, batch_index, admit, m_mem, kind)
    return p_t, g, admit, cand
