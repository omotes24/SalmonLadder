"""Candidate-set helpers shared by the exploration and the dev2 confirmation (pure numpy)."""
import numpy as np


def class_r_all(queries, support, m, chunk=1024):
    """m-th smallest cosine distance from every query to every class's support set -> (N, n_cls)."""
    n_cls, n_sup, dim = support.shape
    flat = support.reshape(n_cls * n_sup, dim)
    out = np.empty((len(queries), n_cls))
    for lo in range(0, len(queries), chunk):
        dist = 1.0 - (queries[lo:lo + chunk] @ flat.T).reshape(-1, n_cls, n_sup)
        out[lo:lo + chunk] = np.partition(dist, m - 1, axis=-1)[..., m - 1]
    return out


def mask_from_idx(idx, n_cls):
    mask = np.zeros((idx.shape[0], n_cls), dtype=bool)
    np.put_along_axis(mask, idx, True, axis=1)
    return mask


def proto_topk_mask(queries, support, k):
    """Top-k classes by cosine to the visual prototype (mean of the support features, re-normalised)."""
    protos = support.mean(axis=1)
    protos = protos / np.linalg.norm(protos, axis=1, keepdims=True)
    idx = np.argsort(-(queries @ protos.T), axis=1)[:, :k]
    return mask_from_idx(idx, support.shape[0])


def d_over_mask(z, mask):
    return np.where(mask, z, np.inf).min(axis=1)
