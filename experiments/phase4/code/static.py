"""Static (stream-free) prototype view of REPRISE: the frozen vins.r5.proto_view, plus d_all of support rows
(computed with the same prototypes and shrunken statistics) for the static-score propagation baseline."""
import numpy as np

from vins import r5


def support_dall(sup, n0=48, mad_scale=1.4826):
    sup = np.asarray(sup, np.float64)
    tot = sup.sum(axis=1)
    mu = tot / np.linalg.norm(tot, axis=1, keepdims=True)
    loo_mu = tot[:, None, :] - sup
    loo_mu /= np.linalg.norm(loo_mu, axis=2, keepdims=True)
    loo = 1.0 - np.einsum("cnd,cnd->cn", sup, loo_mu)
    n = loo.shape[1]
    med_c = np.median(loo, axis=1)
    mad_c = mad_scale * np.median(np.abs(loo - med_c[:, None]), axis=1)
    med_all = float(np.median(loo))
    mad_all = float(mad_scale * np.median(np.abs(loo - med_all)))
    med_t = (n * med_c + n0 * med_all) / (n + n0)
    mad_t = (n * mad_c + n0 * mad_all) / (n + n0)
    flat = sup.reshape(-1, sup.shape[-1])
    z = ((1.0 - flat @ mu.T) - med_t[None, :]) / mad_t[None, :]
    return z.min(axis=1)


def static_view(sup, cal, stream, cand_cal, cand_stream, n0=48, m=1):
    q = np.concatenate([cal, stream])
    cand = np.concatenate([cand_cal, cand_stream])
    is_cal = np.zeros(len(q), bool)
    is_cal[:len(cal)] = True
    v = r5.proto_view(sup, q, cand, is_cal, n0=n0, m=m)
    return from_proto(v, sup, n0)


def from_proto(v, sup, n0=48):
    return {"d": v["d"], "d_all": v["d_all"], "p": v["p"], "d_cal": v["d_cal"], "d_all_cal": v["d_all_cal"],
            "med": v["stats"]["med_all"], "mad": v["stats"]["mad_all"], "d_all_sup": support_dall(sup, n0)}
