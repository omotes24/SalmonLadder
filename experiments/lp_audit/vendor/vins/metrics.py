"""Metrics (a)-(g) per order seed, aggregation (mean, sd with ddof=1) and the frozen go/no-go rule."""
import math

import numpy as np
from scipy.stats import spearmanr

from . import config as C


def eps_tag(eps):
    return f"{eps * 100:g}".replace(".", "p")


def nom_col(feat, m, eps):
    return f"nom_{feat}_m{m}_eps{eps_tag(eps)}"


def _mean(mask):
    mask = np.asarray(mask)
    return float(mask.mean()) if mask.size else float("nan")


def _ratio(num, den):
    return float(num / den) if den > 0 else float("nan")


def measures(id_scores, ood_scores):
    """Upstream utils.detection_util.get_measures (ID positive, higher = more ID)."""
    from utils.detection_util import get_measures

    auroc, aupr, fpr = get_measures(np.asarray(id_scores, dtype=np.float64), np.asarray(ood_scores, dtype=np.float64))
    return {"AUROC": float(auroc), "FPR95": float(fpr), "AUPR": float(aupr)}


def _spearman(x, y):
    if len(x) < 3:
        return float("nan")
    return float(spearmanr(x, y).statistic)


def stream_independent(dv):
    """(b) nomination rates, (e) ID breakdown, (g) d-only detection. dv: dview rows of the evaluated subset."""
    grp = dv.group.values
    is_id, is_near, is_far = (dv.split.values == "id_dev"), grp == "near", grp == "far"
    out = {}
    for feat in C.FEATURES:
        # candidate sets may depend on the view (explore_candidates writes true_in_K_<feat>)
        in_k = dv[f"true_in_K_{feat}"].values if f"true_in_K_{feat}" in dv else dv.true_in_K.values
        for m in C.M_LIST:
            key = f"{feat}_m{m}"
            d = dv[f"d_{key}"].values
            view = {"g_d": {"ID_vs_near": measures(-d[is_id], -d[is_near]),
                            "ID_vs_far": measures(-d[is_id], -d[is_far])}}
            for eps in C.EPS_LIST:
                nom = dv[nom_col(feat, m, eps)].values
                sel = nom & is_id
                not_in_k = sel & ~in_k
                wrong_in_k = sel & in_k & ~dv.zs_top1_correct.values
                other = sel & dv.zs_top1_correct.values
                n_sel = int(sel.sum())
                view[eps_tag(eps)] = {
                    "b": {"P_nom_ID": _mean(nom[is_id]), "P_nom_near": _mean(nom[is_near]),
                          "P_nom_far": _mean(nom[is_far]), "P_nom_calib": _mean(nom[dv.split.values == "calib"])},
                    "e": {"n_ID_nominated": n_sel,
                          "true_not_in_K": {"n": int(not_in_k.sum()), "frac": _ratio(not_in_k.sum(), n_sel)},
                          "top1_wrong_true_in_K": {"n": int(wrong_in_k.sum()), "frac": _ratio(wrong_in_k.sum(), n_sel)},
                          "other_top1_correct": {"n": int(other.sum()), "frac": _ratio(other.sum(), n_sel)}},
                }
            out[key] = view
    return out


def per_seed(near_frame, far_frame):
    """(a), (c), (d), (f), (g: S_final) for one order seed."""
    fn, ff = near_frame, far_frame
    idn, nr = (fn.group == "ID").values, (fn.group == "near").values
    idf, fr = (ff.group == "ID").values, (ff.group == "far").values
    sn, sf = fn.seeded.values.astype(bool), ff.seeded.values.astype(bool)
    out = {
        "a": {"P_seeded_ID_nearstream": _mean(sn[idn]), "P_seeded_near": _mean(sn[nr]),
              "P_seeded_ID_farstream": _mean(sf[idf]), "P_seeded_far": _mean(sf[fr])},
        "g_S_final": {"ID_vs_near": measures(fn.S_final.values[idn], fn.S_final.values[nr]),
                      "ID_vs_far": measures(ff.S_final.values[idf], ff.S_final.values[fr])},
        "views": {},
    }
    for feat in C.FEATURES:
        for m in C.M_LIST:
            key = f"{feat}_m{m}"
            dn, df_ = fn[f"d_{key}"].values, ff[f"d_{key}"].values
            view = {"f": {"near": _spearman(dn[nr], fn.S_arrival.values[nr]),
                          "ID_nearstream": _spearman(dn[idn], fn.S_arrival.values[idn]),
                          "far": _spearman(df_[fr], ff.S_arrival.values[fr]),
                          "ID_farstream": _spearman(df_[idf], ff.S_arrival.values[idf])}}
            for eps in C.EPS_LIST:
                col = nom_col(feat, m, eps)
                nom_n, nom_f = fn[col].values.astype(bool), ff[col].values.astype(bool)
                nov_near, nov_id = nom_n & ~sn & nr, nom_n & ~sn & idn
                nov_far, nov_id_f = nom_f & ~sf & fr, nom_f & ~sf & idf
                c_rate, id_rate = _ratio(nov_near.sum(), nr.sum()), _ratio(nov_id.sum(), idn.sum())
                f_rate, idf_rate = _ratio(nov_far.sum(), fr.sum()), _ratio(nov_id_f.sum(), idf.sum())
                view[eps_tag(eps)] = {
                    "c": {"rate": c_rate, "count": int(nov_near.sum())},
                    "ID_novel_nearstream": {"rate": id_rate, "count": int(nov_id.sum())},
                    "d": {"precision_dev": _ratio(nov_near.sum(), nov_near.sum() + nov_id.sum()),
                          "precision_1to1": _ratio(c_rate, c_rate + id_rate)},
                    "far_novel": {"rate": f_rate, "count": int(nov_far.sum()),
                                  "precision_dev": _ratio(nov_far.sum(), nov_far.sum() + nov_id_f.sum()),
                                  "precision_1to1": _ratio(f_rate, f_rate + idf_rate)},
                }
            out["views"][key] = view
    return out


def aggregate(seed_dicts):
    """Mean and sd (ddof=1) over seeds for every numeric leaf."""
    first = seed_dicts[0]
    if isinstance(first, dict):
        return {k: aggregate([d[k] for d in seed_dicts]) for k in first}
    values = np.array(seed_dicts, dtype=np.float64)
    sd = float(np.std(values, ddof=1)) if len(values) > 1 else float("nan")
    return {"mean": float(np.mean(values)), "sd": sd, "per_seed": [float(v) for v in values]}


def decide(agg, criteria):
    rows, go = [], False
    for feat in criteria["features"]:
        for m in criteria["m"]:
            for eps in criteria["eps_for_decision"]:
                cell = agg["views"][f"{feat}_m{m}"][eps_tag(eps)]
                c_mean, d_mean = cell["c"]["rate"]["mean"], cell["d"]["precision_1to1"]["mean"]
                pass_c = (not math.isnan(c_mean)) and c_mean >= criteria["X"]
                pass_d = (not math.isnan(d_mean)) and d_mean >= criteria["Y"]
                rows.append({"feature": feat, "m": m, "eps": eps, "c_mean": c_mean,
                             "c_sd": cell["c"]["rate"]["sd"], "d11_mean": d_mean,
                             "d11_sd": cell["d"]["precision_1to1"]["sd"], "pass_c": pass_c,
                             "pass_d": pass_d, "pass": pass_c and pass_d})
                go = go or (pass_c and pass_d)
    return ("GO" if go else "NO-GO"), rows
