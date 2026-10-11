"""Phase 14 core: the cross-fitted ladder read-out.
The calibration images and the test images are split at random into F folds. In fold b the seeds are selected from the test
images of the fold by BH on their calibrated ranks against the calibration images of the same fold (Proposition 3 applies to
every fold). The nodes of fold b' (its calibration AND its test images) are then scored with the seeds of the other folds and
with the calibration images of the other folds as the rank reference, by a statistic s that treats the calibration and the test
images of fold b' identically; the final read-out p_final is the calibrated rank of s among the calibration images of fold b'.
Because every step that builds s is invariant to permutations of the nodes of fold b', p_final is a conformal p-value for the
ID test images of the fold (super-uniform, PRDS) and BH on p_final controls the FDR of the final selection.
Also: the same-feature baselines (prototype rank, kNN rank) and the transductive class-propagation baseline (LGC max)."""
import numpy as np
import torch

import p14common as P
import core12 as C
from engine import _log, p_high, p_low, propagate_converged
from an11 import measures

Graph = C.Graph
knn_rank = C.knn_rank


def cfg_name(rule, q, mode):
    return f"xfit|{rule}|q{q:g}|{mode}" if rule != "none" else f"xfit|{mode}"


def proto_scores(sup, cal, sf, dev="cuda", per_class=12):
    """Cosine to the nearest class prototype (mean of the support images of the class) for the calibration and the test images
    (the static score read by the cross-fit as a calibrated rank; larger = more ID-like)."""
    S = torch.as_tensor(np.asarray(sup, np.float32), device=dev)
    proto = S.reshape(-1, per_class, S.shape[1]).mean(1)
    proto = proto / proto.norm(dim=1, keepdim=True)

    def best(X):
        X = torch.as_tensor(np.asarray(X, np.float32), device=dev)
        return torch.cat([(X[lo:lo + 8192] @ proto.T).max(dim=1).values for lo in range(0, len(X), 8192)]).double().cpu().numpy()

    return best(cal), best(sf)


def view_terms(g, static_cal=None, static_test=None):
    """u+ of all nodes of one view's graph (converged), with the static scores if given (larger = more ID-like)."""
    u = g.mass(np.arange(g.ns))
    return {"u_pos": u, "static": (None if static_cal is None else np.concatenate([np.asarray(static_cal, np.float64), np.asarray(static_test, np.float64)]))}


def xfit(graphs, terms, flag, folds=2, seed=0, rules=P.RULES, qs=P.QS14, modes=P.MODES, lam=P.LAM, pairs=None, fold_of=None):
    """Cross-fitted ladder over the views (same node set in every view).
    graphs: {view: Graph}; terms: {view: view_terms(...)}; flag: is_ood of the test nodes (for the realised ID fraction; may be None).
    pairs: the (rule, q) pairs to run (default: rules x qs); fold_of: (fc, ft) fixed fold labels (default: random split, seed).
    Returns ({(rule, q, mode): {"s": nt, "pfinal": nt}} plus {("lp"): ...} and {("static"): ...}, meta)."""
    g0 = next(iter(graphs.values()))
    ns, nc, nt, nfix = g0.ns, g0.nc, g0.nt, g0.nfix
    if pairs is None:
        pairs = [(rule, q) for rule in rules for q in qs]
    pairs = list(dict.fromkeys(pairs))
    if fold_of is None:
        rng = np.random.default_rng([14, seed])
        fc, ft = P.fold_split(nc, nt, folds, rng)
    else:
        fc, ft = (np.asarray(x) for x in fold_of)
        folds = int(max(fc.max(), ft.max())) + 1
    cal_nodes = ns + np.arange(nc)                      # node indices of the calibration images
    test_nodes = nfix + np.arange(nt)
    meta = {"folds": folds, "views": {v: {} for v in graphs}}
    # seeds and u- per view, fold, (rule, q)
    uneg = {v: {} for v in graphs}
    for v, g in graphs.items():
        u = terms[v]["u_pos"]
        for b in range(folds):
            Cb, Tb = cal_nodes[fc == b], test_nodes[ft == b]
            p_plus = p_low(u[Cb], u[Tb])
            for rule, q in pairs:
                mask, pi0 = P.select_seeds(p_plus, q, len(Cb), rule)
                idx = np.flatnonzero(mask)
                meta["views"][v][f"fold{b}|{rule}|q{q:g}"] = {"seeds": int(len(idx)), "pi0_hat": float(pi0), "n_test": int(len(Tb)), "n_cal": int(len(Cb)),
                                                               "id_frac": (float(1.0 - flag[Tb[idx] - nfix].mean()) if (flag is not None and len(idx)) else None)}
                uneg[v][(b, rule, q)] = g.mass(Tb[idx], lam=lam) if len(idx) else np.zeros(g.N)
    # statistics of every non-support node, per evaluation fold
    keys = [(rule, q, mode) for (rule, q) in pairs for mode in modes] + [("none", 0.0, "lp"), ("none", 0.0, "static")]
    S = {k: np.full(nc + nt, np.nan) for k in keys}
    for b2 in range(folds):
        own_cal = cal_nodes[fc == b2]
        ref_cal = cal_nodes[fc != b2]
        nodes = np.concatenate([own_cal, test_nodes[ft == b2]])
        pos = np.concatenate([np.flatnonzero(fc == b2), nc + np.flatnonzero(ft == b2)])   # positions in the nc+nt arrays
        lpp, lps = 0.0, 0.0
        lpn = {pair: 0.0 for pair in pairs}
        for v, g in graphs.items():
            u = terms[v]["u_pos"]
            lpp = lpp + _log(p_low(u[ref_cal], u[nodes]))
            st = terms[v]["static"]
            if st is not None:
                lps = lps + _log(p_low(st[np.flatnonzero(fc != b2)], st[pos]))      # st: positions of the nc+nt arrays
            for rule, q in pairs:
                un = sum(uneg[v][(b, rule, q)] for b in range(folds) if b != b2)
                lpn[(rule, q)] = lpn[(rule, q)] + (_log(p_high(un[ref_cal], un[nodes])) if np.any(un) else 0.0)
        S[("none", 0.0, "lp")][pos] = lpp
        S[("none", 0.0, "static")][pos] = lps if np.ndim(lps) else np.nan
        for rule, q in pairs:
            base = lpp + lpn[(rule, q)]
            for mode in modes:
                if mode == "pp":
                    S[(rule, q, mode)][pos] = base
                elif mode == "pps":
                    S[(rule, q, mode)][pos] = base + lps
                elif mode == "min":
                    S[(rule, q, mode)][pos] = np.minimum(base, lps) if np.ndim(lps) else base
    out = {}
    for k, s in S.items():
        if np.isnan(s).all():
            continue
        pfinal = np.empty(nt)
        for b2 in range(folds):
            cal_pos, test_pos = np.flatnonzero(fc == b2), np.flatnonzero(ft == b2)
            pfinal[test_pos] = p_low(s[cal_pos], s[nc + test_pos])
        out[k] = {"s": s[nc:], "pfinal": pfinal, "s_cal": s[:nc]}
    meta["fold_of_test"] = ft
    meta["fold_of_cal"] = fc
    return out, meta


def validity(pfinal, flag, ft, q=0.2, alphas=(0.01, 0.05, 0.1, 0.2)):
    """Realised ID false-alarm rate and OOD recall of the rule p_final <= alpha, and the FDP of BH on p_final per fold at level q."""
    res = {}
    idm, oodm = ~flag, flag
    for a in alphas:
        res[f"fpr@{a:g}"] = float((pfinal[idm] <= a).mean())
        res[f"tpr@{a:g}"] = float((pfinal[oodm] <= a).mean())
    V, R = 0, 0
    for b in np.unique(ft):
        sel = ft == b
        mask = P.plain_bh(pfinal[sel], q)
        V += int((mask & idm[sel]).sum())
        R += int(mask.sum())
    res["bh_fdp"] = float(V / max(R, 1))
    res["bh_rejections"] = int(R)
    res["bh_recall"] = float(R and (R - V) / max(int(oodm.sum()), 1))
    return res


def score_all(graphs, terms, feats, flag, bases, labels_sup, n_classes, tag, device, pairs=None, modes=P.MODES, save_cfg=None,
              v1_qs=P.QS14, fold_of=None, with_lgc=True):
    """Every configuration of one task: the cross-fit (pairs x modes, p+ only, static), the Phase 12 read-out (asymmetric seeds,
    Storey, per q), kNN rank, LGC max; standalone and with every base in bases ({(name, seed): log base score aligned to the test
    images}). Returns (rows, meta, arrays of the saved configuration)."""
    out, meta = xfit(graphs, terms, flag, folds=2, seed=0, pairs=pairs, modes=modes, fold_of=fold_of)
    rows, arrays = [], {}

    def add(name, x, extra=None):
        au, fp = measures(x, flag)
        rows.append({**tag, "cfg": name, "base": "none", "seed": -1, "AUROC": au, "FPR95": fp, **(extra or {})})
        for (b, seed), lb in bases.items():
            au, fp = measures(x + lb, flag)
            rows.append({**tag, "cfg": name, "base": b, "seed": int(seed), "AUROC": au, "FPR95": fp})
    for (rule, q, mode), e in out.items():
        name = cfg_name(rule, q, mode)
        # the read-out of the method is p_final (log for the product with a base score); the fold-pooled statistic s is a diagnostic
        add(name, _log(e["pfinal"]), validity(e["pfinal"], flag, meta["fold_of_test"], q=(q if rule != "none" else P.Q_PAPER)))
        add(name + "|pooled", e["s"])
        if save_cfg and name == save_cfg:
            arrays["s"], arrays["pfinal"] = np.asarray(e["s"], np.float32), np.asarray(e["pfinal"], np.float32)
            arrays["fold"] = np.asarray(meta["fold_of_test"], np.int8)
    lp, ln, knn, lgc = 0.0, {q: 0.0 for q in v1_qs}, 0.0, 0.0
    for v, g in graphs.items():
        o, m = C.read_out(g, flag, qs=v1_qs)
        lp = lp + o["lp"]
        for q in v1_qs:
            ln[q] = ln[q] + o[f"ln|q{q:g}"]
        sup, cal, sf = feats[v]
        knn = knn + C.knn_rank(sup, cal, sf, dev=device)
        if with_lgc and labels_sup is not None:
            sc = lgc_max(g, labels_sup, n_classes)
            lgc = lgc + _log(p_low(sc[g.ns:g.nfix], sc[g.nfix:]))
        meta["views"][v]["v1"] = m
    add("v1|Tlp", lp)
    for q in v1_qs:
        add(f"v1|storey|q{q:g}|pp", lp + ln[q])
    add("knn", knn)
    if with_lgc and labels_sup is not None:
        add("lgcmax", lgc)
    meta.pop("fold_of_test", None)
    meta.pop("fold_of_cal", None)
    return rows, meta, arrays


@torch.no_grad()
def lgc_max(g, labels_sup, n_classes, lam=P.LAM):
    """Transductive baseline: label propagation of the class labels of the support images (Zhou et al. 2004 / TPN-style) on the
    same graph; score = the largest class mass of a node (larger = more ID-like). Returns the scores of all nodes."""
    M = g.matrix()
    labels = torch.as_tensor(np.asarray(labels_sup), device=g.dev)
    best = torch.full((g.N,), -np.inf, dtype=torch.float64, device=g.dev)
    chunk = 256                                                   # classes per pass (the columns are independent; memory ~ N x chunk)
    for c0 in range(0, n_classes, chunk):
        cs = min(chunk, n_classes - c0)
        Y = torch.zeros((g.N, cs), dtype=torch.float64, device=g.dev)
        sel = (labels >= c0) & (labels < c0 + cs)
        Y[torch.arange(g.ns, device=g.dev)[sel], labels[sel] - c0] = 1.0
        rhs = (1.0 - lam) * Y
        den = torch.linalg.vector_norm(rhs, dim=0).clamp_min(1e-30)
        U = torch.zeros_like(Y)
        for it in range(3000):
            U = lam * torch.sparse.mm(M, U) + rhs
            if it % 10 == 9:
                res = torch.linalg.vector_norm(U - lam * torch.sparse.mm(M, U) - rhs, dim=0) / den
                if float(res.max()) <= 1e-6:
                    break
        best = torch.maximum(best, U.max(dim=1).values)
        del Y, rhs, U
    torch.cuda.empty_cache()
    return best.cpu().numpy()
