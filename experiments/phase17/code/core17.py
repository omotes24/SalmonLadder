"""Phase 17 core (registered, prereg_p17.json).

(1) sigma_ref: the paper's read-out with reference images instead of folds.
    Graph, views, lambda, k_g and the support images are those of the paper (per view the symmetric k_g-NN graph over
    [support; 4 calibration images per class; test], u+ = converged propagation of the support mass, eq. lp, lambda = 0.9).
    Labelled split (fixed by position, as in Phase 16): of the 4 calibration images of each class the first 2 are the reference R,
    the last 2 the calibration set C. Pool U = C u T.
      p_R,v(x)  = (1 + #{r in R : u+_v(r) <= u+_v(x)}) / (|R| + 1)                  for x in U
      seeds_v   = Storey-BH (q = 0.2, uncapped pi0_hat, lambda_S = floor(|R|/2)/(|R|+1)) on p_R,v over U (calibration images are
                  candidates too: the seeds are a symmetric function of the unordered pool)
      u-_v      = converged propagation of the seed mass;  u~-_v = u-_v - (1 - lambda) 1_seeds = lambda S u-_v (a node's own
                  injected mass is not counted, so every node of U is scored by the same rule)
      sigma_ref = sum_v [ log p_R,v(x) + log (1 + #{r in R : u~-_v(r) >= u~-_v(x)}) / (|R| + 1) ]
      p_fin     = (1 + #{c in C : sigma_ref(c) <= sigma_ref(x)}) / (|C| + 1)                    (no folds)
    sigma_ref is invariant to permutations of U (R, S fixed), so p_fin is a conformal p-value (AdaDetect Thm 3.3).
(2) Descriptive (no selection): the decomposition u = sqrt(delta) * q, q = u / sqrt(delta), of the paper's statistic.
(3) Descriptive: the separation bounds on the recurrence grid -- Corollary 2 for sigma and sigma_ref, Corollary A1 for h."""
import numpy as np
import torch

import p17common as P
import core16 as H
from engine import _log, p_high, p_low
from an11 import measures

ALPHAS = (0.01, 0.05, 0.1, 0.2)
SIG = "xfit|storey|q0.2|pp|pooled"          # the paper's statistic sigma (cross-fitted; ranking rows)
PF = "xfit|storey|q0.2|pp"                  # its calibrated rank p_fin (validity rows)
REF_S = "ref|storey|q0.2|pp|stat"           # sigma_ref (ranking rows)
REF_P = "ref|storey|q0.2|pp"                # its calibrated rank p_fin (validity rows)


def degrees(g):
    """W = max(A, A^T) of the view's graph (k_g = 10; the arithmetic of the propagation matrix) as a CPU CSR, and the degrees."""
    W, d = H.sym_W(g.graph)
    W = W.cpu()
    return {"crow": W.crow_indices().numpy(), "col": W.col_indices().numpy(), "val": W.values().numpy(), "deg": d.cpu().numpy()}


def row(Wd, i):
    a, b = Wd["crow"][i], Wd["crow"][i + 1]
    return Wd["col"][a:b], Wd["val"][a:b]


def validity(p, flag, q=0.2):
    res = {}
    for a in ALPHAS:
        res[f"fpr@{a:g}"] = float((p[~flag] <= a).mean())
        res[f"tpr@{a:g}"] = float((p[flag] <= a).mean())
    mask = P.P.plain_bh(p, q)
    V, Rn = int((mask & ~flag).sum()), int(mask.sum())
    res["bh_fdp"] = float(V / max(Rn, 1))
    res["bh_rejections"] = Rn
    res["bh_recall"] = float((Rn - V) / max(int(flag.sum()), 1))
    return res


# ---------------------------------------------------------------- (1) sigma_ref
def sigma_ref(graphs, terms, q=P.Q17, lam=P.LAM17, rule="storey"):
    g0 = next(iter(graphs.values()))
    ns, nc, nt, nfix = g0.ns, g0.nc, g0.nt, g0.nfix
    R, C = H.split_cal(nc, ns, P.N_REF, P.N_CAL_PER_CLASS)
    test_nodes = nfix + np.arange(nt)
    pool = np.concatenate([C, test_nodes])
    lpp, lpn = 0.0, 0.0
    info, seeds_of, uneg = {"views": {}}, {}, {}
    for v, g in graphs.items():
        u = terms[v]["u_pos"]
        pR = p_low(u[R], u[pool])
        mask, pi0 = P.M15.select_seeds(pR, q, len(R), rule)
        seeds = pool[mask]
        if len(seeds):
            un = g.mass(seeds, lam=lam).copy()
            un[seeds] -= (1.0 - lam)                      # leave-self-out: u~- = lambda S u-
            un = np.maximum(un, 0.0)
        else:
            un = np.zeros(g.N)
        lpp = lpp + _log(pR)
        lpn = lpn + (_log(p_high(un[R], un[pool])) if len(seeds) else 0.0)
        seeds_of[v], uneg[v] = seeds, un
        info["views"][v] = {"seeds": int(len(seeds)), "cal_seeds": int(np.isin(seeds, C).sum()), "pi0_hat": float(pi0)}
    s = lpp + lpn
    lp_only = np.asarray(lpp, np.float64)
    pfin = p_low(s[: len(C)], s[len(C):])
    pfin_lp = p_low(lp_only[: len(C)], lp_only[len(C):])
    info.update({"n_ref": int(len(R)), "n_cal": int(len(C)), "nfix": int(nfix)})
    return {"s": s[len(C):], "s_cal": s[: len(C)], "pfin": pfin, "lp": lp_only[len(C):], "pfin_lp": pfin_lp, "R": R, "C": C,
            "seeds": seeds_of, "uneg": uneg, "info": info}


def ref_rows(out, flag, bases, tag):
    rows = []
    nfix = out["info"]["nfix"]
    seed_tag = {}
    for v, sd in out["seeds"].items():
        idf = float(np.mean([(i < nfix) or (not flag[i - nfix]) for i in sd])) if len(sd) else None
        seed_tag.update({f"seeds_ref_{v}": int(len(sd)), f"cal_seeds_ref_{v}": out["info"]["views"][v]["cal_seeds"], f"seed_idfrac_ref_{v}": idf})
    for name, x, extra in ((REF_S, out["s"], {}), (REF_P, _log(out["pfin"]), validity(out["pfin"], flag)),
                           ("ref|lp|stat", out["lp"], {}), ("ref|lp", _log(out["pfin_lp"]), validity(out["pfin_lp"], flag))):
        au, fp = measures(x, flag)
        rows.append({**tag, "cfg": name, "base": "none", "seed": -1, "AUROC": au, "FPR95": fp, **extra, **seed_tag})
        for (b, seed), lb in bases.items():
            au, fp = measures(x + lb, flag)
            rows.append({**tag, "cfg": name, "base": b, "seed": int(seed), "AUROC": au, "FPR95": fp})
    return rows


# ---------------------------------------------------------------- the paper's cross-fit, recomputed with its internals
def xfit_internals(graphs, terms, seed=0, q=0.2, rule="storey", lam=P.LAM17):
    """Phase 14 xfit for (storey, 0.2, pp), folds 2, the same fold split (rng [14, seed]); returns the fold labels, the seeds and u-
    of every (view, fold) and sigma of the calibration and test nodes (must equal the paper's statistic)."""
    g0 = next(iter(graphs.values()))
    ns, nc, nt, nfix = g0.ns, g0.nc, g0.nt, g0.nfix
    rng = np.random.default_rng([14, seed])
    fc, ft = P.fold_split(nc, nt, 2, rng)
    cal_nodes, test_nodes = ns + np.arange(nc), nfix + np.arange(nt)
    seeds, uneg = {}, {}
    for v, g in graphs.items():
        u = terms[v]["u_pos"]
        for b in range(2):
            Cb, Tb = cal_nodes[fc == b], test_nodes[ft == b]
            mask, _ = P.select_seeds(p_low(u[Cb], u[Tb]), q, len(Cb), rule)
            seeds[(v, b)] = Tb[np.flatnonzero(mask)]
            uneg[(v, b)] = g.mass(seeds[(v, b)], lam=lam) if len(seeds[(v, b)]) else np.zeros(g.N)
    return {"fc": fc, "ft": ft, "cal_nodes": cal_nodes, "test_nodes": test_nodes, "seeds": seeds, "uneg": uneg, "nc": nc, "nt": nt}


def xfit_stat(graphs, terms, xi, transform=None):
    """sigma (fold-pooled statistic, calibration then test positions) with the read-out ranks taken on transform(u, v)."""
    tr = transform or (lambda x, v: x)
    fc, ft, cal_nodes, test_nodes, nc, nt = xi["fc"], xi["ft"], xi["cal_nodes"], xi["test_nodes"], xi["nc"], xi["nt"]
    s = np.full(nc + nt, np.nan)
    for b2 in range(2):
        own_cal, ref_cal = cal_nodes[fc == b2], cal_nodes[fc != b2]
        nodes = np.concatenate([own_cal, test_nodes[ft == b2]])
        pos = np.concatenate([np.flatnonzero(fc == b2), nc + np.flatnonzero(ft == b2)])
        tot = 0.0
        for v in graphs:
            up, un_raw = tr(terms[v]["u_pos"], v), xi["uneg"][(v, 1 - b2)]
            un = tr(un_raw, v)
            tot = tot + _log(p_low(up[ref_cal], up[nodes]))
            tot = tot + (_log(p_high(un[ref_cal], un[nodes])) if np.any(un_raw) else 0.0)
        s[pos] = tot
    return s


# ---------------------------------------------------------------- (2) decomposition u = sqrt(delta) * q
def decomposition_rows(graphs, terms, xi, Wd, flag, tag):
    """Descriptive rows (standalone): the density factor sqrt(delta) alone, the probability factor q = u+/sqrt(delta) alone and
    their product u+ (each the sum over views of the log rank among all calibration images), and the paper's statistic with
    u+- replaced by u+-/sqrt(delta) (seeds unchanged)."""
    g0 = next(iter(graphs.values()))
    ns, nfix = g0.ns, g0.nfix
    C, T = np.arange(ns, nfix), np.arange(nfix, g0.N)
    rows, sc = [], {"dec|deg": 0.0, "dec|q": 0.0, "dec|u": 0.0}
    for v in graphs:
        u, rd = terms[v]["u_pos"], np.sqrt(np.maximum(Wd[v]["deg"], 1e-300))
        for key, x in (("dec|deg", rd), ("dec|q", u / rd), ("dec|u", u)):
            sc[key] = sc[key] + _log(p_low(x[C], x[T]))
    nc = xi["nc"]
    sc["dec|sigma_q"] = xfit_stat(graphs, terms, xi, lambda x, v: x / np.sqrt(np.maximum(Wd[v]["deg"], 1e-300)))[nc:]
    for name, x in sc.items():
        au, fp = measures(x, flag)
        rows.append({**tag, "cfg": name, "base": "none", "seed": -1, "AUROC": au, "FPR95": fp})
    return rows


# ---------------------------------------------------------------- (3) separation bounds
def class_escape(Wd, nodes, members, rho):
    """For a node set Q (members, bool over all nodes) and its nodes: shares e (weight to outside Q), the matrix P_QQ and the
    discounted escape probability pi = smallest non-negative solution of pi = rho (e + P_QQ pi) (rho scalar or per node)."""
    k = len(nodes)
    loc = {int(i): j for j, i in enumerate(nodes)}
    Pq = np.zeros((k, k))
    e = np.zeros(k)
    out_nb = set()
    for j, i in enumerate(nodes):
        cols, vals = row(Wd, i)
        d = Wd["deg"][i]
        inq = members[cols]
        e[j] = vals[~inq].sum() / d
        for c, w in zip(cols[inq], vals[inq]):
            Pq[j, loc[int(c)]] += w / d
        out_nb.update(int(c) for c in cols[~inq])
    rr = np.full(k, rho) if np.isscalar(rho) else np.asarray(rho, np.float64)
    pi = np.linalg.solve(np.eye(k) - rr[:, None] * Pq, rr * e)
    return np.clip(pi, 0.0, 1.0), e, Pq, np.array(sorted(out_nb), dtype=np.int64)


def corollary1_b(Wd, u, nodes, members, lam):
    """Corollary 1 (paper, discrete walk stopped with probability 1 - lambda): b_i = pi_i sqrt(delta_i) max_{j in dQ} u_j/sqrt(delta_j)."""
    pi, e, _, nb = class_escape(Wd, nodes, members, lam)
    rd = np.sqrt(Wd["deg"])
    mx = float(np.max(u[nb] / rd[nb])) if len(nb) else 0.0
    return pi * rd[nodes] * mx, pi, e


def seed_lower(Wd, y, seed_mask, lam):
    """c(y) = lambda (1 - lambda) sum_{j seed} W(y, j) / sqrt(delta_y delta_j) <= lambda (S u-)(y)  (u-_j >= 1 - lambda on seeds)."""
    cols, vals = row(Wd, y)
    m = seed_mask[cols]
    if not m.any():
        return 0.0
    d = Wd["deg"]
    return float(lam * (1.0 - lam) * np.sum(vals[m] / np.sqrt(d[y] * d[cols[m]])))


def _plow_sorted(ref_sorted, x):
    return (1 + np.searchsorted(ref_sorted, x, side="right")) / (len(ref_sorted) + 1)


def _phigh_sorted(ref_sorted, x):
    return (1 + len(ref_sorted) - np.searchsorted(ref_sorted, x, side="left")) / (len(ref_sorted) + 1)


def corollary2_sigma(graphs, terms, xi, Wd, s_all, cls_of_test, lam=P.LAM17):
    """Corollary 2 for the paper's sigma (cross-fit). For every unknown test image y (class Q = all test images of its class;
    Q contains no support image), with its own fold b and the other fold b':
        u+_v(y) <= b_v(y)  (Corollary 1);  u-_{v,b'}(y) >= c_v(y)  (y is not a seed of fold b');
        sigma(y) <= B(y) = sum_v [log p_low(u+_v[C_b'], b_v(y)) + log p_high(u-_{v,b'}[C_b'], c_v(y))];
        p_fin(y) <= pbar(y) = p_low(sigma[C_b], B(y)).
    Returns arrays over the unknown test images: sigma, B, pbar, p_fin (actual)."""
    g0 = next(iter(graphs.values()))
    N, nc = g0.N, xi["nc"]
    fc, ft, cal_nodes, test_nodes = xi["fc"], xi["ft"], xi["cal_nodes"], xi["test_nodes"]
    unk = np.flatnonzero(cls_of_test >= 0)
    up_ref, un_ref, smask, own_cal = {}, {}, {}, {}
    for b in range(2):
        ref = cal_nodes[fc != b]                      # the reference calibration images of the nodes of fold b
        own_cal[b] = np.sort(s_all[np.flatnonzero(fc == b)])
        for v in graphs:
            up_ref[(v, b)] = np.sort(terms[v]["u_pos"][ref])
            sd = xi["seeds"][(v, 1 - b)]
            m = np.zeros(N, bool)
            m[sd] = True
            smask[(v, b)] = m
            un_ref[(v, b)] = np.sort(xi["uneg"][(v, 1 - b)][ref]) if len(sd) else None
    B = np.zeros(len(unk))
    for k in np.unique(cls_of_test[unk]):
        sel = np.flatnonzero(cls_of_test[unk] == k)
        loc = unk[sel]
        nodes = test_nodes[loc]
        members = np.zeros(N, bool)
        members[nodes] = True
        for v in graphs:
            b_v, _, _ = corollary1_b(Wd[v], terms[v]["u_pos"], nodes, members, lam)
            for j, (t, y) in enumerate(zip(loc, nodes)):
                fb = int(ft[t])
                B[sel[j]] += float(np.log(_plow_sorted(up_ref[(v, fb)], b_v[j])))
                if un_ref[(v, fb)] is not None:
                    c = seed_lower(Wd[v], y, smask[(v, fb)], lam)
                    B[sel[j]] += float(np.log(_phigh_sorted(un_ref[(v, fb)], c)))
    folds = ft[unk].astype(int)
    pbar = np.array([_plow_sorted(own_cal[f], x) for f, x in zip(folds, B)])
    pfin = np.array([_plow_sorted(own_cal[f], x) for f, x in zip(folds, s_all[nc + unk])])
    return {"sigma": s_all[nc + unk], "B": B, "pbar": pbar, "pfin": pfin}


def corollary2_ref(graphs, terms, ref, Wd, cls_of_test, lam=P.LAM17):
    """Corollary 2 for sigma_ref: reference R for both ranks, seeds from the pool, u~- = lambda S u- (so c(y) bounds it for every
    y, seed or not); p_fin(y) <= p_low(sigma_ref[C], B(y))."""
    g0 = next(iter(graphs.values()))
    N, nfix = g0.N, g0.nfix
    R = ref["R"]
    unk = np.flatnonzero(cls_of_test >= 0)
    up_ref, un_ref, smask = {}, {}, {}
    for v in graphs:
        up_ref[v] = np.sort(terms[v]["u_pos"][R])
        sd = ref["seeds"][v]
        m = np.zeros(N, bool)
        m[sd] = True
        smask[v] = m
        un_ref[v] = np.sort(ref["uneg"][v][R]) if len(sd) else None
    B = np.zeros(len(unk))
    for k in np.unique(cls_of_test[unk]):
        sel = np.flatnonzero(cls_of_test[unk] == k)
        nodes = nfix + unk[sel]
        members = np.zeros(N, bool)
        members[nodes] = True
        for v in graphs:
            b_v, _, _ = corollary1_b(Wd[v], terms[v]["u_pos"], nodes, members, lam)
            for j, y in enumerate(nodes):
                B[sel[j]] += float(np.log(_plow_sorted(up_ref[v], b_v[j])))
                if un_ref[v] is not None:
                    c = seed_lower(Wd[v], y, smask[v], lam)
                    B[sel[j]] += float(np.log(_phigh_sorted(un_ref[v], c)))
    cal_sorted = np.sort(ref["s_cal"])
    pbar = np.array([_plow_sorted(cal_sorted, x) for x in B])
    return {"sigma": ref["s"][unk], "B": B, "pbar": pbar, "pfin": ref["pfin"][unk]}


def h16_internals(graphs, lam=P.LAM16, grel=P.GREL16, q=P.Q16):
    """The registered Phase 16 read-out h (core16.h16) with its internals: union graph (CPU CSR), degrees, gamma, seeds, h, s."""
    g0 = next(iter(graphs.values()))
    ns, nc, nt, nfix, N = g0.ns, g0.nc, g0.nt, g0.nfix, g0.N
    dev = g0.dev
    W, d, _ = H.union_graph(graphs)
    gamma = grel * float(d.median())
    R, C = H.split_cal(nc, ns)
    test_nodes = nfix + np.arange(nt)
    pool = np.concatenate([C, test_nodes])
    S = torch.zeros(N, dtype=torch.bool, device=dev)
    S[:ns] = True
    none = torch.zeros(N, dtype=torch.bool, device=dev)
    h0, _, _ = H.solve_h(W, d, S, none, lam, gamma)
    h0 = h0.cpu().numpy()
    mask, _ = P.M14.select_seeds(p_low(h0[R], h0[pool]), q, len(R), "storey")
    seeds = pool[mask]
    Z = torch.zeros(N, dtype=torch.bool, device=dev)
    if len(seeds):
        Z[torch.as_tensor(seeds, device=dev)] = True
    h, _, _ = H.solve_h(W, d, S, Z, lam, gamma)
    s = (lam * torch.sparse.mm(W, h[:, None])[:, 0] / (d + gamma)).cpu().numpy()
    Wc = W.cpu()
    Wd = {"crow": Wc.crow_indices().numpy(), "col": Wc.col_indices().numpy(), "val": Wc.values().numpy(), "deg": d.cpu().numpy()}
    return {"Wd": Wd, "gamma": gamma, "lam": lam, "R": R, "C": C, "seeds": seeds, "h": h.cpu().numpy(), "s": s,
            "p": p_low(s[C], s[test_nodes]), "ns": ns, "nfix": nfix, "N": N}


def corollaryA1(hi, cls_of_test):
    """Corollary A1 for h: for every non-seed unknown test image y, Q = the non-seed test images of its class,
        pi_Q(y) (Theorem A3, exact), phibar = phi(max rho, max e, min zeta) (Theorem A3 bound),
        pbar_l(y) = (1 + #{c in C : l(c) <= pi_Q(y)}) / (|C| + 1) with l(c) = rho_c omega_c (calibration-side lower bound, A4),
        pbar_s(y) = (1 + #{c in C : s(c) <= pi_Q(y)}) / (|C| + 1) (the actual calibration scores), and the actual p(y);
    on the calibration images: l(c) / s(c)."""
    Wd, gam, lam = hi["Wd"], hi["gamma"], hi["lam"]
    d = Wd["deg"]
    rho = lam * d / (d + gam)
    N, ns, nfix = hi["N"], hi["ns"], hi["nfix"]
    seed_mask = np.zeros(N, bool)
    seed_mask[hi["seeds"]] = True
    C = hi["C"]
    omega = np.zeros(N)
    for c in C:
        cols, vals = row(Wd, c)
        omega[c] = vals[cols < ns].sum() / d[c]
    ell = rho[C] * omega[C]
    s_cal = hi["s"][C]
    out = {"h": [], "s": [], "pi": [], "phibar": [], "pbar_l": [], "pbar_s": [], "p": [], "cls": []}
    test_nodes = nfix + np.arange(len(cls_of_test))
    for k in np.unique(cls_of_test[cls_of_test >= 0]):
        nodes = test_nodes[(cls_of_test == k)]
        nodes = nodes[~seed_mask[nodes]]
        if len(nodes) == 0:
            continue
        members = np.zeros(N, bool)
        members[nodes] = True
        k_ = len(nodes)
        loc = {int(i): j for j, i in enumerate(nodes)}
        Pq = np.zeros((k_, k_))
        e = np.zeros(k_)
        zeta = np.zeros(k_)
        for j, i in enumerate(nodes):
            cols, vals = row(Wd, i)
            inq, sd = members[cols], seed_mask[cols]
            e[j] = vals[~inq & ~sd].sum() / d[i]
            zeta[j] = vals[sd].sum() / d[i]
            for cc, w in zip(cols[inq], vals[inq]):
                Pq[j, loc[int(cc)]] += w / d[i]
        r = rho[nodes]
        pi = np.clip(np.linalg.solve(np.eye(k_) - r[:, None] * Pq, r * e), 0.0, 1.0)
        rb, eb, zb = r.max(), e.max(), zeta.min()
        phib = (rb * eb / (1.0 - rb * (1.0 - eb - zb))) if eb > 0 else 0.0
        for j, i in enumerate(nodes):
            out["h"].append(hi["h"][i]); out["s"].append(hi["s"][i]); out["pi"].append(pi[j]); out["phibar"].append(phib)
            out["pbar_l"].append(float(p_low(ell, np.array([pi[j]]))[0]))
            out["pbar_s"].append(float(p_low(s_cal, np.array([pi[j]]))[0]))
            out["p"].append(float(p_low(s_cal, np.array([hi["s"][i]]))[0]))
            out["cls"].append(int(k))
    out = {k: np.asarray(v) for k, v in out.items()}
    out["ell_over_s_cal"] = ell / np.maximum(s_cal, 1e-300)
    return out
