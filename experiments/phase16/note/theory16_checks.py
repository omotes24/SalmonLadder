"""Numerical checks of every statement of the Phase 16 theory note (synthetic data, CPU, no OpenOOD data).
Object: h(x) = E_x[lam^N exp(-gamma T); tau_A < tau_Z] for the continuous-time random walk with jump rates W_ij
(holding time Exp(delta_i), jump to j with probability W_ij / delta_i), A = support images (value 1), Z = seeds (value 0).
Writes checks16.json (read by make_note16.py)."""
import json
import sys
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spl
from scipy.stats import spearmanr

OUT = Path(__file__).with_name("checks16.json")


def l2n(x):
    return x / np.linalg.norm(x, axis=1, keepdims=True)


def knn_graph(X, k):
    S = X @ X.T
    np.fill_diagonal(S, -np.inf)
    I = np.argpartition(-S, k, axis=1)[:, :k]
    V = np.take_along_axis(S, I, 1).clip(0)
    n = len(X)
    A = sp.csr_matrix((V.ravel(), (np.repeat(np.arange(n), k), I.ravel())), shape=(n, n))
    return A.maximum(A.T).tocsr()


def solve(W, one, zero, lam, gam, rhs=None):
    """(D + gam I - lam W)_FF h_F = lam W_F,one 1 (or a given right-hand side on F); h = 1 on one, 0 on zero."""
    n = W.shape[0]
    d = np.asarray(W.sum(1)).ravel()
    bnd = np.zeros(n, bool)
    bnd[np.asarray(one, int)] = True
    bnd[np.asarray(zero, int)] = True
    F = np.flatnonzero(~bnd)
    M = (sp.diags(d[F] + gam) - lam * W[F][:, F]).tocsc()
    if rhs is None:
        b = lam * np.asarray(W[F][:, np.asarray(one, int)].sum(1)).ravel()
    else:
        b = rhs(F, d)
    h = np.zeros(n)
    h[np.asarray(one, int)] = 1.0
    h[F] = spl.spsolve(M, b)
    return h, F


def instance(rng, C=100, D=64, n_ref=2, n_pool=2400, n_cal=400, n_unk=10, r=20, n_far=200, s_id=1.0, s_unk=0.6):
    """C ID classes (12 support + n_ref reference images each), an ID pool split at random into calibration and test images
    (exchangeable by construction), n_unk recurring unknown classes of r images (tight) and n_far diffuse unknowns."""
    mu = l2n(rng.normal(size=(C + n_unk, D)))

    def draw(c, n, s):
        return l2n(mu[c] + s * rng.normal(size=(n, D)) / np.sqrt(D))
    sup = np.concatenate([draw(c, 12, s_id) for c in range(C)])
    ref = np.concatenate([draw(c, n_ref, s_id) for c in range(C)])
    cls = rng.integers(C, size=n_pool)
    pool_id = np.stack([draw(c, 1, s_id)[0] for c in cls])
    perm = rng.permutation(n_pool)
    cal_i, tid_i = perm[:n_cal], perm[n_cal:]
    unk = np.concatenate([draw(C + j, r, s_unk) for j in range(n_unk)])
    far = l2n(rng.normal(size=(n_far, D)))
    X = np.concatenate([sup, ref, pool_id[cal_i], pool_id[tid_i], unk, far])
    ns, nr, nc, nti, nu, nf = len(sup), len(ref), n_cal, len(tid_i), len(unk), n_far
    o = np.cumsum([0, ns, nr, nc, nti, nu, nf])
    idx = {"S": np.arange(o[0], o[1]), "R": np.arange(o[1], o[2]), "C": np.arange(o[2], o[3]), "Tid": np.arange(o[3], o[4]),
           "U": np.arange(o[4], o[5]), "Far": np.arange(o[5], o[6])}
    idx["T"] = np.arange(o[3], o[6])
    flag = np.r_[np.zeros(nti, bool), np.ones(nu + nf, bool)]
    labels = {"S": np.repeat(np.arange(C), 12), "cal": cls[cal_i], "tid": cls[tid_i], "unk": np.repeat(np.arange(n_unk), r), "r": r, "n_unk": n_unk}
    return X, idx, flag, labels


def rank_low(ref, x):
    ref = np.sort(ref)
    return (1.0 + np.searchsorted(ref, x, side="right")) / (len(ref) + 1.0)


def storey_bh(p, q, n_ref):
    m = len(p)
    K = n_ref // 2
    lam_s = K / (n_ref + 1.0)
    pi0 = (1.0 + np.sum(p >= lam_s)) / (m * (1.0 - lam_s))          # not capped at 1 (the AdaDetect estimator, as on dev1)
    o = np.argsort(p, kind="stable")
    ok = np.flatnonzero(p[o] <= q * np.arange(1, m + 1) / (pi0 * m))
    k = ok.max() + 1 if len(ok) else 0
    mask = np.zeros(m, bool)
    mask[o[:k]] = True
    return mask, pi0


def bh(p, q):
    m = len(p)
    o = np.argsort(p, kind="stable")
    ok = np.flatnonzero(p[o] <= q * np.arange(1, m + 1) / m)
    k = ok.max() + 1 if len(ok) else 0
    mask = np.zeros(m, bool)
    mask[o[:k]] = True
    return mask


def method(X, idx, lam, gam_rel, q=0.2, k=10, design="pool"):
    """The Phase 16 read-out on one instance: stage 1 (A = S, no seeds), seeds by Storey-BH over the pool C u T against R,
    stage 2 (A = S, Z = seeds), score s(x) = rho_x (P h)(x) for every unlabelled x, final p = rank among C."""
    W = knn_graph(X, k)
    d = np.asarray(W.sum(1)).ravel()
    gam = gam_rel * np.median(d)
    S, R, Cn, T = idx["S"], idx["R"], idx["C"], idx["T"]
    g1, _ = solve(W, S, [], lam, gam)
    if design == "pool":                      # invariant: candidates C u T, reference R
        pool = np.concatenate([Cn, T])
        mask, _ = storey_bh(rank_low(g1[R], g1[pool]), q, len(R))
        seeds = pool[mask]
    else:                                     # not invariant: candidates T only, reference C (seeds see which nodes are C)
        mask, _ = storey_bh(rank_low(g1[Cn], g1[T]), q, len(Cn))
        seeds = T[mask]
    h, _ = solve(W, S, seeds, lam, gam)
    s = lam * (W @ h) / (d + gam)              # one formula for every unlabelled node (= h off the seeds)
    p = rank_low(s[Cn], s[T])
    return W, d, gam, h, s, p, seeds


def main(reps):
    rng = np.random.default_rng(16)
    out = {}
    X, idx, flag, lab = instance(rng)
    W = knn_graph(X, 10)
    d = np.asarray(W.sum(1)).ravel()
    P = (sp.diags(1 / d) @ W).tocsr()
    S = idx["S"]
    lam, gam = 0.95, 0.0625 * float(np.median(d))
    # 1. linear system vs Monte Carlo of the walk (holding Exp(delta), killing per jump 1 - lam, per time gam)
    h0, _ = solve(W, S, [], lam, gam)
    mc_rows = []
    for x in (idx["C"][3], idx["U"][2], idx["Far"][5]):
        tot, n = 0.0, 20000
        for _ in range(n):
            v, w = x, 1.0
            while True:
                w *= np.exp(-gam * rng.exponential(1 / d[v]))
                row = P.getrow(v)
                v = rng.choice(row.indices, p=row.data)
                w *= lam
                if v < len(S):
                    tot += w
                    break
                if w < 1e-7:
                    break
        mc_rows.append({"node": int(x), "solve": float(h0[x]), "mc": tot / n, "mc_se": float(np.sqrt(h0[x] * (1 - h0[x]) / n))})
    out["mc"] = mc_rows
    # 2. Dirichlet principle: h minimises E(f) = lam sum_{i<j} W_ij (f_i - f_j)^2 + sum_i ((1 - lam) delta_i + gam) f_i^2
    Wc = sp.triu(W, 1).tocoo()

    def energy(f):
        return lam * np.sum(Wc.data * (f[Wc.row] - f[Wc.col]) ** 2) + np.sum(((1 - lam) * d + gam) * f ** 2)
    F = np.setdiff1d(np.arange(len(X)), S)
    e0 = energy(h0)
    worse = []
    for t in range(200):
        f = h0.copy()
        f[F] += 1e-3 * rng.normal(size=len(F))
        worse.append(energy(f) > e0)
    # Kirchhoff residual (electrical form) at the free nodes
    kcl = lam * (d * h0 - W @ h0) + ((1 - lam) * d + gam) * h0
    out["energy"] = {"E_min": float(e0), "all_perturbations_higher": bool(all(worse)), "n_perturbations": 200,
                     "kcl_max_residual": float(np.abs(kcl[F]).max())}
    # 3. scale covariance: h(cW; lam, c gam) = h(W; lam, gam)
    hc, _ = solve(3.7 * W, S, [], lam, 3.7 * gam)
    out["scale"] = {"max_abs_diff": float(np.abs(hc - h0).max())}
    # 4. three fates: P(reach A first) + P(reach Z first) + P(killed first) = 1
    Z = idx["U"][::3]
    hA, _ = solve(W, S, Z, lam, gam)
    hZ, _ = solve(W, Z, S, lam, gam)
    dd, F2 = solve(W, np.array([], int), np.concatenate([S, Z]), lam, gam, rhs=lambda F, d_: gam + (1 - lam) * d_[F])
    out["fates"] = {"max_abs_dev_from_1": float(np.abs((hA + hZ + dd)[F2] - 1).max())}
    # 5. additivity over the classes of A
    parts = np.zeros(len(X))
    for c in np.unique(lab["S"]):
        Sc = S[lab["S"] == c]
        hc_, _ = solve(W, Sc, np.concatenate([S[lab["S"] != c], Z]), lam, gam)
        parts += np.where(np.isin(np.arange(len(X)), S), 0.0, hc_)
    out["additivity"] = {"max_abs_diff": float(np.abs(parts[F2] - hA[F2]).max())}
    # 6. exit decomposition (exact) and the escape bound for every unknown cluster, with seeds inside
    inv = np.zeros(len(X), bool)
    esc = []
    for j in range(lab["n_unk"]):
        Qall = idx["U"][j * lab["r"]:(j + 1) * lab["r"]]
        seeds = Qall[::5]
        Q = np.setdiff1d(Qall, seeds)
        h, _ = solve(W, S, seeds, lam, gam)
        inQ = np.zeros(len(X), bool)
        inQ[Q] = True
        nbr = np.unique(W[Q].indices)
        boundary = nbr[~inQ[nbr]]
        # H_Q(y, z): discounted probability that the walk leaves Q for the first time at z
        Hsum = np.zeros(len(Q))
        pi_exact = np.zeros(len(Q))
        for z in boundary:
            others = np.setdiff1d(np.flatnonzero(~inQ), [z])
            Hz, _ = solve(W, [z], others, lam, gam)
            Hsum += Hz[Q] * h[z]
            if z not in set(seeds.tolist()):
                pi_exact += Hz[Q]
        e = np.array([W[y][:, np.flatnonzero(~inQ & ~np.isin(np.arange(len(X)), seeds))].sum() / d[y] for y in Q])
        s_ = np.array([W[y][:, seeds].sum() / d[y] for y in Q])
        rho = lam * d[Q] / (d[Q] + gam)
        rb, eb, sm = rho.max(), e.max(), s_.min()
        phi = rb * eb / (1 - rb * (1 - eb - sm))
        # gamma = 0, no seeds: the closed form reduces to psi(e) = lam e / (1 - lam + lam e)
        esc.append({"cluster": j, "exit_decomp_max_err": float(np.abs(Hsum - h[Q]).max()), "max_h": float(h[Q].max()),
                    "max_pi": float(pi_exact.max()), "phi": float(phi), "h_le_pi": bool(np.all(h[Q] <= pi_exact + 1e-12)),
                    "pi_le_phi": bool(pi_exact.max() <= phi + 1e-12), "rho_max": float(rb), "e_max": float(eb), "s_min": float(sm)})
    out["escape"] = esc
    # 7. capture lower bound for the unlabelled ID images of each class (no seeds there)
    cap = []
    T_id = np.concatenate([idx["C"], idx["Tid"]])
    lab_id = np.concatenate([lab["cal"], lab["tid"]])
    for c in range(8):
        Rp = T_id[lab_id == c]
        inR = np.zeros(len(X), bool)
        inR[Rp] = True
        notS = ~np.isin(np.arange(len(X)), S)
        zero = np.flatnonzero(~inR & notS)
        kap, _ = solve(W, S, zero, lam, gam)
        om = np.array([W[y][:, S].sum() / d[y] for y in Rp])
        ep = np.array([W[y][:, np.flatnonzero(~inR & notS)].sum() / d[y] for y in Rp])
        rl = (lam * d[Rp] / (d[Rp] + gam)).min()
        cf = rl * om.min() / (1 - rl * (1 - om.min() - ep.max()))
        cap.append({"class": c, "min_kappa": float(kap[Rp].min()), "closed_form": float(cf), "kappa_ge_cf": bool(kap[Rp].min() >= cf - 1e-12),
                    "h_ge_kappa": bool(np.all(hA[Rp] >= kap[Rp] - 1e-12)), "one_step_ge": bool(np.all(kap[Rp] >= (lam * d[Rp] / (d[Rp] + gam)) * om - 1e-12))})
    out["capture"] = cap
    # 8. separation corollary on the full read-out: p(y) <= (1 + #{c : rho_c omega_c <= pi_Q(y)}) / (|C| + 1)
    W_, d_, gam_, h_, s_all, p_all, seeds_all = method(X, idx, lam, 0.0625)
    Tn = idx["T"]
    pos = {int(t): i for i, t in enumerate(Tn)}
    omega = np.asarray(W_[:, S].sum(1)).ravel() / d_
    rho_all = lam * d_ / (d_ + gam_)
    low_c = np.sort(rho_all[idx["C"]] * omega[idx["C"]])
    sep = {"n": 0, "violations": 0, "frac_bound_floor": 0.0, "frac_actual_floor": 0.0, "frac_bound_le005": 0.0, "frac_actual_le005": 0.0}
    rows = []
    sset = set(seeds_all.tolist())
    # Q = connected components of the unknown images that are not seeds (analysis only: uses the unknown labels)
    from scipy.sparse.csgraph import connected_components
    ufree = np.array([y for y in np.concatenate([idx["U"], idx["Far"]]) if int(y) not in sset])
    ncomp, comp = connected_components(W_[ufree][:, ufree], directed=False)
    sizes = []
    for j in range(ncomp):
        Q = ufree[comp == j]
        sizes.append(len(Q))
        inQ = np.zeros(len(X), bool)
        inQ[Q] = True
        one = np.flatnonzero(~inQ & ~np.isin(np.arange(len(X)), seeds_all))
        piQ, _ = solve(W_, one, seeds_all, lam, gam_)
        for y in Q:
            pb = (1.0 + np.searchsorted(low_c, piQ[y], side="right")) / (len(idx["C"]) + 1.0)
            pa = p_all[pos[int(y)]]
            rows.append((pa, pb))
    rows = np.array(rows)
    floor = 1.0 / (len(idx["C"]) + 1.0)
    out["separation"] = {"n_unknown_nonseed": int(len(rows)), "violations": int(np.sum(rows[:, 0] > rows[:, 1] + 1e-12)),
                         "frac_actual_floor": float(np.mean(rows[:, 0] <= floor + 1e-12)), "frac_bound_floor": float(np.mean(rows[:, 1] <= floor + 1e-12)),
                         "frac_actual_le005": float(np.mean(rows[:, 0] <= 0.05)), "frac_bound_le005": float(np.mean(rows[:, 1] <= 0.05)),
                         "n_seeds": int(len(seeds_all)), "n_components": int(ncomp), "max_component": int(max(sizes))}
    # 9. first-order limit: lam = 1 - a eps, gam = b eps  =>  (1 - h) / eps -> a E[N_A] + b E[tau_A]
    Fn = np.setdiff1d(np.arange(len(X)), S)
    EN = np.zeros(len(X))
    EN[Fn] = spl.spsolve((sp.identity(len(Fn)) - P[Fn][:, Fn]).tocsc(), np.ones(len(Fn)))
    ET = np.zeros(len(X))
    ET[Fn] = spl.spsolve((sp.diags(d[Fn]) - W[Fn][:, Fn]).tocsc(), np.ones(len(Fn)))
    lim = []
    for a, b in ((1.0, 0.0), (0.0, 1.0), (1.0, 2.0)):
        for eps in (1e-2, 1e-3, 1e-4):
            hh, _ = solve(W, S, [], 1 - a * eps, b * eps)
            tgt = a * EN + b * ET
            lim.append({"a": a, "b": b, "eps": eps, "max_rel_err": float(np.max(np.abs((1 - hh[Fn]) / eps - tgt[Fn]) / tgt[Fn]))})
    out["limit"] = lim
    # 10. the LGC source mass tends to sqrt(delta) (degree only) as lam -> 1
    Sm = (sp.diags(d ** -0.5) @ W @ sp.diags(d ** -0.5)).tocsc()
    y = np.zeros(len(X))
    y[S] = 1.0
    lgc = []
    for lm in (0.9, 0.99, 0.999, 0.9999):
        u = (1 - lm) * spl.spsolve((sp.identity(len(X)) - lm * Sm).tocsc(), y)
        ratio = u[Fn] / np.sqrt(d[Fn])
        lgc.append({"lam": lm, "spearman_u_sqrtdeg": float(spearmanr(u[Fn], np.sqrt(d[Fn]))[0]),
                    "rel_spread_u_over_sqrtdeg": float((ratio.max() - ratio.min()) / ratio.mean())})
    out["lgc_limit"] = lgc
    # 11. finite validity: pool design (invariant) vs seeds from the test images against C (not invariant)
    val = {}
    for design in ("pool", "test_only"):
        rng2 = np.random.default_rng(161)
        rec = []
        for _ in range(reps):
            Xr, ir, fr, _ = instance(rng2)
            *_, p, seeds = method(Xr, ir, 0.95, 0.0625, design=design)
            pid = p[~fr]
            sel = bh(p, 0.2)
            rec.append([np.mean(pid <= 0.05), np.mean(pid <= 0.1), np.mean(pid <= 0.2), (sel & ~fr).sum() / max(sel.sum(), 1), np.mean(p[fr] <= 0.05)])
        rec = np.array(rec)
        val[design] = {"reps": reps, **{k: [float(rec[:, i].mean()), float(rec[:, i].std(ddof=1) / np.sqrt(reps))]
                                       for i, k in enumerate(["fpr005", "fpr010", "fpr020", "fdp_bh020", "tpr005"])}}
    out["validity"] = val
    OUT.write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 400)
