"""Figure of the theory note: one toy 2-D point cloud (Gaussian-weighted symmetric k-NN graph), the same quantity h under the two
prices. Two ID classes with support images, a tight unknown cluster between them (near-like) and sparse scattered unknowns
(far-like). (a) price per jump only, (b) price per time only, (c) both. Writes gen/fig_toy.pdf and gen/fig_toy.json."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import scipy.sparse as sp  # noqa: E402
import scipy.sparse.linalg as spl  # noqa: E402

HERE = Path(__file__).resolve().parent
GEN = HERE / "gen"
GEN.mkdir(exist_ok=True)
plt.rcParams["font.family"] = ["Noto Sans CJK JP", "DejaVu Sans"]
plt.rcParams["pdf.fonttype"] = 42

rng = np.random.default_rng(5)
A = rng.normal([0.0, 0.0], 0.6, size=(160, 2))
B = rng.normal([3.4, 0.0], 0.6, size=(160, 2))
c_near = 1.5 * np.array([np.cos(0.3 * np.pi), np.sin(0.3 * np.pi)])
near = rng.normal(c_near, 0.1, size=(40, 2))                      # a tight unknown cluster next to class A (dense)
th, rad = rng.uniform(0, 2 * np.pi, 30), rng.uniform(1.5, 2.2, 30)
cen = np.where(rng.random(30) < 0.5, 0.0, 3.4)
far = np.c_[cen + rad * np.cos(th), rad * np.sin(th)]               # sparse unknowns scattered around the ID classes
X = np.concatenate([A, B, near, far])
n = len(X)
grp = np.r_[np.zeros(320), 2 * np.ones(40), 3 * np.ones(30)].astype(int)
sup = np.r_[rng.choice(160, 30, replace=False), 160 + rng.choice(160, 30, replace=False)]
k = 8
D2 = ((X[:, None, :] - X[None, :, :]) ** 2).sum(-1)
np.fill_diagonal(D2, np.inf)
I = np.argsort(D2, axis=1)[:, :k]
dist = np.sqrt(np.take_along_axis(D2, I, 1))
bw = 0.5 * np.median(dist)
V = np.exp(-dist ** 2 / (2 * bw ** 2))
Am = sp.csr_matrix((V.ravel(), (np.repeat(np.arange(n), k), I.ravel())), shape=(n, n))
W = Am.maximum(Am.T).tocsr()
d = np.asarray(W.sum(1)).ravel()


def solve(one, zero, lam, gam):
    bnd = np.zeros(n, bool)
    bnd[one] = True
    bnd[zero] = True
    F = np.flatnonzero(~bnd)
    M = (sp.diags(d[F] + gam) - lam * W[F][:, F]).tocsc()
    h = np.zeros(n)
    h[one] = 1.0
    h[F] = spl.spsolve(M, lam * np.asarray(W[F][:, one].sum(1)).ravel())
    return h


med = float(np.median(d))
cfgs = [("(a) 歩数の代価だけ", 0.8, 0.0), ("(b) 時間の代価だけ", 1.0, 0.3 * med), ("(c) 二つの代価", 0.8, 0.03 * med)]
hs = [solve(sup, [], lam, g) for _, lam, g in cfgs]
seeds = np.array([], int)

fig, axes = plt.subplots(1, 3, figsize=(9.0, 2.9), constrained_layout=True)
coo = sp.triu(W, 1).tocoo()
cmap = plt.get_cmap("Blues")
summ = []
for ax, (title, lam, g), h in zip(axes, cfgs, hs):
    for i, j in zip(coo.row, coo.col):
        ax.plot(X[[i, j], 0], X[[i, j], 1], color="#d9d9d9", lw=0.4, zorder=1)
    unl = np.setdiff1d(np.arange(n), sup)
    hv = h.copy()
    # rank-normalise the unlabelled values per panel (the read-out is a rank), support drawn separately
    r = np.empty(len(unl))
    r[np.argsort(hv[unl])] = np.linspace(0, 1, len(unl))
    rk = np.full(n, np.nan)
    rk[unl] = r
    idu = np.flatnonzero(grp <= 1)
    idu = idu[~np.isin(idu, sup)]
    sc = ax.scatter(X[idu, 0], X[idu, 1], c=rk[idu], cmap=cmap, vmin=-0.15, vmax=1.0, s=9, edgecolors="#969696", linewidths=0.25, zorder=3)
    ax.scatter(X[sup, 0], X[sup, 1], marker="s", s=10, c="#08306b", edgecolors="white", linewidths=0.3, zorder=4)
    g2, g3 = np.flatnonzero(grp == 2), np.flatnonzero(grp == 3)
    ax.scatter(X[g3, 0], X[g3, 1], marker="D", c=rk[g3], cmap=cmap, vmin=-0.15, vmax=1.0, s=26, edgecolors="#252525", linewidths=0.7, zorder=5)
    ax.scatter(X[g2, 0], X[g2, 1], marker="^", c=rk[g2], cmap=cmap, vmin=-0.15, vmax=1.0, s=22, edgecolors="#252525", linewidths=0.6, zorder=6)
    rr = dict(zip(unl, r))
    m_near = float(np.mean([rr[i] for i in np.flatnonzero(grp == 2) if i in rr]))
    m_far = float(np.mean([rr[i] for i in np.flatnonzero(grp == 3)]))
    m_id = float(np.mean([rr[i] for i in np.flatnonzero(grp <= 1) if i in rr]))
    summ.append({"panel": title, "lambda": lam, "gamma": g, "mean_rank_id": m_id, "mean_rank_near": m_near, "mean_rank_far": m_far})
    ax.set_title(title, fontsize=9)
    ax.text(0.02, 0.02, f"順位の平均：ID {m_id:.2f}，塊▲ {m_near:.2f}，散在◆ {m_far:.2f}", transform=ax.transAxes, fontsize=7.5, color="#252525")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlim(-2.6, 6.0)
    ax.set_ylim(-2.6, 2.7)
    for sp_ in ax.spines.values():
        sp_.set_color("#bdbdbd")
cb = fig.colorbar(sc, ax=axes, shrink=0.8, pad=0.01)
cb.set_label("ラベルなし画像の中での$h$の順位", fontsize=7)
cb.ax.tick_params(labelsize=6)
fig.savefig(GEN / "fig_toy.pdf")
fig.savefig(GEN / "fig_toy.png", dpi=170)
(GEN / "fig_toy.json").write_text(json.dumps({"median_degree": {"id": float(np.median(d[grp <= 1])), "cluster": float(np.median(d[grp == 2])), "scattered": float(np.median(d[grp == 3]))}, "panels": summ}, indent=1) + "\n")
print(json.dumps(summ, indent=1))
