"""Salmon Ladder (the paper of October 2026): the post-stream two-sided read-out on a synthetic two-view stream (CPU, seconds).

The script calls the archived implementation directly:
  experiments/phase4/code/engine.py                    PrefixTopK (exact kNN lists), matrix (W = max(A, A^T), D^-1/2 W D^-1/2),
                                                       propagate_converged, p_low / p_high (calibrated ranks)
  experiments/phase10_11/phase11/code/sweep11.py       storey_bh (Storey-BH selection of the seeds)
  experiments/phase10_11/phase11/code/final11.py       the same steps as run on the benchmark streams (view_terms)

Per view, with the support, calibration and stream images as the nodes of one k_g = 10 graph:
  u+   mass propagated from the support images (lambda = 0.9, to convergence)   -> p+ = rank among the calibration images (low = unknown)
  Q    seeds: Storey-BH at q = 0.1 on p+ of the stream images
  u-   mass propagated from the seeds on the same graph                         -> p- = rank among the calibration images (high mass = unknown)
The detector is the sum over the views of log p+ + log p-; with a base detector s0, log s0 is added with weight 1.

The stream is synthetic (groups of sibling classes with one unknown class per group). It demonstrates the interface and the
behaviour under recurrence; it is not a benchmark and no number printed here appears in the paper.
"""
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
for rel in ("experiments/lp_audit/vendor", "experiments/phase4/code"):
    sys.path.insert(0, str(ROOT / rel))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import engine as E  # noqa: E402
import synth5  # noqa: E402
from metrics_p4 import metrics  # noqa: E402

K_G, LAM, Q = 10, 0.9, 0.1


def storey_bh(p, q):
    """experiments/phase10_11/phase11/code/sweep11.py: storey_bh (copied so that the example imports no server-path module)."""
    p = np.asarray(p, np.float64)
    n = len(p)
    pi0 = min(1.0, (np.sum(p > 0.5) + 1) / (n * 0.5))
    order = np.argsort(p)
    ps = p[order]
    thr = q * np.arange(1, n + 1) / (n * pi0)
    ok = np.flatnonzero(ps <= thr)
    mask = np.zeros(n, bool)
    if len(ok):
        mask[order[: ok[-1] + 1]] = True
    return mask


def rereading(sup, cal, sf, device="cpu"):
    """One view: log p+ and log p- of the stream images (experiments/phase10_11/phase11/code/final11.py: view_terms)."""
    ns, nc = len(sup), len(cal)
    nfix = ns + nc
    X = np.concatenate([sup, cal, sf]).astype(np.float32)
    graph = E.PrefixTopK(X, device=device, kmax=max(K_G, 20))
    M = graph.matrix(K_G, 1.0, "sym")

    def mass(idx):
        y = torch.zeros((len(X), 1), dtype=torch.float64, device=device)
        y[torch.as_tensor(idx, device=device), 0] = 1.0
        return E.propagate_converged(M, y, LAM, max_iter=3000)[0][:, 0].cpu().numpy()

    u_pos = mass(np.arange(ns))
    p_pos = E.p_low(u_pos[ns:nfix], u_pos[nfix:])
    seeds = np.flatnonzero(storey_bh(p_pos, Q))
    if len(seeds) == 0:
        return E._log(p_pos), np.zeros(len(sf)), seeds
    u_neg = mass(nfix + seeds)
    p_neg = E.p_high(u_neg[ns:nfix], u_neg[nfix:])
    return E._log(p_pos), E._log(p_neg), seeds


def main():
    torch.manual_seed(0)
    stream = synth5.make(seed=0, batch=64)            # 60 ID classes, 20 unknown classes of 50 images each
    is_ood = stream["is_ood"]
    total = {"p+": 0.0, "p+ x p-": 0.0}
    for name, view in stream["views"].items():
        sup = view["sup"].reshape(-1, view["sup"].shape[-1])          # (classes x 12, D) in class order
        lp, ln, seeds = rereading(sup, view["cal"], view["sf"])
        total["p+"] += lp
        total["p+ x p-"] += lp + ln
        print(f"  view {name}: {len(seeds)} seeds, of which {100 * is_ood[seeds].mean():.1f}% from unknown classes")
    print(f"synthetic stream: {len(is_ood)} images, {int(is_ood.sum())} from unknown classes; scores assigned after the whole stream")
    for label, score in total.items():
        m = metrics(score, is_ood)                    # the score is large for ID-looking images
        print(f"  {label:10s} AUROC {m['AUROC']:6.2f}   FPR95 {m['FPR95']:6.2f}")
    print("Synthetic feature example finished. No benchmark metric is reported.")


if __name__ == "__main__":
    main()
