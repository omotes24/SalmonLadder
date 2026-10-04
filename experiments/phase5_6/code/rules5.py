"""Phase 5: combination rules for per-view p-values (evaluation on saved development scores; labels used here only)."""
import numpy as np

from analyze5 import FROZEN, REFT, diff, draw_means, load, summarize, views_of
from metrics_p4 import metrics


def stack(z, keys, views=None):
    return np.stack([z[f"s::{v}::{k}"] for v in (views or views_of(z)) for k in keys])      # log p, (components, N)


def combine(L, rule, tau=1.0, K=None, w=None):
    """L: log p (components x N). Returns an ID-high score."""
    if w is not None:
        L = L * np.asarray(w, float)[:, None]
    if rule == "prod":
        return L.sum(0)
    if rule == "tpm":                                    # truncated product: only p <= tau count
        return np.minimum(L - np.log(tau), 0.0).sum(0)
    if rule == "min":
        return L.min(0)
    if rule == "rtp":                                    # product of the K smallest
        return np.sort(L, axis=0)[:K].sum(0)
    if rule == "cauchy":
        P = np.clip(np.exp(L), 1e-300, 1 - 1e-12)
        return -np.tan((0.5 - P) * np.pi).mean(0)
    if rule == "hmp":
        return -np.log(np.exp(-L).mean(0))
    raise ValueError(rule)


def evaluate(cfg, keys, rule, dev="dev1", views=None, base="none", **kw):
    out = {}
    for s in ("near", "far"):
        met = {}
        for t, z in load(cfg, dev, s).items():
            x = combine(stack(z, keys, views), rule, **kw)
            if base == "TINS":
                x = x + z["logS"]
            met[t] = metrics(x, z["is_ood"])
        out[s] = met
    return out


def report(cfg, rows, dev="dev1", views=None, bases=("none",), ref_keys=None):
    ref_keys = ref_keys or [k for k, _ in FROZEN]
    for base in bases:
        R = evaluate(cfg, ref_keys, "prod", dev, views, base)
        print(f"{'base':5s} {'components / rule':52s} | near AUROC  FPR95   dFPR [95% CI]         | far AUROC  FPR95   dFPR")
        for name, (keys, rule, kw) in rows.items():
            E = evaluate(cfg, keys, rule, dev, views, base, **kw)
            n, f = summarize(E["near"]), summarize(E["far"])
            d, df = diff(E["near"], R["near"]), diff(E["far"], R["far"])
            print(f"{base:5s} {name:52s} | {n['AUROC']:6.2f} {n['FPR95']:6.2f} {d['mean']:+6.2f} [{d['lo']:+6.2f},{d['hi']:+6.2f}] |"
                  f" {f['AUROC']:6.2f} {f['FPR95']:6.2f} {df['mean']:+6.2f}", flush=True)
