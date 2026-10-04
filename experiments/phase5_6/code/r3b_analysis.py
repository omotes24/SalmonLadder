"""Round-3b (dev1): truncation of the seed-mass factor (penalise only), weights, and x TINS for the leading candidates."""
import itertools
import numpy as np
from analyze5 import FROZEN, REFT, diff, draw_means, load, summarize, tci
from metrics_p4 import metrics
import diag5

lp, lp0 = f"lp|{REFT}", f"lp0|{REFT}"


def score(z, spec, views=("B14", "L14")):
    """spec: list of (key, weight, tau); tau < 1 truncates the factor from above (min(log p - log tau, 0))."""
    x = 0.0
    for v in views:
        for key, w, tau in spec:
            a = z[f"s::{v}::{key}"]
            if tau < 1.0:
                a = np.minimum(a - np.log(tau), 0.0)
            x = x + w * a
    return x


def ev(cfg, spec, base="none"):
    out = {}
    for s in ("near", "far"):
        met = {}
        for t, z in load(cfg, "dev1", s).items():
            x = score(z, spec)
            if base == "TINS":
                x = x + z["logS"]
            met[t] = metrics(x, z["is_ood"])
        out[s] = met
    return out


def line(name, cfg, spec, R, base="none"):
    E = ev(cfg, spec, base)
    n, f = summarize(E["near"]), summarize(E["far"])
    d, df = diff(E["near"], R["near"]), diff(E["far"], R["far"])
    print(f"{base:5s} {name:58s} | {n['AUROC']:6.2f} {n['FPR95']:6.2f} {d['mean']:+6.2f} [{d['lo']:+6.2f},{d['hi']:+6.2f}] |"
          f" {f['AUROC']:6.2f} {f['FPR95']:6.2f} {df['mean']:+6.2f}", flush=True)
    return n["FPR95"], df["mean"]


cfg = "r3"
FR = [("M0", 1, 1), (lp0, 1, 1)]
for base in ("none", "TINS"):
    R = ev(cfg, FR, base)
    print(f"{'base':5s} {'combination':58s} | near AUROC  FPR95   dFPR [95% CI]         | far AUROC  FPR95   dFPR")
    line("frozen", cfg, FR, R, base)
    for D in ("Dlp0.05m0", "Dlp0.1m0", "Dlp0.2m0", "Dlp0.1m3", "Dlp0.2m3"):
        line(f"M0 x lp x neg:{D}", cfg, [("M0", 1, 1), (lp, 1, 1), (f"neg:{D}", 1, 1)], R, base)
        line(f"M0 x {D} x lp x neg", cfg, [("M0", 1, 1), (D, 1, 1), (lp, 1, 1), (f"neg:{D}", 1, 1)], R, base)
        if base == "none":
            for tau in (0.5, 0.3, 0.2, 0.1):
                line(f"M0 x lp x neg:{D} [neg<= {tau}]", cfg, [("M0", 1, 1), (lp, 1, 1), (f"neg:{D}", 1, tau)], R, base)
            for tau in (0.5, 0.2):
                line(f"M0 x {D}[<= {tau}] x lp x neg[<= {tau}]", cfg, [("M0", 1, 1), (D, 1, tau), (lp, 1, 1), (f"neg:{D}", 1, tau)], R, base)
            for w in (0.5, 1.5, 2.0):
                line(f"M0 x lp x neg:{D}^{w}", cfg, [("M0", 1, 1), (lp, 1, 1), (f"neg:{D}", w, 1)], R, base)
            line(f"M0^0.5 x {D}^0.5 x lp x neg", cfg, [("M0", 0.5, 1), (D, 0.5, 1), (lp, 1, 1), (f"neg:{D}", 1, 1)], R, base)
            line(f"M0 x lp^0.5 x lp0^0.5 x neg:{D}", cfg, [("M0", 1, 1), (lp, 0.5, 1), (lp0, 0.5, 1), (f"neg:{D}", 1, 1)], R, base)
            line(f"M0 x lp x neg:{D} x static^0.25", cfg, [("M0", 1, 1), (lp, 1, 1), (f"neg:{D}", 1, 1), ("static", 0.25, 1)], R, base)
