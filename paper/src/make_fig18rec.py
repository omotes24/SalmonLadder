"""Figure: recurrence x mixing rate (Phase 14 grid) -> figs/fig18_rec.pdf.
(a),(b): FPR95 (standalone, DINOv2 L/14 + DINOv3 L/16) of p_fin (solid) and p+ only (dashed) against the images per unknown class r,
one line per number of unknown classes c; for the largest c also the static rank (dotted), p+ p- without cross-fitting (thin grey)
and p_fin with the static rank in the statistic (dash-dot). (c),(d): the share of the unknown images selected as seeds (DINOv3 L/16,
Storey q = 0.2, both folds) against r."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
S14 = json.load(open(HERE / "data" / "p14_summary.json"))
REC = S14["rec"]
S15 = json.load(open(HERE / "data" / "p15_summary.json")) if (HERE / "data" / "p15_summary.json").exists() else {}
CAPREC = S15.get("cap_rec") or {}
CAP = "xfit|storeyc|q0.2|pp|pooled|none"
SEL = S14["selection"]["chosen"]                       # xfit|storey|q0.2|pp|none
CFG = SEL.rsplit("|", 1)[0]
RULE, QK = CFG.split("|")[1:3]
MAIN, LP, STATIC, V1, PPS = CFG + "|pooled|none", "xfit|lp|pooled|none", "xfit|static|pooled|none", "v1|storey|q0.2|pp|none", f"xfit|{RULE}|{QK}|pps|pooled|none"
plt.rcParams.update({"font.size": 7, "axes.linewidth": 0.6, "xtick.major.width": 0.5, "ytick.major.width": 0.5, "pdf.fonttype": 42, "font.family": "DejaVu Sans"})
COLORS = ["#1f5fbf", "#3a8f5a", "#b07020", "#8a3fa0", "#c0392b", "#555555"]


def grid(ds):
    out = {}
    for k, e in REC.items():
        d, c, r = k.split("|")
        if d == ds:
            out[(int(c[1:]), int(r[1:]))] = e
    return out


def main():
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.0))
    axes = axes.reshape(1, 2)
    for j, (ds, title) in enumerate((("ninco", "NINCO (64 classes, 45,000 ID images)"), ("ssb_hard", "SSB-hard (980 classes, 45,000 ID images)"))):
        g = grid(ds)
        cs = sorted({c for c, _ in g})
        rs = sorted({r for _, r in g})
        ax = axes[0, j]
        for i, c in enumerate(cs):
            xs = [r for r in rs if (c, r) in g]
            ax.plot(xs, [g[(c, r)]["cfg"][MAIN]["FPR95"] for r in xs], "-o", color=COLORS[i % len(COLORS)], ms=3, lw=1.2, label=f"c={c}")
            ax.plot(xs, [g[(c, r)]["cfg"][LP]["FPR95"] for r in xs], "--", color=COLORS[i % len(COLORS)], lw=0.9)
        cmax = max(cs)
        xs = [r for r in rs if (cmax, r) in g]
        ax.plot(xs, [g[(cmax, r)]["cfg"][STATIC]["FPR95"] for r in xs], ":", color="#777777", lw=1.0, marker="x", ms=3, label=f"static p (c={cmax})")
        ax.plot(xs, [g[(cmax, r)]["cfg"][V1]["FPR95"] for r in xs], "-", color="#aaaaaa", lw=0.8, label=f"no cross-fit (c={cmax})")
        ax.plot(xs, [g[(cmax, r)]["cfg"][PPS]["FPR95"] for r in xs], "-.", color="#333333", lw=1.0, marker="s", ms=2.5, label=f"sigma+static (c={cmax})")
        capc = {int(k.split("|")[2][1:]): e for k, e in CAPREC.items() if k.startswith(f"{ds}|c{cmax}|")}
        if capc:
            xs2 = [r for r in xs if r in capc]
            ax.plot(xs2, [capc[r]["cfg"][CAP]["FPR95"] for r in xs2], "-", color="#e377c2", lw=1.0, marker="^", ms=2.5, label=f"capped seeds (c={cmax})")
        ax.set_xscale("log")
        ax.set_xticks(rs)
        ax.set_xticklabels([str(r) for r in rs])
        ax.set_xlabel("images per unknown class r")
        ax.set_ylabel("FPR95 (%)  ←  better")
        ax.set_ylim(0, 102)
        ax.set_yticks([0, 20, 40, 60, 80, 100])
        ax.set_title(("(a) " if j == 0 else "(b) ") + title, fontsize=7.5)
        ax.grid(True, linewidth=0.3, color="#dddddd")
        ax.set_axisbelow(True)
        ax.plot([], [], "-", color="black", lw=1.2, label="sigma")
        ax.plot([], [], "--", color="black", lw=0.9, label="p+ only")
        ax.legend(fontsize=4.6, frameon=False, ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.40), handlelength=1.3, columnspacing=0.5, handletextpad=0.25, borderaxespad=0.0)
        for s in ("top", "right"):
            axes[0, j].spines[s].set_visible(False)
    fig.subplots_adjust(left=0.065, right=0.995, top=0.88, bottom=0.46, wspace=0.2)
    fig.savefig(HERE / "figs" / "fig18_rec.pdf", bbox_inches="tight", pad_inches=0.02)
    print("wrote figs/fig18_rec.pdf")


if __name__ == "__main__":
    main()
