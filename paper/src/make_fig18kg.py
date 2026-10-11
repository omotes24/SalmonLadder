"""Figure: the recurrence grid with the neighbour count k_g varied (Phase 15) -> figs/fig18_kg.pdf.
(a) NINCO c = 64, (b) SSB-hard c = 980: FPR95 (standalone) of sigma (solid) and p+ only (dashed) against r, one colour per k_g;
the static rank (dotted, independent of k_g)."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
S15 = json.load(open(HERE / "data" / "p15_summary.json"))
KG = S15["kg"]
S14 = json.load(open(HERE / "data" / "p14_summary.json"))
SEL = S14["selection"]["chosen"]
CFG = SEL.rsplit("|", 1)[0]
MAIN, LP, STATIC = CFG + "|pooled|none", "xfit|lp|pooled|none", "xfit|static|pooled|none"
plt.rcParams.update({"font.size": 7, "axes.linewidth": 0.6, "xtick.major.width": 0.5, "ytick.major.width": 0.5, "pdf.fonttype": 42, "font.family": "DejaVu Sans"})
COLORS = {3: "#8a3fa0", 5: "#1f5fbf", 10: "#c0392b", 20: "#3a8f5a", 40: "#b07020"}


def grid(ds):
    out = {}
    for k, e in KG.items():
        d, c, r, kg = k.split("|")
        if d == ds:
            out[(int(r[1:]), int(kg[2:]))] = e
    return out


def main():
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 1.85))
    for j, (ds, title) in enumerate((("ninco", "NINCO (c = 64, 45,000 ID images)"), ("ssb_hard", "SSB-hard (c = 980, 45,000 ID images)"))):
        g = grid(ds)
        rs = sorted({r for r, _ in g})
        kgs = sorted({k for _, k in g})
        ax = axes[j]
        for kg in kgs:
            xs = [r for r in rs if (r, kg) in g]
            ax.plot(xs, [g[(r, kg)]["cfg"][MAIN]["FPR95"] for r in xs], "-o", color=COLORS.get(kg, "#555"), ms=3, lw=1.2 if kg == 10 else 0.9, label=f"k_g = {kg}")
            ax.plot(xs, [g[(r, kg)]["cfg"][LP]["FPR95"] for r in xs], "--", color=COLORS.get(kg, "#555"), lw=0.8)
        xs = [r for r in rs if (r, kgs[0]) in g]
        ax.plot(xs, [g[(r, kgs[0])]["cfg"][STATIC]["FPR95"] for r in xs], ":", color="#777777", lw=1.0, marker="x", ms=3, label="static p")
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
        ax.plot([], [], "--", color="black", lw=0.8, label="p+ only")
        ax.legend(fontsize=5.0, frameon=False, ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.40), handlelength=1.5, columnspacing=0.7, handletextpad=0.3, borderaxespad=0.0)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    fig.subplots_adjust(left=0.065, right=0.995, top=0.88, bottom=0.40, wspace=0.2)
    fig.savefig(HERE / "figs" / "fig18_kg.pdf", bbox_inches="tight", pad_inches=0.02)
    print("wrote figs/fig18_kg.pdf")


if __name__ == "__main__":
    main()
