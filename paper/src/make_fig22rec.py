"""Figure: recurrence x mixing rate (Phase 14/17 grid) -> figs/fig22_rec.pdf.
FPR95 (standalone, DINOv2 L/14 + DINOv3 L/16) of sigma_ref (solid) and p+ only read with the reference images (dashed) against the
images per unknown class r, one line per number of unknown classes c; for the largest c also the static rank (dotted, Phase 14 on the
same cells) and the cross-fitted sigma (thin grey). Data: data/p17_summary.json (rec), data/p14_summary.json (static p)."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
S17 = json.load(open(HERE / "data" / "p17_summary.json"))["rec"]
REC14 = json.load(open(HERE / "data" / "p14_summary.json"))["rec"]
REF, RLP, SIG, STATIC = "ref|storey|q0.2|pp|stat|none", "ref|lp|stat|none", "xfit|storey|q0.2|pp|pooled|none", "xfit|static|pooled|none"
plt.rcParams.update({"font.size": 7, "axes.linewidth": 0.6, "xtick.major.width": 0.5, "ytick.major.width": 0.5, "pdf.fonttype": 42, "font.family": "DejaVu Sans"})
COLORS = ["#1f5fbf", "#3a8f5a", "#b07020", "#8a3fa0", "#c0392b", "#555555"]


def v17(ds, c, r, key):
    return S17[f"{ds}|c{c}|r{r}|{key}"]["FPR95"]


def main():
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.0))
    for j, (ds, title) in enumerate((("ninco", "NINCO (64 classes, 45,000 ID images)"), ("ssb_hard", "SSB-hard (980 classes, 45,000 ID images)"))):
        cells = sorted({(int(k.split("|")[1][1:]), int(k.split("|")[2][1:])) for k in S17 if k.startswith(ds + "|")})
        cs = sorted({c for c, _ in cells})
        rs = sorted({r for _, r in cells})
        ax = axes[j]
        for i, c in enumerate(cs):
            xs = [r for r in rs if (c, r) in cells]
            ax.plot(xs, [v17(ds, c, r, REF) for r in xs], "-o", color=COLORS[i % len(COLORS)], ms=3, lw=1.2, label=f"c={c}")
            ax.plot(xs, [v17(ds, c, r, RLP) for r in xs], "--", color=COLORS[i % len(COLORS)], lw=0.9)
        cmax = max(cs)
        xs = [r for r in rs if (cmax, r) in cells]
        ax.plot(xs, [REC14[f"{ds}|c{cmax}|r{r}"]["cfg"][STATIC]["FPR95"] for r in xs], ":", color="#777777", lw=1.0, marker="x", ms=3, label=f"static p (c={cmax})")
        ax.plot(xs, [v17(ds, cmax, r, SIG) for r in xs], "-", color="#aaaaaa", lw=0.8, label=f"sigma, cross-fitted (c={cmax})")
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
        ax.plot([], [], "-", color="black", lw=1.2, label="sigma_ref")
        ax.plot([], [], "--", color="black", lw=0.9, label="p+ only")
        ax.legend(fontsize=4.8, frameon=False, ncol=5, loc="upper center", bbox_to_anchor=(0.5, -0.40), handlelength=1.3, columnspacing=0.6, handletextpad=0.25, borderaxespad=0.0)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    fig.subplots_adjust(left=0.065, right=0.995, top=0.88, bottom=0.42, wspace=0.2)
    fig.savefig(HERE / "figs" / "fig22_rec.pdf", bbox_inches="tight", pad_inches=0.02)
    print("wrote figs/fig22_rec.pdf")


if __name__ == "__main__":
    main()
