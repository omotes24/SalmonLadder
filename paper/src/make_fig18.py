"""Figure 2 of the public-benchmark paper: AUROC vs FPR95 of every method of Table 1 on OpenOOD v1.5 ImageNet-1K
(near-OOD and far-OOD), from data/published.json and data/p11_summary.json (same rows as make_p18.py).
Labels are placed by hand (data coordinates) with thin leader lines; only the main (Ours) row is labelled, the two
smaller stars are the previous configuration and the standalone one (named in the caption)."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
S = json.load(open(HERE / "data" / "p11_summary.json"))
S14 = json.load(open(HERE / "data" / "p14_summary.json"))
CFG = S14["selection"]["chosen"].rsplit("|", 1)[0]
PUB = json.load(open(HERE / "data" / "published.json"))
OURS = {"rows": [{"key": CFG + "|none", "fig": "Ours, standalone"}, {"key": CFG + "|TINS", "fig": "Salmon Ladder (Ours)", "main": True}]}
OO_NEAR, OO_FAR = ["ssb_hard", "ninco"], ["inaturalist", "textures", "openimageo"]
BLUE, LIGHT = "#1f5fbf", "#8fb3e6"
plt.rcParams.update({"font.size": 7, "axes.linewidth": 0.6, "xtick.major.width": 0.5, "ytick.major.width": 0.5,
                     "pdf.fonttype": 42, "font.family": "DejaVu Sans"})
STYLE = {"sup": dict(marker="s", color="#777777", label="trained on ImageNet-1K labels (reported)"),
         "unsup": dict(marker="D", color="#b07020", label="unsupervised, DisCoPatch (reported)"),
         "vlm": dict(marker="o", color="#3a8f5a", label="VLM, CLIP ViT-B/16 (reported; TANL measured)"),
         "ours2": dict(marker="*", color=LIGHT, label="Salmon Ladder, standalone"),
         "ours": dict(marker="*", color=BLUE, label="Salmon Ladder (Ours)")}
# label positions (data coordinates) per panel
POS_NEAR = {"MCM": (61.0, 76.0), "NegLabel": (66.0, 67.0), "AdaNeg": (69.5, 63.5), "ASH": (79.5, 73.5), "InterNeg": (83.2, 66.3),
            "SCALE": (72.5, 60.0), "LAPT": (72.5, 56.5), "TINS": (84.5, 59.3), "TANL (measured)": (84.5, 55.0),
            "MOODv2": (89.8, 49.5), "NNGuide": (83.0, 40.0), "RankFeat": (90.5, 44.5), "DisCoPatch": (85.0, 29.4),
            "Salmon Ladder (Ours)": (82.0, 12.0)}
POS_FAR = {"MCM": (85.6, 66.0), "NNGuide": (90.2, 7.0), "TINS": (90.2, 10.8), "MOODv2": (90.2, 14.4), "SCALE": (90.2, 18.0),
           "InterNeg": (90.2, 21.6), "AdaNeg": (90.2, 25.2), "DisCoPatch": (90.2, 28.8), "ASH": (90.2, 32.4),
           "TANL (measured)": (90.2, 36.0), "RankFeat": (85.6, 42.0), "LAPT": (85.6, 46.0), "NegLabel": (85.6, 50.0),
           "Salmon Ladder (Ours)": (91.5, 2.8)}


def mean(v):
    return sum(v) / len(v)


def points():
    pts = []   # (name, group, near(a,f), far(a,f))
    for m, g in [("MOODv2", "sup"), ("NNGuide", "sup"), ("RankFeat", "sup"), ("SCALE", "sup"), ("ASH", "sup"), ("DisCoPatch", "unsup")]:
        p = PUB["openood"][m]["per"]
        pts.append((m, g, (mean([p[d][0] for d in OO_NEAR]), mean([p[d][1] for d in OO_NEAR])),
                    (mean([p[d][0] for d in OO_FAR]), mean([p[d][1] for d in OO_FAR]))))
    for m in ["MCM", "NegLabel", "LAPT", "AdaNeg", "InterNeg", "TINS"]:
        e = PUB["openood"][m]
        pts.append((m, "vlm", tuple(e["near"]), tuple(e["far"])))
    k = "-|-|TANL"
    pts.append(("TANL (measured)", "vlm", (S["openood"]["near"][k]["AUROC"], S["openood"]["near"][k]["FPR95"]),
                (S["openood"]["far"][k]["AUROC"], S["openood"]["far"][k]["FPR95"])))
    for r in OURS["rows"]:
        k = r["key"]
        D = S14["test_train"]["openood"]
        pts.append((r.get("fig", "Salmon Ladder (Ours)"), "ours" if r.get("main") else "ours2",
                    (D["near"][k]["AUROC"], D["near"][k]["FPR95"]), (D["far"][k]["AUROC"], D["far"][k]["FPR95"])))
    return pts


def panel(ax, pts, which, title, pos):
    for name, g, near, far in pts:
        a, f = near if which == "near" else far
        st = STYLE[g]
        ax.scatter(a, f, marker=st["marker"], s=(150 if g == "ours" else 60 if g == "ours2" else 22), color=st["color"],
                   edgecolor="white" if g.startswith("ours") else "none", linewidth=0.5, zorder=5 if g.startswith("ours") else 3)
        if name in pos:
            tx, ty = pos[name]
            ax.annotate(name, (a, f), xytext=(tx, ty), fontsize=5.6, color=(BLUE if g == "ours" else "black"),
                        fontweight=("bold" if g == "ours" else "normal"), va="center", ha="left", zorder=6,
                        arrowprops=dict(arrowstyle="-", lw=0.4, color="#9a9a9a", shrinkA=0, shrinkB=2.5))
    ax.set_xlabel("AUROC (%)  →  better")
    ax.set_ylabel("FPR95 (%)  ←  better")
    ax.invert_yaxis()
    ax.set_title(title, fontsize=7.5)
    ax.grid(True, linewidth=0.3, color="#dddddd")
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def main():
    pts = points()
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.05))
    panel(axes[0], pts, "near", "(a) near-OOD: SSB-hard, NINCO", POS_NEAR)
    panel(axes[1], pts, "far", "(b) far-OOD: iNaturalist, Textures, OpenImage-O", POS_FAR)
    axes[0].set_xlim(58, 100)
    axes[0].set_ylim(84, 8)
    axes[1].set_xlim(84, 100.4)
    axes[1].set_ylim(72, 0)
    handles = [plt.Line2D([], [], marker=STYLE[g]["marker"], color=STYLE[g]["color"], linestyle="", markersize=(9 if g == "ours" else 6 if g == "ours2" else 5),
                          label=STYLE[g]["label"]) for g in ("sup", "unsup", "vlm", "ours", "ours2")]
    fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=6, frameon=False, bbox_to_anchor=(0.5, -0.02), handletextpad=0.3, columnspacing=1.2)
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    (HERE / "figs").mkdir(exist_ok=True)
    fig.savefig(HERE / "figs" / "fig18_scatter.pdf")
    print("wrote figs/fig18_scatter.pdf")


if __name__ == "__main__":
    main()
