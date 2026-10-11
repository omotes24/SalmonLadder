"""Tables and number macros of the paper v18 from
  data/p14_summary.json   Phase 14 (the cross-fitted ladder: dev-split selection, benchmarks with 6 draws, exchangeable calibration,
                          grid, ImageNet-200, backbone families, CIFAR-100, arrival-time mode)
  data/p12_summary.json   Phase 12 (view selection on the dev splits; the test-set exploration history of the earlier phases)
  data/p13_summary.json   Phase 13 (arrival-time mode of the read-out without cross-fitting)
  data/p11_summary.json   measured base detectors on the same streams
  data/published.json     reported values of other methods
Writes sections/t18_*.tex and sections/numbers.tex (+ data/numbers.json). Conventions: \\sota = column best, \\cellcolor{upd}+bold =
beats every non-ours row, \\gap = margin to the best non-ours value after the main row."""
import json
import os
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
S14 = json.load(open(HERE / "data" / "p14_summary.json"))
S15 = json.load(open(HERE / "data" / "p15_summary.json")) if (HERE / "data" / "p15_summary.json").exists() else {}
BASE15 = {"knn_all_k50": "kNN，全学習画像（$k{=}50$）", "ncm_all": "クラス平均，全学習画像", "mcm_clip": "MCM~\\cite{ming2022mcm}，ゼロショット", "neglabel_clip": "NegLabel~\\cite{jiang2024neglabel}，ゼロショット"}
CAPJA = {"storeyc": "$\\sigma$，上限付きの種（Storey型）", "bhc": "$\\sigma$，上限付きの種（単純BH法）"}
S12 = json.load(open(HERE / "data" / "p12_summary.json"))
S13 = json.load(open(HERE / "data" / "p13_summary.json"))
S11 = json.load(open(HERE / "data" / "p11_summary.json"))
PUB = json.load(open(HERE / "data" / "published.json"))
OO_NEAR, OO_FAR = ["ssb_hard", "ninco"], ["inaturalist", "textures", "openimageo"]
FOUR = ["inat", "sun", "places", "dtd"]
CIFAR_NEAR, CIFAR_FAR = ["cifar10", "tin"], ["mnist", "svhn", "texture", "places365"]
SEL = S14["selection"]["chosen"]                                   # e.g. xfit|storey|q0.2|pp|none (the registered rule's choice)
CFG = SEL.rsplit("|", 1)[0]                                        # xfit|storey|q0.2|pp
if not os.environ.get("P17_DEV"):
    assert CFG == "xfit|storey|q0.2|pp", CFG
RULE, QK, MODE = CFG.split("|")[1:]
SC = CFG + "|pooled"                                               # the statistic sigma (fold-pooled): the score s = s0 exp(sigma)
MAIN, ALONE = SC + "|TINS", SC + "|none"                           # ranking rows use sigma; validity rows use its calibrated rank p_fin (CFG)
PF_MAIN, PF_ALONE = CFG + "|TINS", CFG + "|none"                   # the registered form (p_fin x base): reported in the ablation
V1 = "v1|storey|q0.2|pp"
LP, STATIC = "xfit|lp|pooled", "xfit|static|pooled"
TT, TV = S14["test_train"], S14["test_val"]
GRP = {"near": ("openood", "near"), "far": ("openood", "far"), "four": ("fourood", "fourood")}
DS_JA = {"ssb_hard": "SSB-hard", "ninco": "NINCO", "inaturalist": "iNaturalist", "textures": "Textures", "openimageo": "OpenImage-O",
         "inat": "iNat. (Four-OOD)", "sun": "SUN", "places": "Places", "dtd": "Textures (Four-OOD)",
         "cifar10": "CIFAR-10", "tin": "TIN", "mnist": "MNIST", "svhn": "SVHN", "texture": "Textures", "places365": "Places365"}


def tt(key, grp):
    part, g = GRP[grp]
    e = TT[part][g].get(key)
    return None if e is None else (e["AUROC"], e["FPR95"])


def tt_e(key, grp):
    part, g = GRP[grp]
    return TT[part][g][key]


def tt_ds(key, part, ds):
    e = TT[part]["per_dataset"].get(f"{key}|{ds}")
    return None if e is None else (e["AUROC"], e["FPR95"])


def meas(name, part, grp):
    e = S11[part][grp][f"-|-|{name}"]
    return e["AUROC"], e["FPR95"]


def meas_ds(name, part, ds):
    e = S11[part]["per_dataset"][f"-|-|{name}|{ds}"]
    return e["AUROC"], e["FPR95"]


def six(key, src=tt):
    parts = [src(key, "near"), src(key, "far"), src(key, "four")]
    if any(x is None for x in parts):
        return None
    return [v for x in parts for v in x]


def tt15(key, grp):
    C = S15.get("cap_test") or {}
    part, g = GRP[grp]
    e = (C.get(part) or {}).get(g, {}).get(key)
    return None if e is None else (e["AUROC"], e["FPR95"])


def six_meas(name):
    return [v for x in (meas(name, "openood", "near"), meas(name, "openood", "far"), meas(name, "fourood", "fourood")) for v in x]


def onl14(key, grp):
    o = S14.get("online")
    part, g = GRP[grp]
    if not o or part not in o or key not in o[part][g]:
        return None
    e = o[part][g][key]
    return e["AUROC"], e["FPR95"]


def onl13(key, grp):
    o = S13.get("online")
    part, g = GRP[grp]
    if not o or part not in o or key not in o[part][g]:
        return None
    e = o[part][g][key]
    return e["AUROC"], e["FPR95"]


def short(label):
    return re.sub(r"（[^）]*）", "", label)


def mean(vs):
    return sum(vs) / len(vs)


def f2(x):
    return f"{x:.2f}"


def f3(x):
    return f"{x:.3f}"


def fmt_table(rows, ncols, fmt, gap_row=None, gap_label="\\quad 最良の比較手法との差", mark_best=True):
    vals = [r[1] for r in rows]
    out, best, best_other = [], [], []
    for j in range(ncols):
        col = [v[j] for v in vals if v is not None and v[j] is not None]
        oth = [v[j] for (lab, v, o) in rows if v is not None and v[j] is not None and not o]
        hi = (j % 2 == 0)
        best.append((max if hi else min)(col) if col else None)
        best_other.append((max if hi else min)(oth) if oth else None)
    for i, (lab, v, o) in enumerate(rows):
        if v is None:
            out.append(lab)
            continue
        cells, gaps = [], []
        for j in range(ncols):
            x = v[j]
            if x is None:
                cells.append("--")
                gaps.append("")
                continue
            hi = (j % 2 == 0)
            s = fmt[j](x)
            better = best_other[j] is not None and ((x > best_other[j] + 1e-9) if hi else (x < best_other[j] - 1e-9))
            if o and better:
                s = f"\\cellcolor{{upd}}\\textbf{{{s}}}"
            if mark_best and best[j] is not None and abs(x - best[j]) < 1e-9:
                s += "\\sota"
            cells.append(s)
            # the gap row is computed from the displayed (rounded) values so that a reader can verify it from the table
            g = ((round(x, 2) - round(best_other[j], 2)) if hi else (round(best_other[j], 2) - round(x, 2))) if best_other[j] is not None else None
            gaps.append(f"\\gap{{{g:+.2f}}}" if g is not None else "")
        out.append(lab + " & " + " & ".join(cells) + " \\\\")
        if gap_row is not None and i in (gap_row if isinstance(gap_row, (list, tuple)) else [gap_row]):
            out.append("{\\scriptsize\\textcolor{black!60}{" + gap_label + "}} & " + " & ".join(gaps) + " \\\\")
    return "\n".join(out)


def gaps(rows, gap_row, names):
    out = {}
    if gap_row is None:
        return out
    v = rows[gap_row][1]
    for j, name in enumerate(names):
        hi = (j % 2 == 0)
        col = [(r[1][j], r[0]) for r in rows if r[1] is not None and not r[2] and r[1][j] is not None]
        if not col or v[j] is None:
            continue
        b = (max if hi else min)(col)
        out[f"gap_{name}"] = (round(v[j], 2) - round(b[0], 2)) if hi else (round(b[0], 2) - round(v[j], 2))     # from the displayed values
        out[f"best_other_{name}"] = b[0]
        out[f"best_other_{name}_label"] = b[1]
    return out


def head6(label, font="7pt", lead="8.2pt", colsep="2.2pt"):
    return (f"{{\\fontsize{{{font}}}{{{lead}}}\\selectfont\\renewcommand{{\\arraystretch}}{{1.0}}%\n\\setlength{{\\tabcolsep}}{{{colsep}}}\\begin{{tabular}}{{@{{}}lcccccc@{{}}}}\n\\hline\n"
            "\\rule[-2.0pt]{0pt}{8.6pt} & \\multicolumn{2}{c}{OpenOOD near-OOD} & \\multicolumn{2}{c}{OpenOOD far-OOD} & \\multicolumn{2}{c}{Four-OOD} \\\\\n"
            f"\\rule[-2.0pt]{{0pt}}{{8.6pt}}{label} & AUROC$\\uparrow$ & FPR95$\\downarrow$ & AUROC$\\uparrow$ & FPR95$\\downarrow$ & AUROC$\\uparrow$ & FPR95$\\downarrow$ \\\\\n\\hline\n")


TAIL = "\n\\hline\n\\end{tabular}}%\n"
OURS = {ALONE: "Salmon Ladder，単体", MAIN: "Salmon Ladder $\\times$TINS"}
BASE_JA = {"TINS": "TINS~\\cite{yang2026tins}", "TANL": "TANL~\\cite{zhang2026tanl}", "NegLabel": "NegLabel~\\cite{jiang2024neglabel}", "AdaNeg": "AdaNeg~\\cite{zhang2024adaneg}"}
SIXN = ["near_AUROC", "near_FPR95", "far_AUROC", "far_FPR95", "four_AUROC", "four_FPR95"]


def macro_key(k):
    return k.replace("+", "p").replace("|", "_").replace(".", "").replace("-", "")


def main():
    sec = HERE / "sections"
    N = {"main_key": MAIN, "cfg": CFG, "rule": RULE, "mode": MODE, "q_main": float(QK[1:]), "n_draws": S14["test_train"]["n_draws"]}
    # ------------------------------------------------------------------ table: main
    rows = []
    rows.append(("\\multicolumn{7}{@{}l}{\\rule{0pt}{6.6pt}\\textit{ImageNet-1Kのラベルで学習した検出器（報告値$^{\\dagger}$，近／遠の平均は本稿で計算）}} \\\\", None, False))
    for m in ["MOODv2", "NNGuide", "SCALE"]:
        p = PUB["openood"][m]["per"]
        rows.append((PUB["openood"][m]["label"], [mean([p[d][0] for d in OO_NEAR]), mean([p[d][1] for d in OO_NEAR]),
                                                   mean([p[d][0] for d in OO_FAR]), mean([p[d][1] for d in OO_FAR]), None, None], False))
    rows.append(("\\multicolumn{7}{@{}l}{\\rule{0pt}{6.6pt}\\textit{教師なし（報告値$^{\\dagger}$）}} \\\\", None, False))
    p = PUB["openood"]["DisCoPatch"]["per"]
    rows.append((PUB["openood"]["DisCoPatch"]["label"], [mean([p[d][0] for d in OO_NEAR]), mean([p[d][1] for d in OO_NEAR]),
                                                          mean([p[d][0] for d in OO_FAR]), mean([p[d][1] for d in OO_FAR]), None, None], False))
    rows.append(("\\multicolumn{7}{@{}l}{\\rule{0pt}{6.6pt}\\textit{VLM，到着時採点（CLIP ViT-B/16，報告値$^{\\dagger}$）}} \\\\", None, False))
    for m in ["NegLabel", "AdaNeg", "InterNeg", "TINS"]:
        e = PUB["openood"][m]
        f = PUB["fourood"].get(m, {}).get("avg")
        rows.append((e["label"], [e["near"][0], e["near"][1], e["far"][0], e["far"][1]] + (list(f) if f else [None, None]), False))
    rows.append(("\\multicolumn{7}{@{}l}{\\rule{0pt}{6.6pt}\\textit{同じテスト集合での実測（公式実装，到着時採点，学習なし）}} \\\\", None, False))
    for m in ["NegLabel", "AdaNeg", "TANL", "TINS"]:
        rows.append((BASE_JA[m], six_meas(m), False))
    rows.append(("\\multicolumn{7}{@{}l}{\\rule{0pt}{6.6pt}\\textit{Salmon Ladder（Ours，学習なし，クラスあたり16枚，事後採点；6通りの抽出の平均）}} \\\\", None, False))
    rows.append((OURS[ALONE], six(ALONE), True))
    gap_alone = len(rows) - 1
    rows.append(("\\multicolumn{7}{@{}l}{\\rule{0pt}{6.6pt}\\textit{\\quad ベース検出器との積 $s_0\\exp\\sigma$（登録時の形$\\pfin\\times s_0$ではない；\\secref{score}）}} \\\\", None, False))
    for b in ("NegLabel", "AdaNeg", "TANL"):
        v = six(SC + "|" + b)
        if v is not None:
            rows.append((f"\\quad $\\times${b}", v, True))
    rows.append(("\\quad $\\times$TINS", six(MAIN), True))
    gap_row = len(rows) - 1
    body = fmt_table(rows, 6, [f2] * 6, gap_row=[gap_alone, gap_row])
    (sec / "t18_main.tex").write_text(head6("手法", font="6.8pt", lead="8.0pt") + body + TAIL)
    for grp in ("near", "far", "four"):
        for key, tag in ((MAIN, ""), (ALONE, "alone_"), (V1 + "|TINS", "v1_"), (V1 + "|none", "v1_alone_"), (LP + "|TINS", "lp_"), (LP + "|none", "lp_alone_"), (PF_MAIN, "pf_"), (PF_ALONE, "pf_alone_")):
            e = tt_e(key, grp)
            N[f"{tag}{grp}_AUROC"], N[f"{tag}{grp}_FPR95"] = e["AUROC"], e["FPR95"]
            if e.get("FPR95_ci") and e["FPR95_ci"].get("lo") is not None:
                N[f"{tag}{grp}_FPR95_lo"], N[f"{tag}{grp}_FPR95_hi"] = e["FPR95_ci"]["lo"], e["FPR95_ci"]["hi"]
                N[f"{tag}{grp}_AUROC_lo"], N[f"{tag}{grp}_AUROC_hi"] = e["AUROC_ci"]["lo"], e["AUROC_ci"]["hi"]
            for ref in ("v1", "lp", "alone"):
                d = e.get(f"dFPR95_vs_{ref}")
                if d:
                    N[f"{tag}{grp}_dFPR95_vs_{ref}"], N[f"{tag}{grp}_dFPR95_vs_{ref}_lo"], N[f"{tag}{grp}_dFPR95_vs_{ref}_hi"] = d["mean"], d["lo"], d["hi"]
                    da = e.get(f"dAUROC_vs_{ref}")
                    N[f"{tag}{grp}_dAUROC_vs_{ref}"] = da["mean"]
        for b in ("NegLabel", "AdaNeg", "TANL"):
            e = TT[GRP[grp][0]][GRP[grp][1]].get(SC + "|" + b)
            if e:
                N[f"base_{b}_{grp}_AUROC"], N[f"base_{b}_{grp}_FPR95"] = e["AUROC"], e["FPR95"]
                part, g = GRP[grp]
                N[f"meas_{b}_{grp}_AUROC"], N[f"meas_{b}_{grp}_FPR95"] = meas(b, part, g)
                N[f"base_{b}_{grp}_gain_FPR95"] = N[f"meas_{b}_{grp}_FPR95"] - e["FPR95"]
        part, g = GRP[grp]
        N[f"meas_TINS_{grp}_AUROC"], N[f"meas_TINS_{grp}_FPR95"] = meas("TINS", part, g)
        N[f"tins_gain_{grp}_FPR95"] = N[f"meas_TINS_{grp}_FPR95"] - N[f"{grp}_FPR95"]
    others = [r for r in rows if r[1] is not None and not r[2]]
    for j, name in enumerate(SIXN):
        col = [(r[1][j], r[0]) for r in others if r[1][j] is not None]
        hi = j % 2 == 0
        b = (max if hi else min)(col)
        N[f"best_other_{name}"], N[f"best_other_{name}_label"] = b[0], b[1]
        r2 = lambda x: round(x, 2)                                                   # gaps from the displayed (rounded) values
        N[f"gap_{name}"] = (r2(N[name]) - r2(b[0])) if hi else (r2(b[0]) - r2(N[name]))
        N[f"alone_gap_{name}"] = (r2(N[f"alone_{name}"]) - r2(b[0])) if hi else (r2(b[0]) - r2(N[f"alone_{name}"]))
        for bb in ("NegLabel", "AdaNeg", "TANL"):
            if f"base_{bb}_{name}" in N:
                N[f"base_{bb}_gap_{name}"] = (r2(N[f"base_{bb}_{name}"]) - r2(b[0])) if hi else (r2(b[0]) - r2(N[f"base_{bb}_{name}"]))
    for tag in ("", "alone_"):
        for grp in ("near", "far", "four"):
            if f"{tag}{grp}_dFPR95_vs_lp" in N:
                N[f"{tag}{grp}_dFPR95_vs_lp_abs"] = abs(N[f"{tag}{grp}_dFPR95_vs_lp"])
    N["t1_alone_wins"] = sum(1 for n in SIXN if N[f"alone_gap_{n}"] > 0)
    N["t1_base_rows_all_win"] = int(all(N.get(f"base_{bb}_gap_{n}", -1) > 0 for bb in ("NegLabel", "AdaNeg", "TANL") for n in SIXN if f"base_{bb}_{n}" in N))
    N["near_FPR95_ratio_vs_best_other"] = N["best_other_near_FPR95"] / N["near_FPR95"]
    N["alone_near_FPR95_ratio_vs_best_other"] = N["best_other_near_FPR95"] / N["alone_near_FPR95"]
    # ------------------------------------------------------------------ table: per dataset OpenOOD
    rows = []
    for m in ["MOODv2", "NNGuide", "RankFeat", "SCALE", "DisCoPatch", "TINS"]:
        p = PUB["openood"][m]["per"]
        rows.append((short(PUB["openood"][m]["label"]) + "$^{\\dagger}$", sum(([p[d][0], p[d][1]] for d in OO_NEAR + OO_FAR), []), False))
    rows.append(("TINS~\\cite{yang2026tins}（実測）", sum((list(meas_ds("TINS", "openood", d)) for d in OO_NEAR + OO_FAR), []), False))
    rows.append((OURS[ALONE], sum((list(tt_ds(ALONE, "openood", d)) for d in OO_NEAR + OO_FAR), []), True))
    rows.append((OURS[MAIN], sum((list(tt_ds(MAIN, "openood", d)) for d in OO_NEAR + OO_FAR), []), True))
    gap_row = len(rows) - 1
    body = fmt_table(rows, 10, [f2] * 10, gap_row=gap_row)
    t = ("{\\fontsize{7pt}{8.2pt}\\selectfont\\renewcommand{\\arraystretch}{1.0}%\n\\setlength{\\tabcolsep}{2.0pt}\\begin{tabular}{@{}lcccccccccc@{}}\n\\hline\n"
         "\\rule[-2.0pt]{0pt}{8.6pt} & \\multicolumn{4}{c}{near-OOD} & \\multicolumn{6}{c}{far-OOD} \\\\\n"
         "\\rule[-2.0pt]{0pt}{8.6pt} & \\multicolumn{2}{c}{SSB-hard} & \\multicolumn{2}{c}{NINCO} & \\multicolumn{2}{c}{iNaturalist} & \\multicolumn{2}{c}{Textures} & \\multicolumn{2}{c}{OpenImage-O} \\\\\n"
         "\\rule[-2.0pt]{0pt}{8.6pt}手法 & AUROC & FPR95 & AUROC & FPR95 & AUROC & FPR95 & AUROC & FPR95 & AUROC & FPR95 \\\\\n\\hline\n" + body + TAIL)
    (sec / "t18_perds.tex").write_text(t)
    N.update({f"t2_{k}": v for k, v in gaps(rows, gap_row, sum(([f"{d}_AUROC", f"{d}_FPR95"] for d in OO_NEAR + OO_FAR), [])).items()})
    N["t2_wins"] = sum(1 for k, v in N.items() if k.startswith("t2_gap_") and v > 0)
    for d in OO_NEAR + OO_FAR:
        N[f"{d}_AUROC"], N[f"{d}_FPR95"] = tt_ds(MAIN, "openood", d)
        N[f"alone_{d}_AUROC"], N[f"alone_{d}_FPR95"] = tt_ds(ALONE, "openood", d)
        pd_ = PUB["openood"]["DisCoPatch"]["per"][d]
        N[f"{d}_vs_discopatch_AUROC"], N[f"{d}_vs_discopatch_FPR95"] = N[f"{d}_AUROC"] - pd_[0], pd_[1] - N[f"{d}_FPR95"]
        N[f"{d}_vs_discopatch_FPR95_abs"] = abs(pd_[1] - N[f"{d}_FPR95"])
        ci = TT["openood"].get("per_dataset_ci", {}).get(f"{MAIN}|{d}")
        if ci:
            N[f"{d}_FPR95_lo"], N[f"{d}_FPR95_hi"] = ci["FPR95_ci"]["lo"], ci["FPR95_ci"]["hi"]
            N[f"{d}_AUROC_lo"], N[f"{d}_AUROC_hi"] = ci["AUROC_ci"]["lo"], ci["AUROC_ci"]["hi"]
    # ------------------------------------------------------------------ table: Four-OOD per dataset
    rows = []
    for m in ["MCM", "NegLabel", "CSP", "OODD", "AdaNeg", "InterNeg", "TINS"]:
        e = PUB["fourood"][m]
        rows.append((short(e["label"]) + "$^{\\dagger}$", sum(([e["per"][d][0], e["per"][d][1]] for d in FOUR), []) + list(e["avg"]), False))
    rows.append(("TINS~\\cite{yang2026tins}（実測）", sum((list(meas_ds("TINS", "fourood", d)) for d in FOUR), []) + list(meas("TINS", "fourood", "fourood")), False))
    rows.append((OURS[ALONE], sum((list(tt_ds(ALONE, "fourood", d)) for d in FOUR), []) + list(tt(ALONE, "four")), True))
    rows.append((OURS[MAIN], sum((list(tt_ds(MAIN, "fourood", d)) for d in FOUR), []) + list(tt(MAIN, "four")), True))
    gap_row = len(rows) - 1
    body = fmt_table(rows, 10, [f2] * 10, gap_row=gap_row)
    t = ("{\\fontsize{7pt}{8.2pt}\\selectfont\\renewcommand{\\arraystretch}{1.0}%\n\\setlength{\\tabcolsep}{2.0pt}\\begin{tabular}{@{}lcccccccccc@{}}\n\\hline\n"
         "\\rule[-2.0pt]{0pt}{8.6pt} & \\multicolumn{2}{c}{iNaturalist} & \\multicolumn{2}{c}{SUN} & \\multicolumn{2}{c}{Places} & \\multicolumn{2}{c}{Textures} & \\multicolumn{2}{c}{平均} \\\\\n"
         "\\rule[-2.0pt]{0pt}{8.6pt}手法 & AUROC & FPR95 & AUROC & FPR95 & AUROC & FPR95 & AUROC & FPR95 & AUROC & FPR95 \\\\\n\\hline\n" + body + TAIL)
    (sec / "t18_four.tex").write_text(t)
    N.update({f"t3_{k}": v for k, v in gaps(rows, gap_row, sum(([f"{d}_AUROC", f"{d}_FPR95"] for d in FOUR), []) + ["avg_AUROC", "avg_FPR95"]).items()})
    N["t3_wins"] = sum(1 for k, v in N.items() if k.startswith("t3_gap_") and v > 0)
    N["t3_wins_ds"] = sum(1 for k, v in N.items() if k.startswith("t3_gap_") and "_avg_" not in k and v > 0)
    N["t23_wins_ds"], N["t23_cols_ds"] = N["t2_wins"] + N["t3_wins_ds"], 2 * len(OO_NEAR + OO_FAR + FOUR)
    # ------------------------------------------------------------------ table: read-outs on the same features (ablation), x TINS and standalone
    ABL = [(STATIC, "静的$p$（原型への余弦の較正順位，グラフなし）"), ("knn", "kNN~\\cite{sun2022knn}（支持画像への$k{=}10$近傍の較正順位，グラフなし）"),
           ("lgcmax", "LGC-max（支持画像のクラスラベルを伝播，最大クラス質量の較正順位）"), (LP, "$\\pP$のみ（支持画像からの伝播，種なし，交差適合）"),
           (V1, "$\\pP\\pN$，交差適合なし（v16までの読み出し；保証は種だけ）"), (SC, "$\\sigma$：交差適合した$\\pP\\pN$（本稿）"),
           (CFG, "$\\pfin$：$\\sigma$の較正順位をそのままスコアにしたもの（登録時の形）")]
    ABLX = [(f"xfit|bh|{QK}|pp|pooled", "$\\sigma$，種を単純BH法で選択"), (f"xfit|{RULE}|{QK}|pps|pooled", "$\\sigma$に静的$p$を加える（\\secref{rec}）"),
            (f"xfit|{RULE}|{QK}|min|pooled", "$\\min(\\sigma,\\,\\log$静的$p)$"), (f"xfit|bh|{QK}|pps|pooled", "単純BH法の種，静的$p$を加える"),
            (f"xfit|storey|q0.1|pp|pooled", "$\\sigma$，$q{=}0.1$"), (f"xfit|storey|q0.05|pp|pooled", "$\\sigma$，$q{=}0.05$"), (f"xfit|bh|q0.1|pp|pooled", "単純BH法の種，$q{=}0.1$"),
            ("v1|storey|q0.1|pp", "$\\pP\\pN$，交差適合なし，$q{=}0.1$"), ("v1|storey|q0.05|pp", "$\\pP\\pN$，交差適合なし，$q{=}0.05$")]
    rows, keys = [], []
    for base, title in (("TINS", "同じ特徴（DINOv2 L/14 + DINOv3 L/16）での読み出し，$\\times$TINS（単体は付録\\ref{sec:app_sens}の\\tabref{ablx}）"),):
        rows.append((f"\\multicolumn{{7}}{{@{{}}l}}{{\\rule{{0pt}}{{6.6pt}}\\textit{{{title}}}}} \\\\", None, False))
        for k, lab in ABL:
            v = six(f"{k}|{base}")
            if v is None:
                continue
            rows.append((lab, v, False))
            keys.append(f"{k}|{base}")
            if k == SC:
                for rule, clab in CAPJA.items():
                    v15 = six(f"xfit|{rule}|{QK}|pp|pooled|{base}", tt15)
                    if v15:
                        rows.append((clab + "（\\secref{cap}）", v15, False))
        o13 = six(f"L14+D3L|T|{base}", onl13)
        if o13:
            rows.append(("$\\pP\\pN$，到着時採点，交差適合なし（付録\\ref{sec:app_online}）", o13, False))
        o14 = six(f"xonl|{RULE}|{QK}|{MODE}|pooled|{base}", onl14)
        if o14:
            rows.append(("$\\sigma$，到着時採点（付録\\ref{sec:app_online}）", o14, False))
    body = fmt_table(rows, 6, [f2] * 6)
    (sec / "t18_abl.tex").write_text(head6("構成", font="6.8pt", lead="8.0pt", colsep="2.0pt") + body + TAIL)
    rows = []
    for base, title in (("none", "同じ特徴での読み出し，単体"), ("TINS", "第2段の他の候補，$\\times$TINS")):
        rows.append((f"\\multicolumn{{7}}{{@{{}}l}}{{\\rule{{0pt}}{{6.6pt}}\\textit{{{title}}}}} \\\\", None, False))
        for k, lab in (ABL if base == "none" else [(SC, "$\\sigma$（本稿：Storey型，$q{=}0.2$）")]) + ABLX:
            v = six(f"{k}|{base}")
            if v is None:
                continue
            rows.append((short(lab) if base == "none" else lab, v, False))
            keys.append(f"{k}|{base}")
            if k == SC:
                for rule, clab in CAPJA.items():
                    v15 = six(f"xfit|{rule}|{QK}|pp|pooled|{base}", tt15)
                    if v15:
                        rows.append((clab, v15, False))
        if base == "none":
            o13 = six(f"L14+D3L|T|{base}", onl13)
            if o13:
                rows.append(("$\\pP\\pN$，到着時採点，交差適合なし", o13, False))
            o14 = six(f"xonl|{RULE}|{QK}|{MODE}|pooled|{base}", onl14)
            if o14:
                rows.append(("$\\sigma$，到着時採点", o14, False))
    body = fmt_table(rows, 6, [f2] * 6)
    (sec / "t18_ablx.tex").write_text(head6("構成", font="6.8pt", lead="8.0pt", colsep="2.0pt") + body + TAIL)
    C15T = S15.get("cap_test")
    if C15T:
        for rule in CAPJA:
            for base in ("TINS", "none"):
                for grp in ("near", "far", "four"):
                    part, g = GRP[grp]
                    e = C15T[part][g].get(f"xfit|{rule}|{QK}|pp|pooled|{base}")
                    if e:
                        tag = f"cap_{rule}_{'' if base == 'TINS' else 'alone_'}{grp}"
                        N[f"{tag}_AUROC"], N[f"{tag}_FPR95"] = e["AUROC"], e["FPR95"]
                        pe = C15T[part].get(f"paired_{g}", {}).get(f"xfit|{rule}|{QK}|pp|pooled|{base}", {})
                        for ref in ("lp", "storey"):
                            if f"dFPR95_vs_{ref}" in pe:
                                d = pe[f"dFPR95_vs_{ref}"]
                                N[f"{tag}_dFPR95_vs_{ref}"], N[f"{tag}_dFPR95_vs_{ref}_lo"], N[f"{tag}_dFPR95_vs_{ref}_hi"] = d["mean"], d["lo"], d["hi"]
            for d in OO_NEAR + OO_FAR:
                for rl in (rule, "storey", "bh"):
                    s = C15T["openood"]["seeds"].get(f"{d}|D3L|{rl}|{QK}")
                    if s:
                        N[f"cap_seed_{d}_{rl}_ratio"], N[f"cap_seed_{d}_{rl}_idfrac"] = 100.0 * s["seeds_over_n_test"], (s["id_frac"] if s["id_frac"] is not None else 0.0)
                e = C15T["openood"]["per_dataset"].get(f"xfit|{rule}|{QK}|pp|pooled|TINS|{d}")
                if e:
                    N[f"cap_{rule}_{d}_FPR95"] = e["FPR95"]
                e = C15T["openood"]["per_dataset"].get(f"xfit|{rule}|{QK}|pp|pooled|none|{d}")
                if e:
                    N[f"cap_{rule}_alone_{d}_FPR95"] = e["FPR95"]
    for key in keys:
        kk = "abl_" + macro_key(key)
        for grp in ("near", "far", "four"):
            e = tt_e(key, grp)
            N[f"{kk}_{grp}_AUROC"], N[f"{kk}_{grp}_FPR95"] = e["AUROC"], e["FPR95"]
            for ref in ("v1", "lp", "alone"):
                d = e.get(f"dFPR95_vs_{ref}")
                if d:
                    N[f"{kk}_{grp}_dFPR95_vs_{ref}"], N[f"{kk}_{grp}_dFPR95_vs_{ref}_lo"], N[f"{kk}_{grp}_dFPR95_vs_{ref}_hi"] = d["mean"], d["lo"], d["hi"]
    for base in ("TINS", "none"):
        for grp in ("near", "far", "four"):
            o = onl14(f"xonl|{RULE}|{QK}|{MODE}|pooled|{base}", grp)
            if o:
                N[f"onl14_{base}_{grp}_AUROC"], N[f"onl14_{base}_{grp}_FPR95"] = o
            o = onl13(f"L14+D3L|T|{base}", grp)
            if o:
                N[f"onl13_{base}_{grp}_AUROC"], N[f"onl13_{base}_{grp}_FPR95"] = o
    OL = S14.get("online") or {}
    if OL.get("iters_mean"):
        N["onl_iters_mean"], N["onl_iters_max"], N["onl_seconds_per_stream"] = OL["iters_mean"], OL["iters_max"], OL["seconds_per_stream_mean"]
    # ------------------------------------------------------------------ validity of p_final: exchangeable calibration (test_val) and benchmark calibration (test_train)
    VK = ("fpr@0.01", "fpr@0.05", "fpr@0.1", "fpr@0.2", "bh_fdp", "bh_recall")
    rows = []
    for src, tagsrc in ((TV, "vcal"), (TT, "vtrain")):
        allv = []
        for part, dss in (("openood", OO_NEAR + OO_FAR), ("fourood", FOUR)):
            val = src[part]["validity"]["per_dataset"]
            for d in dss:
                e = val.get(f"{CFG}|{d}")                                             # validity rows are keyed by the configuration (standalone)
                if e is None:
                    continue
                allv.append(e)
                for k in VK:
                    N[f"{tagsrc}_{d}_{k.replace('@', '').replace('.', '')}"] = e[k]
                if tagsrc == "vcal":
                    rows.append((DS_JA[d], [e[k] for k in VK], False))
        if allv:
            for k in VK:
                xs = [e[k] for e in allv]
                kk = k.replace("@", "").replace(".", "")
                N[f"{tagsrc}_{kk}_min"], N[f"{tagsrc}_{kk}_max"], N[f"{tagsrc}_{kk}_mean"] = min(xs), max(xs), mean(xs)
        # the realised ID fraction among the seeds per fold (chosen rule and plain BH), over data sets and views
        for rule in ("storey", "bh"):
            fr, mx = [], []
            for part in ("openood", "fourood"):
                for k, e in src[part]["seeds"].items():
                    if k.endswith(f"|{rule}|{QK}") and e["id_frac"] is not None:
                        fr.append(e["id_frac"])
                        mx.append(e["id_frac_max"])
            if fr:
                N[f"{tagsrc}_idfrac_{rule}_min"], N[f"{tagsrc}_idfrac_{rule}_max"], N[f"{tagsrc}_idfrac_{rule}_mean"], N[f"{tagsrc}_idfrac_{rule}_maxdraw"] = min(fr), max(fr), mean(fr), max(mx)
        for d in OO_NEAR + OO_FAR + FOUR:
            part = "openood" if d in OO_NEAR + OO_FAR else "fourood"
            for v in ("L14", "D3L"):
                e = src[part]["seeds"].get(f"{d}|{v}|{RULE}|{QK}")
                if e:
                    N[f"{tagsrc}_seed_{d}_{v}_idfrac"], N[f"{tagsrc}_seed_{d}_{v}_n"], N[f"{tagsrc}_seed_{d}_{v}_pi0"] = e["id_frac"], e["seeds"], e["pi0_hat"]
    if rows:
        lines = [lab + " & " + " & ".join(f3(x) for x in v) + " \\\\" for lab, v, _ in rows]
        t = ("{\\fontsize{6.5pt}{7.6pt}\\selectfont\\renewcommand{\\arraystretch}{1.0}%\n\\setlength{\\tabcolsep}{1.7pt}\\begin{tabular}{@{}lcccc|cc@{}}\n\\hline\n"
             "\\rule[-2.0pt]{0pt}{8.6pt} & \\multicolumn{4}{c|}{IDの誤警報率 $\\Pr(\\pfin\\le\\alpha)$} & \\multicolumn{2}{c}{BH法（$q{=}" + QK[1:] + "$）} \\\\\n"
             "\\rule[-2.0pt]{0pt}{8.6pt}データセット & $\\alpha{=}0.01$ & $0.05$ & $0.1$ & $0.2$ & 偽発見割合 & 再現率 \\\\\n\\hline\n" + "\n".join(lines) + TAIL)
        (sec / "t18_val.tex").write_text(t)
    N["vcal_n_splits"] = TV["n_splits"]
    # metrics under the exchangeable calibration
    for grp in ("near", "far", "four"):
        part, g = GRP[grp]
        for key, tag in ((MAIN, "main"), (ALONE, "alone"), (V1 + "|none", "v1_alone")):
            e = TV[part]["metrics"][g].get(key)
            if e:
                N[f"vcal_{tag}_{grp}_AUROC"], N[f"vcal_{tag}_{grp}_FPR95"] = e["AUROC"], e["FPR95"]
    # ------------------------------------------------------------------ seeds table (appendix): benchmark calibration, chosen rule, per fold mean
    rows = []
    for part, dss in (("openood", OO_NEAR + OO_FAR), ("fourood", FOUR)):
        for d in dss:
            cells = []
            for v in ("L14", "D3L"):
                e = TT[part]["seeds"].get(f"{d}|{v}|{RULE}|{QK}")
                eb = TT[part]["seeds"].get(f"{d}|{v}|bh|{QK}")
                cells += [e["seeds"], e["id_frac"], eb["seeds"], eb["id_frac"]] if (e and eb) else [None] * 4
            rows.append((DS_JA[d], cells, False))
    lines = []
    for lab, v, _ in rows:
        cells = [("--" if x is None else (f"{x:,.0f}" if i % 2 == 0 else f"{x:.3f}")) for i, x in enumerate(v)]
        lines.append(lab + " & " + " & ".join(cells) + " \\\\")
    t = ("{\\fontsize{6.0pt}{7.1pt}\\selectfont\\renewcommand{\\arraystretch}{1.0}%\n\\setlength{\\tabcolsep}{1.3pt}\\begin{tabular}{@{}lcccc|cccc@{}}\n\\hline\n"
         "\\rule[-2.0pt]{0pt}{8.2pt} & \\multicolumn{4}{c|}{DINOv2 L/14} & \\multicolumn{4}{c}{DINOv3 L/16} \\\\\n"
         "\\rule[-2.0pt]{0pt}{8.2pt} & \\multicolumn{2}{c}{Storey型} & \\multicolumn{2}{c|}{単純BH} & \\multicolumn{2}{c}{Storey型} & \\multicolumn{2}{c}{単純BH} \\\\\n"
         "\\rule[-2.0pt]{0pt}{8.2pt}データセット & 種 & ID率 & 種 & ID率 & 種 & ID率 & 種 & ID率 \\\\\n\\hline\n" + "\n".join(lines) + TAIL)
    (sec / "t18_fdp.tex").write_text(t)
    # ------------------------------------------------------------------ recurrence x mixing grid (numbers, appendix tables)
    REC = S14.get("rec") or {}
    GRIDCFG = {"static": f"{STATIC}|none", "lp": f"{LP}|none", "v1": f"{V1}|none", "main": ALONE, "pps": f"xfit|{RULE}|{QK}|pps|pooled|none", "min": f"xfit|{RULE}|{QK}|min|pooled|none", "bh": f"xfit|bh|{QK}|pp|pooled|none", "mainT": MAIN, "pf": PF_ALONE}
    if REC:
        for ds, cmax, tag in (("ninco", 64, "ninco"), ("ssb_hard", 980, "ssb")):
            for k, e in REC.items():
                if k.startswith(f"{ds}|c{cmax}|"):
                    r = int(k.split("|")[2][1:])
                    for name, cfg in GRIDCFG.items():
                        if cfg in e["cfg"]:
                            N[f"rec_{tag}_c{cmax}_r{r}_{name}_FPR95"] = e["cfg"][cfg]["FPR95"]
                    N[f"rec_{tag}_c{cmax}_r{r}_mix"] = 100 * e["mixing_rate"]
                    if "fpr@0.05" in e["cfg"].get(PF_ALONE, {}):
                        N[f"rec_{tag}_c{cmax}_r{r}_fpr005"] = e["cfg"][PF_ALONE]["fpr@0.05"]
        def cond(pred):
            return [k for k, e in REC.items() if pred(k, e)]
        rle5 = cond(lambda k, e: int(k.split("|")[2][1:]) <= 5)
        rge10 = cond(lambda k, e: int(k.split("|")[2][1:]) >= 10)
        N["rec_n_r_le5"], N["rec_n_r_ge10"] = len(rle5), len(rge10)
        N["rec_n_r_le5_main_worse_static"] = sum(1 for k in rle5 if REC[k]["cfg"][ALONE]["FPR95"] > REC[k]["cfg"][GRIDCFG["static"]]["FPR95"])
        N["rec_n_r_le5_pps_worse_static"] = sum(1 for k in rle5 if REC[k]["cfg"][GRIDCFG["pps"]]["FPR95"] > REC[k]["cfg"][GRIDCFG["static"]]["FPR95"])
        N["rec_n_r_le5_min_worse_static"] = sum(1 for k in rle5 if REC[k]["cfg"][GRIDCFG["min"]]["FPR95"] > REC[k]["cfg"][GRIDCFG["static"]]["FPR95"])
        N["rec_n_r_ge10_main_better_static"] = sum(1 for k in rge10 if REC[k]["cfg"][ALONE]["FPR95"] < REC[k]["cfg"][GRIDCFG["static"]]["FPR95"])
        N["rec_n_r_ge10_pps_worse_main"] = sum(1 for k in rge10 if REC[k]["cfg"][GRIDCFG["pps"]]["FPR95"] > REC[k]["cfg"][ALONE]["FPR95"])
        N["rec_n_main_le_v1"] = sum(1 for k in REC if REC[k]["cfg"][ALONE]["FPR95"] <= REC[k]["cfg"][GRIDCFG["v1"]]["FPR95"] + 1e-9)
        N["rec_n_main_lt_v1"] = sum(1 for k in REC if REC[k]["cfg"][ALONE]["FPR95"] < REC[k]["cfg"][GRIDCFG["v1"]]["FPR95"] - 1e-9)
        N["rec_n_all"] = len(REC)
        N["rec_max_main_minus_v1"] = max(REC[k]["cfg"][ALONE]["FPR95"] - REC[k]["cfg"][GRIDCFG["v1"]]["FPR95"] for k in REC)
        N["rec_min_main_minus_v1"] = min(REC[k]["cfg"][ALONE]["FPR95"] - REC[k]["cfg"][GRIDCFG["v1"]]["FPR95"] for k in REC)
        N["rec_n_main_within1_v1"] = sum(1 for k in REC if abs(REC[k]["cfg"][ALONE]["FPR95"] - REC[k]["cfg"][GRIDCFG["v1"]]["FPR95"]) <= 1.0 + 1e-9)
        N["rec_max_main_minus_static_le5"] = max(REC[k]["cfg"][ALONE]["FPR95"] - REC[k]["cfg"][GRIDCFG["static"]]["FPR95"] for k in rle5)
        N["rec_max_pps_minus_static_le5"] = max(REC[k]["cfg"][GRIDCFG["pps"]]["FPR95"] - REC[k]["cfg"][GRIDCFG["static"]]["FPR95"] for k in rle5)
        fp = [REC[k]["cfg"][PF_ALONE]["fpr@0.05"] for k in REC if "fpr@0.05" in REC[k]["cfg"][PF_ALONE]]
        if fp:
            N["rec_fpr005_min"], N["rec_fpr005_max"] = min(fp), max(fp)
        for ds, lab in (("ninco", "NINCO"), ("ssb_hard", "SSB-hard")):
            lines = []
            keys = sorted((k for k in REC if k.startswith(ds + "|")), key=lambda k: (int(k.split("|")[1][1:]), int(k.split("|")[2][1:])))
            for k in keys:
                e = REC[k]
                c, r = k.split("|")[1][1:], k.split("|")[2][1:]
                cf = e["cfg"]
                def pair(key):
                    return f"{cf[key]['AUROC']:.1f}／{cf[key]['FPR95']:.1f}" if key in cf else "--"
                fp5 = cf.get(PF_ALONE, {}).get("fpr@0.05")
                sd = (e.get("view_D3L") or {}).get(f"{RULE}_{QK}") or {}
                idf = sd.get("id_frac") if sd else None
                seeds_s = (f"{sd['seeds']:,.0f} & " + ("--" if idf is None else f"{100 * idf:.0f}") + f" & {100 * sd['ood_seeds'] / max(e['n_ood'], 1):.0f}") if sd else "-- & -- & --"
                lines.append(f"{c} & {r} & {e['n_ood']:,} & {100 * e['mixing_rate']:.2f} & {pair(GRIDCFG['static'])} & {pair(GRIDCFG['lp'])} & {pair(GRIDCFG['v1'])} & {pair(ALONE)} & {pair(GRIDCFG['pps'])} & {pair(MAIN)} & "
                             f"{('--' if fp5 is None else f'{fp5:.3f}')} & {seeds_s} \\\\")
            t = ("{\\fontsize{6.0pt}{7.1pt}\\selectfont\\renewcommand{\\arraystretch}{1.0}%\n\\setlength{\\tabcolsep}{1.4pt}\\begin{tabular}{@{}rrrr|cccccc|c|rrr@{}}\n\\hline\n"
                 "\\rule[-2.0pt]{0pt}{8.2pt}$c$ & $r$ & 未知 & 混入率 & 静的$p$ & $\\pP$ & $\\pP\\pN$（交差なし） & $\\sigma$（本稿） & $\\sigma$+静的$p$ & $\\sigma\\times$TINS & $\\Pr(\\pfin{\\le}0.05)$ & 種 & ID & 再現 \\\\\n\\hline\n" + "\n".join(lines) + TAIL)
            (sec / f"t18_rec_{ds}.tex").write_text(t)
            N[f"rec_{ds}_n"] = len(keys)
    # ------------------------------------------------------------------ ImageNet-200
    I2 = S14.get("in200")
    if I2:
        PUB200 = PUB.get("imagenet200", {})
        rows = [("\\multicolumn{5}{@{}l}{\\rule{0pt}{6.6pt}\\textit{ImageNet-200で学習したResNet-18（報告値$^{\\dagger}$）}} \\\\", None, False)]
        rows += [(e["label"], [e["near"][0], e["near"][1], e["far"][0], e["far"][1]], False) for name, e in PUB200.items() if not name.startswith("_")]
        B15 = (S15.get("base") or {}).get("imagenet200")
        if B15:
            rows.append(("\\multicolumn{5}{@{}l}{\\rule{0pt}{6.6pt}\\textit{全学習画像（" + f"{B15['n_train']:,}" + "枚）またはゼロショット（本稿で実測）}} \\\\", None, False))
            for m, lab in BASE15.items():
                if m in B15["near"]:
                    rows.append((lab, [B15["near"][m]["AUROC"], B15["near"][m]["FPR95"], B15["far"][m]["AUROC"], B15["far"][m]["FPR95"]], False))
        rows.append(("\\multicolumn{5}{@{}l}{\\rule{0pt}{6.6pt}\\textit{凍結特徴，クラスあたり16枚，学習なし，単体（" + f"{I2['n_draws']}" + "通りの抽出の平均）}} \\\\", None, False))
        def in2(key, src=None):
            src = src or I2
            a, b = src["near"][key], src["far"][key]
            return [a["AUROC"], a["FPR95"], b["AUROC"], b["FPR95"]]
        for key, lab, ours in ((f"{STATIC}|none", "静的$p$（同じ特徴，グラフなし）", False), ("knn|none", "kNN（同じ特徴，グラフなし）", False), ("lgcmax|none", "LGC-max（同じグラフ）", False),
                               (f"{LP}|none", "$\\pP$のみ（種なし）", True), (f"{V1}|none", "$\\pP\\pN$，交差適合なし", True), (ALONE, "$\\sigma$（本稿の構成）", True), (f"xfit|bh|{QK}|pp|pooled|none", "$\\sigma$，単純BH法の種", True)):
            if key in I2["near"]:
                rows.append((lab, in2(key), ours))
        C15 = S15.get("cap_in200")
        if C15:
            for rule, lab in CAPJA.items():
                key = f"xfit|{rule}|{QK}|pp|pooled|none"
                if key in C15["near"]:
                    rows.append((lab, in2(key, C15), True))
        body = fmt_table(rows, 4, [f2] * 4)
        t = ("{\\fontsize{6.8pt}{8pt}\\selectfont\\renewcommand{\\arraystretch}{1.0}%\n\\setlength{\\tabcolsep}{2pt}\\begin{tabular}{@{}lcccc@{}}\n\\hline\n"
             "\\rule[-2.0pt]{0pt}{8.6pt} & \\multicolumn{2}{c}{near-OOD} & \\multicolumn{2}{c}{far-OOD} \\\\\n"
             "\\rule[-2.0pt]{0pt}{8.6pt}手法 & AUROC$\\uparrow$ & FPR95$\\downarrow$ & AUROC$\\uparrow$ & FPR95$\\downarrow$ \\\\\n\\hline\n" + body + TAIL)
        (sec / "t18_in200.tex").write_text(t)
        for key, tag in ((ALONE, "main"), (f"{LP}|none", "lp"), (f"{STATIC}|none", "static"), ("knn|none", "knn"), ("lgcmax|none", "lgc"), (f"{V1}|none", "v1"), (f"xfit|bh|{QK}|pp|pooled|none", "bh"), (f"xfit|{RULE}|{QK}|pps|pooled|none", "pps"), (PF_ALONE, "pf")):
            for g in ("near", "far"):
                e = I2[g].get(key)
                if e:
                    N[f"in200_{tag}_{g}_AUROC"], N[f"in200_{tag}_{g}_FPR95"] = e["AUROC"], e["FPR95"]
                    if e["FPR95_ci"].get("lo") is not None:
                        N[f"in200_{tag}_{g}_FPR95_lo"], N[f"in200_{tag}_{g}_FPR95_hi"] = e["FPR95_ci"]["lo"], e["FPR95_ci"]["hi"]
        for g in ("near", "far"):
            e = I2["validity"][g].get(CFG)
            if e:
                N[f"in200_val_{g}_fpr005"], N[f"in200_val_{g}_fpr005_max"], N[f"in200_val_{g}_bhfdp"], N[f"in200_val_{g}_bhfdp_max"] = e["fpr@0.05"], e["fpr@0.05_max"], e["bh_fdp"], e["bh_fdp_max"]
        for d in OO_NEAR + OO_FAR:
            for rule in ("storey", "bh"):
                e = I2["seeds"].get(f"{d}|D3L|{rule}|{QK}")
                if e:
                    N[f"in200_seed_{d}_{rule}_pi0"], N[f"in200_seed_{d}_{rule}_idfrac"], N[f"in200_seed_{d}_{rule}_ratio"] = e["pi0_hat"], (e["id_frac"] if e["id_frac"] is not None else 0.0), e["seeds_over_n_test"]
            e = I2["per_dataset"].get(f"{ALONE}|{d}")
            if e:
                N[f"in200_main_{d}_AUROC"], N[f"in200_main_{d}_FPR95"] = e["AUROC"], e["FPR95"]
        for d, sz in (I2.get("sizes") or {}).items():
            N[f"in200_mix_{d}"] = 100.0 * sz["n_ood"] / (sz["n_id"] + sz["n_ood"])
        P200 = {k: v for k, v in PUB200.items() if not k.startswith("_")}
        N["in200_best_pub_near_AUROC"] = max(e["near"][0] for e in P200.values())
        N["in200_best_pub_far_AUROC"] = max(e["far"][0] for e in P200.values())
        N["in200_best_pub_near_FPR95"] = min(e["near"][1] for e in P200.values() if e["near"][1] is not None)
        N["in200_best_pub_far_FPR95"] = min(e["far"][1] for e in P200.values() if e["far"][1] is not None)
        if B15:
            for m in BASE15:
                for g in ("near", "far"):
                    if m in B15[g]:
                        N[f"in200_{m}_{g}_AUROC"], N[f"in200_{m}_{g}_FPR95"] = B15[g][m]["AUROC"], B15[g][m]["FPR95"]
            N["in200_n_train"] = B15["n_train"]
        if C15:
            for rule in CAPJA:
                for g in ("near", "far"):
                    e = C15[g].get(f"xfit|{rule}|{QK}|pp|pooled|none")
                    if e:
                        N[f"in200_{rule}_{g}_AUROC"], N[f"in200_{rule}_{g}_FPR95"] = e["AUROC"], e["FPR95"]
                        pe = C15.get(f"paired_{g}", {}).get(f"xfit|{rule}|{QK}|pp|pooled|none", {})
                        for ref in ("lp", "storey"):
                            if f"dFPR95_vs_{ref}" in pe:
                                d = pe[f"dFPR95_vs_{ref}"]
                                N[f"in200_{rule}_{g}_dFPR95_vs_{ref}"], N[f"in200_{rule}_{g}_dFPR95_vs_{ref}_lo"], N[f"in200_{rule}_{g}_dFPR95_vs_{ref}_hi"] = d["mean"], d["lo"], d["hi"]
                for d in OO_NEAR + OO_FAR:
                    e = C15["seeds"].get(f"{d}|D3L|{rule}|{QK}")
                    if e:
                        N[f"in200_seed_{d}_{rule}_idfrac"], N[f"in200_seed_{d}_{rule}_ratio"] = (e["id_frac"] if e["id_frac"] is not None else 0.0), e["seeds_over_n_test"]
        N["in200_n_draws"] = I2["n_draws"]
    # ------------------------------------------------------------------ backbone families (appendix table + numbers)
    VW = S14.get("views")
    if VW:
        VJA = {"B14": "DINOv2 B/14", "L14": "DINOv2 L/14", "D3B": "DINOv3 B/16", "D3L": "DINOv3 L/16", "CLIP": "CLIP ViT-B/16（画像埋め込み）", "DINO1": "DINO (v1) ViT-B/16"}
        rows = []
        for v in ["B14", "L14", "D3B", "D3L", "DINO1", "CLIP"]:
            if v not in VW:
                continue
            e = VW[v]
            def g4(key):
                a, b = e["near"].get(key), e["far"].get(key)
                return None if (a is None or b is None) else [a["AUROC"], a["FPR95"], b["AUROC"], b["FPR95"]]
            VC = (S15.get("views_c") or {}).get(v)
            def c4(key):
                if not VC:
                    return None
                a, b = VC["near"].get(key), VC["far"].get(key)
                return None if (a is None or b is None) else [a["AUROC"], a["FPR95"], b["AUROC"], b["FPR95"]]
            vals = [g4(f"{STATIC}|none"), c4("static|centred|none"), g4("knn|none"), c4("knn|centred|none"), g4("lgcmax|none"), g4(f"{LP}|none"), g4(f"{V1}|none"), g4(ALONE), g4(MAIN)]
            for key, tag in (("static|centred|none", "staticc"), ("knn|centred|none", "knnc"), ("static|raw|none", "staticraw"), ("knn|raw|none", "knnraw")):
                x = c4(key)
                if x:
                    N[f"view_{v}_{tag}_near_AUROC"], N[f"view_{v}_{tag}_near_FPR95"], N[f"view_{v}_{tag}_far_AUROC"], N[f"view_{v}_{tag}_far_FPR95"] = x
            for key, tag in ((f"{STATIC}|none", "static"), ("knn|none", "knn"), ("lgcmax|none", "lgc"), (f"{LP}|none", "lp"), (f"{V1}|none", "v1"), (ALONE, "main"), (MAIN, "mainT"), (PF_ALONE, "pf")):
                x = g4(key)
                if x:
                    N[f"view_{v}_{tag}_near_AUROC"], N[f"view_{v}_{tag}_near_FPR95"], N[f"view_{v}_{tag}_far_AUROC"], N[f"view_{v}_{tag}_far_FPR95"] = x
            cells = []
            for x in vals:
                cells += (["--"] * 2 if x is None else [f"{x[1]:.2f}", f"{x[3]:.2f}"])
            rows.append(f"{VJA[v]}（{VW['dim'][v]}次元） & " + " & ".join(cells) + " \\\\")
        t = ("{\\fontsize{6.0pt}{7.1pt}\\selectfont\\renewcommand{\\arraystretch}{1.0}%\n\\setlength{\\tabcolsep}{1.3pt}\\begin{tabular}{@{}lcc|cc|cc|cc|cc|cc|cc|cc|cc@{}}\n\\hline\n"
             "\\rule[-2.0pt]{0pt}{8.2pt} & \\multicolumn{2}{c|}{静的$p$} & \\multicolumn{2}{c|}{静的$p$，中心化} & \\multicolumn{2}{c|}{kNN} & \\multicolumn{2}{c|}{kNN，中心化} & \\multicolumn{2}{c|}{LGC-max} & \\multicolumn{2}{c|}{$\\pP$のみ} & \\multicolumn{2}{c|}{$\\pP\\pN$（交差なし）} & \\multicolumn{2}{c|}{$\\sigma$（本稿）} & \\multicolumn{2}{c}{$\\sigma\\times$TINS} \\\\\n"
             "\\rule[-2.0pt]{0pt}{8.2pt}ビュー & 近 & 遠 & 近 & 遠 & 近 & 遠 & 近 & 遠 & 近 & 遠 & 近 & 遠 & 近 & 遠 & 近 & 遠 & 近 & 遠 \\\\\n\\hline\n" + "\n".join(rows) + TAIL)
        (sec / "t18_views.tex").write_text(t)
        # every single view: p_fin at or below p+ only and below the static rank (near)?
        cone_all = json.load(open(HERE / "data" / "clip_cone.json"))
        cone = cone_all["CLIP"]["mean_cos_all_protos"]
        N["view_CLIP_mean_cos_id"], N["view_CLIP_mean_cos_ood"] = cone["id"], cone["textures"]     # data/clip_cone.json (bridge job 701)
        for vv, tag in (("CLIP", "CLIP"), ("L14", "L14"), ("DINO1", "DINO1")):
            mp = cone_all[vv]["max_proto_cos"]
            N[f"view_{tag}_max_cos_id_mean"], N[f"view_{tag}_max_cos_id_std"] = mp["id"]
            N[f"view_{tag}_max_cos_ood_mean"], N[f"view_{tag}_max_cos_ood_std"] = mp["textures"]
        # the ID threshold at 95 % acceptance under a normal approximation and the fraction of Textures above it (illustration only)
        import math
        mu, sd = cone_all["CLIP"]["max_proto_cos"]["id"]; mo, so = cone_all["CLIP"]["max_proto_cos"]["textures"]
        thr = mu - 1.645 * sd
        N["view_CLIP_max_cos_thr95"] = thr
        N["view_CLIP_textures_above_thr95_pct"] = 100.0 * 0.5 * (1 + math.erf((mo - thr) / (so * math.sqrt(2))))
        N["views_n"] = len([v for v in VW["views"]])
        VCall = S15.get("views_c") or {}
        dif = [abs(VCall[v][g]["static|raw|none"]["FPR95"] - VW[v][g][f"{STATIC}|none"]["FPR95"]) for v in VW["views"] if v in VCall for g in ("near", "far")]
        if dif:
            N["views_raw_vs_table_maxabs"] = max(dif)
            for v in VW["views"]:
                if v in VCall:
                    for g in ("near", "far"):
                        N[f"view_{v}_centred_minus_raw_{g}_FPR95"] = VCall[v][g]["static|centred|none"]["FPR95"] - VCall[v][g]["static|raw|none"]["FPR95"]
                        N[f"view_{v}_knncentred_minus_raw_{g}_FPR95"] = VCall[v][g]["knn|centred|none"]["FPR95"] - VCall[v][g]["knn|raw|none"]["FPR95"]
            dino = [v for v in VW["views"] if v in VCall and v != "CLIP"]
            N["views_dino_centred_maxabs_FPR95"] = max(abs(N[f"view_{v}_centred_minus_raw_{g}_FPR95"]) for v in dino for g in ("near", "far"))
            N["views_dino_knncentred_maxabs_FPR95"] = max(abs(N[f"view_{v}_knncentred_minus_raw_{g}_FPR95"]) for v in dino for g in ("near", "far"))
            N["views_support_mean_norm_CLIP"] = (VCall.get("support_mean_norm") or {}).get("CLIP", 0.0)
            N["views_support_mean_norm_dino_min"] = min(x for k, x in (VCall.get("support_mean_norm") or {}).items() if k != "CLIP") if VCall.get("support_mean_norm") else 0.0
            N["views_support_mean_norm_dino_max"] = max(x for k, x in (VCall.get("support_mean_norm") or {}).items() if k != "CLIP") if VCall.get("support_mean_norm") else 0.0
        N["views_n_main_below_lp_near"] = sum(1 for v in VW["views"] if VW[v]["near"][ALONE]["FPR95"] < VW[v]["near"][f"{LP}|none"]["FPR95"])
        N["views_n_main_below_static_near"] = sum(1 for v in VW["views"] if VW[v]["near"][ALONE]["FPR95"] < VW[v]["near"][f"{STATIC}|none"]["FPR95"])
        N["views_n_main_below_lgc_near"] = sum(1 for v in VW["views"] if VW[v]["near"][ALONE]["FPR95"] < VW[v]["near"]["lgcmax|none"]["FPR95"])
        N["views_n_main_le_v1_near"] = sum(1 for v in VW["views"] if VW[v]["near"][ALONE]["FPR95"] <= VW[v]["near"][f"{V1}|none"]["FPR95"] + 0.005)
        N["views_max_main_minus_v1_near"] = max(VW[v]["near"][ALONE]["FPR95"] - VW[v]["near"][f"{V1}|none"]["FPR95"] for v in VW["views"])
        N["views_min_main_minus_v1_near"] = min(VW[v]["near"][ALONE]["FPR95"] - VW[v]["near"][f"{V1}|none"]["FPR95"] for v in VW["views"])
    # ------------------------------------------------------------------ CIFAR-100
    CF = S14.get("cifar")
    if CF:
        PUBC = {k: v for k, v in PUB.get("cifar100", {}).items() if not k.startswith("_")}
        rows = []
        rows.append(("\\multicolumn{5}{@{}l}{\\rule{0pt}{6.6pt}\\textit{CIFAR-100で学習したResNet-18（報告値$^{\\dagger}$）}} \\\\", None, False))
        for name in ("MSP", "MLS", "RMDS", "VOS", "G-ODIN", "RotPred", "OE"):
            e = PUBC[name]
            rows.append((short(e["label"]) + ("（外れ値データで学習）" if e["group"] == "train_outlier" else ""), [e["near"][0], e["near"][1], e["far"][0], e["far"][1]], False))
        B15C = (S15.get("base") or {}).get("cifar100")
        if B15C:
            rows.append(("\\multicolumn{5}{@{}l}{\\rule{0pt}{6.6pt}\\textit{全学習画像（" + f"{B15C['n_train']:,}" + "枚）またはゼロショット（本稿で実測）}} \\\\", None, False))
            for m, lab in BASE15.items():
                if m in B15C["near"]:
                    rows.append((lab, [B15C["near"][m]["AUROC"], B15C["near"][m]["FPR95"], B15C["far"][m]["AUROC"], B15C["far"][m]["FPR95"]], False))
        rows.append(("\\multicolumn{5}{@{}l}{\\rule{0pt}{6.6pt}\\textit{凍結特徴，クラスあたり16枚，学習なし，単体（6通りの抽出の平均）}} \\\\", None, False))
        def c4(key):
            a, b = CF["near"].get(key), CF["far"].get(key)
            return None if (a is None or b is None) else [a["AUROC"], a["FPR95"], b["AUROC"], b["FPR95"]]
        for key, lab, ours in ((f"L14+D3L|{STATIC}|none", "静的$p$（グラフなし）", False), ("L14+D3L|knn|none", "kNN（グラフなし）", False), ("L14+D3L|lgcmax|none", "LGC-max", False),
                               (f"L14+D3L|{LP}|none", "$\\pP$のみ", True), (f"L14+D3L|{V1}|none", "$\\pP\\pN$，交差適合なし", True), (f"L14+D3L|{ALONE}", "$\\sigma$（本稿の構成）", True),
                               (f"L14+D3L|xfit|{RULE}|{QK}|pps|pooled|none", "$\\sigma$に静的$p$を加える", True), (f"D3L|{ALONE}", "$\\sigma$，DINOv3 L/16のみ", True)):
            x = c4(key)
            if x:
                rows.append((lab, x, ours))
        C15C = S15.get("cap_cifar")
        if C15C:
            for rule, lab in CAPJA.items():
                key = f"xfit|{rule}|{QK}|pp|pooled|none"
                if key in C15C["near"]:
                    rows.append((lab, [C15C["near"][key]["AUROC"], C15C["near"][key]["FPR95"], C15C["far"][key]["AUROC"], C15C["far"][key]["FPR95"]], True))
        body = fmt_table(rows, 4, [f2] * 4)
        t = ("{\\fontsize{6.8pt}{8pt}\\selectfont\\renewcommand{\\arraystretch}{1.0}%\n\\setlength{\\tabcolsep}{2pt}\\begin{tabular}{@{}lcccc@{}}\n\\hline\n"
             "\\rule[-2.0pt]{0pt}{8.6pt} & \\multicolumn{2}{c}{near-OOD} & \\multicolumn{2}{c}{far-OOD} \\\\\n"
             "\\rule[-2.0pt]{0pt}{8.6pt}手法 & AUROC$\\uparrow$ & FPR95$\\downarrow$ & AUROC$\\uparrow$ & FPR95$\\downarrow$ \\\\\n\\hline\n" + body + TAIL)
        (sec / "t18_cifar.tex").write_text(t)
        for key, tag in ((f"L14+D3L|{ALONE}", "main"), (f"L14+D3L|{LP}|none", "lp"), (f"L14+D3L|{STATIC}|none", "static"), ("L14+D3L|knn|none", "knn"), ("L14+D3L|lgcmax|none", "lgc"),
                         (f"L14+D3L|{V1}|none", "v1"), (f"L14+D3L|xfit|{RULE}|{QK}|pps|pooled|none", "pps"), (f"D3L|{ALONE}", "d3l"), (f"L14|{ALONE}", "l14"), (f"L14+D3L|{PF_ALONE}", "pf")):
            for g in ("near", "far"):
                e = CF[g].get(key)
                if e:
                    N[f"cifar_{tag}_{g}_AUROC"], N[f"cifar_{tag}_{g}_FPR95"] = e["AUROC"], e["FPR95"]
                    if e.get("FPR95_ci", {}).get("lo") is not None:
                        N[f"cifar_{tag}_{g}_FPR95_lo"], N[f"cifar_{tag}_{g}_FPR95_hi"] = e["FPR95_ci"]["lo"], e["FPR95_ci"]["hi"]
                    for ref in ("v1", "lp"):
                        d = e.get(f"dFPR95_vs_{ref}")
                        if d:
                            N[f"cifar_{tag}_{g}_dFPR95_vs_{ref}"], N[f"cifar_{tag}_{g}_dFPR95_vs_{ref}_lo"], N[f"cifar_{tag}_{g}_dFPR95_vs_{ref}_hi"] = d["mean"], d["lo"], d["hi"]
        for d in CIFAR_NEAR + CIFAR_FAR:
            e = CF["per_dataset"].get(f"L14+D3L|{ALONE}|{d}")
            if e:
                N[f"cifar_main_{d}_AUROC"], N[f"cifar_main_{d}_FPR95"] = e["AUROC"], e["FPR95"]
        for g in ("near", "far"):
            e = CF["validity"][g].get(f"L14+D3L|{CFG}")
            if e:
                N[f"cifar_val_{g}_fpr005"], N[f"cifar_val_{g}_fpr005_max"], N[f"cifar_val_{g}_bhfdp"], N[f"cifar_val_{g}_bhfdp_max"], N[f"cifar_val_{g}_recall"] = e["fpr@0.05"], e["fpr@0.05_max"], e["bh_fdp"], e["bh_fdp_max"], e["bh_recall"]
        for g in ("near", "far"):
            best = max(PUBC.items(), key=lambda kv: kv[1][g][0])
            N[f"cifar_best_pub_{g}_AUROC"], N[f"cifar_best_pub_{g}_label"] = best[1][g][0], best[0]
            N[f"cifar_main_vs_best_pub_{g}_AUROC"] = N[f"cifar_main_{g}_AUROC"] - best[1][g][0]
            no_out = max((kv for kv in PUBC.items() if kv[1]["group"] != "train_outlier"), key=lambda kv: kv[1][g][0])
            N[f"cifar_best_pub_noout_{g}_AUROC"], N[f"cifar_best_pub_noout_{g}_label"] = no_out[1][g][0], no_out[0]
        N["cifar_n_draws"] = CF["n_draws"]
        for g in ("near", "far"):
            bestf = min(PUBC.items(), key=lambda kv: kv[1][g][1] if kv[1][g][1] is not None else 1e9)
            N[f"cifar_best_pub_{g}_FPR95"], N[f"cifar_best_pub_{g}_FPR95_label"] = bestf[1][g][1], bestf[0]
        if B15C:
            for m in BASE15:
                for g in ("near", "far"):
                    if m in B15C[g]:
                        N[f"cifar_{m}_{g}_AUROC"], N[f"cifar_{m}_{g}_FPR95"] = B15C[g][m]["AUROC"], B15C[g][m]["FPR95"]
            N["cifar_n_train"] = B15C["n_train"]
        if C15C:
            for rule in CAPJA:
                for g in ("near", "far"):
                    key = f"xfit|{rule}|{QK}|pp|pooled|none"
                    e = C15C[g].get(key)
                    if e:
                        N[f"cifar_{rule}_{g}_AUROC"], N[f"cifar_{rule}_{g}_FPR95"] = e["AUROC"], e["FPR95"]
                        pe = C15C.get(f"paired_{g}", {}).get(key, {})
                        for ref in ("lp", "storey"):
                            if f"dFPR95_vs_{ref}" in pe:
                                d = pe[f"dFPR95_vs_{ref}"]
                                N[f"cifar_{rule}_{g}_dFPR95_vs_{ref}"], N[f"cifar_{rule}_{g}_dFPR95_vs_{ref}_lo"], N[f"cifar_{rule}_{g}_dFPR95_vs_{ref}_hi"] = d["mean"], d["lo"], d["hi"]
                for d in CIFAR_NEAR + CIFAR_FAR:
                    e = C15C["per_dataset"].get(f"xfit|{rule}|{QK}|pp|pooled|none|{d}")
                    if e:
                        N[f"cifar_{rule}_{d}_FPR95"] = e["FPR95"]
                    s = C15C["seeds"].get(f"{d}|D3L|{rule}|{QK}")
                    if s:
                        N[f"cifar_seed_{d}_{rule}_ratio"], N[f"cifar_seed_{d}_{rule}_idfrac"] = 100.0 * s["seeds_over_n_test"], s["id_frac"]
        # sizes (mixing rate in %) and seeds (DINOv3 L/16 view, q = 0.2, both folds, all draws): share of the fold selected and ID share among the seeds
        if CF.get("sizes") and CF.get("seeds"):
            for d, e in CF["sizes"].items():
                N[f"cifar_mix_{d}"] = 100.0 * e["mixing"]
                N[f"cifar_nood_{d}"] = e["n_ood"]
                for rule in ("storey", "bh"):
                    s = CF["seeds"][f"{d}|D3L|{rule}|q0.2"]
                    N[f"cifar_seed_{d}_{rule}_ratio"], N[f"cifar_seed_{d}_{rule}_idfrac"], N[f"cifar_seed_{d}_{rule}_pi0"] = 100.0 * s["seeds_over_n_test"], s["id_frac"], s["pi0_hat"]
            far_hi = [d for d in CIFAR_FAR if CF["sizes"][d]["mixing"] > 0.6]                 # the three far sets where the unknowns dominate
            N["cifar_mix_far_hi_min"], N["cifar_mix_far_hi_max"] = min(N[f"cifar_mix_{d}"] for d in far_hi), max(N[f"cifar_mix_{d}"] for d in far_hi)
            N["cifar_seed_far_hi_storey_ratio_min"], N["cifar_seed_far_hi_storey_ratio_max"] = min(N[f"cifar_seed_{d}_storey_ratio"] for d in far_hi), max(N[f"cifar_seed_{d}_storey_ratio"] for d in far_hi)
            N["cifar_seed_storey_idfrac_min"], N["cifar_seed_storey_idfrac_max"] = min(N[f"cifar_seed_{d}_storey_idfrac"] for d in CIFAR_NEAR + CIFAR_FAR), max(N[f"cifar_seed_{d}_storey_idfrac"] for d in CIFAR_NEAR + CIFAR_FAR)
            N["cifar_seed_near_storey_ratio_min"], N["cifar_seed_near_storey_ratio_max"] = min(N[f"cifar_seed_{d}_storey_ratio"] for d in CIFAR_NEAR), max(N[f"cifar_seed_{d}_storey_ratio"] for d in CIFAR_NEAR)
            N["cifar_n_far_hi"] = len(far_hi)
        for d in CIFAR_NEAR + CIFAR_FAR:                                                          # per-data-set FPR95 of p+ only, BH seeds, static and LGC-max
            for key, tag in ((f"L14+D3L|{LP}|none", "lp"), (f"L14+D3L|xfit|bh|{QK}|pp|pooled|none", "bh"), (f"L14+D3L|{STATIC}|none", "static"), ("L14+D3L|lgcmax|none", "lgc")):
                e = CF["per_dataset"].get(f"{key}|{d}")
                if e:
                    N[f"cifar_{tag}_{d}_AUROC"], N[f"cifar_{tag}_{d}_FPR95"] = e["AUROC"], e["FPR95"]
        for g in ("near", "far"):
            e = CF[g].get(f"L14+D3L|xfit|bh|{QK}|pp|pooled|none")
            if e:
                N[f"cifar_bh_{g}_AUROC"], N[f"cifar_bh_{g}_FPR95"] = e["AUROC"], e["FPR95"]
    # ------------------------------------------------------------------ selection: Phase 12 (views, q) and Phase 14 (rule, q, mode)
    sel12 = S12["selection"]["_selection"]
    N["sel_n_candidates"], N["sel_n_feasible"] = sel12["n_candidates"], sel12["n_feasible"]
    N["sel_dev1_near_FPR95"], N["sel_dev2_near_FPR95"] = sel12["chosen_dev1"]["near"]["FPR95"], sel12["chosen_dev2"]["near"]["FPR95"]
    d2 = {k: v["near"]["FPR95"] for k, v in S12["selection"]["dev2"].items() if "|T|" in k and k.endswith("|none")}
    order2 = sorted(d2, key=d2.get)
    N["sel_dev2_rank"] = order2.index(sel12["chosen"]) + 1
    N["sel_dev2_joint_near_FPR95"] = d2["L14xD3L+D3L|T|q0.2|none"]
    s14 = S14["selection"]
    N["selq_n_candidates"], N["selq_n_feasible"] = s14["n_candidates"], s14["n_feasible"]
    N["selq_dev1_near_FPR95"], N["selq_dev1_far_FPR95"] = s14["chosen_dev1"]["near"]["FPR95"], s14["chosen_dev1"]["far"]["FPR95"]
    N["selq_dev2_near_FPR95"], N["selq_dev2_far_FPR95"] = s14["chosen_dev2"]["near"]["FPR95"], s14["chosen_dev2"]["far"]["FPR95"]
    N["selq_ref_dev1_near_FPR95"], N["selq_ref_dev1_far_FPR95"] = s14["reference"]["near"]["FPR95"], s14["reference"]["far"]["FPR95"]
    N["selq_dev1_near_minus_ref"] = N["selq_dev1_near_FPR95"] - N["selq_ref_dev1_near_FPR95"]
    rank = s14["ranking_dev1_near"]
    N["selq_dev1_rank_of_chosen"] = [c for c, _, _ in rank].index(SEL) + 1
    N["selq_dev1_best_near_FPR95"], N["selq_dev1_best_label"] = rank[0][1], rank[0][0]
    if "selection_dev" in S14:                                                                     # full dev tables when available
        pass
    # Phase 14 dev table (appendix) from sel14_summary if present
    SEL14 = None
    f = HERE / "data" / "sel14_summary.json"
    if f.exists():
        SEL14 = json.load(open(f))
        rows = []
        MJA = {"pp": "$\\pP\\pN$", "pps": "$\\pP\\pN\\times$静的$p$", "min": "$\\min(\\pP\\pN,$静的$p)$"}
        RJA = {"storey": "Storey型", "bh": "単純BH"}
        for rule in ("storey", "bh"):
            for q in ("0.2", "0.1", "0.05"):
                for mode in ("pp", "pps", "min"):
                    c = f"xfit|{rule}|q{q}|{mode}|none"
                    d1, d2_ = SEL14["dev1"].get(c), SEL14["dev2"].get(c)
                    if not d1:
                        continue
                    v = [d1["near"]["AUROC"], d1["near"]["FPR95"], d1["far"]["AUROC"], d1["far"]["FPR95"], d2_["near"]["AUROC"], d2_["near"]["FPR95"], d2_["far"]["AUROC"], d2_["far"]["FPR95"]]
                    rows.append((f"{RJA[rule]}，$q{{=}}{q}$，{MJA[mode]}" + ("（選択）" if c == SEL else ""), v, False))
        for c, lab in ((f"{V1}|none", "参照：$\\pP\\pN$，交差適合なし（Storey型，$q{=}0.2$）"), ("xfit|lp|none", "$\\pP$のみ"), ("xfit|static|none", "静的$p$"), ("knn|none", "kNN"), ("lgcmax|none", "LGC-max"), (f"{SC}|none", "選択した構成の$\\sigma$（順位付けの統計量）"), (f"{SC}|TINS", "選択した構成の$\\sigma\\times$TINS"), (f"{CFG}|TINS", "選択した構成の$\\pfin\\times$TINS")):
            d1, d2_ = SEL14["dev1"].get(c), SEL14["dev2"].get(c)
            if d1:
                rows.append((lab, [d1["near"]["AUROC"], d1["near"]["FPR95"], d1["far"]["AUROC"], d1["far"]["FPR95"], d2_["near"]["AUROC"], d2_["near"]["FPR95"], d2_["far"]["AUROC"], d2_["far"]["FPR95"]], False))
        body = fmt_table(rows, 8, [f2] * 8)
        t = ("{\\fontsize{6.6pt}{7.8pt}\\selectfont\\renewcommand{\\arraystretch}{1.0}%\n\\setlength{\\tabcolsep}{2.0pt}\\begin{tabular}{@{}lcccccccc@{}}\n\\hline\n"
             "\\rule[-2.0pt]{0pt}{8.6pt} & \\multicolumn{4}{c}{dev1（選択）} & \\multicolumn{4}{c}{dev2（確認）} \\\\\n"
             "\\rule[-2.0pt]{0pt}{8.6pt} & \\multicolumn{2}{c}{near} & \\multicolumn{2}{c}{far} & \\multicolumn{2}{c}{near} & \\multicolumn{2}{c}{far} \\\\\n"
             "\\rule[-2.0pt]{0pt}{8.6pt}候補（$\\pfin$，単体） & AUROC & FPR95 & AUROC & FPR95 & AUROC & FPR95 & AUROC & FPR95 \\\\\n\\hline\n" + body + TAIL)
        (sec / "t18_sel14.tex").write_text(t)
        v = SEL14["dev1"].get(SEL, {}).get("validity")
        if v:
            N["selq_dev1_fpr005"], N["selq_dev1_bhfdp"] = v["fpr@0.05"], v["bh_fdp"]
        N["selq_dev1_lgc_near_FPR95"] = SEL14["dev1"]["lgcmax|none"]["near"]["FPR95"]
        N["selq_dev1_pps_near_FPR95"], N["selq_dev1_pps_far_FPR95"] = SEL14["dev1"][f"xfit|{RULE}|{QK}|pps|none"]["near"]["FPR95"], SEL14["dev1"][f"xfit|{RULE}|{QK}|pps|none"]["far"]["FPR95"]
    # ------------------------------------------------------------------ appendix: Phase 12 dev-split selection table (views x q) and the test-set exploration history
    VIEWS_JA = {"L14": "DINOv2 L/14", "D3L": "DINOv3 L/16", "L14xD3L": "連結ビュー", "L14+D3L": "DINOv2 L/14 + DINOv3 L/16", "L14xD3L+D3L": "連結ビュー + DINOv3 L/16",
                "D3B+D3L": "DINOv3 B/16 + L/16", "B14+L14": "DINOv2 B/14 + L/14", "L14+D3B+D3L": "DINOv2 L/14 + DINOv3 B/16 + L/16"}
    selall = S12["selection"]
    rows = []
    for vs in VIEWS_JA:
        for q in ("0.05", "0.1", "0.2"):
            c = f"{vs}|T|q{q}|none"
            d1, d2_ = selall["dev1"].get(c), selall["dev2"].get(c)
            if not d1:
                continue
            v = [d1["near"]["AUROC"], d1["near"]["FPR95"], d1["far"]["AUROC"], d1["far"]["FPR95"], d2_["near"]["AUROC"], d2_["near"]["FPR95"], d2_["far"]["AUROC"], d2_["far"]["FPR95"]]
            rows.append((f"{VIEWS_JA[vs]}，$q{{=}}{q}$" + ("（選択）" if c == sel12["chosen"] else ""), v, False))
    body = fmt_table(rows, 8, [f2] * 8)
    t = ("{\\fontsize{6.6pt}{7.8pt}\\selectfont\\renewcommand{\\arraystretch}{1.0}%\n\\setlength{\\tabcolsep}{2.0pt}\\begin{tabular}{@{}lcccccccc@{}}\n\\hline\n"
         "\\rule[-2.0pt]{0pt}{8.6pt} & \\multicolumn{4}{c}{dev1（選択）} & \\multicolumn{4}{c}{dev2（確認）} \\\\\n"
         "\\rule[-2.0pt]{0pt}{8.6pt} & \\multicolumn{2}{c}{near} & \\multicolumn{2}{c}{far} & \\multicolumn{2}{c}{near} & \\multicolumn{2}{c}{far} \\\\\n"
         "\\rule[-2.0pt]{0pt}{8.6pt}構成（$\\pP\\pN$，単体） & AUROC & FPR95 & AUROC & FPR95 & AUROC & FPR95 & AUROC & FPR95 \\\\\n\\hline\n" + body + TAIL)
    (sec / "t18_sel.tex").write_text(t)
    DR = S12["draws"]
    rows = []
    for vs in ("L14", "D3L", "L14xD3L", "L14+D3L", "L14xD3L+D3L"):
        for ro, q in (("Tq0.05", "0.05"), ("T", "0.1"), ("Tq0.2", "0.2")):
            for b in ("none", "TINS"):
                c = f"{vs}|{ro}|{b}"
                if c not in DR["openood"]["near"]:
                    continue
                vals = [DR["openood"]["near"][c]["AUROC"], DR["openood"]["near"][c]["FPR95"], DR["openood"]["far"][c]["AUROC"], DR["openood"]["far"][c]["FPR95"], DR["fourood"]["fourood"][c]["AUROC"], DR["fourood"]["fourood"][c]["FPR95"]]
                rows.append((f"{VIEWS_JA[vs]}，$q{{=}}{q}$，" + ("単体" if b == "none" else "$\\times$TINS"), vals, False))
    body = fmt_table(rows, 6, [f2] * 6)
    (sec / "t18_sens.tex").write_text(head6("構成（交差適合なしの$\\pP\\pN$）") + body + TAIL)
    # ------------------------------------------------------------------ appendix: arrival-time mode per data set (cross-fit; and without cross-fit from Phase 13)
    O14, O13 = S14.get("online"), S13.get("online")
    if O14 and "openood" in O14 and "fourood" in O14:
        rows = []
        for key, lab in ((f"xonl|{RULE}|{QK}|{MODE}|pooled|TINS", "$\\sigma\\times$TINS"), (f"xonl|{RULE}|{QK}|{MODE}|pooled|none", "$\\sigma$，単体")):
            pdd, pff = O14["openood"]["per_dataset"], O14["fourood"]["per_dataset"]
            if f"{key}|ssb_hard" in pdd:
                rows.append((lab, sum(([pdd[f"{key}|{d}"]["AUROC"], pdd[f"{key}|{d}"]["FPR95"]] for d in OO_NEAR + OO_FAR), []) + sum(([pff[f"{key}|{d}"]["AUROC"], pff[f"{key}|{d}"]["FPR95"]] for d in FOUR), []), False))
        if O13 and "openood" in O13:
            for c, lab in (("L14+D3L|T|TINS", "$\\pP\\pN\\times$TINS，交差適合なし"), ("L14+D3L|T|none", "$\\pP\\pN$，単体，交差適合なし")):
                pdd, pff = O13["openood"]["per_dataset"], O13["fourood"]["per_dataset"]
                if f"{c}|ssb_hard" in pdd:
                    rows.append((lab, sum(([pdd[f"{c}|{d}"]["AUROC"], pdd[f"{c}|{d}"]["FPR95"]] for d in OO_NEAR + OO_FAR), []) + sum(([pff[f"{c}|{d}"]["AUROC"], pff[f"{c}|{d}"]["FPR95"]] for d in FOUR), []), False))
        body = fmt_table(rows, 18, [f2] * 18)
        t = ("{\\fontsize{5.8pt}{6.9pt}\\selectfont\\renewcommand{\\arraystretch}{1.0}%\n\\setlength{\\tabcolsep}{0.5pt}\\begin{tabular}{@{}l" + "c" * 18 + "@{}}\n\\hline\n"
             "\\rule[-2.0pt]{0pt}{8.6pt} & \\multicolumn{2}{c}{SSB-hard} & \\multicolumn{2}{c}{NINCO} & \\multicolumn{2}{c}{iNat.} & \\multicolumn{2}{c}{Textures} & \\multicolumn{2}{c}{OImO} & \\multicolumn{2}{c}{iNat.} & \\multicolumn{2}{c}{SUN} & \\multicolumn{2}{c}{Places} & \\multicolumn{2}{c}{Textures} \\\\\n\\hline\n" + body + TAIL)
        (sec / "t18_online.tex").write_text(t)
        for part, g, grp in (("openood", "near", "near"), ("openood", "far", "far"), ("fourood", "fourood", "four")):
            e = O14[part]["validity"][g].get(f"xonl|{RULE}|{QK}|{MODE}")
            if e:
                N[f"onl14_val_{grp}_fpr005"], N[f"onl14_val_{grp}_fpr005_max"], N[f"onl14_val_{grp}_bhfdp"] = e["fpr@0.05"], e["fpr@0.05_max"], e["bh_fdp"]
    # ------------------------------------------------------------------ Phase 15: k_g grid, capped seeds on the grid and on dev
    KG = S15.get("kg")
    if KG:
        for key, e in KG.items():
            ds, c, r, kg = key.split("|")
            cf = e["cfg"]
            tag = f"kg_{ds}_{kg}_{r}"
            N[f"{tag}_main_FPR95"], N[f"{tag}_lp_FPR95"], N[f"{tag}_static_FPR95"] = cf[ALONE]["FPR95"], cf[f"{LP}|none"]["FPR95"], cf[f"{STATIC}|none"]["FPR95"]
        for ds in ("ninco", "ssb_hard"):
            for kg in S15["kgs"]:
                rs = sorted(int(k.split("|")[2][1:]) for k in KG if k.startswith(f"{ds}|") and k.endswith(f"|kg{kg}"))
                cross = next((r for r in rs if N[f"kg_{ds}_kg{kg}_r{r}_main_FPR95"] < N[f"kg_{ds}_kg{kg}_r{r}_static_FPR95"]), 0)
                N[f"kg_{ds}_kg{kg}_cross_r"] = cross
                N[f"kg_{ds}_kg{kg}_rmax_main_FPR95"] = N[f"kg_{ds}_kg{kg}_r{rs[-1]}_main_FPR95"]
                N[f"kg_{ds}_kg{kg}_r1_main_FPR95"] = N[f"kg_{ds}_kg{kg}_r1_main_FPR95"]
        for ds in ("ninco", "ssb_hard"):
            rs = sorted({int(k.split("|")[2][1:]) for k in KG if k.startswith(f"{ds}|")})
            rmax = rs[-1]
            for r, tagr in ((1, "r1"), (rmax, "rmax")):
                best = min(S15["kgs"], key=lambda kg: N[f"kg_{ds}_kg{kg}_r{r}_main_FPR95"])
                N[f"kg_{ds}_best_kg_{tagr}"] = best
                N[f"kg_{ds}_kg10_minus_best_{tagr}_FPR95"] = N[f"kg_{ds}_kg10_r{r}_main_FPR95"] - N[f"kg_{ds}_kg{best}_r{r}_main_FPR95"]
            N[f"kg_{ds}_rmax"] = rmax
            # the smallest r at which sigma is at least 10 points below the static rank, per k_g (0 = never)
            for kg in S15["kgs"]:
                N[f"kg_{ds}_kg{kg}_onset_r"] = next((r for r in rs if N[f"kg_{ds}_kg{kg}_r{r}_main_FPR95"] <= N[f"kg_{ds}_kg{kg}_r{r}_static_FPR95"] - 10.0), 0)
        N["kg_n_reps"] = next(iter(KG.values()))["n_reps"]
    CR = S15.get("cap_rec")
    if CR:
        rows = []
        for ds in ("ninco", "ssb_hard"):
            rows.append((f"\\multicolumn{{10}}{{@{{}}l}}{{\\rule{{0pt}}{{6.6pt}}\\textit{{{'NINCO' if ds == 'ninco' else 'SSB-hard'}}}}} \\\\", None, False))
            for key in sorted((k for k in CR if k.startswith(ds + "|")), key=lambda k: (int(k.split("|")[1][1:]), int(k.split("|")[2][1:]))):
                e = CR[key]; cf = e["cfg"]; _, c, r = key.split("|")
                sc = e.get("view_D3L", {}).get("storeyc", {})
                vals = [cf[k]["FPR95"] for k in (f"xfit|storey|{QK}|pp|pooled|none", f"xfit|storeyc|{QK}|pp|pooled|none", f"xfit|bh|{QK}|pp|pooled|none", f"xfit|bhc|{QK}|pp|pooled|none", f"{LP}|none", f"{STATIC}|none")]
                mr = 100 * e["mixing_rate"]
                rows.append((f"{c[1:]} & {r[1:]} & {mr:.1f}" if mr < 10 else f"{c[1:]} & {r[1:]} & {mr:.0f}", vals + [sc.get("seeds"), (100 * sc["id_frac"]) if sc.get("id_frac") is not None else None], False))
                N[f"caprec_{ds}_{c}_{r}_mix"] = mr
                N[f"caprec_{ds}_{c}_{r}_storeyc_idfrac"] = sc.get("id_frac") if sc.get("id_frac") is not None else 0.0
                st = e.get("view_D3L", {}).get("storey", {})
                N[f"caprec_{ds}_{c}_{r}_storey_idfrac"] = st.get("id_frac") if st.get("id_frac") is not None else 0.0
                for rl in ("storey", "storeyc", "bh", "bhc"):
                    N[f"caprec_{ds}_{c}_{r}_{rl}_FPR95"] = cf[f"xfit|{rl}|{QK}|pp|pooled|none"]["FPR95"]
                N[f"caprec_{ds}_{c}_{r}_lp_FPR95"] = cf[f"{LP}|none"]["FPR95"]
        lines = []
        for lab, v, _ in rows:
            if v is None:
                lines.append(lab)
            else:
                lines.append(lab + " & " + " & ".join(["--" if x is None else (f"{x:.1f}" if i < 6 else (f"{x:.0f}" if i == 6 else f"{x:.1f}")) for i, x in enumerate(v)]) + " \\\\")
        tcap = ("{\\fontsize{6.3pt}{7.4pt}\\selectfont\\renewcommand{\\arraystretch}{1.0}%\n\\setlength{\\tabcolsep}{2.2pt}\\begin{tabular}{@{}rrr|cc|cc|c|c|cc@{}}\n\\hline\n"
                "\\rule[-2.0pt]{0pt}{8.2pt} & & & \\multicolumn{2}{c|}{Storey型} & \\multicolumn{2}{c|}{単純BH法} & & & \\multicolumn{2}{c}{種（上限付きStorey型）} \\\\\n"
                "\\rule[-2.0pt]{0pt}{8.2pt}$c$ & $r$ & 混入率 & 上限なし & $\\tau{=}0.05$ & 上限なし & $\\tau{=}0.05$ & $\\pP$のみ & 静的$p$ & 枚数 & ID \\\\\n\\hline\n" + "\n".join(lines) + TAIL)
        (sec / "t18_caprec.tex").write_text(tcap)
        def stats(rule, cond):
            cells = [e for e in CR.values() if cond(e)]
            d = [e["cfg"][f"xfit|{rule}|{QK}|pp|pooled|none"]["FPR95"] - e["cfg"][f"{LP}|none"]["FPR95"] for e in cells]
            return len(cells), sum(1 for x in d if x > 1.0), (max(d) if d else 0.0), (min(d) if d else 0.0)
        for rule in ("storey", "storeyc", "bh", "bhc"):
            for cname, cond in (("all", lambda e: True), ("hi", lambda e: e["mixing_rate"] >= 0.3), ("lo", lambda e: e["mixing_rate"] < 0.3)):
                n, worse, mx, mn = stats(rule, cond)
                N[f"caprec_{rule}_{cname}_n"], N[f"caprec_{rule}_{cname}_n_worse_lp1"], N[f"caprec_{rule}_{cname}_max_minus_lp"], N[f"caprec_{rule}_{cname}_min_minus_lp"] = n, worse, mx, mn
        lo_cells = [e for k, e in CR.items() if e["mixing_rate"] < 0.3 and int(k.split("|")[2][1:]) >= 10]
        d2 = [e["cfg"][f"xfit|storeyc|{QK}|pp|pooled|none"]["FPR95"] - e["cfg"][f"xfit|storey|{QK}|pp|pooled|none"]["FPR95"] for e in lo_cells]
        N["caprec_lo_r10_n"], N["caprec_lo_r10_max_storeyc_minus_storey"], N["caprec_lo_r10_min_storeyc_minus_storey"] = len(lo_cells), (max(d2) if d2 else 0.0), (min(d2) if d2 else 0.0)
        N["caprec_n_cells"] = len(CR)
        N["caprec_n_hi"] = sum(1 for e in CR.values() if e["mixing_rate"] >= 0.3)
    CD = S15.get("cap_dev")
    if CD:
        for dev in ("dev1", "dev2"):
            for rule in ("storey", "storeyc", "bh", "bhc"):
                for stream in ("near", "far"):
                    e = CD[dev].get(f"xfit|{rule}|{QK}|pp|pooled|none", {}).get(stream)
                    if e:
                        N[f"capdev_{dev}_{rule}_{stream}_FPR95"] = e["FPR95"]
                e = CD[dev].get(f"xfit|{rule}|{QK}|pp|pooled|TINS", {}).get("near")
                if e:
                    N[f"capdev_{dev}_{rule}_tins_near_FPR95"] = e["FPR95"]
    # ------------------------------------------------------------------ Phase 15 (f): the Corollary-1 check on the grid (descriptive)
    CORP = HERE / "data" / "p15_cor_summary.json"
    if CORP.exists():
        COR = json.load(open(CORP))
        N["cor_n_rows"], N["cor_max_ratio_pi"], N["cor_max_excess_pi"] = COR["n_rows"], COR["max_ratio_pi"], COR["max_excess_pi"]
        N["cor_all_valid"] = int(COR["all_valid_psi"] and COR["all_valid_pi"] and COR["all_pi_le_psi"])
        N["cor_n_clusters_total"] = sum(v["n_clusters"] for v in COR["cells"].values())
        for key, e in COR["cells"].items():
            ds, r, kg, v = key.split("|")
            if v != "D3L":
                continue
            tag = f"cor_{ds}_{r}_{kg}"
            N[f"{tag}_pi_med"], N[f"{tag}_ebar_med"], N[f"{tag}_p_act_med"], N[f"{tag}_p_bound_pi_med"] = e["pi_med"], e["ebar_med"], e["p_act_med"], e["p_bound_pi_med"]
            N[f"{tag}_frac_act_le005_pct"], N[f"{tag}_frac_bound_pi_le005_pct"], N[f"{tag}_frac_bound_psi_le005_pct"] = 100 * e["frac_act_le005"], 100 * e["frac_bound_pi_le005"], 100 * e["frac_bound_psi_le005"]
            N[f"{tag}_n_boundary_med"] = e["n_boundary_med"]
        rows = []
        for ds, lab, rs in (("ninco", "NINCO", (10, 20, 30)), ("ssb_hard", "SSB-hard", (10, 20, 50))):
            for r in rs:
                cells = []
                for kg in (3, 5, 10, 20, 40):
                    e = COR["cells"][f"{ds}|r{r}|kg{kg}|D3L"]
                    cells.append(f"{e['pi_med']:.2f} & {100 * e['frac_act_le005']:.0f} & {100 * e['frac_bound_pi_le005']:.0f}")
                rows.append(f"{lab} & {r} & " + " & ".join(cells) + " \\\\")
        tcor = ("{\\fontsize{6.3pt}{7.4pt}\\selectfont\\renewcommand{\\arraystretch}{1.0}%\n\\setlength{\\tabcolsep}{1.6pt}\\begin{tabular}{@{}lr|rrr|rrr|rrr|rrr|rrr@{}}\n\\hline\n"
                "\\rule[-2.0pt]{0pt}{8.2pt} & & \\multicolumn{3}{c|}{$k_g{=}3$} & \\multicolumn{3}{c|}{$k_g{=}5$} & \\multicolumn{3}{c|}{$k_g{=}10$} & \\multicolumn{3}{c|}{$k_g{=}20$} & \\multicolumn{3}{c}{$k_g{=}40$} \\\\\n"
                "\\rule[-2.0pt]{0pt}{8.2pt} & $r$ & $\\pi$ & 実 & 系 & $\\pi$ & 実 & 系 & $\\pi$ & 実 & 系 & $\\pi$ & 実 & 系 & $\\pi$ & 実 & 系 \\\\\n\\hline\n" + "\n".join(rows) + TAIL)
        (sec / "t18_cor.tex").write_text(tcor)
    # ------------------------------------------------------------------ helpers: absolute differences, percent ratios
    for k in list(N):
        if isinstance(N[k], float) and (k.endswith(("_dFPR95_vs_lp", "_dFPR95_vs_storey", "_dFPR95_vs_lp_lo", "_dFPR95_vs_lp_hi", "_dFPR95_vs_storey_lo", "_dFPR95_vs_storey_hi")) or "_minus_lp" in k or "_minus_raw_" in k) and f"{k}_abs" not in N:
            N[f"{k}_abs"] = abs(N[k])
        if isinstance(N[k], float) and k.startswith("in200_seed_") and k.endswith("_ratio") and f"{k}_pct" not in N:
            N[f"{k}_pct"] = 100.0 * N[k]
        if isinstance(N[k], float) and k.endswith("_idfrac") and f"{k}_pct" not in N:
            N[f"{k}_pct"] = 100.0 * N[k]
    # ------------------------------------------------------------------ numbers.tex
    lines = []
    digits = {"0": "Zero", "1": "One", "2": "Two", "3": "Three", "4": "Four", "5": "Five", "6": "Six", "7": "Seven", "8": "Eight", "9": "Nine"}
    for k, v in N.items():
        kk = k.replace("FPR95", "FPR").replace("t2_", "tableTwo_").replace("t3_", "tableThree_").replace("t1_", "tableOne_")
        name = "N" + "".join(digits.get(ch, ch) for ch in "".join(ch for ch in kk.title() if ch.isalnum()))
        if isinstance(v, float):
            if any(k.startswith(p) for p in ("vcal_", "vtrain_")) and not k.endswith(("_AUROC", "_FPR95", "_n", "_pi0")):
                dec = 3                                                           # realised rates: three decimals
            elif k == "cor_max_ratio_pi":
                dec = 7
            elif k.startswith("cor_") and k.endswith("_pct"):
                dec = 0
            elif k.startswith("cor_") and (k.endswith("_p_act_med") or k.endswith("_p_bound_pi_med")):
                dec = 3
            elif k.endswith("_idfrac_pct") or k.endswith("_above_thr95_pct"):
                dec = 1
            elif "fpr005" in k or "bhfdp" in k or "idfrac" in k or "mean_cos" in k or "max_cos" in k or k.endswith("_recall") or "_val_" in k and k.endswith("_max"):
                dec = 3
            elif k.startswith("onl_iters") or k.startswith("onl_seconds") or k.startswith("cifar_mix_") or (k.startswith("cifar_seed_") and "_ratio" in k) or k.endswith("_ratio_pct"):
                dec = 0
            elif (k.startswith("rec_") or k.startswith("kg_") or k.startswith("caprec_")) and not ("fpr005" in k or "min_mix" in k):
                dec = 1
            elif k.endswith("_ratio_vs_best_other"):
                dec = 1
            else:
                dec = 2
            lines.append(f"\\newcommand\\{name}{{{v:.{dec}f}}}")
        elif isinstance(v, int):
            lines.append(f"\\newcommand\\{name}{{{v}}}")
        else:
            lines.append(f"% {name} = {v}")
    (sec / "numbers.tex").write_text("\n".join(lines) + "\n")
    json.dump(N, open(HERE / "data" / "numbers.json", "w"), indent=1, ensure_ascii=False)
    print(json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in N.items() if not k.startswith(("abl_", "view_", "rec_", "vtrain_seed", "vcal_seed"))}, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
