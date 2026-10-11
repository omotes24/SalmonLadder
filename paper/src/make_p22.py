"""Paper v22: the read-out with reference images sigma_ref (Phase 17, registered; adopted as the main read-out by the registered
interval rule on dev2) is the main configuration; sigma (cross-fitted) is reported as the read-out of v18-v21.
Writes sections/numbers22.tex (+ data/numbers22.json) and the tables sections/t22_*.tex from
  data/p17_summary.json     registered summary of Phase 17 (ana17_fix.py --summary)
  data/p17_adoption.json    the registered adoption rule on dev2 (ana17.py --dev2-adopt)
  data/p17_pairs.json       paired comparisons formed by derive_p17.py from the per-task CSVs (descriptive)
  data/p17_rec_seeds.json   seeds of sigma_ref on the recurrence grid (derive_p17.py; descriptive)
  data/devtest.json, data/lowmix.json, data/folds_check.json   offline analyses of existing results (experiments/phase17/offline)
and, for the rows that were not rerun (static p, LGC-max, capped seeds, arrival-time mode, published and measured comparison rows),
the inputs of make_p18.py (imported as a module; its main() still writes numbers.tex and the t18_* tables used by the appendix).
Conventions as in make_p18: \\sota = column best, \\cellcolor{upd}+bold = beats every non-ours row, \\gap = margin to the best
non-ours value (from the displayed values)."""
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import make_p18 as M  # noqa: E402

D = HERE / "data"
SEC = HERE / "sections"
S = json.load(open(D / "p17_summary.json"))
AD = json.load(open(D / "p17_adoption.json"))
PR = json.load(open(D / "p17_pairs.json"))
RS = json.load(open(D / "p17_rec_seeds.json"))
DT = json.load(open(D / "devtest.json"))
LM = json.load(open(D / "lowmix.json"))
FC = json.load(open(D / "folds_check.json"))
S16 = json.load(open(D / "p16_summary.json"))

SIG = "xfit|storey|q0.2|pp|pooled"      # sigma (cross-fitted; v18-v21), ranking rows
PF = "xfit|storey|q0.2|pp"              # its p_fin
REF = "ref|storey|q0.2|pp|stat"         # sigma_ref (main), ranking rows
REFP = "ref|storey|q0.2|pp"             # its p_fin
RLP = "ref|lp|stat"                     # p+ only, read with the reference images
XLP = "xfit|lp|pooled"                  # p+ only, cross-fitted
V1 = "v1|storey|q0.2|pp"                # p+ p- without folds or references (v16)
DEC = ("dec|deg", "dec|q", "dec|u", "dec|sigma_q")
GRP = {"near": ("test_openood", "near"), "far": ("test_openood", "far"), "four": ("test_fourood", "fourood")}
SIXN = M.SIXN
N, FMT = {}, {}


def put(name, val, dec=2):
    assert name.isalpha() and name not in N, name
    assert val is not None and (isinstance(val, (bool, str)) or np.isfinite(val)), (name, val)
    N[name] = val
    FMT[name] = dec


def g17(key, grp):
    sec, g = GRP[grp]
    return S[sec][g].get(key)


def six17(key):
    out = []
    for grp in ("near", "far", "four"):
        e = g17(key, grp)
        if e is None:
            return None
        out += [e["AUROC"], e["FPR95"]]
    return out


def six14(key):
    return M.six(key)


def ds17(key, part, ds):
    e = S[f"test_{part}"]["per_dataset"].get(f"{key}|{ds}")
    return None if e is None else [e["AUROC"], e["FPR95"]]


def r2(x):
    return round(x, 2)


def fmt_table22(rows, ncols, fmt, gap_row=None, gap_label="\\quad 最良の比較手法との差", mark_best=True):
    """make_p18.fmt_table with the 'beats every non-ours row' shading decided on the displayed (2-decimal) values, so that the shading
    agrees with the gap row (a tie at two decimals is neither shaded nor counted as a win)."""
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
        cells, gaps_ = [], []
        for j in range(ncols):
            x = v[j]
            if x is None:
                cells.append("--")
                gaps_.append("")
                continue
            hi = (j % 2 == 0)
            s = fmt[j](x)
            better = best_other[j] is not None and ((r2(x) > r2(best_other[j])) if hi else (r2(x) < r2(best_other[j])))
            if o and better:
                s = f"\\cellcolor{{upd}}\\textbf{{{s}}}"
            if mark_best and best[j] is not None and abs(x - best[j]) < 1e-9:
                s += "\\sota"
            cells.append(s)
            g = ((r2(x) - r2(best_other[j])) if hi else (r2(best_other[j]) - r2(x))) if best_other[j] is not None else None
            gaps_.append(f"\\gap{{{g:+.2f}}}" if g is not None else "")
        out.append(lab + " & " + " & ".join(cells) + " \\\\")
        if gap_row is not None and i in (gap_row if isinstance(gap_row, (list, tuple)) else [gap_row]):
            out.append("{\\scriptsize\\textcolor{black!60}{" + gap_label + "}} & " + " & ".join(gaps_) + " \\\\")
    return "\n".join(out)



GN = {"near": "Near", "far": "Far", "four": "Four"}
# ===================================================================== sigma_ref reproduces nothing it should not: sigma equals Phase 14
for grp in ("near", "far", "four"):
    for base in ("none", "TINS", "TANL", "NegLabel", "AdaNeg"):
        a, b = g17(f"{SIG}|{base}", grp), M.tt_e(f"{SIG}|{base}", grp)
        assert abs(a["FPR95"] - b["FPR95"]) < 1e-4 and abs(a["AUROC"] - b["AUROC"]) < 1e-4, (grp, base)

# ===================================================================== table: main (OpenOOD v1.5 ImageNet-1K and Four-OOD)
rows = []
rows.append(("\\multicolumn{7}{@{}l}{\\rule{0pt}{6.6pt}\\textit{ImageNet-1Kのラベルで学習した検出器（報告値$^{\\dagger}$，近／遠の平均は本稿で計算）}} \\\\", None, False))
for m in ["MOODv2", "NNGuide", "SCALE"]:
    p = M.PUB["openood"][m]["per"]
    rows.append((M.PUB["openood"][m]["label"], [M.mean([p[d][0] for d in M.OO_NEAR]), M.mean([p[d][1] for d in M.OO_NEAR]),
                                                 M.mean([p[d][0] for d in M.OO_FAR]), M.mean([p[d][1] for d in M.OO_FAR]), None, None], False))
rows.append(("\\multicolumn{7}{@{}l}{\\rule{0pt}{6.6pt}\\textit{教師なし（報告値$^{\\dagger}$）}} \\\\", None, False))
p = M.PUB["openood"]["DisCoPatch"]["per"]
rows.append((M.PUB["openood"]["DisCoPatch"]["label"], [M.mean([p[d][0] for d in M.OO_NEAR]), M.mean([p[d][1] for d in M.OO_NEAR]),
                                                        M.mean([p[d][0] for d in M.OO_FAR]), M.mean([p[d][1] for d in M.OO_FAR]), None, None], False))
rows.append(("\\multicolumn{7}{@{}l}{\\rule{0pt}{6.6pt}\\textit{VLM，到着時採点（CLIP ViT-B/16，報告値$^{\\dagger}$）}} \\\\", None, False))
for m in ["NegLabel", "AdaNeg", "InterNeg", "TINS"]:
    e = M.PUB["openood"][m]
    f = M.PUB["fourood"].get(m, {}).get("avg")
    rows.append((e["label"], [e["near"][0], e["near"][1], e["far"][0], e["far"][1]] + (list(f) if f else [None, None]), False))
rows.append(("\\multicolumn{7}{@{}l}{\\rule{0pt}{6.6pt}\\textit{同じテスト集合での実測（公式実装，到着時採点，学習なし）}} \\\\", None, False))
for m in ["NegLabel", "AdaNeg", "TANL", "TINS"]:
    rows.append((M.BASE_JA[m], M.six_meas(m), False))
rows.append(("\\multicolumn{7}{@{}l}{\\rule{0pt}{6.6pt}\\textit{Salmon Ladder（Ours，学習なし，クラスあたり16枚，事後採点；6通りの抽出の平均）}} \\\\", None, False))
rows.append(("Salmon Ladder，単体", six17(f"{REF}|none"), True))
gap_alone = len(rows) - 1
rows.append(("\\multicolumn{7}{@{}l}{\\rule{0pt}{6.6pt}\\textit{\\quad ベース検出器との積 $s_0\\exp\\sR$（\\secref{score}）}} \\\\", None, False))
for b in ("NegLabel", "AdaNeg", "TANL"):
    rows.append((f"\\quad $\\times${b}", six17(f"{REF}|{b}"), True))
rows.append(("\\quad $\\times$TINS", six17(f"{REF}|TINS"), True))
gap_tins = len(rows) - 1
body = fmt_table22(rows, 6, [M.f2] * 6, gap_row=[gap_alone, gap_tins])
(SEC / "t22_main.tex").write_text(M.head6("手法", font="6.8pt", lead="8.0pt") + body + M.TAIL)
others = [r for r in rows if r[1] is not None and not r[2]]
best = {}
for j, name in enumerate(SIXN):
    col = [(r[1][j], r[0]) for r in others if r[1][j] is not None]
    hi = j % 2 == 0
    best[name] = (max if hi else min)(col)
for tag, key in (("Alone", f"{REF}|none"), ("Tins", f"{REF}|TINS"), ("TANL", f"{REF}|TANL"), ("NegLabel", f"{REF}|NegLabel"), ("AdaNeg", f"{REF}|AdaNeg")):
    v = six17(key)
    for j, name in enumerate(SIXN):
        grp, met = name.split("_")
        G, Mt = GN[grp], ("Auroc" if met == "AUROC" else "Fpr")
        hi = j % 2 == 0
        put(f"NR{tag}{G}{Mt}", v[j])
        gap = (r2(v[j]) - r2(best[name][0])) if hi else (r2(best[name][0]) - r2(v[j]))
        put(f"NR{tag}Gap{G}{Mt}", gap)
    for grp in ("near", "far", "four"):
        e = g17(key, grp)
        if e.get("FPR95_ci") and e["FPR95_ci"].get("lo") is not None:
            put(f"NR{tag}{GN[grp]}FprLo", e["FPR95_ci"]["lo"])
            put(f"NR{tag}{GN[grp]}FprHi", e["FPR95_ci"]["hi"])
        d = e.get("dFPR95_vs_sigma")
        if d:
            put(f"NR{tag}{GN[grp]}DSig", d["mean"])
            put(f"NR{tag}{GN[grp]}DSigLo", d["lo"])
            put(f"NR{tag}{GN[grp]}DSigHi", d["hi"])
            put(f"NR{tag}{GN[grp]}DAucSig", e["dAUROC_vs_sigma"]["mean"])
put("NRDSigFarFourMaxAbs", max(abs(N[f"NR{t}{G}DSig"]) for t in ("Alone", "Tins") for G in ("Far", "Four")))
put("NRAloneNearFprRatio", best["near_FPR95"][0] / N["NRAloneNearFpr"], 1)
put("NRTinsNearFprRatio", best["near_FPR95"][0] / N["NRTinsNearFpr"], 1)
put("NRAloneWinsMain", sum(1 for n in SIXN if N[f"NRAloneGap{GN[n.split('_')[0]]}{'Auroc' if n.endswith('AUROC') else 'Fpr'}"] > 0), 0)
assert N["NRAloneWinsMain"] == 6
# base detectors measured on the same streams vs their product with sigma_ref (near FPR95)
for b in ("TINS", "TANL", "NegLabel", "AdaNeg"):
    a = M.meas(b, "openood", "near")
    put(f"NRMeas{b}NearFpr", a[1])
    put(f"NRGain{b}NearFpr", a[1] - g17(f"{REF}|{b}", "near")["FPR95"])
# sigma (v18-v21) for the same rows, for the text
for tag, key in (("SigAlone", f"{SIG}|none"), ("SigTins", f"{SIG}|TINS")):
    v = six17(key)
    for j, name in enumerate(SIXN):
        grp, met = name.split("_")
        put(f"NR{tag}{GN[grp]}{'Auroc' if met == 'AUROC' else 'Fpr'}", v[j])

# ===================================================================== per data set (OpenOOD, Four-OOD) with gap rows for both of ours
rows = []
for m in ["MOODv2", "NNGuide", "RankFeat", "SCALE", "DisCoPatch", "TINS"]:
    p = M.PUB["openood"][m]["per"]
    rows.append((M.short(M.PUB["openood"][m]["label"]) + "$^{\\dagger}$", sum(([p[d][0], p[d][1]] for d in M.OO_NEAR + M.OO_FAR), []), False))
rows.append(("TINS~\\cite{yang2026tins}（実測）", sum((list(M.meas_ds("TINS", "openood", d)) for d in M.OO_NEAR + M.OO_FAR), []), False))
rows.append(("Salmon Ladder，単体", sum((ds17(f"{REF}|none", "openood", d) for d in M.OO_NEAR + M.OO_FAR), []), True))
ga = len(rows) - 1
rows.append(("Salmon Ladder $\\times$TINS", sum((ds17(f"{REF}|TINS", "openood", d) for d in M.OO_NEAR + M.OO_FAR), []), True))
gt = len(rows) - 1
body = fmt_table22(rows, 10, [M.f2] * 10, gap_row=[ga, gt])
t = ("{\\fontsize{7pt}{8.2pt}\\selectfont\\renewcommand{\\arraystretch}{1.0}%\n\\setlength{\\tabcolsep}{2.0pt}\\begin{tabular}{@{}lcccccccccc@{}}\n\\hline\n"
     "\\rule[-2.0pt]{0pt}{8.6pt} & \\multicolumn{4}{c}{near-OOD} & \\multicolumn{6}{c}{far-OOD} \\\\\n"
     "\\rule[-2.0pt]{0pt}{8.6pt} & \\multicolumn{2}{c}{SSB-hard} & \\multicolumn{2}{c}{NINCO} & \\multicolumn{2}{c}{iNaturalist} & \\multicolumn{2}{c}{Textures} & \\multicolumn{2}{c}{OpenImage-O} \\\\\n"
     "\\rule[-2.0pt]{0pt}{8.6pt}手法 & AUROC & FPR95 & AUROC & FPR95 & AUROC & FPR95 & AUROC & FPR95 & AUROC & FPR95 \\\\\n\\hline\n" + body + M.TAIL)
(SEC / "t22_perds.tex").write_text(t)
names_oo = sum(([f"{d}_AUROC", f"{d}_FPR95"] for d in M.OO_NEAR + M.OO_FAR), [])
gA = M.gaps(rows, ga, names_oo)
gA_oo_saved = dict(gA)
gT = M.gaps(rows, gt, names_oo)
wins_oo_alone = sum(1 for k, v in gA.items() if k.startswith("gap_") and v > 0)
wins_oo_tins = sum(1 for k, v in gT.items() if k.startswith("gap_") and v > 0)
lost_oo_alone = [k[4:] for k, v in gA.items() if k.startswith("gap_") and v <= 0]
lost_oo_tins = [k[4:] for k, v in gT.items() if k.startswith("gap_") and v <= 0]
for d in M.OO_NEAR + M.OO_FAR:
    for tag, key in (("Alone", f"{REF}|none"), ("Tins", f"{REF}|TINS")):
        a = ds17(key, "openood", d)
        DD = {"ssb_hard": "SsbHard", "ninco": "Ninco", "inaturalist": "Inat", "textures": "Textures", "openimageo": "Openimageo"}[d]
        put(f"NR{tag}{DD}Auroc", a[0])
        put(f"NR{tag}{DD}Fpr", a[1])
        ci = S["test_openood"]["per_dataset_ci"].get(f"{key}|{d}")
        if ci:
            put(f"NR{tag}{DD}FprLo", ci["FPR95_ci"]["lo"])
            put(f"NR{tag}{DD}FprHi", ci["FPR95_ci"]["hi"])
dp = M.PUB["openood"]["DisCoPatch"]["per"]
put("NRDiscoSsbHardAuroc", dp["ssb_hard"][0])
put("NRDiscoSsbHardFpr", dp["ssb_hard"][1])
put("NRDiscoNincoFpr", dp["ninco"][1])
rows = []
for m in ["MCM", "NegLabel", "CSP", "OODD", "AdaNeg", "InterNeg", "TINS"]:
    e = M.PUB["fourood"][m]
    rows.append((M.short(e["label"]) + "$^{\\dagger}$", sum(([e["per"][d][0], e["per"][d][1]] for d in M.FOUR), []) + list(e["avg"]), False))
rows.append(("TINS~\\cite{yang2026tins}（実測）", sum((list(M.meas_ds("TINS", "fourood", d)) for d in M.FOUR), []) + list(M.meas("TINS", "fourood", "fourood")), False))
rows.append(("Salmon Ladder，単体", sum((ds17(f"{REF}|none", "fourood", d) for d in M.FOUR), []) + six17(f"{REF}|none")[4:6], True))
ga = len(rows) - 1
rows.append(("Salmon Ladder $\\times$TINS", sum((ds17(f"{REF}|TINS", "fourood", d) for d in M.FOUR), []) + six17(f"{REF}|TINS")[4:6], True))
gt = len(rows) - 1
body = fmt_table22(rows, 10, [M.f2] * 10, gap_row=[ga, gt])
t = ("{\\fontsize{7pt}{8.2pt}\\selectfont\\renewcommand{\\arraystretch}{1.0}%\n\\setlength{\\tabcolsep}{2.0pt}\\begin{tabular}{@{}lcccccccccc@{}}\n\\hline\n"
     "\\rule[-2.0pt]{0pt}{8.6pt} & \\multicolumn{2}{c}{iNaturalist} & \\multicolumn{2}{c}{SUN} & \\multicolumn{2}{c}{Places} & \\multicolumn{2}{c}{Textures} & \\multicolumn{2}{c}{平均} \\\\\n"
     "\\rule[-2.0pt]{0pt}{8.6pt}手法 & AUROC & FPR95 & AUROC & FPR95 & AUROC & FPR95 & AUROC & FPR95 & AUROC & FPR95 \\\\\n\\hline\n" + body + M.TAIL)
(SEC / "t22_four.tex").write_text(t)
names_f = sum(([f"{d}_AUROC", f"{d}_FPR95"] for d in M.FOUR), []) + ["avg_AUROC", "avg_FPR95"]
gA = M.gaps(rows, ga, names_f)
gT = M.gaps(rows, gt, names_f)
wins_f_alone = sum(1 for k, v in gA.items() if k.startswith("gap_") and "_avg_" not in k and v > 0)
wins_f_tins = sum(1 for k, v in gT.items() if k.startswith("gap_") and "_avg_" not in k and v > 0)
lost_f_alone = [k[4:] for k, v in gA.items() if k.startswith("gap_") and "_avg_" not in k and v <= 0]
lost_f_tins = [k[4:] for k, v in gT.items() if k.startswith("gap_") and "_avg_" not in k and v <= 0]
put("NRPerdsCols", 2 * len(M.OO_NEAR + M.OO_FAR + M.FOUR), 0)
put("NRPerdsWinsAlone", wins_oo_alone + wins_f_alone, 0)
put("NRPerdsWinsTins", wins_oo_tins + wins_f_tins, 0)
put("NRPerdsWinsAloneOO", wins_oo_alone, 0)
put("NRPerdsWinsAloneFour", wins_f_alone, 0)
N["_lost_alone"] = lost_oo_alone + ["four:" + x for x in lost_f_alone]
N["_lost_tins"] = lost_oo_tins + ["four:" + x for x in lost_f_tins]
FMT["_lost_alone"] = FMT["_lost_tins"] = None
# the main text names these six columns (four below, two ties at two decimals) and says that x TINS wins all 18
assert sorted(N["_lost_alone"]) == sorted(["ssb_hard_FPR95", "inaturalist_AUROC", "openimageo_AUROC", "openimageo_FPR95", "four:inat_AUROC", "four:places_AUROC"]), N["_lost_alone"]
assert N["_lost_tins"] == []
assert abs(gA_oo_saved["gap_inaturalist_AUROC"]) < 1e-9 and abs(gA["gap_inat_AUROC"]) < 1e-9          # the two ties
assert all(gA_oo_saved[f"gap_{k}"] < 0 for k in ("ssb_hard_FPR95", "openimageo_AUROC", "openimageo_FPR95")) and gA["gap_places_AUROC"] < 0

# ===================================================================== ablation (x TINS) and its standalone version
ABL = [("p14", M.STATIC, "静的$p$（原型への余弦の較正順位，グラフなし）"),
       ("p14", "knn", "kNN~\\cite{sun2022knn}（支持画像への$k{=}10$近傍の較正順位，グラフなし）"),
       ("p14", "lgcmax", "LGC-max（支持画像のクラスラベルを伝播，最大クラス質量の較正順位）"),
       ("p17", RLP, "$\\pP$のみ（支持画像からの伝播，種なし；参照画像で読む）"),
       ("p14", V1, "$\\pP\\pN$，折も参照画像もなし（保証は種だけ）"),
       ("p14", SIG, "$\\sigma$：2つの折で交差適合した$\\pP\\pN$"),
       ("p17", REF, "$\\sR$：参照画像で種を選び読む$\\pP\\pN$（本稿）"),
       ("p17", REFP, "$\\sR$の較正順位$\\pfin$をそのままスコアにしたもの")]


def six_src(src, key):
    return six17(key) if src == "p17" else six14(key)


def abl_rows(base, short_labels=False):
    rows = []
    for src, k, lab in ABL:
        v = six_src(src, f"{k}|{base}")
        if v is None:
            continue
        rows.append((M.short(lab) if short_labels else lab, v, False))
        if k == SIG:
            for rule, clab in M.CAPJA.items():
                v15 = M.six(f"xfit|{rule}|{M.QK}|pp|pooled|{base}", M.tt15)
                if v15:
                    rows.append((clab + ("" if short_labels else "（付録\\ref{sec:app_cap}）"), v15, False))
    o13 = M.six(f"L14+D3L|T|{base}", M.onl13)
    if o13:
        rows.append(("$\\pP\\pN$，到着時採点，折なし" + ("" if short_labels else "（付録\\ref{sec:app_online}）"), o13, False))
    o14 = M.six(f"xonl|{M.RULE}|{M.QK}|{M.MODE}|pooled|{base}", M.onl14)
    if o14:
        rows.append(("$\\sigma$，到着時採点" + ("" if short_labels else "（付録\\ref{sec:app_online}）"), o14, False))
    return rows


rows = [("\\multicolumn{7}{@{}l}{\\rule{0pt}{6.6pt}\\textit{同じ特徴（DINOv2 L/14 + DINOv3 L/16）での読み出し，$\\times$TINS（単体は付録\\ref{sec:app_sens}の\\tabref{ablx}）}} \\\\", None, False)]
rows += abl_rows("TINS")
(SEC / "t22_abl.tex").write_text(M.head6("構成", font="6.8pt", lead="8.0pt", colsep="2.0pt") + fmt_table22(rows, 6, [M.f2] * 6) + M.TAIL)
ABLX = [(f"xfit|bh|{M.QK}|pp|pooled", "$\\sigma$，種を単純BH法で選択"), (f"xfit|{M.RULE}|{M.QK}|pps|pooled", "$\\sigma$に静的$p$を加える"),
        (f"xfit|{M.RULE}|{M.QK}|min|pooled", "$\\min(\\sigma,\\,\\log$静的$p)$"), (f"xfit|bh|{M.QK}|pps|pooled", "単純BH法の種，静的$p$を加える"),
        ("xfit|storey|q0.1|pp|pooled", "$\\sigma$，$q{=}0.1$"), ("xfit|storey|q0.05|pp|pooled", "$\\sigma$，$q{=}0.05$"), ("xfit|bh|q0.1|pp|pooled", "単純BH法の種，$q{=}0.1$"),
        ("v1|storey|q0.1|pp", "$\\pP\\pN$，折なし，$q{=}0.1$"), ("v1|storey|q0.05|pp", "$\\pP\\pN$，折なし，$q{=}0.05$")]
rows = [("\\multicolumn{7}{@{}l}{\\rule{0pt}{6.6pt}\\textit{同じ特徴での読み出し，単体}} \\\\", None, False)]
rows += abl_rows("none", short_labels=True)
rows.append(("\\multicolumn{7}{@{}l}{\\rule{0pt}{6.6pt}\\textit{$\\sigma$の第2段の他の候補（\\tabref{selq}），$\\times$TINS}} \\\\", None, False))
rows.append(("$\\sigma$（Storey型，$q{=}0.2$）", six14(f"{SIG}|TINS"), False))
for k, lab in ABLX:
    v = six14(f"{k}|TINS")
    if v is not None:
        rows.append((lab, v, False))
(SEC / "t22_ablx.tex").write_text(M.head6("構成", font="6.8pt", lead="8.0pt", colsep="2.0pt") + fmt_table22(rows, 6, [M.f2] * 6) + M.TAIL)
# numbers of the ablation used in the text
P = PR["test_openood"]
PF4 = PR["test_fourood"]
for base, tg in (("none", "Alone"), ("TINS", "Tins")):
    e = P[f"near|{RLP}|{base}"]
    put(f"NRLp{tg}NearFpr", e["FPR95"])
    put(f"NRvsLp{tg}NearD", e["dFPR95_ref_minus"]["mean"])
    put(f"NRvsLp{tg}NearDLo", e["dFPR95_ref_minus"]["lo"])
    put(f"NRvsLp{tg}NearDHi", e["dFPR95_ref_minus"]["hi"])
    put(f"NRvsLp{tg}NearDAbs", abs(e["dFPR95_ref_minus"]["mean"]))
    e = P[f"far|{RLP}|{base}"]
    put(f"NRLp{tg}FarFpr", e["FPR95"])
    put(f"NRvsLp{tg}FarD", e["dFPR95_ref_minus"]["mean"])
    e = PF4[f"fourood|{RLP}|{base}"]
    put(f"NRLp{tg}FourFpr", e["FPR95"])
    put(f"NRvsLp{tg}FourD", e["dFPR95_ref_minus"]["mean"])
    e = P[f"near|{V1}|{base}"]
    put(f"NRvsVOne{tg}NearD", e["dFPR95_ref_minus"]["mean"])
    put(f"NRvsVOne{tg}NearDLo", e["dFPR95_ref_minus"]["lo"])
    put(f"NRvsVOne{tg}NearDHi", e["dFPR95_ref_minus"]["hi"])
    put(f"NRVOne{tg}NearFpr", e["FPR95"])
put("NRPfAloneNearFpr", g17(f"{REFP}|none", "near")["FPR95"])
put("NRPfTinsNearFpr", g17(f"{REFP}|TINS", "near")["FPR95"])
put("NRPfTinsFourFpr", g17(f"{REFP}|TINS", "four")["FPR95"])
put("NRPfTinsNearWorse", g17(f"{REFP}|TINS", "near")["FPR95"] - g17(f"{REF}|TINS", "near")["FPR95"])

# ===================================================================== decomposition u = sqrt(delta) * q (standalone; test, ImageNet-200, CIFAR-100)
DECSETS = [("test_openood", "near", "INNear", "ImageNet-1K near"), ("test_openood", "far", "INFar", "ImageNet-1K far"), ("test_fourood", "fourood", "Four", "Four-OOD"),
           ("in200", "near", "ITNear", "IN-200 near"), ("in200", "far", "ITFar", "IN-200 far"), ("cifar", "near", "CFNear", "CIFAR near"), ("cifar", "far", "CFFar", "CIFAR far")]
DECROWS = [("dec|deg", "Deg", "$\\sqrt{\\delta}$のみ（密度の因子）"), ("dec|q", "Q", "$q=u^{+}/\\sqrt{\\delta}$のみ（確率の因子）"),
           ("dec|u", "U", "$u^{+}=\\sqrt{\\delta}\\,q$（積；$\\pP$，ラベル付き画像4,000枚）"), ("dec|sigma_q", "SigQ", "$\\sigma$の$u^{\\pm}$を$u^{\\pm}/\\sqrt{\\delta}$に置換"),
           (SIG, "Sig", "$\\sigma$（交差適合）"), (REF, "Ref", "$\\sR$（本稿）")]
lines = []
for key, tag, lab in DECROWS:
    cells = []
    for sec, g, st, _ in DECSETS:
        e = S[sec][g][f"{key}|none"]
        cells.append(f"{e['AUROC']:.2f}／{e['FPR95']:.2f}")
        put(f"NRDec{tag}{st}Auroc", e["AUROC"])
        put(f"NRDec{tag}{st}Fpr", e["FPR95"])
    lines.append(lab + " & " + " & ".join(cells) + " \\\\")
t = ("{\\fontsize{6.6pt}{7.8pt}\\selectfont\\renewcommand{\\arraystretch}{1.0}%\n\\setlength{\\tabcolsep}{1.9pt}\\begin{tabular}{@{}lccccccc@{}}\n\\hline\n"
     "\\rule[-2.0pt]{0pt}{8.6pt} & \\multicolumn{3}{c}{ImageNet-1K（OpenOOD, Four-OOD）} & \\multicolumn{2}{c}{ImageNet-200} & \\multicolumn{2}{c}{CIFAR-100} \\\\\n"
     "\\rule[-2.0pt]{0pt}{8.6pt}読み出し（AUROC／FPR95） & near & far & Four-OOD & near & far & near & far \\\\\n\\hline\n" + "\n".join(lines[:4]) + "\n\\hline\n" + "\n".join(lines[4:]) + M.TAIL)
(SEC / "t22_dec.tex").write_text(t)
for sec, g, st, _ in DECSETS:
    pp = PR[sec]
    for key, tag in (("dec|q", "Q"), ("dec|u", "U")):
        e = pp[f"{g}|{key}|none"]
        # difference q - u (positive: the density factor helps) from the per-draw values: (ref - u) - (ref - q) = q - u
    eq, eu = pp[f"{g}|dec|q|none"], pp[f"{g}|dec|u|none"]
    put(f"NRDecQminusU{st}", eq["FPR95"] - eu["FPR95"])
    # for the text: how much the product u+ lowers (Helps) or raises (Hurts) FPR95 relative to q alone (positive numbers)
    if eq["FPR95"] >= eu["FPR95"]:
        put(f"NRDecUHelps{st}", eq["FPR95"] - eu["FPR95"])
    else:
        put(f"NRDecUHurts{st}", eu["FPR95"] - eq["FPR95"])
# per data set of the decomposition (for the appendix text): AUROC of sqrt(delta) alone per OpenOOD data set
for d in M.OO_NEAR + M.OO_FAR:
    e = S["test_openood"]["per_dataset"].get(f"dec|deg|none|{d}")
    eq = S["test_openood"]["per_dataset"].get(f"dec|q|none|{d}")
    eu = S["test_openood"]["per_dataset"].get(f"dec|u|none|{d}")
    DD = {"ssb_hard": "SsbHard", "ninco": "Ninco", "inaturalist": "Inat", "textures": "Textures", "openimageo": "Openimageo"}[d]
    put(f"NRDecDeg{DD}Auroc", e["AUROC"], 1)
    put(f"NRDecQ{DD}Fpr", eq["FPR95"])
    put(f"NRDecU{DD}Fpr", eu["FPR95"])

# ===================================================================== dev splits: the registered adoption rule and dev1
for dev, tg in (("dev2", "DevTwo"), ("dev1_descriptive", "DevOne")):
    e = AD[dev]
    for s, G in (("near", "Near"), ("far", "Far")):
        x = e[s]
        put(f"NR{tg}{G}RefFpr", x["sigma_ref_FPR95"])
        put(f"NR{tg}{G}SigFpr", x["sigma_FPR95"])
        put(f"NR{tg}{G}D", x["diff_ci"]["mean"])
        put(f"NR{tg}{G}DLo", x["diff_ci"]["lo"])
        put(f"NR{tg}{G}DHi", x["diff_ci"]["hi"])
        put(f"NR{tg}{G}RefAuroc", x["sigma_ref_AUROC"])
        put(f"NR{tg}{G}SigAuroc", x["sigma_AUROC"])
        put(f"NR{tg}{G}TinsD", x["x_tins"]["diff_ci"]["mean"])
    put(f"NR{tg}FprFive", e["validity"]["fpr@0.05"], 3)
    put(f"NR{tg}FprTwenty", e["validity"]["fpr@0.2"], 3)
    put(f"NR{tg}BhFdp", e["validity"]["bh_fdp_mean"], 3)
    put(f"NR{tg}SigFprFive", e["validity_sigma"]["fpr@0.05"], 3)
assert AD["adopt_sigma_ref_as_main"] and AD["near_holds"] and AD["far_holds"] and AD["validity_holds"]
N["adopt_sigma_ref"] = True
FMT["adopt_sigma_ref"] = None

# ===================================================================== validity of p_fin(sigma_ref) on the benchmark (training-image calibration)
for grp in ("near", "far", "four"):
    sec, g = GRP[grp]
    vr = S[sec]["validity_ref"][g]
    put(f"NRVal{GN[grp]}FprFive", vr["fpr@0.05"], 3)
    put(f"NRVal{GN[grp]}FprTwenty", vr["fpr@0.2"], 3)
    put(f"NRVal{GN[grp]}BhFdp", vr["bh_fdp"], 3)
    vs = S[sec]["validity"][g].get(PF)
    put(f"NRValSig{GN[grp]}FprFive", vs["fpr@0.05"], 3)

# ===================================================================== ImageNet-200 and CIFAR-100 (sigma_ref rows; the other rows as in v21)
for sec, tg in (("in200", "IT"), ("cifar", "CF")):
    for g, G in (("near", "Near"), ("far", "Far")):
        e = S[sec][g][f"{REF}|none"]
        put(f"NR{tg}{G}Auroc", e["AUROC"])
        put(f"NR{tg}{G}Fpr", e["FPR95"])
        d = e["dFPR95_vs_sigma"]
        put(f"NR{tg}{G}DSig", d["mean"])
        put(f"NR{tg}{G}DSigLo", d["lo"])
        put(f"NR{tg}{G}DSigHi", d["hi"])
        pe = PR[sec][f"{g}|{RLP}|none"]
        put(f"NR{tg}{G}LpFpr", pe["FPR95"])
        put(f"NR{tg}{G}DLp", pe["dFPR95_ref_minus"]["mean"])
        put(f"NR{tg}{G}DLpLo", pe["dFPR95_ref_minus"]["lo"])
        put(f"NR{tg}{G}DLpHi", pe["dFPR95_ref_minus"]["hi"])
        es = S[sec][g][f"{SIG}|none"]
        put(f"NR{tg}{G}SigFpr", es["FPR95"])
        vr = S[sec]["validity_ref"].get(g)
        if vr:
            put(f"NR{tg}{G}ValFive", vr["fpr@0.05"], 3)


def in200_cifar_table(kind):
    if kind == "in200":
        I2 = M.S14["in200"]
        PUB200 = M.PUB.get("imagenet200", {})
        rows = [("\\multicolumn{5}{@{}l}{\\rule{0pt}{6.6pt}\\textit{ImageNet-200で学習したResNet-18（報告値$^{\\dagger}$）}} \\\\", None, False)]
        rows += [(e["label"], [e["near"][0], e["near"][1], e["far"][0], e["far"][1]], False) for name, e in PUB200.items() if not name.startswith("_")]
        B = (M.S15.get("base") or {}).get("imagenet200")
        src, cap, pre = I2, M.S15.get("cap_in200"), ""
        nd = I2["n_draws"]
    else:
        CF = M.S14["cifar"]
        PUBC = {k: v for k, v in M.PUB.get("cifar100", {}).items() if not k.startswith("_")}
        rows = [("\\multicolumn{5}{@{}l}{\\rule{0pt}{6.6pt}\\textit{CIFAR-100で学習したResNet-18（報告値$^{\\dagger}$）}} \\\\", None, False)]
        for name in ("MSP", "MLS", "RMDS", "VOS", "G-ODIN", "RotPred", "OE"):
            e = PUBC[name]
            rows.append((M.short(e["label"]) + ("（外れ値データで学習）" if e["group"] == "train_outlier" else ""), [e["near"][0], e["near"][1], e["far"][0], e["far"][1]], False))
        B = (M.S15.get("base") or {}).get("cifar100")
        src, cap, pre = CF, M.S15.get("cap_cifar"), "L14+D3L|"
        nd = CF["n_draws"]
    if B:
        rows.append(("\\multicolumn{5}{@{}l}{\\rule{0pt}{6.6pt}\\textit{全学習画像（" + f"{B['n_train']:,}" + "枚）またはゼロショット（本稿で実測）}} \\\\", None, False))
        for m, lab in M.BASE15.items():
            if m in B["near"]:
                rows.append((lab, [B["near"][m]["AUROC"], B["near"][m]["FPR95"], B["far"][m]["AUROC"], B["far"][m]["FPR95"]], False))
    rows.append(("\\multicolumn{5}{@{}l}{\\rule{0pt}{6.6pt}\\textit{凍結特徴，クラスあたり16枚，学習なし，単体（" + f"{nd}" + "通りの抽出の平均）}} \\\\", None, False))

    def g4(key):
        a, b = src["near"].get(key), src["far"].get(key)
        return None if (a is None or b is None) else [a["AUROC"], a["FPR95"], b["AUROC"], b["FPR95"]]

    def g4_17(key):
        a, b = S[kind]["near"][key], S[kind]["far"][key]
        return [a["AUROC"], a["FPR95"], b["AUROC"], b["FPR95"]]
    for key, lab, ours in ((f"{pre}{M.STATIC}|none", "静的$p$（同じ特徴，グラフなし）", False), (f"{pre}knn|none", "kNN（同じ特徴，グラフなし）", False),
                           (f"{pre}lgcmax|none", "LGC-max（同じグラフ）", False)):
        x = g4(key)
        if x:
            rows.append((lab, x, ours))
    rows.append(("$\\pP$のみ（種なし，参照画像で読む）", g4_17(f"{RLP}|none"), True))
    x = g4(f"{pre}{V1}|none")
    if x:
        rows.append(("$\\pP\\pN$，折も参照画像もなし", x, True))
    rows.append(("$\\sigma$（交差適合）", g4(f"{pre}{SIG}|none"), True))
    rows.append(("$\\sR$（本稿の構成）", g4_17(f"{REF}|none"), True))
    if cap:
        for rule, lab in M.CAPJA.items():
            key = f"xfit|{rule}|{M.QK}|pp|pooled|none"
            if key in cap["near"]:
                rows.append((lab, [cap["near"][key]["AUROC"], cap["near"][key]["FPR95"], cap["far"][key]["AUROC"], cap["far"][key]["FPR95"]], True))
    body = fmt_table22(rows, 4, [M.f2] * 4)
    t = ("{\\fontsize{6.8pt}{8pt}\\selectfont\\renewcommand{\\arraystretch}{1.0}%\n\\setlength{\\tabcolsep}{2pt}\\begin{tabular}{@{}lcccc@{}}\n\\hline\n"
         "\\rule[-2.0pt]{0pt}{8.6pt} & \\multicolumn{2}{c}{near-OOD} & \\multicolumn{2}{c}{far-OOD} \\\\\n"
         "\\rule[-2.0pt]{0pt}{8.6pt}手法 & AUROC$\\uparrow$ & FPR95$\\downarrow$ & AUROC$\\uparrow$ & FPR95$\\downarrow$ \\\\\n\\hline\n" + body + M.TAIL)
    (SEC / f"t22_{kind}.tex").write_text(t)
    # sigma_ref beats every reported row?
    pub = [r[1] for r in rows if r[1] is not None and not r[2]]
    v = g4_17(f"{REF}|none")
    return all((v[j] > max(p_[j] for p_ in pub if p_[j] is not None)) if j % 2 == 0 else (v[j] < min(p_[j] for p_ in pub if p_[j] is not None)) for j in range(4))


N["in200_ref_beats_all_other_rows"] = in200_cifar_table("in200")
N["cifar_ref_beats_all_other_rows"] = in200_cifar_table("cifar")
FMT["in200_ref_beats_all_other_rows"] = FMT["cifar_ref_beats_all_other_rows"] = None

# ===================================================================== recurrence grid (sigma_ref; static p from Phase 14 on the same cells)
REC14 = M.S14["rec"]
REC17 = S["rec"]
cells = sorted({tuple(k.split("|")[:3]) for k in REC17}, key=lambda x: (x[0], int(x[1][1:]), int(x[2][1:])))
diffs_sig = []
for ds, c, r in cells:
    k = f"{ds}|{c}|{r}"
    a = REC17[f"{k}|{SIG}|none"]["FPR95"]
    b = REC14[k]["cfg"][f"{SIG}|none"]["FPR95"]
    assert abs(a - b) < 1e-6, (k, a, b)           # the same cells as Phase 14 (so its static p applies)
put("NRRecNCells", len(cells), 0)


def rv(k, key, src="p17"):
    return REC17[f"{k}|{key}|none"]["FPR95"] if src == "p17" else REC14[k]["cfg"][f"{key}|none"]["FPR95"]


le5 = [f"{ds}|{c}|{r}" for ds, c, r in cells if int(r[1:]) <= 5]
ge10 = [f"{ds}|{c}|{r}" for ds, c, r in cells if int(r[1:]) >= 10]
put("NRRecNLeFive", len(le5), 0)
put("NRRecNGeTen", len(ge10), 0)
put("NRRecLeFiveWorseStatic", sum(1 for k in le5 if rv(k, REF) > rv(k, M.STATIC, "p14")), 0)
put("NRRecLeFiveMeanD", float(np.mean([rv(k, REF) - rv(k, M.STATIC, "p14") for k in le5])), 1)
put("NRRecLeFiveMaxD", float(np.max([rv(k, REF) - rv(k, M.STATIC, "p14") for k in le5])), 1)
for rr, W in ((1, "One"), (2, "Two"), (5, "Five")):
    ks = [k for k in le5 if int(k.split("|")[2][1:]) == rr]
    put(f"NRRecR{W}MeanD", float(np.mean([rv(k, REF) - rv(k, M.STATIC, "p14") for k in ks])), 1)
    put(f"NRRecR{W}N", len(ks), 0)
    put(f"NRRecR{W}Worse", sum(1 for k in ks if rv(k, REF) > rv(k, M.STATIC, "p14")), 0)
put("NRRecGeTenBetterStatic", sum(1 for k in ge10 if rv(k, REF) < rv(k, M.STATIC, "p14")), 0)
put("NRRecGeTenMeanD", float(np.mean([rv(k, REF) - rv(k, M.STATIC, "p14") for k in ge10])), 1)
dref = [rv(f"{ds}|{c}|{r}", REF) - rv(f"{ds}|{c}|{r}", SIG) for ds, c, r in cells]
put("NRRecMaxAbsDiffSig", float(np.max(np.abs(dref))), 1)
put("NRRecNWithinOneSig", sum(1 for x in dref if abs(x) <= 1.0 + 1e-9), 0)
put("NRRecMeanDSig", float(np.mean(dref)), 2)
for k, tag in (("ninco|c64|r1", "NincoCSixtyFourROne"), ("ninco|c64|r10", "NincoCSixtyFourRTen"), ("ninco|c64|r30", "NincoCSixtyFourRThirty"),
               ("ssb_hard|c980|r10", "SsbCNineEightyRTen"), ("ssb_hard|c980|r50", "SsbCNineEightyRFifty")):
    put(f"NRRec{tag}", rv(k, REF), 1)
    put(f"NRRec{tag}Lp", rv(k, RLP), 1)
    put(f"NRRec{tag}Static", rv(k, M.STATIC, "p14"), 1)
    put(f"NRRec{tag}Sig", rv(k, SIG), 1)
assert abs(N["NRRecNincoCSixtyFourROne"] - N["NRRecNincoCSixtyFourROneLp"]) < 1e-9     # the text says 'sigma_ref and p+ only both'
# validity on the grid: p_fin(sigma_ref) at alpha 0.05 from the per-cell rows is not in the summary; report the benchmark validity instead
# h (Phase 16) on the same cells against sigma_ref, r <= 2
H16 = S16["rec"]
dh = [H16[f"{ds}|{c}|{r}|h16|s|none"]["FPR95"] - rv(f"{ds}|{c}|{r}", REF) for ds, c, r in cells if int(r[1:]) <= 2 and f"{ds}|{c}|{r}|h16|s|none" in H16]
put("NRRecHvsRefLeTwoMeanD", float(np.mean(dh)), 1)
# cells with mixing rate >= 10 %: sigma_ref vs p+ only (the conclusion's recommendation)
hm = [(ds, c, r) for ds, c, r in cells if REC17[f"{ds}|{c}|{r}|{REF}|none"]["mixing_rate"] >= 0.10]
put("NRRecHiMixN", len(hm), 0)
put("NRRecHiMixMax", 100 * max(REC17[f"{ds}|{c}|{r}|{REF}|none"]["mixing_rate"] for ds, c, r in hm), 0)
assert all(rv(f"{ds}|{c}|{r}", REF) < rv(f"{ds}|{c}|{r}", RLP) for ds, c, r in hm)
put("NRRecHvsRefLeTwoN", len(dh), 0)
# seeds of sigma_ref against the mixing rate (derive_p17.py)
for b, tg in (("lt1", "LtOne"), ("1to10", "OneToTen"), ("ge10", "GeTen")):
    e = RS[b]
    put(f"NRSeed{tg}CellsViews", e["cells_views"], 0)
    put(f"NRSeed{tg}WithSeeds", e["with_seeds"], 0)
    put(f"NRSeed{tg}IdFrac", e["idfrac_mean"], 2)
    put(f"NRSeed{tg}IdFracMin", e["idfrac_min"], 2)
    put(f"NRSeed{tg}IdFracMax", e["idfrac_max"], 2)
    put(f"NRSeed{tg}FdpUncond", e["fdp_uncond_mean"], 2)
    put(f"NRSeed{tg}ShareReps", 100 * e["share_reps_with_seeds"], 0)
# seeds of sigma_ref on the benchmarks (per data set and view, mean over the draws)
_bf = [e["idfrac"] for e in RS["bench"].values()]
put("NRSeedBenchIdMin", min(_bf), 2)
put("NRSeedBenchIdMax", max(_bf), 2)
# the same for sigma (offline, Phase 14 grid)
for b, tg in (("lt1", "LtOne"), ("1to10", "OneToTen"), ("ge10", "GeTen")):
    e = LM[b]
    put(f"NRSeedSig{tg}WithSeeds", e["cells_views"], 0)
    put(f"NRSeedSig{tg}IdFrac", e["id_frac_mean"], 2)

# per-cell table of sigma_ref on the grid (appendix)
for ds, lab in (("ninco", "NINCO"), ("ssb_hard", "SSB-hard")):
    lines = []
    for d_, c, r in [x for x in cells if x[0] == ds]:
        k = f"{d_}|{c}|{r}"
        e17 = REC17[f"{k}|{REF}|none"]
        sd = RS["cells"][f"{k}|D3L"]
        idf = "--" if sd["idfrac"] is None else f"{100 * sd['idfrac']:.0f}"
        n_ood = REC14[k]["n_ood"]
        lines.append(f"{c[1:]} & {r[1:]} & {n_ood:,} & {100 * e17['mixing_rate']:.2f} & {rv(k, M.STATIC, 'p14'):.1f} & {rv(k, RLP):.1f} & {rv(k, SIG):.1f} & "
                     f"{rv(k, REF):.1f} & {REC17[f'{k}|{REF}|TINS']['FPR95']:.1f} & {sd['seeds']:,.0f} & {idf} \\\\")
    t = ("{\\fontsize{6.2pt}{7.2pt}\\selectfont\\renewcommand{\\arraystretch}{1.0}%\n\\setlength{\\tabcolsep}{2.6pt}\\begin{tabular}{@{}rrrr|ccccc|rr@{}}\n\\hline\n"
         "\\rule[-2.0pt]{0pt}{8.2pt}$c$ & $r$ & 未知 & 混入率 & 静的$p$ & $\\pP$（参照） & $\\sigma$ & $\\sR$ & $\\sR\\times$TINS & 種 & ID \\\\\n\\hline\n" + "\n".join(lines) + M.TAIL)
    (SEC / f"t22_recref_{ds}.tex").write_text(t)

# seeds of sigma_ref on the benchmarks (appendix table; registered per-task columns, mean over the six draws)
DSJ = {"ssb_hard": "SSB-hard", "ninco": "NINCO", "inaturalist": "iNaturalist", "textures": "Textures", "openimageo": "OpenImage-O",
       "inat": "iNat.（Four-OOD）", "sun": "SUN", "places": "Places", "dtd": "Textures（Four-OOD）"}
lines = []
for part, dss in (("openood", M.OO_NEAR + M.OO_FAR), ("fourood", M.FOUR)):
    for ds in dss:
        cells_ = []
        for v in ("L14", "D3L"):
            e = RS["bench"][f"{part}|{ds}|{v}"]
            cells_ += [f"{e['seeds']:,.0f}", f"{100 * e['seeds'] / e['pool']:.1f}", f"{e['cal_seeds']:.0f}", f"{e['idfrac']:.3f}"]
        lines.append(f"{DSJ[ds]} & " + " & ".join(cells_) + " \\\\")
t = ("{\\fontsize{6.5pt}{7.6pt}\\selectfont\\renewcommand{\\arraystretch}{1.0}%\n\\setlength{\\tabcolsep}{2.8pt}\\begin{tabular}{@{}l|rrrr|rrrr@{}}\n\\hline\n"
     "\\rule[-2.0pt]{0pt}{8.2pt} & \\multicolumn{4}{c|}{DINOv2 L/14} & \\multicolumn{4}{c}{DINOv3 L/16} \\\\\n"
     "\\rule[-2.0pt]{0pt}{8.2pt}データセット & 種 & プール\\% & 較正 & ID率 & 種 & プール\\% & 較正 & ID率 \\\\\n\\hline\n" + "\n".join(lines) + M.TAIL)
(SEC / "t22_seeds.tex").write_text(t)

# ===================================================================== separation bounds on the grid (Corollary 2 for sigma / sigma_ref, Corollary A1 for h)
BD = S["bounds"]
bkeys = sorted(BD, key=lambda k: (k.split("|")[0], int(k.split("|")[1][1:]), int(k.split("|")[2][1:])))
assert all(BD[k][t]["violations"] == 0 for k in BD for t in ("c2", "c2r"))
assert all(BD[k]["a1"]["violations"] == 0 and BD[k]["a1"]["phibar_violations"] == 0 for k in BD if "a1" in BD[k])
put("NRBdNCells", len(bkeys), 0)
for k, tag in (("ninco|c64|r30", "NincoCSixtyFourRThirty"), ("ninco|c64|r1", "NincoCSixtyFourROne"), ("ssb_hard|c980|r50", "SsbCNineEightyRFifty"),
               ("ninco|c10|r30", "NincoCTenRThirty"), ("ssb_hard|c10|r50", "SsbCTenRFifty")):
    for t, tt_ in (("c2r", "Ref"), ("c2", "Sig")):
        put(f"NRBd{tt_}{tag}Cert", 100 * BD[k][t]["cert_005"], 0 if 100 * BD[k][t]["cert_005"] >= 10 else 1)
        put(f"NRBd{tt_}{tag}Act", 100 * BD[k][t]["actual_005"], 0)
    put(f"NRBdHOne{tag}CertS", 100 * BD[k]["a1"]["cert_s_005"], 0 if 100 * BD[k]["a1"]["cert_s_005"] >= 10 else 1)
    put(f"NRBdHOne{tag}Act", 100 * BD[k]["a1"]["actual_005"], 0)
r30 = [k for k in bkeys if k.startswith("ninco|") and k.endswith("|r30")]
put("NRBdRefNincoRThirtyCertMin", 100 * min(BD[k]["c2r"]["cert_005"] for k in r30), 0)
put("NRBdRefNincoRThirtyCertMax", 100 * max(BD[k]["c2r"]["cert_005"] for k in r30), 0)
put("NRBdRefNincoRThirtyActMin", 100 * min(BD[k]["c2r"]["actual_005"] for k in r30), 0)
put("NRBdRefNincoRThirtyActMax", 100 * max(BD[k]["c2r"]["actual_005"] for k in r30), 0)
le2 = [k for k in bkeys if int(k.split("|")[2][1:]) <= 5]
put("NRBdRefLeFiveCertMax", 100 * max(BD[k]["c2r"]["cert_005"] for k in le2), 0 if 100 * max(BD[k]["c2r"]["cert_005"] for k in le2) >= 10 else 1)
put("NRBdRefSlackMin", min(BD[k]["c2r"]["slack_median"] for k in bkeys), 1)
put("NRBdRefSlackMax", max(BD[k]["c2r"]["slack_median"] for k in bkeys), 1)
put("NRBdHOneCertLMax", 100 * max(BD[k]["a1"]["cert_l_005"] for k in bkeys), 0)
put("NRBdHOneCertSMax", 100 * max(BD[k]["a1"]["cert_s_005"] for k in bkeys), 0)
put("NRBdHOneEllOverSMedMin", min(BD[k]["a1"]["ell_over_s_cal_median"] for k in bkeys), 2)
put("NRBdHOneEllOverSMedMax", max(BD[k]["a1"]["ell_over_s_cal_median"] for k in bkeys), 2)
put("NRBdHOneHOverPiMin", min(BD[k]["a1"]["h_over_pi_median"] for k in bkeys), 2)
put("NRBdHOneHOverPiMax", max(BD[k]["a1"]["h_over_pi_median"] for k in bkeys), 2)
assert N["NRBdHOneCertLMax"] == 0
lines = []
for ds, lab in (("ninco", "NINCO"), ("ssb_hard", "SSB-hard")):
    for k in [x for x in bkeys if x.startswith(ds + "|")]:
        c, r = k.split("|")[1][1:], k.split("|")[2][1:]
        e = BD[k]
        lines.append(f"{lab} & {c} & {r} & {100 * e['c2r']['cert_005']:.1f} & {100 * e['c2r']['actual_005']:.1f} & {e['c2r']['slack_median']:.1f} & "
                     f"{100 * e['c2']['cert_005']:.1f} & {100 * e['c2']['actual_005']:.1f} & "
                     f"{100 * e['a1']['cert_l_005']:.1f} & {100 * e['a1']['cert_s_005']:.1f} & {100 * e['a1']['actual_005']:.1f} & {e['a1']['pi_over_phibar_median']:.2f} & {e['a1']['ell_over_s_cal_median']:.2f} \\\\")
half = (len(lines) + 1) // 2
head = ("{\\fontsize{6.2pt}{7.2pt}\\selectfont\\renewcommand{\\arraystretch}{1.0}%\n\\setlength{\\tabcolsep}{2.2pt}\\begin{tabular}{@{}lrr|ccc|cc|ccccc@{}}\n\\hline\n"
        "\\rule[-2.0pt]{0pt}{8.2pt} & & & \\multicolumn{3}{c|}{$\\sR$（系2）} & \\multicolumn{2}{c|}{$\\sigma$（系2）} & \\multicolumn{5}{c}{$h$（系A1）} \\\\\n"
        "\\rule[-2.0pt]{0pt}{8.2pt}未知 & $c$ & $r$ & 保証 & 実 & 余裕 & 保証 & 実 & 下界 & 実較正 & 実 & $\\pi/\\bar\\phi$ & $\\ell/s$ \\\\\n\\hline\n")
(SEC / "t22_bounds.tex").write_text(head + "\n".join(lines) + M.TAIL)

# ===================================================================== covariate-shifted ID (ImageNet-V2, ImageNet-R)
CS = S["csid"]
CSN = {"imagenet_v2": ("ImageNet-V2", "VTwo"), "imagenet_r": ("ImageNet-R", "R")}
SCN = [("sigma_ref", "$\\sR$（本稿）", "Ref"), ("sigma", "$\\sigma$（交差適合）", "Sig"), ("lp", "$\\pP$のみ（交差適合）", "Lp"), ("static_all", "静的$p$", "Static"), ("knn", "kNN", "Knn")]
lines = []
for cs, (lab, ct) in CSN.items():
    lines.append(f"\\multicolumn{{11}}{{@{{}}l}}{{\\rule{{0pt}}{{6.6pt}}\\textit{{IDに{lab}を加える（{'10,000' if cs == 'imagenet_v2' else '30,000'}枚）}}}} \\\\")
    for name, nl, nt in SCN:
        cells = []
        for g in ("near", "far"):
            e = CS[f"{cs}|{g}|{name}"]
            cells += [f"{e['fs_AUROC']:.2f}／{e['fs_FPR95']:.2f}", f"{e['csid_ood_AUROC']:.2f}／{e['csid_ood_FPR95']:.2f}", f"{e['id_ood_FPR95']:.2f}", f"{e['id_csid_AUROC']:.1f}"]
            for met in ("fs_AUROC", "fs_FPR95", "csid_ood_AUROC", "csid_ood_FPR95", "id_ood_FPR95", "id_csid_AUROC"):
                tagm = {"fs_AUROC": "FsAuroc", "fs_FPR95": "FsFpr", "csid_ood_AUROC": "CoAuroc", "csid_ood_FPR95": "CoFpr", "id_ood_FPR95": "IoFpr", "id_csid_AUROC": "IcAuroc"}[met]
                put(f"NRCs{ct}{'Near' if g == 'near' else 'Far'}{nt}{tagm}", e[met], 1 if met == "id_csid_AUROC" else 2)
        lines.append(nl + " & " + " & ".join(cells) + " \\\\")
    for g in ("near", "far"):
        v = CS[f"{cs}|{g}|validity"]
        G = "Near" if g == "near" else "Far"
        for a, A in (("0.05", "Five"), ("0.2", "Twenty")):
            put(f"NRCs{ct}{G}FprId{A}", v[f"fpr_id@{a}|sigma_ref"], 3)
            put(f"NRCs{ct}{G}FprCs{A}", v[f"fpr_csid@{a}|sigma_ref"], 3)
        put(f"NRCs{ct}{G}SeedShare", 100 * v["seeds_ref_csid"] / v["seeds_ref_all"], 0)
t = ("{\\fontsize{6.4pt}{7.5pt}\\selectfont\\renewcommand{\\arraystretch}{1.0}%\n\\setlength{\\tabcolsep}{1.8pt}\\begin{tabular}{@{}l|cccc|cccc@{}}\n\\hline\n"
     "\\rule[-2.0pt]{0pt}{8.2pt} & \\multicolumn{4}{c|}{OOD = near-OOD（SSB-hard, NINCO）} & \\multicolumn{4}{c}{OOD = far-OOD（iNaturalist, Textures, OpenImage-O）} \\\\\n"
     "\\rule[-2.0pt]{0pt}{8.2pt}読み出し & 全体 & シフト対OOD & ID対OOD & ID対シフト & 全体 & シフト対OOD & ID対OOD & ID対シフト \\\\\n"
     "\\rule[-2.0pt]{0pt}{8.2pt} & AUROC／FPR95 & AUROC／FPR95 & FPR95 & AUROC & AUROC／FPR95 & AUROC／FPR95 & FPR95 & AUROC \\\\\n\\hline\n"
     + "\n".join(lines).replace("\\multicolumn{11}", "\\multicolumn{9}") + M.TAIL)
(SEC / "t22_csid.tex").write_text(t)

# ===================================================================== development splits vs the benchmark (offline, existing results) and the fold assignment
DS_ = DT["sigma"]
for g, G in (("near", "Near"), ("far", "Far")):
    e = DS_[g]
    put(f"NRDt{G}SpDevTwoTest", e["spearman_dev2_test"], 2)
    put(f"NRDt{G}SpDevOneTest", e["spearman_dev1_test"], 2)
    put(f"NRDt{G}SpDevOneDevTwo", e["spearman_dev1_dev2"], 2)
    put(f"NRDt{G}RankOnTest", e["rank_on_test_of_dev1_best"], 0)
    put(f"NRDt{G}PairAgree", e["pair_sign_agreement_dev2_test"], 2)
    put(f"NRDt{G}StoreyBhDevOne", e["storey_vs_bh"]["dev1"], 2)
    put(f"NRDt{G}StoreyBhDevTwo", e["storey_vs_bh"]["dev2"], 2)
    put(f"NRDt{G}StoreyBhTest", e["storey_vs_bh"]["test"], 2)
    put(f"NRDt{G}QDevOne", e["q0.2_vs_q0.1"]["dev1"], 2)
    put(f"NRDt{G}QDevTwo", e["q0.2_vs_q0.1"]["dev2"], 2)
    put(f"NRDt{G}QTest", e["q0.2_vs_q0.1"]["test"], 2)
    put(f"NRDt{G}SpPpOnly", e["spearman_dev2_test_pp_only"], 2)
put("NRDtNCand", DS_["n"], 0)
for c_, tg in (("xfit|storey|q0.05|pp", "QFive"), ("xfit|storey|q0.2|pp", "QTwenty")):
    v = DS_["near"]["values"][c_]
    put(f"NRDt{tg}DevOneNear", v[0], 2)
    put(f"NRDt{tg}DevTwoNear", v[1], 2)
    put(f"NRDt{tg}TestNear", v[2], 2)
assert N["NRDtQFiveDevOneNear"] - N["NRDtQTwentyDevOneNear"] > 3 and N["NRDtQFiveTestNear"] < N["NRDtQTwentyTestNear"]   # the reversal the text states
for key, tg in (("dev1|near", "DevOneNear"), ("dev1|far", "DevOneFar"), ("dev2|near", "DevTwoNear"), ("dev2|far", "DevTwoFar")):
    e = FC[key]
    put(f"NRFc{tg}PFourteen", e["mean_p14_sigma"], 2)
    put(f"NRFc{tg}PSixteen", e["mean_p16_sigma"], 2)
    put(f"NRFc{tg}Range", e["mean_range_over_orders"], 2)
put("NRFcRangeMin", min(FC[k]["mean_range_over_orders"] for k in FC), 2)
# the spread over the four fold assignments per draw (three of Phase 14 + the one of Phases 16/17)
_r4, _dev = [], []
for _k, _e in FC.items():
    for _row in _e["rows"]:
        _vals = _row["p14_sigma_per_order"] + [_row["p16_sigma"]]
        _r4.append(max(_vals) - min(_vals))
        _dev.append(abs(_row["p16_sigma"] - float(np.mean(_row["p14_sigma_per_order"]))))
put("NRFcRangeFourMean", float(np.mean(_r4)), 2)
put("NRFcRangeFourMax", float(np.max(_r4)), 2)
put("NRFcDevSixteenMax", float(np.max(_dev)), 2)
put("NRFcRangeMax", max(FC[k]["mean_range_over_orders"] for k in FC), 2)
# the Phase 16 adoption rule judged with interval upper ends (what an interval rule would have decided)
CONF16 = json.load(open(D / "p16_dev2_confirmation.json"))
put("NRHsUpperNear", S16["dev_dev2"]["near"]["h16|s|none"]["dFPR95_vs_sigma"]["hi"])
put("NRHsUpperFar", S16["dev_dev2"]["far"]["h16|s|none"]["dFPR95_vs_sigma"]["hi"])
assert N["NRHsUpperNear"] > 0.5 and N["NRHsUpperFar"] > 0.5
N19 = json.load(open(D / "numbers19.json"))
_hd = [N19["NHsTestNearD"], N19["NHsTestFarD"], N19["NHsTestFourD"]]
put("NRHsTestMinDiff", min(_hd), 2)
put("NRHsTestMaxDiff", max(_hd), 2)
put("NRHsTestTinsNearDiff", N19["NHsTestNearTinsD"], 2)
for _k in ("NHsTestNearDAuc", "NHsTestFarDAuc", "NHsTestFourDAuc", "NHsInTwoHundredNearDAuc", "NHsInTwoHundredNearD"):
    assert N19[_k] < 0, _k
    put("NRAbs" + _k[1:] + "Val", -N19[_k], 2)

# ===================================================================== dev table (appendix): sigma_ref vs sigma on dev2 (rule), dev1, and the benchmark
lines = []
for lab, key in (("dev2 near（採否の規則）", "DevTwoNear"), ("dev2 far（採否の規則）", "DevTwoFar"), ("dev1 near", "DevOneNear"), ("dev1 far", "DevOneFar")):
    lines.append(f"{lab} & {N['NR' + key + 'SigAuroc']:.2f} & {N['NR' + key + 'SigFpr']:.2f} & {N['NR' + key + 'RefAuroc']:.2f} & {N['NR' + key + 'RefFpr']:.2f} & "
                 f"{N['NR' + key + 'D']:+.2f} [{N['NR' + key + 'DLo']:+.2f}, {N['NR' + key + 'DHi']:+.2f}] \\\\")
lines.append("\\hline")
for lab, sec, g, base in (("ImageNet-1K near-OOD", "test_openood", "near", "none"), ("ImageNet-1K far-OOD", "test_openood", "far", "none"), ("Four-OOD", "test_fourood", "fourood", "none"),
                          ("ImageNet-200 near-OOD", "in200", "near", "none"), ("ImageNet-200 far-OOD", "in200", "far", "none"),
                          ("CIFAR-100 near-OOD", "cifar", "near", "none"), ("CIFAR-100 far-OOD", "cifar", "far", "none"),
                          ("ImageNet-1K near-OOD $\\times$TINS", "test_openood", "near", "TINS"), ("ImageNet-1K far-OOD $\\times$TINS", "test_openood", "far", "TINS"),
                          ("Four-OOD $\\times$TINS", "test_fourood", "fourood", "TINS")):
    a, b = S[sec][g][f"{SIG}|{base}"], S[sec][g][f"{REF}|{base}"]
    d = b["dFPR95_vs_sigma"]
    if lab.startswith("ImageNet-1K near-OOD $\\times$"):
        lines.append("\\hline")
    lines.append(f"{lab} & {a['AUROC']:.2f} & {a['FPR95']:.2f} & {b['AUROC']:.2f} & {b['FPR95']:.2f} & {d['mean']:+.2f} [{d['lo']:+.2f}, {d['hi']:+.2f}] \\\\")
t = ("{\\fontsize{6.5pt}{7.6pt}\\selectfont\\renewcommand{\\arraystretch}{1.0}%\n\\setlength{\\tabcolsep}{2.2pt}\\begin{tabular}{@{}lcc|cc|c@{}}\n\\hline\n"
     "\\rule[-2.0pt]{0pt}{8.6pt} & \\multicolumn{2}{c|}{$\\sigma$（交差適合）} & \\multicolumn{2}{c|}{$\\sR$（本稿）} & \\\\\n"
     "\\rule[-2.0pt]{0pt}{8.6pt}設定 & \\AUC & \\FPR & \\AUC & \\FPR & \\FPR{}の差 [95\\%区間] \\\\\n\\hline\n" + "\n".join(lines) + M.TAIL)
(SEC / "t22_dev.tex").write_text(t)

# ===================================================================== dev-test agreement table (appendix): the 18 Phase 14 candidates
vals = DS_["near"]["values"]
valsf = DS_["far"]["values"]
order = sorted(vals, key=lambda c: vals[c][0])
RJ = {"storey": "Storey型", "bh": "単純BH"}
MJ = {"pp": "$\\pP\\pN$", "pps": "$\\pP\\pN\\times$静的$p$", "min": "$\\min(\\cdot)$"}
lines = []
te_rank = {c: i + 1 for i, c in enumerate(sorted(vals, key=lambda c: vals[c][2]))}
te_rank_f = {c: i + 1 for i, c in enumerate(sorted(valsf, key=lambda c: valsf[c][2]))}
for c in order:
    _, rule, qk, mode = c.split("|")
    v, vf = vals[c], valsf[c]
    lines.append(f"{RJ[rule]}，$q{{=}}{qk[1:]}$，{MJ[mode]} & {v[0]:.2f} & {v[1]:.2f} & {v[2]:.2f} & {te_rank[c]} & {vf[0]:.2f} & {vf[1]:.2f} & {vf[2]:.2f} & {te_rank_f[c]} \\\\")
t = ("{\\fontsize{6.4pt}{7.5pt}\\selectfont\\renewcommand{\\arraystretch}{1.0}%\n\\setlength{\\tabcolsep}{2.4pt}\\begin{tabular}{@{}l|cccc|cccc@{}}\n\\hline\n"
     "\\rule[-2.0pt]{0pt}{8.2pt} & \\multicolumn{4}{c|}{near（\\FPR{}[\\%]）} & \\multicolumn{4}{c}{far（\\FPR{}[\\%]）} \\\\\n"
     "\\rule[-2.0pt]{0pt}{8.2pt}候補（$\\sigma$，dev1の順） & dev1 & dev2 & テスト & 順位 & dev1 & dev2 & テスト & 順位 \\\\\n\\hline\n" + "\n".join(lines) + M.TAIL)
(SEC / "t22_devtest.tex").write_text(t)
put("NRDtFarRankChosen", te_rank_f["xfit|storey|q0.2|pp"], 0)
put("NRDtNearRankChosen", te_rank["xfit|storey|q0.2|pp"], 0)
assert N["NRDtNearRankChosen"] == N["NRDtNearRankOnTest"]   # near: the chosen candidate is the dev1-near best

# ===================================================================== relabelled copies of tables of the sections that were not rerun (sigma is no longer the paper's read-out)
for src, dst in (("t18_rec_ninco.tex", "t22x_rec_ninco.tex"), ("t18_rec_ssb_hard.tex", "t22x_rec_ssb_hard.tex"), ("t18_views.tex", "t22x_views.tex"), ("t19_h16.tex", "t22x_h16.tex")):
    t = (SEC / src).read_text()
    n_before = t.count("（本稿）") + t.count("（本文）")
    t = t.replace("$\\sigma$（本稿）", "$\\sigma$（交差適合）").replace("$\\sigma$（本文）", "$\\sigma$（交差適合）")
    assert t.count("（本稿）") + t.count("（本文）") == 0 and n_before > 0, src
    (SEC / dst).write_text(t)

# ===================================================================== write numbers22.tex
SIGNED_SUFFIX = ("D", "DLo", "DHi", "DSig", "DSigLo", "DSigHi", "DAucSig", "TinsD", "MeanD", "DvsRef", "DLp", "DLpLo", "DLpHi", "QminusU",
                 "StoreyBhDevOne", "StoreyBhDevTwo", "StoreyBhTest", "QDevOne", "QDevTwo", "QTest", "MeanDSig")
lines = []
for k, v in N.items():
    if FMT.get(k) is None:
        lines.append(f"% {k} = {v}")
    elif FMT[k] == 0:
        lines.append(f"\\newcommand\\{k}{{{int(round(v))}}}")
    elif k.endswith(SIGNED_SUFFIX) and not k.endswith("Abs"):
        lines.append(f"\\newcommand\\{k}{{{v:+.{FMT[k]}f}}}")
    else:
        lines.append(f"\\newcommand\\{k}{{{v:.{FMT[k]}f}}}")
(SEC / "numbers22.tex").write_text("\n".join(lines) + "\n")
(D / "numbers22.json").write_text(json.dumps({k: v for k, v in N.items()}, indent=1, default=str) + "\n")
print(json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in N.items()}, indent=0, ensure_ascii=False))
