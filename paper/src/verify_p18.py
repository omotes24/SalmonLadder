"""Checks every number and claim stated in the text of the paper v18 against the data files.
Macros (\\N...) are expanded from sections/numbers.tex; literal numbers are checked against data/published.json,
data/p11_summary.json (measured baselines), data/p14_summary.json (Phase 14) and data/numbers.json.
Exit code 1 and a list of mismatches when something differs."""
import json
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
N = json.load(open(HERE / "data" / "numbers.json"))
PUB = json.load(open(HERE / "data" / "published.json"))
S11 = json.load(open(HERE / "data" / "p11_summary.json"))
S12 = json.load(open(HERE / "data" / "p12_summary.json"))
S13 = json.load(open(HERE / "data" / "p13_summary.json"))
S14 = json.load(open(HERE / "data" / "p14_summary.json"))
SEL14 = json.load(open(HERE / "data" / "sel14_summary.json"))
OO_NEAR, OO_FAR, FOUR = ["ssb_hard", "ninco"], ["inaturalist", "textures", "openimageo"], ["inat", "sun", "places", "dtd"]
CFG = "xfit|storey|q0.2|pp"                      # selected configuration: p_fin rows
SC = CFG + "|pooled"                              # the fold-pooled statistic sigma: the score of every table
MAIN, ALONE = SC + "|TINS", SC + "|none"
PF_MAIN, PF_ALONE = CFG + "|TINS", CFG + "|none"
V1, LP, STATIC = "v1|storey|q0.2|pp", "xfit|lp|pooled", "xfit|static|pooled"
TT, TV = S14["test_train"], S14["test_val"]


def mean(v):
    return sum(v) / len(v)


def expand(text):
    macros = dict(re.findall(r"\\newcommand\\(N[A-Za-z]+)\{([^}]*)\}", (HERE / "sections" / "numbers.tex").read_text()))
    for k in sorted(macros, key=len, reverse=True):
        text = text.replace("\\" + k + "{}", macros[k]).replace("\\" + k, macros[k])
    return text


FILES = ["body_p18.tex", "sections/method18.tex", "sections/rec18.tex", "sections/fdp18.tex", "sections/appendix18.tex", "sections/in200_18.tex",
         "sections/views18.tex", "sections/fig_overview18.tex", "sections/fig_graph18.tex", "preamble_p18.tex",
         "sections/cap18.tex", "sections/kg18.tex", "sections/kgmain18.tex", "sections/caprec18.tex"]
S15 = json.load(open(HERE / "data" / "p15_summary.json"))
texts = {f: expand((HERE / f).read_text(encoding="utf-8")) for f in FILES}
abstract = (HERE / "sections" / "abstract18.tex").read_text(encoding="utf-8")
text = "\n".join(texts.values()) + "\n" + abstract
bad = []
assert N["main_key"] == MAIN and S14["selection"]["chosen"] == CFG + "|none"
assert TT["n_draws"] == 6 and "6通りの抽出" in text


def need(s, where=None):
    if s not in (where if where is not None else text):
        bad.append(f"missing: {s}")


def f2(x):
    return f"{x:.2f}"


def f3(x):
    return f"{x:.3f}"


def A(k, g="near", part="openood"):
    return TT[part][g if part == "openood" else "fourood"][k]["FPR95"]


m = N
# --- headline numbers (text + abstract)
need(f"{f2(m['alone_near_AUROC'])}\\%，\\FPR{{}} {f2(m['alone_near_FPR95'])}\\%")
need(f"AUROC {f2(m['alone_near_AUROC'])}\\%・FPR95 {f2(m['alone_near_FPR95'])}\\%", abstract)
need(f"far-OOD {f2(m['alone_far_AUROC'])}\\%・{f2(m['alone_far_FPR95'])}\\%", abstract)
need(f"Four-OOD {f2(m['alone_four_AUROC'])}\\%・{f2(m['alone_four_FPR95'])}\\%", abstract)
need(f"\\AUC{{}}で{f2(m['alone_gap_near_AUROC'])}ポイント，\\FPR{{}}で{f2(m['alone_gap_near_FPR95'])}ポイント更新した")
need(f"\\FPR{{}} {f2(m['near_FPR95'])}\\%，far-OOD {f2(m['far_AUROC'])}\\%，{f2(m['far_FPR95'])}\\%，Four-OOD {f2(m['four_AUROC'])}\\%，{f2(m['four_FPR95'])}\\%")   # the product paragraph
assert m["alone_gap_near_AUROC"] == round(m["alone_near_AUROC"], 2) - round(m["best_other_near_AUROC"], 2)      # gaps from the displayed values
need("82.65\\%", abstract)
need(f"{f2(m['best_other_near_AUROC'])}\\%，{f2(m['best_other_near_FPR95'])}\\%")
assert "DisCoPatch" in m["best_other_near_AUROC_label"] and "DisCoPatch" in m["best_other_near_FPR95_label"]
assert "NNGuide" in m["best_other_far_AUROC_label"] and "NNGuide" in m["best_other_far_FPR95_label"]
assert "TINS" in m["best_other_four_FPR95_label"]
need(f"NNGuide，{f2(m['best_other_far_AUROC'])}\\%，{f2(m['best_other_far_FPR95'])}\\%")
ratio = m["best_other_near_FPR95"] / m["alone_near_FPR95"]
assert abs(m["alone_near_FPR95_ratio_vs_best_other"] - ratio) < 1e-9 and f"{ratio:.1f}分の1" in text, ratio
assert abs(m["four_FPR95"] / m["best_other_four_FPR95"] - 0.5) < 0.05 and "ほぼ半分" in text, "half"
assert m["alone_four_FPR95"] < m["best_other_four_FPR95"] and "を下回る" in text          # standalone Four-OOD beats TINS too
assert m["alone_four_FPR95"] > m["four_FPR95"]
assert m["alone_near_FPR95_hi"] - m["alone_near_FPR95_lo"] <= 1.2 and "1ポイント程度" in text
e = TT["openood"]["near"][MAIN]
assert abs(e["FPR95"] - m["near_FPR95"]) < 1e-9 and e["n"] == 6
assert abs(TT["openood"]["near"][ALONE]["FPR95"] - m["alone_near_FPR95"]) < 1e-9
# --- reported values quoted literally
t = PUB["openood"]["TINS"]
need(f"\\AUC{{}}は{f2(t['near'][0])}\\%，\\FPR{{}}は{f2(t['near'][1])}\\%")
need(f"TINS（{f2(PUB['fourood']['TINS']['avg'][0])}\\%，{f2(PUB['fourood']['TINS']['avg'][1])}\\%）")
dp = PUB["openood"]["DisCoPatch"]["per"]
need(f"DisCoPatch（{dp['ssb_hard'][0]:.1f}\\%／{dp['ssb_hard'][1]:.1f}\\%）")
need(f"DisCoPatch（{dp['ninco'][1]:.1f}\\%）")
need(f"報告値の最良（{f2(m['best_other_near_FPR95'])}\\%）")
# --- measured bases and the gains
gains, meas_ = {}, {}
for b, key in (("TINS", "TINS"), ("NegLabel", "NegLabel"), ("AdaNeg", "AdaNeg"), ("TANL", "TANL")):
    tm = S11["openood"]["near"][f"-|-|{key}"]
    assert abs(m[f"meas_{b}_near_FPR95"] - tm["FPR95"]) < 1e-9, b
    row = TT["openood"]["near"][f"{SC}|{b}"]
    gain = tm["FPR95"] - row["FPR95"]
    assert abs(m["tins_gain_near_FPR95" if b == "TINS" else f"base_{b}_near_gain_FPR95"] - gain) < 1e-9 and gain > 30, b
    gains[b], meas_[b] = gain, tm["FPR95"]
raw_body = (HERE / "body_p18.tex").read_text(encoding="utf-8")
assert min(meas_, key=meas_.get) == "TINS" and max(meas_, key=meas_.get) == "NegLabel" and "\\NMeasTinsNearFpr{}--\\NMeasNeglabelNearFpr" in raw_body
assert min(gains, key=gains.get) == "TANL" and max(gains, key=gains.get) == "NegLabel" and "\\NBaseTanlNearGainFpr{}--\\NBaseNeglabelNearGainFpr" in raw_body
assert m["t1_base_rows_all_win"] == 1 and "との積もすべての列で報告値を上回り" in text
# --- intro range of FPR95 of the trained / unsupervised detectors cited there
rng = [mean([PUB["openood"][k]["per"][d][1] for d in OO_NEAR]) for k in ("NNGuide", "MOODv2", "DisCoPatch")]
need(f"{round(min(rng))}--{round(max(rng))}\\%残る")
# --- wins per data set: 17 of the 18 columns (SSB-hard FPR95 is the one loss, inside the interval)
assert m["t2_wins"] == 9 and m["t3_wins_ds"] == 8 and m["t23_wins_ds"] == 17 and m["t23_cols_ds"] == 18 and "18列のうち17列" in text
losses = [k for k, v in N.items() if (k.startswith("t2_gap_") or k.startswith("t3_gap_")) and v <= 0]
assert losses == ["t2_gap_ssb_hard_FPR95"], losses
assert m["ssb_hard_FPR95_lo"] <= dp["ssb_hard"][1] <= m["ssb_hard_FPR95_hi"] and "19.8\\%を含み" in text
assert m["ssb_hard_vs_discopatch_AUROC"] > 0 and m["ninco_vs_discopatch_FPR95"] > 0
assert m["t1_alone_wins"] == 6 and "6列すべてで報告値の最良を上回る" in text

assert all(v > 0 for k, v in N.items() if (k.startswith("t2_gap_") or k.startswith("t3_gap_")) and k.endswith("_AUROC"))
# --- same-feature comparison statements (Sec. 4.2)
for k in (f"{STATIC}|TINS", f"{STATIC}|none", "knn|TINS", "knn|none"):
    assert A(k) > m["best_other_near_FPR95"], k                       # the static detectors do not reach the best reported value
for b in ("TINS", "none"):                                             # LGC-max: better than the static detectors, far behind sigma (>= 5 points)
    assert A(f"{STATIC}|{b}") > A(f"lgcmax|{b}") > A(f"{SC}|{b}") + 5, b
assert "静的な検出器よりは良いが$\\pP$に遠く及ばない" in text
assert A(f"{LP}|TINS") < m["best_other_near_FPR95"]                    # propagation alone does
for b in ("TINS", "none"):
    e = TT["openood"]["near"][f"{SC}|{b}"]
    assert e["dFPR95_vs_lp"]["hi"] < 0, b                              # seeds help on near-OOD (interval excludes 0)
    assert A(f"{LP}|{b}", "far") > A(f"{SC}|{b}", "far") and A(f"{LP}|{b}", "fourood", "fourood") > A(f"{SC}|{b}", "fourood", "fourood")
    d = e["dFPR95_vs_v1"]
    assert -0.5 < d["mean"] < 0.5, (b, d)                              # cross-fitting costs at most half a point on near-OOD
assert "far-OODとFour-OODでも種は\\FPR{}を下げる" in text
assert m["far_dFPR95_vs_v1"] < 0 and m["four_dFPR95_vs_v1"] < 0 and "ポイントと改善する" in text
# p_fin as the score: same standalone, worse x TINS on near-OOD, better on Four-OOD
assert abs(m["pf_alone_near_FPR95"] - m["alone_near_FPR95"]) < 0.1 and m["pf_near_FPR95"] > m["near_FPR95"] + 2 and m["pf_four_FPR95"] < m["four_FPR95"]
assert m["pf_near_FPR95"] > m["alone_near_FPR95"] and "と単体より悪化したのを見て$\\sigma$に替えた" in text
assert "これは登録からの逸脱であり，単体の値は登録どおりである" in text and "登録時の形$\\pfin\\times s_0$ではない" in texts["body_p18.tex"] + (HERE / "sections" / "t18_main.tex").read_text()
bh = TT["openood"]["near"][f"xfit|bh|q0.2|pp|pooled|TINS"]["FPR95"]
assert bh < m["near_FPR95"] and m["near_FPR95"] - bh < 0.5 and "テスト集合ではStorey型よりわずかに良い" in text
assert A("xfit|storey|q0.2|pps|pooled|TINS") > A(MAIN) and A("xfit|storey|q0.2|pps|pooled|TINS", "far") < A(MAIN, "far") and "と悪化し" in text and "と改善する" in text
# --- arrival-time mode (appendix) quoted in Sec. 4.2
onl = S14["online"]["openood"]["near"]
k14 = f"xonl|storey|q0.2|pp|pooled|TINS"
assert onl[k14]["FPR95"] > m["near_FPR95"] and onl[k14]["FPR95"] < m["best_other_near_FPR95"] and onl[k14]["AUROC"] > m["best_other_near_AUROC"]
assert onl[k14]["n"] == 5 and S14["online"]["fourood"]["fourood"][k14]["n"] == 3 and "OpenOODは5通り，Four-OODは3通りの到着順" in text
assert S14["online"]["iters_max"] <= 3000 and "不動点" in text and abs(m["onl_iters_mean"] - S14["online"]["iters_mean"]) < 1e-9
assert abs(m["onl14_TINS_near_FPR95"] - onl[k14]["FPR95"]) < 1e-9 and abs(m["onl14_none_near_FPR95"] - onl[k14.replace("TINS", "none")]["FPR95"]) < 1e-9 and "後に届く画像を使えない分だけ劣るが，報告値の最良は上回る" in text
for grp, part, g in (("near", "openood", "near"), ("far", "openood", "far"), ("four", "fourood", "fourood")):
    v = S14["online"][part]["validity"][g]["xonl|storey|q0.2|pp"]
    assert abs(m[f"onl14_val_{grp}_fpr005"] - v["fpr@0.05"]) < 1e-9 and 0.04 < v["fpr@0.05"] < 0.08
# --- selection (Sec. 3)
sel = S12["selection"]
assert sel["_selection"]["chosen"] == "L14+D3L|T|q0.2|none" and m["sel_n_candidates"] == 24 and "24候補" in text
s14 = S14["selection"]
assert s14["chosen"] == CFG + "|none" and s14["n_candidates"] == 18 and "18候補" in text
assert s14["chosen_dev1"]["far"]["FPR95"] <= s14["reference"]["far"]["FPR95"] + 0.5
assert s14["chosen_dev1"]["near"]["FPR95"] < s14["reference"]["near"]["FPR95"] and s14["chosen_dev2"]["near"]["FPR95"] < SEL14["dev2"]["v1|storey|q0.2|pp|none"]["near"]["FPR95"]
assert "より良い" in texts["sections/appendix18.tex"]
for k, v in SEL14["dev1"].items():                                       # q = 0.05 is the worst q for every (rule, mode) on dev1 near
    if "|q0.05|" in k and k.startswith("xfit|") and k.endswith("|none") and "|pooled|" not in k:
        assert v["near"]["FPR95"] > max(SEL14["dev1"][k.replace("q0.05", "q0.1")]["near"]["FPR95"], SEL14["dev1"][k.replace("q0.05", "q0.2")]["near"]["FPR95"]), k
pps = SEL14["dev1"]["xfit|storey|q0.2|pps|none"]
assert pps["far"]["FPR95"] == min(v["far"]["FPR95"] for k, v in SEL14["dev1"].items() if k.startswith("xfit|") and k.endswith("|none") and "|pooled|" not in k and k.count("|") == 4)
assert pps["near"]["FPR95"] > s14["chosen_dev1"]["near"]["FPR95"]
assert "テスト集合上の探索を行ったことも開示し" in text
# --- validity (Sec. 4.4)
assert TV["n_splits"] == 3 and "\\NVcalNSplits" not in text
for a, key in (("001", 0.01), ("005", 0.05), ("01", 0.1), ("02", 0.2)):
    lo, hi = m[f"vcal_fpr{a}_min"], m[f"vcal_fpr{a}_max"]
    assert abs(lo - key) < max(0.012, 0.3 * key) and abs(hi - key) < max(0.012, 0.3 * key), (a, lo, hi)
assert m["vcal_bh_fdp_max"] <= 0.2 + 0.02 and "水準以下" in text
assert m["vcal_idfrac_storey_max"] <= 0.2 + 0.03
assert abs(m["vcal_main_near_FPR95"] - m["near_FPR95"]) < 0.6 and abs(m["vcal_alone_near_FPR95"] - m["alone_near_FPR95"]) < 0.6
assert "命題4の保証は実測でも成り立つ" in text and abs(m["vcal_alone_near_FPR95"] - m["alone_near_FPR95"]) < 0.6
assert m["vtrain_fpr005_max"] > 0.05 and m["vtrain_bh_fdp_max"] > 0.2 and "水準を超える" in text
# --- recurrence grid statements (Sec. 4.3)
R = S14["rec"]
assert m["rec_n_r_le5"] == 30 and m["rec_n_r_ge10"] == 30 and m["rec_n_r_le5_main_worse_static"] >= 25 and m["rec_n_r_ge10_main_better_static"] >= 25
assert m["rec_ninco_c64_r1_main_FPR95"] > m["rec_ninco_c64_r1_static_FPR95"]
assert m["rec_ninco_c64_r10_main_FPR95"] > m["rec_ninco_c64_r30_main_FPR95"] and m["rec_ninco_c64_r30_main_FPR95"] < m["rec_ninco_c64_r30_static_FPR95"]
assert m["rec_n_main_within1_v1"] / m["rec_n_all"] >= 0.8 and m["rec_max_main_minus_v1"] < 3.0 and m["rec_min_main_minus_v1"] > -5.0 and "条件で1ポイント以内である" in text
assert m["rec_n_r_le5_pps_worse_static"] < m["rec_n_r_le5_main_worse_static"] and m["rec_ninco_c64_r1_pps_FPR95"] < m["rec_ninco_c64_r1_main_FPR95"]
assert m["rec_ninco_c64_r30_pps_FPR95"] > m["rec_ninco_c64_r30_main_FPR95"] and m["rec_max_pps_minus_static_le5"] < m["rec_max_main_minus_static_le5"]
assert m["rec_fpr005_min"] > 0.03 and m["rec_fpr005_max"] < 0.08 and "0.05をやや超える" in text
e = R["ssb_hard|c980|r50"]
assert e["n_ood"] == 49000 and "SSB-hard（980クラス，各50枚）" in text and "NINCO（64クラス）" in text
for key in (ALONE, f"{LP}|none", f"{STATIC}|none", f"{V1}|none", "xfit|storey|q0.2|pps|pooled|none"):
    assert key in R["ninco|c64|r30"]["cfg"], key                         # the figure reads these rows
# --- ImageNet-200 (Sec. 4.5): beats the reported best and the static detectors, but seeds do not help (p+ only and LGC-max are better)
I2 = S14["in200"]
assert I2["n_draws"] == 6
for g in ("near", "far"):
    e = I2[g][ALONE]
    assert e["AUROC"] > m[f"in200_best_pub_{g}_AUROC"]
    for b in (f"{STATIC}|none", "knn|none"):
        assert e["AUROC"] > I2[g][b]["AUROC"] and e["FPR95"] < I2[g][b]["FPR95"], (g, b)
    assert I2[g][f"{LP}|none"]["FPR95"] < e["FPR95"] and I2[g]["lgcmax|none"]["FPR95"] < e["FPR95"], g
assert "$\\pN$を加えた本稿の構成はこれらより悪い" in text
assert I2["near"]["xfit|bh|q0.2|pp|pooled|none"]["FPR95"] < I2["near"][ALONE]["FPR95"] and abs(I2["near"]["xfit|bh|q0.2|pp|pooled|none"]["FPR95"] - I2["near"][f"{LP}|none"]["FPR95"]) < 0.5
sb = I2["seeds"]["ssb_hard|D3L|storey|q0.2"]
assert sb["seeds_over_n_test"] > 0.99 and sb["pi0_hat"] < 0.25 and I2["sizes"]["ssb_hard"]["n_id"] == 9000 and "9,000枚" in text
assert I2["seeds"]["ssb_hard|D3L|bh|q0.2"]["seeds_over_n_test"] < 0.95
assert m["in200_best_pub_near_AUROC"] == 84.84 and m["in200_best_pub_far_AUROC"] == 95.01
assert abs(m["in200_mix_ssb_hard"] - 100 * 49000 / 58000) < 0.5
# --- CIFAR-100 (Sec. 4.5)
CF = S14["cifar"]
assert CF["n_draws"] == 6 and "重ならない6通りの抽出，較正画像$n{=}400$" in text
for g in ("near", "far"):
    e = CF[g][f"L14+D3L|{ALONE}"]
    assert e["AUROC"] > m[f"cifar_best_pub_{g}_AUROC"] + 10, g                                   # "10ポイント以上上回り"
    for b in (f"L14+D3L|{STATIC}|none", "L14+D3L|knn|none"):
        assert e["FPR95"] < CF[g][b]["FPR95"] and e["AUROC"] > CF[g][b]["AUROC"], (g, b)
assert m["cifar_best_pub_near_label"] == "OE" and m["cifar_best_pub_far_label"] == "RotPred" and m["cifar_best_pub_near_AUROC"] == 88.30 and m["cifar_best_pub_far_AUROC"] == 88.40
assert m["cifar_best_pub_near_FPR95"] == 30.73 and m["cifar_best_pub_far_FPR95"] == 34.50 and m["in200_best_pub_near_FPR95"] == 52.30 and m["in200_best_pub_far_FPR95"] == 21.22
assert "非公開" not in text
assert PUB["cifar100"]["OE"]["group"] == "train_outlier" and "外れ値データで学習" in text
e = CF["near"][f"L14+D3L|{ALONE}"]["dFPR95_vs_lp"]
assert e["lo"] < 0 < e["hi"] and abs(e["mean"]) < 0.1, e                                           # near: no difference to p+ only
e = CF["far"][f"L14+D3L|{ALONE}"]["dFPR95_vs_lp"]
assert e["lo"] > 0.5, e                                                                           # far: seeds hurt
assert abs(e["mean"] - (CF["far"][f"L14+D3L|{ALONE}"]["FPR95"] - CF["far"][f"L14+D3L|{LP}|none"]["FPR95"])) < 1e-9       # paired mean = difference of the means (same draws)
assert "far-OODでは$\\pN$が害になる" in text and "ここでも利得は$\\pP$の伝播にあり，種は効かない" in text
pdd = CF["per_dataset"]
assert pdd[f"L14+D3L|{LP}|none|places365"]["FPR95"] < pdd["L14+D3L|xfit|bh|q0.2|pp|pooled|none|places365"]["FPR95"] < pdd[f"L14+D3L|{ALONE}|places365"]["FPR95"]
hi = [d for d in ("mnist", "svhn", "places365") if CF["sizes"][d]["mixing"] > 0.6]
assert len(hi) == 3 and m["cifar_n_far_hi"] == 3 and CF["sizes"]["texture"]["mixing"] < 0.5 and CF["sizes"]["cifar10"]["mixing"] < 0.6 and CF["sizes"]["tin"]["mixing"] < 0.5
assert m["cifar_seed_storey_idfrac_max"] <= 0.2 and m["cifar_seed_far_hi_storey_ratio_min"] > 90, (m["cifar_seed_storey_idfrac_max"], m["cifar_seed_far_hi_storey_ratio_min"])
assert CF["sizes"]["mnist"]["n_ood"] == 70000 and CF["sizes"]["places365"]["n_ood"] == 33773 and CF["sizes"]["svhn"]["n_ood"] == 26032 and CF["sizes"]["cifar10"]["n_id"] == 9000
for g in ("near", "far"):
    v = CF["validity"][g][f"L14+D3L|{CFG}"]
    assert abs(m[f"cifar_val_{g}_fpr005"] - v["fpr@0.05"]) < 1e-9 and 0.04 < v["fpr@0.05"] < 0.07 and v["bh_fdp"] < 0.2
# --- backbone families (appendix)
VW = S14["views"]
assert set(VW["views"]) == {"B14", "L14", "D3B", "D3L", "CLIP", "DINO1"} and m["views_n"] == 6
assert m["views_n_main_below_static_near"] == 6 and m["views_n_main_below_lgc_near"] == 6 and m["views_n_main_below_lp_near"] == 6
assert m["views_min_main_minus_v1_near"] < 0 < m["views_max_main_minus_v1_near"] < 1.0
# centring (Phase 15, H6 refuted): CLIP's static read-out stays far from the DINO range; far improves by > 10 points, near by < 5; DINO views move < 10 (static)
VC = S15["views_c"]
assert m["view_CLIP_staticc_near_FPR95"] > 70 and m["view_CLIP_staticc_far_FPR95"] > 40, "H6 would hold"
assert m["view_CLIP_centred_minus_raw_far_FPR95"] < -10 and -5 < m["view_CLIP_centred_minus_raw_near_FPR95"] < 0
assert m["views_dino_centred_maxabs_FPR95"] < 10 < m["views_dino_knncentred_maxabs_FPR95"] < 11 and "否定" in texts["sections/views18.tex"]
assert m["views_raw_vs_table_maxabs"] < 0.5
assert m["views_support_mean_norm_CLIP"] > 0.6 > m["views_support_mean_norm_dino_max"]
assert m["view_CLIP_static_near_FPR95"] > 90 and m["view_CLIP_static_far_FPR95"] > 90 and m["view_CLIP_mean_cos_ood"] > m["view_CLIP_mean_cos_id"]
assert m["view_CLIP_max_cos_id_mean"] > m["view_CLIP_max_cos_ood_mean"] and m["view_CLIP_textures_above_thr95_pct"] > 90
for v in VW["views"]:                                                              # the centred columns of the views table are the Phase 15 values
    tv = (HERE / "sections" / "t18_views.tex").read_text()
    assert f"{VC[v]['near']['static|centred|none']['FPR95']:.2f} & {VC[v]['far']['static|centred|none']['FPR95']:.2f}" in tv, v
# --- Phase 15: capped seed rule (Sec. 4.4) -------------------------------------------------------------------------------
C15 = S15["cap_test"]
sc_alone = C15["openood"]["near"]["xfit|storeyc|q0.2|pp|pooled|none"]
assert abs(sc_alone["FPR95"] - m["cap_storeyc_alone_near_FPR95"]) < 1e-9 and sc_alone["n"] == 6
for tag in ("cap_storeyc_alone_near", "cap_storeyc_near", "cap_storeyc_alone_far", "cap_storeyc_alone_four"):           # within 0.25 of the uncapped rule (H2: 0.5)
    assert abs(m[f"{tag}_dFPR95_vs_storey"]) <= 0.25, tag
assert "いずれも上限なしと0.25ポイント以内" in text
assert m["cap_seed_ssb_hard_storey_ratio"] > m["cap_seed_ssb_hard_storeyc_ratio"] + 10 and m["cap_seed_ssb_hard_storeyc_idfrac"] < 0.1 < m["cap_seed_ssb_hard_storey_idfrac"]
assert m["cap_seed_openimageo_storey_ratio"] > m["cap_seed_openimageo_storeyc_ratio"] + 1 and abs(m["cap_seed_ninco_storey_ratio"] - m["cap_seed_ninco_storeyc_ratio"]) < 1e-9 and abs(m["cap_seed_textures_storey_ratio"] - m["cap_seed_textures_storeyc_ratio"]) < 1e-9
assert 0 < m["cap_seed_inaturalist_storey_ratio"] - m["cap_seed_inaturalist_storeyc_ratio"] < 3 and "（iNaturalistではわずか）" in text
I2c, CFc = S15["cap_in200"], S15["cap_cifar"]
assert m["in200_storeyc_near_dFPR95_vs_lp_hi"] < 0 and m["in200_storeyc_near_dFPR95_vs_storey_hi"] < -1 and abs(m["in200_storeyc_far_dFPR95_vs_lp"]) < 0.1   # IN-200: below p+ only on near, equal on far
assert m["in200_storeyc_near_FPR95"] < m["in200_lp_near_FPR95"] < m["in200_main_near_FPR95"] and "$\\pP$だけと同等以上になる" in text
assert 0 < m["cifar_storeyc_far_dFPR95_vs_lp"] <= 0.5 and m["cifar_storeyc_near_dFPR95_vs_lp_hi"] < 0 and m["cifar_storeyc_far_dFPR95_vs_storey_hi"] < 0            # H1: within +0.5 of p+ only
assert m["cifar_storeyc_near_FPR95"] < m["cifar_lp_near_FPR95"] and "ポイントに縮む" in text
for rule in ("storeyc", "bhc"):                                                    # hypotheses as registered
    assert m[f"in200_{rule}_near_dFPR95_vs_lp"] <= 0.5 and m[f"cifar_{rule}_far_dFPR95_vs_lp"] <= 0.5                 # H1 (within +0.5 of p+ only)
assert abs(m["cap_storeyc_alone_near_dFPR95_vs_storey"]) <= 0.5 and abs(m["cap_storeyc_near_dFPR95_vs_storey"]) <= 0.5   # H2
assert m["caprec_storeyc_hi_n_worse_lp1"] == 0 and m["caprec_bhc_hi_n_worse_lp1"] == 0 and m["caprec_lo_r10_max_storeyc_minus_storey"] <= 1.0 and m["caprec_n_hi"] == 2 and m["caprec_lo_r10_n"] == 28   # H3
assert m["caprec_storeyc_all_n_worse_lp1"] == 2 and m["caprec_bhc_all_n_worse_lp1"] == 0 and "NINCO $c{=}10,20$の$r{=}10$" in text
for dev in ("dev1", "dev2"):
    assert abs(m[f"capdev_{dev}_storeyc_near_FPR95"] - m[f"capdev_{dev}_storey_near_FPR95"]) < 0.1 and abs(m[f"capdev_{dev}_storeyc_tins_near_FPR95"] - m[f"capdev_{dev}_storey_tins_near_FPR95"]) < 0.1
assert m["caprec_ssb_hard_c980_r50_storey_idfrac"] > 0.2 > 0.1 > m["caprec_ssb_hard_c980_r50_storeyc_idfrac"] and m["caprec_ssb_hard_c980_r50_mix"] > 50
assert "本稿の主構成は登録どおり上限なしのStorey型" in text and "混入率が未知の運用には上限付きを推す" in text
tin, tcf = (HERE / "sections" / "t18_in200.tex").read_text(), (HERE / "sections" / "t18_cifar.tex").read_text()
for rule in ("storeyc", "bhc"):
    e = I2c["near"][f"xfit|{rule}|q0.2|pp|pooled|none"]; f_ = I2c["far"][f"xfit|{rule}|q0.2|pp|pooled|none"]
    assert f"{e['AUROC']:.2f}" in tin and f"{e['FPR95']:.2f}" in tin and f"{f_['FPR95']:.2f}" in tin
    e = CFc["near"][f"xfit|{rule}|q0.2|pp|pooled|none"]; f_ = CFc["far"][f"xfit|{rule}|q0.2|pp|pooled|none"]
    assert f"{e['FPR95']:.2f}" in tcf and f"{f_['FPR95']:.2f}" in tcf
# --- Phase 15: many-label and zero-shot baselines (Sec. 4.5) -----------------------------------------------------------
B = S15["base"]
assert B["imagenet200"]["n_train"] == 258951 and B["cifar100"]["n_train"] == 50000 and "258,951枚" in text and "50,000枚" in text
assert 75 <= 258951 / 3200 <= 85 and "約80倍" in text
for bench, tag in (("imagenet200", "in200"), ("cifar100", "cifar")):
    for meth, mk in (("knn_all_k50", "knn_all_k50"), ("ncm_all", "ncm_all"), ("mcm_clip", "mcm_clip"), ("neglabel_clip", "neglabel_clip")):
        for g in ("near", "far"):
            e = B[bench][g][meth]
            assert abs(m[f"{tag}_{mk}_{g}_FPR95"] - e["FPR95"]) < 1e-9 and abs(m[f"{tag}_{mk}_{g}_AUROC"] - e["AUROC"]) < 1e-9, (bench, meth, g)
            assert f"{e['FPR95']:.2f}" in (tin if tag == "in200" else tcf)
assert max(abs(m[f"in200_{k}_{g}_FPR95"] - m[f"in200_static_{g}_FPR95"]) for k in ("knn_all_k50", "ncm_all") for g in ("near", "far")) <= 2.5 and "と同程度にとどまり" in text
assert max(abs(m[f"cifar_{k}_{g}_FPR95"] - m[f"cifar_static_{g}_FPR95"]) for k in ("knn_all_k50", "ncm_all") for g in ("near", "far")) <= 4.0 and "から数ポイント以内にとどまり" in text
assert m["in200_knn_all_k50_near_FPR95"] > m["in200_lp_near_FPR95"] + 10 and m["in200_knn_all_k50_near_FPR95"] > m["in200_main_near_FPR95"]      # H4 second half
assert m["in200_knn_all_k50_near_FPR95"] > m["in200_static_near_FPR95"] and m["cifar_knn_all_k50_near_FPR95"] < m["cifar_static_near_FPR95"] and m["cifar_knn_all_k50_far_FPR95"] > m["cifar_static_far_FPR95"]   # H4 first half refuted
assert min(m[f"cifar_{k}_{g}_FPR95"] for k in ("mcm_clip", "neglabel_clip") for g in ("near", "far")) > 60 and "MCMとNegLabelは弱い" in text
# --- Phase 15: k_g grid (Sec. 4.3 / appendix) ----------------------------------------------------------------------------
KG = S15["kg"]
assert S15["kgs"] == [3, 5, 10, 20, 40] and m["kg_n_reps"] == 3
assert m["kg_ninco_kg40_r1_main_FPR95"] < m["kg_ninco_kg10_r1_main_FPR95"] and m["kg_ninco_kg40_r30_main_FPR95"] > m["kg_ninco_kg10_r30_main_FPR95"] + 10
assert m["kg_ninco_best_kg_rmax"] == 10 and m["kg_ssb_hard_best_kg_rmax"] == 5 and m["kg_ssb_hard_kg10_minus_best_rmax_FPR95"] < 0.5
assert m["kg_ninco_kg40_cross_r"] == 1 and m["kg_ninco_kg3_cross_r"] == m["kg_ninco_kg10_cross_r"] == 10 and m["kg_ssb_hard_kg3_cross_r"] == 2 and m["kg_ssb_hard_kg20_cross_r"] == 1     # H5 first half refuted
assert m["kg_ninco_kg3_r1_main_FPR95"] == max(m[f"kg_ninco_kg{k}_r1_main_FPR95"] for k in (3, 5, 10, 20, 40))
assert "否定され" in texts["sections/kgmain18.tex"] and "$k_g{=}10$は登録どおり変えない" in text
# --- Phase 15 registration statements
assert "83ea4061" in text and "8ef9c8d" in text and "登録からの逸脱はない" in text
for h in ("H1", "H2", "H3", "H4", "H5", "H6"):
    assert f"\\textbf{{{h}}}" in texts["sections/appendix18.tex"], h
# --- constants in the method
assert "約$0.25\\%$" in text and abs(1 / (0.2 * 2001) - 0.0025) < 1e-5
assert "$n_b=|\\mathcal{C}_b|=2{,}000$" in text
# --- vocabulary and hygiene
assert "$k{=}10$" in text and "kNN" in text
title = re.search(r"\\title\{(.*?)\}", texts["preamble_p18.tex"], re.S).group(1).replace("\\titleicon", "").replace("\\\\", " ")
assert len(title.split()) <= 14 and title.startswith(" Salmon Ladder: Conformal Label Propagation") and "Self-Labelled Seeds" in title, title
for word in ("前版", "pM", "pLP", "オンライン", "メモリ", "確かに未知", "確かな未知", "DisCoNet", "ストリーム", "15回", "再来", "Recurring"):
    assert word not in text.replace("\\pP", "").replace("\\pN", ""), word
for f, tx in texts.items():
    assert not re.search(r"\\N[A-Z][A-Za-z]+", tx), (f, "unexpanded macro", re.search(r"\\N[A-Z][A-Za-z]+", tx).group(0))
assert "omotes24/SalmonLadder" in text
abs_plain = abstract.replace("\\%", "%").strip()
assert len(abs_plain) <= 400, len(abs_plain)
log = (HERE / "main.log").read_text(errors="ignore")
assert not re.search(r"^! ", log, re.M) and "Overfull" not in log, "build log has errors or overfull boxes"
assert "undefined" not in log.replace("Font shape `JY1/mc/bx/it' undefined", "").replace("Font shape `JT1/mc/b/it' undefined", "").replace("Font shape `JY1/mc/b/it' undefined", ""), "undefined references"
# main text (incl. references) within 9 pages: the appendix starts on page 10
import subprocess
pages = subprocess.run(["pdftotext", "-layout", str(HERE / "main.pdf"), "-"], capture_output=True, text=True).stdout.split("\f")
last_ref = re.findall(r"\\bibitem\{[^}]+\} ([^:]+):", (HERE / "sections" / "refs18.tex").read_text())[-1].replace("~", " ")
assert any("参考文献" in p for p in pages[:9]) and any(last_ref[:14] in p for p in pages[:9]) and not any(last_ref[:14] in p for p in pages[9:]) and ("A." in pages[9] or "表 A" in pages[9]), "references must end on page 9"
if bad:
    print("\n".join(bad))
    raise SystemExit(1)
print(f"verify_p18: 0 mismatches; abstract {len(abs_plain)} chars; pages {len([p for p in pages if p.strip()])}")
