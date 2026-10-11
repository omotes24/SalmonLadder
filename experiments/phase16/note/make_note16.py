"""Tables and numbers of the Phase 16 theory note, generated from the dev1 exploration CSVs (harm16*.py on the server) and
from checks16.json (theory16_checks.py). Nothing in the note is typed by hand: every number is a macro written here.
Inputs : ../results_raw/all/results/{dev1,dev1_b,dev1_c,dev1_d}/dev1_draw<k>_<far|near>.csv, dev1_e/degree_dev1.csv, *.meta.json
Outputs: gen/numbers16.tex, gen/t_main.tex, gen/t_grid.tex, gen/t_deg.tex, gen/t_checks.tex, gen/numbers16.json"""
import glob
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
RAW = HERE.parent / "exploration" / "results"
GEN = HERE / "gen"
GEN.mkdir(exist_ok=True)
ROUNDS = ["dev1", "dev1_b", "dev1_c", "dev1_d"]

rows = []
for rd in ROUNDS:
    for f in sorted(glob.glob(str(RAW / rd / "dev1_draw*_*.csv"))):
        m = re.search(r"draw(\d)_(far|near)\.csv$", f)
        df = pd.read_csv(f)
        df["round"], df["draw"], df["stream"] = rd, int(m.group(1)), m.group(2)
        rows.append(df)
df = pd.concat(rows, ignore_index=True)
# every configuration must have 5 draws x 2 streams; identical configurations across rounds must agree (same code path)
cnt = df.groupby(["round", "cfg"]).size()
assert (cnt == 10).all(), cnt[cnt != 10]
dup = df.groupby(["cfg", "draw", "stream"]).FPR95.agg(["min", "max", "count"])
assert ((dup["max"] - dup["min"]).abs() < 1e-9).all(), "a configuration differs between rounds"
df = df.drop_duplicates(["cfg", "draw", "stream"])
g = df.groupby(["cfg", "stream"]).agg(AUROC=("AUROC", "mean"), FPR95=("FPR95", "mean"), f05=("fpr@0.05", "mean"), t05=("tpr@0.05", "mean"),
                                      f20=("fpr@0.2", "mean"), seeds=("seeds", "mean"), idf=("seed_idfrac", "mean"), cal_seeds=("cal_seeds", "mean"),
                                      n=("draw", "count"))
assert (g.n == 5).all()
REF = {"cur_sigma", "cur_lp"}
n_cfg = len(set(df.cfg) - REF)


def v(cfg, stream, col):
    return float(g.loc[(cfg, stream), col])


def f2(x):
    return f"{x:.2f}"


def f3(x):
    return f"{x:.3f}"


macros, nums = {}, {}


def mac(name, val, fmt=f2):
    assert re.fullmatch(r"[A-Za-z]+", name), name
    macros[name] = fmt(val) if not isinstance(val, str) else val
    nums[name] = val


KEYS = {  # macro stem -> configuration
    "Sig": "cur_sigma", "Lp": "cur_lp",
    "AbsOne": "abs1_union_l0.9", "AbsOneHi": "abs1_union_l0.99", "AbsXfit": "abs2_xfit_union_l0.90", "AbsPool": "abs2_pool_union_l0.90",
    "Harm": "abs2_pool_union_l1.00", "Hit": "hit_union", "Dir": "dir1_union_l0.9", "DirPool": "dir2_pool_union_l0.90",
    "CtHit": "ct_hit_union", "GndOne": "gnd1_union_g1.0", "GndXfit": "gnd2_xfit_union_g1.0",
    "JgOne": "jg1_union_l0.95_g1.0", "JgXfitU": "jg2_xfit_union_l0.95_g1.0", "JgXfitP": "jg2_xfit_prod_l0.95_g1.0",
    "XfitN": "xfit_unionN_l0.95_gr0.0625", "XfitP": "xfit_prod_l0.95_gr0.0625", "PoolOne": "pool1_unionN_l0.95_gr0.0625",
    "PoolP": "pool2_prod_l0.95_gr0.0625", "PoolU": "pool2_union_l0.95_gr0.0625", "Fin": "pool2_unionN_l0.95_gr0.0625",
    "FinLo": "pool2_unionN_l0.95_gr0.02", "FinLam": "pool2_unionN_l0.9_gr0.0625",
}
for stem, cfg in KEYS.items():
    for st, S in (("far", "Far"), ("near", "Near")):
        mac(f"N{stem}{S}", v(cfg, st, "FPR95"))
        mac(f"N{stem}{S}Auc", v(cfg, st, "AUROC"))
        if not np.isnan(v(cfg, st, "f05")):
            mac(f"N{stem}{S}Val", v(cfg, st, "f05"), f3)
            mac(f"N{stem}{S}ValTwo", v(cfg, st, "f20"), f3)
            mac(f"N{stem}{S}Tpr", 100 * v(cfg, st, "t05"), lambda x: f"{x:.1f}")
        if not np.isnan(v(cfg, st, "seeds")):
            mac(f"N{stem}{S}Seeds", v(cfg, st, "seeds"), lambda x: f"{x:,.0f}")
            mac(f"N{stem}{S}Idf", 100 * v(cfg, st, "idf"), lambda x: f"{x:.1f}")
        if not np.isnan(v(cfg, st, "cal_seeds")):
            mac(f"N{stem}{S}CalSeeds", v(cfg, st, "cal_seeds"), lambda x: f"{x:,.0f}")
for st, S in (("far", "Far"), ("near", "Near")):
    mac(f"NFinMinusSig{S}", v(KEYS["Fin"], st, "FPR95") - v("cur_sigma", st, "FPR95"), lambda x: f"{x:+.2f}")
    mac(f"NFinMinusSig{S}Auc", v(KEYS["Fin"], st, "AUROC") - v("cur_sigma", st, "AUROC"), lambda x: f"{x:+.2f}")
    mac(f"NAbsOneMinusSig{S}", v(KEYS["AbsOne"], st, "FPR95") - v("cur_sigma", st, "FPR95"), lambda x: f"{x:+.2f}")
mac("NNCfg", n_cfg, lambda x: str(int(x)))
for st, S in (("far", "Far"), ("near", "Near")):
    mac(f"NFinAucGap{S}", v("cur_sigma", st, "AUROC") - v(KEYS["Fin"], st, "AUROC"))
    mac(f"NAbsOneGap{S}", v(KEYS["AbsOne"], st, "FPR95") - v("cur_sigma", st, "FPR95"))

# paired differences of the unified configuration vs sigma (5 draws, paired t interval)
from scipy import stats  # noqa: E402
for st, S in (("far", "Far"), ("near", "Near")):
    a = df[(df.cfg == KEYS["Fin"]) & (df.stream == st)].sort_values("draw").FPR95.to_numpy()
    b = df[(df.cfg == "cur_sigma") & (df.stream == st)].sort_values("draw").FPR95.to_numpy()
    dlt = a - b
    se = dlt.std(ddof=1) / np.sqrt(len(dlt))
    t = stats.t.ppf(0.975, len(dlt) - 1)
    mac(f"NFinDiff{S}Lo", dlt.mean() - t * se, lambda x: f"{x:+.2f}")
    mac(f"NFinDiff{S}Hi", dlt.mean() + t * se, lambda x: f"{x:+.2f}")

# degree medians (meta of round 4)
md = {"union": [], "L14": [], "D3L": []}
for f in sorted(glob.glob(str(RAW / "dev1_d" / "*.meta.json"))):
    m = json.loads(Path(f).read_text())
    for k in md:
        md[k].append(m["deg_median"][k])
    mac("NNid", m["n_id"], lambda x: f"{int(x):,}")
    mac("NNs", m["ns"], lambda x: f"{int(x):,}")
    mac("NNc", m["nc"], lambda x: f"{int(x):,}")
far_meta = json.loads((RAW / "dev1_d" / "dev1_draw0_far.meta.json").read_text())
near_meta = json.loads((RAW / "dev1_d" / "dev1_draw0_near.meta.json").read_text())
mac("NNoodFar", far_meta["n_ood"], lambda x: f"{int(x):,}")
mac("NNoodNear", near_meta["n_ood"], lambda x: f"{int(x):,}")
mac("NMedDegLo", min(md["union"]), lambda x: f"{x:.1f}")
mac("NMedDegHi", max(md["union"]), lambda x: f"{x:.1f}")
mac("NMedDegL", float(np.mean(md["L14"])), lambda x: f"{x:.1f}")
mac("NMedDegD", float(np.mean(md["D3L"])), lambda x: f"{x:.1f}")

# floor of the seed certification (Proposition 4): fraction 1 / (q (|R| + 1)) of the pool, |R| = 2 per class on dev1
mac("NFloorFrac", 100.0 / (0.2 * (nums["NNc"] / 2 + 1)), lambda x: f"{x:.2f}")
its = []
for rd in ROUNDS:
    for f in glob.glob(str(RAW / rd / "*.meta.json")):
        its += [x[0] for x in json.loads(Path(f).read_text()).get("solver", {}).values()]
mac("NItMax", max(its), lambda x: str(int(x)))

# degree decomposition (dev1_e)
de = pd.read_csv(RAW / "dev1_e" / "degree_dev1.csv").groupby(["stream", "view"]).mean(numeric_only=True)
for st, S in (("far", "Far"), ("near", "Near")):
    for vw, V in (("L14", "L"), ("D3L", "D")):
        r = de.loc[(st, vw)]
        mac(f"NDegSp{S}{V}", r.spearman_u_sqrtdeg)
        mac(f"NDegAuc{S}{V}", r.deg_AUROC, lambda x: f"{x:.1f}")
        mac(f"NDegFpr{S}{V}", r.deg_FPR95, lambda x: f"{x:.1f}")
        mac(f"NUFpr{S}{V}", r.u_FPR95, lambda x: f"{x:.1f}")
        mac(f"NUnFpr{S}{V}", r.u_over_sqrtdeg_FPR95, lambda x: f"{x:.1f}")
        mac(f"NUAuc{S}{V}", r.u_AUROC, lambda x: f"{x:.1f}")
        mac(f"NUnAuc{S}{V}", r.u_over_sqrtdeg_AUROC, lambda x: f"{x:.1f}")


# ---------------------------------------------------------------- main table
def cell(cfg, st, col, fmt=f2):
    x = v(cfg, st, col)
    return "--" if np.isnan(x) else fmt(x)


def best_marks(cfgs):
    out = {}
    for st in ("far", "near"):
        fp = {c: v(c, st, "FPR95") for c in cfgs}
        au = {c: v(c, st, "AUROC") for c in cfgs}
        out[(st, "FPR95")] = min(fp, key=fp.get)
        out[(st, "AUROC")] = max(au, key=au.get)
    return out


GROUPS = [
    ("v18の読み出し（参照）", [("cur_sigma", r"$\sigma$（$\pP\pN$，2折）"), ("cur_lp", r"$\pP$のみ")]),
    (r"歩数の代価だけ（$\gamma{=}0$）", [("abs1_union_l0.9", r"種なし，$\lambda{=}0.9$"), ("abs2_xfit_union_l0.90", r"種（2折），$\lambda{=}0.9$"),
                                   ("abs2_pool_union_l0.90", r"種（折なし），$\lambda{=}0.9$"), ("abs2_pool_union_l1.00", r"種（折なし），$\lambda{\to}1$：調和関数"),
                                   ("hit_union", r"種なし，$\lambda{\to}1$：$-\mathbb{E}[N_{\mathcal{S}}]$"), ("dir1_union_l0.9", r"有向$k_g$近傍，種なし")]),
    (r"時間の代価だけ（$\lambda{=}1$）", [("gnd1_union_g1.0", r"種なし，$\gamma{=}1$"), ("gnd2_xfit_union_g1.0", r"種（2折），$\gamma{=}1$"),
                                    ("ct_hit_union", r"種なし，$\gamma{\to}0$：$-\mathbb{E}[T_{\mathcal{S}}]$")]),
    (r"二つの代価（$\lambda{=}0.95$，$\gamma\approx\mathrm{median}(\delta)/16$）", [("jg1_union_l0.95_g1.0", r"種なし"), ("jg2_xfit_union_l0.95_g1.0", r"種（2折），和グラフ"),
                                                             ("jg2_xfit_prod_l0.95_g1.0", r"種（2折），ビューごと＋順位の積"),
                                                             ("pool1_unionN_l0.95_gr0.0625", r"折なし，$\mathcal{S}$11/$\mathcal{R}$1/$\mathcal{C}$4"),
                                                             ("pool2_prod_l0.95_gr0.0625", r"折なし，12/2/2，ビューごと"),
                                                             ("pool2_unionN_l0.95_gr0.0625", r"\textbf{折なし，12/2/2，正規化和：一つの量}")]),
]
allc = [c for _, rs in GROUPS for c, _ in rs]
bm = best_marks(allc)
L = [r"\begin{tabular}{lcccccc}", r"\toprule",
     r" & \multicolumn{2}{c}{dev1 far} & \multicolumn{2}{c}{dev1 near} & \multicolumn{2}{c}{誤警報率（$\alpha{=}0.05$）}\\",
     r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}",
     r"構成 & \AUC$\uparrow$ & \FPR$\downarrow$ & \AUC$\uparrow$ & \FPR$\downarrow$ & far & near\\", r"\midrule"]
for gi, (gname, rs) in enumerate(GROUPS):
    L.append(r"\multicolumn{7}{l}{\textit{%s}}\\" % gname)
    for ri, (c, lab) in enumerate(rs):
        cells = []
        for st in ("far", "near"):
            for col in ("AUROC", "FPR95"):
                x = cell(c, st, col)
                if bm[(st, col)] == c:
                    x = r"\textbf{" + x + "}"
                cells.append(x)
        val = [cell(c, st, "f05", f3) for st in ("far", "near")]
        L.append(r"\quad " + f"{lab} & " + " & ".join(cells + val) + r"\\")
    L.append(r"\midrule" if gi < len(GROUPS) - 1 else r"\bottomrule")
L.append(r"\end{tabular}")
(GEN / "t_main.tex").write_text("\n".join(L) + "\n")

# ---------------------------------------------------------------- lambda x gamma grid (round 3, one walk on the union graph)
G = [r"\begin{tabular}{lccc|ccc}", r"\toprule",
     r" & \multicolumn{3}{c|}{種なし（$\mathcal{S}$だけが吸収壁）} & \multicolumn{3}{c}{種（2折）}\\",
     r"$\lambda$ \textbackslash{} $\gamma$ & 0.3 & 1 & 3 & 0.3 & 1 & 3\\", r"\midrule"]
for lam in ("0.8", "0.9", "0.95"):
    cs = []
    for fam in ("jg1_union", "jg2_xfit_union"):
        for gm in ("0.3", "1.0", "3.0"):
            c = f"{fam}_l{lam}_g{gm}"
            cs.append(f"{v(c, 'far', 'FPR95'):.1f}\\,/\\,{v(c, 'near', 'FPR95'):.1f}")
    G.append(f"{lam} & " + " & ".join(cs) + r"\\")
G += [r"\midrule", r"\multicolumn{7}{l}{参照：$\sigma$ %s\,/\,%s，$\pP$のみ %s\,/\,%s，$\gamma{=}0$・$\lambda{=}0.9$の種なし %s\,/\,%s}\\" % (
    f"{v('cur_sigma', 'far', 'FPR95'):.1f}", f"{v('cur_sigma', 'near', 'FPR95'):.1f}", f"{v('cur_lp', 'far', 'FPR95'):.1f}",
    f"{v('cur_lp', 'near', 'FPR95'):.1f}", f"{v('abs1_union_l0.9', 'far', 'FPR95'):.1f}", f"{v('abs1_union_l0.9', 'near', 'FPR95'):.1f}"),
      r"\bottomrule", r"\end{tabular}"]
(GEN / "t_grid.tex").write_text("\n".join(G) + "\n")

# ---------------------------------------------------------------- degree decomposition
Dg = [r"\begin{tabular}{llccccc}", r"\toprule",
      r"未知 & ビュー & Spearman$(u^{+},\sqrt{\delta})$ & \AUC$(\delta)$ & \FPR$(\delta)$ & \FPR$(u^{+})$ & \FPR$(u^{+}/\sqrt{\delta})$\\", r"\midrule"]
for st in ("far", "near"):
    for vw, name in (("L14", "DINOv2 L/14"), ("D3L", "DINOv3 L/16")):
        r = de.loc[(st, vw)]
        Dg.append(f"{st} & {name} & {r.spearman_u_sqrtdeg:.2f} & {r.deg_AUROC:.1f} & {r.deg_FPR95:.1f} & {r.u_FPR95:.1f} & {r.u_over_sqrtdeg_FPR95:.1f}" + r"\\")
Dg += [r"\bottomrule", r"\end{tabular}"]
(GEN / "t_deg.tex").write_text("\n".join(Dg) + "\n")

# ---------------------------------------------------------------- synthetic checks
ck = json.loads((HERE / "checks16.json").read_text())
mc = ck["mc"]
esc, cap = ck["escape"], ck["capture"]
lim = {(r["a"], r["b"], r["eps"]): r["max_rel_err"] for r in ck["limit"]}
lg = {r["lam"]: r for r in ck["lgc_limit"]}
val = ck["validity"]
sep = ck["separation"]
mac("NCkMcMaxZ", max(abs(r["solve"] - r["mc"]) / r["mc_se"] for r in mc), lambda x: f"{x:.1f}")
mac("NCkKcl", ck["energy"]["kcl_max_residual"], lambda x: f"{x:.0e}")
mac("NCkScale", ck["scale"]["max_abs_diff"], lambda x: f"{x:.0e}")
mac("NCkFates", ck["fates"]["max_abs_dev_from_1"], lambda x: f"{x:.0e}")
mac("NCkAdd", ck["additivity"]["max_abs_diff"], lambda x: f"{x:.0e}")
mac("NCkExit", max(r["exit_decomp_max_err"] for r in esc), lambda x: f"{x:.0e}")
mac("NCkPiMax", max(r["max_pi"] for r in esc))
mac("NCkPhiMin", min(r["phi"] for r in esc))
mac("NCkPhiMax", max(r["phi"] for r in esc))
sepE = ck["separation_escape"]
mac("NCkSepN", sep["n_unknown_nonseed"], lambda x: str(int(x)))
mac("NCkSepViol", sep["violations"], lambda x: str(int(x)))
mac("NCkSepEN", sepE["n"], lambda x: str(int(x)))
mac("NCkSepEViol", sepE["violations"], lambda x: str(int(x)))
mac("NCkSepELeBnd", 100 * sepE["frac_bound_le005"], lambda x: f"{x:.0f}")
mac("NCkSepELeAct", 100 * sepE["frac_actual_le005"], lambda x: f"{x:.0f}")
mac("NCkSepEPi", sepE["median_pi"], lambda x: f"{x:.3f}")
mac("NCkSepERhoOm", sepE["median_rho_omega_cal"], lambda x: f"{x:.2f}")
for k, nm in ((1e-2, "Two"), (1e-4, "Four")):
    mac(f"NCkLim{nm}", max(lim[(a, b, k)] for (a, b, e) in lim if e == k), lambda x: f"{x:.2g}")
mac("NCkLgcSpLo", lg[0.9]["spearman_u_sqrtdeg"])
mac("NCkLgcSpHi", lg[0.9999]["spearman_u_sqrtdeg"], lambda x: f"{x:.4f}")
mac("NCkLgcSpreadHi", lg[0.9999]["rel_spread_u_over_sqrtdeg"], lambda x: f"{x:.0e}")
mac("NCkReps", val["pool"]["reps"], lambda x: str(int(x)))
# theory16_checks.instance(): 2,400 ID pool images of which 400 calibration -> m0 = 2,000 ID test images, m = m0 + 200 + 200 unknowns
mac("NCkQmm", 0.2 * 2000 / 2400, f3)
for d_, D_ in (("pool", "Pool"), ("test_only", "Tonly")):
    for k, K in (("fpr005", "Five"), ("fpr010", "Ten"), ("fpr020", "Twenty"), ("fdp_bh020", "Fdp")):
        mac(f"NCk{D_}{K}", val[d_][k][0], f3)
        mac(f"NCk{D_}{K}Se", val[d_][k][1], f3)
chk = [r"\begin{tabular}{p{0.30\linewidth}p{0.42\linewidth}p{0.20\linewidth}}", r"\toprule", r"主張 & 測った量 & 結果\\", r"\midrule",
       r"(1) 線形方程式 $=$ ランダムウォーク & 解とモンテカルロ（2万回）の差の最大（標準誤差の倍数） & %s\\" % macros["NCkMcMaxZ"],
       r"(2) エネルギー最小・キルヒホッフ則 & 自由節点の電流の収支の最大残差；摂動200回がすべて増加 & %s；%s\\" % (macros["NCkKcl"], "はい" if ck["energy"]["all_perturbations_higher"] else "いいえ"),
       r"(3) 尺度の共変性 $h(cW;\lambda,c\gamma)=h(W;\lambda,\gamma)$ & 最大差 & %s\\" % macros["NCkScale"],
       r"(4) 三つの行き先の確率の和 $=1$ & 1からのずれの最大 & %s\\" % macros["NCkFates"],
       r"(5) クラスごとの到達確率の和 $=h$ & 最大差 & %s\\" % macros["NCkAdd"],
       r"(6) 出口分解（等式），$h\le\pi_Q\le\varphi$ & 分解の誤差の最大；%dの塊で$h\le\pi_Q$，$\pi_Q\le\varphi$ & %s；%s\\" % (len(esc), macros["NCkExit"], "すべて成立" if all(r["h_le_pi"] and r["pi_le_phi"] for r in esc) else "不成立あり"),
       r"(7) 捕まる確率の下界 & %dクラスで$h\ge\kappa\ge\varphi$，$\kappa\ge\rho\omega$ & %s\\" % (len(cap), "すべて成立" if all(r["kappa_ge_cf"] and r["h_ge_kappa"] and r["one_step_ge"] for r in cap) else "不成立あり"),
       r"(8) 分離（系1）& 密な未知の塊（種は各塊の5枚に1枚）の種でない%s枚：違反数，上界だけで$p\le0.05$が保証される割合（実際）；全手続きの種でない未知%s枚での違反数 & %s，%s\%%（%s\%%）；%s\\" % (macros["NCkSepEN"], macros["NCkSepN"], macros["NCkSepEViol"], macros["NCkSepELeBnd"], macros["NCkSepELeAct"], macros["NCkSepViol"]),
       r"(9) 1次の極限 $(1-h)/\varepsilon\to\mu\mathbb{E}N+\nu\mathbb{E}T$ & 相対誤差の最大（$\varepsilon{=}10^{-2}$，$10^{-4}$）& %s，%s\\" % (macros["NCkLimTwo"], macros["NCkLimFour"]),
       r"(10) LGCの$u^{+}\to\sqrt{\delta}$（$\lambda\to1$）& Spearman$(u^{+},\sqrt\delta)$（$\lambda{=}0.9\to0.9999$）& %s$\to$%s\\" % (macros["NCkLgcSpLo"], macros["NCkLgcSpHi"]),
       r"(11) 有効性（定理2），%s回 & $\alpha{=}0.05/0.1/0.2$でのIDの誤警報率；BH($q{=}0.2$)の偽発見割合 & %s/%s/%s；%s\\" % (
           macros["NCkReps"], macros["NCkPoolFive"], macros["NCkPoolTen"], macros["NCkPoolTwenty"], macros["NCkPoolFdp"]),
       r"\quad 対照：種を$\mathcal{C}$で認定（置換不変でない）& 同上 & %s/%s/%s；%s\\" % (macros["NCkTonlyFive"], macros["NCkTonlyTen"], macros["NCkTonlyTwenty"], macros["NCkTonlyFdp"]),
       r"\bottomrule", r"\end{tabular}"]
(GEN / "t_checks.tex").write_text("\n".join(chk) + "\n")

(GEN / "numbers16.tex").write_text("".join(f"\\newcommand{{\\{k}}}{{{x}}}\n" for k, x in sorted(macros.items())))
(GEN / "numbers16.json").write_text(json.dumps({k: (x if isinstance(x, (str, int)) else float(x)) for k, x in nums.items()}, indent=1) + "\n")
print(f"{len(macros)} macros; {n_cfg} configurations explored (excluding the two references)")
for k in ("NSigFar", "NSigNear", "NFinFar", "NFinNear", "NFinFarAuc", "NFinNearAuc", "NSigFarAuc", "NSigNearAuc", "NFinFarVal", "NFinNearVal",
          "NAbsOneFar", "NAbsOneNear", "NCtHitFar", "NCtHitNear", "NFinDiffFarLo", "NFinDiffFarHi", "NFinDiffNearLo", "NFinDiffNearHi"):
    print(k, macros[k])
