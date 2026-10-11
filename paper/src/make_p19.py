"""Paper v19 = v18 + the registered Phase 16 variant (the one-quantity read-out h16) in the appendix.
Runs nothing of v18 (make_p18.py still writes sections/numbers.tex and the t18_* tables); writes the Phase 16 numbers
(sections/numbers19.tex, data/numbers19.json) and the table sections/t19_h16.tex from
  data/p16_summary.json            (ana16.py --summary of the registered runs),
  data/p16_dev1_selection.json     (the registered base-combination rule on dev1),
  data/p16_dev2_confirmation.json  (H1-H3 on dev2 and the registered adoption flag).
Every number of the Phase 16 appendix text comes from these macros."""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
D = HERE / "data"
S = json.loads((D / "p16_summary.json").read_text())
SEL = json.loads((D / "p16_dev1_selection.json").read_text())
CONF = json.loads((D / "p16_dev2_confirmation.json").read_text())
REF = "xfit|storey|q0.2|pp|pooled"
PF = "xfit|storey|q0.2|pp"
H = "h16|s"
COMB = SEL["chosen"]                      # base combination chosen on dev1 ('h16|p' = s0 * p)
assert COMB in ("h16|s", "h16|p")
N, FMT = {}, {}


def put(name, val, dec=2):
    assert name.isalpha() and name not in N, name
    N[name] = val
    FMT[name] = dec


def grp(section, g, key):
    return S[section][g][key]


def block(prefix, section, g, base="none", hkey=H):
    e_h, e_s = grp(section, g, f"{hkey}|{base}"), grp(section, g, f"{REF}|{base}")
    put(prefix + "Fpr", e_h["FPR95"])
    put(prefix + "Auroc", e_h["AUROC"])
    put(prefix + "SigFpr", e_s["FPR95"])
    put(prefix + "SigAuroc", e_s["AUROC"])
    d = e_h["dFPR95_vs_sigma"]
    put(prefix + "D", d["mean"])
    put(prefix + "Lo", d["lo"])
    put(prefix + "Hi", d["hi"])
    put(prefix + "DAuc", e_h["dAUROC_vs_sigma"]["mean"])
    return e_h, e_s


# ---- dev1 selection (registered rule) and dev2 confirmation (H1-H3)
put("NHsDevOneTinsNearS", SEL["candidates"]["h16|s"]["near_x_tins"])
put("NHsDevOneTinsFarS", SEL["candidates"]["h16|s"]["far_x_tins"])
put("NHsDevOneTinsNearP", SEL["candidates"]["h16|p"]["near_x_tins"])
put("NHsDevOneTinsFarP", SEL["candidates"]["h16|p"]["far_x_tins"])
put("NHsDevOneTinsSigNear", SEL["sigma_x_tins"]["near"])
put("NHsDevOneTinsSigFar", SEL["sigma_x_tins"]["far"])
for g, G, Hk in (("near", "Near", "H1"), ("far", "Far", "H2")):
    block(f"NHsDevTwo{G}", "dev_dev2", g)
    assert abs(N[f"NHsDevTwo{G}D"] - CONF[Hk]["diff_mean"]) < 1e-9
    put(f"NHsDevTwo{G}DAucLo", CONF[Hk]["dAUROC_ci"]["lo"])
    put(f"NHsDevTwo{G}DAucHi", CONF[Hk]["dAUROC_ci"]["hi"])
    put(f"NHsDevTwo{G}TinsFpr", CONF[Hk]["x_tins"]["h16"])
    put(f"NHsDevTwo{G}TinsSigFpr", CONF[Hk]["x_tins"]["sigma"])
put("NHsDevTwoFprFive", CONF["H3"]["fpr@0.05"], 3)
put("NHsDevTwoFprTwenty", CONF["H3"]["fpr@0.2"], 3)
N["adopt_by_registered_rule"] = bool(CONF["adopt_h16_as_main"])
assert CONF["H1"]["holds"] and CONF["H2"]["holds"] and CONF["H3"]["holds"] and CONF["adopt_h16_as_main"]

# ---- benchmarks (6 draws), standalone and x TINS with the dev1-chosen combination
for section, g, G in (("test_openood", "near", "Near"), ("test_openood", "far", "Far"), ("test_fourood", "fourood", "Four")):
    block(f"NHsTest{G}", section, g)
    block(f"NHsTest{G}Tins", section, g, base="TINS", hkey=COMB)
    v = S[section]["validity"][g]
    put(f"NHsTest{G}ValFive", v["h16|p"]["fpr@0.05"], 3)
    put(f"NHsTest{G}SigValFive", v[PF]["fpr@0.05"], 3)
for section, g, G in (("in200", "near", "InTwoHundredNear"), ("in200", "far", "InTwoHundredFar"), ("cifar", "near", "CifarNear"), ("cifar", "far", "CifarFar")):
    block(f"NHs{G}", section, g)

# ---- recurrence grid: cells where h16 beats sigma (standalone), and the singleton cells
rec = S["rec"]
cells = sorted({tuple(k.split("|")[:3]) for k in rec})
better, total, sing = 0, 0, []
for ds, c, r in cells:
    a, b = rec.get(f"{ds}|{c}|{r}|{H}|none"), rec.get(f"{ds}|{c}|{r}|{REF}|none")
    if a is None or b is None:
        continue
    total += 1
    better += int(a["FPR95"] < b["FPR95"])
    if int(r[1:]) <= 2:
        sing.append(a["FPR95"] - b["FPR95"])
put("NHsRecNCells", total, 0)
put("NHsRecNBetter", better, 0)
put("NHsRecSingletonMeanD", sum(sing) / len(sing), 1)
put("NHsRecNSingleton", len(sing), 0)
for tag, key in (("NHsRecNincoCSixtyFourROne", "ninco|c64|r1"), ("NHsRecNincoCSixtyFourRThirty", "ninco|c64|r30")):
    put(tag, rec[f"{key}|{H}|none"]["FPR95"], 1)
    put(tag + "Sig", rec[f"{key}|{REF}|none"]["FPR95"], 1)

# ---- the dev1 exploration (Phase 16, before the registration), from the theory note's generated numbers
X = json.loads((D / "p16_exploration_numbers.json").read_text())
put("NHsExplNCfg", X["NNCfg"], 0)
for src, dst in (("NAbsOneFar", "NHsExplAbsOneFar"), ("NAbsOneNear", "NHsExplAbsOneNear"), ("NSigFar", "NHsExplSigFar"), ("NSigNear", "NHsExplSigNear"),
                 ("NFinFar", "NHsExplFinFar"), ("NFinNear", "NHsExplFinNear"), ("NDegAucFarL", "NHsExplDegAucFarVTwo"), ("NDegAucFarD", "NHsExplDegAucFarVThree"),
                 ("NDegAucNearL", "NHsExplDegAucNearVTwo"), ("NDegAucNearD", "NHsExplDegAucNearVThree")):
    put(dst, float(X[src]), 1 if "DegAuc" in dst else 2)

SIGNED = ("D", "Lo", "Hi", "DAuc", "DAucLo", "DAucHi", "MeanD")
# ---- write numbers19.tex
lines = []
for k, v in N.items():
    if isinstance(v, bool):
        lines.append(f"% {k} = {v}")
    elif FMT.get(k, 2) == 0:
        lines.append(f"\\newcommand\\{k}{{{int(round(v))}}}")
    elif k.endswith(SIGNED) and not k.endswith("Fpr"):
        lines.append(f"\\newcommand\\{k}{{{v:+.{FMT[k]}f}}}")
    else:
        lines.append(f"\\newcommand\\{k}{{{v:.{FMT[k]}f}}}")
(HERE / "sections" / "numbers19.tex").write_text("\n".join(lines) + "\n")
(D / "numbers19.json").write_text(json.dumps(N, indent=1) + "\n")


# ---- table t19_h16.tex
def f2(x):
    return f"{x:.2f}"


def sg(x):
    return f"{x:+.2f}"


rows = [("dev2 near（登録した確認）", "NHsDevTwoNear"), ("dev2 far（登録した確認）", "NHsDevTwoFar"),
        ("ImageNet-1K near-OOD", "NHsTestNear"), ("ImageNet-1K far-OOD", "NHsTestFar"), ("Four-OOD", "NHsTestFour"),
        ("ImageNet-200 near-OOD", "NHsInTwoHundredNear"), ("ImageNet-200 far-OOD", "NHsInTwoHundredFar"),
        ("CIFAR-100 near-OOD", "NHsCifarNear"), ("CIFAR-100 far-OOD", "NHsCifarFar"),
        (r"ImageNet-1K near-OOD $\times$TINS", "NHsTestNearTins"), (r"ImageNet-1K far-OOD $\times$TINS", "NHsTestFarTins"),
        (r"Four-OOD $\times$TINS", "NHsTestFourTins")]
L = [r"{\fontsize{6.5pt}{7.6pt}\selectfont\renewcommand{\arraystretch}{1.0}%",
     r"\setlength{\tabcolsep}{1.7pt}\begin{tabular}{@{}lcc|cc|c@{}}", r"\hline",
     r"\rule[-2.0pt]{0pt}{8.6pt} & \multicolumn{2}{c|}{$\sigma$（本文）} & \multicolumn{2}{c|}{一つの量$h$} & \\",
     r"\rule[-2.0pt]{0pt}{8.6pt}設定 & \AUC & \FPR & \AUC & \FPR & \FPR{}の差 [95\%区間] \\", r"\hline"]
for i, (label, p) in enumerate(rows):
    if i in (2, 5, 9):
        L.append(r"\hline")
    L.append(f"{label} & {f2(N[p + 'SigAuroc'])} & {f2(N[p + 'SigFpr'])} & {f2(N[p + 'Auroc'])} & {f2(N[p + 'Fpr'])} & "
             f"{sg(N[p + 'D'])} [{sg(N[p + 'Lo'])}, {sg(N[p + 'Hi'])}] \\\\")
L += [r"\hline", r"\end{tabular}}%"]
(HERE / "sections" / "t19_h16.tex").write_text("\n".join(L) + "\n")
print(json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in N.items()}, indent=0, ensure_ascii=False))
