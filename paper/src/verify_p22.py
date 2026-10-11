"""Checks of the paper v22 (sigma_ref as the main read-out).
(1) Recomputes the Phase 17 numbers of the text independently from the per-task result files (experiments/phase17/results: CSV and
    meta.json; not via the registered summary) and compares them with the macros of sections/numbers22.tex (data/numbers22.json).
(2) Checks the claims that the text states in words (counts, orderings, 'better than', 'within', ...).
(3) Checks the form: title, authors, abstract length, forbidden words, the page limit of the main text, the one-column appendix.
The v18/v19 numbers that v22 still uses (sigma, the sections that were not rerun) are those of numbers.tex / numbers19.tex, checked by
verify_p19.py on the same data. Exit code 1 and a list of mismatches when something differs."""
import glob
import json
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

HERE = Path(__file__).resolve().parent
RES = Path(sys.argv[1]) if len(sys.argv) > 1 else (HERE.parent.parent / "experiments" / "phase17" / "results" if (HERE.parent.parent / "experiments" / "phase17" / "results").exists() else Path("/home/claude/SalmonLadder/experiments/phase17/results"))
N = json.load(open(HERE / "data" / "numbers22.json"))
PUB = json.load(open(HERE / "data" / "published.json"))
S14 = json.load(open(HERE / "data" / "p14_summary.json"))
S16 = json.load(open(HERE / "data" / "p16_summary.json"))
SIG, REF, REFP, RLP, V1 = "xfit|storey|q0.2|pp|pooled", "ref|storey|q0.2|pp|stat", "ref|storey|q0.2|pp", "ref|lp|stat", "v1|storey|q0.2|pp"
OO = {"near": ["ssb_hard", "ninco"], "far": ["inaturalist", "textures", "openimageo"]}
FOUR = ["inat", "sun", "places", "dtd"]
bad = []


def close(name, val, tol=1e-6):
    if name not in N:
        bad.append(f"macro missing: {name}")
        return
    if abs(N[name] - val) > tol:
        bad.append(f"{name}: macro {N[name]} vs recomputed {val}")


def tci(x):
    x = np.asarray(x, np.float64)
    m = float(x.mean())
    h = float(stats.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x)))
    return m, m - h, m + h


def load(pattern):
    return pd.concat([pd.read_csv(f) for f in sorted(glob.glob(str(RES / pattern)))], ignore_index=True)


# ======================================================================= (1) recomputation from the per-task files
# --- benchmarks: per draw, mean over the base orders, then over the data sets of a group
T = {"openood": load("p17_test/openood/*/*.csv"), "fourood": load("p17_test/fourood/*/*.csv")}
for k in T:
    T[k]["draw"] = T[k]["draw"].astype(str)
    assert T[k].draw.nunique() == 6, k


def group(part, sets, cfg, base):
    d = T[part]
    x = d[(d.cfg == cfg) & (d.base == base) & d.ds.isin(sets)].groupby(["draw", "ds"])[["AUROC", "FPR95"]].mean()
    assert len(x) == 6 * len(sets), (part, cfg, base, len(x))
    return x.groupby(level=0).mean()                 # per draw


GS = {"Near": ("openood", OO["near"]), "Far": ("openood", OO["far"]), "Four": ("fourood", FOUR)}
for tag, base in (("Alone", "none"), ("Tins", "TINS"), ("TANL", "TANL"), ("NegLabel", "NegLabel"), ("AdaNeg", "AdaNeg")):
    for G, (part, sets) in GS.items():
        r = group(part, sets, REF, base)
        close(f"NR{tag}{G}Auroc", r.AUROC.mean())
        close(f"NR{tag}{G}Fpr", r.FPR95.mean())
        if f"NR{tag}{G}DSig" in N:
            s = group(part, sets, SIG, base)
            m, lo, hi = tci((r.FPR95 - s.FPR95).values)
            close(f"NR{tag}{G}DSig", m)
            close(f"NR{tag}{G}DSigLo", lo)
            close(f"NR{tag}{G}DSigHi", hi)
        if f"NR{tag}{G}FprLo" in N:
            m, lo, hi = tci(r.FPR95.values)
            close(f"NR{tag}{G}FprLo", lo)
            close(f"NR{tag}{G}FprHi", hi)
# sigma recomputed here equals the Phase 14 table (the same graphs and folds)
for G, (part, sets) in GS.items():
    g = {"Near": "near", "Far": "far", "Four": "fourood"}[G]
    for base in ("none", "TINS"):
        s = group(part, sets, SIG, base)
        e = S14["test_train"][part][g][f"{SIG}|{base}"]
        if abs(s.FPR95.mean() - e["FPR95"]) > 1e-4 or abs(s.AUROC.mean() - e["AUROC"]) > 1e-4:
            bad.append(f"sigma {G} {base} differs from Phase 14")
# p+ only (reference images), v16 read-out, p_fin of sigma_ref: values and paired differences
for base, tg in (("none", "Alone"), ("TINS", "Tins")):
    r = group("openood", OO["near"], REF, base)
    lp = group("openood", OO["near"], RLP, base)
    close(f"NRLp{tg}NearFpr", lp.FPR95.mean())
    m, lo, hi = tci((r.FPR95 - lp.FPR95).values)
    close(f"NRvsLp{tg}NearD", m)
    close(f"NRvsLp{tg}NearDLo", lo)
    close(f"NRvsLp{tg}NearDHi", hi)
    v1 = group("openood", OO["near"], V1, base)
    m, lo, hi = tci((r.FPR95 - v1.FPR95).values)
    close(f"NRvsVOne{tg}NearD", m)
    close(f"NRVOne{tg}NearFpr", v1.FPR95.mean())
    close(f"NRLp{tg}FarFpr", group("openood", OO["far"], RLP, base).FPR95.mean())
    close(f"NRLp{tg}FourFpr", group("fourood", FOUR, RLP, base).FPR95.mean())
close("NRPfAloneNearFpr", group("openood", OO["near"], REFP, "none").FPR95.mean())
close("NRPfTinsNearFpr", group("openood", OO["near"], REFP, "TINS").FPR95.mean())
close("NRPfTinsFourFpr", group("fourood", FOUR, REFP, "TINS").FPR95.mean())
# per data set
DSN = {"ssb_hard": "SsbHard", "ninco": "Ninco", "inaturalist": "Inat", "textures": "Textures", "openimageo": "Openimageo"}
for ds, nm in DSN.items():
    for tg, base in (("Alone", "none"), ("Tins", "TINS")):
        x = group("openood", [ds], REF, base)
        close(f"NR{tg}{nm}Auroc", x.AUROC.mean())
        close(f"NR{tg}{nm}Fpr", x.FPR95.mean())
# decomposition (test, ImageNet-200, CIFAR-100)
I2 = load("p17_in200/*/*.csv")
CF = load("p17_cifar/*/*.csv")
for d in (I2, CF):
    d["draw"] = d["draw"].astype(str)
CFG_ = {"near": ["cifar10", "tin"], "far": ["mnist", "svhn", "texture", "places365"]}


def gmean(d, sets, cfg, base="none"):
    x = d[(d.cfg == cfg) & (d.base == base) & d.ds.isin(sets)].groupby(["draw", "ds"])[["AUROC", "FPR95"]].mean()
    assert len(x) == 6 * len(sets)
    return x.groupby(level=0).mean()


SETS = {"INNear": ("openood", OO["near"]), "INFar": ("openood", OO["far"]), "Four": ("fourood", FOUR)}
for key, tag in (("dec|deg", "Deg"), ("dec|q", "Q"), ("dec|u", "U"), ("dec|sigma_q", "SigQ"), (SIG, "Sig"), (REF, "Ref")):
    for st, (part, sets) in SETS.items():
        x = gmean(T[part], sets, key)
        close(f"NRDec{tag}{st}Auroc", x.AUROC.mean())
        close(f"NRDec{tag}{st}Fpr", x.FPR95.mean())
    for st, (d, sets) in {"ITNear": (I2, OO["near"]), "ITFar": (I2, OO["far"]), "CFNear": (CF, CFG_["near"]), "CFFar": (CF, CFG_["far"])}.items():
        x = gmean(d, sets, key)
        close(f"NRDec{tag}{st}Auroc", x.AUROC.mean())
        close(f"NRDec{tag}{st}Fpr", x.FPR95.mean())
for st in ("INNear", "INFar", "Four", "ITNear", "ITFar", "CFNear", "CFFar"):
    q, u = N[f"NRDecQ{st}Fpr"], N[f"NRDecU{st}Fpr"]
    if q >= u:
        close(f"NRDecUHelps{st}", q - u)
    else:
        close(f"NRDecUHurts{st}", u - q)
# ImageNet-200 and CIFAR-100 rows of sigma_ref
for tg, d, groups in (("IT", I2, OO), ("CF", CF, CFG_)):
    for g, G in (("near", "Near"), ("far", "Far")):
        r, s, lp = gmean(d, groups[g], REF), gmean(d, groups[g], SIG), gmean(d, groups[g], RLP)
        close(f"NR{tg}{G}Auroc", r.AUROC.mean())
        close(f"NR{tg}{G}Fpr", r.FPR95.mean())
        close(f"NR{tg}{G}LpFpr", lp.FPR95.mean())
        m, lo, hi = tci((r.FPR95 - lp.FPR95).values)
        close(f"NR{tg}{G}DLp", m)
        close(f"NR{tg}{G}DLpLo", lo)
        close(f"NR{tg}{G}DLpHi", hi)
        m, lo, hi = tci((r.FPR95 - s.FPR95).values)
        close(f"NR{tg}{G}DSig", m)
# --- dev splits: the registered adoption rule
for dev, tg in (("dev2", "DevTwo"), ("dev1", "DevOne")):
    d = load(f"p17_dev/{dev}/*.csv")
    d["draw"] = d["draw"].astype(str)
    for s_, G in (("near", "Near"), ("far", "Far")):
        def per_draw(cfg, base):
            x = d[(d.cfg == cfg) & (d.base == base) & (d.stream == s_)].groupby("draw")[["AUROC", "FPR95"]].mean()
            assert len(x) == 5
            return x
        a, b = per_draw(REF, "none"), per_draw(SIG, "none")
        close(f"NR{tg}{G}RefFpr", a.FPR95.mean())
        close(f"NR{tg}{G}SigFpr", b.FPR95.mean())
        m, lo, hi = tci((a.FPR95 - b.FPR95).values)
        close(f"NR{tg}{G}D", m)
        close(f"NR{tg}{G}DLo", lo)
        close(f"NR{tg}{G}DHi", hi)
        if tg == "DevTwo" and hi > 0.5:
            bad.append(f"dev2 {s_}: the interval rule does not hold (upper end {hi})")
    v = d[(d.cfg == REFP) & (d.base == "none")]
    assert len(v) == 10
    close(f"NR{tg}FprFive", v["fpr@0.05"].mean())
    close(f"NR{tg}FprTwenty", v["fpr@0.2"].mean())
    close(f"NR{tg}BhFdp", v["bh_fdp"].mean())
    if tg == "DevTwo" and not (v["fpr@0.05"].mean() <= 0.06 and v["fpr@0.2"].mean() <= 0.21):
        bad.append("dev2: the validity part of the rule does not hold")
# --- benchmark validity of p_fin(sigma_ref): mean over the tasks of a group
for G, (part, sets) in GS.items():
    d = T[part]
    v = d[(d.cfg == REFP) & (d.base == "none") & d.ds.isin(sets)]
    close(f"NRVal{G}FprFive", v["fpr@0.05"].mean())
# --- recurrence grid
R = load("p17_rec/*/*.csv")
cells = R.groupby(["ds", "c", "r", "cfg", "base"])[["FPR95"]].mean()
REC14 = S14["rec"]


def cell(ds, c, r, cfg, base="none"):
    return float(cells.loc[(ds, c, r, cfg, base)].FPR95)


keys = sorted({(ds, c, r) for ds, c, r, _, _ in cells.index})
assert len(keys) == 60
le5 = [k for k in keys if k[2] <= 5]
ge10 = [k for k in keys if k[2] >= 10]
st = {k: REC14[f"{k[0]}|c{k[1]}|r{k[2]}"]["cfg"]["xfit|static|pooled|none"]["FPR95"] for k in keys}
for k in keys:      # the cells are those of Phase 14 (so its static p applies)
    if abs(cell(*k, SIG) - REC14[f"{k[0]}|c{k[1]}|r{k[2]}"]["cfg"][f"{SIG}|none"]["FPR95"]) > 1e-6:
        bad.append(f"grid cell {k}: sigma differs from Phase 14")
close("NRRecLeFiveWorseStatic", sum(cell(*k, REF) > st[k] for k in le5))
close("NRRecLeFiveMeanD", float(np.mean([cell(*k, REF) - st[k] for k in le5])))
close("NRRecLeFiveMaxD", float(np.max([cell(*k, REF) - st[k] for k in le5])))
for rr, W in ((1, "One"), (2, "Two"), (5, "Five")):
    ks = [k for k in le5 if k[2] == rr]
    close(f"NRRecR{W}MeanD", float(np.mean([cell(*k, REF) - st[k] for k in ks])))
close("NRRecGeTenBetterStatic", sum(cell(*k, REF) < st[k] for k in ge10))
close("NRRecGeTenMeanD", float(np.mean([cell(*k, REF) - st[k] for k in ge10])))
dd = [cell(*k, REF) - cell(*k, SIG) for k in keys]
close("NRRecMaxAbsDiffSig", float(np.max(np.abs(dd))))
close("NRRecNWithinOneSig", sum(abs(x) <= 1.0 + 1e-9 for x in dd))
for k, tag in ((("ninco", 64, 1), "NincoCSixtyFourROne"), (("ninco", 64, 10), "NincoCSixtyFourRTen"), (("ninco", 64, 30), "NincoCSixtyFourRThirty"),
               (("ssb_hard", 980, 10), "SsbCNineEightyRTen"), (("ssb_hard", 980, 50), "SsbCNineEightyRFifty")):
    close(f"NRRec{tag}", cell(*k, REF))
    close(f"NRRec{tag}Static", st[k])
dh = [S16["rec"][f"{k[0]}|c{k[1]}|r{k[2]}|h16|s|none"]["FPR95"] - cell(*k, REF) for k in keys if k[2] <= 2]
close("NRRecHvsRefLeTwoMeanD", float(np.mean(dh)))
# seeds of sigma_ref against the mixing rate (cell x view: seeds if any repetition selected one; FDP = 0 when no seed)
x = R[(R.cfg == REFP) & (R.base == "none")]
for b, tg, lo, hi in (("lt1", "LtOne", 0, 0.01), ("1to10", "OneToTen", 0.01, 0.10), ("ge10", "GeTen", 0.10, 1.01)):
    sub = x[(x.mixing_rate >= lo) & (x.mixing_rate < hi)]
    fd, fr, withs = [], [], 0
    for v in ("L14", "D3L"):
        fd += [(f if s > 0 else 0.0) for f, s in zip(sub[f"seed_idfrac_ref_{v}"].fillna(0.0), sub[f"seeds_ref_{v}"])]
        g = sub.groupby(["ds", "c", "r"])
        for _, gg in g:
            idf = gg[f"seed_idfrac_ref_{v}"].dropna()
            if len(idf):
                withs += 1
                fr.append(idf.mean())
    close(f"NRSeed{tg}FdpUncond", float(np.mean(fd)))
    close(f"NRSeed{tg}WithSeeds", withs)
    close(f"NRSeed{tg}IdFrac", float(np.mean(fr)))
# --- seeds of sigma_ref on the benchmarks (per data set and view, mean over draws): the range of the ID fraction
_bf = []
for part in ("openood", "fourood"):
    x = T[part][(T[part].cfg == REFP) & (T[part].base == "none")]
    for ds, g in x.groupby("ds"):
        for v in ("L14", "D3L"):
            _bf.append(g[f"seed_idfrac_ref_{v}"].mean())
close("NRSeedBenchIdMin", min(_bf), 0.005)
close("NRSeedBenchIdMax", max(_bf), 0.005)
if not min(_bf) > 0.2:
    bad.append("benchmark seeds: the ID fraction is not above q everywhere")
# --- covariate shift (per-task meta.json; mean over the data sets of a group within a draw, then over draws)
rows = []
for f in sorted(glob.glob(str(RES / "p17_csid" / "*" / "*" / "*.meta.json"))):
    mt = json.loads(open(f).read())
    cs, dr, ds = f.split("/")[-3], f.split("/")[-2], f.split("/")[-1].split(".")[0]
    sub = mt["csid_metrics"]
    for name in ("sigma_ref", "sigma", "lp", "static_all", "knn"):
        rows.append({"cs": cs, "draw": dr, "ds": ds, "score": name, **sub[name]})
    rows.append({"cs": cs, "draw": dr, "ds": ds, "score": "_v", "fcs5": sub["fpr_csid@0.05"]["sigma_ref"], "fid5": sub["fpr_id@0.05"]["sigma_ref"],
                 "seeds_all": sub["seeds_ref"]["all"], "seeds_cs": sub["seeds_ref"]["csid"]})
C = pd.DataFrame(rows)
assert C.groupby("cs").draw.nunique().tolist() == [6, 6]
for cs, ct in (("imagenet_v2", "VTwo"), ("imagenet_r", "R")):
    for g, G in (("near", "Near"), ("far", "Far")):
        for name, nt in (("sigma_ref", "Ref"), ("sigma", "Sig"), ("static_all", "Static")):
            y = C[(C.cs == cs) & C.ds.isin(OO[g]) & (C.score == name)].groupby("draw").mean(numeric_only=True)
            for met, tm in (("fs_FPR95", "FsFpr"), ("id_csid_AUROC", "IcAuroc"), ("fs_AUROC", "FsAuroc")):
                close(f"NRCs{ct}{G}{nt}{tm}", float(y[met].mean()))
        y = C[(C.cs == cs) & C.ds.isin(OO[g]) & (C.score == "_v")].groupby("draw").mean(numeric_only=True)
        close(f"NRCs{ct}{G}FprCsFive", float(y.fcs5.mean()))
        close(f"NRCs{ct}{G}FprIdFive", float(y.fid5.mean()))
        close(f"NRCs{ct}{G}SeedShare", 100 * float(y.seeds_cs.mean()) / float(y.seeds_all.mean()))
# --- separation bounds: the registered summary (per-image arrays are on the server); logical checks
B = json.load(open(HERE / "data" / "p17_summary.json"))["bounds"]
for k, e in B.items():
    for t in ("c2", "c2r"):
        if e[t]["violations"] != 0 or e[t]["cert_005"] > e[t]["actual_005"] + 1e-12:
            bad.append(f"bounds {k} {t}: violation or certified > actual")
    a = e["a1"]
    if a["violations"] != 0 or a["phibar_violations"] != 0 or a["cert_l_005"] > a["cert_s_005"] + 1e-12 or a["cert_s_005"] > a["actual_005"] + 1e-12:
        bad.append(f"bounds {k} a1: inconsistent")
r30 = [k for k in B if k.startswith("ninco|") and k.endswith("|r30")]
close("NRBdRefNincoRThirtyCertMin", 100 * min(B[k]["c2r"]["cert_005"] for k in r30))
close("NRBdRefNincoRThirtyActMax", 100 * max(B[k]["c2r"]["actual_005"] for k in r30))
close("NRBdHOneCertLMax", 100 * max(B[k]["a1"]["cert_l_005"] for k in B))

# ======================================================================= (2) claims in words
def gt(a, b, what):
    if not N[a] > N[b]:
        bad.append(f"claim fails: {what} ({a}={N[a]} vs {b}={N[b]})")


body = (HERE / "body_p22.tex").read_text(encoding="utf-8")
app = (HERE / "sections" / "appendix22.tex").read_text(encoding="utf-8")
rec = (HERE / "sections" / "rec22.tex").read_text(encoding="utf-8")
in2 = (HERE / "sections" / "in200_22.tex").read_text(encoding="utf-8")
meth = (HERE / "sections" / "method22.tex").read_text(encoding="utf-8")
# headline: standalone beats every reported value in all six columns of Table 1; x TINS too
if N["NRAloneWinsMain"] != 6:
    bad.append("standalone does not win all 6 columns")
for nm in ("NearAuroc", "FarAuroc", "FourAuroc", "NearFpr", "FarFpr", "FourFpr"):
    if not (N[f"NRAloneGap{nm}"] > 0 and N[f"NRTinsGap{nm}"] > 0):
        bad.append(f"gap not positive: {nm}")
# per data set: 12 / 18 standalone, 18 / 18 x TINS
if not (N["NRPerdsWinsAlone"] == 12 and N["NRPerdsWinsTins"] == 18 and N["NRPerdsCols"] == 18):
    bad.append("per-data-set counts")
# 'Four-OOD: the x TINS FPR95 about half of TINS (reported)'
half = N["NRTinsFourFpr"] / PUB["fourood"]["TINS"]["avg"][1]
if not 0.45 <= half <= 0.55:
    bad.append(f"'ほぼ半分' fails: {half}")
# sigma_ref vs sigma: better on ImageNet-1K near (interval below 0); far / Four-OOD within the stated bound
if not (N["NRAloneNearDSigHi"] < 0 and N["NRTinsNearDSigHi"] < 0):
    bad.append("sigma_ref not better than sigma on near")
for t in ("Alone", "Tins"):
    for G in ("Far", "Four"):
        if abs(N[f"NR{t}{G}DSig"]) > N["NRDSigFarFourMaxAbs"] + 1e-12:
            bad.append(f"far/four bound {t}{G}")
        if not (N[f"NR{t}{G}DSigLo"] < 0 < N[f"NR{t}{G}DSigHi"]):
            bad.append(f"'他の設定では区間が0を含む' fails at {t}{G}")
for tg in ("IT", "CF"):
    for G in ("Near", "Far"):
        if not (N[f"NR{tg}{G}DSigLo"] < 0 < N[f"NR{tg}{G}DSigHi"]):
            bad.append(f"'他の設定では区間が0を含む' fails at {tg}{G}")
# the seeds help on ImageNet-1K (intervals below 0) and hurt on ImageNet-200 / CIFAR-100 far (intervals above 0)
if not (N["NRvsLpAloneNearDHi"] < 0 and N["NRvsLpTinsNearDHi"] < 0 and N["NRvsLpAloneFarD"] < 0 and N["NRvsLpAloneFourD"] < 0):
    bad.append("seeds do not help on ImageNet-1K")
if not (N["NRITNearDLpLo"] > 0 and N["NRITFarDLpLo"] > 0 and N["NRCFFarDLpLo"] > 0):
    bad.append("seeds do not hurt on ImageNet-200 / CIFAR-100 far")
# conclusion: p+ only is 'equal or better' than sigma_ref on ImageNet-200 and CIFAR-100
for tg in ("IT", "CF"):
    for G in ("Near", "Far"):
        if not N[f"NR{tg}{G}DLpHi"] > 0:
            bad.append(f"p+ only not >= sigma_ref at {tg}{G}")
# ImageNet-200: sigma_ref worse than LGC-max on both groups ('にも及ばない')
num = dict(re.findall(r"\\newcommand\\(N[A-Za-z]+)\{([^}]*)\}", (HERE / "sections" / "numbers.tex").read_text()))
if not (N["NRITNearFpr"] > float(num["NInTwoZeroZeroLgcNearFpr"]) and N["NRITFarFpr"] > float(num["NInTwoZeroZeroLgcFarFpr"])):
    bad.append("ImageNet-200: sigma_ref not worse than LGC-max")
# CIFAR-100: AUROC more than 10 points above the reported best (near, far)
if not (N["NRCFNearAuroc"] - float(num["NCifarBestPubNearAuroc"]) > 10 and N["NRCFFarAuroc"] - float(num["NCifarBestPubFarAuroc"]) > 10):
    bad.append("CIFAR-100: not 10 points above the reported best")
if not (N["NRITNearAuroc"] > float(num["NInTwoZeroZeroBestPubNearAuroc"]) + 10 and N["NRITNearFpr"] < float(num["NInTwoZeroZeroBestPubNearFpr"]) - 10):
    bad.append("ImageNet-200: not far above the reported best")
# decomposition: q alone better than u on ImageNet near, worse on far; sqrt(delta) alone uninformative or reversed
if not ("NRDecUHurtsINNear" in N and "NRDecUHurtsITNear" in N and "NRDecUHelpsINFar" in N and "NRDecUHelpsFour" in N and "NRDecUHelpsITFar" in N and "NRDecUHelpsCFNear" in N):
    bad.append("decomposition: the directions stated in the text do not hold")
if not (N["NRDecDegINNearAuroc"] < 50 and N["NRDecDegINFarAuroc"] < 55):
    bad.append("decomposition: density alone is not uninformative / reversed")
if not (N["NRDecSigQINNearFpr"] > N["NRDecSigINNearFpr"] and N["NRDecSigQINFarFpr"] > N["NRDecSigINFarFpr"]):
    bad.append("decomposition: sigma with u/sqrt(delta) not worse on ImageNet near and far")
# covariate shift: static p better than sigma_ref on far (full spectrum); V2 hardly distinguished; R almost all flagged
if not N["NRCsRFarStaticFsFpr"] < N["NRCsRFarRefFsFpr"]:
    bad.append("csid: static p not better on R far")
if not (N["NRCsVTwoNearRefIcAuroc"] < 60 and N["NRCsRNearRefIcAuroc"] > 95 and N["NRCsRNearFprCsFive"] > 0.8):
    bad.append("csid: V2 / R statements")
# grid: r <= 5 worse than static in most cells, r >= 10 better in most; sigma_ref and p+ equal at NINCO 64 x 1
if not (N["NRRecLeFiveWorseStatic"] >= 25 and N["NRRecGeTenBetterStatic"] >= 28):
    bad.append("grid counts")
if abs(N["NRRecNincoCSixtyFourROne"] - N["NRRecNincoCSixtyFourROneLp"]) > 1e-9:
    bad.append("grid: sigma_ref and p+ only not equal at NINCO 64 x 1")
# seeds at low mixing: ID fraction above q when seeds are selected; FDP with zeros below q at < 1 %
if not (N["NRSeedLtOneIdFrac"] > 0.2 and N["NRSeedLtOneFdpUncond"] < 0.2):
    bad.append("low-mixing seeds statements")
# bounds: about half at NINCO r = 30, almost nothing at SSB-hard 980 x 50, nothing for h's calibration-side lower bound
if not (40 <= N["NRBdRefNincoRThirtyCertMin"] and N["NRBdRefNincoRThirtyCertMax"] <= 60 and N["NRBdRefSsbCNineEightyRFiftyCert"] < 1 and N["NRBdHOneCertLMax"] == 0):
    bad.append("bounds statements")
# the h adoption: interval upper ends above +0.5 (an interval rule would not have adopted h)
if not (N["NRHsUpperNear"] > 0.5 and N["NRHsUpperFar"] > 0.5):
    bad.append("h upper ends")
# development splits: the chosen candidate's test ranks
if not (N["NRDtNearRankOnTest"] == 10 and N["NRDtFarRankChosen"] == 18):
    bad.append("dev-test ranks")
# p_fin x TINS worse on near, better on Four-OOD than sigma_ref x TINS
if not (N["NRPfTinsNearFpr"] > N["NRTinsNearFpr"] and N["NRPfTinsFourFpr"] < N["NRTinsFourFpr"]):
    bad.append("p_fin product statements")
# the measured base detectors: smallest / largest near FPR95 and gains as written (TINS..NegLabel; TANL..NegLabel)
meas = {b: N[f"NRMeas{b}NearFpr"] for b in ("TINS", "TANL", "NegLabel", "AdaNeg")}
gain = {b: N[f"NRGain{b}NearFpr"] for b in ("TINS", "TANL", "NegLabel", "AdaNeg")}
if not (min(meas, key=meas.get) == "TINS" and max(meas, key=meas.get) == "NegLabel" and min(gain, key=gain.get) == "TANL" and max(gain, key=gain.get) == "NegLabel"):
    bad.append("base detector range statements")
if "\\NRMeasTINSNearFpr--\\NRMeasNegLabelNearFpr" not in body or "\\NRGainTANLNearFpr--\\NRGainNegLabelNearFpr" not in body:
    bad.append("base detector range macros not in the text")
# x TINS SSB-hard FPR95 below DisCoPatch but its interval contains DisCoPatch's value
if not (N["NRTinsSsbHardFpr"] < N["NRDiscoSsbHardFpr"] and N["NRTinsSsbHardFprLo"] < N["NRDiscoSsbHardFpr"] < N["NRTinsSsbHardFprHi"]):
    bad.append("SSB-hard x TINS interval statement")
# literal numbers of the main text
for lit, val in (("82.65", PUB["openood"]["TINS"]["near"][0]), ("57.88", PUB["openood"]["TINS"]["near"][1])):
    if f"{val:.2f}" != lit or lit not in body:
        bad.append(f"literal {lit}")
if "報告値の最良（29.40\\%）" not in body or abs(float(num["NBestOtherNearFpr"]) - 29.40) > 1e-9:
    bad.append("literal 29.40")
rng = [np.mean([PUB["openood"][k]["per"][d][1] for d in OO["near"]]) for k in ("NNGuide", "MOODv2", "DisCoPatch")]
if f"{round(min(rng))}--{round(max(rng))}\\%残る" not in body:
    bad.append("literal range 29--48")

# ======================================================================= (3) form
pre = (HERE / "preamble_p22.tex").read_text(encoding="utf-8")
title = re.search(r"\\title\{\\titleicon (.*?)\}\n", pre, re.S).group(1)
if " ".join(title.replace("\\\\", " ").split()) != "Salmon Ladder: Conformal Label Propagation for OOD Detection":
    bad.append(f"title: {title}")
for s in ("\\affiliate{keio}{慶應義塾大学　総合政策学部3年\\\\中澤・大越研究室d-hacks　72402035}", "\\affiliate{keio2}{慶應義塾大学院　政策・メディア研究科}",
          "\\author{表 紘太朗}{Kotaro Omote}{keio}", "\\author{尾崎 慶次郎}{Keijiro Ozaki}{keio2}"):
    if s not in pre:
        bad.append(f"author block: {s}")
abstract = (HERE / "sections" / "abstract22.tex").read_text(encoding="utf-8").strip()
macros = dict(re.findall(r"\\newcommand\\(N[A-Za-z]+)\{([^}]*)\}", (HERE / "sections" / "numbers22.tex").read_text()))
ab = re.sub(r"\\(N[A-Za-z]+)", lambda m_: macros[m_.group(1)], abstract).replace("\\%", "%").replace("$", "").replace("{", "").replace("}", "")
if len(ab) > 400:
    bad.append(f"abstract {len(ab)} characters")
if "\\clearpage\n\\onecolumn\n\\appendix\n" not in body:
    bad.append("one-column appendix after \\clearpage")
FORB = ["オンライン", "ストリーム", "メモリ", "確かに未知", "確かな未知", "再来", "Recurring", "15回", "酔歩", "魚道", "見立て", "名の由来", "段構え", "pM", "pLP", "DisCoNet", "前版"]
main_tex = (HERE / "main.tex").read_text(encoding="utf-8")


def expand_inputs(text):
    out = []
    for line in text.splitlines():
        m_ = re.search(r"\\input\{([^}]+)\}", line)
        if m_:
            p = HERE / m_.group(1)
            p = p if p.suffix == ".tex" else p.with_suffix(".tex")
            out.append(expand_inputs(p.read_text(encoding="utf-8")))
        else:
            out.append(line)
    return "\n".join(out)


full = expand_inputs(main_tex)
full_nocomment = "\n".join(re.split(r"(?<!\\)%", l, maxsplit=1)[0] for l in full.splitlines())
full_text = re.sub(r"\\[A-Za-z]+", " ", full_nocomment)        # words, not control sequences (\\NVcalBhFdpMin, $\\pm$ contain 'pM')
for w in FORB:
    if w in full_text:
        bad.append(f"forbidden word in the sources: {w}")
pdf = subprocess.run(["pdftotext", "-layout", str(HERE / "main.pdf"), "-"], capture_output=True, text=True).stdout
for w in FORB:
    if w in pdf:
        bad.append(f"forbidden word in the PDF: {w}")
pages = pdf.split("\f")
app_page = next(i for i, p in enumerate(pages) if any(re.fullmatch(r"\s*付\s+録\s*", line) for line in p.splitlines()))   # the heading, not an in-text reference
if app_page + 1 != 10:
    bad.append(f"the appendix starts on page {app_page + 1} (the main text and references must fit in 9 pages)")
if "参考文献" not in pages[8]:
    bad.append("the main reference list is not on page 9")
# the name is not explained; 'Salmon Ladder' appears only as a proper noun
for w in ("鮭", "サケ", "はしご"):
    if w in full_nocomment:
        bad.append(f"the name is explained: {w}")
# every macro used in the sources is defined
defined = set(re.findall(r"\\newcommand\\(N[A-Za-z]+)\{", "".join((HERE / "sections" / f).read_text() for f in ("numbers.tex", "numbers19.tex", "numbers22.tex"))))
used = set(re.findall(r"\\(N[A-Z][A-Za-z]+)", full_nocomment))
missing = sorted(u for u in used if u not in defined and u not in ("NeurIPS",))
if missing:
    bad.append(f"undefined macros: {missing[:10]}")
# tables carry the macro values
t = (HERE / "sections" / "t22_main.tex").read_text()
for nm in ("NRAloneNearFpr", "NRAloneNearAuroc", "NRTinsNearFpr", "NRTinsFourFpr"):
    if f"{N[nm]:.2f}" not in t:
        bad.append(f"t22_main lacks {nm}")
t = (HERE / "sections" / "t22_dec.tex").read_text()
for nm in ("NRDecQINNear", "NRDecUINFar", "NRDecDegINNear", "NRDecRefCFFar"):
    if f"{N[nm + 'Auroc']:.2f}／{N[nm + 'Fpr']:.2f}" not in t:
        bad.append(f"t22_dec lacks {nm}")

print(json.dumps({"mismatches": len(bad)}, ensure_ascii=False))
for b in bad:
    print(" -", b)
sys.exit(1 if bad else 0)
