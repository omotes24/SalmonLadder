"""Write docs/RESULTS.md and the results table of README.md, recomputed from the result files archived under experiments/.

  python tools/make_results_tables.py            # rewrite docs/RESULTS.md and the generated block of README.md
  python tools/make_results_tables.py --check    # exit 1 if either differs from what the result files give

Per-stream metric tables are aggregated here (unit means, paired 95% t intervals); summary JSON files written by the
analysis scripts on the server are read where no per-stream table is archived. Nothing is typed by hand.
"""
import argparse
import glob
import json
import sys
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import t as student_t

ROOT = Path(__file__).resolve().parents[1]
E = ROOT / "experiments"
P4, P5, P7, R5 = E / "phase4", E / "phase5_6", E / "phase7", E / "r5_final/r5"
OUT = ROOT / "docs/RESULTS.md"


def J(path):
    return json.loads(Path(path).read_text())


def tci(values):
    v = np.asarray(values, float)
    m = float(v.mean())
    h = float(student_t.ppf(0.975, len(v) - 1) * v.std(ddof=1) / np.sqrt(len(v)))
    return {"mean": m, "lo": m - h, "hi": m + h, "n": len(v)}


def af(a, f):
    return f"{a:.2f} / {f:.2f}"


def afd(d):
    return af(d["AUROC"], d["FPR95"])


def ci(d, nd=2):
    if isinstance(d, (list, tuple)):
        d = {"mean": d[0], "lo": d[1], "hi": d[2]}
    return f"{d['mean']:+.{nd}f} [{d['lo']:+.{nd}f}, {d['hi']:+.{nd}f}]"


def gain(a, b):
    """a - b with two decimals, rounded half up on the decimal values (the stored means are exact binary fractions)."""
    d = (Decimal(repr(a)) - Decimal(repr(b))).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
    return f"{d:+}"


def cell(o, key=None):
    o = o[key] if key else o
    return af(o["AUROC"]["mean"], o["FPR95"]["mean"])


def table(header, rows, align=None):
    align = align or ["l"] + ["r"] * (len(header) - 1)
    line = {"l": ":--", "r": "--:", "c": ":-:"}
    out = ["| " + " | ".join(header) + " |", "|" + "|".join(line[a] for a in align) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return out + [""]


# ------------------------------------------------------------------------------------------------ 1. Phase 4: U1 / U2
UNIT = {"U1": "split", "U2": "draw", "U3": "draw"}
BASES = ["none", "MCM", "NegLabel", "AdaNeg", "TANL", "TINS"]


def section_unused():
    BL = pd.read_parquet(P4 / "results/baselines.parquet")
    EV = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(str(P4 / "results/eval/*.parquet")))], ignore_index=True)
    lock = J(P4 / "selection_lock.json")
    sel = lock["selected"]
    summary = J(P4 / "results/eval_summary.json")
    ext = J(P7 / "results/an_ext_bases.json")["banks"]
    res = J(P4 / "results/resources.json")
    meta = J(P4 / "results/vlm_tta_meta.json")

    def bl(bank, base, visual):
        d = BL[(BL.bank == bank) & (BL.base == base) & (BL.visual == visual)]
        u = d.groupby(UNIT[bank])[["AUROC", "FPR95"]].mean()
        assert len(u) == 5, (bank, base, visual)
        return float(u.AUROC.mean()), float(u.FPR95.mean())

    def bl_diff(bank, base, a, b, metric="FPR95"):
        d = BL[(BL.bank == bank) & (BL.base == base)].pivot_table(index=["split", "draw", "seed"], columns="visual", values=metric)
        u = (d[a] - d[b]).groupby(level=UNIT[bank]).mean()
        assert len(u) == 5
        return tci(u.values)

    L = ["## 1. Evaluation on images not used for selection (Phase 4)", "",
         "U1: five new class splits of ImageNet-1K (900 ID classes; the 100 unknown classes have sibling classes among the ID classes; "
         "18,000 ID and 5,000 unknown images per stream; 3 draws of the 16 labelled images x 3 arrival orders per split = 45 streams; unit of the "
         "interval: split). U2: ImageNet-O against all 1,000 classes (5 draws x 3 orders = 15 streams; unit: draw). Batches of 256. "
         "zeta (robust z of the zero-start propagation) is the registered comparator: among the read-outs of the same graph that use neither the "
         "entrance nor the memory, it had the lowest near-OOD FPR95 on the development split, where every read-out received the same 27 graph "
         "configurations; the choice was locked before U1 and U2 existed. `x` is the product of scores (sum of log scores, weight 1).", "",
         "### 1.1 Base detectors alone, x zeta, x Salmon Ladder, x extension (AUROC / FPR95)", "",
         "`x extension` is the read-out locked in Phase 7 after Salmon Ladder had been evaluated on U1 and U2 (Section 6); it is not part of the "
         "registered Phase 4 comparison.", ""]
    for bank in ("U1", "U2"):
        rows = []
        for base in BASES:
            alone = "–" if base == "none" else af(*bl(bank, base, "none"))
            d = bl_diff(bank, base, "REPRISE", "minimal")
            x = ext[bank]["bases"][base]
            assert abs(x["reprise"]["FPR95"]["mean"] - bl(bank, base, "REPRISE")[1]) < 5e-3, (bank, base)       # the two phases agree
            rows.append([base if base != "none" else "none (standalone)", alone, af(*bl(bank, base, "minimal")), af(*bl(bank, base, "REPRISE")), ci(d),
                         cell(x, "ext"), ci(x["ext-reprise"]["FPR95"])])
        L += [f"**{bank}**", ""] + table(["base detector s0", "s0 only", "x zeta", "x Salmon Ladder", "Salmon Ladder - zeta (FPR95)", "x extension",
                                           "extension - Salmon Ladder (FPR95)"], rows)
    oodd = {bank: af(*bl(bank, "MCM+OODD_b64", "none")) for bank in ("U1", "U2")}
    L += [f"MCM+OODD (re-implementation of the official procedure, batch 64) alone: U1 {oodd['U1']}, U2 {oodd['U2']}. AdaNeg and TANL are the "
          "official post-processors of OpenOOD-VLM, NegLabel is the vanilla score of the AdaNeg code, TINS is the upstream code.", ""]

    L += ["### 1.2 Registered endpoints", "",
          "E1: standalone FPR95 on U1, frozen Salmon Ladder minus zeta. E2: FPR95 on U1 with TINS, frozen Salmon Ladder (weight 1) minus zeta with its "
          "development-selected weight a* = " + f"{summary['a_star']:g}" + ". Decision rule registered before the evaluation: Salmon Ladder is the main "
          "method if the upper limit of the interval is below 0. U2 and U3 are registered replications.", ""]
    rows = []
    for bank in ("U1", "U2", "U3"):
        b = summary["banks"][bank]
        rows.append([bank, f"{b['E1_FPR95']['A_mean']:.2f}", f"{b['E1_FPR95']['B_mean']:.2f}", ci(b["E1_FPR95"]), ci(b["E1_AUROC"]),
                     f"{b['E2_FPR95']['A_mean']:.2f}", f"{b['E2_FPR95']['B_mean']:.2f}", ci(b["E2_FPR95"]), ci(b["E2_AUROC"])])
    L += table(["bank", "E1 Salmon Ladder", "E1 zeta", "E1 difference (FPR95)", "E1 difference (AUROC)", "E2 Salmon Ladder", "E2 zeta", "E2 difference (FPR95)",
                "E2 difference (AUROC)"], rows)
    L += ["U3 is a five-vs-five species split of a private wildlife data set. It was registered and evaluated; its image list cannot be released and "
          "it is not reported in the paper. The FPR95 intervals of U3 include 0; the AUROC differences do not.", ""]

    # propagation-only variants and Salmon Ladder variants on the same graph
    def ev(bank, family, config, base="none", role=None):
        d = EV[(EV.bank == bank) & (EV.family == family) & (EV.base == base) & np.isclose(EV.a, 1.0)]
        if config is not None:
            d = d[d.config == config]
        d = d[d.role == role] if role is not None else d[d.role != "weighted"]
        u = d.groupby(UNIT[bank])[["AUROC", "FPR95"]].mean()
        assert len(u) == 5, (bank, family, config, base, role, len(u))
        return float(u.AUROC.mean()), float(u.FPR95.mean())

    ref = "k10g1l0.9"
    fam = [("standard propagation, raw u", "raw", sel["raw"]["config"], None), ("  solved to convergence", "raw_L2", sel["raw"]["config"], None),
           ("  random-walk normalisation", "rw", sel["rw"]["config"], None), ("mass correction u N_t / N_s", "mass", sel["mass"]["config"], None),
           ("calibration rank, zero start", "cdf", sel["cdf"]["config"], None), ("calibration rank, warm start", "cdf_L0", sel["cdf_L0"]["config"], None),
           ("robust z (zeta)", "z", sel["z"]["config"], None), ("propagation of the static distance", "sprop", sel["sprop"]["config"], None),
           ("static p x rank, zero start", "stat_cdf", sel["stat_cdf"]["config"], None),
           ("static p x rank, warm start", "stat_cdf_L0", sel["stat_cdf_L0"]["config"], None), ("memory p_M only", "Mpt", None, None),
           ("Salmon Ladder (frozen)", "rep_L0", ref, "frozen"), ("  re-tuned under the same budget", "rep_L0", sel["rep_L0"]["config"], "selected"),
           ("  zero start, re-tuned", "rep_L1", sel["rep_L1"]["config"], "selected")]
    def ev_streams(bank, family, config, role):
        d = EV[(EV.bank == bank) & (EV.family == family) & (EV.base == "none") & np.isclose(EV.a, 1.0)]
        if config is not None:
            d = d[d.config == config]
        d = d[d.role == role] if role is not None else d[d.role != "weighted"]
        return d.drop_duplicates(["split", "draw", "seed"]).set_index(["split", "draw", "seed"]).FPR95

    def to_reprise(bank, family, config, role):
        """FPR95 of the frozen Salmon Ladder minus this read-out, paired by stream, interval over the units of the bank."""
        d = (ev_streams(bank, "rep_L0", ref, "frozen") - ev_streams(bank, family, config, role)).groupby(level=UNIT[bank]).mean()
        assert len(d) == 5 and not d.isna().any(), (bank, family)
        return ci(tci(d.values))

    rows = []
    for lab, f, cfg, role in fam:
        tins = "–" if f == "Mpt" else f"{ev('U1', f, cfg, 'TINS', role)[1]:.2f}"
        frozen = f == "rep_L0" and role == "frozen"
        rows.append([lab, cfg.replace("k", "k=").replace("g", ", gamma=").replace("l", ", lambda=") if cfg else "–", af(*ev("U1", f, cfg, "none", role)), tins,
                     af(*ev("U2", f, cfg, "none", role)), "–" if frozen else to_reprise("U1", f, cfg, role), "–" if frozen else to_reprise("U2", f, cfg, role)])
    L += ["### 1.3 Read-outs of the same graph (standalone)", "",
          "Every family received the same 27 graph configurations on the development split (rule: lowest near FPR95 subject to far FPR95 <= "
          "reference + 0.5); the selected configuration was locked before the evaluation. `zero start`: all nodes iterated from 0 at every batch; "
          "`warm start`: continued from the values of the previous batch (15 sweeps in both cases).", ""]
    L += table(["read-out", "selected graph", "U1 AUROC / FPR95", "U1 x TINS FPR95", "U2 AUROC / FPR95", "Salmon Ladder - read-out, U1 FPR95",
                "Salmon Ladder - read-out, U2 FPR95"], rows)
    L += ["The last two columns are descriptive (only the comparison with zeta was registered). On U2 the product of the static rank and the "
          "propagation rank has a lower FPR95 than zeta; the frozen Salmon Ladder is below every read-out without entrance and memory on both sets.", ""]

    runs = [v for v in meta.values() if v["bank"] == "U1"]
    cost = lambda m: (float(np.mean([v["methods"][m]["ms_per_image"] for v in runs])), float(np.max([v["methods"][m]["alloc_MiB"] for v in runs])) / 1024)
    rows = [["TINS", f"{res['TINS']['ms_per_image']:.2f}", f"{res['TINS']['alloc_MiB'] / 1024:.2f}"]]
    rows += [[m, f"{cost(m)[0]:.2f}", f"{cost(m)[1]:.2f}"] for m in ("AdaNeg", "TANL", "NegLabel")]
    rows += [["Salmon Ladder (frozen)", f"{res['REPRISE_frozen']['ms_per_image']:.2f}", f"{res['REPRISE_frozen']['alloc_MiB'] / 1024:.2f}"],
             ["zeta", f"{res['minimal_z']['ms_per_image']:.2f}", f"{res['minimal_z']['alloc_MiB'] / 1024:.2f}"]]
    enc = res["encoders"]
    L += ["### 1.4 Cost on U1 (one RTX 2080 Ti; method-specific processing only, image encoding excluded)", ""]
    L += table(["method", "ms per image", "allocated GPU memory [GiB]"], rows)
    L += [f"Image encoding: CLIP ViT-B/16 {enc['CLIP']['ms_per_image']:.2f} ms, DINOv2 ViT-B/14 {enc['B14']['ms_per_image']:.2f} ms, "
          f"ViT-L/14 {enc['L14']['ms_per_image']:.2f} ms per image.", ""]
    return L


# ------------------------------------------------------------------------------------------------ 2. OpenOOD / Four-OOD
OO = ["ssb_hard", "ninco", "inaturalist", "textures", "openimageo"]
FOUR = ["inat", "sun", "places", "dtd"]
S5, S3 = [123, 124, 125, 126, 127], [123, 124, 125]


def section_test():
    run = {("openood", d, s): J(R5 / f"phase3/openood/{d}_seed{s}.json")["metrics"] for d in OO for s in S5}
    run.update({("fourood", d, s): J(R5 / f"phase3/fourood/{d}_seed{s}.json")["metrics"] for d in FOUR for s in S3})
    base = J(R5 / "paper/paper_data.json")["base_detectors"]
    for (part, d, s), m in run.items():
        for k in ("mcm", "mcm_x_v5", "neglabel", "neglabel_x_v5"):
            m[k] = base[f"{part}|{d}|{s}"][k]

    def per_order(part, method, dss, metric):
        return [float(np.mean([run[(part, d, s)][method][metric] for d in dss])) for s in (S5 if part == "openood" else S3)]

    def c(part, method, dss):
        return af(float(np.mean(per_order(part, method, dss, "AUROC"))), float(np.mean(per_order(part, method, dss, "FPR95"))))

    near, far = OO[:2], OO[2:]
    rows_def = [("MCM", "mcm"), ("NegLabel", "neglabel"), ("TINS", "tins"), None,
                ("kNN, 16 labelled images, two views, x TINS", "knn16_2view"), ("Mahalanobis++, 16 labelled images, two views, x TINS", "maha16_2view"),
                ("Mahalanobis++, all training images, ViT-L/14, x TINS", "mahaF_l14"), ("AdaNeg-type memory (no entrance), x TINS", "adaneg"),
                ("OODD-type memory, x TINS", "oodd"), None,
                ("static p only, x TINS", "v5_static"), ("memory p_M only, x TINS", "v5_mem"), ("propagation p_LP only, x TINS", "v5_lp"), None,
                ("MCM x Salmon Ladder", "mcm_x_v5"), ("NegLabel x Salmon Ladder", "neglabel_x_v5"), ("Salmon Ladder, standalone", "v5_vis"), ("**TINS x Salmon Ladder**", "v5")]
    rows = [[lab, c("openood", m, near), c("openood", m, far)] + [c("openood", m, [d]) for d in OO] for lab, m in [r for r in rows_def if r]]
    L = ["## 2. OpenOOD v1.5 ImageNet-1K and Four-OOD (final test of the frozen method)", "",
         "The frozen configuration was evaluated under a registration written before the run (R5 Phase 3; the fifth registered evaluation on this "
         "test split, see docs/PREREGISTRATION.md). ID: 45,000 validation images; five arrival orders; values are means over the orders "
         "(AUROC / FPR95, upstream TINS metric convention). The comparison rows use the same DINOv2 features and the same 16 labelled images per class.", ""]
    L += table(["method", "near-OOD", "far-OOD", "SSB-hard", "NINCO", "iNaturalist", "Textures", "OpenImage-O"], rows)
    d = tci(np.array(per_order("openood", "v5", near, "FPR95")) - np.array(per_order("openood", "adaneg", near, "FPR95")))
    d2 = tci(np.array(per_order("openood", "v5", near, "FPR95")) - np.array(per_order("openood", "v5_lp", near, "FPR95")))
    L += [f"Near-OOD FPR95, TINS x Salmon Ladder minus the strongest same-feature comparison (AdaNeg-type memory): {ci(d)}; minus propagation only: {ci(d2)} "
          "(paired over the five orders).", ""]
    rows = [[lab, c("fourood", m, FOUR)] + [c("fourood", m, [d]) for d in FOUR]
            for lab, m in (("MCM", "mcm"), ("NegLabel", "neglabel"), ("TINS", "tins"), ("Mahalanobis++, all training images, x TINS", "mahaF_l14"),
                           ("propagation p_LP only, x TINS", "v5_lp"), ("Salmon Ladder, standalone", "v5_vis"), ("**TINS x Salmon Ladder**", "v5"))]
    L += ["**Four-OOD** (ImageNet-1K against iNaturalist, SUN, Places, Textures; three orders)", ""]
    L += table(["method", "mean", "iNaturalist", "SUN", "Places", "Textures"], rows)

    cub = {}
    for f in sorted(glob.glob(str(R5 / "phase3/cub/eval_*_seed*.json"))):
        cub.setdefault(Path(f).stem.split("_")[1], []).append(J(f)["metrics"])
    cc = lambda lv, m: af(float(np.mean([x[m]["AUROC"] for x in cub[lv]])), float(np.mean([x[m]["FPR95"] for x in cub[lv]])))
    rows = [[lab] + [cc(lv, m) for lv in ("Easy", "Medium", "Hard")]
            for lab, m in (("MCM", "mcm"), ("TINS", "tins"), ("kNN x TINS", "knn16_2view"), ("Mahalanobis++ x TINS", "maha16_2view"), ("**TINS x Salmon Ladder**", "v5"))]
    L += ["**CUB-200-2011, Semantic Shift Benchmark split** (100 known classes; three orders)", ""]
    L += table(["method", "Easy", "Medium", "Hard"], rows)

    s = J(P5 / "results_final/summary_openood.json")
    f4 = J(P5 / "results_final/summary_fourood.json")
    rows = []
    for lab, o in (("OpenOOD near-OOD, x TINS", s["near_TINS"]), ("OpenOOD far-OOD, x TINS", s["far_TINS"]), ("OpenOOD near-OOD, standalone", s["near_none"]),
                   ("OpenOOD far-OOD, standalone", s["far_none"]), ("Four-OOD, x TINS", f4["fourood_TINS"]), ("Four-OOD, standalone", f4["fourood_none"])):
        rows.append([lab, afd(o["frozen_v5"]), afd(o["zeta"]), afd(o["locked"]), ci(o["locked_minus_frozen_v5_FPR95"]), ci(o["locked_minus_frozen_v5_AUROC"])])
    L += ["**Sixth use of the test split (Phase 5).** The two-sided variant locked in Phase 5 was evaluated once; the frozen method was recomputed in "
          "the same run and reproduces the values above. The variant is not adopted (Section 5).", ""]
    L += table(["setting", "Salmon Ladder (frozen)", "zeta", "two-sided variant", "variant - frozen (FPR95)", "variant - frozen (AUROC)"], rows)

    # every evaluation of this project on the OpenOOD test split, in order (all x TINS)
    T = E / "r5_final/test_eval"
    m1, m2 = J(T / "results/metrics.json")["means"], J(T / "results_clavism/metrics.json")["means"]
    m3, m4 = J(T / "results_m2/metrics.json")["table"], J(T / "results_m3/metrics.json")["table"]
    frac = lambda d: af(100 * d["AUROC"], 100 * d["FPR95_upstream"])
    tab = lambda d: af(d["AUROC"]["mean"], d["FPR95"]["mean"])
    assert tab(m3["near"]["tins"]) == tab(m4["near"]["tins"]) == c("openood", "tins", near)              # the same TINS runs in uses 3-6
    assert afd(s["near_TINS"]["frozen_v5"]) == c("openood", "v5", near) and afd(s["far_TINS"]["frozen_v5"]) == c("openood", "v5", far)
    rows = [["–", "TINS alone", "1", frac(m1["nearood"]["tins"]), frac(m1["farood"]["tins"])],
            ["1", "static rank of the class-conditional nearest-neighbour distance (ViT-B/14, 5 candidate classes)", "1", frac(m1["nearood"]["fused"]), frac(m1["farood"]["fused"])],
            ["2", "+ memory of earlier images", "1", frac(m2["nearood"]["clavism"]), frac(m2["farood"]["clavism"])],
            ["–", "TINS alone", "5", c("openood", "tins", near), c("openood", "tins", far)],
            ["3", "+ second view (ViT-L/14), two-stage entrance", "5", tab(m3["near"]["clavism_m2"]), tab(m3["far"]["clavism_m2"])],
            ["4", "+ class prototypes, label propagation, three-stage entrance (0.4, 0.3, 0.1): first Salmon Ladder configuration", "5", tab(m4["near"]["clavism_m3"]), tab(m4["far"]["clavism_m3"])],
            ["5", "**frozen Salmon Ladder**: shrinkage 48, 20 candidate classes, nearest member, gamma = 1, entrance (0.3, 0.2, 0.1019); registered final test", "5",
             c("openood", "v5", near), c("openood", "v5", far)],
            ["6", "two-sided variant of Phase 5 (not adopted)", "5", afd(s["near_TINS"]["locked"]), afd(s["far_TINS"]["locked"])]]
    L += ["**Registered evaluations on the OpenOOD test split.** A newly frozen configuration was evaluated on the split six times, each under a "
          "registration written before the run. The configurations of uses 1–4 were steps of the development. Between these uses, analysis scripts "
          "scored further variants, ablations and baselines on the same test data for the drafts of that time (`experiments/r5_final/analysis/`), "
          "and the earlier controls used test images of two of its data sets. The split is therefore not independent of the design of the method; "
          "the evaluation in Section 1 exists for that reason. All rows are products with TINS; `orders` is the number of arrival orders averaged.", ""]
    L += table(["use", "configuration", "orders", "near-OOD", "far-OOD"], rows, align=["c", "l", "r", "r", "r"])
    return L


# ------------------------------------------------------------------------------------------------ 3. recurrence intervention
def section_intervention():
    x = J(P4 / "results/exp4_summary.json")
    cur = {(r["family"], r["cond"], r["r"]): r["AUROC"] for r in x["curves"]}
    rs = [0, 1, 2, 5, 10, 20]
    fams = [("static p", "static"), ("memory p_M only", "Mpt"), ("propagation p_LP only", "pLP"), ("static p x p_LP (no memory)", "static_pLP"),
            ("zeta", "best"), ("Salmon Ladder", "REPRISE")]
    L = ["## 3. Recurrence intervention (Phase 4, experiment 4)", "",
         f"A fixed history (768 ID and 256 unknown images) is processed; r of its unknown images are replaced by images of the class under test; "
         f"then query images of that class and common ID queries are scored one at a time on a clone of the state. {x['n_classes']} "
         "(split, class) pairs from U1, standalone AUROC. Deviation from the registration: 64 common ID queries instead of the registered 128 (an error "
         "in the design code, `exp4_design.py`; see docs/PREREGISTRATION.md).", ""]
    L += table(["read-out"] + [f"r = {r}" for r in rs], [[lab] + [f"{cur[(f, 'same', r)]:.2f}" for r in rs] for lab, f in fams])
    L += ["Gain over r = 0 (from the unrounded values):", ""]
    L += table(["read-out"] + [f"r = {r}" for r in rs[1:]],
               [[lab] + [gain(cur[(f, "same", r)], cur[(f, "same", 0)]) for r in rs[1:]] for lab, f in fams if f != "static"])
    rows = [[lab] + [ci(x[f"gain20_{f}"][cd]) for cd in ("same", "dup", "near", "far")]
            for lab, f in (("Salmon Ladder", "REPRISE"), ("zeta", "best"), ("static p x p_LP", "static_pLP"), ("propagation p_LP only", "pLP"), ("memory p_M only", "Mpt"))]
    L += ["Gain in AUROC from r = 0 to r = 20 by the kind of inserted image (class bootstrap, 95%):", ""]
    L += table(["read-out", "same class", "augmented copies of one image", "nearest other unknown class", "farthest other unknown class"], rows)
    e = x["estimand_REPRISE_vs_staticpLP"]
    L += [f"Registered estimand (gain of Salmon Ladder minus gain of static p x p_LP, same class, r = 20): {ci(e['class_bootstrap'])} "
          f"(clustered by split: {ci(e['split_clustered'])}); positive for {100 * e['frac_positive']:.1f}% of the pairs.", ""]
    return L


# ------------------------------------------------------------------------------------------------ 4. operating conditions
def section_operating():
    op = P4 / "results/operating"
    L = ["## 4. Operating conditions (Phase 4, experiments 5–8)", ""]
    rows = []
    for B in (1, 16, 64, 256):
        d = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(str(op / f"exp5_B{B}_seed*.parquet")))])
        r = d[d.family == "REPRISE"]
        rows.append([B, af(r.AUROC.mean(), r.FPR95.mean()), f"{r['ms_per_image'].mean():.2f}"] +
                    [f"{r[f'idFA@{a}'].mean():.2f}" for a in ("0.001", "0.01", "0.05")] + [f"{r['oodTPR@0.01'].mean():.1f}"])
    L += ["**Batch size** (U1, split 1, three orders). Thresholds were fixed on the development split at nominal ID false-alarm rates of 0.1%, 1% and 5%; "
          "`ms` is the stream time per image without encoding and without the static distances.", ""]
    L += table(["batch", "AUROC / FPR95", "ms per image", "ID false alarms @0.1% [%]", "@1% [%]", "@5% [%]", "unknown detected @1% [%]"], rows)
    conds = [("random", "random order"), ("id_first9000", "9,000 ID images first"), ("id_burst10", "ID in runs of 10 images of one class"),
             ("ood5pct", "5% unknown"), ("ood1pct", "1% unknown"), ("burst10_ood5pct", "ID runs and 5% unknown")]
    fams = [("Salmon Ladder", "REPRISE"), ("zeta", "best"), ("static p x p_LP", "static_pLP"), ("memory p_M only", "Mpt")]
    rows = []
    for key, lab in conds:
        d = pd.concat([pd.read_parquet(op / f"exp6_s{split}_{key}.parquet") for split in range(1, 6)])
        assert d[d.family == "REPRISE"].shape[0] == 5, key
        rows.append([lab] + [f"{d[d.family == f].FPR95.mean():.2f}" for _, f in fams] + [f"{d[d.family == 'REPRISE']['idFA@0.01'].mean():.2f}"])
    L += ["**Arrival conditions** (U1, five splits, FPR95):", ""]
    L += table(["condition"] + [lab for lab, _ in fams] + ["Salmon Ladder ID false alarms @1% [%]"], rows)
    rows = []
    for cap in (0, 32768, 8192, 1000):
        for policy in (("fifo",) if cap == 0 else ("fifo", "random", "diversity")):
            b = J(P4 / f"results/bounded/cap{cap}_{policy}.json")
            m = b["metrics"]
            rows.append(["unbounded" if cap == 0 else f"{cap:,}", "–" if cap == 0 else policy, f"{m['REPRISE']['all']['FPR95']:.2f}",
                         f"{m['REPRISE']['last20k']['FPR95']:.2f}", f"{m['best_z']['all']['FPR95']:.2f}", f"{b['final_graph_nodes']:,}",
                         f"{max(v['allocated_GiB'] for v in b['peaks'].values()):.2f}"])
    L += ["**Bounded history** (one long stream: 72,000 ID and 5,000 unknown images; the cap applies to the stream nodes of the graph and to each "
          "entrance stage; eviction: `fifo` oldest first, `random`, `diversity` most redundant first; FPR95; `last 20k`: the last 20,000 images):", ""]
    L += table(["cap", "eviction", "Salmon Ladder", "Salmon Ladder, last 20k", "zeta", "graph nodes at the end", "peak allocated GPU memory [GiB]"], rows,
               align=["r", "l", "r", "r", "r", "r", "r"])
    ex = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(str(P4 / "results/exp8/*.parquet")))])
    budgets = [("b4", "3 + 1"), ("b8", "6 + 2"), ("b16", "12 + 4 (frozen configuration)"), ("b32", "24 + 8"), ("b16_8+8", "8 + 8"), ("b16_14+2", "14 + 2"),
               ("L14_noK", "12 + 4, ViT-L/14 only, no candidate classes")]
    assert {c for c, _ in budgets} == set(ex.cond.unique())
    rows = []
    for cond, lab in budgets:
        g = ex[ex.cond == cond]
        r, z, st = g[g.family == "REPRISE"], g[g.family == "best_z"], g[g.family == "static"]
        assert len(r) == len(z) == len(st) == 5, cond
        rows.append([lab, af(st.AUROC.mean(), st.FPR95.mean()), af(z.AUROC.mean(), z.FPR95.mean()), af(r.AUROC.mean(), r.FPR95.mean())])
    L += ["**Labelled images per class** (support + calibration; U1, five splits, draw 0, order 123):", ""]
    L += table(["support + calibration", "static p", "zeta", "Salmon Ladder"], rows)
    return L


# ------------------------------------------------------------------------------------------------ 5. Phase 5 / 6
def section_phase56():
    c = J(P5 / "results/confirm_dev2.json")
    u = J(P5 / "results_final/summary_u4.json")
    d3 = J(P5 / "results/d3_summary_dev2.json")
    L = ["## 5. Improvement loop (Phase 5) and DINOv3 swap (Phase 6)", "",
         "Phase 5 searched on the development split for a variant that lowers the standalone near FPR95 by a registered margin, confirmed the locked "
         "variant once on the second development split, and evaluated it once on a new unused bank (U4: five new ImageNet-1K class splits, 45 streams) "
         "and on the test split. The locked variant (`two-sided`: zero-start propagation plus a self-labelled seed set of earlier stream images, "
         "selected by two rounds of Storey-BH) improves U4 by a large margin but does not transfer to OpenOOD (Section 2), so the frozen method "
         "remains the main method.", ""]
    rows = [["dev2 near, standalone (confirmation)", afd(c["near_none"]["frozen"]), afd(c["near_none"]["locked"]), ci(c["near_none"]["dFPR95"])],
            ["dev2 near, x TINS", afd(c["near_TINS"]["frozen"]), afd(c["near_TINS"]["locked"]), ci(c["near_TINS"]["dFPR95"])],
            ["dev2 far, standalone", afd(c["far_none"]["frozen"]), afd(c["far_none"]["locked"]), ci(c["far_none"]["dFPR95"])],
            ["dev2 far, x TINS", afd(c["far_TINS"]["frozen"]), afd(c["far_TINS"]["locked"]), ci(c["far_TINS"]["dFPR95"])],
            ["U4, standalone", afd(u["near_none"]["frozen_v5"]), afd(u["near_none"]["locked"]), ci(u["near_none"]["locked_minus_frozen_v5_FPR95"])],
            ["U4, x TINS", afd(u["near_TINS"]["frozen_v5"]), afd(u["near_TINS"]["locked"]), ci(u["near_TINS"]["locked_minus_frozen_v5_FPR95"])]]
    L += table(["evaluation", "Salmon Ladder (frozen)", "two-sided variant", "difference (FPR95)"], rows)
    L += [f"U4 reference values (standalone): zeta {afd(u['near_none']['zeta'])}; frozen memory x zero-start propagation {afd(u['near_none']['M0xlp_zero'])}.", "",
          "Phase 6 replaced only the two views (DINOv2 ViT-B/14 + ViT-L/14 by DINOv3 ViT-B/16 + ViT-L/16, CLS token, 256 px) under a registration "
          "written before the weights were obtained; development splits only (difference: DINOv3 minus DINOv2, unit: draw):", ""]
    cells = d3["cells"]
    rows = []
    for stream in ("near", "far"):
        for method in ("v5", "two-sided"):
            a, b = cells[f"{stream}|none|{method}|DINOv2"], cells[f"{stream}|none|{method}|DINOv3@256"]
            rows.append([f"dev2 {stream}, {'Salmon Ladder (frozen)' if method == 'v5' else 'two-sided variant'}", af(a["AUROC"], a["FPR95"]), af(b["AUROC"], b["FPR95"]), ci(b["dFPR95"])])
    L += table(["evaluation (standalone)", "DINOv2 B/14 + L/14", "DINOv3 B/16 + L/16", "difference (FPR95)"], rows)
    return L


# ------------------------------------------------------------------------------------------------ 6. Phase 7
BINS = ["1-1", "2-2", "3-5", "6-10", "11-20", "21-30", "31-50"]
VIEW = {"S14": "DINOv2 ViT-S/14", "B14": "DINOv2 ViT-B/14", "L14": "DINOv2 ViT-L/14", "G14": "DINOv2 ViT-g/14", "D3S": "DINOv3 ViT-S/16",
        "D3SP": "DINOv3 ViT-S+/16", "D3B": "DINOv3 ViT-B/16", "D3L": "DINOv3 ViT-L/16", "DINO1": "DINO ViT-B/16", "MAE": "MAE ViT-B/16",
        "RN50": "CLIP RN50", "CLIP": "CLIP ViT-B/16", "CLIPL": "CLIP ViT-L/14", "SIG2L": "SigLIP 2 ViT-L/16"}
SINGLE = ["S14", "B14", "L14", "G14", "D3S", "D3SP", "D3B", "D3L", "DINO1", "MAE", "RN50", "CLIP", "CLIPL", "SIG2L"]
PAIRS = ["B14+L14", "S14+B14", "S14+D3S", "S14+RN50", "S14+CLIP", "B14+CLIP", "L14+CLIP", "L14+G14", "D3S+D3SP", "D3B+D3L", "L14+D3L", "CLIP+CLIPL"]
POSTHOC = {"D3S", "D3SP", "RN50", "S14+D3S", "D3S+D3SP", "S14+RN50"}
DS = {"U1": "ImageNet-1K (U1)", "insk": "ImageNet-Sketch", "inr": "ImageNet-R", "places365": "Places365", "cub": "CUB-200", "cifar100": "CIFAR-100",
      "cubssb-Easy": "CUB (SSB, Easy)", "cubssb-Medium": "CUB (SSB, Medium)", "cubssb-Hard": "CUB (SSB, Hard)"}
PATTERNS = [("random", "random order"), ("emerging", "unknown classes emerge one after another"), ("id_first", "ID only in the first half"),
            ("ood_early", "unknown images concentrated early"), ("id_burst10", "ID in runs of 10 images of one class"),
            ("two_visits", "each unknown class in two short visits"), ("ood_burst", "unknown images in runs of one class")]


def section_phase7():
    R = P7 / "results"
    lock, retro, u1, cases = J(R / "an_lock_eval.json"), J(R / "an_retro_u1r.json"), J(R / "an_u1.json"), J(R / "an_cases_u1.json")
    xb, c1, B, D, Ee = J(R / "an_ext_bases.json")["banks"], J(R / "c1_summary.json"), J(R / "an_b.json"), J(R / "an_d.json")["rows"], J(R / "an_e.json")
    ncm, bench = J(R / "ncm_u1.json"), J(R / "bench7.json")
    spec = J(P7 / "selection_lock_p7.json")["spec"]
    L = ["## 6. Additional experiments (Phase 7)", "",
         "The frozen configuration is unchanged. `extension` is the read-out locked by amendment 02 "
         f"(`{spec}`: per view, a one-sided memory term that also reads members admitted from the same batch, times the calibration rank of the "
         "zero-start propagation), selected on the first development split, confirmed once on the second, locked, and then evaluated. It was "
         "designed after Salmon Ladder had been evaluated on U1 and U2. The OpenOOD test split was not used in this phase.", "",
         "### 6.1 Miss rate by arrival order within the unknown class (U1, 45 streams, standalone)", "",
         "Miss rate at each detector's own 95%-ID threshold. `re-scored`: the issued scores are never changed; the image is scored again with the "
         "graph and memory after 5 further batches, or at the end of the stream.", ""]
    w, rep = lock["u1w"], retro["fam"]["reprise"]
    rows = [["static p"] + [f"{w['bins'][b]['static']['mean']:.2f}" for b in BINS],
            ["zeta"] + [f"{u1['bins']['zeta'][b]['miss']['mean']:.2f}" for b in BINS],
            ["Salmon Ladder"] + [f"{w['bins'][b]['frozen']['mean']:.2f}" for b in BINS],
            ["extension"] + [f"{w['bins'][b]['cand']['mean']:.2f}" for b in BINS],
            ["Salmon Ladder, re-scored after 5 batches"] + [f"{rep['5']['miss'][b]['mean']:.2f}" for b in BINS],
            ["Salmon Ladder, re-scored at the end"] + [f"{rep['end']['miss'][b]['mean']:.2f}" for b in BINS]]
    L += table(["detector"] + ["k = " + b.replace("-", "–") if b.split("-")[0] != b.split("-")[1] else "k = " + b.split("-")[0] for b in BINS], rows)
    rows = [[lab, af(rep[k]["AUROC"]["mean"], rep[k]["FPR95"]["mean"]), f"{rep[k]['miss']['1-1']['mean']:.2f}"]
            for lab, k in (("as issued", "0"), ("after 1 batch", "1"), ("after 2 batches", "2"), ("after 5 batches", "5"), ("after 10 batches", "10"),
                           ("after 20 batches", "20"), ("at the end of the stream", "end"))]
    L += ["Overall metrics of Salmon Ladder under delayed re-scoring:", ""] + table(["re-scored", "AUROC / FPR95", "miss rate at k = 1"], rows)

    rows = []
    for k in ("R", "C", "N1", "N+"):
        cs = w["cases"][k]
        rows.append([k, f"{cases['share'][k]['mean']:.2f}", f"{cases['fam']['static']['case_AUROC'][k]:.2f}", f"{cases['fam']['zeta']['case_AUROC'][k]:.2f}",
                     f"{cases['fam']['reprise']['case_AUROC'][k]:.2f}", f"{cs['frozen_miss']['mean']:.2f}", f"{cs['cand_miss']['mean']:.2f}", ci(cs["miss_diff"])])
    f = w["none|FPR95"]
    rows.append(["all", "100.00", f"{cases['fam']['static']['AUROC']['mean']:.2f}", f"{cases['fam']['zeta']['AUROC']['mean']:.2f}",
                 f"{cases['fam']['reprise']['AUROC']['mean']:.2f}", f"{f['frozen']['mean']:.2f}", f"{f['cand']['mean']:.2f}", ci(f["diff"])])
    ce = cases["ceiling"]
    L += ["### 6.2 Unknown images by the evidence available at arrival (U1)", "",
          "R: an image of the same class from an earlier batch is in the memory. C: none, but an admitted image of the same class is in the same batch. "
          "N1: neither, first image of its class. N+: neither, although the class appeared before (none of its images was admitted). The split uses "
          "labels and is a diagnostic, not a detector. AUROC of a case is computed against all ID images.", ""]
    L += table(["case", "share of unknown images [%]", "static p AUROC", "zeta AUROC", "Salmon Ladder AUROC", "Salmon Ladder miss rate", "extension miss rate",
                "extension - Salmon Ladder (miss rate)"], rows)
    L += [f"Ceiling: if every non-R image were detected as well as an R image, the overall AUROC would rise from {ce['overall']:.2f} to {ce['all_as_R']:.2f}; "
          f"if only first images (N1) were, to {ce['k1_as_R']:.2f}.", ""]

    rows = []
    for key, lab in (("within_only", "memory also reads the same batch (only)"), ("zero_only", "zero-start propagation (only)"),
                     ("hinge2_zero", "one-sided memory + zero-start propagation"), ("hinge2_within", "one-sided memory + same batch"), ("ext", "extension (locked)")):
        rows.append([lab] + [ci(xb[bank]["decomposition"][key][m]) for bank in ("U1", "U2") for m in ("AUROC", "FPR95")])
    L += ["### 6.3 Components of the extension (difference to Salmon Ladder, standalone; descriptive)", ""]
    L += table(["read-out", "U1 AUROC", "U1 FPR95", "U2 AUROC", "U2 FPR95"], rows)
    L += ["The extension multiplied by each base detector is listed in Section 1.1. On U4 the extension gives "
          f"{af(lock['u4w']['none|AUROC']['cand']['mean'], lock['u4w']['none|FPR95']['cand']['mean'])} "
          f"against {af(lock['u4w']['none|AUROC']['frozen']['mean'], lock['u4w']['none|FPR95']['frozen']['mean'])} for Salmon Ladder "
          f"(FPR95 difference {ci(lock['u4w']['none|FPR95']['diff'])}).", ""]

    u1s, u2s = B["U1"]["sets"], B["U2"]["sets"]
    assert set(SINGLE + PAIRS) == set(u1s) == set(u2s), "the table must list every evaluated feature set"

    def brow(name):
        o, q, vs = u1s[name], u2s[name], name.split("+")
        lab = " x ".join(VIEW[v] for v in vs) + (" †" if name in POSTHOC else "")
        return [lab, f"{sum(bench[v]['params_M'] for v in vs):.0f}", f"{sum(bench[v]['ms_per_image'] for v in vs):.1f}", f"{ncm[name]['mean']:.1f}" if name in ncm else "–",
                cell(o, "static|none"), cell(o, "zeta|none"), cell(o, "reprise|none"), ci(o["reprise-zeta"]["FPR95"]), cell(o, "ext|none"), ci(o["ext-reprise"]["FPR95"]),
                f"{q['zeta|none']['FPR95']['mean']:.2f}", f"{q['reprise|none']['FPR95']['mean']:.2f}", ci(q["reprise-zeta"]["FPR95"]),
                f"{q['ext|none']['FPR95']['mean']:.2f}", ci(q["ext-reprise"]["FPR95"])]

    L += ["### 6.4 Feature extractors (frozen hyper-parameters; U1: unit split, U2: unit draw)", "",
          "Parameters [millions] and encoding time per image [ms] (RTX 2080 Ti, single precision; CLIP RN50 and ViT-B/16 run in half precision as in "
          "the OpenAI implementation) refer to the image encoder. NCM: accuracy [%] of the nearest class mean of the 12 support images on the ID images "
          "of U1. † marks feature sets added after the first results had been seen (amendment 03); all of them are reported.", ""]
    head = ["features", "params", "ms", "NCM", "U1 static p", "U1 zeta", "U1 Salmon Ladder", "U1 Salmon Ladder - zeta (FPR95)", "U1 extension", "U1 extension - Salmon Ladder (FPR95)",
            "U2 zeta FPR95", "U2 Salmon Ladder FPR95", "U2 Salmon Ladder - zeta", "U2 extension FPR95", "U2 extension - Salmon Ladder"]
    L += table(head, [brow(n) for n in SINGLE] + [brow(n) for n in PAIRS])
    hb1, hb2 = B["U1"]["H_B"], B["U2"]["H_B"]
    L += [f"Registered hypothesis H-B (DINOv2 ViT-L/14 x DINOv3 ViT-L/16 against the main configuration, Salmon Ladder, standalone FPR95): "
          f"U1 {ci(hb1['standalone_FPR95'])}, U2 {ci(hb2['standalone_FPR95'])}; confirmed: {hb1['confirmed']}.", ""]

    rows = []
    for r in [x for x in c1["table"] if x["cond"] == "same"]:
        c = r["cells"]
        rows.append([r["r"], f"{c['c00']:.2f}", f"{c['cr0']:.2f}", f"{c['c0r']:.2f}", f"{c['crr']:.2f}", ci(r["E_M"]), ci(r["E_G"]), ci(r["E_MG"]), ci(r["I"])])
    L += ["### 6.5 Memory and propagation separated", "",
          "Intervention 2 x 2 (the 250 pairs of Section 3; the r images of the class under test are shown to the memory only, to the graph only, or to both; "
          "AUROC, class bootstrap):", ""]
    L += table(["r", "neither", "memory only", "graph only", "both", "effect of the memory", "effect of the graph", "effect of both", "interaction"], rows,
               align=["r"] * 9)
    ov = u1["overall"]
    rows = [[lab, cell(ov[k]["none"])] for lab, k in (("no history (static p x batch-only propagation)", "static_lpB"), ("memory history only", "mem_lpB"),
                                                      ("propagation history only", "static_lpzero"), ("both (memory x zero-start propagation)", "mem_lpzero"),
                                                      ("Salmon Ladder (memory x warm-start propagation)", "reprise"), ("zeta", "zeta"), ("static p", "static"))]
    L += ["Stream 2 x 2 on U1 (the history of each path is switched off separately):", ""] + table(["read-out", "AUROC / FPR95"], rows)

    def drow(r):
        return [DS[r["ds"]], f"{r['C']:.0f}", f"{r['U']:.0f}", f"{r['m']:.0f}", f"{r['pi']:.1f}", cell(r, "knn|none"), cell(r, "maha|none"), cell(r, "static|none"),
                cell(r, "zeta|none"), cell(r, "reprise|none"), ci(r["reprise-zeta"]["FPR95"]), cell(r, "ext|none"), ci(r["ext-reprise"]["FPR95"]), r["units"]]

    L += ["### 6.6 Data sets (frozen hyper-parameters, DINOv2 ViT-B/14 x ViT-L/14)", "",
          "C: ID classes, U: unknown classes, m: median number of images per unknown class. kNN and Mahalanobis++ are static detectors on the same "
          "features and the same 16 labelled images. Unit: split (five), except the SSB split of CUB (three orders).", ""]
    head = ["data set", "C", "U", "m", "unknown [%]", "kNN", "Mahalanobis++", "static p", "zeta", "Salmon Ladder", "Salmon Ladder - zeta (FPR95)", "extension",
            "extension - Salmon Ladder (FPR95)", "units"]
    n = 0
    for cond, title in (("nat", "Natural streams (all evaluation images of the data set)"), ("mat", "Matched composition (40 unknown classes x 25 images, 4,000 ID images)")):
        rows = [drow(r) for k in DS for r in D if r["cond"] == cond and r["ds"] == k]
        n += len(rows)
        L += [f"**{title}**", ""] + table(head, rows)
    assert n == len(D) == 14

    e1 = sorted([r for r in Ee["e1"] if r["cell"]["U"] == 100], key=lambda r: (r["cell"]["n_id"], r["cell"]["m"]))
    rows = [[f"{r['cell']['n_id']:,}", r["cell"]["m"], f"{r['pi']:.2f}", f"{r['static|none']['FPR95']['mean']:.2f}", f"{r['zeta|none']['FPR95']['mean']:.2f}",
             f"{r['reprise|none']['FPR95']['mean']:.2f}", f"{r['ext|none']['FPR95']['mean']:.2f}", ci(r["reprise-static"]["FPR95"]), ci(r["reprise-zeta"]["FPR95"])] for r in e1]
    L += ["### 6.7 Stream conditions (U1 images, five splits, FPR95)", "",
          "Composition: 100 unknown classes with m images each and the given number of ID images, random order, batches of 256 "
          f"({len(Ee['e1'])} compositions were run in total; the remaining ones are in `experiments/phase7/results/an_e.json`).", ""]
    L += table(["ID images", "m", "unknown [%]", "static p", "zeta", "Salmon Ladder", "extension", "Salmon Ladder - static p", "Salmon Ladder - zeta"], rows, align=["r"] * 9)
    e2 = {(r["cell"]["comp"], r["cell"]["pattern"]): r for r in Ee["e2"]}
    rows = [[comp, lab] + [f"{e2[(comp, pk)][f'{f}|none']['FPR95']['mean']:.2f}" for f in ("static", "zeta", "mem", "reprise", "ext")] +
            [ci(e2[(comp, pk)]["reprise-zeta"]["FPR95"]), ci(e2[(comp, pk)]["ext-reprise"]["FPR95"])]
            for comp in ("native", "sparse") for pk, lab in PATTERNS]
    L += ["Arrival order (18,000 ID images; `native`: 100 unknown classes x 50 images, `sparse`: x 10 images):", ""]
    L += table(["composition", "order", "static p", "zeta", "memory only", "Salmon Ladder", "extension", "Salmon Ladder - zeta", "extension - Salmon Ladder"], rows,
               align=["l", "l"] + ["r"] * 7)
    e3 = {(r["cell"]["pattern"], r["cell"]["batch"]): r for r in Ee["e3"]}
    rows = [[lab, b] + [f"{e3[(pk, b)][f'{f}|none']['FPR95']['mean']:.2f}" for f in ("zeta", "mem", "reprise", "ext")] +
            [ci(e3[(pk, b)]["reprise-zeta"]["FPR95"]), ci(e3[(pk, b)]["ext-reprise"]["FPR95"])]
            for pk, lab in (("random", "random order"), ("ood_burst", "unknown images in runs of one class")) for b in (1, 16, 64, 256)]
    L += ["Batch size (100 unknown classes x 50 images, 18,000 ID images):", ""]
    L += table(["order", "batch", "zeta", "memory only", "Salmon Ladder", "extension", "Salmon Ladder - zeta", "extension - Salmon Ladder"], rows, align=["l"] + ["r"] * 7)
    return L


# ------------------------------------------------------------------------------------------------ 7. propagation identities
def section_identities():
    t = J(P4 / "results/theory_checks.json")
    t3 = J(P4 / "results/theory_t3.json")["T3"]
    T1 = t["T1"]
    ratio = [r["mean_u_L2"] / r["Ns_over_Nt"] for r in T1]
    L = ["## 7. Checks of the propagation identities (development split)", "",
         f"One development stream, ViT-L/14, {len(T1)} batches (`experiments/phase4/results/theory_checks.json`). The share of support nodes "
         f"N_s / N_t falls from {T1[0]['Ns_over_Nt']:.2f} to {T1[-1]['Ns_over_Nt']:.2f} while the mean of the converged solution stays between "
         f"{min(ratio):.2f} and {max(ratio):.2f} times that share. The weighted mass sum_i sqrt(d_i) u_i divided by the mass of the support nodes "
         f"is {np.mean([r['mass_L2'] for r in T1]):.3f} for the converged solution, {np.mean([r['mass_L0'] for r in T1]):.3f} for 15 warm-start sweeps "
         f"and {np.mean([r['mass_L1'] for r in T1]):.3f} for 15 sweeps from zero.", ""]
    rows = []
    for key, lab in (("near|B14|L0", "near, ViT-B/14, warm start"), ("near|L14|L0", "near, ViT-L/14, warm start"), ("far|B14|L0", "far, ViT-B/14, warm start"),
                     ("far|L14|L0", "far, ViT-L/14, warm start"), ("near|B14|L1", "near, ViT-B/14, zero start"), ("near|L14|L1", "near, ViT-L/14, zero start")):
        a, b3 = t["T2"][key], t3[key]
        rows.append([lab, f"{a['rho_consecutive_mean']:.3f}", f"{a['rho_first_vs_last_mean']:.3f}", f"{b3['consecutive_mean_abs_dp']:.4f}",
                     f"{b3['consecutive_bound']:.4f}", f"{b3['first_last_mean_abs_dp']:.3f}", f"{b3['first_last_bound']:.3f}"])
    L += ["Calibration images across batches (Spearman correlation of their masses; mean change of their leave-one-out rank and the bound from the "
          "number of order flips):", ""]
    L += table(["stream, view, start", "rho, consecutive batches", "rho, first vs last batch", "rank change, consecutive", "bound",
                "rank change, first vs last", "bound"], rows)
    return L



# ------------------------------------------------------------------------------------------------ 8. Phases 10 and 11: public benchmarks
P1011 = E / "phase10_11"
OO_NEAR, OO_FAR, FOUR_SETS = ["ssb_hard", "ninco"], ["inaturalist", "textures", "openimageo"], ["inat", "sun", "places", "dtd"]
MAIN11 = "L14xD3L+D3L|T|TINS"
VIEW11 = {"L14": "DINOv2 ViT-L/14", "D3L": "DINOv3 ViT-L/16", "L14xD3L": "joint view (DINOv2 L/14 (+) DINOv3 L/16)", "B14+L14": "DINOv2 B/14 + L/14",
          "L14+D3L": "DINOv2 L/14 + DINOv3 L/16", "D3B+D3L": "DINOv3 B/16 + L/16", "L14xD3L+D3L": "joint view + DINOv3 L/16 (the paper)",
          "L14+D3B+D3L": "DINOv2 L/14 + DINOv3 B/16 + L/16", "B14+L14+D3L": "DINOv2 B/14 + L/14 + DINOv3 L/16", "B14+L14+D3B+D3L": "all four encoders"}
READ11 = {"Tlp": "p+ only (no seeds)", "Tlp20": "p+ only, k_g = 20", "Tq0.05": "p+ x p-, q = 0.05", "T": "p+ x p-, q = 0.1 (the paper)",
          "Tq0.2": "p+ x p-, q = 0.2", "Tk20": "p+ x p-, k_g = 20", "Tnn": "p+ x p- x rank of the distance to the nearest seed"}


def rows1011(name):
    d = pd.read_csv(P1011 / name)
    d["ds"] = d.stream.str.replace(r"_seed\d+$", "", regex=True)
    d["seed"] = d.stream.str.extract(r"seed(\d+)$", expand=False).astype(int)
    d["cfg"] = d.views + "|" + d.readout + "|" + d.base
    return d


def agg1011(d):
    """Per configuration: mean over stream orders of the mean over the data sets of the group (near, far, fourood), the per-data-set
    means, and the paired FPR95 difference to the main configuration over stream orders."""
    out = {}
    for part, sets in (("openood", {"near": OO_NEAR, "far": OO_FAR}), ("fourood", {"fourood": FOUR_SETS})):
        dd = d[d.part == part]
        for g, names in sets.items():
            per_seed = dd[dd.ds.isin(names)].groupby(["cfg", "seed", "ds"])[["AUROC", "FPR95"]].mean().groupby(["cfg", "seed"]).mean()
            ref = per_seed.loc[MAIN11] if MAIN11 in per_seed.index.get_level_values(0) else None
            for c in per_seed.index.get_level_values(0).unique():
                v = per_seed.loc[c]
                e = {"AUROC": float(v.AUROC.mean()), "FPR95": float(v.FPR95.mean()), "n": int(len(v))}
                if ref is not None and len(v) == len(ref) and len(v) > 1:
                    e["dFPR95"] = tci((v.FPR95 - ref.FPR95).values)
                out[(c, g)] = e
        for (c, ds), v in dd.groupby(["cfg", "ds"])[["AUROC", "FPR95"]].mean().iterrows():
            out[(c, ds)] = {"AUROC": float(v.AUROC), "FPR95": float(v.FPR95)}
    return out


def section_phase1011():
    p10, p11 = agg1011(rows1011("phase10/results/p10_rows.csv")), agg1011(rows1011("phase11/results/final/p11_rows.csv"))

    def three(src, c):
        return [af(src[(c, g)]["AUROC"], src[(c, g)]["FPR95"]) for g in ("near", "far", "fourood")]

    def ds_row(src, c, names):
        return [af(src[(c, n)]["AUROC"], src[(c, n)]["FPR95"]) for n in names]

    def d_near(c):
        e = p11[(c, "near")]
        return ci(e["dFPR95"]) if "dFPR95" in e else ""

    L = ["## 8. Public benchmarks: Phase 10 (DINOv3 views, streaming) and Phase 11 (post-stream read-out; the paper of October 2026)", "",
         "Streams of the OpenOOD v1.5 ImageNet-1K benchmark (45,000 ID validation images mixed with one OOD data set; near-OOD: SSB-hard, NINCO; "
         "far-OOD: iNaturalist, Textures, OpenImage-O; five stream orders each) and of Four-OOD (50,000 ID images with iNaturalist, SUN, Places, "
         "Textures; three orders each). 16 labelled ImageNet-1K training images per class (12 support + 4 calibration). Means over the data sets "
         "of a group, then over the orders; the per-stream tables are `experiments/phase10_11/phase10/results/p10_rows.csv` and `phase11/results/final/p11_rows.csv`. The base detectors "
         "(official implementations, no training) were measured on the same streams in Phase 10. Phase 11 scores every image after the whole "
         "stream has arrived (Section 1 of `METHOD.md`); its configuration was selected on these streams, see `PREREGISTRATION.md`.", "",
         "### 8.1 Main result", ""]
    rows = [[f"{b} alone (measured)"] + three(p10, f"-|-|{b}") + [""] for b in ("MCM", "NegLabel", "AdaNeg", "TANL", "TINS")]
    rows += [["Salmon Ladder, standalone"] + three(p11, "L14xD3L+D3L|T|none") + [d_near("L14xD3L+D3L|T|none")],
             ["Salmon Ladder x TANL"] + three(p11, "L14xD3L+D3L|T|TANL") + [d_near("L14xD3L+D3L|T|TANL")],
             ["**Salmon Ladder x TINS** (the paper)"] + three(p11, MAIN11) + [""]]
    L += table(["detector", "near-OOD", "far-OOD", "Four-OOD", "near FPR95 - main (95% interval, unit: order)"], rows)
    L += ["### 8.2 Per data set", ""]
    names = OO_NEAR + OO_FAR
    rows = [["TINS alone (measured)"] + ds_row(p10, "-|-|TINS", names), ["Salmon Ladder, standalone"] + ds_row(p11, "L14xD3L+D3L|T|none", names),
            ["Salmon Ladder x TINS"] + ds_row(p11, MAIN11, names)]
    L += table(["OpenOOD v1.5", "SSB-hard", "NINCO", "iNaturalist", "Textures", "OpenImage-O"], rows)
    rows = [["TINS alone (measured)"] + ds_row(p10, "-|-|TINS", FOUR_SETS), ["Salmon Ladder, standalone"] + ds_row(p11, "L14xD3L+D3L|T|none", FOUR_SETS),
            ["Salmon Ladder x TINS"] + ds_row(p11, MAIN11, FOUR_SETS)]
    L += table(["Four-OOD", "iNaturalist", "SUN", "Places", "Textures"], rows)
    L += ["### 8.3 Views (p+ x p-, q = 0.1, k_g = 10, x TINS)", ""]
    rows = [[VIEW11[v]] + three(p11, f"{v}|T|TINS") + [d_near(f"{v}|T|TINS")] for v in VIEW11]
    L += table(["views", "near-OOD", "far-OOD", "Four-OOD", "near FPR95 - main"], rows)
    L += ["### 8.4 Read-outs (joint view + DINOv3 L/16, x TINS)", ""]
    rows = [[READ11[r]] + three(p11, f"L14xD3L+D3L|{r}|TINS") + [d_near(f"L14xD3L+D3L|{r}|TINS")] for r in READ11]
    L += table(["read-out", "near-OOD", "far-OOD", "Four-OOD", "near FPR95 - main"], rows)
    L += ["### 8.5 Seeds of the main configuration", "",
          "Storey-BH at q = 0.1 on the calibrated rank p+ of every stream image, per view. The ID fraction is the share of ID images among the "
          "selected seeds (the realised false discovery proportion), averaged over the orders; the guarantee of Proposition 3 of the paper assumes "
          "that the calibration images (ImageNet-1K training images) and the ID test images (validation images) are exchangeable.", ""]
    seeds = {}
    for f in sorted(P1011.glob("phase11/results/final/*/*.meta.json")):
        m = J(f)
        ds = m["stream"].rsplit("_seed", 1)[0]
        for v in ("L14xD3L", "D3L"):
            e = m["views"][v]["q0.1"]
            seeds.setdefault((m["part"], ds, v), []).append((e["seeds"], e["id_frac"], m["n_ood"]))
    rows = []
    for (part, ds, v), vals in seeds.items():
        a = np.array(vals, float)
        rows.append([part, ds, v, f"{a[:, 0].mean():,.0f}", f"{a[:, 2].mean():,.0f}", f"{a[:, 1].mean():.3f}"])
    L += table(["part", "data set", "view", "seeds", "OOD images", "ID fraction among the seeds"], rows, ["l", "l", "l", "r", "r", "r"])
    L += ["### 8.6 The streaming configuration on the same streams (Phase 10)", "",
          "The online read-out of Phases 4-9 (entrance memory p_M x warm-start propagation p_LP, `METHOD.md` Section 2) with the Phase 10 views, "
          "scored at arrival. It is superseded by the post-stream read-out above and is not part of the paper.", ""]
    rows = [[lab] + three(p10, c) for lab, c in (("DINOv2 B/14 + L/14, x TINS", "B14+L14|SL|TINS"), ("DINOv2 L/14 + DINOv3 L/16, standalone", "L14+D3L|SL|none"),
                                                   ("DINOv2 L/14 + DINOv3 L/16, x TINS", "L14+D3L|SL|TINS"))]
    L += table(["streaming Salmon Ladder (Phase 10)", "near-OOD", "far-OOD", "Four-OOD"], rows)
    return L

# ------------------------------------------------------------------------------------------------ README block
README = ROOT / "README.md"
BEGIN, END = "<!-- BEGIN GENERATED: results -->", "<!-- END GENERATED: results -->"


def readme_block():
    BL = pd.read_parquet(P4 / "results/baselines.parquet")
    summary = J(P4 / "results/eval_summary.json")
    s = J(P5 / "results_final/summary_openood.json")

    def bl(bank, base, visual):
        d = BL[(BL.bank == bank) & (BL.base == base) & (BL.visual == visual)]
        u = d.groupby(UNIT[bank])[["AUROC", "FPR95"]].mean()
        return af(float(u.AUROC.mean()), float(u.FPR95.mean()))

    def diff(bank, base):
        d = BL[(BL.bank == bank) & (BL.base == base)].pivot_table(index=["split", "draw", "seed"], columns="visual", values="FPR95")
        return ci(tci((d["REPRISE"] - d["minimal"]).groupby(level=UNIT[bank]).mean().values))

    names = {"U1": "U1: new class splits of ImageNet-1K", "U2": "U2: ImageNet-O"}
    rows = []
    for bank in ("U1", "U2"):
        rows.append([f"{names[bank]}, standalone", "–", bl(bank, "none", "minimal"), bl(bank, "none", "REPRISE"), diff(bank, "none")])
        rows.append([f"{bank}, with TINS", bl(bank, "TINS", "none"), bl(bank, "TINS", "minimal"), bl(bank, "TINS", "REPRISE"), diff(bank, "TINS")])
    for part, lab in (("near", "near-OOD"), ("far", "far-OOD")):
        o = s[f"{part}_TINS"]
        rows.append([f"OpenOOD v1.5 ImageNet-1K {lab}, with TINS", afd(o["base_only"]), afd(o["zeta"]), afd(o["frozen_v5"]), ""])
    e = summary["banks"]["U1"]
    p10, p11 = agg1011(rows1011("phase10/results/p10_rows.csv")), agg1011(rows1011("phase11/results/final/p11_rows.csv"))

    def three(src, c):
        return [af(src[(c, g)]["AUROC"], src[(c, g)]["FPR95"]) for g in ("near", "far", "fourood")]

    lines = ["Salmon Ladder (Phase 11, the paper): scores assigned after the whole stream has arrived, two views, 16 labelled images per class.", ""]
    lines += table(["public benchmark streams", "OpenOOD near-OOD", "OpenOOD far-OOD", "Four-OOD"],
                   [["TINS alone (measured on the same streams)"] + three(p10, "-|-|TINS"), ["Salmon Ladder, standalone"] + three(p11, "L14xD3L+D3L|T|none"),
                    ["**Salmon Ladder x TINS**"] + three(p11, MAIN11)])
    lines += ["The online variant of Phases 4-7 (entrance memory and warm-start propagation, scored at arrival), on the data that no selection had used:", ""]
    lines += table(["evaluation", "base detector alone", "x zeta", "x online variant", "online variant - zeta (FPR95, 95% interval)"], rows)
    lines += [f"Registered endpoints on U1: E1 is the standalone row. E2 uses TINS and gives zeta its development-selected weight "
              f"(FPR95 {e['E2_FPR95']['A_mean']:.2f} against {e['E2_FPR95']['B_mean']:.2f}): {ci(e['E2_FPR95'])}. The rows with TINS above use weight 1 "
              "for both methods.", ""]
    return "\n".join(lines)


def with_block(text, block):
    head, rest = text.split(BEGIN)
    return head + BEGIN + "\n" + block + END + rest.split(END)[1]


def build():
    L = ["# Results", "",
         "This file is generated by `tools/make_results_tables.py` from the result files archived under `experiments/` "
         "(`python tools/make_results_tables.py --check` verifies that it is up to date). Do not edit it by hand.", "",
         "Conventions: AUROC (ID as the positive class) / FPR95, both in %. A difference `a - b` is `a` minus `b` with a paired 95% t interval over "
         "the stated units (bootstrap intervals are marked); a negative FPR95 difference means that `a` is better. Salmon Ladder is the frozen "
         "configuration: DINOv2 ViT-B/14 and ViT-L/14 CLS features, 16 labelled images per class (12 support + 4 calibration), entrance thresholds "
         "(0.3, 0.2, 0.1019), graph k = 10, gamma = 1, lambda = 0.9, 15 sweeps, warm start. Which evaluation was registered before it was run, and how "
         "often the public test split was used, is recorded in `docs/PREREGISTRATION.md`.", ""]
    for section in (section_unused, section_test, section_intervention, section_operating, section_phase56, section_phase7, section_identities, section_phase1011):
        L += section()
    return "\n".join(L).rstrip() + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--check", action="store_true", help="compare with docs/RESULTS.md instead of writing it")
    args = parser.parse_args()
    text = build()
    readme = README.read_text(encoding="utf-8")
    new_readme = with_block(readme, readme_block())
    if args.check:
        same = OUT.is_file() and OUT.read_text(encoding="utf-8") == text
        print("docs/RESULTS.md is up to date" if same else "docs/RESULTS.md differs from the result files")
        print("the results table of README.md is up to date" if new_readme == readme else "the results table of README.md differs from the result files")
        sys.exit(0 if same and new_readme == readme else 1)
    OUT.write_text(text, encoding="utf-8")
    README.write_text(new_readme, encoding="utf-8")
    print(f"wrote {OUT.relative_to(ROOT)} ({len(text.splitlines())} lines) and the results table of README.md")


if __name__ == "__main__":
    main()
