"""The locked extension (amendment 02) multiplied by the base detectors of the paper's Table 1, on U1 and U2.
Descriptive (after the lock; nothing is selected): the registered evaluation of the extension is the standalone score
and the product with TINS on U1 / U4 (an_lock_eval.py). Base scores: MCM and TINS from the stream files,
NegLabel / AdaNeg / TANL from Phase 4 (results/vlm_tta, the official postprocessors on the same sample order).
Units: split (U1, 9 streams each) and support draw (U2, 3 orders each); paired 95% t intervals (df = 4).
The zeta and REPRISE columns must reproduce Table 1 of the paper (checked against Phase 4's baselines_summary.json)."""
import json

import numpy as np

import an7 as A
from metrics_p4 import metrics, tci
from p7common import P4

VIEWS = ["B14", "L14"]
BASES = ["none", "MCM", "NegLabel", "AdaNeg", "TANL", "TINS"]
VIS = ["zeta", "reprise", "ext"]
EXTRA = {"within_only": ("M@0", 0.0, "lp0", 1.0), "zero_only": ("M", 0.0, "lp", 1.0), "hinge2_zero": ("Mh2", 0.0, "lp", 1.0), "hinge2_within": ("Mh2@0", 0.0, "lp0", 1.0)}


def main():
    spec = A.locked_spec()
    assert spec is not None, "no lock"
    out = {"spec": list(spec), "banks": {}}
    ref = P4 / "results" / "baselines_summary.json"
    ref = {(r["bank"], r["base"], r["visual"]): r for r in json.loads(ref.read_text())["rows"]} if ref.exists() else {}
    worst = 0.0
    for bank, exp, unit in (("U1", "u1w", "split"), ("U2", "u2w", "draw")):
        ts = [t for t in A.tasks(exp, VIEWS[0]) if (A.RESULTS / exp / VIEWS[1] / f"{t}.npz").exists()]
        if not ts:
            continue
        rows = []
        for t in ts:
            info = A.parse_std(t)
            s = A.Stream(exp, t, VIEWS)
            v = np.load(P4 / "results" / "vlm_tta" / f"{bank}_s{info['split']}_seed{info['seed']}.npz", allow_pickle=True)
            assert list(v["sample_id"]) == list(s.z[VIEWS[0]]["sample_id"]), t
            assert np.array_equal(v["is_ood"].astype(bool), s.is_ood), t
            base = {"none": 0.0, "MCM": s.logM, "TINS": s.logS, "NegLabel": np.log(v["NegLabel"]), "AdaNeg": np.log(v["AdaNeg"]), "TANL": np.log(v["TANL"])}
            vis = {"zeta": s.family("zeta"), "reprise": s.family("reprise"), "ext": s.cand(spec)}
            row = dict(info)
            for n, sp in EXTRA.items():                     # descriptive decomposition of the extension (standalone)
                m = metrics(s.cand(sp), s.is_ood)
                row[f"none|{n}|AUROC"], row[f"none|{n}|FPR95"] = m["AUROC"], m["FPR95"]
            row["static|AUROC"], row["static|FPR95"] = (lambda m: (m["AUROC"], m["FPR95"]))(metrics(s.family("static"), s.is_ood))
            for b in BASES:
                if b != "none":
                    m = metrics(base[b], s.is_ood)
                    row[f"{b}|s0|AUROC"], row[f"{b}|s0|FPR95"] = m["AUROC"], m["FPR95"]
                for f in VIS:
                    m = metrics(base[b] + vis[f], s.is_ood)
                    row[f"{b}|{f}|AUROC"], row[f"{b}|{f}|FPR95"] = m["AUROC"], m["FPR95"]
            rows.append(row)
        g = lambda key: A.unit_mean(rows, unit, key)[0]
        res = {"n_streams": len(rows), "unit": unit, "bases": {}}
        print(f"##### {bank} ({exp}, {len(rows)} streams, unit {unit}): AUROC / FPR95")
        print(f"{'base':9s} | s0 only       | x zeta        | x REPRISE     | x extension   | ext - REPRISE: AUROC              FPR95                     | ext - zeta FPR95")
        for b in BASES:
            o = {}
            for f in (["s0"] if b != "none" else []) + VIS:
                o[f] = {m: tci(g(f"{b}|{f}|{m}")) for m in ("AUROC", "FPR95")}
            o["ext-reprise"] = {m: tci(g(f"{b}|ext|{m}") - g(f"{b}|reprise|{m}")) for m in ("AUROC", "FPR95")}
            o["ext-zeta"] = {m: tci(g(f"{b}|ext|{m}") - g(f"{b}|zeta|{m}")) for m in ("AUROC", "FPR95")}
            o["reprise-zeta"] = {m: tci(g(f"{b}|reprise|{m}") - g(f"{b}|zeta|{m}")) for m in ("AUROC", "FPR95")}
            res["bases"][b] = o
            c = lambda f: f"{o[f]['AUROC']['mean']:5.2f} / {o[f]['FPR95']['mean']:5.2f}" if f in o else " " * 13
            print(f"{b:9s} | {c('s0')} | {c('zeta')} | {c('reprise')} | {c('ext')} | {A.fmt_ci(o['ext-reprise']['AUROC']):22s} {A.fmt_ci(o['ext-reprise']['FPR95']):25s} | {A.fmt_ci(o['ext-zeta']['FPR95'])}")
            for f, vname in (("zeta", "minimal"), ("reprise", "REPRISE"), ("s0", "none")):
                r = ref.get((bank, b, vname))
                if r is not None and f in o:
                    worst = max(worst, abs(r["AUROC"][0] - o[f]["AUROC"]["mean"]), abs(r["FPR95"][0] - o[f]["FPR95"]["mean"]))
        res["decomposition"] = {n: {m: tci(g(f"none|{n}|{m}") - g(f"none|reprise|{m}")) for m in ("AUROC", "FPR95")} for n in list(EXTRA) + ["ext"]}
        res["static"] = {m: tci(g(f"static|{m}")) for m in ("AUROC", "FPR95")}
        print("  decomposition (standalone, difference to REPRISE): " + "; ".join(
            f"{n}: AUROC {A.fmt_ci(v['AUROC'])}, FPR95 {A.fmt_ci(v['FPR95'])}" for n, v in res["decomposition"].items()))
        out["banks"][bank] = res
    out["max_abs_diff_to_phase4_table"] = worst
    print(f"largest |difference| of the s0 / zeta / REPRISE cells to Phase 4's baselines_summary.json: {worst:.4f}")
    A.dumpj(A.RESULTS / "an_ext_bases.json", out)
    print("written", A.RESULTS / "an_ext_bases.json")


if __name__ == "__main__":
    main()
