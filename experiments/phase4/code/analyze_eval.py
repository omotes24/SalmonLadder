"""Confirmatory analysis of Phase 4 (registered endpoints E1/E2 and secondary comparisons)."""
import json

import numpy as np
import pandas as pd

from common import RESULTS, ROOT, dump, utc
from evaluate import load_lock
from metrics_p4 import tci

REF = "k10g1l0.9"


def load():
    fs = sorted((RESULTS / "eval").glob("*_seed*.parquet"))
    return pd.concat([pd.read_parquet(f) for f in fs], ignore_index=True)


def pick(df, family, config, base, a=1.0, role=None):
    d = df[(df.family == family) & (df.config == config) & (df.base == base) & (np.isclose(df.a, a))]
    if role:
        d = d[d.role == role]
    return d.set_index(["bank", "split", "draw", "seed"])[["AUROC", "FPR95"]]


def paired(df, bank, a_sel, b_sel, metric="FPR95"):
    A = pick(df, *a_sel).loc[bank]
    B = pick(df, *b_sel).loc[bank]
    d = (A[metric] - B[metric]).dropna()
    unit = "split" if bank == "U1" else "draw"
    vals = d.groupby(level=unit).mean()
    return {**tci(vals.values), "unit": unit, "n_streams": int(len(d)), "A_mean": float(A[metric].mean()),
            "B_mean": float(B[metric].mean()), "per_unit": {str(k): float(v) for k, v in vals.items()}}


def main():
    df = load()
    lock = load_lock()
    sel = lock["selected"]
    best = lock["best_minimal"]
    wa = lock["weighted"][best]["a"]
    frozen_none = ("rep_L0", REF, "none")
    frozen_tins = ("rep_L0", REF, "TINS")
    res = {"utc": utc(), "best_minimal": best, "best_config": sel[best]["config"], "a_star": wa, "banks": {}}
    for bank in ("U1", "U2", "U3"):
        if bank not in set(df.bank):
            continue
        r = {}
        for metric in ("FPR95", "AUROC"):
            r[f"E1_{metric}"] = paired(df, bank, frozen_none, (best, sel[best]["config"], "none"), metric)
            r[f"E2_{metric}"] = paired(df, bank, frozen_tins, (best, sel[best]["config"], "TINS", wa, "weighted"), metric)
        comps = {}
        for f in sel:
            for base in ("none", "TINS", "MCM"):
                comps[f"{f}@{sel[f]['config']}|{base}"] = {m: paired(df, bank, (f, sel[f]["config"], base), ("rep_L0", REF, base), m)
                                                             for m in ("FPR95", "AUROC")}
        comps[f"raw_L2@{sel['raw']['config']}|none"] = {m: paired(df, bank, ("raw_L2", sel["raw"]["config"], "none"), frozen_none, m)
                                                          for m in ("FPR95", "AUROC")}
        r["vs_frozen"] = comps
        means = df[df.bank == bank].groupby(["family", "config", "role", "base", "a"])[["AUROC", "FPR95"]].mean().reset_index()
        r["means"] = means.to_dict(orient="records")
        res["banks"][bank] = r
    dump(RESULTS / "eval_summary.json", res)
    lines = [f"# Phase 4 confirmatory results ({utc()})", "", f"best minimal: {best} @ {sel[best]['config']} (a* = {wa})", ""]
    for bank, r in res["banks"].items():
        lines.append(f"## {bank}")
        for key in ("E1_FPR95", "E1_AUROC", "E2_FPR95", "E2_AUROC"):
            e = r[key]
            lines.append(f"- {key}: REPRISE-frozen minus best = {e['mean']:+.2f} [{e['lo']:+.2f}, {e['hi']:+.2f}] "
                         f"(n={e['n']} {e['unit']}s; A={e['A_mean']:.2f}, B={e['B_mean']:.2f})")
        lines.append("")
        lines.append("| family@config | base | dFPR95 vs frozen | dAUROC vs frozen |")
        lines.append("|---|---|---|---|")
        for k, v in r["vs_frozen"].items():
            f_, b_ = k.split("|")
            lines.append(f"| {f_} | {b_} | {v['FPR95']['mean']:+.2f} [{v['FPR95']['lo']:+.2f}, {v['FPR95']['hi']:+.2f}] | "
                         f"{v['AUROC']['mean']:+.2f} [{v['AUROC']['lo']:+.2f}, {v['AUROC']['hi']:+.2f}] |")
        lines.append("")
    (RESULTS / "eval_summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
