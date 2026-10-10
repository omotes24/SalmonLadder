"""Phase 10 analysis: every (views x read-out x base detector) configuration of the frozen Salmon Ladder on the
public-benchmark streams, with the measured baselines, from
  P5/results_final/<part>/<stream>.npz        frozen views B14, L14 (Phase 5 final run; logS = TINS, logM = MCM)
  P10/results/<part>/D3B+D3L/<stream>.npz      new views D3B, D3L (same streams; final10.py)
  P10/results/vlm/<part>/<stream>.npz          official TANL / AdaNeg / AdaNeg_TA / NegLabel scores (vlm10.py)
Metrics: the upstream TINS measures (AUROC, FPR at 95% TPR), as every earlier phase. Aggregation: per dataset the mean
over orders; OpenOOD near = mean(SSB-hard, NINCO) and far = mean(iNaturalist, Textures, OpenImage-O) per order, then
the mean over the 5 orders with the paired t interval of the difference to the frozen reference (SL x TINS on B14+L14);
Four-OOD = mean of the four sets per order (3 orders).
Output: P10/results/p10_summary.json, p10_rows.csv (one row per configuration x stream), p10_table.txt."""
import argparse
import glob
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

import p10common as P

REFT = "k10g1b1l0.9"
READOUTS = {"SL": [("M0", 1.0), (f"lp0|{REFT}", 1.0)],         # frozen Salmon Ladder read-out: p_M x p_LP
            "M0": [("M0", 1.0)],                                 # memory only
            "lp0": [(f"lp0|{REFT}", 1.0)],                        # propagation only (warm start)
            "static": [("static", 1.0)],                          # static distance rank (no stream state)
            "z": [(f"z|{REFT}", 1.0)],                            # robust z of the zero-start propagation (zeta)
            "M0z": [("M0", 1.0), (f"z|{REFT}", 1.0)],
            "SLs": [("M0", 1.0), (f"lp0|{REFT}", 1.0), ("static", 1.0)]}
VIEWSETS = [("B14", "L14"), ("L14", "D3L"), ("D3B", "D3L"), ("D3L",), ("L14",), ("B14",), ("D3B",),
            ("B14", "L14", "D3L"), ("L14", "D3B", "D3L"), ("B14", "L14", "D3B", "D3L"), ("B14", "D3L")]
BASES = ["none", "TINS", "MCM", "TANL", "AdaNeg", "AdaNeg_TA", "NegLabel"]
REF = ("B14+L14", "SL", "TINS")
_UP = {}


def measures(x, f):
    if "ok" not in _UP:
        from vins.tins_dev import import_tins
        import_tins()
        _UP["ok"] = True
    from vins.metrics import measures as m_
    m = m_(x[~f], x[f])
    return 100 * m["AUROC"], 100 * m["FPR95"]


def load_stream(part, name, new_tag="D3B+D3L"):
    z5 = np.load(P.P5 / "results_final" / part / f"{name}.npz", allow_pickle=True)
    z10 = np.load(P.RESULTS / part / new_tag / f"{name}.npz", allow_pickle=True)
    zv = np.load(P.RESULTS / "vlm" / part / f"{name}.npz", allow_pickle=True)
    assert np.array_equal(z5["sample_id"], z10["sample_id"]) and np.array_equal(z5["sample_id"], zv["sample_id"]), name
    assert np.array_equal(z5["is_ood"], z10["is_ood"]) and np.array_equal(z5["is_ood"], zv["is_ood"]), name
    S = {"is_ood": z5["is_ood"].astype(bool), "base::none": None, "base::TINS": z5["logS"], "base::MCM": z5["logM"]}
    for z, views in ((z5, P.BASE_VIEWS), (z10, P.NEW_VIEWS)):
        for v in views:
            for ro in {k for c in READOUTS.values() for k, _ in c}:
                S[f"s::{v}::{ro}"] = z[f"s::{v}::{ro}"].astype(np.float64)
    for m in BASES[3:]:
        S[f"base::{m}"] = np.log(np.maximum(zv[m].astype(np.float64), 1e-300))
    return S


def score(S, views, readout, base):
    x = np.zeros_like(S["is_ood"], dtype=np.float64)
    for v in views:
        for key, w in READOUTS[readout]:
            x = x + w * S[f"s::{v}::{key}"]
    if base != "none":
        x = x + S[f"base::{base}"]
    return x


def tci(d):
    d = np.asarray(d, np.float64)
    n = len(d)
    from scipy import stats
    m = d.mean()
    if n < 2:
        return {"mean": float(m), "lo": float("nan"), "hi": float("nan"), "n": n}
    h = stats.t.ppf(0.975, n - 1) * d.std(ddof=1) / np.sqrt(n)
    return {"mean": float(m), "lo": float(m - h), "hi": float(m + h), "n": n}


def part_rows(part):
    rows = []
    files = sorted(glob.glob(str(P.P5 / "results_final" / part / "*.npz")))
    for f in files:
        name = Path(f).stem
        S = load_stream(part, name)
        flag = S["is_ood"]
        for b in BASES[1:]:
            au, fp = measures(S[f"base::{b}"], flag)
            rows.append({"part": part, "stream": name, "views": "-", "readout": "-", "base": b, "AUROC": au, "FPR95": fp})
        for vs in VIEWSETS:
            for ro in READOUTS:
                for b in BASES:
                    au, fp = measures(score(S, vs, ro, b), flag)
                    rows.append({"part": part, "stream": name, "views": "+".join(vs), "readout": ro, "base": b, "AUROC": au, "FPR95": fp})
        print(json.dumps({"part": part, "stream": name, "rows": len(rows)}), flush=True)
    return rows


def aggregate(df):
    df = df.copy()
    df["ds"] = df.stream.str.replace(r"_seed\d+$", "", regex=True)
    df["seed"] = df.stream.str.extract(r"seed(\d+)$", expand=False).astype(int)
    df["cfg"] = df.views + "|" + df.readout + "|" + df.base
    out = {}
    for part, d in df.groupby("part"):
        groups = {"near": P.OO_NEAR, "far": P.OO_FAR} if part == "openood" else {"fourood": P.FOUR}
        per_ds = d.groupby(["cfg", "ds"])[["AUROC", "FPR95"]].mean()
        res = {"per_dataset": {f"{c}|{ds}": {k: float(v) for k, v in r.items()} for (c, ds), r in per_ds.iterrows()}}
        for g, sets in groups.items():
            dd = d[d.ds.isin(sets)]
            per_seed = dd.groupby(["cfg", "seed", "ds"])[["AUROC", "FPR95"]].mean().groupby(["cfg", "seed"]).mean()
            ref_key = "|".join(REF)
            ref = per_seed.loc[ref_key] if ref_key in per_seed.index.get_level_values(0) else None
            gres = {}
            for c in per_seed.index.get_level_values(0).unique():
                v = per_seed.loc[c]
                e = {"AUROC": float(v.AUROC.mean()), "FPR95": float(v.FPR95.mean()), "n_orders": int(len(v))}
                if ref is not None and len(v) == len(ref):
                    e["dAUROC_vs_ref"] = tci((v.AUROC - ref.AUROC).values)
                    e["dFPR95_vs_ref"] = tci((v.FPR95 - ref.FPR95).values)
                gres[c] = e
            res[g] = gres
        out[part] = res
    return out


def table(summary):
    lines = []
    for part, res in summary.items():
        if part.startswith("_"):
            continue
        for g in [k for k in res if k != "per_dataset"]:
            lines.append(f"== {part} {g}: configurations sorted by FPR95 (top 40) ==")
            items = sorted(res[g].items(), key=lambda kv: kv[1]["FPR95"])
            for c, e in items[:40]:
                d = e.get("dFPR95_vs_ref")
                extra = f"  dFPR95 {d['mean']:+.2f} [{d['lo']:+.2f},{d['hi']:+.2f}]" if d else ""
                lines.append(f"{c:40s} {e['AUROC']:6.2f} / {e['FPR95']:6.2f}{extra}")
            lines.append(f"-- {part} {g}: baselines and the frozen reference --")
            for c in [k for k in res[g] if k.startswith("-|-|")] + ["|".join(REF)]:
                if c in res[g]:
                    e = res[g][c]
                    lines.append(f"{c:40s} {e['AUROC']:6.2f} / {e['FPR95']:6.2f}")
            lines.append("")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parts", default="openood,fourood")
    ap.add_argument("--table-only", action="store_true", help="rewrite p10_table.txt from the existing p10_summary.json")
    a = ap.parse_args()
    if a.table_only:
        summary = json.loads((P.RESULTS / "p10_summary.json").read_text())
        t = table(summary)
        (P.RESULTS / "p10_table.txt").write_text(t + "\n")
        print(t)
        print("ANALYSIS10_DONE", flush=True)
        return
    rows = []
    for part in a.parts.split(","):
        rows += part_rows(part)
    df = pd.DataFrame(rows)
    P.RESULTS.mkdir(parents=True, exist_ok=True)
    df.to_csv(P.RESULTS / "p10_rows.csv", index=False)
    summary = aggregate(df)
    summary["_meta"] = {"utc": P.utc(), "readouts": {k: [list(c) for c in v] for k, v in READOUTS.items()},
                        "viewsets": ["+".join(v) for v in VIEWSETS], "bases": BASES, "reference": "|".join(REF),
                        "n_rows": int(len(df))}
    (P.RESULTS / "p10_summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    t = table(summary)
    (P.RESULTS / "p10_table.txt").write_text(t + "\n")
    print(t)
    print("ANALYSIS10_DONE", flush=True)


if __name__ == "__main__":
    main()
