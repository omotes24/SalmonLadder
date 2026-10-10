"""Phase 11 analysis: streaming vs transductive read-outs (and their mixtures) on the public-benchmark streams.
Reads P11/results/<part>/<stream>.npz (trans11.py) and the official VLM scores (P10/results/vlm). Same metrics and
aggregation as analysis10 (upstream TINS measures; near/far/Four-OOD per order; paired t interval vs the streaming
reference L14+D3L|SL|TINS). Output: P11/results/p11_summary.json, p11_table.txt."""
import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd

import p11common as P

REFT = P.REFT
READOUTS = {
    "SL": [("M0", 1.0), (f"lp0|{REFT}", 1.0)],                       # streaming (the paper)
    "SLfull": [("Mfull", 1.0), ("LPfull", 1.0)],                     # transductive memory x transductive propagation
    "Mfull_lp0": [("Mfull", 1.0), (f"lp0|{REFT}", 1.0)],
    "M0_LPfull": [("M0", 1.0), ("LPfull", 1.0)],
    "SLfull_neg": [("Mfull", 1.0), ("LPfull", 1.0), ("NEGfull", 1.0)],
    "SL_neg": [("M0", 1.0), (f"lp0|{REFT}", 1.0), ("NEGfull", 1.0)],
    "zfull": [("zfull", 1.0)],
    "Mfull_zfull": [("Mfull", 1.0), ("zfull", 1.0)],
    "SL_SLfull": [("M0", 1.0), (f"lp0|{REFT}", 1.0), ("Mfull", 1.0), ("LPfull", 1.0)],
    "Mfull": [("Mfull", 1.0)],
    "LPfull": [("LPfull", 1.0)],
    "NEGfull": [("NEGfull", 1.0)],
    "static": [("static", 1.0)],
}
VIEWSETS = [("L14", "D3L"), ("B14", "D3L"), ("D3L",), ("D3B", "D3L"), ("B14", "L14"), ("B14", "L14", "D3L"), ("L14", "D3B", "D3L"),
            ("B14", "L14", "D3B", "D3L")]
BASES = ["none", "TINS", "TANL"]
REF = "L14+D3L|SL|TINS"
_UP = {}


def measures(x, f):
    if "ok" not in _UP:
        from vins.tins_dev import import_tins
        import_tins()
        _UP["ok"] = True
    from vins.metrics import measures as m_
    m = m_(x[~f], x[f])
    return 100 * m["AUROC"], 100 * m["FPR95"]


def load_stream(part, name):
    z = np.load(P.RESULTS11 / part / f"{name}.npz", allow_pickle=True)
    zv = np.load(P.RESULTS / "vlm" / part / f"{name}.npz", allow_pickle=True)
    assert np.array_equal(z["sample_id"], zv["sample_id"])
    S = {k: z[k].astype(np.float64) for k in z.files if k.startswith("s::")}
    S["is_ood"] = z["is_ood"].astype(bool)
    S["base::none"] = None
    S["base::TINS"] = z["logS"].astype(np.float64)
    S["base::TANL"] = np.log(np.maximum(zv["TANL"].astype(np.float64), 1e-300))
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
    from scipy import stats
    d = np.asarray(d, np.float64)
    n = len(d)
    m = d.mean()
    if n < 2:
        return {"mean": float(m), "lo": float("nan"), "hi": float("nan"), "n": n}
    h = stats.t.ppf(0.975, n - 1) * d.std(ddof=1) / np.sqrt(n)
    return {"mean": float(m), "lo": float(m - h), "hi": float(m + h), "n": n}


def main():
    rows = []
    for part in ("openood", "fourood"):
        for f in sorted(glob.glob(str(P.RESULTS11 / part / "*.npz"))):
            name = Path(f).stem
            S = load_stream(part, name)
            flag = S["is_ood"]
            for vs in VIEWSETS:
                for ro in READOUTS:
                    for b in BASES:
                        au, fp = measures(score(S, vs, ro, b), flag)
                        rows.append({"part": part, "stream": name, "views": "+".join(vs), "readout": ro, "base": b, "AUROC": au, "FPR95": fp})
            print(json.dumps({"part": part, "stream": name, "rows": len(rows)}), flush=True)
    df = pd.DataFrame(rows)
    df["ds"] = df.stream.str.replace(r"_seed\d+$", "", regex=True)
    df["seed"] = df.stream.str.extract(r"seed(\d+)$", expand=False).astype(int)
    df["cfg"] = df.views + "|" + df.readout + "|" + df.base
    df.to_csv(P.RESULTS11 / "p11_rows.csv", index=False)
    out = {}
    lines = []
    for part, d in df.groupby("part"):
        groups = {"near": P.OO_NEAR, "far": P.OO_FAR} if part == "openood" else {"fourood": P.FOUR}
        per_ds = d.groupby(["cfg", "ds"])[["AUROC", "FPR95"]].mean()
        res = {"per_dataset": {f"{c}|{ds}": {k: float(v) for k, v in r.items()} for (c, ds), r in per_ds.iterrows()}}
        for g, sets in groups.items():
            dd = d[d.ds.isin(sets)]
            per_seed = dd.groupby(["cfg", "seed", "ds"])[["AUROC", "FPR95"]].mean().groupby(["cfg", "seed"]).mean()
            ref = per_seed.loc[REF]
            gres = {}
            for c in per_seed.index.get_level_values(0).unique():
                v = per_seed.loc[c]
                e = {"AUROC": float(v.AUROC.mean()), "FPR95": float(v.FPR95.mean()), "n_orders": int(len(v))}
                e["dAUROC_vs_ref"] = tci((v.AUROC - ref.AUROC).values)
                e["dFPR95_vs_ref"] = tci((v.FPR95 - ref.FPR95).values)
                gres[c] = e
            res[g] = gres
            lines.append(f"== {part} {g}: top 30 by FPR95 ==")
            for c, e in sorted(gres.items(), key=lambda kv: kv[1]["FPR95"])[:30]:
                dd_ = e["dFPR95_vs_ref"]
                lines.append(f"{c:42s} {e['AUROC']:6.2f} / {e['FPR95']:6.2f}   dFPR {dd_['mean']:+.2f} [{dd_['lo']:+.2f},{dd_['hi']:+.2f}]")
            lines.append(f"-- reference {REF}: {gres[REF]['AUROC']:.2f} / {gres[REF]['FPR95']:.2f}")
            lines.append("")
        out[part] = res
    out["_meta"] = {"utc": P.utc(), "readouts": {k: [list(c) for c in v] for k, v in READOUTS.items()}, "reference": REF}
    (P.RESULTS11 / "p11_summary.json").write_text(json.dumps(out, indent=1) + "\n")
    t = "\n".join(lines)
    (P.RESULTS11 / "p11_table.txt").write_text(t + "\n")
    print(t)
    print("AN11_DONE", flush=True)


if __name__ == "__main__":
    main()
