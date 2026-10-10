"""Phase 11b analysis of the transductive sweep (sweep11.py). Output: P11/results/sweep/p11b_summary.json, p11b_table.txt."""
import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd

import p11common as P
from an11 import measures, tci

TAGS = [f"k{k}b{b:g}l{l:g}" for k in (10, 20) for b in (1.0, 0.0) for l in (0.9, 0.95, 0.99)]
READOUTS = {}
for t in TAGS:
    READOUTS[f"LP|{t}"] = [(f"LP|{t}", 1.0)]
    READOUTS[f"z|{t}"] = [(f"z|{t}", 1.0)]
for q in (0.1, 0.2):
    READOUTS[f"LPxNEG|q{q:g}"] = [("LP|k10b1l0.9", 1.0), (f"NEG|q{q:g}", 1.0)]
    READOUTS[f"RAT|q{q:g}"] = [(f"RAT|q{q:g}", 1.0)]
    READOUTS[f"LPxRAT|q{q:g}"] = [("LP|k10b1l0.9", 1.0), (f"RAT|q{q:g}", 1.0)]
    READOUTS[f"zxNEG|q{q:g}"] = [("z|k10b1l0.9", 1.0), (f"NEG|q{q:g}", 1.0)]
VIEWSETS = [("L14", "D3L"), ("B14", "L14", "D3L"), ("L14", "D3B", "D3L"), ("D3L",), ("B14", "L14", "D3B", "D3L"),
            ("L14xD3L",), ("B14xL14xD3L",), ("L14xD3L", "D3L"), ("L14xD3L", "L14", "D3L")]
BASES = ["none", "TINS"]
REF = "L14+D3L|LP|k10b1l0.9|TINS"


def main():
    rows = []
    for part in ("openood", "fourood"):
        for f in sorted(glob.glob(str(P.RESULTS11 / "sweep" / part / "*.npz"))):
            name = Path(f).stem
            z = np.load(f, allow_pickle=True)
            flag = z["is_ood"].astype(bool)
            logS = z["logS"].astype(np.float64)
            cache = {}
            for vs in VIEWSETS:
                for ro, combo in READOUTS.items():
                    x = np.zeros(len(flag))
                    for v in vs:
                        for key, w in combo:
                            x = x + w * z[f"s::{v}::{key}"].astype(np.float64)
                    for b in BASES:
                        y = x + logS if b == "TINS" else x
                        au, fp = measures(y, flag)
                        rows.append({"part": part, "stream": name, "views": "+".join(vs), "readout": ro, "base": b, "AUROC": au, "FPR95": fp})
            print(json.dumps({"part": part, "stream": name, "rows": len(rows)}), flush=True)
    df = pd.DataFrame(rows)
    df["ds"] = df.stream.str.replace(r"_seed\d+$", "", regex=True)
    df["seed"] = df.stream.str.extract(r"seed(\d+)$", expand=False).astype(int)
    df["cfg"] = df.views + "|" + df.readout + "|" + df.base
    df.to_csv(P.RESULTS11 / "sweep" / "p11b_rows.csv", index=False)
    out, lines = {}, []
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
                gres[c] = {"AUROC": float(v.AUROC.mean()), "FPR95": float(v.FPR95.mean()), "n_orders": int(len(v)),
                           "dFPR95_vs_ref": tci((v.FPR95 - ref.FPR95).values), "dAUROC_vs_ref": tci((v.AUROC - ref.AUROC).values)}
            res[g] = gres
            lines.append(f"== {part} {g}: top 40 by FPR95 ==")
            for c, e in sorted(gres.items(), key=lambda kv: kv[1]["FPR95"])[:40]:
                dd_ = e["dFPR95_vs_ref"]
                lines.append(f"{c:50s} {e['AUROC']:6.2f} / {e['FPR95']:6.2f}   dFPR {dd_['mean']:+.2f} [{dd_['lo']:+.2f},{dd_['hi']:+.2f}]")
            lines.append(f"-- reference {REF}: {gres[REF]['AUROC']:.2f} / {gres[REF]['FPR95']:.2f}")
            lines.append("")
        out[part] = res
    out["_meta"] = {"utc": P.utc(), "reference": REF}
    (P.RESULTS11 / "sweep" / "p11b_summary.json").write_text(json.dumps(out, indent=1) + "\n")
    t = "\n".join(lines)
    (P.RESULTS11 / "sweep" / "p11b_table.txt").write_text(t + "\n")
    print(t)
    print("AN11B_DONE", flush=True)


if __name__ == "__main__":
    main()
