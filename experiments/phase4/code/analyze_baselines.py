"""Exp 2 summary: original test-time methods (NegLabel, AdaNeg, AdaNeg-TA, TANL, MCM+OODD) and TINS/MCM on the
confirmatory streams, alone and multiplied by the frozen REPRISE visual part and by the best minimal configuration.
Pairing: identical sample order (asserted). Units as the confirmatory analysis (U1: split means of 9 draw x order
conditions -> t interval over 5 splits; U2/U3: 5 draw values, orders averaged). Descriptive (post-lock, no selection).
Output: results/baselines_summary.json and .md"""
import json

import numpy as np
import pandas as pd
import torch
from scipy import stats

from bank_data import Features, pos_text
from common import RESULTS, ROOT, utc
from metrics_p4 import metrics
from oodd_eval import oodd_scores

REF = "k10g1l0.9"


def lock():
    return json.loads((ROOT / "selection_lock.json").read_text())


def tci(v):
    v = np.asarray(v, float)
    m = v.mean()
    if len(v) < 2:
        return m, np.nan, np.nan
    h = stats.t.ppf(0.975, len(v) - 1) * v.std(ddof=1) / np.sqrt(len(v))
    return m, m - h, m + h


def main():
    L = lock()
    zcfg = L["selected"]["z"]["config"]
    F = Features()
    rows = []
    for f in sorted((RESULTS / "eval").glob("*_scores.npz")):
        bank, s, d, seed = f.name.replace("_scores.npz", "").split("_")
        split, draw, seed = int(s[1:]), int(d[1:]), int(seed[4:])
        z = np.load(f, allow_pickle=True)
        ids, flag = list(z["sample_id"]), z["is_ood"].astype(bool)
        v = np.load(RESULTS / "vlm_tta" / f"{bank}_s{split}_seed{seed}.npz", allow_pickle=True)
        assert list(v["sample_id"]) == ids and np.array_equal(v["is_ood"].astype(bool), flag)
        rep = z[f"rep_L0__{REF}"]
        mini = z[f"z__{zcfg}"]
        clip = F.get("CLIP", ids)
        pt = pos_text(bank, split)
        mcm = torch.softmax(torch.as_tensor(clip @ pt.T, dtype=torch.float64), dim=1).max(1).values.numpy()
        bases = {"TINS": z["logS"], "MCM": z["logM"], "NegLabel": np.log(v["NegLabel"]), "AdaNeg": np.log(v["AdaNeg"]),
                 "AdaNeg_TA": np.log(v["AdaNeg_TA"]), "TANL": np.log(v["TANL"])}
        oodd = {b: oodd_scores(clip, mcm, batch=b) for b in (64, 256)}
        key = dict(bank=bank, split=split, draw=draw, seed=seed)
        for b, x in bases.items():
            rows.append({**key, "base": b, "visual": "none", **metrics(x, flag)})
            rows.append({**key, "base": b, "visual": "REPRISE", **metrics(x + rep, flag)})
            rows.append({**key, "base": b, "visual": "minimal", **metrics(x + mini, flag)})
        for b, x in oodd.items():
            rows.append({**key, "base": f"MCM+OODD_b{b}", "visual": "none", **metrics(x, flag)})
        rows.append({**key, "base": "none", "visual": "REPRISE", **metrics(rep, flag)})
        rows.append({**key, "base": "none", "visual": "minimal", **metrics(mini, flag)})
    df = pd.DataFrame(rows)
    df.to_parquet(RESULTS / "baselines.parquet", index=False)
    out = {"utc": utc(), "rows": []}
    for (bank, base, vis), g in df.groupby(["bank", "base", "visual"]):
        unit = g.groupby("split" if bank == "U1" else "draw")[["AUROC", "FPR95"]].mean()
        a, f = tci(unit.AUROC), tci(unit.FPR95)
        out["rows"].append({"bank": bank, "base": base, "visual": vis, "AUROC": a, "FPR95": f, "n_units": len(unit)})
    # paired: base x REPRISE - base, base x REPRISE - base x minimal
    pair = []
    for (bank, base), g in df[df.base != "none"].groupby(["bank", "base"]):
        if base.startswith("MCM+OODD"):
            continue
        piv = g.pivot_table(index=["split", "draw", "seed"], columns="visual", values="FPR95")
        unitcol = "split" if bank == "U1" else "draw"
        u = piv.groupby(level=unitcol).mean()
        pair.append({"bank": bank, "base": base, "REPRISE_minus_none": tci(u["REPRISE"] - u["none"]),
                     "REPRISE_minus_minimal": tci(u["REPRISE"] - u["minimal"])})
    out["paired_FPR95"] = pair
    (RESULTS / "baselines_summary.json").write_text(json.dumps(out, indent=1, default=float))
    lines = ["| bank | base | visual | AUROC | FPR95 |", "|---|---|---|---|---|"]
    for r in out["rows"]:
        lines.append(f"| {r['bank']} | {r['base']} | {r['visual']} | {r['AUROC'][0]:.2f} [{r['AUROC'][1]:.2f}, {r['AUROC'][2]:.2f}] | "
                     f"{r['FPR95'][0]:.2f} [{r['FPR95'][1]:.2f}, {r['FPR95'][2]:.2f}] |")
    lines += ["", "| bank | base | xREPRISE - base (FPR95) | xREPRISE - xminimal (FPR95) |", "|---|---|---|---|"]
    for p in pair:
        a, b = p["REPRISE_minus_none"], p["REPRISE_minus_minimal"]
        lines.append(f"| {p['bank']} | {p['base']} | {a[0]:.2f} [{a[1]:.2f}, {a[2]:.2f}] | {b[0]:.2f} [{b[1]:.2f}, {b[2]:.2f}] |")
    (RESULTS / "baselines_summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
