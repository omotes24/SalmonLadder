"""Apply the registered selection rule to the dev1 tuning results and lock the selections (before any U score)."""
import json

import numpy as np
import pandas as pd

from common import RESULTS, ROOT, dump, sha_file, utc

REF = "k10g1l0.9"
MINIMAL = ["raw", "cdf", "cdf_L0", "z", "mass", "rw", "sprop", "stat_cdf", "stat_cdf_L0"]
TUNED = MINIMAL + ["rep_L0", "rep_L1"]


def load():
    fs = sorted((RESULTS / "dev1_tune").glob("draw*_seed*.parquet"))
    assert len(fs) == 30, len(fs)
    return pd.concat([pd.read_parquet(f) for f in fs], ignore_index=True)


def table(df, base, a):
    d = df[(df.base == base) & (df.a == a)]
    m = d.groupby(["family", "config", "stream"])[["AUROC", "FPR95"]].mean().unstack("stream")
    m.columns = [f"{s}_{k}" for k, s in m.columns]
    return m.reset_index()


def pick(m, fam, ref=REF, key="near_FPR95"):
    s = m[m.family == fam].copy()
    far_ref = float(s[s.config == ref].far_FPR95.iloc[0])
    feas = s[s.far_FPR95 <= far_ref + 0.5 + 1e-12].copy()
    feas["is_ref"] = feas.config == ref
    feas = feas.sort_values([key, "near_AUROC", "is_ref"], ascending=[True, False, False])
    r = feas.iloc[0]
    return {"config": r.config, "near_FPR95": float(r.near_FPR95), "far_FPR95": float(r.far_FPR95),
            "near_AUROC": float(r.near_AUROC), "far_AUROC": float(r.far_AUROC), "far_ref": far_ref,
            "n_feasible": int(len(feas))}


def main():
    df = load()
    m0 = table(df, "none", 1.0)
    sel = {f: pick(m0, f) for f in TUNED}
    best = min(MINIMAL, key=lambda f: (sel[f]["near_FPR95"], sel[f]["far_FPR95"]))
    # weighted control on dev1 x TINS: a in {0.5, 1, 1.5, 2, 3} at the selected configuration
    wsel = {}
    for f in [best, "rep_L0", "rep_L1"]:
        cfg = sel[f]["config"]
        rows = []
        for a in (0.5, 1.0, 1.5, 2.0, 3.0):
            m = table(df, "TINS", a)
            r = m[(m.family == f) & (m.config == cfg)].iloc[0]
            rows.append({"a": a, "near_FPR95": float(r.near_FPR95), "far_FPR95": float(r.far_FPR95), "near_AUROC": float(r.near_AUROC)})
        far1 = [r for r in rows if r["a"] == 1.0][0]["far_FPR95"]
        feas = [r for r in rows if r["far_FPR95"] <= far1 + 0.5 + 1e-12]
        feas.sort(key=lambda r: (r["near_FPR95"], -r["near_AUROC"], r["a"] != 1.0))
        wsel[f] = {"a": feas[0]["a"], "grid": rows}
    frozen = m0[(m0.family == "rep_L0") & (m0.config == REF)].iloc[0]
    lock = {"utc": utc(), "rule": "min dev1 standalone near FPR95 s.t. far FPR95 <= reference + 0.5 (registered)",
            "prereg_sha256": (ROOT / "prereg_p4.sha256").read_text().strip(),
            "selected": sel, "best_minimal": best, "weighted": wsel,
            "frozen_reprise_dev1": {k: float(frozen[k]) for k in ("near_FPR95", "far_FPR95", "near_AUROC", "far_AUROC")},
            "n_rows": int(len(df))}
    dump(ROOT / "selection_lock.json", lock)
    (ROOT / "selection_lock.sha256").write_text(sha_file(ROOT / "selection_lock.json") + "\n")
    print(json.dumps({"best_minimal": best, "selected": {f: sel[f]["config"] for f in sel},
                      "weighted": {f: wsel[f]["a"] for f in wsel}}, indent=1))


if __name__ == "__main__":
    main()
