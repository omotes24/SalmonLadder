"""Intervention (registered Exp 4 design, same-class content) re-run with every engine7 read-out: AUROC per
(split, class) pair (8 class queries vs 64 common ID queries) for r in {0, 1, 2, 5, 10, 20}. Descriptive.
The hinge read-outs were NOT confirmed on dev2 (amendment 01); they are shown for information only."""
import json

import numpy as np
import pandas as pd

import an7 as A
from p7common import P4, SEED

SRC = A.RESULTS / "x4"
NB = 10000
FAMS = {"static": ["static"], "mem": ["M"], "rho_only": ["Mrho"], "plp": ["lp0"], "lp_zero": ["lp"], "zeta": ["z"],
        "static_plp": ["static", "lp0"], "reprise": ["M", "lp0"], "mem_lpzero": ["M", "lp"], "static_rho_plp": ["static", "Mrho", "lp0"],
        "hinge0_plp": ["Mh0", "lp0"], "hinge1_plp": ["Mh1", "lp0"], "hinge2_plp": ["Mh2", "lp0"], "hinge3_plp": ["Mh3", "lp0"], "hinge4_plp": ["Mh4", "lp0"],
        "hinge2_lpzero": ["Mh2", "lp"], "static_reprise": ["static", "M", "lp0"]}


def auroc(id_s, ood_s):
    a = np.asarray(id_s)[:, None]
    b = np.asarray(ood_s)[None, :]
    return 100 * (np.mean(a > b) + 0.5 * np.mean(a == b))


def boot(x, rng):
    x = np.asarray(x, float)
    idx = rng.integers(0, len(x), size=(NB, len(x)))
    v = x[idx].mean(1)
    return {"mean": float(x.mean()), "lo": float(np.quantile(v, 0.025)), "hi": float(np.quantile(v, 0.975)), "n": int(len(x))}


def main():
    rows = []
    for k in range(1, 6):
        dz = json.loads((P4 / "banks" / "exp4" / f"split{k}.json").read_text())
        idq = list(dz["id_queries"])
        b = pd.read_parquet(SRC / f"s{k}_r0.parquet").pivot(index="query", columns="key", values="score")
        for w in dz["probe"]:
            f = SRC / f"s{k}_{w}.parquet"
            if not f.exists():
                continue
            qs = list(dz["probes"][w]["queries"])
            p = pd.read_parquet(f)
            tabs = {0: b}
            for r, g in p.groupby("r"):
                tabs[int(r)] = g.pivot(index="query", columns="key", values="score")
            for r, t in tabs.items():
                row = {"split": k, "wnid": w, "r": r}
                for fam, keys in FAMS.items():
                    xi = sum(t.loc[idq, key].values for key in keys)
                    xo = sum(t.loc[qs, key].values for key in keys)
                    row[fam] = auroc(xi, xo)
                rows.append(row)
    T = pd.DataFrame(rows)
    T.to_parquet(A.RESULTS / "x4_auroc.parquet", index=False)
    rng = np.random.default_rng([SEED, 41])
    n_pairs = int(T[T.r == 0].shape[0])
    out = {"n_pairs": n_pairs, "mean": {}, "vs_static_r0": {}, "vs_reprise": {}, "gain_r20": {}}
    print(f"pairs: {n_pairs}; mean AUROC by r = 0, 1, 2, 5, 10, 20")
    P = {r: T[T.r == r].sort_values(["split", "wnid"]).reset_index(drop=True) for r in (0, 1, 2, 5, 10, 20)}
    for fam in FAMS:
        out["mean"][fam] = {r: float(P[r][fam].mean()) for r in P}
        out["vs_static_r0"][fam] = boot(P[0][fam] - P[0]["static"], rng)
        out["vs_reprise"][fam] = {r: boot(P[r][fam] - P[r]["reprise"], rng) for r in P}
        out["gain_r20"][fam] = boot(P[20][fam] - P[0][fam], rng)
        d0 = out["vs_static_r0"][fam]
        print(f"  {fam:15s} " + " ".join(f"{out['mean'][fam][r]:6.2f}" for r in P) + f" | r=0 minus static p {d0['mean']:+.2f} [{d0['lo']:+.2f},{d0['hi']:+.2f}]"
              f" | r=20 minus REPRISE {out['vs_reprise'][fam][20]['mean']:+.2f} [{out['vs_reprise'][fam][20]['lo']:+.2f},{out['vs_reprise'][fam][20]['hi']:+.2f}]")
    A.dumpj(A.RESULTS / "an_x4.json", out)


if __name__ == "__main__":
    main()
