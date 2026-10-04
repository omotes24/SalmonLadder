"""Exp 4 analysis: per probe class AUROC (8 class queries vs 64 common ID queries), registered estimand and curves."""
import json

import numpy as np
import pandas as pd

from common import RESULTS, dump, utc

OUT = RESULTS / "exp4"
FAMS = ["REPRISE", "static_pLP", "best", "pLP", "Mpt", "static"]


def auroc(id_s, ood_s):
    a = np.asarray(id_s)[:, None]
    b = np.asarray(ood_s)[None, :]
    return 100 * (np.mean(a > b) + 0.5 * np.mean(a == b))


def main():
    base = pd.concat([pd.read_parquet(f) for f in sorted(OUT.glob("s*_r0.parquet"))], ignore_index=True)
    per = pd.concat([pd.read_parquet(f) for f in sorted(OUT.glob("s*_n*.parquet"))], ignore_index=True)
    designs = {k: json.loads((RESULTS.parent / "banks" / "exp4" / f"split{k}.json").read_text()) for k in range(1, 6)}
    rows = []
    for k, dz in designs.items():
        idq = set(dz["id_queries"])
        b = base[base.split == k]
        for w in dz["probe"]:
            qs = set(dz["probes"][w]["queries"])
            for f in FAMS:
                bf = b[b.family == f]
                au0 = auroc(bf[bf["query"].isin(idq)].score, bf[bf["query"].isin(qs)].score)
                for cond in ("same", "near", "far", "dup"):
                    rows.append({"split": k, "wnid": w, "family": f, "cond": cond, "r": 0, "AUROC": au0})
            p = per[(per.split == k) & (per.wnid == w)]
            for (cond, r, f), g in p.groupby(["cond", "r", "family"]):
                rows.append({"split": k, "wnid": w, "family": f, "cond": cond, "r": int(r),
                             "AUROC": auroc(g[g["query"].isin(idq)].score, g[g["query"].isin(qs)].score)})
    A = pd.DataFrame(rows)
    A.to_parquet(OUT / "auroc_table.parquet", index=False)
    P = A.pivot_table(index=["split", "wnid"], columns=["family", "cond", "r"], values="AUROC")

    def gain(f, cond="same", r=20):
        return P[(f, cond, r)] - P[(f, cond, 0)]

    rng = np.random.default_rng(20260928)
    res = {"utc": utc(), "n_classes": int(len(P))}

    def boot(x, clustered=False):
        x = x.dropna()
        vals = []
        splits = x.index.get_level_values("split").values
        for _ in range(2000):
            if clustered:
                ss = rng.choice(np.unique(splits), size=len(np.unique(splits)), replace=True)
                parts = [x.values[splits == s] for s in ss]
                sample = np.concatenate([rng.choice(pp, size=len(pp), replace=True) for pp in parts])
            else:
                sample = rng.choice(x.values, size=len(x), replace=True)
            vals.append(sample.mean())
        return {"mean": float(x.mean()), "lo": float(np.quantile(vals, 0.025)), "hi": float(np.quantile(vals, 0.975)), "n": int(len(x))}

    G = gain("REPRISE") - gain("static_pLP")
    res["estimand_REPRISE_vs_staticpLP"] = {"class_bootstrap": boot(G), "split_clustered": boot(G, True),
                                            "frac_positive": float((G > 0).mean()), "frac_negative": float((G < 0).mean())}
    for other in ("best", "pLP"):
        Go = gain("REPRISE") - gain(other)
        res[f"REPRISE_vs_{other}"] = {"class_bootstrap": boot(Go), "frac_positive": float((Go > 0).mean())}
    curves = A.groupby(["family", "cond", "r"]).AUROC.mean().reset_index()
    res["curves"] = curves.to_dict(orient="records")
    for f in FAMS:
        res[f"gain20_{f}"] = {c: boot(gain(f, c)) for c in ("same", "near", "far", "dup")}
    res["first_appearance_r0"] = {f: float(P[(f, "same", 0)].mean()) for f in FAMS}
    dump(OUT.parent / "exp4_summary.json", res)
    lines = [f"# Exp 4 ({utc()}), {res['n_classes']} probe classes", "",
             "estimand (REPRISE gain - static x pLP gain, same, r=20 vs 0): "
             f"{res['estimand_REPRISE_vs_staticpLP']['class_bootstrap']}", ""]
    for f in FAMS:
        lines.append(f"- {f}: " + ", ".join(f"{c} {v['mean']:+.2f} [{v['lo']:+.2f},{v['hi']:+.2f}]" for c, v in res[f"gain20_{f}"].items()))
    (OUT.parent / "exp4_summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
