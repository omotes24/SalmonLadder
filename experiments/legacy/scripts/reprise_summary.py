"""Summaries of the post-hoc REPRISE analyses (analysis/reprise/<part>/*.npz). Output: analysis/reprise/summary.json

Score variants (per stream and order):
  tins              S_final
  reprise           S * pt3_B * pt3_L * plp_B * plp_L                (the frozen method)
  visual_only       pt3_B * pt3_L * plp_B * plp_L                    (no text score)
  static            S * p_B * p_L                                    (no memory, no propagation)
  memory_only       S * pt3_B * pt3_L                                (no propagation)
  lp_only           S * plp_B * plp_L                                (no memory)
  single_entrance   S * pt1_B * pt1_L * plp_B * plp_L                (single-stage entrance p_all <= 0.10)
  B14_only / L14_only   S * pt3 * plp of one view
  mcm, neglabel and <base>_reprise = base * visual part
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from vins import config as C  # noqa: E402
from vins.tins_dev import import_tins  # noqa: E402

A = C.WORK / "analysis" / "reprise"
GROUPS = {"openood": {"near": ["ssb_hard", "ninco"], "far": ["inaturalist", "textures", "openimageo"]},
          "fourood": {"four": ["inat", "sun", "places", "dtd"]}}


def variants(z, base):
    S = z["S"].astype(np.float64)
    g = lambda k: z[k].astype(np.float64)  # noqa: E731
    vis = g("pt3_B14") * g("pt3_L14") * g("plp_B14") * g("plp_L14")
    out = {"tins": S, "reprise": S * vis, "visual_only": vis, "static": S * g("p_B14") * g("p_L14"),
           "memory_only": S * g("pt3_B14") * g("pt3_L14"), "lp_only": S * g("plp_B14") * g("plp_L14"),
           "single_entrance": S * g("pt1_B14") * g("pt1_L14") * g("plp_B14") * g("plp_L14"),
           "B14_only": S * g("pt3_B14") * g("plp_B14"), "L14_only": S * g("pt3_L14") * g("plp_L14")}
    for name, b in base.items():
        out[name] = b
        out[f"{name}_reprise"] = b * vis
    return out


def main():
    t = import_tins()

    def met(s, is_id):
        a, _, f = t.get_measures(s[is_id], s[~is_id])
        return {"AUROC": 100 * float(a), "FPR95": 100 * float(f)}

    summary = {}
    for part, groups in GROUPS.items():
        files = sorted((A / part).glob("*_seed*.npz"))
        if not files:
            continue
        bl = np.load(C.WORK / "analysis" / "baselines" / ("openood.npz" if part == "openood" else "fourood.npz"))
        bpos = {s: i for i, s in enumerate(bl["sample_id"].tolist())}
        per, calib, admit, timing = {}, [], [], []
        for f in files:
            z = np.load(f, allow_pickle=True)
            stream, seed = f.stem.rsplit("_seed", 1)
            is_id = ~z["is_ood"].astype(bool)
            k = np.array([bpos[x] for x in z["sample_id"]])
            base = {"mcm": bl["mcm"][k].astype(np.float64), "neglabel": bl["neglabel"][k].astype(np.float64)}
            per[(stream, int(seed))] = {name: met(s, is_id) for name, s in variants(z, base).items()}
            calib.append({f"{p}_{v}@{a}": float((z[f"{p}_{v}"][is_id] <= a).mean())
                          for p in ("p", "pt3", "plp") for v in ("B14", "L14") for a in (0.01, 0.05, 0.1)})
            admit.append({f"{m}_{v}_{w}": float(z[f"{m}_{v}"][sel].mean()) for m in ("adm1", "adm3") for v in ("B14", "L14")
                          for w, sel in (("ID", is_id), ("OOD", ~is_id))})
            tm = json.loads(str(z["times"]))
            timing.append({"n": int(len(is_id)), "stream": stream,
                           **{f"{kk}_{v}": tm[v][kk] for v in tm for kk in tm[v]}})
        seeds = sorted({s for _, s in per})
        methods = list(next(iter(per.values())).keys())
        table = {}
        streams = sorted({s for s, _ in per})
        for key in streams:
            table[key] = {m: {mm: {"mean": float(np.mean([per[(key, s)][m][mm] for s in seeds])),
                                   "sd": float(np.std([per[(key, s)][m][mm] for s in seeds], ddof=1))}
                              for mm in ("AUROC", "FPR95")} for m in methods}
        for grp, names in groups.items():
            table[grp] = {m: {mm: {"mean": float(np.mean([np.mean([per[(n, s)][m][mm] for n in names]) for s in seeds])),
                                   "sd": float(np.std([np.mean([per[(n, s)][m][mm] for n in names]) for s in seeds], ddof=1))}
                              for mm in ("AUROC", "FPR95")} for m in methods}
        tsum = {}
        for v in ("B14", "L14"):
            for kk in ("memory_single_s", "memory_reprise_s", "lp_s"):
                tot_s = sum(r[f"{kk}_{v}"] for r in timing)
                tot_n = sum(r["n"] for r in timing)
                tsum[f"{kk}_{v}_ms_per_image"] = 1000 * tot_s / tot_n
        summary[part] = {"seeds": seeds, "table": table,
                         "calibration_ID": {k: float(np.mean([c[k] for c in calib])) for k in calib[0]},
                         "admission": {k: float(np.mean([a[k] for a in admit])) for k in admit[0]},
                         "timing": tsum, "per_run": {f"{s}|{sd}": v for (s, sd), v in per.items()}}
        print(f"== {part} (seeds {seeds})")
        for grp in groups:
            for m in methods:
                r = table[grp][m]
                print(f"  {grp:5s} {m:18s} {r['AUROC']['mean']:6.2f} ± {r['AUROC']['sd']:.2f} / {r['FPR95']['mean']:6.2f}")
        print("  calibration:", {k: round(100 * v, 2) for k, v in summary[part]["calibration_ID"].items() if k.endswith("@0.1")})
        print("  admission:", {k: round(100 * v, 2) for k, v in summary[part]["admission"].items()})
        print("  timing ms/image:", {k: round(v, 2) for k, v in tsum.items()})
    (A / "summary.json").write_text(json.dumps(summary, indent=1) + "\n")


if __name__ == "__main__":
    main()
