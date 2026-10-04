"""Phase 3 summaries: OpenOOD / Four-OOD tables (mean over orders; near = SSB-hard & NINCO, far = iNaturalist,
Textures, OpenImage-O), paired order-level differences, E2-E4 test streams, CUB (F2) and LoCoOp.
Output: <R5>/phase3/summary.json and summary.md
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from vins import r5  # noqa: E402

P3 = r5.R5 / "phase3"
OO = {"near": ["ssb_hard", "ninco"], "far": ["inaturalist", "textures", "openimageo"]}
FOUR = ["inat", "sun", "places", "dtd"]
COMPARE = [("v5", "v5_lp"), ("v5", "v4"), ("v5", "tins"), ("v4", "v4_lp"), ("v5", "reprise_clip"), ("v5", "adaneg"),
           ("v5", "oodd"), ("v5", "maha16_2view"), ("v5", "knn16_2view"), ("v5", "mahaF_l14"), ("v5", "knnF_l14"),
           ("v5", "protoF_l14"), ("v5_m2b", "v5"), ("v5_dino1", "v5"), ("v5_noK", "v5"), ("v5_mem_pure", "v5"),
           ("v5_mem_allood", "v5"), ("v5_lp_keep_id", "v5"), ("v5_lp_keep_ood", "v5")]


def load(part, streams, seeds):
    runs = {}
    for s in streams:
        for sd in seeds:
            f = P3 / part / f"{s}_seed{sd}.json"
            if f.exists():
                runs[(s, sd)] = json.loads(f.read_text())
    return runs


def ci(v):
    return r5.t_interval(v) if len(v) > 1 else {"mean": float(np.mean(v)), "lo": float("nan"), "hi": float("nan"),
                                                "n": len(v), "n_positive": int(np.sum(np.array(v) > 0)),
                                                "n_negative": int(np.sum(np.array(v) < 0)), "sd": float("nan")}


def fmt(x):
    return f"{x['mean']:+.2f} [{x['lo']:+.2f}, {x['hi']:+.2f}]"


def main():
    res, L = {}, ["# Phase 3 results (test; run once)", ""]
    # OpenOOD
    seeds = [123, 124, 125, 126, 127]
    runs = load("openood", OO["near"] + OO["far"], seeds)
    if runs:
        keys = sorted(next(iter(runs.values()))["metrics"])
        tab = {}
        for k in keys:
            row = {}
            for g, dss in OO.items():
                for m in ("AUROC", "FPR95"):
                    per_seed = [np.mean([runs[(d, sd)]["metrics"][k][m] for d in dss if (d, sd) in runs and k in runs[(d, sd)]["metrics"]])
                                for sd in seeds]
                    row[f"{g}_{m}"] = float(np.mean(per_seed))
                    row[f"{g}_{m}_sd"] = float(np.std(per_seed, ddof=1))
            for d in OO["near"] + OO["far"]:
                for m in ("AUROC", "FPR95"):
                    vals = [runs[(d, sd)]["metrics"][k][m] for sd in seeds if (d, sd) in runs and k in runs[(d, sd)]["metrics"]]
                    if vals:
                        row[f"{d}_{m}"] = float(np.mean(vals))
            tab[k] = row
        res["openood"] = {"table": tab}
        L += ["## OpenOOD v1.5 ImageNet-1K (5 orders)", "",
              "| method | near AUROC | near FPR95 | far AUROC | far FPR95 | SSB-hard | NINCO | iNat | Textures | OpenImage-O |",
              "|---|---|---|---|---|---|---|---|---|---|"]
        for k in keys:
            r = tab[k]
            L.append(f"| {k} | {r['near_AUROC']:.2f} | {r['near_FPR95']:.2f} | {r['far_AUROC']:.2f} | {r['far_FPR95']:.2f} | "
                     + " | ".join(f"{r.get(d + '_AUROC', float('nan')):.2f}/{r.get(d + '_FPR95', float('nan')):.2f}"
                                  for d in OO["near"] + OO["far"]) + " |")
        comp = {}
        L += ["", "| comparison (order-level paired) | near FPR95 | near AUROC | far FPR95 | far AUROC |", "|---|---|---|---|---|"]
        for a, b in COMPARE:
            if a not in keys or b not in keys:
                continue
            c = {}
            for g, dss in OO.items():
                for m in ("FPR95", "AUROC"):
                    c[f"{g}_{m}"] = ci([np.mean([runs[(d, sd)]["metrics"][a][m] - runs[(d, sd)]["metrics"][b][m] for d in dss])
                                        for sd in seeds])
            comp[f"{a} - {b}"] = c
            L.append(f"| {a} - {b} | {fmt(c['near_FPR95'])} | {fmt(c['near_AUROC'])} | {fmt(c['far_FPR95'])} | {fmt(c['far_AUROC'])} |")
        res["openood"]["compare"] = comp
        ent = {}
        for tag in ("v4", "v5", "v5_m2b"):
            vals = [r["entrance"][tag] for r in runs.values() if tag in r["entrance"]]
            if vals:
                ent[tag] = {k: float(np.mean([v[k] for v in vals])) for k in vals[0]}
        res["openood"]["entrance"] = ent
        L += ["", "entrance / validity (mean over OpenOOD streams): " + json.dumps(ent), ""]
    # Four-OOD
    seeds4 = [123, 124, 125]
    runs4 = load("fourood", FOUR, seeds4)
    if runs4:
        keys = sorted(next(iter(runs4.values()))["metrics"])
        tab = {}
        for k in keys:
            row = {}
            for m in ("AUROC", "FPR95"):
                per_seed = [np.mean([runs4[(d, sd)]["metrics"][k][m] for d in FOUR]) for sd in seeds4]
                row[f"avg_{m}"], row[f"avg_{m}_sd"] = float(np.mean(per_seed)), float(np.std(per_seed, ddof=1))
                for d in FOUR:
                    row[f"{d}_{m}"] = float(np.mean([runs4[(d, sd)]["metrics"][k][m] for sd in seeds4]))
            tab[k] = row
        res["fourood"] = {"table": tab}
        L += ["## Four-OOD (3 orders)", "", "| method | avg AUROC | avg FPR95 | iNat | SUN | Places | Textures |", "|---|---|---|---|---|---|---|"]
        for k in keys:
            r = tab[k]
            L.append(f"| {k} | {r['avg_AUROC']:.2f} | {r['avg_FPR95']:.2f} | " +
                     " | ".join(f"{r[d + '_AUROC']:.2f}/{r[d + '_FPR95']:.2f}" for d in FOUR) + " |")
        L.append("")
    # E2-E4 on test
    sdir = P3 / "streams"
    if sdir.exists():
        L += ["## E2-E4 on test (3 orders each): v5 - v5_lp (entrance + memory over LP)", "",
              "| stream | OOD/total | TINS FPR95 | v5_lp FPR95 | v5 FPR95 | v5 - v5_lp FPR95 | v5 - v5_lp AUROC | ID adm. % |",
              "|---|---|---|---|---|---|---|---|"]
        res["streams"] = {}
        for d in sorted(p for p in sdir.iterdir() if p.is_dir()):
            ev = [json.loads(f.read_text()) for f in sorted(d.glob("eval_seed*.json"))]
            if not ev:
                continue
            dm = lambda k, m: [e["metrics"][k][m] for e in ev]  # noqa: E731
            diff = {m: ci([a - b for a, b in zip(dm("v5", m), dm("v5_lp", m))]) for m in ("FPR95", "AUROC")}
            adm = 100 * float(np.mean([e["logs"]["v5"]["id_admission"] for e in ev]))
            res["streams"][d.name] = {"diff": diff, "tins_FPR95": float(np.mean(dm("tins", "FPR95"))),
                                      "v5_FPR95": float(np.mean(dm("v5", "FPR95"))), "v5_lp_FPR95": float(np.mean(dm("v5_lp", "FPR95"))),
                                      "id_admission": adm}
            L.append(f"| {d.name} | {ev[0]['n_ood']}/{ev[0]['n']} | {np.mean(dm('tins', 'FPR95')):.2f} | {np.mean(dm('v5_lp', 'FPR95')):.2f} | "
                     f"{np.mean(dm('v5', 'FPR95')):.2f} | {fmt(diff['FPR95'])} | {fmt(diff['AUROC'])} | {adm:.1f} |")
        L.append("")
    # CUB
    cdir = P3 / "cub"
    ce = {(lv, sd): json.loads((cdir / f"eval_{lv}_seed{sd}.json").read_text()) for lv in ("Easy", "Medium", "Hard")
          for sd in (123, 124, 125) if (cdir / f"eval_{lv}_seed{sd}.json").exists()}
    if ce:
        keys = sorted(next(iter(ce.values()))["metrics"])
        L += ["## F2: SSB CUB (3 orders)", "", "| method | Easy AUROC/FPR95 | Medium | Hard |", "|---|---|---|---|"]
        res["cub"] = {}
        for k in keys:
            cells = []
            for lv in ("Easy", "Medium", "Hard"):
                a = np.mean([ce[(lv, sd)]["metrics"][k]["AUROC"] for sd in (123, 124, 125) if (lv, sd) in ce])
                f = np.mean([ce[(lv, sd)]["metrics"][k]["FPR95"] for sd in (123, 124, 125) if (lv, sd) in ce])
                res["cub"].setdefault(k, {})[lv] = {"AUROC": float(a), "FPR95": float(f)}
                cells.append(f"{a:.2f}/{f:.2f}")
            L.append(f"| {k} | " + " | ".join(cells) + " |")
        L.append("")
    # LoCoOp
    lf = P3 / "locoop_metrics.json"
    if lf.exists():
        res["locoop"] = json.loads(lf.read_text())
        L += ["## LoCoOp (official 16-shot checkpoints, seeds 1-3; static scores)", "", "```", json.dumps(res["locoop"], indent=1), "```"]
    (P3 / "summary.json").write_text(json.dumps(res, indent=1) + "\n")
    (P3 / "summary.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
