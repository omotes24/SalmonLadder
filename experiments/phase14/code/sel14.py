"""Phase 14 (a): selection of the seed rule (Storey vs plain BH), the level q and the combination with the static rank for the
cross-fitted ladder on the development splits (dev1: selection; dev2: confirmation). Views fixed to DINOv2 L/14 + DINOv3 L/16
(selected in Phase 12), k_g, lambda fixed. Candidates: rule in {storey, bh} x q in {0.05, 0.1, 0.2} x mode in {pp, pps, min}
(18); references: p+ only, static rank, kNN rank, LGC-max, and the Phase 12 read-out (asymmetric seeds, Storey, per q).
Output: P14/results/sel/<dev>/<dev>_draw<d>_<stream>_seed<s>.csv ; --aggregate -> sel14_summary.json, sel14_table.txt."""
import argparse
import glob
import json
import sys
import time

import numpy as np
import pandas as pd
import torch

import p14common as P
import core14 as K

sys.path.insert(0, "/home/omote/reprise_p5_20261002/code")
import dev5  # noqa: E402

VIEWS = ["L14", "D3L"]


def feats_of(D, draw, name, rows):
    sup, cal, sf = dev5.raw_view(D, draw, name, rows)
    return P.l2n(sup.reshape(-1, sup.shape[-1])), P.l2n(cal), P.l2n(sf)


def run_dev(dev, worker, nworkers, device="cuda"):
    out_dir = P.RESULTS14 / "sel" / dev
    out_dir.mkdir(parents=True, exist_ok=True)
    D = dev5.load(dev)
    keys = sorted(D["tins"].keys())[worker::nworkers]
    todo = [k for k in keys if not (out_dir / f"{dev}_draw{k[0]}_{k[1]}_seed{k[2]}.csv").exists()]
    print(json.dumps({"dev": dev, "feats": sorted(D["feats"].keys()), "tasks": len(todo), "utc": P.utc()}), flush=True)
    for (draw, stream, seed) in todo:
        name = f"{dev}_draw{draw}_{stream}_seed{seed}"
        t0 = time.time()
        sid, flag, S, bidx = D["tins"][(draw, stream, seed)]
        rows = np.array([D["stream_row"][s] for s in sid])
        logS = np.log(np.maximum(S.astype(np.float64), 1e-300))
        graphs, terms, feats = {}, {}, {}
        n_classes = None
        for v in VIEWS:
            sup, cal, sf = feats_of(D, draw, v, rows)
            n_classes = len(sup) // 12
            g = K.Graph(sup, cal, sf, device)
            sc, st = K.proto_scores(sup, cal, sf, device)
            graphs[v], terms[v], feats[v] = g, K.view_terms(g, sc, st), (sup, cal, sf)
        labels_sup = np.repeat(np.arange(n_classes), 12)
        tag = {"dev": dev, "draw": draw, "stream": stream, "seed": seed, "n": int(len(sid)), "n_ood": int(flag.sum())}
        rows_out, meta, _ = K.score_all(graphs, terms, feats, flag, {("TINS", seed): logS}, labels_sup, n_classes, tag, device)
        for g in graphs.values():
            g.free()
        pd.DataFrame(rows_out).to_csv(out_dir / f"{name}.csv", index=False)
        (out_dir / f"{name}.meta.json").write_text(json.dumps(meta, indent=1, default=str) + "\n")
        pick = {r["cfg"]: round(r["FPR95"], 2) for r in rows_out if r["base"] == "none" and r["cfg"] in ("xfit|storey|q0.2|pp", "xfit|bh|q0.2|pp", "xfit|bh|q0.2|min", "v1|storey|q0.2|pp", "xfit|lp", "xfit|static", "lgcmax")}
        print(json.dumps({"task": name, "FPR95": pick, "seconds": round(time.time() - t0, 1), "utc": P.utc()}), flush=True)


RULE = ("min dev1 near FPR95 over the 18 cross-fit candidates (rule x q x mode; views L14+D3L; standalone) subject to "
        "dev1 far FPR95 <= that of the Phase 12 read-out (v1|storey|q0.2|pp, standalone) + 0.5; dev2 reported as confirmation only")


def aggregate():
    base = P.RESULTS14 / "sel"
    df = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(str(base / "dev*" / "*.csv")))])
    df["key"] = df.cfg + "|" + df.base
    out, lines = {}, []
    for dev, d in df.groupby("dev"):
        res = {}
        for stream, dd in d.groupby("stream"):
            per_draw = dd.groupby(["key", "draw"])[["AUROC", "FPR95"]].mean()           # unit: draw (seeds averaged first)
            for c in per_draw.index.get_level_values(0).unique():
                v = per_draw.loc[c]
                res.setdefault(c, {})[stream] = {"AUROC": float(v.AUROC.mean()), "FPR95": float(v.FPR95.mean()), "n_draws": int(len(v)),
                                                 "FPR95_per_draw": [float(x) for x in v.FPR95.values]}
        if "fpr@0.05" in d:
            for c, dd in d[d["fpr@0.05"].notna()].groupby("key"):
                res.setdefault(c, {})["validity"] = {k: float(dd[k].mean()) for k in ("fpr@0.01", "fpr@0.05", "fpr@0.1", "fpr@0.2", "tpr@0.05", "bh_fdp", "bh_recall") if k in dd}
        out[dev] = res
    d1 = out["dev1"]
    ref = d1["v1|storey|q0.2|pp|none"]
    # candidates: the 18 (rule, q, mode) read-outs p_final, standalone  (xfit|<rule>|q<q>|<mode>|none; the |pooled diagnostics excluded)
    cands = {c: e for c, e in d1.items() if c.startswith("xfit|") and c.endswith("|none") and "|pooled|" not in c and c.count("|") == 4}
    feasible = {c: e for c, e in cands.items() if e["far"]["FPR95"] <= ref["far"]["FPR95"] + 0.5}

    def order(c):                       # ties (to 0.01 FPR95 points): lower far FPR95, then the simpler read-out (pp < pps < min), Storey before BH, larger q
        _, rule, qk, mode, _ = c.split("|")
        e = feasible[c]
        return (round(e["near"]["FPR95"], 2), round(e["far"]["FPR95"], 2), P.MODES.index(mode), P.RULES.index(rule), -float(qk[1:]))
    chosen = min(feasible, key=order) if feasible else None
    out["_selection"] = {"rule": RULE, "reference": ref, "n_candidates": len(cands), "n_feasible": len(feasible), "chosen": chosen,
                         "chosen_dev1": d1.get(chosen), "chosen_dev2": out.get("dev2", {}).get(chosen),
                         "ranking_dev1_near": sorted(((c, e["near"]["FPR95"], e["far"]["FPR95"]) for c, e in cands.items()), key=lambda t: t[1]),
                         "utc": P.utc()}
    for dev in sorted(k for k in out if not k.startswith("_")):
        for stream in ("near", "far"):
            lines.append(f"== {dev} {stream}: configurations by FPR95 (mean over draws) ==")
            items = sorted(((c, e[stream]) for c, e in out[dev].items() if stream in e), key=lambda ce: ce[1]["FPR95"])
            for c, e in items:
                lines.append(f"{c:40s} {e['AUROC']:6.2f} / {e['FPR95']:6.2f}   (draws {e['n_draws']})")
            lines.append("")
        lines.append(f"== {dev} validity of p_final (mean over tasks) ==")
        for c, e in sorted(out[dev].items()):
            if "validity" in e:
                lines.append(f"{c:40s} " + " ".join(f"{k}={v:.3f}" for k, v in e["validity"].items()))
        lines.append("")
    lines.append(f"SELECTED: {chosen}  (feasible {len(feasible)} of {len(cands)})")
    P.write_json(base / "sel14_summary.json", out)
    (base / "sel14_table.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print("SEL14_DONE", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dev", choices=["dev1", "dev2"])
    ap.add_argument("--worker", type=int, default=0)
    ap.add_argument("--nworkers", type=int, default=1)
    ap.add_argument("--aggregate", action="store_true")
    a = ap.parse_args()
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    if a.aggregate:
        aggregate()
    else:
        run_dev(a.dev, a.worker, a.nworkers)
