"""Phase 12 (a): selection of the views, q and the base detector on the development splits (dev1: selection; dev2: confirmation),
with the post-stream two-sided read-out. Candidates: view sets x q in {0.05, 0.1, 0.2} x base in {none, TINS}; the same-feature
baselines (static prototype rank, kNN rank, p+ only) are scored alongside. Every (dev, draw, stream, seed) stream of the
R5 development caches is scored after the whole stream has arrived.
Output: P12/results/sel/<dev>/<dev>_draw<d>_<stream>_seed<s>.csv  (one row per configuration)
        P12/results/sel/sel12_summary.json, sel12_table.txt (--aggregate; the selection rule is applied there)."""
import argparse
import glob
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import p12common as P
import core12 as C
import engine5 as E5
from engine import _log
from an11 import measures, tci

sys.path.insert(0, "/home/omote/reprise_p5_20261002/code")
import dev5  # noqa: E402
from static import support_dall  # noqa: E402
from vins import r5  # noqa: E402

VIEWS = ["B14", "L14", "D3B", "D3L"]
VIEWSETS = [("D3L",), ("L14",), ("L14xD3L",), ("L14", "D3L"), ("L14xD3L", "D3L"), ("D3B", "D3L"), ("B14", "L14"), ("L14", "D3B", "D3L")]
BASES = ("none", "TINS")


def l2n(x):
    x = np.asarray(x, np.float32)
    return x / np.linalg.norm(x, axis=1, keepdims=True)


def feats_of(D, draw, name, rows):
    """Support (C x 12 x D), calibration (4C x D) and stream features of one view for one draw; joint view by concatenation."""
    if "x" in name:
        a, b = name.split("x")
        pa, pb = feats_of(D, draw, a, rows), feats_of(D, draw, b, rows)
        return tuple(l2n(np.concatenate([l2n(pa[i].reshape(-1, pa[i].shape[-1])), l2n(pb[i].reshape(-1, pb[i].shape[-1]))], 1))
                     .reshape(pa[i].shape[:-1] + (-1,)) for i in range(3))
    sup, cal, sf = dev5.raw_view(D, draw, name, rows)
    return l2n(sup.reshape(-1, sup.shape[-1])).reshape(sup.shape), l2n(cal), l2n(sf)


def run_dev(dev, worker, nworkers, device="cuda"):
    out_dir = P.RESULTS12 / "sel" / dev
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
        dr = D["draws"][str(draw)]
        cand_cal = np.argsort(-dr["cal_sims"], axis=1)[:, :P.V5["K"]]
        cand_sf = D["cand_stream"][rows]
        terms, meta = {}, {"dev": dev, "draw": draw, "stream": stream, "seed": seed, "n": int(len(sid)), "n_ood": int(flag.sum()), "views": {}}
        for v in VIEWS + ["L14xD3L"]:
            if v not in D["feats"] and "x" not in v:
                continue
            sup3, cal, sf = feats_of(D, draw, v, rows)
            sup = sup3.reshape(-1, sup3.shape[-1])
            g = C.Graph(sup, cal, sf, device)
            out, m = C.read_out(g, flag)
            g.free()
            terms[(v, "lp")] = out["lp"]
            for q in P.QS:
                terms[(v, f"ln|q{q:g}")] = out[f"ln|q{q:g}"]
            terms[(v, "knn")] = C.knn_rank(sup, cal, sf, dev=device)
            if "x" not in v:
                st = E5.static5(r5.proto_view, support_dall, sup3, cal, sf, cand_cal, cand_sf, P.V5["n0"], P.V5["m"])
                terms[(v, "static")] = _log(np.asarray(st["p"])[len(cal):])          # proto_view returns [calibration; stream]
            meta["views"][v] = m
        rows_out = []
        for vs in VIEWSETS:
            if any((v, "lp") not in terms for v in vs):
                continue
            combos = {"T|q0.05": ["lp", "ln|q0.05"], "T|q0.1": ["lp", "ln|q0.1"], "T|q0.2": ["lp", "ln|q0.2"], "Tlp": ["lp"], "knn": ["knn"], "static": ["static"]}
            for ro, ks in combos.items():
                if any(terms.get((v, k)) is None for v in vs for k in ks):
                    continue
                x = sum(terms[(v, k)] for v in vs for k in ks)
                for b in BASES:
                    au, fp = measures(x + (logS if b == "TINS" else 0.0), flag)
                    rows_out.append({**{k: meta[k] for k in ("dev", "draw", "stream", "seed")}, "views": "+".join(vs), "readout": ro, "base": b,
                                     "AUROC": au, "FPR95": fp})
        pd.DataFrame(rows_out).to_csv(out_dir / f"{name}.csv", index=False)
        (out_dir / f"{name}.meta.json").write_text(json.dumps(meta, indent=1) + "\n")
        print(json.dumps({"task": name, "rows": len(rows_out), "seconds": round(time.time() - t0, 1), "utc": P.utc()}), flush=True)


def aggregate():
    base = P.RESULTS12 / "sel"
    df = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(str(base / "dev*" / "*.csv")))])
    df["cfg"] = df.views + "|" + df.readout + "|" + df.base
    out, lines = {}, []
    for dev, d in df.groupby("dev"):
        res = {}
        # unit: draw (streams and seeds of a draw averaged first), as the Phase 4 intervals
        for stream, dd in d.groupby("stream"):
            per_draw = dd.groupby(["cfg", "draw"])[["AUROC", "FPR95"]].mean()
            for c in per_draw.index.get_level_values(0).unique():
                v = per_draw.loc[c]
                res.setdefault(c, {})[stream] = {"AUROC": float(v.AUROC.mean()), "FPR95": float(v.FPR95.mean()), "n_draws": int(len(v)),
                                                 "FPR95_per_draw": [float(x) for x in v.FPR95.values]}
        out[dev] = res
    # selection rule (registered in the paper's text before the test results of these configurations were looked at under this rule):
    # among the view sets x q (read-out T), standalone (base none), minimise the near FPR95 on dev1 subject to far FPR95 on dev1 at most
    # the reference (L14+D3L, q = 0.1) + 0.5; confirm on dev2 (reported, not used for selection).
    d1 = out["dev1"]
    ref = d1["L14+D3L|T|q0.1|none"]
    cands = {c: e for c, e in d1.items() if c.endswith("|none") and "|T|" in c}
    feasible = {c: e for c, e in cands.items() if e["far"]["FPR95"] <= ref["far"]["FPR95"] + 0.5}
    chosen = min(feasible, key=lambda c: feasible[c]["near"]["FPR95"])
    out["_selection"] = {"rule": "min dev1 near FPR95 (standalone, read-out T) s.t. dev1 far FPR95 <= reference (L14+D3L|T|q0.1|none) + 0.5",
                         "reference": ref, "n_candidates": len(cands), "n_feasible": len(feasible), "chosen": chosen,
                         "chosen_dev1": d1[chosen], "chosen_dev2": out.get("dev2", {}).get(chosen),
                         "utc": P.utc()}
    for dev in sorted(k for k in out if not k.startswith("_")):
        for stream in ("near", "far"):
            lines.append(f"== {dev} {stream}: configurations by FPR95 (mean over draws) ==")
            items = sorted(((c, e[stream]) for c, e in out[dev].items() if stream in e), key=lambda ce: ce[1]["FPR95"])
            for c, e in items:
                lines.append(f"{c:40s} {e['AUROC']:6.2f} / {e['FPR95']:6.2f}   (draws {e['n_draws']})")
            lines.append("")
    lines.append(f"SELECTED: {chosen}  (feasible {len(feasible)} of {len(cands)})")
    P.write_json(base / "sel12_summary.json", out)
    (base / "sel12_table.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print("SEL12_DONE", flush=True)


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
