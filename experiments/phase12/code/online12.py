"""Phase 12 (d): the arrival-time (online) mode of the read-out on the 37 benchmark streams (batches of 256 as the stream files
fix them): the graph grows batch by batch, u+ and u- continue from the previous values (15 sweeps), the seeds are re-selected from
the current p+ of every image so far (Storey-AdaDetect, q = 0.1), and the score of a batch is emitted once (never revised).
Views L14, D3L and the joint view; base none / TINS. Output: P12/results/online/<part>/<stream>.npz, .csv, .meta.json."""
import argparse
import json
import os
import time

import numpy as np
import pandas as pd
import torch

import p12common as P
import core12 as C
from an11 import measures

VIEWS = ["L14", "D3L"]
VIEWSETS = [("D3L",), ("L14", "D3L"), ("L14xD3L", "D3L")]


def l2n(x):
    x = np.asarray(x, np.float32)
    return x / np.linalg.norm(x, axis=1, keepdims=True)


def run_part(part, worker, nworkers, q=P.Q_MAIN, device="cuda"):
    out_dir = P.RESULTS12 / ("online" if q == 0.1 else f"online_q{q:g}") / part
    out_dir.mkdir(parents=True, exist_ok=True)
    tasks = P.stream_tasks(part)
    todo = [(ds, s, p) for (ds, s, p) in tasks[worker::nworkers] if not (out_dir / f"{ds}_seed{s}.csv").exists()]
    if not todo:
        return
    D = P.load_p3(part)
    P.add_views(D, part, ["D3L"])
    shots = {v: (l2n(D["shots"][v][:12000]), l2n(D["shots"][v][12000:16000])) for v in VIEWS}
    for ds, seed, path in todo:
        name = f"{ds}_seed{seed}"
        t0 = time.time()
        z = np.load(path, allow_pickle=True)
        ids, flag, bidx = z["sample_id"], z["is_ood"].astype(bool), z["batch_index"]
        logS = np.log(np.maximum(z["S_final"].astype(np.float64), 1e-300))
        rows = np.array([D["row"][x] for x in ids])
        feats = {v: (shots[v][0], shots[v][1], l2n(D["ev"][v][rows - 4000])) for v in VIEWS}
        feats["L14xD3L"] = tuple(l2n(np.concatenate([feats["L14"][i], feats["D3L"][i]], 1)) for i in range(3))
        terms, meta = {}, {"part": part, "stream": name, "q": q, "n": int(len(ids)), "n_ood": int(flag.sum()), "batches": int(bidx.max()) + 1, "views": {}}
        for v, (sup, cal, sf) in feats.items():
            lp, ln, stats = C.online_view(sup, cal, sf, bidx, q=q, dev=device, flag=flag)
            terms[(v, "lp")], terms[(v, "ln")] = lp, ln
            last = stats[-1]
            meta["views"][v] = {"seeds_last_batch": last[1], "id_frac_last_batch": last[2],
                                "seeds_by_batch": [s[1] for s in stats[::max(1, len(stats) // 20)]]}
        arrays = {"sample_id": ids, "is_ood": flag, "bidx": bidx}
        for (v, k), x in terms.items():
            arrays[f"s::{v}::{k}"] = np.asarray(x, np.float32)
        rows_out = []
        for vs in VIEWSETS:
            for ro, ks in (("T", ["lp", "ln"]), ("Tlp", ["lp"])):
                x = sum(terms[(v, k)] for v in vs for k in ks)
                for base, add in (("none", 0.0), ("TINS", logS)):
                    au, fp = measures(x + add, flag)
                    rows_out.append({"part": part, "stream": name, "views": "+".join(vs), "readout": ro, "base": base, "AUROC": au, "FPR95": fp})
        tmp = out_dir / f"{name}.{os.getpid()}.tmp.npz"
        np.savez_compressed(tmp, meta=json.dumps(meta), **arrays)
        os.replace(tmp, out_dir / f"{name}.npz")
        (out_dir / f"{name}.meta.json").write_text(json.dumps(meta, indent=1) + "\n")
        pd.DataFrame(rows_out).to_csv(out_dir / f"{name}.csv", index=False)
        main = [r for r in rows_out if r["views"] == "D3L" and r["readout"] == "T" and r["base"] == "TINS"][0]
        print(json.dumps({"part": part, "stream": name, "D3L|T|TINS": [round(main["AUROC"], 2), round(main["FPR95"], 2)],
                          "seconds": round(time.time() - t0, 1), "utc": P.utc()}), flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", required=True, choices=["openood", "fourood"])
    ap.add_argument("--worker", type=int, default=0)
    ap.add_argument("--nworkers", type=int, default=1)
    ap.add_argument("--q", type=float, default=P.Q_MAIN)
    a = ap.parse_args()
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    run_part(a.part, a.worker, a.nworkers, q=a.q)
