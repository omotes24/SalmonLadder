"""Phase 15 (c): the recurrence grid with the neighbour count k_g of the graph varied (3, 5, 10, 20, 40; the paper uses 10).
Same test sets as rec14.py (the Phase 12/13 random choice of classes and images) for NINCO c = 64 and SSB-hard c = 980, r in the
grid, 3 repetitions; the registered configuration (Storey q = 0.2, p+ p-), p+ only, static, the Phase 12 read-out, kNN;
views L14 + D3L, standalone and x TINS. One graph (exact top-40 lists) per test set, all k_g evaluated on it.
Output: P15/results/kg/<ds>/r<r>_rep<i>_kg<k>.{csv,meta.json}."""
import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import p15common as P
import core14 as K
import core15 as K15
from rec14 import GRID, REPS, VIEWS

CMAX = {"ninco": 64, "ssb_hard": 980}


def run(worker, nworkers, device="cuda"):
    P.install()
    sel = P.selected()
    pairs = [(sel[0], sel[1])]
    D = P.load_p3("openood")
    P.add_views(D, "openood", ["D3L"])
    paths = P.image_paths("openood")[16000:]
    ids = list(D["ids"])
    cls = {s: Path(pth).parts[-2] for s, pth in zip(ids, paths)}
    sets = {}
    for ds, seed, path in P.stream_tasks("openood"):
        if ds not in ("ninco", "ssb_hard") or seed != 123:
            continue
        z = np.load(path, allow_pickle=True)
        sid, flag, S = z["sample_id"], z["is_ood"].astype(bool), np.log(np.maximum(z["S_final"].astype(np.float64), 1e-300))
        sets[ds] = {"id": sid[~flag], "ood": sid[flag], "logS": dict(zip(sid, S))}
    id_ids = sets["ninco"]["id"]
    shots = {v: (P.l2n(D["shots"][v][:12000]), P.l2n(D["shots"][v][12000:16000])) for v in VIEWS}
    labels_sup = np.repeat(np.arange(1000), 12)
    tasks = [(ds, c, r, rep) for ds in GRID for (c, r) in GRID[ds] if c == CMAX[ds] for rep in range(REPS)][worker::nworkers]
    for ds, c, r, rep in tasks:
        out_dir = P.RESULTS15 / "kg" / ds
        out_dir.mkdir(parents=True, exist_ok=True)
        names = {k: f"r{r}_rep{rep}_kg{k}" for k in P.KGS}
        if all((out_dir / f"{n}.csv").exists() for n in names.values()):
            continue
        t0 = time.time()
        rng = np.random.default_rng([12, int(c), int(r), rep, 0 if ds == "ninco" else 1])          # the Phase 12/13/14 test sets
        by_class = defaultdict(list)
        for s in sets[ds]["ood"]:
            by_class[cls[s]].append(s)
        eligible = sorted(k for k, v in by_class.items() if len(v) >= r)
        chosen = rng.choice(eligible, size=c, replace=False)
        ood_ids = []
        for k in chosen:
            ood_ids.extend(rng.choice(np.array(by_class[k]), size=r, replace=False).tolist())
        test_ids = np.concatenate([id_ids, np.array(ood_ids)])
        flag = np.r_[np.zeros(len(id_ids), bool), np.ones(len(ood_ids), bool)]
        rows = np.array([D["row"][x] for x in test_ids])
        logS = np.array([sets[ds]["logS"][x] for x in test_ids])
        graphs, feats, statics = {}, {}, {}
        for v in VIEWS:
            sup, cal = shots[v]
            sf = P.l2n(D["ev"][v][rows - 4000])
            graphs[v] = K15.Graph15(sup, cal, sf, device, kmax=max(P.KGS))
            statics[v] = K.proto_scores(sup, cal, sf, device)
            feats[v] = (sup, cal, sf)
        for kg in P.KGS:
            if (out_dir / f"{names[kg]}.csv").exists():
                continue
            t1 = time.time()
            for g in graphs.values():
                g.k = kg
            terms = {v: K.view_terms(graphs[v], *statics[v]) for v in VIEWS}
            tag = {"ds": ds, "c": c, "r": r, "rep": rep, "kg": kg, "n_id": int(len(id_ids)), "n_ood": int(len(ood_ids)), "mixing_rate": float(len(ood_ids) / len(test_ids))}
            rows_out, meta, _ = K.score_all(graphs, terms, feats, flag, {("TINS", 123): logS}, labels_sup, 1000, tag, device, pairs=pairs, modes=("pp",), v1_qs=(P.Q_PAPER,), with_lgc=False)
            meta["kg"], meta["selected"], meta["seconds"] = kg, list(sel), round(time.time() - t1, 1)
            pd.DataFrame(rows_out).to_csv(out_dir / f"{names[kg]}.csv", index=False)
            (out_dir / f"{names[kg]}.meta.json").write_text(json.dumps(meta, indent=1, default=str) + "\n")
            pick = {r_["cfg"]: round(r_["FPR95"], 1) for r_ in rows_out if r_["base"] == "none" and r_["cfg"] in (K.cfg_name(*sel) + "|pooled", "xfit|lp|pooled", "xfit|static|pooled", "v1|storey|q0.2|pp")}
            print(json.dumps({"ds": ds, "c": c, "r": r, "rep": rep, "kg": kg, "FPR95": pick, "seconds": meta["seconds"], "utc": P.utc()}), flush=True)
        for g in graphs.values():
            g.free()
        print(json.dumps({"ds": ds, "r": r, "rep": rep, "all_kg_seconds": round(time.time() - t0, 1)}), flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--worker", type=int, default=0)
    ap.add_argument("--nworkers", type=int, default=1)
    a = ap.parse_args()
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    run(a.worker, a.nworkers)
