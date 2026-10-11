"""Phase 14 (c): the recurrence x mixing-rate grid of Phases 12/13 (same test sets: same random choice of classes and images) with
the cross-fitted ladder: the two fixed references (Storey q = 0.2, plain BH q = 0.2) and the dev-selected pair, all three modes
(pp, pps, min); p+ only; static (prototype rank); the Phase 12 read-out; kNN; LGC max. Views L14 + D3L, standalone and x TINS.
Output: P14/results/rec/<ds>/c<c>_r<r>_rep<i>.{csv,meta.json}."""
import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import p14common as P
import core14 as K

VIEWS = ["L14", "D3L"]
GRID = {"ninco": [(c, r) for c in (5, 10, 20, 40, 64) for r in (1, 2, 5, 10, 20, 30)],
        "ssb_hard": [(c, r) for c in (10, 30, 100, 300, 980) for r in (1, 2, 5, 10, 20, 50)]}
REPS = 3


def run(worker, nworkers, device="cuda"):
    sel = P.selected()
    pairs = P.pairs_for(sel)
    D = P.load_p3("openood")
    P.add_views(D, "openood", ["D3L"])
    paths = P.image_paths("openood")[16000:]
    ids = list(D["ids"])
    assert len(paths) == len(ids)
    cls = {s: Path(pth).parts[-2] for s, pth in zip(ids, paths)}
    sets = {}
    for ds, seed, path in P.stream_tasks("openood"):
        if ds not in ("ninco", "ssb_hard") or seed != 123:
            continue
        z = np.load(path, allow_pickle=True)
        sid, flag, S = z["sample_id"], z["is_ood"].astype(bool), np.log(np.maximum(z["S_final"].astype(np.float64), 1e-300))
        sets[ds] = {"id": sid[~flag], "ood": sid[flag], "logS": dict(zip(sid, S))}
    id_ids = sets["ninco"]["id"]
    assert set(id_ids) == set(sets["ssb_hard"]["id"]) and len(id_ids) == 45000
    shots = {v: (P.l2n(D["shots"][v][:12000]), P.l2n(D["shots"][v][12000:16000])) for v in VIEWS}
    labels_sup = np.repeat(np.arange(1000), 12)
    tasks = [(ds, c, r, rep) for ds in GRID for (c, r) in GRID[ds] for rep in range(REPS)][worker::nworkers]
    for ds, c, r, rep in tasks:
        out_dir = P.RESULTS14 / "rec" / ds
        out_dir.mkdir(parents=True, exist_ok=True)
        name = f"c{c}_r{r}_rep{rep}"
        if (out_dir / f"{name}.csv").exists():
            continue
        t0 = time.time()
        rng = np.random.default_rng([12, int(c), int(r), rep, 0 if ds == "ninco" else 1])          # the Phase 12/13 test sets
        by_class = defaultdict(list)
        for s in sets[ds]["ood"]:
            by_class[cls[s]].append(s)
        eligible = sorted(k for k, v in by_class.items() if len(v) >= r)
        if len(eligible) < c:
            continue
        chosen = rng.choice(eligible, size=c, replace=False)
        ood_ids = []
        for k in chosen:
            ood_ids.extend(rng.choice(np.array(by_class[k]), size=r, replace=False).tolist())
        test_ids = np.concatenate([id_ids, np.array(ood_ids)])
        flag = np.r_[np.zeros(len(id_ids), bool), np.ones(len(ood_ids), bool)]
        rows = np.array([D["row"][x] for x in test_ids])
        logS = np.array([sets[ds]["logS"][x] for x in test_ids])
        graphs, terms, feats = {}, {}, {}
        for v in VIEWS:
            sup, cal = shots[v]
            sf = P.l2n(D["ev"][v][rows - 4000])
            g = K.Graph(sup, cal, sf, device)
            sc, st = K.proto_scores(sup, cal, sf, device)
            graphs[v], terms[v], feats[v] = g, K.view_terms(g, sc, st), (sup, cal, sf)
        tag = {"ds": ds, "c": c, "r": r, "rep": rep, "n_id": int(len(id_ids)), "n_ood": int(len(ood_ids)), "mixing_rate": float(len(ood_ids) / len(test_ids))}
        rows_out, meta, _ = K.score_all(graphs, terms, feats, flag, {("TINS", 123): logS}, labels_sup, 1000, tag, device, pairs=pairs, v1_qs=(P.Q_PAPER,))
        for g in graphs.values():
            g.free()
        meta["selected"] = list(sel)
        meta["seconds"] = round(time.time() - t0, 1)
        pd.DataFrame(rows_out).to_csv(out_dir / f"{name}.csv", index=False)
        (out_dir / f"{name}.meta.json").write_text(json.dumps(meta, indent=1, default=str) + "\n")
        pick = {r_["cfg"]: round(r_["FPR95"], 1) for r_ in rows_out if r_["base"] == "none" and r_["cfg"] in (K.cfg_name(*sel), "xfit|storey|q0.2|pp", "xfit|bh|q0.2|pp", "v1|storey|q0.2|pp", "xfit|lp", "xfit|static")}
        print(json.dumps({"ds": ds, "c": c, "r": r, "rep": rep, "n_ood": len(ood_ids), "FPR95": pick, "seconds": meta["seconds"], "utc": P.utc()}), flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--worker", type=int, default=0)
    ap.add_argument("--nworkers", type=int, default=1)
    a = ap.parse_args()
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    run(a.worker, a.nworkers)
