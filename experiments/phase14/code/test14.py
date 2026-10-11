"""Phase 14 (b): the cross-fitted ladder on the public benchmarks (OpenOOD v1.5 ImageNet-1K near / far, Four-OOD).
--cal train : the benchmark protocol (calibration = 4 training images per class), six draws of the labelled images; every
              configuration (rule x q x mode, p+ only, static, kNN, LGC max, the Phase 12 read-out) standalone and x base
              (TINS per stream order; TANL / NegLabel / AdaNeg from the Phase 10 official scores); the per-image scores of the
              dev-selected configuration are saved.
--cal val   : exchangeable calibration (4 ID validation images per class, the rest of the ID images as the test set), three
              splits, the paper's draw of the support images; validity of p_final (false-alarm rate at alpha, FDP of BH).
Output: P14/results/test_<cal>/<part>/<draw or split>/<ds>.{csv,meta.json,npz}."""
import argparse
import json
import os
import time

import numpy as np
import pandas as pd
import torch

import p14common as P
import core14 as K
from draws12 import pool_shots, datasets, DRAWS

VIEWS = ["L14", "D3L"]
VLM = ("TANL", "NegLabel", "AdaNeg")
N_CAL_VAL, SPLITS = 4, 3


def vlm_bases(part, ds, ids):
    """log scores of the official VLM post-processors per stream order, aligned to ids (skipped when a score is not positive)."""
    out = {}
    for d2, seed, path in P.stream_tasks(part):
        if d2 != ds:
            continue
        f = P.RESULTS / "vlm" / part / f"{ds}_seed{seed}.npz"
        if not f.exists():
            continue
        zv = np.load(f, allow_pickle=True)
        pos = {x: i for i, x in enumerate(zv["sample_id"])}
        order = np.array([pos[x] for x in ids])
        for m in VLM:
            if m not in zv.files:
                continue
            sc = zv[m].astype(np.float64)[order]
            if (sc <= 0).any():
                continue
            out[(m, int(seed))] = np.log(sc)
    return out


def save(out_dir, ds, rows_out, meta, arrays, ids, flag):
    pd.DataFrame(rows_out).to_csv(out_dir / f"{ds}.csv", index=False)
    (out_dir / f"{ds}.meta.json").write_text(json.dumps(meta, indent=1, default=str) + "\n")
    if arrays:
        tmp = out_dir / f"{ds}.{os.getpid()}.tmp.npz"
        np.savez_compressed(tmp, sample_id=ids, is_ood=flag, **arrays)
        os.replace(tmp, out_dir / f"{ds}.npz")


def run_train_cal(part, worker, nworkers, save_cfg, device="cuda"):
    D = P.load_p3(part)
    P.add_views(D, part, ["D3L"])
    sets = datasets(part)
    tasks = [(d, ds) for d in DRAWS for ds in sorted(sets)][worker::nworkers]
    out_root = P.RESULTS14 / "test_train" / part
    todo = [(d, ds) for d, ds in tasks if not (out_root / d / f"{ds}.csv").exists()]
    if not todo:
        return
    shots = {}
    for d in sorted(set(d for d, _ in todo)):
        shots[d] = {v: (P.l2n(D["shots"][v][:12000]), P.l2n(D["shots"][v][12000:16000])) for v in VIEWS} if d == "p3" else pool_shots(d)
    labels_sup = np.repeat(np.arange(1000), 12)
    for d, ds in todo:
        (out_root / d).mkdir(parents=True, exist_ok=True)
        t0 = time.time()
        ids, flag, logS = sets[ds]
        rows = np.array([D["row"][x] for x in ids])
        bases = {("TINS", int(seed)): S for seed, S in logS.items()}
        bases.update(vlm_bases(part, ds, ids))
        graphs, terms, feats = {}, {}, {}
        for v in VIEWS:
            sup, cal = shots[d][v]
            sf = P.l2n(D["ev"][v][rows - 4000])
            g = K.Graph(sup, cal, sf, device)
            sc, st = K.proto_scores(sup, cal, sf, device)
            graphs[v], terms[v], feats[v] = g, K.view_terms(g, sc, st), (sup, cal, sf)
        tag = {"part": part, "draw": d, "ds": ds, "n": int(len(ids)), "n_ood": int(flag.sum())}
        rows_out, meta, arrays = K.score_all(graphs, terms, feats, flag, bases, labels_sup, 1000, tag, device, save_cfg=save_cfg)
        for g in graphs.values():
            g.free()
        meta["seconds"] = round(time.time() - t0, 1)
        save(out_root / d, ds, rows_out, meta, arrays, ids, flag)
        pick = {r["cfg"]: round(r["FPR95"], 2) for r in rows_out if r["base"] == "none" and r["cfg"] in ("xfit|storey|q0.2|pp", "xfit|bh|q0.2|pp", "xfit|bh|q0.2|min", "v1|storey|q0.2|pp", "xfit|lp", "xfit|static", "lgcmax")}
        print(json.dumps({"part": part, "draw": d, "ds": ds, "FPR95": pick, "bases": sorted({b for (b, s) in bases}), "seconds": meta["seconds"], "utc": P.utc()}), flush=True)


def run_val_cal(part, worker, nworkers, save_cfg, device="cuda"):
    D = P.load_p3(part)
    P.add_views(D, part, ["D3L"])
    labels, _ = P.id_labels(part, D)
    sets = datasets(part)
    tasks = [(s, ds) for s in range(SPLITS) for ds in sorted(sets)][worker::nworkers]
    out_root = P.RESULTS14 / "test_val" / part
    todo = [(s, ds) for s, ds in tasks if not (out_root / f"split{s}" / f"{ds}.csv").exists()]
    if not todo:
        return
    sup_all = {v: P.l2n(D["shots"][v][:12000]) for v in VIEWS}
    id_pos = {x: i for i, x in enumerate(D["ids"])}
    labels_sup = np.repeat(np.arange(1000), 12)
    for s, ds in todo:
        out_dir = out_root / f"split{s}"
        out_dir.mkdir(parents=True, exist_ok=True)
        t0 = time.time()
        ids, flag, logS = sets[ds]
        id_ids = ids[~flag]
        lab = np.array([labels[id_pos[x]] for x in id_ids])
        assert (lab >= 0).all()
        rng = np.random.default_rng([13, s])                                  # the Phase 13 splits
        cal_mask = np.zeros(len(id_ids), bool)
        for c in range(1000):
            cal_mask[rng.choice(np.flatnonzero(lab == c), size=N_CAL_VAL, replace=False)] = True
        cal_ids = id_ids[cal_mask]
        cal_ids = cal_ids[np.argsort(lab[cal_mask], kind="stable")]
        keep = ~np.isin(ids, cal_ids)
        test_ids, test_flag = ids[keep], flag[keep]
        bases = {("TINS", int(seed)): S[keep] for seed, S in logS.items()}
        for (m, seed), lb in vlm_bases(part, ds, ids).items():
            bases[(m, seed)] = lb[keep]
        rows_t, rows_c = np.array([D["row"][x] for x in test_ids]), np.array([D["row"][x] for x in cal_ids])
        graphs, terms, feats = {}, {}, {}
        for v in VIEWS:
            sup, cal, sf = sup_all[v], P.l2n(D["ev"][v][rows_c - 4000]), P.l2n(D["ev"][v][rows_t - 4000])
            g = K.Graph(sup, cal, sf, device)
            sc, st = K.proto_scores(sup, cal, sf, device)
            graphs[v], terms[v], feats[v] = g, K.view_terms(g, sc, st), (sup, cal, sf)
        tag = {"part": part, "split": s, "ds": ds, "n": int(len(test_ids)), "n_ood": int(test_flag.sum()), "n_id_test": int((~test_flag).sum())}
        rows_out, meta, arrays = K.score_all(graphs, terms, feats, test_flag, bases, labels_sup, 1000, tag, device, save_cfg=save_cfg)
        for g in graphs.values():
            g.free()
        meta["seconds"] = round(time.time() - t0, 1)
        save(out_dir, ds, rows_out, meta, arrays, test_ids, test_flag)
        pick = {r["cfg"]: {k: round(r[k], 3) for k in ("fpr@0.05", "tpr@0.05", "bh_fdp")} for r in rows_out if r["base"] == "none" and r["cfg"] in ("xfit|storey|q0.2|pp", "xfit|bh|q0.2|pp")}
        print(json.dumps({"part": part, "split": s, "ds": ds, "validity": pick, "seconds": meta["seconds"], "utc": P.utc()}), flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", required=True, choices=["openood", "fourood"])
    ap.add_argument("--cal", default="train", choices=["train", "val"])
    ap.add_argument("--worker", type=int, default=0)
    ap.add_argument("--nworkers", type=int, default=1)
    ap.add_argument("--save-cfg", default=None, help="configuration whose per-image scores are saved (default: the dev-selected one)")
    a = ap.parse_args()
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    save_cfg = a.save_cfg or K.cfg_name(*P.selected())
    (run_train_cal if a.cal == "train" else run_val_cal)(a.part, a.worker, a.nworkers, save_cfg)
