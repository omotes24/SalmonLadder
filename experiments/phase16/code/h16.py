"""Phase 16 (registered, prereg_p16.json): the one-quantity read-out h (core16.h16) next to the Phase 14 reference sigma
(cross-fitted Storey q = 0.2, p+ p-, pooled; recomputed in the same run on the same graphs), evaluated on
  --step dev    : the development splits dev1 (selection of the base combination) and dev2 (confirmation: H1-H3),
                  every draw x stream, the three TINS orders as bases of one graph,
  --step test   : OpenOOD v1.5 ImageNet-1K near / far and Four-OOD, the benchmark calibration, the six draws, standalone and
                  x bases (TINS per order, TANL / NegLabel / AdaNeg official scores),
  --step in200  : ImageNet-200, six draws,
  --step cifar  : CIFAR-100, six draws, views L14 + D3L,
  --step rec    : the recurrence x mixing-rate grid, 60 cells x 3 repetitions.
Steps other than dev are run only after the dev2 confirmation has been computed (orchestrate16.sh) and the registration has been
committed publicly. Output: P16/results/h16_<step>/... (one CSV + meta.json per task, an npz with the per-image scores)."""
import argparse
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import p16common as P
import core14 as K
import core16 as H
from draws12 import pool_shots, datasets, DRAWS
from test14 import vlm_bases
from in200_14 import layout
from rec14 import GRID, REPS
from extract14 import CIFAR, OOD as CIFAR_OOD

sys.path.insert(0, "/home/omote/reprise_p5_20261002/code")
import dev5  # noqa: E402

VIEWS = list(P.VIEWS16)
PAIRS = [("storey", P.Q_PAPER)]
REF = "xfit|storey|q0.2|pp|pooled"


def build(sup_cal, sf_of, device):
    graphs, terms, feats = {}, {}, {}
    for v in VIEWS:
        sup, cal = sup_cal[v]
        sf = sf_of(v)
        g = K.Graph(sup, cal, sf, device)
        graphs[v], terms[v], feats[v] = g, K.view_terms(g), (sup, cal, sf)
    return graphs, terms, feats


def score_task(graphs, terms, feats, flag, bases, labels_sup, n_classes, tag, device):
    rows, meta, arrays = K.score_all(graphs, terms, feats, flag, bases, labels_sup, n_classes, tag, device, pairs=PAIRS, modes=("pp",),
                                     v1_qs=(P.Q_PAPER,), with_lgc=False, save_cfg="xfit|storey|q0.2|pp")
    out = H.h16(graphs)
    rows += H.h16_rows(out, flag, bases, tag)
    meta["h16"] = out["info"]
    arrays.update({"h16_s": np.asarray(out["s"], np.float32), "h16_p": np.asarray(out["p"], np.float32),
                   "h16_h0": np.asarray(out["h0_test"], np.float32), "h16_seeds": np.asarray(out["seed_nodes"], np.int64)})
    return rows, meta, arrays


def finish(out_dir, name, rows_out, meta, arrays, t0, ids=None, flag=None, extra_meta=None):
    out_dir.mkdir(parents=True, exist_ok=True)
    meta.update(extra_meta or {})
    meta["seconds"] = round(time.time() - t0, 1)
    meta["utc"] = P.utc()
    pd.DataFrame(rows_out).to_csv(out_dir / f"{name}.csv", index=False)
    (out_dir / f"{name}.meta.json").write_text(json.dumps(meta, indent=1, default=str) + "\n")
    tmp = out_dir / f"{name}.{os.getpid()}.tmp.npz"
    np.savez_compressed(tmp, **({} if ids is None else {"sample_id": np.asarray(ids), "is_ood": np.asarray(flag)}), **arrays)
    os.replace(tmp, out_dir / f"{name}.npz")
    pick = {r["cfg"]: round(r["FPR95"], 2) for r in rows_out if r["base"] == "none" and r["cfg"] in (REF, "h16|s", "h16|p")}
    print(json.dumps({"task": str(out_dir.relative_to(P.RESULTS16) / name), "FPR95": pick, "seeds_h16": meta["h16"]["seeds"],
                      "seconds": meta["seconds"], "utc": meta["utc"]}), flush=True)


def run_dev(devs, worker, nworkers, device):
    for dev in devs:
        assert dev in ("dev1", "dev2")
        out_dir = P.RESULTS16 / "h16_dev" / dev
        D = dev5.load(dev)
        groups = defaultdict(list)
        for (draw, stream, seed) in sorted(D["tins"].keys()):
            groups[(draw, stream)].append(seed)
        tasks = sorted(groups)[worker::nworkers]
        for (draw, stream) in tasks:
            name = f"{dev}_draw{draw}_{stream}"
            if (out_dir / f"{name}.csv").exists():
                continue
            t0 = time.time()
            seeds = groups[(draw, stream)]
            sid0, flag0, _, _ = D["tins"][(draw, stream, seeds[0])]
            order = np.argsort(np.asarray(sid0))
            ids, flag = np.asarray(sid0)[order], np.asarray(flag0, bool)[order]
            bases = {}
            for sd in seeds:
                sid, fl, S, _ = D["tins"][(draw, stream, sd)]
                pos = {x: i for i, x in enumerate(sid)}
                idx = np.array([pos[x] for x in ids])
                assert np.array_equal(np.asarray(fl, bool)[idx], flag)
                bases[("TINS", int(sd))] = np.log(np.maximum(np.asarray(S, np.float64)[idx], 1e-300))
            rows = np.array([D["stream_row"][s] for s in ids])
            raw = {v: dev5.raw_view(D, draw, v, rows) for v in VIEWS}
            sup_cal = {v: (P.l2n(raw[v][0].reshape(-1, raw[v][0].shape[-1])), P.l2n(raw[v][1])) for v in VIEWS}
            graphs, terms, feats = build(sup_cal, lambda v: P.l2n(raw[v][2]), device)
            n_classes = len(sup_cal[VIEWS[0]][0]) // 12
            tag = {"dev": dev, "draw": draw, "stream": stream, "n": int(len(ids)), "n_ood": int(flag.sum())}
            rows_out, meta, arrays = score_task(graphs, terms, feats, flag, bases, np.repeat(np.arange(n_classes), 12), n_classes, tag, device)
            for g in graphs.values():
                g.free()
            finish(out_dir, name, rows_out, meta, arrays, t0, ids, flag, {"tins_orders": [int(s) for s in seeds]})


def run_test(worker, nworkers, device):
    for part in ("openood", "fourood"):
        D = P.load_p3(part)
        P.add_views(D, part, ["D3L"])
        sets = datasets(part)
        tasks = [(d, ds) for d in DRAWS for ds in sorted(sets)][worker::nworkers]
        out_root = P.RESULTS16 / "h16_test" / part
        todo = [(d, ds) for d, ds in tasks if not (out_root / d / f"{ds}.csv").exists()]
        shots = {}
        for d in sorted(set(d for d, _ in todo)):
            shots[d] = {v: (P.l2n(D["shots"][v][:12000]), P.l2n(D["shots"][v][12000:16000])) for v in VIEWS} if d == "p3" else pool_shots(d)
        labels_sup = np.repeat(np.arange(1000), 12)
        for d, ds in todo:
            t0 = time.time()
            ids, flag, logS = sets[ds]
            rows = np.array([D["row"][x] for x in ids])
            bases = {("TINS", int(seed)): S for seed, S in logS.items()}
            bases.update(vlm_bases(part, ds, ids))
            graphs, terms, feats = build(shots[d], lambda v: P.l2n(D["ev"][v][rows - 4000]), device)
            tag = {"part": part, "draw": d, "ds": ds, "n": int(len(ids)), "n_ood": int(flag.sum())}
            rows_out, meta, arrays = score_task(graphs, terms, feats, flag, bases, labels_sup, 1000, tag, device)
            for g in graphs.values():
                g.free()
            finish(out_root / d, ds, rows_out, meta, arrays, t0, ids, flag, {"bases": sorted({b for (b, s) in bases})})


def run_in200(worker, nworkers, device):
    D = P.load_p3("openood")
    P.add_views(D, "openood", ["D3L"])
    id_ids, sup_rows, cal_rows = layout(D, P.image_paths("openood"))
    sets = datasets("openood")
    tasks = [(d, ds) for d in DRAWS for ds in sorted(sets)][worker::nworkers]
    out_root = P.RESULTS16 / "h16_in200"
    todo = [(d, ds) for d, ds in tasks if not (out_root / d / f"{ds}.csv").exists()]
    shots = {}
    for d in sorted(set(d for d, _ in todo)):
        full = {v: (P.l2n(D["shots"][v][:12000]), P.l2n(D["shots"][v][12000:16000])) for v in VIEWS} if d == "p3" else pool_shots(d)
        shots[d] = {v: (full[v][0][sup_rows], full[v][1][cal_rows - 12000]) for v in VIEWS}
    labels_sup = np.repeat(np.arange(200), 12)
    for d, ds in todo:
        t0 = time.time()
        ids_all, flag_all, _ = sets[ds]
        ood_ids = ids_all[flag_all]
        test_ids = np.concatenate([id_ids, ood_ids])
        flag = np.r_[np.zeros(len(id_ids), bool), np.ones(len(ood_ids), bool)]
        rows = np.array([D["row"][x] for x in test_ids])
        graphs, terms, feats = build(shots[d], lambda v: P.l2n(D["ev"][v][rows - 4000]), device)
        tag = {"benchmark": "imagenet200", "draw": d, "ds": ds, "n": int(len(test_ids)), "n_id": int(len(id_ids)), "n_ood": int(len(ood_ids)), "n_cal": int(len(cal_rows))}
        rows_out, meta, arrays = score_task(graphs, terms, feats, flag, {}, labels_sup, 200, tag, device)
        for g in graphs.values():
            g.free()
        finish(out_root / d, ds, rows_out, meta, arrays, t0, test_ids, flag)


def run_cifar(worker, nworkers, device):
    t = pd.read_parquet(CIFAR / "table.parquet")
    F = {v: np.load(CIFAR / "features" / f"{v}.npy", mmap_mode="r") for v in VIEWS}
    id_test = t.index[(t.split == "test")].values
    tasks = [(d, ds) for d in range(6) for ds in CIFAR_OOD][worker::nworkers]
    out_root = P.RESULTS16 / "h16_cifar"
    labels_sup = np.repeat(np.arange(100), 12)
    for d, ds in tasks:
        out_dir = out_root / f"draw{d}"
        if (out_dir / f"{ds}.csv").exists():
            continue
        t0 = time.time()
        tr = t[(t.split == "train") & (t.draw == d)].sort_values(["label", "pos"])
        sup_rows, cal_rows = tr.index[tr.role == "sup"].values, tr.index[tr.role == "cal"].values
        ood_rows = t.index[t.dataset == ds].values
        test_rows = np.concatenate([id_test, ood_rows])
        flag = np.r_[np.zeros(len(id_test), bool), np.ones(len(ood_rows), bool)]
        sup_cal = {v: (P.l2n(np.asarray(F[v][sup_rows])), P.l2n(np.asarray(F[v][cal_rows]))) for v in VIEWS}
        graphs, terms, feats = build(sup_cal, lambda v: P.l2n(np.asarray(F[v][test_rows])), device)
        tag = {"benchmark": "cifar100", "draw": d, "ds": ds, "group": CIFAR_OOD[ds], "views": "L14+D3L", "n": int(len(test_rows)), "n_ood": int(len(ood_rows))}
        rows_out, meta, arrays = score_task(graphs, terms, feats, flag, {}, labels_sup, 100, tag, device)
        for g in graphs.values():
            g.free()
        finish(out_dir, ds, rows_out, meta, arrays, t0, test_rows, flag, {"n_id": int(len(id_test)), "n_ood": int(len(ood_rows))})


def run_rec(worker, nworkers, device):
    D = P.load_p3("openood")
    P.add_views(D, "openood", ["D3L"])
    paths = P.image_paths("openood")[16000:]
    cls = {s: Path(pth).parts[-2] for s, pth in zip(list(D["ids"]), paths)}
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
    tasks = [(ds, c, r, rep) for ds in GRID for (c, r) in GRID[ds] for rep in range(REPS)][worker::nworkers]
    for ds, c, r, rep in tasks:
        out_dir = P.RESULTS16 / "h16_rec" / ds
        name = f"c{c}_r{r}_rep{rep}"
        if (out_dir / f"{name}.csv").exists():
            continue
        t0 = time.time()
        rng = np.random.default_rng([12, int(c), int(r), rep, 0 if ds == "ninco" else 1])     # the Phase 12-15 test sets
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
        graphs, terms, feats = build(shots, lambda v: P.l2n(D["ev"][v][rows - 4000]), device)
        tag = {"ds": ds, "c": c, "r": r, "rep": rep, "n_id": int(len(id_ids)), "n_ood": int(len(ood_ids)), "mixing_rate": float(len(ood_ids) / len(test_ids))}
        rows_out, meta, arrays = score_task(graphs, terms, feats, flag, {("TINS", 123): logS}, labels_sup, 1000, tag, device)
        for g in graphs.values():
            g.free()
        finish(out_dir, name, rows_out, meta, arrays, t0, test_ids, flag)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--step", required=True, choices=["dev", "test", "in200", "cifar", "rec"])
    ap.add_argument("--devs", default="dev1,dev2")
    ap.add_argument("--worker", type=int, default=0)
    ap.add_argument("--nworkers", type=int, default=1)
    a = ap.parse_args()
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    if a.step == "dev":
        run_dev(a.devs.split(","), a.worker, a.nworkers, "cuda")
    else:
        {"test": run_test, "in200": run_in200, "cifar": run_cifar, "rec": run_rec}[a.step](a.worker, a.nworkers, "cuda")
    print("H16_DONE", flush=True)
