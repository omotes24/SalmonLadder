"""Phase 17 (registered, prereg_p17.json). Every task builds the paper's graphs once and computes, on the same graphs:
  the paper's statistic sigma (cross-fitted Storey q = 0.2, p+ p-, pooled) and its p_fin, p+ only, static p, kNN  (core14.score_all,
  exactly as in Phases 14-16), and sigma_ref (core17.sigma_ref; registered). Per step also:
  --step dev   : dev1, dev2 (the registered adoption rule is evaluated on dev2 by ana17.py), the three TINS orders as bases;
  --step test  : OpenOOD v1.5 ImageNet-1K near / far, Four-OOD, six draws, standalone and x bases; + the decomposition rows;
  --step in200 : ImageNet-200, six draws; + decomposition;     --step cifar : CIFAR-100, six draws; + decomposition;
  --step rec   : the recurrence x mixing grid (60 cells x 3 reps); + the separation bounds (Corollary 2 for sigma and sigma_ref,
                 Corollary A1 for the Phase 16 read-out h), per unknown image;
  --step csid  : covariate-shifted ID: the ImageNet-1K ID test images + ImageNet-V2 or ImageNet-R (treated as ID) + one OpenOOD
                 OOD set, six draws, standalone (no base scores exist for these images).
Output: P17/results/p17_<step>/... (CSV + meta.json + npz per task)."""
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

import p17common as P
import core14 as K
import core17 as X
import core12 as C12
from engine import _log, p_low
from an11 import measures
from draws12 import pool_shots, datasets, DRAWS
from test14 import vlm_bases
from in200_14 import layout
from rec14 import GRID, REPS
from extract14 import CIFAR, OOD as CIFAR_OOD

sys.path.insert(0, "/home/omote/reprise_p5_20261002/code")
import dev5  # noqa: E402

VIEWS = list(P.VIEWS17)
PAIRS = [("storey", 0.2)]


def build(sup_cal, sf_of, device):
    graphs, terms, feats = {}, {}, {}
    for v in VIEWS:
        sup, cal = sup_cal[v]
        sf = sf_of(v)
        g = K.Graph(sup, cal, sf, device)
        graphs[v], terms[v], feats[v] = g, K.view_terms(g), (sup, cal, sf)
    return graphs, terms, feats


def score_task(graphs, terms, feats, flag, bases, labels_sup, n_classes, tag, device, decomp=False, bounds=None):
    rows, meta, arrays = K.score_all(graphs, terms, feats, flag, bases, labels_sup, n_classes, tag, device, pairs=PAIRS, modes=("pp",),
                                     v1_qs=(0.2,), with_lgc=False, save_cfg=X.PF)
    ref = X.sigma_ref(graphs, terms)
    rows += X.ref_rows(ref, flag, bases, tag)
    meta["ref"] = ref["info"]
    arrays.update({"ref_s": np.asarray(ref["s"], np.float64), "ref_pfin": np.asarray(ref["pfin"], np.float64)})
    for v in VIEWS:
        arrays[f"ref_seeds_{v}"] = np.asarray(ref["seeds"][v], np.int64)
    if decomp or bounds is not None:
        xi = X.xfit_internals(graphs, terms)
        s_all = X.xfit_stat(graphs, terms, xi)
        meta["sigma_recomputed_maxdiff"] = float(np.max(np.abs(s_all[xi["nc"]:] - arrays["s"])))
        assert meta["sigma_recomputed_maxdiff"] < 1e-4, meta["sigma_recomputed_maxdiff"]     # score_all stores sigma as float32
        Wd = {v: X.degrees(g) for v, g in graphs.items()}
        if decomp:
            rows += X.decomposition_rows(graphs, terms, xi, Wd, flag, tag)
        if bounds is not None:
            cls_of_test = bounds
            c2 = X.corollary2_sigma(graphs, terms, xi, Wd, s_all, cls_of_test)
            c2r = X.corollary2_ref(graphs, terms, ref, Wd, cls_of_test)
            hi = X.h16_internals(graphs)
            a1 = X.corollaryA1(hi, cls_of_test)
            meta["bounds"] = {"cor2_sigma_violations": int(np.sum(c2["sigma"] > c2["B"] + 1e-9)),
                              "cor2_ref_violations": int(np.sum(c2r["sigma"] > c2r["B"] + 1e-9)),
                              "corA1_violations": int(np.sum(a1["h"] > a1["pi"] + 1e-9)) if len(a1["h"]) else 0,
                              "corA1_phibar_violations": int(np.sum(a1["pi"] > a1["phibar"] + 1e-9)) if len(a1["pi"]) else 0}
            for k, val in c2.items():
                arrays[f"c2_{k}"] = np.asarray(val, np.float64)
            for k, val in c2r.items():
                arrays[f"c2r_{k}"] = np.asarray(val, np.float64)
            for k, val in a1.items():
                arrays[f"a1_{k}"] = np.asarray(val, np.float64)
            arrays["cls_of_test"] = np.asarray(cls_of_test, np.int64)
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
    print(json.dumps({"task": str(out_dir.relative_to(P.RESULTS17) / name), "seconds": meta["seconds"], "utc": meta["utc"],
                      "bounds": meta.get("bounds")}), flush=True)


def run_dev(devs, worker, nworkers, device):
    for dev in devs:
        assert dev in ("dev1", "dev2")
        out_dir = P.RESULTS17 / "p17_dev" / dev
        D = dev5.load(dev)
        groups = defaultdict(list)
        for (draw, stream, seed) in sorted(D["tins"].keys()):
            groups[(draw, stream)].append(seed)
        for (draw, stream) in sorted(groups)[worker::nworkers]:
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
        out_root = P.RESULTS17 / "p17_test" / part
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
            rows_out, meta, arrays = score_task(graphs, terms, feats, flag, bases, labels_sup, 1000, tag, device, decomp=True)
            for g in graphs.values():
                g.free()
            finish(out_root / d, ds, rows_out, meta, arrays, t0, ids, flag, {"bases": sorted({b for (b, s) in bases})})


def run_in200(worker, nworkers, device):
    D = P.load_p3("openood")
    P.add_views(D, "openood", ["D3L"])
    id_ids, sup_rows, cal_rows = layout(D, P.image_paths("openood"))
    sets = datasets("openood")
    tasks = [(d, ds) for d in DRAWS for ds in sorted(sets)][worker::nworkers]
    out_root = P.RESULTS17 / "p17_in200"
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
        rows_out, meta, arrays = score_task(graphs, terms, feats, flag, {}, labels_sup, 200, tag, device, decomp=True)
        for g in graphs.values():
            g.free()
        finish(out_root / d, ds, rows_out, meta, arrays, t0, test_ids, flag)


def run_cifar(worker, nworkers, device):
    t = pd.read_parquet(CIFAR / "table.parquet")
    F = {v: np.load(CIFAR / "features" / f"{v}.npy", mmap_mode="r") for v in VIEWS}
    id_test = t.index[(t.split == "test")].values
    tasks = [(d, ds) for d in range(6) for ds in CIFAR_OOD][worker::nworkers]
    out_root = P.RESULTS17 / "p17_cifar"
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
        rows_out, meta, arrays = score_task(graphs, terms, feats, flag, {}, labels_sup, 100, tag, device, decomp=True)
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
        out_dir = P.RESULTS17 / "p17_rec" / ds
        name = f"c{c}_r{r}_rep{rep}"
        if (out_dir / f"{name}.csv").exists():
            continue
        t0 = time.time()
        rng = np.random.default_rng([12, int(c), int(r), rep, 0 if ds == "ninco" else 1])     # the Phase 12-16 test sets
        by_class = defaultdict(list)
        for s in sets[ds]["ood"]:
            by_class[cls[s]].append(s)
        eligible = sorted(k for k, v in by_class.items() if len(v) >= r)
        if len(eligible) < c:
            continue
        chosen = rng.choice(eligible, size=c, replace=False)
        ood_ids, ood_cls = [], []
        for j, k in enumerate(chosen):
            pick = rng.choice(np.array(by_class[k]), size=r, replace=False).tolist()
            ood_ids.extend(pick)
            ood_cls.extend([j] * len(pick))
        test_ids = np.concatenate([id_ids, np.array(ood_ids)])
        flag = np.r_[np.zeros(len(id_ids), bool), np.ones(len(ood_ids), bool)]
        cls_of_test = np.r_[np.full(len(id_ids), -1), np.array(ood_cls)]
        rows = np.array([D["row"][x] for x in test_ids])
        logS = np.array([sets[ds]["logS"][x] for x in test_ids])
        graphs, terms, feats = build(shots, lambda v: P.l2n(D["ev"][v][rows - 4000]), device)
        tag = {"ds": ds, "c": c, "r": r, "rep": rep, "n_id": int(len(id_ids)), "n_ood": int(len(ood_ids)), "mixing_rate": float(len(ood_ids) / len(test_ids))}
        rows_out, meta, arrays = score_task(graphs, terms, feats, flag, {("TINS", 123): logS}, labels_sup, 1000, tag, device, bounds=cls_of_test)
        for g in graphs.values():
            g.free()
        finish(out_dir, name, rows_out, meta, arrays, t0, test_ids, flag)


def csid_metrics(score, is_ood, is_csid):
    """Sub-metrics of one per-image score (larger = more ID-like) on a test set of ID, csid and OOD images."""
    out = {}
    au, fp = measures(score, is_ood)                                  # full spectrum: ID and csid are both ID
    out.update({"fs_AUROC": au, "fs_FPR95": fp})
    m = ~(~is_ood & ~is_csid)                                         # csid vs OOD
    au, fp = measures(score[m], is_ood[m])
    out.update({"csid_ood_AUROC": au, "csid_ood_FPR95": fp})
    m = ~is_csid                                                      # ID vs OOD in the same run
    au, fp = measures(score[m], is_ood[m])
    out.update({"id_ood_AUROC": au, "id_ood_FPR95": fp})
    m = ~is_ood                                                       # ID vs csid: does the read-out call shifted ID unknown?
    au, fp = measures(score[m], is_csid[m])
    out.update({"id_csid_AUROC": au, "id_csid_FPR95": fp})
    return out


def run_csid(worker, nworkers, device):
    D = P.load_p3("openood")
    P.add_views(D, "openood", ["D3L"])
    sets = datasets("openood")
    id0 = None
    for ds in sorted(sets):
        ids, flag, _ = sets[ds]
        if id0 is None:
            id0 = ids[~flag]
        assert np.array_equal(np.sort(ids[~flag]), np.sort(id0))
    CF = {name: {v: np.load(P.FEATS17 / name / f"{v}.npy", mmap_mode="r") for v in VIEWS} for name in P.CSID_ROOTS}
    tasks = [(d, cs, ds) for d in DRAWS for cs in sorted(P.CSID_ROOTS) for ds in sorted(sets)][worker::nworkers]
    out_root = P.RESULTS17 / "p17_csid"
    labels_sup = np.repeat(np.arange(1000), 12)
    for d, cs, ds in tasks:
        out_dir = out_root / cs / d
        if (out_dir / f"{ds}.csv").exists():
            continue
        t0 = time.time()
        shots = {v: (P.l2n(D["shots"][v][:12000]), P.l2n(D["shots"][v][12000:16000])) for v in VIEWS} if d == "p3" else pool_shots(d)
        ids, flag, _ = sets[ds]
        id_ids, ood_ids = ids[~flag], ids[flag]
        ncs = len(CF[cs][VIEWS[0]])
        is_ood = np.r_[np.zeros(len(id_ids) + ncs, bool), np.ones(len(ood_ids), bool)]
        is_csid = np.r_[np.zeros(len(id_ids), bool), np.ones(ncs, bool), np.zeros(len(ood_ids), bool)]
        rid, rood = np.array([D["row"][x] for x in id_ids]), np.array([D["row"][x] for x in ood_ids])

        def sf_of(v):
            return P.l2n(np.concatenate([np.asarray(D["ev"][v][rid - 4000]), np.asarray(CF[cs][v]), np.asarray(D["ev"][v][rood - 4000])]))
        graphs, terms, feats = build(shots, sf_of, device)
        tag = {"csid": cs, "draw": d, "ds": ds, "n": int(len(is_ood)), "n_id": int(len(id_ids)), "n_csid": int(ncs), "n_ood": int(len(ood_ids))}
        rows_out, meta, arrays = score_task(graphs, terms, feats, is_ood, {}, labels_sup, 1000, tag, device)
        # per-image scores for the sub-metrics: sigma (paper), sigma_ref, p+ only (cross-fit), static p, kNN (all calibration)
        xi = X.xfit_internals(graphs, terms)
        s_all = X.xfit_stat(graphs, terms, xi)
        assert np.max(np.abs(s_all[xi["nc"]:] - arrays["s"])) < 1e-4                          # float32 storage
        lp_all = X.xfit_stat(graphs, terms, {**xi, "uneg": {k: np.zeros_like(v) for k, v in xi["uneg"].items()}})
        st, kn = 0.0, 0.0
        for v in VIEWS:
            sup, cal, sf = feats[v]
            sc, stt = K.proto_scores(sup, cal, sf, device)
            st = st + _log(p_low(sc, stt))
            kn = kn + C12.knn_rank(sup, cal, sf, dev=device)
        scores = {"sigma": s_all[xi["nc"]:], "sigma_ref": arrays["ref_s"], "lp": lp_all[xi["nc"]:], "static_all": st, "knn": kn}
        sub = {name: csid_metrics(np.asarray(x, np.float64), is_ood, is_csid) for name, x in scores.items()}
        pf, pr = arrays["pfinal"], arrays["ref_pfin"]
        for a in (0.05, 0.2):
            sub[f"fpr_id@{a:g}"] = {"sigma": float((pf[~is_ood & ~is_csid] <= a).mean()), "sigma_ref": float((pr[~is_ood & ~is_csid] <= a).mean())}
            sub[f"fpr_csid@{a:g}"] = {"sigma": float((pf[is_csid] <= a).mean()), "sigma_ref": float((pr[is_csid] <= a).mean())}
        nfix = graphs[VIEWS[0]].nfix
        sd_csid, sd_all = 0, 0
        for (v, b), sd in xi["seeds"].items():
            t = sd - nfix
            sd_all += len(t)
            sd_csid += int(is_csid[t].sum())
        sub["seeds_sigma"] = {"all": sd_all, "csid": sd_csid}
        r_all, r_csid = 0, 0
        for v in VIEWS:
            sd = arrays[f"ref_seeds_{v}"]
            t = sd[sd >= nfix] - nfix
            r_all += len(sd)
            r_csid += int(is_csid[t].sum())
        sub["seeds_ref"] = {"all": r_all, "csid": r_csid}
        meta["csid_metrics"] = sub
        arrays.update({"is_csid": is_csid, "lp_stat": lp_all[xi["nc"]:].astype(np.float32), "static_all": np.asarray(st, np.float32), "knn": np.asarray(kn, np.float32)})
        for g in graphs.values():
            g.free()
        finish(out_dir, ds, rows_out, meta, arrays, t0, None, None)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--step", required=True, choices=["dev", "test", "in200", "cifar", "rec", "csid"])
    ap.add_argument("--devs", default="dev1,dev2")
    ap.add_argument("--worker", type=int, default=0)
    ap.add_argument("--nworkers", type=int, default=1)
    a = ap.parse_args()
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    if a.step == "dev":
        run_dev(a.devs.split(","), a.worker, a.nworkers, "cuda")
    else:
        {"test": run_test, "in200": run_in200, "cifar": run_cifar, "rec": run_rec, "csid": run_csid}[a.step](a.worker, a.nworkers, "cuda")
    print("P17_DONE", flush=True)
