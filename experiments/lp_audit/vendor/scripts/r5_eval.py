"""R5 Stage A and (1): the frozen v4 components and same-feature baselines under matched conditions (dev only).

One invocation = one dev split (VINS_WORK) x one shot draw (0..4, or orig = the shots used so far).
For each stream (near, far) and order seed (123, 124, 125), with the draw's TINS scores (r5_tins.py):
  views (DINOv2 B/14 and L/14, prototype view of iter3_eval.custom_view, frozen):
    p      static conformal p                     p_all  all-class p (entrance stage 1)
    pt3    memory p, v4 iterated entrance (0.40 / 0.30 / 0.10)
    pt1    memory p, static one-stage entrance (p_all <= 0.10)
    plp    v4 conformal LP (batch joins before it is scored: images of a batch refer to each other)
    plpi   the same LP state, but every image scored alone against the graph of earlier batches
    knn<k> conformal p of the k-th NN cosine distance to the pooled support (k in KS)
    maha<l> conformal p of Mahalanobis++ with shared-covariance shrinkage l (l in LAMS)
    raw scores d, g3 (memory), u / ui (LP mass) for the conformal decomposition
  metrics per variant: AUROC, FPR95 (upstream get_measures), within-batch AUROC; entrance statistics.
Labels (is_ood) are used only for metrics and the entrance statistics.
Output: <R5>/<dev>/eval/draw<k>/<stream>_seed<s>.npz (per sample) and .json (metrics)
"""
import argparse
import hashlib
import importlib.util
import json
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

from vins import config as C  # noqa: E402
from vins import r5  # noqa: E402
from vins.features import load_features  # noqa: E402

FROZEN_SHA = "40a9bf336c7ea02cb062b2b78fc48492445add34168c309857f12df670c161ad"
KS = (1, 2, 3, 5, 10, 20, 50)
LAMS = (0.0, 0.001, 0.01, 0.05, 0.1, 0.2, 0.5)
VIEWS = (("B14", "dino.pt", "dino"), ("L14", "dino_vitl14.pt", "dino_vitl14"))
G = {}


def dev_name():
    return "dev2" if C.WORK.name == "dev2" else "dev1"


def load_ev():
    code = ROOT / "scripts" / "iter3_eval.py"
    assert hashlib.sha256(code.read_bytes()).hexdigest() == FROZEN_SHA, "iter3_eval.py changed"
    spec = importlib.util.spec_from_file_location("iter3_eval", code)
    ev = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ev)
    return ev


def shot_table(draw, id_classes, samples):
    """support ids (n_id * 12, class-major) and calibration ids (n_id * 4, class-major) of the draw."""
    order = {c["idx_1k"]: c["id_idx"] for c in id_classes}
    if draw == "orig":
        s = samples[samples.split.isin(["support", "calib"])].copy()
        s["id_idx"] = s.class_idx_id.astype(int)
        s["pos"] = s.groupby(["split", "id_idx"]).cumcount()
        sup = s[s.split == "support"].sort_values(["id_idx", "pos"])
        cal = s[s.split == "calib"].sort_values(["id_idx", "pos"])
        return sup.sample_id.values, cal.sample_id.values, "features"
    d = pd.read_parquet(r5.R5 / "shots" / "draws.parquet")
    d = d[(d.draw == int(draw)) & d.idx_1k.isin(order)].copy()
    d["id_idx"] = d.idx_1k.map(order)
    sup = d[d.role == "support"].sort_values(["id_idx", "pos"])
    cal = d[d.role == "calib"].sort_values(["id_idx", "pos"])
    return sup.sample_id.values, cal.sample_id.values, "r5"


def build(draw, ev, baselines=True):
    samples = pd.read_parquet(C.SPLITS_DIR / "samples.parquet")
    dview = pd.read_parquet(C.RUNS_DIR / "dview" / "dview.parquet").set_index("sample_id")
    id_classes = json.loads((C.SPLITS_DIR / "id_classes.json").read_text())
    n_id = len(id_classes)
    sup_ids, cal_ids, src = shot_table(draw, id_classes, samples)
    assert len(sup_ids) == 12 * n_id and len(cal_ids) == 4 * n_id
    stream_ids = samples.sample_id.values[samples.split.isin(["id_dev", "near_dev", "far_dev"]).values]
    # candidate sets K(x): CLIP zero-shot top-5 over the ID classes (TINS positive text features)
    k_stream = np.array(dview.loc[stream_ids].K_id.tolist())
    if src == "features":
        k_cal = np.array(dview.loc[cal_ids].K_id.tolist())
    else:
        pos_text = torch.load(C.RUNS_DIR / "setup" / "tins_setup.pt", map_location="cpu")["positive_features"].float()
        clip_f, blob = load_features(r5.R5 / "features" / "shots.clip.pt")
        pos = {s: i for i, s in enumerate(blob["sample_id"])}
        k_cal = (clip_f[[pos[s] for s in cal_ids]].float() @ pos_text.T).topk(C.K_TOP, dim=1).indices.numpy()
    cand = np.concatenate([k_cal, k_stream])
    cal_cls = np.repeat(np.arange(n_id), 4)
    views = {}
    for name, dev_file, shot_file in VIEWS:
        dev_f, blob = load_features(C.FEATURES_DIR / dev_file)
        dpos = {s: i for i, s in enumerate(blob["sample_id"])}
        if src == "features":
            sup_f = dev_f[[dpos[s] for s in sup_ids]]
            cal_f = dev_f[[dpos[s] for s in cal_ids]]
        else:
            sh_f, sblob = load_features(r5.R5 / "features" / f"shots.{shot_file}.pt")
            spos = {s: i for i, s in enumerate(sblob["sample_id"])}
            sup_f = sh_f[[spos[s] for s in sup_ids]]
            cal_f = sh_f[[spos[s] for s in cal_ids]]
        st_f = dev_f[[dpos[s] for s in stream_ids]]
        support = sup_f.numpy().astype(np.float32).reshape(n_id, 12, -1)
        q = np.concatenate([cal_f.numpy(), st_f.numpy()]).astype(np.float32)
        is_cal = np.zeros(len(q), dtype=bool)
        is_cal[:len(cal_ids)] = True
        v = ev.custom_view(support, q, cand, is_cal, cal_cls, m=2, proto=True)
        v["support_arr"] = v["support"]
        v["q"] = q
        if baselines:
            kd = r5.knn_distance(support.reshape(-1, support.shape[-1]), q, KS)
            v["knn"] = {k: r5.pval_high(kd[k][is_cal], kd[k]) for k in KS}
            md = r5.maha_pp(support, q, LAMS)
            v["maha"] = {lam: r5.pval_high(md[lam][is_cal], md[lam]) for lam in LAMS}
        views[name] = v
    row = {s: len(cal_ids) + i for i, s in enumerate(stream_ids)}
    return views, row


def tins_run(draw, stream, seed):
    f = r5.R5 / dev_name() / "tins" / f"draw{draw}" / f"{stream}_seed{seed}.npz"
    z = np.load(f, allow_pickle=True)
    return z["sample_id"], z["is_ood"].astype(bool), z["S_final"].astype(np.float64), z["batch_index"]


def measures(id_s, ood_s):
    from vins.metrics import measures as upstream

    m = upstream(id_s, ood_s)
    return {"AUROC": 100 * m["AUROC"], "FPR95": 100 * m["FPR95"]}


def entrance_stats(mask, is_ood):
    n_adm = int(mask.sum())
    return {"id_rate": float(mask[~is_ood].mean()), "ood_rate": float(mask[is_ood].mean()),
            "purity": float(is_ood[mask].mean()) if n_adm else float("nan"), "size": n_adm}


def run_task(task):
    stream, seed = task
    ev, draw, check = G["ev"], G["draw"], G["check"]
    sid, is_ood, S, bidx = tins_run(draw, stream, seed)
    q = np.array([G["row"][s] for s in sid])
    arr, times, ent = {"S": S}, {}, {}
    for name, v in G["views"].items():
        sf, d, p_all = v["q"][q], v["d"][q], v["p_all"][q]
        med, mad = v["stats"]["med_all"], v["stats"]["mad_all"]
        t0 = time.time()
        m3 = r5.entrance(sf, d, p_all, bidx, v["cal_feats"], v["d_cal"], med, mad, r5.V4_ENTRANCE)
        pt3, g3 = r5.memory_p(sf, d, bidx, m3[-1], v["cal_feats"], v["d_cal"], med, mad)
        m1 = r5.entrance(sf, d, p_all, bidx, v["cal_feats"], v["d_cal"], med, mad, (0.10,))
        pt1, _ = r5.memory_p(sf, d, bidx, m1[-1], v["cal_feats"], v["d_cal"], med, mad)
        t1 = time.time()
        lp = r5.lp_run(v["support_arr"], v["cal_feats"], sf, bidx, **r5.V4_LP, init="warm", within_batch=True)
        t2 = time.time()
        lpi = r5.lp_run(v["support_arr"], v["cal_feats"], sf, bidx, **r5.V4_LP, init="warm", within_batch=False)
        t3 = time.time()
        if check:           # the R5 re-implementation must equal the frozen v4 functions on real data
            assert np.array_equal(m3[-1], ev.admit_mask(v, sf, d, p_all, bidx, "cand40_30_10")), "entrance != v4"
            assert np.array_equal(pt3, ev.p_memory(v, sf, d, bidx, m3[-1])[0]), "memory != v4"
            assert np.array_equal(lp["p"], ev.lp_pvalues(v, sf, bidx, 10, 0.9, 3.0, 15)), "LP != v4"
        arr.update({f"p_{name}": v["p"][q], f"pall_{name}": p_all, f"pt3_{name}": pt3, f"pt1_{name}": pt1,
                    f"plp_{name}": lp["p"], f"plpi_{name}": lpi["p"], f"d_{name}": d, f"g3_{name}": g3,
                    f"u_{name}": lp["u"], f"ui_{name}": lpi["u"]})
        for k in KS:
            arr[f"knn{k}_{name}"] = v["knn"][k][q]
        for lam in LAMS:
            arr[f"maha{lam:g}_{name}"] = v["maha"][lam][q]
        ent[name] = {"v4_stage1": entrance_stats(m3[0], is_ood), "v4_stage2": entrance_stats(m3[1], is_ood),
                     "v4_M": entrance_stats(m3[2], is_ood), "single_M": entrance_stats(m1[0], is_ood)}
        times[name] = {"memory_s": t1 - t0, "lp_s": t2 - t1, "lp_ind_s": t3 - t2}
    P = lambda k: arr[f"{k}_B14"] * arr[f"{k}_L14"]  # noqa: E731
    vis = {"static": P("p"), "mem": P("pt3"), "mem1": P("pt1"), "lp": P("plp"), "lpi": P("plpi"),
           "full": P("pt3") * P("plp"), "single_lp": P("pt1") * P("plp"), "full_i": P("pt3") * P("plpi")}
    for k in KS:
        vis[f"knn{k}"] = P(f"knn{k}")
    for lam in LAMS:
        vis[f"maha{lam:g}"] = P(f"maha{lam:g}")
    scores = {"tins": S}
    for key, val in vis.items():
        scores[f"v_{key}"] = val
        scores[f"s_{key}"] = S * val
    for name in ("B14", "L14"):          # single-view raw vs conformal (decomposition of the conformal step)
        scores[f"{name}_d"] = -arr[f"d_{name}"]
        scores[f"{name}_p"] = arr[f"p_{name}"]
        scores[f"{name}_g3"] = -arr[f"g3_{name}"]
        scores[f"{name}_pt3"] = arr[f"pt3_{name}"]
        scores[f"{name}_u"] = arr[f"u_{name}"]
        scores[f"{name}_plp"] = arr[f"plp_{name}"]
    scores["raw_u_prod"] = arr["u_B14"] * arr["u_L14"]
    scores["raw_g_sum"] = -(arr["g3_B14"] + arr["g3_L14"])
    metrics = {}
    for key, s in scores.items():
        m = measures(s[~is_ood], s[is_ood])
        m["AUROC_within_batch"] = 100 * r5.within_batch_auroc(s, is_ood, bidx)
        metrics[key] = m
    out = r5.R5 / dev_name() / "eval" / f"draw{draw}"
    np.savez_compressed(out / f"{stream}_seed{seed}.npz", sample_id=sid, is_ood=is_ood, batch_index=bidx,
                        **{k: np.asarray(v, dtype=np.float32) for k, v in arr.items()})
    (out / f"{stream}_seed{seed}.json").write_text(json.dumps({"metrics": metrics, "entrance": ent, "times": times,
                                                               "n": int(len(sid)), "n_ood": int(is_ood.sum())}) + "\n")
    return stream, seed, times


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--draw", required=True)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--check-v4", action="store_true")
    opts = parser.parse_args()
    start = time.time()
    ev = load_ev()
    from vins.tins_dev import import_tins

    import_tins()                                   # upstream utils (get_measures) become importable
    views, row = build(opts.draw, ev)
    (r5.R5 / dev_name() / "eval" / f"draw{opts.draw}").mkdir(parents=True, exist_ok=True)
    G.update(ev=ev, views=views, row=row, draw=opts.draw, check=opts.check_v4)
    tasks = [(s, sd) for s in ("near", "far") for sd in C.ORDER_SEEDS]
    with mp.get_context("fork").Pool(opts.workers) as pool:
        for stream, seed, times in pool.imap_unordered(run_task, tasks):
            print(json.dumps({"done": f"{dev_name()}/draw{opts.draw}/{stream}_seed{seed}", "times": times}), flush=True)
    print(json.dumps({"draw": opts.draw, "dev": dev_name(), "seconds": round(time.time() - start, 1)}))


if __name__ == "__main__":
    main()
