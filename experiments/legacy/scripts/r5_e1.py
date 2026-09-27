"""R5 Phase 1 / E1: is the gain due to the extra DINOv2 features? (dev only, same 15 conditions as r5_eval)

Per dev split x shot draw x stream x order seed (TINS scores from r5_tins.py):
  REPRISE-CLIP : the frozen v4 pipeline with a single view = CLIP ViT-B/16 image features (the features TINS itself
                 uses; no extra backbone): static p, v4 entrance + memory p_t, conformal LP p_LP
  REPRISE-L14  : the same with the single DINOv2 L/14 view (single-view reference)
  AdaNeg-type  : visual negative-proxy memory without conformal control (L/14): s_id(x) = max_c cos(f, mu_c);
                 a test image joins the negative memory when s_id < tau (fixed threshold, tuned on dev1);
                 score = s_id(x) - lam * mean top-5 cos to the negative memory (memory of earlier batches)
  OODD-type    : dynamic dictionary without conformal control (L/14): the K test features with the lowest s_id seen
                 so far (priority queue, updated after each batch); score = s_id(x) - mean top-5 cos to the dictionary
  For x TINS, every score is turned into a per-batch conformal p against the calibration shots scored under the same
  memory (the same combination tool as REPRISE), then multiplied with S_final.
Grids (tuned on dev1 by mean near FPR95 of x TINS): tau in {0.30, 0.35, 0.40, 0.45, 0.50}, lam in {0.5, 1, 2};
K in {256, 1024, 4096}.
Decision rule of the roadmap: keep the method paper if REPRISE-CLIP retains at least half of the 2-view REPRISE gain
over TINS (near FPR95 reduction), otherwise switch to an analysis paper.
Output: <R5>/<dev>/e1/draw<k>/<stream>_seed<s>.json
"""
import argparse
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
from vins.memory import TopM, pvalues_sorted, top_sims  # noqa: E402

TAUS = (0.30, 0.35, 0.40, 0.45, 0.50)
LAMS = (0.5, 1.0, 2.0)
KS = (256, 1024, 4096)
G = {}


def load_mod(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def build(draw):
    """Views for CLIP (single) and L/14 (single), queries = calibration + dev stream samples."""
    E = load_mod("r5_eval")
    ev = E.load_ev()
    samples = pd.read_parquet(C.SPLITS_DIR / "samples.parquet")
    dview = pd.read_parquet(C.RUNS_DIR / "dview" / "dview.parquet").set_index("sample_id")
    id_classes = json.loads((C.SPLITS_DIR / "id_classes.json").read_text())
    n_id = len(id_classes)
    sup_ids, cal_ids, src = E.shot_table(draw, id_classes, samples)
    stream_ids = samples.sample_id.values[samples.split.isin(["id_dev", "near_dev", "far_dev"]).values]
    k_stream = np.array(dview.loc[stream_ids].K_id.tolist())
    if src == "features":
        k_cal = np.array(dview.loc[cal_ids].K_id.tolist())
    else:
        pos_text = torch.load(C.RUNS_DIR / "setup" / "tins_setup.pt", map_location="cpu")["positive_features"].float()
        clip_f, blob = load_features(r5.R5 / "features" / "shots.clip.pt")
        pos = {s: i for i, s in enumerate(blob["sample_id"])}
        k_cal = (clip_f[[pos[s] for s in cal_ids]].float() @ pos_text.T).topk(C.K_TOP, dim=1).indices.numpy()
    cand = np.concatenate([k_cal, k_stream])
    views = {}
    for name, dev_file, shot_file in (("CLIP", "clip.pt", "clip"), ("L14", "dino_vitl14.pt", "dino_vitl14")):
        dev_f, blob = load_features(C.FEATURES_DIR / dev_file)
        dpos = {s: i for i, s in enumerate(blob["sample_id"])}
        if src == "features":
            sup_f, cal_f = dev_f[[dpos[s] for s in sup_ids]], dev_f[[dpos[s] for s in cal_ids]]
        else:
            sh_f, sblob = load_features(r5.R5 / "features" / f"shots.{shot_file}.pt")
            spos = {s: i for i, s in enumerate(sblob["sample_id"])}
            sup_f, cal_f = sh_f[[spos[s] for s in sup_ids]], sh_f[[spos[s] for s in cal_ids]]
        st_f = dev_f[[dpos[s] for s in stream_ids]]
        support = sup_f.float().numpy().astype(np.float32).reshape(n_id, 12, -1)
        support /= np.linalg.norm(support, axis=2, keepdims=True)
        q = np.concatenate([cal_f.float().numpy(), st_f.float().numpy()]).astype(np.float32)
        q /= np.linalg.norm(q, axis=1, keepdims=True)
        is_cal = np.zeros(len(q), dtype=bool)
        is_cal[:len(cal_ids)] = True
        v = ev.custom_view(support, q, cand, is_cal, np.repeat(np.arange(n_id), 4), m=2, proto=True)
        v["support_arr"], v["q"] = v["support"], q
        mu = support.sum(axis=1)
        v["mu"] = (mu / np.linalg.norm(mu, axis=1, keepdims=True)).astype(np.float32)
        views[name] = v
    return views, {s: len(cal_ids) + i for i, s in enumerate(stream_ids)}


def memory_score_p(s_id_stream, s_id_cal, sf, cal, bidx, admit_fn, lam=1.0, k=5, cap=None):
    """Raw memory score s = s_id - lam * mean top-k cos to the memory (earlier batches) and its per-batch conformal p
    against the calibration shots scored under the same memory. admit_fn(rows, s_id) -> boolean mask of the batch
    rows that join the memory (AdaNeg-type) or None for the OODD priority queue with capacity cap."""
    sf = np.ascontiguousarray(sf, dtype=np.float32)
    mem = np.zeros((0, sf.shape[1]), dtype=np.float32)
    mem_key = np.zeros(0)
    score = np.empty(len(sf))
    p = np.empty(len(sf))
    cal_top = TopM(cal, k) if cap is None else None           # the AdaNeg-type memory only grows
    for b in np.unique(bidx):
        rows = np.flatnonzero(bidx == b)
        if len(mem):
            kk = min(k, len(mem))
            ms = top_sims(sf[rows], mem, kk).mean(axis=1)
            if cal_top is not None:
                mc = cal_top.top[:, :kk].astype(np.float64).mean(axis=1)
            else:
                mc = top_sims(cal, mem, kk).mean(axis=1)
        else:
            ms, mc = np.zeros(len(rows)), np.zeros(len(cal))
        s = s_id_stream[rows] - lam * ms
        sc = np.sort(-(s_id_cal - lam * mc))                        # OOD direction for pvalues_sorted
        score[rows] = s
        p[rows] = pvalues_sorted(sc, -s)
        if cap is None:
            new = rows[admit_fn(s_id_stream[rows])]
            mem = np.concatenate([mem, sf[new]])
            cal_top.add(sf[new])
        else:                                                       # keep the cap lowest-s_id features seen so far
            mem = np.concatenate([mem, sf[rows]])
            mem_key = np.concatenate([mem_key, s_id_stream[rows]])
            if len(mem) > cap:
                keep = np.argpartition(mem_key, cap - 1)[:cap]
                mem, mem_key = mem[keep], mem_key[keep]
    return score, p


def task(job):
    stream, seed = job
    from vins.metrics import measures as upstream

    dev, draw = G["dev"], G["draw"]
    z = np.load(r5.R5 / dev / "tins" / f"draw{draw}" / f"{stream}_seed{seed}.npz", allow_pickle=True)
    sid, is_ood, S, bidx = z["sample_id"], z["is_ood"].astype(bool), z["S_final"].astype(np.float64), z["batch_index"]
    q = np.array([G["row"][s] for s in sid])
    sc = {"tins": S}
    for name, v in G["views"].items():
        sf, d, p_all = v["q"][q], v["d"][q], v["p_all"][q]
        med, mad = v["stats"]["med_all"], v["stats"]["mad_all"]
        if not G["baselines_only"]:
            M = r5.entrance(sf, d, p_all, bidx, v["cal_feats"], v["d_cal"], med, mad, r5.V4_ENTRANCE)[-1]
            pt, _ = r5.memory_p(sf, d, bidx, M, v["cal_feats"], v["d_cal"], med, mad)
            plp = r5.lp_run(v["support_arr"], v["cal_feats"], sf, bidx, **r5.V4_LP)["p"]
            for key, val in (("static", v["p"][q]), ("mem", pt), ("lp", plp), ("full", pt * plp)):
                sc[f"v_{key}_{name}"], sc[f"s_{key}_{name}"] = val, S * val
        if name == "L14":
            s_id = (sf @ v["mu"].T).max(axis=1)
            s_id_cal = (v["cal_feats"] @ v["mu"].T).max(axis=1)
            for tau in G["taus"]:
                for lam in G["lams"]:
                    raw, p = memory_score_p(s_id, s_id_cal, sf, v["cal_feats"], bidx, lambda s, t=tau: s < t, lam=lam)
                    sc[f"v_adaneg_t{tau:g}_l{lam:g}"], sc[f"s_adaneg_t{tau:g}_l{lam:g}"] = raw, S * p
            for cap in G["ks"]:
                raw, p = memory_score_p(s_id, s_id_cal, sf, v["cal_feats"], bidx, None, lam=1.0, cap=cap)
                sc[f"v_oodd_K{cap}"], sc[f"s_oodd_K{cap}"] = raw, S * p
            sc["v_proto_L14"] = s_id
    metrics = {}
    for k, s in sc.items():
        m = upstream(s[~is_ood], s[is_ood])
        metrics[k] = {"AUROC": 100 * m["AUROC"], "FPR95": 100 * m["FPR95"]}
    out = r5.R5 / dev / G["sub"] / f"draw{draw}"
    (out / f"{stream}_seed{seed}.json").write_text(json.dumps({"metrics": metrics}) + "\n")
    return job


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--draw", required=True)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--taus", type=float, nargs="+", default=list(TAUS))
    parser.add_argument("--lams", type=float, nargs="+", default=list(LAMS))
    parser.add_argument("--ks", type=int, nargs="+", default=list(KS))
    parser.add_argument("--baselines-only", action="store_true", help="grid extension: skip the REPRISE variants")
    parser.add_argument("--sub", default="e1", help="output sub-directory under <R5>/<dev>/")
    opts = parser.parse_args()
    start = time.time()
    from vins.tins_dev import import_tins

    import_tins()
    dev = "dev2" if C.WORK.name == "dev2" else "dev1"
    views, row = build(opts.draw)
    (r5.R5 / dev / opts.sub / f"draw{opts.draw}").mkdir(parents=True, exist_ok=True)
    G.update(views=views, row=row, dev=dev, draw=opts.draw, taus=opts.taus, lams=opts.lams, ks=opts.ks,
             baselines_only=opts.baselines_only, sub=opts.sub)
    jobs = [(s, sd) for s in ("near", "far") for sd in C.ORDER_SEEDS]
    with mp.get_context("fork").Pool(opts.workers) as pool:
        for j in pool.imap_unordered(task, jobs):
            print(json.dumps({"done": j}), flush=True)
    print(json.dumps({"seconds": round(time.time() - start, 1)}))


if __name__ == "__main__":
    main()
