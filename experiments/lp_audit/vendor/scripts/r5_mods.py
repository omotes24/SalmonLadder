"""R5 Phase 2: candidate modifications (M1, M3), decompositions (E5) and sensitivity, on the 15 matched conditions.

--what m1m3e5 (dev1 designs, dev2 confirms):
  M2a  stage-specific calibration: shot j of each class calibrates stage j (p_all, p_A1, p_A2, p_t); control:
       shot 0 reused at every stage (same count)
  M1   entrance by per-batch Benjamini-Hochberg: every stage at level q (bh_all_q) or only the final stage with the
       v4 candidate thresholds (bh_last_q), q in {0.05, 0.1, 0.2}; x TINS with and without LP
  M3   conformalised TINS score p_T (per batch, against the calibration shots scored with the same negatives) combined
       with the four visual p-values by product (Fisher order), Cauchy combination, harmonic mean p, and "within-view
       first" (Fisher chi-square p per view, then product with p_T)
  M4   (added after (1)) Mahalanobis++ static p of both views (shrinkage 0.9, chosen on dev1) as an extra factor
       of v4 (and of LP only)
  E5   no K(x) (d_all everywhere); LP graph keeping only ID / only OOD / all stream nodes (label oracle, others
       inserted only when scored); memory = M with false admissions removed (oracle purity) / all OOD arrivals
--what sens (dev1): one factor at a time around v4: entrance thresholds, n0, |K(x)|, k_g, gamma, lambda, sweeps,
       memory window (last W admitted), graph window (stream nodes of the last W batches); timing per image.
Oracle variants use labels on purpose and are diagnostics, never candidate methods.
Output: <R5>/<dev>/mods/<what>/draw<k>/<stream>_seed<s>.json
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
from scipy.stats import chi2  # noqa: E402

from vins import config as C  # noqa: E402
from vins import r5  # noqa: E402

G = {}


def load_mod(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def measures(id_s, ood_s):
    from vins.metrics import measures as upstream

    m = upstream(id_s, ood_s)
    return {"AUROC": 100 * m["AUROC"], "FPR95": 100 * m["FPR95"]}


def ent_stats(M, is_ood):
    return {"id_rate": float(M[~is_ood].mean()), "ood_rate": float(M[is_ood].mean()),
            "purity": float(is_ood[M].mean()) if M.any() else float("nan")}


def memory_window_p(sf, d, bidx, admit, cal_feats, d_cal, med, mad, window, m=2):
    """memory p with only the last `window` admitted samples in the memory (sliding window)."""
    sf = np.ascontiguousarray(sf, dtype=np.float32)
    mem = np.zeros((0, sf.shape[1]), dtype=np.float32)
    p = np.empty(len(sf))
    for b in np.unique(bidx):
        rows = np.flatnonzero(bidx == b)
        top_s = r5.top_sims(sf[rows], mem, m)
        top_c = r5.top_sims(cal_feats, mem, m)
        r_s = 1.0 - top_s[:, m - 1] if top_s.shape[1] >= m else np.full(len(rows), r5.EMPTY)
        r_c = 1.0 - top_c[:, m - 1] if top_c.shape[1] >= m else np.full(len(cal_feats), r5.EMPTY)
        g = d[rows] - (r_s - med) / mad
        gc = np.sort(d_cal - (r_c - med) / mad)
        p[rows] = r5.pvalues_sorted(gc, g)
        mem = np.concatenate([mem, sf[rows[admit[rows]]]])[-window:]
    return p


def lp_window(support, cal, sf, bidx, window_batches, **kw):
    """LP whose graph keeps only the stream nodes of the last `window_batches` batches (recomputed per window)."""
    p = np.empty(len(sf))
    batches = np.unique(bidx)
    for i, b in enumerate(batches):
        lo = batches[max(0, i - window_batches + 1)]
        sel = (bidx >= lo) & (bidx <= b)
        rows = np.flatnonzero(sel)
        out = r5.lp_run(support, cal, sf[rows], bidx[rows], **kw)["p"]
        cur = bidx[rows] == b
        p[rows[cur]] = out[cur]
    return p


def task(job):
    stream, seed = job
    dev, draw, what = G["dev"], G["draw"], G["what"]
    z = np.load(r5.R5 / dev / "tins" / f"draw{draw}" / f"{stream}_seed{seed}.npz", allow_pickle=True)
    e = np.load(r5.R5 / dev / "eval" / f"draw{draw}" / f"{stream}_seed{seed}.npz", allow_pickle=True)
    sid, is_ood, S, bidx = z["sample_id"], z["is_ood"].astype(bool), z["S_final"].astype(np.float64), z["batch_index"]
    assert (e["sample_id"] == sid).all()
    q = np.array([G["row"][s] for s in sid])
    E = {k: e[k].astype(np.float64) for k in e.files if k not in ("sample_id", "is_ood", "batch_index")}
    lp = E["plp_B14"] * E["plp_L14"]
    sc, st, times = {"tins": S, "v4_full": S * E["pt3_B14"] * E["pt3_L14"] * lp}, {}, {}
    if what == "m1m3e5":
        variants = {f"bh_all_{q_:g}": dict(qs=(q_, q_, q_), fixed=(None, None, None)) for q_ in (0.05, 0.1, 0.2)}
        variants.update({f"bh_last_{q_:g}": dict(qs=(q_, q_, q_), fixed=(0.4, 0.3, None)) for q_ in (0.05, 0.1, 0.2)})
        pts = {k: [] for k in variants}
        m2a = {"split1": [], "reuse1": []}
        pt_noK, pt_pure, pt_allood = [], [], []
        lp_or = {"id": [], "ood": []}
        for name, v in G["views"].items():
            sf, d, p_all = v["q"][q], v["d"][q], v["p_all"][q]
            med, mad = v["stats"]["med_all"], v["stats"]["mad_all"]
            for k, kw in variants.items():
                masks = r5.entrance_bh(sf, d, p_all, bidx, v["cal_feats"], v["d_cal"], med, mad, **kw)
                pt, _ = r5.memory_p(sf, d, bidx, masks[-1], v["cal_feats"], v["d_cal"], med, mad)
                pts[k].append(pt)
                st[f"{name}|{k}"] = ent_stats(masks[-1], is_ood)
            # M2a: stage-specific calibration (shot j of every class at stage j; p_t with shot 3) and the
            # same-count control (shot 0 reused at every stage); the LP keeps the 4 shots
            n_id = len(v["d_cal"]) // 4
            cset = lambda j: (v["cal_feats"][np.arange(n_id) * 4 + j], v["d_cal"][np.arange(n_id) * 4 + j],  # noqa: E731
                              v["d_all"][np.arange(n_id) * 4 + j])
            for tag, sets in (("split1", [cset(0), cset(1), cset(2), cset(3)]), ("reuse1", [cset(0)] * 4)):
                masks, _, ptx = r5.entrance_cal(sf, d, v["d_all"][q], bidx, sets, med, mad)
                m2a[tag].append(ptx)
                st[f"{name}|m2a_{tag}"] = ent_stats(masks[-1], is_ood)
            # E5: no K(x): d_all replaces d in the static view and the memory score
            da, dac = v["d_all"][q], v["d_all"][:len(v["d_cal"])]
            Mn = r5.entrance(sf, da, p_all, bidx, v["cal_feats"], dac, med, mad)[-1]
            ptn, _ = r5.memory_p(sf, da, bidx, Mn, v["cal_feats"], dac, med, mad)
            pt_noK.append((ptn, r5.pval_high(dac, da)))
            M = r5.entrance(sf, d, p_all, bidx, v["cal_feats"], v["d_cal"], med, mad)[-1]
            ptp, _ = r5.memory_p(sf, d, bidx, M & is_ood, v["cal_feats"], v["d_cal"], med, mad)
            pta, _ = r5.memory_p(sf, d, bidx, is_ood.copy(), v["cal_feats"], v["d_cal"], med, mad)
            pt_pure.append(ptp)
            pt_allood.append(pta)
            for tag, mask in (("id", ~is_ood), ("ood", is_ood)):
                lp_or[tag].append(r5.lp_oracle(v["support_arr"], v["cal_feats"], sf, bidx, mask, **r5.V4_LP))
        for k in variants:
            ptk = pts[k][0] * pts[k][1]
            sc[f"m1_{k}_mem"], sc[f"m1_{k}_full"] = S * ptk, S * ptk * lp
        for tag in m2a:
            sc[f"m2a_{tag}_mem"] = S * m2a[tag][0] * m2a[tag][1]
            sc[f"m2a_{tag}_full"] = S * m2a[tag][0] * m2a[tag][1] * lp
        sc["e5_noK_full"] = S * pt_noK[0][0] * pt_noK[1][0] * lp
        sc["e5_noK_static"] = S * pt_noK[0][1] * pt_noK[1][1]
        sc["e5_mem_pure_full"] = S * pt_pure[0] * pt_pure[1] * lp
        sc["e5_mem_allood_full"] = S * pt_allood[0] * pt_allood[1] * lp
        for tag in ("id", "ood"):
            lpo = lp_or[tag][0] * lp_or[tag][1]
            sc[f"e5_lp_keep_{tag}"] = S * lpo
            sc[f"e5_lp_keep_{tag}_full"] = S * E["pt3_B14"] * E["pt3_L14"] * lpo
        # M3: conformalised TINS score and combination rules
        cal_s = z["cal_scores"].astype(np.float64)                   # (n_batches, n_cal)
        pT = np.empty(len(S))
        for bi, b in enumerate(np.unique(bidx)):
            rows = np.flatnonzero(bidx == b)
            pT[rows] = r5.pval_low(cal_s[bi], S[rows])
        vis = [E["pt3_B14"], E["pt3_L14"], E["plp_B14"], E["plp_L14"]]
        for rule in ("product", "cauchy", "hmp"):
            sc[f"m3_{rule}"] = r5.combine_p([pT] + vis, rule)
            sc[f"m3_vis_{rule}"] = r5.combine_p(vis, rule)
        per_view = [chi2.sf(-2 * np.log(np.clip(E[f"pt3_{n}"] * E[f"plp_{n}"], 1e-300, 1)), 4) for n in ("B14", "L14")]
        sc["m3_withinview_product"] = r5.combine_p([pT] + per_view, "product")
        sc["m3_withinview_cauchy"] = r5.combine_p([pT] + per_view, "cauchy")
        sc["m3_pT_only"] = pT
        # M4: the same-feature static baseline that was strongest on far OOD in (1), as an extra factor
        pm = G["maha"]["B14"][q] * G["maha"]["L14"][q]
        sc["m4_full_x_maha"] = S * E["pt3_B14"] * E["pt3_L14"] * lp * pm
        sc["m4_lp_x_maha"] = S * lp * pm
        sc["m4_maha_only"] = S * pm
        st["pT_pr_le_0.1_ID"] = float((pT[~is_ood] <= 0.1).mean())
    else:  # sensitivity (one factor at a time around v4)
        G["acc"] = {}
        for name, v in G["views"].items():
            sf, d, p_all = v["q"][q], v["d"][q], v["p_all"][q]
            med, mad = v["stats"]["med_all"], v["stats"]["mad_all"]
            base_M = r5.entrance(sf, d, p_all, bidx, v["cal_feats"], v["d_cal"], med, mad)[-1]
            for th in ((0.3, 0.3, 0.1), (0.5, 0.3, 0.1), (0.4, 0.2, 0.1), (0.4, 0.4, 0.1), (0.4, 0.3, 0.05), (0.4, 0.3, 0.2)):
                M = r5.entrance(sf, d, p_all, bidx, v["cal_feats"], v["d_cal"], med, mad, th)[-1]
                pt, _ = r5.memory_p(sf, d, bidx, M, v["cal_feats"], v["d_cal"], med, mad)
                G["acc"].setdefault(f"eps{th}", []).append(pt)
                st[f"{name}|eps{th}"] = ent_stats(M, is_ood)
            for w in (500, 2000):
                pt = memory_window_p(sf, d, bidx, base_M, v["cal_feats"], v["d_cal"], med, mad, w)
                G["acc"].setdefault(f"memwin{w}", []).append(pt)
            for key, vv in G["alt_views"][name].items():                 # n0 and |K(x)| variants
                sfa, da, pa = vv["q"][q], vv["d"][q], vv["p_all"][q]
                M = r5.entrance(sfa, da, pa, bidx, vv["cal_feats"], vv["d_cal"], vv["stats"]["med_all"],
                                vv["stats"]["mad_all"])[-1]
                pt, _ = r5.memory_p(sfa, da, bidx, M, vv["cal_feats"], vv["d_cal"], vv["stats"]["med_all"],
                                    vv["stats"]["mad_all"])
                G["acc"].setdefault(f"{key}_mem", []).append(pt)
                G["acc"].setdefault(f"{key}_static", []).append(vv["p"][q])
            for key, kw in (("kg5", dict(k=5)), ("kg20", dict(k=20)), ("gamma1", dict(gamma=1.0)),
                            ("gamma5", dict(gamma=5.0)), ("lam0.8", dict(alpha=0.8)), ("lam0.95", dict(alpha=0.95)),
                            ("sweeps5", dict(iters=5)), ("sweeps30", dict(iters=30))):
                args = dict(r5.V4_LP)
                args.update(kw)
                tick = time.time()
                G["acc"].setdefault(key, []).append(r5.lp_run(v["support_arr"], v["cal_feats"], sf, bidx, **args)["p"])
                times[f"{name}|{key}_ms_per_image"] = 1000 * (time.time() - tick) / len(sf)
            tick = time.time()
            G["acc"].setdefault("lpwin20", []).append(lp_window(v["support_arr"], v["cal_feats"], sf, bidx, 20, **r5.V4_LP))
            times[f"{name}|lpwin20_ms_per_image"] = 1000 * (time.time() - tick) / len(sf)
        acc = G.pop("acc")
        for key, (pb, pl) in acc.items():
            if key.startswith("eps") or key.startswith("memwin") or key.endswith("_mem"):
                sc[f"sens_{key}_full"] = S * pb * pl * lp
            elif key.endswith("_static"):
                sc[f"sens_{key}"] = S * pb * pl
            else:
                sc[f"sens_{key}_full"] = S * E["pt3_B14"] * E["pt3_L14"] * pb * pl
    metrics = {k: measures(s[~is_ood], s[is_ood]) for k, s in sc.items()}
    out = r5.R5 / dev / "mods" / what / f"draw{draw}"
    (out / f"{stream}_seed{seed}.json").write_text(json.dumps({"metrics": metrics, "stats": st, "times": times}) + "\n")
    return job


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--what", choices=["m1m3e5", "sens"], required=True)
    parser.add_argument("--draw", required=True)
    parser.add_argument("--workers", type=int, default=6)
    opts = parser.parse_args()
    start = time.time()
    from vins.tins_dev import import_tins

    import_tins()
    E = load_mod("r5_eval")
    dev = "dev2" if C.WORK.name == "dev2" else "dev1"
    views, row = E.build(opts.draw, E.load_ev(), baselines=False)
    alt = {}
    if opts.what == "sens":
        import pandas as pd
        import torch

        samples = pd.read_parquet(C.SPLITS_DIR / "samples.parquet")
        dview = pd.read_parquet(C.RUNS_DIR / "dview" / "dview.parquet").set_index("sample_id")
        id_classes = json.loads((C.SPLITS_DIR / "id_classes.json").read_text())
        sup_ids, cal_ids, src = E.shot_table(opts.draw, id_classes, samples)
        stream_ids = samples.sample_id.values[samples.split.isin(["id_dev", "near_dev", "far_dev"]).values]
        clip_dev, blob = __import__("vins.features", fromlist=["load_features"]).load_features(C.FEATURES_DIR / "clip.pt")
        dpos = {s: i for i, s in enumerate(blob["sample_id"])}
        pos_text = torch.load(C.RUNS_DIR / "setup" / "tins_setup.pt", map_location="cpu")["positive_features"].float()
        if src == "features":
            cal_clip = clip_dev[[dpos[s] for s in cal_ids]]
        else:
            sc_, sb = __import__("vins.features", fromlist=["load_features"]).load_features(r5.R5 / "features" / "shots.clip.pt")
            sp = {s: i for i, s in enumerate(sb["sample_id"])}
            cal_clip = sc_[[sp[s] for s in cal_ids]]
        sims = torch.cat([cal_clip.float(), clip_dev[[dpos[s] for s in stream_ids]].float()]) @ pos_text.T
        for name, v in views.items():
            q_all = v["q"]
            is_cal = np.zeros(len(q_all), bool)
            is_cal[:len(v["d_cal"])] = True
            alt[name] = {}
            for n0 in (6, 24):
                alt[name][f"n0_{n0}"] = r5.proto_view(v["support_arr"], q_all, sims.topk(5, dim=1).indices.numpy(), is_cal, n0=n0)
            for kk in (3, 10):
                alt[name][f"K{kk}"] = r5.proto_view(v["support_arr"], q_all, sims.topk(kk, dim=1).indices.numpy(), is_cal, n0=12)
            for vv in alt[name].values():
                vv["q"] = q_all
    maha = {}
    if opts.what == "m1m3e5":        # M4: Mahalanobis++ static p per view (shrinkage 0.9, the dev1 choice in (1))
        for name, v in views.items():
            md = r5.maha_pp(v["support_arr"], v["q"], (0.9,))[0.9]
            nc = len(v["d_cal"])
            maha[name] = r5.pval_high(md[:nc], md)
    (r5.R5 / dev / "mods" / opts.what / f"draw{opts.draw}").mkdir(parents=True, exist_ok=True)
    G.update(views=views, row=row, dev=dev, draw=opts.draw, what=opts.what, alt_views=alt, maha=maha)
    jobs = [(s, sd) for s in ("near", "far") for sd in C.ORDER_SEEDS]
    with mp.get_context("fork").Pool(opts.workers) as pool:
        for j in pool.imap_unordered(task, jobs):
            print(json.dumps({"done": j}), flush=True)
    print(json.dumps({"seconds": round(time.time() - start, 1)}))


if __name__ == "__main__":
    main()
