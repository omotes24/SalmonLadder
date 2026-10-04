"""Phase 3 (final evaluation; pre-registered in r5/phase3/prereg_phase3.json; run once).

--part openood : OpenOOD v1.5 ImageNet-1K, one stream per OOD dataset (ID = test_imagenet), orders 123..127
--part fourood : Four-OOD (ID = ImageNet-1K val 50,000; iNaturalist / SUN / Places / DTD), orders 123..125
Shots = the TINS 16-shot list used throughout (12 support + 4 calibration per class, 1000 classes). TINS scores come
from the existing unchanged-upstream runs of these streams (S_final and per-batch calibration scores).
Methods (all hyper-parameters fixed before this run, see the pre-registration):
  tins, v4, v5 (+ its static / memory / LP parts and the visual-only product), reprise_clip (v4 with the single
  CLIP ViT-B/16 image view), E1 baselines on DINOv2 L/14 (AdaNeg-type, OODD-type memories; 16-shot kNN and
  Mahalanobis++ with 1 and 2 views; full-train kNN, Mahalanobis++ and class means), each alone and x TINS,
  v5_m2b (OpenOOD only: calibration = 4 images per class from the OpenOOD ID-val list, support unchanged),
  v5_dino1 (the single view DINO ViT-B/16 trained on ImageNet-1K only), and the E5 decomposition of v5 on test
  (no K(x); memory with false admissions removed / all OOD arrivals; LP keeping only ID / only OOD stream nodes).
Output: <R5>/phase3/<part>/<stream>_seed<s>.json (metrics) and .npz (per-sample scores of the main methods)
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
import torch  # noqa: E402

from vins import config as C  # noqa: E402
from vins import r5  # noqa: E402

P3 = r5.R5 / "phase3"
RES = C.WORK / "test_eval" / "results"
FEAT = C.WORK / "extra_feats"
OO = ["ssb_hard", "ninco", "inaturalist", "textures", "openimageo"]
FOUR = ["inat", "sun", "places", "dtd"]
G = {}


def load_script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def prereg():
    f = P3 / "prereg_phase3.json"
    blob = f.read_bytes()
    return json.loads(blob), hashlib.sha256(blob).hexdigest()


def topk(feats, pos, k, chunk=8192):
    pos = pos.float()
    return torch.cat([(feats[lo:lo + chunk].float() @ pos.T).topk(k, dim=1).indices
                      for lo in range(0, len(feats), chunk)]).numpy()


def load_part(part):
    """Features of the shots and of every evaluated image, in one row space: rows [0, 4000) = calibration shots,
    rows 4000.. = evaluated images (the order of `ids`)."""
    b = torch.load(RES / "features.pt", map_location="cpu")
    lt = torch.load(RES / "features_vitl14.pt", map_location="cpu")
    shots = {"B14": b["dino"][:16000].numpy(), "L14": lt["dino"][:16000].numpy()}
    extra = torch.load(P3 / "shots.extra.pt", map_location="cpu")          # CLIP (TINS path) + DINO v1 of the shots
    shots["CLIP"] = extra["clip"].numpy()
    shots["DINO1"] = extra["dino1"].numpy()
    pos = torch.load(C.WORK / "analysis" / "baselines" / "pos_tins.pt")["pos"]
    if part == "openood":
        ids = sorted({x for ds in OO for x in np.load(C.WORK / "test_runs" / "default" / f"{ds}_seed123.npz")["sample_id"].tolist()})
        assert len(ids) == len(b["dino"]) - 16000
        ev = {"B14": b["dino"][16000:].numpy(), "L14": lt["dino"][16000:].numpy(),
              "CLIP": b["clip_q"][4000:].numpy()}
        d1 = torch.load(P3 / "test.dino1.openood.pt", map_location="cpu")
        assert d1["ids"] == ids
        ev["DINO1"] = d1["features"].numpy()
    else:
        names = ["in_val"] + FOUR
        dn = {n: torch.load(FEAT / f"{n}.dino.pt", map_location="cpu") for n in names}
        dl = {n: torch.load(FEAT / f"{n}.dinol14.pt", map_location="cpu") for n in names}
        cl = {n: torch.load(FEAT / f"{n}.clipb16.pt", map_location="cpu") for n in names}
        ids = sum((dn[n]["ids"] for n in names), [])
        assert ids == sum((dl[n]["ids"] for n in names), []) == sum((cl[n]["ids"] for n in names), [])
        ev = {"B14": torch.cat([dn[n]["features"] for n in names]).numpy(),
              "L14": torch.cat([dl[n]["features"] for n in names]).numpy(),
              "CLIP": torch.cat([cl[n]["features"] for n in names]).numpy()}
        d1 = torch.load(P3 / "test.dino1.fourood.pt", map_location="cpu")
        assert d1["ids"] == ids
        ev["DINO1"] = d1["features"].numpy()
    for k in ev:
        ev[k] = ev[k].astype(np.float32)
        ev[k] /= np.linalg.norm(ev[k], axis=1, keepdims=True)
        shots[k] = shots[k].astype(np.float32)
        shots[k] /= np.linalg.norm(shots[k], axis=1, keepdims=True)
    cal_clip = b["clip_q"][:4000]                                          # TINS-path CLIP of the calibration shots
    cands = {kk: np.concatenate([topk(cal_clip, pos, kk), topk(torch.from_numpy(ev["CLIP"]), pos, kk)])
             for kk in (5, 10, 20)}
    base_cand = np.asarray(b["cand"])                                       # the candidates v4 used on test
    cands[5] = base_cand if part == "openood" else np.concatenate([base_cand[:4000], cands[5][4000:]])
    return {"ids": ids, "row": {s: 4000 + i for i, s in enumerate(ids)}, "shots": shots, "ev": ev, "cands": cands}


def view(D, name, n0, K, m, cal=None):
    """Prototype view over [calibration; evaluated images]; cal = optional replacement calibration features."""
    sup = D["shots"][name][:12000].reshape(1000, 12, -1)
    c = D["shots"][name][12000:16000] if cal is None else cal
    q = np.concatenate([c, D["ev"][name]])
    cand = D["cands"][K] if cal is None else np.concatenate([G["m2b_cand"][K], D["cands"][K][4000:]])
    is_cal = np.zeros(len(q), dtype=bool)
    is_cal[:len(c)] = True
    v = r5.proto_view(sup, q, cand, is_cal, n0=n0, m=m)
    v["support_arr"], v["q"] = sup, q
    return v


def reprise(D, views, rows, bidx, cfg, is_ood=None, oracle=None):
    """p_t and p_LP of each view for configuration cfg (and the oracle / no-K variants of E5 when asked)."""
    out = {}
    th = tuple(cfg["thresholds"])
    for name, v in views.items():
        sf, d, p_all = v["q"][rows], v["d"][rows], v["p_all"][rows]
        med, mad = v["stats"]["med_all"], v["stats"]["mad_all"]
        masks = r5.entrance(sf, d, p_all, bidx, v["cal_feats"], v["d_cal"], med, mad, th, cfg["m"])
        pt, _ = r5.memory_p(sf, d, bidx, masks[-1], v["cal_feats"], v["d_cal"], med, mad, cfg["m"])
        lp = r5.lp_run(v["support_arr"], v["cal_feats"], sf, bidx, k=cfg["kg"], alpha=cfg["lam"], gamma=cfg["gamma"],
                       iters=15)["p"]
        out[name] = {"p": v["p"][rows], "pt": pt, "plp": lp, "M": masks[-1]}
        if oracle:
            da, dac = v["d_all"][rows], v["d_all_cal"]
            Mn = r5.entrance(sf, da, p_all, bidx, v["cal_feats"], dac, med, mad, th, cfg["m"])[-1]
            out[name]["pt_noK"], _ = r5.memory_p(sf, da, bidx, Mn, v["cal_feats"], dac, med, mad, cfg["m"])
            out[name]["pt_pure"], _ = r5.memory_p(sf, d, bidx, masks[-1] & is_ood, v["cal_feats"], v["d_cal"], med, mad,
                                                  cfg["m"])
            out[name]["pt_allood"], _ = r5.memory_p(sf, d, bidx, is_ood.copy(), v["cal_feats"], v["d_cal"], med, mad,
                                                    cfg["m"])
            kw = dict(k=cfg["kg"], alpha=cfg["lam"], gamma=cfg["gamma"], iters=15)
            out[name]["plp_keep_id"] = r5.lp_oracle(v["support_arr"], v["cal_feats"], sf, bidx, ~is_ood, **kw)
            out[name]["plp_keep_ood"] = r5.lp_oracle(v["support_arr"], v["cal_feats"], sf, bidx, is_ood, **kw)
    return out


def task(job):
    part, stream, seed, path = job
    from vins.metrics import measures as upstream

    D, pr = G["D"], G["prereg"]
    run = np.load(path, allow_pickle=True)
    ids, is_ood = run["sample_id"], run["is_ood"].astype(bool)
    S, bidx = run["S_final"].astype(np.float64), run["batch_index"]
    rows = np.array([D["row"][x] for x in ids])
    sc, arr, t0 = {"tins": S}, {"S": S}, time.time()
    v4, v5 = pr["configs"]["v4"], pr["configs"]["v5"]
    views4 = {n: G["views"][("v4", n)] for n in ("B14", "L14")}
    views5 = {n: G["views"][("v5", n)] for n in ("B14", "L14")}
    r4 = reprise(D, views4, rows, bidx, v4)
    r5_ = reprise(D, views5, rows, bidx, v5, is_ood=is_ood, oracle=True)
    P = lambda r, k: r["B14"][k] * r["L14"][k]  # noqa: E731
    sc["v4"] = S * P(r4, "pt") * P(r4, "plp")
    sc["v4_lp"] = S * P(r4, "plp")
    sc["v5"] = S * P(r5_, "pt") * P(r5_, "plp")
    sc["v5_static"], sc["v5_mem"], sc["v5_lp"] = S * P(r5_, "p"), S * P(r5_, "pt"), S * P(r5_, "plp")
    sc["v5_vis"] = P(r5_, "pt") * P(r5_, "plp")
    sc["v5_noK"] = S * P(r5_, "pt_noK") * P(r5_, "plp")
    sc["v5_mem_pure"] = S * P(r5_, "pt_pure") * P(r5_, "plp")
    sc["v5_mem_allood"] = S * P(r5_, "pt_allood") * P(r5_, "plp")
    sc["v5_lp_keep_id"] = S * P(r5_, "pt") * P(r5_, "plp_keep_id")
    sc["v5_lp_keep_ood"] = S * P(r5_, "pt") * P(r5_, "plp_keep_ood")
    for n in ("B14", "L14"):
        arr.update({f"v5_pt_{n}": r5_[n]["pt"], f"v5_plp_{n}": r5_[n]["plp"], f"v5_p_{n}": r5_[n]["p"],
                    f"v4_pt_{n}": r4[n]["pt"], f"v4_plp_{n}": r4[n]["plp"]})
    ent = {k: {"id_rate": float(np.mean([r[n]["M"][~is_ood].mean() for n in ("B14", "L14")])),
               "ood_rate": float(np.mean([r[n]["M"][is_ood].mean() for n in ("B14", "L14")]))} for k, r in (("v4", r4), ("v5", r5_))}
    # REPRISE-CLIP (v4 configuration, one CLIP view) and the DINO (IN-1k) view (v5 configuration, one view)
    rc = reprise(D, {"CLIP": G["views"][("v4", "CLIP")]}, rows, bidx, v4)["CLIP"]
    sc["reprise_clip"], sc["reprise_clip_vis"] = S * rc["pt"] * rc["plp"], rc["pt"] * rc["plp"]
    rd = reprise(D, {"DINO1": G["views"][("v5", "DINO1")]}, rows, bidx, v5)["DINO1"]
    sc["v5_dino1"], sc["v5_dino1_vis"] = S * rd["pt"] * rd["plp"], rd["pt"] * rd["plp"]
    sc["v5_l14only"] = S * r5_["L14"]["pt"] * r5_["L14"]["plp"]
    if part == "openood":                                   # M2b: calibration from the OpenOOD ID-val list
        vm = {n: G["views"][("v5m2b", n)] for n in ("B14", "L14")}
        rm = reprise(D, vm, rows, bidx, v5)
        sc["v5_m2b"] = S * P(rm, "pt") * P(rm, "plp")
        ent["v5_m2b"] = {"id_rate": float(np.mean([rm[n]["M"][~is_ood].mean() for n in ("B14", "L14")])),
                         "ood_rate": float(np.mean([rm[n]["M"][is_ood].mean() for n in ("B14", "L14")])),
                         "pr_pt_le_0.1_id": float(np.mean([(rm[n]["pt"][~is_ood] <= 0.1).mean() for n in ("B14", "L14")]))}
        ent["v5"]["pr_pt_le_0.1_id"] = float(np.mean([(r5_[n]["pt"][~is_ood] <= 0.1).mean() for n in ("B14", "L14")]))
    # E1 baselines (L/14 and two views), static conformal p against the calibration shots
    E1 = G["e1mod"]
    b = pr["baselines"]
    vL = views5["L14"]
    sf, qcal = vL["q"][rows], vL["cal_feats"]
    mu = G["muL"]
    s_id, s_id_cal = (sf @ mu.T).max(axis=1), (qcal @ mu.T).max(axis=1)
    raw, p = E1.memory_score_p(s_id, s_id_cal, sf, qcal, bidx, lambda s, t=b["adaneg_tau"]: s < t, lam=b["adaneg_lam"])
    sc["adaneg"], sc["adaneg_vis"] = S * p, raw
    raw, p = E1.memory_score_p(s_id, s_id_cal, sf, qcal, bidx, None, lam=1.0, cap=b["oodd_K"])
    sc["oodd"], sc["oodd_vis"] = S * p, raw
    for key, pv in G["static"].items():                     # 16-shot and full-train static baselines (p per image)
        sc[key], sc[f"{key}_vis"] = S * pv[rows], pv[rows]
    metrics = {}
    for k, s in sc.items():
        m = upstream(s[~is_ood], s[is_ood])
        metrics[k] = {"AUROC": 100 * m["AUROC"], "FPR95": 100 * m["FPR95"],
                      "AUROC_within_batch": 100 * r5.within_batch_auroc(s, is_ood, bidx)}
    out = P3 / part
    np.savez_compressed(out / f"{stream}_seed{seed}.npz", sample_id=ids, is_ood=is_ood, batch_index=bidx,
                        **{k: np.asarray(v, dtype=np.float32) for k, v in arr.items()},
                        **{f"score_{k}": np.asarray(sc[k], dtype=np.float64) for k in ("v4", "v5", "v5_m2b", "reprise_clip")
                           if k in sc})
    (out / f"{stream}_seed{seed}.json").write_text(json.dumps({"metrics": metrics, "entrance": ent,
                                                               "seconds": round(time.time() - t0, 1),
                                                               "n": int(len(ids)), "n_ood": int(is_ood.sum())}) + "\n")
    return stream, seed


def static_baselines(part):
    """16-shot kNN / Mahalanobis++ (B14 x L14 and L14 alone) and full-train kNN / Mahalanobis++ / class means on L/14
    (raw distances from p3_static.py in the same row space) -> conformal p (large = ID) against the calibration shots."""
    z = np.load(P3 / f"static_{part}.npz")
    pv = {k: r5.pval_high(z[k][:4000], z[k]) for k in z.files if k != "ids"}
    return {"knn16_2view": pv["knn16_B14"] * pv["knn16_L14"], "maha16_2view": pv["maha16_B14"] * pv["maha16_L14"],
            "knn16_l14": pv["knn16l_L14"], "maha16_l14": pv["maha16l_L14"],
            "knnF_l14": pv["knnF"], "mahaF_l14": pv["mahaF"], "protoF_l14": pv["protoF"]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--part", choices=["openood", "fourood"], required=True)
    parser.add_argument("--workers", type=int, default=12)
    opts = parser.parse_args()
    start = time.time()
    pr, digest = prereg()
    assert pr["sha256_of_this_file_is_logged"], "pre-registration incomplete"
    from vins.tins_dev import import_tins

    import_tins()
    D = load_part(opts.part)
    G.update(D=D, prereg=pr, e1mod=load_script("r5_e1"))
    v4, v5 = pr["configs"]["v4"], pr["configs"]["v5"]
    views = {}
    for tag, cfg in (("v4", v4), ("v5", v5)):
        for n in ("B14", "L14"):
            views[(tag, n)] = view(D, n, cfg["n0"], cfg["K"], cfg["m"])
    views[("v4", "CLIP")] = view(D, "CLIP", v4["n0"], v4["K"], v4["m"])
    views[("v5", "DINO1")] = view(D, "DINO1", v5["n0"], v5["K"], v5["m"])
    if opts.part == "openood":
        m2b = torch.load(P3 / "m2b_calibration.pt", map_location="cpu")    # ID-val images, 4 per class
        G["m2b_cand"] = {kk: np.asarray(m2b["cand"][str(kk)]) for kk in (5, 10, 20)}
        for n in ("B14", "L14"):
            c = m2b[n].numpy().astype(np.float32)
            c /= np.linalg.norm(c, axis=1, keepdims=True)
            views[("v5m2b", n)] = view(D, n, v5["n0"], v5["K"], v5["m"], cal=c)
    G["views"] = views
    sup = D["shots"]["L14"][:12000].reshape(1000, 12, -1).sum(axis=1)
    G["muL"] = (sup / np.linalg.norm(sup, axis=1, keepdims=True)).astype(np.float32)
    G["static"] = static_baselines(opts.part)
    out = P3 / opts.part
    out.mkdir(parents=True, exist_ok=True)
    if opts.part == "openood":
        tasks = [(opts.part, ds, s, C.WORK / "test_runs" / "default" / f"{ds}_seed{s}.npz")
                 for ds in OO for s in (123, 124, 125, 126, 127)]
    else:
        tasks = [(opts.part, o, s, C.WORK / "extra_runs" / "fourood" / f"imagenet_{o}_seed{s}.npz")
                 for o in FOUR for s in (123, 124, 125)]
    tasks = [t for t in tasks if not (out / f"{t[1]}_seed{t[2]}.json").exists()]
    print(json.dumps({"part": opts.part, "prereg_sha256": digest, "tasks": len(tasks),
                      "setup_s": round(time.time() - start, 1)}), flush=True)
    with mp.get_context("fork").Pool(opts.workers) as pool:
        for stream, seed in pool.imap_unordered(task, tasks):
            print(json.dumps({"done": f"{stream}_seed{seed}"}), flush=True)
    print(json.dumps({"part": opts.part, "seconds": round(time.time() - start, 1)}))


if __name__ == "__main__":
    main()
