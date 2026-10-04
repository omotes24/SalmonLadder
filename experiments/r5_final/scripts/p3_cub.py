"""Phase 3 / F2: SSB CUB (Vaze et al.; known 100 classes, unknown Easy / Medium / Hard) under the same protocol:
16 shots per known class from the CUB train split (12 support + 4 calibration, seeded), streams = CUB test images of
the known classes (ID) + test images of one unknown difficulty (OOD), upstream TINS mixing order, seeds 123..125.
Nothing is tuned on CUB: TINS keeps its ImageNet hyper-parameters (prototypes = normalised mean CLIP feature of the 16
shots, as upstream; negatives mined by the upstream code for the CUB label set), REPRISE keeps the frozen v4 / v5
configurations. The SSB split file is the authors' cub_osr_splits.pkl.

--stage setup : splits (r5/phase3/cub/splits.json)
--stage feats : CLIP ViT-B/16 (TINS path), DINOv2 B/14 and L/14 of every used image (GPU)
--stage tins  : TINS setup for the CUB labels and the streams (GPU)
--stage eval  : TINS, v4, v5 (+ parts), 16-shot kNN / Mahalanobis++ (2 views) x TINS, MCM (CPU)
"""
import argparse
import json
import multiprocessing as mp
import os
import pickle
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

CUB = Path("/home/omote/reprise_controls_20260927/data/CUB_200_2011")
SSB = ROOT / "ext" / "ssb" / "cub_osr_splits.pkl"
OUT = r5.R5 / "phase3" / "cub"
LEVELS = ("Easy", "Medium", "Hard")
SEEDS = (123, 124, 125)
G = {}


def read_table(name):
    return [line.split() for line in (CUB / name).read_text().splitlines() if line.strip()]


def setup():
    ssb = pickle.load(open(SSB, "rb"))
    classes = {int(i) - 1: n for i, n in read_table("classes.txt")}
    images = {int(i): p for i, p in read_table("images.txt")}
    label = {int(i): int(c) - 1 for i, c in read_table("image_class_labels.txt")}
    train = {int(i): int(t) for i, t in read_table("train_test_split.txt")}
    known = sorted(ssb["known_classes"])
    names = [classes[c].split(".", 1)[1].replace("_", " ") for c in known]
    shots = {}
    for j, c in enumerate(known):
        ids = sorted(i for i in images if label[i] == c and train[i] == 1)
        pick = np.random.default_rng([20260927, 81, c]).permutation(len(ids))[:16]
        shots[j] = [images[ids[k]] for k in pick]
    test_id = [images[i] for i in sorted(images) if train[i] == 0 and label[i] in known]
    ood = {lv: [images[i] for i in sorted(images) if train[i] == 0 and label[i] in set(ssb["unknown_classes"][lv])]
           for lv in LEVELS}
    OUT.mkdir(parents=True, exist_ok=True)
    blob = {"known": known, "names": names, "support": {j: s[:12] for j, s in shots.items()},
            "calib": {j: s[12:] for j, s in shots.items()}, "test_id": test_id, "ood": ood,
            "test_id_label": [known.index(label[i]) for i in sorted(images) if train[i] == 0 and label[i] in known]}
    (OUT / "splits.json").write_text(json.dumps(blob) + "\n")
    print(json.dumps({"n_test_id": len(test_id), **{f"n_{lv}": len(v) for lv, v in ood.items()}}))


def all_paths(sp):
    rel = [p for j in range(100) for p in sp["support"][str(j)]] + [p for j in range(100) for p in sp["calib"][str(j)]]
    rel += sp["test_id"] + [p for lv in LEVELS for p in sp["ood"][lv]]
    return list(dict.fromkeys(rel))


@torch.no_grad()
def feats():
    from vins.features import dino_encode_fn, dino_transform, encode, guard_from_seal, load_dino
    from vins.tins_dev import import_tins, load_clip, make_args

    sp = json.loads((OUT / "splits.json").read_text())
    rel = all_paths(sp)
    paths = [str(CUB / "images" / p) for p in rel]
    guard = guard_from_seal()
    t = import_tins()
    args = make_args(t, OUT / "clip_cache", "vins_p3_cub")
    net, preprocess = load_clip(t, args)
    out = {"ids": rel, "clip": encode(paths, preprocess, net.encode_image, guard,
                                      256, 8, "clip")}
    del net
    base = load_dino()
    out["dino"] = encode(paths, dino_transform(), dino_encode_fn(base), guard,
                         128, 8, "dino")
    del base
    large = torch.hub.load(str(C.DINO_HUB), "dinov2_vitl14", source="local", pretrained=False)
    large.load_state_dict(torch.load(C.HOME / ".cache/torch/hub/checkpoints/dinov2_vitl14_pretrain.pth",
                                     map_location="cpu"), strict=True)
    large = large.eval().cuda()
    out["dino_vitl14"] = encode(paths, dino_transform(), dino_encode_fn(large), guard,
                                64, 8, "dino_vitl14")
    torch.save(out, OUT / "feats.pt")
    print(json.dumps({"n": len(rel)}))


def tins():
    from vins.tins_dev import Recorder, build_order, get_logger, import_tins, load_clip, make_args, run_stream, setup_to_device

    sp = json.loads((OUT / "splits.json").read_text())
    F = torch.load(OUT / "feats.pt", map_location="cpu")
    pos_of = {s: i for i, s in enumerate(F["ids"])}
    clip = F["clip"].float()
    t = import_tins()
    args = make_args(t, OUT / "tins_cache", "vins_p3_cub_tins")
    t.setup_seed(args.seed)
    log = get_logger(OUT / "tins.log")
    net, _ = load_clip(t, args)
    device = next(net.parameters()).device
    labels = sp["names"]
    shots = [[p for p in sp["support"][str(j)] + sp["calib"][str(j)]] for j in range(100)]
    protos = torch.stack([clip[[pos_of[p] for p in s]].mean(dim=0) for s in shots])
    protos = (protos / protos.norm(dim=-1, keepdim=True)).to(device)
    positive = t.encode_texts(net, [args.pos_prompt.format(x) for x in labels], batch_size=args.text_batch_size,
                              device=device, desc="pos").to(device)
    neg, neg_texts, neg_words, _ = t.load_or_build_negative_bank(args=args, model=net, positive_labels=labels,
                                                               positive_features=positive, class_prototypes=protos.cpu(), log=log)
    init = t.build_inversion_init_candidates(args, net, neg_words, protos, device, log)
    setup = {"positive_features": positive.cpu(), "negative_features": neg.cpu(), "class_prototypes": protos.cpu(),
             "base_sim": (positive * protos).sum(dim=1).cpu(),
             "init_candidates": {k: (v.cpu() if torch.is_tensor(v) else v) for k, v in init.items()}}
    torch.save({**setup, "negative_words": neg_words}, OUT / "tins_setup.pt")
    dev_setup = setup_to_device(setup)
    id_feats = clip[[pos_of[p] for p in sp["test_id"]]]
    for lv in LEVELS:
        ood_feats = clip[[pos_of[p] for p in sp["ood"][lv]]]
        for seed in SEEDS:
            order = build_order(len(id_feats), len(ood_feats), seed)
            ids = [sp["test_id"][i] if o == 0 else sp["ood"][lv][i] for o, i in order]
            flags = np.array([o for o, _ in order])
            feats_s = torch.stack([id_feats[i] if o == 0 else ood_feats[i] for o, i in order])
            rec = Recorder()
            args.stream_seed = seed
            t.setup_seed(args.seed)
            scores = run_stream(t, args, net, dev_setup, feats_s, hook=rec)
            per = rec.per_sample(len(ids))
            assert np.array_equal(per["S_final"], scores)
            np.savez_compressed(OUT / f"tins_{lv}_seed{seed}.npz", sample_id=np.array(ids), is_ood=flags,
                                S_final=per["S_final"], batch_index=per["batch_index"])
            print(json.dumps({"tins": f"{lv}/{seed}", "n": len(ids)}), flush=True)


def eval_task(job):
    lv, seed = job
    from vins.metrics import measures as upstream

    D, pr = G["D"], G["prereg"]
    z = np.load(OUT / f"tins_{lv}_seed{seed}.npz", allow_pickle=True)
    ids, is_ood, S, bidx = z["sample_id"], z["is_ood"].astype(bool), z["S_final"].astype(np.float64), z["batch_index"]
    rows = 400 + np.array([D["row"][x] for x in ids])
    sc = {"tins": S, "mcm": D["mcm"][rows - 400]}
    for tag in ("v4", "v5"):
        cfg = pr["configs"][tag]
        pts, lps, ps = [], [], []
        for name in ("B14", "L14"):
            v = G["views"][(tag, name)]
            sf, d, p_all = v["q"][rows], v["d"][rows], v["p_all"][rows]
            med, mad = v["stats"]["med_all"], v["stats"]["mad_all"]
            M = r5.entrance(sf, d, p_all, bidx, v["cal_feats"], v["d_cal"], med, mad, tuple(cfg["thresholds"]), cfg["m"])[-1]
            pt, _ = r5.memory_p(sf, d, bidx, M, v["cal_feats"], v["d_cal"], med, mad, cfg["m"])
            lp = r5.lp_run(v["support_arr"], v["cal_feats"], sf, bidx, k=cfg["kg"], alpha=cfg["lam"], gamma=cfg["gamma"],
                           iters=15)["p"]
            pts.append(pt)
            lps.append(lp)
            ps.append(v["p"][rows])
        sc[tag] = S * pts[0] * pts[1] * lps[0] * lps[1]
        sc[f"{tag}_lp"] = S * lps[0] * lps[1]
        sc[f"{tag}_static"] = S * ps[0] * ps[1]
        sc[f"{tag}_vis"] = pts[0] * pts[1] * lps[0] * lps[1]
    for key in ("knn16", "maha16"):
        sc[f"{key}_2view"] = S * D["static"][key][rows]
    metrics = {}
    for k, s in sc.items():
        m = upstream(s[~is_ood], s[is_ood])
        metrics[k] = {"AUROC": 100 * m["AUROC"], "FPR95": 100 * m["FPR95"]}
    (OUT / f"eval_{lv}_seed{seed}.json").write_text(json.dumps({"metrics": metrics, "n": len(ids),
                                                                "n_ood": int(is_ood.sum())}) + "\n")
    return job


def evaluate(workers):
    from vins.tins_dev import import_tins

    import_tins()
    pr = json.loads((r5.R5 / "phase3" / "prereg_phase3.json").read_text())
    sp = json.loads((OUT / "splits.json").read_text())
    F = torch.load(OUT / "feats.pt", map_location="cpu")
    pos_of = {s: i for i, s in enumerate(F["ids"])}
    sup_rel = [p for j in range(100) for p in sp["support"][str(j)]]
    cal_rel = [p for j in range(100) for p in sp["calib"][str(j)]]
    ev_rel = sp["test_id"] + [p for lv in LEVELS for p in sp["ood"][lv]]
    ev_rel = list(dict.fromkeys(ev_rel))
    setup = torch.load(OUT / "tins_setup.pt", map_location="cpu")
    pos_text = setup["positive_features"].float()
    clip = F["clip"].float()
    q_clip = torch.cat([clip[[pos_of[p] for p in cal_rel]], clip[[pos_of[p] for p in ev_rel]]])
    logits = q_clip @ pos_text.T
    mcm = torch.softmax(logits[400:], dim=1).max(dim=1).values.numpy()          # MCM, temperature 1 on cosines
    views, static = {}, {}
    per = {}
    for name, key in (("B14", "dino"), ("L14", "dino_vitl14")):
        f = F[key].float().numpy()
        f /= np.linalg.norm(f, axis=1, keepdims=True)
        sup = f[[pos_of[p] for p in sup_rel]].reshape(100, 12, -1).astype(np.float32)
        q = np.concatenate([f[[pos_of[p] for p in cal_rel]], f[[pos_of[p] for p in ev_rel]]]).astype(np.float32)
        is_cal = np.zeros(len(q), dtype=bool)
        is_cal[:400] = True
        for tag in ("v4", "v5"):
            cfg = pr["configs"][tag]
            cand = logits.topk(cfg["K"], dim=1).indices.numpy()
            v = r5.proto_view(sup, q, cand, is_cal, n0=cfg["n0"], m=cfg["m"])
            v["support_arr"], v["q"] = sup, q
            views[(tag, name)] = v
        b = pr["baselines"]
        kd = r5.knn_distance(sup.reshape(-1, sup.shape[-1]), q, (b["knn_k"],))[b["knn_k"]]
        md = r5.maha_pp(sup, q, (b["maha_lam"],))[b["maha_lam"]]
        per[name] = {"knn16": r5.pval_high(kd[:400], kd), "maha16": r5.pval_high(md[:400], md)}
    static = {k: per["B14"][k] * per["L14"][k] for k in ("knn16", "maha16")}
    G.update(D={"row": {s: i for i, s in enumerate(ev_rel)}, "mcm": mcm, "static": static}, prereg=pr, views=views)
    jobs = [(lv, s) for lv in LEVELS for s in SEEDS]
    with mp.get_context("fork").Pool(workers) as pool:
        for j in pool.imap_unordered(eval_task, jobs):
            print(json.dumps({"done": j}), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["setup", "feats", "tins", "eval"], required=True)
    parser.add_argument("--workers", type=int, default=9)
    opts = parser.parse_args()
    start = time.time()
    {"setup": setup, "feats": feats, "tins": tins, "eval": lambda: evaluate(opts.workers)}[opts.stage]()
    print(json.dumps({"stage": opts.stage, "seconds": round(time.time() - start, 1)}))


if __name__ == "__main__":
    main()
