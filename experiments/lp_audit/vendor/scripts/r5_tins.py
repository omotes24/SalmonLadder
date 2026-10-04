"""R5 step 3: TINS for each shot draw on the dev streams (upstream functions unchanged).

Run once per dev split (VINS_WORK selects dev1 = base work dir, dev2 = <base>/dev2).
--stage setup  --draws 0 1 ...        TINS inputs with the draw's 16 shots as class prototypes, built with the same
                                      upstream calls as prepare_tins_clip.py (fresh caches per draw):
                                      <R5>/<dev>/tins_setup/draw<k>.pt
--stage run    --draws .. --seeds .. --streams ..
                                      TINS on the saved upstream-path stream features (runs/main/stream_feats_*.pt,
                                      identical images and orders for every draw). Per batch the draw's calibration
                                      shots are scored with the negatives that produced S_final (for a later p_T).
                                      <R5>/<dev>/tins/draw<k>/<stream>_seed<s>.npz
--draws orig uses runs/setup/tins_setup.pt and must reproduce runs/main bit for bit (runner check).
Labels (flags) are only saved for evaluation; TINS never receives them.
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

from vins import config as C  # noqa: E402
from vins.features import guard_from_seal, load_features  # noqa: E402
from vins.r5 import R5  # noqa: E402
from vins.tins_dev import Recorder, get_logger, import_tins, load_clip, make_args, run_stream, setup_to_device  # noqa: E402


def dev_name():
    return "dev2" if C.WORK.name == "dev2" else "dev1"


def build_setup(t, draw, out):
    id_classes = json.loads((C.SPLITS_DIR / "id_classes.json").read_text())
    cache_dir = out.parent / f"cache_draw{draw}"
    if cache_dir.exists():
        raise RuntimeError(f"{cache_dir} exists; caches ignore the label set and shot list")
    args = make_args(t, cache_dir, f"vins_r5_{dev_name()}_draw{draw}")
    args.train_imglist = str(R5 / "shots" / "lists" / f"draw{draw}_train16.txt")
    t.setup_seed(args.seed)
    log = get_logger(out.parent / "setup.log")
    net, preprocess = load_clip(t, args)
    device = next(net.parameters()).device
    guard = guard_from_seal()
    with open(args.train_imglist) as handle:
        for line in handle:
            if line.strip():
                guard.check(Path(args.root_dir) / "ImageNet" / line.split()[0])
    protos_1000, proto_meta, proto_cache = t.load_or_build_class_prototypes(args, net, preprocess, log)
    rows = torch.tensor([c["idx_1k"] for c in id_classes])
    protos = protos_1000[rows].to(device)
    ref = torch.load(C.RUNS_DIR / "setup" / "tins_setup.pt", map_location="cpu")["class_prototypes"].float()
    cos = torch.nn.functional.cosine_similarity(protos.float().cpu(), ref, dim=1)
    log.info(f"draw {draw}: cos(new prototypes, original prototypes) mean {cos.mean():.4f} min {cos.min():.4f}")
    if float(cos.mean()) > 0.9999:
        raise RuntimeError("prototypes identical to the original ones: the draw list was not used")
    labels = [c["clean_name"] for c in id_classes]
    upstream_labels = [str(x) for x in t.get_test_labels(args, None)]
    assert labels == [upstream_labels[i] for i in rows.tolist()]
    positive_features = t.encode_texts(net, [args.pos_prompt.format(label) for label in labels],
                                       batch_size=args.text_batch_size, device=device,
                                       desc="Encoding positive labels").to(device)
    base_sim = (positive_features * protos).sum(dim=1)
    negative_features, neg_texts, neg_words, neg_cache = t.load_or_build_negative_bank(
        args=args, model=net, positive_labels=labels, positive_features=positive_features,
        class_prototypes=protos.cpu(), log=log)
    init = t.build_inversion_init_candidates(args, net, neg_words, protos, device, log)
    torch.save({
        "draw": draw, "positive_labels": labels, "id_wnids": [c["wnid"] for c in id_classes],
        "positive_features": positive_features.cpu(), "class_prototypes": protos.cpu(), "base_sim": base_sim.cpu(),
        "negative_features": negative_features.cpu(), "selected_negative_texts": neg_texts,
        "selected_negative_words": neg_words,
        "init_candidates": {k: (v.cpu() if torch.is_tensor(v) else v) for k, v in init.items()},
        "meta": {"prototype_meta": proto_meta, "prototype_cache": str(proto_cache), "negative_cache": str(neg_cache),
                 "train_imglist": args.train_imglist, "cos_to_original_mean": float(cos.mean()),
                 "n_negatives_shared_with_original": len(set(neg_words) & set(torch.load(
                     C.RUNS_DIR / "setup" / "tins_setup.pt", map_location="cpu")["selected_negative_words"]))},
    }, out)


class CalRecorder(Recorder):
    """Recorder that also scores fixed calibration features under the negatives of each batch's final scoring."""

    def __init__(self, capture, cal_feats, score_fn):
        super().__init__()
        self.capture, self.cal, self.score_fn, self.cal_scores = capture, cal_feats, score_fn, []

    def on_final(self, start, scores, bank_size, buffer_size):
        kw = self.capture["last"]
        with torch.no_grad():
            cal = self.score_fn(image_features=self.cal, positive_features=kw["positive_features"],
                                negative_features=kw["negative_features"], logit_scale=kw["logit_scale"],
                                group_num=kw["group_num"], random_permute=kw["random_permute"])
        self.cal_scores.append(cal.float().cpu().numpy())
        super().on_final(start, scores, bank_size, buffer_size)


def calib_ids(draw):
    """The draw's calibration shots of this dev's ID classes, ordered by ID class then position."""
    id_classes = json.loads((C.SPLITS_DIR / "id_classes.json").read_text())
    if draw == "orig":
        dview = pd.read_parquet(C.RUNS_DIR / "dview" / "dview.parquet")
        return dview.sample_id.values[(dview.split == "calib").values]
    d = pd.read_parquet(R5 / "shots" / "draws.parquet")
    d = d[(d.draw == int(draw)) & (d.role == "calib")]
    order = {c["idx_1k"]: c["id_idx"] for c in id_classes}
    d = d[d.idx_1k.isin(order)].assign(id_idx=lambda x: x.idx_1k.map(order)).sort_values(["id_idx", "pos"])
    assert len(d) == 4 * len(id_classes)
    return d.sample_id.values


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["setup", "run"], required=True)
    parser.add_argument("--draws", nargs="+", required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(C.ORDER_SEEDS))
    parser.add_argument("--streams", nargs="+", default=["near", "far"])
    opts = parser.parse_args()
    dev = dev_name()
    t = import_tins()
    if opts.stage == "setup":
        for draw in opts.draws:
            out = R5 / dev / "tins_setup" / f"draw{draw}.pt"
            out.parent.mkdir(parents=True, exist_ok=True)
            tick = time.time()
            build_setup(t, int(draw), out)
            print(json.dumps({"setup": f"{dev}/draw{draw}", "seconds": round(time.time() - tick, 1)}), flush=True)
        return

    samples = pd.read_parquet(C.SPLITS_DIR / "samples.parquet")
    args = make_args(t, R5 / dev / "tins_run_cache", f"vins_r5_{dev}_run")
    t.setup_seed(args.seed)
    net, _ = load_clip(t, args)
    orig = t.compute_grouped_positive_score
    capture = {"last": None}

    def wrapped(**kw):
        capture["last"] = kw
        return orig(**kw)

    t.compute_grouped_positive_score = wrapped
    shot_clip = None
    pending = []
    for draw in opts.draws:
        if draw == "orig":
            setup = torch.load(C.RUNS_DIR / "setup" / "tins_setup.pt", map_location="cpu")
            clip, blob = load_features(C.FEATURES_DIR / "clip.pt")
            pos = {s: i for i, s in enumerate(blob["sample_id"])}
            cids = calib_ids("orig")
            cal = clip[[pos[s] for s in cids]]
        else:
            setup = torch.load(R5 / dev / "tins_setup" / f"draw{draw}.pt", map_location="cpu")
            if shot_clip is None:
                shot_clip = load_features(R5 / "features" / "shots.clip.pt")
            pos = {s: i for i, s in enumerate(shot_clip[1]["sample_id"])}
            cids = calib_ids(draw)
            cal = shot_clip[0][[pos[s] for s in cids]]
        dev_setup = setup_to_device(setup)
        cal = cal.to(dev_setup["positive_features"].device)
        out = R5 / dev / "tins" / f"draw{draw}"
        out.mkdir(parents=True, exist_ok=True)
        for stream in opts.streams:
            for seed in opts.seeds:
                target = out / f"{stream}_seed{seed}.npz"
                if draw != "orig":
                    if target.exists():
                        print(json.dumps({"skip_existing": str(target)}), flush=True)
                        continue
                    try:                                  # several processes may share the list of runs
                        os.close(os.open(out / f"{stream}_seed{seed}.lock", os.O_CREAT | os.O_EXCL | os.O_WRONLY))
                    except FileExistsError:
                        pending.append(target)
                        continue
                blobs = torch.load(C.RUNS_DIR / "main" / f"stream_feats_{stream}_seed{seed}.pt", map_location="cpu")
                feats, rows, flags = blobs["features"], np.asarray(blobs["rows"]), np.asarray(blobs["flags"])
                sid = samples.sample_id.values[rows]
                rec = CalRecorder(capture, cal, orig)
                args.stream_seed = seed
                t.setup_seed(args.seed)
                tick = time.time()
                scores = run_stream(t, args, net, dev_setup, feats, hook=rec)
                per = rec.per_sample(len(rows))
                assert np.array_equal(per["S_final"], scores)
                info = {"n": int(len(rows)), "seconds": round(time.time() - tick, 1), "n_seeded": int(per["seeded"].sum())}
                if draw == "orig":
                    ref = pd.read_parquet(C.RUNS_DIR / "main" / f"stream_{stream}_seed{seed}.parquet").sort_values("position")
                    assert (ref.sample_id.values == sid).all()
                    info["bitwise_equal_main"] = bool(np.array_equal(ref.S_final.values.astype(np.float32), per["S_final"]))
                tmp = out / f"{stream}_seed{seed}.tmp.npz"
                np.savez_compressed(tmp, sample_id=sid, is_ood=flags,
                                    S_arrival=per["S_arrival"], S_final=per["S_final"], seeded=per["seeded"],
                                    batch_index=per["batch_index"],
                                    bank_size=np.array([b["bank_size"] for b in rec.batches]),
                                    cal_scores=np.stack(rec.cal_scores).astype(np.float32), cal_ids=cids)
                os.replace(tmp, target)
                (out / f"{stream}_seed{seed}.lock").unlink(missing_ok=True)
                print(json.dumps({f"{dev}/draw{draw}/{stream}_seed{seed}": info}), flush=True)
    for target in pending:                                # runs taken by another process: wait for them
        waited = 0
        while not target.exists():
            if waited > 5400:
                raise RuntimeError(f"{target} was locked by another process and never appeared")
            time.sleep(15)
            waited += 15
        print(json.dumps({"done_by_other": str(target), "waited": waited}), flush=True)


if __name__ == "__main__":
    main()
