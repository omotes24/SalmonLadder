"""TINS on the additional benchmarks (post-test analyses), with the unchanged upstream scoring functions.

--task fourood  : ID = ImageNet-1K val (50,000), OOD = iNaturalist / SUN / Places / DTD (MOS Four-OOD), CLIP ViT-B/16,
                  setup identical to the OpenOOD runs (Codex's cached prototypes / static negatives / init candidates)
--task acrossid : ID = ImageNet-V2 / -R / -Sketch (--id in_v2|in_r|in_sketch), OOD = Four-OOD, following TINS's
                  across-ID protocol: 4 proxy images per class drawn from the ID set (random.Random(0)) build the
                  prototypes, the rest is the ID test set; static negatives are selected for that label set
--task l14      : OpenOOD v1.5 ImageNet-1K test streams with CLIP ViT-L/14 (setup rebuilt with ViT-L/14)
Streams are ID+OOD shuffled with random.Random(seed) as upstream. Per-image CLIP features come from extract_extra.py.
"""
import argparse
import json
import random
import shutil
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from vins import config as C  # noqa: E402
from vins.tins_dev import get_logger, import_tins  # noqa: E402
from scripts.run_tins_test import TestRecorder, codex_args  # noqa: E402

F = C.WORK / "extra_feats"
FOUR = ["inat", "sun", "places", "dtd"]
OO = ["ssb_hard", "ninco", "inaturalist", "textures", "openimageo"]


def load_feats(name, model="clipb16"):
    blob = torch.load(F / f"{name}.{model}.pt", map_location="cpu")
    return blob


def load_clip_any(t, args, arch):
    root = C.CLIP_WEIGHTS_DIR if arch == "ViT-B/16" else C.WORK / "weights"
    clip_mod = t.official_clip.clip
    original = clip_mod._download
    clip_mod._download = lambda url, r=None: original(url, str(root))
    try:
        return t.load_official_clip(args)
    finally:
        clip_mod._download = original


def synset_names():
    raw = json.loads((C.TINS_DIR / "data" / "ImageNet" / "imagenet_class_index.json").read_text())
    return {v[0]: v[1].replace("_", " ") for _, v in sorted(raw.items(), key=lambda kv: int(kv[0]))}


def proxy_split(targets, n_cls, seed=0, per_class=4):
    """Identical to eval_tins_w_init_across_id.split_id_test_dataset_for_proxies."""
    by_class = defaultdict(list)
    for index, label in enumerate(targets):
        by_class[int(label)].append(index)
    rng = random.Random(seed)
    proxy, test = [], []
    for label in range(n_cls):
        idx = list(by_class[label])
        assert len(idx) > per_class, (label, len(idx))
        rng.shuffle(idx)
        proxy += idx[:per_class]
        test += idx[per_class:]
    return sorted(proxy), sorted(test)


def run_streams(t, args, net, setup, id_feats, id_ids, ood, seeds, out, tag, cal_feats=None, capture=None, score_fn=None):
    device = next(net.parameters()).device
    for name, (o_feats, o_ids) in ood.items():
        for seed in seeds:
            order = [(0, i) for i in range(len(id_ids))] + [(1, j) for j in range(len(o_ids))]
            random.Random(seed).shuffle(order)
            feats = torch.stack([id_feats[k] if o == 0 else o_feats[k] for o, k in order])
            ids = [id_ids[i] if o == 0 else o_ids[i] for o, i in order]
            is_ood = np.array([o for o, _ in order], dtype=np.int32)
            args.stream_seed = seed
            rec = TestRecorder(capture, cal_feats, score_fn) if cal_feats is not None else None
            tick = time.time()
            scores = t.compute_tins_scores_from_image_features(
                image_features=feats, args=args, model=net, positive_features=setup["pos"],
                negative_features=setup["neg"].to(device), inversion_init_candidates=setup["init"],
                class_prototypes=setup["protos"], base_sim=setup["base_sim"], hook=rec)
            extra = {}
            if rec is not None:
                per = rec.per_sample(len(ids))
                extra = {k: per[k] for k in ("S_arrival", "seeded", "inv_pass", "batch_index")}
                extra["bank_size"] = np.array([b["bank_size"] for b in rec.batches])
                extra["cal_scores"] = np.stack(rec.cal_scores).astype(np.float32)
            np.savez_compressed(out / f"{tag}_{name}_seed{seed}.npz", sample_id=np.array(ids), is_ood=is_ood,
                                S_final=scores, **extra)
            print(json.dumps({f"{tag}_{name}_seed{seed}": {"n": len(ids), "seconds": round(time.time() - tick, 1)}}),
                  flush=True)


def build_setup(t, args, net, labels, protos, log):
    device = next(net.parameters()).device
    pos = t.encode_texts(net, [args.pos_prompt.format(l) for l in labels], batch_size=args.text_batch_size,
                         device=device, desc="pos").to(device)
    protos = protos.to(device)
    neg, texts, words, _ = t.load_or_build_negative_bank(args=args, model=net, positive_labels=labels, positive_features=pos,
                                                         class_prototypes=protos.cpu(), log=log)
    init = t.build_inversion_init_candidates(args, net, words, protos, device, log)
    return {"pos": pos, "neg": neg, "init": init, "protos": protos, "base_sim": (pos * protos).sum(dim=1),
            "labels": labels, "neg_texts": texts}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=["fourood", "acrossid", "l14"], required=True)
    parser.add_argument("--id", default=None)
    parser.add_argument("--seeds", type=int, nargs="+", default=[123])
    parser.add_argument("--datasets", nargs="+", default=None, help="l14: subset of the OpenOOD streams")
    opts = parser.parse_args()
    out = C.WORK / "extra_runs" / opts.task
    out.mkdir(parents=True, exist_ok=True)
    t = import_tins()
    log = get_logger(out / f"run_{opts.id or opts.task}.log")

    if opts.task == "fourood":
        args = codex_args(t, C.WORK / "test_runs" / "tins_cache", "vins_fourood", 123)
        t.setup_seed(args.seed)
        net, preprocess = load_clip_any(t, args, "ViT-B/16")
        device = next(net.parameters()).device
        protos = t.load_or_build_class_prototypes(args, net, preprocess, log)[0].to(device)
        labels = [str(x) for x in t.get_test_labels(args, None)]
        setup = build_setup(t, args, net, labels, protos, log)
        cal = torch.load(C.WORK / "test_runs" / "cal_clip.pt")["features"].to(device)
        capture = {"last": None}
        orig = t.compute_grouped_positive_score

        def wrapped(**kw):
            capture["last"] = kw
            return orig(**kw)

        t.compute_grouped_positive_score = wrapped
        idb = load_feats("in_val")
        ood = {n: (load_feats(n)["features"], load_feats(n)["ids"]) for n in FOUR}
        run_streams(t, args, net, setup, idb["features"], idb["ids"], ood, opts.seeds, out, "imagenet",
                    cal_feats=cal, capture=capture, score_fn=orig)

    elif opts.task == "acrossid":
        cache = C.WORK / "extra_runs" / f"tins_cache_{opts.id}"
        args = codex_args(t, cache, f"vins_acrossid_{opts.id}", 123)
        t.setup_seed(args.seed)
        net, _ = load_clip_any(t, args, "ViT-B/16")
        device = next(net.parameters()).device
        idb = load_feats(opts.id)
        if opts.id == "in_v2":
            labels = [str(x) for x in t.get_test_labels(args, None)]
            targets = np.array(idb["labels"])
        else:
            classes = sorted(set(idb["wnids"]))
            names = synset_names()
            labels = [names[w] for w in classes]
            pos_of = {w: i for i, w in enumerate(classes)}
            targets = np.array([pos_of[w] for w in idb["wnids"]])
        proxy, test = proxy_split(targets, len(labels), seed=args.seed, per_class=4)
        feats = idb["features"]
        protos = torch.zeros(len(labels), feats.shape[1])
        counts = torch.zeros(len(labels))
        for i in proxy:
            protos[targets[i]] += feats[i]
            counts[targets[i]] += 1
        protos = protos / counts[:, None]
        setup = build_setup(t, args, net, labels, protos, log)
        torch.save({"labels": labels, "pos": setup["pos"].cpu(), "proxy": proxy, "test": test,
                    "targets": targets}, out / f"setup_{opts.id}.pt")
        ood = {n: (load_feats(n)["features"], load_feats(n)["ids"]) for n in FOUR}
        run_streams(t, args, net, setup, feats[test], [idb["ids"][i] for i in test], ood, opts.seeds, out, opts.id)

    else:  # l14
        cache = C.WORK / "extra_runs" / "tins_cache_l14"
        args = codex_args(t, cache, "vins_l14", 123)
        args.CLIP_ckpt = "ViT-L/14"
        t.setup_seed(args.seed)
        net, preprocess = load_clip_any(t, args, "ViT-L/14")
        device = next(net.parameters()).device
        protos = t.load_or_build_class_prototypes(args, net, preprocess, log)[0].to(device)
        labels = [str(x) for x in t.get_test_labels(args, None)]
        setup = build_setup(t, args, net, labels, protos, log)
        blob = load_feats("oo_test", "clipl14")
        feats = blob["features"]
        ref = torch.load(C.WORK / "test_eval" / "results" / "features.pt", map_location="cpu")["paths"][16000:]
        assert blob["paths"] == ref
        meta = {}
        for ds in OO:
            group = "nearood" if ds in ("ssb_hard", "ninco") else "farood"
            z = np.load(C.CODEX_TINS / f"scores_openood_{group}_{ds}.npz", allow_pickle=True)
            meta[ds] = (z["sample_id"].astype(str), z["is_ood"].astype(np.int32))
        allids = sorted({s for sid, _ in meta.values() for s in sid})
        row = {s: i for i, s in enumerate(allids)}
        assert len(allids) == feats.shape[0]
        for ds in (opts.datasets or OO):
            sid, is_ood123 = meta[ds]
            n_id, n_ood = int((is_ood123 == 0).sum()), int((is_ood123 == 1).sum())
            prefix = {x.rsplit("_", 1)[0] for x, o in zip(sid, is_ood123) if o == 1}.pop()
            id_ids = [f"imagenet_{i:06d}" for i in range(n_id)]
            o_ids = [f"{prefix}_{i:06d}" for i in range(n_ood)]
            id_f = feats[torch.tensor([row[s] for s in id_ids])]
            o_f = feats[torch.tensor([row[s] for s in o_ids])]
            run_streams(t, args, net, setup, id_f, id_ids, {ds: (o_f, o_ids)}, opts.seeds, out, "l14")
        shutil.copy(__file__, out / "run_tins_extra.py")


if __name__ == "__main__":
    main()
