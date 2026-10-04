"""Build the 900-class TINS inputs (upstream functions) and CLIP features for every dev sample.

Outputs: runs/setup/tins_setup.pt, runs/setup/static_negative_leak_check.json,
         features/clip.pt, features/zeroshot.parquet, runs/setup/timings.json
"""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402
import torch  # noqa: E402

from vins import config as C  # noqa: E402
from vins.features import encode, guard_from_seal, save_features  # noqa: E402
from vins.tins_dev import get_logger, import_tins, leak_check, load_clip, make_args  # noqa: E402


def main():
    timings, start = {}, time.time()
    out = C.RUNS_DIR / "setup"
    out.mkdir(parents=True, exist_ok=True)
    C.FEATURES_DIR.mkdir(parents=True, exist_ok=True)
    samples = pd.read_parquet(C.SPLITS_DIR / "samples.parquet")
    id_classes = json.loads((C.SPLITS_DIR / "id_classes.json").read_text())
    heldout = json.loads((C.SPLITS_DIR / "heldout.json").read_text())
    guard = guard_from_seal()

    cache_dir = out / "tins_cache"
    if (cache_dir / "negative_bank").exists():
        raise RuntimeError("negative-bank cache exists; its key ignores the label set, refusing to reuse it")
    t = import_tins()
    args = make_args(t, cache_dir, "vins_gonogo_setup")
    t.setup_seed(args.seed)
    log = get_logger(out / "setup.log")
    torch.cuda.reset_peak_memory_stats()
    net, preprocess = load_clip(t, args)
    device = next(net.parameters()).device

    # Prototypes: upstream code on the full TINS 16-shot list (1000 classes); keep the 900 ID rows.
    with open(args.train_imglist) as handle:
        for line in handle:
            if line.strip():
                guard.check(Path(args.root_dir) / "ImageNet" / line.split()[0])
    protos_1000, proto_meta, proto_cache = t.load_or_build_class_prototypes(args, net, preprocess, log)
    rows = torch.tensor([c["idx_1k"] for c in id_classes])
    protos = protos_1000[rows].to(device)
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
    leak = leak_check(t, args, labels, neg_words, neg_texts, heldout)
    (out / "static_negative_leak_check.json").write_text(json.dumps(leak, indent=1) + "\n")
    torch.save({
        "positive_labels": labels,
        "id_wnids": [c["wnid"] for c in id_classes],
        "positive_features": positive_features.cpu(),
        "class_prototypes": protos.cpu(),
        "base_sim": base_sim.cpu(),
        "negative_features": negative_features.cpu(),
        "selected_negative_texts": neg_texts,
        "selected_negative_words": neg_words,
        "init_candidates": {k: (v.cpu() if torch.is_tensor(v) else v) for k, v in init.items()},
        "meta": {"prototype_meta": proto_meta, "prototype_cache": str(proto_cache),
                 "negative_cache": str(neg_cache), "logit_scale": float(net.logit_scale.exp().item()),
                 "model_dtype": str(net.dtype), "preprocess": repr(preprocess), "args": vars(args)},
    }, out / "tins_setup.pt")
    timings["tins_setup_s"] = round(time.time() - start, 1)

    # CLIP image features (TINS preprocessing, L2-normalised) for every dev sample.
    tick = time.time()
    feats = encode(samples.path.tolist(), preprocess, net.encode_image, guard, batch_size=256,
                   num_workers=8, desc="clip")
    save_features(C.FEATURES_DIR / "clip.pt", samples.sample_id.tolist(), feats,
                  {"model": "CLIP ViT-B/16 (openai, TINS loader)", "preprocess": repr(preprocess)})
    timings["clip_features_s"] = round(time.time() - tick, 1)

    # Zero-shot over the 900 ID classes with the TINS positive text features ("The nice {}.").
    with torch.no_grad():
        sims = feats.to(device) @ positive_features.T
        top = sims.topk(C.K_TOP, dim=1).indices.cpu().numpy()
    zeroshot = pd.DataFrame({"sample_id": samples.sample_id, "zs_top1_id": top[:, 0],
                             "K_id": [list(map(int, r)) for r in top]})
    zeroshot.to_parquet(C.FEATURES_DIR / "zeroshot.parquet", index=False)
    timings["cuda_peak_MiB"] = round(torch.cuda.max_memory_allocated() / 2**20, 1)
    timings["total_s"] = round(time.time() - start, 1)
    (out / "timings.json").write_text(json.dumps(timings, indent=1) + "\n")
    print(json.dumps({"timings": timings, "n_samples": len(samples),
                      "leak_heldout_in_selected": leak["n_heldout_with_name_in_selected_static_negatives"],
                      "leak_heldout_in_pool": leak["n_heldout_with_name_in_candidate_pool"]}, indent=1))


if __name__ == "__main__":
    main()
