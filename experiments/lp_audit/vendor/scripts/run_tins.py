"""Step 3: run TINS (unchanged, logging hooks only) on (ID dev + near dev) and (ID dev + far dev) streams.

Usage: run_tins.py --tag smoke --seeds 123                       (smoke subset)
       run_tins.py --tag main  --seeds 123 124 125               (full dev)
--features upstream (default): TINS input features are encoded by upstream's own stream path
    (ImageListDataset -> build_mixed_stream_loader -> load_or_cache_stream_features_and_gt), in stream order.
--features ours: per-sample features from features/clip.pt (fp16-rounding-level differences).
Outputs per (stream, seed): stream_<stream>_seed<seed>.parquet, bank_*.json, stream_feats_*.pt
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from vins import config as C  # noqa: E402
from vins.features import guard_from_seal, load_features  # noqa: E402
from vins.tins_dev import (Recorder, build_order, get_logger, import_tins, load_clip, make_args,  # noqa: E402
                           run_stream, setup_to_device)

KEEP = ["sample_id", "group", "wnid", "class_name", "class_idx_id", "near_parents", "near_id_siblings"]


def stream_members(samples, tag):
    use = samples.smoke.values if tag.startswith("smoke") else np.ones(len(samples), dtype=bool)
    pick = lambda split: np.flatnonzero((samples.split.values == split) & use)  # noqa: E731
    return pick("id_dev"), {"near": pick("near_dev"), "far": pick("far_dev")}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", required=True)
    parser.add_argument("--seeds", type=int, nargs="+", required=True)
    parser.add_argument("--streams", nargs="+", default=["near", "far"])
    parser.add_argument("--features", choices=["upstream", "ours"], default="upstream")
    opts = parser.parse_args()

    out = C.RUNS_DIR / opts.tag
    out.mkdir(parents=True, exist_ok=True)
    samples = pd.read_parquet(C.SPLITS_DIR / "samples.parquet")
    dview = pd.read_parquet(C.RUNS_DIR / "dview" / "dview.parquet").set_index("sample_id")
    clip, blob = load_features(C.FEATURES_DIR / "clip.pt")
    assert blob["sample_id"] == samples.sample_id.tolist()
    id_rows, ood_rows = stream_members(samples, opts.tag)

    t = import_tins()
    args = make_args(t, out / "tins_cache", f"vins_{opts.tag}")
    assert args.no_image_feature_cache          # upstream's stream cache key ignores the order seed
    t.setup_seed(args.seed)
    log = get_logger(out / "run_tins.log")
    net, preprocess = load_clip(t, args)
    setup = torch.load(C.RUNS_DIR / "setup" / "tins_setup.pt", map_location="cpu")
    dev_setup = setup_to_device(setup)

    loaders = {}
    if opts.features == "upstream":
        guard = guard_from_seal()
        lists = out / "lists"
        lists.mkdir(exist_ok=True)

        def make_loader(name, rows):
            path = lists / f"{name}.txt"
            with open(path, "w") as handle:     # absolute (already verified) paths; os.path.join keeps them
                for r in rows:
                    handle.write(f"{guard.check(samples.path.iloc[r])} {int(samples.class_idx_1k.iloc[r])}\n")
            dataset = t.ImageListDataset("/", path, preprocess)
            return DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

        loaders["id"] = make_loader("id_dev", id_rows)
        for stream in opts.streams:
            loaders[stream] = make_loader(f"{stream}_dev", ood_rows[stream])

    timings = {}
    for stream in opts.streams:
        for seed in opts.seeds:
            order = build_order(len(id_rows), len(ood_rows[stream]), seed)
            flags = np.array([o for o, _ in order], dtype=np.int32)
            rows = np.array([id_rows[i] if is_ood == 0 else ood_rows[stream][i] for is_ood, i in order])
            tick = time.time()
            if opts.features == "upstream":
                args.stream_seed = seed
                name = f"vins_{opts.tag}_{stream}_seed{seed}"
                mixed = t.build_mixed_stream_loader(args, loaders["id"], loaders[stream], name)
                assert mixed.dataset.order == order                 # identical to our bookkeeping
                feats, is_ood = t.load_or_cache_stream_features_and_gt(net, mixed, args, log, name)
                assert np.array_equal(is_ood, flags)
            else:
                feats = clip[torch.from_numpy(rows)]
            encode_seconds = time.time() - tick
            feat_diff = float((feats - clip[torch.from_numpy(rows)]).abs().max())
            torch.save({"features": feats, "rows": rows, "flags": flags, "feature_path": opts.features},
                       out / f"stream_feats_{stream}_seed{seed}.pt")
            recorder = Recorder()
            torch.cuda.reset_peak_memory_stats()
            tick = time.time()
            scores = run_stream(t, args, net, dev_setup, feats, hook=recorder)
            seconds = time.time() - tick
            per = recorder.per_sample(len(rows))
            assert np.array_equal(per["S_final"], scores), "recorder S_final != returned TINS scores"
            assert np.array_equal(per["seeded"], per["S_arrival"] < np.float32(args.ood_threshold))
            frame = samples.iloc[rows][KEEP].reset_index(drop=True)
            frame.insert(0, "position", np.arange(len(rows)))
            frame.insert(0, "order_seed", seed)
            frame.insert(0, "stream", stream)
            for key, values in per.items():
                frame[key] = values
            joined = dview.loc[frame.sample_id].reset_index(drop=True)
            extra = [c for c in joined.columns if c not in frame.columns]
            frame = pd.concat([frame, joined[extra]], axis=1)
            frame.to_parquet(out / f"stream_{stream}_seed{seed}.parquet", index=False)
            (out / f"bank_{stream}_seed{seed}.json").write_text(json.dumps(recorder.bank_trace()) + "\n")
            timings[f"{stream}_seed{seed}"] = {
                "n": int(len(rows)), "batches": len(recorder.batches), "seconds": round(seconds, 1),
                "sec_per_batch": round(seconds / max(1, len(recorder.batches)), 2),
                "encode_seconds": round(encode_seconds, 1), "feature_path": opts.features,
                "feature_max_abs_diff_vs_dview": feat_diff,
                "cuda_peak_MiB": round(torch.cuda.max_memory_allocated() / 2**20, 1),
                "n_seeded": int(per["seeded"].sum()),
            }
            print(json.dumps({f"{stream}_seed{seed}": timings[f"{stream}_seed{seed}"]}), flush=True)
    (out / "timings_tins.json").write_text(json.dumps(timings, indent=1) + "\n")


if __name__ == "__main__":
    main()
