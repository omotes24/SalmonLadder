"""DINOv2 ViT-B/14 CLS features (standard eval preprocessing, L2-normalised) for every dev sample."""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402
import torch  # noqa: E402

from vins import config as C  # noqa: E402
from vins.features import dino_encode_fn, dino_transform, encode, guard_from_seal, load_dino, save_features  # noqa: E402


def main():
    start = time.time()
    samples = pd.read_parquet(C.SPLITS_DIR / "samples.parquet")
    guard = guard_from_seal()
    torch.cuda.reset_peak_memory_stats()
    model = load_dino()
    transform = dino_transform()
    feats = encode(samples.path.tolist(), transform, dino_encode_fn(model), guard, batch_size=128,
                   num_workers=8, desc="dino")
    save_features(C.FEATURES_DIR / "dino.pt", samples.sample_id.tolist(), feats,
                  {"model": "DINOv2 ViT-B/14 (torch hub, local)", "preprocess": repr(transform),
                   "embedding": "x_norm_clstoken"})
    timings = {"dino_features_s": round(time.time() - start, 1),
               "cuda_peak_MiB": round(torch.cuda.max_memory_allocated() / 2**20, 1)}
    (C.RUNS_DIR / "setup" / "timings_dino.json").write_text(json.dumps(timings, indent=1) + "\n")
    print(json.dumps(timings))


if __name__ == "__main__":
    main()
