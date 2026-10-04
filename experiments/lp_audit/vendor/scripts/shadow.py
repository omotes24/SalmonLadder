"""Optional: TINS inversion in shadow mode (nothing is added to any bank).

For near novel seeds (nominated & not seeded; m=1, eps=1%, first order seed; at most 500 samples) we run
the unchanged upstream inversion + ID-prototype-separated criterion and compare the pass rate with the
pass rate logged for TINS's own seeds in the same stream. Shadow runs on TINS's own seeds are a control
for the shadow procedure itself.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

from vins import config as C  # noqa: E402
from vins.features import load_features  # noqa: E402
from vins.tins_dev import import_tins, load_clip, make_args, setup_to_device, shadow_inversion  # noqa: E402

NOM = {"clip": "nom_clip_m1_eps1", "dino": "nom_dino_m1_eps1"}


def sample_rows(frame, mask, limit, seed):
    idx = np.flatnonzero(mask)
    if len(idx) > limit:
        idx = np.sort(np.random.default_rng(seed).choice(idx, size=limit, replace=False))
    return idx


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", required=True)
    parser.add_argument("--seed", type=int, default=C.ORDER_SEEDS[0])
    parser.add_argument("--limit", type=int, default=500)
    opts = parser.parse_args()
    out = C.RUNS_DIR / opts.tag
    frame = pd.read_parquet(out / f"stream_near_seed{opts.seed}.parquet")
    stream_feats = out / f"stream_feats_near_seed{opts.seed}.pt"
    if stream_feats.exists():      # exactly the features TINS saw, in stream (= frame) order
        clip = torch.load(stream_feats, map_location="cpu")["features"]
        assert clip.shape[0] == len(frame)
    else:
        clip, _ = load_features(C.FEATURES_DIR / "clip.pt", frame.sample_id.tolist())

    t = import_tins()
    args = make_args(t, out / "tins_cache_shadow", f"vins_{opts.tag}_shadow")
    t.setup_seed(args.seed)
    net, _ = load_clip(t, args)
    dev_setup = setup_to_device(torch.load(C.RUNS_DIR / "setup" / "tins_setup.pt", map_location="cpu"))

    near = (frame.group == "near").values
    ident = (frame.group == "ID").values
    seeded = frame.seeded.values
    logged = frame.inv_pass.values
    result = {"order_seed": opts.seed, "limit": opts.limit, "config": "m=1, eps=1%",
              "tins_own_seeds_logged": {
                  "all": {"n": int(seeded.sum()), "pass_rate": float((logged[seeded] == 1).mean()) if seeded.any() else None},
                  "near": {"n": int((seeded & near).sum()),
                           "pass_rate": float((logged[seeded & near] == 1).mean()) if (seeded & near).any() else None},
                  "ID": {"n": int((seeded & ident).sum()),
                         "pass_rate": float((logged[seeded & ident] == 1).mean()) if (seeded & ident).any() else None},
              }}

    def shadow(mask, salt):
        idx = sample_rows(frame, mask, opts.limit, salt)
        if len(idx) == 0:
            return {"n": 0, "pass_rate": None}
        passes, deltas = shadow_inversion(t, args, net, dev_setup, clip[torch.from_numpy(idx)])
        return {"n": int(len(idx)), "pass_rate": float(passes.mean()), "median_delta": float(np.median(deltas))}

    result["shadow_on_tins_own_seeds"] = {"near": shadow(seeded & near, 11), "ID": shadow(seeded & ident, 12)}
    result["shadow_on_near_novel"] = {}
    for feat, col in NOM.items():
        novel = near & frame[col].values & ~seeded
        result["shadow_on_near_novel"][feat] = {"n_novel_total": int(novel.sum()), **shadow(novel, 13)}
    (out / "shadow.json").write_text(json.dumps(result, indent=1) + "\n")
    print(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
