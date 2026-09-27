"""Regression: pristine upstream (194759d) vs patched module with hook=None and with a Recorder.

Usage: python -m vins.regression default|deterministic
Two configurations are checked on real dev features (smoke near stream, order seed 123):
  production  : Codex's hyper-parameters (bank 2000, buffer 2000)
  small_bank  : bank 4, buffer 4 so that overflow and Flash are exercised within a few batches
Writes runs/regression/result_<mode>.json.
"""
import hashlib
import json
import os
import subprocess
import sys

MODE = sys.argv[1] if len(sys.argv) > 1 else "default"
if MODE == "deterministic":
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

from . import config as C  # noqa: E402
from .features import load_features  # noqa: E402
from .tins_dev import Recorder, build_order, import_tins, load_clip, make_args, setup_to_device  # noqa: E402

PRISTINE_SHA256 = "018ffecba19a14abdde2a370556170a7e56f0e32eda82b72c69d36c70ca16449"


def call(module, args, net, dev, feats, hook="absent"):
    kwargs = dict(image_features=feats, args=args, model=net,
                  positive_features=dev["positive_features"], negative_features=dev["negative_features"],
                  inversion_init_candidates=dev["init_candidates"], class_prototypes=dev["class_prototypes"],
                  base_sim=dev["base_sim"])
    if hook != "absent":
        kwargs["hook"] = hook
    return module.compute_tins_scores_from_image_features(**kwargs)


def main():
    if MODE == "deterministic":
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)
        torch.use_deterministic_algorithms(True)
    out = C.RUNS_DIR / "regression"
    out.mkdir(parents=True, exist_ok=True)
    source = subprocess.check_output(["git", "show", f"{C.TINS_COMMIT}:eval_tins_w_init.py"], cwd=C.TINS_DIR)
    assert hashlib.sha256(source).hexdigest() == PRISTINE_SHA256
    (C.TINS_DIR / "_pristine_eval_tins_w_init.py").write_bytes(source)

    samples = pd.read_parquet(C.SPLITS_DIR / "samples.parquet")
    clip, blob = load_features(C.FEATURES_DIR / "clip.pt")
    id_rows = np.flatnonzero((samples.split.values == "id_dev") & samples.smoke.values)
    near_rows = np.flatnonzero((samples.split.values == "near_dev") & samples.smoke.values)
    order = build_order(len(id_rows), len(near_rows), 123)
    rows = np.array([id_rows[i] if o == 0 else near_rows[i] for o, i in order])
    feats = clip[torch.from_numpy(rows)]

    patched = import_tins("eval_tins_w_init.py", "eval_tins_w_init")
    pristine = import_tins("_pristine_eval_tins_w_init.py", "tins_pristine")
    assert patched.official_clip is pristine.official_clip
    dev = setup_to_device(torch.load(C.RUNS_DIR / "setup" / "tins_setup.pt", map_location="cpu"))
    result = {"mode": MODE, "configs": []}
    net = None
    for name, bank, buffer in (("production", 2000, 2000), ("small_bank", 4, 4)):
        args = make_args(patched, out / "cache", f"vins_regression_{MODE}")
        args.extra_text_length, args.bank_buffer_size = bank, buffer
        if net is None:
            net, _ = load_clip(patched, args)
        p1 = call(pristine, args, net, dev, feats)
        p2 = call(pristine, args, net, dev, feats)
        off = call(patched, args, net, dev, feats, hook=None)
        recorder = Recorder()
        on = call(patched, args, net, dev, feats, hook=recorder)
        per = recorder.per_sample(len(rows))
        trace = recorder.bank_trace()
        flashes, prev_bank, prev_buf = 0, 0, 0
        for item in trace:        # a Flash is the only way the buffer can differ from prev + overflow
            overflow = max(0, prev_bank + item["n_pass"] - bank)
            flashes += int(item["buffer_size"] != prev_buf + overflow)
            prev_bank, prev_buf = item["bank_size"], item["buffer_size"]
        result["configs"].append({
            "name": name, "n": int(len(rows)), "bank": bank, "buffer": buffer,
            "pristine_repeatable": bool(np.array_equal(p1, p2)),
            "off_equals_pristine": bool(np.array_equal(p1, off)),
            "on_equals_off": bool(np.array_equal(off, on)),
            "max_abs_diff": {"p1_p2": float(np.abs(p1 - p2).max()), "p1_off": float(np.abs(p1 - off).max()),
                             "off_on": float(np.abs(off - on).max())},
            "recorder_final_matches": bool(np.array_equal(per["S_final"], on)),
            "recorder_seeded_matches": bool(np.array_equal(per["seeded"], per["S_arrival"] < np.float32(args.ood_threshold))),
            "n_seeded": int(per["seeded"].sum()),
            "n_batches_updated": int(sum(1 for b in recorder.batches if "inv_pass" in b and b["inv_pass"].any())),
            "n_flash": int(flashes),
        })
    (out / f"result_{MODE}.json").write_text(json.dumps(result, indent=1) + "\n")
    print(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
