"""Iteration 2 (dev only): TINS on the dev streams with alternative seed rules + calibration TINS scores (p_T).

The upstream functions are unchanged. The existing on_activate hook receives TINS's activation mask
(S_arrival < beta) as a tensor; a rule other than "default" edits that mask in place before inversion:
  default          : S_arrival < beta                                   (upstream)
  gate:E           : (S_arrival < beta) AND p_all <= E                  (visual gate on TINS's seeds)
  vis:E            : (S_arrival < beta) OR  p_all <= E                  (extra visual seeds)
  gate:E1+vis:E2   : ((S_arrival < beta) AND p_all <= E1) OR p_all <= E2
p_all is the frozen conformal p-value of d_all (min over all ID classes), fixed before the stream.
Per batch, the 3,600 calibration shots are scored under the negatives that produced S_final (for p_T).
Stream features are the upstream-path features saved by run_tins.py (runs/main/stream_feats_*.pt).
Output: <WORK>/iter2/tins/<rule-tag>/<stream>_seed<seed>.npz
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

from vins import config as C  # noqa: E402
from vins.clavism import dev_views  # noqa: E402
from vins.features import load_features  # noqa: E402
from vins.tins_dev import Recorder, import_tins, load_clip, make_args, run_stream, setup_to_device  # noqa: E402


def parse_rule(rule):
    gate = vis = None
    if rule != "default":
        for part in rule.split("+"):
            key, val = part.split(":")
            if key == "gate":
                gate = float(val)
            elif key == "vis":
                vis = float(val)
            else:
                raise ValueError(rule)
    return gate, vis


def rule_tag(rule):
    return rule.replace(":", "").replace("+", "_").replace(".", "p")


class GateRecorder(Recorder):
    def __init__(self, capture, cal_feats, score_fn, keep=None, add=None):
        super().__init__()
        self.capture, self.cal, self.score_fn, self.cal_scores = capture, cal_feats, score_fn, []
        self.keep, self.add = keep, add
        self._sl = None

    def on_arrival(self, start, scores):
        super().on_arrival(start, scores)
        self._sl = slice(int(start), int(start) + int(scores.shape[0]))

    def on_activate(self, mask):
        if self.keep is not None:
            mask.logical_and_(torch.from_numpy(self.keep[self._sl]).to(mask.device))
        if self.add is not None:
            mask.logical_or_(torch.from_numpy(self.add[self._sl]).to(mask.device))
        super().on_activate(mask)

    def on_final(self, start, scores, bank_size, buffer_size):
        kw = self.capture["last"]
        with torch.no_grad():
            cal = self.score_fn(image_features=self.cal, positive_features=kw["positive_features"],
                                negative_features=kw["negative_features"], logit_scale=kw["logit_scale"],
                                group_num=kw["group_num"], random_permute=kw["random_permute"])
        self.cal_scores.append(cal.float().cpu().numpy())
        super().on_final(start, scores, bank_size, buffer_size)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rules", nargs="+", required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(C.ORDER_SEEDS))
    parser.add_argument("--streams", nargs="+", default=["near", "far"])
    parser.add_argument("--check-main", action="store_true", help="assert default S_final == runs/main (bitwise)")
    opts = parser.parse_args()

    samples = pd.read_parquet(C.SPLITS_DIR / "samples.parquet")
    dview = pd.read_parquet(C.RUNS_DIR / "dview" / "dview.parquet")
    v = dev_views(m=2)
    p_all_of = v["frame"].p_all
    clip, blob = load_features(C.FEATURES_DIR / "clip.pt")
    assert blob["sample_id"] == samples.sample_id.tolist()
    row_of = pd.Series(np.arange(len(samples)), index=samples.sample_id.values)
    cal_ids = dview.sample_id.values[(dview.split == "calib").values]          # dview order (= visual cal order)
    t = import_tins()
    args = make_args(t, C.WORK / "iter2" / "tins_cache", "vins_iter2")
    t.setup_seed(args.seed)
    net, _ = load_clip(t, args)
    dev_setup = setup_to_device(torch.load(C.RUNS_DIR / "setup" / "tins_setup.pt", map_location="cpu"))
    device = dev_setup["positive_features"].device
    cal_feats = clip[torch.from_numpy(row_of.loc[cal_ids].values)].to(device)

    orig = t.compute_grouped_positive_score
    capture = {"last": None}

    def wrapped(**kw):
        capture["last"] = kw
        return orig(**kw)

    t.compute_grouped_positive_score = wrapped
    summary = {}
    for rule in opts.rules:
        gate, vis = parse_rule(rule)
        out = C.WORK / "iter2" / "tins" / rule_tag(rule)
        out.mkdir(parents=True, exist_ok=True)
        for stream in opts.streams:
            for seed in opts.seeds:
                blobs = torch.load(C.RUNS_DIR / "main" / f"stream_feats_{stream}_seed{seed}.pt", map_location="cpu")
                feats, rows, flags = blobs["features"], np.asarray(blobs["rows"]), np.asarray(blobs["flags"])
                sid = samples.sample_id.values[rows]
                p_all = p_all_of.loc[sid].values
                keep = (p_all <= gate) if gate is not None else None
                add = (p_all <= vis) if vis is not None else None
                rec = GateRecorder(capture, cal_feats, orig, keep=keep, add=add)
                args.stream_seed = seed
                tick = time.time()
                scores = run_stream(t, args, net, dev_setup, feats, hook=rec)
                per = rec.per_sample(len(rows))
                assert np.array_equal(per["S_final"], scores)
                info = {"n": int(len(rows)), "seconds": round(time.time() - tick, 1),
                        "n_seeded": int(per["seeded"].sum()),
                        "seeded_ID": int(per["seeded"][flags == 0].sum()), "seeded_OOD": int(per["seeded"][flags == 1].sum())}
                if rule == "default" and opts.check_main:
                    ref = pd.read_parquet(C.RUNS_DIR / "main" / f"stream_{stream}_seed{seed}.parquet").sort_values("position")
                    assert (ref.sample_id.values == sid).all()
                    info["bitwise_equal_main"] = bool(np.array_equal(ref.S_final.values.astype(np.float32), per["S_final"]))
                np.savez_compressed(out / f"{stream}_seed{seed}.npz", sample_id=sid, is_ood=flags,
                                    S_arrival=per["S_arrival"], S_final=per["S_final"], seeded=per["seeded"],
                                    inv_pass=per["inv_pass"], inv_delta=per["inv_delta"], batch_index=per["batch_index"],
                                    bank_size=np.array([b["bank_size"] for b in rec.batches]),
                                    cal_scores=np.stack(rec.cal_scores).astype(np.float32), cal_ids=cal_ids)
                summary[f"{rule}|{stream}_seed{seed}"] = info
                print(json.dumps({f"{rule}|{stream}_seed{seed}": info}), flush=True)
    tag = "_".join(rule_tag(r) for r in opts.rules)
    (C.WORK / "iter2" / "tins" / f"summary_{tag}.json").write_text(json.dumps(summary, indent=1) + "\n")


if __name__ == "__main__":
    main()
