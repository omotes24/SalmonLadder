"""TINS on the OpenOOD v1.5 ImageNet-1K test streams for several order seeds and diagnostic variants.

The frozen methods are not changed; this only produces the TINS side for the uncertainty / diagnostic analyses.
Setup (prototypes, static negatives, inversion init) and stream features come from Codex's unchanged-upstream
reproduction (arguments.json; its cache is copied, the source is never written). Stream features are cached in
seed-123 order; a per-image feature does not depend on the order, so every other order is a permutation of rows.
Saved per (variant, stream, seed): sample_id, is_ood, S_arrival, S_final, seeded, inv_pass, inv_delta, batch index,
bank size per batch, and per-batch TINS scores of the 4,000 calibration shots under the negatives of S_final (p_T).
Variants: default | permfirst (all negatives permuted before the tail is dropped, as the paper describes)
Streams : ssb_hard ninco inaturalist textures openimageo | near_all (ID+SSB-hard+NINCO) | far_all (ID+iNat+Tex+OIO)
"""
import argparse
import glob
import json
import random
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402
from torch.utils.data import DataLoader, Dataset  # noqa: E402

from vins import config as C  # noqa: E402
from vins.tins_dev import Recorder, get_logger, import_tins, load_clip  # noqa: E402

CODEX = C.CODEX_TINS
GROUP = {"ssb_hard": "nearood", "ninco": "nearood", "inaturalist": "farood", "textures": "farood", "openimageo": "farood"}
COMBINED = {"near_all": ["ssb_hard", "ninco"], "far_all": ["inaturalist", "textures", "openimageo"]}


def codex_args(t, cache_dir, name, stream_seed):
    argv = list(json.loads((CODEX / "arguments.json").read_text()))
    argv[argv.index("--cache-dir") + 1] = str(cache_dir)
    argv[argv.index("--name") + 1] = name
    argv[argv.index("--stream-seed") + 1] = str(stream_seed)
    argv = [a for a in argv if a != "--save-stream-scores"] + ["--no-image-feature-cache"]
    saved = sys.argv
    sys.argv = ["eval_tins_w_init.py"] + argv
    try:
        return t.process_args()
    finally:
        sys.argv = saved


class Shots(Dataset):
    def __init__(self, paths, transform):
        self.paths, self.transform = paths, transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        return self.transform(Image.open(self.paths[i]).convert("RGB")), i


@torch.no_grad()
def encode_like_upstream(net, paths, preprocess):
    device = next(net.parameters()).device
    out = []
    for images, _ in DataLoader(Shots(paths, preprocess), batch_size=256, shuffle=False, num_workers=8):
        f = net.encode_image(images.to(device)).float()
        out.append((f / f.norm(dim=-1, keepdim=True)).cpu())
    return torch.cat(out)


def permfirst_score(image_features, positive_features, negative_features, logit_scale, group_num, random_permute):
    """compute_grouped_positive_score with the permutation applied BEFORE the tail is dropped."""
    pos_logits = logit_scale * (image_features @ positive_features.T)
    neg_logits = logit_scale * (image_features @ negative_features.T)
    if random_permute:
        torch.manual_seed(0)
        torch.cuda.manual_seed(0)
        neg_logits = neg_logits[:, torch.randperm(neg_logits.shape[1], device=image_features.device)]
    drop = neg_logits.shape[1] % group_num
    if drop > 0:
        neg_logits = neg_logits[:, :-drop]
    grouped = neg_logits.reshape(pos_logits.shape[0], group_num, -1)
    log_c = torch.log(torch.tensor(float(pos_logits.shape[1]), device=pos_logits.device, dtype=pos_logits.dtype))
    log_pos = torch.logsumexp(pos_logits, dim=-1)
    scores = []
    for g in range(group_num):
        grp = grouped[:, g, :]
        log_neg = torch.logsumexp(grp, dim=-1) - np.log(float(grp.shape[1])) + log_c
        scores.append(torch.exp(log_pos - torch.logaddexp(log_pos, log_neg)).unsqueeze(-1))
    return torch.cat(scores, dim=-1).mean(dim=-1)


class TestRecorder(Recorder):
    """Recorder + TINS scores of the calibration shots under the negatives that produced S_final."""

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


def load_stream_cache(ds):
    z = np.load(CODEX / f"scores_openood_{GROUP[ds]}_{ds}.npz", allow_pickle=True)
    path = glob.glob(str(CODEX / "cache" / "image_features" / f"imgfeat_openood_{GROUP[ds]}_{ds}_stream_*.pt"))
    assert len(path) == 1, path
    blob = torch.load(path[0], map_location="cpu")
    assert np.array_equal(blob["is_ood"].numpy(), z["is_ood"]), ds
    return z["sample_id"].astype(str), z["is_ood"].astype(np.int32), blob["image_features"], z["ID_score"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=["default", "permfirst"], default="default")
    parser.add_argument("--seeds", type=int, nargs="+", required=True)
    parser.add_argument("--streams", nargs="+", default=list(GROUP))
    parser.add_argument("--feature-file", default=None, help="per-image features (sorted OpenOOD test ids) replacing Codex's")
    parser.add_argument("--tag", default=None, help="output sub-directory (default: the variant)")
    opts = parser.parse_args()
    out = C.WORK / "test_runs" / (opts.tag or opts.variant)
    out.mkdir(parents=True, exist_ok=True)
    cache = C.WORK / "test_runs" / "tins_cache"
    for sub in ("class_prototypes", "negative_bank", "inversion_init"):
        if not (cache / sub).exists():
            shutil.copytree(CODEX / "cache" / sub, cache / sub)
    t = import_tins()
    args = codex_args(t, cache, "vins_test_runs", 123)
    t.setup_seed(args.seed)
    log = get_logger(out / "run.log")
    net, preprocess = load_clip(t, args)
    device = next(net.parameters()).device
    protos = t.load_or_build_class_prototypes(args, net, preprocess, log)[0].to(device)
    labels = [str(x) for x in t.get_test_labels(args, None)]
    pos = t.encode_texts(net, [args.pos_prompt.format(l) for l in labels], batch_size=args.text_batch_size,
                         device=device, desc="pos").to(device)
    base_sim = (pos * protos).sum(dim=1)
    neg, _, words, _ = t.load_or_build_negative_bank(args=args, model=net, positive_labels=labels, positive_features=pos,
                                                     class_prototypes=protos.cpu(), log=log)
    init = t.build_inversion_init_candidates(args, net, words, protos, device, log)

    shots = [line.split() for line in C.PROTO_LIST_SRC.read_text().splitlines() if line.strip()]
    by_label = {}
    for rel, label in shots:
        by_label.setdefault(int(label), []).append(str(C.IMAGENET_ROOT / rel))
    cal_paths = [p for c in range(1000) for p in by_label[c][C.N_SUPPORT:]]
    cal_file = C.WORK / "test_runs" / "cal_clip.pt"
    if cal_file.exists():
        cal_feats = torch.load(cal_file)["features"]
    else:
        cal_feats = encode_like_upstream(net, cal_paths, preprocess)
        torch.save({"paths": cal_paths, "features": cal_feats}, cal_file)
    cal_feats = cal_feats.to(device)

    orig_score = t.compute_grouped_positive_score
    score_fn = permfirst_score if opts.variant == "permfirst" else orig_score
    capture = {"last": None}

    def wrapped(**kw):
        capture["last"] = kw
        return score_fn(**kw)

    t.compute_grouped_positive_score = wrapped

    cached = {}

    def get(ds):
        if ds not in cached:
            cached[ds] = load_stream_cache(ds)
            if opts.feature_file:
                sid, is_ood, _, s_codex = cached[ds]
                if "table" not in cached:
                    fb = torch.load(opts.feature_file, map_location="cpu")
                    ref = torch.load(C.WORK / "test_eval" / "results" / "features.pt", map_location="cpu")["paths"][16000:]
                    assert fb["paths"] == ref
                    allids = sorted({x for d in GROUP for x in np.load(
                        CODEX / f"scores_openood_{GROUP[d]}_{d}.npz", allow_pickle=True)["sample_id"].astype(str)})
                    cached["table"] = (fb["features"], {x: i for i, x in enumerate(allids)})
                feats_all, row = cached["table"]
                cached[ds] = (sid, is_ood, feats_all[torch.tensor([row[x] for x in sid])], s_codex)
        return cached[ds]

    summary = {}
    for stream in opts.streams:
        members = COMBINED.get(stream, [stream])
        # per-image feature tables keyed by sample id
        feat_of, id_rows, ood_ids = {}, None, []
        for ds in members:
            sid, is_ood, feats, _ = get(ds)
            ood_j = np.flatnonzero(is_ood == 1)
            for j in ood_j:
                feat_of[sid[j]] = feats[j]
            ood_ids += sorted(sid[ood_j].tolist())
            if id_rows is None:
                id_rows = {sid[j]: feats[j] for j in np.flatnonzero(is_ood == 0)}
        id_ids = sorted(id_rows)
        assert len(ood_ids) == len(feat_of) == len(set(ood_ids)), stream
        for seed in opts.seeds:
            if len(members) == 1:
                sid123, is_ood123, feats123, s_codex = get(stream)
                n_id, n_ood = int((is_ood123 == 0).sum()), int((is_ood123 == 1).sum())
                # upstream order: (0, i) -> imagenet_{i}, (1, i) -> <ds>_{i}; recover i from ids
                order = [(0, i) for i in range(n_id)] + [(1, i) for i in range(n_ood)]
                random.Random(seed).shuffle(order)
                prefix = {x.rsplit("_", 1)[0] for x, o in zip(sid123, is_ood123) if o == 1}.pop()
                ids = [f"imagenet_{i:06d}" if o == 0 else f"{prefix}_{i:06d}" for o, i in order]
                if seed == 123:
                    assert ids == list(sid123), f"{stream}: reconstructed seed-123 order differs from Codex's"
                row = {s: k for k, s in enumerate(sid123)}
                idx = torch.tensor([row[s] for s in ids])
                feats, is_ood = feats123[idx], np.array([o for o, _ in order], dtype=np.int32)
            else:
                order = [(0, s) for s in id_ids] + [(1, s) for s in ood_ids]
                random.Random(seed).shuffle(order)
                ids = [s for _, s in order]
                is_ood = np.array([o for o, _ in order], dtype=np.int32)
                feats = torch.stack([id_rows[s] if o == 0 else feat_of[s] for o, s in order])
                s_codex = None
            args.stream_seed = seed
            rec = TestRecorder(capture, cal_feats, score_fn)
            tick = time.time()
            scores = t.compute_tins_scores_from_image_features(
                image_features=feats, args=args, model=net, positive_features=pos, negative_features=neg.to(device),
                inversion_init_candidates=init, class_prototypes=protos, base_sim=base_sim, hook=rec)
            per = rec.per_sample(len(ids))
            assert np.array_equal(per["S_final"], scores)
            info = {"n": len(ids), "seconds": round(time.time() - tick, 1), "n_seeded": int(per["seeded"].sum())}
            if s_codex is not None and seed == 123 and opts.variant == "default" and not opts.feature_file:
                info["bitwise_equal_codex"] = bool(np.array_equal(scores, s_codex))
                info["max_abs_diff_codex"] = float(np.abs(scores - s_codex).max())
            np.savez_compressed(out / f"{stream}_seed{seed}.npz", sample_id=np.array(ids), is_ood=is_ood,
                                S_arrival=per["S_arrival"], S_final=per["S_final"], seeded=per["seeded"],
                                inv_pass=per["inv_pass"], inv_delta=per["inv_delta"], batch_index=per["batch_index"],
                                bank_size=np.array([b["bank_size"] for b in rec.batches]),
                                cal_scores=np.stack(rec.cal_scores).astype(np.float32))
            summary[f"{stream}_seed{seed}"] = info
            print(json.dumps({f"{opts.variant}/{stream}_seed{seed}": info}), flush=True)
    (out / f"summary_{'_'.join(map(str, opts.seeds))}_{'_'.join(opts.streams)}.json").write_text(
        json.dumps(summary, indent=1) + "\n")


if __name__ == "__main__":
    main()
