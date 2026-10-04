"""TINS (pinned upstream, fixed hyper-parameters) on U4, from cached CLIP features (Phase 4 tins_bank.py with U4 paths).
Per (split, draw): ID names + the draw's 16-shot CLIP prototypes -> negative bank and inversion set-up; per order
seed: one stream in the TINS order. Only ID names reach the detector."""
import argparse
import json
import os

import numpy as np
import torch

from common import utc
from u4_bank import Features, shots_of, spec, tins_path
from u4_common import TINS_DIR, require_lock
from vins.tins_dev import Recorder, build_order, get_logger, import_tins, load_clip, make_args, run_stream, setup_to_device


def job(split, draw, seeds):
    TINS_DIR.mkdir(parents=True, exist_ok=True)
    todo = [s for s in seeds if not tins_path(split, draw, s).exists()]
    if not todo:
        return
    F = Features(("CLIP",))
    names, shots, ide, ood, _ = spec(split)
    C = len(names)
    sup_ids, cal_ids = shots_of(shots, draw, C)
    t = import_tins()
    args = make_args(t, TINS_DIR / "cache", f"p5_U4_s{split}_d{draw}")
    t.setup_seed(args.seed)
    log = get_logger(TINS_DIR / f"U4_s{split}_d{draw}.log")
    net, _ = load_clip(t, args)
    device = next(net.parameters()).device
    clip_shots = torch.as_tensor(np.concatenate([F.get("CLIP", sup_ids).reshape(C, 12, -1),
                                                 F.get("CLIP", cal_ids).reshape(C, 4, -1)], 1))
    protos = clip_shots.mean(1)
    protos = (protos / protos.norm(dim=-1, keepdim=True)).to(device)
    positive = t.encode_texts(net, [args.pos_prompt.format(x) for x in names], batch_size=args.text_batch_size,
                              device=device, desc="pos").to(device)
    pos_file = TINS_DIR / f"U4_s{split}_pos.npy"
    pf = (positive / positive.norm(dim=-1, keepdim=True)).float().cpu().numpy()
    if not pos_file.exists():
        tmp = str(pos_file) + f".{os.getpid()}.npy"
        np.save(tmp, pf)
        os.replace(tmp, pos_file)
    neg, _, words, _ = t.load_or_build_negative_bank(args=args, model=net, positive_labels=names, positive_features=positive,
                                                     class_prototypes=protos.cpu(), log=log)
    init = t.build_inversion_init_candidates(args, net, words, protos, device, log)
    setup = {"positive_features": positive.cpu(), "negative_features": neg.cpu(), "class_prototypes": protos.cpu(),
             "base_sim": (positive * protos).sum(dim=1).cpu(),
             "init_candidates": {k: (v.cpu() if torch.is_tensor(v) else v) for k, v in init.items()}}
    dev_setup = setup_to_device(setup)
    for seed in todo:
        order = build_order(len(ide), len(ood), seed)
        ids = [ide[i] if o == 0 else ood[i] for o, i in order]
        feats = torch.as_tensor(F.get("CLIP", ids))
        rec = Recorder()
        args.stream_seed = seed
        t.setup_seed(args.seed)
        scores = run_stream(t, args, net, dev_setup, feats, hook=rec)
        per = rec.per_sample(len(ids))
        assert np.array_equal(per["S_final"], scores)
        out = tins_path(split, draw, seed)
        tmp = str(out) + f".{os.getpid()}.npz"
        np.savez_compressed(tmp, sample_id=np.array(ids), is_ood=np.array([o for o, _ in order], np.int8),
                            S_final=per["S_final"], batch_index=per["batch_index"])
        os.replace(tmp, out)
        print(json.dumps({"tins": out.name, "n": len(ids), "utc": utc()}), flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", required=True, help="comma list split:draw")
    ap.add_argument("--seeds", default="123,124,125")
    a = ap.parse_args()
    require_lock()
    seeds = [int(s) for s in a.seeds.split(",")]
    for j in a.jobs.split(","):
        s, d = j.split(":")
        job(int(s), int(d), seeds)
