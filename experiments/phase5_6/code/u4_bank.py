"""Inputs of the unused bank U4 (Phase 5): class lists, shots, ordered streams (TINS order rule), candidate sets
K(x) and MCM. Mirrors Phase 4's bank_data.py for U1. Evaluation labels leave this module only as is_ood flags."""
import json

import numpy as np
import pandas as pd
import torch

from u4_common import BANKS, FEATS, TINS_DIR
from vins.tins_dev import build_order


class Features:
    def __init__(self, names=("B14", "L14", "CLIP")):
        t = pd.read_parquet(BANKS / "feature_table.parquet")
        self.row = dict(zip(t.sample_id, range(len(t))))
        self.F = {n: np.load(FEATS / f"{n}.npy", mmap_mode="r") for n in names}

    def get(self, name, ids):
        return np.asarray(self.F[name][np.array([self.row[s] for s in ids])], np.float32)


def spec(split):
    pool = pd.read_parquet(BANKS / "imagenet_pool.parquet")
    cls = pool.drop_duplicates("wnid").sort_values("idx_1k")
    cname = dict(zip(cls.wnid, cls.clean_name))
    sp = json.loads((BANKS / "U4" / f"split{split}.json").read_text())
    idw, held = sp["id_wnids"], set(sp["heldout"])
    rank = {w: i for i, w in enumerate(idw)}
    names = [cname[w] for w in idw]
    shots = pool[(pool.role == "shot") & pool.wnid.isin(rank)].copy()
    shots["class_rank"] = shots.wnid.map(rank)
    ide = pool[(pool.role == "ideval") & pool.wnid.isin(rank)].sort_values(["idx_1k", "slot"])
    ood = pool[(pool.role == "oodeval") & pool.wnid.isin(held)].sort_values(["idx_1k", "slot"])
    wn = dict(zip(pool.sample_id, pool.wnid))
    return names, shots[["class_rank", "draw", "pos", "sample_id"]], ide.sample_id.tolist(), ood.sample_id.tolist(), wn


def shots_of(shots, draw, ncls):
    s = shots[shots.draw == draw].sort_values(["class_rank", "pos"])
    assert len(s) == 16 * ncls and (s.groupby("class_rank").size() == 16).all()
    return s[s.pos < 12].sample_id.tolist(), s[s.pos >= 12].sample_id.tolist()


def tins_path(split, draw, seed):
    return TINS_DIR / f"U4_s{split}_d{draw}_seed{seed}.npz"


def pos_text(split):
    return np.load(TINS_DIR / f"U4_s{split}_pos.npy")


def stream_inputs(F, split, draw, seed, views=("B14", "L14"), K=20):
    names, shots, ide, ood, wn = spec(split)
    C = len(names)
    sup_ids, cal_ids = shots_of(shots, draw, C)
    order = build_order(len(ide), len(ood), seed)
    ids = [ide[i] if o == 0 else ood[i] for o, i in order]
    flag = np.array([o for o, _ in order], bool)
    out = {"names": names, "ids": ids, "flag": flag, "C": C, "wnid": np.array([wn[s] for s in ids])}
    for v in views:
        out[f"sup_{v}"] = F.get(v, sup_ids).reshape(C, 12, -1)
        out[f"cal_{v}"] = F.get(v, cal_ids)
        out[f"sf_{v}"] = F.get(v, ids)
    pt = pos_text(split)
    clip_s, clip_c = F.get("CLIP", ids), F.get("CLIP", cal_ids)
    kk = min(K, C)
    sims_s, sims_c = clip_s @ pt.T, clip_c @ pt.T
    out["cand_s"] = np.argsort(-sims_s, axis=1)[:, :kk]
    out["cand_c"] = np.argsort(-sims_c, axis=1)[:, :kk]
    out["mcm"] = torch.softmax(torch.as_tensor(sims_s, dtype=torch.float64), dim=1).max(1).values.numpy()
    z = np.load(tins_path(split, draw, seed), allow_pickle=True)
    assert list(z["sample_id"]) == ids
    out["S"] = z["S_final"].astype(np.float64)
    out["bidx"] = z["batch_index"]
    return out
