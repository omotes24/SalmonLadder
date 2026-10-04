"""Inputs of the unused banks U1 (ImageNet new splits), U2 (ImageNet-O), U3 (WILD): class lists, shots,
ordered streams (TINS order rule), candidate sets K(x) and MCM. Evaluation labels leave this module only as
the evaluator's is_ood flags."""
import json

import numpy as np
import pandas as pd
import torch

from common import BANKS, FEATS, ROOT
from vins.tins_dev import build_order

WILD_NAMES = {"buffalo": "buffalo", "cheetah": "cheetah", "elephant": "elephant", "giraffe": "giraffe",
              "hippo": "hippopotamus"}
TINS_DIR = ROOT / "tins"


class Features:
    def __init__(self):
        t = pd.read_parquet(BANKS / "feature_table.parquet")
        self.row = dict(zip(t.sample_id, range(len(t))))
        self.F = {n: np.load(FEATS / f"{n}.npy", mmap_mode="r") for n in ("B14", "L14", "CLIP")}

    def get(self, name, ids):
        return np.asarray(self.F[name][np.array([self.row[s] for s in ids])], np.float32)


def spec(bank, split=0):
    """Returns names (ID classes in label order), shots frame (class_rank, draw, pos, sample_id), eval ID ids,
    eval OOD ids (each in a fixed canonical order)."""
    if bank in ("U1", "U2"):
        pool = pd.read_parquet(BANKS / "imagenet_pool.parquet")
        cls = pool.drop_duplicates("wnid").sort_values("idx_1k")
        cname = dict(zip(cls.wnid, cls.clean_name))
        if bank == "U1":
            sp = json.loads((BANKS / "U1" / f"split{split}.json").read_text())
            idw, held = sp["id_wnids"], set(sp["heldout"])
        else:
            idw, held = cls.wnid.tolist(), set()
        rank = {w: i for i, w in enumerate(idw)}
        names = [cname[w] for w in idw]
        shots = pool[(pool.role == "shot") & pool.wnid.isin(rank)].copy()
        shots["class_rank"] = shots.wnid.map(rank)
        ide = pool[(pool.role == "ideval") & pool.wnid.isin(rank)].sort_values(["idx_1k", "slot"])
        if bank == "U2":
            ide = ide.groupby("wnid", group_keys=False).head(10)
            ood = pd.read_parquet(BANKS / "U2_imagenet_o.parquet").sort_values("sample_id").sample_id.tolist()
        else:
            ood = pool[(pool.role == "oodeval") & pool.wnid.isin(held)].sort_values(["idx_1k", "slot"]).sample_id.tolist()
        return names, shots[["class_rank", "draw", "pos", "sample_id"]], ide.sample_id.tolist(), ood
    if bank == "U3":
        w = pd.read_parquet(BANKS / "U3_wild.parquet")
        species = sorted(w[w.group == "id"].species.unique())
        rank = {s: i for i, s in enumerate(species)}
        shots = w[w.role == "shot"].copy()
        shots["class_rank"] = shots.species.map(rank)
        shot_hash = set(shots.sha256)
        ev = w[(w.role != "shot") & w.main_stream & ~w.sha256.isin(shot_hash)].drop_duplicates("sha256")
        ide = ev[ev.role == "ideval"].sort_values("sample_id").sample_id.tolist()
        ood = ev[ev.role == "oodeval"].sort_values("sample_id").sample_id.tolist()
        return [WILD_NAMES[s] for s in species], shots[["class_rank", "draw", "pos", "sample_id"]], ide, ood
    raise ValueError(bank)


def shots_of(shots, draw, ncls):
    s = shots[shots.draw == draw].sort_values(["class_rank", "pos"])
    assert len(s) == 16 * ncls and (s.groupby("class_rank").size() == 16).all()
    sup = s[s.pos < 12].sample_id.tolist()
    cal = s[s.pos >= 12].sample_id.tolist()
    return sup, cal


def tins_path(bank, split, draw, seed):
    return TINS_DIR / f"{bank}_s{split}_d{draw}_seed{seed}.npz"


def pos_text(bank, split):
    return np.load(TINS_DIR / f"{bank}_s{split}_pos.npy")


def stream_inputs(F, bank, split, draw, seed, views=("B14", "L14"), K=20, need_tins=True):
    names, shots, ide, ood = spec(bank, split)
    C = len(names)
    sup_ids, cal_ids = shots_of(shots, draw, C)
    order = build_order(len(ide), len(ood), seed)
    ids = [ide[i] if o == 0 else ood[i] for o, i in order]
    flag = np.array([o for o, _ in order], bool)
    out = {"names": names, "ids": ids, "flag": flag, "C": C}
    for v in views:
        out[f"sup_{v}"] = F.get(v, sup_ids).reshape(C, 12, -1)
        out[f"cal_{v}"] = F.get(v, cal_ids)
        out[f"sf_{v}"] = F.get(v, ids)
    pt = pos_text(bank, split)
    clip_s, clip_c = F.get("CLIP", ids), F.get("CLIP", cal_ids)
    kk = min(K, C)
    sims_s, sims_c = clip_s @ pt.T, clip_c @ pt.T
    out["cand_s"] = np.argsort(-sims_s, axis=1)[:, :kk]
    out["cand_c"] = np.argsort(-sims_c, axis=1)[:, :kk]
    out["mcm"] = torch.softmax(torch.as_tensor(sims_s, dtype=torch.float64), dim=1).max(1).values.numpy()
    if need_tins:
        z = np.load(tins_path(bank, split, draw, seed), allow_pickle=True)
        assert list(z["sample_id"]) == ids
        out["S"] = z["S_final"].astype(np.float64)
        out["bidx"] = z["batch_index"]
    else:
        out["bidx"] = np.arange(len(ids)) // 256
    return out
