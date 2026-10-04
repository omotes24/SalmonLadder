"""Phase 7 data layer: one interface for every bank (U1, U2, U4, the dataset banks of D, SSB CUB).

A Problem is one labelled set-up: C ID classes (names only reach the detector through CLIP text features), the 12 + 4
shots of a draw, feature access per view and the evaluation pools. Labels (is_ood, classes) are returned for the
evaluation scripts and the stream builders only; the engine never receives them."""
import json

import numpy as np
import pandas as pd
import torch

from p7common import BANKS, FEATS, GONOGO, P4, P5


class Store:
    """Memory-mapped feature files addressed by sample_id; several tables may contribute views."""

    def __init__(self):
        self.parts = []

    def add(self, table, files):
        t = pd.read_parquet(table, columns=["sample_id"])
        row = dict(zip(t.sample_id, range(len(t))))
        maps = {}
        for v, f in files.items():
            if f.exists():
                maps[v] = np.load(f, mmap_mode="r")
                assert len(maps[v]) == len(t), (str(f), len(maps[v]), len(t))
        self.parts.append((row, maps))
        return self

    def views(self):
        return sorted({v for _, m in self.parts for v in m})

    def get(self, view, ids):
        for row, maps in self.parts:
            if view in maps:
                idx = np.array([row[s] for s in ids])
                order = np.argsort(idx, kind="stable")
                out = np.empty((len(idx), maps[view].shape[1]), np.float32)
                out[order] = np.asarray(maps[view][idx[order]], np.float32)
                return out
        raise KeyError(view)


class Problem:
    def __init__(self, name, names, sup_ids, cal_ids, pos, store, id_ids, ood_ids, cls_of):
        self.name, self.names, self.C = name, names, len(names)
        self.sup_ids, self.cal_ids, self.pos, self.store = list(sup_ids), list(cal_ids), np.asarray(pos, np.float32), store
        self.id_ids, self.ood_ids, self.cls_of = list(id_ids), list(ood_ids), cls_of
        assert len(self.sup_ids) == 12 * self.C and len(self.cal_ids) == 4 * self.C and len(self.pos) == self.C

    def feat(self, view, ids):
        return self.store.get(view, ids)

    def cls(self, ids):
        return np.array([self.cls_of[s] for s in ids], dtype=object)


_CACHE = {}


def _pool(which):
    if which not in _CACHE:
        root = P4 if which == "p4" else P5
        pool = pd.read_parquet(root / "banks" / "imagenet_pool.parquet")
        _CACHE[which] = pool
    return _CACHE[which]


def _store(which):
    key = ("store", which)
    if key not in _CACHE:
        s = Store()
        if which == "p4":
            s.add(P4 / "banks" / "feature_table.parquet", {v: P4 / "features" / f"{v}.npy" for v in ("B14", "L14", "CLIP")})
            x = FEATS / "u1x"
            if (x / "table.parquet").exists():
                s.add(x / "table.parquet", {f.stem: f for f in x.glob("*.npy")})
        elif which == "u4":
            s.add(P5 / "banks" / "feature_table.parquet", {v: P5 / "features_u4" / f"{v}.npy" for v in ("B14", "L14", "CLIP")})
        else:
            x = FEATS / which
            s.add(BANKS / which / "images.parquet", {f.stem: f for f in x.glob("*.npy")})
        _CACHE[key] = s
    return _CACHE[key]


def _shots(shots, draw, C):
    s = shots[shots.draw == draw].sort_values(["class_rank", "pos"])
    assert len(s) == 16 * C and (s.groupby("class_rank").size() == 16).all()
    return s[s.pos < 12].sample_id.tolist(), s[s.pos >= 12].sample_id.tolist()


def u1(split, draw):
    """U1 (Phase 4): ImageNet-1K sibling split. Extra attribute `pool`: every pool image with wnid and role (for E)."""
    pool = _pool("p4")
    sp = json.loads((P4 / "banks" / "U1" / f"split{split}.json").read_text())
    idw, held = sp["id_wnids"], set(sp["heldout"])
    cls = pool.drop_duplicates("wnid").sort_values("idx_1k")
    cname = dict(zip(cls.wnid, cls.clean_name))
    rank = {w: i for i, w in enumerate(idw)}
    shots = pool[(pool.role == "shot") & pool.wnid.isin(rank)].copy()
    shots["class_rank"] = shots.wnid.map(rank)
    sup, cal = _shots(shots, draw, len(idw))
    ide = pool[(pool.role == "ideval") & pool.wnid.isin(rank)].sort_values(["idx_1k", "slot"]).sample_id.tolist()
    ood = pool[(pool.role == "oodeval") & pool.wnid.isin(held)].sort_values(["idx_1k", "slot"]).sample_id.tolist()
    P = Problem(f"U1_s{split}_d{draw}", [cname[w] for w in idw], sup, cal, np.load(P4 / "tins" / f"U1_s{split}_pos.npy"), _store("p4"),
                ide, ood, dict(zip(pool.sample_id, pool.wnid)))
    P.pool, P.id_wnids, P.heldout = pool, idw, sorted(held)
    P.tins = lambda seed: P4 / "tins" / f"U1_s{split}_d{draw}_seed{seed}.npz"
    return P


def u2(draw):
    pool = _pool("p4")
    cls = pool.drop_duplicates("wnid").sort_values("idx_1k")
    idw = cls.wnid.tolist()
    rank = {w: i for i, w in enumerate(idw)}
    shots = pool[pool.role == "shot"].copy()
    shots["class_rank"] = shots.wnid.map(rank)
    sup, cal = _shots(shots, draw, 1000)
    ide = pool[pool.role == "ideval"].sort_values(["idx_1k", "slot"]).groupby("wnid", group_keys=False).head(10).sample_id.tolist()
    o = pd.read_parquet(P4 / "banks" / "U2_imagenet_o.parquet").sort_values("sample_id")
    cls_of = dict(zip(pool.sample_id, pool.wnid))
    cls_of.update(dict(zip(o.sample_id, o.wnid.astype(str))))
    P = Problem(f"U2_s0_d{draw}", cls.clean_name.tolist(), sup, cal, np.load(P4 / "tins" / "U2_s0_pos.npy"), _store("p4"), ide,
                o.sample_id.tolist(), cls_of)
    P.tins = lambda seed: P4 / "tins" / f"U2_s0_d{draw}_seed{seed}.npz"
    return P


def u4(split, draw):
    pool = _pool("p5")
    sp = json.loads((P5 / "banks" / "U4" / f"split{split}.json").read_text())
    idw, held = sp["id_wnids"], set(sp["heldout"])
    cls = pool.drop_duplicates("wnid").sort_values("idx_1k")
    cname = dict(zip(cls.wnid, cls.clean_name))
    rank = {w: i for i, w in enumerate(idw)}
    shots = pool[(pool.role == "shot") & pool.wnid.isin(rank)].copy()
    shots["class_rank"] = shots.wnid.map(rank)
    sup, cal = _shots(shots, draw, len(idw))
    ide = pool[(pool.role == "ideval") & pool.wnid.isin(rank)].sort_values(["idx_1k", "slot"]).sample_id.tolist()
    ood = pool[(pool.role == "oodeval") & pool.wnid.isin(held)].sort_values(["idx_1k", "slot"]).sample_id.tolist()
    P = Problem(f"U4_s{split}_d{draw}", [cname[w] for w in idw], sup, cal, np.load(P5 / "tins_u4" / f"U4_s{split}_pos.npy"), _store("u4"),
                ide, ood, dict(zip(pool.sample_id, pool.wnid)))
    P.tins = lambda seed: P5 / "tins_u4" / f"U4_s{split}_d{draw}_seed{seed}.npz"
    return P


def ds(name, split):
    """Dataset bank of build7.py (one draw of shots per split)."""
    f = pd.read_parquet(BANKS / name / f"split{split}.parquet")
    meta = json.loads((BANKS / name / f"split{split}.json").read_text())
    sup = f[f.role == "sup"].sort_values(["cls_rank", "pos"]).sample_id.tolist()
    cal = f[f.role == "cal"].sort_values(["cls_rank", "pos"]).sample_id.tolist()
    ide = f[f.role == "id"].sort_values(["cls_rank", "pos"]).sample_id.tolist()
    ood = f[f.role == "ood"].sort_values(["cls", "pos"]).sample_id.tolist()
    P = Problem(f"{name}_s{split}", meta["names"], sup, cal, np.load(BANKS / name / f"pos_s{split}.npy"), _store(name), ide, ood,
                dict(zip(f.sample_id, f.cls)))
    P.meta = meta
    return P


class _DictStore:
    def __init__(self, feats, row):
        self.feats, self.row = feats, row

    def get(self, view, ids):
        return self.feats[view][[self.row[s] for s in ids]].astype(np.float32)

    def views(self):
        return sorted(self.feats)


def cub_ssb(level):
    """SSB CUB (authors' split; Phase 3 files): 100 known classes, OOD = the Easy / Medium / Hard test images."""
    base = GONOGO / "r5" / "phase3" / "cub"
    key = ("cub_ssb",)
    if key not in _CACHE:
        sp = json.loads((base / "splits.json").read_text())
        F = torch.load(base / "feats.pt", map_location="cpu")
        row = {s: i for i, s in enumerate(F["ids"])}
        feats = {}
        for v, k in (("B14", "dino"), ("L14", "dino_vitl14"), ("CLIP", "clip")):
            x = F[k].float().numpy()
            feats[v] = (x / np.linalg.norm(x, axis=1, keepdims=True)).astype(np.float32)
        setup = torch.load(base / "tins_setup.pt", map_location="cpu")
        pos = setup["positive_features"].float()
        pos = (pos / pos.norm(dim=-1, keepdim=True)).numpy()
        _CACHE[key] = (sp, _DictStore(feats, row), pos)
    sp, store, pos = _CACHE[key]
    sup = [p for j in range(100) for p in sp["support"][str(j)]]
    cal = [p for j in range(100) for p in sp["calib"][str(j)]]
    cls_of = {p: f"known{l:03d}" for p, l in zip(sp["test_id"], sp["test_id_label"])}
    for lv in ("Easy", "Medium", "Hard"):
        for p in sp["ood"][lv]:
            cls_of[p] = p.split("/")[0]
    P = Problem(f"cubssb_{level}", sp["names"], sup, cal, pos, store, sp["test_id"], sp["ood"][level], cls_of)
    P.tins = lambda seed: base / f"tins_{level}_seed{seed}.npz"
    return P
