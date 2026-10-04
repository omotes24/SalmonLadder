"""Frozen features (DINOv2 B/14, L/14 CLS; CLIP ViT-B/16 image) for every row of the U4 feature table.
Same models, checkpoints and transforms as the frozen REPRISE caches (Phase 4 extract.py with U4 paths).
Usage: python u4_extract.py --shard i --nshards 4 ; python u4_extract.py --merge --nshards 4"""
import argparse
import json
import os
import time

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from common import dump, sha_file, utc
from u4_common import BANKS, FEATS, require_lock
from vins import config as C
from vins.features import dino_transform
from vins.sealing import Guard, load_sealed
from vins.tins_dev import import_tins, load_clip, make_args


class Images(Dataset):
    def __init__(self, paths, tf_dino, tf_clip, guard):
        self.paths, self.tf_dino, self.tf_clip, self.guard = list(paths), tf_dino, tf_clip, guard

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        with Image.open(self.guard.check(self.paths[i])) as im:
            im = im.convert("RGB")
            return self.tf_dino(im), self.tf_clip(im), i


def load_models():
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    models = {}
    for name, tag in (("B14", "vitb14"), ("L14", "vitl14")):
        m = torch.hub.load(str(C.DINO_HUB), "dinov2_" + tag, source="local", pretrained=False)
        m.load_state_dict(torch.load(C.HOME / ".cache/torch/hub/checkpoints" / f"dinov2_{tag}_pretrain.pth",
                                     map_location="cpu"), strict=True)
        models[name] = m.eval().cuda()
    t = import_tins()
    args = make_args(t, FEATS / "clip_cache", "p5_encoder")
    clip, pre = load_clip(t, args)
    models["CLIP"] = clip.eval()
    return models, pre


@torch.no_grad()
def run_shard(shard, nshards):
    table = pd.read_parquet(BANKS / "feature_table.parquet")
    rows = np.arange(len(table))[shard::nshards]
    FEATS.mkdir(parents=True, exist_ok=True)
    out = FEATS / f"shard{shard}of{nshards}.npz"
    if out.exists():
        print(json.dumps({"exists": str(out)}))
        return
    models, pre = load_models()
    sealed_paths, _, _ = load_sealed()
    ds = Images(table.path.values[rows], dino_transform(), pre, Guard(sealed_paths))
    dl = DataLoader(ds, batch_size=128, shuffle=False, num_workers=6, pin_memory=True)
    buf = {k: [] for k in models}
    order, t0 = [], time.time()
    for n, (xd, xc, idx) in enumerate(dl):
        xd = xd.cuda(non_blocking=True)
        xc = xc.cuda(non_blocking=True)
        for name in ("B14", "L14"):
            chunks = [models[name].forward_features(xd[lo:lo + 64])["x_norm_clstoken"].float() for lo in range(0, len(xd), 64)]
            f = torch.cat(chunks)
            buf[name].append((f / f.norm(dim=-1, keepdim=True)).cpu().numpy().astype(np.float32))
        f = models["CLIP"].encode_image(xc.to(next(models["CLIP"].parameters()).dtype)).float()
        buf["CLIP"].append((f / f.norm(dim=-1, keepdim=True)).cpu().numpy().astype(np.float32))
        order.append(idx.numpy())
        if n % 20 == 0:
            done = sum(len(o) for o in order)
            print(json.dumps({"shard": shard, "done": done, "total": len(rows), "img_per_s": round(done / (time.time() - t0), 1)}), flush=True)
    order = np.concatenate(order)
    assert np.array_equal(order, np.arange(len(rows)))
    tmp = str(out) + f".{os.getpid()}.npz"
    np.savez(tmp, rows=rows, **{k: np.concatenate(v) for k, v in buf.items()})
    os.replace(tmp, out)
    print(json.dumps({"shard": shard, "complete": str(out), "seconds": round(time.time() - t0, 1)}), flush=True)


def merge(nshards):
    table = pd.read_parquet(BANKS / "feature_table.parquet")
    parts = [np.load(FEATS / f"shard{s}of{nshards}.npz") for s in range(nshards)]
    for name in ("B14", "L14", "CLIP"):
        dim = parts[0][name].shape[1]
        full = np.zeros((len(table), dim), np.float32)
        filled = np.zeros(len(table), bool)
        for z in parts:
            full[z["rows"]] = z[name]
            filled[z["rows"]] = True
        assert filled.all()
        np.save(FEATS / f"{name}.npy", full)
    dump(FEATS / "features_info.json", {"utc": utc(), "n": len(table), "table_sha256": sha_file(BANKS / "feature_table.parquet"),
                                        "files": {n: sha_file(FEATS / f"{n}.npy") for n in ("B14", "L14", "CLIP")}})
    print("merged", len(table))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", type=int)
    ap.add_argument("--nshards", type=int, default=4)
    ap.add_argument("--merge", action="store_true")
    a = ap.parse_args()
    require_lock()
    if a.merge:
        merge(a.nshards)
    else:
        run_shard(a.shard, a.nshards)
