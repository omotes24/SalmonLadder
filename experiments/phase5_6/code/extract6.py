"""Phase 6 (encoder swap, no tuning): DINOv3 CLS features for the R5 shots and the dev streams. Sealed images are refused.

Encoders (timm, Hugging Face hub `timm/<name>`, DINOv3 License): ViT-B/16 and ViT-L/16 distilled, LVD-1689M.
Views (one image decode per sample):
  D3B,  D3L  : 256 px -- Resize(293, bicubic) + CenterCrop(256): the centre-crop ratio of the DINOv2 pipeline (0.875)
               and as many patch tokens (256) as DINOv2 /14 at 224 px.                      [pre-registered primary]
  D3Bs, D3Ls : the exact DINOv2 input -- Resize(256, bicubic) + CenterCrop(224) (196 tokens). [sensitivity]
Feature: CLS token after the final norm (forward_features()[:, 0]), L2-normalised float32; fp32 inference, TF32 off.
Output: <P5>/features/<set>.<name>.shard<i>of<n>.pt  (sample_id, features)
"""
import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from common import AUDIT  # noqa: F401  (adds the vendored vins package to sys.path)
from vins import r5
from vins.features import dino_transform, guard_from_seal

P5 = Path("/home/omote/reprise_p5_20261002")
BASE = Path.home() / "vins_gonogo_20260925"
MODELS = {"B": "vit_base_patch16_dinov3.lvd1689m", "L": "vit_large_patch16_dinov3.lvd1689m"}
NAMES = ["D3B", "D3L", "D3Bs", "D3Ls"]


def items(which):
    if which == "shots":
        d = pd.read_parquet(r5.R5 / "shots" / "draws.parquet")
        return d.sample_id.tolist(), d.path.tolist()
    work = BASE if which == "dev1" else BASE / "dev2"
    s = pd.read_parquet(work / "splits" / "samples.parquet")
    s = s[s.split.isin(["id_dev", "near_dev", "far_dev"])]
    return s.sample_id.tolist(), s.path.tolist()


def transform256():
    from torchvision import transforms
    from vins.features import IMAGENET_MEAN, IMAGENET_STD

    return transforms.Compose([
        transforms.Resize(293, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(256),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


class Images(Dataset):
    def __init__(self, paths, transforms, guard):
        self.paths, self.transforms, self.guard = paths, transforms, guard

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        img = Image.open(self.guard.check(self.paths[i])).convert("RGB")
        return tuple(t(img) for t in self.transforms) + (i,)


def norm(x):
    x = x.float()
    return (x / x.norm(dim=-1, keepdim=True)).cpu()


def load_models(dev):
    import timm

    out = {}
    for tag, name in MODELS.items():
        m = timm.create_model(name, pretrained=True, num_classes=0)
        out[tag] = m.eval().to(dev)
    return out


def cls(model, x):
    return model.forward_features(x)[:, 0]


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", required=True, choices=["dev1", "dev2", "shots"])
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--workers", type=int, default=5)
    a = ap.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    dev = "cuda"
    ids, paths = items(a.set)
    sel = np.arange(len(ids))[a.shard::a.nshards]
    if a.limit:
        sel = sel[:a.limit]
    ids, paths = [ids[i] for i in sel], [paths[i] for i in sel]
    out_dir = P5 / "features"
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = f"shard{a.shard}of{a.nshards}" + (f".limit{a.limit}" if a.limit else "")
    if all((out_dir / f"{a.set}.{n}.{tag}.pt").exists() for n in NAMES):
        return
    guard = guard_from_seal()
    t0 = time.time()
    models = load_models(dev)
    loader = DataLoader(Images(paths, (transform256(), dino_transform()), guard), batch_size=a.batch, shuffle=False,
                        num_workers=a.workers, pin_memory=True)
    chunks = {n: [] for n in NAMES}
    order = []
    for x256, x224, idx in loader:
        x256, x224 = x256.to(dev, non_blocking=True), x224.to(dev, non_blocking=True)
        for k, m in models.items():
            chunks[f"D3{k}"].append(norm(cls(m, x256)))
            chunks[f"D3{k}s"].append(norm(cls(m, x224)))
        order.append(idx)
        if len(order) % 50 == 0:
            print(json.dumps({"set": a.set, "shard": a.shard, "done": len(order) * a.batch, "of": len(paths),
                              "seconds": round(time.time() - t0, 1)}), flush=True)
    assert torch.equal(torch.cat(order), torch.arange(len(paths)))
    for n in NAMES:
        f = torch.cat(chunks[n])
        assert torch.isfinite(f).all()
        tmp = out_dir / f"{a.set}.{n}.{tag}.{os.getpid()}.tmp"
        torch.save({"sample_id": ids, "features": f,
                    "meta": {"set": a.set, "name": n, "shard": a.shard, "nshards": a.nshards,
                             "model": MODELS[n[2]], "input": "resize293-crop256" if len(n) == 3 else "resize256-crop224"}}, tmp)
        os.replace(tmp, out_dir / f"{a.set}.{n}.{tag}.pt")
    print(json.dumps({"set": a.set, "shard": a.shard, "n": len(ids), "seconds": round(time.time() - t0, 1),
                      "dims": {n: int(chunks[n][0].shape[1]) for n in NAMES}}), flush=True)


if __name__ == "__main__":
    main()
