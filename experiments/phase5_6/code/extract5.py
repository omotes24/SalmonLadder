"""Phase 5, second stage (prepared in advance; used only if the first stage misses the stop rule):
frozen features of additional encoders for the dev streams and the R5 shots. Sealed images are refused.

Groups (one image decode per sample and group):
  dino : DINOv2 ViT-B/14, ViT-L/14, ViT-g/14 (torch hub, local checkpoints), standard eval preprocessing.
         Saved: B14pm, L14pm (mean of the normalised patch tokens), G14 (CLS), G14pm; B14 / L14 CLS only as a check.
  vlm  : SigLIP2 large patch16-256 and CLIP ViT-L/14 image embeddings (transformers, local cache).
Output: <P5>/features/<set>.<name>.shard<i>of<n>.pt  (sample_id, L2-normalised float32 features)
"""
import argparse
import json
import os
import sys
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
CKPT = Path.home() / ".cache" / "torch" / "hub" / "checkpoints"
HUB = Path.home() / ".cache" / "torch" / "hub" / "facebookresearch_dinov2_main"
BASE = Path.home() / "vins_gonogo_20260925"


def items(which):
    if which == "shots":
        d = pd.read_parquet(r5.R5 / "shots" / "draws.parquet")
        return d.sample_id.tolist(), d.path.tolist()
    work = BASE if which == "dev1" else BASE / "dev2"
    s = pd.read_parquet(work / "splits" / "samples.parquet")
    s = s[s.split.isin(["id_dev", "near_dev", "far_dev"])]
    return s.sample_id.tolist(), s.path.tolist()


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


def dino_models(dev):
    out = {}
    for tag, name in (("B14", "dinov2_vitb14"), ("L14", "dinov2_vitl14"), ("G14", "dinov2_vitg14")):
        m = torch.hub.load(str(HUB), name, source="local", pretrained=False)
        m.load_state_dict(torch.load(CKPT / f"{name}_pretrain.pth", map_location="cpu"), strict=True)
        out[tag] = m.eval().to(dev)
    return out


def vlm_models(dev):
    from transformers import AutoModel, AutoProcessor, CLIPModel, CLIPProcessor

    def load(cls, name):
        try:
            return cls.from_pretrained(name, local_files_only=True)
        except Exception:
            return cls.from_pretrained(name)

    sig = load(AutoModel, "google/siglip2-large-patch16-256").eval().to(dev)
    sigp = load(AutoProcessor, "google/siglip2-large-patch16-256")
    clip = load(CLIPModel, "openai/clip-vit-large-patch14").eval().to(dev)
    clipp = load(CLIPProcessor, "openai/clip-vit-large-patch14")
    t_sig = lambda img: sigp(images=img, return_tensors="pt")["pixel_values"][0]
    t_clip = lambda img: clipp(images=img, return_tensors="pt")["pixel_values"][0]
    return (sig, clip), (t_sig, t_clip)


def pooled(x):
    return x if torch.is_tensor(x) else (x.pooler_output if getattr(x, "pooler_output", None) is not None else x[0])


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", required=True, choices=["dev1", "dev2", "shots"])
    ap.add_argument("--group", required=True, choices=["dino", "vlm"])
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--workers", type=int, default=6)
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
    names = ["B14", "B14pm", "L14", "L14pm", "G14", "G14pm"] if a.group == "dino" else ["SIG2L", "CLIPL"]
    if all((out_dir / f"{a.set}.{n}.{tag}.pt").exists() for n in names):
        return
    guard = guard_from_seal()
    t0 = time.time()
    if a.group == "dino":
        models = dino_models(dev)
        transforms = (dino_transform(),)
    else:
        (sig, clip), transforms = vlm_models(dev)
    loader = DataLoader(Images(paths, transforms, guard), batch_size=a.batch, shuffle=False, num_workers=a.workers, pin_memory=True)
    chunks = {n: [] for n in names}
    order = []
    for batch in loader:
        idx = batch[-1]
        if a.group == "dino":
            x = batch[0].to(dev, non_blocking=True)
            for k, m in models.items():
                f = m.forward_features(x)
                chunks[k].append(norm(f["x_norm_clstoken"]))
                chunks[k + "pm"].append(norm(f["x_norm_patchtokens"].mean(1)))
        else:
            chunks["SIG2L"].append(norm(pooled(sig.get_image_features(pixel_values=batch[0].to(dev, non_blocking=True)))))
            chunks["CLIPL"].append(norm(pooled(clip.get_image_features(pixel_values=batch[1].to(dev, non_blocking=True)))))
        order.append(idx)
        if len(order) % 100 == 0:
            print(json.dumps({"set": a.set, "group": a.group, "shard": a.shard, "done": len(order) * a.batch, "of": len(paths),
                              "seconds": round(time.time() - t0, 1)}), flush=True)
    assert torch.equal(torch.cat(order), torch.arange(len(paths)))
    for n in names:
        f = torch.cat(chunks[n])
        tmp = out_dir / f"{a.set}.{n}.{tag}.{os.getpid()}.tmp"
        torch.save({"sample_id": ids, "features": f, "meta": {"set": a.set, "name": n, "shard": a.shard, "nshards": a.nshards}}, tmp)
        os.replace(tmp, out_dir / f"{a.set}.{n}.{tag}.pt")
    print(json.dumps({"set": a.set, "group": a.group, "shard": a.shard, "n": len(ids), "seconds": round(time.time() - t0, 1),
                      "dims": {n: int(chunks[n][0].shape[1]) for n in names}}), flush=True)


if __name__ == "__main__":
    main()
