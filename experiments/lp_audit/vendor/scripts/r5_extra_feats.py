"""R5 Phase 0: extra frozen image features (no labels, no stream order involved).

--what l14_train --shard i --of n
    DINOv2 ViT-L/14 CLS features (fp32 forward, standard eval preprocessing, L2-normalised, stored as float16) of every
    ImageNet-1K train image (1000 classes, sorted wnid / file order), together with the sha256 of each file (the bytes
    are read once and decoded from memory). Work is saved in parts of 20,000 images, so a restart resumes.
    Output: <R5>/features/in1k_train_l14/shard<i>of<n>.pt  {sample_id, sha256, features, meta}
    Use: full-train kNN / Mahalanobis++ baselines (E1). At use time the images of the dev streams, the R5 shots and any
    file whose sha256 is sealed are dropped.
--what dino1
    DINO (v1) ViT-B/16 CLS features (timm vit_base_patch16_224.dino: self-supervised on ImageNet-1K only), same
    preprocessing, L2-normalised float32, for the dev1 and dev2 samples and the R5 shots.
    Output: <WORK>/features/dino1.pt for dev1 and dev2, <R5>/features/shots.dino1.pt
Every image goes through the sealing guard.
"""
import argparse
import hashlib
import io
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402
from torch.utils.data import DataLoader, Dataset  # noqa: E402

from vins import config as C  # noqa: E402
from vins.features import GuardedImages, dino_transform, guard_from_seal, save_features  # noqa: E402
from vins.r5 import R5  # noqa: E402
from vins.splits import list_class_files, load_imagenet_classes  # noqa: E402

CKPT_DIR = C.HOME / ".cache" / "torch" / "hub" / "checkpoints"
PART = 20000


class HashedImages(Dataset):
    def __init__(self, paths, transform, guard):
        self.paths, self.transform, self.guard = list(paths), transform, guard

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        real = self.guard.check(self.paths[index])
        with open(real, "rb") as handle:
            data = handle.read()
        image = Image.open(io.BytesIO(data)).convert("RGB")
        return self.transform(image), index, hashlib.sha256(data).hexdigest()


def load_l14():
    ckpt = CKPT_DIR / "dinov2_vitl14_pretrain.pth"
    model = torch.hub.load(str(C.DINO_HUB), "dinov2_vitl14", source="local", pretrained=False)
    model.load_state_dict(torch.load(ckpt, map_location="cpu"), strict=True)
    return model.eval().cuda(), hashlib.sha256(ckpt.read_bytes()).hexdigest()


@torch.no_grad()
def l14_train(shard, of, workers):
    wnids, _, _ = load_imagenet_classes()
    ids = [f"in1k/train/{w}/{f}" for w in wnids for f in list_class_files(w)]
    ids = ids[shard::of]
    paths = [str(C.IMAGENET_ROOT / s.split("/", 1)[1]) for s in ids]
    out = R5 / "features" / "in1k_train_l14"
    out.mkdir(parents=True, exist_ok=True)
    model, digest = load_l14()
    guard = guard_from_seal()
    start = time.time()
    for j, lo in enumerate(range(0, len(ids), PART)):
        part = out / f"shard{shard}of{of}_part{j:03d}.pt"
        if part.exists():
            continue
        loader = DataLoader(HashedImages(paths[lo:lo + PART], dino_transform(), guard), batch_size=64, shuffle=False,
                            num_workers=workers, pin_memory=True)
        feats, order, hashes = [], [], []
        for images, index, hx in loader:
            f = model.forward_features(images.cuda(non_blocking=True))["x_norm_clstoken"].float()
            feats.append((f / f.norm(dim=-1, keepdim=True)).half().cpu())
            order.append(index)
            hashes += list(hx)
        idx = torch.cat(order)
        assert torch.equal(idx, torch.arange(min(PART, len(ids) - lo)))
        torch.save({"sample_id": ids[lo:lo + PART], "sha256": hashes, "features": torch.cat(feats)}, part)
        print(json.dumps({"part": part.name, "done": lo + len(hashes), "of": len(ids),
                          "img_per_s": round((lo + len(hashes)) / (time.time() - start), 1)}), flush=True)
    parts = sorted(out.glob(f"shard{shard}of{of}_part*.pt"))
    blobs = [torch.load(p, map_location="cpu") for p in parts]
    sid = [s for b in blobs for s in b["sample_id"]]
    assert sid == ids, "parts do not cover the shard in order"
    torch.save({"sample_id": sid, "sha256": [h for b in blobs for h in b["sha256"]],
                "features": torch.cat([b["features"] for b in blobs]),
                "meta": {"model": "DINOv2 ViT-L/14 (torch hub, local)", "embedding": "x_norm_clstoken",
                         "dtype": "fp32 forward, stored float16", "ckpt_sha256": digest,
                         "shard": shard, "of": of}}, out / f"shard{shard}of{of}.pt")
    for p in parts:
        p.unlink()
    print(json.dumps({"shard": shard, "n": len(sid), "seconds": round(time.time() - start, 1)}))


@torch.no_grad()
def dino1(workers):
    import timm

    model = timm.create_model("vit_base_patch16_224.dino", pretrained=True, num_classes=0).eval().cuda()
    cfg = model.pretrained_cfg
    assert tuple(cfg["mean"]) == (0.485, 0.456, 0.406) and tuple(cfg["std"]) == (0.229, 0.224, 0.225), cfg
    guard = guard_from_seal()
    base = Path.home() / "vins_gonogo_20260925"
    jobs = []
    for work in (base, base / "dev2"):
        s = pd.read_parquet(work / "splits" / "samples.parquet")
        jobs.append((work / "features" / "dino1.pt", s.sample_id.tolist(), s.path.tolist()))
    d = pd.read_parquet(R5 / "shots" / "draws.parquet")
    jobs.append((R5 / "features" / "shots.dino1.pt", d.sample_id.tolist(), d.path.tolist()))
    for out, ids, paths in jobs:
        if out.exists():
            continue
        start = time.time()
        loader = DataLoader(GuardedImages(paths, dino_transform(), guard), batch_size=128, shuffle=False,
                            num_workers=workers, pin_memory=True)
        feats, order = [], []
        for images, index in loader:
            f = model(images.cuda(non_blocking=True)).float()          # CLS token after the final LayerNorm
            feats.append((f / f.norm(dim=-1, keepdim=True)).cpu())
            order.append(index)
        assert torch.equal(torch.cat(order), torch.arange(len(paths)))
        save_features(out, ids, torch.cat(feats), {"model": "DINO ViT-B/16 (timm vit_base_patch16_224.dino)",
                                                   "embedding": "CLS after final norm", "pretrained_cfg": str(cfg)})
        print(json.dumps({"out": str(out), "n": len(ids), "seconds": round(time.time() - start, 1)}), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--what", choices=["l14_train", "dino1"], required=True)
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--of", type=int, default=1)
    parser.add_argument("--workers", type=int, default=8)
    opts = parser.parse_args()
    if opts.what == "l14_train":
        l14_train(opts.shard, opts.of, opts.workers)
    else:
        dino1(opts.workers)


if __name__ == "__main__":
    main()
