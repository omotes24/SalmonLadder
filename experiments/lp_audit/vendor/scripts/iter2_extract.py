"""Iteration 2: DINOv2 CLS features of other sizes (e.g. L/14 fp32, g/14 fp16) for every dev sample, one image pass.

Usage: VINS_WORK=<dev work> python scripts/iter2_extract.py --archs vitl14 vitg14 [--half vitg14]
Output: <WORK>/features/dino_<arch>.pt (same layout as features/dino.pt; standard eval preprocessing, L2-normalised).
The sealing guard refuses any sealed (OpenOOD test / val_imagenet) image.
"""
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402
import torch  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from vins import config as C  # noqa: E402
from vins.features import GuardedImages, dino_transform, guard_from_seal, save_features  # noqa: E402

CKPT_DIR = C.HOME / ".cache" / "torch" / "hub" / "checkpoints"


def load_arch(arch, half):
    ckpt = CKPT_DIR / f"dinov2_{arch}_pretrain.pth"
    digest = hashlib.sha256(ckpt.read_bytes()).hexdigest()
    model = torch.hub.load(str(C.DINO_HUB), f"dinov2_{arch}", source="local", pretrained=False)
    model.load_state_dict(torch.load(ckpt, map_location="cpu"), strict=True)
    model = model.eval().cuda()
    return (model.half() if half else model), digest


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--archs", nargs="+", required=True)
    parser.add_argument("--half", nargs="*", default=[])
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=8)
    opts = parser.parse_args()
    start = time.time()
    samples = pd.read_parquet(C.SPLITS_DIR / "samples.parquet")
    models = {a: load_arch(a, a in opts.half) for a in opts.archs}
    loader = DataLoader(GuardedImages(samples.path.tolist(), dino_transform(), guard_from_seal()),
                        batch_size=opts.batch_size, shuffle=False, num_workers=opts.workers, pin_memory=True)
    chunks = {a: [] for a in opts.archs}
    order = []
    for images, index in loader:
        images = images.cuda(non_blocking=True)
        for a, (model, _) in models.items():
            x = images.half() if a in opts.half else images
            f = model.forward_features(x)["x_norm_clstoken"].float()
            chunks[a].append((f / f.norm(dim=-1, keepdim=True)).cpu())
        order.append(index)
    order = torch.cat(order)
    assert torch.equal(order, torch.arange(len(samples)))
    for a, (_, digest) in models.items():
        save_features(C.FEATURES_DIR / f"dino_{a}.pt", samples.sample_id.tolist(), torch.cat(chunks[a]),
                      {"model": f"DINOv2 {a} (torch hub, local)", "ckpt_sha256": digest, "half": a in opts.half,
                       "embedding": "x_norm_clstoken"})
    print(json.dumps({"archs": opts.archs, "n": len(samples), "seconds": round(time.time() - start, 1),
                      "sha256": {a: d[:16] for a, (_, d) in models.items()}}))


if __name__ == "__main__":
    main()
