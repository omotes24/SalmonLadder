"""THIRD test use (pre-registered in test_eval/prereg_test3.json): DINOv2 ViT-L/14 CLS features of exactly the
image list stored in test_eval/results/features.pt (support/calibration shots + OpenOOD test images).

Usage: python scripts/iter2_extract_test.py --shard K --of N      (one process per GPU)
       python scripts/iter2_extract_test.py --merge --of N         (writes features_vitl14.pt and its sha256)
"""
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402
from PIL import Image  # noqa: E402
from torch.utils.data import DataLoader, Dataset  # noqa: E402

from vins import config as C  # noqa: E402
from vins.features import dino_transform  # noqa: E402

RES = C.WORK / "test_eval" / "results"
PREREG_SHA = "4b6cacce2ffd7f69433f658173b9f3c91dee90a2e66e0eefc0e0c7b4deb4b7a7"
CKPT = C.HOME / ".cache" / "torch" / "hub" / "checkpoints" / "dinov2_vitl14_pretrain.pth"


class Paths(Dataset):
    def __init__(self, paths, transform):
        self.paths, self.transform = paths, transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        return self.transform(Image.open(self.paths[i]).convert("RGB")), i


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard", type=int)
    parser.add_argument("--of", type=int, required=True)
    parser.add_argument("--merge", action="store_true")
    parser.add_argument("--workers", type=int, default=6)
    opts = parser.parse_args()
    prereg = C.WORK / "test_eval" / "prereg_test3.json"
    assert hashlib.sha256(prereg.read_bytes()).hexdigest() == PREREG_SHA, "prereg_test3.json changed or missing"
    paths = torch.load(RES / "features.pt", map_location="cpu")["paths"]
    n = len(paths)
    bounds = [round(k * n / opts.of) for k in range(opts.of + 1)]
    if opts.merge:
        feats = []
        for k in range(opts.of):
            blob = torch.load(RES / f"vitl14_shard{k}.pt", map_location="cpu")
            assert blob["lo"] == bounds[k] and blob["hi"] == bounds[k + 1]
            feats.append(blob["features"])
        feats = torch.cat(feats)
        assert feats.shape[0] == n
        out = RES / "features_vitl14.pt"
        torch.save({"paths": paths, "dino": feats, "meta": {"model": "DINOv2 ViT-L/14", "ckpt": str(CKPT),
                                                            "preprocess": "Resize256 bicubic, CenterCrop224, ImageNet norm",
                                                            "embedding": "x_norm_clstoken, L2-normalised"}}, out)
        digest = hashlib.sha256(out.read_bytes()).hexdigest()
        (RES / "features_vitl14.sha256").write_text(digest + "\n")
        print(json.dumps({"merged": n, "sha256": digest}))
        return
    lo, hi = bounds[opts.shard], bounds[opts.shard + 1]
    start = time.time()
    model = torch.hub.load(str(C.DINO_HUB), "dinov2_vitl14", source="local", pretrained=False)
    model.load_state_dict(torch.load(CKPT, map_location="cpu"), strict=True)
    model = model.eval().cuda()
    loader = DataLoader(Paths(paths[lo:hi], dino_transform()), batch_size=64, shuffle=False,
                        num_workers=opts.workers, pin_memory=True)
    chunks, order = [], []
    for images, index in loader:
        f = model.forward_features(images.cuda(non_blocking=True))["x_norm_clstoken"].float()
        chunks.append((f / f.norm(dim=-1, keepdim=True)).cpu())
        order.append(index)
    assert torch.equal(torch.cat(order), torch.arange(hi - lo))
    torch.save({"lo": lo, "hi": hi, "features": torch.cat(chunks)}, RES / f"vitl14_shard{opts.shard}.pt")
    print(json.dumps({"shard": opts.shard, "lo": lo, "hi": hi, "seconds": round(time.time() - start, 1)}))


if __name__ == "__main__":
    main()
