"""One read pass over the unique OpenOOD test images (post-test analyses) producing two feature sets:
  oo_test.clipl14.pt        : CLIP ViT-L/14 with CLIP's own preprocessing (TINS with a larger CLIP)
  oo_test.clipb16_oodpre.pt : CLIP ViT-B/16 with OpenOOD-style preprocessing (Resize 256, CenterCrop 224, CLIP mean/std)
                              to test whether preprocessing explains the gap to the published TINS numbers
Images are listed in the order of test_eval/results/features.pt (sorted OpenOOD test sample ids).
"""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402
from PIL import Image  # noqa: E402
from torch.utils.data import DataLoader, Dataset  # noqa: E402
from torchvision import transforms  # noqa: E402

from vins import config as C  # noqa: E402
from scripts.extract_extra import load_clip_model  # noqa: E402

CLIP_MEAN, CLIP_STD = (0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)


class Dual(Dataset):
    def __init__(self, paths, ta, tb):
        self.paths, self.ta, self.tb = paths, ta, tb

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        img = Image.open(self.paths[i]).convert("RGB")
        return self.ta(img), self.tb(img), i


@torch.no_grad()
def main():
    out = C.WORK / "extra_feats"
    blob = torch.load(C.WORK / "test_eval" / "results" / "features.pt", map_location="cpu")
    paths = blob["paths"][16000:]
    l14, pre_l14 = load_clip_model("ViT-L/14")
    b16, _ = load_clip_model("ViT-B/16")
    oodpre = transforms.Compose([transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
                                 transforms.CenterCrop(224), transforms.ToTensor(), transforms.Normalize(CLIP_MEAN, CLIP_STD)])
    fa, fb, order = [], [], []
    tick = time.time()
    for xa, xb, idx in DataLoader(Dual(paths, pre_l14, oodpre), batch_size=128, shuffle=False, num_workers=6,
                                  pin_memory=True):
        a = l14.encode_image(xa.cuda(non_blocking=True)).float()
        b = b16.encode_image(xb.cuda(non_blocking=True)).float()
        fa.append((a / a.norm(dim=-1, keepdim=True)).cpu())
        fb.append((b / b.norm(dim=-1, keepdim=True)).cpu())
        order.append(idx)
    assert torch.equal(torch.cat(order), torch.arange(len(paths)))
    ids = [f"oo_test_{k:06d}" for k in range(len(paths))]
    for name, feats in (("clipl14", fa), ("clipb16_oodpre", fb)):
        torch.save({"ids": ids, "paths": paths, "labels": [-1] * len(paths), "wnids": [""] * len(paths),
                    "features": torch.cat(feats)}, out / f"oo_test.{name}.pt")
    print(json.dumps({"n": len(paths), "seconds": round(time.time() - tick, 1)}))


if __name__ == "__main__":
    main()
