"""Phase 3 feature extraction (run after the pre-registration is frozen).

--what shots : CLIP ViT-B/16 (TINS loader and preprocessing) and DINO ViT-B/16 (timm vit_base_patch16_224.dino,
               ImageNet-1K only) of the 16,000 TINS shots (ImageNet train; not sealed) -> <R5>/phase3/shots.extra.pt
--what test  : DINO ViT-B/16 of the evaluated test images (OpenOOD v1.5 test streams; Four-OOD ID and OOD sets),
               in the row order of p3_eval.py -> test.dino1.<part>.pt. Sealed images: requires --unseal <sha256 of the
               frozen pre-registration>.
"""
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402
from torch.utils.data import DataLoader, Dataset  # noqa: E402

from vins import config as C  # noqa: E402
from vins import r5  # noqa: E402
from vins.features import dino_transform, guard_from_seal  # noqa: E402

P3 = r5.R5 / "phase3"
OO = ["ssb_hard", "ninco", "inaturalist", "textures", "openimageo"]
FOUR = ["inat", "sun", "places", "dtd"]


class Images(Dataset):
    def __init__(self, paths, transform, check):
        self.paths, self.transform, self.check = list(paths), transform, check

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        return self.transform(Image.open(self.check(self.paths[i])).convert("RGB")), i


@torch.no_grad()
def run(paths, transform, fn, check, batch=128):
    feats, order = [], []
    for x, i in DataLoader(Images(paths, transform, check), batch_size=batch, num_workers=8, shuffle=False, pin_memory=True):
        f = fn(x.cuda(non_blocking=True)).float()
        feats.append((f / f.norm(dim=-1, keepdim=True)).cpu())
        order.append(i)
    assert torch.equal(torch.cat(order), torch.arange(len(paths)))
    return torch.cat(feats)


def dino1_model():
    import timm

    return timm.create_model("vit_base_patch16_224.dino", pretrained=True, num_classes=0).eval().cuda()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--what", choices=["shots", "test"], required=True)
    parser.add_argument("--part", choices=["openood", "fourood"], default=None)
    parser.add_argument("--unseal", default=None)
    opts = parser.parse_args()
    start = time.time()
    b = torch.load(C.WORK / "test_eval" / "results" / "features.pt", map_location="cpu")
    if opts.what == "shots":
        guard = guard_from_seal()
        paths = b["paths"][:16000]
        from vins.tins_dev import import_tins, load_clip, make_args

        t = import_tins()
        args = make_args(t, P3 / "clip_cache", "vins_p3_shots")
        net, preprocess = load_clip(t, args)
        clip = run(paths, preprocess, net.encode_image, guard.check, 256)
        del net
        dino1 = run(paths, dino_transform(), dino1_model(), guard.check, 128)
        torch.save({"paths": paths, "clip": clip, "dino1": dino1}, P3 / "shots.extra.pt")
        print(json.dumps({"what": "shots", "n": len(paths), "seconds": round(time.time() - start, 1)}))
        return
    digest = hashlib.sha256((P3 / "prereg_phase3.json").read_bytes()).hexdigest()
    if opts.unseal != digest:
        raise SystemExit("the unseal token does not match the frozen Phase 3 pre-registration")
    if opts.part == "openood":
        ids = sorted({x for ds in OO for x in np.load(C.WORK / "test_runs" / "default" / f"{ds}_seed123.npz")["sample_id"].tolist()})
        paths = b["paths"][16000:]
        assert len(paths) == len(ids)
    else:
        names = ["in_val"] + FOUR
        blobs = {n: torch.load(C.WORK / "extra_feats" / f"{n}.dino.pt", map_location="cpu") for n in names}
        ids = sum((blobs[n]["ids"] for n in names), [])
        paths = sum((list(blobs[n]["paths"]) for n in names), [])
    feats = run(paths, dino_transform(), dino1_model(), str, 128)
    torch.save({"ids": ids, "paths": paths, "features": feats, "prereg_sha256": digest}, P3 / f"test.dino1.{opts.part}.pt")
    print(json.dumps({"what": "test", "part": opts.part, "n": len(ids), "seconds": round(time.time() - start, 1)}))


if __name__ == "__main__":
    main()
