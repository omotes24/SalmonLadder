"""R5 step 2: frozen features of the drawn shots (all 5 draws, 1000 classes; sealed images are refused).

--model clip : CLIP ViT-B/16 image features with TINS's loader and preprocessing (as features/clip.pt)
--model dino : DINOv2 ViT-B/14 and ViT-L/14 CLS features, standard eval preprocessing, one image pass
               (as features/dino.pt and features/dino_vitl14.pt)
All features are L2-normalised float32. Output: <R5>/features/shots.<clip|dino|dino_vitl14>.pt
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
from vins.features import GuardedImages, dino_transform, encode, guard_from_seal, load_dino, save_features  # noqa: E402
from vins.r5 import R5  # noqa: E402

CKPT_DIR = C.HOME / ".cache" / "torch" / "hub" / "checkpoints"


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["clip", "dino"], required=True)
    opts = parser.parse_args()
    start = time.time()
    draws = pd.read_parquet(R5 / "shots" / "draws.parquet")
    ids, paths = draws.sample_id.tolist(), draws.path.tolist()
    guard = guard_from_seal()
    out = R5 / "features"
    out.mkdir(parents=True, exist_ok=True)
    if opts.model == "clip":
        from vins.tins_dev import import_tins, load_clip, make_args

        t = import_tins()
        args = make_args(t, R5 / "features" / "clip_cache", "vins_r5_clip")
        net, preprocess = load_clip(t, args)
        feats = encode(paths, preprocess, net.encode_image, guard, batch_size=256, num_workers=8, desc="clip")
        save_features(out / "shots.clip.pt", ids, feats, {"model": "CLIP ViT-B/16 (openai, TINS loader)",
                                                         "preprocess": repr(preprocess)})
    else:
        base = load_dino()
        ckpt = CKPT_DIR / "dinov2_vitl14_pretrain.pth"
        large = torch.hub.load(str(C.DINO_HUB), "dinov2_vitl14", source="local", pretrained=False)
        large.load_state_dict(torch.load(ckpt, map_location="cpu"), strict=True)
        large = large.eval().cuda()
        loader = DataLoader(GuardedImages(paths, dino_transform(), guard), batch_size=64, shuffle=False,
                            num_workers=8, pin_memory=True)
        chunks = {"dino": [], "dino_vitl14": []}
        order = []
        for images, index in loader:
            images = images.cuda(non_blocking=True)
            for key, model in (("dino", base), ("dino_vitl14", large)):
                f = model.forward_features(images)["x_norm_clstoken"].float()
                chunks[key].append((f / f.norm(dim=-1, keepdim=True)).cpu())
            order.append(index)
        assert torch.equal(torch.cat(order), torch.arange(len(paths)))
        save_features(out / "shots.dino.pt", ids, torch.cat(chunks["dino"]),
                      {"model": "DINOv2 ViT-B/14 (torch hub, local)", "embedding": "x_norm_clstoken"})
        save_features(out / "shots.dino_vitl14.pt", ids, torch.cat(chunks["dino_vitl14"]),
                      {"model": "DINOv2 ViT-L/14 (torch hub, local)", "embedding": "x_norm_clstoken",
                       "ckpt_sha256": hashlib.sha256(ckpt.read_bytes()).hexdigest()})
    print(json.dumps({"model": opts.model, "n": len(ids), "seconds": round(time.time() - start, 1)}))


if __name__ == "__main__":
    main()
