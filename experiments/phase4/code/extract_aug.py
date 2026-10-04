"""Near-duplicate control of Exp 4: 20 weak augmentations (seeded) of one probe-class image per probe class,
encoded with the same three frozen encoders. Output features/aug_{B14,L14,CLIP}.npy + banks/exp4/aug_table.parquet."""
import hashlib
import json
import os

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torchvision import transforms

from common import BANKS, FEATS
from extract import load_models
from vins.features import dino_transform

def augment(im, seed):
    torch.manual_seed(seed)
    w, h = im.size
    t = transforms.Compose([transforms.RandomResizedCrop((h, w), scale=(0.8, 1.0), ratio=(0.9, 1.1)),
                            transforms.RandomHorizontalFlip(0.5),
                            transforms.ColorJitter(0.1, 0.1, 0.1)])
    return t(im)


@torch.no_grad()
def main():
    pool = pd.read_parquet(BANKS / "imagenet_pool.parquet").set_index("sample_id")
    rows = []
    for k in range(1, 6):
        dz = json.loads((BANKS / "exp4" / f"split{k}.json").read_text())
        for w in dz["probe"]:
            src = dz["probes"][w]["dup_src"]
            for j in range(20):
                rows.append({"sample_id": f"aug/s{k}/{w}/{j}", "src": src, "path": pool.loc[src, "path"], "split": k,
                             "wnid": w, "j": j, "seed": int(hashlib.sha256(f"{k}/{w}/{j}".encode()).hexdigest()[:8], 16)})
    table = pd.DataFrame(rows)
    models, pre = load_models()
    tf = dino_transform()
    out = {n: [] for n in models}
    for lo in range(0, len(table), 64):
        chunk = table.iloc[lo:lo + 64]
        ims = []
        for r in chunk.itertuples():
            with Image.open(r.path) as im:
                ims.append(augment(im.convert("RGB"), r.seed))
        xd = torch.stack([tf(im) for im in ims]).cuda()
        xc = torch.stack([pre(im) for im in ims]).cuda()
        for n in ("B14", "L14"):
            f = models[n].forward_features(xd)["x_norm_clstoken"].float()
            out[n].append((f / f.norm(dim=-1, keepdim=True)).cpu().numpy())
        f = models["CLIP"].encode_image(xc.to(next(models["CLIP"].parameters()).dtype)).float()
        out["CLIP"].append((f / f.norm(dim=-1, keepdim=True)).cpu().numpy())
    for n in out:
        np.save(FEATS / f"aug_{n}.npy", np.concatenate(out[n]).astype(np.float32))
    table.to_parquet(BANKS / "exp4" / "aug_table.parquet", index=False)
    print(json.dumps({"aug": len(table)}))


if __name__ == "__main__":
    main()
