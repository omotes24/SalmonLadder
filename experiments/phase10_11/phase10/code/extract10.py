"""Phase 10 features of the public-benchmark images for the new frozen views (DINOv3 ViT-B/16 and ViT-L/16; CLS after
the final norm, Resize 293 + CenterCrop 256, fp32, TF32 off, L2-normalised float32), exactly the Phase 6/7 read-out and
input. Models and transforms come from Phase 7's extract7.load (cached weights only, HF_HUB_OFFLINE).
  --part openood : the 16,000 shots + every evaluated image of the OpenOOD v1.5 ImageNet-1K test streams
                   (the row order of features.pt)
  --part fourood : the evaluated images of Four-OOD (in_val, inat, sun, places, dtd; the order of p3_extract.py)
The images of the public test sets are opened on the user's instruction (Phase 10 evaluates on them); no sealing
guard is applied here. Output: P10/features/<part>.<model>.shard<i>of<n>.npz -> merged <part>.<model>.npy."""
import argparse
import hashlib
import json
import time

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

import p10common as P
from extract7 import load as load_model  # Phase 7 encoders (DINOv2 hub checkpoints, DINOv3 timm cache, ...)


class Images(Dataset):
    def __init__(self, paths, tf):
        self.paths, self.tf = list(paths), tf

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        with Image.open(self.paths[i]) as im:
            im = im.convert("RGB")
        return self.tf(im), i


def sha_list(paths):
    return hashlib.sha256("\n".join(paths).encode()).hexdigest()


@torch.no_grad()
def extract(fn, tf, paths, batch, dev):
    feats, order = [], []
    for x, i in DataLoader(Images(paths, tf), batch_size=batch, num_workers=6, shuffle=False, pin_memory=True):
        f = fn(x.to(dev, non_blocking=True)).float()
        feats.append((f / f.norm(dim=-1, keepdim=True)).cpu().numpy().astype(np.float32))
        order.append(i.numpy())
    order = np.concatenate(order)
    assert np.array_equal(order, np.arange(len(paths)))
    return np.concatenate(feats)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", required=True, choices=["openood", "fourood"])
    ap.add_argument("--models", default=",".join(P.NEW_VIEWS))
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--merge", action="store_true")
    ap.add_argument("--batch", type=int, default=64)
    a = ap.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    P.FEAT.mkdir(parents=True, exist_ok=True)
    paths = P.image_paths(a.part)
    n = len(paths)
    digest = sha_list(paths)
    if a.merge:
        for m in a.models.split(","):
            parts = [np.load(P.FEAT / f"{a.part}.{m}.shard{i}of{a.nshards}.npz") for i in range(a.nshards)]
            out = np.zeros((n, parts[0]["feat"].shape[1]), np.float32)
            seen = np.zeros(n, bool)
            for z in parts:
                assert str(z["paths_sha256"]) == digest
                rows = z["rows"]
                out[rows] = z["feat"]
                seen[rows] = True
            assert seen.all(), m
            np.save(P.FEAT / f"{a.part}.{m}.npy", out)
            P.dump(P.FEAT / f"{a.part}.{m}.json", {"part": a.part, "model": m, "rows": n, "dim": int(out.shape[1]),
                                                      "paths_sha256": digest, "utc": P.utc(), "nshards": a.nshards})
            print(json.dumps({"merged": f"{a.part}.{m}", "rows": n, "dim": int(out.shape[1])}), flush=True)
        return
    rows = np.arange(n)[a.shard::a.nshards]
    dev = "cuda"
    for m in a.models.split(","):
        f_out = P.FEAT / f"{a.part}.{m}.shard{a.shard}of{a.nshards}.npz"
        if f_out.exists():
            continue
        t0 = time.time()
        fn, key, tf, nparam = load_model(m, dev, str(P.P10 / "cache"))
        feat = extract(fn, tf, [paths[i] for i in rows], a.batch, dev)
        np.savez(f_out, rows=rows, feat=feat, paths_sha256=digest, transform=key)
        print(json.dumps({"part": a.part, "model": m, "shard": a.shard, "rows": int(len(rows)), "dim": int(feat.shape[1]),
                          "params_M": round(nparam / 1e6, 1), "img_per_s": round(len(rows) / (time.time() - t0), 1), "utc": P.utc()}), flush=True)
        del fn
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
