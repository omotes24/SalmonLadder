"""Phase 7 frozen features for a table of images (sample_id, path). Sealed images are refused.

Models (all fp32, TF32 off, L2-normalised float32 output; cached weights only):
  S14 B14 L14 G14  DINOv2 ViT-S/B/L/g-14 CLS (torch hub, local checkpoints); Resize 256 bicubic + CenterCrop 224
  D3B D3L          DINOv3 ViT-B/16, ViT-L/16 CLS after the final norm; Resize 293 + CenterCrop 256 (as Phase 6)
  D3S D3SP         DINOv3 ViT-S/16, ViT-S+/16, the same read-out and input (amendment 03)
  DINO1 MAE        timm vit_base_patch16_224.dino / .mae (ImageNet-1K images, no labels), pooled CLS; DINOv2 input
  CLIP             CLIP ViT-B/16 image embedding through the pinned TINS loader and preprocessing (the frozen path)
  CLIPL SIG2L      CLIP ViT-L/14 and SigLIP2 large patch16-256 image embeddings (transformers, own processors)
  RN50             CLIP RN50 image embedding (OpenAI weights through the CLIP package vendored in TINS; fp16 on the GPU
                   as the CLIP view; its own preprocessing) (amendment 03)
Usage: extract7.py --table T.parquet --out DIR --models A,B --shard i --nshards n ; extract7.py --merge ...
Output: DIR/<tag>.shard<i>of<n>.npz, merged into DIR/<model>.npy (rows of the table)."""
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

from p7common import HOME, P7, dump, sha_file, utc
from vins.features import IMAGENET_MEAN, IMAGENET_STD, dino_transform
from vins.sealing import Guard, load_sealed

CKPT = HOME / ".cache" / "torch" / "hub" / "checkpoints"
HUB = HOME / ".cache" / "torch" / "hub" / "facebookresearch_dinov2_main"
DINOV2 = {"S14": "vits14", "B14": "vitb14", "L14": "vitl14", "G14": "vitg14"}
TIMM_CLS0 = {"D3B": "vit_base_patch16_dinov3.lvd1689m", "D3L": "vit_large_patch16_dinov3.lvd1689m",
             "D3S": "vit_small_patch16_dinov3.lvd1689m", "D3SP": "vit_small_plus_patch16_dinov3.lvd1689m"}
TIMM_POOL = {"DINO1": "vit_base_patch16_224.dino", "MAE": "vit_base_patch16_224.mae"}
HF = {"CLIPL": "openai/clip-vit-large-patch14", "SIG2L": "google/siglip2-large-patch16-256"}
OPENAI = {"RN50": ("RN50", "afeb0e10f9e5a86da6080e35cf09123aca3b358a0c3e3b6c78a7b63bc04b6762")}   # architecture, sha256 of the checkpoint
WEIGHTS = P7 / "weights"
ALL = list(DINOV2) + list(TIMM_CLS0) + list(TIMM_POOL) + ["CLIP"] + list(HF) + list(OPENAI)


def openai_clip_model(arch, dev):
    """OpenAI CLIP through the package vendored in TINS (the loader of the CLIP view), checkpoint kept under P7/weights.
    The package verifies the sha256 of the file; on the GPU the weights are half precision, as in the CLIP view."""
    import sys

    from vins import config as C

    tp = str(C.TINS_DIR / "third_party")
    if tp not in sys.path:
        sys.path.insert(0, tp)
    import openai_clip

    WEIGHTS.mkdir(parents=True, exist_ok=True)
    original = openai_clip.clip._download
    openai_clip.clip._download = lambda url, r=None: original(url, str(WEIGHTS))
    try:
        model, preprocess = openai_clip.load(arch, device=dev, jit=False)
    finally:
        openai_clip.clip._download = original
    return model.eval(), preprocess


def transform256():
    from torchvision import transforms

    return transforms.Compose([
        transforms.Resize(293, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(256),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def pooled(x):
    return x if torch.is_tensor(x) else (x.pooler_output if getattr(x, "pooler_output", None) is not None else x[0])


def load(name, dev, cache_dir):
    """Returns (encode function on a batch tensor, transform key, transform, number of parameters used)."""
    if name in DINOV2:
        tag = DINOV2[name]
        m = torch.hub.load(str(HUB), "dinov2_" + tag, source="local", pretrained=False)
        m.load_state_dict(torch.load(CKPT / f"dinov2_{tag}_pretrain.pth", map_location="cpu"), strict=True)
        m = m.eval().to(dev)
        return (lambda x: m.forward_features(x)["x_norm_clstoken"]), "dino224", dino_transform(), sum(p.numel() for p in m.parameters())
    if name in TIMM_CLS0 or name in TIMM_POOL:
        import timm

        m = timm.create_model({**TIMM_CLS0, **TIMM_POOL}[name], pretrained=True, num_classes=0).eval().to(dev)
        n = sum(p.numel() for p in m.parameters())
        if name in TIMM_CLS0:
            return (lambda x: m.forward_features(x)[:, 0]), "dino256", transform256(), n
        assert getattr(m, "global_pool", "token") == "token"
        return (lambda x: m(x)), "dino224", dino_transform(), n
    if name == "CLIP":
        from vins.tins_dev import import_tins, load_clip, make_args

        cwd = os.getcwd()
        t = import_tins()
        args = make_args(t, Path(cache_dir) / "clip_cache", "p7_encoder")
        clip, pre = load_clip(t, args)
        os.chdir(cwd)
        clip = clip.eval()
        dt = next(clip.parameters()).dtype
        return (lambda x: clip.encode_image(x.to(dt))), "clip_tins", pre, sum(p.numel() for p in clip.visual.parameters())
    if name in OPENAI:
        clip, pre = openai_clip_model(OPENAI[name][0], dev)
        dt = next(clip.parameters()).dtype
        return (lambda x: clip.encode_image(x.to(dt))), "clip_" + name.lower(), pre, sum(p.numel() for p in clip.visual.parameters())
    if name == "CLIPL":
        from transformers import CLIPModel, CLIPProcessor

        full = CLIPModel.from_pretrained(HF[name], local_files_only=True).eval()
        proc = CLIPProcessor.from_pretrained(HF[name], local_files_only=True)
        vm, vp = full.vision_model.to(dev), full.visual_projection.to(dev)
        tf = lambda img: proc(images=img, return_tensors="pt")["pixel_values"][0]
        n = sum(p.numel() for p in vm.parameters()) + sum(p.numel() for p in vp.parameters())
        return (lambda x: vp(vm(pixel_values=x).pooler_output)), "hf_clipl", tf, n
    if name == "SIG2L":
        from transformers import AutoModel, AutoProcessor

        full = AutoModel.from_pretrained(HF[name], local_files_only=True).eval()
        proc = AutoProcessor.from_pretrained(HF[name], local_files_only=True)
        vm = full.vision_model.to(dev)
        tf = lambda img: proc(images=img, return_tensors="pt")["pixel_values"][0]
        return (lambda x: vm(pixel_values=x).pooler_output), "hf_sig2l", tf, sum(p.numel() for p in vm.parameters())
    raise ValueError(name)


class Images(Dataset):
    def __init__(self, paths, tfs, guard):
        self.paths, self.tfs, self.guard = list(paths), tfs, guard

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        with Image.open(self.guard.check(self.paths[i])) as im:
            im = im.convert("RGB")
            return tuple(t(im) for t in self.tfs) + (i,)


@torch.no_grad()
def run_shard(a):
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    names = a.models.split(",")
    out_dir = Path(a.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = "-".join(names)
    par = f"p{a.parent.replace(',', 'of')}." if a.parent else ""
    out = out_dir / (f"{tag}.{par}shard{a.shard}of{a.nshards}" + (f".limit{a.limit}" if a.limit else "") + ".npz")
    if out.exists():
        print(json.dumps({"exists": str(out)}))
        return
    table = pd.read_parquet(a.table)
    rows = np.arange(len(table))
    if a.parent:                                   # re-split one shard of an earlier sharding: rows[i::n][shard::nshards]
        pi, pn = (int(x) for x in a.parent.split(","))
        rows = rows[pi::pn]
    rows = rows[a.shard::a.nshards]
    if a.limit:
        rows = rows[:a.limit]
    dev = "cuda"
    enc, tkey, nparam = {}, {}, {}
    tfs = {}
    for n in names:
        enc[n], tkey[n], tf, nparam[n] = load(n, dev, out_dir)
        tfs.setdefault(tkey[n], tf)
    keys = list(tfs)
    sealed_paths, _, _ = load_sealed()
    ds = Images(table.path.values[rows], [tfs[k] for k in keys], Guard(sealed_paths))
    dl = DataLoader(ds, batch_size=a.batch, shuffle=False, num_workers=a.workers, pin_memory=True)
    buf = {n: [] for n in names}
    order, t0 = [], time.time()
    for it, batch in enumerate(dl):
        x = {k: batch[j].to(dev, non_blocking=True) for j, k in enumerate(keys)}
        for n in names:
            xi = x[tkey[n]]
            f = torch.cat([pooled(enc[n](xi[lo:lo + a.sub])).float() for lo in range(0, len(xi), a.sub)])
            buf[n].append((f / f.norm(dim=-1, keepdim=True)).cpu().numpy().astype(np.float32))
        order.append(batch[-1].numpy())
        if it % 25 == 0:
            done = sum(len(o) for o in order)
            print(json.dumps({"tag": tag, "shard": a.shard, "done": done, "total": len(rows),
                              "img_per_s": round(done / (time.time() - t0), 1), "utc": utc()}), flush=True)
    order = np.concatenate(order)
    assert np.array_equal(order, np.arange(len(rows)))
    arrays = {n: np.concatenate(v) for n, v in buf.items()}
    for n, f in arrays.items():
        assert np.isfinite(f).all(), n
    tmp = str(out) + f".{os.getpid()}.tmp.npz"
    np.savez(tmp, rows=rows, **arrays)
    os.replace(tmp, out)
    print(json.dumps({"tag": tag, "shard": a.shard, "complete": str(out), "n": len(rows), "seconds": round(time.time() - t0, 1),
                      "dims": {n: int(f.shape[1]) for n, f in arrays.items()}, "params_M": {n: round(v / 1e6, 1) for n, v in nparam.items()}}), flush=True)


def merge(a):
    names = a.models.split(",")
    tag = "-".join(names)
    out_dir = Path(a.out)
    table = pd.read_parquet(a.table)
    files = sorted(f for f in out_dir.glob(f"{tag}.*shard*of*.npz") if ".limit" not in f.name and ".tmp" not in f.name)
    parts = [np.load(f) for f in files]
    print("merging", [f.name for f in files])
    info = {"utc": utc(), "n": len(table), "table_sha256": sha_file(a.table), "files": {}}
    for n in names:
        full = np.zeros((len(table), parts[0][n].shape[1]), np.float32)
        filled = np.zeros(len(table), int)
        for z in parts:
            full[z["rows"]] = z[n]
            filled[z["rows"]] += 1
        assert (filled == 1).all(), (n, int((filled == 0).sum()), int((filled > 1).sum()))
        assert np.allclose(np.linalg.norm(full, axis=1), 1.0, atol=1e-4)
        np.save(out_dir / f"{n}.npy", full)
        info["files"][n] = {"sha256": sha_file(out_dir / f"{n}.npy"), "dim": int(full.shape[1])}
    dump(out_dir / f"info_{tag}.json", info)
    print("merged", tag, len(table), {n: v["dim"] for n, v in info["files"].items()})


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--table", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--models", required=True)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=4)
    ap.add_argument("--parent", default="", help="'i,n': take rows[i::n] first (re-split of one shard)")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--sub", type=int, default=64)
    ap.add_argument("--workers", type=int, default=5)
    ap.add_argument("--merge", action="store_true")
    a = ap.parse_args()
    if a.merge:
        merge(a)
    else:
        run_shard(a)
