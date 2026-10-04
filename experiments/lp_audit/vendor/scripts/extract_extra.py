"""Frozen features for the additional evaluation sets (post-test analyses; images are read on purpose).

Datasets are listed in the order torchvision's ImageFolder (or TINS's NumericImageFolderDataset) would use:
  in_val     ImageNet-1K val 50,000 (OpenOOD test_imagenet + val_imagenet lists), sorted by (label, file)
  inat sun places  MOS subsets (single folder "images"), dtd (DTD, 47 class folders)
  in_v2 in_r in_sketch  ImageNet-V2 matched-frequency, ImageNet-R (200 classes), ImageNet-Sketch
  oo_test    the unique OpenOOD test images in the order of test_eval/results/features.pt (for CLIP ViT-L/14)
Models: dino (DINOv2 ViT-B/14 CLS), dinol14 (DINOv2 ViT-L/14 CLS), clipb16 / clipl14 (CLIP image encoder, upstream TINS encoding path).
Output: <WORK>/extra_feats/<dataset>.<model>.pt with ids, paths, labels (1000-class index or -1), wnids, features.
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402
from PIL import Image  # noqa: E402
from torch.utils.data import DataLoader, Dataset  # noqa: E402

from vins import config as C  # noqa: E402
from vins.features import dino_encode_fn, dino_transform, load_dino  # noqa: E402

X = Path.home() / "datasets" / "extra_ood"
IMG = C.OPENOOD_IMAGES
LISTS = C.OPENOOD_IMGLIST_DIR
EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def wnids_1k():
    return sorted(p.name for p in (C.IMAGENET_ROOT / "train").iterdir() if p.is_dir())


def image_files(folder):
    return sorted(str(p) for p in Path(folder).iterdir() if p.is_file() and p.suffix.lower() in EXTS)


def imagefolder(root):
    classes = sorted(p.name for p in Path(root).iterdir() if p.is_dir())
    items = [(f, c) for c in classes for f in image_files(Path(root) / c)]
    return classes, items


def build(name):
    w = wnids_1k()
    idx = {x: i for i, x in enumerate(w)}
    if name == "in_val":
        rows = []
        for lst in ("test_imagenet.txt", "val_imagenet.txt"):
            for line in (LISTS / lst).read_text().splitlines():
                if line.strip():
                    rel, lab = line.split()
                    rows.append((int(lab), str(IMG / rel)))
        rows.sort()
        return [p for _, p in rows], [lab for lab, _ in rows], [w[lab] for lab, _ in rows]
    if name in ("inat", "sun", "places"):
        folder = {"inat": "iNaturalist", "sun": "SUN", "places": "Places"}[name]
        classes, items = imagefolder(X / folder)
        return [f for f, _ in items], [-1] * len(items), [c for _, c in items]
    if name == "dtd":
        classes, items = imagefolder(X / "dtd" / "images")
        return [f for f, _ in items], [-1] * len(items), [c for _, c in items]
    if name == "in_v2":
        root = X / "imagenetv2-matched-frequency-format-val"
        classes = sorted((p for p in root.iterdir() if p.is_dir()), key=lambda p: int(p.name))
        items = [(str(f), int(c.name)) for c in classes for f in sorted(c.rglob("*")) if f.is_file() and f.suffix.lower() in EXTS]
        return [f for f, _ in items], [lab for _, lab in items], [w[lab] for _, lab in items]
    if name in ("in_r", "in_sketch"):
        root = X / "imagenet-r" if name == "in_r" else X / "imagenet-sketch" / "sketch"
        classes, items = imagefolder(root)
        return [f for f, _ in items], [idx[c] for _, c in items], [c for _, c in items]
    if name == "oo_test":
        blob = torch.load(C.WORK / "test_eval" / "results" / "features.pt", map_location="cpu")
        paths = blob["paths"][16000:]
        return paths, [-1] * len(paths), [""] * len(paths)
    raise ValueError(name)


class Images(Dataset):
    def __init__(self, paths, transform):
        self.paths, self.transform = paths, transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        return self.transform(Image.open(self.paths[i]).convert("RGB")), i


@torch.no_grad()
def run(paths, transform, fn, batch):
    out, order = [], []
    for images, index in DataLoader(Images(paths, transform), batch_size=batch, shuffle=False, num_workers=10,
                                    pin_memory=True):
        f = fn(images.cuda(non_blocking=True)).float()
        out.append((f / f.norm(dim=-1, keepdim=True)).cpu())
        order.append(index)
    assert torch.equal(torch.cat(order), torch.arange(len(paths)))
    return torch.cat(out)


def load_clip_model(arch):
    sys.path.insert(0, str(C.TINS_DIR / "third_party"))
    import openai_clip  # noqa: E402  (TINS's vendored OpenAI CLIP, same loader as upstream)

    root = {"ViT-B/16": C.CLIP_WEIGHTS_DIR, "ViT-L/14": C.WORK / "weights"}[arch]
    original = openai_clip.clip._download
    openai_clip.clip._download = lambda url, r=None: original(url, str(root))
    try:
        model, preprocess = openai_clip.load(arch, device="cuda", jit=False)
    finally:
        openai_clip.clip._download = original
    return model.eval(), preprocess


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", required=True)
    parser.add_argument("--models", nargs="+", default=["dino", "clipb16"])
    opts = parser.parse_args()
    out = C.WORK / "extra_feats"
    out.mkdir(parents=True, exist_ok=True)
    for model_name in opts.models:
        if model_name == "dino":
            model, transform = load_dino(), dino_transform()
            fn, batch = dino_encode_fn(model), 128
        elif model_name == "dinol14":
            model = torch.hub.load(str(C.DINO_HUB), "dinov2_vitl14", source="local", pretrained=False)
            ckpt = C.HOME / ".cache" / "torch" / "hub" / "checkpoints" / "dinov2_vitl14_pretrain.pth"
            model.load_state_dict(torch.load(ckpt, map_location="cpu"), strict=True)
            model, transform = model.eval().cuda(), dino_transform()
            fn, batch = dino_encode_fn(model), 64
        else:
            arch = {"clipb16": "ViT-B/16", "clipl14": "ViT-L/14"}[model_name]
            model, transform = load_clip_model(arch)
            fn, batch = model.encode_image, (256 if arch == "ViT-B/16" else 96)
        for name in opts.datasets:
            target = out / f"{name}.{model_name}.pt"
            if target.exists():
                continue
            paths, labels, wn = build(name)
            tick = time.time()
            feats = run(paths, transform, fn, batch)
            ids = [f"{name}_{k:06d}" for k in range(len(paths))]
            torch.save({"ids": ids, "paths": paths, "labels": labels, "wnids": wn, "features": feats}, target)
            print(json.dumps({"dataset": name, "model": model_name, "n": len(paths),
                              "seconds": round(time.time() - tick, 1)}), flush=True)
        del model
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
