"""Frozen image features for every dev sample (computed once, stream independent)."""
import hashlib
import time

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from . import config as C
from .sealing import Guard

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class GuardedImages(Dataset):
    def __init__(self, paths, transform, guard):
        self.paths = list(paths)
        self.transform = transform
        self.guard = guard

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path = self.guard.check(self.paths[index])
        image = Image.open(path).convert("RGB")      # same as upstream ImageListDataset
        return self.transform(image), index


@torch.no_grad()
def encode(paths, transform, encode_fn, guard, batch_size=256, num_workers=8, desc=""):
    loader = DataLoader(GuardedImages(paths, transform, guard), batch_size=batch_size, shuffle=False,
                        num_workers=num_workers, pin_memory=True)
    chunks, order = [], []
    start = time.time()
    for images, index in loader:
        feats = encode_fn(images.cuda(non_blocking=True)).float()
        feats = feats / feats.norm(dim=-1, keepdim=True)
        chunks.append(feats.cpu())
        order.append(index)
    order = torch.cat(order)
    assert torch.equal(order, torch.arange(len(paths)))
    out = torch.cat(chunks)
    print(f"[features] {desc}: {len(paths)} images in {time.time() - start:.1f}s", flush=True)
    return out


def dino_transform():
    from torchvision import transforms

    return transforms.Compose([
        transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def load_dino():
    digest = hashlib.sha256(C.DINO_CKPT.read_bytes()).hexdigest()
    if digest != C.DINO_SHA256:
        raise RuntimeError(f"unexpected DINOv2 checkpoint sha256 {digest}")
    model = torch.hub.load(str(C.DINO_HUB), "dinov2_vitb14", source="local", pretrained=False)
    state = torch.load(C.DINO_CKPT, map_location="cpu")
    model.load_state_dict(state, strict=True)
    return model.eval().cuda()


def dino_encode_fn(model):
    def fn(images):
        return model.forward_features(images)["x_norm_clstoken"]   # CLS after the final LayerNorm
    return fn


def save_features(path, sample_ids, feats, meta):
    torch.save({"sample_id": list(sample_ids), "features": feats, "meta": meta}, path)


def load_features(path, sample_ids=None):
    blob = torch.load(path, map_location="cpu")
    feats = blob["features"]
    if sample_ids is not None:
        pos = {s: i for i, s in enumerate(blob["sample_id"])}
        feats = feats[[pos[s] for s in sample_ids]]
    return feats, blob


def guard_from_seal():
    from .sealing import load_sealed

    sealed_paths, _, _ = load_sealed()
    return Guard(sealed_paths)
