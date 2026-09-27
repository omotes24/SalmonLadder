"""Rebuild control-bank features in the archived manifest order, using the frozen encoders."""
import argparse
import hashlib
import json
import os
import sys
import tarfile
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--workspace", type=Path, required=True)
    p.add_argument("--bank", choices=["cub", "cifar", "dev", "openood"], required=True)
    p.add_argument("--model", choices=["CLIP", "B14", "L14"], required=True)
    a = p.parse_args()
    workspace = a.workspace.resolve()
    control = workspace / "reprise_controls_20260927"
    sys.path.insert(0, str(control / "src"))
    import numpy as np
    import pandas as pd
    import torch
    from PIL import Image
    from data import pixel_hash, dump
    from bridge import Encoder
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    base = control / "data" / a.bank
    frame = pd.read_parquet(base / "images.parquet")
    info_path = base / "manifest_frozen.json"
    if not info_path.exists():
        info_path = base / "ready.json"
    info = json.loads(info_path.read_text())
    if a.bank == "cub" and not Path(frame.path.iloc[0]).exists():
        archive = control / "data/CUB_200_2011.tgz"
        digest = hashlib.md5()
        with archive.open("rb") as stream:
            for part in iter(lambda: stream.read(2**20), b""):
                digest.update(part)
        if digest.hexdigest() != "97eceeb196236b17998738112f37df78":
            raise ValueError("Official CUB archive checksum mismatch")
        with tarfile.open(archive) as source:
            source.extractall(control / "data", filter="data")
    if a.bank == "cifar":
        from torchvision.datasets import CIFAR100
        for train in [True, False]:
            source = CIFAR100(str(workspace / "datasets/cifar100"), train=train, download=False)
            for row in frame[frame.train == train].itertuples():
                path = Path(row.path)
                if not path.exists():
                    path.parent.mkdir(parents=True, exist_ok=True)
                    Image.fromarray(source.data[int(row.source_index)]).save(path)
    encoder = Encoder([a.model])
    chunks = []
    for lo in range(0, len(frame), 64):
        rows = frame.iloc[lo:lo+64]
        if "pixel_sha256" in rows:
            for row in rows.itertuples():
                if isinstance(row.pixel_sha256, str) and pixel_hash(row.path) != row.pixel_sha256:
                    raise ValueError(f"Image pixels differ from frozen manifest: {row.sample_id}")
        features, _ = encoder.encode(rows.path.tolist())
        chunks.append(features[a.model])
        if lo % 640 == 0:
            print(json.dumps({"bank": a.bank, "model": a.model, "done": lo+len(rows), "total": len(frame)}), flush=True)
    features = np.concatenate(chunks)
    tmp = base / f"{a.model}.{os.getpid()}.npy"
    np.save(tmp, features)
    os.replace(tmp, base / f"{a.model}.npy")
    if a.model == "CLIP":
        if a.bank == "dev":
            labels = [c["clean_name"] for c in json.loads((workspace / "vins_gonogo_20260925/splits/id_classes.json").read_text())]
        elif a.bank == "openood":
            labels = np.load(workspace / "vins_gonogo_20260925/tins/data/ImageNet/imagenet_class_clean.npy").tolist()
        else:
            labels = info["id_names"]
        pos = encoder.text(labels)
        np.save(base / "pos.npy", pos)
        candidates = np.concatenate([np.argsort(-(features[lo:lo+1024] @ pos.T), axis=1)[:, :5] for lo in range(0, len(features), 1024)])
        np.save(base / "cand.npy", candidates)
    if all((base / f"{m}.npy").exists() for m in ["CLIP", "B14", "L14"]) and (base / "cand.npy").exists():
        dump(base / "ready.json", info)
    dump(base / f"rebuilt_{a.model}.json", {"rows": len(features), "dtype": str(features.dtype), "manifest_order_preserved": True,
                                           "pixel_hash_check": "pixel_sha256" in frame})


if __name__ == "__main__":
    main()
