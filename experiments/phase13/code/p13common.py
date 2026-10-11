"""Salmon Ladder Phase 13 (paper v15): (a) arrival-time mode with converged propagation, (b) exchangeable calibration
(calibration images drawn from the ID validation images) for the realised FDR check, (c) the product with a static rank
on the benchmarks and on the recurrence grid, (d) the OpenOOD ImageNet-200 benchmark. Shared paths and helpers."""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "/home/omote/reprise_p12_20261011/code")
import p12common as P  # noqa: E402  (Phase 12 -> Phase 11/10 paths, Phase 5 code, vendored vins package)
from p12common import *  # noqa: E402,F401,F403

P13 = Path("/home/omote/reprise_p13_20261011")
RESULTS13 = P13 / "results"
LOGS13 = P13 / "logs"
IMGLIST = Path("/home/omote/openood_simple_sota/data/benchmark_imglist")
Q_PAPER = 0.2


def l2n(x):
    x = np.asarray(x, np.float32)
    return x / np.linalg.norm(x, axis=1, keepdims=True)


def id_labels(part, D):
    """OpenOOD label (0..999, sorted-wnid index) of every ID test image of the part, aligned to D['ids'] (-1 for OOD images).
    openood: test_imagenet.txt (45,000 val images); fourood: the 50,000 val images (label from the validation ground truth
    through the same list plus val_imagenet.txt)."""
    lab = {}
    for name in ("test_imagenet.txt", "val_imagenet.txt"):
        for line in (IMGLIST / "imagenet" / name).read_text().splitlines():
            if line.strip():
                p, y = line.split()
                lab[Path(p).name] = int(y)
    paths = P.image_paths(part)
    paths = paths[16000:] if part == "openood" else paths
    assert len(paths) == len(D["ids"]), (len(paths), len(D["ids"]))
    out = np.full(len(paths), -1, np.int64)
    for i, p in enumerate(paths):
        if "imagenet_1k/val/" in p or "/val/ILSVRC2012_val_" in p:
            out[i] = lab[Path(p).name]
    return out, paths


def proto_rank(sup, cal, sf, dev="cuda"):
    """Plain static baseline: cosine similarity to the nearest class prototype (mean of the 12 support features of the class),
    read as the calibrated rank among the calibration images (small = unknown). Returns log p for the test images."""
    import torch
    from engine import _log, p_low
    S = torch.as_tensor(np.asarray(sup, np.float32), device=dev)
    proto = S.reshape(-1, 12, S.shape[1]).mean(1)
    proto = proto / proto.norm(dim=1, keepdim=True)

    def best(X):
        X = torch.as_tensor(np.asarray(X, np.float32), device=dev)
        vals = []
        for lo in range(0, len(X), 8192):
            vals.append((X[lo:lo + 8192] @ proto.T).max(dim=1).values)
        return torch.cat(vals).cpu().numpy().astype(np.float64)

    return _log(p_low(best(cal), best(sf)))


def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=1, default=str) + "\n")
