"""Salmon Ladder Phase 10: the frozen method with stronger frozen views and the official VLM base detectors on the
public benchmarks (OpenOOD v1.5 ImageNet-1K near / far, Four-OOD). Shared paths. Everything new lives under P10.
The Phase 5 engine (engine5.run5, frozen V5 read-outs per view) is used unchanged; a view's read-outs do not depend on
the other views of the same run (checked in Phase 7), so every combination of views is formed in the analysis."""
import json
import os
import sys
import time
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

HOME = Path("/home/omote")
P10 = HOME / "reprise_p10_20261010"
P5 = HOME / "reprise_p5_20261002"
P4 = HOME / "reprise_p4_20260928"
P7 = HOME / "reprise_p7_20261003"
for _p in (P5 / "code", P7 / "code"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
import common as C5  # noqa: E402  (puts the vendored vins package on sys.path; V5, GONOGO, utc)

GONOGO = C5.GONOGO
V5 = C5.V5
FEAT = P10 / "features"
RESULTS = P10 / "results"
LOGS = P10 / "logs"
OO = ["ssb_hard", "ninco", "inaturalist", "textures", "openimageo"]
OO_NEAR, OO_FAR = ["ssb_hard", "ninco"], ["inaturalist", "textures", "openimageo"]
FOUR = ["inat", "sun", "places", "dtd"]
OO_SEEDS, FOUR_SEEDS = (123, 124, 125, 126, 127), (123, 124, 125)
BASE_VIEWS = ["B14", "L14"]                 # the frozen views (features of Phase 3)
NEW_VIEWS = ["D3B", "D3L"]                  # DINOv3 ViT-B/16, ViT-L/16 (Phase 6/7 read-out and input)
utc = C5.utc
dump = C5.dump


def stream_tasks(part):
    """(dataset, seed, path of the Phase 3 stream file with sample_id / is_ood / S_final (TINS) / batch_index)."""
    from vins import config as C
    if part == "openood":
        return [(ds, s, C.WORK / "test_runs" / "default" / f"{ds}_seed{s}.npz") for ds in OO for s in OO_SEEDS]
    return [(o, s, C.WORK / "extra_runs" / "fourood" / f"imagenet_{o}_seed{s}.npz") for o in FOUR for s in FOUR_SEEDS]


def load_p3(part):
    """Phase 3 feature space of the part: shots (16,000 rows), evaluated images, CLIP candidates, row map."""
    import importlib.util
    spec_ = importlib.util.spec_from_file_location("p3_eval", GONOGO / "scripts" / "p3_eval.py")
    P3E = importlib.util.module_from_spec(spec_)
    spec_.loader.exec_module(P3E)
    return P3E.load_part(part)


def image_paths(part):
    """Paths of the images in the row order of load_p3(part): openood = 16,000 shots + evaluated images (the order of
    features.pt); fourood = the evaluated images only (in_val, inat, sun, places, dtd), as p3_extract.py."""
    import torch
    from vins import config as C
    b = torch.load(C.WORK / "test_eval" / "results" / "features.pt", map_location="cpu")
    if part == "openood":
        return [str(p) for p in b["paths"]]
    names = ["in_val"] + FOUR
    blobs = {n: torch.load(C.WORK / "extra_feats" / f"{n}.dino.pt", map_location="cpu") for n in names}
    return sum((list(map(str, blobs[n]["paths"])) for n in names), [])


def add_views(D, part, views):
    """Adds the Phase 10 views (L2-normalised float32) to the Phase 3 feature space D."""
    import numpy as np
    oo = {v: np.load(FEAT / f"openood.{v}.npy", mmap_mode="r") for v in views}
    for v in views:
        shots = np.array(oo[v][:16000], dtype=np.float32)          # copies (the memory map is read-only)
        ev = np.array(oo[v][16000:], dtype=np.float32) if part == "openood" else np.load(FEAT / f"fourood.{v}.npy").astype(np.float32)
        for a in (shots, ev):
            a /= np.linalg.norm(a, axis=1, keepdims=True)
        assert len(ev) == len(D["ids"]), (v, len(ev), len(D["ids"]))
        D["shots"][v], D["ev"][v] = shots, ev
    return D
