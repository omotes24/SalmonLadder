"""Test-split sealing.

The OpenOOD ImageNet-1K test lists and val_imagenet (together: all of ImageNet val plus the
OpenOOD near/far test sets) are sealed. The dev pipeline never opens those images. We only read
the path/sha256 manifest built earlier (no pixels) to (1) refuse any sealed path at load time and
(2) drop dev candidates that are byte-identical to a sealed image.
"""
import json
import os
from collections import Counter

from . import config as C

# manifest 'dataset' names that correspond to SEALED_LIST_FILES
MANIFEST_DATASETS = {
    "imagenet": "test_imagenet.txt",
    "ssb_hard": "test_ssb_hard.txt",
    "ninco": "test_ninco.txt",
    "inaturalist": "test_inaturalist.txt",
    "textures": "test_textures.txt",
    "openimage_o": "test_openimage_o.txt",
    "imagenet_calibration": "val_imagenet.txt",
}


class SealedAccessError(RuntimeError):
    pass


def load_sealed(manifest=None):
    manifest = manifest or C.SEAL_MANIFEST
    paths, hashes, counts = set(), set(), Counter()
    with open(manifest) as handle:
        for line in handle:
            row = json.loads(line)
            paths.add(os.path.realpath(row["path"]))
            hashes.add(row["sha256"])
            counts[row["dataset"]] += 1
    return paths, hashes, dict(counts)


def sealed_list_counts():
    """Line counts of the sealed list files (paths only), to prove the manifest covers them."""
    out = {}
    for name in C.SEALED_LIST_FILES:
        with open(C.OPENOOD_IMGLIST_DIR / name) as handle:
            out[name] = sum(1 for line in handle if line.strip())
    return out


def check_manifest_coverage(counts):
    lists = sealed_list_counts()
    problems = []
    for dataset, list_name in MANIFEST_DATASETS.items():
        if counts.get(dataset) != lists[list_name]:
            problems.append((dataset, counts.get(dataset), list_name, lists[list_name]))
    return problems


class Guard:
    """Refuses to hand out any sealed path. Every image read in this project goes through check()."""

    def __init__(self, sealed_paths):
        self.sealed = sealed_paths

    def check(self, path):
        real = os.path.realpath(path)
        if real in self.sealed:
            raise SealedAccessError(f"sealed image requested: {real}")
        return real
