"""REPRISE Phase 4 (minimal configuration + independent confirmation). Shared paths and helpers.

Read-only inputs: the frozen REPRISE/TINS vendor code of the LP audit, the dev1 caches and ImageNet train.
All new files live under ROOT. No OpenOOD test image is ever opened (sealed manifest guard).
"""
import hashlib
import json
import os
import sys
import time
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

HOME = Path("/home/omote")
ROOT = HOME / "reprise_p4_20260928"
AUDIT = HOME / "reprise_lp_audit_20260927"
VENDOR = AUDIT / "vendor"
GONOGO = HOME / "vins_gonogo_20260925"
R5 = GONOGO / "r5"
IMAGENET = Path("/home/omote/datasets/openood_official/images_largescale/imagenet_1k")
IMAGENET_O = Path("/home/omote/datasets/ood_data/official_imagenet_o/extracted")
WILD = HOME / "WILD_DATA2"
BANKS = ROOT / "banks"
FEATS = ROOT / "features"
RESULTS = ROOT / "results"

if str(VENDOR) not in sys.path:
    sys.path.insert(0, str(VENDOR))
if str(AUDIT) not in sys.path:
    sys.path.insert(0, str(AUDIT))

V5 = {"n0": 48, "K": 20, "m": 1, "kg": 10, "gamma": 1.0, "lam": 0.9,
      "thresholds": (0.3, 0.2, 0.10191613435745239)}
DRAWS = 5
ORDER_SEEDS = (123, 124, 125)


def dump(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=1, allow_nan=False) + "\n")
    os.replace(tmp, path)


def sha_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def utc():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
