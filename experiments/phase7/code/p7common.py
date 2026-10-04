"""REPRISE Phase 7 (additional experiments). Shared paths and helpers.

Read-only inputs: the Phase 4 / Phase 5 code, banks, features and scores. New files live under P7.
No OpenOOD test image, ImageNet val image or Four-OOD image is opened (sealed manifest guard in every extraction)."""
import os
import sys
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")            # cached weights only: nothing is downloaded in this phase
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

HOME = Path("/home/omote")
P4 = HOME / "reprise_p4_20260928"
P5 = HOME / "reprise_p5_20261002"
P7 = HOME / "reprise_p7_20261003"
GONOGO = HOME / "vins_gonogo_20260925"
for _p in (str(P4 / "code"), str(P5 / "code")):         # P5 first: engine, engine5, static, metrics_p4, common, dev5 ...
    if _p in sys.path:
        sys.path.remove(_p)
    sys.path.insert(0, _p)

import common  # noqa: E402,F401  (Phase 4/5 module: adds the vendored vins package to sys.path)
from common import V5, dump, sha_file, utc  # noqa: E402,F401

BANKS = P7 / "banks"
FEATS = P7 / "features"
RESULTS = P7 / "results"
SEED = 20261003
