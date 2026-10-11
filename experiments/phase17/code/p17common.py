"""Salmon Ladder Phase 17 (registered, prereg_p17.json): (1) sigma_ref, the read-out of the paper with reference images instead of
folds (12 support / 2 reference / 2 calibration images per class; seeds chosen from all calibration and test images by Storey-BH on
the rank among the reference images; one calibration set, no folds); (2) descriptive analyses without selection: the decomposition
u = sqrt(delta) x (probability) on the benchmarks, the separation bounds (Corollary 2 for sigma / sigma_ref, Corollary A1 for h)
on the recurrence grid, and covariate-shifted ID images (ImageNet-V2, ImageNet-R). Shared paths and the fixed configuration."""
import json
import sys
from pathlib import Path

sys.path.insert(0, "/home/omote/reprise_p16_20261011/code")
import p16common as _M16  # noqa: E402  (Phase 16 -> 15 -> 14 -> 13 -> 12 -> 11/10 paths, Phase 5 code, vendored vins package)
from p16common import *  # noqa: E402,F401,F403
M16 = _M16
M15 = _M16.M15
P = _M16.P                 # Phase 13's namespace (storey_adadetect, plain_bh, load_p3, add_views, l2n, utc, ...)

P17 = Path("/home/omote/reprise_p17_20261011")
RESULTS17 = P17 / "results"
LOGS17 = P17 / "logs"
TABLES17 = P17 / "tables"
FEATS17 = P17 / "features"

# ---- the fixed configuration of sigma_ref (nothing is selected: every value is the paper's or Phase 16's)
VIEWS17 = ("L14", "D3L")   # the paper's two views (selected in Phase 12)
LAM17 = 0.9                # the paper's lambda of the propagation (eq. lp)
Q17 = 0.2                  # the paper's level of the Storey-BH seed rule (uncapped pi0_hat, AdaDetect Cor. 3.7)
N_REF = 2                  # of the 4 calibration images of each class (stored class-major order): positions 0, 1 -> reference R
N_CAL_PER_CLASS = 4        #                                                                            positions 2, 3 -> calibration C
                           # (the same split as Phase 16)
# ---- the registered adoption rule (dev2, 5 draws, paired 95% t interval over draws)
ADOPT_MARGIN = 0.5         # upper bound of the interval of FPR95(sigma_ref) - FPR95(sigma) must be <= +0.5 on near and on far
ADOPT_FPR = {0.05: 0.06, 0.2: 0.21}   # false-alarm rate of p_fin(sigma_ref) at alpha on dev2 (near and far pooled per draw, mean)

EXTRA = Path("/home/omote/datasets/extra_ood")
CSID_ROOTS = {"imagenet_v2": EXTRA / "imagenetv2-matched-frequency-format-val", "imagenet_r": EXTRA / "imagenet-r"}
CSID_LISTS = {"imagenet_v2": "test_imagenet_v2.txt", "imagenet_r": "test_imagenet_r.txt"}   # OpenOOD v1.5 full-spectrum lists


def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=1, default=str) + "\n")
