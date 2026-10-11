"""Salmon Ladder Phase 16 (registered): one quantity h — the probability that the random walk from an image reaches a support
image before a seed and before it vanishes, with a price per jump (1 - lambda) and a price per unit time (gamma). Shared paths
and the fixed configuration (prereg_p16.json)."""
import json
import sys
from pathlib import Path

sys.path.insert(0, "/home/omote/reprise_p15_20261011/code")
import p15common as _M15  # noqa: E402  (Phase 15 -> 14 -> 13 -> 12 -> 11/10 paths, Phase 5 code, vendored vins package)
from p15common import *  # noqa: E402,F401,F403
M15 = _M15
P = _M15.P                 # Phase 13's namespace (storey_adadetect, plain_bh, load_p3, add_views, l2n, utc, ...)

P16 = Path("/home/omote/reprise_p16_20261011")
RESULTS16 = P16 / "results"
LOGS16 = P16 / "logs"

# ---- the fixed configuration (registered; chosen on dev1 during the Phase 16 exploration, see prereg_p16.json)
LAM16 = 0.95               # survival per jump (price per jump 1 - lambda)
GREL16 = 1.0 / 16.0        # gamma = GREL16 * median degree of the degree-normalised union graph (price per unit time)
Q16 = 0.2                  # Storey-BH level of the seed selection (uncapped pi0_hat, AdaDetect Cor. 3.7)
N_REF = 2                  # of the 4 calibration images of each class (stored class-major order): positions 0, 1 -> reference R
N_CAL_PER_CLASS = 4        #                                                                           positions 2, 3 -> calibration C
VIEWS16 = ("L14", "D3L")


def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=1, default=str) + "\n")
