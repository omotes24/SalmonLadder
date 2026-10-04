"""Phase 5 final evaluation: paths of the unused bank U4 and the lock guard.

U4 may be built, featurised and scored only after selection_lock_p5.json exists (pre-registration). Every U4 script
calls require_lock() first."""
import json
from pathlib import Path

from common import sha_file  # noqa: F401  (also puts the vendored vins package on sys.path)

P5 = Path("/home/omote/reprise_p5_20261002")
P4 = Path("/home/omote/reprise_p4_20260928")
BANKS = P5 / "banks"
FEATS = P5 / "features_u4"
TINS_DIR = P5 / "tins_u4"
RESULTS = P5 / "results_u4"
LOCK = P5 / "selection_lock_p5.json"
SEED = 20261002
N_SPLITS, N_HELD = 5, 100
N_DRAWS, N_IDEVAL, N_OODEVAL, N_SPARE = 3, 20, 50, 12
N_SHOT = 16 * N_DRAWS


def require_lock():
    if not LOCK.exists():
        raise SystemExit("selection_lock_p5.json is missing: U4 must not be built or scored before the lock")
    want = (P5 / "selection_lock_p5.sha256").read_text().split()[0]
    if sha_file(LOCK) != want:
        raise SystemExit("selection lock changed after it was written")
    return json.loads(LOCK.read_text())
