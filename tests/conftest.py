"""CPU tests of the archived engines on synthetic streams.

These tests were written together with the Phase 5 and Phase 7 code and run on a workstation during development; they
are not part of the server snapshot under experiments/. They import the archived modules directly (no server path is
needed) and never produce a number that is reported in the paper.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# the same import order as on the server: the vendored `vins` package, then the Phase 4 / Phase 5 / Phase 7 code
for rel in ("experiments/phase7/code", "experiments/phase5_6/code", "experiments/phase4/code", "experiments/lp_audit/vendor"):
    path = str(ROOT / rel)
    if path not in sys.path:
        sys.path.insert(0, path)
sys.path.insert(0, str(ROOT / "examples"))                 # synth5: the synthetic stream generator
