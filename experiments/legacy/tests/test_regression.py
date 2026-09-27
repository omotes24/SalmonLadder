"""Hook OFF reproduces pristine upstream TINS scores; hook ON changes nothing (runs vins.regression)."""
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from vins import config as C

pytestmark = [pytest.mark.hades, pytest.mark.gpu]
ROOT = Path(__file__).resolve().parents[1]


def run_mode(mode):
    env = dict(os.environ, PYTHONPATH=str(ROOT))
    subprocess.run([sys.executable, "-m", "vins.regression", mode], cwd=ROOT, env=env, check=True)
    return json.loads((C.RUNS_DIR / "regression" / f"result_{mode}.json").read_text())


def _check_common(cfg):
    assert cfg["recorder_final_matches"] and cfg["recorder_seeded_matches"]
    assert cfg["n_seeded"] > 0 and cfg["n_batches_updated"] > 0     # inversion + bank path exercised
    if cfg["name"] == "small_bank":
        assert cfg["n_flash"] > 0                                   # overflow + Flash exercised


def test_deterministic_mode_bitwise():
    result = run_mode("deterministic")
    for cfg in result["configs"]:
        _check_common(cfg)
        assert cfg["pristine_repeatable"]
        assert cfg["off_equals_pristine"] and cfg["on_equals_off"]


def test_default_mode():
    result = run_mode("default")
    for cfg in result["configs"]:
        _check_common(cfg)
        if cfg["pristine_repeatable"]:          # bitwise when upstream itself is repeatable
            assert cfg["off_equals_pristine"] and cfg["on_equals_off"]
