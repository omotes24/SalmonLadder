"""Salmon Ladder Phase 14 (paper v17): the cross-fitted ladder (self-labelled seeds selected in one fold, scores of the other
fold read as calibrated ranks -> the final read-out is itself a conformal p-value), dev-split selection of the seed rule
(Storey vs plain BH), the level q and the combination with the static rank; transductive baselines; backbone families;
other base detectors; the OpenOOD CIFAR-100 benchmark. Shared paths and helpers."""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "/home/omote/reprise_p13_20261011/code")
import p13common as _P13  # noqa: E402  (Phase 13 -> 12 -> 11/10 paths, Phase 5 code, vendored vins package)
from p13common import *  # noqa: E402,F401,F403
P = _P13                  # the star import brings an older phase's `P` along; the helpers below address Phase 13's namespace

P14 = Path("/home/omote/reprise_p14_20261011")
RESULTS14 = P14 / "results"
LOGS14 = P14 / "logs"
Q_PAPER = 0.2
RULES = ("storey", "bh")
QS14 = (0.05, 0.1, 0.2)
MODES = ("pp", "pps", "min")           # p+ p- ; p+ p- x static ; min(p+ p-, static)   (all read as calibrated ranks of the statistic)


def select_seeds(p, q, n_cal, rule):
    """Seeds of one fold: BH on the calibrated ranks p of the fold's test images, with the Storey-AdaDetect level (rule 'storey',
    Marandon et al. 2024 Cor. 3.7) or the plain BH level (rule 'bh', Cor. 3.5: FDR <= q m0/m). Returns (mask, pi0_hat)."""
    if rule == "storey":
        return P.storey_adadetect(p, q, n_cal)
    if rule == "bh":
        return P.plain_bh(p, q), 1.0
    raise ValueError(rule)


def fold_split(n_cal, n_test, folds, rng):
    """Random assignment of the calibration and the test nodes to `folds` folds (independent of everything else)."""
    fc = rng.permutation(np.arange(n_cal) % folds)
    ft = rng.permutation(np.arange(n_test) % folds)
    return fc, ft


DEFAULT_SEL = ("bh", 0.2, "pp")


def selected(default=DEFAULT_SEL):
    """The configuration chosen by the registered rule on the development splits (sel14.py --aggregate), as (rule, q, mode);
    the default when the selection has not been run (the smoke test)."""
    f = RESULTS14 / "sel" / "sel14_summary.json"
    if not f.exists():
        return tuple(default)
    name = json.loads(f.read_text())["_selection"]["chosen"]
    if not name:
        return tuple(default)
    _, rule, q, mode, _ = name.split("|")          # xfit|<rule>|q<q>|<mode>|none
    return rule, float(q[1:]), mode


def pairs_for(sel):
    """(rule, q) pairs run on the secondary benchmarks: the paper's two fixed references and the selected configuration."""
    return list(dict.fromkeys([("storey", Q_PAPER), ("bh", Q_PAPER), (sel[0], sel[1])]))


def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=1, default=str) + "\n")
