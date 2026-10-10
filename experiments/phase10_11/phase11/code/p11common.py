"""Salmon Ladder Phase 11: transductive (whole-stream) read-outs on the public benchmarks. Paths and shared helpers.
Everything new lives under P11; features and streaming results are read from Phase 3/5/10."""
import sys
from pathlib import Path

import p10common as P10  # noqa: F401  (puts Phase 5/7 code and the vendored vins package on sys.path)
from p10common import *  # noqa: F401,F403

P11 = Path("/home/omote/reprise_p11_20261010")
RESULTS11 = P11 / "results"
LOGS11 = P11 / "logs"
REFT = "k10g1b1l0.9"


def stream_result_file(part, name, view):
    """The streaming (online) result file holding s::<view>::* and a::<view>::M0 of the stream."""
    if view in ("B14", "L14"):
        return P5 / "results_final" / part / f"{name}.npz"
    return RESULTS / part / "D3B+D3L" / f"{name}.npz"
