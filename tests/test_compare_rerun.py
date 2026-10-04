"""tools/compare_rerun.py: a re-run that equals the archive passes, a changed metric fails, score files are compared array by array."""
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
ARCHIVED = ROOT / "experiments/phase4/results/eval"
STREAMS = ("U1_s1_d0_seed123", "U2_s0_d0_seed123")


def compare(*args):
    done = subprocess.run([sys.executable, "tools/compare_rerun.py", *map(str, args)], cwd=ROOT, capture_output=True, text=True)
    return done.returncode, json.loads(done.stdout)


def test_rerun_is_compared_with_the_archive(tmp_path):
    workspace, stored = tmp_path / "workspace", tmp_path / "stored"
    target = workspace / "reprise_p4_20260928/results/eval"
    target.mkdir(parents=True)
    for stream in STREAMS:
        shutil.copy(ARCHIVED / f"{stream}.parquet", target)
    code, report = compare("--workspace", workspace)
    assert code == 0 and report["metric_tables_compared"] == 2 and report["largest_metric_difference"] == 0 and report["within_tolerance"]
    assert report["metric_tables"][STREAMS[0]]["rows_with_identical_metrics"] == report["metric_tables"][STREAMS[0]]["rows"] == 63

    table = pd.read_parquet(target / f"{STREAMS[0]}.parquet")
    table.loc[0, "FPR95"] += 1.0                                         # one detector, one percentage point
    table.to_parquet(target / f"{STREAMS[0]}.parquet", index=False)
    code, report = compare("--workspace", workspace)
    assert code == 1 and abs(report["largest_metric_difference"] - 1.0) < 1e-9 and not report["within_tolerance"]
    assert compare("--workspace", workspace, "--tolerance", 2)[0] == 0
    table.iloc[1:].to_parquet(target / f"{STREAMS[0]}.parquet", index=False)   # a row is missing
    code, report = compare("--workspace", workspace)
    assert code == 1 and report["metric_tables"][STREAMS[0]] == {"same_rows": False} and report["largest_metric_difference"] is None

    table = pd.read_parquet(ARCHIVED / f"{STREAMS[0]}.parquet")
    table.loc[0, ["AUROC", "FPR95"]] = float("nan")                      # a metric that the re-run could not compute
    table.to_parquet(target / f"{STREAMS[0]}.parquet", index=False)
    code, report = compare("--workspace", workspace)
    assert code == 1 and report["metric_tables"][STREAMS[0]]["max_abs_diff"] == {"AUROC": None, "FPR95": None} and not report["within_tolerance"]
    code, report = compare("--workspace", tmp_path / "no_such_workspace")
    assert code == 1 and report["metric_tables_compared"] == 0 and not report["within_tolerance"] and "nothing was compared" in report["error"]

    shutil.copy(ARCHIVED / f"{STREAMS[0]}.parquet", target)
    scores = np.linspace(0, 1, 50)
    for base, bump in ((workspace, 0.0), (stored, 1e-3)):
        directory = base / "reprise_p7_20261003/results/u1w/B14"
        directory.mkdir(parents=True)
        changed = scores.copy()
        changed[:3] += bump
        np.savez(directory / "U1_s1_d0_seed123.npz", same=scores, ranks=changed, flag=np.arange(50) % 2 == 0, meta=np.array("run at " + str(bump)))
    code, report = compare("--workspace", workspace, "--stored", stored)
    files = report["score_files"]
    assert code == 0 and list(files) == ["reprise_p7_20261003/results/u1w/B14/U1_s1_d0_seed123.npz"]
    found = files["reprise_p7_20261003/results/u1w/B14/U1_s1_d0_seed123.npz"]
    assert found["arrays"] == 4 and found["identical"] == 2 and found["not_compared"] == ["meta"]
    assert found["differing"]["ranks"]["entries"] == 3 and abs(found["differing"]["ranks"]["max_abs_diff"] - 1e-3) < 1e-12
