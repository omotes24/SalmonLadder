"""Compare a re-run in a materialized workspace with the archive.

  python tools/compare_rerun.py --workspace /path/to/workspace
  python tools/compare_rerun.py --workspace W --stored S --record provenance/rerun_check.json --note "what was run"

1. Metric tables of Phase 4 (results/eval/<stream>.parquet, one row per detector, base detector and weight): every
   table that the re-run wrote is compared with the archived table of the same stream (experiments/phase4/results/eval).
   Exit code 1 if AUROC or FPR95 of any row differs by more than --tolerance percentage points (a value that is
   missing in one of the two counts as a difference), if a table has other rows than the archived one, or if nothing
   could be compared.
2. With --stored: the score files of Phase 4 (results/eval/<stream>_scores.npz) and Phase 7 (results/<experiment>/
   <view>/<stream>.npz) are compared array by array with the files of the same name below the directory S, which has
   the layout of a workspace (on the experiment server: the directory that holds the original runs). Score files are
   not part of the archive, so this part is for whoever holds the originals. It is reported and never an error.

A re-run on a GPU is not bit-identical. provenance/rerun_check_20261004.json records what such a comparison gave on
the experiment server.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
KEY = ["family", "config", "role", "base", "a"]
P4, P7 = "reprise_p4_20260928", "reprise_p7_20261003"


def gap(new, old):
    """Largest absolute difference of two columns; infinite if a value is missing in one of them only."""
    new, old = new.to_numpy(float), old.to_numpy(float)
    both_missing = np.isnan(new) & np.isnan(old)
    difference = np.where(both_missing, 0.0, np.abs(new - old))
    return float("inf") if np.isnan(difference).any() else float(difference.max())


def metric_tables(workspace):
    """{stream: comparison} for the Phase 4 tables of the workspace that the archive also holds, and the largest difference."""
    out, worst = {}, 0.0
    for new_path in sorted((workspace / P4 / "results/eval").glob("*.parquet")):
        old_path = ROOT / "experiments/phase4/results/eval" / new_path.name
        if not old_path.is_file():
            continue
        new = pd.read_parquet(new_path).sort_values(KEY).reset_index(drop=True)
        old = pd.read_parquet(old_path).sort_values(KEY).reset_index(drop=True)
        if list(new.columns) != list(old.columns) or len(new) != len(old) or not (new[KEY].values == old[KEY].values).all():
            out[new_path.stem], worst = {"same_rows": False}, float("inf")
            continue
        diff = {m: gap(new[m], old[m]) for m in ("AUROC", "FPR95")}
        identical = int(((new[["AUROC", "FPR95"]].values == old[["AUROC", "FPR95"]].values).all(axis=1)).sum())
        out[new_path.stem] = {"same_rows": True, "rows": len(new), "rows_with_identical_metrics": identical,
                              "max_abs_diff": {m: (d if np.isfinite(d) else None) for m, d in diff.items()}}
        frozen = (new.family == "rep_L0") & (new.role == "frozen") & (new.base == "none")      # the frozen method, standalone
        if frozen.any():
            out[new_path.stem]["frozen_standalone"] = {"rerun": new.loc[frozen, ["AUROC", "FPR95"]].iloc[0].round(6).to_dict(),
                                                       "archived": old.loc[frozen, ["AUROC", "FPR95"]].iloc[0].round(6).to_dict()}
        worst = max(worst, *diff.values())
    return out, worst


def score_arrays(new_path, old_path):
    """Array by array: identical, or the number of differing entries and the largest absolute difference."""
    new, old = np.load(new_path, allow_pickle=False), np.load(old_path, allow_pickle=False)
    report = {"arrays": len(new.files), "identical": 0, "differing": {}, "not_compared": sorted(set(new.files) ^ set(old.files))}
    for name in sorted(set(new.files) & set(old.files)):
        x, y = new[name], old[name]
        if x.shape != y.shape or x.dtype.kind not in "fiub" or y.dtype.kind not in "fiub":
            if x.shape == y.shape and np.array_equal(x, y):
                report["identical"] += 1
            else:
                report["not_compared"].append(name)                     # text (run metadata) or another shape
            continue
        x, y = x.astype(np.float64), y.astype(np.float64)
        differs = ~((x == y) | (np.isnan(x) & np.isnan(y)))
        if not differs.any():
            report["identical"] += 1
        else:
            gap = np.abs(x[differs] - y[differs])
            report["differing"][name] = {"entries": int(differs.sum()), "of": int(x.size), "max_abs_diff": float(np.nanmax(gap)) if np.isfinite(gap).any() else None}
    return report


def score_files(workspace, stored):
    pairs = [(path, stored / path.relative_to(workspace)) for path in sorted((workspace / P4 / "results/eval").glob("*_scores.npz"))]
    pairs += [(path, stored / path.relative_to(workspace)) for path in sorted((workspace / P7 / "results").glob("*/*/*.npz"))]
    return {new.relative_to(workspace).as_posix(): score_arrays(new, old) for new, old in pairs if old.is_file()}


def versions():
    found = {"numpy": np.__version__, "pandas": pd.__version__}
    try:
        import torch
        found["torch"] = torch.__version__
        if torch.cuda.is_available():
            found["gpu"] = torch.cuda.get_device_name(0)
    except ImportError:
        pass
    return found


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--workspace", type=Path, required=True, help="materialized workspace in which the re-run was made")
    parser.add_argument("--stored", type=Path, help="directory with the layout of a workspace that holds original score files")
    parser.add_argument("--tolerance", type=float, default=0.05, help="largest accepted difference of AUROC or FPR95 in percentage points")
    parser.add_argument("--record", type=Path, help="write the comparison to this JSON file")
    parser.add_argument("--note", default="", help="what was re-run and what was reused; stored in the record")
    args = parser.parse_args()
    tables, worst = metric_tables(args.workspace.expanduser().resolve())
    record = {"checked_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "note": args.note, "versions": versions(),
              "tolerance_percentage_points": args.tolerance, "metric_tables_compared": len(tables),
              "largest_metric_difference": worst if np.isfinite(worst) else None,
              "within_tolerance": worst <= args.tolerance, "metric_tables": tables}
    if args.stored:
        record["score_files"] = score_files(args.workspace.expanduser().resolve(), args.stored.expanduser().resolve())
    compared = len(tables) + len(record.get("score_files", {}))
    if not compared:
        record["within_tolerance"] = False
        record["error"] = "nothing was compared: the workspace holds no re-run output that the archive or --stored also holds"
    text = json.dumps(record, indent=1) + "\n"
    if args.record:
        args.record.write_text(text, encoding="utf-8")
    print(text, end="")
    sys.exit(0 if compared and worst <= args.tolerance else 1)


if __name__ == "__main__":
    main()
