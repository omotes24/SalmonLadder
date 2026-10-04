"""Run every check that needs neither data, model weights nor a GPU, and optionally record the outcome.

  python tools/run_cpu_checks.py                                   # run, exit 1 on the first failure
  python tools/run_cpu_checks.py --record provenance/checks.json   # also write the commands, return codes and versions

The commands are the ones listed in the README; the continuous-integration workflow calls this script.
"""
import argparse
import importlib
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# (command, working directory relative to the repository, extra environment)
CHECKS = [
    ("python tools/verify_snapshot.py", ".", {}),
    ("python tools/make_results_tables.py --check", ".", {}),
    ("python tools/make_registration_timeline.py --check", ".", {}),
    ("python -m pytest -q tests", ".", {}),
    ("python -m pytest -q experiments/controls_456/src/test_controls.py", ".", {}),
    ("python -m pytest -q experiments/legacy/tests/test_r5.py", ".", {"PYTHONPATH": "experiments/legacy"}),
    ("python -m unittest discover -s unit_tests", "experiments/lp_audit", {}),
    ("python examples/feature_stream.py", ".", {}),
    ("python examples/salmon_ladder_stream.py", ".", {}),
]
PACKAGES = ["torch", "numpy", "scipy", "pandas", "pyarrow", "sklearn", "pytest"]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--record", type=Path, help="write the outcome to this JSON file")
    args = parser.parse_args()
    base = {**os.environ, "CUDA_VISIBLE_DEVICES": "", "PYTHONDONTWRITEBYTECODE": "1"}
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        base.setdefault(name, "2")
    results = []
    for command, directory, extra in CHECKS:
        words = [sys.executable] + command.split()[1:]
        if "pytest" in command:
            words[3:3] = ["-p", "no:cacheprovider"]
        start = time.time()
        done = subprocess.run(words, cwd=ROOT / directory, env={**base, **extra}, capture_output=True, text=True)
        tail = [line for line in (done.stdout if done.stdout.strip() else done.stderr).strip().splitlines() if line.strip()][-2:]
        results.append({"command": command, "directory": directory, "returncode": done.returncode, "seconds": round(time.time() - start, 1), "last_output": tail})
        print(f"{'ok  ' if done.returncode == 0 else 'FAIL'} {time.time() - start:6.1f}s  {command}")
        if done.returncode:
            print(done.stdout[-4000:] + done.stderr[-4000:])
            break
    passed = len(results) == len(CHECKS) and all(r["returncode"] == 0 for r in results)
    if args.record:
        versions = {name: getattr(importlib.import_module(name), "__version__", "?") for name in PACKAGES}
        record = {"checked_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "python": sys.version.split()[0], "platform": platform.platform(terse=True),
                  "packages": versions, "GPU_benchmark_rerun": False, "checks": results, "all_passed": passed}
        args.record.write_text(json.dumps(record, indent=1) + "\n", encoding="utf-8")
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
