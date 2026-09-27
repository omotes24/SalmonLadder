"""Verify archived experiment sources without importing scientific dependencies."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    checked = 0
    errors = []
    for filename in ["server_snapshot.json", "extra_snapshot.json", "dinov2_snapshot.json", "reference_inputs.json"]:
        record = json.loads((ROOT / "provenance" / filename).read_text())
        entries = record.get("files", record)
        for rel, info in entries.items():
            path = ROOT / rel
            if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != info["sha256"]:
                errors.append(rel)
            checked += 1
    base = ROOT / "experiments/controls_456"
    for rel, digest in json.loads((base / "SOURCE_SHA256.json").read_text()).items():
        if hashlib.sha256((base / rel).read_bytes()).hexdigest() != digest:
            errors.append("controls frozen manifest: " + rel)
    legacy = ROOT / "experiments/legacy"
    frozen = json.loads((legacy / "iter3/frozen_m3.json").read_text())
    if hashlib.sha256((legacy / "scripts/iter3_eval.py").read_bytes()).hexdigest() != frozen["code"]["sha256"]:
        errors.append("legacy frozen iter3_eval.py")
    print(json.dumps({"archived_files_checked": checked, "mismatches": errors}, indent=2))
    raise SystemExit(bool(errors))


if __name__ == "__main__":
    main()
