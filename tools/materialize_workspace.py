"""Create a separate executable copy, preserving the archived experiment sources."""
import argparse
import hashlib
import gzip
import json
import re
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, required=True)
    args = parser.parse_args()
    target = args.workspace.expanduser().resolve()
    if not re.fullmatch(r"[A-Za-z0-9_./-]+", str(target)):
        parser.error("Use a workspace path without spaces or shell metacharacters for the historical launchers.")
    if target.exists() and any(target.iterdir()):
        parser.error("Workspace must be absent or empty; existing experiments are never overwritten.")
    import pandas as pd
    target.mkdir(parents=True, exist_ok=True)
    legacy = target / "vins_gonogo_20260925"
    controls = target / "reprise_controls_20260927"
    reference = target / "ood_large_best_20260923/tins_20260925"
    shutil.copytree(ROOT / "experiments/legacy", legacy)
    shutil.copytree(ROOT / "experiments/controls_456", controls)
    shutil.copytree(ROOT / "experiments/tins_reference", reference)
    shutil.copytree(ROOT / "third_party/tins", legacy / "tins")
    shutil.copytree(ROOT / "third_party/LoCoOp", legacy / "ext/LoCoOp")
    shutil.copytree(ROOT / "third_party/dinov2", target / ".cache/torch/hub/facebookresearch_dinov2_main")
    shutil.copytree(ROOT / "third_party/tins/data", controls / "src/vendor/tins/data", dirs_exist_ok=True)
    changed = {}
    for path in target.rglob("*"):
        if not path.is_file() or path.suffix not in [".py", ".sh", ".json", ".jsonl", ".txt"]:
            continue
        # Hash-checked decision records remain verbatim. They are not live path configuration.
        if "prereg" in path.name or "frozen" in path.name or "sha256" in path.name.lower():
            continue
        old = path.read_text()
        new = old.replace("/home/omote/granood_ke/.venv/bin/python", sys.executable)
        new = new.replace("/home/omote/", str(target) + "/")
        if path.suffix == ".py":
            new = new.replace("Path.home()", "Path(" + repr(str(target)) + ")")
        if path.suffix == ".sh":
            new = new.replace("${HOME}", str(target)).replace("$HOME", str(target))
        if new != old:
            path.write_text(new)
            changed[str(path.relative_to(target))] = {
                "original_sha256": hashlib.sha256(old.encode()).hexdigest(),
                "materialized_sha256": hashlib.sha256(new.encode()).hexdigest(),
            }
    mappings = {"<CONTROLS_ROOT>": str(controls), "<LEGACY_ROOT>": str(legacy), "<DATA_HOME>": str(target)}
    manifests = {}
    for name in ["cub", "cifar", "openood", "dev"]:
        source = ROOT / "manifests" / f"{name}_images.csv.gz"
        frame = pd.read_csv(source, dtype={"sample_id": str, "class": str})
        for token, value in mappings.items():
            frame["path"] = frame.path.str.replace(token, value, regex=False)
        dest = controls / "data" / name / "images.parquet"
        frame.to_parquet(dest, index=False)
        manifests[name] = {"rows": len(frame), "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                           "materialized_parquet_sha256": hashlib.sha256(dest.read_bytes()).hexdigest()}
    for name, dest in [("splits", legacy / "splits"), ("dev2_splits", legacy / "dev2/splits")]:
        frame = pd.read_csv(ROOT / "manifests" / f"{name}_samples.csv.gz")
        for token, value in mappings.items():
            frame["path"] = frame.path.str.replace(token, value, regex=False)
        frame.to_parquet(dest / "samples.parquet", index=False)
    for dest in (legacy / "runs/main").glob("stream_*.parquet"):
        frame = pd.read_parquet(dest)
        if "path" in frame:
            frame["path"] = frame.path.str.replace("/home/omote/", str(target) + "/", regex=False)
            frame.to_parquet(dest, index=False)
    manifest = gzip.decompress((ROOT / "manifests/reference_manifest.jsonl.gz").read_bytes())
    moved = manifest.decode().replace("/home/omote/", str(target) + "/").encode()
    (reference.parent / "manifest.jsonl").write_bytes(moved)
    audit = json.loads(gzip.decompress((ROOT / "manifests/reference_data_audit.json.gz").read_bytes()))
    audit["original_manifest_sha256_before_path_relocation"] = audit["manifest_sha256"]
    audit["manifest_sha256"] = hashlib.sha256(moved).hexdigest()
    (reference.parent / "data_audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    record = {"workspace": str(target), "path_tokens": mappings, "transformed_files": changed, "manifests": manifests,
              "original_manifest_hashes_preserved": True, "automatic_experiment_start": False}
    (target / "materialization.json").write_text(json.dumps(record, indent=2) + "\n")
    print(f"Materialized {target}. Put data/checkpoints in place and build feature banks before running experiments.")


if __name__ == "__main__":
    main()
