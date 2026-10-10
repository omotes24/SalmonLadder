"""Create a separate executable copy, preserving the archived experiment sources.

The workspace reproduces the directory layout of the experiment server:

  <workspace>/vins_gonogo_20260925        experiments/legacy, overlaid with experiments/r5_final
  <workspace>/reprise_controls_20260927   experiments/controls_456
  <workspace>/reprise_lp_audit_20260927   experiments/lp_audit
  <workspace>/reprise_p4_20260928         experiments/phase4
  <workspace>/reprise_p5_20261002         experiments/phase5_6
  <workspace>/reprise_p7_20261003         experiments/phase7
  <workspace>/reprise_p10_20261010        experiments/phase10_11/phase10
  <workspace>/reprise_p11_20261010        experiments/phase10_11/phase11

Server-specific paths in the executable copies are replaced by the workspace path; image tables are rebuilt from
manifests/. The archived outputs of the later phases (results, logs, failure records) are placed under
<workspace>/archived_results/<directory>/ instead of the executable directories, because several scripts skip a task
whose output file already exists: a re-run must compute its own outputs, which can then be compared with the archived
ones. Nothing inside the repository is modified, and no experiment is started.
"""
import argparse
import fnmatch
import hashlib
import gzip
import json
import re
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SERVER_HOME = "/home/omote"
PHASES = {"experiments/lp_audit": "reprise_lp_audit_20260927", "experiments/phase4": "reprise_p4_20260928",
          "experiments/phase5_6": "reprise_p5_20261002", "experiments/phase7": "reprise_p7_20261003",
          "experiments/phase10_11/phase10": "reprise_p10_20261010", "experiments/phase10_11/phase11": "reprise_p11_20261010"}
# archived outputs, relative to the server directory: kept out of the executable directories (see the module docstring)
OUTPUTS = {"reprise_lp_audit_20260927": ["reports", "failures", "unit_tests/*.log"],
           "reprise_p4_20260928": ["results", "logs", "failures"],
           "reprise_p5_20261002": ["results", "results_final", "logs", "failures", "confirm_dev2_attempt1.json"],
           "reprise_p7_20261003": ["results", "logs", "failures", "features"],
           "reprise_p10_20261010": ["results", "logs", "STATUS", "STATUS.run1"],
           "reprise_p11_20261010": ["results", "logs", "STATUS"],
           "vins_gonogo_20260925": ["r5/phase3/openood", "r5/phase3/fourood", "r5/phase3/cub/eval_*.json", "r5/phase3/streams", "r5/phase3/summary.*",
                                    "r5/phase3/locoop_metrics.json", "r5/round2/search.json", "r5/summary", "r5/audit", "r5/paper",
                                    "r5/entrance/tune_table.json", "r5/entrance/eval", "r5/entrance/inject", "test_eval/results*",
                                    "test_eval/*.log", "analysis", "iter*/dev2_looks.jsonl"]}
# decision records whose bytes are hash-checked: never rewritten
VERBATIM = ("prereg", "frozen", "sha256", "amendment", "selection_lock")
# manifest (under manifests/) -> parquet file of the workspace
TABLES = {"phase4_imagenet_pool.csv.gz": "reprise_p4_20260928/banks/imagenet_pool.parquet",
          "phase4_imagenet_o.csv.gz": "reprise_p4_20260928/banks/U2_imagenet_o.parquet",
          "phase4_exp4_aug_table.csv.gz": "reprise_p4_20260928/banks/exp4/aug_table.parquet",
          "phase5_imagenet_pool.csv.gz": "reprise_p5_20261002/banks/imagenet_pool.parquet",
          "phase7_cub_images.csv.gz": "reprise_p7_20261003/banks/cub/images.parquet",
          "phase7_cifar100_images.csv.gz": "reprise_p7_20261003/banks/cifar100/images.parquet",
          "phase7_inr_images.csv.gz": "reprise_p7_20261003/banks/inr/images.parquet",
          "phase7_insk_images.csv.gz": "reprise_p7_20261003/banks/insk/images.parquet",
          "phase7_places365_images.csv.gz": "reprise_p7_20261003/banks/places365/images.parquet"}


def sha_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def is_output(server_dir, rel):
    """True if `rel` (posix path below the server directory) is an archived output of that directory."""
    return any(fnmatch.fnmatchcase(rel, pattern) or rel.startswith(pattern.rstrip("/") + "/") for pattern in OUTPUTS.get(server_dir, []))


def copy_split(source, server_dir, target):
    """Copy an archived directory: inputs and code into <target>/<server_dir>, archived outputs into archived_results."""
    moved = 0
    for path in sorted(source.rglob("*")):
        if not path.is_file() or "__pycache__" in path.parts:
            continue
        rel = path.relative_to(source).as_posix()
        output = is_output(server_dir, rel)
        dest = (target / "archived_results" / server_dir / rel) if output else (target / server_dir / rel)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, dest)
        moved += output
    return moved


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
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
    # the same server directory at the second export: files that the first export does not contain or that changed
    archived = {"vins_gonogo_20260925": copy_split(ROOT / "experiments/r5_final", "vins_gonogo_20260925", target)}
    shutil.copytree(ROOT / "experiments/controls_456", controls)
    shutil.copytree(ROOT / "experiments/tins_reference", reference)
    shutil.copytree(ROOT / "third_party/tins", legacy / "tins")
    shutil.copytree(ROOT / "third_party/LoCoOp", legacy / "ext/LoCoOp")
    shutil.copytree(ROOT / "third_party/dinov2", target / ".cache/torch/hub/facebookresearch_dinov2_main")
    shutil.copytree(ROOT / "third_party/tins/data", controls / "src/vendor/tins/data", dirs_exist_ok=True)
    for source, name in PHASES.items():
        archived[name] = copy_split(ROOT / source, name, target)
        (target / name / "logs").mkdir(exist_ok=True)                    # the launchers redirect their output there
    changed = {}
    for path in target.rglob("*"):
        if not path.is_file() or path.suffix not in [".py", ".sh", ".json", ".jsonl", ".txt", ".yaml"]:
            continue
        if path.relative_to(target).parts[0] == "archived_results":      # records for comparison: left as archived
            continue
        # Hash-checked decision records remain verbatim. They are not live path configuration.
        if any(word in path.name.lower() for word in VERBATIM):
            continue
        old = path.read_text()
        new = old.replace(SERVER_HOME + "/granood_ke/.venv/bin/python", sys.executable)
        new = new.replace(SERVER_HOME + "/", str(target) + "/")
        if path.suffix == ".py":
            new = new.replace("Path.home()", "Path(" + repr(str(target)) + ")")
            new = new.replace('Path("' + SERVER_HOME + '")', "Path(" + repr(str(target)) + ")")
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
            frame["path"] = frame.path.str.replace(SERVER_HOME + "/", str(target) + "/", regex=False)
            frame.to_parquet(dest, index=False)
    manifest = gzip.decompress((ROOT / "manifests/reference_manifest.jsonl.gz").read_bytes())
    moved = manifest.decode().replace(SERVER_HOME + "/", str(target) + "/").encode()
    (reference.parent / "manifest.jsonl").write_bytes(moved)
    audit = json.loads(gzip.decompress((ROOT / "manifests/reference_data_audit.json.gz").read_bytes()))
    audit["original_manifest_sha256_before_path_relocation"] = audit["manifest_sha256"]
    audit["manifest_sha256"] = hashlib.sha256(moved).hexdigest()
    (reference.parent / "data_audit.json").write_text(json.dumps(audit, indent=2) + "\n")

    # Image tables of the later phases: same columns and row order as the server tables (see the provenance record).
    record = json.loads((ROOT / "provenance/server_snapshot_20261004.json").read_text())["tables"]
    frames = {}
    for name, rel in TABLES.items():
        source = ROOT / "manifests" / name
        columns = record[f"manifests/{name}"]["columns"]
        frame = pd.read_csv(source, dtype={c: (str if t == "str" else t) for c, t in columns.items()}, keep_default_na=False)
        assert list(frame.columns) == list(columns) and len(frame) == record[f"manifests/{name}"]["rows"], name
        if "path" in frame:
            frame["path"] = frame.path.str.replace("<DATA_HOME>", str(target), regex=False)
        dest = target / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(dest, index=False)
        frames[name] = frame
        manifests[name] = {"rows": len(frame), "source_sha256": sha_file(source), "materialized_parquet": rel,
                           "materialized_parquet_sha256": sha_file(dest),
                           "server_parquet_sha256": record[f"manifests/{name}"]["server_parquet_sha256"]}
    # Feature-row tables (the row order of the feature arrays). The server table of Phase 4 ended with the rows of a
    # private wildlife dataset (U3); those rows are not archived, so U3 cannot be rebuilt.
    pool4, o4, pool5 = frames["phase4_imagenet_pool.csv.gz"], frames["phase4_imagenet_o.csv.gz"], frames["phase5_imagenet_pool.csv.gz"]
    table4 = pd.concat([pool4[["sample_id", "path"]].assign(source="imagenet"), o4[["sample_id", "path"]].assign(source="imagenet_o")],
                       ignore_index=True)
    table4.to_parquet(target / "reprise_p4_20260928/banks/feature_table.parquet", index=False)
    pool5[["sample_id", "path"]].to_parquet(target / "reprise_p5_20261002/banks/feature_table.parquet", index=False)
    derived = {"reprise_p4_20260928/banks/feature_table.parquet": {"rows": len(table4), "without": "the U3 rows of the server table"},
               "reprise_p5_20261002/banks/feature_table.parquet": {"rows": len(pool5)}}

    record = {"workspace": str(target), "path_tokens": mappings, "transformed_files": changed, "manifests": manifests,
              "derived_tables": derived, "archived_output_files": archived, "original_manifest_hashes_preserved": True,
              "automatic_experiment_start": False}
    (target / "materialization.json").write_text(json.dumps(record, indent=2) + "\n")
    print(f"Materialized {target}. Put data/checkpoints in place and build feature banks before running experiments.")


if __name__ == "__main__":
    main()
