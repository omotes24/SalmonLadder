"""Verify the archive without importing scientific dependencies.

Errors (exit code 1):
  1. an archived file differs from the sha256 (or the size) recorded when it was exported from the experiment server,
     or a file under experiments/, manifests/ or third_party/ is listed in no record;
  2. an image table under manifests/ differs from its record;
  3. the controls' frozen source manifest or the frozen hash of the original implementation does not match;
  4. a registration, amendment or selection lock differs from the hash file written next to it;
  5. a hash that one decision record quotes from another (the list QUOTED below) does not match the quoted file;
  6. a script of the final test differs from the hash in its registration and no documented deviation explains it;
  7. the engine of Phase 7 differs from the hash in its selection lock, or a source of the propagation audit differs
     from the audit's final source manifest.

Reported without being errors, because the records themselves say so (see docs/PREREGISTRATION.md): scripts changed
after the Phase 7 lock, and sources of the propagation audit that changed after its registration and amendments.

In an anonymized export the records hold the hashes and sizes of the exported files, so the same checks apply there;
provenance/anonymized_export.json lists the files that differ from the originals.
"""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SNAPSHOTS = ["server_snapshot.json", "extra_snapshot.json", "dinov2_snapshot.json", "reference_inputs.json", "server_snapshot_20261004.json",
             "server_snapshot_20261011.json", "third_party_licenses.json"]
ARCHIVE_DIRECTORIES = ["experiments", "manifests", "third_party"]
E = "experiments/"
# (hash file, hashed file); a hash file holds the hex digest, optionally followed by the file name
HASH_FILES = [
    (E + "lp_audit/preregistration.sha256", E + "lp_audit/preregistration.json"),
    (E + "r5_final/r5/phase3/prereg_phase3.sha256", E + "r5_final/r5/phase3/prereg_phase3.json"),
    (E + "phase4/prereg_p4.sha256", E + "phase4/prereg_p4.json"),
    (E + "phase4/selection_lock.sha256", E + "phase4/selection_lock.json"),
    (E + "phase5_6/code/prereg_p5.sha256", E + "phase5_6/code/prereg_p5.json"),
    (E + "phase5_6/selection_lock_p5.sha256", E + "phase5_6/selection_lock_p5.json"),
    (E + "phase5_6/prereg_p6.sha256", E + "phase5_6/prereg_p6.json"),
    (E + "phase7/prereg_p7.sha256", E + "phase7/prereg_p7.json"),
    (E + "phase7/amendment_01.sha256", E + "phase7/amendment_01.json"),
    (E + "phase7/amendment_02.sha256", E + "phase7/amendment_02.json"),
    (E + "phase7/amendment_03.sha256", E + "phase7/amendment_03.json"),
    (E + "phase7/selection_lock_p7.sha256", E + "phase7/selection_lock_p7.json"),
]
# (record, key path inside the record, quoted file, number of hex digits quoted). For a .jsonl record every line is
# checked. The value is the digest itself or a sentence that contains it (or its first digits).
QUOTED = [
    # the four evaluations on the test split during development: result -> registration -> frozen configuration, rule, code
    (E + "r5_final/test_eval/results/metrics.json", "prereg_sha256", E + "legacy/test_eval/prereg_test.json", 64),
    (E + "r5_final/test_eval/results_clavism/metrics.json", "prereg_sha256", E + "legacy/test_eval/prereg_test2.json", 64),
    (E + "r5_final/test_eval/results_m2/metrics.json", "prereg_sha256", E + "legacy/test_eval/prereg_test3.json", 64),
    (E + "r5_final/test_eval/results_m3/metrics.json", "prereg_sha256", E + "legacy/test_eval/prereg_test4.json", 64),
    (E + "legacy/test_eval/prereg_test2.json", "frozen_candidate_sha256", E + "r5_final/iter/frozen_c1.json", 64),
    (E + "legacy/test_eval/prereg_test2.json", "stopping_rule_sha256", E + "r5_final/iter/prereg_iter.json", 64),
    (E + "legacy/test_eval/prereg_test3.json", "frozen_candidate.sha256", E + "r5_final/iter2/frozen_m2.json", 64),
    (E + "legacy/test_eval/prereg_test3.json", "iteration_prereg.sha256", E + "r5_final/iter2/prereg_iter2.json", 64),
    (E + "legacy/test_eval/prereg_test4.json", "frozen_candidate.sha256", E + "legacy/iter3/frozen_m3.json", 64),
    (E + "legacy/test_eval/prereg_test4.json", "code.sha256", E + "legacy/scripts/iter3_eval.py", 64),
    (E + "legacy/test_eval/prereg_test4.json", "iteration_prereg.sha256", E + "legacy/iter3/prereg_iter3.json", 64),
    (E + "r5_final/iter2/frozen_m2.json", "prereg", E + "r5_final/iter2/prereg_iter2.json", 64),
    (E + "legacy/iter3/frozen_m3.json", "prereg", E + "legacy/iter3/prereg_iter3.json", 64),
    # the look at the second development split that preceded each of the evaluations 2-4
    (E + "r5_final/iter/dev2_looks.jsonl", "frozen_sha256", E + "r5_final/iter/frozen_c1.json", 64),
    (E + "r5_final/iter/dev2_looks.jsonl", "prereg_sha256", E + "r5_final/iter/prereg_iter.json", 64),
    (E + "r5_final/iter2/dev2_looks.jsonl", "frozen_sha256", E + "r5_final/iter2/frozen_m2.json", 64),
    (E + "r5_final/iter2/dev2_looks.jsonl", "prereg_sha256", E + "r5_final/iter2/prereg_iter2.json", 64),
    (E + "r5_final/iter3/dev2_looks.jsonl", "frozen_sha256", E + "legacy/iter3/frozen_m3.json", 64),
    (E + "r5_final/iter3/dev2_looks.jsonl", "prereg_sha256", E + "legacy/iter3/prereg_iter3.json", 64),
    # the matched comparison and the final test
    (E + "legacy/r5/prereg_r5.json", "frozen_method.code_sha256", E + "legacy/scripts/iter3_eval.py", 64),
    (E + "r5_final/r5/phase3/prereg_phase3.json", "provenance.prereg_phase2_round2_sha256", E + "r5_final/r5/prereg_phase2_round2.json", 64),
    (E + "r5_final/r5/phase3/prereg_phase3.json", "provenance.round2_decision_sha256", E + "r5_final/r5/round2/decision.json", 64),
    # later phases: lock -> registration, registration and amendments -> what they build on
    (E + "phase4/selection_lock.json", "prereg_sha256", E + "phase4/prereg_p4.json", 64),
    (E + "phase5_6/selection_lock_p5.json", "prereg_sha256", E + "phase5_6/code/prereg_p5.json", 64),
    (E + "phase5_6/banks/build_info.json", "lock_sha256", E + "phase5_6/selection_lock_p5.json", 64),
    (E + "phase5_6/prereg_p6.json", "unchanged", E + "phase5_6/selection_lock_p5.json", 64),
    (E + "phase7/amendment_01.json", "to", E + "phase7/prereg_p7.json", 64),
    (E + "phase7/amendment_02.json", "to", E + "phase7/prereg_p7.json", 8),
    (E + "phase7/amendment_02.json", "to", E + "phase7/amendment_01.json", 8),
    (E + "phase7/amendment_03.json", "to", E + "phase7/prereg_p7.json", 8),
    (E + "phase7/amendment_03.json", "to", E + "phase7/amendment_01.json", 8),
    (E + "phase7/amendment_03.json", "to", E + "phase7/amendment_02.json", 8),
]
# (record, key that holds {source file: sha256}): sources of the propagation audit at four moments
AUDIT_CODE_RECORDS = [
    ("preregistration.json", "source_sha256"),
    ("amendment_01.json", "code_sha256"),
    ("execution_code_manifest.json", "files"),
    ("provenance/amendment_03.json", "code_sha256"),
    ("provenance/reporting_amendment.json", "sha256"),
]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def lookup(obj, key_path):
    for key in key_path.split("."):
        obj = obj[key]
    return obj


def main():
    errors = []

    def matches(rel, want, size=None):
        path = ROOT / rel
        return path.is_file() and sha(path) == want and size in (None, path.stat().st_size)

    checked = tables = 0
    listed = set()
    for filename in SNAPSHOTS:
        record = json.loads((ROOT / "provenance" / filename).read_text())
        for rel, info in record.get("files", record).items():
            if not matches(rel, info["sha256"], info.get("bytes")):
                errors.append(rel)
            listed.add(rel)
            checked += 1
        for rel, info in record.get("tables", {}).items():
            if not matches(rel, info["sha256"], info.get("bytes")):
                errors.append("manifest table: " + rel)
            listed.add(rel)
            tables += 1
    for directory in ARCHIVE_DIRECTORIES:
        for path in sorted((ROOT / directory).rglob("*")):
            rel = path.relative_to(ROOT).as_posix()
            if path.is_file() and rel not in listed and "__pycache__" not in path.parts and ".pytest_cache" not in path.parts:
                errors.append("listed in no provenance record: " + rel)
    base = "experiments/controls_456"
    for rel, digest in json.loads((ROOT / base / "SOURCE_SHA256.json").read_text()).items():
        if not matches(f"{base}/{rel}", digest):
            errors.append("controls frozen manifest: " + rel)
    legacy = "experiments/legacy"
    frozen = json.loads((ROOT / legacy / "iter3/frozen_m3.json").read_text())
    if not matches(f"{legacy}/scripts/iter3_eval.py", frozen["code"]["sha256"]):
        errors.append("legacy frozen iter3_eval.py")
    for hash_file, target in HASH_FILES:
        if not matches(target, (ROOT / hash_file).read_text().split()[0]):
            errors.append("registration hash: " + target)
    for record, key_path, target, digits in QUOTED:
        text = (ROOT / record).read_text()
        entries = [json.loads(line) for line in text.splitlines() if line.strip()] if record.endswith(".jsonl") else [json.loads(text)]
        if not entries:
            errors.append(f"{record}: empty record")
        for entry in entries:
            if sha(ROOT / target)[:digits] not in lookup(entry, key_path):
                errors.append(f"{record}: {key_path} does not quote the hash of {target}")
    # the scripts that read the test data: hashes written into the registration before the final test was run
    phase3 = ROOT / "experiments/r5_final/r5/phase3"
    deviations = {d["file"]: d for d in json.loads((phase3 / "deviations.json").read_text())}
    final_test = {"as_registered": 0, "documented_deviations": []}
    for name, digest in json.loads((phase3 / "prereg_phase3.json").read_text())["code"].items():
        rel = f"experiments/r5_final/{name}" if (ROOT / "experiments/r5_final" / name).is_file() else f"experiments/legacy/{name}"
        if matches(rel, digest):
            final_test["as_registered"] += 1
        elif name in deviations and deviations[name]["old_sha256"] == digest and matches(rel, deviations[name]["new_sha256"]):
            final_test["documented_deviations"].append(name)
        else:
            errors.append("final-test code differs from the registration: " + name)
    if not final_test["as_registered"]:
        errors.append("final test: the registration lists no script")
    lock7 = json.loads((ROOT / "experiments/phase7/selection_lock_p7.json").read_text())
    lock_code = {"unchanged_since_lock": [], "changed_after_lock": []}
    for name, digest in lock7["code_sha256"].items():
        lock_code["unchanged_since_lock" if matches(f"experiments/phase7/code/{name}", digest) else "changed_after_lock"].append(name)
    if "engine7.py" not in lock_code["unchanged_since_lock"]:
        errors.append("phase7 engine7.py differs from the hash recorded in the selection lock")
    # the propagation audit: its final source manifest must match; earlier records are compared for information
    audit = "experiments/lp_audit"
    audit_sources = {"final_manifest": {"matching": 0, "not_archived": []}, "earlier_records": {}}
    for name, digest in json.loads((ROOT / audit / "provenance/current_source_manifest.json").read_text())["files"].items():
        if not (ROOT / audit / name).is_file():
            audit_sources["final_manifest"]["not_archived"].append(name)
        elif matches(f"{audit}/{name}", digest):
            audit_sources["final_manifest"]["matching"] += 1
        else:
            errors.append("lp_audit source differs from its final manifest: " + name)
    if not audit_sources["final_manifest"]["matching"]:
        errors.append("lp_audit: the final source manifest lists no archived file")
    for record, key in AUDIT_CODE_RECORDS:
        sources = json.loads((ROOT / audit / record).read_text())[key]
        changed = sorted(name for name, digest in sources.items() if not matches(f"{audit}/{name}", digest))
        audit_sources["earlier_records"][record] = {"sources": len(sources), "changed_later": changed}
    print(json.dumps({"archived_files_checked": checked, "manifest_tables_checked": tables, "hash_files_checked": len(HASH_FILES),
                      "quoted_hashes_checked": len(QUOTED), "final_test_code": final_test, "phase7_lock_code_hashes": lock_code,
                      "lp_audit_sources": audit_sources, "anonymized_export": (ROOT / "provenance/anonymized_export.json").is_file(),
                      "mismatches": errors}, indent=2))
    raise SystemExit(bool(errors))


if __name__ == "__main__":
    main()
