"""The documentation states only what the archive contains.

 1. the archive verifies (hashes of every archived file, registration and quoted hash)
 2. docs/RESULTS.md, the results table of README.md and the timeline of docs/PREREGISTRATION.md follow from the archived files
 3. the outcomes quoted in the prose of docs/PREREGISTRATION.md and the cost quoted in docs/METHOD.md are recomputed from
    the result files they come from; every other quoted number appears in docs/RESULTS.md
 4. the times quoted in the prose of docs/PREREGISTRATION.md are times of the archived records
 5. the constants of docs/METHOD.md are those of the frozen configuration in the code
 6. relative links of the documentation point to existing files
 7. what docs/REPRODUCIBILITY.md says about the re-run on the server is what the record of that re-run holds
 8. what docs/PREREGISTRATION.md says about the typed times of test uses 3 and 4 follows from the timeline
"""
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
E = ROOT / "experiments"
PAIR = re.compile(r"\d{2,3}\.\d{2} / \d{1,3}\.\d{2}")                           # AUROC / FPR95
INTERVAL = re.compile(r"[-+]\d+\.\d{2} \[[-+]\d+\.\d{2}, [-+]\d+\.\d{2}\]")     # difference [lo, hi]
CLOCK = re.compile(r"(?<![\d:])\d{2}:\d{2}:\d{2}(?![\d:])")
LINK = re.compile(r"\]\(([^)#\s]+)(?:#[^)]*)?\)")


def read(rel):
    return (ROOT / rel).read_text(encoding="utf-8").replace("−", "-")


def load(rel):
    return json.loads((E / rel).read_text())


def run(*args):
    return subprocess.run([sys.executable, *args], cwd=ROOT, capture_output=True, text=True)


def interval(d):
    return f"{d['mean']:+.2f} [{d['lo']:+.2f}, {d['hi']:+.2f}]"


def test_archive_verifies():
    done = run("tools/verify_snapshot.py")
    assert done.returncode == 0, done.stdout[-2000:]
    report = json.loads(done.stdout)
    assert report["archived_files_checked"] > 1900 and report["quoted_hashes_checked"] >= 32 and report["final_test_code"]["as_registered"] == 12


def test_generated_documents_are_current():
    for tool in ("tools/make_results_tables.py", "tools/make_registration_timeline.py"):
        done = run(tool, "--check")
        assert done.returncode == 0, (tool, done.stdout, done.stderr[-2000:])


def test_quoted_outcomes_come_from_the_result_files():
    prose = read("docs/PREREGISTRATION.md")
    u1 = load("phase4/results/eval_summary.json")["banks"]["U1"]
    lock = load("phase7/results/an_lock_eval.json")["u1w"]
    far = load("phase5_6/results_final/summary_openood.json")["far_TINS"]["locked_minus_frozen_v5_FPR95"]
    u2 = load("phase7/results/an_ext_bases.json")["banks"]["U2"]["bases"]["none"]["ext-reprise"]["FPR95"]
    assert f"E1 {interval(u1['E1_FPR95'])}, E2 {interval(u1['E2_FPR95'])}" in prose
    assert u1["E1_FPR95"]["hi"] < 0 and u1["E2_FPR95"]["hi"] < 0                       # the registered decision rule
    assert f"({interval(lock['none|AUROC']['diff'])}, {interval(lock['none|FPR95']['diff'])})" in prose
    assert lock["W1"] and lock["W2"] and lock["none|AUROC"]["diff"]["lo"] > 0 and lock["none|FPR95"]["diff"]["hi"] < 0
    assert f"far-OOD FPR95 {far['mean']:+.2f} with TINS" in prose and far["lo"] > 0
    assert u2["lo"] < 0 < u2["hi"] and "On U2 the extension does not differ from the frozen method" in prose
    cost = load("phase4/results/resources.json")["REPRISE_frozen"]
    assert f"{cost['ms_per_image']:.2f} ms per image and {cost['alloc_MiB'] / 1024:.2f} GiB" in read("docs/METHOD.md")
    results = read("docs/RESULTS.md")                                                  # everything else that looks like a result
    for rel in ("README.md", "docs/PREREGISTRATION.md"):
        text = read(rel)
        missing = sorted({q for q in PAIR.findall(text) + INTERVAL.findall(text) if q not in results})
        assert not missing, (rel, missing)


def test_quoted_times_are_times_of_the_records():
    text = read("docs/PREREGISTRATION.md")
    table = text[text.index("<!-- BEGIN GENERATED: timeline -->"):text.index("<!-- END GENERATED: timeline -->")]
    prose = text.replace(table, "")
    record = json.loads((ROOT / "provenance/server_snapshot_20261004.json").read_text())
    known = {info["server_mtime_utc"][11:19] for info in record["files"].values()} | {t[11:19] for t in record["legacy_server_mtime_utc"].values()}
    known |= set(CLOCK.findall(table))
    known |= {load("lp_audit/execution_code_manifest.json")["created_utc"][11:19], load("lp_audit/provenance/current_source_manifest.json")["utc"][11:19]}
    quoted = set(CLOCK.findall(prose))
    assert len(quoted) >= 10 and quoted <= known, sorted(quoted - known)


def test_method_constants():
    from common import V5                                  # experiments/phase4/code/common.py
    assert V5 == {"n0": 48, "K": 20, "m": 1, "kg": 10, "gamma": 1.0, "lam": 0.9, "thresholds": (0.3, 0.2, 0.10191613435745239)}
    method = read("docs/METHOD.md")
    for stated in ("`n0 = 48`", "`K = 20`", "`m = 1`", "0.3, 0.2, 0.10191613435745239", "`k = 10`, `gamma = 1`, `lambda = 0.9`, 15 sweeps"):
        assert stated in method, stated


def test_relative_links_exist():
    documents = [ROOT / "README.md", ROOT / "THIRD_PARTY.md"] + sorted((ROOT / "docs").glob("*.md"))
    broken, checked = [], 0
    for document in documents:
        for target in LINK.findall(document.read_text(encoding="utf-8")):
            if "://" in target or target.startswith("mailto:"):
                continue
            checked += 1
            if not (document.parent / target).exists():
                broken.append((document.name, target))
    assert checked > 20 and not broken, broken


def test_rerun_statement_matches_the_record():
    record = json.loads((ROOT / "provenance/rerun_check_20261004.json").read_text())
    text = read("docs/REPRODUCIBILITY.md")
    tables = record["metric_tables"]
    assert record["within_tolerance"] and sorted(name[:2] for name in tables) == ["U1", "U2"] and "(one stream of U1, one of U2)" in text
    assert f"all {sum(t['rows'] for t in tables.values())} rows of the two metric tables" in text
    assert max(t["max_abs_diff"]["FPR95"] for t in tables.values()) == 0 and "FPR95 was identical" in text
    assert max(t["max_abs_diff"]["AUROC"] for t in tables.values()) < 0.00002 and "AUROC differed by less than 0.00002 percentage points" in text
    files = record["score_files"]
    phase7 = sorted(name.split("/")[2:4] for name in files if name.startswith("reprise_p7"))
    assert phase7 == [["u1w", "B14"], ["u1w", "L14"], ["u4w", "B14"], ["u4w", "L14"]] and "(one stream of U1 and of U4, both views)" in text
    differing = [v for f in files.values() for v in f["differing"].values()]
    counts = (sum(f["arrays"] for f in files.values()), sum(f["identical"] for f in files.values()), sum(len(f["not_compared"]) for f in files.values()))
    assert counts[0] == counts[1] + counts[2] + len(differing)
    assert f"Of {counts[0]} score arrays, {counts[1]} were identical" in text and f"{counts[2]} hold run metadata" in text and f"and {len(differing)} differed" in text
    few, every = [v for v in differing if v["entries"] <= 100], [v for v in differing if v["entries"] > 100]
    assert len(few) + len(every) == len(differing) and min(v["entries"] for v in every) > 10000        # either a few images or all of them
    assert f"at most {max(v['entries'] for v in few)} of 23,000 images" in text and max(v["of"] for v in few) == 23000
    assert f"log score {max(v['max_abs_diff'] for v in few):.4f})" in text
    assert max(v["max_abs_diff"] for v in every) < 3e-5 and "continuous scores by at most 3e-5" in text
    # the table without the rows of U3
    derived = json.loads((ROOT / "provenance/server_snapshot_20261004.json").read_text())["derived_tables"]["experiments/phase4/banks/feature_table.parquet"]
    assert f"{derived['rows'] - derived['wild_rows']:,} rows" in text and f"{derived['wild_rows']:,} further rows for U3" in text


def test_typed_times_statement_matches_the_timeline():
    text = read("docs/PREREGISTRATION.md")
    from datetime import datetime, timedelta

    def row(name):
        found = re.search(r"\| (\d{4}-\d\d-\d\d \d\d:\d\d:\d\d) \| `" + re.escape(name) + r"` \| ([^|]*)\|", text)
        typed = re.match(r"\d{4}-\d\d-\d\d \d\d:\d\d:\d\d", found.group(2).strip())
        return datetime.fromisoformat(found.group(1)), (datetime.fromisoformat(typed.group(0)) if typed else None)
    result3, result4 = row("r5_final/test_eval/results_m2/metrics.json")[0], row("r5_final/test_eval/results_m3/metrics.json")[0]
    file3, typed3 = row("legacy/test_eval/prereg_test3.json")
    file4, typed4 = row("legacy/test_eval/prereg_test4.json")
    frozen2 = row("r5_final/iter2/frozen_m2.json")[0]
    local = timedelta(hours=9)
    assert file3 < result3 and file4 < result4                                           # the registrations precede their results
    assert typed3 - result3 == timedelta(seconds=57) and "for use 3 by 57 s" in text and typed3 - local < frozen2
    assert timedelta(minutes=5) <= typed4 - local - result4 < timedelta(minutes=6) and "for use 4 by 5 min" in text
    assert timedelta(hours=9) <= typed4 - result4 < timedelta(hours=10) and "by 9 h as UTC" in text
    for analysis in ("extra/analysis_extra.json", "reprise/summary.json"):               # the two post hoc analyses that scored Four-OOD
        assert "fourood" in load("r5_final/analysis/" + analysis)
    assert "Four-OOD in two post hoc analyses" in text
