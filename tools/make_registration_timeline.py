"""Write the generated timeline in docs/PREREGISTRATION.md from the archived decision records.

  python tools/make_registration_timeline.py            # rewrite the block between the GENERATED markers
  python tools/make_registration_timeline.py --check    # exit 1 if the block differs from what the records give

Each row is one archived file, the earliest of a group of result files, or one line of a status log. The first column
is the modification time of the file on the experiment server (UTC), read from the provenance records (identical copies
of a record count with their earliest time), or the time written in the log line. The second time is the one written
inside the record. Rows are sorted by the first column.
"""
import argparse
import datetime
import fnmatch
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOC = ROOT / "docs/PREREGISTRATION.md"
BEGIN, END = "<!-- BEGIN GENERATED: timeline -->", "<!-- END GENERATED: timeline -->"
TIME_KEYS = ("frozen_utc", "frozen_at_utc", "registered_utc", "written_utc", "created_utc", "locked_utc", "time_utc", "utc")
E = "experiments/"
# ("record" | "first" | "log", path or pattern below experiments/ [log: path and the text that identifies the line],
#  what the file fixes or shows)
ROWS = [
    ("record", "legacy/criteria.json", "go / no-go criterion of the first project stage"),
    ("record", "r5_final/dev2/prereg_confirm.json", "confirmation on a second development split, then one evaluation on the test split"),
    ("record", "legacy/test_eval/prereg_test.json", "**test use 1**: registration"),
    ("first", "r5_final/test_eval/results/metrics.json", "test use 1: result"),
    ("record", "r5_final/iter/prereg_iter.json", "improvement loop 1: stopping rule"),
    ("record", "r5_final/iter/frozen_c1.json", "loop 1: frozen configuration"),
    ("record", "r5_final/iter/dev2_looks.jsonl", "loop 1: look at the second development split (quotes both hashes)"),
    ("record", "legacy/test_eval/prereg_test2.json", "**test use 2**: registration"),
    ("first", "r5_final/test_eval/results_clavism/metrics.json", "test use 2: result"),
    ("first", "r5_final/analysis/test/*.json", "post hoc: variants and ablations scored on the test split for the draft of that time"),
    ("first", "r5_final/analysis/extra/*.json", "post hoc: baselines on the test split, Four-OOD, other ID sets"),
    ("record", "r5_final/iter2/prereg_iter2.json", "improvement loop 2: stopping rule"),
    ("record", "r5_final/iter2/frozen_m2.json", "loop 2: frozen configuration"),
    ("record", "r5_final/iter2/dev2_looks.jsonl", "loop 2: look at the second development split"),
    ("record", "legacy/test_eval/prereg_test3.json", "**test use 3**: registration"),
    ("first", "r5_final/test_eval/results_m2/metrics.json", "test use 3: result"),
    ("record", "legacy/iter3/prereg_iter3.json", "improvement loop 3: stopping rule"),
    ("record", "legacy/iter3/frozen_m3.json", "loop 3: frozen configuration (`v4`, the first under the working name REPRISE)"),
    ("record", "legacy/test_eval/prereg_test4.json", "**test use 4**: registration (to run only if the look below passes)"),
    ("record", "r5_final/iter3/dev2_looks.jsonl", "loop 3: look at the second development split"),
    ("first", "r5_final/test_eval/results_m3/metrics.json", "test use 4: result"),
    ("first", "r5_final/analysis/reprise/*.json", "post hoc: analyses of the `v4` scores on the test split and on Four-OOD"),
    ("record", "legacy/r5/prereg_r5.json", "matched comparison (R5): protocol, data policy"),
    ("record", "legacy/r5/entrance/frozen.json", "entrance thresholds tuned on the first development split"),
    ("first", "r5_final/r5/entrance/eval/dev2_*.json", "R5: 15 entrance configurations scored on the second development split"),
    ("record", "legacy/r5/prereg_phase2.json", "R5 round 1: candidates, selection and confirmation rule"),
    ("record", "r5_final/r5/prereg_phase2_round2.json", "R5 round 2: coordinate search on the first development split, confirmation on the second"),
    ("record", "r5_final/r5/round2/decision.json", "round 2 decision: the frozen configuration (`v5`)"),
    ("record", "r5_final/r5/phase3/prereg_phase3.json", "**test use 5** (final test of the frozen method): registration with the hashes of the scripts"),
    ("first", "r5_final/r5/phase3/openood/*.json", "test use 5: first result on the test split"),
    ("record", "lp_audit/preregistration.json", "propagation audit on the first development split: registration"),
    ("record", "lp_audit/amendment_01.json", "propagation audit: amendment 1 (before any primary run)"),
    ("first", "lp_audit/reports/all_run_metrics.csv", "propagation audit: metrics of all runs"),
    ("record", "phase4/prereg_p4.json", "Phase 4: registration (families, 27 configurations each, selection rule, endpoints E1 and E2)"),
    ("first", "phase4/results/dev1_tune/*.parquet", "Phase 4: first tuning result on the first development split"),
    ("record", "phase4/amendment_01.json", "Phase 4: amendment 1 (the first attempt to draw the class splits of U1 stopped; no image selected)"),
    ("record", "phase4/selection_lock.json", "Phase 4: **selection lock** (quotes the registration hash)"),
    ("record", "phase4/amendment_03.json", "Phase 4: amendments 2 and 3 on the class splits (no image selected yet)"),
    ("first", "phase4/banks/U1/split?.json", "Phase 4: class splits of U1 written"),
    ("log", "phase4/logs/phase4.status", "STAGE_A_START", "Phase 4: feature extraction of U1-U3 started"),
    ("first", "phase4/results/eval/*.parquet", "Phase 4: first score on U1-U3"),
    ("record", "phase5_6/code/prereg_p5.json", "Phase 5: registration of the improvement loop"),
    ("record", "phase5_6/selection_lock_p5.json", "Phase 5: **selection lock** of the two-sided variant"),
    ("record", "phase5_6/confirm_dev2_attempt1.json", "Phase 5: confirmation on the second development split, attempt 1"),
    ("record", "phase5_6/banks/build_info.json", "Phase 5: evaluation set U4 built (quotes the lock hash)"),
    ("first", "phase5_6/results_final/*.json", "Phase 5: first final result (U4; then **test use 6** and Four-OOD)"),
    ("record", "phase5_6/prereg_p6.json", "Phase 6: registration of the DINOv3 swap (development splits only)"),
    ("first", "phase5_6/results/d3_summary_dev?.json", "Phase 6: first result"),
    ("record", "phase7/prereg_p7.json", "Phase 7: registration of the additional experiments A-E"),
    ("first", "phase7/results/a_dev1_selection.json", "Phase 7 A, first attempt: selection on the first development split"),
    ("first", "phase7/results/a_dev2_confirm_1.json", "Phase 7 A, first attempt: not confirmed on the second development split"),
    ("record", "phase7/amendment_01.json", "Phase 7: amendment 1 (nothing locked; delayed re-scoring analysis)"),
    ("first", "phase7/results/an_retro_u?r.json", "Phase 7: delayed re-scoring of the frozen method on U1 and U4"),
    ("record", "phase7/amendment_02.json", "Phase 7: amendment 2 (case analysis, within-batch memory)"),
    ("log", "phase7/logs/engine_W.status", "start:", "Phase 7: one dispatcher started with the steps dev1, dev2, U1, U4 (within-batch read-outs)"),
    ("first", "phase7/results/an_cases_u?.json", "Phase 7: case analysis of the frozen method on U1 and U4"),
    ("first", "phase7/results/a2_dev1_selection.json", "Phase 7 A, within-batch memory: selection on the first development split"),
    ("log", "phase7/logs/engine_W.status", "dev2w done", "Phase 7: scores of the second development split finished; the dispatcher went on to U1"),
    ("first", "phase7/results/a2_dev2_confirm.json", "Phase 7 A, within-batch memory: confirmed on the second development split"),
    ("record", "phase7/selection_lock_p7.json", "Phase 7: **selection lock** of the extension"),
    ("log", "phase7/logs/engine_W.status", "u1w done", "Phase 7: within-batch scores on U1 finished"),
    ("first", "phase7/results/an_lock_eval.json", "Phase 7: first metrics of the locked extension on U1 and U4"),
    ("record", "phase7/amendment_03.json", "Phase 7: amendment 3 (three further encoders after permission to download)"),
]


def parse(stamp):
    stamp = stamp.replace("Z", "+00:00")
    t = datetime.datetime.fromisoformat(stamp)
    return t if t.tzinfo else t.replace(tzinfo=datetime.timezone.utc)


def written_time(path):
    if path.suffix not in (".json", ".jsonl"):
        return None
    text = path.read_text()
    obj = json.loads(text.splitlines()[0]) if path.suffix == ".jsonl" else json.loads(text)
    if isinstance(obj, dict):
        for key in TIME_KEYS:
            if isinstance(obj.get(key), str):
                return obj[key]
    return None


def build():
    record = json.loads((ROOT / "provenance/server_snapshot_20261004.json").read_text())
    times = {rel: info["server_mtime_utc"] for rel, info in record["files"].items()}
    times.update(record["legacy_server_mtime_utc"])
    digest = {rel: info["sha256"] for rel, info in record["files"].items()}      # hashes as recorded at export
    for earlier in ("server_snapshot.json", "extra_snapshot.json"):
        listed = json.loads((ROOT / "provenance" / earlier).read_text())
        digest.update({rel: info["sha256"] for rel, info in listed.get("files", listed).items()})
    by_digest = {}
    for rel, d in digest.items():
        by_digest.setdefault(d, []).append(rel)
    rows = []
    for kind, pattern, *rest in ROWS:
        label = rest[-1]
        if kind == "log":
            lines = [line for line in (ROOT / E / pattern).read_text().splitlines() if rest[0] in line]
            assert len(lines) == 1, (pattern, rest[0])
            stamp = re.search(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z", lines[0]).group(0)
            rows.append((stamp, f"`{pattern}`: line `{rest[0]}`", "–", "–", label))
        elif kind == "record":
            rel = E + pattern
            copies = [c for c in by_digest[digest[rel]] if c in times]
            server = min(times[c] for c in copies)                    # identical copies: the earliest one
            written = written_time(ROOT / rel)
            late = written is not None and abs((parse(written) - parse(server)).total_seconds()) > 120
            shown = "–" if written is None else written[:19].replace("T", " ") + (" †" if late else "")
            rows.append((server, f"`{pattern}`", shown, f"`{digest[rel][:12]}`", label))
        else:
            found = sorted((t, rel) for rel, t in times.items() if fnmatch.fnmatchcase(rel, E + pattern))
            assert found, pattern
            rows.append((found[0][0], f"`{pattern}`" + (f" ({len(found)} files)" if len(found) > 1 else ""), "–", "–", label))
    order = sorted(range(len(rows)), key=lambda i: (rows[i][0], i))
    out = ["| server time (UTC) | file below `experiments/` | time written in the record | sha256 | content |", "|:--|:--|:--|:--|:--|"]
    out += ["| " + " | ".join((rows[i][0].replace("T", " ").rstrip("Z"),) + rows[i][1:]) + " |" for i in order]
    return "\n".join(out)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--check", action="store_true", help="compare with the block in docs/PREREGISTRATION.md instead of writing it")
    args = parser.parse_args()
    text = DOC.read_text(encoding="utf-8")
    head, rest = text.split(BEGIN)
    old, tail = rest.split(END)
    new = "\n" + build() + "\n"
    if args.check:
        print("timeline is up to date" if old == new else "timeline differs from the archived records")
        sys.exit(0 if old == new else 1)
    DOC.write_text(head + BEGIN + new + END + tail, encoding="utf-8")
    print(f"wrote the timeline of {DOC.relative_to(ROOT)} ({len(new.splitlines()) - 3} rows)")


if __name__ == "__main__":
    main()
