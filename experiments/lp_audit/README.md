# REPRISE Phase 0-2: matched LP and validity audit

This directory is an isolated addition. Existing paper, Git history, legacy results and the server source tree are not overwritten. The source used for the paper is preserved under `vendor/` and in `provenance/server_source.tgz`. The GitHub repository HEAD at start was `965e74420e6395cda902486aa62ed8fe04d6bcae`.

The requested `REPRISE_paper_v2_1.pdf` has SHA256 `784f7ee7f574b977625d930a4aaaa572ef7bfcfc8b2d74f102bab7f2dd82aa27`, identical to `reprise_v5/main.pdf`. Source/configuration correspondence and the documented original code repairs are audited against the paper's frozen Phase-3 registration. The newly requested Phase 0-2 are named separately from that historical "Phase 3".

## Execution

Runtime on hades: `/home/omote/granood_ke/.venv/bin/python`. Read-only input root: `/home/omote/vins_gonogo_20260925`. Output root: `/home/omote/reprise_lp_audit_20260927`.

```sh
python audit.py
python finalize_audit.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m unittest discover -s unit_tests -v
CUDA_VISIBLE_DEVICES=0 python pilot.py --n 512
CUDA_VISIBLE_DEVICES=0 python pilot.py --n 4096 --classwise
# The existing registration is immutable; do not regenerate it over completed runs.
python synthetic.py
python real_calibration.py --prepare
python schedule.py --limit-hours 4 --classwise
python post_schedule.py
python report.py
python equivalence.py
python analyze.py
python static_factor_diagnostic.py
python write_reports.py
python verify_results.py
python summarize_execution.py
python make_summary.py
```

The actual launch waits for the physical image/input hash audit before starting primary GPU experiments. `schedule.py` dispatches 30 matched draw/stream/order jobs. `post_schedule.py` waits for this dispatcher to finish, then evaluates 50 real-calibration redraws and the registered baseline/capacity/tuning comparisons. After measured full-run costs were available, `provenance/amendment_02.json` placed capacity before tuning and added whole-stage budget admission. Skipped tasks remain individually listed in the ledger. It enforces a cumulative 8 allocated GPU-hour ceiling, including failed jobs and a conservative 200 GPU-second allowance for pilots/startup/TINS prefix replay. CPU-only audit and 200 independent simulations are listed separately.

## Interpretation

- Primary new evaluation is the already used **dev1**, not a new test: 900 ID classes, 18,000 ID images, 5,000 near OOD images, 1,763 far OOD images. Five genuinely distinct 12+4 shot draws and three matched stream orders are used. Dev2 and prior test have used status. New holdout is not opened.
- Same graph scores: raw binary u, support-median normalization, frozen pre-stream calibration CDF, current-calibration rank pLP, memory pt, and REPRISE. Every score has a standalone and matched TINS product version. All 900 class channels of standard classwise LP are computed in FP64 blocks, not a selected class subset.
- L0 reproduces the original y initialization at the first batch and warm history thereafter; L1 is all-zero 15 iterations; L2 uses equation-relative residual 1e-8. Graph and calibration remain shared. Initial implementation tests that failed because the direct-solver reference multiplied a float32 matrix by lambda before promoting it were preserved, then corrected to FP64 reference arithmetic.
- Common support and calibration have 16 labels/class total. Frozen CDF uses those same calibration images before the stream; M1 partitions the four shots/class into four roles. No extra labels.
- Numeric zero products keep exact ties with log(0)=-inf and rank-based AUROC, per `amendment_01.json`. Positive products are also checked in ordinary and log arithmetic. The log-floor entry in the original registration is superseded by this pre-primary arithmetic correction.
- Three parameter configurations per named family (including the original LP setting), all selected only on dev1. The original paper's larger/asymmetric development history is retained; equal new budget does not erase it. Incomplete grids are not treated as fully tuned. `baseline_scope.md` distinguishes the Mahalanobis++ adaptation and the simplified negative-cache/dictionary surrogates from official AdaNeg/OODD. Results on these surrogates do not establish superiority to those original methods.
- A post-primary read-only diagnostic compares REPRISE with static-p times pLP, because REPRISE-minus-pLP also includes a static distance factor. Its timing and definition are recorded in `provenance/static_factor_diagnostic_registration.json`. It is not added to the four registered candidates and does not change their selection rule. The matched static-factor control reveals a near/far tradeoff that is hidden by comparing only against pLP.
- Bounded mode retains support/calibration and at most 8,192 stream nodes, plus at most 8,192 members of each A1/A2/M. Exact deletion updates recompute the neighbor lists affected by eviction and are checked against a full rebuild. No approximate neighbor search is used.
- Intervals average three orders within a shot draw, then use five paired draw values. They do not cover uncertainty over entirely new classes; selected development winners have exploratory, not confirmatory, intervals. The 10-draw/3-order confirmation and Holm testing are reserved, not executed here.
- Timings after cached features are not end-to-end latency. Shared graph construction and each solver's incremental time are reported separately. A per-image batch average is not batch-size-one response latency.

See `validity_lemmas.md` for mathematical scope, `dependency_audit.md` for causal paths, `reports/completion_status.json` for the exact completed/planned counts, `failures/` and dispatcher ledgers for failed or budget-stopped runs. No positive-result stopping rule is used.

Start with `summary.md` after final aggregation. It distinguishes measured runs, conservative GPU budget charges, and individually skipped tasks. `reports/result_manifest.json` hashes the server-side raw results; `reports/derived_cache_manifest.json` hashes derived feature/identity preparation artifacts. Large raw result/feature files remain on hades and are not Git-tracked.
