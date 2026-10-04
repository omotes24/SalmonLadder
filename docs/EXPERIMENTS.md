# Experiment map

All paths below are relative to a materialized workspace (`tools/materialize_workspace.py`). Run original commands from its `vins_gonogo_20260925` directory, and control commands from `reprise_controls_20260927`. Do not run every historical search script to select a new method on test results.

| part of the paper | section below | archive |
|---|---|---|
| frozen method and its final test (OpenOOD, Four-OOD, CUB) | [R5 round 2 and the final test](#r5-round-2-and-the-final-test) | `experiments/legacy` + `experiments/r5_final` |
| comparison with label propagation on the development split | [Propagation audit](#propagation-audit) | `experiments/lp_audit` |
| evaluation on images not used for selection, baselines, intervention, operating conditions | [Phase 4](#phase-4-evaluation-on-new-data) | `experiments/phase4` |
| two-sided variant, DINOv3 swap | [Phase 5 and Phase 6](#phase-5-and-phase-6) | `experiments/phase5_6` |
| first appearances and the extension, encoders, components, data sets, stream conditions | [Phase 7](#phase-7-additional-experiments) | `experiments/phase7` |
| earlier development and controls | the sections "Original implementation" and ④–⑥ | `experiments/legacy`, `experiments/controls_456` |

[RESULTS.md](RESULTS.md) holds the result tables; it is generated from the archived result files. [PREREGISTRATION.md](PREREGISTRATION.md) gives the order of registrations, locks and evaluations.

## Original implementation

`iter3/frozen_m3.json` specifies the first configuration of the method under its working name REPRISE (`v4`; the frozen configuration of the paper, `v5`, was fixed later in R5 round 2). `scripts/iter3_eval.py` defines prototype distances, the iterated admission rule, memory scoring, and label propagation. `scripts/confirm_iter3.py` checks the development setting. `scripts/reprise_analysis.py` evaluates the frozen method with saved TINS streams and feature caches:

```bash
python scripts/reprise_analysis.py --part openood --workers 4
python scripts/reprise_analysis.py --part fourood --workers 4
python scripts/reprise_summary.py
python scripts/reprise_recurrence2.py
```

These commands require the inputs listed in the reproduction guide. They are not image-download commands. Some historical scripts assert the original preregistration/source hashes. Preserve these files; changing paths inside them invalidates the original receipts.

R5 scripts (`scripts/r5_*.py` and `scripts/r5_pipeline*.sh`) preserve the separate matched-comparison/development study. Later `p3_locoop.py` is a LoCoOp comparison utility. Its inclusion records available comparison code; it does not assert that every later-stage experiment completed or belongs to the final method. The archived `deploy.sh` records the original deployment and should not be used to install this repository.

## ④ Controlled recurrence: 576 conditions

- Four eligible classes from NINCO and four from SSB-hard, eight fixed query images per class, and a common set of 128 ID queries.
- Two fixed histories per target class: 1,024 images, comprising 768 ID and 256 OOD images.
- Same-class recurrence counts 0, 1, 2, 5, 10, and 20. Replace other-class OOD images at the same positions, matching static-score difficulty where possible.
- History batches 1, 16, and 256; original warm-start 15-sweep LP and a cold-start convergence control.
- Each evaluation query is scored from a cloned pre-query state. Queries never become history for another query. Separate reserved peers isolate within-batch similarity.

`plans/*.npz` stores exact row selections, while `plans/tasks.json` fixes the full condition list. `run.py --task <id>` executes one condition. `report.py` produces recurrence curves and paired full-versus-LP contrasts. Class-bootstrap intervals condition on the recorded histories and do not resimulate adaptive streams. Four classes per dataset support a limited diagnostic, not a broad population guarantee.

## ⑤ Prevalence, recurrence, retention: 592 + 24 conditions

Four panels contain two unknown classes each. OOD fractions are 0, 0.1, 1, 5, and 20%; recurrence counts are 1, 5, and 20; schedules are random, burst, and return after a long gap. Retention limits are 1,000, 5,000, 20,000, and unlimited stream arrivals. The same unique OOD sequence is retained while unique ID images are inserted. No stream fills missing images by repeating an image.

The window applies to A1, A2, M, and graph stream nodes; support and calibration nodes remain fixed. A long-gap schedule with only one occurrence cannot demonstrate recurrence and is marked accordingly. Zero-OOD streams report false alarms; AUROC and FPR95 are undefined there.

`thresholds.json` freezes per-method thresholds using old-dev ID scores. Outputs include ID alarms/1,000, OOD detection rate, first-detection delays, and never-detected-class fraction. The 24 `runtime` conditions separately measure raw-image decoding, feature extraction, TINS, neighbor search, and graph/LP processing. Cached-feature timings from the other conditions are explicitly not end-to-end timings. Model/state initialization is separate; fixed-support feature extraction and arrival-queue waiting are not part of stream latency.

## ⑥ New datasets and feature-model controls: 18 conditions

- CUB: 100 ID / 100 OOD classes. CIFAR-100: 50 ID / 50 OOD classes.
- Official training images supply 12 support + four calibration images per ID class. Official test images are evaluated. Additional reserved ID-training rows are not test queries.
- B/14 image-only, CLIP image-only, and the original two-DINO-view reference; three stream orders each.
- Primary comparisons use the same visual features. The kNN rank and Mahalanobis++ shrinkage setting are fixed in `baseline_selection.json`, selected only on the old development split.

Exact RGB duplicates of fixed support/calibration images and earlier evaluation rows are excluded. Resized or otherwise altered near-duplicates are not guaranteed absent. CUB and CIFAR manifest rows, class partitions, exclusions, and original hashes are included in this repository.

## Outputs and failures

Per-condition arrays and metadata are written to `results/`; diagnostics to `status/`; errors to `failures/`; and tables/plots to `reports/`. Missing or failed conditions are not treated as zero-valued results. `report.py --watch` generates partial reports as jobs finish. The controls ④–⑥ evaluate the earlier configuration `v4`. They finished after the first export; their result files are not archived here, and the recurrence intervention, operating conditions and new data sets reported in the paper are those of Phase 4 and Phase 7, which use the frozen configuration `v5`.

## R5 round 2 and the final test

Workspace directory `vins_gonogo_20260925`; the files of this stage are archived in `experiments/r5_final` and are laid over `experiments/legacy` at materialization.

- `scripts/r5_round2.py` runs the registered coordinate search on the first development split and the confirmation on the second (`r5/prereg_phase2_round2.json`); `r5/round2/decision.json` fixes the frozen configuration `v5`.
- `scripts/p3_prereg.py` writes `r5/phase3/prereg_phase3.json` with the hashes of every script that reads test data. `scripts/p3_pipeline.sh` refuses to start if the registration differs from its hash and then runs, in this order: `p3_extract.py` (features of the test images), `p3_m2b.py` (calibration on ID validation images), `p3_streams.py --stage build|augfeat|tins|eval` (stream conditions on test data), `p3_static.py` (static scores and same-feature baselines), `p3_locoop_run.py`, `p3_eval.py --part openood|fourood` (one JSON per data set and arrival order), `p3_cub.py` (CUB with the Semantic Shift Benchmark split) and `p3_report.py` (`r5/phase3/summary.json`).
- `scripts/p4_paper_data.py` reads the saved scores and writes `r5/paper/paper_data.json` (MCM and NegLabel as base detectors, detection by number of earlier appearances, calibration curves). Nothing is selected from it.
- `iter/`, `iter2/`, `iter3/`, `dev2/` and `test_eval/` hold the decision records, metrics and run logs of the four evaluations on the test split that preceded the final test.

## Propagation audit

Workspace directory `reprise_lp_audit_20260927` (`experiments/lp_audit`). The audit asks whether the gain of Salmon Ladder over label propagation survives when propagation is given the same calibration, solver and tuning budget. It uses the first development split only and opens no new data. `vendor/` is the copy of the original implementation that the audit ran against; `core.py` re-implements the frozen method with an exact incremental graph and three solvers (warm start, zero start, converged), and `reports/frozen_v5_equivalence.csv` compares it with the original scores.

The execution order is listed in `experiments/lp_audit/README.md`. `reports/*.csv` hold every run and every paired difference; `validity_lemmas.md` states under which assumptions the calibration ranks are valid and where the implementation leaves them; `dependency_audit.md` traces which inputs each score depends on. `unit_tests/` (16 tests) run on a CPU.

## Phase 4: evaluation on new data

Workspace directory `reprise_p4_20260928` (`experiments/phase4`); all scripts are in `code/` and are run from there.

| step | scripts | output |
|---|---|---|
| registration and tuning | `launch_tune.sh` → `dev1_tune.py` (27 graph configurations for every read-out family on the first development split); `select_lock.py` | `results/dev1_tune/`, `selection_lock.json` |
| evaluation sets | `build_banks.py` (U1: five class splits of ImageNet-1K; U2: ImageNet-O; U3) | `banks/` |
| features | `stageA.sh` → `extract.py` (four shards, then merge) | `features/` |
| base detector | `tins_bank.py` (TINS per stream) | `tins/` |
| registered evaluation | `evaluate.py` (eight workers), `analyze_eval.py` | `results/eval/*.parquet`, `results/eval_summary.json` |
| original baselines | `vlm_tta.py` (AdaNeg, TANL, NegLabel through the OpenOOD-VLM post-processors in `code/vlm_tta/`), `oodd_eval.py`, `analyze_baselines.py` | `results/baselines.parquet`, `results/vlm_tta_meta.json` |
| recurrence intervention (experiment 4) | `exp4_design.py`, `extract_aug.py`, `exp4_run.py`, `analyze_exp4.py` | `banks/exp4/`, `results/exp4_summary.json` |
| operating conditions (experiments 5–8) | `operating.py --exp exp5` (batch size, fixed thresholds), `--exp exp6` (arrival conditions), `bounded_p4.py` (bounded history), `exp8.py` (labelled images per class) | `results/operating/`, `results/bounded/`, `results/exp8/` |
| checks and cost | `theory_checks.py`, `prefix_check.py`, `test_dev1_equiv.py`, `resources.py` | `results/theory_checks.json`, `results/resources.json` |

`run_phase4.sh` is the orchestrator that ran the steps after the features existed; `logs/phase4.status` records when each step finished. `engine.py` is the streaming engine (Section 2–5 of [METHOD.md](METHOD.md)); `state.py` is the same engine with a clonable state for the intervention; `static.py` computes the static scores. `failures/` keeps the two failed launches (a missing directory, and a memory overflow of the unmodified AdaNeg post-processor on 11 GB GPUs, after which `vlm_tta/openood/postprocessors/adaneg_chunked.py` computes the same quantity in chunks).

U3 is a split of a private wildlife data set. Its registration and aggregated results are archived; its image list is not, so U3 cannot be rebuilt from this repository.

## Phase 5 and Phase 6

Workspace directory `reprise_p5_20261002` (`experiments/phase5_6`).

**Phase 5** (improvement loop). `dev5.py` scores candidate configurations on a development split through `engine5.py`, which adds two-sided propagation, self-labelled seed sets and several explored variants to the frozen engine (`rules5.py`, `diag5.py`, `analyze5.py`, `r2_analysis.py`–`r6_analysis.py` are the analyses of the successive rounds). `make_lock.py` writes `selection_lock_p5.json`; `confirm5.py` runs the confirmation on the second development split. `run_final.sh` then builds the new evaluation set U4 (`u4_build.py`, `u4_extract.py`, `u4_tins.py`) and scores U4, the OpenOOD test split and Four-OOD once each (`final5.py --part u4|openood|fourood`, `final_analysis.py`), writing `results_final/summary_*.json`.

**Phase 6** (encoder swap). `run_d3.sh` checks the registration hash and runs `probe6.py` (weights and read-out check), `extract6.py` and `merge6.py` (DINOv3 ViT-B/16 and ViT-L/16 features of the labelled images and of both development splits), the sanity gate `ncm6.py`, the development runs (`dev5.py` with the swapped views) and `d3_analysis.py`. `run_posthoc.sh` and `d3_posthoc.py` add single-view and mixed-view variants, labelled post hoc.

## Phase 7: additional experiments

Workspace directory `reprise_p7_20261003` (`experiments/phase7`). `engine7.py` wraps the frozen engine and adds the read-outs of Section 6 of [METHOD.md](METHOD.md); `data7.py` gives one interface to every evaluation set; `an7.py` holds the analysis helpers.

| experiment | scripts | results |
|---|---|---|
| preparation | `run_stage1.sh`, `run_stage1b.sh`: `u1x_table.py`, `extract7.py` (further encoders), `build7.py` (class splits of the five data sets), `text7.py` (CLIP text features of the class names) | `banks/`, `features/*/info_*.json` |
| A: first appearances, descriptive | `run7.py --exp u1std\|u4std` with `an_u1.py` (miss rate by arrival order within a class); `--exp u1r\|u4r` with `an_retro.py` (delayed re-scoring); `an_cases.py`; `x4.py` with `an_x4.py` (intervention re-run) | `results/an_u1.json`, `results/an_retro_*.json`, `results/an_cases_*.json`, `results/an_x4.json` |
| A: first attempt (not confirmed) | `dev7.py --dev dev1\|dev2`, `an_a_dev.py select\|confirm` | `results/a_dev1_selection.json`, `results/a_dev2_confirm_1.json` |
| A: within-batch memory (the extension) | `dev7.py --within`, `an_within.py`, `make_lock7.py`, then `run7.py --exp u1w\|u4w` and `an_lock_eval.py` | `results/a2_dev1_selection.json`, `results/a2_dev2_confirm.json`, `selection_lock_p7.json`, `results/an_lock_eval.json` |
| B: encoders | `run7.py --exp u1std\|u2std --views <view>` (and `u1w\|u2w` for the extension), `an_b.py`, `ncm7.py u1` (nearest-class-mean accuracy), `bench7.py` (encoding time) | `results/an_b.json`, `results/ncm_u1.json`, `results/bench7*.json` |
| C: memory and propagation separated | `an_c1.py` (2 x 2 from the intervention scores); `run7.py --exp u1std` (propagation on the current batch only) and `--exp u1alone` (on the image alone) | `results/c1_summary.json`, `results/an_u1alone.json` |
| D: data sets | `run_ds.sh`: `run7.py --exp dsw`, `base7.py` (kNN and Mahalanobis++ on the same features), `ncm7.py ds`, `an_d.py` | `results/an_d.json`, `results/ncm_ds.json` |
| E: stream conditions | `streams7.py` (stream builders), `run7.py --exp e1\|e2\|e3` and `e1w\|e2w\|e3w`, `an_e.py` | `results/an_e.json` |
| base detectors with the extension | `an_ext_bases.py` | `results/an_ext_bases.json` |
| verification | `test7_hades.py` (frozen read-outs against the stored scores), `verify7.py` (independent re-computation of the headline numbers) | `results/test7_hades.json`, `logs/verify7*.log` |

`run_engine.sh <tag> "<gpus>" <workers> <steps>` is the dispatcher used for every `run7.py` and `dev7.py` step; `logs/engine_*.status` record what each dispatcher ran and when. Experiment A selects on the development splits only; B–E apply frozen hyper-parameters. `run_p7b.sh` is amendment 3 (DINOv3 ViT-S/16 and ViT-S+/16, CLIP RN50).
