# Experiment map

All paths below are relative to a materialized workspace. Run original commands from its `vins_gonogo_20260925` directory, and control commands from `reprise_controls_20260927`. Do not run every historical search script to select a new method on test results.

## Original REPRISE

`iter3/frozen_m3.json` specifies the final configuration. `scripts/iter3_eval.py` defines prototype distances, the iterated admission rule, memory scoring, and label propagation. `scripts/confirm_iter3.py` checks the development setting. `scripts/reprise_analysis.py` evaluates the frozen method with saved TINS streams and feature caches:

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

Per-condition arrays and metadata are written to `results/`; diagnostics to `status/`; errors to `failures/`; and tables/plots to `reports/`. Missing or failed conditions are not treated as zero-valued results. `report.py --watch` generates partial reports as jobs finish. No final numerical claims for the pending control sweep are added by this code release.
