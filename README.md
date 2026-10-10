<p align="center"><img src="docs/assets/salmon_ladder.png" alt="Salmon Ladder" height="200"></p>

# Salmon Ladder

Salmon Ladder is a training-free detector of unknown classes that recur in a set of unlabelled test images. This repository holds its code, the records of every experiment, and tools that verify both on a CPU.

The paper (in Japanese, seven pages; OpenOOD v1.5 ImageNet-1K and Four-OOD) is [paper/Salmon_Ladder.pdf](paper/Salmon_Ladder.pdf). <!-- maintainers -->

## The idea💡

Among the test images, an unknown class rarely appears once: like a salmon, it comes back. Salmon Ladder uses these returns as evidence. It trains nothing: two frozen encoders, 16 labelled images per known class (12 support and 4 calibration images), and one nearest-neighbour graph per encoder whose nodes are the support, calibration and test images.

1. **Mass from the support images.** The label mass of the support images is propagated over the graph. An unknown image among other unknown images receives little mass; every value is read as its rank among the calibration images, a conformal p-value `p+`.
2. **Self-labelled seeds.** The Benjamini-Hochberg procedure on `p+` (with Storey's estimate of the ID fraction, q = 0.1) selects the images that are confidently unknown, with the false discovery rate controlled under exchangeability.
3. **Mass from the seeds.** The mass of the seeds is propagated over the same graph and read as a rank `p-`. The unknowns found first pull the rest of their class up the ladder.

The score is `s0(x) * prod_v p+_v(x) p-_v(x)` over the views `v` (DINOv3 ViT-L/16 and the concatenation of DINOv2 ViT-L/14 with DINOv3 ViT-L/16); `s0` is any base detector, TINS in the paper. Scores are assigned after the whole stream has arrived. [docs/METHOD.md](docs/METHOD.md) states every step exactly as the code computes it and names the function that implements it.

**History.** Phases 4-9 developed an online variant that scores each image at arrival with an entrance memory and warm-start propagation; its records, registrations and evaluations on unused data remain archived and documented (`docs/METHOD.md`, Section 2; `docs/PREREGISTRATION.md`). The experiments were run under the working name REPRISE: archived scripts, registrations, result files and the directory names of the experiment server keep that name, because their content is fixed by hashes.

## Results at a glance🙆

AUROC / FPR95 in %. [docs/RESULTS.md](docs/RESULTS.md) holds all result tables (Section 8 for the public benchmarks of the paper). It and the tables below are generated from the archived result files. Values that the paper quotes from other publications are not part of it. `zeta` is the registered comparator of the online variant: among the detectors that use propagation without the entrance and the memory, it was the strongest on the development split, where every detector was tuned with the same budget.

<!-- BEGIN GENERATED: results -->
Salmon Ladder (Phase 11, the paper): scores assigned after the whole stream has arrived, two views, 16 labelled images per class.

| public benchmark streams | OpenOOD near-OOD | OpenOOD far-OOD | Four-OOD |
|:--|--:|--:|--:|
| TINS alone (measured on the same streams) | 81.48 / 56.37 | 97.04 / 12.23 | 98.52 / 6.85 |
| Salmon Ladder, standalone | 97.95 / 11.02 | 98.78 / 5.23 | 98.78 / 4.39 |
| **Salmon Ladder x TINS** | 98.07 / 10.22 | 99.03 / 4.28 | 99.06 / 3.36 |

The online variant of Phases 4-7 (entrance memory and warm-start propagation, scored at arrival), on the data that no selection had used:

| evaluation | base detector alone | x zeta | x online variant | online variant - zeta (FPR95, 95% interval) |
|:--|--:|--:|--:|--:|
| U1: new class splits of ImageNet-1K, standalone | – | 92.40 / 29.43 | 93.86 / 26.27 | -3.16 [-4.15, -2.16] |
| U1, with TINS | 64.95 / 85.67 | 92.27 / 30.17 | 93.79 / 26.48 | -3.69 [-4.67, -2.72] |
| U2: ImageNet-O, standalone | – | 89.93 / 44.15 | 91.81 / 38.20 | -5.96 [-6.75, -5.16] |
| U2, with TINS | 78.62 / 68.46 | 90.48 / 42.32 | 92.07 / 37.26 | -5.06 [-5.84, -4.28] |
| OpenOOD v1.5 ImageNet-1K near-OOD, with TINS | 81.48 / 56.37 | 96.44 / 16.54 | 96.30 / 16.76 |  |
| OpenOOD v1.5 ImageNet-1K far-OOD, with TINS | 97.04 / 12.23 | 98.74 / 5.79 | 98.85 / 5.55 |  |

Registered endpoints on U1: E1 is the standalone row. E2 uses TINS and gives zeta its development-selected weight (FPR95 26.48 against 29.70): -3.22 [-4.21, -2.24]. The rows with TINS above use weight 1 for both methods.
<!-- END GENERATED: results -->

The configuration of the paper (views, q, base detector) was chosen by comparisons on the OpenOOD streams themselves, after sweeps of the post-stream read-out on the same streams; the paper says so, and [docs/PREREGISTRATION.md](docs/PREREGISTRATION.md) lists every use of the test split. U1 and U2 were built after the configurations of the online variant and of zeta had been locked (by the file times of the records); the registered claim about the entrance and the memory rests on them.

## What the repository contains😃

The experiment code is archived byte for byte as it was on the experiment server when it was exported, together with the registrations and selection locks that fix the order of decisions and the aggregated result files.

| location | contents |
|---|---|
| [`paper/`](paper/) | the paper (in Japanese): `Salmon_Ladder.pdf` <!-- maintainers --> |
| [`experiments/phase10_11/`](experiments/phase10_11/) | Phase 10 (DINOv3 views, base detectors and the online variant on the OpenOOD and Four-OOD streams) and Phase 11 (the post-stream read-out of the paper: sweeps and the final run) |
| [`docs/METHOD.md`](docs/METHOD.md) | the method, with the function that implements each step |
| [`docs/RESULTS.md`](docs/RESULTS.md) | all result tables, generated by `tools/make_results_tables.py` |
| [`docs/PREREGISTRATION.md`](docs/PREREGISTRATION.md) | registrations, selection locks, every use of the test split, deviations and unkept commitments |
| [`docs/EXPERIMENTS.md`](docs/EXPERIMENTS.md) | which script produced which result |
| [`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md) | environment, data, model weights, order of execution |
| [`experiments/phase4/`](experiments/phase4/) | evaluation on new data (U1, U2), original baselines, recurrence intervention, operating conditions; `code/engine.py` is the frozen streaming engine |
| [`experiments/phase7/`](experiments/phase7/) | first appearances and the extension, encoders, components, data sets, stream conditions; `code/engine7.py` adds the extension |
| [`experiments/phase5_6/`](experiments/phase5_6/) | improvement loop (two-sided variant, not adopted) and DINOv3 swap |
| [`experiments/lp_audit/`](experiments/lp_audit/) | audit of the comparison with label propagation; `vendor/` is the frozen implementation that the later phases import |
| [`experiments/legacy/`](experiments/legacy/), [`experiments/r5_final/`](experiments/r5_final/) | original implementation and development history; matched comparison, final configuration and the final test on OpenOOD, Four-OOD and CUB |
| [`experiments/controls_456/`](experiments/controls_456/), [`experiments/tins_reference/`](experiments/tins_reference/) | earlier controls and the TINS reproduction |
| [`third_party/`](third_party/) | TINS, DINOv2 hub source, LoCoOp and licences ([THIRD_PARTY.md](THIRD_PARTY.md)) |
| [`manifests/`](manifests/) | image identifiers, splits and portable image paths; no image pixels |
| [`provenance/`](provenance/) | hashes, sizes and server times of every archived file, package versions, records of the checks |
| [`tests/`](tests/), [`examples/`](examples/) | CPU tests and a runnable example on synthetic features |
| [`tools/`](tools/) | verification, result tables, materialization of an executable workspace, comparison of a re-run |

An anonymized export of this repository contains `provenance/anonymized_export.json`. The files listed there differ from the server copies: names and paths are replaced, the hashes and sizes of changed files are updated, and the hashes and sizes of files outside the archive are replaced by pseudonyms and null.

## Verify on a CPU✌️

Python 3.12.3, PyTorch 2.5.1 and torchvision 0.20.1 were used on the server. Nothing below needs data, model weights or a GPU.

```bash
python -m venv .venv
source .venv/bin/activate
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements-core.txt

python tools/verify_snapshot.py                    # every archived file, registration and listed quoted hash
python tools/make_results_tables.py --check        # docs/RESULTS.md follows from the archived result files
python tools/make_registration_timeline.py --check
python -m pytest -q tests                          # frozen engine, extension, variants, documentation
python examples/salmon_ladder_rereading.py         # Salmon Ladder (the post-stream read-out of the paper) on a synthetic stream
python examples/salmon_ladder_stream.py            # the online variant and its extension on a synthetic stream
```

The tests of the earlier stages run the same way:

```bash
python -m pytest -q experiments/controls_456/src/test_controls.py
PYTHONPATH=experiments/legacy python -m pytest -q experiments/legacy/tests/test_r5.py
(cd experiments/lp_audit && python -m unittest discover -s unit_tests)
python examples/feature_stream.py
```

`python tools/run_cpu_checks.py` runs all of the commands above. The examples use synthetic features. They show the interface and the behaviour under recurrence; they are not benchmark results.

## Run the experiments🍤

The archived scripts keep the paths of the server. `python tools/materialize_workspace.py --workspace /absolute/path` writes a separate copy with the server layout, resolves the paths, and rebuilds the image tables from `manifests/`. The archive itself is never edited. Data and model weights are obtained from their providers; [docs/REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md) lists what is needed and the order of execution. The archived selection locks make it possible to rerun an evaluation without repeating the selection.

`python tools/compare_rerun.py --workspace /absolute/path` compares the metric tables of a re-run with the archived ones. `provenance/rerun_check_20261004.json` records such a check on the experiment server.

## Conventions🦀

- Scores are large for in-distribution images. AUROC takes ID as the positive class; FPR95 is the share of unknown images accepted at the threshold that accepts 95% of the ID images.
- `x` denotes the product of scores (the sum of log scores with weight 1). `standalone` means no base detector.
- Calibration ranks are detection scores. In the post-stream read-out of the paper the calibration images and the test images are nodes of the same fixed graph, so `p+` is a conformal p-value when the ID test images and the calibration images are exchangeable (the seeds are selected from it by Benjamini–Hochberg). In the online variant the calibration images are reused by every batch and the memory adapts to the stream, so there the ranks are not presented as distribution-free p-values.
- The original evaluation scripts use the FPR95 convention of the TINS code; the controls and later phases use a threshold that accepts at least 95% of the ID images and count ties as accepted. The conventions are not mixed within a table.
- Intervals are paired 95% t intervals over the stated units. Analyses that were not registered are labelled descriptive or post hoc.
- The working names in the archived files (`REPRISE`, `CLAVIS`, `v4`, `v5`, `minimal`) are explained at the end of [docs/METHOD.md](docs/METHOD.md).

## Third-party code and licences📃

[THIRD_PARTY.md](THIRD_PARTY.md) lists the upstream revisions and licences of the vendored code. Datasets and pretrained models are not redistributed. A licence for the code of Salmon Ladder itself has not been selected yet.

Maintainers: [docs/SUBMISSION.md](docs/SUBMISSION.md) describes how to build the anonymized archive for double-blind review; `CITATION.cff` identifies this repository. <!-- maintainers -->
