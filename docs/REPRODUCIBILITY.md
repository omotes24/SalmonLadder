# Reproduction guide

## Environment

The complete recorded package versions and Python version are in `provenance/server_snapshot.json` (first export) and `provenance/server_snapshot_20261004.json` (second export; the same versions, on Ubuntu 24.04 with four NVIDIA RTX 2080 Ti, 11 GB each). Phases 10 and 11 ran in the same environment (`provenance/server_snapshot_20261011.json`, third export). Core matching and propagation tests run on CPU. TINS inversion and image feature extraction require a CUDA-capable GPU in this implementation.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements-reproduction.txt
```

The requirements distributed inside the upstream TINS repository describe its own environment. Use the environment recorded above for this snapshot.

## Source integrity

`python tools/verify_snapshot.py` checks the exported server files against SHA-256 receipts. It also checks the original `iter3_eval.py` frozen hash, the controls' source manifest, every registration and selection lock against its hash file, and the hashes that decision records quote from each other (the list `QUOTED` in the tool; [PREREGISTRATION.md](PREREGISTRATION.md) explains the chains). A file under `experiments/`, `manifests/` or `third_party/` that no record lists is an error. The snapshot export time identifies when these files were copied; it does not establish when a research decision was preregistered. Some historical JSON registration timestamps have a previously noted UTC inconsistency. Keep the original records and use version history and source hashes for traceability.

The original TINS adapter was verified against its upstream scoring function on 512 old-development images: bitwise equality, maximum absolute difference 0. See `experiments/controls_456/tests/tins_equivalence.json`. This verifies that adapter on that sample, not every dataset or GPU configuration. The outcome of the CPU checks is recorded under `provenance/`: `submission_checks.json` for the first export and `submission_checks_20261004.json` for the second (`python tools/run_cpu_checks.py --record <file>`).

## Data and models

Obtain licensed data from the respective official providers:

- ImageNet-1K and the [OpenOOD ImageNet benchmark](https://github.com/Jingkang50/OpenOOD). Use the official lists matching the stored identifiers, not an arbitrary replacement split.
- [CUB-200-2011](https://www.vision.caltech.edu/datasets/cub_200_2011/) / [official archive](https://data.caltech.edu/records/65de6-vp158). Expected archive MD5: `97eceeb196236b17998738112f37df78`.
- [CIFAR-100](https://www.cs.toronto.edu/~kriz/cifar.html), official train/test split.
- Four-OOD additionally uses the TINS upstream ImageNet/iNaturalist/SUN/Places/DTD setup. Its DTD and ImageNet populations differ from the OpenOOD protocol.

Frozen feature models: [DINOv2](https://github.com/facebookresearch/dinov2) ViT-B/14 and ViT-L/14, and [OpenAI CLIP](https://github.com/openai/CLIP) ViT-B/16. Checkpoint paths and available checksums are in `vins/config.py` and the TINS model receipt. The original DINO hub source is also archived under `third_party/dinov2` when present; use that source for the local hub loader. LoCoOp is optional and needs the authors' 16-shot ImageNet checkpoint for each requested seed.

Images, model checkpoints, feature tensors, evolving memory state, and completed-score archives are intentionally not placed in Git. Image manifests and exact control-stream indices are included. `manifests/*_images.csv.gz` preserves the control banks' row order, so the saved plan arrays remain meaningful. Path tokens stand for a materialized data workspace. The manifest tables contain labels for construction and metrics; detector APIs receive features and the permitted ID labels only.

## Materialize a working copy

```bash
python tools/materialize_workspace.py --workspace /absolute/path/salmon-ladder-run
```

This creates the original directory layout in a separate directory (six experiment directories, see the table below), resolves server-specific paths in executable copies and stream tasks, reconstructs manifest parquet tables, and records every transformed file. The repository snapshots and hash-checked frozen records remain unchanged. The workspace must not already contain an experiment; this avoids overwriting running work.

Place the datasets and checkpoints in the resulting layout, or make symlinks from that layout to your storage. `materialization.json` records the mapping. The archived files provide the exact original entry points; no training or evaluation is automatically started by materialization.

For the controls, the manifest-based feature builder regenerates bank files directly from images. For example:

```bash
python tools/encode_manifest.py --workspace /absolute/path/salmon-ladder-run --bank cub --model CLIP
python tools/encode_manifest.py --workspace /absolute/path/salmon-ladder-run --bank cub --model B14
python tools/encode_manifest.py --workspace /absolute/path/salmon-ladder-run --bank cub --model L14
```

Run the same three model commands for `cifar`, `dev`, and `openood`. CIFAR PNGs are materialized from its official Python archive. CUB paths refer to the extracted official archive. The builder checks stored RGB hashes, where available, before encoding. It writes all rows in the archived manifest order. It does not refit thresholds or use OOD labels to select a model.

After all bank feature/candidate files and checkpoint inputs exist, individual control jobs can be run from the materialized `reprise_controls_20260927`:

```bash
python src/run.py --task R_ninco_c00_rep0_r0_b256_warm15 --device cpu
CUDA_VISIBLE_DEVICES=0 python src/run.py --task H_cub_B14_s123
python src/report.py
```

For a multi-GPU sweep, the original `launch.sh` and `scheduler.py` are included. Their four-GPU resource guard reflects the original server and an earlier R5 pipeline. Review GPU availability and the workspace's resource policy before launching them elsewhere. Do not execute launchers directly inside the archived source tree.

## Historical benchmark reproduction

The original benchmark scripts expect caches generated by the original feature/TINS stages. `experiments/tins_reference/prepare.py` and `run.py` provide the initial TINS reproduction; `scripts/prepare_tins_clip.py`, `extract_dino.py`, `compute_dview.py`, `iter2_extract.py`, `test_eval/*.py`, and `iter2_extract_test.py` provide the feature/development stages. `scripts/run_tins_test.py` generates the five OpenOOD order seeds; `scripts/run_tins_extra.py` covers Four-OOD. The final analysis commands are listed in `EXPERIMENTS.md`.

The historical pipeline is not a one-command download of proprietary data. It expects the exact original image manifests, prototype list, benchmark lists, and checkpoint files. Missing cache errors should be resolved by the corresponding preparation stage, rather than substituting unrelated data. The lightweight verification performed for this repository release does not rerun the full GPU benchmark.

The initial TINS reproduction checks an actual pristine Git revision, unlike the later instrumented copy. Materialization therefore leaves its `repo/` absent. Before its preparation stage, clone the upstream repository there and check out `194759d716b27534bf2e8eeb0d71f5c4f0dabc40`. Do not substitute the hook revision from `third_party/tins`. The parent benchmark manifest is provided in `manifests/reference_manifest.jsonl.gz`; materialization records its original hash and recomputes its transport hash after changing only local path prefixes.

## Later phases (second export)

### Layout of a materialized workspace

| workspace directory | archive | contents |
|---|---|---|
| `vins_gonogo_20260925` | `experiments/legacy` overlaid with `experiments/r5_final` | original implementation, R5, final test |
| `reprise_controls_20260927` | `experiments/controls_456` | controls ④–⑥ |
| `reprise_lp_audit_20260927` | `experiments/lp_audit` | propagation audit; its `vendor/` directory is the copy of the frozen implementation that Phases 4–7 import |
| `reprise_p4_20260928` | `experiments/phase4` | Phase 4 |
| `reprise_p5_20261002` | `experiments/phase5_6` | Phase 5 and Phase 6 |
| `reprise_p7_20261003` | `experiments/phase7` | Phase 7 |
| `reprise_p10_20261010` | `experiments/phase10_11/phase10` | Phase 10 (third export, `provenance/server_snapshot_20261011.json`) |
| `reprise_p11_20261010` | `experiments/phase10_11/phase11` | Phase 11, the paper of October 2026 (third export) |

The scripts of one phase import the code of earlier phases through these directory names (`phase7/code/p7common.py` puts the Phase 5 and Phase 4 code on the import path; both import `reprise_lp_audit_20260927/vendor`; `phase10/code/p10common.py` and `phase11/code/p11common.py` import the Phase 5 code and read the Phase 3 feature space of `vins_gonogo_20260925`). Keep the names.

Materialization rebuilds the image tables of these phases from `manifests/phase*_*.csv.gz` with the workspace path in place of the token `<DATA_HOME>`: `banks/imagenet_pool.parquet`, `banks/U2_imagenet_o.parquet`, `banks/exp4/aug_table.parquet` and `banks/feature_table.parquet` for Phase 4, `banks/imagenet_pool.parquet` and `banks/feature_table.parquet` for Phase 5, and `banks/<data set>/images.parquet` for Phase 7. `materialization.json` lists each rebuilt table with the hash of the server table it corresponds to. With the recorded versions of pandas and pyarrow, a table rebuilt with the server home directory as workspace has the same bytes as the server table. The class splits (`banks/U1/split?.json`, `banks/U4/split?.json`, `banks/<data set>/split?.parquet`) are archived as they were, so a re-run uses the same classes, labelled images and streams without calling the build scripts again.

### Data

Place or link the data below the workspace at the paths the code expects:

| path below the workspace | data | used by |
|---|---|---|
| `datasets/openood_official/images_largescale/imagenet_1k/` | ImageNet-1K in the OpenOOD layout (training images for the labelled images and for U1 and U4; validation images only in the final test) | all phases |
| `datasets/ood_data/official_imagenet_o/extracted/` | [ImageNet-O](https://github.com/hendrycks/natural-adv-examples) | U2 |
| `reprise_controls_20260927/data/CUB_200_2011/` | CUB-200-2011 | Phase 7 D, final test |
| `datasets/cifar100/cifar-100-python/` | CIFAR-100, official Python archive | Phase 7 D |
| `datasets/extra_ood/imagenet-r/`, `datasets/extra_ood/imagenet-sketch/sketch/` | [ImageNet-R](https://github.com/hendrycks/imagenet-r), [ImageNet-Sketch](https://github.com/HaohanWang/ImageNet-Sketch) | Phase 7 D |
| `datasets/places365_val256/` | [Places365](http://places2.csail.mit.edu/download.html) validation images, 256 px | Phase 7 D |

`build7.py` writes the CIFAR-100 images as PNG files from the official archive. The image tables contain file names and, where they were computed, SHA-256 hashes of the image files, so a different copy of a data set is detected. The wildlife data set of U3 is private and is not needed for any number of the paper.

### Model weights

None is redistributed. The paths are those of `phase7/code/extract7.py` and `phase5_6/code/extract6.py` after materialization.

| model | source | used for |
|---|---|---|
| DINOv2 ViT-B/14, ViT-L/14 (and ViT-S/14, ViT-g/14 in Phase 7 B) | [facebookresearch/dinov2](https://github.com/facebookresearch/dinov2) checkpoints, loaded through the hub source archived in `third_party/dinov2` | the two views of the method |
| CLIP ViT-B/16 (and RN50, ViT-L/14 in Phase 7 B) | [OpenAI CLIP](https://github.com/openai/CLIP); `openai/clip-vit-large-patch14` on the Hugging Face hub | candidate classes, MCM, TINS, NegLabel, AdaNeg, TANL |
| DINOv3 ViT-S/16, ViT-S+/16, ViT-B/16, ViT-L/16 (LVD-1689M) | `timm/vit_*_patch16_dinov3.lvd1689m` on the Hugging Face hub; released under the DINOv3 License, which has to be accepted by the user | Phase 6, Phase 7 B; ViT-L/16 is one view of the paper (Phases 10–11) |
| DINO ViT-B/16, MAE ViT-B/16 | `timm/vit_base_patch16_224.dino`, `timm/vit_base_patch16_224.mae` | Phase 7 B |
| SigLIP 2 ViT-L/16 (256 px) | `google/siglip2-large-patch16-256` | Phase 7 B |

The Phase 7 scripts set `HF_HUB_OFFLINE=1`: they read cached weights and download nothing. `probe7b.py` is the only script of that phase that fetches weights (three files, listed with their hashes in `experiments/phase7/results/weights_p7b.json`).

### Order of execution

Each phase is run from the `code/` directory of its workspace directory (the propagation audit from its top directory). The launchers (`*.sh`) record the order that was used on the server, including its four-GPU scheduling; [EXPERIMENTS.md](EXPERIMENTS.md) lists the steps and their outputs. In short:

```text
Phase 4   build_banks.py -> extract.py (stageA.sh) -> run_phase4.sh
            (exp4_design.py, extract_aug.py, tins_bank.py, evaluate.py, analyze_eval.py, vlm_tta.py,
             analyze_baselines.py, operating.py, exp4_run.py, exp8.py, analyze_exp4.py, bounded_p4.py, resources.py)
Phase 5   run_final.sh   (u4_build.py -> u4_extract.py -> u4_tins.py -> final5.py --part u4|openood|fourood -> final_analysis.py)
Phase 6   run_d3.sh      (probe6.py -> extract6.py -> merge6.py -> ncm6.py -> dev5.py -> d3_analysis.py)
Phase 7   run_stage1.sh  (u1x_table.py, extract7.py, build7.py, text7.py)
          run_engine.sh <tag> "<gpus>" <workers per gpu> <steps>   (dev7.py, x4.py, run7.py --exp ...)
          run_ds.sh, an_*.py, verify7.py
Phase 10  orchestrate10.sh (extract10.py --part openood|fourood -> vlm10.py --text-only, vlm10.py --part ... ->
                            final10.py --part ... -> analysis10.py)
Phase 11  orchestrate11.sh, orchestrate11b.sh ... orchestrate11e.sh (the sweeps), then
          orchestrate11f.sh (final11.py --part openood|fourood --worker w --nworkers n -> final11.py --aggregate)
```

Phases 10 and 11 need the Phase 3 stream files and feature caches of `vins_gonogo_20260925` (the final test of the online variant, `scripts/p3_pipeline.sh`), the Phase 5 code, and the cached DINOv3 ViT-B/16 and ViT-L/16 weights (`HF_HUB_OFFLINE=1`). The final run alone (`orchestrate11f.sh`) takes a few minutes on four RTX 2080 Ti once the features exist.

U3 cannot be re-run: its images are private and its image table is not archived. In a materialized workspace, skip `build_banks.py` (the image tables and class splits come from the archive), pass `--banks U1,U2` to `evaluate.py`, and leave the `U3:` jobs out of the job lists of the launchers and of `vlm_tta.py`. `banks/feature_table.parquet` then has 177,699 rows, and `extract.py` writes feature arrays with these rows; on the server the table and the arrays had 18,535 further rows for U3 at the end.

The selection steps (`dev1_tune.py` and `select_lock.py` in Phase 4, `dev5.py` and `make_lock.py` in Phase 5, `dev7.py`, `an_within.py` and `make_lock7.py` in Phase 7) need the feature caches of the development splits, which the R5 stage of the original directory produces (`scripts/r5_*.py`). The archived locks make it possible to skip them: the evaluation scripts read the lock, check it against its hash file, and apply it.

### Checking a reproduction

- `python tools/make_results_tables.py --check` recomputes every table of [RESULTS.md](RESULTS.md) from the archived result files.
- Materialization places the archived outputs (results, logs, failure records) under `<workspace>/archived_results/<directory>/` and not in the executable directories, because several scripts skip a task whose output file exists. A re-run writes the same file names into the executable directory.
- `python tools/compare_rerun.py --workspace <workspace>` compares the Phase 4 metric tables that a re-run wrote (`results/eval/<stream>.parquet`) with the archived tables of the same streams and fails if AUROC or FPR95 of any detector differs by more than 0.05 percentage points. With `--stored <directory>` it also compares score files with originals, for whoever holds them.
- A re-run on a GPU is not bit-identical, even on the same machine. `provenance/rerun_check_20261004.json` records a check on the experiment server: a workspace was materialized from this archive, and `evaluate.py` (one stream of U1, one of U2) and `run7.py` (one stream of U1 and of U4, both views) were run in it with the feature arrays and TINS scores of the original runs. FPR95 was identical in all 126 rows of the two metric tables, and AUROC differed by less than 0.00002 percentage points. Of 226 score arrays, 150 were identical (among them the static and the memory read-outs), 4 hold run metadata and were not compared, and 72 differed. The 72 are the read-outs of the propagation, which enter the score of the method: the rank read-outs differed for at most 16 of 23,000 images in a stream (largest difference of a log score 0.0074), and the continuous scores by at most 3e-5. `experiments/phase7/results/test7_hades.json` records the same behaviour for the engine of Phase 7 on the stored Phase 4 and Phase 5 scores. The check does not cover feature extraction or the TINS runs, and larger differences on other GPUs or library versions are possible.
- The registered order cannot be reproduced after the fact: a re-run verifies the numbers, while the order of decisions is documented by the archived records.

### Not archived

Image files, feature arrays, model weights, per-image score files (about 40 GB) and per-worker run logs are not in the repository; `provenance/server_snapshot_20261004.json` lists what was left out and why, and `provenance/server_snapshot_20261011.json` does the same for Phases 10 and 11 (feature arrays and per-image score files of about 6 GB; the per-stream metric tables, the aggregated results and the analysis logs are archived). The narrative working notes of the propagation audit (four Markdown files that its README mentions) are not archived either; the tables they were written from are in `experiments/lp_audit/reports/`.
