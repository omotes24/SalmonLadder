# Reproduction guide

## Environment

The complete recorded package versions and Python version are in `provenance/server_snapshot.json`. The original GPUs were NVIDIA RTX 2080 Ti (11 GB). Core matching and propagation tests run on CPU. TINS inversion and image feature extraction require a CUDA-capable GPU in this implementation.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements-reproduction.txt
```

The requirements distributed inside the upstream TINS repository describe its own environment. Use the recorded REPRISE environment above for this snapshot.

## Source integrity

`python tools/verify_snapshot.py` checks the exported server files against SHA-256 receipts. It also checks the original `iter3_eval.py` frozen hash and the controls' source manifest. The snapshot export time identifies when these files were copied; it does not establish when a research decision was preregistered. Some historical JSON registration timestamps have a previously noted UTC inconsistency. Keep the original records and use version history and source hashes for traceability.

The original TINS adapter was verified against its upstream scoring function on 512 old-development images: bitwise equality, maximum absolute difference 0. See `experiments/controls_456/tests/tins_equivalence.json`. This verifies that adapter on that sample, not every dataset or GPU configuration. Submission-time CPU check results are recorded separately under `provenance/`.

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
python tools/materialize_workspace.py --workspace /absolute/path/reprise-run
```

This creates the original directory layout in a separate directory, resolves server-specific paths in executable copies and stream tasks, reconstructs manifest parquet tables, and records every transformed file. The repository snapshots and hash-checked frozen records remain unchanged. The workspace must not already contain an experiment; this avoids overwriting running work.

Place the datasets and checkpoints in the resulting layout, or make symlinks from that layout to your storage. `materialization.json` records the mapping. The archived files provide the exact original entry points; no training or evaluation is automatically started by materialization.

For the controls, the manifest-based feature builder regenerates bank files directly from images. For example:

```bash
python tools/encode_manifest.py --workspace /absolute/path/reprise-run --bank cub --model CLIP
python tools/encode_manifest.py --workspace /absolute/path/reprise-run --bank cub --model B14
python tools/encode_manifest.py --workspace /absolute/path/reprise-run --bank cub --model L14
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
