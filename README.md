# REPRISE

Research code for detecting recurring out-of-distribution images in an online stream.

**Authors:** d-hacks B3 omote / 親: zackeyさん.

REPRISE combines a fixed visual prototype score, a memory of previously observed images, and label propagation. The reference experiment configuration uses frozen DINOv2 ViT-B/14 and ViT-L/14 features, with CLIP ViT-B/16 selecting five candidate ID classes. Each ID class supplies 12 support images and four calibration images. An optional TINS score multiplies the visual score. Larger scores indicate ID.

This repository archives the code actually used on the experiment server, together with frozen configurations, split manifests, stream plans, dependency versions, and verification records. The original implementation uses the internal name **CLAVIS-M3**. Development scripts retain their historical names. The additional controls were still running when this code snapshot was prepared; this repository does not represent their results as final.

## Repository map

| Location | Contents |
|---|---|
| [`experiments/legacy/`](experiments/legacy/) | Original REPRISE implementation, feature extraction, OpenOOD/Four-OOD evaluation, development history, R5 controls, and tests |
| [`experiments/legacy/scripts/iter3_eval.py`](experiments/legacy/scripts/iter3_eval.py) | Frozen prototype, memory-admission, and propagation implementation |
| [`experiments/legacy/iter3/frozen_m3.json`](experiments/legacy/iter3/frozen_m3.json) | Original final configuration and source hash |
| [`experiments/controls_456/src/`](experiments/controls_456/src/) | Online implementation, controlled recurrence, prevalence/retention studies, new-dataset evaluation, and reporting |
| [`experiments/controls_456/protocol.json`](experiments/controls_456/protocol.json) | Fixed design for the 1,210 additional conditions |
| [`experiments/tins_reference/`](experiments/tins_reference/) | Initial TINS reproduction and ImageNet prototype-selection code |
| [`third_party/`](third_party/) | TINS, its dependencies, and the LoCoOp comparison code, with attribution |
| [`manifests/`](manifests/) | Image identifiers, splits, and portable image paths; no image pixels |
| [`provenance/`](provenance/) | Server snapshot hashes, exact dependency versions, and submission checks |
| [`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md) | Environment, data, execution order, and expected outputs |

## Quick start: CPU verification

Python **3.12.3**, PyTorch **2.5.1+cu121**, and torchvision **0.20.1+cu121** were used on the server. The CPU verification does not require datasets, checkpoints, or a GPU.

```bash
python -m venv .venv
source .venv/bin/activate
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements-core.txt
python tools/verify_snapshot.py
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 CUDA_VISIBLE_DEVICES='' \
  python -m pytest -q experiments/controls_456/src/test_controls.py
PYTHONPATH=experiments/legacy python -m pytest -q experiments/legacy/tests/test_r5.py
python examples/feature_stream.py
```

The example uses synthetic normalized features to demonstrate the actual online implementation. It is an API smoke test, **not an OOD benchmark result**.

For GPU experiments, install the CUDA 12.1 PyTorch wheels instead, then the recorded full environment from `requirements-reproduction.txt`. Read the [reproduction guide](docs/REPRODUCIBILITY.md) before using the archived launchers: those launchers preserve the original machine paths and scheduling policy. A workspace-materialization tool creates a separate path-adapted copy without editing the archived scientific code.

## Evaluation conventions

- The original OpenOOD and Four-OOD test sets had been used during method development. A frozen configuration does not make those data newly independent.
- CUB-200-2011 and CIFAR-100 were confirmed unused in REPRISE method development before the new splits were fixed. This does not imply that a pretrained feature model has never seen related or overlapping images.
- The controls compare static prototypes, memory only, propagation only, and full REPRISE with `s0=1`. Single-encoder conditions use image prototypes for candidate selection. TINS-composed scores are identified separately.
- History-based memory scores use previous batches. Label propagation also includes the current batch. Independent-query controls restore the pre-query state after each query.
- The original evaluation uses the TINS upstream FPR95 convention. The new controls use a threshold accepting at least 95% of ID samples, accepting ties and recording achieved TPR and tie counts. Do not mix these conventions silently.
- Calibration ranks and their products are detection scores. Adaptive reuse of calibration data is not presented here as an established distribution-free guarantee.

See [the experiment map](docs/EXPERIMENTS.md) for commands and the distinction between development, historical tests, and the new controls. Data and pretrained model files must be obtained from their respective providers. They are not redistributed here.

## Attribution

See [THIRD_PARTY.md](THIRD_PARTY.md) for upstream revisions and licenses. The repository has no assigned paper venue, DOI, or final-results release. Cite the exact Git commit used for a submission; `CITATION.cff` identifies this code repository.
