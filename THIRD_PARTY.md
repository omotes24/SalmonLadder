# Third-party code and attribution

The code and experiment records of Salmon Ladder are distinguished from the following upstream components. No new blanket license is applied to these components by this repository.

| Component | Source / revision | Use |
|---|---|---|
| TINS | https://github.com/zxk1212/tins, upstream `194759d716b27534bf2e8eeb0d71f5c4f0dabc40`, server hook revision `dca37b0194a0d4ca4ea19fcc2466a92a87db6684` | Test-time negative-text adaptation; original reproduction and comparison |
| OpenAI CLIP | https://github.com/openai/CLIP; copies vendored by TINS | CLIP encoder/tokenizer; MIT license in `third_party/OPENAI_CLIP_LICENSE` |
| DINOv2 | https://github.com/facebookresearch/dinov2; exact server hub-source file hashes in `provenance/dinov2_snapshot.json` | Frozen visual encoders; upstream license retained in `third_party/dinov2/LICENSE` |
| LoCoOp | https://github.com/AtsuMiyai/LoCoOp, `a311b3917d1e6e1e02050aefba4fd09746b65dd4` | Optional local/global CLIP comparison; upstream license retained with the source |
| OpenOOD-VLM | https://github.com/YBZh/OpenOOD-VLM, `c6fef2f5ed890df2f59f9c6c3f9ae14fb5a72ab9` | Official AdaNeg and TANL post-processors, run as baselines in Phase 4; MIT license in `third_party/OPENOOD_VLM_LICENSE` |
| NegLabel | https://github.com/XueJiang16/NegLabel, `3253db684075b2db47844676eccdfda40d67a573` | WordNet word lists from which the negative labels of NegLabel, AdaNeg and TANL are mined; Apache-2.0 license in `third_party/NEGLABEL_LICENSE` |

The TINS export contains its code, text resources, tokenizer vocabulary, and small class-list metadata. It does not include dataset images, checkpoints, or results. A top-level TINS license was not present in the server checkout at export; consult the upstream repository for its applicable terms. Included upstream file headers and dataset-specific notices are preserved.

The additional seven-line TINS hook patch is under `experiments/legacy/patches/`. The hooks record internal quantities and are inactive when no hook is supplied. `experiments/controls_456/src/vendor/` retains the exact dependency copy used by the control experiments, even where it duplicates the main third-party snapshot.

`experiments/phase4/code/vlm_tta/` is the subset of OpenOOD-VLM that the two post-processors need, with the revisions recorded in `SOURCE_COMMIT_OpenOOD-VLM` and `SOURCE_COMMIT_NegLabel`. Eleven files are identical to upstream, including the copy of the OpenAI CLIP model code that OpenOOD-VLM vendors. Three package `__init__.py` files are empty so that only these modules are imported, and `openood/utils/comm.py` is a two-line stub in place of the upstream helpers for distributed training. `openood/postprocessors/adaneg_chunked.py` was written for this project: it subclasses the official AdaNeg post-processor and evaluates its sample-adaptive term in chunks of 32 images, because the unmodified code exceeds 11 GB of GPU memory at a batch size of 256; the failed unmodified run is kept in `experiments/phase4/failures/run2_vlm_oom`. The 27 files in `data/txtfiles/` are identical to `txtfiles/` of the NegLabel repository.

The later phases also use pretrained models that are not redistributed: DINOv3 (timm weights on the Hugging Face hub, DINOv3 License), DINO, MAE, SigLIP 2 and further CLIP and DINOv2 variants. [docs/REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md) lists their sources.

Original upstream README files may refer to images or external assets not distributed in this code archive. Checkpoints and datasets remain governed by their providers' terms. A license for the code of Salmon Ladder itself has not been selected in this submission snapshot.
