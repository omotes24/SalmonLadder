# Third-party code and attribution

REPRISE-specific code and experiment records are distinguished from the following upstream components. No new blanket license is applied to these components by this repository.

| Component | Source / revision | Use |
|---|---|---|
| TINS | https://github.com/zxk1212/tins, upstream `194759d716b27534bf2e8eeb0d71f5c4f0dabc40`, server hook revision `dca37b0194a0d4ca4ea19fcc2466a92a87db6684` | Test-time negative-text adaptation; original reproduction and comparison |
| OpenAI CLIP | https://github.com/openai/CLIP; copies vendored by TINS | CLIP encoder/tokenizer; MIT license in `third_party/OPENAI_CLIP_LICENSE` |
| DINOv2 | https://github.com/facebookresearch/dinov2; exact server hub-source file hashes in `provenance/dinov2_snapshot.json` | Frozen visual encoders; upstream license retained in `third_party/dinov2/LICENSE` |
| LoCoOp | https://github.com/AtsuMiyai/LoCoOp, `a311b3917d1e6e1e02050aefba4fd09746b65dd4` | Optional local/global CLIP comparison; upstream license retained with the source |

The TINS export contains its code, text resources, tokenizer vocabulary, and small class-list metadata. It does not include dataset images, checkpoints, or results. A top-level TINS license was not present in the server checkout at export; consult the upstream repository for its applicable terms. Included upstream file headers and dataset-specific notices are preserved.

The additional seven-line TINS hook patch is under `experiments/legacy/patches/`. The hooks record internal quantities and are inactive when no hook is supplied. `experiments/controls_456/src/vendor/` retains the exact dependency copy used by the control experiments, even where it duplicates the main third-party snapshot.

Original upstream README files may refer to images or external assets not distributed in this code archive. Checkpoints and datasets remain governed by their providers' terms. A license for the REPRISE-specific code has not been selected in this submission snapshot.
