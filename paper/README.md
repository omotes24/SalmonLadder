# Paper

`Salmon_Ladder.pdf` is the paper in Japanese, version 18 (main text and references on nine pages, appendix after
`\clearpage`): Salmon Ladder as conformal label propagation for transductive OOD detection with self-labelled seeds,
evaluated on OpenOOD v1.5 ImageNet-1K, Four-OOD, ImageNet-200 and CIFAR-100.

`src/` holds the LaTeX source (MIRU2025 class), the data files every number is generated from (`src/data/`), the
generators (`make_p18.py`, `make_fig18*.py`, `order_refs18.py`) and the checker (`verify_p18.py`, which compares
every literal number and claim of the text with the data files, the build log and the page budget). Build:
`cd src && ./build.sh` (pLaTeX + dvipdfmx). The Phase 12–15 numbers come from `experiments/phase12/` …
`experiments/phase15/` (Phase 15 results, logs and the registration with its outcome are archived in full under
`experiments/phase15/`). The values of other methods that the paper quotes from their publications are listed in
`src/data/published.json` and are not part of the results of this repository.

This directory names the authors. `tools/make_submission_zip.py` leaves it out of the anonymized archive.
