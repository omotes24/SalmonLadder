# Paper

`Salmon_Ladder.pdf` is the paper in Japanese, version 20, titled "Salmon Ladder: Conformal Label Propagation for OOD
Detection" (main text and references on nine pages in two columns, appendix in one column after `\clearpage`): Salmon
Ladder as conformal label propagation for transductive OOD detection with self-labelled seeds, evaluated on OpenOOD v1.5
ImageNet-1K, Four-OOD, ImageNet-200 and CIFAR-100. Version 19 kept the main text of version 18 and added, in Appendix
A.13–A.14, the registered Phase 16 variant (one quantity: the probability that a random walk reaches a support image before
a self-labelled seed) with its theorems and all its registered results, and the disclosed deviation from its registered
adoption rule (`experiments/phase16/prereg_p16_amendments.json`). Version 20 shortens the title and sets the appendix in
one column; the body text and every number are those of version 19.

`src/` holds the LaTeX source (MIRU2025 class), the data files every number is generated from (`src/data/`), the
generators (`make_p18.py`, `make_fig18*.py`, `make_p19.py`, `make_fig19toy.py`, `order_refs19.py`) and the checker
(`verify_p19.py`, which compares every literal number and claim of the text with the data files, the build log and the
page budget). Build: `cd src && python3 make_p18.py && python3 make_fig19toy.py && python3 make_p19.py && cat preamble_p19.tex
body_p19.tex > main.tex && python3 order_refs19.py && ./build.sh && python3 verify_p19.py` (pLaTeX + dvipdfmx). The Phase 12–15 numbers come from `experiments/phase12/` …
`experiments/phase15/` (Phase 15 results, logs and the registration with its outcome are archived in full under
`experiments/phase15/`; Phase 16 under `experiments/phase16/`). The values of other methods that the paper quotes from their publications are listed in
`src/data/published.json` and are not part of the results of this repository.

This directory names the authors. `tools/make_submission_zip.py` leaves it out of the anonymized archive.
