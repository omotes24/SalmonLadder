# Paper

`Salmon_Ladder.pdf` is the paper in Japanese, version 22, titled "Salmon Ladder: Conformal Label Propagation for OOD
Detection" (main text and references on nine pages in two columns, appendix in one column after `\clearpage`): Salmon
Ladder as conformal label propagation for transductive OOD detection with self-labelled seeds, evaluated on OpenOOD v1.5
ImageNet-1K, Four-OOD, ImageNet-200 and CIFAR-100. In version 22 the main read-out is sigma_ref (seeds chosen and read
against labelled reference images instead of two cross-fitting folds), which the registered interval rule of Phase 17
made the main configuration on dev2 before any benchmark value existed (`experiments/phase17/prereg_p17.json`,
`prereg_p17_outcome.json`); the cross-fitted sigma of versions 18–21 is reported alongside. Version 22 also adds the
registered descriptive analyses of Phase 17 (the decomposition u = sqrt(delta) x probability, the separation bound
(Corollary 2) and Corollary A1 measured on the recurrence grid, covariate-shifted ID images from ImageNet-V2 and
ImageNet-R), the agreement between the development splits and the benchmark, the disclosure of the two deviations from
registered rules, and a recommended configuration. Versions 19–21 added the registered Phase 16 variant (Appendix A.18–A.19),
shortened the title, and set the appendix in one column.

`src/` holds the LaTeX source (MIRU2025 class), the data files every number is generated from (`src/data/`), the
generators (`make_p18.py` for the cross-fitted sigma and the sections that were not rerun, `make_p19.py` for the Phase 16
variant, `derive_p17.py` and `make_p22.py` for sigma_ref and the Phase 17 analyses, `make_fig22*.py`, `order_refs22.py`) and
the checker `verify_p22.py`, which recomputes the Phase 17 numbers of the text from the per-task result files in
`experiments/phase17/results` and checks the stated claims, the abstract length and the page budget. Build: `cd src &&
python3 make_p18.py && python3 make_p19.py && python3 derive_p17.py && python3 make_p22.py && python3 make_fig22rec.py &&
python3 make_fig22.py && cat preamble_p22.tex body_p22.tex > main.tex && python3 order_refs22.py && sh build22.sh &&
python3 verify_p22.py` (pLaTeX + dvipdfmx). The Phase 12–17 numbers come from `experiments/phase12/` … `experiments/phase17/`.
The values of other methods that the paper quotes from their publications are listed in
`src/data/published.json` and are not part of the results of this repository.

This directory names the authors. `tools/make_submission_zip.py` leaves it out of the anonymized archive.
