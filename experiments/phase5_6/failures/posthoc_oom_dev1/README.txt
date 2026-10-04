2026-10-03 (Phase 6, post hoc): worker 4 of the dev1 run (cfgs l14,d3L,d3mixB,d3mixL; two workers per GPU) stopped with
CUDA out of memory on GPU 0 (the other worker on that GPU held 9.1 GiB). Three d3mixL tasks were missing; run_posthoc.sh
re-ran them with a single worker on GPU 0 (logs/dev1_phc_g0.log). Completed tasks are unaffected (one file per task,
written atomically). No setting was changed.
