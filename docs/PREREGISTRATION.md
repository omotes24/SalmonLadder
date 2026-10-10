# Registrations, selection locks and the use of evaluation data

The claims of the paper depend on the order in which decisions were taken: which configuration was fixed before which data were scored. This file lists every record that fixes a decision, shows how the records are chained by hashes, and states where the order is weaker than the word "registered" suggests.

`python tools/verify_snapshot.py` checks the hashes. `python tools/make_registration_timeline.py --check` checks that the table below still follows from the archived records. In an anonymized export the hashes of archived files are those of the exported files, and hashes of files outside the archive are pseudonyms (`provenance/anonymized_export.json`); the chains are the same.

## Data and their role

| data | role | what happened |
|---|---|---|
| dev1: 900 ID classes of ImageNet-1K, 100 held-out classes, far set OpenImage-O (validation list) | exploration and every selection | scored without limit |
| dev2: a second split of the same construction (other held-out classes; the same far set) | confirmation | In loops 1–3 one logged look per frozen configuration. In R5 both splits were scored for the matched comparison and for 15 entrance configurations before the round-1 rule was written; from then on dev2 is a second development split on which a choice made on dev1 has to hold, not a held-out set. Phases 5 and 7 use it for a limited number of confirmations of a locked candidate. |
| OpenOOD v1.5 ImageNet-1K test split (SSB-hard, NINCO, iNaturalist, Textures, OpenImage-O) with ImageNet-1K validation images as ID | evaluation | six registered evaluations and several post hoc analyses (next but one section) |
| Four-OOD (iNaturalist, SUN, Places, Textures), CUB-200-2011 with the Semantic Shift Benchmark split | evaluation | Four-OOD in two post hoc analyses of earlier configurations (those of test uses 2 and 4), in the final test, and again in Phase 5; CUB in the final test and again in Phase 7 D |
| U1: five new class splits of ImageNet-1K; U2: ImageNet-O against all 1,000 classes | evaluation | built after the Phase 4 selection lock; scored in Phase 4 and read again by later descriptive analyses |
| U3: five-vs-five species split of a private wildlife data set | evaluation | registered and scored in Phase 4; image list not released; not reported in the paper |
| U4: five further class splits of ImageNet-1K from images never used before | evaluation | built after the Phase 5 selection lock; scored in Phase 5 and read again in Phase 7 |
| CUB-200-2011, CIFAR-100, ImageNet-R, ImageNet-Sketch, Places365 (Phase 7 D) | evaluation | frozen hyper-parameters; scored in Phase 7 |

A data set that has been scored once is called **not used for selection** from then on, not "unused": later analyses on it are descriptive, and a configuration chosen after looking at it would make it development data. The selection rules of every phase name the development splits only.

## Timeline

The first column is the modification time of the file on the experiment server (clock in UTC), recorded when the files were exported (`provenance/server_snapshot_20261004.json`), or the time written in a status-log line. File times are file-system metadata. The hashes are what fixes the order: a record that quotes the hash of another record cannot have been written before it.

<!-- BEGIN GENERATED: timeline -->
| server time (UTC) | file below `experiments/` | time written in the record | sha256 | content |
|:--|:--|:--|:--|:--|
| 2026-09-25 14:58:32 | `legacy/criteria.json` | 2026-09-25 14:58:26 | `4e4404647d85` | go / no-go criterion of the first project stage |
| 2026-09-26 01:29:13 | `r5_final/dev2/prereg_confirm.json` | 2026-09-26 01:29:01 | `ffa6153d9da4` | confirmation on a second development split, then one evaluation on the test split |
| 2026-09-26 01:31:51 | `legacy/test_eval/prereg_test.json` | 2026-09-26 01:31:43 | `6286847dba2d` | **test use 1**: registration |
| 2026-09-26 02:11:02 | `r5_final/test_eval/results/metrics.json` | – | – | test use 1: result |
| 2026-09-26 04:45:03 | `r5_final/iter/prereg_iter.json` | 2026-09-26 04:45:03 | `a3a20677636b` | improvement loop 1: stopping rule |
| 2026-09-26 04:57:51 | `r5_final/iter/frozen_c1.json` | 2026-09-26 04:57:51 | `40e27cb0e59d` | loop 1: frozen configuration |
| 2026-09-26 04:58:24 | `r5_final/iter/dev2_looks.jsonl` | 2026-09-26 04:58:24 | `62147f5cb8f6` | loop 1: look at the second development split (quotes both hashes) |
| 2026-09-26 05:02:50 | `legacy/test_eval/prereg_test2.json` | 2026-09-26 05:02:50 | `8bb64c4c2f69` | **test use 2**: registration |
| 2026-09-26 05:04:02 | `r5_final/test_eval/results_clavism/metrics.json` | – | – | test use 2: result |
| 2026-09-26 07:25:59 | `r5_final/analysis/test/*.json` (3 files) | – | – | post hoc: variants and ablations scored on the test split for the draft of that time |
| 2026-09-26 10:02:07 | `r5_final/analysis/extra/*.json` | – | – | post hoc: baselines on the test split, Four-OOD, other ID sets |
| 2026-09-26 10:34:29 | `r5_final/iter2/prereg_iter2.json` | 2026-09-26 10:45:00 † | `1cbdcdc4da5f` | improvement loop 2: stopping rule |
| 2026-09-26 11:14:48 | `r5_final/iter2/frozen_m2.json` | 2026-09-26 11:30:00 † | `3efbb961baa7` | loop 2: frozen configuration |
| 2026-09-26 11:17:33 | `r5_final/iter2/dev2_looks.jsonl` | 2026-09-26 11:17:33 | `7bd2d22bb7cd` | loop 2: look at the second development split |
| 2026-09-26 11:19:41 | `legacy/test_eval/prereg_test3.json` | 2026-09-26 11:45:00 † | `4b6cacce2ffd` | **test use 3**: registration |
| 2026-09-26 11:44:03 | `r5_final/test_eval/results_m2/metrics.json` | – | – | test use 3: result |
| 2026-09-26 14:31:45 | `legacy/iter3/prereg_iter3.json` | 2026-09-26 12:10:00 † | `05f7dee9e191` | improvement loop 3: stopping rule |
| 2026-09-26 15:01:33 | `legacy/iter3/frozen_m3.json` | 2026-09-27 00:30:00 † | `ddd948af299f` | loop 3: frozen configuration (`v4`, the first under the working name REPRISE) |
| 2026-09-26 15:02:51 | `legacy/test_eval/prereg_test4.json` | 2026-09-27 00:45:00 † | `bdaae9d968a6` | **test use 4**: registration (to run only if the look below passes) |
| 2026-09-26 15:12:43 | `r5_final/iter3/dev2_looks.jsonl` | 2026-09-26 15:12:43 | `b152f47225b3` | loop 3: look at the second development split |
| 2026-09-26 15:40:00 | `r5_final/test_eval/results_m3/metrics.json` | – | – | test use 4: result |
| 2026-09-26 16:31:03 | `r5_final/analysis/reprise/*.json` (3 files) | – | – | post hoc: analyses of the `v4` scores on the test split and on Four-OOD |
| 2026-09-26 18:08:51 | `legacy/r5/prereg_r5.json` | 2026-09-27 04:00:00 † | `1848e86347ed` | matched comparison (R5): protocol, data policy |
| 2026-09-26 19:54:09 | `legacy/r5/entrance/frozen.json` | – | `897b67cbbfe0` | entrance thresholds tuned on the first development split |
| 2026-09-26 19:58:39 | `r5_final/r5/entrance/eval/dev2_*.json` (30 files) | – | – | R5: 15 entrance configurations scored on the second development split |
| 2026-09-26 20:30:38 | `legacy/r5/prereg_phase2.json` | 2026-09-26 20:30:38 | `5de00e698901` | R5 round 1: candidates, selection and confirmation rule |
| 2026-09-27 02:35:25 | `r5_final/r5/prereg_phase2_round2.json` | 2026-09-27 02:35:25 | `bc49cc233b49` | R5 round 2: coordinate search on the first development split, confirmation on the second |
| 2026-09-27 03:42:49 | `r5_final/r5/round2/decision.json` | – | `d865c063cd1a` | round 2 decision: the frozen configuration (`v5`) |
| 2026-09-27 03:46:09 | `r5_final/r5/phase3/prereg_phase3.json` | 2026-09-27 03:46:09 | `0a2475160b4d` | **test use 5** (final test of the frozen method): registration with the hashes of the scripts |
| 2026-09-27 05:36:06 | `r5_final/r5/phase3/openood/*.json` (25 files) | – | – | test use 5: first result on the test split |
| 2026-09-27 11:13:36 | `lp_audit/preregistration.json` | 2026-09-27 11:13:36 | `5e8c2cc907f4` | propagation audit on the first development split: registration |
| 2026-09-27 11:20:44 | `lp_audit/amendment_01.json` | 2026-09-27 11:20:44 | `a08b13709d08` | propagation audit: amendment 1 (before any primary run) |
| 2026-09-27 13:21:29 | `lp_audit/reports/all_run_metrics.csv` | – | – | propagation audit: metrics of all runs |
| 2026-09-27 21:40:28 | `phase4/prereg_p4.json` | 2026-09-27 21:40:28 | `157647e16f69` | Phase 4: registration (families, 27 configurations each, selection rule, endpoints E1 and E2) |
| 2026-09-27 21:43:28 | `phase4/results/dev1_tune/*.parquet` (30 files) | – | – | Phase 4: first tuning result on the first development split |
| 2026-09-27 21:54:36 | `phase4/amendment_01.json` | – | `c1d4ae4faeb3` | Phase 4: amendment 1 (the first attempt to draw the class splits of U1 stopped; no image selected) |
| 2026-09-27 21:55:01 | `phase4/selection_lock.json` | 2026-09-27 21:55:01 | `e7c6a9dbf0b7` | Phase 4: **selection lock** (quotes the registration hash) |
| 2026-09-27 21:57:08 | `phase4/amendment_03.json` | – | `ac0f0cd1a2ec` | Phase 4: amendments 2 and 3 on the class splits (no image selected yet) |
| 2026-09-27 22:30:55 | `phase4/banks/U1/split?.json` (5 files) | – | – | Phase 4: class splits of U1 written |
| 2026-09-28 01:29:31 | `phase4/logs/phase4.status`: line `STAGE_A_START` | – | – | Phase 4: feature extraction of U1-U3 started |
| 2026-09-28 02:33:14 | `phase4/results/eval/*.parquet` (75 files) | – | – | Phase 4: first score on U1-U3 |
| 2026-10-02 08:18:31 | `phase5_6/code/prereg_p5.json` | 2026-10-02 08:18:31 | `6244100a78c1` | Phase 5: registration of the improvement loop |
| 2026-10-02 11:08:12 | `phase5_6/selection_lock_p5.json` | 2026-10-02 11:08:12 | `8fea3cbd7710` | Phase 5: **selection lock** of the two-sided variant |
| 2026-10-02 11:15:47 | `phase5_6/confirm_dev2_attempt1.json` | – | `f4009331122f` | Phase 5: confirmation on the second development split, attempt 1 |
| 2026-10-02 11:38:25 | `phase5_6/banks/build_info.json` | 2026-10-02 11:38:25 | `f80aa1c2a634` | Phase 5: evaluation set U4 built (quotes the lock hash) |
| 2026-10-02 12:41:14 | `phase5_6/results_final/*.json` (3 files) | – | – | Phase 5: first final result (U4; then **test use 6** and Four-OOD) |
| 2026-10-03 02:23:31 | `phase5_6/prereg_p6.json` | 2026-10-03 02:23:31 | `f796aac99d15` | Phase 6: registration of the DINOv3 swap (development splits only) |
| 2026-10-03 03:49:06 | `phase5_6/results/d3_summary_dev?.json` (2 files) | – | – | Phase 6: first result |
| 2026-10-03 12:38:42 | `phase7/prereg_p7.json` | 2026-10-03 12:38:42 | `345b97046f3e` | Phase 7: registration of the additional experiments A-E |
| 2026-10-03 13:07:32 | `phase7/results/a_dev1_selection.json` | – | – | Phase 7 A, first attempt: selection on the first development split |
| 2026-10-03 13:17:24 | `phase7/results/a_dev2_confirm_1.json` | – | – | Phase 7 A, first attempt: not confirmed on the second development split |
| 2026-10-03 13:18:16 | `phase7/amendment_01.json` | 2026-10-03 13:18:16 | `9832dd117ca7` | Phase 7: amendment 1 (nothing locked; delayed re-scoring analysis) |
| 2026-10-03 13:37:02 | `phase7/results/an_retro_u?r.json` (2 files) | – | – | Phase 7: delayed re-scoring of the frozen method on U1 and U4 |
| 2026-10-03 14:00:38 | `phase7/amendment_02.json` | 2026-10-03 14:00:38 | `dca23389fda7` | Phase 7: amendment 2 (case analysis, within-batch memory) |
| 2026-10-03 14:00:54 | `phase7/logs/engine_W.status`: line `start:` | – | – | Phase 7: one dispatcher started with the steps dev1, dev2, U1, U4 (within-batch read-outs) |
| 2026-10-03 14:01:17 | `phase7/results/an_cases_u?.json` (2 files) | – | – | Phase 7: case analysis of the frozen method on U1 and U4 |
| 2026-10-03 14:05:45 | `phase7/results/a2_dev1_selection.json` | – | – | Phase 7 A, within-batch memory: selection on the first development split |
| 2026-10-03 14:09:02 | `phase7/logs/engine_W.status`: line `dev2w done` | – | – | Phase 7: scores of the second development split finished; the dispatcher went on to U1 |
| 2026-10-03 14:09:24 | `phase7/results/a2_dev2_confirm.json` | – | – | Phase 7 A, within-batch memory: confirmed on the second development split |
| 2026-10-03 14:10:15 | `phase7/selection_lock_p7.json` | 2026-10-03 14:10:15 | `2ec669bbdeed` | Phase 7: **selection lock** of the extension |
| 2026-10-03 14:14:28 | `phase7/logs/engine_W.status`: line `u1w done` | – | – | Phase 7: within-batch scores on U1 finished |
| 2026-10-03 14:21:04 | `phase7/results/an_lock_eval.json` | – | – | Phase 7: first metrics of the locked extension on U1 and U4 |
| 2026-10-03 17:46:43 | `phase7/amendment_03.json` | 2026-10-03 17:46:43 | `09adb9ff0a68` | Phase 7: amendment 3 (three further encoders after permission to download) |
<!-- END GENERATED: timeline -->

† The time typed into the record differs from the file time by more than two minutes. These seven early records carry times entered by hand as round values. Three of them (`iter3/frozen_m3.json`, `test_eval/prereg_test4.json`, `r5/prereg_r5.json`) are about nine and a half hours ahead of the file time, which fits a local time (UTC+9) entered as UTC and rounded. The records are left unchanged because later records quote their hashes. What this means for the order:

- The registrations of test uses 3 and 4 carry typed times that are later than the file times of their results: for use 3 by 57 s, if the typed time is UTC (as a local time it would precede the frozen configuration that the registration quotes); for use 4 by 5 min as a local time and by 9 h as UTC. The typed times are therefore not the times at which these files were written. Their file times are earlier than the results (11:19:41 before 11:44:03; 15:02:51 before 15:40:00), and each result file quotes the hash of its registration, so each registration existed in its archived form when the result was written.
- `iter3/prereg_iter3.json` was last modified at 14:31:45, 2 h 21 min after the time typed in it. The archive shows that this content predates the frozen configuration (15:01:33) and the look at the second development split (15:12:43), which quote its hash. Whether an earlier version existed at the typed time cannot be shown.

## Test split

Up to Phase 7, a newly frozen configuration was evaluated on the OpenOOD test split six times, each under a registration written before the run (`docs/RESULTS.md`, Section 2, lists all six results). Phases 10 and 11, which produced the paper of October 2026, scored the same test streams without registration (see the end of this section).

| use | configuration | chain checked by `tools/verify_snapshot.py` |
|:-:|---|---|
| 1 | static rank of one view | result → registration |
| 2 | plus a memory | result → registration → frozen configuration and stopping rule; the look at dev2 quotes both |
| 3 | plus a second view, two-stage entrance | the same chain |
| 4 | plus prototypes and propagation (`v4`) | the same chain, and the registration quotes the hash of the frozen implementation `scripts/iter3_eval.py` |
| 5 | **the frozen method (`v5`)**, with its comparisons, Four-OOD and CUB | registration → round-2 registration and decision; hashes of all 14 scripts that read test data |
| 6 | the two-sided variant of Phase 5 (not adopted); the frozen method recomputed in the same run | lock → registration; the evaluation set U4 quotes the lock |

These six are not the only times test data were scored:

- After uses 2 and 4, analysis scripts computed further variants, ablations, baselines and significance tests on the test scores for the drafts of that time (`experiments/r5_final/analysis/test`, `analysis/extra`, `analysis/reprise`; rows "post hoc" in the timeline). The registration of loop 2 discloses that these results were known.
- The controls ④ and ⑤ (`experiments/controls_456`) are diagnostics of the earlier configuration `v4` on test images of NINCO and SSB-hard.
- Use 5 itself scored the comparison methods and ablations listed in its registration.

The registrations of uses 1–4 each state that nothing is changed after the test. The rule of loop 1 states that the loop stops after the test; the rules of loops 2 and 3 state that the test split is evaluated at most once more, and loop 3 calls itself the final loop; the round-1 rule of R5 allows at most one modification. Development continued after each of them: loops 2 and 3, R5 with a second round and a fifth use of the test split, and Phases 4–7 with a sixth. These statements were not kept.

The test split is therefore **not independent of the design of the method**: its results were known when every later configuration was designed, even though each selection rule refers to the development splits only. This is the reason Phase 4 exists. The numbers of use 5 were reported in the earlier versions of the paper as results on a public benchmark under a frozen configuration, and the claim that the entrance and the memory add to propagation rests on U1 and U2.

**Phases 10 and 11 (October 2026, the paper).** No registration was written, and the test streams of use 5 were scored repeatedly:

- Phase 10 scored the online read-out with the DINOv3 views (every pairing of the four encoders, three read-out variants, six base detectors) and the base detectors themselves on the OpenOOD and Four-OOD streams, and a configuration was chosen from that table (`experiments/phase10_11/phase10/results/p10_table.txt`).
- Phase 11 scored the post-stream read-out on the same streams in five sweeps (`results/p11_table.txt`, `sweep/p11b_table.txt`, `sweep_c/p11c_table.txt`, `sweep_d/p11d_table.txt`, `sweep_e/p11e_table.txt`: graph sizes, mutual graphs, `lambda`, the level `q` and the construction of the seeds, a nearest-seed term, smoothing, term weights, view sets, base detectors) and then ran the chosen configuration with its ablations once more (`results/final/`). The views, `q` and the base detector of the paper were chosen on these tables; `k_g` and `lambda` are the values of the frozen configuration `v5`.

The paper states that its configuration was selected on the benchmark and that scores are assigned after the whole stream has arrived. The result tables of these phases (`docs/RESULTS.md`, Section 8) are therefore results of a configuration selected on the data it is reported on, not a test on unused data; no development split and no unused set was scored in these phases, and nothing in them was registered. The generated timeline above ends with Phase 7; the server file times of Phases 10 and 11 are in `provenance/server_snapshot_20261011.json`.

Two scripts of use 5 differ from the hashes in the registration. Both changes are recorded in `experiments/r5_final/r5/phase3/deviations.json` with the old and the new hash: a wrong cache name that made one stream builder crash before any score existed, and a wrong prefix that left one comparison metric empty (recomputed from saved scores; no model was re-run). The verification reports them as `documented_deviations`.

## What each phase selected, and on what

**Development loops 1–3 and R5.** Each loop explored on dev1 without limit, froze one configuration, looked at dev2 once, and then evaluated the test split once. R5 compared the method with baselines under matched conditions on both development splits, tuned the entrance thresholds and six further constants on dev1 in two rounds, required the result to hold on dev2, and fixed `v5`.

**Propagation audit** (`experiments/lp_audit`). An audit of the comparison between Salmon Ladder and label propagation, on dev1 only. Its registration states that results on dev1 are exploratory. Cost pilots preceded the registration (`reports/pilot_*.json`, stage `cost_pilot_not_selection`). Several scripts changed after the registration. The changes made before the primary runs are recorded with hashes (amendment 1, seven minutes after the registration, and the execution manifest written at 11:26:25 with the note "before primary results"); a fix after a failed run is recorded in amendment 3; report scripts changed after the results were in. The final state of all sources is fixed by `provenance/current_source_manifest.json` (12:41:11). The verification checks the final manifest and reports which sources differ from each earlier record.

**Phase 4.** Registration (21:40:28) → tuning of every read-out family on dev1 with the same 27 graph configurations (21:43–21:49) → selection lock (21:55:01) → class splits of U1 written (22:30:55) → features (from 01:29 on the next day) → scores (from 02:33). The first attempt to draw the class splits was started before the lock was written and stopped because the registered rule could not produce five disjoint splits (amendment 1, 21:54:36); amendments 2 and 3 followed at 21:57. The amendments state that only the WordNet structure had been read and that no image had been selected. That U1–U3 were built after the lock rests on these file times and statements; unlike U4, the sets do not quote the hash of the lock. Registered endpoints and decision rule: Salmon Ladder is the main method if the upper limit of the 95% interval of the FPR95 difference to the strongest propagation-only read-out is below 0, standalone (E1) and with TINS (E2), on U1. Outcome: E1 −3.16 [−4.15, −2.16], E2 −3.22 [−4.21, −2.24]. One deviation: the recurrence intervention used 64 common ID queries instead of the registered 128 (an error in the design code).

**Phase 5.** A search on dev1 for a variant with a large gain, with a registered margin, one locked candidate, at most three confirmations on dev2, and one final evaluation on a new set U4 and on the test split (use 6). The locked two-sided variant met the margin on dev2 and on U4 and did not carry over to OpenOOD (far-OOD FPR95 +1.22 with TINS). It is reported and not adopted. Phase 5 was motivated by the Phase 4 results, so its re-reading of U1 and U2 is descriptive.

**Phase 6.** A swap of the two encoders for DINOv3, registered before the weights were obtained, on dev1 and dev2 only. No evaluation set was read.

**Phase 7.** Five additional experiments registered together.

- A, first appearances. The first registered attempt (one-sided memory without within-batch members) was selected on dev1 and **not confirmed** on dev2; nothing was locked and the planned evaluation was not run (amendment 1). Amendment 2 registered the within-batch memory with ten candidates, a selection rule on dev1 and one confirmation on dev2. One dispatcher computed the read-outs of all ten candidates on dev1, dev2, U1 and U4 in sequence: the scoring of U1 began at 14:09:02, when dev2 was finished, 73 s before the lock was written (14:10:15). The first metric of the extension on U1 and U4 was computed at 14:21:04, after the lock; the lock itself says "before any metric". Before the selection, U1, U2 and U4 had been scored with the frozen method, the delayed re-scoring analysis of amendment 1 had been read on U1 and U4 (13:37), and the case analysis of amendment 2, which splits the unknown images by the evidence available at arrival, had been computed for the frozen method on U1 and U4 (14:01). The evaluation of the extension on these sets is therefore an evaluation of a locked configuration on data **not used for selection**; it is not a test on unused data. Registered criteria on U1: AUROC difference with lower limit above 0 and FPR95 difference with upper limit below 0; both met (+0.12 [+0.08, +0.15], −0.77 [−0.86, −0.69]). On U2 the extension does not differ from the frozen method (`docs/RESULTS.md`, Section 1.1).
- B–E (encoders, separation of memory and propagation, data sets, stream conditions) apply frozen hyper-parameters and select nothing. They are descriptive. Amendment 3 added three encoders after the weights could be obtained.

**Phases 10 and 11.** Selection on the test streams, as stated above; nothing registered. The code of the final run is archived with the hashes computed on the server at export (`provenance/server_snapshot_20261011.json`); no hash was taken before the run.

## The code that produced the results

The archive holds every script as it was on the server when it was exported (an anonymized export replaces names and paths in them). Whether a script is byte-identical to the version that produced a given result is recorded only where hashes were taken at the time:

- final test: the registration holds the hashes of the 14 scripts that read test data; 12 match and 2 are the documented deviations;
- Phase 7: the lock holds the hashes of six scripts. `engine7.py`, which computes every score, and three others match. `run7.py` and `an7.py` changed after the lock (task lists and a helper that reports the locked read-out in B, D and E). `engine7.py` itself was modified at 13:16:11, after the first attempt of A had been scored on dev1 (13:07:32);
- propagation audit: see above.

For Phase 4 and Phase 5 no code hashes were recorded. 22 of the 33 Phase 4 scripts carry file times of 22:01–22:04 on 2026-09-27, after the tuning and the lock, and `dev5.py` of Phase 5 was extended for Phase 6 after the Phase 5 results. The archive does not show whether their content changed. What it does show is functional: `experiments/phase7/results/test7_hades.json` records that the archived engine reproduces the stored Phase 4 and Phase 5 scores (static and memory read-outs identically, propagation ranks up to a few images per stream).

## Statistical conventions

Intervals are paired 95% t intervals over the stated units (class splits, draws of the labelled images, or arrival orders); streams that share a unit are averaged first. The five splits of U1 may share held-out classes (amendment 3 of Phase 4), which the intervals do not model. Bootstrap intervals are marked where they are used. Analyses that were not registered are labelled descriptive or post hoc in `docs/RESULTS.md` and in the paper.
