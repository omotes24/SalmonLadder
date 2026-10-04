# Method

This file states the frozen method exactly as the archived code computes it, and names the function that implements each step. Symbols are defined where they first appear. Scores are oriented so that **large means in-distribution (ID)**; a detector flags an image as unknown when its score is small.

## Setting

Images arrive in batches. For each of the `C` known classes, 16 labelled images are available: the first 12 are the **support** images and the last 4 are the **calibration** images. No other labelled data and no training are used. Every image is scored when its batch arrives, and an issued score is never revised.

Two frozen encoders give two **views** of an image: the L2-normalised CLS feature of DINOv2 ViT-B/14 and of DINOv2 ViT-L/14. A third frozen model, CLIP ViT-B/16, only selects **candidate classes**: the `K = 20` classes with the highest zero-shot similarity to the image. All steps below are computed per view; the final score multiplies the views.

## 1. Static score (no test-time information)

`experiments/lp_audit/vendor/vins/r5.py: proto_view`, called through `experiments/phase4/code/static.py: static_view`.

1. Prototype of class `c`: `mu_c`, the normalised sum of its 12 support features.
2. Leave-one-out distances: for each support image `s` of class `c`, `1 - cos(s, mu_c^(-s))`, where `mu_c^(-s)` is the prototype without `s`. Their median `med_c` and scaled median absolute deviation `MAD_c` (factor 1.4826) describe the spread of class `c`; `med_all` and `MAD_all` are the same statistics over all classes pooled.
3. Shrinkage with `n0 = 48`: `med~_c = (12 med_c + n0 med_all) / (12 + n0)`, and likewise `MAD~_c`. Twelve images per class give noisy scale estimates; the pooled value stabilises them.
4. Standardised distance to class `c`: `z_c(x) = ((1 - cos(x, mu_c)) - med~_c) / MAD~_c`.
5. `d(x) = min` of `z_c(x)` over the 20 candidate classes of `x`; `d_all(x) = min` over all classes. Large `d` means far from every plausible class.
6. Static calibration rank: `p_static(x) = (1 + #{calibration images with d >= d(x)}) / (n_cal + 1)`, where `n_cal = 4C`.

## 2. Entrance and memory

`experiments/phase4/code/engine.py: class Memory` (the frozen configuration uses thresholds `(0.3, 0.2, 0.10191613435745239)`).

The memory stores stream images that looked unknown when they arrived. A later image that lies close to a stored image is pushed towards "unknown". To keep ID images out of the memory, admission is decided by three tests with the sets `A1`, `A2` and `M`: the first uses the static distance alone, and the second and third are computed against the set that the test before admitted.

- Distance scale: `med`, `MAD` are the pooled median and scaled MAD of the leave-one-out nearest-neighbour cosine distances among the support images of a class (`vins/dview.py: loo_stats`, `m = 1`).
- For a set `S` of stored images, `rho_S(x) = 1 - max` cosine similarity between `x` and the members of `S` (2 if `S` is empty), and `g_S(x) = d(x) - (rho_S(x) - med) / MAD`. A near stored neighbour makes `rho_S` small and therefore `g_S` large. The same quantity is computed for every calibration image against the same set, and `p_S(x) = (1 + #{calibration images with g_S >= g_S(x)}) / (n_cal + 1)`.
- Admission rules, evaluated on the state left by the **earlier batches only**:
  - `x` enters `A1` if the rank of `d_all(x)` among the calibration `d_all` values is at most 0.3;
  - `x` enters `A2` if `p_A1(x) <= 0.2`;
  - `x` enters `M` if `p_A2(x) <= 0.1019`.
- The three conditions are evaluated independently for every image. The sets are therefore not nested: an image enters `M` without entering `A2` when its test against `A2` passes and its test against `A1` fails.
- Read-out: `p_M(x)`, computed against `M` before the images of the current batch are admitted.

The third threshold was tuned on the first development split so that 10% of its ID images are admitted to `M` (`experiments/legacy/r5/entrance/frozen.json`, entry `E3m1@0.1`); it is a constant of the frozen configuration and is not re-estimated on evaluation data.

## 3. Label propagation

`experiments/phase4/code/engine.py: PrefixTopK, propagate, run_view`.

- Nodes: the support images, the calibration images, and every stream image up to and including the current batch.
- Edges: at every batch the graph is the exact `k`-nearest-neighbour graph (`k = 10`, cosine similarity, self excluded) over all nodes present; the neighbour lists are updated incrementally when a batch arrives, without approximate search. Weight `max(cos, 0)^gamma` with `gamma = 1`; the graph is symmetrised by taking the larger weight and normalised as `S = D^(-1/2) W D^(-1/2)`.
- Labels: `Y = 1` on support nodes and 0 elsewhere. One label channel is propagated ("how much support mass reaches this node"), not one channel per class.
- Iteration: `U <- lambda S U + (1 - lambda) Y` with `lambda = 0.9`, 15 sweeps per batch. **Warm start**: the sweeps continue from the values of the previous batch; nodes of the new batch start at 0 and the first batch starts from `Y`.
- Read-out: `p_LP(x) = (1 + #{calibration images with u <= u(x)}) / (n_cal + 1)`, the rank of the mass `u(x)` among the masses of the calibration images **in the same graph state**. An unknown image surrounded by other unknown images receives little mass.

## 4. Score

`log s(x) = sum over the two views of [ log p_M(x) + log p_LP(x) ]`.

With a base detector `s0` (TINS, MCM, NegLabel, AdaNeg, TANL) the product `s0(x) * s(x)` is reported as `s0 x Salmon Ladder`; in logarithms the base score is added with weight 1. `Salmon Ladder, standalone` uses `s0 = 1`.

The calibration ranks are used as detection scores. The calibration images are reused by every batch and the memory adapts to the stream, so the ranks are **not** presented as distribution-free p-values; `experiments/lp_audit/validity_lemmas.md` states what does and does not hold.

## 5. Comparison read-outs on the same graph

The same graph gives detectors that use propagation without the entrance and the memory. They are implemented in the same function (`run_view`, argument `families`) so that the comparison shares every other component.

| name in the paper and in `docs/RESULTS.md` | key in the code | definition |
|---|---|---|
| static p | `static` | `p_static` of Section 1 |
| memory only | `Mpt` (Phase 4), `M` (Phase 7) | `p_M` |
| propagation only | `cdf_L0` / `lp0` | `p_LP`, warm start |
| propagation, zero start | `cdf` / `lp` | `p_LP` with all nodes iterated from 0 at every batch |
| zeta | `z` | `Phi((u(x) - median) / MAD)` of the zero-start mass, median and MAD taken over the calibration images; the read-out without entrance and memory that was strongest on the development split, and therefore the registered comparator |
| standard propagation | `raw`, `raw_L2`, `rw`, `mass` | the mass `u` itself, the converged solution, random-walk normalisation, and `u` rescaled by the node count |
| Salmon Ladder | `rep_L0` = `Mpt + cdf_L0` | Section 4 |

In Phase 4 every family received the same 27 graph configurations on the development split; the selected configurations were locked before the evaluation data existed (`experiments/phase4/selection_lock.json`).

## 6. Extension evaluated in Phase 7

`experiments/phase7/code/engine7.py: run_view7`, keys `Mh2@0` and `lp`; locked in `experiments/phase7/selection_lock_p7.json`.

The frozen memory cannot help the first images of an unknown class, and it cannot use neighbours that arrive in the same batch. The extension changes two read-outs and leaves the entrance, the graph and every threshold unchanged:

- **Within-batch, one-sided memory.** `rho(x)` is the distance to the nearest member of `M` admitted up to **and including** the current batch (the image itself is excluded). `g(x) = d(x) + max(0, 2 - (rho(x) - med) / MAD)`: a near stored neighbour raises `g`, and a missing neighbour does not lower it. The rank is taken against the calibration images under the same formula.
- **Zero-start propagation rank** (`lp`) instead of the warm-start rank.

`log s_ext(x) = sum over the views of [ log p_Mh2@0(x) + log p_lp(x) ]`. The extension was selected on the development splits after the frozen method had been evaluated on U1 and U2; its results on those sets are reported as an evaluation of a locked configuration on data that had already been scored once (see `docs/PREREGISTRATION.md`).

## Frozen constants

`experiments/phase4/code/common.py: V5`

| constant | value |
|---|---|
| labelled images per class | 12 support + 4 calibration |
| candidate classes `K` | 20 (CLIP ViT-B/16 zero-shot) |
| shrinkage `n0` | 48 |
| memory neighbour | nearest member (`m = 1`) |
| entrance thresholds | 0.3, 0.2, 0.10191613435745239 |
| graph | `k = 10`, `gamma = 1`, `lambda = 0.9`, 15 sweeps, warm start |
| batch size in the main experiments | 256 |

## Names in the archived code

Salmon Ladder is the name of the method in the paper. It was developed under working names, and the archived files keep them because their content is fixed by hashes: CLAVIS (static score of one view), CLAVIS-M (plus a memory), CLAVIS-M2 (two views, two-stage entrance), CLAVIS-M3 or `v4` (prototypes, three-stage entrance, propagation), and REPRISE, the name of the method from `v4` on and during all later experiments. `v5` is the frozen configuration above. Registrations, scripts, result keys (`REPRISE`, `rep_L0`, `ext-reprise`) and the directories of the experiment server (`reprise_p4_20260928` and the like) therefore say REPRISE where the documentation says Salmon Ladder. `minimal` denotes a read-out without entrance and memory, and `best_minimal` the one selected on the development split (zeta). `VINS` is the name of the earliest project stage.

## Cost

On one RTX 2080 Ti, with image encoding excluded, the frozen method takes 0.21 ms per image and 0.38 GiB of GPU memory on streams of 23,000 images (`docs/RESULTS.md`, Section 1.4). The graph grows with the stream; `docs/RESULTS.md`, Section 4 reports the effect of bounding the history.

## Minimal use

`examples/salmon_ladder_stream.py` runs Sections 1–6 on a synthetic stream on a CPU:

```python
static = static_view(support, calibration, stream, cand_calibration, cand_stream, n0=48, m=1)
out, aux = engine7.run_view7(support, calibration, stream, batch_index, static, (0.3, 0.2, 0.10191613435745239),
                             k=10, gamma=1.0, lam=0.9, iters=15, device="cpu", retro=(0,))
log_score = out["M"] + out["lp0"]            # Salmon Ladder; summed over the views
log_extension = out["Mh2@0"] + out["lp"]
```

`support` has shape `(C, 12, D)`, `calibration` `(4C, D)` in class order, `stream` `(N, D)` in arrival order, `batch_index` gives the batch of each stream row, and the two `cand_*` arrays hold the candidate class indices.
