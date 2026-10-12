"""Salmon Ladder Phase 18 (descriptive, NOT registered; nothing is selected from these results): the analyses asked for by the
review of the 4-page version. Every configuration below is fixed in this file before any of it is run; the paper's configuration
(views L14 + D3L, k_g = 10, lambda = 0.9, Storey-BH q = 0.2, 12 support / 2 reference R / 2 calibration C images per class) is
unchanged, and p+ / sigma_ref computed here must equal the registered Phase 17 values of the same task (checked per task).
  (a) read-outs on the same features and the same 16 labels, all read as ranks among the reference images R (as p+):
      static p (cosine to the nearest class prototype), kNN (cosine to the 10th nearest support image), the one-step propagation
      D^-1 W y+ (the share of a node's neighbourhood weight that goes to support images), LGC-max (class-label propagation on the
      same graph, largest class mass), p+ (= Phase 17 ref|lp|stat) and sigma_ref (= Phase 17 ref|storey|q0.2|pp|stat);
      per view and summed over the two views (single view L14 / D3L = 'views' of the paper);
  (b) other backbones, one view: CLIP ViT-B/16 image features (the backbone of TINS) and DINO ViT-B/16 (self-supervised on
      ImageNet-1K only: never saw ImageNet-21K, from which SSB-hard is drawn); the paper's draw of the 16 images (p3) only,
      because the pool draws have no features of these two backbones that are verified identical to the benchmark ones;
  (c) R and C from the OpenOOD ImageNet-1K validation list (val_imagenet.txt: 5 images per class, disjoint from the ID test list
      test_imagenet.txt): 4 per class (2 R, 2 C), support unchanged (12 training images per class, 16 labels per class in all);
      OpenOOD test sets unchanged, Four-OOD ID test without these 4,000 images; validity of p_fin (false-alarm rate, BH FDP) and
      the ID fraction of the seeds;
  (d) sensitivity of p+ and sigma_ref: k_g in {5, 20}, lambda in {0.5, 0.99}, support images per class in {4, 8};
  (e) compute time of the method on the largest task (OpenOOD SSB-hard: 12,000 + 4,000 + 94,000 nodes per view)."""
import json
import sys
from pathlib import Path

sys.path.insert(0, "/home/omote/reprise_p17_20261011/code")
import p17common as _M17  # noqa: E402  (Phase 17 -> 16 -> ... paths, Phase 5 code, vendored vins package)
from p17common import *  # noqa: E402,F401,F403
M17 = _M17

P18 = Path("/home/omote/reprise_p18_20261012")
RESULTS18 = P18 / "results"
LOGS18 = P18 / "logs"

VIEWS18 = ("L14", "D3L")           # the paper's two views
BACKBONES18 = ("CLIP", "DINO1")    # (b): one view each, draw p3
KG18 = 10                          # the paper's k_g
LAM18 = 0.9                        # the paper's lambda
Q18 = 0.2                          # the paper's Storey-BH level
KNN_K = 10                         # kNN baseline: the 10th nearest support image (Sun et al. 2022; as Phases 12-17)
SENS18 = {"k5": (5, 0.9), "k20": (20, 0.9), "lam0.5": (10, 0.5), "lam0.99": (10, 0.99)}   # (d): (k_g, lambda)
SHOTS18 = (4, 8)                   # (d): support images per class (the first ones in the stored order); R, C unchanged
VAL_RNG = 18                       # (c): rng [VAL_RNG, draw index, class] permutes the 5 validation images of a class; the first
                                   #      2 are R, the next 2 C (class-major order, so core16.split_cal applies unchanged)
TIMING_REPS = 3                    # (e)


def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=1, default=str) + "\n")
