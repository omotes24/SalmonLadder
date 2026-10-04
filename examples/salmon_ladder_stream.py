"""Salmon Ladder (frozen configuration v5) and the locked extension on a synthetic two-view stream (CPU, a few seconds).

The script calls the archived implementation directly:
  experiments/phase4/code/static.py    static prototype view: d(x), d_all(x) and the static calibration rank p
  experiments/phase4/code/engine.py    entrance and memory (class Memory), prefix kNN graph, label propagation
  experiments/phase7/code/engine7.py   run_view7: the frozen read-outs plus the read-outs of the extension

Per view the engine returns log calibration ranks (small = looks unknown). A detector is the sum over the views:
  static p     static
  zeta         z            robust z of the zero-start propagation (the strongest propagation-only configuration)
  Salmon Ladder  M + lp0    memory rank x warm-start propagation rank (the frozen method)
  extension    Mh2@0 + lp   one-sided memory that also reads members admitted from the same batch x zero-start rank

The stream is synthetic (groups of sibling classes with one unknown class per group). It demonstrates the interface
and the behaviour under recurrence; it is not a benchmark and no number printed here appears in the paper.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for rel in ("experiments/lp_audit/vendor", "experiments/phase4/code", "experiments/phase7/code"):
    sys.path.insert(0, str(ROOT / rel))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import engine7 as E7  # noqa: E402
import synth5  # noqa: E402
from metrics_p4 import metrics  # noqa: E402
from static import static_view  # noqa: E402

# the frozen configuration (experiments/phase4/code/common.py: V5)
V5 = {"n0": 48, "m": 1, "thresholds": (0.3, 0.2, 0.10191613435745239), "k": 10, "gamma": 1.0, "lam": 0.9, "sweeps": 15}


def main():
    stream = synth5.make(seed=0, batch=64)            # 60 ID classes, 20 unknown classes of 50 images each, batches of 64
    total = {}
    for view in stream["views"].values():
        # support: (classes, 12, D); calibration: (4 x classes, D); stream features in arrival order; candidate classes
        static = static_view(view["sup"], view["cal"], view["sf"], stream["cand_cal"], stream["cand_sf"], V5["n0"], V5["m"])
        out, aux = E7.run_view7(view["sup"], view["cal"], view["sf"], stream["bidx"], static, V5["thresholds"], k=V5["k"],
                                gamma=V5["gamma"], lam=V5["lam"], iters=V5["sweeps"], device="cpu", retro=(0,))
        for key in ("static", "z", "M", "lp0", "lp", "Mh2@0"):
            total[key] = total.get(key, 0.0) + out[key]
    detectors = {"static p": total["static"], "zeta (propagation only)": total["z"], "Salmon Ladder (frozen v5)": total["M"] + total["lp0"],
                 "extension (locked)": total["Mh2@0"] + total["lp"]}
    is_ood = stream["is_ood"]
    print(f"synthetic stream: {len(is_ood)} images, {int(is_ood.sum())} from unknown classes, {int(stream['bidx'].max()) + 1} batches")
    for name, score in detectors.items():
        m = metrics(score, is_ood)                    # the score is large for ID-looking images
        print(f"  {name:26s} AUROC {m['AUROC']:6.2f}   FPR95 {m['FPR95']:6.2f}")
    print(f"  memory admission in the last view: {100 * aux['admit'][is_ood, 2].mean():.1f}% of the unknown-class images, "
          f"{100 * aux['admit'][~is_ood, 2].mean():.1f}% of the ID images")
    print("Synthetic feature example finished. No benchmark metric is reported.")


if __name__ == "__main__":
    main()
