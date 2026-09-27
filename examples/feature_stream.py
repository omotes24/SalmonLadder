"""Dataset-free smoke example using the exact experiment implementation."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments/controls_456/src"))
from core import Detector, norm, static_view


def main():
    rng = np.random.default_rng(123)
    classes, dim = 6, 32
    support = norm(rng.normal(size=(classes, 12, dim)))
    calibration = norm(rng.normal(size=(classes * 4, dim)))
    query = norm(rng.normal(size=(16, dim)))
    prototype = norm(support.mean(axis=1))
    candidates = lambda x: np.argsort(-(x @ prototype.T), axis=1)[:, :5]
    view = static_view(support, calibration, query, candidates(calibration), candidates(query))
    detector = Detector(view, device="cpu", cap=1000, mode="warm15")
    for start in range(0, len(query), 4):
        stop = start + 4
        scores = detector.step(query[start:stop], view["d"][start:stop], view["pall"][start:stop])
        assert np.isfinite(scores["full"]).all()
        print(f"batch {start // 4}: REPRISE ID scores {scores['full'].round(5).tolist()}")
    state = detector.clone()
    state.step(query[:1], view["d"][:1], view["pall"][:1])
    assert detector.seen == 16 and state.seen == 17
    print("Synthetic feature smoke test passed. No benchmark metric is reported.")


if __name__ == "__main__":
    main()
