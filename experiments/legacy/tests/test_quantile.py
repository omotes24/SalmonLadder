"""Split-conformal quantile: q_eps is the ceil((n+1)(1-eps))-th smallest calibration score."""
import math

import numpy as np
import pytest

from vins.dview import conformal_rank, conformal_threshold, nominate


def test_ranks_for_the_dev_calibration_size():
    assert conformal_rank(3600, 0.005) == 3583
    assert conformal_rank(3600, 0.01) == 3565
    assert conformal_rank(3600, 0.02) == 3529


def test_rank_matches_integer_arithmetic():
    # eps in thousandths: ceil((n+1)(1000-e)/1000) computed with integers only
    for n in range(1, 5000, 7):
        for milli in (5, 10, 20, 50, 100, 200):
            exact = -((-(n + 1) * (1000 - milli)) // 1000)
            assert conformal_rank(n, milli / 1000) == exact, (n, milli)


def test_threshold_is_order_statistic_and_strict_nomination():
    cal = np.arange(1, 101, dtype=float)[::-1]           # 100 values, unsorted
    q = conformal_threshold(cal, 0.05)                    # rank ceil(101 * 0.95) = 96
    assert q == 96.0
    d = np.array([95.0, 96.0, 96.5, 100.0, 1000.0])
    assert nominate(d, q).tolist() == [False, False, True, True, True]


def test_infinite_threshold_when_rank_exceeds_n():
    cal = np.arange(10, dtype=float)
    assert conformal_rank(10, 0.05) == 11
    assert math.isinf(conformal_threshold(cal, 0.05))
    assert not nominate(np.array([1e9]), conformal_threshold(cal, 0.05)).any()


@pytest.mark.parametrize("eps", [0.005, 0.01, 0.02])
def test_marginal_coverage_on_exchangeable_data(eps):
    rng = np.random.default_rng(1)
    n, trials, m = 3600, 400, 2000
    rates = []
    for _ in range(trials):
        cal = rng.normal(size=n)
        test = rng.normal(size=m)
        rates.append(nominate(test, conformal_threshold(cal, eps)).mean())
    mean_rate = float(np.mean(rates))
    # split-conformal: E[rate] <= eps and >= eps - 1/(n+1)
    assert eps - 1 / (n + 1) - 0.002 <= mean_rate <= eps + 0.002
