import numpy as np
import pytest
from scipy import stats

from overfit import (
    annualise,
    deflated_sharpe_ratio,
    expected_max_sharpe,
    probabilistic_sharpe_ratio,
    sharpe_ratio,
    sharpe_std_error,
)


def test_sharpe_ratio_known_value():
    r = np.array([0.01, 0.03, 0.02, 0.02])
    assert sharpe_ratio(r) == pytest.approx(r.mean() / r.std(ddof=1))


def test_sharpe_ratio_columnwise():
    rng = np.random.default_rng(1)
    r = rng.normal(size=(200, 5))
    assert sharpe_ratio(r).shape == (5,)


def test_annualise():
    assert annualise(0.1, 252) == pytest.approx(0.1 * np.sqrt(252))


def test_std_error_gaussian_reduces_to_classic_formula():
    sr, n = 0.3, 500
    assert sharpe_std_error(sr, n) == pytest.approx(np.sqrt((1 + sr**2 / 2) / (n - 1)))


def test_std_error_penalises_fat_tails_and_negative_skew():
    base = sharpe_std_error(0.2, 500)
    assert sharpe_std_error(0.2, 500, skew=-1.0) > base
    assert sharpe_std_error(0.2, 500, kurtosis=8.0) > base


def test_psr_is_half_at_benchmark_and_monotone():
    assert probabilistic_sharpe_ratio(0.1, 0.1, 300) == pytest.approx(0.5)
    assert probabilistic_sharpe_ratio(0.2, 0.0, 300) > probabilistic_sharpe_ratio(0.1, 0.0, 300)


def test_expected_max_zero_for_single_trial():
    assert expected_max_sharpe(1, 0.01) == 0.0


def test_expected_max_grows_with_trials_and_scales_with_sd():
    vals = [expected_max_sharpe(n, 1.0) for n in (2, 10, 100, 1000)]
    assert vals == sorted(vals)
    assert expected_max_sharpe(50, 4.0) == pytest.approx(2 * expected_max_sharpe(50, 1.0))


@pytest.mark.parametrize("n", [10, 100, 1000])
def test_expected_max_matches_monte_carlo(n):
    rng = np.random.default_rng(42)
    sims = rng.normal(size=(4000, n)).max(axis=1).mean()
    assert expected_max_sharpe(n, 1.0) == pytest.approx(sims, rel=0.06)


def test_single_trial_dsr_equals_psr_against_zero():
    assert deflated_sharpe_ratio(0.1, 250, 1, 0.01) == pytest.approx(
        probabilistic_sharpe_ratio(0.1, 0.0, 250)
    )


def test_dsr_decreases_with_more_trials():
    a = deflated_sharpe_ratio(0.15, 250, 10, 0.004)
    b = deflated_sharpe_ratio(0.15, 250, 1000, 0.004)
    assert b < a
