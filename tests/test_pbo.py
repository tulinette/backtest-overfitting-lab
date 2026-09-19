import numpy as np
import pytest

from overfit import cscv_pbo, simulate_returns


def test_pure_noise_gives_pbo_near_half_on_average():
    # PBO of a single dataset is itself random (sd ~ 0.2 with 40 strategies), so
    # the claim "pure noise gives PBO ~ 0.5" is about the average over datasets.
    pbos = [
        cscv_pbo(simulate_returns(1024, 40, seed=s), n_blocks=12, max_splits=200).pbo
        for s in range(20)
    ]
    assert 0.4 <= float(np.mean(pbos)) <= 0.6


def test_in_sample_winner_is_lucky_in_noise():
    r = simulate_returns(n_obs=1024, n_strategies=40, seed=3)
    res = cscv_pbo(r, n_blocks=12)
    # the winner looks good in-sample, and that goodness does not carry over
    assert res.is_best_sharpe.mean() > 0.05
    assert res.oos_sharpe_of_best.mean() < res.is_best_sharpe.mean() / 2


def test_strong_skilled_strategy_gives_low_pbo():
    r = simulate_returns(n_obs=1024, n_strategies=20, n_skilled=1, skilled_annual_sharpe=6.0, seed=5)
    res = cscv_pbo(r, n_blocks=12)
    assert res.pbo < 0.1


def test_number_of_splits_and_sampling():
    r = simulate_returns(n_obs=320, n_strategies=5, seed=0)
    assert cscv_pbo(r, n_blocks=8).n_splits == 70   # C(8, 4)
    assert cscv_pbo(r, n_blocks=8, max_splits=20).n_splits == 20


@pytest.mark.parametrize(
    "kwargs",
    [dict(n_blocks=3), dict(n_blocks=1)],
)
def test_bad_block_count(kwargs):
    with pytest.raises(ValueError):
        cscv_pbo(np.random.default_rng(0).normal(size=(100, 4)), **kwargs)


def test_bad_shapes():
    with pytest.raises(ValueError):
        cscv_pbo(np.zeros(10))
    with pytest.raises(ValueError):
        cscv_pbo(np.random.default_rng(0).normal(size=(100, 1)))
    with pytest.raises(ValueError):
        cscv_pbo(np.random.default_rng(0).normal(size=(20, 4)), n_blocks=16)
