import numpy as np

from overfit import deflated_sharpe_of_best, simulate_leaderboard, simulate_returns


def test_simulate_shape_and_reproducibility():
    a = simulate_returns(100, 7, seed=1)
    b = simulate_returns(100, 7, seed=1)
    assert a.shape == (100, 7)
    assert np.array_equal(a, b)


def test_skilled_columns_have_higher_mean():
    r = simulate_returns(20000, 4, n_skilled=1, skilled_annual_sharpe=2.0, seed=2)
    assert r[:, 0].mean() > r[:, 1:].mean(axis=0).max()


def test_deflated_sharpe_of_best_keys_and_ordering():
    r = simulate_returns(500, 100, seed=4)
    out = deflated_sharpe_of_best(r)
    assert out["deflated_sr"] <= out["naive_psr"]
    assert out["sr0"] > 0


def test_leaderboard_winner_regresses():
    ranks = [simulate_leaderboard(1000, 300, 700, seed=s).private_rank_of_public_winner for s in range(30)]
    assert np.median(ranks) > 1
