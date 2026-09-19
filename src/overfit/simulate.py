"""Synthetic data generators for the experiments."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def simulate_returns(
    n_obs: int,
    n_strategies: int,
    n_skilled: int = 0,
    skilled_annual_sharpe: float = 1.0,
    daily_vol: float = 0.01,
    periods_per_year: int = 252,
    seed: int = 0,
) -> np.ndarray:
    """Daily returns of `n_strategies` independent strategies, shape (n_obs, n_strategies).

    Most strategies have exactly zero expected return (pure luck). The first
    `n_skilled` columns get a genuine positive drift matching `skilled_annual_sharpe`.
    """
    rng = np.random.default_rng(seed)
    r = rng.normal(0.0, daily_vol, size=(n_obs, n_strategies))
    if n_skilled > 0:
        daily_mu = skilled_annual_sharpe / np.sqrt(periods_per_year) * daily_vol
        r[:, :n_skilled] += daily_mu
    return r


@dataclass
class LeaderboardResult:
    true_skill: np.ndarray
    public_score: np.ndarray
    private_score: np.ndarray
    public_winner: int
    private_rank_of_public_winner: int  # 1 = best
    n_teams: int


def simulate_leaderboard(
    n_teams: int,
    n_public: int,
    n_private: int,
    skill_sd: float = 0.01,
    noise_sd: float = 0.5,
    seed: int = 0,
) -> LeaderboardResult:
    """A Kaggle-style competition: each team has a true skill; the public and the
    private leaderboards measure it on different finite test samples.

    score = true_skill + noise, with noise_sd / sqrt(sample size) standard deviation.
    """
    rng = np.random.default_rng(seed)
    skill = rng.normal(0.0, skill_sd, size=n_teams)
    public = skill + rng.normal(0.0, noise_sd / np.sqrt(n_public), size=n_teams)
    private = skill + rng.normal(0.0, noise_sd / np.sqrt(n_private), size=n_teams)
    winner = int(np.argmax(public))
    rank = int((private > private[winner]).sum()) + 1
    return LeaderboardResult(skill, public, private, winner, rank, n_teams)
