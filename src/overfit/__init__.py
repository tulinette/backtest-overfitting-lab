"""Backtest-overfitting lab: measuring how much of a "best strategy" is luck."""

from .pbo import PBOResult, cscv_pbo
from .sharpe import (
    annualise,
    deflated_sharpe_of_best,
    deflated_sharpe_ratio,
    expected_max_sharpe,
    probabilistic_sharpe_ratio,
    sharpe_ratio,
    sharpe_std_error,
)
from .simulate import simulate_leaderboard, simulate_returns

__all__ = [
    "PBOResult",
    "annualise",
    "cscv_pbo",
    "deflated_sharpe_of_best",
    "deflated_sharpe_ratio",
    "expected_max_sharpe",
    "probabilistic_sharpe_ratio",
    "sharpe_ratio",
    "sharpe_std_error",
    "simulate_leaderboard",
    "simulate_returns",
]
