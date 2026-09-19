"""Sharpe-ratio statistics that account for selection bias.

References
----------
Bailey, D. H. and Lopez de Prado, M. (2014). "The Deflated Sharpe Ratio:
Correcting for Selection Bias, Backtest Overfitting and Non-Normality".
Journal of Portfolio Management 40(5).

Mertens, E. (2002). "Comments on variance of the IID estimator in Lo (2002)."
"""

from __future__ import annotations

import numpy as np
from scipy import stats

EULER_GAMMA = 0.5772156649015329


def sharpe_ratio(returns: np.ndarray, axis: int = 0) -> np.ndarray:
    """Per-period (not annualised) Sharpe ratio: sample mean / sample std (ddof=1)."""
    r = np.asarray(returns, dtype=float)
    return r.mean(axis=axis) / r.std(axis=axis, ddof=1)


def annualise(sr: np.ndarray | float, periods_per_year: int = 252) -> np.ndarray | float:
    """Annualise a per-period Sharpe ratio (valid for i.i.d. returns)."""
    return np.asarray(sr) * np.sqrt(periods_per_year)


def sharpe_std_error(sr: float, n_obs: int, skew: float = 0.0, kurtosis: float = 3.0) -> float:
    """Asymptotic standard error of a Sharpe estimate.

    Uses the non-normality correction of Mertens (2002): for i.i.d. Gaussian
    returns (skew=0, kurtosis=3) it reduces to sqrt((1 + sr^2 / 2) / (n_obs - 1)).
    `kurtosis` is the raw (not excess) kurtosis.
    """
    var = (1.0 - skew * sr + (kurtosis - 1.0) / 4.0 * sr**2) / (n_obs - 1)
    return float(np.sqrt(var))


def probabilistic_sharpe_ratio(
    sr: float, benchmark_sr: float, n_obs: int, skew: float = 0.0, kurtosis: float = 3.0
) -> float:
    """P(true Sharpe > benchmark_sr) given the observed per-period Sharpe `sr`."""
    se = sharpe_std_error(sr, n_obs, skew, kurtosis)
    return float(stats.norm.cdf((sr - benchmark_sr) / se))


def expected_max_sharpe(n_trials: int, sr_variance: float) -> float:
    """Expected maximum Sharpe among `n_trials` independent zero-skill strategies.

    `sr_variance` is the variance of the Sharpe estimates across the trials.
    This is the "SR_0" threshold of Bailey and Lopez de Prado (2014), built on
    the extreme-value approximation of the maximum of N standard normals.
    """
    if n_trials < 2:
        return 0.0
    z1 = stats.norm.ppf(1.0 - 1.0 / n_trials)
    z2 = stats.norm.ppf(1.0 - 1.0 / (n_trials * np.e))
    return float(np.sqrt(sr_variance) * ((1.0 - EULER_GAMMA) * z1 + EULER_GAMMA * z2))


def deflated_sharpe_ratio(
    sr: float,
    n_obs: int,
    n_trials: int,
    sr_variance: float,
    skew: float = 0.0,
    kurtosis: float = 3.0,
) -> float:
    """Probability that the selected strategy's true Sharpe exceeds what luck alone
    would produce after trying `n_trials` strategies."""
    sr0 = expected_max_sharpe(n_trials, sr_variance)
    return probabilistic_sharpe_ratio(sr, sr0, n_obs, skew, kurtosis)


def deflated_sharpe_of_best(returns: np.ndarray) -> dict:
    """Select the best in-sample strategy from a (T, N) return matrix and deflate it.

    Returns per-period Sharpe of the winner, its naive PSR against zero, and its
    deflated Sharpe ratio (which accounts for the N strategies that were tried).
    """
    r = np.asarray(returns, dtype=float)
    n_obs, n_trials = r.shape
    srs = sharpe_ratio(r)
    best = int(np.argmax(srs))
    x = r[:, best]
    skew = float(stats.skew(x))
    kurt = float(stats.kurtosis(x, fisher=False))
    sr_var = float(np.var(srs, ddof=1)) if n_trials > 1 else 0.0
    return {
        "best_index": best,
        "best_sharpe": float(srs[best]),
        "naive_psr": probabilistic_sharpe_ratio(srs[best], 0.0, n_obs, skew, kurt),
        "deflated_sr": deflated_sharpe_ratio(srs[best], n_obs, n_trials, sr_var, skew, kurt),
        "sr0": expected_max_sharpe(n_trials, sr_var),
    }
