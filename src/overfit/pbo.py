"""Probability of Backtest Overfitting via Combinatorially Symmetric Cross-Validation.

Reference
---------
Bailey, D. H., Borwein, J., Lopez de Prado, M. and Zhu, Q. J. (2017).
"The Probability of Backtest Overfitting". Journal of Computational Finance 20(4).

Idea: split the (T, N) matrix of strategy returns into S time blocks. For every way
of choosing S/2 blocks as "in-sample" (the other S/2 being "out-of-sample"), pick the
strategy with the best in-sample Sharpe and look at where it ranks out-of-sample.
If the in-sample winner is at or below the out-of-sample median in a large share of
splits, the selection procedure is overfitting.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass

import numpy as np


@dataclass
class PBOResult:
    pbo: float                 # share of splits where the IS winner is at/below the OOS median
    logits: np.ndarray         # logit of the OOS relative rank of the IS winner, one per split
    is_best_sharpe: np.ndarray # in-sample Sharpe of the winner, one per split
    oos_sharpe_of_best: np.ndarray  # out-of-sample Sharpe of that same strategy, one per split
    n_splits: int


def cscv_pbo(
    returns: np.ndarray,
    n_blocks: int = 16,
    max_splits: int | None = None,
    seed: int = 0,
) -> PBOResult:
    """Compute the probability of backtest overfitting for a (T, N) return matrix."""
    r = np.asarray(returns, dtype=float)
    if r.ndim != 2:
        raise ValueError("returns must be a 2-D array of shape (T, N)")
    if n_blocks < 2 or n_blocks % 2:
        raise ValueError("n_blocks must be an even integer >= 2")
    n_obs, n_strats = r.shape
    if n_strats < 2:
        raise ValueError("need at least 2 strategies")
    block_len = n_obs // n_blocks
    if block_len < 3:
        raise ValueError("not enough observations per block")

    blocks = r[: block_len * n_blocks].reshape(n_blocks, block_len, n_strats)
    s1 = blocks.sum(axis=1)          # (S, N) sum of returns per block
    s2 = (blocks**2).sum(axis=1)     # (S, N) sum of squared returns per block

    def sharpe_on(idx: np.ndarray) -> np.ndarray:
        n = block_len * len(idx)
        mean = s1[idx].sum(axis=0) / n
        var = (s2[idx].sum(axis=0) - n * mean**2) / (n - 1)
        return mean / np.sqrt(var)

    splits = list(itertools.combinations(range(n_blocks), n_blocks // 2))
    if max_splits is not None and len(splits) > max_splits:
        rng = np.random.default_rng(seed)
        keep = rng.choice(len(splits), size=max_splits, replace=False)
        splits = [splits[i] for i in keep]

    everything = np.arange(n_blocks)
    logits, is_best, oos_of_best = [], [], []
    for combo in splits:
        is_idx = np.array(combo)
        oos_idx = np.setdiff1d(everything, is_idx)
        sr_is = sharpe_on(is_idx)
        sr_oos = sharpe_on(oos_idx)
        best = int(np.argmax(sr_is))
        rank = int((sr_oos < sr_oos[best]).sum()) + 1   # 1 (worst) .. N (best)
        omega = rank / (n_strats + 1)                   # relative rank in (0, 1)
        logits.append(np.log(omega / (1.0 - omega)))
        is_best.append(sr_is[best])
        oos_of_best.append(sr_oos[best])

    logits_arr = np.array(logits)
    return PBOResult(
        pbo=float((logits_arr <= 0).mean()),
        logits=logits_arr,
        is_best_sharpe=np.array(is_best),
        oos_sharpe_of_best=np.array(oos_of_best),
        n_splits=len(splits),
    )
