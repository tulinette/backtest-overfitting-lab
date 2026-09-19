"""Reproduce every number and figure quoted in the README.

Run from the repository root:  python experiments/run_experiments.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from overfit import (
    annualise,
    cscv_pbo,
    deflated_sharpe_of_best,
    expected_max_sharpe,
    sharpe_ratio,
    simulate_leaderboard,
    simulate_returns,
)

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "figures"
FIG.mkdir(exist_ok=True)

T_OBS = 1250            # about five years of daily returns
N_LIST = [1, 10, 100, 1000]
REPS = 300
THRESH = 0.95           # "significant at 95%"


def exp1_false_positives() -> dict:
    """Try N strategies with ZERO true skill, keep the best. How often does the
    naive test call it significant, versus the deflated test?"""
    out = {}
    for n in N_LIST:
        naive_hits, dsr_hits, best_sr_ann = 0, 0, []
        for rep in range(REPS):
            r = simulate_returns(T_OBS, n, seed=10_000 * n + rep)
            res = deflated_sharpe_of_best(r)
            naive_hits += res["naive_psr"] > THRESH
            dsr_hits += res["deflated_sr"] > THRESH
            best_sr_ann.append(float(annualise(res["best_sharpe"])))
        out[str(n)] = {
            "mean_best_annualised_sharpe": float(np.mean(best_sr_ann)),
            "naive_false_positive_rate": naive_hits / REPS,
            "deflated_false_positive_rate": dsr_hits / REPS,
        }
    return out


def exp2_pbo() -> dict:
    """PBO averaged over datasets, for pure noise and with one genuinely skilled strategy."""
    res = {}
    for label, kw in {
        "noise_only": dict(n_skilled=0),
        "one_skilled_sharpe_1.0": dict(n_skilled=1, skilled_annual_sharpe=1.0),
        "one_skilled_sharpe_3.0": dict(n_skilled=1, skilled_annual_sharpe=3.0),
    }.items():
        pbos = []
        for s in range(30):
            r = simulate_returns(T_OBS, 100, seed=500 + s, **kw)
            pbos.append(cscv_pbo(r, n_blocks=12, max_splits=200, seed=s).pbo)
        res[label] = {"mean_pbo": float(np.mean(pbos)), "sd_across_datasets": float(np.std(pbos))}
    return res


def exp3_leaderboard() -> dict:
    """Sweep the ratio of true-skill spread to test-set noise. With noise_sd=0.5 the
    public/private noise sd is about 0.029/0.019 (300/700 test samples)."""
    out = {}
    for skill_sd in (0.01, 0.03, 0.1):
        ranks = np.array(
            [simulate_leaderboard(1000, 300, 700, skill_sd=skill_sd, seed=s).private_rank_of_public_winner for s in range(500)]
        )
        out[f"skill_sd_{skill_sd}"] = {
            "median_private_rank_of_public_winner": float(np.median(ranks)),
            "share_still_first": float((ranks == 1).mean()),
            "share_in_top_10": float((ranks <= 10).mean()),
        }
    return out


def figures(exp1: dict) -> None:
    # 1) expected best Sharpe under pure luck
    ns = np.unique(np.logspace(0.3, 4, 60).astype(int))
    sd = 1 / np.sqrt(T_OBS - 1)  # std of a per-period Sharpe estimate at zero skill
    best = [expected_max_sharpe(int(n), sd**2) * np.sqrt(252) for n in ns]
    fig, ax = plt.subplots(figsize=(5.5, 3.6))
    ax.semilogx(ns, best)
    ax.set_xlabel("number of strategies tried (all zero true skill)")
    ax.set_ylabel("expected best annualised Sharpe")
    ax.set_title(f"Luck alone buys a good backtest ({T_OBS} daily obs)")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG / "luck_vs_trials.png", dpi=150)
    plt.close(fig)

    # 2) PBO logit distribution, noise vs skilled
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4), sharey=True)
    for ax, (title, kw) in zip(
        axes,
        [("100 zero-skill strategies (one dataset)", {}), ("+ one with true Sharpe 3", dict(n_skilled=1, skilled_annual_sharpe=3.0))],
    ):
        r = simulate_returns(T_OBS, 100, seed=7, **kw)
        res = cscv_pbo(r, n_blocks=12, max_splits=400, seed=1)
        ax.hist(res.logits, bins=30)
        ax.axvline(0, color="k", lw=1)
        ax.set_title(f"{title}\nPBO = {res.pbo:.2f}")
        ax.set_xlabel("logit of out-of-sample rank of the in-sample winner")
    axes[0].set_ylabel("splits")
    fig.tight_layout()
    fig.savefig(FIG / "pbo_logits.png", dpi=150)
    plt.close(fig)

    # 3) leaderboard shake-up
    lb = simulate_leaderboard(1000, 300, 700, seed=1)
    fig, ax = plt.subplots(figsize=(4.6, 4.2))
    ax.scatter(lb.public_score, lb.private_score, s=6, alpha=0.5)
    w = lb.public_winner
    ax.scatter([lb.public_score[w]], [lb.private_score[w]], color="C3", s=40, label=f"public #1 -> private #{lb.private_rank_of_public_winner}")
    ax.set_xlabel("public score")
    ax.set_ylabel("private score")
    ax.legend(loc="upper left")
    ax.set_title("Public vs private leaderboard (simulated)")
    fig.tight_layout()
    fig.savefig(FIG / "leaderboard_shakeup.png", dpi=150)
    plt.close(fig)


def main() -> None:
    exp1 = exp1_false_positives()
    exp2 = exp2_pbo()
    exp3 = exp3_leaderboard()
    figures(exp1)
    results = {
        "settings": {"T_obs": T_OBS, "reps_exp1": REPS, "threshold": THRESH},
        "exp1_false_positives": exp1,
        "exp2_pbo": exp2,
        "exp3_leaderboard": exp3,
    }
    (ROOT / "results.json").write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
