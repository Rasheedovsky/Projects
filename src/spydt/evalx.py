"""Evaluation: economics, deflated statistics, model-comparison tests.

Every model is mapped through the IDENTICAL trading rule: position =
sign(forecast), unit notional, entry 15:31 open, MOC exit, identical costs.
"""
from __future__ import annotations

import numpy as np
from itertools import combinations
from scipy import stats

ANNUAL = 252


def strategy_net_returns(prob: np.ndarray, r_exec: np.ndarray,
                         cost_per_side_bp: float) -> np.ndarray:
    """Daily net log returns of the sign rule at the given per-side cost."""
    pos = np.where(prob >= 0.5, 1.0, -1.0)
    return pos * r_exec - 2.0 * cost_per_side_bp * 1e-4


def hit_rate(prob: np.ndarray, y: np.ndarray) -> float:
    return float(np.mean(np.where(prob >= 0.5, 1.0, -1.0) == y))


def auc(prob: np.ndarray, y: np.ndarray) -> float:
    from sklearn.metrics import roc_auc_score

    y01 = (y > 0).astype(int)
    if y01.min() == y01.max():
        return 0.5
    return float(roc_auc_score(y01, prob))


def sharpe(r: np.ndarray) -> float:
    s = r.std(ddof=1)
    return float(r.mean() / s * np.sqrt(ANNUAL)) if s > 0 else 0.0


def sortino(r: np.ndarray) -> float:
    dn = r[r < 0].std(ddof=1) if (r < 0).any() else 0.0
    return float(r.mean() / dn * np.sqrt(ANNUAL)) if dn > 0 else 0.0


def max_drawdown(r: np.ndarray) -> float:
    eq = np.cumsum(r)
    return float((np.maximum.accumulate(eq) - eq).max())


def psr(r: np.ndarray, sr_star_annual: float = 0.0) -> float:
    """Probabilistic Sharpe Ratio (skew/kurtosis-adjusted), vs annual SR*."""
    n = len(r)
    if n < 20 or r.std(ddof=1) == 0:
        return 0.5
    sr = r.mean() / r.std(ddof=1)                    # per-period
    sr_star = sr_star_annual / np.sqrt(ANNUAL)
    g3 = stats.skew(r)
    g4 = stats.kurtosis(r, fisher=False)
    denom = np.sqrt(max(1 - g3 * sr + (g4 - 1) / 4 * sr**2, 1e-12) / (n - 1))
    return float(stats.norm.cdf((sr - sr_star) / denom))


def effective_trials(returns_matrix: np.ndarray) -> float:
    """Effective number of independent trials from the correlation spectrum
    (participation ratio). returns_matrix: (T, N) daily net returns."""
    if returns_matrix.shape[1] < 2:
        return 1.0
    c = np.corrcoef(returns_matrix.T)
    c = np.nan_to_num(c, nan=0.0)
    np.fill_diagonal(c, 1.0)
    ev = np.linalg.eigvalsh(c)
    ev = np.clip(ev, 0, None)
    return float(ev.sum() ** 2 / (ev**2).sum())


def expected_max_sharpe(n_trials: float, var_sr: float) -> float:
    """E[max SR] under N independent standard trials (Bailey & LdP 2014)."""
    if n_trials <= 1 or var_sr <= 0:
        return 0.0
    gamma = 0.5772156649
    n = max(n_trials, 1.0 + 1e-9)
    z1 = stats.norm.ppf(1 - 1.0 / n)
    z2 = stats.norm.ppf(1 - 1.0 / (n * np.e))
    return float(np.sqrt(var_sr) * ((1 - gamma) * z1 + gamma * z2))


def dsr(r: np.ndarray, trial_returns_matrix: np.ndarray) -> float:
    """Deflated Sharpe Ratio: PSR against the expected max SR of the trials."""
    srs = []
    for j in range(trial_returns_matrix.shape[1]):
        col = trial_returns_matrix[:, j]
        s = col.std(ddof=1)
        srs.append(col.mean() / s if s > 0 else 0.0)
    var_sr = float(np.var(srs, ddof=1)) if len(srs) > 1 else 0.0
    n_eff = effective_trials(trial_returns_matrix)
    sr_star_period = expected_max_sharpe(n_eff, var_sr)
    return psr(r, sr_star_annual=sr_star_period * np.sqrt(ANNUAL))


def dm_test(r_a: np.ndarray, r_b: np.ndarray, lag: int = 5) -> tuple[float, float]:
    """HAC (Newey-West) test on the daily net-return differential a - b.
    Returns (t_stat, p_value); positive t favors a."""
    d = r_a - r_b
    n = len(d)
    mu = d.mean()
    e = d - mu
    g0 = float(e @ e) / n
    s = g0
    for k in range(1, min(lag, n - 1) + 1):
        gk = float(e[k:] @ e[:-k]) / n
        s += 2 * (1 - k / (lag + 1)) * gk
    se = np.sqrt(max(s, 1e-18) / n)
    t = mu / se
    p = 2 * (1 - stats.norm.cdf(abs(t)))
    return float(t), float(p)


def mcnemar(hits_a: np.ndarray, hits_b: np.ndarray) -> tuple[float, float]:
    """Exact McNemar on disagreeing days. Returns (stat=b01, p)."""
    b01 = int(np.sum(hits_a & ~hits_b))
    b10 = int(np.sum(~hits_a & hits_b))
    n = b01 + b10
    if n == 0:
        return 0.0, 1.0
    p = 2 * min(stats.binom.cdf(min(b01, b10), n, 0.5), 0.5)
    return float(b01), float(min(p, 1.0))


def cscv_pbo(returns_matrix: np.ndarray, s_partitions: int = 8) -> float:
    """Probability of Backtest Overfitting via CSCV (Bailey et al. 2017).

    returns_matrix: (T, N) daily net returns of ALL registry trials over the
    same calendar. Partition T into S blocks; for every S/2-subset, pick the
    in-sample winner and record its out-of-sample rank.
    """
    t, n = returns_matrix.shape
    if n < 2:
        return 0.0
    blocks = np.array_split(np.arange(t), s_partitions)
    below_median = 0
    total = 0
    for combo in combinations(range(s_partitions), s_partitions // 2):
        is_idx = np.concatenate([blocks[i] for i in combo])
        oos_idx = np.concatenate([blocks[i] for i in range(s_partitions)
                                  if i not in combo])
        is_sr = returns_matrix[is_idx].mean(axis=0) / (
            returns_matrix[is_idx].std(axis=0, ddof=1) + 1e-12)
        oos_sr = returns_matrix[oos_idx].mean(axis=0) / (
            returns_matrix[oos_idx].std(axis=0, ddof=1) + 1e-12)
        winner = int(np.argmax(is_sr))
        rank = stats.rankdata(oos_sr)[winner] / n
        below_median += int(rank <= 0.5)
        total += 1
    return float(below_median / total)
