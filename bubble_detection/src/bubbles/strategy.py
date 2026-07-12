"""Event-driven strategy backtest: trade only when the system has a *good*
signal, hold through the episode, compare against buy-and-hold.

Rules (all information available at decision time; positions execute on the
NEXT bar's close — one full bar of delay):

  enter/hold  pos = sign(tau_hat)   when  bubble_flag == 1
                                    and   tau_hat is out-of-sample (test fold)
                                    and   the signal is "good":
                                            90% CI excludes 0  (default)
                                            and |tau_hat| >= min_abs_tau
  flat        otherwise.

Costs are charged per unit of turnover (|position change| * cost_bps).
Because tau_hat exists only on walk-forward test bars, every position in the
backtest is out-of-sample.  Benchmarks over the same evaluation span:
  * buy & hold
  * long-whenever-flagged (episode timing without the causal direction) —
    isolates the value the causal forest adds on top of the PSY detector.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def bars_per_year(index: pd.DatetimeIndex) -> float:
    span_years = (index[-1] - index[0]).total_seconds() / (365.25 * 24 * 3600)
    return (len(index) - 1) / max(span_years, 1e-9)


def _equity_stats(r: np.ndarray, bpy: float, active: np.ndarray | None = None) -> dict:
    r = np.asarray(r, dtype=float)
    mu, sd = r.mean(), r.std(ddof=1) if r.size > 2 else np.nan
    downside = float(np.sqrt(np.mean(np.minimum(r, 0.0) ** 2)))
    eq = np.cumsum(r)
    dd = eq - np.maximum.accumulate(eq)
    mdd = float(dd.min()) if r.size else 0.0
    ann = float(mu * bpy)
    act = r[active] if active is not None else r
    act = act[act != 0.0] if active is None else act
    return {
        "total_log_return": float(eq[-1]) if r.size else 0.0,
        "total_simple_return": float(np.expm1(eq[-1])) if r.size else 0.0,
        "ann_return": ann,
        "ann_vol": float(sd * np.sqrt(bpy)),
        "sharpe": float(mu / sd * np.sqrt(bpy)) if sd and np.isfinite(sd) and sd > 0 else np.nan,
        "sortino": float(mu / downside * np.sqrt(bpy)) if downside > 0 else np.nan,
        "calmar": float(ann / abs(mdd)) if mdd < 0 else np.nan,
        "max_drawdown_log": mdd,
        "bar_hit_rate": float(np.mean(act > 0)) if act.size else np.nan,
    }


def backtest(close: pd.Series, flag: np.ndarray, tau: np.ndarray,
             tau_lb: np.ndarray, tau_ub: np.ndarray,
             cost_bps: float = 2.0, require_ci: bool = True,
             min_abs_tau: float = 0.0, exit_mode: str = "episode",
             max_hold: int | None = None) -> dict:
    """exit_mode:
      'episode' — in a position only while the explosiveness flag is on and
                  the signal is good (original rule; very short holds).
      'flip'    — enter on a good signal, HOLD until a good signal of the
                  opposite sign appears (always in the market after the
                  first signal).
      'horizon' — enter on a good signal, hold up to ``max_hold`` bars
                  (refreshed by same-sign signals), flip early on an
                  opposite signal, flat when the clock runs out.
    """
    idx = close.index
    n = len(close)
    r = np.zeros(n)
    r[1:] = np.diff(np.log(close.to_numpy()))

    good = np.isfinite(tau) & (np.abs(tau) >= min_abs_tau)
    if require_ci:
        good &= (tau_lb > 0) | (tau_ub < 0)
    sig = np.where((flag == 1.0) & good, np.sign(tau), 0.0)
    sig = np.nan_to_num(sig)

    if exit_mode == "episode":
        target = sig
    elif exit_mode == "flip":
        target = np.zeros(n)
        cur = 0.0
        for t in range(n):
            if sig[t] != 0.0:
                cur = sig[t]
            target[t] = cur
    elif exit_mode == "horizon":
        hold = max_hold or 78
        target = np.zeros(n)
        cur, timer = 0.0, 0
        for t in range(n):
            if sig[t] != 0.0:
                cur, timer = sig[t], hold
            elif timer > 0:
                timer -= 1
            else:
                cur = 0.0
            target[t] = cur
    else:
        raise ValueError(f"unknown exit_mode {exit_mode!r}")

    # evaluation span = where out-of-sample predictions exist at all
    te = np.where(np.isfinite(tau))[0]
    if te.size < 10:
        return {"error": "not enough out-of-sample bars"}
    a, b = te[0], te[-1]

    pos = np.zeros(n)
    pos[1:] = target[:-1]                       # one-bar execution delay
    pos[: a + 1] = 0.0
    cost = cost_bps * 1e-4
    turn = np.abs(np.diff(np.concatenate([[0.0], pos])))
    strat_r = pos * r - turn * cost

    # long-whenever-flagged benchmark (same delay, same costs)
    lpos = np.zeros(n)
    lpos[1:] = np.where(flag[:-1] == 1.0, 1.0, 0.0)
    lpos[: a + 1] = 0.0
    lpos[np.isnan(lpos)] = 0.0
    lturn = np.abs(np.diff(np.concatenate([[0.0], lpos])))
    long_ep_r = lpos * r - lturn * cost

    sl = slice(a, b + 1)
    bpy = bars_per_year(idx[sl])

    # trades = maximal runs of constant non-zero position
    trades = []
    t0 = None
    for t in range(a, b + 2):
        cur = pos[t] if t <= b else 0.0
        prev = pos[t - 1] if t > a else 0.0
        if cur != prev:
            if prev != 0.0 and t0 is not None:
                pnl = float(np.sum(strat_r[t0:t]))
                trades.append({"entry": str(idx[t0]), "exit": str(idx[min(t, n - 1)]),
                               "side": "long" if prev > 0 else "short",
                               "bars": t - t0, "pnl_log": pnl})
            t0 = t if cur != 0.0 else None

    wins = [t["pnl_log"] for t in trades if t["pnl_log"] > 0]
    losses = [t["pnl_log"] for t in trades if t["pnl_log"] <= 0]
    out = {
        "span": [str(idx[a]), str(idx[b])],
        "bars_evaluated": int(b - a + 1),
        "bars_per_year": float(bpy),
        "cost_bps_per_side": cost_bps,
        "exposure_frac": float(np.mean(pos[sl] != 0.0)),
        "n_trades": len(trades),
        "n_long": sum(1 for t in trades if t["side"] == "long"),
        "n_short": sum(1 for t in trades if t["side"] == "short"),
        "win_rate": float(len(wins) / len(trades)) if trades else np.nan,
        "avg_trade_pnl_log": float(np.mean([t["pnl_log"] for t in trades])) if trades else np.nan,
        "avg_win_log": float(np.mean(wins)) if wins else np.nan,
        "avg_loss_log": float(np.mean(losses)) if losses else np.nan,
        "profit_factor": float(sum(wins) / abs(sum(losses))) if losses and sum(losses) < 0 else np.nan,
        "avg_trade_bars": float(np.mean([t["bars"] for t in trades])) if trades else np.nan,
        "strategy": _equity_stats(strat_r[sl], bpy, active=(pos[sl] != 0.0)),
        "buy_hold": _equity_stats(r[sl], bpy, active=np.ones(b - a + 1, dtype=bool)),
        "long_in_episode": _equity_stats(long_ep_r[sl], bpy, active=(lpos[sl] != 0.0)),
        "trades": trades,
        "_series": pd.DataFrame({"strat": np.cumsum(strat_r[sl]),
                                 "bh": np.cumsum(r[sl]),
                                 "long_ep": np.cumsum(long_ep_r[sl]),
                                 "pos": pos[sl]}, index=idx[sl]),
    }
    return out
