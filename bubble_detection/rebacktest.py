"""Re-run the strategy backtest from a completed run's saved out-of-sample
predictions under different exit rules — no model refitting needed.

Usage: python rebacktest.py --results results_spy --csv data/SPY_5m.csv \
                            [--cost-bps 1] [--horizon 78]
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from bubbles.data import load_yf_csv
from bubbles.strategy import backtest

C = {"episode": "#0072B2", "flip": "#E69F00", "horizon": "#009E73", "bh": "#767676"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results_spy")
    ap.add_argument("--csv", default="data/SPY_5m.csv")
    ap.add_argument("--cost-bps", type=float, default=1.0)
    ap.add_argument("--horizon", type=int, default=78)
    args = ap.parse_args()

    base = os.path.dirname(os.path.abspath(__file__))
    res = os.path.join(base, args.results)
    pred_path = os.path.join(res, "features_predictions.csv.gz")
    if not os.path.exists(pred_path):
        pred_path = pred_path[:-3]
    df = load_yf_csv(os.path.join(base, args.csv))
    P = pd.read_csv(pred_path, index_col=0, parse_dates=True)
    P = P.reindex(df.index)

    flag = P["bubble_flag"].to_numpy()
    tau = P["tau"].to_numpy()
    lb = P["tau_lb"].to_numpy()
    ub = P["tau_ub"].to_numpy()

    out, series = {}, {}
    for mode in ("episode", "flip", "horizon"):
        bt = backtest(df["Close"], flag, tau, lb, ub, cost_bps=args.cost_bps,
                      require_ci=True, exit_mode=mode, max_hold=args.horizon)
        series[mode] = bt.pop("_series")
        bt.pop("trades")
        out[mode] = bt

    # table
    cols = ["total_simple_return", "ann_return", "ann_vol", "sharpe", "sortino",
            "calmar", "max_drawdown_log", "bar_hit_rate"]
    lines = ["| exit rule | exposure | trades | win rate | PF | avg hold | total ret | ann ret | Sharpe | Sortino | Calmar | maxDD(log) |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for mode, bt in out.items():
        s = bt["strategy"]
        lines.append(
            f"| {mode} | {100*bt['exposure_frac']:.1f}% | {bt['n_trades']} | "
            f"{bt['win_rate']:.2f} | {bt.get('profit_factor', float('nan')):.2f} | "
            f"{bt.get('avg_trade_bars', float('nan')):.0f} bars | "
            f"{100*s['total_simple_return']:+.1f}% | {s['ann_return']:+.3f} | "
            f"{s['sharpe']:+.2f} | {s['sortino']:+.2f} | {s['calmar']:+.2f} | "
            f"{s['max_drawdown_log']:+.3f} |")
    bh = out["episode"]["buy_hold"]
    lines.append(f"| buy & hold | 100% | 1 | — | — | — | {100*bh['total_simple_return']:+.1f}% | "
                 f"{bh['ann_return']:+.3f} | {bh['sharpe']:+.2f} | {bh['sortino']:+.2f} | "
                 f"{bh['calmar']:+.2f} | {bh['max_drawdown_log']:+.3f} |")
    table = "\n".join(lines)

    with open(os.path.join(res, "exit_modes.md"), "w") as fh:
        fh.write(f"# Exit-rule comparison ({args.cost_bps:.0f} bps/side, 1-bar delay)\n\n{table}\n")
    with open(os.path.join(res, "exit_modes.json"), "w") as fh:
        json.dump(out, fh, indent=2, default=float)

    fig, ax = plt.subplots(figsize=(11, 4.2))
    eq0 = series["episode"]
    ax.plot(eq0.index, eq0["bh"], color=C["bh"], lw=1.2, label="buy & hold")
    for mode in ("episode", "flip", "horizon"):
        ax.plot(series[mode].index, series[mode]["strat"], color=C[mode], lw=1.3,
                label=f"{mode} exit")
    ax.grid(True, alpha=0.25, linewidth=0.6)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.set_title("Cumulative OOS log return by exit rule vs buy & hold")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(res, "exit_modes.png"), dpi=140)
    print(table)


if __name__ == "__main__":
    main()
