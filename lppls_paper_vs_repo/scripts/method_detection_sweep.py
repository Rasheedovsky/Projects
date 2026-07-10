"""Per-method bubble-detection sweep over SPY 1998-2010: can EACH algorithm
independently catch multiple bubbles?

Every SWEEP_STEP trading days, every method fits an ensemble of
SWEEP_WINDOWS window lengths; its confidence = share of fits passing the
standard qualification filters (b < 0 -> positive-bubble, b > 0 -> negative).
Output: per-method confidence timelines + a detection scorecard against the
realised extremes found by the census.

Outputs: results/method_sweep_spy.csv, results/fig_method_sweep_spy.png,
         results/method_detection_scorecard.csv
"""

import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from deep_lppls import kan, lm, mlnn, plnn
from deep_lppls.core import minmax_scale, solve_linear
from lppls.lppls import LPPLS
from spy_bubble_scan import qualify

N = 252
SWEEP_WINDOWS = [60, 100, 152, 226, 350]
SWEEP_STEP = 20
CONF_THR = 0.25
LOOKBACK = 120  # trading days before an event within which a flag counts

# realised extremes on SPY 1998-2010 (from the census + public record)
EVENTS = [
    ("2004-03-05", "pos", "2004 rally top"),
    ("2007-10-09", "pos", "GFC top"),
    ("2010-04-23", "pos", "flash-crash top"),
    ("2001-09-21", "neg", "9/11 trough"),
    ("2002-10-09", "neg", "2002 bottom"),
    ("2009-03-09", "neg", "GFC bottom"),
]

METHODS = ["NM", "LM", "M-LNN", "M-LNN-KAN", "P-LNN"]
COLORS = {"NM": "tab:red", "LM": "tab:blue", "M-LNN": "tab:orange",
          "M-LNN-KAN": "tab:cyan", "P-LNN": "tab:purple"}


def norm_classify(tc, m, w, x_scaled, t2_idx, L):
    """Qualification in normalised units; returns (sign, qualified)."""
    t_norm = np.linspace(0.0, 1.0, N)
    a, b, c1, c2 = solve_linear(t_norm, x_scaled, tc, m, w)
    tc_idx = t2_idx + (tc - 1.0) * (L - 1)
    denom = tc - 1.0
    O = (w / (2 * np.pi)) * np.log(tc / denom) if denom > 0 else np.inf
    c_abs = float(np.hypot(c1, c2))
    D = (m * abs(b)) / (w * c_abs) if c_abs > 0 else np.inf
    ok = (
        (t2_idx - min(60, 0.5 * (L - 1))) < tc_idx < (t2_idx + min(252, 0.5 * (L - 1)))
        and 0 < m < 1 and 2 < w < 15 and O > 2.5 and D > 0.5
    )
    return ("pos" if b < 0 else "neg"), bool(ok)


def main():
    df = pd.read_csv(ROOT / "data" / "SPY_daily_1998_2021.csv", parse_dates=["Date"])
    df = df[df.Date <= "2010-12-31"].sort_values("Date").reset_index(drop=True)
    df["logp"] = np.log(df["Close"])
    logp = df["logp"].values
    t_all = np.arange(len(df), dtype=float)
    model = LPPLS(observations=np.array([t_all, logp]))
    plnn_params = plnn.load_params(ROOT / "models" / "P-LNN-100K.npz")
    t_norm = np.linspace(0.0, 1.0, N)

    rows = []
    t2_grid = list(range(SWEEP_WINDOWS[-1], len(df), SWEEP_STEP))
    t_start = time.time()
    for k, t2_idx in enumerate(t2_grid):
        counts = {m: {"pos": 0, "neg": 0, "pos_q": 0, "neg_q": 0} for m in METHODS}
        for L in SWEEP_WINDOWS:
            t1_idx = t2_idx - L + 1
            if t1_idx < 0:
                continue
            obs = np.array([t_all[t1_idx : t2_idx + 1], logp[t1_idx : t2_idx + 1]])
            tc, m, w, a, b, c, c1, c2, O, D = model.fit(max_searches=25, obs=obs)
            if tc != 0:
                sign = "pos" if b < 0 else "neg"
                counts["NM"][sign] += 1
                if qualify(tc, m, w, b, c, O, D, float(t1_idx), float(t2_idx)):
                    counts["NM"][sign + "_q"] += 1

            win = logp[t1_idx : t2_idx + 1]
            x = np.interp(t_norm, np.linspace(0, 1, len(win)), win)
            x_scaled, _ = minmax_scale(x)
            seed = t2_idx * 1000 + L

            fits = {
                "LM": lm.fit_lm(t_norm, x_scaled, seed=seed),
                "M-LNN": mlnn.fit_mlnn(x_scaled, seed=seed),
                "M-LNN-KAN": kan.fit_mlnn_kan(x_scaled, seed=seed),
            }
            p = plnn.predict(plnn_params, x_scaled.astype(np.float32))[0]
            fits["P-LNN"] = dict(tc=float(p[0]), m=float(p[1]), w=float(p[2]))
            for name, r in fits.items():
                sign, ok = norm_classify(r["tc"], r["m"], r["w"], x_scaled, t2_idx, L)
                counts[name][sign] += 1
                if ok:
                    counts[name][sign + "_q"] += 1
        for name, c in counts.items():
            rows.append(dict(
                t2_idx=t2_idx, date=df.loc[t2_idx, "Date"], method=name,
                pos_conf=c["pos_q"] / c["pos"] if c["pos"] else 0.0,
                neg_conf=c["neg_q"] / c["neg"] if c["neg"] else 0.0,
            ))
        if (k + 1) % 10 == 0:
            print(f"  {k + 1}/{len(t2_grid)} ({time.time() - t_start:.0f}s)", flush=True)

    sw = pd.DataFrame(rows)
    sw.to_csv(ROOT / "results" / "method_sweep_spy.csv", index=False)

    # detection scorecard: max confidence within LOOKBACK days before event
    score_rows = []
    for date_s, sign, label in EVENTS:
        ev_idx = int((df["Date"] - pd.Timestamp(date_s)).abs().idxmin())
        col = "pos_conf" if sign == "pos" else "neg_conf"
        for name in METHODS:
            g = sw[(sw.method == name) & (sw.t2_idx >= ev_idx - LOOKBACK) & (sw.t2_idx <= ev_idx)]
            mx = float(g[col].max()) if len(g) else 0.0
            score_rows.append(dict(event=label, date=date_s, sign=sign, method=name,
                                   max_conf=round(mx, 2), detected=mx >= CONF_THR))
    sc = pd.DataFrame(score_rows)
    sc.to_csv(ROOT / "results" / "method_detection_scorecard.csv", index=False)
    print(sc.pivot_table(index=["event", "date"], columns="method", values="max_conf").round(2).to_string(), flush=True)

    # figure: price + one confidence row per method
    fig, axes = plt.subplots(len(METHODS) + 1, 1, figsize=(15, 12), sharex=True,
                             gridspec_kw={"height_ratios": [2.2] + [1] * len(METHODS), "hspace": 0.06})
    axes[0].plot(df["Date"], df["Close"], color="black", lw=0.9)
    axes[0].set_yscale("log")
    axes[0].set_ylabel("SPY (log)")
    for date_s, sign, label in EVENTS:
        c = "red" if sign == "pos" else "green"
        for ax in axes:
            ax.axvline(pd.Timestamp(date_s), color=c, ls="--", lw=0.9, alpha=0.6)
    axes[0].set_title("Per-method LPPLS bubble detection sweep, SPY 1998-2010 "
                      "(dashed: realised tops red / bottoms green)")
    for ax, name in zip(axes[1:], METHODS):
        g = sw[sw.method == name]
        ax.plot(g["date"], g["pos_conf"], color=COLORS[name], lw=1.2)
        ax.plot(g["date"], g["neg_conf"], color="green", lw=1.0, alpha=0.7)
        ax.axhline(CONF_THR, color="grey", ls=":", lw=0.8)
        ax.set_ylim(0, 1)
        ax.set_ylabel(name, fontsize=9)
        ax.grid(alpha=0.2)
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    fig.savefig(ROOT / "results" / "fig_method_sweep_spy.png", dpi=140, bbox_inches="tight")
    print("figure saved", flush=True)


if __name__ == "__main__":
    main()
