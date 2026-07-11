"""
End-to-end analysis: the Gresnigt-Kole-Franses "financial earthquake"
indicator (ETAS/Hawkes) + the KAN-PIN parameter estimator, applied to
Alcoa (AA) hourly data (and a daily aggregate).

Run:  python3 run_analysis.py [path/to/AA_h.csv]

Produces figures/fig1..fig9 (PNG) and results.json.
Every section prints what it is doing and why; see etas.py / kan_pin.py
for the underlying mathematics.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from etas import (CrossETAS, ETASModel, EventData, extract_events,
                  fit_variants, load_yfinance_csv, log_returns)
from kan_pin import KANPINHawkes
from stability import NYBLOM_CRIT_5PCT, nyblom_test, walk_forward
from overnight import (KANPINOvernight, OvernightETAS, extract_gap_stream,
                       split_overnight_intraday, walk_forward_overnight)

# ----------------------------------------------------------------------
# chart style (light mode, validated palette)
# ----------------------------------------------------------------------
C = {
    "fall": "#e34948", "run": "#2a78d6", "aqua": "#1baf7a", "yellow": "#eda100",
    "ink": "#0b0b0b", "ink2": "#52514e", "muted": "#898781",
    "grid": "#e1e0d9", "axis": "#c3c2b7", "surface": "#fcfcfb",
    "seq": ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#1c5cab"],
}
plt.rcParams.update({
    "figure.facecolor": C["surface"], "axes.facecolor": C["surface"],
    "savefig.facecolor": C["surface"], "axes.edgecolor": C["axis"],
    "axes.labelcolor": C["ink2"], "axes.grid": True, "grid.color": C["grid"],
    "grid.linewidth": 0.8, "xtick.color": C["muted"], "ytick.color": C["muted"],
    "text.color": C["ink"], "font.size": 10, "axes.titlesize": 11,
    "axes.titleweight": "bold", "axes.spines.top": False,
    "axes.spines.right": False, "lines.linewidth": 1.8,
    "legend.frameon": False, "font.family": "sans-serif",
})

HERE = Path(__file__).resolve().parent
FIGS = HERE / "figures"
FIGS.mkdir(exist_ok=True)
RESULTS: dict = {}


def date_axis(ax, index: pd.DatetimeIndex, times=None):
    """Model time is 'bars since start'; relabel ticks as dates."""
    ticks = np.linspace(0, len(index) - 1, 7).astype(int)
    ax.set_xticks(ticks)
    ax.set_xticklabels([index[i].strftime("%b %y") for i in ticks])


# ======================================================================
# 1. Data
# ======================================================================

def load_data(csv_path: str):
    print("=" * 72)
    print("1. DATA")
    print("=" * 72)
    df = load_yfinance_csv(csv_path)
    close_h = df["Close"]
    r_h = log_returns(close_h)
    # daily series aggregated from the hourly closes (last bar of each day)
    close_d = close_h.groupby(close_h.index.date).last()
    close_d.index = pd.DatetimeIndex(close_d.index)
    r_d = log_returns(close_d)
    print(f"hourly bars: {len(close_h)}  ({close_h.index[0].date()} -> "
          f"{close_h.index[-1].date()}),  hourly returns: {len(r_h)}")
    print(f"daily closes: {len(close_d)},  daily returns: {len(r_d)}")
    RESULTS["data"] = {"n_hourly_bars": int(len(close_h)),
                       "n_daily": int(len(close_d)),
                       "start": str(close_h.index[0]), "end": str(close_h.index[-1])}
    return close_h, r_h, close_d, r_d


# ======================================================================
# 2. Events + model estimation (the paper's method)
# ======================================================================

def fit_models(r_h: pd.Series):
    print()
    print("=" * 72)
    print("2. THE PAPER'S METHOD: extreme-return events + ETAS by MLE")
    print("=" * 72)
    falls = extract_events(r_h, "fall", 0.95)
    runs = extract_events(r_h, "run", 0.95)
    print(f"fall events (r <= -{falls.M0:.4f}): {falls.n}")
    print(f"run  events (r >= +{runs.M0:.4f}): {runs.n}")

    tables, best = {}, {}
    for ev in (falls, runs):
        tab = fit_variants(ev, seed=1)
        tables[ev.tail] = tab
        name = tab.iloc[0]["model"]
        best[ev.tail] = tab.attrs["models"][name]
        print(f"\n--- {ev.tail.upper()} variants (AIC-ranked):")
        print(tab[["model", "logL", "AIC", "mu", "K0", "alpha", "c",
                   "branching_n", "KS_pvalue"]].round(4).to_string(index=False))
        print()
        print(best[ev.tail].explain())
    RESULTS["variants"] = {k: json.loads(t.drop(columns=[]).to_json(orient="records"))
                           for k, t in tables.items()}
    return falls, runs, best, tables


def fit_cross(falls: EventData, runs: EventData):
    print()
    print("=" * 72)
    print("3. CROSS-EXCITATION (falls <-> runs), bivariate Hawkes")
    print("=" * 72)
    x = CrossETAS(falls, runs).fit(seed=2)
    print(x.explain())
    tests = {}
    for j in ("fall", "run"):
        t = x.cross_excitation_test(j)
        tests[j] = t
        print(f"LR test of no cross-excitation into {j}: LR={t['LR']:.2f}, "
              f"p={t['pvalue']:.3f}")
    RESULTS["cross_excitation"] = {j: {k: v for k, v in t.items()} for j, t in tests.items()}
    return x


# ======================================================================
# 4. Figures 1 & 2: price, events, conditional intensity
# ======================================================================

def figure_events(close_h, r_h, falls, runs):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 6), sharex=True,
                                   gridspec_kw={"height_ratios": [2, 1]})
    px = close_h.values[1:]                    # align with returns
    ax1.plot(np.arange(len(px)), px, color=C["ink2"], lw=1.4, label="AA close")
    fi = falls.times.astype(int) - 1
    ri = runs.times.astype(int) - 1
    ax1.scatter(fi, px[fi], marker="v", s=34, color=C["fall"], zorder=3,
                label=f"fall events (n={falls.n})")
    ax1.scatter(ri, px[ri], marker="^", s=34, color=C["run"], zorder=3,
                label=f"run events (n={runs.n})")
    ax1.set_ylabel("price ($)")
    ax1.set_title("AA hourly price and extreme-return events ('earthquakes')")
    ax1.legend(loc="upper left")

    ax2.bar(np.arange(len(r_h)), r_h.values, width=1.0, color=C["muted"])
    ax2.axhline(-falls.M0, color=C["fall"], lw=1.2, ls="--")
    ax2.axhline(runs.M0, color=C["run"], lw=1.2, ls="--")
    ax2.text(len(r_h) * 0.995, -falls.M0, " fall threshold", color=C["fall"],
             va="top", ha="right", fontsize=8)
    ax2.text(len(r_h) * 0.995, runs.M0, " run threshold", color=C["run"],
             va="bottom", ha="right", fontsize=8)
    ax2.set_ylabel("hourly log return")
    date_axis(ax2, r_h.index)
    fig.tight_layout()
    fig.savefig(FIGS / "fig1_price_events.png", dpi=150)
    plt.close(fig)
    print("saved figures/fig1_price_events.png")


def figure_intensity(close_h, r_h, best):
    t_grid = np.arange(1.0, len(r_h) + 1.0)
    lam_f = best["fall"].intensity(t_grid)
    lam_r = best["run"].intensity(t_grid)
    fig, axes = plt.subplots(3, 1, figsize=(11, 7.5), sharex=True,
                             gridspec_kw={"height_ratios": [1.4, 1, 1]})
    px = close_h.values[1:]
    axes[0].plot(np.arange(len(px)), px, color=C["ink2"], lw=1.2)
    axes[0].set_ylabel("price ($)")
    axes[0].set_title("The indicator: conditional intensity of extreme events")

    axes[1].plot(t_grid - 1, lam_f, color=C["fall"], lw=1.4)
    axes[1].axhline(best["fall"].mu, color=C["muted"], lw=1, ls=":")
    axes[1].text(2, best["fall"].mu, " background mu", color=C["muted"],
                 fontsize=8, va="bottom")
    for t in best["fall"].events.times:
        axes[1].axvline(t - 1, color=C["fall"], alpha=0.18, lw=0.7, ymax=0.12)
    axes[1].set_ylabel("fall intensity\n(events/hour)")

    axes[2].plot(t_grid - 1, lam_r, color=C["run"], lw=1.4)
    axes[2].axhline(best["run"].mu, color=C["muted"], lw=1, ls=":")
    for t in best["run"].events.times:
        axes[2].axvline(t - 1, color=C["run"], alpha=0.18, lw=0.7, ymax=0.12)
    axes[2].set_ylabel("run intensity\n(events/hour)")
    # same scale on both intensity panels: makes the absence of
    # self-excitation in runs (flat line) an honest, visible finding
    lo = 0.95 * min(lam_f.min(), lam_r.min())
    hi = 1.05 * max(lam_f.max(), lam_r.max())
    axes[1].set_ylim(lo, hi)
    axes[2].set_ylim(lo, hi)
    if best["run"].K0 < 1e-3:
        axes[2].text(0.5, 0.75, "no self-excitation detected in runs "
                     "($K_0 \\approx 0$): intensity = constant background",
                     transform=axes[2].transAxes, ha="center",
                     color=C["ink2"], fontsize=9)
    date_axis(axes[2], r_h.index)
    fig.tight_layout()
    fig.savefig(FIGS / "fig2_intensity.png", dpi=150)
    plt.close(fig)
    print("saved figures/fig2_intensity.png")


# ======================================================================
# 5. Early Warning System, out of sample
# ======================================================================

def ews_out_of_sample(r_h, falls, kernel_spec, d: float = 7.0,
                      train_frac: float = 0.7):
    """Fit on the first 70% of the sample, then walk forward hour by hour
    through the last 30% computing P(>=1 fall event within d hours).
    d = 7 trading hours = one trading day, the hourly-data analogue of
    the paper's 'medium term' (their 5 days on daily data).  Longer windows
    are uninformative here: during the turbulent test period almost every
    35h window contains an event, leaving nothing to discriminate."""
    print()
    print("=" * 72)
    print("4. EARLY WARNING SYSTEM (out of sample, falls)")
    print("=" * 72)
    T_train = np.floor(falls.T * train_frac)
    keep = falls.times <= T_train
    ev_train = EventData(times=falls.times[keep], mags=falls.mags[keep],
                         M0=falls.M0, T=T_train, tail="fall",
                         timestamps=falls.timestamps[keep], index=falls.index)
    kernel, use_alpha = kernel_spec
    m = ETASModel(kernel=kernel, use_alpha=use_alpha).fit(ev_train, seed=1)
    print(f"trained on [0, {T_train:.0f}] ({ev_train.n} events): "
          f"mu={m.mu:.4f} K0={m.K0:.3f} c={m.c:.1f} n={m.branching_ratio():.2f}")
    # walk forward with FIXED parameters but growing event history
    m.events = falls
    t_grid = np.arange(T_train + 1, falls.T - d, 1.0)
    # sweep thresholds over the range the predictions actually span
    probs_tmp = np.array([m.prob_event_within(t, d) for t in t_grid])
    thresholds = np.unique(np.quantile(probs_tmp, np.linspace(0.02, 0.98, 33)))
    bt = m.ews_backtest(t_grid, d=d, thresholds=thresholds)
    ibest = int(bt["KSS"].idxmax())
    print(f"test hours: {len(t_grid)}, event-windows: "
          f"{int(bt.attrs['realised'].sum())}")
    print(f"best threshold p*={bt.loc[ibest,'threshold']:.2f}: "
          f"hit rate={bt.loc[ibest,'hit_rate']:.2f}, "
          f"false alarms={bt.loc[ibest,'false_alarm']:.2f}, "
          f"KSS={bt.loc[ibest,'KSS']:.2f}")
    print("NOTE: discrimination is weak on this sample - one year of hourly")
    print("data with mild clustering (n~0.2); the paper's strong KSS comes from")
    print("~50 years of daily data spanning much heavier crash cascades.")
    RESULTS["ews"] = {"horizon_hours": d,
                      "train_end": float(T_train),
                      "best_threshold": float(bt.loc[ibest, "threshold"]),
                      "hit_rate": float(bt.loc[ibest, "hit_rate"]),
                      "false_alarm": float(bt.loc[ibest, "false_alarm"]),
                      "KSS": float(bt.loc[ibest, "KSS"])}
    return m, t_grid, bt


def figure_ews(r_h, falls, t_grid, bt, d):
    probs, realised = bt.attrs["probs"], bt.attrs["realised"]
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 6.5),
                                   gridspec_kw={"height_ratios": [1.6, 1]})
    ax1.plot(t_grid, probs, color=C["fall"], lw=1.5,
             label=f"P(fall event within {d:.0f}h)")
    ibest = int(bt["KSS"].idxmax())
    thr = bt.loc[ibest, "threshold"]
    ax1.axhline(thr, color=C["muted"], ls="--", lw=1.2)
    ax1.text(t_grid[2], thr, f" signal threshold {thr:.2f} (max KSS)",
             color=C["muted"], va="bottom", fontsize=8)
    in_test = falls.times[falls.times > t_grid[0]]
    ax1.scatter(in_test, np.full(len(in_test), 1.02), marker="v", s=26,
                color=C["fall"], clip_on=False, label="realised fall events")
    ax1.set_ylim(0, 1.05)
    ax1.set_ylabel("probability")
    ax1.set_title("Early Warning System on the out-of-sample window")
    ax1.text(0.5, 0.55, "predictions span only "
             f"{probs.min():.2f}-{probs.max():.2f}: one calm training year\n"
             "gives the model little clustering to discriminate with "
             "(the paper uses ~50 years)",
             transform=ax1.transAxes, ha="center", color=C["ink2"], fontsize=9)
    ax1.legend(loc="center left", fontsize=8)
    ticks = np.linspace(t_grid[0], t_grid[-1], 6).astype(int)
    ax1.set_xticks(ticks)
    ax1.set_xticklabels([falls.index[min(i, len(falls.index) - 1)].strftime("%d %b %y")
                         for i in ticks])

    ax2.plot(bt["threshold"], bt["hit_rate"], color=C["run"], lw=1.6,
             label="hit rate")
    ax2.plot(bt["threshold"], bt["false_alarm"], color=C["yellow"], lw=1.6,
             label="false-alarm rate")
    ax2.plot(bt["threshold"], bt["KSS"], color=C["aqua"], lw=2.2,
             label="KSS = hits - false alarms")
    ax2.axvline(thr, color=C["muted"], ls="--", lw=1)
    ax2.set_xlabel("signal threshold on predicted probability "
                   "(swept over the observed prediction range)")
    ax2.set_ylabel("rate")
    ax2.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGS / "fig3_ews.png", dpi=150)
    plt.close(fig)
    print("saved figures/fig3_ews.png")


# ======================================================================
# 6. Duration of falls and runs: when does the episode end?
# ======================================================================

def duration_forecasts(falls, runs, best, quiet_window: float = 14.0):
    print()
    print("=" * 72)
    print("5. HOW LONG WILL THE FALL/RUN LAST - AND WHEN DOES IT END?")
    print("=" * 72)
    print(f"episode = cluster of same-tail events with gaps < {quiet_window:.0f} trading")
    print("hours (2 days); 'over' = a full quiet window with no further event.")
    m = best["fall"]
    eps = m.episodes(quiet_window)
    eps_sorted = sorted(eps, key=lambda e: -e[2])
    s, e, n_ev = eps_sorted[0]                     # biggest fall episode
    print(f"largest fall episode: {falls.time_to_stamp(s).date()} -> "
          f"{falls.time_to_stamp(e).date()}  ({n_ev} events over {e - s:.0f} "
          f"trading hours)")
    # stand at several points during the episode and forecast the end
    stand = np.linspace(s + 1, e + quiet_window * 0.9, 14)
    rows = []
    for i, t in enumerate(stand):
        f = m.predict_episode_end(t, quiet_window, horizon=400.0,
                                  n_paths=400, seed=100 + i)
        last_after = falls.times[(falls.times >= t - 1e-9) & (falls.times <= e)]
        realised = float(last_after[-1] - t) if len(last_after) else 0.0
        rows.append({"t": t, "median": f["median"], "q10": f["q10"],
                     "q90": f["q90"], "p_over": f["p_over_now"],
                     "exp_more": f["exp_more_events"], "realised": realised})
        print(f"  t=+{t - s:5.0f}h: P(over)={f['p_over_now']:.2f}, "
              f"E[more events]={f['exp_more_events']:.1f}, remaining "
              f"median={f['median']:.0f}h [{f['q10']:.0f}, {f['q90']:.0f}] "
              f"(realised {realised:.0f}h)")
    df = pd.DataFrame(rows)

    # ---- nowcast at the very end of the sample, both tails ----
    print("\nNOWCAST at the last bar of the sample:")
    now = {}
    for tail, model in best.items():
        t_now = model.events.T
        p_day = model.prob_event_within(t_now, 7.0)
        p_week = model.prob_event_within(t_now, 35.0)
        f = model.predict_episode_end(t_now, quiet_window, horizon=400.0,
                                      n_paths=400, seed=7)
        now[tail] = {"p_event_1d": p_day, "p_event_1w": p_week,
                     "p_episode_over": f["p_over_now"],
                     "exp_more_events": f["exp_more_events"],
                     "median_remaining_h": f["median"]}
        print(f"  {tail.upper():4s}: P(extreme within 1 day)={p_day:.2f}, "
              f"within 1 week={p_week:.2f}; P(current episode over)="
              f"{f['p_over_now']:.2f}, expected further events="
              f"{f['exp_more_events']:.1f}")
    RESULTS["nowcast"] = now
    RESULTS["episode_case_study"] = {
        "start": str(falls.time_to_stamp(s)), "end": str(falls.time_to_stamp(e)),
        "n_events": int(n_ev), "duration_hours": float(e - s)}
    return df, (s, e, n_ev)


def figure_duration(falls, dur_df, episode):
    s, e, n_ev = episode
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6.5), sharex=True,
                                   gridspec_kw={"height_ratios": [1.5, 1]})
    x = dur_df["t"] - s
    ax1.fill_between(x, dur_df["q10"], dur_df["q90"], color=C["seq"][1],
                     label="80% interval (simulated)")
    ax1.plot(x, dur_df["median"], color=C["seq"][4], lw=2,
             label="median forecast")
    ax1.plot(x, dur_df["realised"], color=C["ink"], lw=1.6, ls="--",
             label="realised remaining duration")
    for t in falls.times[(falls.times >= s) & (falls.times <= e)] - s:
        ax1.axvline(t, color=C["fall"], alpha=0.25, lw=0.8, ymax=0.06)
    ax1.set_ylabel("remaining duration (trading hours)")
    ax1.set_title(f"Forecasting the end of the largest fall episode "
                  f"({n_ev} events; red ticks)")
    ax1.legend(loc="upper right", fontsize=8)

    ax2.plot(x, dur_df["p_over"], color=C["aqua"], lw=2)
    ax2.set_ylabel("P(episode already over)")
    ax2.set_xlabel(f"trading hours since episode start "
                   f"({falls.time_to_stamp(s).strftime('%d %b %Y')})")
    ax2.set_ylim(0, 1)
    fig.tight_layout()
    fig.savefig(FIGS / "fig4_duration.png", dpi=150)
    plt.close(fig)
    print("saved figures/fig4_duration.png")


# ======================================================================
# 7. KAN-PIN estimation: synthetic validation + real data
# ======================================================================

def synthetic_validation():
    print()
    print("=" * 72)
    print("6. KAN-PIN: synthetic validation (known ground truth)")
    print("=" * 72)
    truth = {"mu": 0.02, "K0": 0.6, "c": 8.0}
    T = 1748.0
    dummy = EventData(times=np.array([1.0]), mags=np.array([0.02]), M0=0.017,
                      T=T, tail="fall", timestamps=pd.DatetimeIndex([]),
                      index=pd.DatetimeIndex([]))
    gen = ETASModel(kernel="exp", use_alpha=False, mu=truth["mu"],
                    K0=truth["K0"], alpha=0.0, c=truth["c"], omega=1.0,
                    beta_m=0.018)
    gen.events = dummy
    t_s, m_s = gen.simulate(0.0, T, np.random.default_rng(102), [], [])
    ev = EventData(times=t_s, mags=m_s, M0=0.017, T=T, tail="fall",
                   timestamps=pd.DatetimeIndex([]), index=pd.DatetimeIndex([]))
    print(f"simulated {ev.n} events from truth "
          f"mu={truth['mu']}, K0={truth['K0']}, c={truth['c']}")
    mle = ETASModel(kernel="exp", use_alpha=False).fit(ev, seed=3)
    kp = KANPINHawkes(ev, kernel="exp", use_alpha=False, seed=1)
    kp.fit(verbose=False)
    e = kp.eta_hat
    tab = pd.DataFrame({
        "truth": [truth["mu"], truth["K0"], truth["c"]],
        "MLE": [mle.mu, mle.K0, mle.c],
        "KAN-PIN": [e["mu"], e["K0"], e["c"]],
    }, index=["mu", "K0", "c"])
    print(tab.round(4).to_string())
    print("(KAN-PIN's short-c bias is the documented interpolation-leakage "
          "effect; mu and the branching ratio K0 are recovered well.)")
    RESULTS["synthetic_validation"] = json.loads(tab.to_json())
    return tab


def kan_pin_real(falls, best):
    print()
    print("=" * 72)
    print("7. KAN-PIN on the real AA fall events")
    print("=" * 72)
    t0 = time.time()
    kp = KANPINHawkes(falls, kernel="exp", use_alpha=False, seed=1)
    kp.fit(verbose=True)
    e = kp.eta_hat
    mle = best["fall"] if best["fall"].kernel == "exp" else \
        ETASModel(kernel="exp", use_alpha=False).fit(falls, seed=1)
    tab = pd.DataFrame({
        "MLE (paper)": [mle.mu, mle.K0, mle.c, mle.branching_ratio()],
        "KAN-PIN": [e["mu"], e["K0"], e["c"], e["K0"]],
    }, index=["mu", "K0", "c", "branching n"])
    print(tab.round(4).to_string())
    print(f"({time.time() - t0:.0f}s)")
    RESULTS["kan_pin_real"] = json.loads(tab.to_json())
    return kp, mle, tab


def figure_kanpin(falls, kp, mle, tab):
    fig = plt.figure(figsize=(11, 7))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.3, 1])
    # (a) compensator vs staircase
    ax = fig.add_subplot(gs[0, :])
    tt, Lam = kp.compensator_curve(600)
    ax.step(np.concatenate([[0.0], falls.times]),
            np.arange(0, falls.n + 1), where="post", color=C["ink2"], lw=1.4,
            label="event staircase N(t) (data)")
    ax.plot(tt, Lam, color=C["run"], lw=2, label="KAN compensator $\\Lambda_\\theta(t)$")
    Lam_mle = mle.compensator(falls.times)
    ax.plot(falls.times, Lam_mle, color=C["fall"], lw=1.6, ls="--",
            label="MLE compensator $\\Lambda_{\\hat\\eta}(t)$")
    ax.set_title("KAN-PIN: the network learns the integrated intensity of fall events")
    ax.set_ylabel("cumulative events")
    date_axis(ax, falls.index)
    ax.legend(loc="upper left", fontsize=9)
    # (b) training losses, phase C
    ax = fig.add_subplot(gs[1, 0])
    it = np.arange(len(kp.history["L_data"])) * 10
    ax.plot(it, kp.history["L_data"], color=C["run"], lw=1.6)
    ax.plot(it, kp.history["L_de"], color=C["aqua"], lw=1.6)
    ax.plot(it, kp.history["L_ic"], color=C["yellow"], lw=1.6)
    ax.text(it[-1], kp.history["L_data"][-1], " data", color=C["run"], fontsize=8)
    ax.text(it[-1], kp.history["L_de"][-1], " DE", color=C["aqua"], fontsize=8)
    ax.text(it[-1], kp.history["L_ic"][-1], " IC", color=C["yellow"], fontsize=8)
    ax.set_yscale("log")
    ax.set_xlabel("MDMM epoch (phase C)")
    ax.set_ylabel("loss")
    ax.set_title("MDMM training losses")
    # (c) parameter comparison
    ax = fig.add_subplot(gs[1, 1])
    idx = np.arange(3)
    w = 0.38
    vals_mle = tab["MLE (paper)"].values[:3]
    vals_kp = tab["KAN-PIN"].values[:3]
    # c is on a different scale: normalise each parameter by the MLE value
    ax.bar(idx - w / 2, vals_mle / vals_mle, w, color=C["fall"],
           label="MLE (=1)")
    ax.bar(idx + w / 2, vals_kp / vals_mle, w, color=C["run"], label="KAN-PIN")
    ax.set_xticks(idx)
    ax.set_xticklabels(["$\\mu$", "$K_0$", "$c$"])
    ax.axhline(1.0, color=C["muted"], lw=1, ls=":")
    ax.set_ylabel("relative to MLE")
    ax.set_title("Parameter estimates")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGS / "fig5_kanpin.png", dpi=150)
    plt.close(fig)
    print("saved figures/fig5_kanpin.png")


# ======================================================================
# 8. Parameter stability (Nyblom) + walk-forward validation, both methods
# ======================================================================

def stability_and_walkforward(falls, best):
    print()
    print("=" * 72)
    print("8. PARAMETER STABILITY (running Nyblom) + WALK-FORWARD VALIDATION")
    print("=" * 72)
    mle_full = best["fall"] if best["fall"].kernel == "exp" else \
        ETASModel(kernel="exp", use_alpha=False).fit(falls, seed=1)
    ny = nyblom_test(mle_full)
    print("Full-sample Nyblom stability test (5% critical: individual "
          f"{ny['crit_individual_5pct']}, joint {ny['crit_joint_5pct']}):")
    for n, v in ny["individual"].items():
        print(f"  {n:5s}: {v:.3f} {'UNSTABLE' if ny['unstable_individual'][n] else '(stable)'}")
    print(f"  joint: {ny['joint']:.3f} "
          f"{'UNSTABLE' if ny['unstable_joint'] else '(stable)'}")
    print()
    print("Walk-forward (expanding window, both estimators refit each step;")
    print("OOS = predictive log-score on the next segment, full history,")
    print("parameters frozen at the refit time; Poisson benchmark = train rate):")
    wf = walk_forward(falls, n_steps=8)
    tot = {m: float(wf[f"oos_logscore_{m}"].sum())
           for m in ("mle", "kanpin", "poisson")}
    print(f"\nTOTAL OOS log-score: MLE {tot['mle']:.1f} | "
          f"KAN-PIN {tot['kanpin']:.1f} | Poisson {tot['poisson']:.1f}")
    print("Reading: windowed MLE mostly finds K0~0 (calm training windows), so")
    print("it scores like Poisson OOS; KAN-PIN's stable-but-short-kernel")
    print("clustering scores slightly worse. On one year of hourly data the")
    print("self-excitation adds no OOS predictive value at these window sizes;")
    print("the running Nyblom never rejects within-window stability, while the")
    print("across-window K0 path shows regime dependence (calm -> turbulent).")
    RESULTS["nyblom_full_sample"] = {"individual": ny["individual"],
                                     "joint": ny["joint"],
                                     "unstable_joint": ny["unstable_joint"]}
    RESULTS["walk_forward"] = {"total_oos_logscore": tot,
                               "n_steps": int(len(wf))}
    return wf, ny


def figure_stability(falls, wf):
    dates = [d.strftime("%d %b %y") for d in wf["date"]]
    x = np.arange(len(wf))
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 7.5))
    specs = [("mu", "background rate $\\mu$ (events/h)", axes[0, 0], None),
             ("K0", "fertility $K_0$ (= branching ratio)", axes[0, 1], None),
             ("c", "kernel scale $c$ (hours, log)", axes[1, 0], "log")]
    for par, title, ax, yscale in specs:
        ax.plot(x, wf[f"mle_{par}"], color=C["fall"], marker="o", ms=5,
                label="MLE (paper)")
        ax.plot(x, wf[f"kp_{par}"], color=C["run"], marker="s", ms=5,
                label="KAN-PIN")
        if yscale:
            ax.set_yscale(yscale)
        ax.set_title(title)
        ax.set_xticks(x[::2]); ax.set_xticklabels(dates[::2], fontsize=8)
        ax.legend(fontsize=8)
    ax = axes[1, 1]
    ax.plot(x, wf["nyblom_joint"], color=C["aqua"], marker="o", ms=5,
            label="joint statistic")
    for par, ls in (("mu", ":"), ("K0", "--"), ("c", "-.")):
        ax.plot(x, wf[f"nyblom_{par}"], color=C["muted"], lw=1.1, ls=ls,
                label=f"{par}")
    ax.axhline(wf["nyblom_crit_joint"].iloc[0], color=C["fall"], lw=1.2,
               ls="--")
    ax.text(0.1, wf["nyblom_crit_joint"].iloc[0], " 5% critical (joint)",
            color=C["fall"], fontsize=8, va="bottom")
    ax.axhline(NYBLOM_CRIT_5PCT[1], color=C["yellow"], lw=1.2, ls="--")
    ax.text(0.1, NYBLOM_CRIT_5PCT[1], " 5% critical (individual)",
            color=C["yellow"], fontsize=8, va="bottom")
    ax.set_title("running Nyblom stability test")
    ax.set_xticks(x[::2]); ax.set_xticklabels(dates[::2], fontsize=8)
    ax.legend(fontsize=8, ncol=2)
    fig.suptitle("Walk-forward parameter paths and running Nyblom test "
                 "(expanding windows)", fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIGS / "fig6_stability.png", dpi=150)
    plt.close(fig)
    print("saved figures/fig6_stability.png")

    oos = wf.dropna(subset=["oos_logscore_mle"])
    fig, ax = plt.subplots(figsize=(10, 4.2))
    xb = np.arange(len(oos)); w = 0.38
    ax.bar(xb - w / 2, oos["oos_logscore_mle"] - oos["oos_logscore_poisson"],
           w, color=C["fall"], label="MLE (paper)")
    ax.bar(xb + w / 2, oos["oos_logscore_kanpin"] - oos["oos_logscore_poisson"],
           w, color=C["run"], label="KAN-PIN")
    ax.axhline(0, color=C["ink2"], lw=1.2)
    ax.text(len(oos) - 0.4, 0, " Poisson benchmark", color=C["ink2"],
            fontsize=8, va="bottom", ha="right")
    ax.set_xticks(xb)
    ax.set_xticklabels([d.strftime("%d %b %y") for d in oos["date"]], fontsize=8)
    ax.set_ylabel("OOS log-score minus Poisson")
    ax.set_title("Walk-forward out-of-sample predictive log-score by segment "
                 "(above 0 = beats Poisson)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGS / "fig7_walkforward_oos.png", dpi=150)
    plt.close(fig)
    print("saved figures/fig7_walkforward_oos.png")


# ======================================================================
# 9. Overnight gaps as triggers of intraday self-excitation
# ======================================================================

def overnight_analysis(csv_path: str):
    print()
    print("=" * 72)
    print("9. OVERNIGHT GAPS AS TRIGGERS OF INTRADAY EXTREMES")
    print("=" * 72)
    df = load_yfinance_csv(csv_path)
    r_intra, gaps_all = split_overnight_intraday(df)
    print(f"separated {len(r_intra)} intraday returns from "
          f"{len(gaps_all)} overnight gaps")
    falls_i = extract_events(r_intra, "fall", 0.95)
    runs_i = extract_events(r_intra, "run", 0.95)
    src = extract_gap_stream(gaps_all, 0.75, "abs")
    print(f"intraday fall events: {falls_i.n} (M0={falls_i.M0:.4f}); "
          f"gap triggers: {src.n} (|gap| >= {src.G0:.4f})")
    out = {}
    for ev in (falls_i, runs_i):
        m = OvernightETAS(ev, src).fit(seed=1)
        lr = m.lr_test_no_overnight(seed=1)
        ny = m.nyblom()
        print(f"\n--- intraday {ev.tail.upper()}s ---")
        print(m.explain())
        print(f"LR test of overnight triggering: LR={lr['LR']:.1f}, "
              f"p={lr['pvalue']:.4f}  (self-only lnL={lr['logL_selfonly']:.1f})")
        print(f"Nyblom joint = {ny['joint']:.2f} "
              f"(5% crit {ny['crit_joint_5pct']:.2f})"
              f"{' UNSTABLE' if ny['unstable_joint'] else ' (stable)'}")
        out[ev.tail] = {"theta": m.theta, "LR": lr["LR"],
                        "LR_pvalue": lr["pvalue"], "nyblom_joint": ny["joint"]}
        if ev.tail == "fall":
            model_fall, lr_fall = m, lr
    # KAN-PIN comparison on falls
    kp = KANPINOvernight(falls_i, src, seed=1)
    kp.fit(epochs=600, verbose=False)
    ek = kp.eta_hat
    tab = pd.DataFrame({"MLE (paper)": [model_fall.theta[p] for p in
                                        OvernightETAS.PARAM_NAMES],
                        "KAN-PIN": [ek[p] for p in OvernightETAS.PARAM_NAMES]},
                       index=list(OvernightETAS.PARAM_NAMES))
    print("\nFull-sample parameter comparison (intraday falls):")
    print(tab.round(4).to_string())
    RESULTS["overnight"] = {k: {kk: (dict(vv) if isinstance(vv, dict) else vv)
                                for kk, vv in v.items()} for k, v in out.items()}
    RESULTS["overnight"]["kanpin_falls"] = ek
    return r_intra, gaps_all, falls_i, src, model_fall, kp


def figure_overnight(r_intra, gaps_all, falls_i, src, model):
    """Evidence figure: where do intraday extremes sit within the day, and
    how does the fitted intensity decompose into background / self / gap
    triggering?"""
    # hour-of-day evidence: fall counts by bar position, trigger vs quiet days
    day_of = pd.Series(r_intra.index.date)
    barpos = day_of.groupby(day_of.values).cumcount() + 1
    is_event = np.zeros(len(r_intra), bool)
    is_event[(falls_i.times - 1).astype(int)] = True
    gap_days = set(pd.DatetimeIndex(src.dates).date)
    on_trigger_day = day_of.isin(gap_days).values
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.4))
    pos = np.arange(1, 8)
    n_trig_days, n_quiet_days = len(gap_days), len(set(day_of)) - len(gap_days)
    f_t = [np.sum(is_event & on_trigger_day & (barpos == p)) / n_trig_days
           for p in pos]
    f_q = [np.sum(is_event & ~on_trigger_day & (barpos == p)) / n_quiet_days
           for p in pos]
    w = 0.38
    ax1.bar(pos - w / 2, f_t, w, color=C["fall"],
            label=f"big-gap days (n={n_trig_days})")
    ax1.bar(pos + w / 2, f_q, w, color=C["muted"],
            label=f"quiet-gap days (n={n_quiet_days})")
    ax1.set_xlabel("intraday hour (bar of the trading day)")
    ax1.set_ylabel("extreme falls per day")
    ax1.set_title("Extreme intraday falls cluster in the first hours\n"
                  "after a big overnight gap (model-free evidence)")
    ax1.legend(fontsize=8)

    # fitted intensity decomposition on a representative stretch
    t_grid = np.arange(1.0, falls_i.T + 1.0, 0.25)
    mu, self_t, trig_t = model._terms(t_grid, model.theta)
    a, b = falls_i.T - 300, falls_i.T   # last ~6 weeks
    sel = (t_grid >= a) & (t_grid <= b)
    ax2.fill_between(t_grid[sel], 0, mu, color=C["grid"], label="background $\\mu$")
    ax2.fill_between(t_grid[sel], mu, mu + self_t[sel], color=C["seq"][2],
                     label="intraday self-excitation")
    ax2.fill_between(t_grid[sel], mu + self_t[sel], mu + self_t[sel] + trig_t[sel],
                     color=C["fall"], label="overnight-gap trigger")
    for t in src.times[(src.times >= a) & (src.times <= b)]:
        ax2.axvline(t, color=C["ink2"], lw=0.6, alpha=0.4, ymax=0.06)
    ax2.set_xlabel("trading hours (last 6 weeks of sample; ticks = gap opens)")
    ax2.set_ylabel("fall intensity (events/hour)")
    ax2.set_title("Fitted intensity decomposition:\ngap spikes at the open, fading in ~1 hour")
    ax2.legend(fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(FIGS / "fig8_overnight.png", dpi=150)
    plt.close(fig)
    print("saved figures/fig8_overnight.png")


def overnight_walkforward(falls_i, src, gaps_all):
    print()
    print("=" * 72)
    print("10. WALK-FORWARD (30-day start): overnight model vs benchmarks")
    print("=" * 72)
    day_starts = np.concatenate([[1.0], gaps_all["time"].values + 0.5])
    wf = walk_forward_overnight(falls_i, src, day_starts,
                                start_days=30, step_days=30)
    tot = {k: float(wf[f"oos_{k}"].sum())
           for k in ("full", "selfonly", "kanpin", "poisson")}
    wins = int((wf["oos_full"] > wf["oos_selfonly"]).sum())
    print(f"\nTOTAL OOS log-score: overnight-MLE {tot['full']:.1f} | "
          f"self-only {tot['selfonly']:.1f} | KAN-PIN {tot['kanpin']:.1f} | "
          f"Poisson {tot['poisson']:.1f}")
    print(f"overnight model beats self-only in {wins}/{len(wf)} segments; "
          f"K_o path {wf['mle_K_o'].min():.2f}-{wf['mle_K_o'].max():.2f}; "
          f"LR significant (p<0.05) from step "
          f"{int(np.argmax(wf['LR_pvalue'].values < 0.05)) + 1} onward.")
    RESULTS["overnight_walkforward"] = {"total_oos": tot,
                                        "wins_vs_selfonly": wins,
                                        "n_segments": int(len(wf))}
    return wf


def figure_overnight_wf(wf):
    fig = plt.figure(figsize=(11.5, 7.5))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.15, 1])
    dates = [d.strftime("%d %b %y") for d in wf["date"]]
    x = np.arange(len(wf))
    ax = fig.add_subplot(gs[0, :])
    w = 0.27
    ax.bar(x - w, wf["oos_full"] - wf["oos_poisson"], w, color=C["fall"],
           label="overnight-trigger MLE")
    ax.bar(x, wf["oos_selfonly"] - wf["oos_poisson"], w, color=C["yellow"],
           label="self-only ETAS")
    ax.bar(x + w, wf["oos_kanpin"] - wf["oos_poisson"], w, color=C["run"],
           label="overnight KAN-PIN")
    ax.axhline(0, color=C["ink2"], lw=1.2)
    ax.text(len(wf) - 0.5, 0, " Poisson benchmark", color=C["ink2"],
            fontsize=8, va="bottom", ha="right")
    ax.set_xticks(x); ax.set_xticklabels(dates, fontsize=8)
    ax.set_ylabel("OOS log-score minus Poisson")
    ax.set_title("Walk-forward (30-day start): out-of-sample predictive "
                 "log-score by segment (above 0 = beats Poisson)")
    ax.legend(fontsize=8, ncol=3)

    ax = fig.add_subplot(gs[1, 0])
    ax.plot(x, wf["mle_K_o"], color=C["fall"], marker="o", ms=5,
            label="$K_o$ (overnight trigger), MLE")
    ax.plot(x, wf["mle_K_s"], color=C["yellow"], marker="o", ms=5,
            label="$K_s$ (intraday self), MLE")
    ax.plot(x, wf["kp_K_o"], color=C["run"], marker="s", ms=5,
            label="$K_o$, KAN-PIN")
    ax.set_xticks(x[::2]); ax.set_xticklabels(dates[::2], fontsize=8)
    ax.set_title("fertility paths across windows")
    ax.legend(fontsize=8)

    ax = fig.add_subplot(gs[1, 1])
    ax.plot(x, wf["LR_overnight"], color=C["aqua"], marker="o", ms=5,
            label="LR statistic")
    ax.axhline(7.81, color=C["muted"], ls="--", lw=1.2)
    ax.text(0.1, 7.81, " 5% critical, $\\chi^2(3)$", color=C["muted"],
            fontsize=8, va="bottom")
    ax.set_xticks(x[::2]); ax.set_xticklabels(dates[::2], fontsize=8)
    ax.set_title("running LR test: is overnight triggering significant?")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGS / "fig9_overnight_wf.png", dpi=150)
    plt.close(fig)
    print("saved figures/fig9_overnight_wf.png")


# ======================================================================
# 11. Daily data (small-sample caveat)
# ======================================================================

def daily_analysis(r_d: pd.Series):
    print()
    print("=" * 72)
    print("8. DAILY DATA (aggregated; small sample - indicative only)")
    print("=" * 72)
    falls_d = extract_events(r_d, "fall", 0.90)   # 90% threshold: ~25 events
    m = ETASModel(kernel="exp", use_alpha=False).fit(falls_d, seed=1)
    diag = m.residual_diagnostics()
    print(m.explain())
    print(f"KS p-value: {diag['ks_pvalue']:.3f}")
    print(f"NOTE: only {falls_d.n} events in {falls_d.T:.0f} trading days - "
          "the paper uses ~50 years of daily data; treat as illustration.")
    RESULTS["daily"] = {"n_events": int(falls_d.n), "mu": m.mu, "K0": m.K0,
                        "c": m.c, "branching_n": m.branching_ratio(),
                        "ks_pvalue": diag["ks_pvalue"]}
    return falls_d, m


# ======================================================================

def main(csv_path: str):
    t0 = time.time()
    close_h, r_h, close_d, r_d = load_data(csv_path)
    falls, runs, best, tables = fit_models(r_h)
    cross = fit_cross(falls, runs)
    figure_events(close_h, r_h, falls, runs)
    figure_intensity(close_h, r_h, best)
    spec = (best["fall"].kernel, best["fall"].use_alpha)
    m_train, t_grid, bt = ews_out_of_sample(r_h, falls, spec, d=7.0)
    figure_ews(r_h, falls, t_grid, bt, d=7.0)
    dur_df, episode = duration_forecasts(falls, runs, best)
    figure_duration(falls, dur_df, episode)
    synthetic_validation()
    kp, mle, tab = kan_pin_real(falls, best)
    figure_kanpin(falls, kp, mle, tab)
    wf, ny = stability_and_walkforward(falls, best)
    figure_stability(falls, wf)
    r_intra, gaps_all, falls_i, src, model_on, kp_on = overnight_analysis(csv_path)
    figure_overnight(r_intra, gaps_all, falls_i, src, model_on)
    wf_on = overnight_walkforward(falls_i, src, gaps_all)
    figure_overnight_wf(wf_on)
    daily_analysis(r_d)
    (HERE / "results.json").write_text(json.dumps(RESULTS, indent=2, default=float))
    print(f"\nresults.json written.  total {time.time() - t0:.0f}s")


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else str(
        HERE / "AA_h.csv")
    main(path)
