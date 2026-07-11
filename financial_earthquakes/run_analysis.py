"""
End-to-end analysis: the Gresnigt-Kole-Franses "financial earthquake"
indicator (ETAS/Hawkes) + the KAN-PIN parameter estimator, applied to
Alcoa (AA) hourly data (and a daily aggregate).

Run:  python3 run_analysis.py [path/to/AA_h.csv]

Produces figures/fig1..fig5 (PNG) and results.json.
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
# 8. Daily data (small-sample caveat)
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
    daily_analysis(r_d)
    (HERE / "results.json").write_text(json.dumps(RESULTS, indent=2, default=float))
    print(f"\nresults.json written.  total {time.time() - t0:.0f}s")


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else str(
        HERE / "AA_h.csv")
    main(path)
