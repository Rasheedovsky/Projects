"""ALCOA (AA) live bubble analysis — HLPPL paper implementation + extensions.

Indicators (P-LNN removed from the live set by design decision):
  HLPPL       paper: rolling 7-param TRF fit, residual score + volume-hype proxy
  HLPPL-KAN   extension: trajectory from M-LNN-KAN on DAE-cleaned windows
  MONO        M-LNN fit ensemble at the live date
  MONO-KAN    M-LNN-KAN fit ensemble at the live date
  MONO-DAE    M-LNN on DAE-cleaned windows at the live date

Outputs:
  results/fig_alcoa_hlppl.png           score series + episodes (paper Figs 1-4)
  results/fig_alcoa_live_densities.png  tc + price_c PDFs at the live date
  results/fig_alcoa_ml_decision.png     5-day score forecasts vs entry bands
  results/alcoa_scores.csv, alcoa_episodes.csv, alcoa_live_ensemble.csv,
  results/alcoa_ml_forecast.csv
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
from scipy.stats import gaussian_kde

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from deep_lppls import dae, hlppl, kan, mlnn
from deep_lppls.core import minmax_scale, solve_linear

W_ROLL = 126                       # rolling window for the HLPPL score
LIVE_WINDOWS = list(range(60, 241, 20))  # ensemble at the live date
N = 252

COLORS = {"HLPPL": "tab:blue", "HLPPL-KAN": "tab:red", "MONO": "tab:orange",
          "MONO-KAN": "tab:cyan", "MONO-DAE": "tab:brown"}


def load_alcoa():
    df = pd.read_csv(ROOT / "data" / "AA_1year.csv", skiprows=[1, 2])
    df = df.rename(columns={"Price": "Date"})
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)
    df["logp"] = np.log(df["Close"])
    return df


def mirror_dae_clean(x_scaled):
    """DAE cleaning with mirroring for declining windows (the DAE was
    trained on rising bubble shapes; a negative bubble is the mirror)."""
    mirrored = x_scaled[-1] < x_scaled[0]
    x_in = 1.0 - x_scaled if mirrored else x_scaled
    out = dae.clean(hlppl._dae_params(), x_in).astype(np.float64)
    return (1.0 - out) if mirrored else out


def rolling_scores(df, engine="trf"):
    """HLPPL score series: for each day, fit the window ending there, take
    the endpoint residual, causally normalise, add the hype term (Eq. 14)."""
    logp = df["logp"].values
    hype = hlppl.hype_proxy(df["Volume"].values)
    eps, dates_idx, fits = [], [], []
    t0 = time.time()
    for t2 in range(W_ROLL - 1, len(df)):
        win = logp[t2 - W_ROLL + 1 : t2 + 1]
        if engine == "trf":
            theta, sse = hlppl.fit_lppl_trf(win, seed=t2)
            if theta is None:
                eps.append(np.nan); dates_idx.append(t2); fits.append(None)
                continue
            traj = hlppl.lppl7(theta, np.arange(1, W_ROLL + 1, dtype=float))
            fits.append(theta)
        else:
            r = hlppl.fit_lppl_kan(win, seed=t2)
            traj = r["traj"]
            fits.append(r)
        eps.append(win[-1] - traj[-1])
        dates_idx.append(t2)
        if (len(eps)) % 40 == 0:
            print(f"  [{engine}] {len(eps)}/{len(df) - W_ROLL + 1} ({time.time() - t0:.0f}s)", flush=True)
    eps = np.array(eps)
    eps_norm = hlppl.causal_normalize(np.nan_to_num(eps, nan=0.0))
    score = hlppl.bubble_score(eps_norm, hype[np.array(dates_idx)])
    return pd.DataFrame(dict(idx=dates_idx, date=df["Date"].values[np.array(dates_idx)],
                             eps=eps, eps_norm=eps_norm, hype=hype[np.array(dates_idx)],
                             score=score)), fits


def live_ensembles(df):
    """All five methods fitted over LIVE_WINDOWS ending at the last date.
    Returns rows with (method, L, sign, tc_days, tc_date, price_c)."""
    logp = df["logp"].values
    t2_idx = len(df) - 1
    rows = []
    t_norm = np.linspace(0.0, 1.0, N)
    cal_per_td = (df["Date"].iloc[-1] - df["Date"].iloc[0]) / (len(df) - 1)

    def norm_record(method, tc, m, w, x_fit, scale, mirrored, L):
        mn, rng_ = scale
        beta = solve_linear(t_norm, x_fit, tc, m, w)
        b = float(beta[1])
        a_sc = float(beta[0])
        if mirrored:
            b = -b
            a_sc = 1.0 - a_sc
        tc_days = (tc - 1.0) * (L - 1)
        price_c = float(np.exp(a_sc * rng_ + mn))
        rows.append(dict(method=method, L=L, sign="positive" if b < 0 else "negative",
                         tc_days_after_now=tc_days,
                         tc_date=df["Date"].iloc[-1] + cal_per_td * tc_days,
                         price_c=price_c, m=m, w=w))

    for L in LIVE_WINDOWS:
        win = logp[t2_idx - L + 1 : t2_idx + 1]
        x = np.interp(t_norm, np.linspace(0, 1, L), win)
        x_scaled, scale = minmax_scale(x)
        mirrored = x_scaled[-1] < x_scaled[0]
        x_mir = 1.0 - x_scaled if mirrored else x_scaled
        x_clean_mir = dae.clean(hlppl._dae_params(), x_mir).astype(np.float64)

        r = mlnn.fit_mlnn(x_scaled, seed=L)
        norm_record("MONO", r["tc"], r["m"], r["w"], x_scaled, scale, False, L)
        r = kan.fit_mlnn_kan(x_scaled, seed=L)
        norm_record("MONO-KAN", r["tc"], r["m"], r["w"], x_scaled, scale, False, L)
        r = mlnn.fit_mlnn(x_clean_mir, seed=L)
        norm_record("MONO-DAE", r["tc"], r["m"], r["w"], x_clean_mir, scale, mirrored, L)
        r = kan.fit_mlnn_kan(x_clean_mir, seed=L)
        norm_record("HLPPL-KAN", r["tc"], r["m"], r["w"], x_clean_mir, scale, mirrored, L)

        theta, sse = hlppl.fit_lppl_trf(win, seed=L)
        if theta is not None:
            A, B, C, m7, w7, phi, tc7 = theta
            tc_days = tc7 - L
            rows.append(dict(method="HLPPL", L=L, sign="positive" if B < 0 else "negative",
                             tc_days_after_now=tc_days,
                             tc_date=df["Date"].iloc[-1] + cal_per_td * tc_days,
                             price_c=float(np.exp(A)), m=m7, w=w7))
        print(f"  live L={L} done", flush=True)
    return pd.DataFrame(rows)


def ml_forecast(scores, df):
    """Walk-forward multi-horizon forecaster of the Bubble Score (paper's
    decision layer; the dual-stream transformer is scaled down to ridge
    regression on lagged features — disclosed: 1 stock-year of data cannot
    train a transformer without overfitting)."""
    from numpy.linalg import lstsq

    s = scores["score"].values
    idx = scores["idx"].values
    ret1 = df["logp"].diff().reindex().values
    feats, targs = [], []
    LAGS = 10
    for i in range(LAGS, len(s) - 5):
        f = list(s[i - LAGS + 1 : i + 1]) + [scores["eps_norm"].values[i], scores["hype"].values[i],
                                             ret1[idx[i]], df["logp"].values[idx[i]] - df["logp"].values[idx[i] - 5]]
        feats.append(f)
        targs.append(s[i + 1 : i + 6])
    X = np.array(feats)
    Y = np.array(targs)
    n_train = int(len(X) * 0.8)
    Xm, Xs = X[:n_train].mean(0), X[:n_train].std(0) + 1e-9
    Xn = (X - Xm) / Xs
    lam = 3.0
    A = np.vstack([Xn[:n_train], np.sqrt(lam) * np.eye(X.shape[1])])
    val_corr = []
    coefs = []
    for h in range(5):
        b = np.concatenate([Y[:n_train, h], np.zeros(X.shape[1])])
        w = lstsq(A, b, rcond=None)[0]
        coefs.append(w)
        pv = Xn[n_train:] @ w
        val_corr.append(float(np.corrcoef(pv, Y[n_train:, h])[0, 1]))
    # live forecast from the latest feature vector
    i = len(s) - 1
    f_live = list(s[i - LAGS + 1 : i + 1]) + [scores["eps_norm"].values[i], scores["hype"].values[i],
                                              ret1[idx[i]], df["logp"].values[idx[i]] - df["logp"].values[idx[i] - 5]]
    f_live = (np.array(f_live) - Xm) / Xs
    fc = np.array([float(f_live @ w) for w in coefs])
    return fc, val_corr


def main():
    df = load_alcoa()
    print(f"ALCOA {df.Date.iloc[0].date()} -> {df.Date.iloc[-1].date()} ({len(df)} days), "
          f"last close {df.Close.iloc[-1]:.2f}", flush=True)
    peak_idx = int(df["Close"].idxmax())
    print(f"max close {df.Close.max():.2f} on {df.Date[peak_idx].date()} "
          f"({100 * (df.Close.iloc[-1] / df.Close.max() - 1):.1f}% below)", flush=True)

    # ---- rolling HLPPL scores, both engines ----
    print("rolling HLPPL (TRF)...", flush=True)
    sc_trf, fits_trf = rolling_scores(df, engine="trf")
    print("rolling HLPPL-KAN (DAE+KAN)...", flush=True)
    sc_kan, _ = rolling_scores(df, engine="kan")
    sc_trf.to_csv(ROOT / "results" / "alcoa_scores.csv", index=False)
    sc_kan.to_csv(ROOT / "results" / "alcoa_scores_kan.csv", index=False)

    alpha = hlppl.ou_alpha(sc_trf["eps"].dropna().values)
    print(f"residual AR(1) mean-reversion alpha = {alpha:.3f} "
          f"({'OK: volatility-confined' if alpha > 0 else 'WARNING: residuals not mean-reverting'})", flush=True)

    eps_all = []
    for name, sc in [("HLPPL", sc_trf), ("HLPPL-KAN", sc_kan)]:
        eps = hlppl.label_episodes(sc["score"].values, list(sc["date"]))
        for e in eps:
            e["engine"] = name
        eps_all += eps
        print(f"  {name}: {len(eps)} episodes", flush=True)
    pd.DataFrame(eps_all).to_csv(ROOT / "results" / "alcoa_episodes.csv", index=False)

    # ---- live ensembles at the last date ----
    print("live ensembles...", flush=True)
    ens = live_ensembles(df)
    ens.to_csv(ROOT / "results" / "alcoa_live_ensemble.csv", index=False)
    summ = ens.groupby("method").agg(
        n=("sign", "size"),
        neg_share=("sign", lambda s: float((s == "negative").mean())),
        med_tc_days=("tc_days_after_now", "median"),
        med_price_c=("price_c", "median"),
    ).round(2)
    print(summ.to_string(), flush=True)

    # ---- ML decision layer ----
    fc, val_corr = ml_forecast(sc_trf, df)
    action, reason = hlppl.trading_decision(fc, position=0)
    pd.DataFrame(dict(h=[1, 2, 3, 4, 5], forecast=fc)).to_csv(ROOT / "results" / "alcoa_ml_forecast.csv", index=False)
    print(f"score now {sc_trf.score.iloc[-1]:+.3f} | forecasts h1..5: "
          + " ".join(f"{v:+.2f}" for v in fc), flush=True)
    print(f"val corr by horizon: " + " ".join(f"{v:.2f}" for v in val_corr), flush=True)
    print(f"DECISION: {action} — {reason}", flush=True)

    plot_scores(df, sc_trf, sc_kan, eps_all, fits_trf)
    plot_densities(df, ens)
    plot_decision(sc_trf, fc, action, reason)
    print("figures saved", flush=True)


def plot_scores(df, sc_trf, sc_kan, eps_all, fits_trf):
    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(14, 8.5), sharex=True,
                                  gridspec_kw={"height_ratios": [2, 1.4], "hspace": 0.05})
    ax.plot(df["Date"], df["logp"], color="black", lw=1.0, label="AA ln(price)")
    theta = fits_trf[-1]
    if theta is not None:
        t = np.arange(1, W_ROLL + 1, dtype=float)
        traj = hlppl.lppl7(theta, t)
        ax.plot(df["Date"].iloc[-W_ROLL:], traj, "--", color="tab:blue", lw=1.4,
                label="HLPPL fit (last window)")
    for e in eps_all:
        if e["engine"] != "HLPPL":
            continue
        color = "red" if e["type"] == "positive" else "green"
        ax.axvspan(e["start"], e["end"], color=color, alpha=0.15)
    ax.set_ylabel("ln(AA close)")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(alpha=0.25)
    ax.set_title("ALCOA: HLPPL bubble detection (shaded: episodes |score|>0.8 for 10+ days; "
                 "red=positive bubble, green=negative)")

    ax2.plot(sc_trf["date"], sc_trf["score"], color="tab:blue", lw=1.2, label="HLPPL score")
    ax2.plot(sc_kan["date"], sc_kan["score"], color="tab:red", lw=1.2, alpha=0.8, label="HLPPL-KAN score")
    for y, ls in [(hlppl.TAU, "-"), (-hlppl.TAU, "-"), (hlppl.THETA1, "--"), (-hlppl.THETA1, "--"),
                  (hlppl.THETA2, ":"), (-hlppl.THETA2, ":")]:
        ax2.axhline(y, color="grey", ls=ls, lw=0.8)
    ax2.set_ylim(-1.35, 1.35)
    ax2.set_ylabel("Bubble Score")
    ax2.legend(loc="upper left", fontsize=9)
    ax2.grid(alpha=0.25)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    fig.savefig(ROOT / "results" / "fig_alcoa_hlppl.png", dpi=140, bbox_inches="tight")


def plot_densities(df, ens):
    fig = plt.figure(figsize=(13.5, 8))
    gs = fig.add_gridspec(2, 2, width_ratios=[5, 1], height_ratios=[1, 3.4],
                          hspace=0.04, wspace=0.03)
    ax_t = fig.add_subplot(gs[0, 0])
    ax = fig.add_subplot(gs[1, 0], sharex=ax_t)
    ax_p = fig.add_subplot(gs[1, 1], sharey=ax)

    seg = df.iloc[-300:] if len(df) > 300 else df
    ax.plot(seg["Date"], seg["Close"], color="black", lw=1.0)
    last_date = df["Date"].iloc[-1]
    ax.axvline(last_date, color="red", ls="-.", lw=1.4)
    x_hi = last_date + pd.Timedelta(days=200)
    ax.axvspan(last_date, x_hi, color="grey", alpha=0.08)
    price_lo, price_hi = seg["Close"].min() * 0.75, seg["Close"].max() * 1.1
    ax.set_xlim(seg["Date"].iloc[0], x_hi)
    ax.set_ylim(price_lo, price_hi)

    t_grid = pd.date_range(last_date - pd.Timedelta(days=30), x_hi, freq="D")
    t_num = mdates.date2num(t_grid)
    p_grid = np.linspace(price_lo, price_hi, 300)
    conf_txt = []
    for method, g in ens.groupby("method"):
        neg_share = (g["sign"] == "negative").mean()
        conf_txt.append(f"{method} {int(100 * neg_share)}% neg")
        tcs = mdates.date2num(pd.to_datetime(g["tc_date"]))
        tcs = tcs[(tcs > t_num[0]) & (tcs < t_num[-1])]
        pcs = g["price_c"].values
        pcs = pcs[(pcs > price_lo) & (pcs < price_hi)]
        if len(np.unique(tcs)) >= 3:
            pdf = gaussian_kde(tcs)(t_num)
            ax_t.plot(t_grid, pdf, color=COLORS[method], lw=1.6)
            ax_t.fill_between(t_grid, 0, pdf, color=COLORS[method], alpha=0.25)
        if len(np.unique(pcs)) >= 3:
            pdfp = gaussian_kde(pcs)(p_grid)
            ax_p.plot(pdfp, p_grid, color=COLORS[method], lw=1.6)
            ax_p.fill_betweenx(p_grid, 0, pdfp, color=COLORS[method], alpha=0.25)

    ax_t.axvline(last_date, color="red", ls="-.", lw=1.2)
    ax_t.set_yticks([])
    ax_t.set_ylabel("PDF($t_c$)")
    ax_t.set_title(f"ALCOA live indicator @ {last_date.date()} — negative-bubble fit share: "
                   + " | ".join(conf_txt), fontsize=9.5)
    ax_p.set_xticks([])
    ax_p.set_xlabel("PDF(price$_c$)")
    plt.setp(ax_p.get_yticklabels(), visible=False)
    plt.setp(ax_t.get_xticklabels(), visible=False)
    ax.set_ylabel("AA close")
    handles = [plt.Line2D([], [], color=c, lw=2) for c in COLORS.values()]
    ax.legend(handles, COLORS.keys(), loc="upper left", fontsize=9)
    ax.grid(alpha=0.25)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    fig.savefig(ROOT / "results" / "fig_alcoa_live_densities.png", dpi=140, bbox_inches="tight")


def plot_decision(sc, fc, action, reason):
    fig, ax = plt.subplots(figsize=(11, 5.5))
    tail = sc.iloc[-60:]
    ax.plot(range(-len(tail) + 1, 1), tail["score"], color="tab:blue", lw=1.4, label="Bubble Score")
    ax.plot(range(1, 6), fc, "o--", color="tab:purple", lw=1.6, ms=8, label="ML forecast (h=1..5)")
    for y, lbl in [(hlppl.THETA1, "short entry +0.7"), (-hlppl.THETA1, "long entry -0.7"),
                   (hlppl.THETA2, "short exit +0.3"), (-hlppl.THETA2, "long exit -0.3")]:
        ax.axhline(y, color="grey", ls="--", lw=0.8)
        ax.text(-58, y + 0.02, lbl, fontsize=7, color="grey")
    ax.axvline(0, color="red", ls="-.", lw=1.2)
    ax.set_ylim(-1.3, 1.3)
    ax.set_xlabel("trading days relative to now")
    ax.set_ylabel("Bubble Score")
    ax.set_title(f"ALCOA — ML decision layer: {action}\n({reason})", fontsize=11)
    ax.legend(loc="lower left", fontsize=9)
    ax.grid(alpha=0.25)
    fig.savefig(ROOT / "results" / "fig_alcoa_ml_decision.png", dpi=140, bbox_inches="tight")


if __name__ == "__main__":
    main()
