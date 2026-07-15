"""End-to-end bubble detection pipeline.

  data -> statistical-test features (causal) -> labels -> purged walk-forward
       -> Causal Random Forest (tau of explosiveness on forward returns)
       -> benchmark RF direction classifier -> metrics, plots, summary

Usage:  python run_pipeline.py [--csv data/AA_h.csv] [--horizon 21]
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
from bubbles.features import FeatureConfig, build_features
from bubbles.labels import build_labels
from bubbles.walkforward import purged_walkforward
from bubbles.causal_model import fit_causal_forest, fit_direction_benchmark
from bubbles.strategy import backtest
from bubbles import psy

# Okabe-Ito CVD-safe hues (validated): blue, orange, green, vermillion
C_BLUE, C_ORANGE, C_GREEN, C_VERM = "#0072B2", "#E69F00", "#009E73", "#D55E00"
GRAY = "#767676"

# Features excluded from the causal forest's X: they are (near-)deterministic
# functions of the treatment definition, so keeping them would destroy
# propensity overlap.  The benchmark classifier still uses all of them.
TREATMENT_DERIVED = {"bubble_flag", "bsadf", "bsadf_gap", "bubble_age"}


def newey_west_tstat(x: np.ndarray, lag: int) -> float:
    """t-stat of the mean with Newey-West (HAC) variance, for overlapping labels."""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    n = x.size
    if n < 10:
        return np.nan
    e = x - x.mean()
    g0 = float(e @ e) / n
    v = g0
    for l in range(1, min(lag, n - 1) + 1):
        w = 1.0 - l / (lag + 1.0)
        v += 2.0 * w * float(e[l:] @ e[:-l]) / n
    return float(x.mean() / np.sqrt(max(v, 1e-18) / n))


def style_ax(ax):
    ax.grid(True, alpha=0.25, linewidth=0.6)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def _bt_section(bt: dict) -> str:
    if "error" in bt:
        return f"skipped: {bt['error']}"
    rows = []
    for key, name in (("strategy", "event strategy"), ("buy_hold", "buy & hold"),
                      ("long_in_episode", "long whenever flagged")):
        s = bt[key]
        rows.append(f"| {name} | {100*s['total_simple_return']:+.1f}% | {s['total_log_return']:+.4f} | "
                    f"{s['ann_return']:+.3f} | {s['ann_vol']:.3f} | {s['sharpe']:+.2f} | "
                    f"{s['sortino']:+.2f} | {s['calmar']:+.2f} | {s['max_drawdown_log']:+.4f} | "
                    f"{s['bar_hit_rate']:.3f} |")
    body = "\n".join(rows)
    tr = (f"trades: {bt['n_trades']} ({bt['n_long']} long / {bt['n_short']} short) · "
          f"win rate {format(bt['win_rate'], '.2f') if bt['n_trades'] else 'n/a'} · "
          f"profit factor {format(bt.get('profit_factor', float('nan')), '.2f')} · "
          f"avg win {format(bt.get('avg_win_log', float('nan')), '+.4f')} / "
          f"avg loss {format(bt.get('avg_loss_log', float('nan')), '+.4f')} (log) · "
          f"avg duration {format(bt.get('avg_trade_bars', float('nan')), '.0f')} bars")
    return (f"span {bt['span'][0]} → {bt['span'][1]} ({bt['bars_evaluated']} bars, "
            f"~{bt['bars_per_year']:.0f} bars/yr) · exposure {100*bt['exposure_frac']:.1f}%\n"
            f"{tr}\n\n"
            "| series | total ret | total log ret | ann ret | ann vol | Sharpe | Sortino | Calmar | max DD (log) | bar hit rate |\n"
            "|---|---|---|---|---|---|---|---|---|---|\n" + body)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=os.path.join(os.path.dirname(__file__), "data", "AA_h.csv"))
    ap.add_argument("--horizon", type=int, default=21, help="label horizon in bars (~3 days hourly)")
    ap.add_argument("--train-min", type=int, default=600)
    ap.add_argument("--test-size", type=int, default=150)
    ap.add_argument("--mc-sims", type=int, default=200)
    ap.add_argument("--ssa", action="store_true", help="SSA-denoise inputs of the stability tests")
    ap.add_argument("--cost-bps", type=float, default=2.0, help="one-way transaction cost")
    ap.add_argument("--min-abs-tau", type=float, default=0.0, help="extra |tau| entry filter")
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "results"))
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    rng_seed = 42

    # ---------------- data & features ----------------
    df = load_yf_csv(args.csv)
    asset = os.path.splitext(os.path.basename(args.csv))[0]
    print(f"Loaded {len(df)} bars: {df.index[0]} .. {df.index[-1]}")
    cfg = FeatureConfig(mc_sims=args.mc_sims, use_ssa=args.ssa,
                        cache_path=os.path.join(args.out, "cache_bsadf_cv.json")
                        ).scale_for_length(len(df))
    feats, meta = build_features(df, cfg)
    labels = build_labels(df, feats, horizon=args.horizon)
    print(f"Features: {list(feats.columns)}")

    # Full-sample GSADF verdict (the formal test behind the real-time flag)
    gsadf, gcv = meta["gsadf_stat"], meta["gsadf_cv"]
    print(f"GSADF stat = {gsadf:.3f} | cv90/95/99 = "
          f"{gcv[0.90]:.3f}/{gcv[0.95]:.3f}/{gcv[0.99]:.3f}")

    min_dur = max(3, int(round(np.log(len(df)))))
    episodes = psy.date_stamp_episodes(feats["bubble_flag"].to_numpy() == 1.0, min_dur)
    print(f"PSY episodes (>= {min_dur} bars): {len(episodes)}")

    # ---------------- usable rows & folds ----------------
    all_cols = list(feats.columns)
    x_cols_causal = [c for c in all_cols if c not in TREATMENT_DERIVED]
    data = feats.join(labels)
    usable = np.where(data[all_cols + ["fwd_ret"]].notna().all(axis=1).to_numpy())[0]
    print(f"Usable rows: {usable.size} of {len(df)}")
    if usable.size < 500:                    # tiny samples: relax treated-arm floor
        from bubbles import causal_model as _cm
        _cm.MIN_TREATED = 10
    # keep the fold count manageable on large samples (~8-10 test blocks)
    test_size = max(args.test_size, usable.size // 10)
    train_min = max(args.train_min, usable.size // 3)
    if test_size != args.test_size or train_min != args.train_min:
        print(f"Auto-scaled folds: train_min={train_min} test_size={test_size}")
    folds = purged_walkforward(usable, train_min, test_size, args.horizon)
    print(f"Walk-forward folds: {len(folds)}")
    if not folds:
        raise SystemExit("Not enough data for walk-forward validation.")

    F = feats.to_numpy()
    Xc = feats[x_cols_causal].to_numpy()
    T = feats["bubble_flag"].to_numpy()
    Y = labels["fwd_ret"].to_numpy()
    D = labels["dir_up"].to_numpy()

    pred = pd.DataFrame(index=df.index,
                        columns=["fold", "proba_up", "tau", "tau_lb", "tau_ub"],
                        dtype=float)
    fold_rows, importances = [], []
    MAX_TRAIN = 60000                       # cap forest training size on huge samples
    for k, (tr, te) in enumerate(folds):
        if len(tr) > MAX_TRAIN:
            tr = tr[-MAX_TRAIN:]            # most recent rows (still fully pre-test)
        bm = fit_direction_benchmark(F[tr], D[tr], F[te], seed=rng_seed)
        pred.iloc[te, pred.columns.get_loc("proba_up")] = bm["proba_up"]
        pred.iloc[te, pred.columns.get_loc("fold")] = k
        importances.append(bm["importances"])

        cf = fit_causal_forest(Xc[tr], T[tr], Y[tr], Xc[te], seed=rng_seed)
        row = {"fold": k, "train": len(tr), "test": len(te),
               "treated_train": int(T[tr].sum()), "treated_test": int(T[te].sum())}
        if cf is not None:
            pred.iloc[te, pred.columns.get_loc("tau")] = cf["tau"]
            pred.iloc[te, pred.columns.get_loc("tau_lb")] = cf["tau_lb"]
            pred.iloc[te, pred.columns.get_loc("tau_ub")] = cf["tau_ub"]
            row.update(ate=cf["ate"], ate_stderr=cf["ate_stderr"])
        else:
            row.update(ate=np.nan, ate_stderr=np.nan)
        fold_rows.append(row)
        print(f"fold {k}: train={len(tr)} test={len(te)} "
              f"treated(train/test)={row['treated_train']}/{row['treated_test']} "
              f"ate={row['ate'] if np.isfinite(row.get('ate', np.nan)) else 'skipped'}")

    # ---------------- evaluation ----------------
    from sklearn.metrics import accuracy_score, roc_auc_score

    te_mask = pred["proba_up"].notna().to_numpy() & np.isfinite(D)
    yhat = (pred["proba_up"].to_numpy()[te_mask] > 0.5).astype(int)
    ytrue = D[te_mask].astype(int)
    bench = {
        "n_test": int(te_mask.sum()),
        "accuracy": float(accuracy_score(ytrue, yhat)),
        "auc": float(roc_auc_score(ytrue, pred["proba_up"].to_numpy()[te_mask]))
        if len(np.unique(ytrue)) > 1 else np.nan,
        "base_rate_up": float(ytrue.mean()),
    }

    tau = pred["tau"].to_numpy()
    treated_te = np.isfinite(tau) & (T == 1.0) & np.isfinite(Y)
    all_te = np.isfinite(tau) & np.isfinite(Y)
    causal = {"n_treated_test_bars": int(treated_te.sum())}
    if all_te.sum() >= 30:
        # Secondary diagnostic: does the CATE surface carry directional
        # information on ALL test bars (not only currently-explosive ones)?
        s_all = np.sign(tau[all_te])
        causal.update({
            "all_bars_dir_hit_rate": float((s_all == np.sign(Y[all_te])).mean()),
            "all_bars_tau_fwdret_corr": float(np.corrcoef(tau[all_te], Y[all_te])[0, 1]),
            "all_bars_ci_excludes_zero_frac": float(np.mean(
                (pred["tau_lb"].to_numpy()[all_te] > 0) |
                (pred["tau_ub"].to_numpy()[all_te] < 0))),
        })
    if treated_te.sum() >= 10:
        s = np.sign(tau[treated_te])
        strat = s * Y[treated_te]
        causal.update({
            "dir_hit_rate": float((s == np.sign(Y[treated_te])).mean()),
            "strategy_mean_fwd_ret": float(strat.mean()),
            "always_long_mean_fwd_ret": float(Y[treated_te].mean()),
            "strategy_nw_tstat": newey_west_tstat(strat, args.horizon),
            "always_long_nw_tstat": newey_west_tstat(Y[treated_te], args.horizon),
            "tau_fwdret_corr": float(np.corrcoef(tau[treated_te], Y[treated_te])[0, 1]),
        })

    # ---------------- event-driven strategy backtest ----------------
    bt = backtest(df["Close"], T, tau,
                  pred["tau_lb"].to_numpy(), pred["tau_ub"].to_numpy(),
                  cost_bps=args.cost_bps, require_ci=True,
                  min_abs_tau=args.min_abs_tau)

    imp = pd.Series(np.mean(importances, axis=0), index=all_cols).sort_values(ascending=False)

    metrics = {
        "n_bars": len(df),
        "horizon": args.horizon,
        "gsadf_stat": gsadf,
        "gsadf_cv": {str(k): v for k, v in gcv.items()},
        "gsadf_reject_95": bool(gsadf > gcv[0.95]),
        "n_episodes": len(episodes),
        "episodes": [[df.index[a].isoformat(), df.index[b].isoformat()] for a, b in episodes],
        "bubble_bar_fraction": float(np.nanmean(T)),
        "folds": fold_rows,
        "benchmark_rf": bench,
        "causal_forest": causal,
        "strategy_backtest": {k: v for k, v in bt.items() if k != "_series"},
        "ssa": bool(args.ssa),
        "feature_importances_top10": imp.head(10).round(4).to_dict(),
    }
    with open(os.path.join(args.out, "metrics.json"), "w") as fh:
        json.dump(metrics, fh, indent=2, default=float)

    out_df = data.join(pred)
    if len(out_df) > 60000:
        out_df.to_csv(os.path.join(args.out, "features_predictions.csv.gz"),
                      compression="gzip")
    else:
        out_df.to_csv(os.path.join(args.out, "features_predictions.csv"))

    # ---------------- plots ----------------
    x = df.index

    fig, ax = plt.subplots(figsize=(11, 4.2))
    ax.plot(x, df["Close"], color=C_BLUE, lw=1.2, label=f"{asset} close")
    for i, (a, b) in enumerate(episodes):
        ax.axvspan(x[a], x[b], color=C_ORANGE, alpha=0.30,
                   label="PSY explosive episode" if i == 0 else None)
    ax.set_title(f"{asset} close with PSY (BSADF > cv95) explosive episodes")
    ax.legend(frameon=False)
    style_ax(ax)
    fig.tight_layout(); fig.savefig(os.path.join(args.out, "price_episodes.png"), dpi=140)

    fig, ax = plt.subplots(figsize=(11, 3.6))
    ax.plot(x, feats["bsadf"], color=C_BLUE, lw=1.0, label="BSADF")
    ax.plot(x, meta["bsadf_cv_series"], color=C_VERM, lw=1.0, ls="--", label="95% critical value (MC)")
    ax.set_title("Backward Sup ADF sequence vs Monte-Carlo critical value")
    ax.legend(frameon=False)
    style_ax(ax)
    fig.tight_layout(); fig.savefig(os.path.join(args.out, "bsadf.png"), dpi=140)

    panel = [("rtadf", "Right-tailed ADF"), ("cusum", "CUSUM (rel. to 5% bound)"),
             ("chow_f", "Chow F (midpoint)"), ("qlr_f", "QLR sup-F"),
             ("bp_nbreaks", "Bai-Perron # breaks"), ("rup_ncp", "PELT # change points"),
             ("hmm_p_bull", "HMM P(bull)"), ("hmm_sig", "HMM E[vol]")]
    fig, axes = plt.subplots(4, 2, figsize=(11, 9), sharex=True)
    for (col, title), ax in zip(panel, axes.ravel()):
        ax.plot(x, feats[col], color=C_BLUE, lw=0.9)
        ax.set_title(title, fontsize=9)
        style_ax(ax)
    fig.suptitle("Statistical-test feature panel (all causal, trailing windows)", y=0.995)
    fig.autofmt_xdate(rotation=30)
    fig.tight_layout(); fig.savefig(os.path.join(args.out, "feature_panel.png"), dpi=140)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 6), sharex=True,
                                   gridspec_kw={"height_ratios": [2, 1.4]})
    ax1.plot(x, df["Close"], color=C_BLUE, lw=1.0, label=f"{asset} close")
    for i, (a, b) in enumerate(episodes):
        ax1.axvspan(x[a], x[b], color=C_ORANGE, alpha=0.30,
                    label="explosive episode" if i == 0 else None)
    ax1.legend(frameon=False); style_ax(ax1)
    m = pred["tau"].notna()
    if m.any():
        ax2.fill_between(x[m], pred.loc[m, "tau_lb"].astype(float),
                         pred.loc[m, "tau_ub"].astype(float),
                         color=GRAY, alpha=0.3, label="90% CI")
        tvals = pred.loc[m, "tau"].astype(float)
        ax2.scatter(x[m], tvals, s=6,
                    c=np.where(tvals >= 0, C_GREEN, C_VERM))
    ax2.axhline(0, color=GRAY, lw=0.8)
    ax2.set_title("Causal forest tau(x): effect of explosiveness on fwd %d-bar return "
                  "(green: continuation, vermillion: reversal)" % args.horizon, fontsize=9)
    ax2.legend(frameon=False); style_ax(ax2)
    fig.tight_layout(); fig.savefig(os.path.join(args.out, "tau_direction.png"), dpi=140)

    if "_series" in bt:
        eq = bt["_series"]
        fig, ax = plt.subplots(figsize=(11, 4.0))
        ax.plot(eq.index, eq["strat"], color=C_BLUE, lw=1.4, label="event strategy (sign tau, CI-filtered)")
        ax.plot(eq.index, eq["bh"], color=C_ORANGE, lw=1.4, label="buy & hold")
        ax.plot(eq.index, eq["long_ep"], color=C_GREEN, lw=1.2, label="long whenever flagged")
        ax.set_title(f"Cumulative log return, out-of-sample span — cost {args.cost_bps:.0f} bps/side, "
                     "positions delayed one bar")
        ax.legend(frameon=False); style_ax(ax)
        fig.tight_layout(); fig.savefig(os.path.join(args.out, "strategy.png"), dpi=140)

    # ---------------- summary ----------------
    ep_lines = "\n".join(f"| {df.index[a]:%Y-%m-%d %H:%M} | {df.index[b]:%Y-%m-%d %H:%M} | {b-a+1} |"
                         for a, b in episodes) or "| (none) | | |"
    fold_lines = "\n".join(
        f"| {r['fold']} | {r['train']} | {r['test']} | {r['treated_train']} | {r['treated_test']} | "
        f"{r['ate']:.5f} ± {r['ate_stderr']:.5f} |" if np.isfinite(r.get("ate", np.nan)) else
        f"| {r['fold']} | {r['train']} | {r['test']} | {r['treated_train']} | {r['treated_test']} | skipped |"
        for r in fold_rows)
    imp_lines = "\n".join(f"| {k} | {v:.4f} |" for k, v in imp.head(10).items())
    cz = causal
    summary = f"""# Bubble detection run summary

Data: `{os.path.basename(args.csv)}` — {len(df)} bars, {df.index[0]:%Y-%m-%d} to {df.index[-1]:%Y-%m-%d}.
Label horizon: {args.horizon} bars. Walk-forward folds: {len(folds)} (purge = horizon).

## Full-sample GSADF test
GSADF = **{gsadf:.3f}** vs Monte-Carlo critical values 90/95/99% = {gcv[0.90]:.3f} / {gcv[0.95]:.3f} / {gcv[0.99]:.3f}
→ explosiveness {"**detected**" if gsadf > gcv[0.95] else "not detected"} at the 5% level over the sample
(sup-statistic cv simulated on the stationary-tail length — a lower bound for samples much longer
than the window cap, so treat borderline rejections cautiously; per-bar flags are unaffected).
Explosive bars (BSADF > cv95): {100*np.nanmean(T):.1f}% of the sample.

## PSY date-stamped episodes (min duration {min_dur} bars)
| start | end | bars |
|---|---|---|
{ep_lines}

## Walk-forward causal forest (treatment = explosiveness flag)
| fold | train | test | treated(train) | treated(test) | ATE on fwd ret |
|---|---|---|---|---|---|
{fold_lines}

Pooled treated test bars: {cz.get('n_treated_test_bars', 0)}
{f'''- direction hit rate of sign(tau): **{cz['dir_hit_rate']:.3f}**
- mean fwd {args.horizon}-bar log return — sign(tau) strategy: **{cz['strategy_mean_fwd_ret']:.5f}** (NW t = {cz['strategy_nw_tstat']:.2f}) vs always-long: {cz['always_long_mean_fwd_ret']:.5f} (NW t = {cz['always_long_nw_tstat']:.2f})
- corr(tau, realized fwd ret): {cz['tau_fwdret_corr']:.3f}''' if 'dir_hit_rate' in cz else '- too few treated test bars for strategy evaluation.'}
{f'''
All test bars (secondary diagnostic — CATE as a conditional direction signal):
- sign(tau) hit rate: {cz['all_bars_dir_hit_rate']:.3f} | corr(tau, fwd ret): {cz['all_bars_tau_fwdret_corr']:.3f} | CI excludes 0 on {100*cz['all_bars_ci_excludes_zero_frac']:.1f}% of bars''' if 'all_bars_dir_hit_rate' in cz else ''}

## Event-driven strategy backtest (out-of-sample span, {args.cost_bps:.0f} bps/side, 1-bar delay)
{_bt_section(bt)}

## Benchmark RF direction classifier (all bars, all features)
accuracy = {bench['accuracy']:.3f} (base rate up = {bench['base_rate_up']:.3f}), AUC = {bench['auc']:.3f}, n = {bench['n_test']}

## Top feature importances (benchmark RF, fold average)
| feature | importance |
|---|---|
{imp_lines}

## Caveats
- Forward labels overlap (h = {args.horizon}); Newey-West t-stats partially correct this, but per-fold sample sizes are small — treat results as a research signal, not a tradable backtest.
- Causal identification is *selection-on-observables*: tau is causal only insofar as the stability/regime features span the confounders of the explosiveness flag.
- Single asset series; upgrade path is a cross-sectional panel.
"""
    with open(os.path.join(args.out, "summary.md"), "w") as fh:
        fh.write(summary)
    print("\n" + summary)


if __name__ == "__main__":
    main()
