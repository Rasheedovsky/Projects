"""
BMNR (BitMine Immersion) - real daily data supplied by the user
(BMNR_5y_data.csv, 5 Jun 2025 -> 14 Jul 2026, 277 sessions).

Daily bars, so: core model = ETAS on extreme close-to-close returns (90%
tail quantile ~ 28 events per tail); the OHLC columns also allow the
overnight-gap -> intraday-move test at daily resolution (gap = ln(O/C_prev)
triggering extreme open->close moves).  One MLE + KAN-PIN comparison, a
daily-refit walk-forward with OOS scoring vs Poisson, an episode/bottom
nowcast, and a single summary figure.  Designed to run once, cheaply.
"""

from __future__ import annotations

import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import run_analysis as RA
from etas import ETASModel, EventData, extract_events, load_yfinance_csv
from kan_pin import KANPINHawkes
from overnight import OvernightETAS, GapStream
from run_analysis import C, HERE, set_ticker
from stability import nyblom_test

Q = 0.90                 # daily extreme threshold (28 events per tail)


def main(csv_path: str):
    set_ticker("BMNR")
    df = load_yfinance_csv(csv_path)
    r = np.log(df["Close"]).diff().dropna()
    falls = extract_events(r, "fall", Q)
    runs = extract_events(r, "run", Q)
    print(f"BMNR daily: {len(r)} returns, {df.index[0].date()} -> "
          f"{df.index[-1].date()}; falls {falls.n} (<= -{falls.M0:.1%}), "
          f"runs {runs.n} (>= +{runs.M0:.1%})")

    # ---------- 1. core ETAS fits (paper method) + diagnostics ----------
    models = {}
    for ev in (falls, runs):
        m = ETASModel(kernel="exp", use_alpha=False).fit(ev, seed=1)
        ny = nyblom_test(m)
        ks = m.residual_diagnostics()["ks_pvalue"]
        models[ev.tail] = m
        print(f"\n{m.explain()}")
        print(f"  KS p={ks:.3f}; Nyblom joint={ny['joint']:.2f} "
              f"(crit {ny['crit_joint_5pct']:.2f})"
              f"{' UNSTABLE' if ny['unstable_joint'] else ' (stable)'}")

    # ---------- 2. KAN-PIN comparison (falls, one fit) ----------
    kp = KANPINHawkes(falls, kernel="exp", use_alpha=False, seed=1)
    kp.fit(verbose=False)
    e = kp.eta_hat
    mf = models["fall"]
    print(f"\nKAN-PIN vs MLE (falls): mu {e['mu']:.4f}/{mf.mu:.4f}  "
          f"K0 {e['K0']:.2f}/{mf.K0:.2f}  c {e['c']:.1f}/{mf.c:.1f} days")

    # ---------- 3. overnight gap -> intraday move (daily resolution) ----
    r_intra = pd.Series(np.log(df["Close"].values / df["Open"].values),
                        index=df.index, name="r_intra")
    ev_i = extract_events(r_intra, "fall", Q)
    g = np.log(df["Open"].values[1:] / df["Close"].values[:-1])
    G0 = float(np.quantile(np.abs(g), 0.75))
    keep = np.abs(g) >= G0
    src = GapStream(times=np.flatnonzero(keep) + 1.5, gaps=g[keep],
                    mags=np.abs(g[keep]), G0=G0,
                    dates=pd.DatetimeIndex(df.index[1:][keep]))
    mo = OvernightETAS(ev_i, src).fit(seed=1)
    lr = mo.lr_test_no_overnight(seed=1)
    print(f"\nOvernight-gap trigger (daily resolution): K_s="
          f"{mo.theta['K_s']:.2f}, K_o={mo.theta['K_o']:.2f}, "
          f"LR={lr['LR']:.1f} (p={lr['pvalue']:.3f})")

    # ---------- 4. daily-refit walk-forward, OOS vs Poisson ----------
    rows = []
    warm = {t: None for t in ("fall", "run")}
    for t_now in np.arange(40.0, falls.T):
        day = {}
        ok = True
        for tail, ev in (("fall", falls), ("run", runs)):
            keep_e = ev.times <= t_now
            if keep_e.sum() < 5:
                ok = False
                break
            ev_w = EventData(times=ev.times[keep_e], mags=ev.mags[keep_e],
                             M0=ev.M0, T=float(t_now), tail=tail,
                             timestamps=ev.timestamps[keep_e], index=ev.index)
            m = ETASModel(kernel="exp", use_alpha=False).fit(
                ev_w, seed=1, n_starts=2 if warm[tail] is not None else 6)
            warm[tail] = True
            m.events = ev            # full history, frozen parameters
            p = m.prob_event_within(t_now, 1.0)
            hit = bool(np.any((ev.times > t_now) & (ev.times <= t_now + 1)))
            lam = m.intensity(ev.times[(ev.times > t_now)
                                       & (ev.times <= t_now + 1)])
            lam = np.atleast_1d(lam)
            score = float(np.sum(np.log(np.maximum(lam, 1e-300)))
                          - (m.compensator(t_now + 1) - m.compensator(t_now)))
            rate = keep_e.sum() / t_now
            pois = (int(hit) * np.log(rate)) - rate
            day[tail] = (p, hit, score, pois, m)
        if not ok:
            continue
        rows.append({"t": t_now, "date": falls.time_to_stamp(t_now + 1),
                     "p_fall": day["fall"][0], "p_run": day["run"][0],
                     "hit_fall": day["fall"][1], "hit_run": day["run"][1],
                     "oos_fall": day["fall"][2], "pois_fall": day["fall"][3]})
    wf = pd.DataFrame(rows)
    adv = float((wf["oos_fall"] - wf["pois_fall"]).sum())
    print(f"\nDaily walk-forward: {len(wf)} OOS days; cumulative fall "
          f"log-score vs Poisson: {adv:+.1f}")

    # ---------- 5. nowcast: episode status and implied bottom ----------
    m = models["fall"]
    T = falls.T
    end = m.predict_episode_end(T, quiet_window=5.0, horizon=90.0,
                                n_paths=400, seed=7)
    px = float(df["Close"].iloc[-1])
    rng = np.random.default_rng(11)
    bots = []
    for _ in range(1500):
        t_s, m_s = m.simulate(T, 90.0, rng)
        tau, drop = T, 0.0
        for tt, mm in zip(t_s, m_s):
            if tt - tau >= 5.0:
                break
            tau, drop = tt, drop + mm
        bots.append(px * np.exp(-drop))
    bots = np.array(bots)
    p1d = m.prob_event_within(T, 1.0)
    p1w = m.prob_event_within(T, 5.0)
    print(f"\nNOWCAST (last close ${px:.2f}): P(extreme fall <=1d)={p1d:.2f},"
          f" <=1w={p1w:.2f}; P(fall episode over)={end['p_over_now']:.2f}, "
          f"E[more falls]={end['exp_more_events']:.1f}")
    print(f"episode bottom: median ${np.median(bots):.2f}, 80% "
          f"[${np.quantile(bots, 0.10):.2f}, ${np.quantile(bots, 0.90):.2f}]")

    # ---------- 6. one figure ----------
    fig, axes = plt.subplots(2, 2, figsize=(12, 7.5))
    ax = axes[0, 0]
    px_all = df["Close"].values[1:]
    ax.plot(np.arange(len(px_all)), px_all, color=C["ink2"], lw=1.2)
    ax.set_yscale("log")
    fi = (falls.times - 1).astype(int)
    ri = (runs.times - 1).astype(int)
    ax.scatter(fi, px_all[fi], marker="v", s=26, color=C["fall"], zorder=3,
               label=f"extreme daily falls (n={falls.n})")
    ax.scatter(ri, px_all[ri], marker="^", s=26, color=C["run"], zorder=3,
               label=f"extreme daily runs (n={runs.n})")
    tk = np.linspace(0, len(px_all) - 1, 6).astype(int)
    ax.set_xticks(tk)
    ax.set_xticklabels([r.index[i].strftime("%b %y") for i in tk], fontsize=8)
    ax.set_ylabel("BMNR ($, log scale)")
    ax.set_title("BMNR daily price and extreme events")
    ax.legend(fontsize=8)

    ax = axes[0, 1]
    tg = np.arange(1.0, falls.T + 1.0)
    ax.plot(tg - 1, models["fall"].intensity(tg), color=C["fall"], lw=1.4,
            label="fall intensity")
    ax.plot(tg - 1, models["run"].intensity(tg), color=C["run"], lw=1.4,
            label="run intensity")
    ax.set_xticks(tk)
    ax.set_xticklabels([r.index[i].strftime("%b %y") for i in tk], fontsize=8)
    ax.set_ylabel("events/day")
    ax.set_title("The indicator: conditional intensity")
    ax.legend(fontsize=8)

    ax = axes[1, 0]
    xw = np.arange(len(wf))
    ax.plot(xw, (wf["oos_fall"] - wf["pois_fall"]).cumsum(), color=C["fall"],
            lw=1.8, label="daily-refit ETAS (falls)")
    ax.axhline(0, color=C["ink2"], lw=1)
    ax.text(len(wf) * 0.98, 0, " Poisson benchmark", color=C["ink2"],
            fontsize=8, ha="right", va="bottom")
    tkw = np.linspace(0, len(wf) - 1, 6).astype(int)
    ax.set_xticks(tkw)
    ax.set_xticklabels([wf["date"].iloc[i].strftime("%b %y") for i in tkw],
                       fontsize=8)
    ax.set_ylabel("cumulative OOS log-score vs Poisson")
    ax.set_title("Daily walk-forward, out of sample")
    ax.legend(fontsize=8)

    ax = axes[1, 1]
    ax.plot(xw, wf["p_fall"], color=C["fall"], lw=1.3, label="P(fall tmrw)")
    ax.plot(xw, wf["p_run"], color=C["run"], lw=1.3, label="P(run tmrw)")
    hf = np.flatnonzero(wf["hit_fall"].values)
    hr = np.flatnonzero(wf["hit_run"].values)
    ax.scatter(hf, np.full(len(hf), 1.00), marker="v", s=12, color=C["fall"],
               clip_on=False)
    ax.scatter(hr, np.full(len(hr), 1.05), marker="^", s=12, color=C["run"],
               clip_on=False)
    ax.set_ylim(0, 1.08)
    ax.set_xticks(tkw)
    ax.set_xticklabels([wf["date"].iloc[i].strftime("%b %y") for i in tkw],
                       fontsize=8)
    ax.set_ylabel("probability")
    ax.set_title("Daily nowcasts (markers = realized extreme days)")
    ax.legend(fontsize=8, loc="center right")
    fig.suptitle("BMNR: financial-earthquake indicator on real daily data",
                 fontweight="bold")
    fig.tight_layout()
    fig.savefig(RA.FIGS / "fig_bmnr.png", dpi=150)
    print(f"saved {RA.FIGS.name}/fig_bmnr.png")

    out = {"falls": {k: float(getattr(models['fall'], k))
                     for k in ("mu", "K0", "c")},
           "runs": {k: float(getattr(models['run'], k))
                    for k in ("mu", "K0", "c")},
           "kanpin_falls": e, "overnight_LR": lr["LR"],
           "overnight_p": lr["pvalue"],
           "wf_days": int(len(wf)), "wf_oos_vs_poisson": adv,
           "nowcast": {"p_fall_1d": p1d, "p_fall_1w": p1w,
                       "p_episode_over": end["p_over_now"],
                       "bottom_median": float(np.median(bots)),
                       "bottom_q10": float(np.quantile(bots, 0.10)),
                       "bottom_q90": float(np.quantile(bots, 0.90))}}
    (HERE / "results_BMNR.json").write_text(json.dumps(out, indent=2,
                                                       default=float))
    return wf, out


if __name__ == "__main__":
    import sys
    main(sys.argv[1] if len(sys.argv) > 1 else "BMNR_d.csv")
