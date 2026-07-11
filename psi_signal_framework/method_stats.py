"""
Statistical summary of the psi-signal method from a saved panel
(panel_<measure>.csv.gz produced by detect_flashcrashes.py).

Reports, per panel:
  - psi distribution: mean, sd, skew, kurtosis, quantiles
  - z distribution vs the standard-normal null: observed tail rates at
    1/2/3/4 sigma against expected, skew/kurtosis
  - n_classes distribution and the multi-class lift (alert vs calm windows)
  - signal persistence: autocorrelation of psi, mean alert-streak length
  - episode taxonomy and catalog coverage
  - forward-looking check: future 30-min realized volatility conditional on
    alert vs calm windows (does the signal carry information about what
    comes NEXT, i.e. is it useful for detecting informed trading?)

Run:  python3 method_stats.py results_full_w60 results_full --measure gross_return
"""

import argparse
import os

import numpy as np
import pandas as pd
from scipy import stats

from detect_flashcrashes import FLASH_CRASH_STRICT, KNOWN_EVENTS

ALERT_Z = 3.0


def summarize(outdir: str, measure: str, bars: pd.DataFrame | None) -> dict:
    panel = pd.read_csv(os.path.join(outdir, f"panel_{measure}.csv.gz"),
                        index_col=0, parse_dates=True)
    ep = pd.read_csv(os.path.join(outdir, f"episodes_{measure}.csv"))
    scored = panel.dropna(subset=["z"])
    psi, z = scored["psi"], scored["z"]

    s = {"windows": len(panel), "scored": len(scored)}
    s["psi_mean"], s["psi_sd"] = psi.mean(), psi.std()
    s["psi_skew"], s["psi_kurt"] = psi.skew(), psi.kurt()
    s["psi_q50"], s["psi_q99"] = psi.quantile(0.5), psi.quantile(0.99)

    s["z_skew"], s["z_kurt"] = z.skew(), z.kurt()
    for k in (1, 2, 3, 4):
        s[f"tail_{k}s_obs"] = (z >= k).mean()
        s[f"tail_{k}s_null"] = 1 - stats.norm.cdf(k)

    s["multi_overall"] = (scored["n_classes"] >= 2).mean()
    s["multi_alert"] = (scored.loc[z >= ALERT_Z, "n_classes"] >= 2).mean()
    s["multi_calm"] = (scored.loc[z.abs() < 1, "n_classes"] >= 2).mean()
    s["lift"] = s["multi_alert"] / s["multi_calm"]

    s["psi_ac1"] = psi.autocorr(1)
    s["psi_ac30"] = psi.autocorr(30)

    disp = ep[ep["type"] == "dispersed"]
    s["episodes"] = len(ep)
    s["dispersed"] = len(disp)
    s["catalog_hits"] = sum((disp["date"].astype(str) == d).any()
                            for d in FLASH_CRASH_STRICT)
    pk = pd.to_timedelta(disp["peak_time"].astype(str))
    s["fomc_slot_share"] = ((pk >= pd.Timedelta("13:55:00"))
                            & (pk <= pd.Timedelta("14:20:00"))).mean()

    # forward information: vol expansion = next-30-min vol / trailing-60-min
    # vol (deseasonalizes the intraday U-shape), by signal state
    if bars is not None:
        r = np.log(bars["close"]).groupby(bars.index.normalize()).diff()
        trail = r.rolling(60, min_periods=40).std()
        fwd = r.iloc[::-1].rolling(30, min_periods=20).std().iloc[::-1].shift(-30)
        expansion = (fwd / trail).reindex(scored.index)
        dispr = scored["sigma"] / scored["sigma_med"]
        groups = {
            "exp_disp_alert": (z >= ALERT_Z) & (dispr >= 1.5),
            "exp_comp_alert": (z >= ALERT_Z) & (dispr <= 0.67),
            "exp_calm": z.abs() < 1,
        }
        for key, m in groups.items():
            e = expansion[m & expansion.notna()]
            s[key] = e.median()
            s[f"{key}_Pup"] = (e > 1).mean()
            s[f"{key}_n"] = len(e)
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("outdirs", nargs="+")
    ap.add_argument("--measure", default="gross_return")
    ap.add_argument("--csv", default="../data/spy_1min_2008_2021_cleaned.csv")
    args = ap.parse_args()

    bars = None
    if args.csv and os.path.exists(args.csv):
        from psi_signal import load_minute_bars
        bars = load_minute_bars(args.csv)

    table = {}
    for d in args.outdirs:
        label = d.rstrip("/").split("/")[-1]
        table[label] = summarize(d, args.measure, bars)
    df = pd.DataFrame(table)
    with pd.option_context("display.float_format", lambda v: f"{v:,.4g}"):
        print(df.to_string())


if __name__ == "__main__":
    main()
