"""Now-cast: train on all labeled history, read the signal at the latest bars.

The walk-forward run in run_pipeline.py scores *historical* test folds.  This
script answers "what is the system saying right now": it fits the causal
forest and the benchmark classifier on every usable (labeled) row, then
evaluates the current state — explosiveness flag, tau(x), regime — on the
most recent bars, which have no labels yet.

Usage: python nowcast.py [--csv data/AA_h.csv] [--horizon 21] [--last 10]
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from bubbles.data import load_yf_csv
from bubbles.features import FeatureConfig, build_features
from bubbles.labels import build_labels
from bubbles.causal_model import fit_causal_forest, fit_direction_benchmark

TREATMENT_DERIVED = {"bubble_flag", "bsadf", "bsadf_gap", "bubble_age"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=os.path.join(os.path.dirname(__file__), "data", "AA_h.csv"))
    ap.add_argument("--horizon", type=int, default=21)
    ap.add_argument("--last", type=int, default=10, help="how many recent bars to score")
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "results"))
    args = ap.parse_args()

    df = load_yf_csv(args.csv)
    cfg = FeatureConfig(cache_path=os.path.join(args.out, "cache_bsadf_cv.json"))
    feats, meta = build_features(df, cfg)
    labels = build_labels(df, feats, horizon=args.horizon)

    all_cols = list(feats.columns)
    x_cols_causal = [c for c in all_cols if c not in TREATMENT_DERIVED]
    n = len(df)

    # training rows: complete features AND a resolved label
    tr = np.where(feats[all_cols].notna().all(axis=1).to_numpy()
                  & labels["fwd_ret"].notna().to_numpy())[0]
    # scoring rows: most recent bars with complete features (labels unresolved)
    score = np.where(feats[all_cols].notna().all(axis=1).to_numpy())[0]
    score = score[score > tr.max()][-args.last:] if (score > tr.max()).any() else score[-args.last:]

    F = feats.to_numpy()
    Xc = feats[x_cols_causal].to_numpy()
    T = feats["bubble_flag"].to_numpy()
    Y = labels["fwd_ret"].to_numpy()
    D = labels["dir_up"].to_numpy()

    cf = fit_causal_forest(Xc[tr], T[tr], Y[tr], Xc[score], seed=42)
    bm = fit_direction_benchmark(F[tr], D[tr], F[score], seed=42)

    cv = meta["bsadf_cv_series"]
    print(f"\n=== Now-cast on {os.path.basename(args.csv)} — last bar "
          f"{df.index[-1]:%Y-%m-%d %H:%M} close={df['Close'].iloc[-1]:.2f} ===")
    print(f"trained on {tr.size} labeled rows; scoring {score.size} recent bars\n")
    hdr = f"{'bar':>18} {'close':>7} {'flag':>4} {'bsadf-cv':>9} {'age':>4} " \
          f"{'tau':>8} {'tau90CI':>18} {'p_up':>5} {'P(bull)':>7} {'P(bear)':>7}"
    print(hdr)
    for k, i in enumerate(score):
        tau = cf["tau"][k] if cf else np.nan
        lb = cf["tau_lb"][k] if cf else np.nan
        ub = cf["tau_ub"][k] if cf else np.nan
        print(f"{df.index[i]:%Y-%m-%d %H:%M} {df['Close'].iloc[i]:7.2f} "
              f"{int(T[i]):>4} {feats['bsadf_gap'].iloc[i]:9.3f} "
              f"{int(feats['bubble_age'].iloc[i]):>4} {tau:8.4f} "
              f"[{lb:7.4f},{ub:7.4f}] {bm['proba_up'][k]:5.2f} "
              f"{feats['hmm_p_bull'].iloc[i]:7.2f} {feats['hmm_p_bear'].iloc[i]:7.2f}")

    i = score[-1]
    print(f"\nlatest-bar context: qlr_f={feats['qlr_f'].iloc[i]:.2f} "
          f"qlr_loc={feats['qlr_loc'].iloc[i]:.2f} cusum={feats['cusum'].iloc[i]:.2f} "
          f"cusum_sq={feats['cusum_sq'].iloc[i]:.2f} bp_nbreaks={feats['bp_nbreaks'].iloc[i]:.0f} "
          f"bp_since={feats['bp_since'].iloc[i]:.0f} bp_dmean={feats['bp_dmean'].iloc[i]:+.5f}")
    print(f"                    rup_ncp={feats['rup_ncp'].iloc[i]:.0f} "
          f"rup_since={feats['rup_since'].iloc[i]:.0f} rup_lvr={feats['rup_lvr'].iloc[i]:+.2f} "
          f"hmm_mu={feats['hmm_mu'].iloc[i]:+.5f} hmm_sig={feats['hmm_sig'].iloc[i]:.4f} "
          f"mom_35={feats['mom_35'].iloc[i]:+.4f} rtadf={feats['rtadf'].iloc[i]:.2f}")
    if cf:
        print(f"\nfull-sample ATE of explosiveness on fwd {args.horizon}-bar return: "
              f"{cf['ate']:+.5f} ± {cf['ate_stderr']:.5f}")


if __name__ == "__main__":
    main()
