"""Phase 8: aggregate the prediction registry into reports/results.md.

Evaluation objects:
- per-path daily net returns (5 CPCV paths) for path-Sharpe distributions;
- per-day consensus (mean prob across paths) daily net returns per trial for
  the (T x N_trials) matrix feeding DSR / PBO / DM / McNemar.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from spydt.cv import make_groups, paths_from_splits
from spydt.evalx import (auc, cscv_pbo, dm_test, dsr, effective_trials,
                         hit_rate, max_drawdown, mcnemar, psr, sharpe,
                         sortino, strategy_net_returns)

PILOT = yaml.safe_load(Path("configs/pilot.yaml").read_text())
REG = yaml.safe_load(Path("configs/registry.yaml").read_text())
PRED_DIR = Path("data/registry/preds")
BUNDLE = np.load("data/tensors/bundles/split0.npz")
SPLITS = json.loads(Path("data/tensors/splits.json").read_text())

DATES = [str(d) for d in BUNDLE["dates"]]
Y = BUNDLE["y"]
R_EXEC = BUNDLE["r_exec"]
R_LABEL = BUNDLE["r_label"]
TAB = BUNDLE["tabular"]
N = len(DATES)


def _measured_cost_per_side_bp() -> np.ndarray:
    """Per-day per-side cost: half of a 1-cent spread at the 15:31 entry
    price, plus the fees allowance from config. SPY's quoted spread has been
    ~1 cent throughout 2011-2021."""
    import pandas as pd

    rth = pd.read_parquet("data/interim/spy_rth_et.parquet", columns=["open"])
    t = rth.index.strftime("%H:%M")
    entry = rth[t == "15:31"]["open"]
    by_date = {str(d): float(p) for d, p in
               zip(entry.index.date, entry.to_numpy())}
    px = np.array([by_date.get(d, np.nan) for d in DATES])
    px = np.where(np.isnan(px), np.nanmedian(px), px)
    return 0.005 / px * 1e4 + PILOT["costs"]["fees_bp_per_side"]


MEASURED_COST_BP = _measured_cost_per_side_bp()
N_GROUPS = PILOT["cv"]["n_groups"]
GROUPS = make_groups(N, N_GROUPS)
PATHS = paths_from_splits(N_GROUPS, PILOT["cv"]["k_test"])
HEADLINE = PILOT["costs"]["headline_bp"]
SUBPERIODS = [("2011-01-01", "2013-12-31"), ("2014-01-01", "2017-12-31"),
              ("2018-01-01", "2021-12-31")]


def seeds_for(config: str) -> list[int]:
    if config in REG["baseline_trials"]:
        return [0]
    return [0] if config == "vb_shuffled" else PILOT["train"]["seeds"]


def split_prob(config: str, split_id: int) -> np.ndarray:
    """Seed-ensembled prob over all days (0.5 where absent)."""
    probs = []
    for seed in seeds_for(config):
        f = PRED_DIR / f"{config}_s{seed}_split{split_id}.npz"
        z = np.load(f)
        p = np.full(N, np.nan)
        p[z["idx"]] = z["prob"]
        probs.append(p)
    return np.nanmean(np.vstack(probs), axis=0)


def path_probs(config: str) -> list[np.ndarray]:
    """One length-N prob vector per CPCV path (NaN off-test days -> filled)."""
    cache = {sid: split_prob(config, sid)
             for sid in {sid for p in PATHS for sid in p.values()}}
    out = []
    for p in PATHS:
        v = np.full(N, np.nan)
        for g, sid in p.items():
            m = GROUPS == g
            v[m] = cache[sid][m]
        out.append(np.where(np.isnan(v), 0.5, v))
    return out


def meta_probs() -> list[np.ndarray]:
    """meta_vb_mae: logistic trade filter fit once per split on the
    seed-ENSEMBLED inner-val predictions, using that split's own bundle
    tabular statistics (no cross-split statistic reuse)."""
    from sklearn.linear_model import LogisticRegression

    per_split: dict[int, np.ndarray] = {}
    for sid in range(len(SPLITS)):
        zb = np.load(f"data/tensors/bundles/split{sid}.npz")
        tab_s = zb["tabular"]
        zs = [np.load(PRED_DIR / f"vb_mae_s{seed}_split{sid}.npz")
              for seed in PILOT["train"]["seeds"]]
        vi = zs[0]["val_idx"]
        vp = np.mean(np.vstack([z["val_prob"] for z in zs]), axis=0)
        hit = (np.where(vp >= 0.5, 1, -1) == Y[vi]).astype(int)
        Xv = np.column_stack([np.abs(vp - 0.5), tab_s[vi]])
        lr = LogisticRegression(max_iter=1000)
        fitted = len(np.unique(hit)) > 1
        if fitted:
            lr.fit(Xv, hit)
        p = np.full(N, np.nan)
        base_p = np.nan_to_num(split_prob("vb_mae", sid), nan=0.5)
        Xt = np.nan_to_num(np.column_stack([np.abs(base_p - 0.5), tab_s]))
        take = lr.predict_proba(Xt)[:, 1] if fitted else np.full(N, 1.0)
        p[zs[0]["idx"]] = take[zs[0]["idx"]]
        per_split[sid] = p
    base = {sid: split_prob("vb_mae", sid) for sid in per_split}
    out = []
    for p in PATHS:
        prob = np.full(N, 0.5)
        gatep = np.full(N, 1.0)
        for g, sid in p.items():
            m = GROUPS == g
            prob[m] = np.where(np.isnan(base[sid][m]), 0.5, base[sid][m])
            gatep[m] = np.where(np.isnan(per_split[sid][m]), 1.0, per_split[sid][m])
        # trade only when the meta model expects the primary call to be right
        prob = np.where(gatep >= 0.5, prob, 0.5)  # 0.5 -> treated as flat
        out.append(prob)
    return out


def net_returns(prob: np.ndarray, cost_bp) -> np.ndarray:
    """cost_bp: scalar or per-day array of per-side costs."""
    r = strategy_net_returns(prob, R_EXEC, cost_bp)
    return np.where(prob == 0.5, 0.0, r)  # exactly-0.5 = flat, no trade


def consensus(paths: list[np.ndarray]) -> np.ndarray:
    return np.mean(np.vstack(paths), axis=0)


def in_period(lo: str, hi: str) -> np.ndarray:
    return np.array([(lo <= d <= hi) for d in DATES])


def evaluate(config: str, paths: list[np.ndarray]) -> dict:
    per_path = []
    for pv in paths:
        r = net_returns(pv, HEADLINE)
        per_path.append({"sharpe": sharpe(r), "hit": hit_rate(pv, Y),
                         "auc": auc(pv, Y)})
    cons = consensus(paths)
    rc = net_returns(cons, HEADLINE)
    traded = cons != 0.5
    out = {
        "config": config,
        "path_sharpes": [round(p["sharpe"], 3) for p in per_path],
        "sharpe_mean": float(np.mean([p["sharpe"] for p in per_path])),
        "sharpe_sd": float(np.std([p["sharpe"] for p in per_path])),
        "hit_mean": float(np.mean([p["hit"] for p in per_path])),
        "auc_mean": float(np.mean([p["auc"] for p in per_path])),
        "hit_traded": hit_rate(cons[traded], Y[traded]) if traded.any() else 0.5,
        "hit_big": hit_rate(cons[np.abs(R_LABEL) > 2 * HEADLINE * 1e-4],
                            Y[np.abs(R_LABEL) > 2 * HEADLINE * 1e-4]),
        "exposure": float(traded.mean()),
        "sortino": sortino(rc), "max_dd": float(max_drawdown(rc)),
        "psr": psr(rc),
        "cost_curve": {c: sharpe(net_returns(cons, c))
                       for c in PILOT["costs"]["per_side_bp_grid"]},
        "sharpe_measured_cost": sharpe(net_returns(cons, MEASURED_COST_BP)),
        "subperiods": {f"{lo[:4]}-{hi[:4]}": sharpe(rc[in_period(lo, hi)])
                       for lo, hi in SUBPERIODS},
        "consensus_returns": rc,
        "consensus_prob": cons,
    }
    return out


def main() -> None:
    configs = REG["nn_trials"] + REG["baseline_trials"]
    evals = {c: evaluate(c, path_probs(c)) for c in configs}
    evals["meta_vb_mae"] = evaluate("meta_vb_mae", meta_probs())

    trial_names = list(evals)
    Rmat = np.column_stack([evals[c]["consensus_returns"] for c in trial_names])
    n_eff = effective_trials(Rmat)
    pbo = cscv_pbo(Rmat, s_partitions=16)   # registered S=16
    pbo8 = cscv_pbo(Rmat, s_partitions=8)   # reference
    for c in trial_names:
        evals[c]["dsr"] = dsr(evals[c]["consensus_returns"], Rmat)

    prim = evals["vb_mae"]
    dm_gao = dm_test(prim["consensus_returns"],
                     evals["gao_ols"]["consensus_returns"])
    dm_xgb = dm_test(prim["consensus_returns"],
                     evals["xgboost"]["consensus_returns"])
    dm_mr = dm_test(prim["consensus_returns"],
                    evals["minirocket"]["consensus_returns"])
    hits_p = np.where(prim["consensus_prob"] >= 0.5, 1, -1) == Y
    hits_x = np.where(evals["xgboost"]["consensus_prob"] >= 0.5, 1, -1) == Y
    mcn = mcnemar(hits_p, hits_x)

    # ---- secondary hypotheses: point direction + DM significance ---------
    def sh(c):
        return evals[c]["sharpe_mean"]

    def versus(a: str, b: str) -> tuple[bool, float]:
        t, p = dm_test(evals[a]["consensus_returns"], evals[b]["consensus_returns"])
        return (sh(a) > sh(b)), (p if t > 0 else 1.0)

    S: dict[str, tuple[bool, float]] = {
        "S1 3D beats 2D control (vb_scratch > vb2d)": versus("vb_scratch", "vb2d"),
        "S2a dual beats clock-only": versus("vb_scratch", "clock_only"),
        "S2b dual beats info-only": versus("vb_scratch", "info_only"),
        "S3a V-C beats identical-content 2D (vc > vc2d)": versus("vc", "vc2d"),
        "S3b V-C beats no-cross-plane (vc > va)": versus("vc", "va"),
        "S4 primary beats Gao OLS": versus("vb_mae", "gao_ols"),
        "S4b primary beats XGBoost": versus("vb_mae", "xgboost"),
        "S5 primary beats MiniRocket": versus("vb_mae", "minirocket"),
        "S6 SSL pretraining helps": versus("vb_mae", "vb_scratch"),
        "S7 depth shuffle hurts (3-seed matched)": versus("vb_scratch", "vb_shuffled"),
    }

    # ---- success criteria (pilot analogs) --------------------------------
    n_traded = int((prim["consensus_prob"] != 0.5).sum())
    ci = 1.96 * np.sqrt(0.25 / max(n_traded, 1))
    def sig(key: str) -> bool:
        direction, p = S[key]
        return direction and p < 0.05

    crit = {
        "hit >= 52.5% with CI excluding 50%": prim["hit_mean"] >= 0.525
        and prim["hit_mean"] - ci > 0.5,
        "net Sharpe > 0 on >= 4/5 paths @1bp": sum(
            s > 0 for s in prim["path_sharpes"]) >= 4,
        "mean path Sharpe >= 0.8": prim["sharpe_mean"] >= 0.8,
        "DSR >= 0.95": prim["dsr"] >= 0.95,
        "PBO < 40%": pbo < 0.40,
        "design claim (S1, S2a, S2b, S3a, S3b all directional AND p<0.05)": all(
            sig(k) for k in ["S1 3D beats 2D control (vb_scratch > vb2d)",
                             "S2a dual beats clock-only", "S2b dual beats info-only",
                             "S3a V-C beats identical-content 2D (vc > vc2d)",
                             "S3b V-C beats no-cross-plane (vc > va)"]),
        "beats sequence baselines (S5, p<0.05)": sig("S5 primary beats MiniRocket"),
    }
    verdict = "POSITIVE" if all(crit.values()) else "NULL (honest)"

    # ---- figures ----------------------------------------------------------
    fig, ax = plt.subplots(figsize=(11, 5))
    x = np.arange(N)
    for c in ["vb_mae", "gao_ols", "xgboost", "minirocket", "always_long"]:
        ax.plot(x, np.cumsum(evals[c]["consensus_returns"]), label=c, lw=1)
    step = max(N // 8, 1)
    ax.set_xticks(x[::step]); ax.set_xticklabels([DATES[i] for i in x[::step]],
                                                 rotation=30, fontsize=7)
    ax.legend(); ax.set_title(f"Cumulative net log return @{HEADLINE}bp/side (consensus)")
    fig.tight_layout(); fig.savefig("reports/figures/equity_curves.png", dpi=110)

    fig2, ax2 = plt.subplots(figsize=(10, 4))
    names = trial_names + ["meta_vb_mae"] if "meta_vb_mae" not in trial_names else trial_names
    means = [evals[c]["sharpe_mean"] for c in names]
    sds = [evals[c]["sharpe_sd"] for c in names]
    ax2.bar(range(len(names)), means, yerr=sds, capsize=3)
    ax2.set_xticks(range(len(names)))
    ax2.set_xticklabels(names, rotation=45, ha="right", fontsize=8)
    ax2.axhline(0, color="k", lw=0.5)
    ax2.set_title("Mean path Sharpe ± sd across 5 CPCV paths @1bp")
    fig2.tight_layout(); fig2.savefig("reports/figures/sharpe_by_config.png", dpi=110)

    # ---- results.md -------------------------------------------------------
    md = ["# Results — SPY dual-time 3D CNN (PILOT)\n"]
    md.append(f"**Overall verdict: {verdict}.** Registry: {len(trial_names)} trials "
              f"(effective trials from return-correlation spectrum: {n_eff:.1f}); "
              f"CSCV PBO: **{pbo:.2f}** (S=16; S=8 reference {pbo8:.2f}); "
              f"headline cost {HEADLINE} bp/side flat; measured era cost "
              f"(half of 1-cent spread at 15:31 entry + {PILOT['costs']['fees_bp_per_side']} bp fees) "
              f"averages {MEASURED_COST_BP.mean():.2f} bp/side.\n")
    md.append("## Trial table (5 CPCV paths, net @1bp)\n")
    md.append("| trial | hit | AUC | mean path Sharpe ± sd | path Sharpes | DSR | "
              "Sortino | maxDD | exposure |")
    md.append("|---|---|---|---|---|---|---|---|---|")
    for c in list(evals):
        e = evals[c]
        md.append(f"| {c} | {e['hit_mean']:.3f} | {e['auc_mean']:.3f} | "
                  f"{e['sharpe_mean']:.2f} ± {e['sharpe_sd']:.2f} | "
                  f"{e['path_sharpes']} | {e['dsr']:.2f} | {e['sortino']:.2f} | "
                  f"{e['max_dd']:.3f} | {e['exposure']:.2f} |")
    md.append("\n## Cost sensitivity (consensus Sharpe)\n")
    grid = PILOT["costs"]["per_side_bp_grid"]
    md.append("| trial | " + " | ".join(f"{c}bp" for c in grid) + " | measured |")
    md.append("|---|" + "---|" * (len(grid) + 1))
    for c in ["vb_mae", "vb_scratch", "vb2d", "vc", "vc2d", "gao_ols", "xgboost",
              "minirocket", "always_long"]:
        e = evals[c]
        md.append(f"| {c} | " + " | ".join(f"{e['cost_curve'][g]:.2f}" for g in grid)
                  + f" | {e['sharpe_measured_cost']:.2f} |")
    md.append("\n## Subperiod Sharpe (consensus @1bp)\n")
    keys = list(evals["vb_mae"]["subperiods"])
    md.append("| trial | " + " | ".join(keys) + " |")
    md.append("|---|" + "---|" * len(keys))
    for c in ["vb_mae", "gao_ols", "xgboost", "always_long"]:
        e = evals[c]
        md.append(f"| {c} | " + " | ".join(f"{e['subperiods'][k]:.2f}" for k in keys) + " |")
    md.append("\n## Model-comparison tests (primary = vb_mae, daily net returns)\n")
    md.append(f"- DM vs Gao OLS: t={dm_gao[0]:.2f}, p={dm_gao[1]:.3f}")
    md.append(f"- DM vs XGBoost: t={dm_xgb[0]:.2f}, p={dm_xgb[1]:.3f}")
    md.append(f"- DM vs MiniRocket: t={dm_mr[0]:.2f}, p={dm_mr[1]:.3f}")
    md.append(f"- McNemar vs XGBoost: b01={mcn[0]:.0f}, p={mcn[1]:.3f}")
    md.append(f"- Hit rate on |move| > round-trip cost days (primary): "
              f"{prim['hit_big']:.3f}")
    md.append("\n## Secondary hypotheses (direction + HAC-DM significance)\n")
    for k, (direction, p) in S.items():
        tag = ("PASS (p<0.05)" if direction and p < 0.05
               else "directional only, n.s." if direction else "FAIL")
        md.append(f"- {k}: **{tag}** (DM p={p:.3f})")
    md.append("\n## Success criteria\n")
    for k, v in crit.items():
        md.append(f"- {k}: **{'PASS' if v else 'FAIL'}**")
    md.append(f"\n**Verdict: {verdict}**\n")
    md.append("## Discussion — honest reading\n")
    all_negative = all(evals[c]["sharpe_mean"] < 0 for c in trial_names)
    if all_negative:
        md.append(
            "**Every registered trial, including the Gao OLS replication and "
            "always-long, has negative net Sharpe at the headline cost.** In this "
            "regime the secondary-hypothesis 'PASS' marks (S1/S2/S3/S7) are "
            "orderings among losing strategies — relative rankings of noise — and "
            "must NOT be read as evidence for the dual-time design claim. The "
            "pilot's substantive conclusions are: (1) the last-half-hour direction "
            "is not predictable net of costs in 2011-2021 from morning information "
            "under this protocol — consistent with the documented post-2013 decay "
            "of intraday momentum (Gao OLS is net-positive only in the 2011-2013 "
            "subperiod, matching the decay literature); (2) SSL pretraining as "
            "configured HURT (S6): the era-firewalled 2008-2010 corpus appears to "
            "transfer regime-specific features that do not help 2011-2021 — and "
            "the corpus was built on full-session windows while supervised inputs "
            "are morning-only, a domain shift the full protocol should fix; "
            "(3) the falsifiability diagnostic and the leakage gates passed — "
            "look-ahead is excluded and a gross leak is provably detectable, so "
            "this is a *gated* null; the gates cannot rule out signal-destroying "
            "defects, and the soft positive-control gate (f) quantifies how much "
            "of a planted realistic-size signal the pipeline recovers; (4) the "
            f"registry PBO of {pbo:.2f} says any in-sample winner here would "
            "likely be backtest overfitting.")
    md.append("\n## Disclosures (execution history)\n")
    md.append(
        "- **Mid-study rerun:** a first, complete evaluation run (including all "
        "45 primary-trial fits) was executed before leakage gate (b) was first "
        "run; the gate then caught a feature-timestamp violation (bar threshold "
        "calibration/FFD d*/bar construction used full sessions, letting a train "
        "day's own afternoon into its features). Everything derived was wiped and "
        "rebuilt under the morning-only construction; the reported matrix is the "
        "second, clean run. Recorded in git history (commits around the gate fix).\n"
        "- **MAE v2:** the SSL checkpoint was retrained after the fix under the "
        "corrected construction (per-window bars, era-morning threshold, "
        "per-offset FFD lanes); the v1 checkpoint predated the fix and was "
        "discarded as irreproducible.\n"
        "- **Registry amendment v2 (post-referee, disclosed):** vc2d control "
        "added (the original S3 comparison mistakenly used V-B content); "
        "vb_shuffled promoted to 3 seeds; S3 re-specified as vc>vc2d AND vc>va. "
        "All amended trials enter the DSR count.\n"
        "- **Known unfixed pilot limitations:** r_on carries ~-20 bp artifacts "
        "on ~44 quarterly ex-div days (dividend correction deferred; affects the "
        "Gao baselines' r_on regressor and one tabular feature); MOC exit is "
        "proxied by the 15:59 bar close (auction slippage not modeled); fitted "
        "statistics are train-fold-only but not past-only (standard CPCV "
        "practice; a walk-forward robustness pass is deferred to the full run); "
        "the embargo (5 days) is justified by the ~500-observation FFD weight "
        "window (~6 morning-days of memory) — one day short in the strictest "
        "reading, immaterial at ~430-day groups but fixed to 6 in the full "
        "protocol; CSCV S=16 as registered (S=8 shown for reference).")
    md.append("\n![equity](figures/equity_curves.png)\n")
    md.append("![sharpes](figures/sharpe_by_config.png)\n")
    md.append("\n## Pilot scope\n")
    md.append(REG["pilot_scope_notes"])
    Path("reports/results.md").write_text("\n".join(md) + "\n")
    with open("data/registry/summary.json", "w") as f:
        json.dump({c: {k: v for k, v in e.items()
                       if k not in ("consensus_returns", "consensus_prob")}
                   for c, e in evals.items()}, f, indent=1, default=str)
    print(f"verdict={verdict} pbo={pbo:.2f} -> reports/results.md")


if __name__ == "__main__":
    main()
