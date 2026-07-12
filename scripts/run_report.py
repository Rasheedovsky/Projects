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
    """meta_vb_mae: logistic trade filter fit on inner-val predictions."""
    from sklearn.linear_model import LogisticRegression

    per_split: dict[int, np.ndarray] = {}
    for sid in range(len(SPLITS)):
        seed_probs = []
        for seed in PILOT["train"]["seeds"]:
            z = np.load(PRED_DIR / f"vb_mae_s{seed}_split{sid}.npz")
            vi, vp = z["val_idx"], z["val_prob"]
            hit = (np.where(vp >= 0.5, 1, -1) == Y[vi]).astype(int)
            Xv = np.column_stack([np.abs(vp - 0.5), TAB[vi]])
            lr = LogisticRegression(max_iter=1000)
            fitted = len(np.unique(hit)) > 1
            if fitted:
                lr.fit(Xv, hit)
            p = np.full(N, np.nan)
            Xt = np.column_stack([np.abs(split_prob("vb_mae", sid) - 0.5), TAB])
            take = lr.predict_proba(Xt)[:, 1] if fitted else np.full(N, 1.0)
            p[z["idx"]] = take[z["idx"]]
            seed_probs.append(p)
        per_split[sid] = np.nanmean(np.vstack(seed_probs), axis=0)
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


def net_returns(prob: np.ndarray, cost_bp: float) -> np.ndarray:
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
    pbo = cscv_pbo(Rmat, s_partitions=8)
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

    # ---- secondary hypotheses -------------------------------------------
    def sh(c):
        return evals[c]["sharpe_mean"]

    S = {
        "S1 3D beats 2D control": sh("vb_scratch") > sh("vb2d"),
        "S2 dual beats single-axis": sh("vb_scratch") > max(sh("clock_only"),
                                                            sh("info_only")),
        "S3 V-C beats 2D control": sh("vc") > sh("vb2d"),
        "S4 primary beats Gao OLS (DM p<0.05)": dm_gao[0] > 0 and dm_gao[1] < 0.05,
        "S4b primary beats XGBoost (DM p<0.05)": dm_xgb[0] > 0 and dm_xgb[1] < 0.05,
        "S5 primary beats MiniRocket": sh("vb_mae") > sh("minirocket"),
        "S6 SSL pretraining helps": sh("vb_mae") > sh("vb_scratch"),
        "S7 depth shuffle hurts": sh("vb_scratch") > sh("vb_shuffled"),
    }

    # ---- success criteria (pilot analogs) --------------------------------
    n_traded = int((prim["consensus_prob"] != 0.5).sum())
    ci = 1.96 * np.sqrt(0.25 / max(n_traded, 1))
    crit = {
        "hit >= 52.5% with CI excluding 50%": prim["hit_mean"] >= 0.525
        and prim["hit_mean"] - ci > 0.5,
        "net Sharpe > 0 on >= 4/5 paths @1bp": sum(
            s > 0 for s in prim["path_sharpes"]) >= 4,
        "mean path Sharpe >= 0.8": prim["sharpe_mean"] >= 0.8,
        "DSR >= 0.95": prim["dsr"] >= 0.95,
        "PBO < 40%": pbo < 0.40,
        "design claim S1+S2+S3": S["S1 3D beats 2D control"]
        and S["S2 dual beats single-axis"] and S["S3 V-C beats 2D control"],
        "beats sequence baselines (S5)": S["S5 primary beats MiniRocket"],
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
              f"CSCV PBO (S=8): **{pbo:.2f}**; headline cost {HEADLINE} bp/side.\n")
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
    md.append("| trial | " + " | ".join(f"{c}bp" for c in grid) + " |")
    md.append("|---|" + "---|" * len(grid))
    for c in ["vb_mae", "vb_scratch", "vb2d", "vc", "gao_ols", "xgboost",
              "minirocket", "always_long"]:
        e = evals[c]
        md.append(f"| {c} | " + " | ".join(f"{e['cost_curve'][g]:.2f}" for g in grid) + " |")
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
    md.append("\n## Secondary hypotheses\n")
    for k, v in S.items():
        md.append(f"- {k}: **{'PASS' if v else 'FAIL'}**")
    md.append("\n## Success criteria\n")
    for k, v in crit.items():
        md.append(f"- {k}: **{'PASS' if v else 'FAIL'}**")
    md.append(f"\n**Verdict: {verdict}**\n")
    md.append("![equity](figures/equity_curves.png)\n")
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
