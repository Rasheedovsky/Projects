"""Causal Random Forest meta-model (Wager & Athey 2018; Athey, Tibshirani &
Wager 2019 — Generalized Random Forests), via econml's CausalForestDML.

Framing
-------
outcome    Y = forward h-bar log return
treatment  T = 1{BSADF exceeds its 95% critical value}  (the market is in a
               statistically detected explosive regime *now*)
covariates X = all other test statistics (stability, breaks, HMM regime
               probabilities, momentum/vol controls)

The forest estimates the conditional average treatment effect
    tau(x) = E[Y | T=1, X=x] - E[Y | T=0, X=x]
i.e. *the direction and magnitude of the price move that the detected
explosiveness adds, given the current stability/regime context*:

    tau(x) > 0  ->  explosiveness predicts continuation (bubble growth leg)
    tau(x) < 0  ->  explosiveness predicts reversal (collapse imminent)

DML residualizes Y and T on X with cross-fitted nuisance models before the
forest is grown, which removes the confounding that plagues a plain
"predict returns from bubble signals" regression.  Nuisance cross-fitting
uses contiguous (non-shuffled) folds to respect serial dependence.
"""
from __future__ import annotations

import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import KFold

try:
    from econml.dml import CausalForestDML
    HAS_ECONML = True
except ImportError:              # pragma: no cover
    HAS_ECONML = False

MIN_TREATED = 25                 # minimum treated/control rows per fold


def fit_causal_forest(X_tr, T_tr, Y_tr, X_te, seed: int = 42):
    """Fit CausalForestDML on a training fold; return CATE estimates on test.

    Returns dict with tau (point), tau_lb/tau_ub (90% CI), ate, ate_stderr —
    or None when the fold lacks treatment variation or econml is missing.
    """
    if not HAS_ECONML:
        return None
    T_tr = np.asarray(T_tr).astype(int)
    n1 = int(T_tr.sum())
    n0 = int((1 - T_tr).sum())
    if n1 < MIN_TREATED or n0 < MIN_TREATED:
        return None

    est = CausalForestDML(
        model_y=RandomForestRegressor(n_estimators=300, min_samples_leaf=20,
                                      max_depth=8, random_state=seed),
        model_t=RandomForestClassifier(n_estimators=300, min_samples_leaf=20,
                                       max_depth=8, random_state=seed),
        discrete_treatment=True,
        cv=KFold(n_splits=3, shuffle=False),      # contiguous blocks, time-aware
        n_estimators=800,
        min_samples_leaf=15,
        max_samples=0.45,
        random_state=seed,
    )
    est.fit(np.asarray(Y_tr), T_tr, X=np.asarray(X_tr))
    tau = est.effect(np.asarray(X_te))
    lb, ub = est.effect_interval(np.asarray(X_te), alpha=0.10)
    inf = est.effect_inference(np.asarray(X_te)).population_summary()
    return {
        "tau": tau, "tau_lb": lb, "tau_ub": ub,
        "ate": float(inf.mean_point), "ate_stderr": float(inf.stderr_mean),
        "n_treated_train": n1, "n_control_train": n0,
    }


def fit_direction_benchmark(X_tr, y_tr, X_te, seed: int = 42):
    """Plain random-forest direction classifier (the non-causal benchmark)."""
    clf = RandomForestClassifier(n_estimators=500, min_samples_leaf=15,
                                 max_depth=10, random_state=seed, n_jobs=-1)
    clf.fit(np.asarray(X_tr), np.asarray(y_tr).astype(int))
    proba = clf.predict_proba(np.asarray(X_te))[:, 1]
    return {"proba_up": proba, "importances": clf.feature_importances_}
