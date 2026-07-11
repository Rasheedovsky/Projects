"""
Parameter-stability testing (Nyblom) and walk-forward validation for the
ETAS "financial earthquake" model, with both estimators - the paper's MLE
and the KAN-PIN physics-informed network - refit at every step for
comparison.

Nyblom test for point processes
-------------------------------
Nyblom (1989) / Hansen (1992) test the null "parameters are constant over
the sample" against the alternative that they follow a random walk (i.e.
drift through time).  It is built from the *score* contributions of the
log-likelihood.  For a point process, the log-likelihood decomposes into
one contribution per event:

    l_i(theta) = ln lambda(t_i | H) - [Lambda(t_i) - Lambda(t_{i-1})]

(the last inter-event stretch to T is appended to the final event), with
sum_i l_i = ln L.  Let s_i = d l_i / d theta at the estimate (computed here
by central finite differences), demeaned so they sum to zero, and let
S_i = sum_{k<=i} s_k be the cumulative score.  The statistics are

    individual:  L_j = sum_i S_ij^2 / (N * V_jj),      V = sum_i s_i s_i'
    joint     :  L   = (1/N) * sum_i S_i' V^{-1} S_i

Under parameter constancy these follow a Cramer-von Mises law; Hansen's
(1992) 5% critical values are 0.470 for an individual parameter and, for
the joint test with p parameters, {1: 0.470, 2: 0.749, 3: 1.01, 4: 1.24,
5: 1.47}.  A statistic above the critical value says the parameter has NOT
been stable over the estimation window.

Intuition for this application: if the branching ratio K0 of crash
aftershocks drifts (calm regime -> turbulent regime), the cumulative score
for K0 wanders instead of hovering around zero, and L_{K0} blows up.

Walk-forward validation
-----------------------
An expanding window is refit at regular steps tau_1 < tau_2 < ... < T; at
each step BOTH estimators are run on data up to tau_k only, and their
parameters are scored out of sample on (tau_k, tau_{k+1}] with the proper
scoring rule for point processes, the predictive log-score

    score = sum_{events in segment} ln lambda_theta(t_i)
            - [Lambda_theta(tau_{k+1}) - Lambda_theta(tau_k)]

(history is always the full realised past; only the *parameters* are
frozen at tau_k).  A homogeneous-Poisson benchmark (rate = training event
rate) is scored identically: an estimator only adds value if it beats it.
The Nyblom test recomputed on every expanding window gives the "running"
stability diagnostic requested alongside estimation.

Note on scope: the Nyblom test is a likelihood-score construction, so it
is evaluated at the MLE (it is a test of the model/window, not of the
optimiser).  KAN-PIN's stability is judged by the same walk-forward
machinery: its parameter path across windows and its out-of-sample
log-scores, directly comparable with the MLE's.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from etas import ETASModel, EventData
from kan_pin import KANPINHawkes

# Hansen (1992) 5% critical values for the joint Nyblom test, by number of
# parameters; individual parameters use p = 1.
NYBLOM_CRIT_5PCT = {1: 0.470, 2: 0.749, 3: 1.01, 4: 1.24, 5: 1.47}


# ----------------------------------------------------------------------
# per-event score contributions and the Nyblom statistic
# ----------------------------------------------------------------------

def _active_params(model: ETASModel):
    """(names, values) of the parameters that were actually estimated."""
    names, vals = ["mu", "K0"], [model.mu, model.K0]
    if model.use_alpha:
        names.append("alpha"); vals.append(model.alpha)
    names.append("c"); vals.append(model.c)
    if model.kernel == "power":
        names.append("omega"); vals.append(model.omega)
    return names, np.array(vals, dtype=float)


def _set_active(model: ETASModel, names, vals):
    for n, v in zip(names, vals):
        setattr(model, n, float(v))


def _event_loglik_contributions(model: ETASModel) -> np.ndarray:
    """l_i = ln lambda(t_i) - [Lambda(t_i) - Lambda(t_{i-1})], with the
    final stretch Lambda(T) - Lambda(t_N) charged to the last event so the
    contributions sum exactly to the log-likelihood."""
    ev = model.events
    lam = np.atleast_1d(model.intensity(ev.times))
    Lam = np.atleast_1d(model.compensator(ev.times))
    dLam = np.diff(np.concatenate([[0.0], Lam]))
    li = np.log(np.maximum(lam, 1e-300)) - dLam
    li[-1] -= model.compensator(ev.T) - Lam[-1]
    return li


def event_scores(model: ETASModel) -> tuple[np.ndarray, list[str]]:
    """(N x p) matrix of per-event score contributions at the fitted
    parameters, by central finite differences."""
    names, theta = _active_params(model)
    N = model.events.n
    S = np.zeros((N, len(names)))
    for j, (n, v) in enumerate(zip(names, theta)):
        h = max(1e-6, 1e-4 * abs(v))
        _set_active(model, [n], [v + h])
        lp = _event_loglik_contributions(model)
        _set_active(model, [n], [v - h])
        lm = _event_loglik_contributions(model)
        _set_active(model, [n], [v])          # restore
        S[:, j] = (lp - lm) / (2 * h)
    return S, names


def nyblom_test(model: ETASModel) -> dict:
    """Nyblom/Hansen parameter-stability test at the fitted MLE.

    Returns individual statistics per parameter, the joint statistic, the
    5% critical values, and boolean 'unstable' verdicts."""
    s, names = event_scores(model)
    s = s - s.mean(axis=0, keepdims=True)     # scores sum to ~0 at the MLE
    N, p = s.shape
    V = s.T @ s
    S = np.cumsum(s, axis=0)
    indiv = {n: float(np.sum(S[:, j] ** 2) / (N * V[j, j]))
             for j, n in enumerate(names)}
    # joint statistic with a ridge for near-singular V (collinear scores)
    Vr = V + 1e-10 * np.trace(V) * np.eye(p)
    joint = float(np.trace(np.linalg.solve(Vr, S.T @ S)) / N)
    crit1, critp = NYBLOM_CRIT_5PCT[1], NYBLOM_CRIT_5PCT.get(p, 1.47)
    return {
        "individual": indiv,
        "joint": joint,
        "crit_individual_5pct": crit1,
        "crit_joint_5pct": critp,
        "unstable_individual": {n: v > crit1 for n, v in indiv.items()},
        "unstable_joint": joint > critp,
        "n_params": p,
    }


# ----------------------------------------------------------------------
# out-of-sample predictive log-score for a parameter set
# ----------------------------------------------------------------------

def segment_log_score(params: dict, events_full: EventData,
                      a: float, b: float, kernel: str = "exp",
                      use_alpha: bool = False) -> tuple[float, int]:
    """Predictive log-score of parameter set `params` on segment (a, b]:
    sum ln lambda(t_i) - [Lambda(b) - Lambda(a)], history = full past."""
    m = ETASModel(kernel=kernel, use_alpha=use_alpha,
                  mu=params["mu"], K0=params["K0"],
                  alpha=params.get("alpha", 0.0), c=params["c"],
                  omega=params.get("omega", 1.0),
                  beta_m=float(np.mean(events_full.mags - events_full.M0)))
    m.events = events_full
    inseg = (events_full.times > a) & (events_full.times <= b)
    lam = np.atleast_1d(m.intensity(events_full.times[inseg]))
    score = float(np.sum(np.log(np.maximum(lam, 1e-300)))
                  - (m.compensator(b) - m.compensator(a)))
    return score, int(inseg.sum())


# ----------------------------------------------------------------------
# walk-forward validation with both estimators
# ----------------------------------------------------------------------

def walk_forward(falls: EventData, n_steps: int = 8,
                 start_frac: float = 0.4, kernel: str = "exp",
                 use_alpha: bool = False, kanpin_epochs: int = 600,
                 seed: int = 1, verbose: bool = True) -> pd.DataFrame:
    """Expanding-window walk-forward.

    At each refit time tau_k (from start_frac*T to T in n_steps steps):
      * fit the paper's MLE on events in [0, tau_k]  -> parameter path
      * fit KAN-PIN on the same window               -> parameter path
      * run the Nyblom test on the window (at the MLE) -> running stability
      * score BOTH parameter sets and a Poisson benchmark out of sample on
        (tau_k, tau_{k+1}]  (last window has no OOS segment)

    Returns a tidy DataFrame, one row per refit step.
    """
    T = falls.T
    taus = np.linspace(start_frac * T, T, n_steps)
    rows = []
    for k, tau in enumerate(taus):
        keep = falls.times <= tau
        ev_w = EventData(times=falls.times[keep], mags=falls.mags[keep],
                         M0=falls.M0, T=float(tau), tail=falls.tail,
                         timestamps=falls.timestamps[keep], index=falls.index)
        # ---- paper method (MLE) ----
        mle = ETASModel(kernel=kernel, use_alpha=use_alpha).fit(ev_w, seed=seed)
        ny = nyblom_test(mle)
        # ---- KAN-PIN ----
        kp = KANPINHawkes(ev_w, kernel=kernel, use_alpha=use_alpha, seed=seed)
        kp.fit(epochs=kanpin_epochs, verbose=False)
        ek = kp.eta_hat
        row = {
            "tau": float(tau), "n_events": ev_w.n,
            "date": falls.time_to_stamp(tau),
            "mle_mu": mle.mu, "mle_K0": mle.K0, "mle_c": mle.c,
            "kp_mu": ek["mu"], "kp_K0": ek["K0"], "kp_c": ek["c"],
            "nyblom_joint": ny["joint"],
            "nyblom_crit_joint": ny["crit_joint_5pct"],
            **{f"nyblom_{n}": v for n, v in ny["individual"].items()},
        }
        # ---- out-of-sample scoring on the next segment ----
        if k < len(taus) - 1:
            a, b = float(tau), float(taus[k + 1])
            s_mle, n_seg = segment_log_score(
                {"mu": mle.mu, "K0": mle.K0, "alpha": mle.alpha, "c": mle.c,
                 "omega": mle.omega}, falls, a, b, kernel, use_alpha)
            s_kp, _ = segment_log_score(ek, falls, a, b, kernel, use_alpha)
            rate = ev_w.n / tau                     # Poisson benchmark
            s_pois = n_seg * np.log(rate) - rate * (b - a)
            row.update(oos_events=n_seg, oos_logscore_mle=s_mle,
                       oos_logscore_kanpin=s_kp, oos_logscore_poisson=s_pois)
        rows.append(row)
        if verbose:
            msg = (f"  tau={tau:6.0f} ({row['date'].date()}, {ev_w.n:3d} ev) "
                   f"MLE(mu={mle.mu:.4f} K0={mle.K0:.2f} c={mle.c:5.1f}) "
                   f"KAN-PIN(mu={ek['mu']:.4f} K0={ek['K0']:.2f} c={ek['c']:4.1f}) "
                   f"Nyblom joint={ny['joint']:.2f}"
                   f"{'*' if ny['unstable_joint'] else ''}")
            if "oos_events" in row:
                msg += (f" | OOS lnS: MLE {row['oos_logscore_mle']:.1f} "
                        f"KP {row['oos_logscore_kanpin']:.1f} "
                        f"Pois {row['oos_logscore_poisson']:.1f}")
            print(msg)
    return pd.DataFrame(rows)
