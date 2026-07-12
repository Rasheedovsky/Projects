"""Walk-forward Gaussian HMM regime features (Hamilton 1989 regime switching,
estimated as an HMM via EM).

Look-ahead is avoided on two fronts:
  1.  The HMM is refit every ``refit_every`` bars using ONLY past returns.
  2.  Between refits, state probabilities are *filtered* — computed with the
      forward recursion P(state_t | r_1..r_t) — never the smoothed
      (forward-backward) posteriors that hmmlearn's ``predict_proba`` returns,
      which would condition on future observations.

Features per bar (states sorted by mean return):
  hmm_p_bull : filtered probability of the highest-mean state
  hmm_p_bear : filtered probability of the lowest-mean state
  hmm_mu     : filtered one-step expected return  sum_s p_s * mu_s
  hmm_sig    : filtered expected volatility       sum_s p_s * sigma_s
  hmm_entropy: entropy of the filtered state distribution (regime ambiguity)
"""
from __future__ import annotations

import numpy as np
from hmmlearn.hmm import GaussianHMM


def _fit_hmm(r_hist: np.ndarray, n_states: int, seed: int):
    model = GaussianHMM(n_components=n_states, covariance_type="diag",
                        n_iter=200, tol=1e-4, random_state=seed)
    model.fit(r_hist[:, None])
    return model


def _filtered_probs(model, r: np.ndarray) -> np.ndarray:
    """Forward-algorithm filtered probabilities P(state_t | r_1..t)."""
    mu = model.means_.ravel()
    sd = np.sqrt(model.covars_.reshape(model.n_components, -1)[:, 0])
    A = model.transmat_
    pi = model.startprob_
    n, k = r.size, mu.size
    out = np.empty((n, k))
    # emission likelihoods
    z = (r[:, None] - mu[None, :]) / sd[None, :]
    em = np.exp(-0.5 * z * z) / (np.sqrt(2 * np.pi) * sd[None, :])
    em = np.maximum(em, 1e-300)
    alpha = pi * em[0]
    alpha /= alpha.sum()
    out[0] = alpha
    for t in range(1, n):
        alpha = (alpha @ A) * em[t]
        s = alpha.sum()
        if not np.isfinite(s) or s <= 0:
            alpha = np.full(k, 1.0 / k)
        else:
            alpha /= s
        out[t] = alpha
    return out


def walkforward_hmm(returns: np.ndarray, n_states: int = 3, min_history: int = 300,
                    refit_every: int = 50, seed: int = 42,
                    max_history: int | None = None, filter_warm: int = 2000):
    """``max_history`` caps the EM training sample (rolling window) and
    ``filter_warm`` bounds the forward-filter burn-in before each scored
    block — filtered probabilities forget their initial condition
    geometrically fast, so a warm start from a uniform distribution
    ``filter_warm`` bars back is indistinguishable from filtering the full
    history while keeping large intraday samples tractable."""
    r = np.asarray(returns, dtype=np.float64)
    n = r.size
    cols = {k: np.full(n, np.nan) for k in
            ("hmm_p_bull", "hmm_p_bear", "hmm_mu", "hmm_sig", "hmm_entropy")}

    t0 = min_history
    while t0 < n:
        t1 = min(t0 + refit_every, n)
        h0 = max(0, t0 - max_history) if max_history else 0
        hist = r[h0:t0]
        scale = hist.std() if hist.std() > 0 else 1.0
        model = None
        for k in (n_states, 2):
            try:
                m = _fit_hmm(hist / scale, k, seed)
                if np.all(np.isfinite(m.transmat_)):
                    model = m
                    break
            except Exception:
                continue
        if model is not None:
            # causal filtering, warm-started shortly before the scored block
            w0 = max(0, t0 - filter_warm)
            probs_seg = _filtered_probs(model, r[w0:t1] / scale)
            probs = np.full((t1, model.n_components), np.nan)
            probs[w0:t1] = probs_seg
            mu = model.means_.ravel() * scale
            sd = np.sqrt(model.covars_.reshape(model.n_components, -1)[:, 0]) * scale
            order = np.argsort(mu)                      # bear ... bull
            for t in range(t0, t1):
                p = probs[t]
                cols["hmm_p_bull"][t] = p[order[-1]]
                cols["hmm_p_bear"][t] = p[order[0]]
                cols["hmm_mu"][t] = float(p @ mu)
                cols["hmm_sig"][t] = float(p @ sd)
                cols["hmm_entropy"][t] = float(-np.sum(p * np.log(np.maximum(p, 1e-12))))
        t0 = t1
    return cols
