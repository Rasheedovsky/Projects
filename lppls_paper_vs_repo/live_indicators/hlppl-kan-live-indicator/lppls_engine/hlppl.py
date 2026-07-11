"""HLPPL — Hyped Log-Periodic Power Law model (Cao, Shao, Yan, Geman,
arXiv:2510.10878), implemented from the paper and aligned with the reference
pipeline (chirindaopensource). Two trajectory engines are provided:

  fit_lppl_trf     the paper's calibration: direct 7-parameter bounded fit
                   [A, B, C, m, omega, phi, t_c], scipy least_squares 'trf',
                   10 multistart seeds, t_c in (W+5, W+250), omega in (2, 20)
  fit_lppl_kan     the HLPPL-KAN variant [extension]: trajectory produced by
                   the M-LNN-KAN network on a DAE-cleaned window (declining
                   windows are mirrored before cleaning, since the DAE was
                   trained on rising bubble shapes; a negative bubble is the
                   mirror image of a positive one)

Score pipeline (paper Sec. 3):
  residual eps(t) = ln p(t) - ln p_hat(t)  (endpoint of the rolling window)
  AR(1)/OU check on residuals (volatility-confined LPPL, Lin-Ren-Sornette)
  eps_norm(t) = eps(t) / max_{s<=t} |eps(s)|   in [-1, 1]   (causal, Eq. 8)
  BubbleScore = eps_norm + a1*H + a2*S if eps_norm > 0
                eps_norm - a1*H + a2*S otherwise              (Eq. 14)

DATA LIMITATION (disclosed): the paper's Hype index H is the stock's share
of financial news coverage and S is FinBERT news sentiment. No news corpus
is available in this environment, so H is proxied by abnormal trading
volume percentile (attention proxy in the sense of Barber-Odean) and S = 0.
Set alpha1 = alpha2 = 0 for the pure residual-only variant of the paper.

Episodes (paper Sec. 4.2.2): |score| > TAU=0.8 sustained >= DMIN=10 days.
Trading rules (paper Sec. 5/6): long entry if score <= -0.7, short entry if
score >= +0.7, exits on crossing -/+0.3, reversal exit when consecutive
horizon forecasts flip sign.
"""

import numpy as np
from scipy.optimize import least_squares

from .core import minmax_scale, solve_linear

# dae / kan are imported lazily inside the functions that use them, so the
# plain-TRF engine works in packages that ship without those modules

TAU = 0.8
DMIN = 10
THETA1 = 0.7
THETA2 = 0.3
ALPHA1 = 0.2   # hype weight (config-level default; not pinned in the paper)
ALPHA2 = 0.1   # sentiment weight (unused here: no news data, S = 0)
N_SEEDS = 10
KAN_N = 252    # resampled length for the KAN engine


def lppl7(theta, t):
    A, B, C, m, omega, phi, tc = theta
    dt = np.maximum(tc - t, 1e-8)
    return A + B * dt**m + C * dt**m * np.cos(omega * np.log(dt) + phi)


def fit_lppl_trf(logp, seed=0):
    """Paper calibration on one window (t = 1..W). Returns (theta, sse)."""
    W = len(logp)
    t = np.arange(1, W + 1, dtype=float)
    lo = [np.min(logp) - 1.0, -10.0, -5.0, 0.01, 2.0, -2 * np.pi, W + 5.0]
    hi = [np.max(logp) + 1.0, 10.0, 5.0, 0.99, 20.0, 2 * np.pi, W + 250.0]
    rng = np.random.default_rng(seed)
    best, best_sse = None, np.inf
    for _ in range(N_SEEDS):
        x0 = np.array([
            rng.normal(np.mean(logp), 0.1),
            rng.uniform(-1.0, 1.0),
            rng.uniform(-0.5, 0.5),
            rng.uniform(0.1, 0.9),
            rng.uniform(2.0, 20.0),
            rng.uniform(-np.pi, np.pi),
            rng.uniform(W + 10.0, W + 100.0),
        ])
        x0 = np.clip(x0, lo, hi)
        try:
            sol = least_squares(lambda th: lppl7(th, t) - logp, x0, bounds=(lo, hi),
                                method="trf", ftol=1e-8, xtol=1e-8, max_nfev=1000)
        except Exception:
            continue
        sse = 2 * sol.cost
        if sse < best_sse:
            best, best_sse = sol.x, sse
    return best, best_sse


def fit_lppl_kan(logp, seed=0):
    """HLPPL-KAN engine: DAE-clean (mirroring declining windows), fit with
    the M-LNN-KAN, return the trajectory over the window plus (tc, m, w,
    beta, scale, mirrored) for extrapolation."""
    from . import dae as dae_mod
    from . import kan as kan_mod
    W = len(logp)
    x = np.interp(np.linspace(0, 1, KAN_N), np.linspace(0, 1, W), logp)
    x_scaled, (mn, rng_) = minmax_scale(x)
    mirrored = x_scaled[-1] < x_scaled[0]  # declining window
    x_in = 1.0 - x_scaled if mirrored else x_scaled
    x_clean = dae_mod.clean(_dae_params(), x_in).astype(np.float64)
    r = kan_mod.fit_mlnn_kan(x_clean, seed=seed)
    tc, m, w = r["tc"], r["m"], r["w"]
    t_norm = np.linspace(0.0, 1.0, KAN_N)
    beta = solve_linear(t_norm, x_clean, tc, m, w)
    from .core import design_matrix
    yhat = design_matrix(t_norm, tc, m, w) @ beta
    if mirrored:
        yhat = 1.0 - yhat
    traj = np.interp(np.linspace(0, 1, W), t_norm, yhat) * rng_ + mn
    return dict(traj=traj, tc=tc, m=m, w=w, beta=beta, scale=(mn, rng_), mirrored=mirrored)


_DAE_PARAMS = None


def _dae_params():
    global _DAE_PARAMS
    if _DAE_PARAMS is None:
        from pathlib import Path
        from . import dae as dae_mod
        _DAE_PARAMS = dae_mod.load_params(Path(__file__).resolve().parents[1] / "models" / "DAE-cleaner.npz")
    return _DAE_PARAMS


def hype_proxy(volume, span=60):
    """Attention proxy: abnormal volume v_t / trailing mean, squashed to
    (0, 1) with 0.5 at average attention. Stated proxy for the paper's
    news-share Hype index (no news corpus available)."""
    v = np.asarray(volume, dtype=float)
    trail = np.array([v[max(0, i - span + 1) : i + 1].mean() for i in range(len(v))])
    rel = v / np.maximum(trail, 1e-12)
    return rel / (1.0 + rel)


def ou_alpha(resid):
    """AR(1) mean-reversion speed of residuals (volatility-confined check):
    delta eps = -alpha*eps + u. alpha > 0 => mean reverting (model consistent)."""
    e = np.asarray(resid)
    if len(e) < 10 or np.std(e[:-1]) < 1e-12:
        return np.nan
    b = np.polyfit(e[:-1], np.diff(e), 1)[0]
    return -b


def bubble_score(eps_norm, hype, sent=0.0, alpha1=ALPHA1, alpha2=ALPHA2):
    """Eq. 14: hype amplifies the prevailing regime, sentiment corrects."""
    sign = np.where(eps_norm > 0, 1.0, -1.0)
    return eps_norm + sign * alpha1 * hype + alpha2 * sent


def causal_normalize(eps):
    """Eq. 8: eps(t) / running max_{s<=t} |eps(s)|."""
    eps = np.asarray(eps, dtype=float)
    runmax = np.maximum.accumulate(np.abs(eps))
    return eps / np.maximum(runmax, 1e-12)


def label_episodes(score, dates, tau=TAU, dmin=DMIN):
    """Paper Sec. 4.2.2: |score| > tau sustained >= dmin days -> episode."""
    sig = np.where(score > tau, 1, np.where(score < -tau, -1, 0))
    episodes, i = [], 0
    while i < len(sig):
        if sig[i] != 0:
            j = i
            while j + 1 < len(sig) and sig[j + 1] == sig[i]:
                j += 1
            if j - i + 1 >= dmin:
                episodes.append(dict(
                    start=dates[i], end=dates[j],
                    type="positive" if sig[i] > 0 else "negative",
                    duration=j - i + 1,
                    intensity=float(np.max(np.abs(score[i : j + 1]))),
                ))
            i = j + 1
        else:
            i += 1
    return episodes


def trading_decision(forecasts, position=0, theta1=THETA1, theta2=THETA2):
    """Paper Sec. 6.1 rules on multi-horizon score forecasts B_hat[h], h=1..5.
    Returns (action, reason). Reversal exit: consecutive horizons flip sign."""
    f = np.asarray(forecasts, dtype=float)
    if position != 0 and np.any(f[:-1] * f[1:] < 0):
        return "EXIT", "prediction reversal: consecutive horizon forecasts flip sign"
    b = f[0]
    if position == 1 and b >= -theta2:
        return "EXIT LONG", f"forecast {b:+.2f} crossed exit threshold -{theta2}"
    if position == -1 and b <= theta2:
        return "EXIT SHORT", f"forecast {b:+.2f} crossed exit threshold +{theta2}"
    if position == 0:
        if np.min(f) <= -theta1:
            h = int(np.argmin(f)) + 1
            return "ENTER LONG", f"forecast horizon {h} reaches {np.min(f):+.2f} <= -{theta1} (oversold / negative bubble)"
        if np.max(f) >= theta1:
            h = int(np.argmax(f)) + 1
            return "ENTER SHORT", f"forecast horizon {h} reaches {np.max(f):+.2f} >= +{theta1} (overpriced / positive bubble)"
        return "STAY FLAT", f"forecasts within (-{theta1}, +{theta1}) band"
    return "HOLD", "no exit condition met"
