"""
Overnight gaps as triggers of intraday self-excitation.

The idea (user's hypothesis)
----------------------------
Overnight information arrives while the market is closed and is released as
one price *gap* at the open.  If the earthquake analogy holds, a large gap
should act like a distant mainshock: it does not belong to the intraday
process itself, but it should *raise the intraday intensity* at the open,
with the effect fading over the trading day.  So we separate the two return
types and let overnight returns enter the intraday model as an EXOGENOUS
marked trigger stream:

  lambda_intra(t) = mu                                    (background)
    + K_s Sum_{intraday events t_i < t}       phi(t - t_i ; c_s)   (self)
    + K_o Sum_{overnight gaps g_d, open_d<=t} e^{alpha_o (|g_d| - G0)}
                                              phi(t - open_d ; c_o) (trigger)

with normalised exponential kernels phi (so K_s and K_o are expected direct
offspring counts), G0 the gap threshold, and alpha_o the gap-magnitude
leverage: a bigger overnight surprise triggers more intraday aftershocks.
The gap stream is *conditioning information*, not a modelled process, so
the intraday likelihood below is a valid conditional likelihood (this is
the same construction as cross-market spillovers in Gresnigt's thesis,
chapter 3, with the "other market" being the overnight session).

Data split
----------
From hourly bars: intraday returns are ln(C_k / C_{k-1}) within a day plus
the first bar's ln(C_1 / O_1); the overnight return of day d is
g_d = ln(O_{d,1} / C_{d-1,last}).  Time runs on the intraday bar clock
(1 unit = 1 intraday return); the gap of day d is placed at the open,
half a bar before the day's first intraday return.

Estimation & testing
--------------------
* MLE with multistart (the paper's method), parameters
  eta = (mu, K_s, c_s, K_o, alpha_o, c_o).
* LR test of "no overnight triggering" (K_o = 0, which nests the plain
  self-only ETAS of etas.py): 3 restricted parameters.
* KAN-PIN (kan_pin.py machinery) with the physics extended by the trigger
  term - same four-phase curriculum, 6-parameter eta.
* Walk-forward validation starting at 30 trading days (expanding window,
  30-day steps), both estimators refit each step, running Nyblom test, and
  out-of-sample predictive log-scores against two benchmarks:
  the self-only ETAS and a homogeneous Poisson.
"""

from __future__ import annotations

from dataclasses import dataclass

import autograd.numpy as anp
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import chi2

from etas import ETASModel, EventData, _Phi, _phi
from kan_pin import KANPINHawkes
from stability import NYBLOM_CRIT_5PCT

# Hansen (1992) 5% joint critical values continued past p=5
NYBLOM_CRIT_5PCT = dict(NYBLOM_CRIT_5PCT)
NYBLOM_CRIT_5PCT.update({6: 1.68, 7: 1.90})


# ----------------------------------------------------------------------
# 1. Separating overnight gaps from intraday returns
# ----------------------------------------------------------------------

@dataclass
class GapStream:
    """The exogenous overnight trigger stream on the intraday bar clock.

    times : gap arrival times (half a bar before each day's first return)
    gaps  : signed overnight log returns
    mags  : |gap|
    G0    : trigger threshold on |gap| (gaps below it are not events)
    dates : the day each gap opens
    """
    times: np.ndarray
    gaps: np.ndarray
    mags: np.ndarray
    G0: float
    dates: pd.DatetimeIndex

    @property
    def n(self):
        return len(self.times)


def split_overnight_intraday(df: pd.DataFrame):
    """Split hourly OHLC bars into intraday returns and overnight gaps.

    Returns (r_intra, gaps_all) where r_intra is a pd.Series of intraday
    log returns (first bar uses open->close so no overnight leaks in) and
    gaps_all is a DataFrame [date, time, gap] with `time` on the intraday
    return clock (1-based, gap sits 0.5 before the day's first return).
    """
    days = df.groupby(df.index.date, sort=True)
    r_vals, r_idx = [], []
    gap_rows = []
    prev_close = None
    for day, bars in days:
        o, c = bars["Open"].values, bars["Close"].values
        t0 = len(r_vals) + 1                      # time of first return today
        if prev_close is not None:
            gap_rows.append({"date": pd.Timestamp(day),
                             "time": t0 - 0.5,
                             "gap": float(np.log(o[0] / prev_close))})
        r_vals.append(float(np.log(c[0] / o[0])))          # first bar: open->close
        r_idx.append(bars.index[0])
        for k in range(1, len(bars)):
            r_vals.append(float(np.log(c[k] / c[k - 1])))
        r_idx.extend(bars.index[1:])
        prev_close = c[-1]
    r_intra = pd.Series(r_vals, index=pd.DatetimeIndex(r_idx), name="r_intra")
    return r_intra, pd.DataFrame(gap_rows)


def extract_gap_stream(gaps_all: pd.DataFrame, quantile: float = 0.75,
                       mode: str = "abs") -> GapStream:
    """Overnight trigger events: gaps whose relevant magnitude exceeds the
    `quantile` of its distribution.

    mode='abs' : any large gap (up or down) is a trigger  (volatility view)
    mode='neg' : only large down-gaps                      (crash view)
    mode='pos' : only large up-gaps
    """
    g = gaps_all["gap"].values
    if mode == "abs":
        x = np.abs(g)
    elif mode == "neg":
        x = np.where(g < 0, -g, 0.0)
    else:
        x = np.where(g > 0, g, 0.0)
    G0 = float(np.quantile(x[x > 0], quantile))
    keep = x >= G0
    return GapStream(times=gaps_all["time"].values[keep].astype(float),
                     gaps=g[keep], mags=x[keep], G0=G0,
                     dates=pd.DatetimeIndex(gaps_all["date"].values[keep]))


# ----------------------------------------------------------------------
# 2. The overnight-trigger ETAS model (the paper's method: MLE)
# ----------------------------------------------------------------------

class OvernightETAS:
    """Intraday ETAS with an exogenous marked overnight trigger stream.

    Parameters: mu, K_s, c_s (intraday self-excitation) and
    K_o, alpha_o, c_o (overnight triggering).  Exponential kernels.
    """

    PARAM_NAMES = ("mu", "K_s", "c_s", "K_o", "alpha_o", "c_o")

    def __init__(self, events: EventData, src: GapStream):
        self.events = events        # intraday extreme events (target)
        self.src = src              # overnight gap triggers (exogenous)
        self.theta: dict[str, float] = {}
        self.loglik = np.nan

    # ----- intensity & compensator (history = both streams) -----

    def _terms(self, t, theta):
        mu, Ks, cs, Ko, ao, co = (theta[k] for k in self.PARAM_NAMES)
        t = np.atleast_1d(np.asarray(t, dtype=float))
        dts = t[:, None] - self.events.times[None, :]
        self_term = Ks * np.where(dts > 0, _phi(np.where(dts > 0, dts, 1.0),
                                                "exp", cs, 1.0), 0.0).sum(axis=1)
        dto = t[:, None] - self.src.times[None, :]
        boost = np.exp(ao * (self.src.mags - self.src.G0))[None, :]
        trig_term = Ko * (np.where(dto > 0, _phi(np.where(dto > 0, dto, 1.0),
                                                 "exp", co, 1.0), 0.0)
                          * boost).sum(axis=1)
        return mu, self_term, trig_term

    def intensity(self, t, theta=None):
        mu, s, o = self._terms(t, theta or self.theta)
        lam = np.atleast_1d(mu + s + o)
        return lam if lam.size != 1 else float(lam[0])

    def compensator(self, t, theta=None):
        th = theta or self.theta
        mu, Ks, cs, Ko, ao, co = (th[k] for k in self.PARAM_NAMES)
        t = np.atleast_1d(np.asarray(t, dtype=float))
        ds = t[:, None] - self.events.times[None, :]
        Ls = Ks * np.where(ds > 0, _Phi(ds, "exp", cs, 1.0), 0.0).sum(axis=1)
        do = t[:, None] - self.src.times[None, :]
        boost = np.exp(ao * (self.src.mags - self.src.G0))[None, :]
        Lo = Ko * (np.where(do > 0, _Phi(do, "exp", co, 1.0), 0.0) * boost).sum(axis=1)
        Lam = mu * t + Ls + Lo
        return Lam if Lam.size > 1 else float(Lam[0])

    # ----- likelihood -----

    def _nll(self, x):
        theta = dict(zip(self.PARAM_NAMES, x))
        lam = np.atleast_1d(self.intensity(self.events.times, theta))
        if np.any(lam <= 0) or not np.all(np.isfinite(lam)):
            return 1e10
        return -(np.log(lam).sum() - self.compensator(self.events.T, theta))

    def fit(self, n_starts: int = 10, seed: int = 1, x0_extra=None):
        ev, src = self.events, self.src
        rate = ev.n / ev.T
        rng = np.random.default_rng(seed)
        mag_sd = max(src.mags.std(), 1e-4)
        bounds = [(1e-8, 5 * rate), (0.0, 0.995), (1e-2, ev.T / 10),
                  (0.0, 3.0), (0.0, 2.0 / mag_sd), (1e-2, ev.T / 10)]
        starts = []
        if x0_extra is not None:               # warm start (daily refits)
            starts.append([float(np.clip(v, lo, hi)) for v, (lo, hi)
                           in zip(x0_extra, bounds)])
        for _ in range(n_starts):
            starts.append([rate * rng.uniform(0.3, 0.9), rng.uniform(0.05, 0.5),
                           np.exp(rng.uniform(0, np.log(ev.T / 20))),
                           rng.uniform(0.05, 0.8), rng.uniform(0.0, 30.0),
                           np.exp(rng.uniform(0, np.log(30.0)))])
        best = None
        for x0 in starts:
            res = minimize(self._nll, x0, method="L-BFGS-B", bounds=bounds)
            if best is None or res.fun < best.fun:
                best = res
        self.theta = dict(zip(self.PARAM_NAMES, best.x))
        self.loglik = -best.fun
        return self

    @property
    def aic(self):
        return 2 * len(self.PARAM_NAMES) - 2 * self.loglik

    def lr_test_no_overnight(self, seed: int = 1) -> dict:
        """LR test of K_o = 0 (which nests the plain self-only ETAS).
        3 restricted parameters (K_o, alpha_o, c_o); chi2(3) p-value with
        the usual boundary caveat (conservative)."""
        restricted = ETASModel(kernel="exp", use_alpha=False).fit(
            self.events, seed=seed)
        lr = 2 * (self.loglik - restricted.loglik)
        return {"LR": float(lr), "pvalue": float(chi2.sf(max(lr, 0), df=3)),
                "logL_full": float(self.loglik),
                "logL_selfonly": float(restricted.loglik),
                "restricted": restricted}

    # ----- interpretation -----

    def gap_offspring(self, gap_mag: float) -> float:
        """Expected intraday extreme events directly triggered by an
        overnight gap of magnitude |g| = gap_mag."""
        th = self.theta
        return th["K_o"] * np.exp(th["alpha_o"] * (gap_mag - self.src.G0))

    def explain(self) -> str:
        th = self.theta
        ev, src = self.events, self.src
        med, big = np.median(src.mags), np.quantile(src.mags, 0.9)
        lines = [
            f"Overnight-trigger ETAS for intraday {ev.tail.upper()} events:",
            f"  {ev.n} intraday events (|r| >= {ev.M0:.4f}), "
            f"{src.n} overnight gap triggers (|gap| >= {src.G0:.4f}).",
            f"  Background mu = {th['mu']:.4f}/h;  intraday self-excitation "
            f"K_s = {th['K_s']:.2f} (half-life {th['c_s']*np.log(2):.0f}h).",
            f"  Overnight trigger: a threshold gap ({src.G0:.1%}) sparks "
            f"K_o = {th['K_o']:.2f} intraday events;",
            f"  a median trigger gap ({med:.1%}) sparks "
            f"{self.gap_offspring(med):.2f}, a top-decile gap ({big:.1%}) "
            f"{self.gap_offspring(big):.2f}",
            f"  (gap leverage alpha_o = {th['alpha_o']:.0f}); the effect "
            f"fades with half-life {th['c_o']*np.log(2):.1f} trading hours.",
            f"  ln L = {self.loglik:.1f}, AIC = {self.aic:.1f}.",
        ]
        return "\n".join(lines)

    # ----- per-event scores for the Nyblom test -----

    def _event_contribs(self, theta) -> np.ndarray:
        lam = np.atleast_1d(self.intensity(self.events.times, theta))
        Lam = np.atleast_1d(self.compensator(self.events.times, theta))
        dLam = np.diff(np.concatenate([[0.0], Lam]))
        li = np.log(np.maximum(lam, 1e-300)) - dLam
        li[-1] -= self.compensator(self.events.T, theta) - Lam[-1]
        return li

    def nyblom(self) -> dict:
        names = list(self.PARAM_NAMES)
        s = np.zeros((self.events.n, len(names)))
        for j, n in enumerate(names):
            v = self.theta[n]
            h = max(1e-6, 1e-4 * abs(v))
            up = dict(self.theta); up[n] = v + h
            dn = dict(self.theta); dn[n] = v - h
            s[:, j] = (self._event_contribs(up) - self._event_contribs(dn)) / (2 * h)
        s = s - s.mean(axis=0, keepdims=True)
        N, p = s.shape
        V = s.T @ s + 1e-12 * np.eye(p)
        S = np.cumsum(s, axis=0)
        indiv = {n: float(np.sum(S[:, j] ** 2) / (N * max(V[j, j], 1e-12)))
                 for j, n in enumerate(names)}
        joint = float(np.trace(np.linalg.solve(
            V + 1e-10 * np.trace(V) * np.eye(p), S.T @ S)) / N)
        return {"individual": indiv, "joint": joint,
                "crit_individual_5pct": NYBLOM_CRIT_5PCT[1],
                "crit_joint_5pct": NYBLOM_CRIT_5PCT.get(p, 1.9),
                "unstable_joint": joint > NYBLOM_CRIT_5PCT.get(p, 1.9)}


# ----------------------------------------------------------------------
# 3. KAN-PIN for the overnight model
# ----------------------------------------------------------------------

class KANPINOvernight(KANPINHawkes):
    """KAN-PIN estimator with the physics extended by the overnight
    trigger term.  Same architecture and four-phase curriculum as
    KANPINHawkes; eta = (mu, K_s, c_s, K_o, alpha_o, c_o)."""

    PARAM_NAMES = ("mu", "K_s", "c_s", "K_o", "alpha_o", "c_o")

    def __init__(self, events: EventData, src: GapStream, **kw):
        self.src = src
        super().__init__(events, kernel="exp", use_alpha=False, **kw)
        rate = self.rho
        mag_sd = max(src.mags.std(), 1e-4)
        self.lower = np.array([1e-6, 1e-6, 1e-2, 0.0, 0.0, 1e-2])
        self.upper = np.array([5 * rate, 0.995, self.T / 10,
                               3.0, 2.0 / mag_sd, self.T / 10])
        # refine collocation after the overnight triggers too - the
        # post-open decay is what identifies (K_o, c_o)
        near = (src.times[:, None] + np.array([0.0, 1.0, 3.0, 7.0])).ravel()
        self.s_nodes = np.unique(np.clip(
            np.concatenate([self.s_nodes, near]), 0.0, self.T))
        left = np.searchsorted(events.times, self.s_nodes, side="left")
        right = np.searchsorted(events.times, self.s_nodes, side="right")
        self.t_data = self.s_nodes
        self.y_data = 0.5 * (left + right) / self.N
        self.params["eta"] = np.array([0.7 * rate, 0.3, self.T / 50,
                                       0.3, 10.0, 4.0])
        self.mult = {"lam": np.zeros(2), "chi": np.zeros(6)}
        self.history = {k: [] for k in ("L_data", "L_de", "L_ic",
                                        *self.PARAM_NAMES)}

    def _integrated_lambda(self, eta, a, b):
        mu, Ks, cs, Ko, ao, co = (eta[i] for i in range(6))
        da = a[:, None] - self.ev.times[None, :]
        db = b[:, None] - self.ev.times[None, :]
        def PHI(u, c):
            up = anp.where(u > 0, u, 0.0)
            return 1.0 - anp.exp(-up / c)
        out = mu * (b - a) + Ks * anp.sum(PHI(db, cs) - PHI(da, cs), axis=1)
        oa = a[:, None] - self.src.times[None, :]
        ob = b[:, None] - self.src.times[None, :]
        boost = anp.exp(ao * (self.src.mags - self.src.G0))[None, :]
        out = out + Ko * anp.sum(boost * (PHI(ob, co) - PHI(oa, co)), axis=1)
        return out

    def _calibrate_eta(self, params, adam_steps: int = 400,
                       eta_start=None) -> np.ndarray:
        layers = params["layers"]
        s = self.s_nodes
        Gs = np.asarray(self._G(layers, s))
        incr_net = (Gs[1:] - Gs[:-1]) * self.N
        ds = s[1:] - s[:-1]
        scale = self.rho * np.maximum(ds, self.T / (4.0 * len(ds)))

        def L_de_of(eta):
            incr_phys = self._integrated_lambda(eta, s[:-1], s[1:])
            return anp.mean(((incr_net - incr_phys) / scale) ** 2)

        if eta_start is not None:
            best = np.asarray(eta_start, dtype=float)
            best_val = float(L_de_of(best))
        else:
            best, best_val = None, np.inf
        for Ks in (() if eta_start is not None else (0.05, 0.3, 0.6)):
            for cs in np.geomspace(1.0, self.T / 12, 5):
                for Ko in (0.05, 0.3, 0.6):
                    for co in (2.0, 7.0, 21.0):
                        eta = np.array([max(self.rho * (1 - Ks) * 0.8,
                                            self.lower[0]),
                                        Ks, cs, Ko, 10.0, co])
                        val = float(L_de_of(eta))
                        if val < best_val:
                            best, best_val = eta, val
        from autograd import grad as agrad
        g_eta = agrad(L_de_of)
        eta = best.copy()
        m = np.zeros_like(eta); v = np.zeros_like(eta)
        for k in range(1, adam_steps + 1):
            g = np.asarray(g_eta(eta))
            m = 0.9 * m + 0.1 * g
            v = 0.999 * v + 0.001 * g * g
            eta = eta - 2e-2 * (m / (1 - 0.9 ** k)) / (np.sqrt(v / (1 - 0.999 ** k)) + 1e-8)
            eta = np.clip(eta, self.lower, self.upper)
        return eta

    @property
    def eta_hat(self) -> dict:
        eta = np.clip(np.asarray(self.params["eta"]), self.lower, self.upper)
        return dict(zip(self.PARAM_NAMES, (float(x) for x in eta)))


# ----------------------------------------------------------------------
# 4. Walk-forward validation (30-day start), both estimators
# ----------------------------------------------------------------------

def _truncate(events: EventData, src: GapStream, tau: float):
    ke = events.times <= tau
    ks = src.times <= tau
    ev_w = EventData(times=events.times[ke], mags=events.mags[ke],
                     M0=events.M0, T=float(tau), tail=events.tail,
                     timestamps=events.timestamps[ke], index=events.index)
    src_w = GapStream(times=src.times[ks], gaps=src.gaps[ks],
                      mags=src.mags[ks], G0=src.G0, dates=src.dates[ks])
    return ev_w, src_w


def _segment_score(theta: dict, events: EventData, src: GapStream,
                   a: float, b: float) -> float:
    """OOS predictive log-score of overnight-model parameters on (a, b]
    (full history, frozen parameters)."""
    m = OvernightETAS(events, src)
    m.theta = theta
    inseg = (events.times > a) & (events.times <= b)
    lam = np.atleast_1d(m.intensity(events.times[inseg]))
    return float(np.sum(np.log(np.maximum(lam, 1e-300)))
                 - (m.compensator(b) - m.compensator(a)))


def walk_forward_overnight(events: EventData, src: GapStream,
                           day_starts: np.ndarray, start_days: int = 30,
                           step_days: int = 30, kanpin_epochs: int = 500,
                           seed: int = 1, verbose: bool = True) -> pd.DataFrame:
    """Expanding-window walk-forward for the overnight-trigger model.

    `day_starts` maps trading days to bar-clock times (time of each day's
    first intraday return); refits happen every `step_days` trading days,
    starting after `start_days` (the requested 30-day burn-in).  At each
    refit: overnight-model MLE, self-only MLE (benchmark), KAN-PIN
    overnight model, running Nyblom on the overnight MLE, then OOS
    predictive log-scores on the next segment (Poisson as the floor).
    """
    n_days = len(day_starts)
    refit_days = list(range(start_days, n_days, step_days))
    taus = [float(day_starts[min(d, n_days - 1)]) for d in refit_days]
    taus.append(float(events.T))
    rows = []
    for k in range(len(taus) - 1):
        tau, tau_next = taus[k], taus[k + 1]
        ev_w, src_w = _truncate(events, src, tau)
        if ev_w.n < 5:
            continue
        full = OvernightETAS(ev_w, src_w).fit(seed=seed)
        lr = full.lr_test_no_overnight(seed=seed)
        selfonly = lr["restricted"]
        ny = full.nyblom()
        kp = KANPINOvernight(ev_w, src_w, seed=seed)
        kp.fit(epochs=kanpin_epochs, verbose=False)
        ek = kp.eta_hat
        # ---- OOS scores on (tau, tau_next] ----
        a, b = tau, tau_next
        inseg = (events.times > a) & (events.times <= b)
        n_seg = int(inseg.sum())
        s_full = _segment_score(full.theta, events, src, a, b)
        s_kp = _segment_score(ek, events, src, a, b)
        s_self = _segment_score(
            {"mu": selfonly.mu, "K_s": selfonly.K0, "c_s": selfonly.c,
             "K_o": 0.0, "alpha_o": 0.0, "c_o": 1.0}, events, src, a, b)
        rate = ev_w.n / tau
        s_pois = n_seg * np.log(rate) - rate * (b - a)
        rows.append({
            "tau": tau, "date": events.time_to_stamp(tau),
            "n_events": ev_w.n, "n_gaps": src_w.n,
            **{f"mle_{p}": full.theta[p] for p in full.PARAM_NAMES},
            **{f"kp_{p}": ek[p] for p in full.PARAM_NAMES},
            "LR_overnight": lr["LR"], "LR_pvalue": lr["pvalue"],
            "nyblom_joint": ny["joint"],
            "nyblom_crit": ny["crit_joint_5pct"],
            "oos_events": n_seg,
            "oos_full": s_full, "oos_selfonly": s_self,
            "oos_kanpin": s_kp, "oos_poisson": s_pois,
        })
        if verbose:
            print(f"  day {refit_days[k]:3d} ({rows[-1]['date'].date()}, "
                  f"{ev_w.n:3d} ev/{src_w.n:2d} gaps) "
                  f"K_s={full.theta['K_s']:.2f} K_o={full.theta['K_o']:.2f} "
                  f"a_o={full.theta['alpha_o']:4.0f} c_o={full.theta['c_o']:5.1f} "
                  f"LR={lr['LR']:5.1f}(p={lr['pvalue']:.3f}) "
                  f"Ny={ny['joint']:.2f} | OOS: full {s_full:6.1f} "
                  f"self {s_self:6.1f} KP {s_kp:6.1f} Pois {s_pois:6.1f}")
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------
# 5. DAILY walk-forward: refit every trading day (warm-started)
# ----------------------------------------------------------------------

def walk_forward_daily(falls: EventData, runs: EventData, src_all: GapStream,
                       gaps_all: pd.DataFrame, start_days: int = 30,
                       kanpin_epochs: int = 60, kanpin_cal: int = 120,
                       seed: int = 1, verbose_every: int = 40) -> pd.DataFrame:
    """Walk-forward with DAILY re-estimation.

    For each trading day d (after a `start_days` burn-in):
      * refit, on data strictly before day d's open (warm-started from the
        previous day): the overnight-trigger MLE for falls AND for runs,
        the self-only ETAS benchmark, and the KAN-PIN overnight model (falls);
      * at day d's open - when the overnight gap g_d IS known - compute each
        model's probability of >= 1 extreme fall / run during day d, and
        score all parameter sets on day d with the predictive log-score
        (Poisson benchmark from the training window's event rate).

    Returns one row per evaluated day.
    """
    day_starts = np.concatenate([[1.0], gaps_all["time"].values + 0.5])
    n_days = len(day_starts)
    day_ends = np.concatenate([day_starts[1:] - 1.0, [falls.T]])
    warm_f = warm_r = None
    warm_kp_eta = None
    rows = []
    rng = np.random.default_rng(seed)
    for d in range(start_days, n_days):
        t_open = day_starts[d] - 0.5          # the gap fires here
        tau = t_open - 0.1                    # fit strictly before today
        b = day_ends[d]
        ev_f, src_w = _truncate(falls, src_all, tau)
        ev_r, _ = _truncate(runs, src_all, tau)
        if ev_f.n < 5 or ev_r.n < 5:
            continue
        # ---- daily refits (warm-started) ----
        mf = OvernightETAS(ev_f, src_w).fit(n_starts=3, seed=seed, x0_extra=warm_f)
        mr = OvernightETAS(ev_r, src_w).fit(n_starts=3, seed=seed, x0_extra=warm_r)
        warm_f = [mf.theta[k] for k in mf.PARAM_NAMES]
        warm_r = [mr.theta[k] for k in mr.PARAM_NAMES]
        so = ETASModel(kernel="exp", use_alpha=False).fit(ev_f, seed=seed,
                                                          n_starts=3)
        kp = KANPINOvernight(ev_f, src_w, seed=seed)
        kp.fit(epochs=kanpin_epochs, calibrate_steps=kanpin_cal,
               warm_eta=warm_kp_eta, verbose=False)
        ek = kp.eta_hat
        warm_kp_eta = np.array([ek[k] for k in OvernightETAS.PARAM_NAMES])
        # ---- day-d forecasts at the open (gap known) ----
        def day_prob(theta, events):
            m = OvernightETAS(events, src_all)  # full streams as history
            m.theta = theta
            dLam = m.compensator(b) - m.compensator(t_open)
            return 1.0 - np.exp(-dLam), dLam
        p_fall, e_fall = day_prob(mf.theta, falls)
        p_run, e_run = day_prob(mr.theta, runs)
        # ---- day-d OOS log-scores (falls target) ----
        s_full = _segment_score(mf.theta, falls, src_all, t_open, b)
        s_kp = _segment_score(ek, falls, src_all, t_open, b)
        s_self = _segment_score({"mu": so.mu, "K_s": so.K0, "c_s": so.c,
                                 "K_o": 0.0, "alpha_o": 0.0, "c_o": 1.0},
                                falls, src_all, t_open, b)
        rate = ev_f.n / tau
        n_seg = int(np.sum((falls.times > t_open) & (falls.times <= b)))
        s_pois = (n_seg * np.log(rate) if n_seg else 0.0) - rate * (b - t_open)
        gap_today = gaps_all.iloc[d - 1]["gap"] if d >= 1 else 0.0
        rows.append({
            "day": d, "date": falls.time_to_stamp(day_starts[d]),
            "gap": float(gap_today), "big_gap": abs(gap_today) >= src_all.G0,
            "p_fall": p_fall, "p_run": p_run,
            "e_fall": e_fall, "e_run": e_run,
            "realized_fall": bool(n_seg),
            "realized_run": bool(np.sum((runs.times > t_open) & (runs.times <= b))),
            "mle_K_o": mf.theta["K_o"], "mle_K_s": mf.theta["K_s"],
            "mle_c_o": mf.theta["c_o"], "kp_K_o": ek["K_o"], "kp_K_s": ek["K_s"],
            "run_K_o": mr.theta["K_o"],
            "oos_full": s_full, "oos_selfonly": s_self,
            "oos_kanpin": s_kp, "oos_poisson": s_pois,
        })
        if verbose_every and (len(rows) % verbose_every == 0):
            r = rows[-1]
            print(f"  day {d:3d} ({r['date'].date()}) gap={r['gap']:+.3f} "
                  f"P(fall)={p_fall:.2f} P(run)={p_run:.2f} "
                  f"K_o={r['mle_K_o']:.2f} | cum OOS-Pois: "
                  f"full {sum(x['oos_full']-x['oos_poisson'] for x in rows):+.1f} "
                  f"self {sum(x['oos_selfonly']-x['oos_poisson'] for x in rows):+.1f} "
                  f"KP {sum(x['oos_kanpin']-x['oos_poisson'] for x in rows):+.1f}")
    return pd.DataFrame(rows)
