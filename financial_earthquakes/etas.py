"""
Financial earthquakes: the ETAS / Hawkes indicator of Gresnigt, Kole & Franses.

Reference
---------
F. Gresnigt, E. Kole, P.H. Franses (2015), "Interpreting financial market
crashes as earthquakes: A new Early Warning System for medium term crashes",
Journal of Banking & Finance 56, 123-139.  (Chapter 2 of Gresnigt's 2020
Erasmus University PhD thesis "Identifying and Predicting Financial
Earthquakes Using Hawkes Processes", https://repub.eur.nl/pub/124777/.)

The idea in one paragraph
-------------------------
Around a crash, financial markets behave like the earth's crust around an
earthquake: a large shock ("mainshock") is followed by a cluster of further
shocks ("aftershocks") whose frequency decays slowly with time (Omori's law),
and big shocks trigger more and bigger aftershocks than small ones.  The paper
therefore models the *arrival times* of extreme returns as a marked
self-exciting (Hawkes) point process of the ETAS type, borrowed from
seismology (Ogata 1988).  The fitted model yields a *conditional intensity*
lambda(t | history): the instantaneous expected number of extreme-return
events per unit of time given everything observed so far.  Because the
intensity is high right after events and decays deterministically between
them, it can be turned into an Early Warning System (EWS): the probability
that at least one more extreme return arrives within the next d periods.
The same machinery answers "how long will this fall (or run) last and when
will it end": simulate the fitted process forward and record when the
aftershock cascade dies out.

The model
---------
Events are the times t_1 < t_2 < ... < t_N in [0, T] at which the return
exceeds an extreme threshold (left tail -> "fall events" / crashes,
right tail -> "run events" / booms), with marks (magnitudes)
m_i = |r_i| >= M0, where M0 is the threshold (e.g. the 95% quantile of |r|
in the relevant tail).  The conditional intensity is

    lambda(t | H_t) = mu + K0 * sum_{i: t_i < t} e^{alpha (m_i - M0)} * phi(t - t_i)

with
    mu     >= 0 : background ("tectonic") rate of spontaneous events,
    K0     >= 0 : fertility - expected number of direct aftershocks
                  triggered by an event of threshold magnitude,
    alpha  >= 0 : how much bigger events trigger more aftershocks
                  (alpha = 0: aftershock productivity ignores magnitude),
    phi(s)      : normalised time-decay kernel, integral_0^inf phi = 1:
      * power law / Omori (as in seismology, heavy tailed):
            phi(s) = (omega / c) * (1 + s/c)^{-(1+omega)}
      * exponential (short memory):
            phi(s) = (1/c) * e^{-s/c}

Because phi integrates to one, K0 * e^{alpha(m - M0)} is exactly the expected
number of *direct* children of an event of magnitude m, and the *branching
ratio*

    n = K0 * E[e^{alpha (m - M0)}]

is the average number of direct aftershocks per event.  The process is
stationary only if n < 1; as n -> 1 the market sits close to criticality and
clusters (falls/runs) become long.  1 / (1 - n) is the expected total cluster
size spawned by one spontaneous event.

Magnitudes above the threshold are modelled (as in the Gutenberg-Richter law
of seismology) with an exponential density

    f(m) = (1/beta_m) * e^{-(m - M0)/beta_m},   m >= M0.

Estimation (the method in the paper)
------------------------------------
Maximum likelihood.  For a marked point process observed on [0, T] the
log-likelihood of the ground process is (Daley & Vere-Jones):

    ln L(theta) = sum_i ln lambda(t_i | H_{t_i}) - integral_0^T lambda(s) ds

and the integral is analytic thanks to the normalised kernels:

    integral_0^T lambda = mu*T + K0 * sum_i e^{alpha(m_i - M0)} * PHI(T - t_i)

where PHI is the kernel CDF, PHI(u) = 1 - (1 + u/c)^{-omega} (power law)
or 1 - e^{-u/c} (exponential).  The magnitude part
sum_i ln f(m_i) separates and beta_m has the closed-form MLE mean(m_i - M0).
The paper estimates several variants (power-law vs exponential decay, with
and without magnitude influence alpha) and selects by AIC and by residual
diagnostics based on the random time change theorem: if the model is right,
the rescaled inter-event times tau_i = Lambda(t_i) - Lambda(t_{i-1}) (with
Lambda the integrated intensity, the "compensator") are i.i.d. Exp(1).

Early Warning System (the indicator)
------------------------------------
The probability that at least one extreme event occurs in (t, t + d] is

    P = 1 - exp( - integral_t^{t+d} lambda(s | H) ds )                  (*)

Evaluating (*) with the history frozen at t (no future offspring) gives a
fast analytic *lower bound*; simulating the fitted process forward with
Ogata's thinning algorithm gives the exact probability including
aftershocks-of-aftershocks.  A signal is issued when P exceeds a threshold;
signals are scored out of sample by the hit rate H (fraction of event-windows
called), the false-alarm rate F (fraction of quiet windows falsely called),
and the Hanssen-Kuiper skill score KSS = H - F, exactly as in the paper.

Duration of falls and runs
--------------------------
Fit the model separately to left-tail events (falls) and right-tail events
(runs).  An *episode* is a cluster of same-tail events separated by less
than a quiet window w.  Standing at time t inside an episode, simulate many
futures; in each, the episode "ends" at the first instant tau >= t from which
a full window w passes with no event.  The simulated distribution of tau - t
is the forecast of the remaining duration of the fall/run, its quantiles give
an interval forecast, and P(no event in (t, t+w]) is the probability that the
episode is *already over*.

Everything above is implemented below, in this order:
data -> events -> model/likelihood -> fit -> diagnostics -> simulation ->
EWS -> episode-duration forecasts.  `ETASModel.explain()` narrates a fitted
model in plain English.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import kstest

# --------------------------------------------------------------------------
# 1. Data handling
# --------------------------------------------------------------------------

def load_yfinance_csv(path: str) -> pd.DataFrame:
    """Load a yfinance-style CSV (3 header rows: Price/Ticker/Datetime).

    Returns a DataFrame indexed by timestamp with columns
    Close, High, Low, Open, Volume.
    """
    df = pd.read_csv(path, skiprows=[1, 2], index_col=0, parse_dates=True)
    df.index.name = "Datetime"
    return df.sort_index()


def log_returns(close: pd.Series) -> pd.Series:
    """Log returns between consecutive observations (trading clock:
    the overnight/weekend gap is just one tick, as in the paper where
    event times are measured in trading periods)."""
    return np.log(close).diff().dropna()


@dataclass
class EventData:
    """Extreme-return events for one tail, on the trading clock.

    times      : event times in *periods since the start of the return
                 series* (fractional not needed - one bar = one time unit)
    mags       : magnitudes m_i = |r_i| (all >= M0)
    M0         : the extreme threshold on |r|
    T          : length of the observation window in periods
    tail       : 'fall' (left tail, r <= -M0) or 'run' (right tail, r >= M0)
    timestamps : the wall-clock timestamps of the events (for reporting)
    index      : timestamps of every bar (to translate model time -> dates)
    """
    times: np.ndarray
    mags: np.ndarray
    M0: float
    T: float
    tail: str
    timestamps: pd.DatetimeIndex
    index: pd.DatetimeIndex

    @property
    def n(self) -> int:
        return len(self.times)

    def time_to_stamp(self, t: float) -> pd.Timestamp:
        """Translate a model time (in bars) back to a wall-clock timestamp."""
        i = int(np.clip(round(t), 0, len(self.index) - 1))
        return self.index[i]


def extract_events(returns: pd.Series, tail: str = "fall",
                   quantile: float = 0.95) -> EventData:
    """Define 'earthquakes' exactly as in Gresnigt et al.: returns whose
    magnitude exceeds the `quantile` of the relevant tail.

    tail='fall': events are r <= Q_{1-quantile}(r)  (large negative returns)
    tail='run' : events are r >= Q_{quantile}(r)    (large positive returns)

    Event times are bar indices 1..K on the trading clock, so decay happens
    in *market time* and is not distorted by nights and weekends.
    """
    r = returns.values
    if tail == "fall":
        M0 = -np.quantile(r, 1.0 - quantile)   # threshold on |r|
        mask = r <= -M0
    elif tail == "run":
        M0 = np.quantile(r, quantile)
        mask = r >= M0
    else:
        raise ValueError("tail must be 'fall' or 'run'")
    idx = np.flatnonzero(mask)
    return EventData(
        times=idx.astype(float) + 1.0,          # bar k gets time k (1-based)
        mags=np.abs(r[idx]),
        M0=float(M0),
        T=float(len(r)),
        tail=tail,
        timestamps=returns.index[idx],
        index=returns.index,
    )


# --------------------------------------------------------------------------
# 2. Triggering kernels phi(s)  (normalised: integral_0^inf phi = 1)
# --------------------------------------------------------------------------

def _phi(s, kernel, c, omega):
    """Kernel density phi(s), s >= 0."""
    if kernel == "power":
        return (omega / c) * np.power(1.0 + s / c, -(1.0 + omega))
    return (1.0 / c) * np.exp(-s / c)            # exponential


def _Phi(u, kernel, c, omega):
    """Kernel CDF  PHI(u) = integral_0^u phi(s) ds."""
    u = np.maximum(u, 0.0)
    if kernel == "power":
        return 1.0 - np.power(1.0 + u / c, -omega)
    return 1.0 - np.exp(-u / c)


# --------------------------------------------------------------------------
# 3. The ETAS model: intensity, likelihood, fitting, interpretation
# --------------------------------------------------------------------------

@dataclass
class ETASModel:
    """Marked Hawkes (ETAS) model for one tail of the return distribution.

    Parameters (theta)
    ------------------
    mu     : background intensity (spontaneous events per period)
    K0     : fertility at threshold magnitude
    alpha  : magnitude leverage on fertility (0 = magnitudes don't matter)
    c      : kernel time scale, in periods
    omega  : power-law tail exponent (only used by the power kernel)
    beta_m : mean excess magnitude, MLE of the exponential mark density
    kernel : 'power' (Omori, heavy tail) or 'exp' (short memory)
    use_alpha : if False, alpha is fixed at 0 (paper's restricted variants)
    """
    kernel: str = "power"
    use_alpha: bool = True
    mu: float = np.nan
    K0: float = np.nan
    alpha: float = 0.0
    c: float = np.nan
    omega: float = 1.0
    beta_m: float = np.nan
    events: EventData | None = None
    loglik: float = np.nan
    n_params: int = 0
    fit_info: dict = field(default_factory=dict)

    # ---------------- intensity & compensator ----------------

    def _boost(self, mags):
        """e^{alpha (m - M0)}: the magnitude multiplier of fertility."""
        return np.exp(self.alpha * (mags - self.events.M0))

    def intensity(self, t, times=None, mags=None):
        """lambda(t | H_t) for scalar or vector t, using events strictly
        before t.  `times`/`mags` default to the fitted event history."""
        ev_t = self.events.times if times is None else times
        ev_m = self.events.mags if mags is None else mags
        t = np.atleast_1d(np.asarray(t, dtype=float))
        # matrix of elapsed times, masked to the past
        dt = t[:, None] - ev_t[None, :]
        past = dt > 0
        contrib = np.where(
            past,
            _phi(np.where(past, dt, 1.0), self.kernel, self.c, self.omega)
            * np.exp(self.alpha * (ev_m[None, :] - self.events.M0)),
            0.0,
        )
        lam = self.mu + self.K0 * contrib.sum(axis=1)
        return lam if lam.size > 1 else float(lam[0])

    def compensator(self, t):
        """Lambda(t) = integral_0^t lambda(s) ds  (analytic)."""
        ev_t, ev_m = self.events.times, self.events.mags
        t = np.atleast_1d(np.asarray(t, dtype=float))
        dt = t[:, None] - ev_t[None, :]
        F = np.where(dt > 0, _Phi(dt, self.kernel, self.c, self.omega), 0.0)
        Lam = self.mu * t + self.K0 * (F * self._boost(ev_m)[None, :]).sum(axis=1)
        return Lam if Lam.size > 1 else float(Lam[0])

    # ---------------- likelihood ----------------

    @staticmethod
    def _neg_loglik(x, ev: EventData, kernel: str, use_alpha: bool):
        """Negative ground-process log-likelihood
        -[ sum_i ln lambda(t_i) - integral_0^T lambda ].
        Parameter vector x = (mu, K0, alpha, c, omega) with alpha/omega
        dropped when unused."""
        mu, K0 = x[0], x[1]
        alpha = x[2] if use_alpha else 0.0
        c = x[3 if use_alpha else 2]
        omega = x[-1] if kernel == "power" else 1.0
        t, m = ev.times, ev.mags
        boost = np.exp(alpha * (m - ev.M0))
        # lambda at each event time (events strictly earlier contribute)
        dt = t[:, None] - t[None, :]
        past = dt > 0
        phi = np.where(past, _phi(np.where(past, dt, 1.0), kernel, c, omega), 0.0)
        lam = mu + K0 * (phi * boost[None, :]).sum(axis=1)
        if np.any(lam <= 0) or not np.all(np.isfinite(lam)):
            return 1e10
        # analytic integral of the intensity over [0, T]
        integral = mu * ev.T + K0 * (boost * _Phi(ev.T - t, kernel, c, omega)).sum()
        return -(np.log(lam).sum() - integral)

    def fit(self, events: EventData, n_starts: int = 8, seed: int = 0):
        """Maximum likelihood with L-BFGS-B multistart (the paper's method;
        multistart guards against the local minima that plague Hawkes MLE).
        """
        self.events = events
        rng = np.random.default_rng(seed)
        rate = events.n / events.T                      # empirical event rate
        # Stationarity requires the branching ratio n < 1 (else clusters never
        # die out and the EWS/duration questions are ill-posed), so K0 < 1.
        # Bounding c and omega away from the flat-kernel corner (omega -> 0,
        # c -> T) removes a well-known mu/K0 identifiability degeneracy.
        bounds = [(1e-8, 5 * rate + 1e-6), (1e-8, 0.995)]
        if self.use_alpha:
            bounds.append((0.0, 2.0 / max(events.mags.std(), 1e-4)))
        bounds.append((1e-2, events.T / 10))            # c
        if self.kernel == "power":
            bounds.append((0.10, 10.0))                 # omega
        best = None
        for k in range(n_starts):
            x0 = np.array([
                rate * rng.uniform(0.2, 0.9),           # mu: part of the rate
                rng.uniform(0.1, 0.9),                  # K0: subcritical start
                *( [rng.uniform(0.0, 20.0)] if self.use_alpha else [] ),
                np.exp(rng.uniform(np.log(0.5), np.log(events.T / 20))),  # c
                *( [rng.uniform(0.2, 3.0)] if self.kernel == "power" else [] ),
            ])
            res = minimize(self._neg_loglik, x0,
                           args=(events, self.kernel, self.use_alpha),
                           method="L-BFGS-B", bounds=bounds)
            if best is None or res.fun < best.fun:
                best = res
        x = best.x
        self.mu, self.K0 = x[0], x[1]
        self.alpha = x[2] if self.use_alpha else 0.0
        self.c = x[3 if self.use_alpha else 2]
        self.omega = x[-1] if self.kernel == "power" else 1.0
        self.beta_m = float(np.mean(events.mags - events.M0))  # mark MLE
        self.loglik = -best.fun
        self.n_params = len(x)
        self.fit_info = {"converged": bool(best.success), "n_starts": n_starts}
        return self

    @property
    def aic(self) -> float:
        return 2 * self.n_params - 2 * self.loglik

    # ---------------- interpretation ----------------

    def branching_ratio(self) -> float:
        """n = K0 * E[e^{alpha(m-M0)}]: mean number of direct aftershocks
        per event.  Uses the exponential-magnitude expectation
        E = 1/(1 - alpha*beta_m) when finite, else the empirical mean."""
        if self.alpha * self.beta_m < 1.0:
            return self.K0 / (1.0 - self.alpha * self.beta_m)
        return self.K0 * float(np.mean(self._boost(self.events.mags)))

    def kernel_half_life(self) -> float:
        """Time for the triggering kernel to release half of its mass:
        how quickly the aftershock danger fades."""
        if self.kernel == "power":
            return self.c * (2.0 ** (1.0 / self.omega) - 1.0)
        return self.c * np.log(2.0)

    def explain(self) -> str:
        """Narrate the fitted model in plain English - the 'self-explaining'
        part of the indicator."""
        ev = self.events
        n = self.branching_ratio()
        lines = [
            f"ETAS model for {ev.tail.upper()} events "
            f"({self.kernel} kernel{'' if self.use_alpha else ', alpha=0'}):",
            f"  Data: {ev.n} extreme {'negative' if ev.tail=='fall' else 'positive'}"
            f" returns (|r| >= {ev.M0:.4f}) over {ev.T:.0f} periods"
            f" -> unconditional rate {ev.n/ev.T:.4f} events/period.",
            f"  Background rate mu = {self.mu:.4f}: about one *spontaneous*"
            f" {ev.tail} event every {1/self.mu:,.0f} periods.",
            f"  Branching ratio n = {n:.2f}: each event triggers on average"
            f" {n:.2f} direct aftershocks"
            f" ({100*n:.0f}% of all events are aftershocks in equilibrium;"
            f" one spontaneous shock spawns a cluster of ~{1/(1-n):.1f} events"
            f" in total)." if n < 1 else
            f"  Branching ratio n = {n:.2f} >= 1: SUPERCRITICAL - clusters"
            " do not die out on their own within the sample.",
            f"  Aftershock decay: half of the triggering pressure is released"
            f" within {self.kernel_half_life():.1f} periods"
            f" (kernel scale c = {self.c:.2f}"
            + (f", Omori exponent 1+omega = {1+self.omega:.2f})."
               if self.kernel == "power" else ").")
        ]
        if self.use_alpha:
            boost10 = np.exp(self.alpha * self.beta_m)
            lines.append(
                f"  Magnitude leverage alpha = {self.alpha:.1f}: an event one"
                f" average-excess bigger than the threshold triggers"
                f" {boost10:.2f}x more aftershocks.")
        lines.append(
            f"  Mean excess magnitude beta_m = {self.beta_m:.4f}"
            f" (magnitudes ~ M0 + Exp(beta_m)).")
        lines.append(f"  ln L = {self.loglik:.1f}, AIC = {self.aic:.1f}.")
        return "\n".join(lines)

    # ---------------- diagnostics (random time change) ----------------

    def residual_diagnostics(self) -> dict:
        """If the model is correct, tau_i = Lambda(t_i) - Lambda(t_{i-1})
        are i.i.d. Exp(1) (Papangelou / time-rescaling theorem) - the
        specification test emphasised in the thesis.  Returns the KS test
        of u_i = 1 - exp(-tau_i) against U(0,1)."""
        Lam = self.compensator(self.events.times)
        tau = np.diff(np.concatenate([[0.0], Lam]))
        u = 1.0 - np.exp(-tau)
        ks = kstest(u, "uniform")
        return {"rescaled_times": tau, "ks_stat": float(ks.statistic),
                "ks_pvalue": float(ks.pvalue)}

    # ---------------- simulation (Ogata thinning) ----------------

    def simulate(self, t_start: float, horizon: float, rng,
                 hist_times=None, hist_mags=None):
        """Simulate the fitted process on (t_start, t_start + horizon]
        conditional on the history, via Ogata's modified thinning.
        Returns (times, mags) of the simulated events."""
        ht = list(self.events.times[self.events.times <= t_start]
                  if hist_times is None else hist_times)
        hm = list(self.events.mags[: len(ht)] if hist_mags is None else hist_mags)
        new_t, new_m = [], []
        t = t_start
        t_end = t_start + horizon
        while t < t_end:
            lam_bar = self.intensity(t + 1e-9,
                                     np.asarray(ht), np.asarray(hm))
            if lam_bar <= 0:
                break
            w = rng.exponential(1.0 / lam_bar)      # candidate wait
            t_cand = t + w
            if t_cand > t_end:
                break
            lam_cand = self.intensity(t_cand, np.asarray(ht), np.asarray(hm))
            if rng.uniform() * lam_bar <= lam_cand:  # accept (intensity only
                m = self.events.M0 + rng.exponential(self.beta_m)  # falls between events)
                ht.append(t_cand); hm.append(m)
                new_t.append(t_cand); new_m.append(m)
            t = t_cand
        return np.array(new_t), np.array(new_m)

    # ---------------- the Early Warning System ----------------

    def prob_event_within(self, t: float, d: float,
                          method: str = "analytic",
                          n_paths: int = 300, seed: int = 0) -> float:
        """P(at least one extreme event in (t, t+d] | history up to t).

        'analytic' freezes the history at t (no future offspring):
            P = 1 - exp(-[Lambda(t+d) - Lambda(t)])   -- fast lower bound,
        'simulate' runs Ogata thinning and counts paths with events --
        the exact probability, used in the paper's EWS."""
        if method == "analytic":
            # restrict the compensator to events known at time t
            keep = self.events.times <= t
            ev_t, ev_m = self.events.times[keep], self.events.mags[keep]
            def Lam(x):
                F = _Phi(x - ev_t, self.kernel, self.c, self.omega)
                return self.mu * x + self.K0 * (F * np.exp(
                    self.alpha * (ev_m - self.events.M0))).sum()
            return 1.0 - np.exp(-(Lam(t + d) - Lam(t)))
        rng = np.random.default_rng(seed)
        keep = self.events.times <= t
        ht, hm = self.events.times[keep], self.events.mags[keep]
        hits = 0
        for _ in range(n_paths):
            st, _ = self.simulate(t, d, rng, ht.copy(), hm.copy())
            hits += len(st) > 0
        return hits / n_paths

    def ews_backtest(self, t_grid: np.ndarray, d: float,
                     thresholds: np.ndarray) -> pd.DataFrame:
        """Walk-forward EWS evaluation as in the paper.

        At each time in `t_grid` compute P(event within d periods); the
        realised outcome is 1 if an event actually arrived in that window.
        For each probability threshold report hit rate H, false-alarm rate F
        and the Hanssen-Kuiper skill score KSS = H - F  (KSS > 0 means the
        system beats an uninformed coin)."""
        probs = np.array([self.prob_event_within(t, d) for t in t_grid])
        ev = self.events.times
        realised = np.array([np.any((ev > t) & (ev <= t + d)) for t in t_grid])
        rows = []
        for thr in thresholds:
            sig = probs >= thr
            tp = np.sum(sig & realised); fn = np.sum(~sig & realised)
            fp = np.sum(sig & ~realised); tn = np.sum(~sig & ~realised)
            H = tp / max(tp + fn, 1); F = fp / max(fp + tn, 1)
            rows.append({"threshold": thr, "hit_rate": H,
                         "false_alarm": F, "KSS": H - F,
                         "signals": int(sig.sum())})
        out = pd.DataFrame(rows)
        out.attrs["probs"] = probs
        out.attrs["realised"] = realised
        return out

    # ---------------- duration of the current fall / run ----------------

    def episodes(self, quiet_window: float) -> list[tuple[float, float, int]]:
        """Split the observed events into episodes (clusters): consecutive
        events less than `quiet_window` apart belong to the same episode.
        Returns [(start_time, end_time, n_events), ...]."""
        t = self.events.times
        out, s, prev, cnt = [], t[0], t[0], 1
        for x in t[1:]:
            if x - prev < quiet_window:
                cnt += 1
            else:
                out.append((s, prev, cnt)); s, cnt = x, 1
            prev = x
        out.append((s, prev, cnt))
        return out

    def predict_episode_end(self, t: float, quiet_window: float,
                            horizon: float, n_paths: int = 500,
                            seed: int = 0) -> dict:
        """Forecast when the current fall/run episode ends.

        Standing at time t, simulate the fitted process forward n_paths
        times.  In each path the episode ends at the first instant tau >= t
        from which `quiet_window` periods pass with no event (tau = t if the
        very next window is already quiet).  Returns the simulated
        distribution of the *remaining* duration tau - t plus:
          p_over_now      : P(no further event within quiet_window)
          p_more_events   : 1 - p_over_now
          exp_more_events : expected number of further events before the end
        """
        rng = np.random.default_rng(seed)
        keep = self.events.times <= t
        ht, hm = self.events.times[keep], self.events.mags[keep]
        remaining, extra_events = [], []
        for _ in range(n_paths):
            st, _ = self.simulate(t, horizon, rng, ht.copy(), hm.copy())
            # end = first moment from which a quiet_window passes eventless
            tau, n_extra = t, 0
            for s in st:
                if s - tau >= quiet_window:      # gap reached before s
                    break
                tau, n_extra = s, n_extra + 1    # episode continues through s
            remaining.append(tau - t)
            extra_events.append(n_extra)
        remaining = np.array(remaining); extra_events = np.array(extra_events)
        q = lambda p: float(np.quantile(remaining, p))
        return {
            "remaining_samples": remaining,
            "extra_event_samples": extra_events,
            "p_over_now": float(np.mean(remaining == 0.0)),
            "p_more_events": float(np.mean(extra_events > 0)),
            "exp_more_events": float(extra_events.mean()),
            "median": q(0.5), "q10": q(0.10), "q25": q(0.25),
            "q75": q(0.75), "q90": q(0.90), "mean": float(remaining.mean()),
        }


# --------------------------------------------------------------------------
# 4. Model selection helper: the paper's grid of variants
# --------------------------------------------------------------------------

def fit_variants(events: EventData, seed: int = 0) -> pd.DataFrame:
    """Fit the four core specifications of the paper (power/exponential
    kernel x with/without magnitude influence) and rank them by AIC.
    Returns a table; the fitted models sit in .attrs['models']."""
    specs = [("power", True), ("power", False), ("exp", True), ("exp", False)]
    rows, models = [], {}
    for kernel, use_alpha in specs:
        name = f"{kernel}{'' if use_alpha else ' (alpha=0)'}"
        m = ETASModel(kernel=kernel, use_alpha=use_alpha).fit(events, seed=seed)
        diag = m.residual_diagnostics()
        rows.append({
            "model": name, "logL": m.loglik, "AIC": m.aic,
            "mu": m.mu, "K0": m.K0, "alpha": m.alpha, "c": m.c,
            "omega": m.omega if kernel == "power" else np.nan,
            "branching_n": m.branching_ratio(),
            "KS_pvalue": diag["ks_pvalue"],
        })
        models[name] = m
    out = pd.DataFrame(rows).sort_values("AIC").reset_index(drop=True)
    out.attrs["models"] = models
    return out


# --------------------------------------------------------------------------
# 5. Cross-excitation: bivariate Hawkes (falls <-> runs)
# --------------------------------------------------------------------------
#
# The univariate fits above tell only half the story: on this data set,
# positive extremes ("runs") show almost no *self*-excitation yet fail the
# Poisson diagnostic - they cluster because they arrive inside turbulent
# episodes started by *falls* (rebounds).  Chapter 3 of Gresnigt's thesis
# tests exactly this kind of spillover ("cross-excitation") between event
# streams.  The natural model is a mutually exciting (bivariate) Hawkes
# process with exponential kernels:
#
#   lambda_j(t) = mu_j + sum_k K_jk * sum_{i in stream k, t_i<t}
#                                       (1/c_jk) e^{-(t - t_i)/c_jk}
#
# for target streams j in {fall, run} and source streams k in {fall, run}.
# K_jk is the expected number of type-j events directly triggered by one
# type-k event.  A useful property: the log-likelihood decomposes by target
# stream, so each equation can be estimated separately by MLE.

class CrossETAS:
    """Two-type mutually exciting Hawkes model on {fall, run} events."""

    def __init__(self, falls: EventData, runs: EventData):
        assert falls.T == runs.T
        self.streams = {"fall": falls, "run": runs}
        self.T = falls.T
        self.params: dict[str, dict] = {}

    # ----- intensity of target stream j given both histories -----

    def intensity(self, j: str, t, hist: dict[str, np.ndarray] | None = None):
        p = self.params[j]
        t = np.atleast_1d(np.asarray(t, dtype=float))
        lam = np.full(t.shape, p["mu"])
        for k in ("fall", "run"):
            src = self.streams[k].times if hist is None else hist[k]
            dt = t[:, None] - src[None, :]
            past = dt > 0
            lam += p[f"K_{k}"] * np.where(
                past, _phi(np.where(past, dt, 1.0), "exp", p[f"c_{k}"], 1.0),
                0.0).sum(axis=1)
        return lam if lam.size > 1 else float(lam[0])

    @staticmethod
    def _neg_loglik_target(x, tj: np.ndarray, sources: dict, T: float):
        """-ln L for one target equation.  x = (mu, K_fall, c_fall, K_run, c_run)."""
        mu, Kf, cf, Kr, cr = x
        lam = np.full(len(tj), mu)
        integral = mu * T
        for (K, c, src) in ((Kf, cf, sources["fall"]), (Kr, cr, sources["run"])):
            dt = tj[:, None] - src[None, :]
            past = dt > 0
            lam += K * np.where(past, _phi(np.where(past, dt, 1.0), "exp", c, 1.0), 0.0).sum(axis=1)
            integral += K * _Phi(T - src, "exp", c, 1.0).sum()
        if np.any(lam <= 0):
            return 1e10
        return -(np.log(lam).sum() - integral)

    def fit(self, n_starts: int = 8, seed: int = 0):
        rng = np.random.default_rng(seed)
        src = {k: self.streams[k].times for k in ("fall", "run")}
        for j in ("fall", "run"):
            tj = self.streams[j].times
            rate = len(tj) / self.T
            bounds = [(1e-8, 5 * rate), (0.0, 0.995), (1e-2, self.T / 10),
                      (0.0, 0.995), (1e-2, self.T / 10)]
            best = None
            for _ in range(n_starts):
                x0 = [rate * rng.uniform(0.2, 0.9),
                      rng.uniform(0.05, 0.6), np.exp(rng.uniform(0, np.log(self.T / 20))),
                      rng.uniform(0.05, 0.6), np.exp(rng.uniform(0, np.log(self.T / 20)))]
                res = minimize(self._neg_loglik_target, x0,
                               args=(tj, src, self.T),
                               method="L-BFGS-B", bounds=bounds)
                if best is None or res.fun < best.fun:
                    best = res
            mu, Kf, cf, Kr, cr = best.x
            self.params[j] = {"mu": mu, "K_fall": Kf, "c_fall": cf,
                              "K_run": Kr, "c_run": cr,
                              "logL": -best.fun, "n_params": 5}
        return self

    def cross_excitation_test(self, j: str) -> dict:
        """Likelihood-ratio test of 'no cross-excitation into stream j'
        (K_{j,other} = 0), the spirit of the thesis's specification tests.
        Approximate chi-square(2) p-value (boundary caveat noted)."""
        from scipy.stats import chi2
        other = "run" if j == "fall" else "fall"
        src = {k: self.streams[k].times for k in ("fall", "run")}
        tj = self.streams[j].times
        # restricted fit: cross kernel switched off
        def nll_restricted(x):
            full = ([x[0], x[1], x[2], 0.0, 1.0] if other == "run"
                    else [x[0], 0.0, 1.0, x[1], x[2]])
            return self._neg_loglik_target(np.array(full), tj, src, self.T)
        rate = len(tj) / self.T
        best = None
        for x0 in ([rate * f, K, c] for f in (0.5, 0.8)
                   for K in (0.1, 0.4) for c in (10.0, 80.0)):
            res = minimize(nll_restricted, x0, method="L-BFGS-B",
                           bounds=[(1e-8, 5 * rate), (0.0, 0.995), (1e-2, self.T / 10)])
            if best is None or res.fun < best.fun:
                best = res
        # Re-polish the full model from the restricted optimum so the nested
        # model can never spuriously beat the full one (multistart noise).
        mu_r, K_r, c_r = best.x
        seed_full = ([mu_r, K_r, c_r, 1e-4, 20.0] if other == "run"
                     else [mu_r, 1e-4, 20.0, K_r, c_r])
        res_full = minimize(self._neg_loglik_target, seed_full,
                            args=(tj, src, self.T), method="L-BFGS-B",
                            bounds=[(1e-8, 5 * rate), (0.0, 0.995),
                                    (1e-2, self.T / 10), (0.0, 0.995),
                                    (1e-2, self.T / 10)])
        if -res_full.fun > self.params[j]["logL"]:
            mu, Kf, cf, Kr, cr = res_full.x
            self.params[j].update(mu=mu, K_fall=Kf, c_fall=cf, K_run=Kr,
                                  c_run=cr, logL=-res_full.fun)
        lr = 2 * (self.params[j]["logL"] - (-best.fun))
        return {"LR": float(lr), "pvalue": float(chi2.sf(max(lr, 0), df=2)),
                "logL_restricted": float(-best.fun)}

    def branching_matrix(self) -> pd.DataFrame:
        """K[j, k] = expected type-j events directly triggered by one
        type-k event.  Spectral radius < 1 <=> stationary."""
        M = pd.DataFrame(
            [[self.params["fall"]["K_fall"], self.params["fall"]["K_run"]],
             [self.params["run"]["K_fall"], self.params["run"]["K_run"]]],
            index=["-> fall", "-> run"], columns=["from fall", "from run"])
        M.attrs["spectral_radius"] = float(
            np.max(np.abs(np.linalg.eigvals(M.values))))
        return M

    def explain(self) -> str:
        M = self.branching_matrix()
        pf, pr = self.params["fall"], self.params["run"]
        hl = lambda c: c * np.log(2.0)
        lines = [
            "Bivariate Hawkes (falls <-> runs), exponential kernels:",
            f"  Background rates: mu_fall = {pf['mu']:.4f}, mu_run = {pr['mu']:.4f} per period.",
            f"  Branching matrix (direct offspring per event):",
            f"    fall -> fall {M.iloc[0,0]:.2f} | run -> fall {M.iloc[0,1]:.2f}",
            f"    fall -> run  {M.iloc[1,0]:.2f} | run -> run  {M.iloc[1,1]:.2f}",
            f"  Spectral radius = {M.attrs['spectral_radius']:.2f} (< 1: stationary).",
            f"  Decay half-lives into falls: from falls {hl(pf['c_fall']):.0f}, from runs {hl(pf['c_run']):.0f} periods;",
            f"  into runs: from falls {hl(pr['c_fall']):.0f}, from runs {hl(pr['c_run']):.0f} periods.",
        ]
        return "\n".join(lines)

    # ----- joint simulation (two-type Ogata thinning) -----

    def simulate(self, t_start: float, horizon: float, rng,
                 hist: dict[str, list] | None = None):
        """Simulate both streams forward; returns dict of new event times."""
        h = {k: list(self.streams[k].times[self.streams[k].times <= t_start])
             for k in ("fall", "run")} if hist is None else {k: list(v) for k, v in hist.items()}
        new = {"fall": [], "run": []}
        t, t_end = t_start, t_start + horizon
        while t < t_end:
            arr = {k: np.asarray(h[k]) for k in h}
            lam = {j: self.intensity(j, t + 1e-9, arr) for j in ("fall", "run")}
            lam_bar = lam["fall"] + lam["run"]
            if lam_bar <= 0:
                break
            t_cand = t + rng.exponential(1.0 / lam_bar)
            if t_cand > t_end:
                break
            lam_c = {j: self.intensity(j, t_cand, arr) for j in ("fall", "run")}
            tot = lam_c["fall"] + lam_c["run"]
            if rng.uniform() * lam_bar <= tot:
                j = "fall" if rng.uniform() < lam_c["fall"] / tot else "run"
                h[j].append(t_cand)
                new[j].append(t_cand)
            t = t_cand
        return {k: np.array(v) for k, v in new.items()}

    def predict_episode_end(self, j: str, t: float, quiet_window: float,
                            horizon: float, n_paths: int = 400,
                            seed: int = 0) -> dict:
        """Same episode-end forecast as ETASModel.predict_episode_end, but
        the simulated future lets falls and runs feed each other, which
        matters when cross-excitation is strong (as it is for runs here)."""
        rng = np.random.default_rng(seed)
        hist = {k: list(self.streams[k].times[self.streams[k].times <= t])
                for k in ("fall", "run")}
        remaining, extra = [], []
        for _ in range(n_paths):
            new = self.simulate(t, horizon, rng, hist)
            tau, n_extra = t, 0
            for s in new[j]:
                if s - tau >= quiet_window:
                    break
                tau, n_extra = s, n_extra + 1
            remaining.append(tau - t); extra.append(n_extra)
        remaining = np.array(remaining); extra = np.array(extra)
        q = lambda p: float(np.quantile(remaining, p))
        return {"remaining_samples": remaining,
                "p_over_now": float(np.mean(remaining == 0.0)),
                "p_more_events": float(np.mean(extra > 0)),
                "exp_more_events": float(extra.mean()),
                "median": q(0.5), "q10": q(0.10), "q90": q(0.90)}
