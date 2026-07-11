# Financial Earthquakes: a Hawkes/ETAS indicator for falls & runs, with a KAN-PIN parameter estimator

Implementation of the **"financial earthquakes" indicator** of Francine Gresnigt's
Erasmus University work — [*Identifying and Predicting Financial Earthquakes Using
Hawkes Processes* (PhD thesis, 2020)](https://repub.eur.nl/pub/124777/), whose core
chapter is Gresnigt, Kole & Franses (2015), *"Interpreting financial market crashes
as earthquakes: A new Early Warning System for medium term crashes"*, **Journal of
Banking & Finance 56**, 123–139 ([Tinbergen DP 14-067](https://ideas.repec.org/p/tin/wpaper/20140067.html),
[SSRN](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2469697)) — applied to
Alcoa (AA) hourly data, plus a **KAN-PIN** (physics-informed Kolmogorov–Arnold
network) architecture that estimates the model's parameters, built from:

- **PINNverse** — Almanstötter, Vetter & Iber (2025), *PINNverse: Accurate parameter
  estimation in differential equations from noisy data with constrained
  physics-informed neural networks* (arXiv:2504.05248): constrained optimization
  via the Modified Differential Method of Multipliers (MDMM).
- **PI-KAN** — Kashefi & Mukerji (2025), *Physics-informed KAN PointNet*: KAN layers
  with learnable Jacobi-polynomial activations inside a physics-informed network.

> Provenance note: this environment's network policy blocks repub.eur.nl / SSRN /
> arXiv downloads, so the ETAS implementation follows the published specification
> of the model (Ogata's ETAS, the paper's EWS design and evaluation criteria) as
> documented in the references above rather than a line-by-line reading of the
> thesis PDF.

---

## 1. What the indicator is (the idea)

Stock prices around a crash behave like the earth's crust around an earthquake:

- a big shock arrives, then **aftershocks cluster after it** — more extreme moves
  follow, with frequency decaying in time (Omori's law);
- **bigger shocks trigger more aftershocks** than small ones;
- shocks also arrive "spontaneously" at a background ("tectonic") rate.

The indicator models the **arrival times of extreme returns** (not the returns
themselves) as a **marked self-exciting point process** — the Epidemic-Type
Aftershock Sequence (ETAS) model from seismology. Extreme negative returns
("**falls**", beyond the 95% tail quantile) and extreme positive returns
("**runs**") each form an event stream with magnitudes `m_i = |r_i|`. The model's
output is the **conditional intensity**

```
λ(t | history) = μ + K₀ · Σ_{tᵢ<t} e^{α(mᵢ−M₀)} · φ(t−tᵢ)
```

= the expected number of extreme events per unit time *right now, given everything
seen so far*. `μ` is the background rate, `K₀` the average number of direct
aftershocks per event, `α` the magnitude leverage, and `φ` a decay kernel
(power-law/Omori `∝(1+s/c)^−(1+ω)` or exponential `∝e^{−s/c}`). Since `∫φ = 1`,
the **branching ratio `n = K₀·E[e^{α(m−M₀)}]`** is the expected number of direct
children per event — the market's "criticality": `n → 1` means every shock spawns
another and turbulence becomes self-sustaining; `1/(1−n)` is the expected cluster
size.

Everything the user asks of the indicator follows from λ:

- **Early Warning System**: P(≥1 extreme event in the next *d* periods)
  `= 1 − exp(−∫λ)`; signal when it crosses a threshold; scored by hit rate H,
  false-alarm rate F and the **Hanssen–Kuiper skill score KSS = H − F** (paper's
  criteria).
- **How long will the fall/run last, and when does it end**: simulate the fitted
  process forward (Ogata thinning); an episode "ends" at the first moment from
  which a full quiet window passes with no further event. The simulation gives the
  full distribution of the remaining duration, P(the episode is already over), and
  the expected number of further shocks.

**Estimation in the paper: maximum likelihood.**
`ln L = Σᵢ ln λ(tᵢ) − ∫₀ᵀ λ(s)ds` (the integral is analytic), four kernel/leverage
variants compared by AIC, and specification checked by the **time-rescaling
theorem**: the transformed inter-event times `Λ(tᵢ)−Λ(tᵢ₋₁)` must be i.i.d. Exp(1)
(KS test) — the specification-testing theme of thesis chapter 4.

Implementation: [`etas.py`](etas.py) — the module docstring contains the full
mathematical story, `ETASModel.explain()` narrates any fitted model in plain
English, and every method's docstring states the formula it implements.

## 2. What the data show (AA, hourly, Jul 2025 → Jul 2026)

1,749 hourly bars → 1,748 returns; 95% tail thresholds give **88 fall events**
(hourly drop ≤ −1.65%) and **88 run events** (≥ +1.78%). Time runs on the trading
clock (1 unit = 1 bar), so decay is measured in market hours, not calendar hours.

| | falls | runs |
|---|---|---|
| best model (AIC) | exponential kernel, α=0 | exponential kernel, α=0 |
| background μ | 0.042 events/h | 0.050 events/h |
| fertility K₀ (= branching n) | **0.18** | **0.00** |
| kernel scale c (half-life) | 79 h (55 h) | — |
| KS p-value (time-rescaling) | 0.38 ✓ | 0.04 ✗ |

**Falls self-excite; runs don't.** Each extreme drop triggers on average 0.18
further extreme drops, spread over roughly two weeks of trading — the earthquake
analogy holds for crashes. Extreme up-moves arrive like a Poisson process; the
market falls in cascades but rises without momentum in its extremes. This
asymmetry is a known stylized fact of equity tails and is exactly why the paper
targets *crashes*.

A **bivariate mutually-exciting Hawkes** (falls ⇄ runs; the thesis's
cross-excitation/spillover theme, implemented in `CrossETAS`) finds **no
significant cross-excitation** on this sample either (LR tests p ≈ 1.0) — the
mild clustering that the KS test detects in runs is not explained by fall→run
triggering; it is more likely slow background variation (regimes) than
event-level contagion.

The magnitude-leverage parameter α is not identified on one year of data (AIC
prefers α = 0), consistent with the paper needing decades of daily data to pin it.

### The Early Warning System, out of sample

Trained on the first 70% (a calm period: n ≈ 0.07), evaluated hour-by-hour on the
last 30% with d = 7 trading hours (1 day — the hourly analogue of the paper's
"sometime in the next 5 days"): best threshold gives **hit rate 0.38 vs false
alarms 0.24, KSS = +0.15** — positive skill, same sign as the paper, but modest:
one calm training year gives the model little clustering to discriminate with,
and predicted probabilities span only 0.27–0.30. The paper's much stronger scores
come from ~50 years of daily S&P data containing 1987/2008-scale cascades. On
this dataset the indicator's real value is the intensity level itself (fig2) and
the episode nowcasts below, not binary alarms.

### Duration of falls and runs (the user's core question)

Episodes = clusters of same-tail events separated by < 14 trading hours (2 days).
Standing anywhere in time, `predict_episode_end()` simulates the fitted process
forward and reports:

- **P(the episode is already over)** — i.e. P(no further extreme in the quiet window),
- the **distribution of the remaining duration** (median + 80% interval),
- the **expected number of further extreme events**.

Case study (fig4): the largest fall episode (15→24 Jun 2026, 7 events in 42
trading hours). With n = 0.18 the model keeps saying "most likely over within a
day or two" (median 1–3h, 80% band up to ~22h) while this particular episode kept
going for 42h — an honest display of what a *mildly* self-exciting model can and
cannot promise: it prices ordinary clustering, and a run of this length was
genuinely in its tail. On 50 years of daily data with n ≈ 0.8–0.95 (the paper's
regime), these forecasts become sharply informative because most events *are*
aftershocks.

Nowcast at the last bar (10 Jul 2026): P(extreme fall within a day) ≈ 0.31,
within a week ≈ 0.84; the early-July fall episode is ~50/50 to be over, with ≈1.2
further extreme drops expected before it dies out. (Runs: nearly identical
numbers, driven purely by the background rate.)

### Parameter stability (running Nyblom test) and walk-forward validation

Estimation now ships with two robustness layers (`stability.py`), and **both
estimators — the paper's MLE and KAN-PIN — are refit at every step for
comparison** (fig6, fig7):

- **Nyblom (1989)/Hansen (1992) stability test**, adapted to point processes:
  the ETAS log-likelihood is decomposed into per-event score contributions;
  cumulative scores give the classic statistic per parameter and jointly
  (5% critical values 0.470 individual / 1.01 joint for 3 parameters). On the
  full sample: μ 0.30, K₀ 0.26, c 0.19, joint 0.48 — **no parameter instability
  detected** within the year.
- **Walk-forward validation**: an expanding window is refit at 8 steps
  (Dec 2025 → Jul 2026). At each step the Nyblom test is recomputed (the
  *running* test — it stays below the 5% line everywhere) and both parameter
  sets are scored **out of sample** on the next segment with the point-process
  predictive log-score, against a homogeneous-Poisson benchmark.

What it shows, honestly:

| | MLE (paper) | KAN-PIN |
|---|---|---|
| K₀ path across windows | 0.00 → 0.18 (calm windows find no clustering; only the full year does) | 0.26–0.37, remarkably stable |
| c path | rides upward, unidentified while K₀ ≈ 0 | 1–3 h (short-biased but stable) |
| total OOS log-score | **−231.1** | −241.5 |
| Poisson benchmark | −231.0 | |

**Neither estimator beats the Poisson benchmark out of sample on this year of
hourly data** — windowed MLE mostly estimates K₀ ≈ 0 and therefore *is* nearly
Poisson, while KAN-PIN's stable-but-short-kernel clustering slightly overfits.
The running Nyblom never rejects *within-window* stability, yet the K₀ path
shows the regime dependence directly: self-excitation only becomes visible
once the turbulent spring/summer 2026 enters the window. Conclusion of the
validation: on this sample the indicator's in-sample structure is genuine
(AIC, KS test) but *predictively* the clustering is too weak to exploit at
these window sizes — decades of data (the paper's setting) are what make the
EWS skill material.

### Daily data

Aggregating the hourly closes to ~250 daily returns (90% threshold → 25 events):
exponential kernel fits with μ = 0.095/day, K₀ = 0.05, c = 8.5 days, KS p = 0.51.
Direction agrees (mild clustering of daily crashes) but 25 events cannot support
serious inference — the paper uses half a century of daily data. Treat as
illustration; the hourly results above are the substantive ones.

## 3. The KAN-PIN architecture (estimating the parameters with your two papers)

The inverse problem "find η = (μ, K₀, α, c, ω) from event data" is recast in
PINNverse's constrained form, with a KAN as the network ([`kan_pin.py`](kan_pin.py)):

- **State variable**: the compensator `Λ(t) = ∫₀ᵗ λ_η(s)ds`, represented as
  `Λ_θ(t) = N·G(t/T)` with `G(x) = x + KAN(x) − KAN(0)`:
  a **depth-1 Jacobi-polynomial KAN** (learnable edge activations
  `ψ(z) = Σᵢ Λᵢ Pᵢ^{(a,b)}(z)` on tanh-squashed inputs, as in PI-KAN) over
  **Fourier features** `[x, sin 2πkx, cos 2πkx]` (PINNverse's remedy for sharp
  solutions); the linear skip starts the network at the exact Poisson compensator,
  and subtracting `KAN(0)` hard-encodes the initial condition Λ(0)=0.
- **Physics (the DE)**: `dΛ/dt = λ_η(t)`, imposed in **integral (weak) form** on
  collocation intervals — `Λ(s_{k+1}) − Λ(s_k) = ∫λ_η` with an analytic right-hand
  side — because a Hawkes intensity spikes at every event and no smooth network
  satisfies the pointwise residual at that frequency (spectral bias).
- **Data**: the counting staircase N(t)/N sampled over the whole domain (at an
  event the unbiased target is i−½, by time rescaling) — the noisy observation of
  the state, exactly PINNverse's setting.
- **Constrained training (MDMM)**: minimize the data loss subject to
  `L_DE = 0` and bounds `η_lower ≤ η ≤ η_upper` (infeasibility functions +
  multipliers), Adam descent on (θ, η), gradient ascent on the multipliers.

**What it took to make it work** (documented in the module docstring — these are
findings, not footnotes): naive joint MDMM training *always* collapsed into the
degenerate Poisson solution (K₀→0), because the flat compensator satisfies the
physics exactly while the data loss is dominated by counting noise — the
concave-Pareto pathology that motivates PINNverse, in point-process form. The
working architecture is a **four-phase curriculum**: (A) exact ridge
least-squares solve of the KAN on the staircase (a depth-1 KAN is linear in its
coefficients); (B) calibrate η alone against the frozen network's increments
(grid + Adam, 5 parameters); (B2) re-solve the KAN on the analytic compensator
Λ_η (projection onto the physics manifold); (C) joint MDMM to negotiate the final
data/physics balance. Total runtime ≈ 15 s.

### Validation on synthetic data (truth known) and on the real fall events

| | truth | MLE | KAN-PIN |
|---|---|---|---|
| μ | 0.020 | 0.0195 | 0.026 |
| K₀ | 0.60 | 0.62 | 0.43 |
| c | 8.0 | 7.0 | 3.1 |

| AA falls (real) | MLE | KAN-PIN |
|---|---|---|
| μ | 0.042 | 0.029 |
| K₀ (= n) | 0.18 | 0.34 |
| c | 79 | 3.4 |

KAN-PIN detects the clustering and recovers μ and the branching ratio K₀ — the
economically meaningful quantities — at the right order; the kernel scale **c is
biased short** by a documented mechanism (*interpolation leakage*: the smooth
network spreads each event's own +1 step into the immediately following
intervals, which the physics can only read as near-instant aftershocks; MLE is
immune because an event's kernel starts strictly after its own arrival). MLE, the
paper's estimator, remains the right tool whenever the likelihood is tractable —
which is exactly PINNverse's own positioning: constrained physics-informed
training earns its keep when the forward model is expensive or the likelihood
intractable (the thesis's chapter 5 estimates Hawkes models from *option prices*
with machine learning — precisely the regime where a KAN-PIN generalizes and MLE
does not).

## 4. Further applications of the indicator

- **Risk management**: λ(t) is a real-time tail-risk gauge — scale VaR/ES or
  de-lever when the fall intensity is multiples of μ; the branching ratio n is a
  market-fragility monitor (distance to criticality).
- **Crash EWS across assets** (thesis ch. 2–3): multivariate Hawkes with
  cross-excitation turns spillovers (index → single stock, US → EU) into advance
  warning; `CrossETAS` here is the two-stream template.
- **Position management inside turbulence**: `predict_episode_end()` answers "is
  the fall over?" probabilistically — re-entry timing, stop-widening, or option
  hedge unwinding while P(more shocks) is high.
- **Specification testing** (thesis ch. 4): the time-rescaling KS machinery in
  `residual_diagnostics()` tests any intensity model, not just ETAS.
- **Derivatives** (thesis ch. 5): self-exciting jump intensities feed jump-diffusion
  option pricing; the KAN-PIN route is the natural estimator when calibrating such
  models to option surfaces where likelihoods are unavailable.
- **Beyond finance**: the same code fits any clustered event stream —
  order-flow bursts, liquidity droughts, cyber-attack waves, social unrest.

## 5. Conclusion

0. **Stability & validation**: the full-sample Nyblom test finds the ETAS
   parameters stable within the year (joint 0.48 < 1.01), and the running
   test never rejects on any expanding window; but walk-forward validation
   shows neither estimator beats a Poisson benchmark out of sample here —
   the clustering signal, while statistically present in-sample, is too weak
   on one year of single-name hourly data to be predictively exploitable.
1. **The paper's method works as advertised on AA hourly data, with honest
   small-sample limits.** Extreme *falls* behave like earthquakes: a fitted ETAS
   model finds significant self-excitation (n ≈ 0.18, ~2-week decay) and passes
   its specification test, while extreme *runs* are indistinguishable from
   Poisson — crashes cascade, rallies don't. The indicator's derived products —
   crash probabilities (out-of-sample KSS +0.15), episode-end forecasts, and
   fragility (branching-ratio) monitoring — all work mechanically, but their
   *sharpness* is limited by one calm year of data; the paper's headline skill
   needs decades that include true crash cascades.
2. **KAN-PIN can estimate the parameters, and the exercise maps the method's
   real boundary.** A depth-1 Jacobi-KAN over Fourier features, trained
   PINNverse-style under integral-form Hawkes constraints, recovers the
   background rate and branching ratio at the right order in ~15 s — but only
   with a curriculum that first makes the network data-consistent and
   physics-feasible; joint constrained training from scratch reliably collapses
   to the degenerate Poisson solution, and the kernel scale inherits a
   short-bias from interpolation leakage.
3. **Method choice**: for this model class, use the paper's MLE for production
   estimates (efficient, unbiased, seconds); use KAN-PIN as an independent
   cross-check and as the forward path to models where MLE dies (option-implied,
   non-affine, expensive-forward settings) — which is precisely where both the
   PINNverse paper and the Gresnigt thesis point.

## Files

| file | contents |
|---|---|
| `etas.py` | the paper's method: events, ETAS intensity/likelihood/MLE, diagnostics, simulation, EWS, episode-duration forecasts, bivariate cross-excitation; self-explaining docstrings + `explain()` |
| `kan_pin.py` | Jacobi-KAN + Fourier features + integral-form physics + MDMM curriculum estimator |
| `stability.py` | Nyblom/Hansen parameter-stability test (per-event scores) + walk-forward validation engine running both estimators |
| `run_analysis.py` | end-to-end pipeline: all figures, tables, results.json (`python3 run_analysis.py`) |
| `Financial_Earthquakes_Hawkes.ipynb` | executed notebook walking through everything |
| `figures/` | fig1 events, fig2 intensity indicator, fig3 EWS, fig4 duration forecasts, fig5 KAN-PIN, fig6 stability/Nyblom, fig7 walk-forward OOS |
| `AA_h.csv` | the hourly input data (daily series is aggregated from it) |
| `results.json` | key numbers from the last run |

Dependencies: `numpy scipy pandas matplotlib autograd` (pure CPU, no torch).
