# The statistical tests: what each one is, what it detects, and how it becomes a feature

Every test below is computed on a **trailing window ending at bar t**, so its
value at t is a function of data up to t only — the whole feature matrix is
walk-forward safe by construction. Labels (forward returns, episode
terminations) are the only objects allowed to look ahead, and the validation
machinery purges any training row whose label window overlaps a test block.

---

## 1. Right-tailed ADF test (Phillips, Wu & Yu 2011)

**What it is.** The Augmented Dickey-Fuller regression

    Δy_t = α + ρ·y_{t−1} + Σ φ_i·Δy_{t−i} + ε_t

on log prices, but with the *right-tailed* alternative: instead of testing
H0: ρ = 0 (unit root) against ρ < 0 (mean reversion), we test against
**ρ > 0 — an explosive autoregressive root**. A large positive t-statistic
of ρ̂ is direct evidence that price is growing faster than a random walk,
i.e. the defining property of a rational bubble (Diba & Grossman 1988).

**Feature.** `rtadf` — the t-stat of ρ̂ over a fixed trailing window
(168 hourly bars ≈ 5 weeks). Rising values = building explosiveness.

## 2. SADF → GSADF / BSADF (Phillips, Shi & Yu 2015)

**What it is.** A single fixed-window ADF misses bubbles that start and
collapse inside the sample. PSY's solution: take the **supremum of ADF
statistics over many windows**. The GSADF statistic sups over *all*
windows (both endpoints vary) and is the most powerful known test for
multiple periodically-collapsing bubbles. Its real-time companion, the
**backward sup ADF (BSADF)**, sups at each bar t over all windows *ending*
at t — this is the object that supports live date-stamping: the market is
in an explosive phase at t when BSADF(t) exceeds its critical value.

**Critical values.** No closed form; we simulate the null (driftless random
walk — ADF t-stats are scale invariant) with the exact same sample size and
window configuration, and take per-bar quantiles of the simulated BSADF
sequences (95% by default; 200 replications, cached to disk).

**Features.** `bsadf` (the statistic), `bsadf_gap` (statistic minus critical
value), `bubble_flag` = 1{BSADF > cv95} — **this flag is the treatment
variable of the causal forest** — and `bubble_age` (bars the flag has been
on). The full-sample GSADF = sup_t BSADF(t) is reported in the summary as
the formal whole-sample test.

**Implementation note.** Naively this is O(n²) OLS fits. We precompute
prefix sums of all regressor cross-products, making each window's
sufficient statistics O(1); all ~600k window regressions and the Monte
Carlo are batched in numpy (verified to 1e-8 against statsmodels).

## 3. CUSUM test (Brown, Durbin & Evans 1975)

**What it is.** Fit the AR(1) model recursively through the window; each
new observation produces a **recursive residual** (its one-step-ahead
standardized prediction error). Under parameter stability the cumulative
sum of these residuals is a driftless martingale that stays inside a
parabolic ±boundary; when the regression parameters shift (a bubble
igniting or bursting changes drift/persistence), the CUSUM wanders out.
The CUSUM-of-squares variant is sensitive to *variance* regime changes.

**Features.** `cusum` — max |CUSUM| relative to the 5% Brown-Durbin-Evans
boundary (values > 1 = significant instability); `cusum_sq` — scaled
maximum deviation of the CUSUM-of-squares from its null expectation.
(In this data-set, `cusum_sq` came out as the single most important feature
of the direction benchmark.)

## 4. Chow test (Chow 1960)

**What it is.** The classical F-test for a structural break at a **known**
candidate date: fit the AR(1) on the full window and on the two sub-samples,
and compare SSRs. It answers "did the data-generating process before and
after the candidate date differ?"

**Feature.** `chow_f` — F-statistic for a break at the trailing window's
midpoint (the conventional agnostic choice). Interpretable as "how
different is the last month from the month before it."

## 5. QLR / sup-F test (Quandt 1960; Andrews 1993)

**What it is.** When the break date is unknown, the correct procedure is
the Chow F maximized over all candidate dates in the central 70% of the
window (15% trimming). Andrews derived its non-standard limit distribution.
It is the canonical single-unknown-break test and much more powerful than a
midpoint Chow when breaks are off-center.

**Features.** `qlr_f` — the sup-F value; `qlr_loc` — the *relative position*
of the argmax break in the window (values near 1 mean "the most likely
break just happened", a natural bubble-inception/collapse indicator; this
was the 2nd most important benchmark feature on this data-set).

## 6. Bai-Perron test (Bai & Perron 1998, 2003)

**What it is.** Generalizes break testing to **multiple simultaneous breaks
at unknown dates**: for each break count m it finds the globally
SSR-minimizing partition of the sample into m+1 regimes (dynamic
programming), then selects m by an information criterion. It answers not
just "did something break" but "how many regimes and where".

**Features.** `bp_nbreaks` (BIC-selected number of breaks in the trailing
window), `bp_since` (bars since the last detected break — freshness of the
current regime), `bp_dmean` (change in mean log return between the last two
regimes — the direction the last break moved the market). Our DP is exact
on a candidate grid with minimum segment length, vectorized across windows.

## 7. `ruptures` change-point detection (Killick et al. 2012; Truong et al. 2020)

**What it is.** The `ruptures` package implements modern penalized
change-point algorithms. We use **PELT** — exact penalized segmentation in
linear time — with the Gaussian cost on standardized log *returns*, so it
reacts to both mean and **variance** shifts. This complements Bai-Perron
(which targets the AR structure of *levels*): volatility-regime changes are
a canonical companion of bubble phases — quiet grind-up, violent collapse.

**Features.** `rup_ncp` (change points in the window), `rup_since` (bars
since the last one), `rup_lvr` (log variance ratio of the newest segment vs
the previous one — positive = vol expansion, the collapse signature).

## 8. Hidden Markov Model regimes (Hamilton 1989)

**What it is.** A Gaussian mixture over *persistent latent states* fitted
to returns by EM — the workhorse regime-switching model. States sort into
low-mean/high-vol ("bear/crash"), mid, and high-mean ("bull/run-up") regimes.

**Look-ahead control (important).** The HMM is refit every 50 bars using
only past returns, and between refits we compute **filtered** probabilities
P(state_t | r_1..r_t) with the forward recursion. Off-the-shelf
`predict_proba` returns *smoothed* (forward-backward) posteriors that
condition on the future — a classic silent leak we deliberately avoid.

**Features.** `hmm_p_bull`, `hmm_p_bear` (filtered regime probabilities),
`hmm_mu`, `hmm_sig` (filtered one-step expected return and vol),
`hmm_entropy` (regime ambiguity — spikes near transitions).

---

# The meta-model: Causal Random Forest (Wager & Athey 2018; Athey, Tibshirani & Wager 2019)

## Why causal rather than predictive

A plain classifier trained on "bubble signal → forward return" learns the
*correlation* between the signal and returns, which is contaminated by
everything that co-moves with both (volatility, momentum, regime). The
causal-forest framing asks the sharper question:

> **Given the current stability/regime context x, what does the presence of
> a statistically detected explosive episode do to the next h-bar return?**

    Y  = forward 21-bar log return                     (outcome)
    T  = bubble_flag = 1{BSADF > cv95}                 (treatment)
    X  = stability + break + HMM + control features    (effect modifiers / confounders)

`CausalForestDML` (econml) first **residualizes** Y and T on X with
cross-fitted nuisance models (double machine learning, Chernozhukov et al.
2018) — removing the part of the return predictable from context alone and
the part of the signal explained by context — then grows an honest forest
on the residuals to estimate the **conditional average treatment effect**
τ(x) with valid confidence intervals.

## Reading τ as bubble direction

- **τ̂(x) > 0** — in this context, explosiveness adds *positive* forward
  return: the bubble is in its growth leg → long.
- **τ̂(x) < 0** — in this context, explosiveness predicts *reversal*:
  collapse risk dominates → short / exit.

This matches the PSY-literature view that an exceedance episode contains
both the run-up and the incipient collapse, and which one you're in is a
function of regime context — exactly the heterogeneity a causal forest is
built to capture.

## Leakage hygiene

- Features mechanically derived from the treatment definition (`bsadf`,
  `bsadf_gap`, `bubble_age`, the flag itself) are **excluded from X** —
  they would make the propensity model degenerate (no overlap).
- Nuisance cross-fitting uses contiguous, unshuffled folds.
- The outer evaluation is purged walk-forward (Lopez de Prado 2018): the
  last h training rows before each test block are dropped because their
  labels overlap the test window.

## Honest caveats

- Identification is *selection-on-observables*: τ is causal only to the
  extent the feature set spans the confounders of the explosiveness flag.
- Explosive bars are rare (~7% here), so per-fold treated counts are small;
  treat single-asset results as a research signal, not a backtest.

---

## References

- Brown, Durbin & Evans (1975), *Techniques for Testing the Constancy of Regression Relationships over Time*, JRSS-B.
- Chow (1960), *Tests of Equality Between Sets of Coefficients in Two Linear Regressions*, Econometrica.
- Quandt (1960), JASA; Andrews (1993), *Tests for Parameter Instability and Structural Change with Unknown Change Point*, Econometrica.
- Bai & Perron (1998, 2003), *Estimating and Testing Linear Models with Multiple Structural Changes*, Econometrica / J. Applied Econometrics.
- Hamilton (1989), *A New Approach to the Economic Analysis of Nonstationary Time Series and the Business Cycle*, Econometrica.
- Phillips, Wu & Yu (2011), *Explosive Behavior in the 1990s NASDAQ*, International Economic Review.
- Phillips, Shi & Yu (2015), *Testing for Multiple Bubbles*, International Economic Review.
- Killick, Fearnhead & Eckley (2012), *Optimal Detection of Changepoints with a Linear Computational Cost*, JASA.
- Truong, Oudre & Vayatis (2020), *Selective Review of Offline Change Point Detection Methods*, Signal Processing.
- Wager & Athey (2018), *Estimation and Inference of Heterogeneous Treatment Effects using Random Forests*, JASA.
- Athey, Tibshirani & Wager (2019), *Generalized Random Forests*, Annals of Statistics.
- Chernozhukov et al. (2018), *Double/Debiased Machine Learning*, Econometrics Journal.
- Lopez de Prado (2018), *Advances in Financial Machine Learning*, Wiley.
