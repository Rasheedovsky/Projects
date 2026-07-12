# Critique response — adversarial referee pass (Phase 8)

An independent adversarial referee reviewed the executed pilot (all reports,
configs, source modules, driver scripts, committed artifacts, and the git
history). Fourteen findings; every one is answered below with the action
taken. Actions marked **[fixed]** are in the final reported run; actions
marked **[disclosed]** are documented limitations deferred to the full
protocol.

## Blocker

**1. MAE checkpoint predated the gate-(b) fix and was irreproducible from
HEAD.** Correct on every particular. **[fixed]** — `src/spydt/ssl.py` was
rewritten (per-window bars, era-morning-calibrated threshold, per-offset FFD
lanes; construction tag "v2" in the provenance file), the v1 checkpoint was
deleted, the MAE retrained from the HEAD code, and all 45 `vb_mae` fits plus
the meta trial rerun against the v2 checkpoint. The reported S6 and primary
results are v2-based. The checkpoint remains gitignored (binary artifact),
but is exactly reproducible via `python scripts/run_pretrain.py` from the
committed code and manifest.

## Major

**2. Undisclosed mid-study rerun after test-fold exposure.** True: a full
evaluation (including all primary fits) completed minutes before gate (b)
first ran and failed; everything derived was wiped and rerun under the fix.
**[fixed]** — now disclosed in `reports/results.md` § Disclosures with the
commit trail. We note the direction of the change was conservative and the
outcome NULL in both runs, but agree disclosure is mandatory, and that the
protocol error was running the gates after the matrix; the full protocol
gates before any test-fold fit.

**3. "Validated null" overclaims; no positive control at realistic effect
size.** **[fixed]** — language changed to "gated null" with an explicit
statement of what the gates do and do not establish; a soft positive-control
gate (f) was added that plants a series-borne signal with Bayes accuracy
≈56% and reports how much the encode+train pipeline recovers. Power caveats
(58% pad rate, always-trade cost floor, SSL domain shift) are listed in the
discussion. The cost-floor point is additionally addressed by the measured
era cost column (finding 7).

**4. Gate (a) underpowered with a fudge factor.** **[fixed]** — gate (a) now
runs on the production split-0 bundle (~1,570 train / ~860 test days), the
pass condition is a clean |acc−0.5| ≤ 95% CI half-width with no additive
slack, and the production `ffd_grid_step` question is mooted for (a) because
the production bundle itself is used. Gates (b)/(g) still rebuild small
bundles with a coarser d* grid for runtime; the properties they assert
(bit-identity under corruption) are grid-independent.

**5. S3 tested a different comparison than registered.** Correct — a genuine
specification error: `vb2d` carries V-B content, not V-C content.
**[fixed]** — `vc2d` (identical V-C 3-slice content, 2D layout) was built and
run (3 seeds × 15 splits), and S3 is now S3a (vc > vc2d) AND S3b (vc > va,
the no-cross-plane control, matching PLAN Phase 2's second ablation). The
amendment is dated and disclosed in `configs/registry.yaml` (amendment_v2)
and the new trials enter the DSR count. The success-criteria row now requires
both S3 components with significance.

**6. Secondary hypotheses adjudicated without statistical tests; S7
confounded by seed count.** **[fixed]** — every S-hypothesis now reports the
HAC-DM p-value on daily net-return differentials alongside direction, and
"PASS" requires p<0.05 (directional-only results are labeled as such);
`vb_shuffled` was rerun with 3 seeds so S7 compares matched ensembles.

**7. Headline cost was flat-assumed, not measured as promised.**
**[fixed]** — a measured era cost series (half of a 1-cent spread at the
daily 15:31 entry price + the configured fee allowance) is computed in
`run_report.py`, reported in the header and as a cost-table column. The
registered success criteria remain evaluated at the registered flat 1 bp
(changing the criterion post hoc would be its own forking path); the measured
column shows the flat assumption's direction: measured costs average below
1 bp/side in this sample, so the flat headline is conservative.

**8. Train-fold-only is not past-only; embargo never verified.**
**[partially fixed / disclosed]** — `assert_no_leakage` now verifies the
embargo window too (with a unit test), and gate (d) exercises it. The
deeper point — thresholds/d*/scalers fit on train folds that postdate a test
block — is standard CPCV practice but real; it is now discussed in the
disclosures, and a walk-forward robustness pass is explicitly deferred to
the full protocol. We note it cannot manufacture directional label
information (it shifts feature *representations*, not future returns), which
is why we class it as a robustness item rather than a leak.

## Minor

**9. Meta trial statistic reuse and feature shift.** **[fixed]** — the meta
model now uses each split's own bundle tabular statistics and fits once per
split on the seed-ensembled validation probabilities (matching what it is
applied to). Platt/isotonic calibration and the sized-bet variant remain
unrun in the pilot — **[disclosed]** in the registry notes.

**10. Registered gate "threshold calibration train-days-only" was never
implemented as an executable proof.** **[fixed]** — gate (g): every non-train
day is corrupted ×1000 and the bar threshold plus all train-row features must
be bit-identical.

**11. r_on dividend correction promised but absent.** **[disclosed]** — the
pilot reports raw r_on; ~44 quarterly ex-div days carry ≈ −20 bp artifacts
into the Gao baselines' r_on regressor and one tabular feature. The audit's
distribution-date scan is the hook for the full-protocol fix (correction from
SPY distribution history). Direction of bias: it adds noise to r_on-based
baselines; it does not manufacture signal for the primary.

**12. Stale pre-fix diagnostic numbers quoted as justification.**
**[fixed]** — `registry.yaml` pilot_scope_notes now carry the post-fix
numbers (29.9% vs 7.4% floor-degeneracy); the bar-scheme decision itself is
unchanged — the post-fix diagnostic reaches the same ordering.

**13. Statistical-procedure deviations (CSCV S=8 vs 16; DSR return object;
effective-trials estimator; hit-rate CI mixing).** **[fixed/disclosed]** —
CSCV now runs the registered S=16 (S=8 kept as reference); the DSR's return
object (consensus across paths) and the participation-ratio effective-trials
estimator are now stated in the disclosures as pilot choices; the hit-rate
criterion's CI is computed on the consensus series whose hit rate is
reported.

**14. MOC proxy and embargo rationale.** **[disclosed]** — 15:59-close proxy
for the MOC print and unmodeled auction slippage are in the disclosures; the
embargo rationale is corrected to the FFD weight-window memory (~6 morning
days), with the one-day shortfall documented and set to 6 in the full
protocol.

## Referee's positive notes (for completeness)

The referee independently confirmed: gate (b) caught a real construction
leak mid-study and the response (full wipe + rebuild) was correct; the
registry freeze is git-verifiable and was honored; the Phase-0 audit
(timezone triangulation, adjustment anchor, volume units, ex-ante exclusion
rules) is solid.

## Bottom line

We accept the referee's summary: the result is a **null at ~1 bp costs,
under an always-trade sign rule, on a majority-padded info axis** — and with
the v2 MAE, matched-seed S7, corrected S3 controls, measured costs, and
hardened gates, that null is now as clean as this pilot's scale allows. No
finding overturned the null; several materially narrowed what it claims.
