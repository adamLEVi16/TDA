# Round 12 Pre-Registration — Optimization Gauntlet

**This file is committed BEFORE any variant is implemented or run.** The git
timestamp of this commit, versus the commit containing results, is the proof
that specs were locked first. Anything not written here does not get tested in
Round 12; any deviation must be reported as a deviation.

## Motivation (from documented diagnostics, not curiosity)
- Whipsaw is the largest measured leak: 66% of ≥50%-cash entries reverse
  within 4 months, costing ~4.5pts each vs staying invested (Round 11).
- Inverse-vol sizing ignores correlations: TLT/IEF (and VUSTX duplication
  risk) double-count bond risk.
- The 2017–24 flat-alpha decade reads as opportunity-set thinning, arguing
  for more uncorrelated sleeves rather than parameter changes.
- The SMA 6–14mo Sharpe plateau (1.15–1.17) says parameter tuning is
  exhausted; only design changes are tested here.

## Variants (exact specs, locked)

**V0 — Baseline (control):** current RP+Trend exactly as in `multi_asset.py`.
Frozen. All comparisons are against V0 on identical data.

**V1 — Continuous trend signal:** replace the binary trend gate with a linear
ramp: scaler `s = clip((P/SMA10 − 0.95) / 0.10, 0, 1)` (0 at 5% below the SMA,
1 at 5% above). Weight = inverse-vol weight × s. The 5% half-width is fixed a
priori; **no other ramp widths will be tried** — if this exact spec fails, V1
fails.

**V2 — Lookback ensemble:** trend scaler = mean of four binary votes
`P > SMA_k` for k ∈ {3, 6, 9, 12} months (fixed set, chosen to bracket the
plateau without including the baseline's 10). Weight = inverse-vol × scaler.
No other lookback sets will be tried.

**V3 — ERC (equal risk contribution) weights:** replace inverse-vol with ERC
solved on the trailing **36-month** covariance of monthly returns
(multiplicative-update iteration, 200 steps; fall back to inverse-vol while
<36 months of history). Trend gate identical to V0 (binary, zeroed when off,
no renormalization). The 36m window is fixed a priori.

**V4 — Universe expansion (8-ETF era only):** add, one at a time, TIP, IWM,
BWX to the 8-ETF universe; evaluation window = common history after each
addition; judged as marginal contribution vs V0 on the same clipped window.
No long-history confirmation exists for these instruments — any positive
result is therefore graded weaker by construction and cannot be "accepted,"
only flagged for future validation.

## Protocol (locked)
1. **Selection window:** 1987-06 → 2005-12 (5-fund universe). Variants with
   selection-window Sharpe < V0's do not advance.
2. **Holdout confirmation:** 2006-01 → 2024-12 (never used for selection).
3. **Cross-instrument confirmation:** full 8-ETF era (2007–2024).
4. **Acceptance criteria (ALL required):** (a) selection Sharpe ≥ V0;
   (b) holdout Sharpe ≥ V0; (c) 8-ETF Sharpe ≥ V0; (d) MaxDD no worse than
   V0 by more than 3pts on any set; (e) annual turnover ≤ 1.5× V0's.
5. **Multiple-testing accounting:** this round adds 4 trials (V4's three
   additions counted as one exploratory trial) on top of ~9 prior project
   hypotheses; any accepted variant's full-sample bootstrap p-value will be
   reported alongside a deflated-Sharpe view using ~13 cumulative trials.
6. **Version discipline:** V0 remains the validated, live (TradingView)
   strategy regardless of outcome. An accepted variant becomes v2-candidate
   and must pass the FULL existing torture suite (torture_test.py +
   extended_tests.py) before replacing anything.
7. **Expected outcome, stated in advance:** most variants fail; a surviving
   variant is expected to add roughly +0.05–0.15 Sharpe via whipsaw/cost
   reduction, not transform the strategy. A result far better than that will
   be treated with suspicion, not celebration.
