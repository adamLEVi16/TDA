# Concept C — Pre-registration (written 2026-09-25 before any test-period evaluation)

Question: do correlation / topology regime indicators improve a LONG-ONLY US equity market position (rest in T-bills)
versus simpler indicators (trailing realised volatility, 10-month trend), out of sample?

Only data loading (`data.py`, `logs_data.txt`) and a ripser timing check were run before this file was written.

## Data
- Market: Ken French daily and monthly `Mkt-RF` + `RF` (market total return = Mkt-RF + RF). 1926-07 .. 2024-12.
- Network: Ken French 49 industry portfolios, daily, value-weighted (first table). Only the **36 industries with no
  missing value 1926-07..2024-12** are used (constant N=36, so H1 counts are comparable through time). Dropped (missing
  early data): Books, FabPr, Gold, Guns, Hlth, LabEq, MedEq, Paper, PerSv, Rubbr, Soda, Softw, Toys.
- Tradable check: Yahoo SPY (auto-adjusted, i.e. dividends reinvested), ^VIX (1990+), ^VIX3M (2006-07+). Cash earns
  French daily RF.

## Periods
- Burn-in: 1926-07 .. 1928-12 (indicator warm-up; expanding thresholds and regressions need history).
- **Design: 1929-01-02 .. 1989-12-29** (weights set at close 1928-12-31). All choices are made here.
- **Test: 1990-01-02 .. 2024-12-31** (weights set at close 1989-12-29). Run once with frozen parameters.
- SPY check: 1994-01-03 .. 2024-12-31. VIX/VIX3M rule: 2006-08-01 .. 2024-12-31 (compared to B&H on the same dates).
- Expanding-window quantities (thresholds, regression coefficients) use all data from 1926 up to t, including test data
  as it arrives (this is real-time information, not look-ahead).

## Indicators (daily, at close t, using data through t only)
Window = last 60 trading days ending at t (inclusive) of the 36 industry daily returns, unless stated.
1. `RV`  : sqrt(252 * mean(r^2)) of daily market total returns over the last 21 trading days.
2. `TREND`: market total-return index at month-end vs mean of the last 10 month-end index values (incl. current); in
   if above. (Weekly-rebalance robustness: index vs its 210-trading-day simple average.)
3. `AC60`: mean off-diagonal pairwise Pearson correlation of the 36 industries.
4. `AR60`: absorption ratio = sum of top 7 eigenvalues (N/5 rounded down) of the 60-day covariance matrix / trace.
5. `H1N` : number of finite H1 intervals from ripser (maxdim=1, full filtration, distance_matrix=True) on
   d_ij = sqrt(2(1-rho_ij)).
6. `H1P` : total H1 persistence = sum(death - birth) over finite H1 intervals.
7. `TOPOVOL`: the student's "topology volatility" = 30-day rolling std of H1N + 30-day rolling std of H1P.
8. `FIED`: Fiedler value (2nd-smallest eigenvalue) of the normalised Laplacian I - D^-1/2 W D^-1/2 with
   W_ij = rho_ij if rho_ij > 0.3 else 0 (zero diagonal; isolated node degree set to 1), exactly as `risk_report.py`.
9. SPY check only: `VIX` level; `VIXTS` = VIX / VIX3M.
Monthly: `ACM` = average pairwise correlation of the 36 industries' daily returns within calendar month m (Pollet-Wilson
style, industries instead of stocks); `RVM` = sqrt(252*mean(r^2)) of daily market returns within calendar month m.

## Strategies (long-only; weight w in the market, 1-w in T-bills)
Rebalance: **monthly**, at the close of the last trading day of each month, using indicator values at that close;
the new weight earns returns from the next trading day. Between rebalances the market weight drifts with returns.
Weekly rebalance (last trading day of each week) is a robustness check.

### Baselines
- `BH`: buy-and-hold market (w=1 always, no trading cost).
- `TREND10`: w = 1 if TREND in else 0.
- `VT-naive`: w = min(cap, sigma* / RV_t) (Moreira-Muir style, trailing vol used directly as forecast).
- `VT-RV`: w = min(cap, sigma* / sigmahat_t), sigmahat from an expanding OLS of log RVM_{m+1} on [1, log RV_m]
  (monthly, non-overlapping, estimated on all month pairs whose outcome is known at t; min 24 pairs);
  sigmahat = exp(fitted + s^2/2).
- `VT-RV x TREND10`: VT-RV weight times TREND10 in/out.

### Regime rules under test
- Binary `B-X` for X in {RV, AC60, AR60, H1N, H1P, TOPOVOL, FIED}: w = 0 (cash) when X is in its risk-off region,
  else 1. Risk-off region: X_t > Q_q(X history through t) if direction = "high", X_t < Q_(1-q)(history) if "low".
  History = all daily values of X from its first value through t (expanding). Grid chosen on DESIGN only:
  direction in {high, low} x q in {0.50, 0.75, 0.90} = 6 variants per indicator, 42 in total. Selection criterion:
  design-period net (2 bp) excess-return Sharpe. The winner per indicator is frozen and carried to test even if it
  does not beat BH in design.
- Continuous `VT-RV+X` for X in {AC60, AR60, H1N, H1P, TOPOVOL, FIED}: as VT-RV but the regression is
  log RVM_{m+1} on [1, log RV_m, X_m]. No free parameters beyond sigma* and cap.
- SPY check adds `B-VIX` (direction and q frozen = those chosen for B-RV on design), `B-VIXTS` (out when
  VIX/VIX3M > 1.0, fixed, no fitting), `VT-VIX` (regression of log next-month FF-market RVM on [1, log VIX_m],
  expanding from 1990-01, min 24 pairs) and `VT-RV+VIX`.
  For SPY, RV and TREND are computed from SPY itself; the VT regressions' coefficients are estimated on the
  French market series (same economic quantity, longer history) and applied to SPY-based inputs; the network
  indicators and their expanding thresholds are the French-industry series.

### Fixed parameters
- sigma* = annualised std of daily French market total returns over 1927-01-01 .. 1989-12-31 (computed in design).
- cap = 1.0 (no leverage: a small cash account). Robustness: cap = 1.5 with borrowing cost RF + 0.50%/yr on (w-1)+.
- Costs: 2 bp one-way x |w_target - w_drifted| at every rebalance (index/SPY default). Robustness: 10 bp.
  BH pays no cost. Break-even costs reported: BE0 = cost at which mean net excess return = 0, and BE_BH = cost at
  which net Sharpe = BH Sharpe.

## Metrics (per period)
- Excess-return Sharpe (annualised, from monthly compounded strategy return minus monthly compounded RF), with
  Newey-West (6 lags) t-stat of the mean monthly excess return and a 95% CI from a circular block bootstrap
  (12-month blocks, 5000 reps, seed 20260925).
- CAGR (total return), max drawdown (daily wealth), Calmar = CAGR/|MDD|, annual vol, average weight,
  turnover (sum |dw| per year), cost drag (%/yr).
- Sharpe difference vs BH: paired circular block bootstrap (same blocks, 12 months, 5000 reps) two-sided p, and
  Jobson-Korkie with Memmel (2003) correction on monthly excess returns. Incremental tests VT-RV+X vs VT-RV and
  B-X vs B-RV / TREND10 the same way. Holm correction reported across the 13 test-period rules vs BH.

## Predictive regressions (the horse race), monthly, non-overlapping
- Vol: y = log RVM_{m+1}. Base = [1, log RV_m]. Augmented = base + X_m for X in {AC60, AR60, H1N, H1P, TOPOVOL, FIED}.
  Plus: base2 = [1, log RV_m, AC60_m], augmented2 = base2 + {H1N, H1P, TOPOVOL, FIED} ("does topology add anything
  beyond RV and average correlation?").
  Design: in-sample OLS, NW(6) t of the added coefficient(s) (Wald for the 4-term block).
  Test: expanding-window one-step forecasts for 1990-01..2024-12 outcomes; OOS R^2 = 1 - SSE_aug/SSE_base;
  Clark-West (2007) one-sided test with NW(6) SE.
- Returns: y = French monthly Mkt-RF in month m+1. Predictors (univariate): ACM, AC60, RVM (as variance RVM^2),
  AR60, H1N, H1P, TOPOVOL, FIED; and Pollet-Wilson bivariate [ACM, RVM^2].
  Design: in-sample OLS NW(6) t. Test: Campbell-Thompson OOS R^2 vs the expanding historical mean, forecasts for
  1990-01..2024-12, expanding from 1926 (min 24 obs), unrestricted and CT-restricted (forecast truncated at 0;
  ACM slope set to 0 if negative, since Pollet-Wilson predict a positive sign). Clark-West test.

## Hypotheses (nulls)
- H1: none of AC60/AR60/H1N/H1P/TOPOVOL/FIED improves OOS forecasts of next-month RV beyond log RV (OOS R^2 <= 0).
- H2: none predicts next-month excess returns OOS (CT OOS R^2 <= 0); includes the Pollet-Wilson ACM claim.
- H3: no binary correlation/topology rule has a higher test-period Sharpe than BH, TREND10 or B-RV.
- H4: VT-RV+X does not beat VT-RV (Sharpe difference = 0).
- H5 (baseline sanity): VT-RV / VT-naive / TREND10 Sharpe = BH Sharpe.

## Robustness (test period, after the main run; all listed here in advance)
10 bp costs; weekly rebalance; cap 1.5 with borrowing cost; subperiods 1990-2007 and 2008-2024; crisis behaviour
(average weight in 2008-09-01..2009-03-31, 2020-02-20..2020-04-30, 2022-01-01..2022-10-31); SPY check monthly and weekly.

## Timing check
Code asserts that each day's portfolio return uses the weight set at the previous close, and a deliberately
look-ahead version (weight applied same day) is computed once as a diagnostic to show the lag matters.

## Variant count
Design grid 42 binary + 1 trend + 3 VT baselines + 6 VT-RV+X + 1 combo; test: 13 frozen rules + 3 baselines + BH.
Any variant added later will be listed in the final report with the count.
