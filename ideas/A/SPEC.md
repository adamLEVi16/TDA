# SPEC — Concept A: monthly network (correlation-graph) momentum

Written 2026-09-25 BEFORE any design- or test-period portfolio evaluation. Only data loading / gap checks
(`data.py`, `logs_data.txt`) have been run so far. Anything added after this point is listed under
"Amendments" at the bottom with a timestamp and reason.

## Hypotheses (directional, pre-stated)

- **H1 (Fama-MacBeth).** In monthly cross-sectional regressions of next-month industry return on
  cross-sectionally z-scored OWN1, OWN12, NET1, NET12, the average NET1 and/or NET12 slope is > 0
  (Newey-West t, 6 lags). This is the cleanest "beyond own momentum" test.
- **H2 (spanning, own momentum).** The NET1 and NET12 quintile long/short portfolios have a positive
  intercept when regressed on the OWN12 L/S and OWN1 L/S portfolios (NW t, 6 lags).
- **H3 (spanning, factors).** Same intercept vs FF5 + UMD (Mkt-RF, SMB, HML, RMW, CMA, UMD).
- **H4 (tradable ETF version).** Long-only top-5 ETFs by NET signal has a higher Sharpe (excess of RF)
  than the equal-weight ETF universe and than own-12-1 top-5, net of 3 bp costs, 2007-2024.
- Null expectation, stated honestly: industry momentum (OWN12, OWN1) is known and the graph neighbours of
  an industry are highly correlated with it, so NET signals may be mostly a noisy copy of OWN signals.

## Universes and data

1. **Industries.** Ken French 49 value-weighted industry portfolios. Monthly returns from
   `49_Industry_Portfolios_CSV.zip` (first table, VW); graph correlations from the daily file
   `49_Industry_Portfolios_daily_CSV.zip` (first table, VW). -99.99/-999 = missing.
   Eligible at month-end t: >= 240 non-missing daily returns in the trailing 252 trading days ending at t,
   and non-missing monthly returns t-11..t. If an eligible industry's t+1 return is missing (only Rubbr has
   interior gaps), it earns 0 that month; the number of such cases is logged.
   Costs: 10 bp one-way.
   - Design period: return months from the first valid formation date through 1989-12.
   - Test period: return months 1990-01 to 2024-12, evaluated ONCE with frozen parameters.
   - Subperiods of test: 1990-01..2007-12 and 2008-01..2024-12.
2. **ETFs.** XLB XLE XLF XLI XLK XLP XLU XLV XLY EWJ EWG EWU EWC EWA EWH EWS EWZ EWW EWT EWY EEM EFA TLT IEF
   SHY LQD HYG GLD DBC VNQ (yfinance, auto_adjust=True; SPY for the benchmark). Monthly returns = month-end
   adjusted close ratios. Eligible at t: >= 240 daily returns in trailing 252 days and 13 month-end prices
   (t-12..t). Return months 2007-01..2024-12. Same frozen parameters, no retuning. Costs 3 bp one-way
   (SPY buy-and-hold 2 bp).
   Survivorship/selection: the ETF list is chosen today from funds that still exist and are famous; this is
   a look-ahead selection bias that favours the universe as a whole (not obviously any one signal).

## Graph (reuses the student's machinery, monthly)

At month-end t: C = Pearson correlation matrix of trailing 252 daily returns ending at t (pairwise
complete). For each asset i, N_k(i) = the k assets j != i with the highest C_ij. Weights
w_ij = max(C_ij, 0) / sum_{j in N_k(i)} max(C_ij, 0) (equal weights if the sum is 0).

## Signals (all known at the close of month t)

- OWN1_i = r_i(t)                                  (1-month; industry 1-month momentum/reversal baseline)
- OWN12_i = prod_{s=t-11}^{t-1}(1 + r_i(s)) - 1    (12-1 momentum baseline, skips month t)
- NET1_i = sum_j w_ij r_j(t)                       (neighbours' last-month return)
- NET12_i = sum_j w_ij OWN12_j                     (neighbours' 12-1 momentum)
- LAPGAP1_i = [(I - 0.5 L)^3 x]_i - x_i, x = r(t)  (original code's smoothing, alpha=0.5, T=3; symmetric
  normalized Laplacian of the symmetrised kNN graph A_ij = max(C_ij,0) if j in N_k(i) or i in N_k(j);
  sign = "catch up to peers", i.e. minus the original residual)
- NET1_PURE_i = residual of cross-sectional OLS of NET1 on [1, OWN1, OWN12] at month t (network component
  orthogonal to own momentum, estimated with month-t data only)

## Parameters

- Correlation window 252 days (fixed, from the task). alpha = 0.5, T = 3 (fixed, from original code).
- **k in {3, 5, 10}: the only tuned parameter.** Chosen on the industry design period as the k that
  maximises the average of net-of-cost Sharpe of the NET1 and NET12 quintile L/S portfolios. Frozen to
  `frozen_params.json` before the test run.
- Quintiles: top/bottom 20% of eligible industries (rounded to nearest integer, min 1). ETFs: N = 5 fixed.

## Portfolios (monthly rebalance at close of t, hold t+1)

- Industries: (a) L/S = +1 equal-weight top quintile, -1 equal-weight bottom quintile (gross 2);
  (b) long-only EW top quintile; baselines: EW all eligible industries (monthly rebalanced), and the market
  (FF Mkt-RF + RF, no cost).
- ETFs: long-only EW top 5; L/S top 5 minus bottom 5; baselines: EW all eligible ETFs, inverse-volatility
  weighted all eligible ETFs (weights ∝ 1/trailing-252-day vol; "realised-vol sizing"), SPY buy-and-hold,
  OWN12 top 5, OWN1 top 5.
- Costs: cost x sum_i |w_target,i - w_drift,i| at every rebalance, where w_drift = w(1+r)/(1+sum w r) is the
  previous month's weights after drift (first month: from zero). Break-even cost = gross mean monthly
  return / mean monthly one-way turnover (bp).

## Timing

Signal at t uses daily data with dates <= t and monthly returns with dates <= t; the position earns the
return of month t+1. Asserted in code (window end <= t < return month).

## Metrics

Annualised mean, vol, Sharpe (long-only: excess of Ken French RF; self-financing L/S: no RF), NW t-stat
(6 lags) of the mean monthly (excess) return with 95% CI, Sharpe 95% CI from a stationary block bootstrap
(mean block 6 months, 2000 draws, seed 0), max drawdown, turnover, break-even cost. Differences
("beats"): paired stationary block bootstrap of the Sharpe difference (same settings), two-sided p.
Spanning regressions: OLS with NW (6 lag) SEs, alpha annualised.

## Robustness (reported, never used to pick parameters)

All k in {3,5,10} on the test period; tercile instead of quintile; subperiods; placebo: NET1/NET12 L/S with
k random neighbours (200 draws, seed 0) to test whether the correlation graph matters.

## Amendments (all added AFTER the runs named; none changes a frozen parameter)

1. After design.py + test_industries.py first ran: `rank_weights` was vectorised for speed (placebo runtime).
   Tie-breaking changed slightly (only OWN1 affected, e.g. test L/S OWN1 gross +3.96% -> +3.93%). Both scripts were
   re-run with identical frozen k=5; the current logs are the re-run; first-run logs kept in cache/*_v1.txt.
2. After test_etfs.py ran: robust_etfs.py (POST-HOC, not pre-registered): (a) N in {3,7} x k in {3,10} grid for
   long-only ETF NET1/NET12 and OWN12 N in {3,7} (21 cells printed, 18 of them extra beyond the pre-registered N=5,k=5 cells);
   (b) random-neighbour placebo for ETFs (200 draws, seed 0); (c) an earlier ETF holdout 2000-01..2006-12 with
   frozen parameters (data never looked at before this run).
3. test_etfs.py also reports descriptive halves 2007-2015 / 2016-2024 and calendar-year returns (not in SPEC).
