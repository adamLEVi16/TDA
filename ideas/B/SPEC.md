# Concept B: pre-registration (written before any strategy return was computed)

Written 2026-09-25, after `data_load.py` had run (it only builds the universe and prints the equal-weight
universe return next to SPY, to size the survivorship bias). No signal or strategy return had been computed.

## Question
Does the student's graph-Laplacian residual have an edge at any horizon (1 day, 1 week, 1 month, 12-1 months)
that survives costs on individual large-cap US stocks? Does graph-neighbour momentum spillover
(neighbours' past returns predicting a stock's next-month return) work? How do both compare with the standard
baselines on the same dates and at the same costs?

## Hypotheses
- H1 (horizon map). For each residual horizon k in {1d, 5d, 21d, 12-1m} and each residual type (Laplacian LR, neighbour
  NR), the quintile long-short on the cumulative residual has a non-zero test-period net mean return. The sign is picked
  on the design period (2005-2014), then frozen.
- H2 (spillover). A stock whose graph neighbours had high past returns (1 month, or 12-1 months) earns a higher next-month
  return. Expected sign a priori: continuation (+). The sign is still picked on the design period, as for every signal.
- H3 (baselines). 12-1 momentum (expected +), 1-week reversal (expected -), 1-month reversal (expected -), and
  industry-adjusted 1-week reversal (expected -). Signs are picked on the design period.
- Null for every strategy: test-period mean net daily return = 0 (two-sided, Newey-West).

## Universe (survivorship-biased by construction)
S&P 100 constituents per the Wikipedia table (page last edited 2026-09-19, fetched 2026-09-25, saved as
`sp100_wikipedia_2026-09-25.csv`, with GICS sectors). I keep a symbol only if Yahoo has a valid adjusted close on
2005-01-03 and on every trading day to 2024-12-31 (forward-fill of at most 5 days). GOOG is dropped as a duplicate of
GOOGL. That leaves **83 stocks**:

AAPL ABT ACN ADBE AMAT AMD AMGN AMT AMZN AXP BA BAC BKNG BLK BMY BNY BRK-B C CAT CMCSA COF COP COST CRM CSCO CVS CVX DE
DHR DIS DUK EMR FDX GD GE GILD GOOGL GS HD IBM INTC INTU ISRG JNJ JPM KO LIN LLY LMT LOW LRCX MCD MDLZ MDT MMM MO MRK MS
MSFT MU NEE NFLX NVDA ORCL PEP PFE PG QCOM RTX SBUX SCHW SO T TMO TXN UNH UNP UPS USB VZ WFC WMT XOM

The following were dropped for starting later than 2005-01-03 (18 in total): ABBV, ANET, AVGO, DELL, GEV, GM, GOOG (duplicate), MA, META,
NOW, PANW, PLTR, PM, SNDK (no Yahoo data), TMUS, TSLA, UBER, V.

These are the stocks that are large today. Stocks that were large in 2005 and later shrank, merged or went bust are
missing. The equal-weight universe returned 14.75%/yr over 2005-2014 against 7.62% for SPY, and 15.83% against 13.02%
over 2015-2024 (`data_load.log`). The report discusses which way this bias pushes each signal.

**Data fix.** Yahoo's DHR series is mis-adjusted for the Fortive spin-off: +61% on 2016-07-05. I set that one return to 0.
Every other move larger than 20% was checked against known events and left alone.

## Periods
- Returns from 2005-01-03 to 2024-12-31 (5033 days).
- **Design period:** strategy returns dated 2005-01-03 to 2014-12-31. Signals need 60 days of warm-up, and 12-1
  signals need 252 days, so each strategy's design returns start when its signal first exists.
- **Test period:** strategy returns dated 2015-01-02 to 2024-12-31. It is evaluated once, with signs frozen from the
  design period.

## Graph and residuals (the original construction from `/home/user/TDA/audit/check_original.py`)
For each day i (with x_i the vector of day-i simple returns):
- C = Pearson correlation of daily returns over days i-60 .. i-1 (60 days, excluding day i).
- A_jk = |C_jk| if |C_jk| > 0.3, else 0, with a zero diagonal. deg = row sums.
- L = I - D^-1/2 A D^-1/2, using D^-1/2 = diag(1/sqrt(deg + 1e-8)).
- **LR (Laplacian residual):** e_i = x_i - (I - 0.5 L)^3 x_i. T = 3 is an integer, so this is the matrix cube, which is
  identical to the original `fractional_matrix_power`. The code checks this numerically.
- **NR (neighbour residual):** n_i,j = x_i,j - sum_k A_jk x_i,k / sum_k A_jk. For a stock with no neighbours (deg = 0), it is
  x_i,j minus the cross-sectional mean of x_i.

## Signals, all known at the close of day t
| id | signal | rebalance |
|---|---|---|
| LR1, NR1 | residual of day t | daily |
| LR5, NR5 | sum of residuals over days t-4..t | weekly (last trading day of each calendar week) |
| LR21, NR21 | sum of residuals over days t-20..t | monthly (last trading day of month) |
| LR12_1, NR12_1 | sum of residuals over days t-251..t-21 | monthly |
| SP21 | spillover: sum_k A_jk R21_k / sum_k A_jk, where R21 = compounded return over t-20..t and A is the graph for day t+1 (correlations over t-59..t) | monthly |
| SP12_1 | the same, using the compounded return over t-251..t-21 | monthly |
| MOM12_1 | own compounded return t-251..t-21 | monthly |
| REV5 | own compounded return t-4..t | weekly |
| REV21 | own compounded return t-20..t | monthly |
| IREV5 | own 5-day return minus the equal-weight 5-day return of the other stocks in the same GICS sector (universe mean for the one-stock sectors, Real Estate and Materials) | weekly |

The spillover signal uses the same |rho| > 0.3 graph. The report gives the share of negative edges, which is expected to be about 0.

## Portfolio
- At each rebalance close t, rank the signal times its sign. The long leg is the top quintile (16 of 83 stocks), equally
  weighted to sum to +1. The short leg is the bottom quintile, weighted to sum to -1. There is no leverage beyond this
  gross exposure of 2 per unit of capital.
- Positions earn returns from t+1. Between rebalances the weights drift: w <- w(1+r)/(1+r_p). The code asserts that every
  daily return is earned on weights set at an earlier close.
- **Costs:** 5 bp one-way times sum |w_target - w_drifted| at each rebalance, including the initial entry. The daily
  net return is the gross return minus the cost booked on that rebalance day's return.
- **Sign selection:** sign = +1 (continuation) if the sign=+1 long-short has a positive design-period **gross** mean,
  else -1 (reversal). Costs are identical for both signs.
- **Long-only variant,** for an account that may not be able to short: the long leg alone (weights sum to 1), compared
  with an equal-weight portfolio of all 83 stocks rebalanced on the same schedule at the same cost.

## Metrics, per strategy and period
- Annualised mean gross and net return. Sharpe = mean/sd x sqrt(252), with no RF subtracted for long-short. For
  long-only, subtract Ken French daily RF.
- Newey-West t-stat of the mean daily net and gross return (10 lags).
- 95% CI for the net Sharpe: circular block bootstrap, 21-day blocks, 2000 resamples, seed 12345.
- Turnover per year (sum |dw|, one-way), cost drag per year, and **break-even cost** in bp = mean gross daily return /
  mean daily turnover.
- Maximum drawdown of net returns.
- FF5 + UMD alpha: daily regression of net returns on Mkt-RF, SMB, HML, RMW, CMA and Mom, with no RF subtracted for
  long-short, NW 10 lags. Reported for every test-period strategy.
- Long-only: excess return over the EW benchmark, with a NW t-stat of the paired daily difference and a block-bootstrap
  CI for the difference in Sharpe.
- **Multiple testing:** 14 test-period long-short strategies. Holm-Bonferroni on the NW p-values of the net mean. A
  "survivor" has a test-period net mean > 0 with a Holm-adjusted p < 0.05.

## Pre-listed robustness checks (run after the single test evaluation)
- R1: costs of 0, 2, 5 and 10 bp.
- R2: test sub-periods 2015-2019 and 2020-2024.
- R3: the original 20-stock universe, restricted to the 18 names with data from 2005 (META and TSLA drop out),
  quintile = 4 stocks, with signs frozen from the main design period.
- R4: implementation lag of one extra day (trade at the close of t+1) for the daily and weekly signals.
- R5: Fama-MacBeth regressions of the next-month return on SP21 and SP12_1, controlling for own REV21 and MOM12_1
  (design and test separately, NW t-stats on the monthly slope series).
- Any variant not listed here will be reported as an addition, with a count.

## Addendum 1: written after the single test evaluation (test.log), before any robustness run
The test produced no survivors. These post-hoc additions are exploratory, are counted as extra variants in the report,
and cannot turn a failed test into a pass.
- A1 (survivorship-free check of the long-horizon signals): LR12_1, NR12_1 and MOM12_1 on the Ken French **49 industry
  portfolios** (daily, value-weighted), using the same graph and residual construction. An industry is dropped on a day
  when its return is missing. Quintile = 10 of 49, with signs frozen from the stock design period (+1, +1, +1). Cost is
  10 bp one-way. Evaluated on 1927-2004 (pre-sample) and on 2005-2014 and 2015-2024.
- A2 (is the residual 12-1 signal just momentum?): the correlation of the LR12_1 and MOM12_1 long-short daily net returns
  in the test period, and a regression of LR12_1 on MOM12_1 (NW).
- A3: long-only LR12_1 and MOM12_1 with the 5 largest contributors to test-period excess return removed from the
  universe, to show how concentrated any excess return is.
