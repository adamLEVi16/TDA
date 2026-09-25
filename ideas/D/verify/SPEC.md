# SPEC — Concept D: variance-risk-premium (VRP) timing of the US market

Pre-registered 2026-09-25, before any design-period or test-period performance number was computed.
The only data looked at so far is `check_availability.py` / `check_availability.log`, which gives date ranges only.

## Hypotheses
- H1 (forecasting). The monthly VRP predicts the next-month (h=1) and next-3-month (h=3) US market excess return
  **out of sample**. Metric: Campbell-Thompson OOS R^2 against the expanding historical-mean forecast. Test: one-sided
  Clark-West MSPE-adjusted test with Newey-West SE (lags = h). Success means OOS R^2 > 0 and CW p < 0.05.
- H2 (trading). A long-only SPY/T-bill rule that scales SPY exposure with the VRP forecast (S1) beats SPY buy-and-hold
  after costs. Metric: difference in annualised Sharpe ratio (excess of RF). Test: paired stationary block bootstrap
  (mean block length 6 months, 10,000 draws, seed 20260925), two-sided 95% CI and one-sided p. The difference in mean
  excess return is reported the same way.
- H3 (value added by VRP). S1 beats the identical rule that uses the historical-mean forecast (S0). This separates VRP
  information from generic mean-variance scaling. Same test as H2.

## Data (all real, cached to `cache/`)
- ^VIX daily close from yfinance, 1990-01-02 on. This is the current CBOE methodology back-filled to 1990.
- Ken French `F-F_Research_Data_Factors_daily` (Mkt-RF, RF, daily) and `F-F_Research_Data_Factors` (monthly), through 2026-07.
- SPY (yfinance, auto_adjust=True, so dividends are reinvested) from 1993-01-29.
- Tail-risk illustrations only, not traded or tested: SVXY (2011+), ^PUT (CBOE S&P 500 PutWrite index, 1996+),
  VXX (2018+). XIV is not available on yfinance (delisted), so it is not used.

## Signal definitions (monthly, sampled at the last trading day of month t)
- Daily market total return: r_d = (Mkt-RF)_d + RF_d from the French daily file.
- Realised variance: RV_t = sum over the trading days d in calendar month t of r_d^2 (monthly variance units).
- Implied variance: IV_t = (VIX_t / 100)^2 / 12, using the VIX close on the last trading day of month t.
- **VRP_t = IV_t − RV_t.** This follows the Bollerslev-Tauchen-Zhou construction, with daily rather than 5-minute
  returns for RV.
- Robustness only (not primary): RV from SPY daily log returns over the trailing 21 trading days (RV21), and VIX-level alone
  (IV_t) as the predictor.

## Forecasting targets
- h=1: y_{t+1} = monthly Mkt-RF for month t+1 (French monthly).
- h=3: y_{t+1:t+3} = sum of monthly Mkt-RF over months t+1..t+3. Overlapping targets use NW lags = 3.
- The regression is y = a + b·VRP_t + e, fitted by OLS on an expanding window that starts with the forecast for 1990-02.
  At forecast origin t, only pairs whose target is fully observed by t are used (s ≤ t−h).
- Benchmark forecast: the mean of y over the same expanding window.
- Also reported: a Campbell-Thompson restricted variant (forecast truncated at 0). It is a secondary result.
- The in-sample full-period regression is reported with NW t-stats and **labelled in-sample**.

## Periods
- Design period: forecast origins 1992-01 … 1999-12 (targets 1992-02 … 2000-01), with a 24-month minimum estimation
  window from 1990-02. The strategy design period is evaluated on the French market (Mkt = Mkt-RF + RF) because SPY begins
  1993-01.
- Test period: forecast origins 1999-12 … 2026-06, so the monthly returns earned are 2000-01 … 2026-07 (h=1).
  For h=3, the last origin is 2026-04. The test period is run **once**, after the parameters below are frozen.

## Trading rules (monthly rebalance at the month-end close; weights set at close t earn month t+1 return)
Assets: SPY (monthly total return from adjusted closes) and T-bills (French monthly RF). The weight on SPY is w_t ∈ [0,1].
There is no leverage and no shorting (small retail account).
- **S1 (primary):** w_t = clip( μ̂_t / (γ · σ̂²_t), 0, 1 ), with μ̂_t the OOS VRP forecast (h=1), σ̂²_t the sample
  variance of monthly Mkt-RF over the prior 60 months (rolling), and γ = 3. γ=3 is a literature convention and is fixed,
  not tuned.
- **S0 (control):** as S1 but μ̂_t = historical-mean forecast.
- **S2 (threshold):** w_t = 0 if VRP_t is below the q-quantile of its own expanding history (data ≤ t, ≥ 24 obs), else 1.
  **q is chosen on the design period only** from {0.10, 0.20, 0.30}, by the highest design-period Sharpe.
- Baselines (standard parameters, not tuned):
  - B1: SPY buy-and-hold (w=1).
  - B2: 10-month SMA trend. w=1 if the SPY price at close t is above its 10-month average of month-end closes, else 0.
    French Mkt is used in the design period.
  - B3: volatility target. w = min(1, 0.15 / sqrt(12·RV_t)).
  - B4: VIX-level threshold, the same as S2 with IV_t in place of VRP_t and the same q. This checks whether VRP adds
    anything beyond VIX.
- Timing check in code: assert that every weight index is strictly earlier than the return month it is applied to.

## Costs
- SPY cost = 2 bp one-way × |w_target,t − w_drift,t|. The drifted weight comes from the weight set at t−1 and the month's
  SPY and T-bill returns. The cost is charged at every rebalance, including drift. The initial purchase is also charged.
- Also reported at 10 bp. Break-even cost = the one-way cost at which the strategy's mean excess return equals B1's.
  A separate break-even is the cost at which the strategy's own mean excess return is 0.

## Metrics (test period, monthly)
- Annualised mean excess return (over RF), volatility and Sharpe (mean/sd·√12). The Sharpe t-stat uses the Lo (2002)
  NW-adjusted SE (6 lags), and the mean t-stat uses NW with 6 lags.
- Max drawdown of the total-return wealth path, average |Δw| turnover per year, average exposure.
- Difference tests vs B1 and S1 vs S0: paired stationary block bootstrap, as in H2.

## Robustness (run after the single test run; all listed here in advance)
R1 sub-periods 2000-2012 and 2013-2026-07; R2 RV21 from SPY; R3 cost 10 bp; R4 excluding 2008-09..2009-06;
R5 VIX-only predictor (OOS R^2); R6 CT sign-restricted forecasts.

## Tail-risk illustration (descriptive, not a strategy test)
- The monthly "short variance swap" proxy payoff is IV_t − RV_{t+1} (implied minus next-month realised variance) per unit
  variance notional. Report its mean, t-stat, skew and worst 5 months. This approximates a VIX-struck variance swap:
  the VIX² is close to the 30-day variance-swap rate, and the approximation is stated as such.
- Report the max drawdown and the worst days of SVXY, ^PUT and VXX over their available history, including 2018-02-05
  and 2020-03.

## Variant accounting
Every variant run will be counted in the final report. Anything not in this file that is added later will be listed as
post-hoc.
