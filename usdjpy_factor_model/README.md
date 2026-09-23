# USD/JPY short-term factor study

Tests whether rates, carry, momentum, risk and positioning factors predict next-week USD/JPY returns.
Everything uses free data (FRED, Japan MoF JGB yields, CFTC COT), so no API keys are needed.

```
pip install pandas numpy statsmodels scikit-learn scipy requests
python run_tests.py --download   # first run; later runs can skip --download
```

## Setup

- Target: log change in USD/JPY (NY noon fix) from week t to t+1, 1990–2026 (1,915 weeks). "spot+carry" adds the 3m UST − 1y JGB carry.
- Every input is taken as of the last close strictly before the FX print. COT is used only from the Friday after its release.
- Significance: Newey-West t-stats. Walk-forward models train from 1990, test 2000–2026, refit quarterly, and pay 1bp per unit traded.

## Results

| Factor (known at week t) | t, 1990–2026 | t, 1990–2007 | t, 2008–2026 | t, 2016–2026 |
|---|---|---|---|---|
| Δ(US2y − JGB2y), last week | 3.40 | 2.42 | 2.47 | 1.04 |
| Δ(US10y − JGB10y), last week | 3.50 | 2.77 | 2.21 | 1.38 |
| Δ(US10y − JGB10y), last 4 weeks | 3.33 | 3.55 | 0.96 | 0.82 |
| Carry (3m UST − 1y JGB) | 1.39 | 2.27 | 0.17 | −0.06 |
| 12-week momentum | 1.32 | 0.92 | 0.87 | 0.48 |
| VIX change | −1.61 | −0.77 | −1.49 | −1.12 |
| FX-vs-rates misalignment | −0.03 | 0.61 | −0.67 | −0.51 |
| CFTC spec positioning (z) | −0.86 | −0.98 | −0.41 | −0.11 |

Oil, Nikkei, JGB 10y changes and 1-week reversal also show nothing (|t| < 1.7 in the full sample).

1. **Rate-spread momentum is the only factor that survives a multiple-testing bar (t > 3).** A widening US–Japan spread last week predicts USD/JPY gains next week, but it explains under 1% of the variance and has faded since 2016.
2. **The Friday-anchored weekly test overstates it.** When the same signal is traded as five staggered weekly positions, one opened each weekday, the Sharpe ratio is 0.42 for 1990–2007, 0.10 for 2008–2026 and −0.09 for 2016–2026.
3. **Multi-factor models do not beat holding USD/JPY long.** OLS, ridge, gradient boosting and a fixed-sign composite have OOS R² between −3.1% and +1.5%. From 2013 to 2026 every model has a lower Sharpe ratio than always long spot+carry (0.65).
4. **Rates explain USD/JPY in the same week but do not predict it.** Same-week changes in the US 10y yield explain 25% of weekly USD/JPY variance since 2008. Using that link to trade requires forecasting US yields.
5. **Daily horizon:** the prior day's rate, VIX and USD/JPY moves do not predict the next day's return (|t| < 1).

## Not tested (needs paid data)

- USD/JPY 1m implied vol and 25-delta risk reversals, as a sentiment and crash-risk factor
- Economic-release surprises (US CPI, NFP) and BoJ/Fed meeting days, as event factors
- JPY cross-currency basis and Japanese life-insurer hedge-ratio flows
- Intraday data, needed for any horizon shorter than a day
