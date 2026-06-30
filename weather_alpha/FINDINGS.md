# Weather → Retail-Stock Signal — Findings

Goal: test whether local weather anomalies at a retailer's HQ metro contain
information about its *future* stock return that the market hasn't priced in
yet — i.e. genuine "new info," not a repackaging of price/volume data the
whole market already sees. Free data only (Open-Meteo ERA5 reanalysis,
1995–2024; Yahoo Finance prices), fully reproducible (`weather_data.py`,
`anomaly.py`, `test_signal.py`).

## Design (committed before running anything)

- **Universe**: 14 weather-plausible consumer/retail/restaurant names, HQ-city
  used as a (acknowledged-imperfect) proxy for regional exposure: HD, LOW,
  CBRL, TXRH, EAT, DRI, CAKE, WEN, UAA, SBUX, JWN, LULU, COLM, KSS.
  Written into `universe.py` *before* any regression — no picking winners
  after peeking.
- **Signal**: weekly weather anomaly vs a trailing 10-year same-week
  climatology (no look-ahead — a week's "normal" only ever uses strictly
  prior years).
- **Two pre-committed hypotheses**:
  - H1 (precipitation): more rain than normal → worse foot traffic →
    negative return, possibly lagged a week if the market doesn't track local
    weather in real time.
  - H2 (temperature *extremity*, i.e. `|deviation|`): too hot or too cold →
    worse foot traffic → negative return, same lag logic.
- **Tests**: cluster-robust panel regression (same-week, and forward
  next-week), a time-shift placebo (circularly shifts each ticker's weather
  series to build an honest empirical null), and — the decisive one — an
  actual tradable cross-sectional long/short basket built directly on the
  same score.

## Results

JWN (Nordstrom) dropped — ticker now 404s on Yahoo (went private 2025).
13 tickers, 1999–2024, 1,351 weeks, 14,835 ticker-weeks.

| Test | panom_z (rain) | tanom_z (signed temp) | textreme_z (\|temp dev\|, the literal H2) |
|---|---|---|---|
| Same-week, p | 0.59 | 0.075 | 0.61 |
| Next-week, p (cluster SE) | 0.31 | **0.051** | 0.18 |
| Next-week, p (time-shift placebo) | 0.27 | **0.007** | 0.06 (wrong sign vs H2) |
| **Basket long/short Sharpe** | **−0.10** | **−0.25** | **−0.07** |

## Honest verdict: null result

- **H1 (rain) is dead on arrival.** No effect at any horizon, any test (all
  p ≥ 0.27). Precipitation at a retailer's HQ does not predict its return.
- **H2 as literally specified (extremity) also fails**, and where it's
  borderline (placebo p=0.06) the sign is *backwards* from the hypothesis —
  that's not even a near-miss, the mechanism doesn't behave as theorized.
- **The one number that looks interesting — signed temperature anomaly,
  next-week, placebo p=0.007 — does NOT survive the most important test: an
  actual long/short basket built on exactly that score loses money**
  (Sharpe −0.25, 49% hit rate, indistinguishable from a coin flip with a
  negative drift). A "significant" pooled-panel coefficient that can't be
  turned into a positive-Sharpe cross-sectional sort is not a tradable
  signal — it's most likely a handful of autocorrelated weeks or one or two
  tickers driving the regression, not a clean monotonic relationship. This
  is the same failure mode this project caught before (multi-asset trend's
  short-sample Sharpe edge looked real until torture-tested) — caught here
  *before* anyone got excited about it.

**Bottom line: this specific design — HQ-city weather, weekly resolution,
1-week lag, 14 tickers — finds no exploitable signal.** This is a real,
useful negative result, not a failure of effort.

## Why it most likely failed (not "weather doesn't matter")

1. **HQ-city is a weak proxy for store footprint.** Home Depot's stores
   aren't concentrated in Atlanta; "Atlanta weather" barely correlates with
   HD's actual nationwide sales-weather exposure. A real version needs
   store-count-weighted regional weather, which requires data (store
   locations) we don't have for free.
2. **Wrong horizon.** The real economic channel is: bad weather → same-store-
   sales miss → investors don't find out until the *next earnings report*,
   weeks or months later — not next week. A weekly lag test is structurally
   too short to catch that. The correct test is an event study: cumulative
   weather anomaly over the trailing fiscal quarter vs. the stock's return
   *on its next earnings date*. That needs a historical earnings-date
   calendar, which is harder to source reliably for free and was not yet
   built.
3. By 2024, anyone trading retail names professionally (including Alan,
   who pays for weather reports) already adjusts for obvious weather shocks
   manually — a *weekly*, market-wide lag is exactly the kind of slow,
   easy-to-arbitrage mispricing that gets competed away first.

## Reproduce
```bash
cd weather_alpha
python3 weather_data.py   # fetch/cache 12 metros, 1995-2024 (Open-Meteo, free, no key)
python3 test_signal.py    # builds panel, runs all tests above
```
