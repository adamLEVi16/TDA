# Pre-registration: "When is the equity premium earned?" (calendar-timed index exposure)

Written before any post-publication (out-of-sample) returns were computed.

## Thesis

Published research says most of the US equity premium is earned on a small set of days that can be known in advance:

1. **FOMC-cycle even weeks** (Cieslak, Morse & Vissing-Jorgensen 2019, JF 74(5):2201–2248, doi:10.1111/jofi.12818).
   - Claim: over 1994–2016, the premium was "earned entirely in weeks 0, 2, 4 and 6 in FOMC cycle time".
   - In-sample strategy B (hold stocks in even weeks only) had Sharpe 0.92 vs 0.45 for always holding.
2. **Pre-FOMC announcement drift** (Lucca & Moench 2015, JF 70(1):329–371, doi:10.1111/jofi.12196).
   - Claim: about 0.49% excess return in the 24 hours before scheduled announcements, Sept 1994 – Mar 2011.
   - Prior evidence against: Kurov, Wolfe & Gilbert (2021), Finance Research Letters 40:101781, doi:10.1016/j.frl.2020.101781, "The disappearing pre-FOMC announcement drift". CMVJ's Appendix Table 1 also says the Lucca–Moench result is absent after publication.
3. **Turn of the month** (McConnell & Xu 2008, FAJ 64(2):49–64, doi:10.2469/faj.v64.n2.11).
   - Claim: over 1926/1987–2005, all of the market's excess return occurred from the last trading day of the month through the first three trading days of the next.
   - Mechanism: month-end institutional cash needs (Etula, Rinne, Suominen & Vaittinen 2020, RFS 33(1):75–111, doi:10.1093/rfs/hhz054).

Caution on calendar effects as a class: Sullivan, Timmermann & White (2001), J. Econometrics 105(1):249–286, doi:10.1016/s0304-4076(01)00077-x. McLean & Pontiff (2016, JF 71(1):5–32, doi:10.1111/jofi.12365) find that anomaly returns fall by roughly half after publication.

**Prior before testing: low.** The most likely outcome is post-publication decay. The test is informative whichever way it comes out, because the rules have no free parameters to tune.

## Data

- **Market:** Ken French daily factors (`F-F_Research_Data_Factors_daily`). Market total return = Mkt-RF + RF; cash earns RF. This is the same source CMVJ used.
- **Tradable check:** SPY adjusted close from yfinance (1994+). Cash earns Ken French RF.
- **FOMC dates:** scheduled meetings from federalreserve.gov. Pages `fomchistorical{1994..2020}.htm` and `fomccalendars.htm` (2021–2026).
  - Include headings containing "Meeting".
  - Exclude "Conference Call", "(unscheduled)" and "notation vote".
  - Keep "(cancelled)" scheduled meetings (March 17–18, 2020), because that was the published schedule an investor would have followed. Sensitivity check: drop it.
  - Day 0 = the last day of a multi-day meeting.
  - Merge entries on consecutive days into one meeting (e.g. 2003-09-15 and 09-16), keeping the later date.

## Definitions (fixed; no parameters are estimated)

- **FOMC cycle days** (CMVJ): count weekdays (Mon–Fri, holidays included, as in the paper) relative to the most recent day 0.
  - The weekday before the next scheduled day 0 is day −1 and belongs to week 0 of the next cycle.
  - Otherwise day d ≥ 0 counted from the last day 0 falls in week floor((d + 1) / 5). So week 0 = days −1..3, week 1 = 4..8, week 2 = 9..13, and so on.
  - Even weeks = weeks 0, 2, 4, 6, 8.
- **FOMC day** (proxy for Lucca–Moench): the close-to-close return on day 0.
  - This is an approximation. Their window is 2pm day −1 to 2pm day 0, which daily data cannot reproduce.
- **TOM:** the last trading day of each month plus the first three trading days of the next month, using trading days in the return data.
- **Strategies:** hold the market on window days and T-bills otherwise.
  - The decision for day t uses only the published calendar, so there is no look-ahead: FOMC calendars are published a year or more ahead, and month-ends are known.
  1. **EVEN:** hold in FOMC even weeks.
  2. **FOMC0:** hold on day 0 only.
  3. **TOM:** hold on turn-of-month days.
  4. **UNION:** hold on EVEN or TOM days.
- **Benchmark:** buy-and-hold market (BH).
- **Costs:** 1 bp one-way per unit of weight change (SPY half-spread is about 0.3 bp). Sensitivity at 5 bp.

## Periods

| Hypothesis | Replication (in-sample, original paper) | Out-of-sample (after the sample / publication) |
|---|---|---|
| EVEN | 1994-01-01 to 2016-12-31 | 2017-01-01 to latest |
| FOMC0 | 1994-09-01 to 2011-03-31 | 2011-04-01 to latest |
| TOM | 1926-07-01 to 2005-12-31 | 2006-01-01 to latest |
| UNION | 1994-01-01 to 2016-12-31 | 2017-01-01 to latest |

## Metrics (reported for every period)

- Mean daily excess return in-window vs out-of-window, with a Newey–West (5 lags) t-stat on the difference (OLS of excess return on a window dummy).
- Share of the total excess return earned in-window, and time in the market.
- Strategy excess-return Sharpe (RF subtracted), CAGR and max drawdown, vs BH on the same dates.
- Sharpe difference vs BH with a paired stationary block bootstrap: mean block 20 days, 5,000 draws, seed 0, two-sided p.
- Success criterion for "worth trading", fixed in advance: the out-of-sample Sharpe exceeds BH with bootstrap p < 0.05, and the result holds at 5 bp and on SPY.

## Amendment 1 (made after the first run; both versions reported)

The replication check did not match the paper's day counts per FOMC week:

| | wk 0 | wk 2 | wk 4 | wk 6 |
|---|---|---|---|---|
| First run | 920 | 919 | 909 | 445 |
| Paper | 920 | 924 | 831 | 120 |

The paper (p. 5) also defines week −1 as days −6 to −2 before the next meeting. Those days take precedence over counting forward from the last meeting.

- **Primary definition, changed to match the paper:** day −1 → week 0; days −6 to −2 → week −1 (odd); otherwise week = floor((d + 1) / 5).
- **Original definition:** kept and reported as "EVEN_v1".

I had already seen the first run's out-of-sample result for EVEN_v1 when I made this change, so both versions are shown. The TOM and FOMC0 definitions are unchanged.
