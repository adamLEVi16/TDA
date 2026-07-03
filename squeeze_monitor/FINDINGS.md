# Squeeze Early-Warning Monitor — Findings

A **risk tool for a consumer/retail short book**, not an alpha strategy: among
names that are already crowded shorts, does an abnormal public-attention spike
warn that the right tail (a squeeze) is coming? Free data (FINRA consolidated
short interest, Wikimedia daily pageviews, Yahoo prices), fully reproducible
(`short_interest.py`, `monitor.py`, `demo.py` → `demo.html`).

## Design discipline

- **Point-in-time short interest**: FINRA publishes ~9 business days after
  settlement; the monitor only uses a report from its publication date onward.
- **Pre-committed rule (V1)**, fixed before evaluation: CROWDED = published
  days-to-cover ≥ 4; SPIKE = weekly Wikipedia views ≥ 2× trailing 8-week
  median; FLAG = both.
- **Survivorship acknowledged**: delisted squeeze names (BBBY, EXPR) can't be
  fetched with free price data.
- **Tradable-universe discipline**: GME and AMC are excluded from the
  headline demo, the significance test, and the episode charts — both are
  banned at the target fund. They appear only in the V2 mechanical-flaw
  discussion below, where they were the historical case that surfaced the
  issue (not a claim this fund could trade them).

## V1 result (the pre-committed test): the tail effect is real

**Ex-banned (28 tradable names, the number the demo leads with)**, 2018–2024:
a flagged crowded short is **2.2× as likely to rip ≥ +25% in the next 4
weeks** (12.0% vs 5.5%; week-clustered bootstrap one-sided **p = 0.032**).

**Full 30-name universe incl. GME/AMC** (for reference / comparability to
earlier rounds), 10,049 name-weeks (4,216 crowded):

| group | n | mean 4-wk | P(≥+15%) | P(≥+25%) |
|---|---|---|---|---|
| **crowded + spike (FLAG)** | 91 | +3.9% | 20.9% | **14.3%** |
| crowded, no spike | 4,125 | +1.7% | 15.0% | **6.1%** |
| not crowded (context) | 5,833 | +2.3% | 11.4% | 4.8% |

Full-universe: 2.3× as likely to rip ≥+25% (ΔP = +8.2%, 95% CI
[+0.7%, +16.0%], p = 0.015). The ex-banned result is weaker (as expected,
since GME/AMC were unusually extreme contributors) but still holds at
p < 0.05. Alarm budget ≈ 13/yr across 30 names — actionable, not noisy.

## The V1 episode check exposed a real mechanical flaw

Testing V1 against the historical GME squeeze (excluded from the tradable
demo, but useful as a stress case since it's the most extreme squeeze on
record) showed it was **never flagged** — because FINRA days-to-cover
*collapsed* (6.1 → 2.1) during the squeeze while short interest was still ~61M
shares: the volume denominator exploded. **Days-to-cover mechanically fails at
exactly the moment it matters.** Weekly Friday sampling also lost the intra-week
attention explosion (Jan 25–26).

## V2 (post-hoc engineering iteration, labeled as such)

Volume-independent crowding (SI level ≥ 75th percentile of its own trailing
2 years of *published* reports) + daily attention timing (views ≥ 3× trailing
56-day median). Validated on the GME/AMC stress cases, then applied to the
tradable universe:

- **GME stress case: first daily flag 2020-11-12** (views 3.2×, SI at 85th+
  pctile) — 76 days before the Jan-27 peak close. **AMC stress case: first
  daily flag 2021-05-11** (views 10×) — 22 days before the Jun-2 peak close.
- **Tradable-universe episodes (the ones in demo.html):** CVNA's own trailing-
  SI percentile stays moderate (~0.51) through its Nov-2023 run — an artifact
  of its 2022 collapse-era SI dominating its own 2-year baseline — so V2's
  daily test doesn't fire there; the weekly V1 rule does (flagged 2023-11-03,
  price +85% by the Dec-19 peak, 46 days). ETSY: flagged 2022-07-01,
  +51% by the Aug-16 peak, also 46 days.
- **Misses, stated**: BYND-2019 (IPO'd too recently for an SI baseline) and
  CVNA's *separate*, earlier July-2023 move (~+50% in two weeks, no attention
  spike — a different episode from the flagged November one).
- Honest caveat: V2's *panel-level* weekly tail difference is **not**
  significant (p ≈ 0.20) — the population-level statistical evidence rests on
  V1 (p = 0.032 ex-banned / 0.015 full universe); V2's value is episode
  timing where it fires. Both stated, neither oversold.

## What it is / isn't

- **Is**: a free, reproducible screen that says "this crowded short is
  attracting abnormal public attention — size down / step aside," with
  significant population-level tail evidence (V1) and vivid canonical-episode
  timing (V2).
- **Isn't**: a return predictor or a strategy. It flags danger, not direction.
- **Upgrade path with desk data**: daily borrow rate / utilization (instead of
  twice-monthly lagged FINRA), options flow, and the fund's own position data
  would sharpen both legs materially — this is exactly the kind of free-data
  prototype that gets better with paid inputs.

## Reproduce
```bash
cd squeeze_monitor
python3 short_interest.py  # FINRA short interest, all names (free, no key)
python3 monitor.py         # V1 pre-committed evaluation + V2 iteration
python3 demo.py            # one-page visual demo -> demo.html
```
