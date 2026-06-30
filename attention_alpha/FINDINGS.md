# Attention → Consumer-Stock Signal — Findings

Goal: find "new info the market is slow to price." Test whether abnormal PUBLIC
ATTENTION to a consumer brand (free Wikipedia pageviews) predicts that brand's
own future stock return. Free data only (Wikimedia REST API + Yahoo prices),
2015–2024, fully reproducible (`wiki_data.py`, `test_signal.py`,
`torture_test.py`).

## Design (committed before running anything)

- **Universe**: 25 consumer-discretionary single names (restaurants, apparel,
  specialty/online retail) — the kind a retail long/short PM actually trades.
  In `universe.py` before any regression. One (ELF) 404'd on the API and was
  excluded, not swapped → 24 names.
- **Signal (no look-ahead)**: ASVI = log(weekly pageviews) − log(median of the
  prior 8 weeks). The week's views are observed by Friday; the median is
  strictly prior weeks. Z-scored *across tickers within each week* to isolate
  the cross-sectional effect from market-wide attention waves.
- **Pre-committed hypotheses** (Da, Engelberg & Gao 2011, "In Search of
  Attention"): H1 attention spike → positive next-week return; H2 it reverses
  over the next month.
- **Tests, in order of how much they matter**: panel regression (cluster-robust
  SE by week) → time-shift placebo → **tradable long/short basket** (decisive)
  → full torture suite (costs, momentum confound, sub-periods, drop-one,
  liquidity, holding period).

## Results — 24 tickers, 2015–2024, 488 weeks, 11,305 ticker-weeks

### The signal is REAL (this is the genuine, positive part)

| Test | Result |
|---|---|
| Forward reg: attention_t → return_{t+1} | coef +0.0013, **t = +2.41**, cluster-robust |
| Time-shift placebo | **p = 0.039** (real link beats shuffled null) |
| **Not just price momentum** — orthogonal to own return_t | coef +0.0013, **t = +2.48, p = 0.013** |
| Not a 2020–21 meme artifact | meme era *hurt* it; works in 2015–19 (+0.35) and 2022–24 (+0.32) |

Abnormal Wikipedia attention to a consumer brand genuinely carries information
about its next-week return, **incremental to and independent of recent price
moves** (the obvious confound — stocks that popped get looked up — is
controlled for and the signal survives). This is a legitimate, placebo-
confirmed informational edge. Unlike the weather test (`../weather_alpha/`,
a pure null), there is a real effect here.

### …but it is NOT systematically tradable after costs (the honest wall)

| Cost (bps/side) | 0 | 5 | 10 | 20 |
|---|---|---|---|---|
| Net Sharpe | **+0.48** | +0.29 | **+0.09** | −0.30 |
| Net ann. return | +12.2% | +7.3% | +2.3% | −7.5% |

- The effect is **short-lived — it decays within ~1 week** (the 4-week-ahead
  regression is null, p=0.47). Capturing it therefore requires **weekly
  rotation = very high turnover** (~1.9× the book/week).
- **Holding longer doesn't rescue it**: hold 1/2/4/8 wks → net Sharpe
  0.09/−0.08/0.10/−0.25. The signal dilutes faster than turnover falls.
- On **liquid large-caps only** (where you could trade in real size), it's
  negative net of 10 bps (Sharpe −0.17). The gross edge lives partly in the
  smaller, costlier-to-trade names.

## Honest verdict

**Real signal, real cost wall.** This is "new info the market underuses" —
attention data is genuinely predictive and momentum-independent — but the
mispricing is too small and too fast-decaying to harvest as a standalone
systematic weekly long/short once you pay to trade it. That is itself why it
persists: it's not worth a quant fund's transaction costs to arbitrage away.

**What it is good for:** a free, real-time **attention monitor** for names a
discretionary PM already follows. When a consumer brand shows an abnormal
attention spike, that has historically preceded a small positive next-week
drift, independent of price action — a supplementary confirmation input that
complements paid spend data (credit-card panels measure *actual* spend;
attention measures *interest/intent* and slightly leads). **It is NOT a
money-making strategy on its own, and is not presented as one.**

## What would raise/lower conviction next

- **+** Combine attention with the fundamental channel: cumulative attention
  over a fiscal quarter vs. the stock's move *on its next earnings date*
  (needs a historical earnings calendar). Attention as an SSS nowcast, priced
  at earnings, is a slower/larger effect than the 1-week pop — and far less
  turnover-sensitive.
- **+** Multi-source attention (Wikipedia + Google Trends + app ranks) to cut
  noise per name.
- **−** Honest risk: 24 names is a thin cross-section; the standalone-trade
  result is already dead, so only the *monitor* use survives scrutiny.

## Follow-up — the earnings-date channel (the slow, low-turnover version): also NULL
`earnings_data.py`, `earnings_study.py`

The weekly signal dies on turnover; the proposed fix was to test the *fundamental*
channel instead — attention as a same-store-sales nowcast, priced at the next
earnings report (a 2-day hold ~4×/yr, so costs are negligible). Built a real
earnings calendar from **SEC EDGAR** (8-K Item 2.02 filing dates — the actual
earnings releases), 550 events across 22 names, 2016–2024.

Does cumulative pre-earnings attention predict the announcement reaction?

| Signal | → 2-day reaction | → drift (+2..+21d) |
|---|---|---|
| attention **level** (mean abnormal attention over the quarter) | t=−0.69, p=0.49 | t=−0.03, p=0.97 |
| attention **YoY growth** (this quarter vs same quarter last year) | t=+0.07, p=0.94 | t=−0.83, p=0.41 |

Placebo confirms (p=0.47). Tradable long/short-into-earnings: Sharpe +0.21,
**p=0.26 — not significant**.

**Null on both specifications.** The earnings-date version clears the cost wall,
but there is no signal there to harvest: by the time a consumer name reports,
analyst consensus has already absorbed whatever demand information sits in its
public attention. So the *only* real attention effect is the 1-week pop — and
that one is precluded by transaction costs. The honest end of the arc: free
public-attention data contains a real but un-monetizable short-horizon edge and
no detectable edge at the fundamental/earnings horizon.

## Reproduce
```bash
cd attention_alpha
python3 wiki_data.py      # fetch/cache pageviews (free, no key), 2015-2024
python3 test_signal.py    # weekly: panel + placebo + basket
python3 torture_test.py   # weekly: costs, momentum control, sub-periods, drop-one, holding
python3 earnings_data.py  # earnings calendar from SEC EDGAR (free, no key)
python3 earnings_study.py # earnings-date event study (level + YoY growth)
```
