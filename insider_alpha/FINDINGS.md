# Insider Cluster-Buy Study — Findings (null)

Test of the most persistent documented insider anomaly (cluster buying — ≥2
distinct insiders making open-market purchases near-simultaneously) on a
pre-committed ~110-name consumer universe, 2014–2024. Data: SEC's structured
quarterly Insider Transactions datasets (free), point-in-time via filing dates.
1,454 open-market purchases → 124 cluster events (112 with full price data).

| Test | Result |
|---|---|
| Event study, +63td market-adjusted | **+0.00%, t=0.00** (+21td t=0.65, +126td t=0.82) |
| Ticker-preserving date-shift placebo | p=0.87 |
| Calendar-time long portfolio (63td hold, 10bps) | CAPM alpha +1.2%/yr, **t=0.13**; MaxDD −68% |
| Conviction cuts (≥$500k, ≥3 insiders) | means positive, medians **negative**, hit ≤44% |

**Verdict: null.** The literature effect (built mostly on broad universes with
small caps and pre-2010 data) does not appear in liquid consumer names
2014–2024 at this sample size. Consistent with this project's running theme:
public-data anomalies are dead where institutional money can easily trade.

Reproduce: `python3 download.py && python3 study.py`
