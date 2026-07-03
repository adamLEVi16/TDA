# Systematic Research Portfolio — Adam Levine

A short portfolio of self-contained quantitative studies. The point of this
collection is **method, not a single magic number**: each study takes a
hypothesis, tests it with no look-ahead, demands statistical significance,
charges realistic transaction costs, and reports honestly whether the idea
survives — including when it doesn't. Every result below is reproducible from
free data with the code in this repo (urllib-based fetchers, no paid feeds, no
API keys).

The three studies are deliberately a *spread of outcomes* — one win, one null,
one "real-but-not-tradable" — because that spread is the honest distribution of
real research, and distinguishing between those three cases is the actual skill.

---

## How each study is evaluated (the standard applied to all three)

- **No look-ahead.** Signals are computed from data available strictly *before*
  the return they predict (e.g. a 1-period implementation lag; climatologies and
  rolling medians use only prior periods).
- **Survivorship is controlled, not hidden.** Where a hand-picked universe is
  unavoidable, an equal-weight-universe benchmark isolates genuine selection
  skill from "these survivors went up."
- **Significance, not just point estimates.** Block-bootstrap confidence
  intervals on Sharpe differences; time-shift/placebo nulls built from the data
  itself; deflated-Sharpe haircuts for multiple testing.
- **Costs are modeled.** Turnover-based transaction costs, stress-tested up to
  levels that reflect real mid-cap execution — because a gross Sharpe that dies
  at 10 bps/side is not a strategy.
- **Confounds are tested directly.** e.g. "is this attention signal just price
  momentum?" is answered by orthogonalizing to recent returns, not assumed away.
- **The decisive test is a tradable portfolio.** A significant pooled-panel
  coefficient that can't be turned into a positive-Sharpe, cost-aware book is
  treated as *not a signal*.

---

## Study 1 — Multi-asset trend + risk parity  → a validated defensive edge
`equity_factor/` · `long_history.py`, `torture_test.py`, `tearsheet.py`

A long-only monthly GTAA: hold 8 liquid asset-class ETFs above their 10-month
trend, inverse-vol weighted, move the rest to T-bill cash. Validated out-of-
sample on dividend-adjusted fund proxies back to **1987 (451 months, 7+ crises,
incl. a rising-rate regime)**.

| | Strategy | US equity (buy & hold) | 60/40 |
|---|---|---|---|
| Sharpe (1987–2024) | **1.16** | 0.74 | 0.93 |
| Max drawdown | **−12.8%** | −51.0% | −29.9% |
| CAGR | 6.8% | 10.5% | 9.2% |
| CAPM alpha | **+1.8%/yr (t=2.30, p=0.022)** | — | — |

- Sharpe edge over buy-and-hold is **statistically significant** (ΔSharpe +0.43,
  95% CI [+0.13, +0.75], block-bootstrap **p = 0.002**), and it clears the formal
  alpha bar too: **CAPM alpha +1.8%/yr (t=2.30)**, confirmed on a 40-year, 4-fund
  extension through the Volcker era (**+2.3%/yr, t=3.08, p=0.002**) and a strict
  spanning test vs. its own 5 underlying assets (p=0.058) — the central result:
  a static-weight mix of the same assets is statistically rejected (p=0.025) as
  an explanation, so the monthly *timing* itself, not the diversification, is
  the return engine.
- Robust: survives 40 bps/side costs, execution 21 sessions late, trend window
  6–14 mo, drop any single asset, rebalance-date shifts; 5yr rolling alpha
  positive in 84% of windows; unaffected by correlation-regime spikes (the
  standard CTA objection). Verified deterministic, no leverage/shorting,
  hand-checked no-look-ahead.
- **Honest limitations, stated up front:** it does *not* out-*return* equities in
  bull markets (6.8% vs 10.5% CAGR); decade-by-decade alpha was ≈0 in 2017–24;
  it underperforms buy-and-hold in 69% of rolling 3-year windows (once for 159
  straight months) — likely *why* the premium persists, since few allocators can
  sit through that. Two-thirds of its cash-flight signals are false alarms that
  cost ~4.5pts vs staying invested; the entire edge rides on the minority that
  catch a real bear market. It is a **modest, real diversifying sleeve** — not a
  return engine, and not for 100% of a book.

**Takeaway:** I can find a genuine edge *and* subject it to the validation that
most backtests skip — and then classify it honestly (defensive beta, not alpha).

---

## Study 2 — Weather → retail-stock returns  → an honest null
`weather_alpha/` · `weather_data.py`, `test_signal.py`

Does a retailer's local weather anomaly (vs a no-look-ahead trailing
climatology) predict its own future return? 12 metros of free ERA5 weather
mapped to 14 weather-plausible retail/restaurant names, HQ-city as a footprint
proxy. **Pre-committed** hypotheses (rain hurts foot traffic; temperature
extremity hurts foot traffic).

- **Null.** Rain: nothing at any horizon (p ≥ 0.27). Temperature extremity:
  null / wrong-signed. The one borderline pooled coefficient (signed temp,
  placebo p=0.007) **fails the decisive test** — a long/short basket built on it
  *loses* money (Sharpe −0.25, ~coin-flip hit rate).
- Documented *why* it most likely failed (HQ-city ≠ store footprint; the real
  channel is a quarterly same-store-sales surprise priced at earnings, not a
  1-week lag) and what a better-designed test would require.

**Takeaway:** I kill my own ideas. A statistically borderline coefficient that
won't trade is reported as a null, not dressed up.

---

## Study 3 — Public attention → consumer stocks  → real signal, real cost wall
`attention_alpha/` · `wiki_data.py`, `test_signal.py`, `torture_test.py`

Does abnormal public attention (free Wikipedia pageviews, 2015–2024) to a
consumer brand lead its stock? 24 consumer-discretionary single names; ASVI
signal (log views − log trailing-8-week median), z-scored cross-sectionally.

**The signal is real:**

- Forward regression attention_t → return_{t+1}: **t = +2.41** (cluster-robust),
  time-shift placebo **p = 0.039**.
- **It is not just price momentum** — the obvious confound (stocks that pop get
  looked up) is controlled by orthogonalizing to the name's own recent return;
  the signal *survives*: **t = +2.48, p = 0.013**.
- Not a 2020–21 meme artifact — that era *hurt* it; it works in 2015–19 and
  2022–24.

**But it is not systematically tradable** — and saying so is the point:

| Cost (bps/side) | 0 | 5 | 10 | 20 |
|---|---|---|---|---|
| Net Sharpe | **+0.48** | +0.29 | **+0.09** | −0.30 |

The edge decays within ~1 week → forces weekly rotation → high turnover. Net of a
realistic 10 bps it's ~0.09, and **negative on the liquid large-caps you could
actually trade in size**. Holding longer doesn't help (signal dilutes faster
than turnover falls). That cost wall is *why* the inefficiency persists.

**Follow-up — the earnings-date channel (also tested, also reported):** since the
weekly edge dies on turnover, the natural fix is the *fundamental* channel —
attention as a same-store-sales nowcast, priced at the next earnings report (a
2-day hold ~4×/yr, so costs are negligible). I built a real earnings calendar
from **SEC EDGAR** (8-K Item 2.02 filing dates) — 550 events, 22 names,
2016–2024 — and tested whether cumulative pre-earnings attention predicts the
announcement reaction, in both *level* and *YoY-growth* specifications.
**Null on both** (reaction t = −0.69 and +0.07; placebo p = 0.47; tradable
long/short p = 0.26). The earnings version clears the cost wall but has no signal
to harvest — by report time, consensus has already absorbed the demand info.

**Takeaway:** the hardest and most valuable judgment in applied quant — the gap
between "statistically significant" and "tradable after costs," and the
discipline to chase the follow-up and report it null. The complete, honest arc:
free public-attention data holds a real but un-monetizable *short-horizon* edge
(killed by costs) and no detectable *fundamental-horizon* edge (killed by
consensus). Its one legitimate use is as a free, real-time *attention monitor*
feeding a discretionary, low-turnover process — explicitly not a standalone
strategy.

---

## Study 4 — PEAD in consumer names, conditioned on attention  → the classic is dead where it matters, and the harness proves it
`attention_alpha/` · `pead_study.py`, `xbrl_data.py`, `pead_diagnostics.py`

Replicate-then-extend on the most-documented anomaly in the literature:
post-earnings-announcement drift. 1,837 earnings events (SEC EDGAR 8-K dates),
25 consumer names, 2005–2024; surprise measured both as the 2-day market-adjusted
announcement reaction and as Bernard–Thomas SUE built from **point-in-time SEC
XBRL EPS** (earliest-filed values — no restatement look-ahead). Extension:
Hirshleifer's limited-attention prediction (drift stronger when announcement
attention is low), tested with the Wikipedia attention data from Study 3.

- Full sample: **no tradable drift** (calendar-time L/S, point-in-time tercile
  assignment, net of costs ≈ 0).
- **The era split recovers the published record**: the drift portfolio earned
  +7.2%/yr net (Sharpe +0.39) through 2016 — then went to **−13%/yr (2017–20)
  and −10%/yr (2021–24)**, with the SUE extremes mildly *reversing* post-2016
  (crowding now front-runs the announcement). This matches the documented
  post-2015 attenuation of PEAD in liquid names (e.g. Martineau 2021), found
  here independently on out-of-sample data and a different universe.
- The attention extension: null — low-attention events drift no more than
  high-attention ones in this universe.

**Takeaway:** the harness doesn't just fail to find false positives — it
*recovers known truths on both sides*: the anomaly where the literature says it
lived, and its death where the literature says it died. And the practical
conclusion is the portfolio's central thesis: in liquid US consumer equities,
the free-public-data edges are arbitraged out — the remaining edge is in
proprietary data (transaction panels, store-level signals), which is exactly
where a systematic harness like this one should be pointed.

## Study 5 — Squeeze early-warning monitor  → a working risk tool (the applied payoff)
`squeeze_monitor/` · `monitor.py`, `short_interest.py`, `demo.py` → `demo.html`

The applied synthesis of the portfolio: the one validated positive signal
(abnormal public attention, Study 3) repurposed where turnover costs don't
apply — as a **risk flag for a consumer/retail short book**. Crowding from
FINRA short interest used **only from its real publication date** (~9 business
days after settlement); attention from daily Wikipedia pageviews.

- **Pre-committed rule, evaluated on a 28-name tradable universe (2018–2024,
  GME/AMC excluded — both banned at the target fund): a flagged crowded short
  is 2.2× as likely to rip ≥ +25% in the next 4 weeks** (12.0% vs 5.5%;
  week-clustered bootstrap p = 0.032). ~13 flags/yr across the universe.
  Two live episodes: **CVNA flagged 2023-11-03, +85% by its 46-day-later
  peak; ETSY flagged 2022-07-01, +51% by its 46-day-later peak.**
- The GME/AMC squeezes were used only as historical *stress cases* during
  development (never as tradable claims) and surfaced a real mechanical
  insight: **days-to-cover fails during the squeeze itself** (GME Jan-2021:
  DTC "fell" to 2.1 because volume exploded while short interest was still
  ~61M shares). The volume-independent V2 crowding measure fixes this on the
  GME/AMC stress tests, though it doesn't reach threshold on CVNA specifically
  (its own 2022 collapse-era SI dominates its 2-year baseline) — so CVNA is
  shown via the weekly V1 rule instead. Misses (BYND, CVNA's separate July-2023
  move) and V2's weaker panel significance (p≈0.20) are stated plainly.
- Framed as what it is: a monitor that says "size down," not a return
  predictor — and a prototype whose crowding leg upgrades directly with desk
  data (daily borrow/utilization instead of lagged twice-monthly FINRA).

## Further nulls (same discipline, briefer writeups)

Three more pre-registered tests, same standard, all null — kept in the record
because a portfolio that only shows hits isn't credible:

- **Insider cluster-buys** (`insider_alpha/`): the most persistent documented
  insider-trading anomaly (≥2 distinct insiders buying in the open market
  within 21 days), built on SEC's free structured Form 4 dataset, 112 events
  in consumer names 2014–2024. Null — 63-day abnormal return exactly 0.00%,
  placebo p=0.87.
- **Dual momentum / GEM** (`equity_factor/dual_momentum.py`): the Antonacci
  US/international/bond-switching rule on the 37-year history. Beats VFINX on
  raw CAGR (11.5% vs 10.5%) but the Sharpe edge isn't significant (p=0.31);
  its outperformance is concentrated entirely in 2000–09.
- **Turn-of-month** (`equity_factor/turn_of_month.py`): one of the most
  durable calendar anomalies in the literature. Real and significant pre-2010
  (t=2.94) — and **dead since** (t=0.25, 2010–2024), exactly the decay pattern
  every other tested public-data edge in this project shows.

## Methods toolkit (shared across studies)

Block bootstrap (Sharpe-difference CIs, preserving autocorrelation) · time-shift
& circular-shuffle placebo nulls · deflated Sharpe (Bailey–López de Prado) ·
Newey-West/HAC and week-clustered standard errors · CAPM / FF5+UMD factor
attribution · turnover-based cost modeling · drop-one and parameter-sweep
robustness · explicit no-look-ahead and determinism checks.

## Reproduce
```bash
# each study is independent; free data, no keys
cd equity_factor   && python3 long_history.py && python3 torture_test.py
cd ../weather_alpha   && python3 weather_data.py && python3 test_signal.py
cd ../attention_alpha && python3 wiki_data.py && python3 test_signal.py && \
                         python3 torture_test.py && python3 earnings_study.py
```
Per-study write-ups: `equity_factor/FINDINGS.md`, `weather_alpha/FINDINGS.md`,
`attention_alpha/FINDINGS.md`.

## What I'd pursue next
The free-data signals are now mapped: a real-but-untradable short-horizon
attention edge, and nulls at the weather and fundamental/earnings horizons. The
honest read is that the remaining edge is unlikely to live in *free, public*
data (which gets arbitraged or absorbed into consensus), but in (a) combining
weak signals into a portfolio of genuinely uncorrelated sleeves — the Study-1
diversification principle applied to *signals* — or (b) richer proprietary
inputs (transaction/credit-card panels, store-level footprint) where the same
rigorous harness here would carry directly over. The methodology transfers; the
data is the constraint.
