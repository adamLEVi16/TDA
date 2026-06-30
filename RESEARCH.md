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

- Sharpe edge over buy-and-hold is **statistically significant** (ΔSharpe +0.43,
  95% CI [+0.13, +0.75], block-bootstrap **p = 0.002**).
- Robust: survives 40 bps/side costs (Sharpe 1.06); trend window 6–14 mo →
  1.15–1.17 (no fitted parameter); drop any single asset → 0.96–1.32; rebalance-
  date shifts ≥1.03. Verified deterministic, no leverage/shorting, hand-checked
  no-look-ahead.
- **Honest limitation, stated up front:** it does *not* out-*return* equities in
  bull markets (6.8% vs 10.5% CAGR). It is **crash insurance / a diversifying
  sleeve** — alternative beta (trend premium), not proprietary alpha. Its job is
  risk reduction, and at that it's real and significant.

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

**Takeaway:** the hardest and most valuable judgment in applied quant — the gap
between "statistically significant" and "tradable after costs." This is a real,
momentum-independent piece of public information that is nonetheless not worth a
fund's transaction costs to harvest. Its legitimate use is as a free, real-time
*attention monitor* feeding a discretionary, low-turnover process — explicitly
not pitched as a standalone strategy.

---

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
cd ../attention_alpha && python3 wiki_data.py   && python3 test_signal.py && python3 torture_test.py
```
Per-study write-ups: `equity_factor/FINDINGS.md`, `weather_alpha/FINDINGS.md`,
`attention_alpha/FINDINGS.md`.

## What I'd pursue next
The attention result points to the higher-conviction follow-up: the *fundamental*
channel rather than the 1-week pop — cumulative attention over a fiscal quarter
vs. the stock's move on its **next earnings date** (attention as a same-store-
sales nowcast). Slower-decaying and far less turnover-sensitive, so it can clear
the cost wall that kills the weekly version. Needs a historical earnings calendar.
