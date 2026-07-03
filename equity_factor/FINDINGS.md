# Equity Factor Project — Findings Log

Goal: beat SPY buy-and-hold on a **risk-adjusted, after-cost** basis, using
cross-sectional selection over US single stocks. Honest evaluation: 1-month
implementation lag, turnover-based costs, and **two** benchmarks — SPY (real) and
an **equal-weight universe (EW)** that controls for survivorship bias (both the
strategy and EW draw from the same survivor set, so beating EW = genuine
selection skill).

## Round 1 & 2 — price-based factors (106 large-caps, 2010-08 → 2024-12, 10 bps/side)

| Portfolio | CAGR | Vol | Sharpe | MaxDD | vs EW? | FF5+UMD alpha (t) |
|---|---|---|---|---|---|---|
| Momentum, long-only top quintile | 16.2% | 15.3% | 1.07 | −20.2% | loses | +0.6% (0.4) |
| Low-vol, long-only top quintile | 10.9% | 12.4% | 0.90 | −19.4% | loses | −2.4% (−1.2) |
| Momentum, long-short (neutral) | −0.6% | 8.2% | −0.03 | −28.1% | — | −2.8% (−2.4) |
| **Equal-weight universe (control)** | 16.3% | 14.7% | **1.10** | −22.0% | — | — |
| SPY buy & hold | 14.4% | 14.4% | 1.01 | −23.9% | — | — |

### Honest conclusions
1. **No price-based factor beat the survivorship-neutral EW benchmark.** Momentum
   long-only *looks* like it beats SPY (1.07 vs 1.01) but loses to EW (1.10) — its
   apparent edge is survivorship + equal-weighting (a size tilt), not selection.
2. **Momentum has zero independent alpha** here (t=0.36) and the **long-short lost
   money with significantly negative alpha** (t=−2.4). Large-cap momentum was
   crowded/dead this decade.
3. **Low-vol** delivered the smallest drawdown but a *lower* Sharpe than SPY.

### The binding constraint
- **Survivorship bias** inflates every single-stock number (and the EW benchmark).
  Fixing it requires point-in-time index membership, which Yahoo doesn't provide.
- **No fundamentals** (book value, earnings, margins) → can't test value/quality,
  which is where much of the durable cross-sectional edge actually lives.
- Price-only factors on a biased large-cap set over a bull decade are a hard place
  to find honest alpha. This is a real result, not a bug.

## Round 3 — trend / regime overlay (risk management, not selection)

10-month SMA rule, cash when below trend (survivorship-free on SPY):

| Portfolio | CAGR | Vol | Sharpe | MaxDD |
|---|---|---|---|---|
| SPY buy & hold | 14.8% | 14.5% | **1.03** | −23.9% |
| SPY + trend overlay | 7.5% | 11.0% | 0.71 | −23.0% |
| EW buy & hold | 17.3% | 15.0% | **1.15** | −22.0% |
| EW + trend overlay | 9.6% | 11.3% | 0.87 | **−15.4%** |

**The overlay lowered Sharpe in both cases.** It cut the EW drawdown nicely
(−15.4% vs −22%) but at a heavy return cost; on SPY it barely helped drawdown
because monthly signals are too slow for fast crashes (2020). 2010–2024's
V-shaped recoveries punish trend filters (whipsaw).

## Overall honest conclusion (rounds 1–3)
Nothing tested — momentum, low-vol, long-short, or a trend overlay — beats
buy-and-hold **risk-adjusted** over 2010–2024 on this universe. This is the
genuine answer for a **price-only, large-cap, survivorship-biased, single-regime**
setup, not a lack of effort. Beating a strong bull market with simple robust rules
is legitimately hard, and most apparent edges are survivorship or overfitting.

To get an honest shot at edge we must change the **inputs**, not keep tuning:
point-in-time universe (kill survivorship), fundamentals (value/quality), a
broader universe incl. small-caps, and a longer multi-regime history.

## Round 4 — multi-asset trend + risk parity (survivorship-FREE, 8 ETFs)

GTAA: each asset trend-filtered (10m SMA, to cash when below), inverse-vol
weighted. 2007–2024, incl. the 2008 crisis. (`multi_asset.py`)

| Portfolio | CAGR | Vol | Sharpe | MaxDD | corr SPY |
|---|---|---|---|---|---|
| RP+Trend | 4.2% | 5.4% | **0.79** | **−7.0%** | 0.42 |
| RP+Trend 2x const lev | 4.0% | 10.8% | 0.41 | −20.7% | 0.42 |
| EW+Trend | 4.4% | 6.0% | 0.75 | −8.5% | 0.53 |
| 60/40 | 7.8% | 9.6% | 0.84 | −29.5% | 0.96 |
| SPY | 10.4% | 15.7% | 0.71 | −50.8% | 1.00 |

**First strategy to beat SPY risk-adjusted (full sample): 0.79 vs 0.71, with a
−7% drawdown vs −51%.** BUT two honest caveats kill the "money machine" reading:

**(a) The edge is ENTIRELY the 2008 crash.** Sub-period Sharpe (RP+Trend vs SPY):
- 2007–2009: **1.18 vs −0.19**  (RP+Trend wins massively)
- 2010–2024: 0.69 vs **0.97**  (SPY wins)
- 2015–2024: 0.64 vs **0.88**  (SPY wins)

In every post-crisis window, plain SPY beat it on Sharpe *and* return. The strategy
is **crash insurance**: it trades bull-market upside for tail protection.

**(b) You can't cheaply lever the gap.** Its ~4% return barely exceeds financing,
so constant leverage adds risk without return (Sharpe 0.41 at 2x). Dynamic
vol-targeting was worse (Sharpe 0.29) — it levered up into drawdowns.

### Honest bottom line
Across single-stock factors, trend overlays, and multi-asset diversification,
**nothing reliably beats buy-and-hold across all regimes.** The one genuine,
survivorship-free win is *drawdown reduction*: a diversified trend strategy cut the
worst-case loss from −51% to −7% and had a better full-cycle Sharpe — valuable if
you fear crashes, but it underperforms in bull markets. That is the real, complete
answer, consistent with 15 years of tactical strategies lagging the S&P.

## Round 5 — productionizing option 1 (cash yield, control, robustness)

Refined the multi-asset strategy: cash now earns the **1-month T-bill** (was 0%),
added a **no-trend control**, and stress-tested. (`multi_asset.py`, `present.py`)

| Portfolio | CAGR | Vol | Sharpe | MaxDD |
|---|---|---|---|---|
| **RP+Trend** (the strategy) | 4.6% | 5.4% | **0.86** | **−6.5%** |
| 60/40 | 7.8% | 9.6% | 0.84 | −29.5% |
| SPY | 10.4% | 15.7% | 0.71 | −50.8% |
| RP static (no trend, control) | 5.1% | 9.2% | 0.58 | −20.7% |

- **The trend filter is the active ingredient:** same 8 assets without it = Sharpe
  0.58 / −20.7% DD. Adding trend → 0.86 / −6.5%.
- **Robust, not cherry-picked:** trend window 6–14m → Sharpe 0.86–0.93; vol window
  6–18m → 0.78–0.86; survives 40 bps/side costs (0.73).
- **Consistent across regimes** (Sharpe): crisis '07–09 = 1.23 (SPY −0.19);
  bull '10–24 = 0.77 (SPY 0.97); recent '15–24 = 0.77 (SPY 0.88). It never blows
  up; SPY out-returns it in calm decades but craters in crises.
- Verified: deterministic re-run, weights+cash=1, no leverage/shorting, and a
  hand-recomputed last-month weight check confirms **no look-ahead**. See
  `presentation.html` (run `python3 present.py`).

**Honest verdict on option 1:** a real, survivorship-free, reproducible strategy that
beats buy-and-hold *risk-adjusted* (higher Sharpe, 1/8th the drawdown). It does NOT
out-*return* SPY in bull markets — its value is consistency and crash protection.

## Round 6 — TORTURE TESTS (trying hard to kill it). `torture_test.py`

Healthy skepticism applied. Results that matter:

| Test | Result | Verdict |
|---|---|---|
| Sharpe diff vs SPY (block bootstrap) | ΔSharpe +0.13, **95% CI [−0.38, +0.64]**, p=0.32 | **NOT significant** |
| Sharpe diff vs 60/40 | ΔSharpe +0.00 | indistinguishable |
| Deflated Sharpe (6–20 trials) | P(true Sharpe>0) ≈ 0.96–0.99 | positive Sharpe is real… |
| Drop-one-asset | 0.73–0.92; drop ALL bonds → 0.70 | robust but bond-dependent |
| Rebalance-date shift ±10d | Sharpe 0.66–0.86 (month-end = best) | **date-luck present** |
| Trend vs static diversification | ΔSharpe +0.26, p=0.08; DD −6.5% vs −20.7% | trend adds value, mostly via DD |

### Honest verdict
**The Sharpe edge over buy-and-hold is NOT statistically significant.** We have a
strategy with a robustly *positive* Sharpe and robust *drawdown reduction*
(−6.5% vs SPY −51%), but its risk-adjusted advantage over SPY/60/40 is within the
noise — and the headline 0.86 is the lucky-rebalance-day, bond-bull, 2008-driven
best case (a fairer figure is ~0.75). What survives torture is **crash protection**,
not market-beating alpha.

### The only test that can settle it
Confidence intervals are huge because the sample has ~2 crashes. The definitive
next step is **extending history to 1990 (or 1973) with index/fund proxies** —
more independent crashes and a rising-rate regime where the bond sleeve can't
help. That, not more tuning, is how you'd actually prove or break this.

## Round 7 — the RIGHT lens: alpha as low correlation (`portfolio_blend.py`)

Judging a 0.42-correlation sleeve by its standalone Sharpe vs SPY is the wrong test.
The right test is its marginal contribution to a portfolio.

- **CAPM alpha** of RP+Trend vs the market: **+1.93%/yr, t = 1.75**, beta 0.14
  (nearly market-neutral). This is "return not explained by market correlation."
- **Adding it to SPY improves the portfolio.** 70% SPY / 30% RP+Trend: CAGR 8.9%
  (vs SPY 10.4%), Sharpe 0.79 (vs 0.71), MaxDD −37.5% (vs −50.8%). Sharpe-max blend
  (~82% sleeve) hits 0.95. Blend-vs-SPY ΔSharpe +0.22, **p = 0.10** — much stronger
  than the standalone test (p = 0.32), though still short of p<0.05.

**Reframe:** standalone = "not proven"; as an uncorrelated sleeve = "promising,
nearly significant." The value is diversification, not standalone outperformance.

**Forward principle (Dalio's "Holy Grail"):** stack 3–5 genuinely uncorrelated
positive-return sleeves rather than over-tuning this one. Correlation does the work.

## Round 8 — LONG-HISTORY VALIDATION (1987–2024): it passes. `long_history.py`

Re-ran the identical engine on dividend-adjusted fund proxies back to 1987 (VFINX,
VWIGX, VUSTX, VGPMX, VWEHX) — adding Black Monday '87, the 1990 recession, the 1994
bond crash, 1998 LTCM, 2000–02 dot-com, plus a high/rising-rate regime. 451 months.

| Portfolio | CAGR | Sharpe | MaxDD |
|---|---|---|---|
| **RP+Trend** | 6.8% | **1.16** | **−12.8%** |
| 60/40 | 9.2% | 0.93 | −29.9% |
| VFINX buy & hold | 10.5% | 0.74 | −51.0% |

- **The Sharpe edge over buy-and-hold is now SIGNIFICANT: ΔSharpe +0.43, 95% CI
  [+0.13, +0.75], p = 0.002** (short sample was p = 0.32). More crashes → the
  confidence interval collapsed. vs 60/40 it's borderline (+0.24, p = 0.065).
- **Robust on every axis:** rebalance-date shifts ≥1.03 (short sample was a fragile
  0.66–0.86); 40 bps costs → 1.06; trend window 6–14m → 1.15–1.17; drop any asset →
  0.96–1.32 (all still beat buy-and-hold).
- **Consistent across regimes** (Sharpe): '87–'99 = 1.34, '00–'09 = 1.27 (VFINX
  0.02!), '10–'24 = 0.94 (≈ VFINX 0.96). Positive returns in 2000/01/02 and ~flat
  in 2008 while equities lost 37%.
- Works on a *different* instrument set than the 8 ETFs → the edge is the
  mechanism (trend + risk parity), not the specific tickers picked.

### Verdict (upgraded, earned)
The strategy is **validated**: a statistically significant, robust improvement in
risk-adjusted return over equity buy-and-hold across 37 years and 7+ crises, driven
by crash avoidance and slow-bear protection. Honest limits remain: it does NOT
out-*return* equities in raging bulls (ties on Sharpe in 2010–24), and its edge over
a strong 60/40 is only borderline-significant. It is a genuine **defensive core /
diversifying sleeve**, now backed by out-of-sample evidence — not a lucky backtest.

## Round 9 — vs the actual competing PRODUCTS (`competitors.py`)

Benchmarked against real funds, not just SPY/60-40.

**vs AQR Managed Futures (AQMNX), 2010–2024:** ours Sharpe 0.83 / DD −6.5% vs
AQR 0.33 / −25.1%. We beat AQR's flagship trend fund over this window — BUT
(a) 2010–2024 was a notoriously bad decade for managed futures and excludes AQR's
huge +2008, and (b) AQR is a *truer* diversifier: it can short, so its equity corr
is −0.21 vs our +0.48. Correlation of ours to AQR MF is only 0.17 — different tools.

**vs static multi-asset funds:** ours beats PRPFX (Permanent Portfolio) and VBIAX
(Vanguard Balanced) on Sharpe and drawdown in every window. Over 37 years (1987–2024)
ours matched PRPFX's return (6.8% vs 6.7%) at much higher Sharpe (1.16 vs 0.80) and
smaller drawdown (−12.8% vs −19.1%).

**Honest structural point:** ours is LONG-ONLY (de-risks to cash, can't short), so it
is a *defensive long allocator* (+0.48 equity corr), NOT a true managed-futures hedge
(AQR's −0.21). It is not a replacement for AQR; it's a simpler, gentler cousin that
happened to have a higher Sharpe in this window.

## Reproduce
```bash
cd equity_factor
python3 data.py        # fetch/caches the price panel (urllib; proxy-safe)
python3 backtest.py    # prints the table above + factor regression
```

## Round 10 — the alpha hunt, concluded (`alpha_test.py`, `dual_momentum.py`, `turn_of_month.py`, plus insider_alpha/, squeeze_monitor/flag_pnl.py)

Systematic sweep for standalone alpha in free public data. Honest nulls, all
pre-committed specs: **flag→long P&L** (concentration, alpha t=0.53); **insider
cluster buys** on 112 consumer events 2014-24 (AR63 = 0.00%, placebo p=0.87);
**dual momentum / GEM** on 37y (beats VFINX CAGR 11.5% vs 10.5% but ΔSharpe
p=0.31, edge entirely 2000-09); **turn-of-month** (strong pre-2010 t=2.94, dead
since, t=0.25). Pattern: every published public-data anomaly is dead or
insignificant where institutions can trade it.

**The one result that formally clears the alpha bar is RP+Trend itself:**

| Test (1987-2024, 451 mo, net of costs) | alpha/yr | t | p |
|---|---|---|---|
| CAPM (vs market) | **+1.80%** | **+2.30** | **0.022** |
| Spanning (vs all 5 own assets — strictest) | +1.40% | +1.89 | 0.058 |
| second half alone | +1.98% | +1.88 | 0.060 |

Significant CAPM alpha at beta 0.24, **stronger in the second half than the
first** (no post-discovery decay), on top of the already-established Sharpe
significance (p=0.002) and timing-vs-static significance (p=0.025). Verdict:
a modest, real, verifiable ~1.8%/yr of defensive timing alpha — the honest
final answer of the whole search.

## Round 11 — extended validation depth (`extended_tests.py`, `cash_conditioning.py`)

**40-year extension** (4-fund variant reaching back through the Volcker rate
shock, 1981-2024, 479 months): CAPM alpha **+2.29%/yr, t = +3.08, p = 0.002**
at β = 0.15 — the alpha claim now clears t=3 on the longest sample free data
allows. Sharpe 1.41 vs equity 0.76.

**Rolling stability**: 5-year rolling CAPM alpha positive in **84% of all
windows** (median +2.0%/yr) — not a two-point artifact. 3-year rolling Sharpe
beats equity in 68% of windows.

**Decade-by-decade alpha** (honest, not smoothed over): +0.3% (1987-96),
+2.9% (1997-2006), +3.3% (2007-16), **-0.3% (2017-24)**. The edge lived in
1997-2016 and was essentially absent in the most recent stretch.

**Regime conditioning**: NBER recessions -2.4%/yr vs equity -13.4%/yr.
Rising-rate regimes (trailing 12mo bond return negative) hurt: Sharpe 0.34 vs
falling-rate 1.52 — a real, stated dependency on the bond sleeve's tailwind.

**Execution-lag stress**: rebalancing 5/10/21 sessions late still yields
Sharpe 1.03-1.11 — no rebalance-date luck.

**Stationary bootstrap**: Sharpe-diff significance holds (p = 0.001-0.006)
across mean block lengths 3/6/12mo — not sensitive to the block-size choice.

**Career risk** (new, important for anyone actually allocating to this):
underperforms buy-and-hold in **69% of rolling 3-year windows**, once for
**159 consecutive months**. This is likely *why* the premium persists — very
few allocators can sit through 13 years of relative underperformance.

**Cash-level conditioning** (`cash_conditioning.py` — does the strategy's OWN
cash level carry information?): in the extreme 90-100%-cash state (n=15,
clustered at major bottoms), equity's own next-12-month return averaged
**+20.3%** (t=2.04) vs +12.2% when fully invested — mean reversion after
crashes — while the strategy itself captured only +8.0% of that window. The
continuous linear regression of cash%% on equity's forward return is **not
significant at any horizon** (t in [-0.35, +0.90]) — this is concentrated at
the extreme bin, not a tradable relationship.

**Whipsaw cost (the mechanical explanation for the known up-capture
shortfall)**: 86 entries into ≥50% cash over 38 years; **66% reverse within 4
months** (false alarms). During those false-alarm episodes the strategy
earned +1.25% vs +5.79% for staying invested — a clean, quantified cost. The
entire edge rides on the ~1-in-3 cash entries that catch a real bear market.

**Correlation-regime risk** (standard CTA objection, tested directly): does
NOT bite — Sharpe in high-correlation regimes (1.22) slightly exceeds
low-correlation regimes (1.10).

### Honest verdict
Depth of testing is now essentially exhaustive for what free data supports.
Nothing broke the core claim; several new numbers *sharpen* it (40yr t=3.08,
84% rolling-window consistency) while others are genuinely uncomfortable and
now on the record (2017-24 alpha ≈ 0, 69% 3-year-window underperformance,
66% whipsaw rate). That combination — real edge, honestly bounded — is the
final, complete answer this project can give with free data alone.
