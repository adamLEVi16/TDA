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

## Reproduce
```bash
cd equity_factor
python3 data.py        # fetch/caches the price panel (urllib; proxy-safe)
python3 backtest.py    # prints the table above + factor regression
```
