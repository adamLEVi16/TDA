# Honest Results — Reproduced from Real Data

These numbers were produced by running the **real-data** scripts (`risk_report.py`,
`strategy_variants_real.py`, `ml_integration_real.py`) on live Yahoo Finance prices
+ Ken French factors. All use a strict 1-day signal lag, walk-forward split
(train 2019–2021, **test 2022–2024**), and 5 bps transaction costs.

> Bottom line: **No version of the strategy is profitable, and none shows alpha.**
> The topology/spectral signal does not predict next-day returns better than a coin flip.

## 1. Verified backtest (`risk_report.py`)

| Strategy | Ann Ret | Ann Vol | Sharpe | Max DD | Hit |
|---|---|---|---|---|---|
| A: Laplacian + Fiedler (TDA) | −30.8% | 12.0% | **−2.57** | −61.6% | 19.4% |
| B: Laplacian + Vol filter | −26.9% | 11.1% | −2.42 | −56.3% | 22.2% |
| C: Laplacian only (no filter) | −37.9% | 13.7% | −2.75 | −69.0% | 16.7% |
| D: Beta-spread + Fiedler (best) | −9.8% | 12.7% | **−0.77** | −40.4% | 47.2% |
| **SPY buy & hold** | **+8.6%** | 17.5% | **+0.49** | −26.2% | 61.1% |

### Factor regression (FF5 + UMD, Newey-West HAC t-stats)

| Strategy | Annualized alpha | t-stat | β(Mkt) | R² |
|---|---|---|---|---|
| A: Laplacian + Fiedler | −34.99% | **−5.14** | 0.05 | 0.007 |
| B: Laplacian + Vol | −31.49% | −5.24 | 0.07 | 0.018 |
| C: Laplacian only | −42.18% | −5.67 | 0.08 | 0.017 |
| D: Beta-spread + Fiedler | −14.00% | −1.88 | −0.10 | 0.074 |

**Alpha is significantly _negative_** (not zero) for the spectral strategies. β≈0,
so these are not disguised market bets — they actively destroy value.

## 2. Strategy variants (`strategy_variants_real.py`)

| Variant | Ann Ret | Sharpe | Max DD |
|---|---|---|---|
| V1: Mean-Reversion | −29.7% | −2.00 | −61.0% |
| V2: Momentum + TDA Hybrid | −16.1% | −1.11 | −44.5% |
| V3: Adaptive Threshold (Z) | −12.5% | −1.06 | −37.4% |
| V4: Ensemble | −19.4% | −2.06 | −45.8% |

## 3. ML integration (`ml_integration_real.py`)

| Model | F1 | **AUC** | Precision | Recall | Sharpe |
|---|---|---|---|---|---|
| Threshold (Fiedler<25pct) | 0.373 | 0.509 | 0.538 | 0.285 | −0.27 |
| Random Forest | 0.606 | **0.498** | 0.521 | 0.723 | nan |
| Gradient Boosting | 0.591 | **0.492** | 0.524 | 0.677 | nan |
| Neural Network | 0.688 | **0.510** | 0.525 | 1.000 | nan |

**AUC ≈ 0.50 across every model = no predictive power.** Feature importances are
nearly uniform (0.14–0.16) across all seven spectral/correlation features — i.e. the
model found noise, not signal. (The NN's recall=1.0 / Sharpe=nan means it degenerately
predicts a single class.)

---

## Why it fails, and where real alpha could *plausibly* come from

These are **hypotheses to test**, not promises. The current evidence sets a high bar:
any new idea must beat (a) SPY buy-and-hold and (b) the plain realized-vol filter (B),
which already matches or beats the topology signal.

1. **Timescale mismatch (most likely culprit).** Features are 60-day rolling
   correlations — slow-moving — but the target is *next-day* return sign. Slow features
   cannot predict fast moves. Test weekly/monthly horizons before concluding anything.

2. **Topology is a risk signal, not a return signal.** The literature supports TDA for
   *crash/stress detection*, not directional alpha. Reframe it as a de-risking overlay on
   a long-only book (hold sector ETF when CV/Fiedler is calm, cut exposure when stressed)
   and measure risk-adjusted return vs buy-and-hold — but it must beat the vol filter to
   add anything.

3. **Universe too small.** 6 stocks/sector gives a topology of ~6 nodes; H1 "loops" and CV
   are statistically thin. A 50–100 name universe gives the persistence diagram something
   to work with.

4. **Test window had no crash.** 2022–2024 was a drawdown but not a 2008/2020-style
   correlation spike, where these methods claim their edge. An honest out-of-sample crisis
   test is worth running — with the caveat that fitting to known crises is its own trap.

5. **The directional mean-reversion bet is the weakest link.** Long losers / short winners
   intraday-to-daily is fighting short-term momentum; that's where the −2 Sharpes come from,
   independent of topology.

**Honest assessment:** as currently constructed, there is no detectable alpha. The
mathematical framework (correlation–CV bound, spectral-gap analysis) may still be a valid
*descriptive* contribution, but the *trading* claim is a clean negative result.
