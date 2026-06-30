# TDA Trading Strategy — Adam Levine

> **Looking for the current research?** See **[`RESEARCH.md`](RESEARCH.md)** — a
> portfolio of three self-contained, reproducible quant studies (a validated
> defensive strategy, an honest null, and a real-signal-killed-by-costs), all on
> free data. The TDA material below is earlier work, kept for the record with an
> honest negative result.

---

Quantitative research into topological data analysis (TDA) applied to equity market regime detection.

## Finding (honest, reproduced from real data)

**No version of the strategy is profitable, and none shows alpha.** Across every
real-data, look-ahead-free, factor-regression-verified backtest (test window 2022–2024):

- Best variant (beta-spread + Fiedler) = Sharpe **−0.77**; the pure spectral/TDA
  strategy = Sharpe **−2.57**. SPY buy-and-hold over the same window = **+0.49**.
- Factor regression shows **significantly negative** alpha (t ≈ −5 for the spectral
  strategies), with β≈0 — they are not disguised market bets, they destroy value.
- ML on the spectral/topology features gives **AUC ≈ 0.50** (a coin flip): no predictive
  power for next-day returns.

Full reproduced tables and an honest discussion of where alpha could *plausibly* come
from are in **[`HONEST_RESULTS.md`](HONEST_RESULTS.md)**.

The mathematical framework (correlation–CV bound, spectral-gap analysis) may still be a
valid *descriptive* contribution, but the *trading* claim is a clean negative result.

## Repository Structure

```
risk_report.py                    Verified backtest — Strategies A–D + FF5/UMD regression  [REAL]
tda_sector_backtest.py            Walk-forward sector backtest with ripser topology        [REAL]
strategy_variants_real.py         Strategy variants V1–V4, real data, 1-day lag            [REAL]
ml_integration_real.py            ML benchmark on real data, 1-day lag                     [REAL]
run_all_real.sh                   One-shot Colab runner for the real pipelines
HONEST_RESULTS.md                 Reproduced real numbers + alpha analysis

thesis_latex/                     LaTeX source for the thesis (figures/appendix tables are
                                  synthetic — see disclosure banners in sec07–sec10)

legacy_simulated/                 QUARANTINE — simulated/fabricated code, NOT validation.
                                  See legacy_simulated/README.md. Includes the figure
                                  generator, validation_tests, null_model_test,
                                  visualizations_3d, and the legacy thesis_expansion/ phases.
```

> Note: yfinance's `curl_cffi` backend may fail behind some corporate/agent proxies.
> On Google Colab it works out of the box. The real scripts are otherwise unmodified.

## Reproducing All Numbers (Google Colab)

```bash
!bash run_all_real.sh
```

This produces:
- `risk_report.pdf` — 6-panel risk report, Strategy A/B/C/D vs SPY (2022–2024)
- `ml_integration_real.{csv,txt}` — F1, AUC, precision, recall, Sharpe per ML model
- `strategy_variants_real.{csv,txt}` — V1/V2/V3/V4 metrics

After running, paste the contents of the `*.txt` files into Sections 8 and 10
of the thesis, then remove the `Simulation Disclosure` and `Projection Disclosure`
banners at the top of `sec08_variants.tex` and `sec10_ml.tex` — the disclosures
are only needed while those sections still reference simulated/projected numbers.

## Compiling the Thesis

```bash
cd thesis_latex
pdflatex thesis_main.tex
bibtex thesis_main
pdflatex thesis_main.tex
pdflatex thesis_main.tex
```

