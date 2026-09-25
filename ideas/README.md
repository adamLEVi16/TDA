# Can the TDA idea be turned into a strategy that makes money?

Four extensions of the correlation-network idea were tested on real data. Each has:

- a pre-registered `SPEC.md`;
- parameters chosen on a design period, then one run on a later test period;
- trading costs and T-bill-adjusted Sharpe ratios.

An independent verifier re-ran every script (logs in each `verify/` folder) and tried to refute the result. All four results stood; the corrections were minor.

**Short answer: none of them beats simply holding an index fund.** The only useful finding is about risk: a plain 10-month trend rule roughly halves the worst drawdown and costs about 1% a year in return. Correlation and topology add nothing to it.

| Folder | Idea | Verified result |
|---|---|---|
| `A/` | Monthly "network momentum": predict next month from neighbours' past returns (49 industries 1927–2024; 30 ETFs 2007–2024) | No edge. Adds nothing beyond ordinary industry momentum after costs (net alpha −0.6%/yr, t −0.38). The top-5 ETF version had Sharpe 0.59 vs SPY 0.62 (p 0.86), and from 2019–2024 it earned +82% vs SPY +158%. |
| `B/` | Your graph-residual signal on S&P 100 stocks at 1-day, 1-week, 1-month and 12-month horizons | No edge. Daily and weekly versions break even only at 1–3 bp of cost. The 12-month version is plain momentum (correlation 0.96) and fails multiple-testing correction. |
| `C/` | Regime detection done right: move a long-only market position to T-bills using volatility, trend, correlation, absorption ratio, H1 topology, Fiedler or VIX | No regime rule beats buy-and-hold significantly. Your topology-volatility filter is at best equal to buy-and-hold; the design-chosen version is significantly worse (Sharpe 0.21 vs 0.57). Trend and volatility rules cut drawdowns. |
| `D/` | Volatility and correlation risk premia (how professionals trade correlation) | Variance-risk-premium timing has not worked out of sample since 2012. Short-volatility products lost 88% in three days (Feb 2018). Not usable in a small or custodial account. |

## Drawdown-reducing rules (the only practical output)

US market, 1990–2024, 2 bp costs, cash earns T-bills (`C/logs_test.txt`):

| Rule | Sharpe | CAGR | Worst drawdown |
|---|---|---|---|
| Buy and hold | 0.57 | 10.7% | −54.6% |
| 10-month trend (hold the index when it is above its 10-month average, else T-bills; checked monthly) | 0.66 | 9.7% | −25.6% |
| Volatility targeting × 10-month trend | 0.68 | 9.7% | −23.4% |

The Sharpe differences are not statistically significant (p ≈ 0.5). The trend rule trailed buy-and-hold in 19 of 35 years. What it buys is a smaller worst case, not a higher return.

SPY, 2000–2026 (`D/test_run.log`): buy-and-hold max drawdown −50.8%, 10-month rule −22.0%, 15% volatility target −35.7%.
