# Can the TDA idea be turned into a strategy that makes money?

Five ideas were tested on real data:

- A–D are extensions of the correlation-network idea.
- E is a newer-research thesis about *when* the stock market pays.

Each test has:

- a pre-registered `SPEC.md`;
- rules fixed before the test period, either chosen on a design period or taken from the original paper;
- trading costs and Sharpe ratios computed on returns above T-bills.

An independent verifier re-ran every script (logs in each `verify/` folder) and tried to refute the result. All five results stood; the corrections were minor.

**Short answer: none of them beats simply holding an index fund.** The only useful finding is about risk. A plain 10-month trend rule roughly halves the worst drawdown and costs about 1% a year in return. Correlation and topology add nothing to it.

| Folder | Idea | Verified result |
|---|---|---|
| `A/` | Monthly "network momentum": predict next month from neighbours' past returns (49 industries 1927–2024; 30 ETFs 2007–2024) | No edge. It adds nothing beyond ordinary industry momentum after costs (net alpha −0.6%/yr, t −0.38). The top-5 ETF version had Sharpe 0.59 vs SPY 0.62 (p 0.86), and from 2019–2024 it earned +82% vs SPY +158%. |
| `B/` | Your graph-residual signal on S&P 100 stocks at 1-day, 1-week, 1-month and 12-month horizons | No edge. The daily and weekly versions break even only at 1–3 bp of cost. The 12-month version is plain momentum (correlation 0.96) and fails multiple-testing correction. |
| `C/` | Regime detection done right: move a long-only market position to T-bills using volatility, trend, correlation, absorption ratio, H1 topology, Fiedler or VIX | No regime rule beats buy-and-hold significantly. Your topology-volatility filter is at best equal to buy-and-hold; the version chosen on the design period is significantly worse (Sharpe 0.21 vs 0.57). Trend and volatility rules cut drawdowns. |
| `D/` | Volatility and correlation risk premia (how professionals trade correlation) | Variance-risk-premium timing has not worked out of sample since 2012. Short-volatility products lost 88% in three days (Feb 2018). Not usable in a small or custodial account. |
| `E/` | When is the premium earned? Hold the index only in FOMC-cycle "even weeks", on Fed decision days, or at the turn of the month (2008–2019 papers) | Every effect replicates in its original sample and disappears after publication. The rules earned 2–7% a year against 11–15% for buy-and-hold, and did worse than holding the same average stock weight every day. |

## Drawdown-reducing rules (the only practical output)

US market, 1990–2024, 2 bp costs, cash earns T-bills (`C/logs_test.txt`):

| Rule | Sharpe | CAGR | Worst drawdown |
|---|---|---|---|
| Buy and hold | 0.57 | 10.7% | −54.6% |
| 10-month trend (hold the index when it is above its 10-month average, else T-bills; checked monthly) | 0.66 | 9.7% | −25.6% |
| Volatility targeting × 10-month trend | 0.68 | 9.7% | −23.4% |

The Sharpe differences are not statistically significant (p ≈ 0.5). The trend rule trailed buy-and-hold in 19 of 35 years. What it buys is a smaller worst case, not a higher return.

SPY, 2000–2026 (`D/test_run.log`): buy-and-hold max drawdown −50.8%, 10-month rule −22.0%, 15% volatility target −35.7%.

## E. When is the equity premium earned? (post-publication test of newer research)

**Thesis.** Published papers say most of the stock market's return comes on predictable days:

- FOMC-cycle "even weeks": Cieslak, Morse & Vissing-Jorgensen (2019), *JF* 74(5), doi:10.1111/jofi.12818.
- Scheduled Fed decision days: Lucca & Moench (2015), *JF* 70(1), doi:10.1111/jofi.12196.
- The turn of the month: McConnell & Xu (2008), *FAJ* 64(2), doi:10.2469/faj.v64.n2.11. A mechanism is proposed in Etula et al. (2020), *RFS* 33(1), doi:10.1093/rfs/hhz054.

If these held, you could hold the index on those days and T-bills otherwise, and get most of the return with half the exposure or less.

**How it was tested.**

- `E/SPEC.md` was written before any post-publication data was examined. The rules have no free parameters.
- FOMC dates are parsed from federalreserve.gov (`E/fomc_dates.py`). The verifier confirmed all 260 scheduled meetings held from 1994 to mid-2026 against the Fed's own statements and minutes.
- Returns are Ken French daily data (the source the FOMC paper used), checked on SPY, with 1 bp and 5 bp costs.
- Each paper is first replicated in its own sample, then the same rule is run after publication (`E/calendar_test.py`, `E/logs_calendar_test.txt`).
- Amendment 1 in `SPEC.md` changed the week definition to match the paper exactly after a first run; both versions are reported, and they agree.

**Results.** Figures are bp per day in-window vs out-of-window unless labelled Sharpe.

| Rule | Replicated in original sample | After publication | Rule vs buy-and-hold after publication (1 bp) |
|---|---|---|---|
| Hold only in FOMC even weeks | 1994–2016: 9.9 vs −2.5 (t 4.03). On the paper's metric, Sharpe 0.93 vs 0.45 (paper: 0.92 vs 0.45). | 2017–Jul 2026: 3.2 vs 7.2 (t −0.79) | Sharpe 0.27 vs 0.71; 5.2% vs 15.2% a year |
| Hold only on Fed decision days | 1994–2011: 36.0 vs 1.7 (t 3.42), a close-to-close proxy for Lucca–Moench | 2011–Jul 2026: 8.7 vs 5.0 (t 0.30); 13.3 vs 4.9 (t 0.75) without the cancelled 18 Mar 2020 meeting | Sharpe 0.14 vs 0.73 |
| Hold only at the turn of the month | 1926–2005: 14.8 vs 0.2 (t 7.96); 95% of the premium | 2006–Jul 2026: 4.7 vs 4.3 (t 0.10) | Sharpe 0.24 vs 0.56; 3.4% vs 11.3% a year |

- **Robustness.** The same holds on SPY, at 5 bp, and when each test starts at the publication date instead of the sample end.
- **Power.** The post-publication data reject the original effect sizes: even weeks p 0.001, Fed days p 0.01, turn of month p < 0.001 (`E/verify/logs_extra.txt`). A half-size effect is rejected for even weeks (p 0.04) but cannot be ruled out for the turn of the month (p 0.08) or Fed days (p 0.26).
- **Timing vs a constant mix, no costs on either side.** Timing did no better than holding the same average stock weight every day:
  - Even-week rule: 5.7% a year vs 8.7% for a constant 47% stock / 53% T-bill mix.
  - Turn-of-month rule: tied on return (3.7% vs 3.8%) but with a worse worst drawdown (−18.7% vs −10.8%).
  - The rules' smaller drawdowns come only from being less invested.

This matches McLean & Pontiff (2016), *JF* 71(1), doi:10.1111/jofi.12365. Across 97 published predictors, returns were 58% lower after publication, and the effects with the strongest in-sample returns declined the most.

The verifier's `E/verify/` scripts ran against the date list before the December 1993 meeting was added. That addition affects only January 1994, inside the replication period.

## F. Bull put spreads on the S&P 500 (`F/put_spreads.py`, `F/logs_put_spreads.txt`)

No free historical option prices exist, so this uses two real-data angles. It has not been independently re-checked like A–E.

**1. Cboe strategy indices (real traded SPX option prices).**

CNDR is the Cboe iron condor index: sell a ~20-delta put and call, buy ~5-delta wings, monthly, held to expiry. Its put side is a bull put spread.

| Series | Period | Return/yr | Sharpe (excess) | Alpha vs S&P 500 | Max drawdown |
|---|---|---|---|---|---|
| CNDR | 1988–2007 | 10.5% | 0.91 | +4.8%/yr (t 3.2) | −16% |
| CNDR | 2008–Aug 2026 | 0.7% | −0.01 | −1.1%/yr (t −0.6) | −20% |
| PUT (ATM put writing) | 2008–Aug 2026 | 7.1% | 0.59 | −0.3%/yr | −37% |
| S&P 500 total return | 2008–Aug 2026 | 11.5% | 0.71 | – | −52% |

Over 1988–2026, CNDR's Sharpe is significantly below the S&P 500's (p 0.01).

**2. Is today's credit worth it?**

- Live SPY chain on 25 Sep 2026, Oct 30 expiry: selling the 743 put (20-delta) and buying the 693 put (5-delta) fills at $2.96. That is 5.9% of the $50 width.
- The same spread, placed the same number of VIX-sigmas below spot, paid out 3.4% of width on average over 1990–2026 (95% CI 2.1–4.6%). The win rate was 90%.
- By era, it paid out 2.9% of width in 1990–2007 and 4.3% in 2008–2026.
- So *if* credits were always as rich as today's, the trade has positive expected value: about +2.4% on capital at risk per ~5-week trade.
- CNDR's flat post-2008 record says historical credits often were not that rich. A single snapshot cannot settle this.

**Sizing decides the outcome.** Today's credit ratio applied to 370 historical trades, with cash earning nothing:

| Account risked per trade | Return/yr | Worst drawdown |
|---|---|---|
| 10% | 2.7% | −13% |
| 20% | 5.2% | −25% |
| 50% | 11.5% | −59% |
| 100% | wiped out | −100% |

At equal drawdown this is roughly S&P-like, not better.
