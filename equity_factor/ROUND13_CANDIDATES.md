# Round 13 candidate hypotheses — BRAINSTORM ONLY, NOTHING TESTED

Committed 2026-07-06, before any candidate has been coded or evaluated.
Same discipline as ROUND12_PREREGISTRATION.md: writing ideas down *before*
looking at their results is what separates research from curve-fitting.

**Status: parked.** The live forward test (IBKR paper, inception 2026-07-06,
base NAV $1,272,650) is the current priority. Decision rule at the bottom.

---

## Known weaknesses these candidates target

(All quantified in FINDINGS.md / tearsheet.)

- **W1 — fast crashes:** monthly signals cannot dodge a one-week collapse
  (Oct-1987 type). The strategy eats the first leg of any fast fall.
- **W2 — V-shaped recoveries:** it re-enters late (2020), giving back part of
  the crash protection it just earned.
- **W3 — whipsaw:** 66% of cash-flights are false alarms costing ~4.5pts each.
- **W4 — regime dependence:** part of the 1987–2021 premium may be the
  falling-rate era; V0's 2017–24 decade alpha was ~0 (V2: +0.4%).
- **W5 — cash drag at 1x:** CAGR 4.6% trails SPY badly in long bull markets;
  honest levered math (financing at T-bill+100bps) gives ~7.5% at 2.5x with
  ~-20% DD, Sharpe FALLING from 0.86 to 0.61.

## Candidates (each costs one trial against the deflated-Sharpe budget)

- **C1 — asymmetric entry/exit.** Exit on the slow signal (10m/ensemble),
  re-enter on a fast one (e.g. 3m vote alone). Targets W2. Prior: mildly
  positive — V2's short lookbacks already helped 2020 re-entry.
- **C2 — intra-month circuit breaker.** A daily stop (e.g. sleeve -X% from
  month-start => cash until next month-end). Targets W1. Prior: SKEPTICAL —
  it trades the rare fast crash against many more whipsaws (W3 gets worse);
  the V1 ramp (a softer de-risking idea) already FAILED Round 12 selection.
- **C3 — vol-targeted book.** Scale total exposure to a constant ex-ante vol
  (e.g. 6% ann from trailing cov). Standard institutional overlay; targets
  risk consistency rather than return. Prior: neutral-positive; adds a
  parameter and modest turnover.
- **C4 — cash-competing hurdle (dual momentum).** Require an on-trend sleeve
  to also beat T-bills over the lookback (excess momentum), so high-rate
  regimes raise the bar to be invested. Directly targets W4, the rate-era
  counter-hypothesis. Prior: positive in the literature (Antonacci), and the
  most economically motivated of the five.
- **C5 — universe breadth.** More uncorrelated trend sleeves (currencies,
  intl bonds, TIPS). Round 12's V4 exploration (+TIP/+IWM/+BWX, 8-ETF era
  only) was inconclusive; free data lacks long history for a real test.
  Prior: right idea, wrong data budget.

## Explicitly rejected (do not revisit without new evidence)

- Hysteresis / ramp entries around the SMA (V1 family): failed pre-registered
  selection in Round 12.
- Fewer ETFs "to cut costs": costs are already ~10bps on tiny turnover; the
  basket breadth IS the mechanism (single-asset trend tested null).
- Naive leverage claims: see W5 numbers; any leverage discussion must include
  financing cost and intramonth drawdown (daily-sampled MaxDD is -7.2%,
  trough 2011-08-08, vs -6.5% monthly-sampled).

## Decision rule (pre-committed)

1. No candidate is tested before **6 months of live forward data** are logged.
2. At most **one** candidate per round, chosen for economic rationale (C4 is
   the current front-runner), tested under a Round-12-style protocol:
   prereg file committed first, 1987–2005 selection / 2006+ holdout, 8-ETF
   cross-confirmation, trial counter incremented (currently 13).
3. V0 stays live throughout; a winner replaces it only by the Round 12 §6 bar.
