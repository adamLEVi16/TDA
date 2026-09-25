# Audit of the TDA trading research

**Scope.** The topological-data-analysis (TDA) trading project in this repo, December 2025 to May 2026:
the original Colab notebooks (commit `7df04e6`), the January "SSRN-ready" paper (v12), the Davidson
paper branch (`claude/review-tda-paper-gGfou`), and the thesis and scripts on `main`. Not audited: the
USD/JPY study (`claude/modest-clarke-nma4gh`) and the non-TDA projects on `claude/code-audit-1bo6sh`
(see section 8).

**Method.** I read every version of the code and the papers, re-ran the original backtest and `main`'s
scripts on fresh Yahoo Finance data, and checked 41 citations against Crossref. The re-run scripts and
their output are in [`audit/`](audit/). Every number below comes from those outputs or from a file and
line cited next to it.

---

## 1. Verdict

**The idea: 3/10.** Using persistent homology of stock-correlation networks to detect market stress
is a real, small research area, so the question was reasonable. The trading design could not work. It
replaces about 70% of a 20-stock long/short book every day, which costs 25–36% a year at 5 bp per trade.
No known daily signal on mega-cap stocks earns that much before costs. That arithmetic was
available before any code was written.

**The execution splits in two.**

- The **December 2025 notebooks are honest.** They reproduce to the second decimal and correctly
  report that the strategy lost money (walk-forward Sharpe −0.56).
- **Almost everything written afterwards does not hold up.** The January paper, the Davidson paper and
  the thesis contain statistics that no code in the repo produces, a headline result computed on
  synthetic data built to contain that result, and citations to papers that do not exist. None of these
  documents should be shown to anyone in their current form.

**Why the strategy actually failed.** None of these three reasons appears in any version of the paper.

1. **There is no signal.** Before costs, the signal's Sharpe over 2007–2024 is −0.19 (t = −0.8).
2. **Trading costs cause the losses.** Before costs the 2022–24 test was slightly positive (Sharpe +0.6
   to +0.9, not significant). After costs it was −0.56.
3. **The topology filter points the wrong way.** It labels calm markets "unstable". It flagged 0 of 52
   trading days during the COVID crash and 60% of the calm 2023–24 rally.

The paper's explanation instead is "scale mismatch between local signals and global topology". Nothing
in the repo tests that idea, and the April re-run of the Phase-3 notebook measured it and got the
opposite of what the paper says (section 5.7).

---

## 2. What the strategy does

**Data.** Daily prices for 20 large US stocks (AAPL, MSFT, NVDA, JPM, XOM, KO, …), 2019–2024.

**Signal (graph Laplacian residual).** Each day:

- Connect two stocks if their 60-day return correlation is above 0.3.
- Compute a "smoothed" return for each stock: a blend of its own return and its neighbours', repeated
  three times. This is the formula h = (I − αL)³x.
- Residual = actual return − smoothed return. It measures how much the stock beat or lagged the stocks
  it usually moves with.

**Trade.** The next day, buy the 5 stocks with the largest residuals and short the 5 with the smallest.
Equal weights, rebuilt every day.

Buying yesterday's relative winners is a *momentum* trade. Every write-up calls it "mean reversion"
and says it "buys underperformers". That is the opposite of what the code does (section 5.3).

**Filter (persistent homology).**

- Treat each stock as a point, with distance √(2(1 − correlation)), so correlated stocks are close.
- Join every pair of points closer than a distance ε, and raise ε from 0. Rings of connections
  ("loops", H1) appear and later fill in. Persistent homology records how many loops appear and how
  long each one lasts.
- "Topology volatility" = 30-day rolling standard deviation of the loop count + 30-day rolling standard
  deviation of total loop lifetime.
- When topology volatility is above its 75th percentile, the day is "unstable" and the strategy holds
  cash.

**Why the filter measures the wrong thing.** In a sell-off all stocks move together, all distances
shrink at once, and few loops form. In calm markets stocks move independently, distances spread out,
and short-lived loops come and go. So the loop count falls in crises, and its variability is highest in
calm markets. The filter is a noisy inverse of average correlation.

**Later variants** (`main`, April 2026) swap persistent homology for the Fiedler value (the
second-smallest eigenvalue of the graph Laplacian, a measure of how well connected the graph is). They
run the trade inside 6-stock sectors and add a "beta-spread" trade (Strategy D). The same inversion
applies: a low Fiedler value means low correlation, which means calm markets, and the code labels it
"stressed" (section 5.6).

## 3. Timeline

| When | Where | What happened | Can you trust it? |
|---|---|---|---|
| Dec 2025 | `7df04e6` (5 Colab notebooks) | Built the pipeline to test someone else's claimed "Sharpe ~1.3" (the notebook names the source as "joshuaaalampour"). Walk-forward result: Sharpe −0.56. All 12 sensitivity runs negative. | **Yes.** This is the honest core. Commit `79ebfd1` (3 May) deleted these notebooks from `main`; they survive only in history. |
| 2 Jan 2026 | `TDA_Revised_v12_SSRN_READY.pdf` | Paper built on the notebook result, plus statistics and claims the notebooks do not contain (section 6). | **No.** |
| 3–12 Jan | `main`, `thesis_expansion/`, `thesis_latex/` | An AI session wrote six "thesis expansion" phases and a LaTeX thesis in about a week. Much of it runs on simulated data. Commit `dc799b8` (7 Jan) is titled "Replace fabricated data with honest simulated values". | **No** (section 7). |
| 13 Jan – 6 Feb | `claude/review-tda-paper-gGfou` | Paper prepared for a Davidson submission. Its "main result" is the ρc ≈ 0.50 threshold, "validated" by a test on synthetic data that has the threshold built in. | **No** (sections 6–7). |
| Apr – May | `main` (PRs #2–#4) | Reframed as an "honest negative result": `risk_report.py` (Sharpe −0.77, alpha t ≈ −5) and "real-data" scripts. | **Partly.** The numbers reproduce, but the Fiedler "stress" label is inverted and the t ≈ −5 comes from costs (section 5.6). The thesis still claims ρc ≈ 0.50 and a "genuine" theory contribution. |
| 29 Jun | `claude/code-audit-1bo6sh` (`4181c52`) | Moved the simulated code into `legacy_simulated/`. Not merged into `main`. | The quarantine is correct. Its `HONEST_RESULTS.md` repeats "actively destroys value" and "timescale mismatch", which section 5 shows are wrong. |

The history is 148 commits: 127 by "Claude", 13 by you (uploads and merges), and 8 by
RevengeOfTheCheesecake (the original notebooks).

## 4. Was the idea good? 3/10

**What was reasonable.**

- TDA on correlation networks is a published area. Gidea & Katz (2018) is real and is the right
  starting point.
- Testing a claim you saw online with walk-forward validation and trading costs was the correct
  instinct.
- The December notebooks did that honestly.

**Why it was never going to work.**

1. **Cost arithmetic.** A 5-long/5-short book on 20 stocks, rebuilt daily, turns over 2–3× its size per
   day. At 5 bp that is 25–36% a year. To break even the signal needs a before-cost Sharpe of roughly
   1.2–1.7 on the most heavily traded stocks in the world. No known daily signal on those stocks comes
   close. You can
   check this before writing a line of code.
2. **The topology has nothing new to add.** The H1 statistics are a complicated, noisy function of a
   20×20 correlation matrix estimated from 60 days. There was no reason to expect them to carry more
   information than average correlation, and they do not (section 5.5). "Topology volatility" was an
   ad hoc formula, and it turned out to point the wrong way.
3. **The test could not distinguish anything.** Two years of daily data leave a Sharpe standard error
   of about ±0.75. Even a real edge could not have been confirmed.
4. **The starting claim was unverified.** "Sharpe ~1.3" came from an unreviewed source. The notebooks
   showed it did not replicate. Replicating that was the result, and it was small.

A version of the question worth asking: *does persistent homology of correlation networks predict
volatility or drawdowns better than average correlation and trailing volatility, out of sample, over
2000–2024?* That is a clean yes/no test. For this data, section 5.5 already suggests the answer is no.

---

## 5. Findings from re-running the code

### 5.1 The original backtest reproduces, and costs explain the loss

`audit/check_original.py` copies the Phase-4 notebook's logic and parameters exactly. It gets the
notebook's numbers: fold Sharpes −0.42 and −1.24, CAGR −15.24% and −11.82%, average daily cost 0.1433%
and 0.0582%, combined Sharpe −0.56. The rows below add controls the notebook and the papers never ran.
All rows cover the same 504 out-of-sample days (2022-03-30 to 2024-04-02) with 5 bp costs.

| Strategy | Sharpe before costs | Sharpe after costs | 95% CI (after costs) | p | Turnover/day | Cost drag/yr |
|---|---|---|---|---|---|---|
| **Original** (long + residual, topology filter) | +0.60 | **−0.56** | [−2.05, +0.93] | 0.43 | 2.0× | 25% |
| Same signal, no filter | +0.94 | −0.60 | [−2.10, +0.91] | 0.40 | 2.9× | 36% |
| Same signal, realised-vol filter | +0.86 | −0.59 | [−2.08, +0.91] | 0.41 | 2.2× | 27% |
| Direction flipped (true mean reversion), topology filter | −0.60 | −1.76 | [−3.97, +0.45] | 0.01 | 2.0× | 25% |

What this shows:

- **Costs, not the signal design, produced the loss.** The book turns over 2–2.9× its gross size per
  day. At 5 bp that is 25–36% a year, larger than any plausible gross return.
- **The paper's significance claims cannot be right.** Two years of daily data give a Sharpe standard
  error of about ±0.76. The correct 95% interval for −0.56 is [−2.05, +0.93], with p = 0.43. The paper
  reports [−0.64, −0.48], t = −14.3, p < 0.001. No code in the repo computes those figures.
- **The topology filter adds nothing.** Filter vs. no filter changes the after-cost Sharpe by 0.04. The
  small improvement comes from sitting in cash (fewer trades). The filter lowers the before-cost Sharpe
  from +0.94 to +0.60, so it removed profitable days.
- **"50% loss reduction from topology" is false.** The paper compares the TDA strategy's walk-forward
  Sharpe (−0.56) with a different signal (z-score reversal, −1.58) over a different period. On matched
  dates, the notebook's own sensitivity run of the TDA strategy scored −1.78, *worse* than the −1.58
  reversal baseline.

### 5.2 The signal has no edge over a longer history

`audit/check_long_history.py` runs the unfiltered signal over 2007–2024 on the 18 original names that
traded the whole period. Using today's winners favours the test, and it still fails.

| | Before costs | After 2 bp | After 5 bp |
|---|---|---|---|
| Sharpe 2007–2024 | −0.19 (t = −0.81) | −0.81 | −1.73 |

By year, the before-cost Sharpe ranges from −2.2 to +3.8 with no pattern. The positive 2022–24 test
result was noise.

### 5.3 The trade direction is described backwards

Residual = actual − smoothed (`TDA_Phase4_StrategyBacktest.ipynb`, `e = x - h`). A positive residual
means the stock beat its neighbours. The notebook goes long the largest residuals (`nlargest(n)` →
`+1/n`). That is a one-day momentum trade.

The v12 paper (page 6) says it goes "long positions in 5 assets with the highest positive residuals
(underperforming relative to correlation neighbors)". That is wrong. The paper then explains the
failure as "mean reversion fails in trending markets", and lists "use momentum instead" as the fix. The
strategy was already momentum.

In April, commit `f4af37f` flipped `risk_report.py` to true mean reversion. Flipping the original
strategy the same way makes it worse: −1.76 after costs (table above).

### 5.4 The topology filter labels calm markets as unstable

The Phase-3 notebook's rule is: flag a day if topology volatility is above its full-sample 75th
percentile. Here is what it selects (`audit/outputs/check_original.txt`):

| | SPY vol (20d, annualised) | SPY vol next 20d | Average stock correlation |
|---|---|---|---|
| Days flagged "unstable" | 12.8% | 13.3% | 0.24 |
| Days flagged "stable" | 18.0% | 17.9% | 0.35 |

- Flagged days are **calmer** than the rest, both at the time and over the next month.
- COVID crash, 15 Feb to 30 Apr 2020: **0 of 52 days flagged.** The loop count fell from 5 to 2 as
  average correlation rose from 0.24 to 0.78.
- Share of each year flagged: 1% in 2020, 50% in 2024.
- In the walk-forward test the filter flagged 0% of the 2022 bear market and 60% of the calm 2023–24
  rally. A plain realised-volatility filter did the reverse (51% and 0%).

The v12 paper's Figure 2 caption ("Successfully detected: COVID crash (March 2020…)") and its "the
topology signal lagged the regime shift" explanation are both wrong. The filter is inverted.

Two smaller problems:

- "Topology volatility" adds a standard deviation of a count (median 1.37) to a standard deviation of a
  lifetime sum (median 0.049). The loop count supplies about 97% of the value.
- The paper says the correlation graph "remains connected for >95% of trading days". It is connected on
  75.1% of days.

### 5.5 "Topology" mostly measures average correlation

- The H1 loop count and total persistence have rank correlations of −0.64 and −0.70 with the average
  pairwise correlation.
- The next-month volatility forecast was tested with non-overlapping 20-day windows. Adding topology
  volatility to trailing volatility moves R² from 0.339 to 0.341, with t = −0.57 on the topology
  term.

The H1 statistics are a deterministic function of a 20×20 correlation matrix estimated from 60 days. The
question the project needed to answer was whether they tell you anything that simpler summaries of the
same matrix (average correlation, largest eigenvalue) do not. In this data they do not, and the repo
never tests it.

### 5.6 `main`'s "verified" backtest: −0.77 and the t ≈ −5 alphas

`risk_report.py` reproduces the thesis's conclusion table exactly (`sec12_conclusion.tex:26–29`). For
example, Strategy A's alpha is −34.98% with t = −5.13, and Strategy D's Sharpe is −0.77. Three problems
change what those numbers mean (`audit/check_risk_report.py`):

**1. The "stress" label in Strategy D is inverted.** `risk_report.py:291` says "low Fiedler =
fragmented graph ≈ … stressed". The Fiedler value has rank correlation +0.81 to +0.89 with
within-sector average correlation. SPY volatility averages 12–13% on "stressed" (low-Fiedler) days and
18–20% on the other days.

| Strategy D variant, 2022–24 | Sharpe |
|---|---|
| As coded | −0.77 |
| Regime label flipped | +0.41 |
| No regime at all (always short the high-beta spread) | +0.22 |

This is not evidence of a working strategy. Flipping a sign after seeing the result is data-snooping,
and none of these differs significantly from zero over three years. The point is that the "best
negative result" is itself an artefact of the inverted label.

**2. The factor regression subtracts the risk-free rate from a zero-cost long/short**
(`risk_report.py:351`). That removes about 3.8% a year (the 2022–24 average) that was never earned or
paid.

**3. The t ≈ −5 alphas come from costs.**

| | Sharpe before costs | Alpha after costs, as coded | Alpha after costs, no RF | Alpha before costs, no RF |
|---|---|---|---|---|
| A: Laplacian + Fiedler | −0.57 | −35.0% (t −5.13) | −31.2% (t −4.56) | −7.2% (t −1.05) |
| B: Laplacian + vol filter | −0.01 | −31.5% (t −5.23) | −27.7% (t −4.60) | −0.9% (t −0.15) |
| C: Laplacian only | −0.38 | −42.2% (t −5.66) | −38.4% (t −5.14) | −5.7% (t −0.75) |
| D: Beta-spread + Fiedler | −0.67 | −14.0% (t −1.87) | −10.2% (t −1.37) | −8.9% (t −1.21) |

The thesis says "the signal actively destroys value" (`sec12_conclusion.tex:37`). It does not. Before
costs the alphas are indistinguishable from zero. Daily trading destroys the value. The thesis's other
claim, "the Fiedler filter reduces losses (A vs C), confirming topology's value as a risk indicator",
fails the same test: before costs the filter makes A worse than C (−0.57 vs −0.38). It only saves
trading costs by sitting in cash.

One more issue: the README calls this backtest "factor-regression-verified". With current
`pandas-datareader` the factor download fails, and the script prints N/A without stopping
(`risk_report.py:336–339`).

### 5.7 Smaller items

- **ML (`ml_integration_real.py`).** AUC is 0.49–0.51 for every model, which is a coin flip. The neural
  network predicts "up" every day (recall 1.000), so its F1 of 0.688 only reflects the base rate. The
  Sharpe column prints `nan` because the last row of each sector keeps a NaN next-day return
  (`ml_integration_real.py:116–118`).
- **Strategy variants (`strategy_variants_real.py`).** All four are negative after costs (−1.06 to
  −2.06).
- **The "scale mismatch" explanation measured itself wrong.** The April Colab re-run of Phase 3 (commit
  `54fe509`) computed a signal timescale of 50 days against a topology timescale of 30 days
  (R_temporal = 0.6×). That is the reverse of "daily signal vs. monthly topology". It also found the
  signal slightly *more* accurate on "unstable" days (51.8% vs 50.1%, p = 0.042).
- **January side experiments** (`151a55f`: VXX topology events, neural-net overlay). The VXX test has
  36 events, and its mean return sits inside the range of its own random baseline. Granger-causality
  p-values on overlapping series say nothing about tradability.

---

## 6. Claims that are false or have no source

"No source" means no code or output in the repo, in any branch or commit, produces the number.

### 6.1 January "SSRN-ready" paper (v12)

| Claim (page) | Status | Evidence |
|---|---|---|
| Sharpe −0.56, "95% CI [−0.64, −0.48]", t = −14.3, p < 0.001, Cohen's d 0.90 (p.11) | **Wrong; no source** | The correct interval is [−2.05, +0.93] with p = 0.43 (section 5.1). The same inflated precision appears on every row of Table 2. |
| Max drawdowns −44.28%, −48.12%, −52.45%, marked "estimated based on cumulative return patterns" (p.11) | **No source** | The notebooks never printed them. |
| "Topology filter reduces losses by 50%" (abstract, p.12, p.16) | **False** | It compares different signals over different periods. On matched dates the TDA strategy was worse (section 5.1). |
| Volatility filter −0.82 and correlation filter −0.71 vs. TDA −0.56 (p.16) | **No source** | No code in any version computes these. My re-run: realised-vol filter −0.59, topology −0.56, a difference well inside noise. |
| Trade "buys underperformers" / "mean reversion" (p.6, p.12, p.14) | **Wrong** | The code buys outperformers (section 5.3). |
| "Successfully detected: COVID crash" (Fig. 2 caption) | **False** | 0 of 52 crash days flagged (section 5.4). |
| Maximum drawdown "occurs in late 2024" (Fig. 3 caption) | **Impossible** | The walk-forward test ends on 2 April 2024. |
| Graph "connected for >95% of trading days" (p.5) | **False** | It is connected on 75.1% of days. |
| Data "cross-referenced with Bloomberg terminal data" (p.5) | **No evidence; self-contradictory** | Page 1 says the work was done "without … access to proprietary data". |
| Deflated Sharpe ratios; Bonferroni "all p < 0.001"; power "exceeding 0.95" (p.9) | **No source** | Nothing computes them, and the p-values they rely on are wrong. |
| Reproduction guide: `src/data_download.py`, `src/backtest_walkforward.py`, `src/generate_figures.py`, `requirements.txt`, tag `v1.0-final`, "all random seeds are fixed" (p.24) | **False** | None of these existed. The repo has never had a tag, and `requirements.txt` first appears in July 2026 in an unrelated project. |
| "Scale mismatch" is the "primary architectural flaw" (throughout) | **Untested** | Nothing isolates it. When it was measured in April, the numbers went the other way (section 5.7). |
| "1,494 observations inadequate for high-dimensional persistent homology" presented as a "quantification of sample-size requirements" (p.17) | **Confused** | Each persistence diagram comes from a 60-day window of 20 stocks. The 1,494-day total is not the relevant sample, and nothing was quantified. |
| AI tools "used only for code debugging and syntax optimization" (p.1, p.23) | **Inaccurate** | See 6.3. |

### 6.2 Citations

Of 41 unique references across the v12 paper, the thesis and the Davidson paper:

- **28 are real.**
- **8 are misattributed.**
- **5 do not appear to exist.** No paper matches the title, and the cited volume or page belongs to an
  unrelated article.

Every problem is in the recent TDA-in-finance citations, which are the ones used to position the work.
I resolved the DOIs myself.

| As cited | What the DOI or volume actually is |
|---|---|
| Bi, Zhao, Cui & Liu (2023), *Chaos Solitons Fractals* 171:113457 | A paper on plant growth under toxicity (Dipesh & Kumar). The closest real paper is Valdivia (2023), *Quant. Finance & Econ.* 7(3). |
| Maji, Arora & Verma (2023), "Why does TDA detect financial bubbles?", *Quant. Finance* 23(7) | The DOI is a paper on VIX-linked variable annuities (MacKay, Vachon & Cui). The real paper is Akingbade, Gidea, Manzi & Nateghi (2024), *CNSNS* 128:107665. |
| Majumdar, Mukherjee & Mitra (2024), *J. Econometrics* 238(1) | The DOI does not exist. The real paper is Souto (2023), *J. Finance & Data Sci.* 9:100107. |
| Goel, Filipović & Pasricha (2025), *JFE*, forthcoming | Real paper, wrong journal: *Quantitative Finance* 25(8), 2025. |
| Frazzini, Israel & Moskowitz (2018), *FAJ* 74(2) / *JFE* | An SSRN working paper; not in either journal. |
| Meng, Xu, Zhou & Sornette (2021), *Physica A* 572:125897 (thesis) | **Does not exist.** That article number belongs to a paper on scale-free network models. |
| Macocco et al. (2023), *Finance Res. Lett.* 56:104116 (thesis) | **Does not exist.** That number belongs to a paper on Twitter and metaverse stocks. |
| Macocco, Wan & Bassoli (2023), *ESWA* 226:120213 (Davidson) | **Does not exist.** |
| Yen & Yen (2012), *Physica A* 391(22):5563 (thesis) | **Does not exist.** Those pages hold two other papers. |
| Majumdar & Lozano-Duran (2024), *J. Computational Finance* 27(3) (Davidson) | **Does not exist.** |
| Yen & Cheong (2023), *PLOS ONE* 18(1):e0279925 (Davidson) | That article number is a veterinary parasitology paper. The same authors' real work is in *Front. Phys.* (2021). |

### 6.3 AI-use statement

The v12 paper and the current thesis (`thesis_main.tex:79`) say AI tools were "used only for code
debugging and syntax optimization". The thesis also says the Section-11 bound and ρc ≈ 0.50 "represent
original contributions developed through this independent research program" (`thesis_main.tex:83`).

The git history does not support that. 127 of 148 commits are authored by "Claude", including every
thesis section, the six expansion phases, the theory section and the verification scripts. Your commits
are file uploads and pull-request merges. Whatever the division of labour was in practice, the
statement as written is inaccurate. An inaccurate AI disclosure is an integrity problem on its own,
separate from the content.

## 7. The thesis on `main`

**The "honest reframe" is not clean.** The current abstract (`thesis_main.tex:100–110`) still claims:

| Claim | Status |
|---|---|
| "Sharpe −0.56, p < 0.001" | **Wrong.** p = 0.43. |
| A "critical correlation threshold ρc ≈ 0.50" is the discovery | **Built into synthetic data** (below). |
| "Topology provides measurable risk-reduction benefit (max drawdown reduced 7–13 points)" | A vs C is 7.4 points, and it comes from sitting in cash. **The 12.7 is the volatility filter (B), not topology.** |
| "Calibrated simulation across 11 market scenarios demonstrates the pattern generalizes" | The simulation was built to contain the pattern. |
| "Machine learning confirms topology captures regime structure (F1 = 0.578)" | **Computed on synthetic regime-switching data.** Section 10's own disclosure (`sec10_ml.tex:14`) says so; the abstract leaves that out. On real data, `ml_integration_real.py` gives AUC 0.49–0.51. |
| "α t-stats −5.1 to −1.9" | Inflated by subtracting the risk-free rate, and driven by costs (section 5.6). |
| A "correlation-stability bound CV(H1) ≤ α/√(ρ(1−ρ))" | **Not a result** (below). |

**The ρc ≈ 0.50 "discovery" was planted.** `validation_tests.py:113–131`:

- Seeds the random generator.
- Generates fake data in which the CV of H1 jumps by 0.2 exactly at ρ = 0.50 (`if rho < 0.50: cv = 0.65 + …  else: cv = 0.45 − …`).
- Runs a Chow test for a break at 0.50 and prints "Results to add to paper".

Re-running it reproduces the Davidson paper's "main result" exactly: F = 32.33, p < 0.001, bootstrap CI
[0.45, 0.65]. On that branch, `thesis_main.tex:115` states the result as "statistically validated via
Chow test (F = 32.33, p < 0.001)". It is the output of a test that found a break someone put there. The
"international validation" in the same script (`validation_tests.py:258+`) downloads 5 country ETFs per
market, skips every market for having too few tickers, and reports only US sectors.

**The "theorem" is not a theorem** (`sec11_theory.tex:160–292`):

1. **Step 3 contradicts the result.** It derives CV(H1) ∝ √(ρ(1−ρ)), then "inverts" this into
   CV(H1) ≤ α/√(ρ(1−ρ)). That step reverses the relationship, and no argument is given for it.
2. **Step 1 has eigenvalue dispersion backwards.** For high ρ, a correlation matrix has one large
   eigenvalue (about nρ) and n−1 small ones. That is *high* dispersion, not low.
3. **"Var[C_ij] ≈ ρ(1−ρ), the variance of a Bernoulli-like variable"** has no basis. Correlations are
   not Bernoulli variables.
4. **The text contradicts its own table.** It calls the high-ρ bound "small, tight", but α/0.3 > α/0.46,
   and its own table gives the largest bound (1.17) at ρ = 0.9.
5. **α is "determined empirically"** by fitting to simulated data. A bound with a free constant fitted
   to the data it is tested on will always "hold".
6. **The ρc derivation fails, then asserts ρc ≈ 0.50 anyway.** It reaches "ρ(1−ρ) ≥ 0.276 (impossible,
   max = 0.25)" and then states "Adjusting for conservative bound … ρc ≈ 0.50 (empirically validated)".
   Because ρ(1−ρ) is symmetric around 0.5, no "below this ρ" threshold can follow from it.

The README's statement that "the mathematical framework (correlation-CV bound, spectral gap analysis,
ghost loop detection) is a genuine contribution" is false. What remains is standard background (the
Marchenko–Pastur law, the Fiedler value), correctly quoted, plus claims resting on simulated data.

---

## 8. Other branches (not audited in depth)

- **`claude/modest-clarke-nma4gh`**: a USD/JPY factor study. It is a separate project and was not
  reviewed.
- **`claude/code-audit-1bo6sh`**: only commit `4181c52` concerns TDA (the quarantine above). The other 49
  commits are separate projects: a multi-asset trend strategy ("RP+Trend"), weather, Wikipedia-attention,
  PEAD, a short-squeeze monitor, a TradingView indicator and a forward test. I only checked them for
  scope. Two points matter if you plan to use that work:
  - **It is a paper account, not real money** (IBKR `DU…` prefix, `forward_track.py:15`). The status
    page reads as if it were live ("LIVE", "before any money moved"); only the footer says "paper".
  - **The headline "Sharpe 1.16 vs 0.74, p = 0.002" omits the T-bill rate.** The Sharpe is computed on
    total returns (`equity_factor/torture_test.py:18`, `backtest.py:112`). For a strategy that often holds half its
    money or more in T-bills, that inflates the Sharpe far more than it does for stocks. Its own
    CAGR and volatility give roughly 0.7 vs 0.6 once T-bills are subtracted. A helper re-running the
    branch code got 0.65 vs 0.55 (p ≈ 0.24), which is not significant. The kill criteria in `9e80f0c`
    were written after month 1 had been observed.

## 9. What to do now

1. **Stop using the v12 paper, the Davidson paper and the thesis as they stand.** They contain made-up
   statistics, results from synthetic data presented as findings, five non-existent citations and an
   inaccurate AI-use statement. If any version was submitted or posted (SSRN, Davidson, a school, a
   university application), send a correction or withdraw it. The fact that an AI tool wrote the text
   does not change whose name is on it.
2. **Put the December notebooks back on `main`** (`git checkout 79ebfd1^ -- '*.ipynb'`; this includes the
   April re-run of Phase 3). They are the
   only part that is both honest and correct.
3. **If you want to keep one thing from this project,** write a two-page note with this content: "I
   tried to replicate a claimed Sharpe 1.3 TDA strategy. It fails out of sample (Sharpe −0.56). Before
   costs it is slightly positive but insignificant, and over 2007–2024 the signal has no edge (Sharpe
   −0.19). The losses come from 2–3× daily turnover. The persistent-homology filter selects calm
   markets and missed the COVID crash." Every number in that note is reproduced in `audit/`. It is a
   small but honest result.
4. **Remove or label the fabricating code on `main`:** test 1 in `validation_tests.py`, the
   non-ripser fallbacks in `null_model_test.py`, and `generate_all_thesis_figures.py`. Correct the
   README, which calls `validation_tests.py` "cross-market validation on real FTSE/DAX/Nikkei" data.
   That test downloads 5 country ETFs per market, skips every market, and reports only US sectors.
5. **For the next project, three habits would have caught every problem here:**
   - Estimate turnover × cost before building anything.
   - Work out how many years of data you need to tell a Sharpe of 0.5 from 0.
   - Put a number in a document only if you can point to the committed output file that produced it.

## 10. Reproducing this audit

```bash
pip install yfinance ripser pandas numpy scipy statsmodels pandas-datareader
cd audit
python check_original.py        # section 5.1, 5.3–5.5  (~15 s)
python check_long_history.py    # section 5.2
python check_risk_report.py     # section 5.6 (imports ../risk_report.py; downloads Ken French factors)
```

The saved output from my runs is in `audit/outputs/`. Yahoo revises adjusted prices from time to time,
so later runs can differ in the third decimal.
