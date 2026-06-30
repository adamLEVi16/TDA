"""
THE TEST: does abnormal public attention (Wikipedia pageviews) to a consumer
brand predict its OWN stock's FUTURE return -- i.e. is attention "new info" the
market is slow to price?

Pre-committed hypotheses (written before running anything), from the attention
literature (Da, Engelberg & Gao 2011, "In Search of Attention"):
  H1 (price-pressure): a spike in abnormal attention this week -> retail buying
      -> POSITIVE return next week. Long high-attention, short low-attention.
  H2 (reversal): that attention-driven pop REVERSES over the following month
      -> NEGATIVE return at a ~4-week horizon.

Signal (no look-ahead): ASVI_t = log(views_t) - log(median of the prior 8 weeks'
views). The current week's views are fully observed by Friday; the median uses
strictly PRIOR weeks. The signal at week t predicts the return in week t+1.

To isolate the CROSS-SECTIONAL effect (not a market-wide attention wave that
moves everything together), ASVI is z-scored ACROSS tickers within each week.

Tests, in order of how much they matter:
  1. contemporaneous panel reg (sanity: is it already priced same-week?)
  2. forward panel reg, cluster-robust SE by week (the actual effect)
  3. time-shift placebo (honest empirical null from the data itself)
  4. tradable long/short basket  <-- the decisive test; a coefficient that
     can't become a positive-Sharpe trade is not a signal.
"""
import warnings; warnings.filterwarnings("ignore")
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "equity_factor"))
import numpy as np, pandas as pd
import statsmodels.api as sm
import data as D                # equity_factor's urllib Yahoo fetcher
import wiki_data as WD
from universe import TICKER_ARTICLE

START = "2015-06-01"
MED_WEEKS = 8


def stock_weekly_returns():
    px = D.get_prices(tickers=list(TICKER_ARTICLE.keys()), start=START, verbose=True)
    return px.resample("W-FRI").last().pct_change()


def asvi(pv_series):
    """Abnormal attention: log(weekly views) - log(trailing 8-week median).
    Strictly trailing -> no look-ahead."""
    wk = pv_series.resample("W-FRI").mean()
    logv = np.log(wk.clip(lower=1))
    trailing_med = logv.shift(1).rolling(MED_WEEKS).median()
    return (logv - trailing_med).rename("asvi")


def build_panel():
    pv = WD.get_pageviews(list(TICKER_ARTICLE.values()), verbose=True)
    rets = stock_weekly_returns()
    art2tk = {v: k for k, v in TICKER_ARTICLE.items()}

    rows = []
    for article, s in pv.items():
        tk = art2tk[article]
        if tk not in rets.columns:
            print(f"  [panel] skip {tk}: no price data")
            continue
        a = asvi(s)
        r = rets[tk]
        idx = a.index.intersection(r.index)
        df = pd.DataFrame({
            "asvi": a.reindex(idx),
            "ret": r.reindex(idx),
            "ret_fwd1": r.reindex(idx).shift(-1),
            "ret_fwd4": r.reindex(idx).shift(-1).rolling(4).sum().shift(-3),  # weeks t+1..t+4
        }).dropna(subset=["asvi", "ret"])
        df["ticker"] = tk
        df["week"] = df.index
        if len(df) < 100:
            print(f"  [panel] skip {tk}: only {len(df)} weeks")
            continue
        rows.append(df)

    panel = pd.concat(rows)
    # cross-sectional z-score of ASVI within each week (isolate the cross-section)
    panel["asvi_z"] = panel.groupby("week")["asvi"].transform(
        lambda x: (x - x.mean()) / x.std() if x.std() > 0 else x * 0)
    return panel.reset_index(drop=True)


def cluster_reg(panel, yvar, xvar, label):
    d = panel.dropna(subset=[yvar, xvar]).copy()
    X = sm.add_constant(d[[xvar]])
    mdl = sm.OLS(d[yvar], X).fit(cov_type="cluster", cov_kwds={"groups": d["week"]})
    print(f"\n  {label}  (n={len(d)} ticker-weeks, {d['ticker'].nunique()} tk, {d['week'].nunique()} wk)")
    for v in ["const", xvar]:
        print(f"    {v:<8} coef={mdl.params[v]:+.5f}  t={mdl.tvalues[v]:+.2f}  p={mdl.pvalues[v]:.3f}")
    return mdl


def shift_placebo(panel, yvar, xvar="asvi_z", n=1000, seed=11):
    rng = np.random.default_rng(seed)
    d = panel.dropna(subset=[yvar, xvar])
    real = sm.OLS(d[yvar], sm.add_constant(d[[xvar]])).fit().params[xvar]
    groups = {tk: g.index.values for tk, g in d.groupby("ticker")}
    null = np.empty(n)
    for k in range(n):
        sh = d.copy()
        for tk, gi in groups.items():
            s = rng.integers(1, len(gi) - 1)
            sh.loc[gi, xvar] = np.roll(d.loc[gi, xvar].values, s)
        null[k] = sm.OLS(sh[yvar], sm.add_constant(sh[[xvar]])).fit().params[xvar]
    p = (np.abs(null) >= abs(real)).mean()
    print(f"\n  PLACEBO ({yvar}, n={n} shifts): real={real:+.5f}  "
          f"null95=[{np.percentile(null,2.5):+.5f},{np.percentile(null,97.5):+.5f}]  p={p:.3f}")
    return p


def basket(panel, yvar="ret_fwd1", n_legs=5, sign=+1):
    """Long high-ASVI quartile, short low-ASVI quartile (sign=+1 = H1). Held over
    the forward window. Dollar-neutral, so no benchmark needed."""
    rows = []
    for wk, g in panel.dropna(subset=[yvar, "asvi"]).groupby("week"):
        if len(g) < 2 * n_legs:
            continue
        gs = g.sort_values("asvi")
        lo, hi = gs.iloc[:n_legs], gs.iloc[-n_legs:]
        rows.append({"week": wk, "ret": sign * (hi[yvar].mean() - lo[yvar].mean())})
    return pd.DataFrame(rows).set_index("week")["ret"].dropna()


def report_basket(s, label, periods_per_yr=52):
    ann = s.mean() * periods_per_yr
    vol = s.std() * np.sqrt(periods_per_yr)
    sharpe = ann / vol if vol > 0 else np.nan
    print(f"  {label:<46} {len(s)} wk  ann={ann:+.2%}  vol={vol:.2%}  "
          f"Sharpe={sharpe:+.2f}  hit={float((s>0).mean()):.0%}")
    return sharpe


def main():
    panel = build_panel()
    panel.to_csv(os.path.join(os.path.dirname(__file__), "panel.csv"), index=False)
    print(f"\n{'='*80}\n  PANEL: {len(panel)} ticker-weeks, {panel['ticker'].nunique()} tickers, "
          f"{panel['week'].min().date()} -> {panel['week'].max().date()}\n{'='*80}")

    print("\n--- [1] CONTEMPORANEOUS (same week): is it already priced? ---")
    cluster_reg(panel, "ret", "asvi_z", "attention_t -> return_t")

    print("\n--- [2] FORWARD: does attention LEAD the stock? ---")
    cluster_reg(panel, "ret_fwd1", "asvi_z", "attention_t -> return_{t+1}  (H1: price-pressure)")
    cluster_reg(panel, "ret_fwd4", "asvi_z", "attention_t -> return_{t+1..t+4}  (H2: reversal)")

    print("\n--- [3] PLACEBO ---")
    shift_placebo(panel, "ret_fwd1")
    shift_placebo(panel, "ret_fwd4")

    print(f"\n{'='*80}\n  [4] TRADABLE LONG/SHORT BASKET (the decisive test)\n{'='*80}")
    report_basket(basket(panel, "ret_fwd1", sign=+1), "H1 long-high/short-low attention, next week")
    report_basket(basket(panel, "ret_fwd1", sign=-1), "  (opposite sign, for reference)")
    report_basket(basket(panel, "ret_fwd4", sign=-1) / 4, "H2 reversal: short-high/long-low, t+1..t+4")


if __name__ == "__main__":
    main()
