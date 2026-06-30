"""
THE TEST: does a retailer's local weather anomaly carry information about its
OWN future stock return that the market hasn't priced in yet?

Two hypotheses, pre-committed (written before running anything):
  H1 (precipitation): more rain than the seasonal norm this week at a
      retailer's HQ metro -> worse foot traffic -> the market underreacts
      because nobody (except Alan, with his paid weather reports) is tracking
      regional weather vs individual tickers -> negative return, possibly
      LAGGED into the following week if it isn't priced same-day.
  H2 (temperature extremity): bigger |deviation from seasonal normal temp|
      (too hot or too cold keeps people inside) -> worse foot traffic ->
      same contemporaneous/lag logic.

Method: pool all 14 tickers into one panel, standardize each ticker's weather
anomaly to a z-score (so a "bad week" means the same thing whether the metro
is Seattle or Dallas), and regress next-week return on this week's anomalies,
clustering standard errors by week (weather and market moves are correlated
across tickers in the same week). Reported alongside a SAME-week regression
(is this already priced same-day, i.e. no edge) and a time-shift PLACEBO test
that builds an honest null distribution from the data itself, rather than
trusting OLS asymptotics on a short, autocorrelated panel.
"""
import warnings; warnings.filterwarnings("ignore")
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "equity_factor"))
import numpy as np, pandas as pd
import statsmodels.api as sm
import data as D                # equity_factor's urllib Yahoo fetcher
import weather_data as WD
import anomaly as SIG
from universe import TICKER_METRO

START = "1999-01-01"


def stock_weekly_returns():
    px = D.get_prices(tickers=list(TICKER_METRO.keys()), start=START, verbose=True)
    wk = px.resample("W-FRI").last()
    return wk.pct_change()


def build_panel():
    wx = WD.get_weather(start="1995-01-01", verbose=True)
    anoms = SIG.build_anomalies(wx)
    rets = stock_weekly_returns()

    rows = []
    for tk, metro in TICKER_METRO.items():
        if tk not in rets.columns or metro not in anoms:
            print(f"  [panel] skip {tk}: missing data")
            continue
        r = rets[tk].dropna()
        a = anoms[metro]
        idx = r.index.intersection(a.index)
        if len(idx) < 100:
            print(f"  [panel] skip {tk}: only {len(idx)} overlapping weeks")
            continue
        df = pd.DataFrame({
            "ret": r.reindex(idx),
            "ret_fwd": r.reindex(idx).shift(-1),
            "tanom": a["tanom"].reindex(idx),
            "panom": a["panom"].reindex(idx),
        }).dropna()
        df["tanom_z"] = (df["tanom"] - df["tanom"].mean()) / df["tanom"].std()
        df["panom_z"] = (df["panom"] - df["panom"].mean()) / df["panom"].std()
        # H2 as literally stated = EXTREMITY (|deviation|), not signed deviation.
        # Built as |tanom_z| of the already-standardized series (mean removed
        # first, so this is a true two-sided extremity measure).
        df["textreme_z"] = df["tanom_z"].abs()
        df["ticker"] = tk
        df["week"] = df.index
        rows.append(df)
    panel = pd.concat(rows).reset_index(drop=True)
    return panel


def cluster_regression(panel, yvar, xvars, label):
    d = panel.dropna(subset=[yvar] + xvars).copy()
    X = sm.add_constant(d[xvars])
    mdl = sm.OLS(d[yvar], X).fit(cov_type="cluster", cov_kwds={"groups": d["week"]})
    print(f"\n  {label}  (n={len(d)} ticker-weeks, {d['ticker'].nunique()} tickers, "
          f"{d['week'].nunique()} weeks)")
    for v in ["const"] + xvars:
        print(f"    {v:<10} coef={mdl.params[v]:+.5f}  t={mdl.tvalues[v]:+.2f}  "
              f"p={mdl.pvalues[v]:.3f}")
    return mdl


def shift_placebo(panel, yvar="ret_fwd", xvars=("panom_z", "tanom_z"), n=1000, seed=7):
    """Null distribution from circularly shifting each ticker's weather series
    by a random offset (destroys true weather<->return alignment but keeps
    each series' own autocorrelation/seasonality intact)."""
    rng = np.random.default_rng(seed)
    xvars = list(xvars)
    real = sm.OLS(panel[yvar], sm.add_constant(panel[xvars])).fit()
    real_coefs = real.params[xvars]

    null_coefs = {v: np.empty(n) for v in xvars}
    groups = {tk: grp.index.values for tk, grp in panel.groupby("ticker")}
    for k in range(n):
        shifted = panel.copy()
        for tk, gidx in groups.items():
            s = rng.integers(1, len(gidx) - 1)
            for v in xvars:
                shifted.loc[gidx, v] = np.roll(panel.loc[gidx, v].values, s)
        m = sm.OLS(shifted[yvar], sm.add_constant(shifted[xvars])).fit()
        for v in xvars:
            null_coefs[v][k] = m.params[v]

    print(f"\n  TIME-SHIFT PLACEBO TEST  (n={n} shuffles, real weather<->return link destroyed)")
    for v in xvars:
        p = (np.abs(null_coefs[v]) >= abs(real_coefs[v])).mean()
        print(f"    {v:<10} real coef={real_coefs[v]:+.5f}   "
              f"null range [{np.percentile(null_coefs[v],2.5):+.5f}, "
              f"{np.percentile(null_coefs[v],97.5):+.5f}]   "
              f"empirical p={p:.3f}")
    return null_coefs


def basket_strategy(panel, score_var, n_legs=4):
    """Cross-sectional weekly sort: long the n_legs tickers with the most
    BENIGN weather (lowest |score|... or for panom_z, lowest=best, since more
    rain = worse), short the n_legs worst. Pure long-short, no benchmark
    needed since it's market/sector neutral by construction (equal $ each side)."""
    rows = []
    for wk, grp in panel.groupby("week"):
        if len(grp) < 2 * n_legs:
            continue
        g = grp.sort_values(score_var)
        worst = g.iloc[:n_legs]   # highest panom_z/tanom_z = worst weather -> short
        best = g.iloc[-n_legs:]   # lowest = best weather -> long
        ret = best["ret_fwd"].mean() - worst["ret_fwd"].mean()
        rows.append({"week": wk, "ret": ret})
    s = pd.DataFrame(rows).set_index("week")["ret"].dropna()
    return s


def main():
    panel = build_panel()
    panel.to_csv(os.path.join(os.path.dirname(__file__), "panel.csv"), index=False)
    print(f"\n{'='*78}\n  PANEL BUILT: {len(panel)} ticker-weeks, "
          f"{panel['ticker'].nunique()} tickers, "
          f"{panel['week'].min().date()} -> {panel['week'].max().date()}\n{'='*78}")

    print("\n--- CONTEMPORANEOUS (same week) -- sanity check: does weather even ")
    print("    correlate with the SAME week's return? (no edge if so -- already priced) ---")
    cluster_regression(panel, "ret", ["panom_z", "tanom_z"], "weather_t -> return_t (directional)")
    cluster_regression(panel, "ret", ["panom_z", "textreme_z"], "weather_t -> return_t (H2 as literally stated: extremity)")

    print("\n--- THE ACTUAL TEST: forward (next week) -- is there a LAG? ---")
    cluster_regression(panel, "ret_fwd", ["panom_z", "tanom_z"],
                        "weather_t -> return_{t+1} (directional)")
    cluster_regression(panel, "ret_fwd", ["panom_z", "textreme_z"],
                        "weather_t -> return_{t+1} (H2 as literally stated: extremity)")

    shift_placebo(panel, xvars=("panom_z", "tanom_z"))
    shift_placebo(panel, xvars=("panom_z", "textreme_z"))

    print(f"\n{'='*78}\n  BASKET LONG/SHORT: long best-weather quartile, short worst-weather\n{'='*78}")
    for score in ["panom_z", "tanom_z", "textreme_z"]:
        strat = basket_strategy(panel, score)
        ann = strat.mean() * 52
        vol = strat.std() * np.sqrt(52)
        sharpe = ann / vol if vol > 0 else np.nan
        print(f"  sort on {score:<10}: {len(strat)} weeks, "
              f"ann.ret={ann:+.2%}  ann.vol={vol:.2%}  Sharpe={sharpe:+.2f}  "
              f"hit={float((strat>0).mean()):.0%}")


if __name__ == "__main__":
    main()
