"""
Multi-asset trend + risk-parity ("Global Tactical Asset Allocation").

Universe: liquid ETFs across asset classes (survivorship-FREE — real instruments).
Each month-end m (decide at m, hold m+1, no look-ahead):
  - trend filter: asset eligible only if price > its 10-month SMA, else that
    sleeve goes to cash;
  - risk weighting: inverse-volatility (trailing 12m) across the eligible assets,
    normalized by the all-asset inverse-vol sum so risk-off = partial/full cash.
Cash earns 0% (conservative; real T-bills would only help).

Two strategy variants + two benchmarks:
  RP+Trend  = inverse-vol, trend-to-cash
  EW+Trend  = equal-weight eligible assets, trend-to-cash (Faber GTAA)
  SPY       = 100% equity buy & hold
  60/40     = 60% SPY / 40% IEF, monthly rebalanced
Win condition: higher Sharpe AND smaller drawdown than both benchmarks.
"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import data as D
from backtest import metrics

ASSETS = ["SPY", "EFA", "EEM", "TLT", "IEF", "GLD", "DBC", "VNQ"]
SMA_MONTHS = 10
VOL_WIN = 12
COST_BPS = 10

def load():
    px = D.get_prices(tickers=ASSETS, start="2005-01-01", end="2024-12-31", verbose=False)
    m = px.resample("ME").last()
    # restrict to common history (all assets present)
    m = m.dropna()
    return m

def run():
    mpx = load()
    mret = mpx.pct_change()
    sma = mpx.rolling(SMA_MONTHS).mean()
    vol = mret.rolling(VOL_WIN).std()
    dates = mpx.index

    prev = pd.Series(0.0, index=ASSETS)
    rows = []
    for i in range(1, len(dates)):
        m, hold = dates[i - 1], dates[i]
        trend_on = (mpx.loc[m] > sma.loc[m])
        v = vol.loc[m]
        if v.isna().any() or trend_on.isna().any():
            continue
        inv = 1.0 / v
        denom = inv.sum()                       # all-asset inverse-vol
        w_rp = (inv / denom).where(trend_on, 0.0)          # risk-off -> cash
        on = trend_on[trend_on].index
        w_ew = pd.Series(0.0, index=ASSETS)
        if len(on) > 0:
            w_ew[on] = 1.0 / len(ASSETS)        # 1/N, off sleeves -> cash

        r = mret.loc[hold]
        turn = (w_rp - prev).abs().sum()
        rp_ret = float((w_rp * r).sum()) - turn * COST_BPS / 1e4
        ew_ret = float((w_ew * r).sum())        # (cost approx on RP only for brevity)
        prev = w_rp
        rows.append({"date": hold, "RP+Trend": rp_ret, "EW+Trend": ew_ret,
                     "SPY": float(r["SPY"]),
                     "60/40": float(0.6 * r["SPY"] + 0.4 * r["IEF"])})
    return pd.DataFrame(rows).set_index("date")

def constant_leverage(rp_ret, lev=2.0, fin_rate=0.04):
    """Scale a positive-Sharpe strategy by a FIXED leverage (preserves Sharpe,
    minus financing). The honest way to raise the risk budget — unlike dynamic
    vol-targeting, which backfired here by levering up into drawdowns."""
    financing = max(lev - 1, 0) * fin_rate / 12
    return (lev * rp_ret - financing).rename(f"RP+Trend ({lev:g}x const)")

if __name__ == "__main__":
    bt = run()
    bt["RP+Trend (2x const)"] = constant_leverage(bt["RP+Trend"], 2.0)
    print("=" * 72)
    print(f"  MULTI-ASSET TREND + RISK PARITY  —  {len(ASSETS)} ETFs, "
          f"{SMA_MONTHS}m SMA, inv-vol")
    print(f"  {bt.index.min().date()} → {bt.index.max().date()}  ({len(bt)} months)"
          f"   cost={COST_BPS}bps/side")
    print("=" * 72)
    print(f"{'Portfolio':<28}{'CAGR':>9}{'Vol':>8}{'Sharpe':>8}{'MaxDD':>9}{'Hit':>7}{'corrSPY':>9}")
    print("-" * 72)
    for col in ["RP+Trend", "RP+Trend (2x const)", "EW+Trend", "60/40", "SPY"]:
        mm = metrics(bt[col].dropna())
        c = bt[col].corr(bt["SPY"])
        print(f"{col:<28}{mm['CAGR']:>8.1%}{mm['Vol']:>8.1%}{mm['Sharpe']:>8.2f}"
              f"{mm['MaxDD']:>9.1%}{mm['Hit']:>7.0%}{c:>9.2f}")
    print("-" * 72)
