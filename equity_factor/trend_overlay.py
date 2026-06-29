"""
Trend / regime overlay — does simple risk management beat buy-and-hold?

Rule (canonical, 1 parameter): at each month-end, if price > its 10-month SMA
(~200-day), be fully invested next month; else hold cash. Signal uses only data
through month m; position held in m+1 (no look-ahead).

Why this matters: stock SELECTION added no value (see FINDINGS.md), but the goal
is risk-adjusted return. A trend filter attacks drawdowns directly. Tested on:
  - SPY  → SURVIVORSHIP-FREE answer to "can a rule beat buy-and-hold?"
  - EW universe → same overlay on the equal-weight book
Cash earns 0% here (a conservative assumption; real T-bills would only help).
"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import data as D
from backtest import metrics, run as run_xs

SMA_MONTHS = 10

def overlay(price_m, ret_m, label):
    sma = price_m.rolling(SMA_MONTHS).mean()
    invested = (price_m > sma).shift(1).fillna(False)   # decide at m, hold m+1
    strat = ret_m.where(invested, 0.0)
    return strat.rename(label), invested

if __name__ == "__main__":
    px = D.get_prices(verbose=False)
    spy_m = px[D.BENCHMARK].resample("ME").last()
    spy_ret = spy_m.pct_change()

    # equal-weight universe monthly return (same survivor set as the factor book)
    stocks_m = px.drop(columns=[D.BENCHMARK]).resample("ME").last()
    ew_ret = stocks_m.pct_change().mean(axis=1)

    spy_tf, inv = overlay(spy_m, spy_ret, "SPY + trend overlay")
    ew_tf, _ = overlay(stocks_m.mean(axis=1), ew_ret, "EW + trend overlay")

    # align all on the traded window
    idx = spy_tf.dropna().index
    series = {
        "SPY buy & hold": spy_ret.reindex(idx),
        "SPY + trend overlay": spy_tf.reindex(idx),
        "EW buy & hold": ew_ret.reindex(idx),
        "EW + trend overlay": ew_tf.reindex(idx),
    }

    print("=" * 70)
    print(f"  TREND OVERLAY ({SMA_MONTHS}-month SMA)  —  cash when below trend")
    print(f"  {idx.min().date()} → {idx.max().date()}  ({len(idx)} months)")
    print("=" * 70)
    print(f"{'Portfolio':<26}{'CAGR':>9}{'Vol':>8}{'Sharpe':>8}{'MaxDD':>9}{'Hit':>7}")
    print("-" * 70)
    for name, r in series.items():
        m = metrics(r)
        print(f"{name:<26}{m['CAGR']:>8.1%}{m['Vol']:>8.1%}{m['Sharpe']:>8.2f}"
              f"{m['MaxDD']:>9.1%}{m['Hit']:>7.0%}")
    print("-" * 70)
    pct_invested = inv.reindex(idx).mean()
    print(f"Time invested (SPY overlay): {pct_invested:.0%}   "
          f"(rest in cash)")
