"""
Long price history (2004+) for the PEAD study, cached in a SEPARATE namespace
(px04_*) so it cannot collide with the 2015+ cache used by the weekly attention
tests (stale-cache range bugs bit us once before in equity_factor).
"""
import os, sys, time
import pandas as pd
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "equity_factor"))
import data as D

CACHE_DIR = os.path.join(os.path.dirname(__file__), "cache")
START, END = "2004-01-01", "2024-12-31"


def get_prices_long(tickers, verbose=True):
    series = {}
    for tk in tickers:
        fp = os.path.join(CACHE_DIR, f"px04_{tk}.csv")
        if os.path.exists(fp):
            s = pd.read_csv(fp, index_col=0, parse_dates=True).iloc[:, 0]
            s.name = tk
        else:
            s = D._fetch_one(tk, START, END)
            time.sleep(0.25)
            if s is not None and len(s) > 0:
                s.to_frame().to_csv(fp)
        if s is not None and len(s) > 0:
            series[tk] = s
            if verbose:
                print(f"  [px04] {tk:<6} {s.index.min().date()} -> {s.index.max().date()} ({len(s)}d)")
        elif verbose:
            print(f"  [px04] {tk:<6} MISSING")
    px = pd.concat(series.values(), axis=1)
    px.columns = list(series.keys())
    return px.sort_index()


if __name__ == "__main__":
    from universe import TICKER_ARTICLE
    px = get_prices_long(list(TICKER_ARTICLE.keys()) + ["SPY"])
    print(f"\n{px.shape[1]} tickers, {px.shape[0]} days")
