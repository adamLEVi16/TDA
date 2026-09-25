"""VERIFIER: fresh yfinance download of a subset of tickers; compare monthly returns 2007-2024 to builder's cached etf_px.csv."""
from pathlib import Path
import numpy as np, pandas as pd, yfinance as yf
HERE = Path(__file__).parent
T = ["SPY", "XLK", "XLE", "EWZ", "TLT", "GLD", "HYG", "DBC", "SHY", "EEM"]
cache = HERE / "cache" / "yf_fresh.csv"
if cache.exists():
    fr = pd.read_csv(cache, index_col=0, parse_dates=True)
else:
    fr = yf.download(T, start="2005-01-01", end="2025-01-01", auto_adjust=True, progress=False)["Close"]
    fr.to_csv(cache)
old = pd.read_csv(HERE / "cache" / "etf_px.csv", index_col=0, parse_dates=True)[T]
mo = lambda p: p.resample("ME").last().pct_change(fill_method=None).loc["2007-01":"2024-12"]
a, b = mo(old), mo(fr[T])
d = (a - b).abs()
print("max |monthly return diff| per ticker, cached vs fresh yfinance (2007-2024):")
print(d.max().round(6).to_string())
print("max monthly |return| per ticker (outlier check):")
print(a.abs().max().round(3).to_string())
dd = old[T].pct_change(fill_method=None).loc["2006":"2024"]
print("daily |return| > 25% count per ticker:", (dd.abs() > 0.25).sum().to_dict())
