"""Shared helpers for the audit checks: cached price download and Sharpe statistics."""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

CACHE = Path(__file__).parent / "_cache"          # *.csv is gitignored

# The 20-stock universe used by the Dec-2025 notebooks.
UNIVERSE = ["AAPL", "MSFT", "AMZN", "NVDA", "META", "GOOG", "TSLA",
            "NFLX", "JPM", "PEP", "CSCO", "ORCL", "DIS", "BAC",
            "XOM", "IBM", "INTC", "AMD", "KO", "WMT"]


def prices(tickers, start, end):
    """Adjusted closes from Yahoo Finance, cached on disk."""
    CACHE.mkdir(exist_ok=True)
    path = CACHE / f"px_{start}_{end}_{len(tickers)}.csv"
    if path.exists():
        return pd.read_csv(path, index_col=0, parse_dates=True)
    import yfinance as yf
    px = yf.download(list(tickers), start=start, end=end, auto_adjust=True, progress=False)["Close"]
    px.to_csv(path)
    return px


def sharpe(r):
    """Annualised Sharpe of a daily return series (no risk-free: the strategies are self-financing)."""
    return r.mean() / r.std() * np.sqrt(252) if r.std() > 0 else 0.0


def sharpe_ci(r):
    """Lo (2002) iid standard error, 95% CI and two-sided p-value for an annualised Sharpe."""
    sr, years = sharpe(r), len(r) / 252
    se = np.sqrt((1 + 0.5 * sr ** 2) / years)
    t = r.mean() / r.std() * np.sqrt(len(r))
    return sr, se, (sr - 1.96 * se, sr + 1.96 * se), 2 * stats.t.sf(abs(t), len(r) - 1)


def cagr(r):
    return (1 + r).prod() ** (252 / len(r)) - 1


def max_drawdown(r):
    c = (1 + r).cumprod()
    return (c / c.cummax() - 1).min()
