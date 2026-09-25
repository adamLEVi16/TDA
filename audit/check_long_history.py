"""Is the notebook's residual signal (no filter) any good outside 2022-24?

Same residual rule as the Dec-2025 notebooks, run on 2007-2024 for the 18 names of the original
universe that traded throughout (META and TSLA listed later). Survivorship bias favours this test:
the list is today's winners.

Run from the repo root:  python audit/check_long_history.py
"""
import numpy as np
import pandas as pd

from common import UNIVERSE, prices, sharpe

px = prices(UNIVERSE + ["SPY"], "2007-01-01", "2024-12-10").drop(columns=["SPY", "META", "TSLA"])
R = px.ffill().dropna().pct_change().dropna()

resid = []
for i in range(60, len(R)):
    adj = np.abs(R.iloc[i - 60:i].corr().values)
    np.fill_diagonal(adj, 0)
    adj[adj < 0.3] = 0
    deg = adj.sum(1)
    d_inv = np.diag(1 / np.sqrt(deg + 1e-8))
    eye = np.eye(len(deg))
    x = R.iloc[i].values
    resid.append(x - np.linalg.matrix_power(eye - 0.5 * (eye - d_inv @ adj @ d_inv), 3) @ x)
E = pd.DataFrame(resid, index=R.index[60:], columns=R.columns)

rank = E.rank(axis=1)
n = R.shape[1]
pos = ((rank > n - 5).astype(float) - (rank <= 5).astype(float)) / 5   # long top-5, short bottom-5
lag = pos.shift(1).dropna()
gross = (lag * R.loc[lag.index]).sum(axis=1)
turnover = lag.diff().abs().sum(axis=1).fillna(0)
out = pd.DataFrame({"gross": gross, "net 2bp": gross - turnover * 0.0002, "net 5bp": gross - turnover * 0.0005})

print(f"{n} stocks, {gross.index[0].date()}..{gross.index[-1].date()}, mean turnover {turnover.mean():.2f}x gross per day")
print("full-period Sharpe:", {k: round(sharpe(v), 2) for k, v in out.items()},
      f"| t-stat of gross mean = {sharpe(gross) * np.sqrt(len(gross) / 252):+.2f}")
print(out.groupby(out.index.year).apply(lambda d: d.apply(sharpe)).round(2).T.to_string())
