"""Spot checks of thesis / Davidson-paper claims that can be settled with data or arithmetic.

1. Section 7: the Phase-2 code (02_compute_sector_topology.py:35,58) runs ripser with thresh=0.3 on
   the distance sqrt(2(1-rho)), so an edge needs rho > 0.955. Count windows with any H1 bar.
2. Section 6: "47 crisis days (VIX > 30 for 3 days, 2023-24)". Count VIX closes above 30.
3. Section 11 eigenvalue table: for any n x n correlation matrix with mean off-diagonal correlation r,
   lambda_1 >= 1 + (n-1) r (Rayleigh quotient with the all-ones vector) and lambda_1 <= n (the trace).
4. Section 10: F1 of a classifier that always predicts "up", as a baseline for the reported F1 = 0.578.

Run from the repo root:  python audit/check_thesis_claims.py
"""
import numpy as np
import pandas as pd
from ripser import ripser

from common import UNIVERSE, prices

px = prices(UNIVERSE + ["SPY"], "2019-01-01", "2024-12-10").drop(columns="SPY")
R = px.ffill().bfill().pct_change().dropna()

print("== 1. H1 at the Phase-2 threshold (ripser thresh=0.3 on sqrt(2(1-rho)))")
for name, tickers in {"Technology (5)": ["AAPL", "MSFT", "NVDA", "AMD", "INTC"], "All 20": UNIVERSE}.items():
    windows = with_h1 = 0
    for i in range(60, len(R), 5):
        d = np.sqrt(np.clip(2 * (1 - R[tickers].iloc[i - 60:i].corr().values), 0, None))
        np.fill_diagonal(d, 0)
        windows += 1
        with_h1 += len(ripser(d, maxdim=1, distance_matrix=True, thresh=0.3)["dgms"][1]) > 0
    print(f"{name:<15} windows with any H1 bar: {with_h1}/{windows}")

print("\n== 2. VIX closes above 30, 2023-2024 (sec06 claims 47 crisis days)")
import yfinance as yf  # noqa: E402
vix = yf.download("^VIX", start="2023-01-01", end="2025-01-01", progress=False, auto_adjust=True)["Close"].squeeze()
print(f"{int((vix > 30).sum())} day(s): {[d.isoformat() for d in vix[vix > 30].index.date]}")

print("\n== 3. sec11 eigenvalue table (rho, lambda_1): 0.3->2.14, 0.5->4.52, 0.7->8.91, 0.9->16.34")
table = {0.3: 2.14, 0.5: 4.52, 0.7: 8.91, 0.9: 16.34}
feasible = [n for n in range(2, 200)
            if all(1 + (n - 1) * r <= lam <= n for r, lam in table.items())]
print(f"matrix sizes n for which every row is possible: {feasible or 'none'}")
print(f"sec07/Davidson 'Financials (3 stocks): lambda_1 = 13.5' -> max possible lambda_1 for n=3 is 3")

print("\n== 4. F1 of always predicting 'up' at base rate p: F1 = 2p/(1+p)")
for p in (0.50, 0.52, 0.533):
    print(f"p={p:.3f}: F1={2 * p / (1 + p):.3f}   (thesis's best model: 0.578)")
