"""
LONG-HISTORY VALIDATION (~1987-2024) — the decisive out-of-sample test.

The 8-ETF backtest (2007-2024) had only ~2 crashes, so confidence intervals were
huge. Here we re-run the IDENTICAL engine on dividend-adjusted mutual-fund proxies
that reach back to ~1987, adding many independent stress events the original sample
never saw: Black Monday 1987, the 1990 recession, the 1994 bond massacre, 1998
LTCM/Asia, and 2000-02 dot-com — plus a high-/rising-rate regime where the bond
sleeve canNOT rely on a tailwind.

Universe (all trend-filtered + inverse-vol weighted, cash earns T-bills):
  VFINX  US equity (S&P500 fund)        VWIGX  international equity
  VUSTX  long Treasuries                VGPMX  precious metals (gold/inflation proxy)
  VWEHX  high-yield credit
Benchmarks: VFINX buy & hold; 60/40 (60% VFINX / 40% VUSTX).
"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import multi_asset as MA
from backtest import metrics
from torture_test import block_bootstrap_sharpe_diff

LH_ASSETS = ["VFINX", "VWIGX", "VUSTX", "VGPMX", "VWEHX"]
LH_BENCH = ("VFINX", "VUSTX")
START = "1984-01-01"
sh = lambda r: r.mean() / r.std() * np.sqrt(12) if r.std() > 0 else np.nan

def run_lh(**kw):
    return MA.run(assets=LH_ASSETS, bench=LH_BENCH, start=START, **kw)

def main():
    bt, w = run_lh()
    rp, spy, sixty = bt["RP+Trend"], bt["SPY"], bt["60/40"]
    print("=" * 76)
    print(f"  LONG-HISTORY VALIDATION  —  {len(LH_ASSETS)} fund proxies")
    print(f"  {bt.index.min().date()} → {bt.index.max().date()}  ({len(bt)} months, "
          f"{len(bt)//12} years)")
    print("=" * 76)
    print(f"{'Portfolio':<26}{'CAGR':>9}{'Vol':>8}{'Sharpe':>8}{'MaxDD':>9}{'corrEq':>8}")
    print("-" * 76)
    for col, lbl in [("RP+Trend", "RP+Trend (strategy)"), ("EW+Trend", "EW+Trend"),
                     ("RP static", "RP static (no trend)"), ("60/40", "60/40"),
                     ("SPY", "VFINX buy & hold")]:
        m = metrics(bt[col])
        print(f"{lbl:<26}{m['CAGR']:>8.1%}{m['Vol']:>8.1%}{m['Sharpe']:>8.2f}"
              f"{m['MaxDD']:>9.1%}{bt[col].corr(spy):>8.2f}")
    print("-" * 76)

    # significance vs both benchmarks
    print("\nSHARPE-DIFFERENCE SIGNIFICANCE (block bootstrap, 5000x):")
    for name, b in [("VFINX", spy), ("60/40", sixty), ("RP static", bt["RP static"])]:
        d, lo, hi, p = block_bootstrap_sharpe_diff(rp, b)
        tag = "SIGNIFICANT" if (lo > 0 or hi < 0) else "not significant"
        print(f"  RP+Trend - {name:<10}: ΔSharpe={d:+.2f}  95% CI [{lo:+.2f},{hi:+.2f}]"
              f"  p={p:.3f}  -> {tag}")

    # regime / sub-period table
    print("\nSUB-PERIOD Sharpe (RP+Trend vs VFINX):")
    periods = {"Full": (None, None), "1987-1999": ("1987", "1999"),
               "2000-2009": ("2000", "2009"), "2010-2024": ("2010", "2024")}
    for lab, (a, b) in periods.items():
        seg = bt.loc[a:b] if a else bt
        print(f"  {lab:<12} RP={sh(seg['RP+Trend']):+.2f}  VFINX={sh(seg['SPY']):+.2f}"
              f"   (RP CAGR {(1+seg['RP+Trend']).prod()**(12/len(seg))-1:+.1%}, "
              f"VFINX {(1+seg['SPY']).prod()**(12/len(seg))-1:+.1%})")

    # crisis-by-crisis (calendar-year worst events)
    yr = (1 + bt).groupby(bt.index.year).prod() - 1
    print("\nCRISIS YEARS (RP+Trend vs VFINX):")
    for y in [1987, 1990, 1994, 2000, 2001, 2002, 2008, 2020, 2022]:
        if y in yr.index:
            print(f"  {y}: RP+Trend={yr.loc[y,'RP+Trend']:+6.1%}   VFINX={yr.loc[y,'SPY']:+6.1%}")

if __name__ == "__main__":
    main()
