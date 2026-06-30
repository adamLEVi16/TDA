"""
Head-to-head vs the actual competing PRODUCTS (not just SPY/60-40).

  AQMNX  AQR Managed Futures Strategy  (the professional trend-following analog, 2010+)
  VBIAX  Vanguard Balanced Index       (the 'just buy a balanced fund' option, 2000+)
  PRPFX  Permanent Portfolio           (real-world static multi-asset fund, 1982+)

Each competitor is compared against OUR strategy computed over the IDENTICAL window,
plus SPY and 60/40 for reference. Fund returns are NAV (already net of fees); our
strategy includes 10 bps trading cost but not the ~0.1%/yr underlying ETF expense.
"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import data as D, multi_asset as MA, long_history as LH
from backtest import metrics

def fund_monthly(tk, start="1999-01-01"):
    s = D.get_prices(tickers=[tk], start=start, verbose=False)[tk]
    return s.resample("ME").last().pct_change()

def table(title, series_dict, window):
    a, b = window
    print("\n" + "=" * 72)
    print(f"  {title}")
    rows = {k: v.loc[a:b].dropna() for k, v in series_dict.items()}
    n = min(len(v) for v in rows.values())
    print(f"  window {a} → {b}   ({n} common months)")
    print("=" * 72)
    print(f"{'Strategy / Fund':<30}{'CAGR':>9}{'Vol':>8}{'Sharpe':>8}{'MaxDD':>9}{'corrSPY':>9}")
    print("-" * 72)
    spy = next((v for k, v in rows.items() if "buy & hold" in k), None)
    # align all to common index
    idx = None
    for v in rows.values():
        idx = v.index if idx is None else idx.intersection(v.index)
    for name, r in rows.items():
        r = r.reindex(idx)
        m = metrics(r)
        c = r.corr(spy.reindex(idx)) if spy is not None else np.nan
        print(f"{name:<30}{m['CAGR']:>8.1%}{m['Vol']:>8.1%}{m['Sharpe']:>8.2f}"
              f"{m['MaxDD']:>9.1%}{c:>9.2f}")
    print("-" * 72)
    return rows, idx

def main():
    # our strategies
    bt8, _ = MA.run()                 # 8-ETF, 2007-2024
    btlh, _ = LH.run_lh()             # fund-proxy, 1987-2024
    ours8 = bt8["RP+Trend"]; spy8 = bt8["SPY"]; sixty8 = bt8["60/40"]
    ourslh = btlh["RP+Trend"]; spylh = btlh["SPY"]; sixtylh = btlh["60/40"]

    # competitor funds (monthly)
    aqmnx = fund_monthly("AQMNX", "2009-06-01")
    vbiax = fund_monthly("VBIAX", "1999-06-01")
    prpfx = fund_monthly("PRPFX", "1981-06-01")

    # 1) vs AQR Managed Futures (the key peer), 2010-2024
    r1, idx1 = table("vs AQR MANAGED FUTURES — the professional trend product (2010–2024)", {
        "OUR RP+Trend (8-ETF)": ours8, "AQMNX AQR Managed Futures": aqmnx,
        "PRPFX Permanent Portfolio": prpfx, "VBIAX Vanguard Balanced": vbiax,
        "60/40": sixty8, "SPY (buy & hold)": spy8}, ("2010-01-01", "2024-12-31"))
    # how similar is ours to AQR's trend product?
    o = ours8.reindex(idx1); a = aqmnx.reindex(idx1)
    print(f"Correlation OUR strategy vs AQR Managed Futures: {o.corr(a):.2f}")

    # 2) full 8-ETF window vs the always-on funds, 2007-2024
    table("vs BALANCED & PERMANENT-PORTFOLIO FUNDS (2007–2024)", {
        "OUR RP+Trend (8-ETF)": ours8, "PRPFX Permanent Portfolio": prpfx,
        "VBIAX Vanguard Balanced": vbiax, "60/40": sixty8,
        "SPY (buy & hold)": spy8}, ("2007-04-01", "2024-12-31"))

    # 3) long-history strategy vs Permanent Portfolio, 1987-2024
    table("LONG-HISTORY: our strategy vs Permanent Portfolio (1987–2024)", {
        "OUR RP+Trend (long-hist)": ourslh, "PRPFX Permanent Portfolio": prpfx,
        "60/40 (VFINX/VUSTX)": sixtylh, "VFINX (buy & hold)": spylh},
        ("1987-06-01", "2024-12-31"))

if __name__ == "__main__":
    main()
