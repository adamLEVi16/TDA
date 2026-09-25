"""Data loaders for Concept A (network momentum). All downloads are cached under ./cache.

Ken French parser adapted from /home/user/TDA/audit/check_risk_report.py (function french), extended to
(a) take only the FIRST table in the file (value-weighted), (b) handle monthly YYYYMM dates, (c) turn
-99.99 / -999 into NaN.
"""
import io
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).parent
CACHE = HERE / "cache"
CACHE.mkdir(exist_ok=True)
FRENCH = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"

ETFS = ["XLB", "XLE", "XLF", "XLI", "XLK", "XLP", "XLU", "XLV", "XLY",
        "EWJ", "EWG", "EWU", "EWC", "EWA", "EWH", "EWS", "EWZ", "EWW", "EWT", "EWY", "EEM", "EFA",
        "TLT", "IEF", "SHY", "LQD", "HYG", "GLD", "DBC", "VNQ"]


def _raw(name):
    p = CACHE / name
    if not p.exists():
        p.write_bytes(urllib.request.urlopen(FRENCH + name, timeout=120).read())
    with zipfile.ZipFile(p) as z:
        return z.read(z.namelist()[0]).decode("latin1").splitlines()


def french(name):
    """First (value-weighted) table of a Ken French CSV zip, in decimal returns, NaN for missing."""
    lines = _raw(name)
    header = next(i for i, line in enumerate(lines) if line.startswith(","))
    cols = [c.strip() for c in lines[header].split(",")[1:]]
    rows = []
    for line in lines[header + 1:]:
        key = line.split(",")[0].strip()
        if not key.isdigit():
            if rows:           # first table ended (blank line / next table title)
                break
            continue
        rows.append([c.strip() for c in line.split(",")])
    dates = [r[0] for r in rows]
    if len(dates[0]) == 6:
        idx = pd.to_datetime(dates, format="%Y%m") + pd.offsets.MonthEnd(0)
    else:
        idx = pd.to_datetime(dates, format="%Y%m%d")
    df = pd.DataFrame([r[1:] for r in rows], columns=cols, index=idx).astype(float)
    df[(df <= -99.99)] = np.nan
    return df / 100


def industries():
    """(daily, monthly) FF49 value-weighted industry returns."""
    d = french("49_Industry_Portfolios_daily_CSV.zip")
    m = french("49_Industry_Portfolios_CSV.zip")
    return d, m


def factors_monthly():
    ff5 = french("F-F_Research_Data_5_Factors_2x3_CSV.zip")
    umd = french("F-F_Momentum_Factor_CSV.zip")
    umd.columns = ["UMD"]
    ff3 = french("F-F_Research_Data_Factors_CSV.zip")
    return ff5.join(umd, how="inner"), ff3


def etf_prices():
    p = CACHE / "etf_px.csv"
    if p.exists():
        return pd.read_csv(p, index_col=0, parse_dates=True)
    import yfinance as yf
    px = yf.download(ETFS + ["SPY"], start="1993-01-01", end="2025-01-01", auto_adjust=True,
                     progress=False)["Close"]
    px.to_csv(p)
    return px


if __name__ == "__main__":
    d, m = industries()
    print("FF49 daily", d.shape, d.index[0].date(), d.index[-1].date(), "NaN share", float(d.isna().mean().mean()))
    print("FF49 monthly", m.shape, m.index[0].date(), m.index[-1].date())
    print("first valid monthly date per industry (late starters):")
    fv = m.apply(lambda s: s.first_valid_index())
    print(fv[fv > m.index[0]].sort_values().to_string())
    f, ff3 = factors_monthly()
    print("FF5+UMD", f.shape, f.index[0].date(), f.index[-1].date(), list(f.columns))
    print("FF3", ff3.shape, ff3.index[0].date(), ff3.index[-1].date())
    px = etf_prices()
    print("ETF prices", px.shape, px.index[0].date(), px.index[-1].date())
    print(px.apply(lambda s: s.first_valid_index().date()).sort_values().to_string())
    # sanity: monthly compounded daily vs monthly file
    comp = (1 + d).resample("ME").prod(min_count=15) - 1
    j = comp.index.intersection(m.index)
    diff = (comp.loc[j] - m.loc[j]).abs()
    print("max |monthly - compounded daily| median over industries:", float(diff.max().median()),
          " mean abs diff:", float(np.nanmean(diff.values)))
