"""Concept C data loaders. Everything cached under ./cache.

Ken French parser adapted from /home/user/TDA/audit/check_risk_report.py (function french), extended to take
only the FIRST table (value-weighted), handle YYYYMM monthly dates and map -99.99/-999 to NaN.
"""
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).parent
CACHE = HERE / "cache"
CACHE.mkdir(exist_ok=True)
FRENCH = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"
END = "2024-12-31"


def _raw(name):
    p = CACHE / name
    if not p.exists():
        p.write_bytes(urllib.request.urlopen(FRENCH + name, timeout=180).read())
    with zipfile.ZipFile(p) as z:
        return z.read(z.namelist()[0]).decode("latin1").splitlines()


def french(name):
    lines = _raw(name)
    header = next(i for i, line in enumerate(lines) if line.startswith(","))
    cols = [c.strip() for c in lines[header].split(",")[1:]]
    rows = []
    for line in lines[header + 1:]:
        key = line.split(",")[0].strip()
        if not key.isdigit():
            if rows:
                break
            continue
        rows.append([c.strip() for c in line.split(",")])
    dates = [r[0] for r in rows]
    if len(dates[0]) == 6:
        idx = pd.to_datetime(dates, format="%Y%m") + pd.offsets.MonthEnd(0)
    else:
        idx = pd.to_datetime(dates, format="%Y%m%d")
    df = pd.DataFrame([r[1:] for r in rows], columns=cols, index=idx).astype(float)
    df[df <= -99.99] = np.nan
    return (df / 100).loc[:END]


def market_daily():
    """Daily Mkt (total) return, RF, Mkt-RF. 1926-07 .. 2024-12."""
    f = french("F-F_Research_Data_Factors_daily_CSV.zip")
    return pd.DataFrame({"mkt": f["Mkt-RF"] + f["RF"], "rf": f["RF"], "mkt_rf": f["Mkt-RF"]})


def market_monthly():
    f = french("F-F_Research_Data_Factors_CSV.zip")
    return pd.DataFrame({"mkt": f["Mkt-RF"] + f["RF"], "rf": f["RF"], "mkt_rf": f["Mkt-RF"]})


def industries_daily():
    """FF49 value-weighted daily returns, restricted to industries with NO missing values 1926-07..2024-12
    (constant N, so topology statistics are comparable over time)."""
    d = french("49_Industry_Portfolios_daily_CSV.zip")
    full = d.columns[d.notna().all()]
    return d[full], list(d.columns.difference(full))


def yahoo(tickers, start="1990-01-01", end="2025-01-01"):
    p = CACHE / f"yf_{'_'.join(t.replace('^', '') for t in tickers)}.csv"
    if p.exists():
        return pd.read_csv(p, index_col=0, parse_dates=True)
    import yfinance as yf
    px = yf.download(list(tickers), start=start, end=end, auto_adjust=True, progress=False)["Close"]
    px.to_csv(p)
    return px


if __name__ == "__main__":
    md, mm = market_daily(), market_monthly()
    print(f"market daily {md.index[0].date()}..{md.index[-1].date()} n={len(md)}; monthly {mm.index[0].date()}..{mm.index[-1].date()} n={len(mm)}")
    ind, dropped = industries_daily()
    print(f"industries daily {ind.index[0].date()}..{ind.index[-1].date()} n={len(ind)}, kept {ind.shape[1]} of 49; dropped (missing data): {dropped}")
    print("daily market vs industries date alignment:", md.index.equals(ind.index))
    y = yahoo(["SPY", "^VIX", "^VIX3M"])
    for c in y.columns:
        s = y[c].dropna()
        print(f"yahoo {c}: {s.index[0].date()}..{s.index[-1].date()} n={len(s)}")
    spy = y["SPY"].pct_change().dropna()
    j = pd.concat([spy, md["mkt"]], axis=1, join="inner").dropna()
    print(f"corr(SPY daily ret, FF mkt daily) {j.index[0].date()}..{j.index[-1].date()}: {j.corr().iloc[0,1]:.4f}")
    # sanity: compounding daily FF market to months vs monthly file
    comp = (1 + md["mkt"]).groupby(md.index.to_period("M")).prod() - 1
    comp.index = comp.index.to_timestamp("M")
    k = pd.concat([comp, mm["mkt"]], axis=1, join="inner").dropna()
    print(f"daily-compounded vs monthly FF mkt: corr {k.corr().iloc[0,1]:.5f}, mean abs diff {np.abs(k.iloc[:,0]-k.iloc[:,1]).mean():.5f}")
