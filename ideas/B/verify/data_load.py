"""Step 1: download and cache data for Concept B.

- S&P 100 constituents: Wikipedia table (page last edited 2026-09-19), saved as sp100_wikipedia_2026-09-25.csv.
- Adjusted closes from Yahoo (auto_adjust=True), 2004-01-01..2024-12-31, for all 101 symbols + SPY
  + the original 20-stock universe (subset, AAPL etc.).
- Universe rule (pre-registered in SPEC.md): keep a symbol if it has a valid close on 2005-01-03 and on
  every trading day through 2024-12-31 (after forward-filling at most 5 days). GOOG dropped as a duplicate share
  class of GOOGL.
- Ken French daily FF5, momentum, RF.
"""
import io
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

D = Path(__file__).parent
CACHE = D / "cache"
CACHE.mkdir(exist_ok=True)
FRENCH = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"


def french(name):
    path = CACHE / name
    if not path.exists():
        path.write_bytes(urllib.request.urlopen(FRENCH + name).read())
    with zipfile.ZipFile(path) as z:
        lines = z.read(z.namelist()[0]).decode("latin1").splitlines()
    header = next(i for i, line in enumerate(lines) if line.startswith(","))
    cols = [c.strip() for c in lines[header].split(",")[1:]]
    rows = []
    for line in lines[header + 1:]:
        if not line[:8].strip().isdigit():
            break            # first table only
        rows.append(line.split(","))
    df = pd.DataFrame([r[1:] for r in rows], columns=cols,
                      index=pd.to_datetime([r[0].strip() for r in rows])).astype(float)
    df = df.mask(df <= -99.99)
    return df / 100


def load_prices():
    path = CACHE / "px_sp100_2004_2024.csv"
    if path.exists():
        return pd.read_csv(path, index_col=0, parse_dates=True)
    import yfinance as yf
    cons = pd.read_csv(D / "sp100_wikipedia_2026-09-25.csv")
    tick = [s.replace(".", "-") for s in cons["Symbol"]] + ["SPY"]
    px = yf.download(tick, start="2004-01-01", end="2025-01-01", auto_adjust=True, progress=False)["Close"]
    px.to_csv(path)
    return px


def universe():
    """Returns (daily simple returns 2005-01-03..2024-12-31 for the kept universe, SPY returns, sector map, dropped)."""
    px = load_prices()
    cons = pd.read_csv(D / "sp100_wikipedia_2026-09-25.csv")
    cons["tk"] = cons["Symbol"].str.replace(".", "-", regex=False)
    sector = dict(zip(cons["tk"], cons["Sector"]))
    px = px.loc["2004-12-31":"2024-12-31"]
    spy = px.pop("SPY")
    keep, dropped = [], {}
    for t in cons["tk"]:
        if t == "GOOG":
            dropped[t] = "duplicate share class of GOOGL"
            continue
        s = px[t] if t in px else pd.Series(dtype=float)
        first = s.first_valid_index()
        if first is None or first > pd.Timestamp("2005-01-03"):
            dropped[t] = f"first price {None if first is None else first.date()}"
            continue
        if s.loc["2005-01-03":].ffill(limit=5).isna().any():
            dropped[t] = "gaps"
            continue
        keep.append(t)
    P = px[keep].ffill(limit=5)
    R = P.pct_change().loc["2005-01-03":]
    R = R.iloc[1:] if R.iloc[0].isna().all() else R
    # Data fix (pre-registered in SPEC.md): Yahoo's DHR series has a broken Fortive spin-off adjustment
    # (adjusted close jumps 42.01 -> 67.73 on 2016-07-05, +61%). The true total return is unknown here; set to 0.
    if "DHR" in R:
        R.loc["2016-07-05", "DHR"] = 0.0
    spy_r = spy.pct_change().reindex(R.index)
    return R, spy_r, {t: sector[t] for t in keep}, dropped


if __name__ == "__main__":
    R, spy_r, sec, dropped = universe()
    print(f"returns {R.index[0].date()}..{R.index[-1].date()}  {R.shape[0]} days x {R.shape[1]} stocks, NaN count {int(R.isna().sum().sum())}")
    print("kept:", " ".join(R.columns))
    print(f"dropped ({len(dropped)}):")
    for k, v in dropped.items():
        print(f"  {k}: {v}")
    print("sector counts:", pd.Series(sec).value_counts().to_dict())
    big = (R.abs() > 0.4)
    print("daily |ret|>40% events:", [(d.date(), t, round(R.loc[d, t], 3)) for d, t in zip(*np.where(big))][:20] if False else
          [(R.index[i].date(), R.columns[j], round(R.iat[i, j], 3)) for i, j in zip(*np.where(big.values))])
    ff = french("F-F_Research_Data_5_Factors_2x3_daily_CSV.zip")
    mom = french("F-F_Momentum_Factor_daily_CSV.zip")
    print("FF5 daily", ff.index[0].date(), ff.index[-1].date(), list(ff.columns))
    print("MOM daily", mom.index[0].date(), mom.index[-1].date(), list(mom.columns))
    # survivorship quantification: EW universe vs SPY
    ew = R.mean(axis=1)
    for a, b in [("2005", "2014"), ("2015", "2024")]:
        e, s = ew.loc[a:b], spy_r.loc[a:b]
        print(f"{a}-{b}: EW survivor universe ann. return {(1+e).prod()**(252/len(e))-1:.2%}  SPY {(1+s).prod()**(252/len(s))-1:.2%}")
