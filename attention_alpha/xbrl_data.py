"""
Quarterly diluted EPS from SEC XBRL (companyconcept API, free, no key) and the
classic Bernard-Thomas SUE (standardized unexpected earnings, seasonal random
walk):  SUE_q = (EPS_q - EPS_{q-4}) / std(last 8 seasonal diffs).

Point-in-time care: when the same fiscal quarter appears in multiple filings
(originals + restatements), we keep the EARLIEST-FILED value -- the number the
market actually saw at announcement time, not a later restated one.
Q4 is not reported directly (10-Ks carry only FY), so Q4 = FY - (Q1+Q2+Q3).
"""
import urllib.request, json, os, time
import numpy as np, pandas as pd
from earnings_data import ticker_cik_map, UA

CACHE_DIR = os.path.join(os.path.dirname(__file__), "cache")
API = ("https://data.sec.gov/api/xbrl/companyconcept/CIK{cik}/us-gaap/"
       "EarningsPerShareDiluted.json")


def _get(url, retries=4):
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA})
            return json.load(urllib.request.urlopen(req, timeout=45))
        except Exception as e:
            if attempt == retries - 1:
                raise
            time.sleep(2 ** attempt)


def quarterly_eps(ticker, cik):
    """DataFrame indexed by fiscal-quarter end date: [eps]. Earliest-filed value
    per period; Q4 derived from FY minus the three reported quarters."""
    try:
        d = _get(API.format(cik=cik))
    except Exception as e:
        print(f"  [xbrl] FAILED {ticker}: {type(e).__name__}")
        return None
    obs = d.get("units", {}).get("USD/shares", [])
    rows = []
    for x in obs:
        if x.get("form") not in ("10-Q", "10-K") or not x.get("start") or not x.get("end"):
            continue
        start, end = pd.Timestamp(x["start"]), pd.Timestamp(x["end"])
        rows.append({"start": start, "end": end, "days": (end - start).days,
                     "val": x["val"], "filed": pd.Timestamp(x["filed"])})
    if not rows:
        return None
    df = pd.DataFrame(rows)
    # earliest-filed value per (start,end) period = point-in-time
    df = df.sort_values("filed").groupby(["start", "end"], as_index=False).first()

    q = df[(df["days"] >= 80) & (df["days"] <= 100)].set_index("end").sort_index()
    fy = df[(df["days"] >= 350) & (df["days"] <= 375)]

    eps = q["val"].to_dict()
    # derive missing Q4s: FY end matches a quarter end; subtract that FY's 3 quarters
    for _, r in fy.iterrows():
        if r["end"] in eps:
            continue
        in_fy = q[(q.index > r["start"]) & (q.index <= r["end"])]
        if len(in_fy) == 3:
            eps[r["end"]] = r["val"] - in_fy["val"].sum()
    s = pd.Series(eps).sort_index()
    s = s[~s.index.duplicated()]
    return s.to_frame("eps")


def sue_series(eps_df, min_hist=6):
    """SUE per quarter end: seasonal diff scaled by trailing std of 8 seasonal
    diffs (needs >= min_hist prior diffs). Uses only data through that quarter."""
    e = eps_df["eps"]
    sdiff = e.diff(4)
    out = {}
    vals = sdiff.dropna()
    for i, (dt, d) in enumerate(zip(vals.index, vals.values)):
        hist = vals.iloc[max(0, i - 8):i]
        if len(hist) >= min_hist and hist.std() > 0:
            out[dt] = d / hist.std()
    return pd.Series(out, name="sue")


def get_sue(tickers, verbose=True):
    """dict[ticker] -> Series indexed by fiscal-quarter END date."""
    cikmap = ticker_cik_map()
    out = {}
    for tk in tickers:
        fp = os.path.join(CACHE_DIR, f"sue_{tk}.csv")
        if os.path.exists(fp):
            s = pd.read_csv(fp, index_col=0, parse_dates=True).iloc[:, 0]
        else:
            cik = cikmap.get(tk)
            eps = quarterly_eps(tk, cik) if cik else None
            time.sleep(0.2)
            s = sue_series(eps) if eps is not None and len(eps) > 12 else pd.Series(dtype=float)
            s.to_frame("sue").to_csv(fp)
        if len(s):
            out[tk] = s
            if verbose:
                print(f"  [sue] {tk:<6} {len(s):>3} quarters  {s.index.min().date()} -> {s.index.max().date()}")
        elif verbose:
            print(f"  [sue] {tk:<6} unavailable")
    return out


if __name__ == "__main__":
    from universe import TICKER_ARTICLE
    sue = get_sue(list(TICKER_ARTICLE.keys()))
    print(f"\n{len(sue)}/{len(TICKER_ARTICLE)} tickers with SUE")
