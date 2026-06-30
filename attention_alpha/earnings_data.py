"""
Historical earnings-announcement dates from SEC EDGAR -- the 8-K filing with
Item 2.02 ("Results of Operations and Financial Condition") IS the earnings
release. Free, no key, government source, reliable back decades. Cached per
ticker. This is the data piece the weekly attention test was missing.
"""
import urllib.request, json, os, time
import pandas as pd

CACHE_DIR = os.path.join(os.path.dirname(__file__), "cache")
os.makedirs(CACHE_DIR, exist_ok=True)
UA = "tda-research adamwasnothere292@gmail.com"


def _get(url):
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    return json.load(urllib.request.urlopen(req, timeout=45))


def ticker_cik_map():
    fp = os.path.join(CACHE_DIR, "_cik_map.json")
    if os.path.exists(fp):
        return json.load(open(fp))
    d = _get("https://www.sec.gov/files/company_tickers.json")
    m = {v["ticker"]: str(v["cik_str"]).zfill(10) for v in d.values()}
    json.dump(m, open(fp, "w"))
    return m


def _earnings_from_submissions(cik, retries=4):
    """Return list of 8-K Item-2.02 filing dates for a CIK, across all
    (recent + older) submission files."""
    dates = []
    base = f"https://data.sec.gov/submissions/CIK{cik}.json"
    for attempt in range(retries):
        try:
            d = _get(base)
            break
        except Exception as e:
            if attempt == retries - 1:
                print(f"  [edgar] FAILED CIK{cik}: {type(e).__name__}")
                return []
            time.sleep(2 ** attempt)

    def harvest(rec):
        forms = rec.get("form", []); fdates = rec.get("filingDate", [])
        items = rec.get("items", [""] * len(forms))
        for f, dt, it in zip(forms, fdates, items):
            if f == "8-K" and "2.02" in (it or ""):
                dates.append(dt)

    harvest(d["filings"]["recent"])
    for extra in d["filings"].get("files", []):
        try:
            harvest(_get(f"https://data.sec.gov/submissions/{extra['name']}"))
            time.sleep(0.2)
        except Exception:
            pass
    return sorted(set(dates))


def get_earnings_dates(tickers, verbose=True):
    """dict[ticker] -> sorted DatetimeIndex of earnings-announcement dates."""
    cikmap = ticker_cik_map()
    out = {}
    for tk in tickers:
        fp = os.path.join(CACHE_DIR, f"earn_{tk}.csv")
        if os.path.exists(fp):
            s = pd.read_csv(fp, parse_dates=["date"])["date"]
        else:
            cik = cikmap.get(tk)
            if cik is None:
                if verbose: print(f"  [edgar] {tk:<6} no CIK")
                continue
            dates = _earnings_from_submissions(cik)
            time.sleep(0.2)
            s = pd.Series(pd.to_datetime(dates), name="date")
            s.to_frame().to_csv(fp, index=False)
        if len(s):
            out[tk] = pd.DatetimeIndex(sorted(s))
            if verbose:
                print(f"  [edgar] {tk:<6} {len(s):>3} earnings dates "
                      f"{out[tk].min().date()} -> {out[tk].max().date()}")
    return out


if __name__ == "__main__":
    from universe import TICKER_ARTICLE
    ed = get_earnings_dates(list(TICKER_ARTICLE.keys()))
    print(f"\n{len(ed)}/{len(TICKER_ARTICLE)} tickers with earnings dates")
