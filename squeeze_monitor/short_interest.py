"""
FINRA consolidated short interest (free, anonymous, twice-monthly) per ticker,
with the field that matters for squeezes: daysToCoverQuantity (short interest /
avg daily volume). Cached per ticker.

POINT-IN-TIME: FINRA disseminates short interest ~9 business days after the
settlement date. `avail` = settlementDate + 9 business days; the monitor must
only use a report from its `avail` date onward, never from its settlement date.
"""
import urllib.request, json, os, time
import pandas as pd
from pandas.tseries.offsets import BDay

CACHE_DIR = os.path.join(os.path.dirname(__file__), "cache")
os.makedirs(CACHE_DIR, exist_ok=True)
API = "https://api.finra.org/data/group/otcMarket/name/consolidatedShortInterest"
PUB_LAG_BDAYS = 9


def _fetch(ticker, retries=4):
    body = json.dumps({
        "limit": 5000,
        "compareFilters": [{"compareType": "EQUAL", "fieldName": "symbolCode",
                            "fieldValue": ticker}],
        "fields": ["settlementDate", "currentShortPositionQuantity",
                   "averageDailyVolumeQuantity", "daysToCoverQuantity"],
    }).encode()
    for attempt in range(retries):
        try:
            req = urllib.request.Request(API, data=body, headers={
                "Content-Type": "application/json", "Accept": "application/json",
                "User-Agent": "tda-research adamwasnothere292@gmail.com"})
            rows = json.load(urllib.request.urlopen(req, timeout=45))
            if not rows:
                return None
            df = pd.DataFrame(rows)
            df["settlementDate"] = pd.to_datetime(df["settlementDate"])
            df = (df.rename(columns={"currentShortPositionQuantity": "si",
                                     "averageDailyVolumeQuantity": "adv",
                                     "daysToCoverQuantity": "dtc"})
                    .sort_values("settlementDate")
                    .drop_duplicates("settlementDate", keep="last"))
            df["avail"] = df["settlementDate"] + BDay(PUB_LAG_BDAYS)
            return df[["settlementDate", "avail", "si", "adv", "dtc"]]
        except Exception as e:
            if attempt == retries - 1:
                print(f"  [finra] FAILED {ticker}: {type(e).__name__} {str(e)[:60]}")
                return None
            time.sleep(2 ** attempt)


def get_short_interest(tickers, verbose=True):
    out = {}
    for tk in tickers:
        fp = os.path.join(CACHE_DIR, f"si_{tk}.csv")
        if os.path.exists(fp):
            df = pd.read_csv(fp, parse_dates=["settlementDate", "avail"])
        else:
            df = _fetch(tk)
            time.sleep(0.3)
            if df is not None:
                df.to_csv(fp, index=False)
        if df is not None and len(df):
            out[tk] = df
            if verbose:
                print(f"  [finra] {tk:<6} {len(df):>3} reports  "
                      f"{df['settlementDate'].min().date()} -> {df['settlementDate'].max().date()}")
        elif verbose:
            print(f"  [finra] {tk:<6} MISSING")
    return out


if __name__ == "__main__":
    from sq_universe import TICKER_ARTICLE
    si = get_short_interest(list(TICKER_ARTICLE.keys()))
    print(f"\n{len(si)}/{len(TICKER_ARTICLE)} tickers with short interest")
