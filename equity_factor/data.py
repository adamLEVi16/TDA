"""
Data layer for the equity factor project.

Fetches daily adjusted-close prices from Yahoo Finance via urllib (works behind
proxies where yfinance's curl_cffi backend fails) and caches to CSV. Honest by
construction: one fixed, transparent universe; no point-in-time membership data,
so survivorship bias is acknowledged and controlled for at the backtest level
(equal-weight-universe benchmark) rather than hidden.
"""
import urllib.request, json, time, os
import pandas as pd

CACHE_DIR = os.path.join(os.path.dirname(__file__), "cache")
os.makedirs(CACHE_DIR, exist_ok=True)

# Fixed large/mega-cap universe, all listed well before the 2010 backtest start,
# spread across sectors. NOTE: this is a survivorship-biased set (these names
# survived and stayed large). We do NOT treat beating SPY as proof of skill on
# its own; the equal-weight-universe benchmark in backtest.py controls for it.
UNIVERSE = [
    # Tech / comms
    "AAPL", "MSFT", "ORCL", "IBM", "INTC", "CSCO", "QCOM", "TXN", "ADBE", "CRM",
    "NVDA", "AMD", "GOOGL", "META", "NFLX", "ADI", "MU", "HPQ", "ACN",
    # Consumer
    "AMZN", "HD", "MCD", "NKE", "SBUX", "TGT", "LOW", "TJX", "BKNG", "DIS",
    "KO", "PEP", "PG", "CL", "WMT", "COST", "MDLZ", "MO", "PM", "KMB",
    # Health care
    "JNJ", "PFE", "MRK", "ABT", "TMO", "UNH", "LLY", "BMY", "AMGN", "GILD",
    "MDT", "CVS", "CI", "ISRG", "SYK",
    # Financials
    "JPM", "BAC", "WFC", "C", "GS", "MS", "AXP", "USB", "PNC", "BK",
    "SCHW", "BLK", "SPGI", "CB", "TRV", "MET", "AIG",
    # Industrials / materials / energy
    "GE", "HON", "UNP", "UPS", "CAT", "DE", "BA", "LMT", "RTX", "MMM",
    "EMR", "ITW", "FDX", "NSC", "XOM", "CVX", "COP", "SLB", "EOG", "PSX",
    "DD", "DOW", "FCX", "NEM", "APD",
    # Utilities / real estate / staples extra
    "NEE", "DUK", "SO", "D", "AEP", "EXC", "SPG", "PLD", "AMT", "O",
]

BENCHMARK = "SPY"


def _fetch_one(ticker, start, end, retries=4):
    p1 = int(pd.Timestamp(start).timestamp())
    p2 = int(pd.Timestamp(end).timestamp())
    url = (f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}"
           f"?period1={p1}&period2={p2}&interval=1d&events=div%2Csplit")
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            j = json.load(urllib.request.urlopen(req, timeout=30))
            r = j["chart"]["result"][0]
            ts = pd.to_datetime(r["timestamp"], unit="s").normalize()
            adj = r["indicators"]["adjclose"][0]["adjclose"]
            return pd.Series(adj, index=ts, name=ticker).dropna()
        except Exception as e:
            if attempt == retries - 1:
                print(f"  [data] FAILED {ticker}: {type(e).__name__} {str(e)[:60]}")
                return None
            time.sleep(2 ** attempt)


def get_prices(tickers=None, start="2009-06-01", end="2024-12-31", verbose=True):
    """Return a daily adjusted-close DataFrame (cols=tickers). Cached per ticker."""
    if tickers is None:
        tickers = UNIVERSE + [BENCHMARK]
    series = {}
    for i, tk in enumerate(tickers):
        fp = os.path.join(CACHE_DIR, f"{tk}.csv")
        if os.path.exists(fp):
            s = pd.read_csv(fp, index_col=0, parse_dates=True).iloc[:, 0]
            s.name = tk
        else:
            s = _fetch_one(tk, start, end)
            time.sleep(0.25)
            if s is not None and len(s) > 0:
                s.to_frame().to_csv(fp)
        if s is not None and len(s) > 0:
            series[tk] = s
        if verbose and (i + 1) % 20 == 0:
            print(f"  [data] {i+1}/{len(tickers)} tickers …")
    px = pd.concat(series.values(), axis=1)
    px.columns = list(series.keys())
    return px.sort_index()


if __name__ == "__main__":
    px = get_prices()
    print(f"\nUniverse: {px.shape[1]} tickers, {px.shape[0]} days, "
          f"{px.index.min().date()} → {px.index.max().date()}")
    missing = [t for t in UNIVERSE + [BENCHMARK] if t not in px.columns]
    print("Missing-data tickers (fetch failed):", missing)
