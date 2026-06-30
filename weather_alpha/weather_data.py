"""
Free historical daily weather (ERA5 reanalysis via Open-Meteo, no API key, no
rate limit issues) for a fixed set of US metros used as HQ/store-footprint
proxies for weather-sensitive consumer/retail/restaurant names. Cached to CSV.
"""
import urllib.request, json, os, time
import pandas as pd

CACHE_DIR = os.path.join(os.path.dirname(__file__), "cache")
os.makedirs(CACHE_DIR, exist_ok=True)

# Metro coordinates (lat, lon). Chosen as the HQ city for each ticker in
# universe.py -- fixed before any regression was run.
METROS = {
    "ATLANTA":    (33.7490, -84.3880),
    "CHARLOTTE":  (35.2271, -80.8431),
    "NASHVILLE":  (36.1627, -86.7816),
    "LOUISVILLE": (38.2527, -85.7585),
    "DALLAS":     (32.7767, -96.7970),
    "ORLANDO":    (28.5383, -81.3792),
    "LOSANGELES": (34.0522, -118.2437),
    "COLUMBUS":   (39.9612, -82.9988),
    "BALTIMORE":  (39.2904, -76.6122),
    "SEATTLE":    (47.6062, -122.3321),
    "PORTLAND":   (45.5152, -122.6784),
    "MILWAUKEE":  (43.0389, -87.9065),
}


def _fetch_metro(name, lat, lon, start, end, retries=4):
    url = ("https://archive-api.open-meteo.com/v1/archive"
           f"?latitude={lat}&longitude={lon}&start_date={start}&end_date={end}"
           "&daily=temperature_2m_max,temperature_2m_min,precipitation_sum"
           "&timezone=America%2FNew_York")
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            j = json.load(urllib.request.urlopen(req, timeout=60))
            d = j["daily"]
            df = pd.DataFrame({
                "tmax": d["temperature_2m_max"],
                "tmin": d["temperature_2m_min"],
                "precip": d["precipitation_sum"],
            }, index=pd.to_datetime(d["time"]))
            return df.dropna(how="all")
        except Exception as e:
            if attempt == retries - 1:
                print(f"  [weather] FAILED {name}: {type(e).__name__} {str(e)[:80]}")
                return None
            time.sleep(2 ** attempt)


def get_weather(start="1995-01-01", end="2024-12-31", verbose=True):
    """Return dict[metro] -> daily DataFrame[tmax,tmin,precip], cached per metro."""
    out = {}
    for name, (lat, lon) in METROS.items():
        fp = os.path.join(CACHE_DIR, f"wx_{name}.csv")
        if os.path.exists(fp):
            df = pd.read_csv(fp, index_col=0, parse_dates=True)
        else:
            df = _fetch_metro(name, lat, lon, start, end)
            time.sleep(0.3)
            if df is not None:
                df.to_csv(fp)
        if df is not None:
            out[name] = df
        if verbose:
            print(f"  [weather] {name}: {'ok ' + str(len(df)) + ' days' if df is not None else 'MISSING'}")
    return out


if __name__ == "__main__":
    wx = get_weather()
    print(f"\n{len(wx)}/{len(METROS)} metros fetched")
    for k, v in wx.items():
        print(f"  {k:<12} {v.index.min().date()} -> {v.index.max().date()}  ({len(v)} days)")
