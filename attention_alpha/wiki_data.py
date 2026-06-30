"""
Free daily Wikipedia pageviews (Wikimedia REST API, no key) for consumer-brand
articles -- a proxy for public attention / consumer demand. Cached per article.

Data starts 2015-07-01 (API limit). Pageviews = how many people looked up a
brand's page that day; the hypothesis is that abnormal attention leads the
stock before the demand shows up in reported numbers.
"""
import urllib.request, urllib.parse, json, os, time
import pandas as pd

CACHE_DIR = os.path.join(os.path.dirname(__file__), "cache")
os.makedirs(CACHE_DIR, exist_ok=True)
UA = "tda-research/1.0 (academic backtest; contact via github)"
API = ("https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article/"
       "en.wikipedia/all-access/all-agents/{article}/daily/{start}/{end}")


def _fetch_article(article, start, end, retries=4):
    art = urllib.parse.quote(article, safe="")
    url = API.format(article=art, start=start, end=end)
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA})
            j = json.load(urllib.request.urlopen(req, timeout=45))
            items = j.get("items", [])
            if not items:
                return None
            idx = pd.to_datetime([it["timestamp"][:8] for it in items], format="%Y%m%d")
            return pd.Series([it["views"] for it in items], index=idx, name=article)
        except urllib.error.HTTPError as e:
            if e.code == 404:
                print(f"  [wiki] 404 (no such article): {article}")
                return None
            if attempt == retries - 1:
                print(f"  [wiki] FAILED {article}: HTTP {e.code}")
                return None
            time.sleep(2 ** attempt)
        except Exception as e:
            if attempt == retries - 1:
                print(f"  [wiki] FAILED {article}: {type(e).__name__} {str(e)[:60]}")
                return None
            time.sleep(2 ** attempt)


def get_pageviews(articles, start="20150701", end="20241231", verbose=True):
    """dict[article] -> daily pageview Series. Cached per article."""
    out = {}
    for a in articles:
        fn = a.replace("/", "_").replace("'", "").replace("&", "and")
        fp = os.path.join(CACHE_DIR, f"pv_{fn}.csv")
        if os.path.exists(fp):
            s = pd.read_csv(fp, index_col=0, parse_dates=True).iloc[:, 0]
            s.name = a
        else:
            s = _fetch_article(a, start, end)
            time.sleep(0.2)
            if s is not None and len(s) > 0:
                s.to_frame().to_csv(fp)
        if s is not None and len(s) > 0:
            out[a] = s
            if verbose:
                print(f"  [wiki] {a:<34} {s.index.min().date()} -> {s.index.max().date()} ({len(s)}d)")
        elif verbose:
            print(f"  [wiki] {a:<34} MISSING")
    return out


if __name__ == "__main__":
    from universe import TICKER_ARTICLE
    pv = get_pageviews(list(TICKER_ARTICLE.values()))
    print(f"\n{len(pv)}/{len(TICKER_ARTICLE)} articles fetched")
