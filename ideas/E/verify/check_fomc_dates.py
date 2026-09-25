"""Independent check of day-0 dates: read the dates out of the FOMC minutes text and the statement release page.

Does not use the analyst's parser or the fomchistorical/fomccalendars headings. URLs are constructed from the
candidate date (and the day before, for two-day meetings); the date is then read from the document text.
"""
import re
import time
import urllib.request
import urllib.error
from datetime import date, timedelta, datetime
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).parent
CACHE = HERE / "fedcheck"
CACHE.mkdir(exist_ok=True)
MONTHS = "January|February|March|April|May|June|July|August|September|October|November|December"


def get(url):
    fn = CACHE / re.sub(r"[^A-Za-z0-9]+", "_", url)[-150:]
    if fn.exists():
        t = fn.read_text(errors="ignore")
        return (None if t == "__404__" else t)
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    try:
        t = urllib.request.urlopen(req, timeout=30).read().decode("utf-8", "ignore")
    except urllib.error.HTTPError as e:
        t = "__404__" if e.code == 404 else None
        if t:
            fn.write_text(t)
        return None
    time.sleep(0.3)
    fn.write_text(t)
    return t


def clean(s):
    return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", s)).strip()


def dates_in(txt):
    out = []
    for m in re.finditer(rf"({MONTHS})\s+(\d{{1,2}}),\s+(\d{{4}})", txt):
        out.append(datetime.strptime(f"{m.group(1)} {m.group(2)} {m.group(3)}", "%B %d %Y").date())
    return out


def minutes_dates(d):
    """Return (url, dates mentioned in the opening 'meeting ... was held' sentence) or (None, None)."""
    cands = []
    for dd in (d, d - timedelta(days=1), d - timedelta(days=3)):   # file may be named by first day of meeting
        s = dd.strftime("%Y%m%d")
        if dd.year <= 1995:
            cands.append(f"https://www.federalreserve.gov/fomc/MINUTES/{dd.year}/{s}min.htm")
        elif dd.year <= 2007:
            cands.append(f"https://www.federalreserve.gov/fomc/minutes/{s}.htm")
        else:
            cands.append(f"https://www.federalreserve.gov/monetarypolicy/fomcminutes{s}.htm")
    for u in cands:
        t = get(u)
        if not t:
            continue
        c = clean(t)
        m = re.search(r"meeting of the Federal Open Market Committee (?:and the Board of Governors[^.]*?)?was held(.{0,600})", c)
        if m:
            seg = m.group(1)
            seg = seg.split(" PRESENT")[0].split("Present:")[0]
            return u, dates_in(seg)
    return None, None


def statement_date(d):
    s = d.strftime("%Y%m%d")
    if d.year >= 2006:
        u = f"https://www.federalreserve.gov/newsevents/pressreleases/monetary{s}a.htm"
    elif d.year >= 2003:
        u = f"https://www.federalreserve.gov/boarddocs/press/monetary/{d.year}/{s}/default.htm"
    else:
        u = f"https://www.federalreserve.gov/boarddocs/press/general/{d.year}/{s}/"
    t = get(u)
    if not t:
        return None, None
    c = clean(t)
    fomc = ("Federal Open Market Committee" in c) or ("FOMC" in c)
    rel = d.strftime("%B ") + str(d.day) + d.strftime(", %Y")
    return u, (fomc and rel in c)


if __name__ == "__main__":
    df = pd.read_csv(HERE.parent / "fomc_dates.csv", parse_dates=["day0"])
    pool = df[(df.day0 <= "2026-07-31") & (~df.cancelled)].day0.dt.date.tolist()
    rs = np.random.default_rng(20260925)
    sample = sorted(rs.choice(len(pool), size=30, replace=False))
    picks = [pool[i] for i in sample]
    special = [date(2003, 9, 16), date(1998, 7, 1), date(2012, 8, 1), date(2013, 5, 1), date(2017, 11, 1),
               date(2023, 2, 1), date(2024, 5, 1), date(1995, 2, 1), date(2006, 1, 31), date(2024, 11, 7),
               date(2020, 4, 29), date(2020, 1, 29)]
    picks = sorted(set(picks) | set(special))
    rows = []
    for d in picks:
        su, sok = statement_date(d)
        mu, md = minutes_dates(d)
        last = max(md) if md else None
        rows.append({"day0": d, "stmt_found_same_date": sok, "minutes_dates": ",".join(str(x) for x in (md or [])),
                     "minutes_last_day": last, "minutes_match": (last == d) if last else None,
                     "random": d in [pool[i] for i in sample]})
        print(rows[-1], flush=True)
    out = pd.DataFrame(rows)
    out.to_csv(HERE / "fomc_check.csv", index=False)
    ok = out.apply(lambda r: bool(r.stmt_found_same_date) or bool(r.minutes_match), axis=1)
    print(f"\nchecked {len(out)} dates ({out.random.sum()} random); confirmed by statement or minutes: {ok.sum()}")
    print("minutes confirm:", (out.minutes_match == True).sum(), " statement confirm:", (out.stmt_found_same_date == True).sum())
    print("NOT confirmed:\n", out[~ok].to_string())
    print("minutes contradict:\n", out[out.minutes_match == False].to_string())
