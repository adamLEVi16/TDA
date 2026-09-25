"""Scheduled FOMC decision days (day 0) from federalreserve.gov, per SPEC.md.

Reads fed/h{1994..2020}.htm (fomchistorical pages) and fed/cal.htm (fomccalendars page, 2021+),
downloading any that are missing, and writes fomc_dates.csv.
"""
import re
import urllib.request
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

HERE = Path(__file__).parent
FED = HERE / "fed"
MONTHS = {m: i for i, m in enumerate(["January", "February", "March", "April", "May", "June", "July", "August",
                                      "September", "October", "November", "December"], 1)}
ABBR = {"Jan": 1, "Feb": 2, "Mar": 3, "Apr": 4, "May": 5, "Jun": 6, "Jul": 7, "Aug": 8, "Sep": 9, "Oct": 10,
        "Nov": 11, "Dec": 12}


def page(name, url):
    FED.mkdir(exist_ok=True)
    p = FED / name
    if not p.exists():
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        p.write_bytes(urllib.request.urlopen(req).read())
    return p.read_text(errors="ignore")


def clean(s):
    return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", "", s)).strip()


rows = []
for y in range(1993, 2021):                                     # 1993 only anchors the Jan-1994 cycle clock
    html = page(f"h{y}.htm", f"https://www.federalreserve.gov/monetarypolicy/fomchistorical{y}.htm")
    for m in re.finditer(r"<h5[^>]*>(.*?)</h5>", html, re.S):
        title = clean(m.group(1))                                  # e.g. "June 30-July 1 Meeting - 1998"
        if "Meeting" not in title or "unscheduled" in title or "notation" in title.lower():
            continue
        cancelled = "cancelled" in title
        body = re.sub(r"\((unscheduled|cancelled)\)", "", title.split(" Meeting")[0]).strip()
        # last day of the meeting: "June 30-July 1" -> July 1; "February 3-4" -> Feb 4; "April/May 30-1" -> May 1
        names = [w for w in re.split(r"[\s/\-]+", body) if w in MONTHS or w[:3] in ABBR]   # also "Jan/Feb 31-1"
        mon, day = MONTHS.get(names[-1]) or ABBR[names[-1][:3]], int(re.findall(r"\d+", body)[-1])
        rows.append({"day0": date(y, mon, day), "source": f"fomchistorical{y}", "title": title, "cancelled": cancelled})

html = page("cal.htm", "https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm")
for ym in re.finditer(r"(\d{4}) FOMC Meetings", html):
    y = int(ym.group(1))
    if y < 2021:
        continue
    nxt = re.search(r"\d{4} FOMC Meetings", html[ym.end():])
    seg = html[ym.end(): ym.end() + (nxt.start() if nxt else len(html))]
    for mm in re.finditer(r'fomc-meeting__month[^>]*>(.*?)</div>\s*<div class="fomc-meeting__date[^>]*>(.*?)</div>', seg, re.S):
        month_txt, date_txt = clean(mm.group(1)), clean(mm.group(2))
        if "unscheduled" in date_txt.lower() or "notation" in date_txt.lower():
            continue
        cancelled = "cancel" in date_txt.lower()
        last_day = int(re.findall(r"\d+", date_txt)[-1])
        mon_names = re.split(r"/", month_txt)                    # "Apr/May" -> last month
        mon_name = mon_names[-1].strip()
        mon = MONTHS.get(mon_name) or ABBR.get(mon_name[:3])
        rows.append({"day0": date(y, mon, last_day), "source": "fomccalendars", "title": f"{month_txt} {date_txt} {y}",
                     "cancelled": cancelled})

df = pd.DataFrame(rows).sort_values("day0").drop_duplicates("day0").reset_index(drop=True)
# merge entries on consecutive days (e.g. 2003-09-15 and 2003-09-16) into one meeting, keeping the later date
keep = [i for i in range(len(df))
        if not (i + 1 < len(df) and df.loc[i + 1, "day0"] - df.loc[i, "day0"] <= timedelta(days=1))]
merged = df.loc[[i for i in range(len(df)) if i not in keep], "day0"].tolist()
df = df.loc[keep].reset_index(drop=True)
df.to_csv(HERE / "fomc_dates.csv", index=False)

print(f"{len(df)} scheduled meetings {df.day0.min()} .. {df.day0.max()}; merged-away earlier days: {merged}")
print("meetings per year:", df.groupby(pd.to_datetime(df.day0).dt.year).size().to_dict())
print("cancelled kept:", df[df.cancelled].day0.tolist())
print(df[pd.to_datetime(df.day0).dt.year.isin([1994, 2016, 2020, 2024, 2026])].to_string(index=False))
