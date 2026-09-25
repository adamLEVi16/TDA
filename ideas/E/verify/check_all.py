import pandas as pd, sys
sys.path.insert(0, ".")
from check_fomc_dates import statement_date, minutes_dates, HERE
df = pd.read_csv(HERE.parent / "fomc_dates.csv", parse_dates=["day0"])
pool = df[(df.day0 <= "2026-07-31") & (~df.cancelled)].day0.dt.date.tolist()
rows = []
for d in pool:
    su, sok = statement_date(d)
    mu, md = (None, None)
    if not sok:
        mu, md = minutes_dates(d)
    last = max(md) if md else None
    rows.append({"day0": d, "stmt": sok, "min_last": last, "min_ok": (last == d) if last else None})
out = pd.DataFrame(rows)
ok = out.apply(lambda r: bool(r.stmt) or bool(r.min_ok), axis=1)
print(f"{len(out)} scheduled day-0 dates 1994-02..2026-07; confirmed: {ok.sum()}")
print("unconfirmed:\n", out[~ok].to_string())
out.to_csv("fomc_check_all.csv", index=False)
