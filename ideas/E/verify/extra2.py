import sys; sys.path.insert(0, ".")
import numpy as np, pandas as pd
from reimpl import ff, day0, fomc_week, tom, report, show, sr, strategy

print("FF Mkt-RF on 2020-03-18: %.2f%%" % (100 * ff.mktrf.loc["2020-03-18"]))
d = ff.loc["2011-04-01":]
dd = d.index.values.astype("datetime64[D]")
f_all = np.isin(dd, day0)
f_drop = np.isin(dd, day0[day0 != np.datetime64("2020-03-18")])
show([report("FOMC0 as analyst (incl. cancelled 3/18/20)", d, f_all), report("FOMC0 excl. cancelled 3/18/20", d, f_drop)],
     "FOMC0 OOS 2011-04..2026-07, cancelled-meeting sensitivity (1 bp; bootstrap 5000)")
# SPY version too
px = pd.read_csv("../cache/spy.csv", index_col=0, parse_dates=True)["SPY"]; spy = px.pct_change().dropna()
idx = spy.loc["2011-04-01":"2026-07-31"].index.intersection(ff.index)
ds = pd.DataFrame({"mkt": spy.reindex(idx), "rf": ff.rf.reindex(idx)})
show([report("SPY FOMC0 incl 3/18", ds, np.isin(idx.values.astype('datetime64[D]'), day0), boot=False),
      report("SPY FOMC0 excl 3/18", ds, np.isin(idx.values.astype('datetime64[D]'), day0[day0 != np.datetime64('2020-03-18')]), boot=False)], "SPY FOMC0")

# constant-mix comparison at equal (zero) cost
print("\n== timing vs constant mix, both at 0 bp")
for lab, a, fl in [("EVEN", "2017-01-01", lambda d: fomc_week(d.index.values, day0) % 2 == 0), ("TOM", "2006-01-01", lambda d: tom(d.index))]:
    d = ff.loc[a:]; f = fl(d); wbar = f.mean()
    s = strategy(d.mkt.values, d.rf.values, f, 0.0); c = wbar * d.mkt.values + (1 - wbar) * d.rf.values
    cagr = lambda x: 100 * (np.prod(1 + x) ** (252 / len(x)) - 1)
    print(f"{lab}: w={wbar:.3f} timing SR {sr(s-d.rf.values):.2f} CAGR {cagr(s):.2f}% | const SR {sr(c-d.rf.values):.2f} CAGR {cagr(c):.2f}%")

# paper Panel C $1 values: does B=$15.22 correspond to earning zero (not T-bills) when out?
d = ff.loc["1994-01-01":"2016-12-31"]
d0x = np.sort(np.append(day0, np.datetime64("1993-12-21")))
w = fomc_week(d.index.values, d0x); even = (w % 2 == 0)
print("\npaper Panel C $1: A=7.68 B=15.22 C=0.51 | ours with cash=RF: B=%.2f C=%.2f | cash=0: B=%.2f C=%.2f" % (
    np.prod(np.where(even, 1 + d.mkt, 1 + d.rf)), np.prod(np.where(~even, 1 + d.mkt, 1 + d.rf)),
    np.prod(np.where(even, 1 + d.mkt, 1.0)), np.prod(np.where(~even, 1 + d.mkt, 1.0))))
