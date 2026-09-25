import sys; sys.path.insert(0, ".")
import numpy as np, pandas as pd
from reimpl import ff, day0, fomc_week, tom, report, show, sr, strategy, newey_west_t

# ---- SPY branch check (analyst's cached yfinance file), own alignment
px = pd.read_csv("../cache/spy.csv", index_col=0, parse_dates=True)["SPY"]
spy = px.pct_change().dropna()
both = pd.concat([spy.rename("spy"), ff.mkt], axis=1, join="inner")
for a, b in [("2006-01-01", "2026-07-31"), ("2017-01-01", "2026-07-31")]:
    x = both.loc[a:b]
    print(f"SPY vs FF mkt {a}..{b}: n={len(x)} corr={x.corr().iloc[0,1]:.4f} mean diff (bp/day)={1e4*(x.spy-x.mkt).mean():.2f}"
          f"  max |ret| spy={x.spy.abs().max():.3f}  zero-ret days spy={(x.spy==0).sum()}")
print("FF dates missing from SPY 2006-2026-07:", len(ff.loc['2006':'2026-07'].index.difference(spy.index)),
      " SPY dates missing from FF:", len(spy.loc['2006':'2026-07-31'].index.difference(ff.index)))
rows = []
for lab, a, rule in [("SPY EVEN 17-", "2017-01-01", "EVEN"), ("SPY TOM 06-", "2006-01-01", "TOM")]:
    idx = spy.loc[a:"2026-07-31"].index.intersection(ff.index)
    d = pd.DataFrame({"mkt": spy.reindex(idx), "rf": ff.rf.reindex(idx)})
    f = fomc_week(idx.values, day0) % 2 == 0 if rule == "EVEN" else tom(idx)
    rows.append(report(lab, d, f, boot=False))
show(rows, "SPY (1 bp)")

# ---- cost sensitivity: zero cost and 5 bp
rows = []
for c in (0.0, 5.0):
    d = ff.loc["2017-01-01":]; rows.append(report(f"EVEN 17- {c:g}bp", d, fomc_week(d.index.values, day0) % 2 == 0, cost_bp=c, boot=False))
    d = ff.loc["2006-01-01":]; rows.append(report(f"TOM 06- {c:g}bp", d, tom(d.index), cost_bp=c, boot=False))
    d = ff.loc["2011-04-01":]; rows.append(report(f"FOMC0 11- {c:g}bp", d, np.isin(d.index.values.astype('datetime64[D]'), day0), cost_bp=c, boot=False))
show(rows, "cost sensitivity")

# ---- post-PUBLICATION start dates instead of post-sample (CMVJ JF Oct 2019; L&M JF Feb 2015; M&X FAJ Mar/Apr 2008)
rows = []
d = ff.loc["2019-10-01":]; rows.append(report("EVEN 2019-10-", d, fomc_week(d.index.values, day0) % 2 == 0, boot=False))
d = ff.loc["2015-02-01":]; rows.append(report("FOMC0 2015-02-", d, np.isin(d.index.values.astype('datetime64[D]'), day0), boot=False))
d = ff.loc["2008-04-01":]; rows.append(report("TOM 2008-04-", d, tom(d.index), boot=False))
show(rows, "post-publication starts (1 bp)")

# ---- is OOS informative? test OOS in/out difference against the in-sample magnitude (and half of it)
print("\n== OOS difference vs in-sample magnitude (NW SE)")
cases = [("EVEN", ("1994-01-01", "2016-12-31"), ("2017-01-01", None), lambda d: fomc_week(d.index.values, day0) % 2 == 0),
         ("FOMC0", ("1994-09-01", "2011-03-31"), ("2011-04-01", None), lambda d: np.isin(d.index.values.astype('datetime64[D]'), day0)),
         ("TOM", ("1926-07-01", "2005-12-31"), ("2006-01-01", None), lambda d: tom(d.index))]
from scipy.stats import norm
for nm, ins, oos, fl in cases:
    di = ff.loc[ins[0]:ins[1]]; do = ff.loc[oos[0]:]
    bi, ti, sei = newey_west_t((di.mkt - di.rf).values, fl(di))
    bo, to, seo = newey_west_t((do.mkt - do.rf).values, fl(do))
    z_full = (bo - bi) / seo; z_half = (bo - bi / 2) / seo
    print(f"{nm:6s} in-sample diff {1e4*bi:6.2f} bp  OOS diff {1e4*bo:6.2f} bp (SE {1e4*seo:5.2f})  "
          f"H0 OOS=in-sample: z={z_full:5.2f} p={2*norm.sf(abs(z_full)):.3f} | H0 OOS=half: z={z_half:5.2f} p={2*norm.sf(abs(z_half)):.3f}"
          f" | 95% CI OOS [{1e4*(bo-1.96*seo):.1f}, {1e4*(bo+1.96*seo):.1f}]")

# ---- week 7+ days in OOS and effect of the 2020-03-18 choice on the paper definition
d = ff.loc["2017-01-01":]
w = fomc_week(d.index.values, day0)
print("\nOOS week counts (paper def):", pd.Series(w).value_counts().sort_index().to_dict())
d0_alt = day0[day0 != np.datetime64("2020-03-18")]
w2 = fomc_week(d.index.values, d0_alt)
print("OOS week counts without 2020-03-18:", pd.Series(w2).value_counts().sort_index().to_dict())
f_alt_paper = (w2 % 2 == 0) & (w2 <= 6)          # paper drops days beyond week 6 -> treat as out
rows = [report("EVEN no-0318 wk<=6", d, f_alt_paper, boot=False), report("EVEN no-0318 as coded", d, w2 % 2 == 0, boot=False)]
show(rows, "cancelled-meeting sensitivity, restricting even weeks to <=6 as in the paper")
