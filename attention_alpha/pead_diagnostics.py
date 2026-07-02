"""
Diagnostics for the PEAD null -- labeled as diagnostics, NOT headline-hunting:
run AFTER the pre-registered tests came back null, to characterize WHERE the
anomaly lives/died, against the published record (PEAD attenuation in liquid
names post-2015, e.g. Martineau 2021). Multiple-testing caveat applies to
everything in this file by construction.
"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import statsmodels.api as sm
import pead_study as PS

def reg(d, sig, yvar, label):
    d = d.dropna(subset=[sig, yvar]).copy()
    d["sz"] = d.groupby("qtr")[sig].transform(
        lambda x: (x - x.mean()) / x.std() if x.std() > 0 else x * 0)
    m = sm.OLS(d[yvar], sm.add_constant(d[["sz"]])).fit(
        cov_type="cluster", cov_kwds={"groups": d["qtr"].astype(str)})
    print(f"  {label:<42} coef={m.params['sz']:+.4f}  t={m.tvalues['sz']:+.2f}  "
          f"p={m.pvalues['sz']:.3f}  (n={len(d)})")


def decile_spread(d, sig, yvar, label):
    d = d.dropna(subset=[sig, yvar]).copy()
    hi = d[d[sig] >= d[sig].quantile(0.9)][yvar]
    lo = d[d[sig] <= d[sig].quantile(0.1)][yvar]
    diff = hi.mean() - lo.mean()
    se = np.sqrt(hi.var()/len(hi) + lo.var()/len(lo))
    print(f"  {label:<42} top-bot decile drift spread={diff:+.3%}  t={diff/se:+.2f} "
          f"(n={len(hi)}+{len(lo)})")


def main():
    ev, px, spy = PS.build_events()
    pre = ev[ev["E"] < "2016-01-01"]
    post = ev[ev["E"] >= "2016-01-01"]
    print(f"events: pre-2016={len(pre)}  post-2016={len(post)}")

    print("\n[D1] ERA SPLIT -- the documented PEAD attenuation")
    for sig in ["reaction", "sue"]:
        reg(pre, sig, "drift61", f"{sig} -> drift61, PRE-2016")
        reg(post, sig, "drift61", f"{sig} -> drift61, POST-2016")

    print("\n[D2] EXTREMES ONLY (top/bottom decile spread, drift61)")
    for sig in ["reaction", "sue"]:
        decile_spread(pre, sig, "drift61", f"{sig}, PRE-2016")
        decile_spread(post, sig, "drift61", f"{sig}, POST-2016")

    print("\n[D3] TRADABLE, PRE-2016 ONLY (was it ever a real portfolio?)")
    s = PS.calendar_portfolio(ev[ev["E"] < "2016-01-01"], px, spy, "reaction")
    PS.perf(s.loc[:"2016-06"], "reaction-sorted L/S net, 2005-2016")
    s2 = PS.calendar_portfolio(ev[ev["E"] < "2016-01-01"], px, spy, "sue")
    PS.perf(s2.loc[:"2016-06"], "sue-sorted L/S net, 2011-2016")


if __name__ == "__main__":
    main()
