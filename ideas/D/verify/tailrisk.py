"""Descriptive tail-risk illustration (SPEC.md): short-variance proxy, SVXY, ^PUT, VXX. Not a strategy test."""
import numpy as np, pandas as pd
from scipy.stats import skew
from vrp_common import *
pd.set_option("display.width", 200)
P = build_panel()
fd = french_daily()

print("== Short 1-month variance swap proxy: payoff_{t+1} = IV_t - RV_{t+1} (VIX^2 as strike; approximation)")
pay = (P["IV"] - P["RV"].shift(-1)).dropna() * 12 * 1e4   # annualised variance points (VIX^2 units)
for lab, s in {"1990-02..2026-07": pay, "2000-01..2026-07": pay.loc["1999-12":]}.items():
    t = s.mean() / nw_se_mean(s.values, 6)
    print(f"{lab}: n={len(s)} mean={s.mean():+.1f} var-pts/month  median={s.median():+.1f}  NW t={t:+.2f}  "
          f"frac>0={(s>0).mean():.1%}  skew={skew(s):+.2f}  min={s.min():+.1f}  max={s.max():+.1f}")
print("Worst 6 months (origin month t; loss realised in t+1):")
w6 = pay.sort_values().head(6)
print(pd.DataFrame({"payoff_varpts": w6.round(1), "VIX_t": P["VIX"].reindex(w6.index).round(1),
                    "realised_vol_next_ann%": (np.sqrt(12 * P["RV"].shift(-1)) * 100).reindex(w6.index).round(1)}).to_string())
cum = pay.cumsum()
print(f"Cumulative payoff: total {pay.sum():+.0f}; worst single month {pay.min():+.0f} = {-pay.min()/pay.mean():.0f} average months of carry")
print(f"Sum of payoffs lost in the worst 1% of months: {pay[pay <= pay.quantile(0.01)].sum():+.0f} vs total {pay.sum():+.0f}")

def dd_report(name, px, rf_d=None):
    px = px.dropna(); r = px.pct_change().dropna()
    dd = px / px.cummax() - 1
    yrs = (px.index[-1] - px.index[0]).days / 365.25
    cagr = (px.iloc[-1] / px.iloc[0]) ** (1 / yrs) - 1
    print(f"\n{name}: {px.index[0].date()}..{px.index[-1].date()} CAGR={cagr:+.2%} ann vol={r.std()*np.sqrt(252):.1%} "
          f"maxDD={dd.min():.1%} (trough {dd.idxmin().date()})")
    print("  worst 5 days:", ", ".join(f"{d.date()} {v:+.1%}" for d, v in r.sort_values().head(5).items()))
    return r

svxy = yf_close("SVXY"); put = yf_close("^PUT"); vxx = yf_close("VXX"); spy = yf_close("SPY")
r_sv = dd_report("SVXY (short VIX-futures ETP; -1x until 2018-02-27, -0.5x after)", svxy)
for d in ["2018-02-02", "2018-02-05", "2018-02-06", "2020-03-09", "2020-03-12", "2020-03-16"]:
    if pd.Timestamp(d) in r_sv.index: print(f"  SVXY {d}: {r_sv[d]:+.1%}")
print(f"  SVXY close 2018-02-02 -> 2018-02-06: {svxy['2018-02-06']/svxy['2018-02-02']-1:+.1%}; "
      f"2018-01-12 -> 2018-02-08: {svxy['2018-02-08']/svxy['2018-01-12']-1:+.1%}")
print(f"  SVXY 2020-02-19 -> 2020-03-18: {svxy['2020-03-18']/svxy['2020-02-19']-1:+.1%}")
dd_report("SPY same window as SVXY", spy.loc[svxy.index[0]:])
r_put = dd_report("CBOE S&P500 PutWrite index ^PUT (price index level; no fees/costs)", put)
r_spy = dd_report("SPY same window as ^PUT", spy.loc[put.index[0]:])
# monthly Sharpe comparison over common window, excess of RF
pm = put.groupby(put.index.to_period("M")).last().pct_change(); sm = spy.groupby(spy.index.to_period("M")).last().pct_change()
pm.index = pm.index.to_timestamp("M"); sm.index = sm.index.to_timestamp("M")
D = pd.DataFrame({"put": pm, "spy": sm, "rf": P["rf"]}).dropna().loc["1996-09":"2026-07"]
for c in ("put", "spy"):
    ex = D[c] - D["rf"]; sr, se = sharpe_lo(ex)
    print(f"  monthly {c}: {D.index[0].date()}..{D.index[-1].date()} Sharpe={sr:.3f} [95% CI {sr-1.96*se:.3f},{sr+1.96*se:.3f}] skew={skew(ex):+.2f} worst month={ex.min():+.1%}")
r = paired_bootstrap((D["put"] - D["rf"]).values, (D["spy"] - D["rf"]).values)
print(f"  ^PUT - SPY: dSharpe={r['d_sr']:+.3f} CI[{r['sr_ci'][0]:+.3f},{r['sr_ci'][1]:+.3f}] p(<=0)={r['p_sr']:.3f}")
print(f"  ^PUT 2008-09..2009-02 return {put['2009-02-27']/put['2008-08-29']-1:+.1%} vs SPY {spy['2009-02-27']/spy['2008-08-29']-1:+.1%}")
print(f"  ^PUT 2020-02-19..2020-03-23 return {put['2020-03-23']/put['2020-02-19']-1:+.1%} vs SPY {spy['2020-03-23']/spy['2020-02-19']-1:+.1%}")
dd_report("VXX (long VIX futures ETN, current series)", vxx)
