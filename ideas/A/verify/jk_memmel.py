"""VERIFIER: Jobson-Korkie test with Memmel (2003) correction for Sharpe differences, as a second opinion on the
builder's paired stationary-bootstrap p-values (iid assumption; monthly Sharpe). Re-builds series via builder engine
(copied into this directory, unmodified)."""
import json, pickle
import numpy as np, pandas as pd
from scipy import stats
import data, engine as E

def jkm(a, b):
    df = pd.concat([a, b], axis=1).dropna().values
    T = len(df); m = df.mean(0); s = df.std(0, ddof=1); r = np.corrcoef(df.T)[0, 1]
    s1, s2 = m / s
    v = (2 - 2 * r + 0.5 * (s1**2 + s2**2 - 2 * s1 * s2 * r**2)) / T
    z = (s1 - s2) / np.sqrt(v)
    return (s1 - s2) * np.sqrt(12), z, 2 * stats.norm.sf(abs(z))

P = json.load(open("frozen_params.json")); K = P["k"]
rf = data.factors_monthly()[1]["RF"]
d, m = data.industries(); m = m.loc[:"2024-12-31"]
sigs = pickle.load(open("cache/ind_signals.pkl", "rb"))["sigs"]
S = {k: v.loc["1989-12-31":"2024-11-30"] for k, v in sigs.items()}
ex = lambda bt: bt["net"] - rf.reindex(bt.index)
EW = ex(E.backtest(E.ew_weights(S["OWN1"]), m, 0.001))
LO = {k: ex(E.backtest(E.rank_weights(S[k], frac=0.2, long_only=True), m, 0.001)) for k in ["OWN12", ("NET1", K), ("NET12", K)]}
print("Industries 1990-2024, LO top quintile, net excess, JK-Memmel:")
for a, b in [(("NET1", K), "EW"), (("NET12", K), "EW"), (("NET1", K), "OWN12"), (("NET12", K), "OWN12"), ("OWN12", "EW")]:
    dsr, z, p = jkm(LO[a], EW if b == "EW" else LO[b])
    print(f"  {str(a):<12} - {b:<6}: dSR {dsr:+.2f} z={z:+.2f} p={p:.3f}")

px = data.etf_prices().loc[:"2024-12-31"]; spy = px.pop("SPY"); px = px[data.ETFS]
daily = px.pct_change(fill_method=None); mpx = px.resample("ME").last()
mo = mpx.pct_change(fill_method=None); mo[mpx.isna()] = np.nan
spym = spy.resample("ME").last().pct_change(fill_method=None)
form = mo.index[(mo.index >= "2006-12-31") & (mo.index <= "2024-11-30")]
ES = E.build_signals(daily, mo, [K], form)
eEW = ex(E.backtest(E.ew_weights(ES["OWN1"]), mo, 0.0003))
SPY = ex(E.backtest(pd.DataFrame({"SPY": 1.0}, index=form), spym.to_frame("SPY"), 0.0002))
eLO = {k: ex(E.backtest(E.rank_weights(ES[k], n=5, long_only=True), mo, 0.0003)) for k in ["OWN12", ("NET1", K), ("NET12", K)]}
print("ETFs 2007-2024, LO top5, net excess, JK-Memmel:")
for a in [("NET1", K), ("NET12", K)]:
    for bn, b in [("EW", eEW), ("SPY", SPY), ("OWN12", eLO["OWN12"])]:
        dsr, z, p = jkm(eLO[a], b)
        print(f"  {str(a):<12} - {bn:<6}: dSR {dsr:+.2f} z={z:+.2f} p={p:.3f}")
# how much of NET12 LO's Sharpe comes from 2008? and cumulative 2019-2024 vs SPY
for nm, x in [("NET12 top5", eLO[("NET12", K)]), ("SPY", SPY), ("EW", eEW)]:
    print(f"  {nm:<10} SR excl. 2008: {E.sharpe(x.drop(x.loc['2008'].index)):+.2f}; "
          f"cum. total return 2019-2024: {float((1+x.loc['2019':'2024']+rf.loc['2019':'2024']).prod()-1):+.1%}")
