"""Step 4: pre-listed robustness checks R1-R5 (SPEC.md) and post-hoc additions A1-A3 (SPEC.md Addendum 1)."""
import json

import numpy as np
import pandas as pd
import statsmodels.api as sm

from lib import (ALPHA, CACHE, D, DESIGN, LOOKBACK, TAU, TEST, backtest, build_signals, factors, ff_alpha, fmt,
                 logcum, nw_t, rebal_dates, sr, summarize)
from data_load import french, universe

pd.set_option("display.width", 250)
signs = json.load(open(D / "frozen_signs.json"))
R, spy, sectors, _ = universe()
sig, _ = build_signals(R, sectors, "full")
F = factors()
rf = F["RF"]
NQ = 16
T0, T1 = TEST

# ---------------------------------------------------------------- R1 cost sensitivity + R2 sub-periods
print("== R1/R2: TEST long-short, net Sharpe by cost level and by sub-period (5 bp)")
rows = []
bts = {}
for name, (s, freq) in sig.items():
    bt0 = backtest(R, s, signs[name], NQ, cost_bp=0.0).loc[T0:T1]
    bts[name] = bt0
    row = {"signal": name}
    for c in (0, 2, 5, 10):
        n = bt0["gross"] - bt0["turnover"] * c / 1e4
        row[f"SR_{c}bp"] = sr(n)
    n5 = bt0["gross"] - bt0["turnover"] * 5 / 1e4
    for a, b in (("2015", "2019"), ("2020", "2024")):
        x = n5.loc[a:b]
        row[f"SR5_{a}-{b}"] = sr(x)
        row[f"t5_{a}-{b}"] = nw_t(x)[0]
        row[f"SRgross_{a}-{b}"] = sr(bt0["gross"].loc[a:b])
    rows.append(row)
print(fmt(pd.DataFrame(rows).set_index("signal")))

# ---------------------------------------------------------------- R4 implementation lag
print("\n== R4: TEST, trade one day late (signal at close t, trade at close t+1), 5 bp")
for name in ("LR1", "NR1", "LR5", "NR5", "REV5", "IREV5"):
    s, _ = sig[name]
    a = summarize(backtest(R, s, signs[name], NQ).loc[T0:T1])
    b = summarize(backtest(R, s, signs[name], NQ, lag=1).loc[T0:T1])
    print(f"{name:<6} lag0: SRgross {a['SR_gross']:+.2f} SRnet {a['SR_net']:+.2f} | lag1: SRgross {b['SR_gross']:+.2f} "
          f"SRnet {b['SR_net']:+.2f} t_net {b['t_net']:+.2f} breakeven {b['breakeven_bp']:+.2f} bp")

# ---------------------------------------------------------------- R3 original 20-stock universe (18 with data from 2005)
orig = ["AAPL", "MSFT", "AMZN", "NVDA", "META", "GOOGL", "TSLA", "NFLX", "JPM", "PEP", "CSCO", "ORCL", "DIS", "BAC",
        "XOM", "IBM", "INTC", "AMD", "KO", "WMT"]            # GOOG in the original -> GOOGL here (same company)
o = [t for t in orig if t in R.columns]
print(f"\n== R3: original universe restricted to {len(o)} names with data from 2005: {o}")
R18 = R[o]
sig18, _ = build_signals(R18, sectors, "orig18")
rows = []
for name, (s, freq) in sig18.items():
    for per, (a, b) in (("design", DESIGN), ("test", TEST)):
        bt = backtest(R18, s, signs[name], 4).loc[a:b]
        m = summarize(bt)
        rows.append({"signal": name, "period": per, "SR_gross": m["SR_gross"], "SR_net": m["SR_net"],
                     "t_net": m["t_net"], "turn_yr": m["turn_yr"], "breakeven_bp": m["breakeven_bp"]})
print(fmt(pd.DataFrame(rows).set_index(["signal", "period"]).unstack("period")))
ew18 = R18.mean(axis=1)
for a, b in (DESIGN, TEST):
    e, s_ = ew18.loc[a:b], spy.loc[a:b]
    print(f"{a[:4]}-{b[:4]}: EW-18 ann mean {e.mean()*252:+.3f}  EW-83 {R.mean(axis=1).loc[a:b].mean()*252:+.3f}  SPY {s_.mean()*252:+.3f}")

# ---------------------------------------------------------------- R5 Fama-MacBeth for spillover
print("\n== R5: Fama-MacBeth, next-month return on cross-sectional z-scores (monthly slopes, NW 3 lags)")
dM = rebal_dates(R.index, "M")
nxt = np.expm1(np.log1p(R).groupby(R.index.to_period("M")).sum())
nxt.index = dM                      # month m return, indexed at month-end m
fwd = nxt.shift(-1)                 # next month's return, indexed at the formation month-end
Z = lambda df: df.sub(df.mean(1), axis=0).div(df.std(1), axis=0)
X = {k: Z(sig[k][0]) for k in ("SP21", "SP12_1", "REV21", "MOM12_1")}
for spec in (["SP21"], ["SP21", "REV21", "MOM12_1"], ["SP12_1"], ["SP12_1", "REV21", "MOM12_1"],
             ["SP21", "SP12_1", "REV21", "MOM12_1"]):
    sl = []
    for d in dM[:-1]:
        y = fwd.loc[d]
        xs = pd.concat([X[k].loc[d] for k in spec], axis=1)
        ok = xs.notna().all(1) & y.notna()
        if ok.sum() < 30:
            continue
        m = sm.OLS(y[ok], sm.add_constant(xs[ok])).fit()
        sl.append(pd.Series(m.params.values[1:], index=spec, name=d))
    sl = pd.DataFrame(sl)
    out = []
    for per, (a, b) in (("design", DESIGN), ("test", TEST)):
        s_ = sl.loc[a:b]
        out.append(per + ": " + "  ".join(f"{k} {s_[k].mean()*1e4:+.1f}bp/mo t={nw_t(s_[k], 3)[0]:+.2f}" for k in spec)
                   + f" (n={len(s_)})")
    print(f"{' + '.join(spec)}\n   " + "\n   ".join(out))

# ---------------------------------------------------------------- A2 is LR12_1 just momentum?
print("\n== A2: TEST, LR12_1 / NR12_1 long-short vs MOM12_1 long-short (net, 5 bp)")
lr = backtest(R, sig["LR12_1"][0], 1, NQ).loc[T0:T1]["net"]
nr = backtest(R, sig["NR12_1"][0], 1, NQ).loc[T0:T1]["net"]
mo = backtest(R, sig["MOM12_1"][0], 1, NQ).loc[T0:T1]["net"]
for nm, x in (("LR12_1", lr), ("NR12_1", nr)):
    m = sm.OLS(x, sm.add_constant(mo)).fit(cov_type="HAC", cov_kwds={"maxlags": 10})
    print(f"{nm}: corr with MOM12_1 {x.corr(mo):+.3f}; {nm} = a + b*MOM12_1: a={m.params['const']*252:+.3f}/yr "
          f"t={m.tvalues['const']:+.2f}, b={m.params.iloc[1]:+.3f}")
    d = x - mo
    print(f"   {nm} minus MOM12_1: {d.mean()*252:+.3f}/yr NW t={nw_t(d)[0]:+.2f}")

# ---------------------------------------------------------------- A3 concentration of long-only excess
print("\n== A3: TEST long-only excess over EW-83 (5 bp): leave-one-out contributions, then drop top 5 jointly")
for name in ("LR12_1", "MOM12_1"):
    def excess(cols):
        Rx = R[cols]
        s = sig[name][0][cols]
        lo = backtest(Rx, s, signs[name], NQ, long_only=True).loc[T0:T1]["net"]
        ew = backtest(Rx, s, 1, NQ, ew=True).loc[T0:T1]["net"]
        return lo - ew
    full = excess(list(R.columns))
    contrib = {c: full.mean() * 252 - excess([x for x in R.columns if x != c]).mean() * 252 for c in R.columns}
    top = sorted(contrib, key=contrib.get, reverse=True)[:5]
    red = excess([c for c in R.columns if c not in top])
    print(f"{name}: full excess {full.mean()*252:+.3f}/yr t={nw_t(full)[0]:+.2f}; top-5 LOO contributors "
          f"{[(c, round(contrib[c], 4)) for c in top]}; without them {red.mean()*252:+.3f}/yr t={nw_t(red)[0]:+.2f}")

# ---------------------------------------------------------------- A1 survivorship-free: 49 industries
print("\n== A1: Ken French 49 industry portfolios (daily VW), long-horizon signals, 10 bp, quintile = 10")
ind = french("49_Industry_Portfolios_daily_CSV.zip")
print(f"industry data {ind.index[0].date()}..{ind.index[-1].date()} {ind.shape}, missing share {ind.isna().mean().mean():.3%}")
ind = ind.loc[:"2024-12-31"]
path = CACHE / "LR_ind49.csv"
if path.exists():
    LRi = pd.read_csv(path, index_col=0, parse_dates=True)
    NRi = pd.read_csv(CACHE / "NR_ind49.csv", index_col=0, parse_dates=True)
else:
    Xv = ind.values
    n = Xv.shape[1]
    lr = np.full_like(Xv, np.nan); nr = np.full_like(Xv, np.nan)
    for i in range(LOOKBACK, len(Xv)):
        w = Xv[i - LOOKBACK:i]
        ok = ~np.isnan(w).any(0) & ~np.isnan(Xv[i])
        if ok.sum() < 20:
            continue
        c = np.corrcoef(w[:, ok], rowvar=False)
        a = np.abs(c); np.fill_diagonal(a, 0); a[a < TAU] = 0
        deg = a.sum(1); k = ok.sum()
        dinv = np.diag(1 / np.sqrt(deg + 1e-8)); f = np.eye(k) - ALPHA * (np.eye(k) - dinv @ a @ dinv)
        x = Xv[i, ok]
        lr[i, ok] = x - f @ f @ f @ x
        loo = (x.sum() - x) / (k - 1)
        nr[i, ok] = x - np.where(deg > 0, (a @ x) / np.where(deg > 0, deg, 1), loo)
    LRi = pd.DataFrame(lr, index=ind.index, columns=ind.columns)
    NRi = pd.DataFrame(nr, index=ind.index, columns=ind.columns)
    LRi.to_csv(path); NRi.to_csv(CACHE / "NR_ind49.csv")
dMi = rebal_dates(ind.index, "M")
isig = {"LR12_1": LRi.rolling(231).sum().shift(21).loc[dMi],
        "NR12_1": NRi.rolling(231).sum().shift(21).loc[dMi],
        "MOM12_1": logcum(ind, 21, 231).loc[dMi]}
Ri = ind.fillna(0.0)
rows = []
for name, s in isig.items():
    bt = backtest(Ri, s, signs[name], 10, cost_bp=10.0)
    for per, (a, b) in (("1927-2004", ("1927", "2004")), ("2005-2014", DESIGN), ("2015-2024", TEST)):
        m = summarize(bt.loc[a:b])
        rows.append({"signal": name, "period": per, "SR_gross": m["SR_gross"], "SR_net": m["SR_net"],
                     "t_net": m["t_net"], "ann_net": m["ann_net"], "turn_yr": m["turn_yr"], "breakeven_bp": m["breakeven_bp"]})
print(fmt(pd.DataFrame(rows).set_index(["signal", "period"])))
bl = {k: backtest(Ri, s, 1, 10, cost_bp=10.0)["net"] for k, s in isig.items()}
for k in ("LR12_1", "NR12_1"):
    for per, (a, b) in (("1927-2004", ("1927", "2004")), ("2015-2024", TEST)):
        xy = pd.concat([bl[k], bl["MOM12_1"]], axis=1, join="inner").loc[a:b]
        x, mo = xy.iloc[:, 0], xy.iloc[:, 1]
        m = sm.OLS(x, sm.add_constant(mo)).fit(cov_type="HAC", cov_kwds={"maxlags": 10})
        print(f"industries {k} vs MOM12_1 {per}: corr {x.corr(mo):+.3f}, intercept {m.params['const']*252:+.3f}/yr t={m.tvalues['const']:+.2f}")
