"""Does anything predict next-week USD/JPY? Run: python run_tests.py [--download]

1. Univariate tests of each factor against next-week returns (Newey-West t-stats, split samples)
2. Same-week fit of rate-spread changes, for contrast (explains moves, but not tradeable)
3. Walk-forward multi-factor models (OLS, ridge, gradient boosting, fixed-sign composite)
4. Best single factor traded in 5 staggered weekly tranches vs. an always-long benchmark
5. Daily horizon: one-day signals vs. next-day return
"""
import sys
import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import norm
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge

import data

warnings.filterwarnings("ignore")
pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 20)
COST = 0.0001  # 1bp per unit of position traded


def nw(f, tgt, lags=4):
    d = pd.concat([f.rename("f"), tgt.rename("y")], axis=1).dropna()
    z = (d.f - d.f.mean()) / d.f.std()
    m = sm.OLS(d.y, sm.add_constant(z)).fit(cov_type="HAC", cov_kwds={"maxlags": lags})
    return m.tvalues.iloc[1], 100 * m.rsquared, len(d)


def univariate(X, y, xs):
    rows = []
    for c in X.columns:
        t, r2, n = nw(X[c], y)
        rows.append({"factor": c, "t_1990_2026": t, "r2_pct": r2, "n": n,
                     "t_1990_2007": nw(X[c][:"2007"], y[:"2007"])[0],
                     "t_2008_2026": nw(X[c]["2008":], y["2008":])[0],
                     "t_2016_2026": nw(X[c]["2016":], y["2016":])[0],
                     "t_spot+carry": nw(X[c], xs)[0]})
    return pd.DataFrame(rows).set_index("factor")


def same_week(y):
    fx = data.fred("DEXJPUS")[data.START:]
    wk = fx.groupby(fx.index.to_period("W-FRI")).tail(1)
    # rates as of the FX date itself (includes the prior close), same-week changes
    p = data.panel(wk.index + pd.Timedelta(days=1))
    p.index = wk.index
    r = np.log(wk).diff()
    out = {}
    for name, s in {"d(US2y-JGB2y)": p["us2"] - p["jgb2"], "d(US10y-JGB10y)": p["us10"] - p["jgb10"],
                    "d(US10y)": p["us10"]}.items():
        d = pd.concat([r, s.diff()], axis=1).dropna()["2008":]
        out[name] = sm.OLS(d.iloc[:, 0], sm.add_constant(d.iloc[:, 1])).fit().rsquared
    return pd.Series(out, name="same-week R2, 2008-2026")


def walk_forward(X, y, xs, start="2000-01-01", refit=13):
    feats = list(X.columns)
    prior = {"carry_short": 1, "mom_12w": 1, "d_spread2y_1w": 1, "misalign_z": -1}
    d = pd.concat([X, y, xs.rename("xs")], axis=1).dropna()
    i0 = d.index.searchsorted(pd.Timestamp(start))
    preds = {m: pd.Series(np.nan, index=d.index) for m in ["fixed_sign_composite", "ols", "ridge", "gbm"]}
    for i in range(i0, len(d), refit):
        tr, te = d.iloc[:i - 1], d.iloc[i:i + refit]  # last train target is realised before te starts
        mu, sd = tr[feats].mean(), tr[feats].std()
        ztr, zte = ((tr[feats] - mu) / sd).clip(-4, 4), ((te[feats] - mu) / sd).clip(-4, 4)
        ytr = tr["ret_next"]
        ctr = sum(s * ztr[k] for k, s in prior.items())
        cte = sum(s * zte[k] for k, s in prior.items())
        preds["fixed_sign_composite"].iloc[i:i + refit] = max(np.polyfit(ctr, ytr, 1)[0], 0) * cte.values
        preds["ols"].iloc[i:i + refit] = sm.OLS(ytr, sm.add_constant(ztr)).fit() \
            .predict(sm.add_constant(zte, has_constant="add")).values
        preds["ridge"].iloc[i:i + refit] = Ridge(alpha=2.0 * len(tr)).fit(ztr, ytr).predict(zte)
        preds["gbm"].iloc[i:i + refit] = HistGradientBoostingRegressor(
            max_depth=2, learning_rate=0.03, max_iter=150, min_samples_leaf=50, random_state=0
        ).fit(ztr, ytr).predict(zte)

    def score(p, tgt, a, b):
        e = pd.concat([p, tgt], axis=1).dropna()[a:b]
        f, r = e.iloc[:, 0], e.iloc[:, 1]
        cw = r ** 2 - ((r - f) ** 2 - f ** 2)  # Clark-West vs. zero forecast
        cwt = sm.OLS(cw, np.ones(len(cw))).fit(cov_type="HAC", cov_kwds={"maxlags": 4}).tvalues.iloc[0]
        pos = np.sign(f)
        pnl = pos * r - COST * pos.diff().abs().fillna(0)
        return {"oos_r2_pct": 100 * (1 - ((r - f) ** 2).sum() / (r ** 2).sum()), "clark_west_p": 1 - norm.cdf(cwt),
                "hit_rate": (np.sign(f) == np.sign(r)).mean(), "sharpe_net": pnl.mean() / pnl.std() * np.sqrt(52)}

    tables = {}
    for tname, tgt in {"spot": d["ret_next"], "spot+carry": d["xs"]}.items():
        for a, b in [("2000", "2026"), ("2000", "2012"), ("2013", "2026")]:
            tab = pd.DataFrame({m: score(p, tgt, a, b) for m, p in preds.items()}).T
            bench = d[a:b]
            tab.loc["benchmark: always long", "sharpe_net"] = \
                bench[{"spot": "ret_next", "spot+carry": "xs"}[tname]].mean() / \
                bench[{"spot": "ret_next", "spot+carry": "xs"}[tname]].std() * np.sqrt(52)
            tables[f"target={tname}, {a}-{b}"] = tab
    return tables


def staggered_spread_momentum():
    """Daily: each day open a 1-week tranche = sign(5-day change in US2y-JGB2y as of the prior close).
    Position = average of the 5 live tranches. P&L is NY-noon to NY-noon."""
    fx = data.fred("DEXJPUS")[data.START:]
    j = data.jgb()
    us2, us3m = data.fred("DGS2"), data.fred("DTB3")
    a = lambda s, b=0: data.asof_before(s, fx.index, b)
    r = np.log(fx).diff().shift(-1)
    carry = (a(us3m) - a(j["jgb1"])) / 100 / 252
    pos = np.sign((a(us2) - a(j["jgb2"])) - (a(us2, 5) - a(j["jgb2"], 5))).rolling(5).mean()
    rows = []
    for lo, hi in [("1990", "2026"), ("1990", "2007"), ("2008", "2026"), ("2016", "2026"), ("2021", "2026")]:
        for name, p, tgt in [("spread momentum (net 1bp)", pos, r + carry),
                             ("always long USD/JPY", pd.Series(1.0, index=fx.index), r + carry)]:
            e = pd.concat([p.rename("p"), tgt.rename("r")], axis=1).dropna()[lo:hi]
            pnl = e.p * e.r - COST * e.p.diff().abs().fillna(0)
            rows.append({"period": f"{lo}-{hi}", "strategy": name,
                         "sharpe": pnl.mean() / pnl.std() * np.sqrt(252),
                         "ann_ret_pct": pnl.mean() * 25200, "ann_vol_pct": pnl.std() * np.sqrt(252) * 100})
    return pd.DataFrame(rows).set_index(["period", "strategy"])


def daily_horizon():
    fx = data.fred("DEXJPUS")[data.START:]
    j = data.jgb()
    a = lambda s, b=0: data.asof_before(s, fx.index, b)
    us2, us10, vix = data.fred("DGS2"), data.fred("DGS10"), data.fred("VIXCLS")
    y = np.log(fx).diff().shift(-1)
    sig = {"d(US2y-JGB2y), prior day": (a(us2) - a(j["jgb2"])) - (a(us2, 1) - a(j["jgb2"], 1)),
           "d(US10y), prior day": a(us10) - a(us10, 1),
           "d log VIX, prior day": np.log(a(vix)) - np.log(a(vix, 1)),
           "USD/JPY return, prior day": np.log(fx).diff()}
    return pd.DataFrame({k: dict(zip(["t", "r2_pct", "n"], nw(s, y, lags=5))) for k, s in sig.items()}).T


if __name__ == "__main__":
    if "--download" in sys.argv:
        data.download()
    X, y, carry = data.weekly_features()
    xs = y + carry
    print("=== 1. Univariate: factor at week t vs USD/JPY log return t -> t+1 ===")
    print(univariate(X, y, xs).round(2))
    print("\n=== 2. Same-week fit (contemporaneous, not tradeable) ===")
    print(same_week(y).round(3))
    print("\n=== 3. Walk-forward, all factors, trained from 1990, refit quarterly ===")
    for k, t in walk_forward(X.drop(columns=[c for c in X if c.startswith("cot")]), y, xs).items():
        print(f"\n-- {k}\n{t.round(3)}")
    print("\n=== 4. Best single factor, 5 staggered weekly tranches (no weekday cherry-picking) ===")
    print(staggered_spread_momentum().round(2))
    print("\n=== 5. Daily horizon: signal known at day t vs USD/JPY return t -> t+1 ===")
    print(daily_horizon().round(3))
