"""Re-run of the Dec-2025 Phase-4 notebook (TDA_Phase4_StrategyBacktest.ipynb, commit 7df04e6)
with the controls the notebook and the Jan-2026 paper never ran.

The residual, topology, regime, signal and cost logic below is the notebook's, with the same
parameters. Additions: gross vs net returns, the same signal with no filter / a realised-vol
filter on identical dates, the trade direction flipped, honest Sharpe standard errors, and
checks on what the topology feature actually measures.

Run from the repo root:  python audit/check_original.py
"""
import numpy as np
import pandas as pd
import statsmodels.api as sm
from ripser import ripser
from scipy import linalg
from scipy.sparse.csgraph import connected_components

from common import UNIVERSE, cagr, max_drawdown, prices, sharpe, sharpe_ci

LOOKBACK, ALPHA, T, TAU, N_POS, COST = 60, 0.5, 3, 0.3, 5, 0.0005

px = prices(UNIVERSE + ["SPY"], "2019-01-01", "2024-12-10")
spy = px.pop("SPY")
returns = px.ffill().bfill().pct_change().dropna()
spy_ret = spy.pct_change().reindex(returns.index)
print(f"data {returns.index[0].date()}..{returns.index[-1].date()}: {len(returns)} days x {returns.shape[1]} stocks")

# --- notebook features: Laplacian residuals and H1 statistics on every day -------------------
resid, h1_loops, h1_pers, connected, avg_corr = [], [], [], [], []
for i in range(LOOKBACK, len(returns)):
    corr = returns.iloc[i - LOOKBACK:i].corr().values
    adj = np.abs(corr.copy())
    np.fill_diagonal(adj, 0)
    adj[adj < TAU] = 0
    deg = adj.sum(1)
    d_inv = np.diag(1.0 / np.sqrt(deg + 1e-8))
    eye = np.eye(len(deg))
    lap = eye - d_inv @ adj @ d_inv
    x = returns.iloc[i].values
    resid.append(x - linalg.fractional_matrix_power(eye - ALPHA * lap, T) @ x)
    connected.append(connected_components(adj > 0, directed=False)[0] == 1)
    avg_corr.append(corr[np.triu_indices_from(corr, 1)].mean())
    dist = np.sqrt(np.clip(2 * (1 - corr), 0, None))
    np.fill_diagonal(dist, 0)
    dgm = ripser(dist, maxdim=1, distance_matrix=True)["dgms"][1]
    dgm = dgm[~np.isinf(dgm).any(axis=1)]
    h1_loops.append(len(dgm))
    h1_pers.append((dgm[:, 1] - dgm[:, 0]).sum() if len(dgm) else 0.0)

idx = returns.index[LOOKBACK:]
RES = pd.DataFrame(resid, index=idx, columns=returns.columns)
F = pd.DataFrame({"h1_loops": h1_loops, "h1_persistence": h1_pers,
                  "connected": connected, "avg_corr": avg_corr}, index=idx)
topo_vol = F["h1_loops"].rolling(30).std() + F["h1_persistence"].rolling(30).std()
spy_vol20 = spy_ret.rolling(20).std().reindex(idx)      # trailing, known at close t


def signals(res, sign=+1):
    """Notebook rule: +1/5 on the 5 largest residuals, -1/5 on the 5 smallest (sign=-1 flips it)."""
    s = pd.DataFrame(0.0, index=res.index, columns=res.columns)
    for d in res.index:
        r = sign * res.loc[d]
        s.loc[d, r.nlargest(N_POS).index] = 1.0 / N_POS
        s.loc[d, r.nsmallest(N_POS).index] = -1.0 / N_POS
    return s


def backtest(sig, unstable=None):
    s = sig.copy()
    if unstable is not None:
        s.loc[unstable[unstable].index] = 0
    lag = s.shift(1)
    gross = (lag * returns.loc[lag.index]).sum(axis=1).dropna()
    turnover = lag.dropna().diff().abs().sum(axis=1)
    return gross, gross - turnover * COST, turnover


# --- notebook walk-forward: 3y train / 1y test, threshold = train 75th percentile ------------
folds, start = [], 3 * 252 + 60
while start + 252 <= len(returns):
    train, test = returns.index[start - 3 * 252:start], returns.index[start:start + 252]
    topo_unstable = topo_vol.loc[test] > topo_vol.loc[train[60:]].quantile(0.75)
    vol_unstable = spy_vol20.loc[test] > spy_vol20.loc[train[60:]].quantile(0.75)
    folds.append((test, topo_unstable, vol_unstable))
    start += 252

print("\n== Walk-forward folds (notebook reported fold Sharpes -0.42 and -1.24, combined -0.56)")
for k, (test, tu, vu) in enumerate(folds, 1):
    g, n, t = backtest(signals(RES.loc[test]), tu)
    print(f"fold {k} {test[0].date()}..{test[-1].date()}: topology flags {tu.mean():.1%} of days, "
          f"realised-vol flags {vu.mean():.1%} | net SR {sharpe(n):+.2f}, gross SR {sharpe(g):+.2f}, "
          f"CAGR {cagr(n):+.2%}, avg daily cost {(t * COST).mean():.4%}")

rows = []
for name, sign, which in [("ORIGINAL: long +resid / short -resid, topology filter", +1, "topo"),
                          ("same signal, no filter", +1, None),
                          ("same signal, realised-vol filter (SPY 20d, train q75)", +1, "vol"),
                          ("direction flipped (true reversal), topology filter", -1, "topo"),
                          ("direction flipped (true reversal), no filter", -1, None)]:
    parts = [backtest(signals(RES.loc[test], sign), {"topo": tu, "vol": vu, None: None}[which])
             for test, tu, vu in folds]
    g, n, t = (pd.concat(p) for p in zip(*parts))
    sr, se, ci, p = sharpe_ci(n)
    rows.append({"strategy": name, "gross SR": sharpe(g), "net SR": sr, "SE": se,
                 "95% CI": f"[{ci[0]:+.2f}, {ci[1]:+.2f}]", "p": p, "net CAGR": cagr(n),
                 "max DD": max_drawdown(n), "turnover/day": t.mean(), "cost drag/yr": (t * COST).mean() * 252})
pd.set_option("display.width", 250)
print("\n== Combined out-of-sample, 5 bp costs, identical dates for every row")
print(pd.DataFrame(rows).to_string(index=False, float_format=lambda v: f"{v:+.3f}"))

# --- paper claims about the graph and the regime filter --------------------------------------
print(f"\n== Graph at tau=0.3 is connected on {F['connected'].mean():.1%} of days (paper v12 says '>95%')")

unstable = topo_vol > topo_vol.quantile(0.75)       # Phase-3 notebook rule (full-sample threshold)
spy_vol_ann = spy_vol20 * np.sqrt(252)
fwd_vol_ann = spy_ret.rolling(20).std().shift(-20).reindex(idx) * np.sqrt(252)
cmp = pd.DataFrame({"flagged unstable": unstable, "SPY vol 20d": spy_vol_ann,
                    "SPY vol next 20d": fwd_vol_ann, "avg corr": F["avg_corr"]}).dropna()
print("\n== What the topology filter selects (Phase-3 rule, full-sample 75th percentile)")
print(cmp.groupby("flagged unstable").mean().round(3).to_string())
print("share of days flagged, by year:", unstable.groupby(unstable.index.year).mean().round(2).to_dict())
covid = unstable.loc["2020-02-15":"2020-04-30"]
print(f"COVID crash 2020-02-15..2020-04-30: {int(covid.sum())} of {len(covid)} days flagged unstable")

print("\n== Spearman correlations: the H1 statistics mostly track (inverse) average correlation")
feat = pd.DataFrame({"h1_loops": F["h1_loops"], "h1_persistence": F["h1_persistence"],
                     "topo_vol": topo_vol, "avg_corr": F["avg_corr"],
                     "spy_vol20": spy_vol20, "fwd_vol20": fwd_vol_ann}).dropna()
print(feat.corr(method="spearman").round(2).to_string())

print("\n== Does topo_vol help forecast next-20d SPY vol once trailing vol is known? (non-overlapping, HC1)")
nov = feat.iloc[::20]
for extra in ([], ["topo_vol"], ["avg_corr"]):
    X = sm.add_constant(pd.concat([np.log(nov[["spy_vol20"]]), nov[extra]], axis=1))
    m = sm.OLS(np.log(nov["fwd_vol20"]), X).fit(cov_type="HC1")
    print(f"log fwd vol ~ log trailing vol {'+ ' + extra[0] if extra else '':<12} R2={m.rsquared:.3f}  "
          + "  ".join(f"t({k})={m.tvalues[k]:+.2f}" for k in m.params.index if k != "const"))
