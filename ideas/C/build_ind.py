"""Build daily indicators (SPEC.md 'Indicators'), each using data through the close of day t only.
Saves cache/ind_daily.pkl and cache/monthly.pkl."""
import time

import numpy as np
import pandas as pd
from ripser import ripser

from data import CACHE, industries_daily, market_daily

W, W_RV, TOPO_W, TAU = 60, 21, 30, 0.3

md = market_daily()
ind, _ = industries_daily()
X = ind.values
N = X.shape[1]
K_AR = N // 5
print(f"N industries={N}, absorption ratio uses top {K_AR} eigenvalues")

t0 = time.time()
rows = []
for i in range(W - 1, len(ind)):
    win = X[i - W + 1:i + 1]                      # rows i-59..i inclusive: data through close of day i
    cov = np.cov(win.T)
    sd = np.sqrt(np.diag(cov))
    c = cov / np.outer(sd, sd)
    iu = np.triu_indices(N, 1)
    ac = c[iu].mean()
    ev = np.sort(np.linalg.eigvalsh(cov))[::-1]
    ar = ev[:K_AR].sum() / ev.sum()
    d = np.sqrt(np.clip(2 * (1 - c), 0, None))
    np.fill_diagonal(d, 0)
    dg = ripser(d, maxdim=1, distance_matrix=True)["dgms"][1]
    dg = dg[np.isfinite(dg).all(axis=1)]
    h1n, h1p = len(dg), float((dg[:, 1] - dg[:, 0]).sum()) if len(dg) else 0.0
    Wm = np.where(c > TAU, c, 0.0)
    np.fill_diagonal(Wm, 0.0)
    deg = Wm.sum(1)
    deg = np.where(deg == 0, 1.0, deg)
    Dm = np.diag(1 / np.sqrt(deg))
    L = np.eye(N) - Dm @ Wm @ Dm
    fied = np.sort(np.linalg.eigvalsh(L))[1]
    rows.append((ind.index[i], ac, ar, h1n, h1p, fied))
print(f"network loop {time.time() - t0:.0f}s")

F = pd.DataFrame(rows, columns=["date", "AC60", "AR60", "H1N", "H1P", "FIED"]).set_index("date")
F["TOPOVOL"] = F["H1N"].rolling(TOPO_W).std() + F["H1P"].rolling(TOPO_W).std()
r = md["mkt"]
F = F.join(np.sqrt(252 * (r ** 2).rolling(W_RV).mean()).rename("RV"), how="outer")
idx = (1 + r).cumprod()
F["IDX"] = idx
F["SMA210"] = idx.rolling(210).mean()
F = F.reindex(md.index)

# monthly quantities
per = md.index.to_period("M")
RVM = np.sqrt(252 * (r ** 2).groupby(per).mean())
acm = {}
for p, g in ind.groupby(per):
    c = np.corrcoef(g.values.T)
    acm[p] = c[np.triu_indices(N, 1)].mean()
ACM = pd.Series(acm)
me = F.groupby(per).tail(1)                       # last trading day of each month
M = me.copy()
M.index = me.index.to_period("M")
M["RVM"], M["ACM"] = RVM, ACM
M["ME_DATE"] = me.index
M["SMA10M"] = M["IDX"].rolling(10).mean()
M["TREND_IN"] = (M["IDX"] > M["SMA10M"]).astype(float).where(M["SMA10M"].notna())
F.to_pickle(CACHE / "ind_daily.pkl")
M.to_pickle(CACHE / "monthly.pkl")

pd.set_option("display.width", 200)
print(F.dropna().describe().T.round(4).to_string())
print("\nfirst full row:", F.dropna().index[0].date())
print("\nSpearman correlations of daily indicators, 1927-1989 (design):")
print(F.loc["1927":"1989", ["RV", "AC60", "AR60", "H1N", "H1P", "TOPOVOL", "FIED"]].corr("spearman").round(2).to_string())
print("\nSpearman correlations of daily indicators, 1990-2024:")
print(F.loc["1990":"2024", ["RV", "AC60", "AR60", "H1N", "H1P", "TOPOVOL", "FIED"]].corr("spearman").round(2).to_string())
tv = F["TOPOVOL"].dropna()
share = (F["H1N"].rolling(30).std() / tv).median()
print(f"\nmedian share of TOPOVOL coming from the H1N term: {share:.3f}")
