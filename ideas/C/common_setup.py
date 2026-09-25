"""Shared loading of indicators / market data and construction of all rule targets."""
import numpy as np
import pandas as pd

from data import CACHE, market_daily, market_monthly
from engine import INDS, NET_INDS, binary_target, vt_target

md = market_daily()
mm = market_monthly()
F = pd.read_pickle(CACHE / "ind_daily.pkl")
M = pd.read_pickle(CACHE / "monthly.pkl")
M["logRV"] = np.log(M["RV"])
M["logRVM"] = np.log(M["RVM"])
M["RVM2"] = M["RVM"] ** 2
ME = pd.DatetimeIndex(M["ME_DATE"].values)
FIRST_X = pd.Period("1926-11", "M")
DESIGN = ("1929-01-01", "1989-12-31")
TEST = ("1990-01-01", "2024-12-31")
SIGMA_STAR = md["mkt"].loc["1927-01-01":"1989-12-31"].std() * np.sqrt(252)

# weekly rebalance dates: last trading day of each calendar week
WK = pd.DatetimeIndex(pd.Series(md.index, index=md.index).groupby(md.index.to_period("W")).last().values)


def monthly_to_dates(s):
    s = s.copy()
    s.index = pd.DatetimeIndex(M.loc[s.index, "ME_DATE"].values) if isinstance(s.index, pd.PeriodIndex) else s.index
    return s


def trend_target(freq="M"):
    if freq == "M":
        t = M["TREND_IN"].copy()
        t.index = ME
        return t
    t = (F["IDX"] > F["SMA210"]).astype(float).where(F["SMA210"].notna())
    return t.reindex(WK)


def vt_weekly(regs, cap):
    """Weekly version: coefficients from the last month-end fit, applied to current daily regressors."""
    from engine import vt_forecasts
    Mx = M.copy()
    Mx["logRV"] = np.log(Mx["RV"])
    f, s2 = vt_forecasts(Mx, regs, FIRST_X)
    # recover coefficients by refitting is costly; instead re-run expanding OLS and store betas
    import statsmodels.api as sm
    Mxx = Mx.loc[FIRST_X:]
    X = sm.add_constant(Mxx[regs]).values
    y = Mxx["logRVM"].shift(-1).values
    betas, s2s = {}, {}
    for j in range(len(Mxx)):
        Xk, yk = X[:j], y[:j]
        ok = np.isfinite(Xk).all(1) & np.isfinite(yk)
        if ok.sum() < 24:
            continue
        b, *_ = np.linalg.lstsq(Xk[ok], yk[ok], rcond=None)
        betas[Mxx["ME_DATE"].iloc[j]] = b
        s2s[Mxx["ME_DATE"].iloc[j]] = (yk[ok] - Xk[ok] @ b).var(ddof=X.shape[1])
    B = pd.DataFrame(betas).T
    S2 = pd.Series(s2s)
    daily = pd.DataFrame({"logRV": np.log(F["RV"])})
    for r in regs:
        if r != "logRV":
            daily[r] = F[r]
    wk = daily.reindex(WK)
    Bw = B.reindex(WK, method="ffill")                    # last month-end fit at or before the weekly date
    S2w = S2.reindex(WK, method="ffill")
    Xw = np.column_stack([np.ones(len(wk))] + [wk[r].values for r in regs])
    f = (Xw * Bw.values).sum(1)
    return pd.Series(np.minimum(cap, SIGMA_STAR / np.exp(f + S2w.values / 2)), index=WK)


def all_targets(frozen=None, cap=1.0, freq="M"):
    """Dict rule -> target weight Series at rebalance dates. If frozen is None, returns the full binary grid."""
    dates = ME if freq == "M" else WK
    T = {}
    T["TREND10"] = trend_target(freq)
    if freq == "M":
        rv_me = F["RV"].reindex(ME)
        T["VT-naive"] = np.minimum(cap, SIGMA_STAR / rv_me)
        T["VT-RV"] = vt_target(M, ["logRV"], SIGMA_STAR, cap, FIRST_X)
        for x in NET_INDS:
            T[f"VT-RV+{x}"] = vt_target(M, ["logRV", x], SIGMA_STAR, cap, FIRST_X)
    else:
        T["VT-naive"] = np.minimum(cap, SIGMA_STAR / F["RV"].reindex(WK))
        T["VT-RV"] = vt_weekly(["logRV"], cap)
        for x in NET_INDS:
            T[f"VT-RV+{x}"] = vt_weekly(["logRV", x], cap)
    T["VT-RV x TREND10"] = T["VT-RV"] * T["TREND10"].reindex(T["VT-RV"].index)
    if frozen is None:
        for x in INDS:
            for d in ("high", "low"):
                for q in (0.5, 0.75, 0.9):
                    T[f"B-{x}|{d}|{q}"] = binary_target(F[x], d, q, dates)
    else:
        for x in INDS:
            d, q = frozen[x]["direction"], frozen[x]["q"]
            T[f"B-{x}"] = binary_target(F[x], d, q, dates)
    return T
