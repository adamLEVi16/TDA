"""Signals, backtest engine and statistics for Concept B (see SPEC.md)."""
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

from data_load import CACHE, french, universe

D = Path(__file__).parent
LOOKBACK, ALPHA, T, TAU = 60, 0.5, 3, 0.3
DESIGN = ("2005-01-01", "2014-12-31")
TEST = ("2015-01-01", "2024-12-31")


# ----------------------------------------------------------------------------------------------
# graph + residuals
def graph(window):
    """Adjacency |rho| thresholded at TAU, zero diagonal (original construction)."""
    c = np.corrcoef(window, rowvar=False)
    a = np.abs(c)
    np.fill_diagonal(a, 0)
    a[a < TAU] = 0
    return a, c


def residuals(R, tag):
    """Daily LR and NR residuals. Day i uses the graph from days i-60..i-1 and x = returns of day i."""
    p_lr, p_nr, p_st = (CACHE / f"{k}_{tag}.csv" for k in ("LR", "NR", "graphstats"))
    if p_lr.exists():
        rd = lambda p: pd.read_csv(p, index_col=0, parse_dates=True)
        return rd(p_lr), rd(p_nr), rd(p_st)
    X = R.values
    n = X.shape[1]
    eye = np.eye(n)
    lr, nr, st = [], [], []
    for i in range(LOOKBACK, len(X)):
        a, c = graph(X[i - LOOKBACK:i])
        deg = a.sum(1)
        dinv = np.diag(1.0 / np.sqrt(deg + 1e-8))
        lap = eye - dinv @ a @ dinv
        f = eye - ALPHA * lap
        x = X[i]
        lr.append(x - f @ f @ f @ x)
        nb = np.where(deg > 0, (a @ x) / np.where(deg > 0, deg, 1), np.nan)
        loo_mean = (x.sum() - x) / (n - 1)
        nr.append(x - np.where(deg > 0, nb, loo_mean))
        off = c[np.triu_indices(n, 1)]
        st.append(((deg == 0).mean(), (off < -TAU).sum() / max((np.abs(off) > TAU).sum(), 1), (np.abs(off) > TAU).mean()))
    idx = R.index[LOOKBACK:]
    LR = pd.DataFrame(lr, index=idx, columns=R.columns)
    NR = pd.DataFrame(nr, index=idx, columns=R.columns)
    ST = pd.DataFrame(st, index=idx, columns=["isolated_share", "neg_edge_share", "edge_density"])
    LR.to_csv(p_lr), NR.to_csv(p_nr), ST.to_csv(p_st)
    return LR, NR, ST


def check_cube(R, days=(100, 2000, 4000)):
    """Numerical check that the matrix cube equals the original fractional_matrix_power."""
    from scipy import linalg
    X, n = R.values, R.shape[1]
    out = []
    for i in days:
        a, _ = graph(X[i - LOOKBACK:i])
        deg = a.sum(1)
        dinv = np.diag(1.0 / np.sqrt(deg + 1e-8))
        f = np.eye(n) - ALPHA * (np.eye(n) - dinv @ a @ dinv)
        out.append(np.abs(np.real(linalg.fractional_matrix_power(f, T)) - f @ f @ f).max())
    return max(out)


def rebal_dates(index, freq):
    s = pd.Series(index, index=index)
    if freq == "D":
        return pd.DatetimeIndex(index)
    per = index.to_period("W-FRI" if freq == "W" else "M")
    return pd.DatetimeIndex(s.groupby(per).max().values)


def logcum(R, start_lag, length):
    """Compounded return over days t-start_lag-length+1 .. t-start_lag (known at close t)."""
    return np.expm1(np.log1p(R).rolling(length).sum().shift(start_lag))


def spillover(R, dates, own):
    """Correlation-weighted neighbour average of `own` (a signal frame) at each date; graph over t-59..t."""
    X = R.values
    pos = {d: i for i, d in enumerate(R.index)}
    rows, iso = [], []
    for d in dates:
        i = pos[d]
        if i + 1 < LOOKBACK or own.loc[d].isna().any():
            rows.append(np.full(R.shape[1], np.nan)); continue
        a, _ = graph(X[i + 1 - LOOKBACK:i + 1])
        deg = a.sum(1)
        v = own.loc[d].values
        loo = (v.sum() - v) / (len(v) - 1)
        rows.append(np.where(deg > 0, (a @ v) / np.where(deg > 0, deg, 1), loo))
        iso.append((deg == 0).mean())
    return pd.DataFrame(rows, index=dates, columns=R.columns), float(np.mean(iso)) if iso else np.nan


def build_signals(R, sectors, tag):
    """Dict id -> (signal frame on rebalance dates, rebalance freq)."""
    LR, NR, ST = residuals(R, tag)
    LR, NR = LR.reindex(R.index), NR.reindex(R.index)
    sig = {}
    dD, dW, dM = rebal_dates(R.index, "D"), rebal_dates(R.index, "W"), rebal_dates(R.index, "M")
    for name, E in (("LR", LR), ("NR", NR)):
        sig[f"{name}1"] = (E.loc[dD], "D")
        sig[f"{name}5"] = (E.rolling(5).sum().loc[dW], "W")
        sig[f"{name}21"] = (E.rolling(21).sum().loc[dM], "M")
        sig[f"{name}12_1"] = (E.rolling(231).sum().shift(21).loc[dM], "M")
    r21, r121, r5 = logcum(R, 0, 21), logcum(R, 21, 231), logcum(R, 0, 5)
    sp21, iso21 = spillover(R, dM, r21)
    sp121, iso121 = spillover(R, dM, r121)
    sig["SP21"] = (sp21, "M")
    sig["SP12_1"] = (sp121, "M")
    sig["MOM12_1"] = (r121.loc[dM], "M")
    sig["REV5"] = (r5.loc[dW], "W")
    sig["REV21"] = (r21.loc[dM], "M")
    sec = pd.Series(sectors).reindex(R.columns)
    ind = pd.DataFrame(index=r5.index, columns=R.columns, dtype=float)
    for s, cols in sec.groupby(sec).groups.items():
        cols = list(cols)
        if len(cols) == 1:
            other = r5.drop(columns=cols).mean(axis=1)
            ind[cols[0]] = r5[cols[0]] - other
        else:
            tot = r5[cols].sum(axis=1)
            for c in cols:
                ind[c] = r5[c] - (tot - r5[c]) / (len(cols) - 1)
    sig["IREV5"] = (ind.loc[dW], "W")
    info = {"isolated_share_daily_graph": float(ST["isolated_share"].mean()),
            "neg_edge_share_daily_graph": float(ST["neg_edge_share"].mean()),
            "edge_density_daily_graph": float(ST["edge_density"].mean()),
            "isolated_share_spillover_graph": iso21}
    return sig, info


# ----------------------------------------------------------------------------------------------
# engine
def target_weights(s, sign, nq, long_only=False):
    s = (sign * s).dropna()
    if len(s) < 2 * nq:
        return None
    order = s.sort_values(kind="mergesort")
    w = pd.Series(0.0, index=s.index)
    w[order.index[-nq:]] = 1.0 / nq
    if not long_only:
        w[order.index[:nq]] = -1.0 / nq
    return w


def backtest(R, signal, sign, nq, cost_bp=5.0, long_only=False, lag=0, ew=False):
    """Daily gross/net returns and turnover. Weights set at close t earn returns from t+1.

    lag=1: a signal known at close t is traded at close t+1 (R4 robustness).
    ew=True: equal-weight all stocks on the same rebalance dates (benchmark)."""
    cols = list(R.columns)
    X = R.values
    idx = R.index
    sig_dates = set(signal.index)
    pos_of = {d: i for i, d in enumerate(idx)}
    trade_at = {}
    for d in signal.index:
        j = pos_of[d] + lag
        if j < len(idx):
            trade_at[j] = d
    w = np.zeros(len(cols))
    formed = pd.NaT
    gross = np.zeros(len(idx)); turn = np.zeros(len(idx)); active = np.zeros(len(idx), bool)
    for i in range(len(idx)):
        g = float(w @ X[i])
        if w.any():
            assert formed < idx[i], "look-ahead: weights formed at/after the return date"
            active[i] = True
            gross[i] = g
            w = w * (1 + X[i]) / (1 + g)
        if i in trade_at:
            d = trade_at[i]
            assert d <= idx[i]
            if ew:
                tw = np.full(len(cols), 1.0 / len(cols))
            else:
                tw = target_weights(signal.loc[d], sign, nq, long_only)
                if tw is None:
                    continue
                tw = tw.reindex(cols).fillna(0).values
            turn[i] = np.abs(tw - w).sum()
            active[i] = active[i] or True
            w = tw
            formed = idx[i]
    out = pd.DataFrame({"gross": gross, "turnover": turn}, index=idx)
    first = np.argmax(active)
    out = out.iloc[first:]
    out["net"] = out["gross"] - out["turnover"] * cost_bp / 1e4
    return out


# ----------------------------------------------------------------------------------------------
# statistics
def nw_t(x, lags=10):
    x = pd.Series(x).dropna()
    m = sm.OLS(x.values, np.ones(len(x))).fit(cov_type="HAC", cov_kwds={"maxlags": lags})
    return float(m.tvalues[0]), float(m.pvalues[0])


def sr(x):
    x = np.asarray(x)
    return x.mean() / x.std(ddof=1) * np.sqrt(252) if x.std() > 0 else 0.0


def boot_idx(n, reps=2000, block=21, seed=12345):
    rng = np.random.default_rng(seed)
    nb = int(np.ceil(n / block))
    starts = rng.integers(0, n, size=(reps, nb))
    idx = (starts[:, :, None] + np.arange(block)[None, None, :]) % n
    return idx.reshape(reps, -1)[:, :n]


def sr_ci(x, reps=2000, block=21, seed=12345):
    x = np.asarray(x)
    b = x[boot_idx(len(x), reps, block, seed)]
    s = b.mean(1) / b.std(1, ddof=1) * np.sqrt(252)
    return np.percentile(s, [2.5, 97.5])


def sr_diff_ci(a, b, reps=2000, block=21, seed=12345):
    a, b = np.asarray(a), np.asarray(b)
    ii = boot_idx(len(a), reps, block, seed)
    A, B = a[ii], b[ii]
    d = A.mean(1) / A.std(1, ddof=1) * np.sqrt(252) - B.mean(1) / B.std(1, ddof=1) * np.sqrt(252)
    return np.percentile(d, [2.5, 97.5]), float(np.mean(d <= 0))


def maxdd(x):
    c = (1 + pd.Series(x)).cumprod()
    return float((c / c.cummax() - 1).min())


def summarize(bt, rf=None):
    g, n, t = bt["gross"], bt["net"], bt["turnover"]
    ex_g, ex_n = (g, n) if rf is None else (g - rf.reindex(g.index).fillna(0), n - rf.reindex(n.index).fillna(0))
    tg, _ = nw_t(ex_g)
    tn, pn = nw_t(ex_n)
    lo, hi = sr_ci(ex_n)
    be = g.mean() / t.mean() * 1e4 if t.mean() > 0 else np.nan
    return {"start": g.index[0].date(), "days": len(g),
            "ann_gross": g.mean() * 252, "ann_net": n.mean() * 252,
            "SR_gross": sr(ex_g), "SR_net": sr(ex_n), "t_gross": tg, "t_net": tn, "p_net": pn,
            "SR_net_CI": f"[{lo:+.2f},{hi:+.2f}]", "turn_yr": t.mean() * 252,
            "cost_yr": (g - n).mean() * 252, "breakeven_bp": be, "maxDD": maxdd(n)}


def factors():
    ff = french("F-F_Research_Data_5_Factors_2x3_daily_CSV.zip")
    mom = french("F-F_Momentum_Factor_daily_CSV.zip").set_axis(["UMD"], axis=1)
    return ff.join(mom, how="inner")


def ff_alpha(x, F, rf_sub=False):
    F = F.reindex(x.index).dropna()
    y = x.reindex(F.index)
    if rf_sub:
        y = y - F["RF"]
    X = sm.add_constant(F[["Mkt-RF", "SMB", "HML", "RMW", "CMA", "UMD"]])
    m = sm.OLS(y, X).fit(cov_type="HAC", cov_kwds={"maxlags": 10})
    return {"alpha_ann": m.params["const"] * 252, "t_alpha": m.tvalues["const"],
            "beta_mkt": m.params["Mkt-RF"], "beta_umd": m.params["UMD"], "R2": m.rsquared}


def fmt(df):
    pd.set_option("display.width", 300)
    pd.set_option("display.max_columns", 40)
    return df.to_string(float_format=lambda v: f"{v:+.3f}")
