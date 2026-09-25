"""Signal construction, portfolio backtest with drift-aware costs, and statistics for Concept A (see SPEC.md)."""
import numpy as np
import pandas as pd
import statsmodels.api as sm

WIN, MIN_OBS, ALPHA, T_STEPS = 252, 240, 0.5, 3
NW_LAGS, BOOT_BLOCK, BOOT_N = 6, 6, 2000


# ----------------------------------------------------------------------------------------------- signals
def knn_weights(C, k, rng=None):
    """Row-normalised weights on the k highest-correlation neighbours (self excluded).
    If rng is given: k RANDOM neighbours (placebo), same weight formula."""
    n = C.shape[0]
    W = np.zeros_like(C)
    for i in range(n):
        c = C[i].copy()
        c[i] = -np.inf
        if rng is None:
            nb = np.argsort(-c)[:k]
        else:
            nb = rng.choice(np.delete(np.arange(n), i), size=min(k, n - 1), replace=False)
        w = np.clip(C[i, nb], 0, None)
        W[i, nb] = w / w.sum() if w.sum() > 0 else 1.0 / len(nb)
    return W


def lap_gap(C, k, x):
    """Original-code smoothing on a symmetrised kNN graph: h=(I-alpha L)^T x, return h - x."""
    n = C.shape[0]
    nbmask = np.zeros((n, n), bool)
    for i in range(n):
        c = C[i].copy(); c[i] = -np.inf
        nbmask[i, np.argsort(-c)[:k]] = True
    A = np.where(nbmask | nbmask.T, np.clip(C, 0, None), 0.0)
    np.fill_diagonal(A, 0)
    deg = A.sum(axis=1)
    dinv = np.diag(1.0 / np.sqrt(deg + 1e-8))
    L = np.eye(n) - dinv @ A @ dinv
    M = np.linalg.matrix_power(np.eye(n) - ALPHA * L, T_STEPS)
    return M @ x - x


def build_signals(daily, monthly, ks, formation_dates, placebo_rng=None, verbose=False):
    """Return {(signal, k): DataFrame[formation t x assets]} plus 'OWN1','OWN12' and 'VOL' (trailing daily vol).
    Uses only daily rows with date <= t and monthly rows with date <= t (asserted)."""
    out = {}
    names = monthly.columns
    rows = {key: {} for key in ["OWN1", "OWN12", "VOL"] +
            [(s, k) for s in ["NET1", "NET12", "LAPGAP1", "NET1_PURE"] for k in ks]}
    dvals, didx = daily.values, daily.index
    for t in formation_dates:
        end = didx.searchsorted(t, side="right")          # daily rows with date <= t
        if end < WIN:
            continue
        win = daily.iloc[end - WIN:end]
        assert win.index[-1] <= t
        mhist = monthly.loc[:t].iloc[-12:]
        if len(mhist) < 12 or mhist.index[-1] != t:
            continue
        assert mhist.index[-1] <= t
        ok = (win.notna().sum() >= MIN_OBS) & mhist.notna().all()
        cols = names[ok.values]
        if len(cols) < 6:
            continue
        own1 = mhist[cols].iloc[-1].values
        own12 = (1 + mhist[cols].iloc[:-1]).prod().values - 1
        C = win[cols].corr().values
        vol = win[cols].std().values * np.sqrt(252)
        rows["OWN1"][t] = pd.Series(own1, cols)
        rows["OWN12"][t] = pd.Series(own12, cols)
        rows["VOL"][t] = pd.Series(vol, cols)
        X = np.column_stack([np.ones(len(cols)), own1, own12])
        for k in ks:
            W = knn_weights(C, k, placebo_rng)
            n1 = W @ own1
            rows[("NET1", k)][t] = pd.Series(n1, cols)
            rows[("NET12", k)][t] = pd.Series(W @ own12, cols)
            if placebo_rng is None:
                rows[("LAPGAP1", k)][t] = pd.Series(lap_gap(C, k, own1), cols)
                beta = np.linalg.lstsq(X, n1, rcond=None)[0]
                rows[("NET1_PURE", k)][t] = pd.Series(n1 - X @ beta, cols)
    for key, d in rows.items():
        if d:
            out[key] = pd.DataFrame(d).T.reindex(columns=names)
    return out


# ------------------------------------------------------------------------------------------- portfolios
def rank_weights(sig, frac=None, n=None, long_only=False):
    """Equal-weight top (and bottom) bucket of each row. frac = quantile fraction, or n = fixed count."""
    V = sig.values
    W = np.zeros_like(V, dtype=float)
    for r in range(V.shape[0]):
        ok = np.where(~np.isnan(V[r]))[0]
        m = n if n is not None else max(1, int(round(frac * len(ok))))
        if len(ok) < (m if long_only else 2 * m):
            continue
        order = ok[np.argsort(-V[r, ok], kind="stable")]
        W[r, order[:m]] = 1.0 / m
        if not long_only:
            W[r, order[-m:]] = -1.0 / m
    return pd.DataFrame(W, index=sig.index, columns=sig.columns)


def ew_weights(sig):
    W = sig.notna().astype(float)
    return W.div(W.sum(axis=1), axis=0).fillna(0.0)


def invvol_weights(vol):
    iv = 1.0 / vol
    return iv.div(iv.sum(axis=1), axis=0).fillna(0.0)


def backtest(W, monthly, cost):
    """W: weights decided at formation month-end t (index). Earns monthly return of the NEXT month.
    Returns DataFrame[return month] with gross, net, turnover. Missing next-month returns earn 0 (counted)."""
    midx = monthly.index
    res, prev_drift, n_missing = [], None, 0
    for t, w in W.iterrows():
        pos = midx.searchsorted(t, side="right")
        if pos >= len(midx):
            break
        t1 = midx[pos]
        assert t1 > t                                   # timing: return strictly after formation
        r = monthly.loc[t1, W.columns]
        held = w != 0
        n_missing += int((r[held].isna()).sum())
        r = r.fillna(0.0)
        wv = w.values
        base = np.zeros_like(wv) if prev_drift is None else prev_drift
        turnover = np.abs(wv - base).sum()
        gross = float(wv @ r.values)
        prev_drift = wv * (1 + r.values) / (1 + gross)
        res.append((t1, gross, gross - cost * turnover, turnover))
    out = pd.DataFrame(res, columns=["month", "gross", "net", "turnover"]).set_index("month")
    out.attrs["n_missing_held"] = n_missing
    return out


# ------------------------------------------------------------------------------------------- statistics
def nw_mean(x, lags=NW_LAGS):
    x = pd.Series(x).dropna()
    fit = sm.OLS(x.values, np.ones(len(x))).fit(cov_type="HAC", cov_kwds={"maxlags": lags})
    m, se = fit.params[0], fit.bse[0]
    return m, fit.tvalues[0], (m - 1.96 * se, m + 1.96 * se)


def stationary_idx(n, rng, block=BOOT_BLOCK):
    p = 1.0 / block
    idx = np.empty(n, int)
    idx[0] = rng.integers(n)
    for i in range(1, n):
        idx[i] = rng.integers(n) if rng.random() < p else (idx[i - 1] + 1) % n
    return idx


def sharpe(x):
    x = np.asarray(x)
    return x.mean() / x.std(ddof=1) * np.sqrt(12) if x.std() > 0 else np.nan


def boot_sharpe(x, seed=0, n=BOOT_N):
    x = np.asarray(pd.Series(x).dropna())
    rng = np.random.default_rng(seed)
    b = [sharpe(x[stationary_idx(len(x), rng)]) for _ in range(n)]
    return np.nanpercentile(b, [2.5, 97.5])


def boot_sharpe_diff(a, b, seed=0, n=BOOT_N):
    """Paired stationary bootstrap of Sharpe(a) - Sharpe(b). Returns diff, 95% CI, two-sided p."""
    df = pd.concat([pd.Series(a), pd.Series(b)], axis=1).dropna().values
    rng = np.random.default_rng(seed)
    d0 = sharpe(df[:, 0]) - sharpe(df[:, 1])
    bs = []
    for _ in range(n):
        i = stationary_idx(len(df), rng)
        bs.append(sharpe(df[i, 0]) - sharpe(df[i, 1]))
    bs = np.array(bs)
    p = 2 * min((bs <= 0).mean(), (bs >= 0).mean())
    return d0, np.percentile(bs, [2.5, 97.5]), p


def max_dd(x):
    c = (1 + pd.Series(x)).cumprod()
    return float((c / c.cummax() - 1).min())


def summarize(bt, rf=None, label="", boot=True):
    """One-line stats. rf: monthly RF series to subtract (long-only); None for self-financing L/S."""
    out = {}
    for col in ["gross", "net"]:
        x = bt[col].copy()
        if rf is not None:
            x = x - rf.reindex(x.index)
        m, t, ci = nw_mean(x)
        out[col] = dict(ann_mean=12 * m, t=t, ci=(12 * ci[0], 12 * ci[1]), sharpe=sharpe(x),
                        vol=x.std() * np.sqrt(12))
    xn = bt["net"] - (rf.reindex(bt.index) if rf is not None else 0)
    sci = boot_sharpe(xn) if boot else (np.nan, np.nan)
    to = bt["turnover"].iloc[1:].mean() if len(bt) > 1 else np.nan
    be = bt["gross"].mean() / to * 1e4 if to > 0 else np.inf
    if rf is not None:
        be = (bt["gross"] - rf.reindex(bt.index)).mean() / to * 1e4 if to > 0 else np.inf
    g, n = out["gross"], out["net"]
    return (f"{label:<34} {bt.index[0]:%Y-%m}..{bt.index[-1]:%Y-%m} n={len(bt):4d} | "
            f"gross {g['ann_mean']:+6.2%} SR {g['sharpe']:+.2f} t={g['t']:+.2f} | "
            f"net {n['ann_mean']:+6.2%} [{n['ci'][0]:+.1%},{n['ci'][1]:+.1%}] SR {n['sharpe']:+.2f} "
            f"[{sci[0]:+.2f},{sci[1]:+.2f}] t={n['t']:+.2f} vol {n['vol']:.1%} | "
            f"TO {to:.2f}/mo BE {be:5.0f}bp MDD {max_dd(bt['net']):.0%}"), out


def fama_macbeth(sigs, monthly, regressors, dates=None):
    """Cross-sectional OLS each month of r(t+1) on z-scored regressors; NW t of the mean slopes."""
    midx = monthly.index
    slopes = {}
    for t in sigs[regressors[0]].index:
        if dates is not None and t not in dates:
            continue
        pos = midx.searchsorted(t, side="right")
        if pos >= len(midx):
            break
        t1 = midx[pos]
        X = pd.concat([sigs[r].loc[t] for r in regressors], axis=1, keys=range(len(regressors))).dropna()
        y = monthly.loc[t1, X.index]
        keep = y.notna()
        X, y = X[keep], y[keep]
        if len(y) < len(regressors) + 5:
            continue
        Z = (X - X.mean()) / X.std(ddof=0)
        Z.insert(0, "c", 1.0)
        slopes[t1] = np.linalg.lstsq(Z.values, y.values, rcond=None)[0][1:]
    S = pd.DataFrame(slopes).T
    S.columns = [str(r) for r in regressors]
    return S


def spanning(y, X, lags=NW_LAGS):
    """OLS y = a + X b with NW SEs. Returns fitted model."""
    df = pd.concat([y.rename("y"), X], axis=1).dropna()
    return sm.OLS(df["y"], sm.add_constant(df.drop(columns="y"))).fit(cov_type="HAC", cov_kwds={"maxlags": lags})
