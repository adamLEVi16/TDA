"""Backtest engine and statistics for Concept C (see SPEC.md)."""
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import optimize, stats

SEED, NBOOT, BLOCK, NWLAG = 20260925, 5000, 12, 6
INDS = ["RV", "AC60", "AR60", "H1N", "H1P", "TOPOVOL", "FIED"]
NET_INDS = ["AC60", "AR60", "H1N", "H1P", "TOPOVOL", "FIED"]


# ----------------------------------------------------------------------------------------------- simulation
def simulate(target, r_m, rf, start, end, borrow_spread=0.0):
    """target: Series of target market weights indexed by rebalance dates (set at that date's CLOSE).
    Returns daily DataFrame (gross return, turnover at that close, weight in effect) for days in [start, end].
    The weight in effect on day t is the one set at the last rebalance close strictly before t, drifted."""
    days = r_m.loc[start:end].index
    tgt = target.dropna()
    first = tgt.index[tgt.index < days[0]]
    assert len(first), "need a rebalance date before the first sample day"
    w = float(tgt.loc[first[-1]])                          # position set at close before sample start
    src_date = first[-1]
    tgt_in = tgt.loc[days[0]:days[-1]]
    rm, rff = r_m.reindex(days).values, rf.reindex(days).values
    out_g, out_t, out_w = np.empty(len(days)), np.zeros(len(days)), np.empty(len(days))
    tgt_map = dict(zip(tgt_in.index, tgt_in.values))
    for k, d in enumerate(days):
        assert src_date < d                                # timing: weight was set strictly before day d
        out_w[k] = w
        g = w * rm[k] + (1 - w) * rff[k] - max(w - 1, 0) * borrow_spread / 252
        out_g[k] = g
        wd = w * (1 + rm[k]) / (1 + g) if (1 + g) != 0 else w     # drift to close of d
        if d in tgt_map and not np.isnan(tgt_map[d]):
            out_t[k] = abs(tgt_map[d] - wd)
            w, src_date = float(tgt_map[d]), d
        else:
            w = wd
    return pd.DataFrame({"gross": out_g, "turn": out_t, "w": out_w, "rf": rff, "rm": rm}, index=days)


def net(sim, cost_bp):
    return sim["gross"] - sim["turn"] * cost_bp / 1e4


def monthly(r):
    return (1 + r).groupby(r.index.to_period("M")).prod() - 1


# ----------------------------------------------------------------------------------------------- statistics
def sharpe_m(ex):
    return ex.mean() / ex.std() * np.sqrt(12)


def nw_t(x, lags=NWLAG):
    m = sm.OLS(np.asarray(x), np.ones(len(x))).fit(cov_type="HAC", cov_kwds={"maxlags": lags})
    return float(m.tvalues[0])


def _block_idx(T, rng):
    nb = int(np.ceil(T / BLOCK))
    starts = rng.integers(0, T, nb)
    return (starts[:, None] + np.arange(BLOCK)[None, :]).ravel()[:T] % T


def boot_indices(T):
    rng = np.random.default_rng(SEED)
    return [_block_idx(T, rng) for _ in range(NBOOT)]


_BI = {}


def _bi(T):
    if T not in _BI:
        _BI[T] = np.array(boot_indices(T))
    return _BI[T]


def boot_sharpe_ci(ex):
    a = np.asarray(ex)
    B = a[_bi(len(a))]
    s = B.mean(1) / B.std(1, ddof=1) * np.sqrt(12)
    return np.percentile(s, [2.5, 97.5])


def boot_diff(ex_a, ex_b):
    """Paired circular block bootstrap of SR_a - SR_b; returns (diff, 95% CI, two-sided p via centred distn)."""
    a, b = np.asarray(ex_a), np.asarray(ex_b)
    I = _bi(len(a))
    A, Bm = a[I], b[I]
    da = A.mean(1) / A.std(1, ddof=1) * np.sqrt(12) - Bm.mean(1) / Bm.std(1, ddof=1) * np.sqrt(12)
    d = sharpe_m(pd.Series(a)) - sharpe_m(pd.Series(b))
    p = np.mean(np.abs(da - d) >= abs(d))
    return d, np.percentile(da, [2.5, 97.5]), p


def jkm(ex_a, ex_b):
    """Jobson-Korkie test with Memmel (2003) correction, per-period Sharpes, two-sided p."""
    a, b = np.asarray(ex_a), np.asarray(ex_b)
    T = len(a)
    sa, sb = a.mean() / a.std(ddof=1), b.mean() / b.std(ddof=1)
    rho = np.corrcoef(a, b)[0, 1]
    V = (2 - 2 * rho + 0.5 * (sa ** 2 + sb ** 2 - 2 * sa * sb * rho ** 2)) / T
    z = (sa - sb) / np.sqrt(V)
    return z, 2 * stats.norm.sf(abs(z))


def max_dd(r):
    c = (1 + r).cumprod()
    return float((c / c.cummax() - 1).min())


def metrics(sim, cost_bp, bh_sr=None):
    r = net(sim, cost_bp)
    rm_, rfm = monthly(r), monthly(sim["rf"])
    ex = rm_ - rfm
    yrs = len(r) / 252
    cagr = (1 + r).prod() ** (1 / yrs) - 1
    mdd = max_dd(r)
    sr = sharpe_m(ex)
    ci = boot_sharpe_ci(ex)
    turn = sim["turn"].sum() / yrs
    gm, gf = monthly(sim["gross"]), rfm

    def sr_at(c):
        x = monthly(sim["gross"] - sim["turn"] * c) - gf
        return x.mean() / x.std() * np.sqrt(12)

    def mean_at(c):
        return (monthly(sim["gross"] - sim["turn"] * c) - gf).mean()

    be0 = be_bh = np.nan
    if sim["turn"].sum() > 0:
        if mean_at(0) > 0:
            be0 = optimize.brentq(mean_at, 0, 0.5) * 1e4 if mean_at(0.5) < 0 else np.inf
        if bh_sr is not None and sr_at(0) > bh_sr:
            be_bh = optimize.brentq(lambda c: sr_at(c) - bh_sr, 0, 0.5) * 1e4 if sr_at(0.5) < bh_sr else np.inf
    return dict(SR=sr, SR_lo=ci[0], SR_hi=ci[1], t_NW=nw_t(ex), CAGR=cagr, vol=r.std() * np.sqrt(252), MDD=mdd,
                Calmar=cagr / abs(mdd), avg_w=sim["w"].mean(), turn_yr=turn,
                cost_drag=(sim["turn"] * cost_bp / 1e4).sum() / yrs, BE0_bp=be0, BE_BH_bp=be_bh), ex


# ----------------------------------------------------------------------------------------------- signals
def expanding_q(x, q):
    """Quantile of all values of x from its first non-NaN value through t (inclusive)."""
    return x.expanding(min_periods=252).quantile(q)


def binary_target(x, direction, q, dates):
    if direction == "high":
        thr = expanding_q(x, q)
        off = x > thr
    else:
        thr = expanding_q(x, 1 - q)
        off = x < thr
    w = (~off).astype(float).where(thr.notna())
    return w.reindex(dates)


def vt_forecasts(M, regs, first_x_month, min_pairs=24, ycol="logRVM"):
    """Expanding OLS of y_{m+1} on [1, regs_m]; at each month m uses pairs (k, k+1) with k+1 <= m.
    Returns (fitted log-vol forecast for m+1 made at end of m, resid var, coef table)."""
    Mx = M.loc[first_x_month:]
    X = sm.add_constant(Mx[regs]).values
    y = Mx[ycol].shift(-1).values                          # y for pair k is the outcome in month k+1
    out_f, out_s2, coefs = np.full(len(Mx), np.nan), np.full(len(Mx), np.nan), []
    for j in range(len(Mx)):
        # pairs k=0..j-1 have outcome month k+1 <= j: known at end of month j
        Xk, yk = X[:j], y[:j]
        ok = np.isfinite(Xk).all(1) & np.isfinite(yk)
        if ok.sum() < min_pairs or not np.isfinite(X[j]).all():
            continue
        b, *_ = np.linalg.lstsq(Xk[ok], yk[ok], rcond=None)
        res = yk[ok] - Xk[ok] @ b
        out_f[j] = X[j] @ b
        out_s2[j] = res.var(ddof=X.shape[1])
        coefs.append(b)
    return pd.Series(out_f, Mx.index), pd.Series(out_s2, Mx.index)


def vt_target(M, regs, sigma_star, cap, first_x_month):
    f, s2 = vt_forecasts(M, regs, first_x_month)
    sig = np.exp(f + s2 / 2)
    w = np.minimum(cap, sigma_star / sig)
    w.index = M.loc[first_x_month:, "ME_DATE"].values
    return w


def fmt_table(rows):
    df = pd.DataFrame(rows).set_index("rule")
    f = {"SR": "{:+.3f}", "SR_lo": "{:+.2f}", "SR_hi": "{:+.2f}", "t_NW": "{:+.2f}", "CAGR": "{:.2%}", "vol": "{:.1%}",
         "MDD": "{:.1%}", "Calmar": "{:.3f}", "avg_w": "{:.2f}", "turn_yr": "{:.2f}", "cost_drag": "{:.3%}",
         "BE0_bp": "{:.0f}", "BE_BH_bp": "{:.0f}", "dSR_vs_BH": "{:+.3f}", "p_boot": "{:.3f}", "p_JKM": "{:.3f}",
         "dSR_lo": "{:+.2f}", "dSR_hi": "{:+.2f}", "p_holm": "{:.3f}"}
    out = df.copy().astype(object)
    for c, fm in f.items():
        if c in df:
            out[c] = [fm.format(v) if pd.notna(v) else "nan" for v in df[c]]
    return out.to_string()
