"""Shared code for Concept D (VRP timing). See SPEC.md for the pre-registered design."""
import io, os, zipfile, urllib.request, json
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "cache"); os.makedirs(CACHE, exist_ok=True)
FRENCH = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"
SEED = 20260925


# ---------------------------------------------------------------- data
def _french_lines(name):
    path = os.path.join(CACHE, name)
    if not os.path.exists(path):
        with open(path, "wb") as f:
            f.write(urllib.request.urlopen(FRENCH + name).read())
    with zipfile.ZipFile(path) as z:
        return z.read(z.namelist()[0]).decode("latin1").splitlines()


def french_daily():
    """First table of the daily 3-factor file; decimals. -99.99/-999 -> NaN."""
    lines = _french_lines("F-F_Research_Data_Factors_daily_CSV.zip")
    header = next(i for i, l in enumerate(lines) if l.startswith(","))
    cols = [c.strip() for c in lines[header].split(",")[1:]]
    rows = []
    for l in lines[header + 1:]:
        f = l.split(",")
        if len(f[0].strip()) == 8 and f[0].strip().isdigit():
            rows.append(f)
        elif rows:
            break
    df = pd.DataFrame([r[1:] for r in rows], columns=cols,
                      index=pd.to_datetime([r[0].strip() for r in rows], format="%Y%m%d")).astype(float)
    return df.mask(df <= -99.99) / 100


def french_monthly():
    """First (monthly) table of the monthly 3-factor file; index = month-end; decimals."""
    lines = _french_lines("F-F_Research_Data_Factors_CSV.zip")
    header = next(i for i, l in enumerate(lines) if l.startswith(","))
    cols = [c.strip() for c in lines[header].split(",")[1:]]
    rows = []
    for l in lines[header + 1:]:
        f = l.split(",")
        if len(f[0].strip()) == 6 and f[0].strip().isdigit():
            rows.append(f)
        elif rows:
            break  # annual table starts after a blank/title line
    idx = pd.to_datetime([r[0].strip() for r in rows], format="%Y%m") + pd.offsets.MonthEnd(0)
    df = pd.DataFrame([r[1:] for r in rows], columns=cols, index=idx).astype(float)
    return df.mask(df <= -99.99) / 100


def yf_close(ticker, start="1985-01-01"):
    safe = ticker.replace("^", "_")
    path = os.path.join(CACHE, f"{safe}.csv")
    if not os.path.exists(path):
        import yfinance as yf
        d = yf.download(ticker, start=start, end="2026-09-25", auto_adjust=True, progress=False)
        c = d["Close"]
        if isinstance(c, pd.DataFrame):
            c = c.iloc[:, 0]
        c.rename(ticker).to_csv(path)
    s = pd.read_csv(path, index_col=0, parse_dates=True).iloc[:, 0]
    return s.dropna()


# ---------------------------------------------------------------- monthly panel
def build_panel(end="2026-07-31"):
    """Monthly panel indexed by month-end. Everything in row t is known at the close of month t,
    except the *target* columns (next-month returns), which are explicitly shifted and named *_next."""
    fd = french_daily()
    fm = french_monthly()
    rd = fd["Mkt-RF"] + fd["RF"]                       # daily market total return
    RV = (rd ** 2).groupby(rd.index.to_period("M")).sum()
    RV.index = RV.index.to_timestamp("M")
    vix = yf_close("^VIX")
    vix_m = vix.groupby(vix.index.to_period("M")).last(); vix_m.index = vix_m.index.to_timestamp("M")
    IV = (vix_m / 100) ** 2 / 12
    spy = yf_close("SPY")
    spy_m = spy.groupby(spy.index.to_period("M")).last(); spy_m.index = spy_m.index.to_timestamp("M")
    # robustness RV: trailing 21 trading days of SPY daily log returns, sampled at month end
    lr = np.log(spy).diff()
    rv21 = (lr ** 2).rolling(21).sum()
    rv21_m = rv21.groupby(rv21.index.to_period("M")).last(); rv21_m.index = rv21_m.index.to_timestamp("M")

    P = pd.DataFrame({"IV": IV, "RV": RV, "RV21": rv21_m, "VIX": vix_m,
                      "mkt_rf": fm["Mkt-RF"], "rf": fm["RF"], "spy_px": spy_m})
    P["mkt"] = P["mkt_rf"] + P["rf"]
    P["spy_ret"] = P["spy_px"].pct_change()
    # the first SPY month (1993-01) is a partial month -> drop its return
    P.loc[P.index <= pd.Timestamp("1993-01-31"), "spy_ret"] = np.nan
    P["VRP"] = P["IV"] - P["RV"]
    P["VRP21"] = P["IV"] - P["RV21"]
    P = P.loc["1990-01-31":end]
    return P


# ---------------------------------------------------------------- OOS forecasting
def oos_forecasts(x, y_m, h, first_origin, last_origin, min_obs=24):
    """x: predictor at origin t (known at t). y_m: monthly excess return indexed by its own month.
    Target for origin t = sum of y over months t+1..t+h. At origin t fit only on origins s with s+h <= t.
    Returns DataFrame indexed by origin t with columns: y (realised target), f_model, f_mean."""
    idx = x.index
    ypos = {d: i for i, d in enumerate(y_m.index)}
    target = pd.Series(np.nan, index=idx)
    for t in idx:
        i = ypos.get(t)
        if i is not None and i + h < len(y_m):
            target[t] = y_m.iloc[i + 1:i + 1 + h].sum()
    out = []
    origins = idx[(idx >= pd.Timestamp(first_origin)) & (idx <= pd.Timestamp(last_origin))]
    for t in origins:
        # estimation sample: origins s whose target window ends <= t
        s_all = idx[idx < t]
        pos_t = idx.get_loc(t)
        s_ok = idx[: max(pos_t - h + 1, 0)]          # s with pos(s)+h <= pos(t)
        X = x.loc[s_ok]; Y = target.loc[s_ok]
        m = X.notna() & Y.notna()
        X, Y = X[m], Y[m]
        if len(Y) < min_obs or pd.isna(x.loc[t]):
            continue
        # timing check: every estimation target ends no later than t
        assert all(idx.get_loc(s) + h <= pos_t for s in X.index)
        A = np.column_stack([np.ones(len(X)), X.values])
        beta = np.linalg.lstsq(A, Y.values, rcond=None)[0]
        out.append((t, target[t], beta[0] + beta[1] * x.loc[t], Y.mean(), beta[1], len(Y)))
    return pd.DataFrame(out, columns=["origin", "y", "f_model", "f_mean", "slope", "n_est"]).set_index("origin")


def nw_se_mean(f, lags):
    f = np.asarray(f, float); f = f - f.mean(); T = len(f)
    g0 = f @ f / T
    s = g0 + 2 * sum((1 - l / (lags + 1)) * (f[l:] @ f[:-l] / T) for l in range(1, lags + 1))
    return np.sqrt(s / T)


def oos_stats(F, restrict=False):
    F = F.dropna(subset=["y"])
    fm = F["f_model"].clip(lower=0) if restrict else F["f_model"]
    e_mod = F["y"] - fm; e_bm = F["y"] - F["f_mean"]
    r2 = 1 - (e_mod ** 2).sum() / (e_bm ** 2).sum()
    return r2, e_mod, e_bm, fm


def clark_west(F, h, restrict=False):
    F = F.dropna(subset=["y"])
    r2, e_mod, e_bm, fm = oos_stats(F, restrict)
    fcw = e_bm ** 2 - (e_mod ** 2 - (F["f_mean"] - fm) ** 2)
    se = nw_se_mean(fcw.values, lags=h)
    t = fcw.mean() / se
    from scipy.stats import norm
    return r2, t, 1 - norm.cdf(t), len(F)


def nw_ols(y, x, lags):
    """OLS y = a + b x with NW t-stat for b (in-sample, reported as such)."""
    m = y.notna() & x.notna(); y, x = y[m].values, x[m].values
    X = np.column_stack([np.ones(len(x)), x]); T = len(y)
    b = np.linalg.lstsq(X, y, rcond=None)[0]; u = y - X @ b
    Xu = X * u[:, None]
    S = Xu.T @ Xu / T
    for l in range(1, lags + 1):
        G = Xu[l:].T @ Xu[:-l] / T
        S += (1 - l / (lags + 1)) * (G + G.T)
    Q = np.linalg.inv(X.T @ X / T)
    V = Q @ S @ Q / T
    r2 = 1 - u.var() / y.var()
    return b[1], b[1] / np.sqrt(V[1, 1]), r2, T


# ---------------------------------------------------------------- backtest
def backtest(w, risky, rf, cost_bp, start, end):
    """w: weights decided at close of month t (index = t). Applied to returns of month t+1.
    risky, rf: monthly returns indexed by own month. Returns DataFrame by return month."""
    months = risky.loc[start:end].index
    c = cost_bp / 1e4
    rows = []; w_drift = 0.0  # start from cash, so initial purchase is charged
    for m in months:
        prev = risky.index[risky.index.get_loc(m) - 1]     # decision month
        assert prev < m                                      # timing check: weight set strictly before return month
        wt = w.get(prev, np.nan)
        if pd.isna(wt):
            raise ValueError(f"missing weight at {prev}")
        tc = c * abs(wt - w_drift)
        r = wt * risky[m] + (1 - wt) * rf[m] - tc
        gross_risky = wt * (1 + risky[m]); gross = gross_risky + (1 - wt) * (1 + rf[m])
        w_drift = gross_risky / gross if gross > 0 else 0.0
        rows.append((m, r, r - rf[m], tc, abs(wt - (rows[-1][5] if rows else 0.0)), wt, w_drift))
    df = pd.DataFrame(rows, columns=["month", "ret", "ex", "tc", "dw_unused", "w", "w_drift"]).set_index("month")
    return df.drop(columns="dw_unused")


def turnover_per_year(bt, c_bp):
    return bt["tc"].sum() / (c_bp / 1e4) / (len(bt) / 12) if c_bp > 0 else np.nan


def sharpe_lo(ex, lags=6):
    """Annualised Sharpe of monthly excess returns and its SE via GMM/delta method with NW (Lo 2002 style)."""
    x = np.asarray(ex, float); T = len(x)
    mu, var = x.mean(), x.var()
    sr = mu / np.sqrt(var)
    g = np.column_stack([x - mu, (x - mu) ** 2 - var])
    S = g.T @ g / T
    for l in range(1, lags + 1):
        G = g[l:].T @ g[:-l] / T
        S += (1 - l / (lags + 1)) * (G + G.T)
    grad = np.array([1 / np.sqrt(var), -mu / (2 * var ** 1.5)])
    se = np.sqrt(grad @ S @ grad / T)
    return sr * np.sqrt(12), se * np.sqrt(12)


def maxdd(ret):
    w = (1 + pd.Series(ret)).cumprod()
    return (w / w.cummax() - 1).min()


def summarize(name, bt, cost_bp):
    ex = bt["ex"]
    sr, se = sharpe_lo(ex)
    mt = ex.mean() / nw_se_mean(ex.values, 6)
    return dict(name=name, ann_ex=ex.mean() * 12, ann_vol=ex.std() * np.sqrt(12), sharpe=sr,
                sr_lo=sr - 1.96 * se, sr_hi=sr + 1.96 * se, t_mean=mt, maxdd=maxdd(bt["ret"]),
                avg_w=bt["w"].mean(), turnover=turnover_per_year(bt, cost_bp) if cost_bp > 0 else np.nan,
                cost_drag=bt["tc"].mean() * 12, n=len(bt))


def stationary_bootstrap_idx(T, mean_block, rng):
    idx = np.empty(T, int); p = 1 / mean_block
    idx[0] = rng.integers(T)
    for i in range(1, T):
        idx[i] = rng.integers(T) if rng.random() < p else (idx[i - 1] + 1) % T
    return idx


def paired_bootstrap(ex_a, ex_b, reps=10000, mean_block=6, seed=SEED):
    """Difference A-B in annualised Sharpe and in annualised mean excess return."""
    a = np.asarray(ex_a); b = np.asarray(ex_b); T = len(a)
    rng = np.random.default_rng(seed)
    def srf(x): return x.mean() / x.std() * np.sqrt(12)
    d_sr = srf(a) - srf(b); d_mu = (a.mean() - b.mean()) * 12
    bs_sr = np.empty(reps); bs_mu = np.empty(reps)
    for k in range(reps):
        ii = stationary_bootstrap_idx(T, mean_block, rng)
        bs_sr[k] = srf(a[ii]) - srf(b[ii]); bs_mu[k] = (a[ii].mean() - b[ii].mean()) * 12
    return dict(d_sr=d_sr, sr_ci=(np.percentile(bs_sr, 2.5), np.percentile(bs_sr, 97.5)),
                p_sr=(bs_sr <= 0).mean(), d_mu=d_mu,
                mu_ci=(np.percentile(bs_mu, 2.5), np.percentile(bs_mu, 97.5)), p_mu=(bs_mu <= 0).mean())


# ---------------------------------------------------------------- weights
def expanding_quantile_rule(sig, q, min_obs=24):
    w = {}
    for i, t in enumerate(sig.index):
        hist = sig.iloc[: i + 1].dropna()          # data <= t only
        if len(hist) < min_obs or pd.isna(sig.iloc[i]):
            continue
        w[t] = 0.0 if sig.iloc[i] < hist.quantile(q) else 1.0
    return pd.Series(w)


def mv_weights(mu, mkt_rf, gamma=3.0, window=60):
    var = mkt_rf.rolling(window).var()             # data <= t
    w = (mu / (gamma * var.reindex(mu.index))).clip(0, 1)
    return w.dropna()


def sma_rule(px, n=10):
    sma = px.rolling(n).mean()
    return (px > sma).astype(float).where(sma.notna()).dropna()


def voltarget(RV, target=0.15):
    return (target / np.sqrt(12 * RV)).clip(upper=1).dropna()


def breakeven_cost(w, risky, rf, start, end, ref_ex_mean=None):
    """One-way cost (bp) where mean excess return = 0 (ref None) or = ref_ex_mean. Linear in c."""
    b0 = backtest(w, risky, rf, 0.0, start, end); b1 = backtest(w, risky, rf, 1.0, start, end)
    per_bp = b0["ex"].mean() - b1["ex"].mean()
    tgt = 0.0 if ref_ex_mean is None else ref_ex_mean
    return (b0["ex"].mean() - tgt) / per_bp if per_bp > 0 else np.inf
