"""Adversarial verification of Concept D (VRP timing). Independent re-implementation of the key numbers
plus extra checks. Uses the builder's cached raw data (copied to ./cache) but NOT the builder's panel/backtest code,
except where explicitly imported for comparison."""
import os, io, zipfile, numpy as np, pandas as pd
from scipy.stats import norm, skew
HERE = os.path.dirname(os.path.abspath(__file__)); C = os.path.join(HERE, "cache")
pd.set_option("display.width", 220)

# ------------------------------------------------------------------ independent data loading
def french(name, datelen):
    with zipfile.ZipFile(os.path.join(C, name)) as z:
        txt = z.read(z.namelist()[0]).decode("latin1")
    rows = []; started = False
    for l in txt.splitlines():
        f = [x.strip() for x in l.split(",")]
        if f[0].isdigit() and len(f[0]) == datelen:
            rows.append(f[:5]); started = True
        elif started:
            break
    df = pd.DataFrame(rows, columns=["d", "Mkt-RF", "SMB", "HML", "RF"])
    for c in df.columns[1:]:
        df[c] = pd.to_numeric(df[c])
    fmt = "%Y%m%d" if datelen == 8 else "%Y%m"
    df.index = pd.to_datetime(df["d"], format=fmt); df = df.drop(columns="d")
    n_bad = (df <= -99.99).sum().sum()
    df = df.mask(df <= -99.99) / 100
    return df, n_bad

fd, bad_d = french("F-F_Research_Data_Factors_daily_CSV.zip", 8)
fm, bad_m = french("F-F_Research_Data_Factors_CSV.zip", 6)
fm.index = fm.index + pd.offsets.MonthEnd(0)
print(f"French daily {fd.index[0].date()}..{fd.index[-1].date()} missing-codes={bad_d}; monthly {fm.index[0].date()}..{fm.index[-1].date()} missing-codes={bad_m}")

def yf(t):
    s = pd.read_csv(os.path.join(C, t.replace('^', '_') + ".csv"), index_col=0, parse_dates=True).iloc[:, 0].dropna()
    return s
vix, spy, put = yf("^VIX"), yf("SPY"), yf("^PUT")

# ------------------------------------------------------------------ 1. date alignment VIX vs French month-ends
last_fr = fd.index.to_series().groupby(fd.index.to_period("M")).max()
last_vx = vix.index.to_series().groupby(vix.index.to_period("M")).max()
al = pd.DataFrame({"fr": last_fr, "vx": last_vx}).dropna().loc["1990-01":"2026-07"]
mis = al[al.fr != al.vx]
print(f"\n[1] Month-end date mismatch VIX vs French daily: {len(mis)} of {len(al)} months")
print(mis.to_string() if len(mis) else "  none")
last_sp = spy.index.to_series().groupby(spy.index.to_period("M")).max()
al2 = pd.DataFrame({"fr": last_fr, "sp": last_sp}).dropna().loc["1993-02":"2026-07"]
print(f"    SPY vs French month-end mismatches: {(al2.fr != al2.sp).sum()} of {len(al2)}")

# ------------------------------------------------------------------ 2. independent monthly panel
rd = fd["Mkt-RF"] + fd["RF"]
RV = (rd ** 2).resample("ME").sum()
IV = ((vix / 100) ** 2 / 12).resample("ME").last()
spy_m = spy.resample("ME").last(); spy_r = spy_m.pct_change()
Pn = pd.DataFrame({"IV": IV, "RV": RV, "mkt_rf": fm["Mkt-RF"], "rf": fm["RF"], "spy_r": spy_r}).loc["1990-01":"2026-07"]
Pn["VRP"] = Pn.IV - Pn.RV
import sys; sys.path.insert(0, HERE)
import vrp_common as vc
P = vc.build_panel()
print(f"\n[2] Independent panel vs builder panel: max|dVRP|={np.nanmax(np.abs(Pn.VRP.values - P.VRP.values)):.2e} "
      f"max|dSPYret| (2000+)={np.nanmax(np.abs(Pn.spy_r.loc['2000':].values - P.spy_ret.loc['2000':].values)):.2e}")
d = pd.DataFrame({"spy": Pn.spy_r, "mkt": Pn.mkt_rf + Pn.rf}).loc["2000-01":"2026-07"]
print(f"    SPY vs French Mkt monthly 2000-01..2026-07: corr={d.corr().iloc[0,1]:.4f} mean diff (SPY-Mkt)={12*(d.spy-d.mkt).mean()*100:+.2f}%/yr "
      f"max |diff|={(d.spy-d.mkt).abs().max()*100:.2f}% on {(d.spy-d.mkt).abs().idxmax().date()}")
dr = spy.pct_change().dropna()
print(f"    SPY daily |ret|>12%: {dr[dr.abs()>0.12].round(3).to_dict()}")

# ------------------------------------------------------------------ 3. independent OOS forecasts (vectorised design)
def oos(x, y, h, o1, o2, min_obs=24):
    idx = x.index; n = len(idx)
    Y = pd.Series([y.iloc[i + 1:i + 1 + h].sum() if i + h < n else np.nan for i in range(n)], index=idx)
    res = []
    for t in idx[(idx >= o1) & (idx <= o2)]:
        p = idx.get_loc(t)
        xs, ys = x.iloc[:p - h + 1], Y.iloc[:p - h + 1]      # targets end at s+h <= p
        assert (ys.index[-1] if len(ys) else t) <= idx[p - h]
        ok = xs.notna() & ys.notna(); xs, ys = xs[ok], ys[ok]
        if len(ys) < min_obs: continue
        b = np.polyfit(xs.values, ys.values, 1)
        res.append((t, Y.iloc[p], b[1] + b[0] * x.iloc[p], ys.mean()))
    return pd.DataFrame(res, columns=["t", "y", "fm", "fb"]).set_index("t")

def nwse(u, L):
    u = np.asarray(u) - np.mean(u); T = len(u); s = u @ u / T
    for l in range(1, L + 1): s += 2 * (1 - l / (L + 1)) * (u[l:] @ u[:-l]) / T
    return np.sqrt(s / T)

def evaluate(F, h, restrict, lags):
    F = F.dropna(); f = F.fm.clip(lower=0) if restrict else F.fm
    em, eb = F.y - f, F.y - F.fb
    r2 = 1 - (em ** 2).sum() / (eb ** 2).sum()
    cw = eb ** 2 - (em ** 2 - (F.fb - f) ** 2); dm = eb ** 2 - em ** 2
    tcw = cw.mean() / nwse(cw, lags); tdm = dm.mean() / nwse(dm, lags)
    return r2, tcw, 1 - norm.cdf(tcw), tdm, 1 - norm.cdf(tdm), len(F)

print("\n[3] Independent OOS R2 (test origins 1999-12..; builder: h1 unres -10.01%, h1 restr +0.69%, h3 unres -14.08%, h3 restr +6.23%)")
FF = {}
for h, last in ((1, "2026-06-30"), (3, "2026-04-30")):
    FF[h] = oos(Pn.VRP, Pn.mkt_rf, h, pd.Timestamp("1999-12-31"), pd.Timestamp(last))
    for r in (False, True):
        for L in ((h,) if h == 1 else (3, 6, 12)):
            r2, tcw, pcw, tdm, pdm, n = evaluate(FF[h], h, r, L)
            print(f"  h={h} {'restr' if r else 'unres'} NWlags={L:2d} n={n} R2={r2*100:+.2f}% CW t={tcw:+.2f} p={pcw:.4f} | "
                  f"DM (unadjusted MSPE diff) t={tdm:+.2f} p={pdm:.4f}")

# Bonferroni over the 12 forecast specifications counted by the builder
r2, tcw, pcw, tdm, pdm, n = evaluate(FF[3], 3, True, 3)
print(f"  h=3 restricted: exact CW p={pcw:.5f}; Bonferroni x12 = {min(1, 12*pcw):.4f}; with NW lags 12: see above. "
      f"DM (no CW adjustment) p={pdm:.4f}, Bonferroni x12 = {min(1, 12*pdm):.4f}")
# sub-period DM for restricted h=3
for a, b in (("1999-12", "2012-11"), ("2012-12", "2026-04")):
    r2s, tcw_s, pcw_s, tdm_s, pdm_s, ns = evaluate(FF[3].loc[a:b], 3, True, 6)
    print(f"  h=3 restr {a}..{b}: R2={r2s*100:+.2f}% CW t={tcw_s:+.2f} DM t={tdm_s:+.2f} (NW 6)")

# ------------------------------------------------------------------ 4. BTZ in-sample replication sanity check (1990-2007)
def nw_ols(y, x, L):
    m = y.notna() & x.notna(); y, x = y[m].values, x[m].values
    X = np.column_stack([np.ones(len(x)), x]); b = np.linalg.lstsq(X, y, rcond=None)[0]; u = y - X @ b
    Xu = X * u[:, None]; T = len(y); S = Xu.T @ Xu / T
    for l in range(1, L + 1):
        G = Xu[l:].T @ Xu[:-l] / T; S += (1 - l / (L + 1)) * (G + G.T)
    Q = np.linalg.inv(X.T @ X / T); V = Q @ S @ Q / T
    return b[1], b[1] / np.sqrt(V[1, 1]), 1 - u.var() / y.var(), T
print("\n[4] In-sample (NOT OOS) sanity check vs Bollerslev-Tauchen-Zhou sample 1990-01..2007-12, VRP in monthly var units")
for h in (1, 3):
    y = sum(Pn.mkt_rf.shift(-k) for k in range(1, h + 1))
    for a, b in (("1990-01", "2007-12"), ("1990-01", "2026-06")):
        s, t, r2, T = nw_ols(y.loc[a:b], Pn.VRP.loc[a:b], L=max(h, 1) + 2)
        print(f"  h={h} {a}..{b}: slope={s:+.3f} NW t={t:+.2f} R2={r2*100:.2f}% T={T}")

# ------------------------------------------------------------------ 5. independent S1 / B1 backtest (vectorised drift)
F1 = FF[1]
var60 = fm["Mkt-RF"].rolling(60).var()
wS1 = (F1.fm / (3 * var60.reindex(F1.index))).clip(0, 1)
def bt(w, cost_bp, a="2000-01", b="2026-07"):
    months = Pn.loc[a:b].index
    wlag = w.reindex(Pn.index).shift(1).loc[a:b]                  # weight set at t applied to t+1
    assert wlag.notna().all()
    r, rf = Pn.spy_r.loc[a:b], Pn.rf.loc[a:b]
    gross = wlag * r + (1 - wlag) * rf
    drift = (wlag * (1 + r) / (1 + gross)).shift(1).fillna(0.0)     # weight before this month's rebalance
    tc = cost_bp / 1e4 * (wlag - drift).abs()
    net = gross - tc
    return net, net - rf, (wlag - drift).abs().sum() / (len(months) / 12)
for name, w in (("B1", pd.Series(1.0, index=Pn.index)), ("S1", wS1)):
    net, ex, to = bt(w, 2)
    sr = ex.mean() / ex.std(ddof=0) * np.sqrt(12)
    wealth = (1 + net).cumprod(); mdd = (wealth / wealth.cummax() - 1).min()
    print(f"\n[5] {name} independent: ann ex={ex.mean()*12*100:.2f}% Sharpe={sr:.3f} maxDD={mdd*100:.1f}% turnover/yr={to:.2f}")
bS1 = pd.read_csv(os.path.join(HERE, "..", "bt_S1.csv"), index_col=0, parse_dates=True)
net, ex, _ = bt(wS1, 2)
print(f"    max |independent S1 net - builder bt_S1.csv ret| = {np.abs(net.values - bS1['ret'].values).max():.2e}")

# ------------------------------------------------------------------ 6. ^PUT data quality
pr = put.pct_change().dropna()
print("\n[6] ^PUT daily |ret|>8%:"); print(pr[pr.abs() > 0.08].round(4).to_string())
print(f"    ^PUT prior peak before 2020-03-13: {put.loc[:'2020-03-12'].max():.2f}; 2020-03-13 print {put.loc['2020-03-13']:.2f}")
gaps = put.index.to_series().diff().dt.days
print(f"    ^PUT gaps > 5 calendar days: {gaps[gaps > 5].to_dict()}")
pm = put.resample("ME").last().pct_change().loc["1996-09":"2026-07"]
print(f"    ^PUT monthly |ret|>15%: {pm[pm.abs() > 0.15].round(3).to_dict()}")
# daily vol excluding the bad print pair
pr_clean = pr.drop([pd.Timestamp("2020-03-13"), pd.Timestamp("2020-03-16")])
print(f"    ^PUT ann vol all days {pr.std()*np.sqrt(252):.1%}; excluding 2020-03-13/16 {pr_clean.std()*np.sqrt(252):.1%}")

# ------------------------------------------------------------------ 7. short-variance proxy wording check
pay = ((Pn.IV - Pn.RV.shift(-1)) * 12 * 1e4).dropna()
worst = pay[pay <= pay.quantile(0.01)]
print(f"\n[7] Short-var proxy: n={len(pay)} mean={pay.mean():.1f} total(net)={pay.sum():.0f} worst-1% months n={len(worst)} sum={worst.sum():.0f}; "
      f"gross carry excl. worst 1% = {pay.drop(worst.index).sum():.0f}; share of that gross lost = {-worst.sum()/pay.drop(worst.index).sum():.1%}; "
      f"loss/net total = {-worst.sum()/pay.sum():.1%}")
print(f"    skew (scipy, biased) = {skew(pay):+.2f}; NW(6) t = {pay.mean()/nwse(pay, 6):+.2f}")
