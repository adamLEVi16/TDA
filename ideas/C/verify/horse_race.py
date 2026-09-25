"""Predictive-regression horse race (SPEC.md 'Predictive regressions'). Monthly, non-overlapping.
Design: in-sample 1929-01..1989-12 outcomes. Test: expanding-window OOS forecasts of 1990-01..2024-12 outcomes."""
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

from common_setup import FIRST_X, M, mm
from engine import NET_INDS, NWLAG

D = M.loc[FIRST_X:].copy()
mm_p = mm.copy()
mm_p.index = mm_p.index.to_period("M")
D["ret_next"] = mm_p["mkt_rf"].shift(-1).reindex(D.index)          # excess return of month m+1
D["logRVM_next"] = D["logRVM"].shift(-1)
# outcome month = m+1
out_month = D.index + 1
DES = (out_month >= pd.Period("1929-01", "M")) & (out_month <= pd.Period("1989-12", "M"))
TST = (out_month >= pd.Period("1990-01", "M")) & (out_month <= pd.Period("2024-12", "M"))


def ols_nw(y, X):
    ok = np.isfinite(X).all(1) & np.isfinite(y)
    return sm.OLS(y[ok], X[ok]).fit(cov_type="HAC", cov_kwds={"maxlags": NWLAG})


def expanding_fc(y, X, min_obs=24):
    """Forecast for row j made with pairs 0..j-1 (all outcomes known at end of month j)."""
    f = np.full(len(y), np.nan)
    b_all = []
    for j in range(len(y)):
        Xk, yk = X[:j], y[:j]
        ok = np.isfinite(Xk).all(1) & np.isfinite(yk)
        if ok.sum() < min_obs or not np.isfinite(X[j]).all():
            b_all.append(None)
            continue
        b, *_ = np.linalg.lstsq(Xk[ok], yk[ok], rcond=None)
        f[j] = X[j] @ b
        b_all.append(b)
    return f, b_all


def clark_west(y, f_base, f_aug):
    adj = (y - f_base) ** 2 - ((y - f_aug) ** 2 - (f_base - f_aug) ** 2)
    t = sm.OLS(adj, np.ones(len(adj))).fit(cov_type="HAC", cov_kwds={"maxlags": NWLAG}).tvalues[0]
    return t, stats.norm.sf(t)


def X_of(cols):
    return sm.add_constant(D[cols], has_constant="add").values


# --------------------------------------------------------------------------------------------- volatility
y = D["logRVM_next"].values
print("== VOLATILITY: y = log RV(month m+1). Base = [1, log RV21_m]")
print(f"design outcomes {DES.sum()} months, test outcomes {TST.sum()} months")
base_cols = ["logRV"]
fb, _ = expanding_fc(y, X_of(base_cols))
mb = ols_nw(y[DES], X_of(base_cols)[DES])
print(f"base design R2 {mb.rsquared:.3f}; slope {mb.params[1]:.3f} (NW t {mb.tvalues[1]:+.1f})")
yt = y[TST]
sse_b = np.nansum((yt - fb[TST]) ** 2)
# OOS R2 of base vs expanding mean of log RV (context)
fm = pd.Series(y).expanding(24).mean().shift(1).values
print(f"base OOS R2 vs expanding mean of log RV, test: {1 - sse_b / np.nansum((yt - fm[TST]) ** 2):.3f}")
print(f"{'added X':<10}{'design coef':>12}{'NW t':>8}{'dR2 design':>12}{'OOS R2 vs base':>16}{'CW t':>8}{'CW p':>8}")
for x in NET_INDS:
    cols = base_cols + [x]
    m = ols_nw(y[DES], X_of(cols)[DES])
    fa, _ = expanding_fc(y, X_of(cols))
    r2 = 1 - np.nansum((yt - fa[TST]) ** 2) / sse_b
    t, p = clark_west(yt, fb[TST], fa[TST])
    print(f"{x:<10}{m.params[2]:>12.4f}{m.tvalues[2]:>+8.2f}{m.rsquared - mb.rsquared:>12.4f}{r2:>16.4f}{t:>+8.2f}{p:>8.3f}")

print("\n-- Does topology add beyond log RV AND average correlation? base2 = [1, log RV, AC60]")
b2 = ["logRV", "AC60"]
topo = ["H1N", "H1P", "TOPOVOL", "FIED"]
m2 = ols_nw(y[DES], X_of(b2)[DES])
m3 = ols_nw(y[DES], X_of(b2 + topo)[DES])
R = np.zeros((4, 7))
R[:, 3:] = np.eye(4)
w = m3.wald_test(R, scalar=True)
print(f"design: R2 {m2.rsquared:.4f} -> {m3.rsquared:.4f}; Wald (NW) on 4 topology terms: stat {float(w.statistic):.2f}, p {float(w.pvalue):.3f}")
print("design topology t-stats: " + ", ".join(f"{c} {t:+.2f}" for c, t in zip(topo, m3.tvalues[3:])))
f2, _ = expanding_fc(y, X_of(b2))
f3, _ = expanding_fc(y, X_of(b2 + topo))
r2 = 1 - np.nansum((yt - f3[TST]) ** 2) / np.nansum((yt - f2[TST]) ** 2)
t, p = clark_west(yt, f2[TST], f3[TST])
print(f"test OOS R2 (base2+topology vs base2) {r2:.4f}; CW t {t:+.2f} p {p:.3f}")
r2a = 1 - np.nansum((yt - f2[TST]) ** 2) / sse_b
t, p = clark_west(yt, fb[TST], f2[TST])
print(f"(AC60 alone vs base, same as above row) OOS R2 {r2a:.4f}; CW t {t:+.2f} p {p:.3f}")

# --------------------------------------------------------------------------------------------- returns
print("\n== RETURNS: y = French Mkt-RF in month m+1. Campbell-Thompson OOS R2 vs expanding historical mean")
yr = D["ret_next"].values
hm = pd.Series(yr).expanding(24).mean().shift(1).values   # mean of outcomes known at end of month m
yrt = yr[TST]
sse_hm = np.nansum((yrt - hm[TST]) ** 2)
specs = {"ACM (Pollet-Wilson)": ["ACM"], "AC60": ["AC60"], "RVM^2": ["RVM2"], "AR60": ["AR60"], "H1N": ["H1N"],
         "H1P": ["H1P"], "TOPOVOL": ["TOPOVOL"], "FIED": ["FIED"], "PW bivariate ACM + RVM^2": ["ACM", "RVM2"]}
print(f"{'predictor':<26}{'design coef':>12}{'NW t':>8}{'R2 des':>8}{'OOS R2':>9}{'CW t':>7}{'CW p':>7}{'OOS R2 CT-restr':>17}{'CW p':>7}")
for name, cols in specs.items():
    m = ols_nw(yr[DES], X_of(cols)[DES])
    f, bs = expanding_fc(yr, X_of(cols))
    r2 = 1 - np.nansum((yrt - f[TST]) ** 2) / sse_hm
    t, p = clark_west(yrt, hm[TST], f[TST])
    fr = f.copy()
    if cols[0] == "ACM":                               # PW sign restriction on the ACM slope
        for j, b in enumerate(bs):
            if b is not None and b[1] < 0:
                fr[j] = hm[j]
    fr = np.maximum(fr, 0)
    r2r = 1 - np.nansum((yrt - fr[TST]) ** 2) / sse_hm
    tr, pr = clark_west(yrt, hm[TST], fr[TST])
    print(f"{name:<26}{m.params[1]:>12.4f}{m.tvalues[1]:>+8.2f}{m.rsquared:>8.4f}{r2:>9.4f}{t:>+7.2f}{p:>7.3f}{r2r:>17.4f}{pr:>7.3f}")

print("\n-- Extra diagnostic (not in SPEC, counted as extra variants): in-sample Pollet-Wilson regression")
for lab, a, b in [("1963-07..2006-12 (PW's sample window)", "1963-07", "2006-12"), ("1927..2024 full", "1927-01", "2024-12")]:
    sel = (out_month >= pd.Period(a, "M")) & (out_month <= pd.Period(b, "M"))
    for cols in (["ACM"], ["ACM", "RVM2"]):
        m = ols_nw(yr[sel], X_of(cols)[sel])
        print(f"{lab:<40} {'+'.join(cols):<10} " + "  ".join(f"{c} coef {m.params[i+1]:+.4f} (NW t {m.tvalues[i+1]:+.2f})"
                                                              for i, c in enumerate(cols)) + f"  n={int(m.nobs)}")
