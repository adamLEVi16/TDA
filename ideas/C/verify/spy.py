"""Tradable check on SPY (SPEC.md): frozen rules, 1994-01-03..2024-12-31, cash at French daily RF, 2 bp costs.
Implementation choices not spelled out in SPEC (stated here and in the report):
 - B-RV threshold: expanding quantile of a spliced RV series (French-market RV21 through 1993-12-31, SPY RV21 after).
 - B-VIX threshold: expanding quantile of VIX from 1990-01-02 (min 252 obs).
 - VT regressions: expanding monthly OLS on the French market (y = log RVM_{m+1}); coefficients as of the last
   month-end at or before the rebalance date, applied to SPY-based log RV21 (and French-industry X / log VIX)."""
import json

import numpy as np
import pandas as pd
import statsmodels.api as sm

from common_setup import F, FIRST_X, M, SIGMA_STAR
from data import market_daily, yahoo
from engine import INDS, NET_INDS, binary_target, boot_diff, fmt_table, jkm, metrics, simulate

FROZEN = json.load(open("frozen_params.json"))
md = market_daily()
y = yahoo(["SPY", "^VIX", "^VIX3M"])
spy_px = y["SPY"].dropna()
r_spy = spy_px.pct_change().dropna()
days = r_spy.index
rf = md["rf"].reindex(days).fillna(0.0)
vix, vix3m = y["^VIX"].dropna(), y["^VIX3M"].dropna()
rv_spy = np.sqrt(252 * (r_spy ** 2).rolling(21).mean())
rv_splice = pd.concat([F["RV"].loc[:"1993-12-31"], rv_spy.loc["1994-01-01":]])
ME_S = pd.DatetimeIndex(pd.Series(days, index=days).groupby(days.to_period("M")).last().values)
WK_S = pd.DatetimeIndex(pd.Series(days, index=days).groupby(days.to_period("W")).last().values)
PER, PER_TS = ("1994-01-01", "2024-12-31"), ("2006-08-01", "2024-12-31")

# monthly frame for VT regressions (French market outcome), with log VIX
Mv = M.copy()
Mv["logRV"] = np.log(Mv["RV"])
Mv["logVIX"] = np.log(vix.reindex(pd.DatetimeIndex(Mv["ME_DATE"].values)).values / 100)


def betas(regs, first):
    Mx = Mv.loc[first:]
    X = sm.add_constant(Mx[regs], has_constant="add").values
    yy = Mx["logRVM"].shift(-1).values
    B, S2 = {}, {}
    for j in range(len(Mx)):
        ok = np.isfinite(X[:j]).all(1) & np.isfinite(yy[:j])
        if ok.sum() < 24:
            continue
        b, *_ = np.linalg.lstsq(X[:j][ok], yy[:j][ok], rcond=None)
        B[Mx["ME_DATE"].iloc[j]] = b
        S2[Mx["ME_DATE"].iloc[j]] = (yy[:j][ok] - X[:j][ok] @ b).var(ddof=X.shape[1])
    return pd.DataFrame(B).T, pd.Series(S2)


def vt_spy(regs, dates, cap=1.0, first=FIRST_X):
    B, S2 = betas(regs, first)
    cur = {"logRV": np.log(rv_spy), "logVIX": np.log(vix / 100)}
    cur.update({x: F[x] for x in NET_INDS})
    Xd = np.column_stack([np.ones(len(dates))] + [cur[r].reindex(dates).values for r in regs])
    Bd = B.reindex(B.index.union(dates)).ffill().reindex(dates).values
    s2 = S2.reindex(S2.index.union(dates)).ffill().reindex(dates).values
    f = (Xd * Bd).sum(1)
    return pd.Series(np.minimum(cap, SIGMA_STAR / np.exp(f + s2 / 2)), index=dates)


def targets(freq):
    dates = ME_S if freq == "M" else WK_S
    T = {}
    if freq == "M":
        me_px = spy_px.reindex(ME_S)
        T["TREND10"] = (me_px > me_px.rolling(10).mean()).astype(float).where(me_px.rolling(10).mean().notna())
    else:
        sma = spy_px.rolling(210).mean()
        T["TREND10"] = (spy_px > sma).astype(float).where(sma.notna()).reindex(dates)
    T["VT-naive"] = np.minimum(1.0, SIGMA_STAR / rv_spy.reindex(dates))
    T["VT-RV"] = vt_spy(["logRV"], dates)
    T["VT-RV x TREND10"] = T["VT-RV"] * T["TREND10"]
    fr = FROZEN["RV"]
    T["B-RV"] = binary_target(rv_splice, fr["direction"], fr["q"], dates)
    for x in NET_INDS:
        T[f"B-{x}"] = binary_target(F[x], FROZEN[x]["direction"], FROZEN[x]["q"], dates)
        T[f"VT-RV+{x}"] = vt_spy(["logRV", x], dates)
    T["B-VIX"] = binary_target(vix, fr["direction"], fr["q"], dates)
    T["VT-VIX"] = vt_spy(["logVIX"], dates, first=pd.Period("1990-01", "M"))
    T["VT-RV+VIX"] = vt_spy(["logRV", "logVIX"], dates, first=pd.Period("1990-01", "M"))
    ts = (vix / vix3m).dropna()
    T["B-VIXTS"] = (ts <= 1.0).astype(float).reindex(dates)
    return T, dates


def run(freq, period, names, label):
    T, dates = targets(freq)
    bh = simulate(pd.Series(1.0, index=dates), r_spy, rf, *period)
    mb, exb = metrics(bh, 0)
    rows, ex = [dict(rule="BH (SPY)", **mb)], {"BH": exb}
    for n in names:
        sim = simulate(T[n], r_spy, rf, *period)
        m, x = metrics(sim, 2.0, mb["SR"])
        d, ci, p = boot_diff(x, exb)
        rows.append(dict(rule=n, **m, dSR_vs_BH=d, dSR_lo=ci[0], dSR_hi=ci[1], p_boot=p, p_JKM=jkm(x, exb)[1]))
        ex[n] = x
    rows = sorted(rows, key=lambda r: -r["SR"])
    print(f"\n== SPY {label}: {period[0]}..{period[1]}, 2 bp, rebalance {freq}")
    print(fmt_table(rows))
    return ex


if __name__ == "__main__":
    pd.set_option("display.width", 300)
    print(f"SPY daily returns {days[0].date()}..{days[-1].date()}; RF missing days filled 0: {md['rf'].reindex(days).isna().sum()}")
    main = (["TREND10", "VT-naive", "VT-RV", "VT-RV x TREND10", "B-RV"] + [f"B-{x}" for x in NET_INDS]
            + [f"VT-RV+{x}" for x in NET_INDS] + ["B-VIX", "VT-VIX", "VT-RV+VIX"])
    ex = run("M", PER, main, "main")
    print("\n-- incremental (paired block bootstrap)")
    for a, b in [("VT-RV+VIX", "VT-RV"), ("VT-VIX", "VT-RV"), ("B-VIX", "B-RV"), ("B-VIX", "TREND10")] + \
                [(f"VT-RV+{x}", "VT-RV") for x in NET_INDS]:
        d, ci, p = boot_diff(ex[a], ex[b])
        print(f"{a:<14} vs {b:<9} dSR {d:+.3f} [{ci[0]:+.2f}, {ci[1]:+.2f}] p_boot {p:.3f}  JKM p {jkm(ex[a], ex[b])[1]:.3f}")
    ex2 = run("M", PER_TS, ["TREND10", "VT-RV", "B-RV", "B-VIX", "VT-VIX", "B-VIXTS", "B-H1N", "B-TOPOVOL"],
              "VIX term-structure window")
    d, ci, p = boot_diff(ex2["B-VIXTS"], ex2["TREND10"])
    print(f"B-VIXTS vs TREND10: dSR {d:+.3f} [{ci[0]:+.2f}, {ci[1]:+.2f}] p_boot {p:.3f}")
    run("W", PER, main, "weekly robustness")
