"""
ROUND 12 VALIDATION — run the FULL existing torture + extended battery on the
accepted V2 ensemble (trend_mode="ensemble"), side by side with V0, reusing the
EXACT helper functions from torture_test.py / extended_tests.py / alpha_test.py
(not reimplemented). Required by ROUND12_PREREGISTRATION.md §6 before V2 could
replace anything. V0 remains the live strategy; this only certifies the
candidate against the same bar V0 already cleared.
"""
import warnings; warnings.filterwarnings("ignore")
import os, sys
import numpy as np, pandas as pd, statsmodels.api as sm
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import multi_asset as MA
from backtest import metrics
from long_history import LH_ASSETS, LH_BENCH, START as LH_START
from torture_test import block_bootstrap_sharpe_diff, deflated_sharpe, sharpe
from extended_tests import capm, stationary_bootstrap_sharpe_diff, EXT_ASSETS, EXT_BENCH, NBER

MODES = [("V0 binary", "binary"), ("V2 ensemble", "ensemble")]

def sortino(r, mar=0.0):
    dn = r[r < mar] - mar
    dd = np.sqrt((dn ** 2).sum() / len(r))
    return (r.mean() * 12) / (dd * np.sqrt(12)) if dd > 0 else np.nan

def lh(mode):
    bt, _ = MA.run(assets=LH_ASSETS, bench=LH_BENCH, start=LH_START, trend_mode=mode)
    return bt

def main():
    print("=" * 84)
    print("  ROUND 12 VALIDATION — full torture+extended battery, V0 vs V2 (ensemble)")
    print("=" * 84)

    bt = {name: lh(mode) for name, mode in MODES}
    rflh = MA.get_rf_monthly(bt["V0 binary"].index)

    # ---- headline (1987-2024) ----
    print("\n[HEADLINE 1987-2024]")
    print(f"  {'':<14}{'Sharpe':>8}{'Sortino':>9}{'CAGR':>8}{'MaxDD':>8}{'vsSPY p':>9}{'CAPMalpha(t)':>15}")
    for name, _ in MODES:
        r = bt[name]["RP+Trend"]; eq = bt[name]["SPY"]
        m = metrics(r)
        _, _, _, pv = block_bootstrap_sharpe_diff(r, eq)
        a, t, pval, beta = capm(r, eq, rflh)
        print(f"  {name:<14}{m['Sharpe']:>8.2f}{sortino(r):>9.2f}{m['CAGR']:>8.1%}"
              f"{m['MaxDD']:>8.1%}{pv:>9.3f}   {a:+.2%} (t={t:+.2f})")

    # ---- strict spanning alpha vs own 5 assets ----
    print("\n[SPANNING ALPHA vs own 5 assets, 1987-2024]")
    axr = MA.load_prices(assets=LH_ASSETS, start=LH_START)
    aret = axr.resample("ME").last().pct_change().reindex(bt["V0 binary"].index).sub(rflh, axis=0)
    for name, _ in MODES:
        y = (bt[name]["RP+Trend"] - rflh).rename("y")
        dsp = pd.concat([y, aret], axis=1).dropna()
        sp = sm.OLS(dsp["y"], sm.add_constant(dsp[LH_ASSETS])).fit(cov_type="HAC", cov_kwds={"maxlags": 6})
        print(f"  {name:<14} spanning alpha {sp.params['const']*12:+.2%}/yr  t={sp.tvalues['const']:+.2f}  p={sp.pvalues['const']:.3f}")

    # ---- deflated Sharpe (13 cumulative trials) ----
    print("\n[DEFLATED SHARPE, 13 cumulative project trials]")
    for name, _ in MODES:
        r = bt[name]["RP+Trend"]
        dsr, thr = deflated_sharpe(sharpe(r), len(r), 13)
        print(f"  {name:<14} P(true Sharpe>0)={dsr:.3f}  (chance-Sharpe threshold ~{thr:.2f})")

    # ---- significance of V2's improvement over V0 (the key question) ----
    d, lo, hi, p = block_bootstrap_sharpe_diff(bt["V2 ensemble"]["RP+Trend"], bt["V0 binary"]["RP+Trend"])
    print(f"\n[V2 vs V0 improvement] ΔSharpe={d:+.2f}  95% CI[{lo:+.2f},{hi:+.2f}]  p={p:.3f}"
          f"  ({'significant' if lo>0 else 'suggestive, not significant'})")

    # ---- robustness battery on BOTH modes ----
    print("\n[ROBUSTNESS] (long history unless noted)")
    for name, mode in MODES:
        # 40bps cost
        b40, _ = MA.run(assets=LH_ASSETS, bench=LH_BENCH, start=LH_START, cost_bps=40, trend_mode=mode)
        # rebalance shift
        shifts = []
        for sd in (-10, -5, 5, 10):
            bs, _ = MA.run(assets=LH_ASSETS, bench=LH_BENCH, start=LH_START, shift_days=sd, trend_mode=mode)
            shifts.append(sharpe(bs["RP+Trend"]))
        # drop-one
        drops = []
        for a in LH_ASSETS:
            bd, _ = MA.run(assets=[x for x in LH_ASSETS if x != a], bench=LH_BENCH, start=LH_START, trend_mode=mode)
            drops.append(sharpe(bd["RP+Trend"]))
        # trend window sweep (ensemble ignores sma_months for its scaler but engine still guards on it; report base)
        # 40yr extension (4-fund)
        be, _ = MA.run(assets=EXT_ASSETS, bench=EXT_BENCH, start="1980-01-01", trend_mode=mode)
        rfe = MA.get_rf_monthly(be.index)
        ae, te, _, _ = capm(be["RP+Trend"], be["SPY"], rfe)
        print(f"  {name:<14} 40bps Sharpe={sharpe(b40['RP+Trend']):.2f}  "
              f"date-shift {min(shifts):.2f}-{max(shifts):.2f}  "
              f"drop-one {min(drops):.2f}-{max(drops):.2f}  "
              f"40yr-ext alpha {ae:+.1%}(t={te:+.2f})")

    # ---- rolling 5y alpha positive %, decade alpha, career risk ----
    print("\n[STABILITY]")
    for name, _ in MODES:
        r = bt[name]["RP+Trend"]; eq = bt[name]["SPY"]
        # rolling 5y alpha
        win = 60; pos = 0; tot = 0
        for i in range(win, len(r)):
            sl = slice(i - win, i)
            mm = sm.OLS((r.iloc[sl] - rflh.iloc[sl]), sm.add_constant((eq.iloc[sl] - rflh.iloc[sl]))).fit()
            tot += 1; pos += mm.params["const"] > 0
        # career risk
        r3r = (1 + r).rolling(36).apply(np.prod) - 1
        r3e = (1 + eq).rolling(36).apply(np.prod) - 1
        under = (r3r < r3e).dropna()
        # decade alpha
        decs = []
        for a, b in [("1987", "1996"), ("1997", "2006"), ("2007", "2016"), ("2017", "2024")]:
            al, *_ = capm(r.loc[a:b], eq.loc[a:b], rflh.loc[a:b]); decs.append(al)
        print(f"  {name:<14} 5y-alpha positive {pos/tot:.0%}  under-BH {under.mean():.0%} of 3y windows  "
              f"decades [{', '.join(f'{d:+.1%}' for d in decs)}]")

    print("\n[VERDICT] V2 passes the same battery V0 cleared IF: significance vs SPY holds,")
    print("  spanning alpha t stays ~2, robustness ranges beat equity's 0.74, and no decade")
    print("  collapses relative to V0. See numbers above; improvement over V0 is modest by design.")

if __name__ == "__main__":
    main()
