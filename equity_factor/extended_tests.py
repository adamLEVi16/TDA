"""
EXTENDED VALIDATION SUITE — deeper and longer than the original torture tests.

[E1] 1981-2024 extension (44 years): a reduced 4-fund variant reaching back
     through the Volcker rate shock and 1981-82 recession — a rising-rate
     stress regime the 1987+ sample barely contains. (Yahoo fund data floor
     is 1980; VFINX/VWNDX equities, VWESX long IG bonds, FGOVX govt bonds.)
[E2] Rolling 3-year Sharpe and rolling 5-year CAPM alpha (stability, not
     just two halves).
[E3] Decade-by-decade CAPM alpha.
[E4] NBER recession conditioning.
[E5] Rate-regime split (rising vs falling rates, proxied by trailing 12-month
     long-Treasury return sign — self-contained, no external yield feed).
[E6] Execution-lag stress: rebalance 5/10/21 sessions late.
[E7] Stationary bootstrap (random block lengths) for the Sharpe difference —
     does p=0.002 depend on the fixed-block choice?
[E8] Career-risk: rolling 3-year windows underperforming buy-and-hold.
[E9] Tail statistics: skew, kurtosis, VaR/CVaR.
Results feed tearsheet.py; run standalone for the full printout.
"""
import warnings; warnings.filterwarnings("ignore")
import os, sys, json
import numpy as np, pandas as pd, statsmodels.api as sm
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import multi_asset as MA
from backtest import metrics
from torture_test import block_bootstrap_sharpe_diff
from long_history import LH_ASSETS, LH_BENCH, START

HERE = os.path.dirname(os.path.abspath(__file__))
EXT_ASSETS = ["VFINX", "VWNDX", "VWESX", "FGOVX"]
EXT_BENCH = ("VFINX", "VWESX")

NBER = [("1981-07", "1982-11"), ("1990-07", "1991-03"), ("2001-03", "2001-11"),
        ("2007-12", "2009-06"), ("2020-02", "2020-04")]

sh = lambda r: r.mean() / r.std() * np.sqrt(12) if r.std() > 0 else np.nan


def capm(y, mkt, rf):
    d = pd.concat([(y - rf).rename("y"), (mkt - rf).rename("m")], axis=1).dropna()
    m = sm.OLS(d["y"], sm.add_constant(d["m"])).fit(cov_type="HAC", cov_kwds={"maxlags": 6})
    return m.params["const"] * 12, m.tvalues["const"], m.pvalues["const"], m.params["m"]


def stationary_bootstrap_sharpe_diff(a, b, mean_block=6, n=5000, seed=3):
    rng = np.random.default_rng(seed)
    a, b = a.values, b.values; T = len(a)
    diffs = np.empty(n)
    for k in range(n):
        idx = np.empty(T, dtype=int); i = 0
        while i < T:
            start = rng.integers(0, T)
            L = min(rng.geometric(1 / mean_block), T - i)
            idx[i:i+L] = (start + np.arange(L)) % T; i += L
        sa, sb = a[idx], b[idx]
        diffs[k] = (sa.mean()/sa.std() - sb.mean()/sb.std()) * np.sqrt(12)
    return diffs.mean(), np.percentile(diffs, 2.5), np.percentile(diffs, 97.5), (diffs <= 0).mean()


def main():
    out = {}
    bt, _ = MA.run(assets=LH_ASSETS, bench=LH_BENCH, start=START)
    rp, eq = bt["RP+Trend"], bt["SPY"]
    rf = MA.get_rf_monthly(bt.index)

    print("=" * 80)
    print("[E1] 1981-2024 EXTENSION (4 funds incl. Volcker era; reduced diversification)")
    bte, _ = MA.run(assets=EXT_ASSETS, bench=EXT_BENCH, start="1980-01-01")
    rpe, eqe = bte["RP+Trend"], bte["SPY"]
    rfe = MA.get_rf_monthly(bte.index)
    for lbl, s in [("RP+Trend (4-fund, 1981+)", rpe), ("VFINX buy & hold", eqe),
                   ("60/40", bte["60/40"])]:
        m = metrics(s)
        print(f"  {lbl:<26} CAGR={m['CAGR']:+.1%} Sharpe={m['Sharpe']:+.2f} MaxDD={m['MaxDD']:.1%}")
    d_, lo, hi, p_ = block_bootstrap_sharpe_diff(rpe, eqe)
    a_, t_, pa_, b_ = capm(rpe, eqe, rfe)
    print(f"  ΔSharpe vs eq {d_:+.2f} CI[{lo:+.2f},{hi:+.2f}] p={p_:.3f} | "
          f"CAPM alpha {a_:+.2%}/yr t={t_:+.2f} p={pa_:.3f} beta={b_:.2f}  "
          f"({len(bte)} months = {len(bte)/12:.0f} years)")
    out["ext"] = {"months": len(bte), "start": str(bte.index.min().date()),
                  "alpha": a_, "t": t_, "p": pa_, "dsharpe_p": p_}

    print("\n[E2] ROLLING STABILITY (1987 5-fund version)")
    roll_sh_rp = rp.rolling(36).apply(sh); roll_sh_eq = eq.rolling(36).apply(sh)
    win = 60
    ra = {}
    for i in range(win, len(rp)):
        seg = slice(i - win, i)
        y = (rp.iloc[seg] - rf.iloc[seg]); x = (eq.iloc[seg] - rf.iloc[seg])
        mm = sm.OLS(y, sm.add_constant(x)).fit()
        ra[rp.index[i]] = mm.params.iloc[1 if False else 0] if False else mm.params["const"] * 12
    roll_alpha = pd.Series(ra)
    print(f"  rolling 3y Sharpe: strategy>equity in {(roll_sh_rp>roll_sh_eq).mean():.0%} of windows")
    print(f"  rolling 5y CAPM alpha: positive in {(roll_alpha>0).mean():.0%} of windows "
          f"(median {roll_alpha.median():+.2%}/yr)")
    pd.DataFrame({"sh_rp": roll_sh_rp, "sh_eq": roll_sh_eq}).to_csv(os.path.join(HERE, "cache/_roll_sh.csv"))
    roll_alpha.to_frame("alpha").to_csv(os.path.join(HERE, "cache/_roll_alpha.csv"))

    print("\n[E3] DECADE-BY-DECADE CAPM ALPHA")
    for lab, a, b in [("1987-1996", "1987", "1996"), ("1997-2006", "1997", "2006"),
                      ("2007-2016", "2007", "2016"), ("2017-2024", "2017", "2024")]:
        al, t, pv, be = capm(rp.loc[a:b], eq.loc[a:b], rf.loc[a:b])
        print(f"  {lab}: alpha={al:+.2%}/yr  t={t:+.2f}  beta={be:.2f}")

    print("\n[E4] NBER RECESSIONS vs EXPANSIONS")
    rec = pd.Series(False, index=rp.index)
    for a, b in NBER:
        rec.loc[a:b] = True
    print(f"  recession months (n={rec.sum()}):  strategy {rp[rec].mean()*12:+.1%}/yr  "
          f"equity {eq[rec].mean()*12:+.1%}/yr")
    print(f"  expansion months (n={(~rec).sum()}): strategy {rp[~rec].mean()*12:+.1%}/yr  "
          f"equity {eq[~rec].mean()*12:+.1%}/yr")
    out["recession"] = {"strat": rp[rec].mean()*12, "eq": eq[rec].mean()*12}

    print("\n[E5] RATE REGIME (trailing 12m long-Treasury return sign)")
    px = MA.load_prices(assets=["VUSTX"], start=START)
    bond12 = px["VUSTX"].resample("ME").last().pct_change(12).reindex(rp.index)
    rising = bond12 < 0
    for lab, msk in [("rising rates", rising), ("falling rates", ~rising)]:
        print(f"  {lab:<14} (n={msk.sum()}): strategy {rp[msk].mean()*12:+.1%}/yr "
              f"Sharpe {sh(rp[msk]):+.2f} | equity {eq[msk].mean()*12:+.1%}/yr {sh(eq[msk]):+.2f}")

    print("\n[E6] EXECUTION-LAG STRESS (rebalance N sessions late)")
    for lag in (5, 10, 21):
        b2, _ = MA.run(assets=LH_ASSETS, bench=LH_BENCH, start=START, shift_days=lag)
        print(f"  {lag:>2} sessions late: Sharpe={sh(b2['RP+Trend']):+.2f}  "
              f"MaxDD={metrics(b2['RP+Trend'])['MaxDD']:.1%}")

    print("\n[E7] STATIONARY BOOTSTRAP (random block lengths) — Sharpe diff vs equity")
    for mb in (3, 6, 12):
        d_, lo, hi, p_ = stationary_bootstrap_sharpe_diff(rp, eq, mean_block=mb)
        print(f"  mean block {mb:>2}mo: ΔSharpe={d_:+.2f}  CI[{lo:+.2f},{hi:+.2f}]  p={p_:.3f}")

    print("\n[E8] CAREER RISK (rolling 3y vs buy-and-hold)")
    r3_rp = (1 + rp).rolling(36).apply(np.prod) - 1
    r3_eq = (1 + eq).rolling(36).apply(np.prod) - 1
    under = (r3_rp < r3_eq).dropna()
    runs, cur = [], 0
    for v in under.values:
        cur = cur + 1 if v else 0
        runs.append(cur)
    print(f"  underperforms equity in {under.mean():.0%} of rolling 3y windows; "
          f"longest stretch {max(runs)} consecutive months")
    out["career"] = {"pct_under3y": float(under.mean()), "longest": int(max(runs))}

    print("\n[E9] TAIL STATS (monthly)")
    for lbl, s in [("strategy", rp), ("equity", eq)]:
        var5, cvar5 = s.quantile(0.05), s[s <= s.quantile(0.05)].mean()
        print(f"  {lbl:<9} skew={s.skew():+.2f} kurt={s.kurt():+.1f} "
              f"VaR5={var5:+.1%} CVaR5={cvar5:+.1%} worst={s.min():+.1%}")

    json.dump(out, open(os.path.join(HERE, "cache/_ext_results.json"), "w"), default=float)
    print("\nSaved rolling series + summary for tearsheet.")


if __name__ == "__main__":
    main()
