"""
THE STRICT ALPHA TEST on the 37-year validated strategy.

Claim being tested: RP+Trend's return is NOT replicable by any static mix of
its own five assets -- i.e. the monthly TIMING adds genuine alpha.

Method: regress the strategy's monthly EXCESS return (over T-bills) on the
excess returns of ALL FIVE underlying funds (US eq, intl eq, long Tsy, gold,
HY credit), 1987-2024, HAC(Newey-West) errors. The intercept is return that no
constant-weight combination of the same ingredients could have produced --
spanning-test alpha. Also reported: plain CAPM alpha, and the two sub-halves.
"""
import warnings; warnings.filterwarnings("ignore")
import os, sys
import numpy as np, pandas as pd, statsmodels.api as sm
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import multi_asset as MA
from long_history import LH_ASSETS, LH_BENCH, START

def main():
    bt, _ = MA.run(assets=LH_ASSETS, bench=LH_BENCH, start=START)
    px = MA.load_prices(assets=LH_ASSETS, start=START)
    aret = px.resample("ME").last().pct_change().reindex(bt.index)
    rf = MA.get_rf_monthly(bt.index)
    y = (bt["RP+Trend"] - rf).rename("strat_ex")
    X = aret.sub(rf, axis=0)
    d = pd.concat([y, X], axis=1).dropna()
    print("=" * 78)
    print(f"  SPANNING / ALPHA TEST — RP+Trend vs its own 5 assets, "
          f"{d.index.min():%Y-%m} -> {d.index.max():%Y-%m} ({len(d)} months)")
    print("=" * 78)

    def reg(cols, label, seg=None):
        dd = d if seg is None else d.loc[seg[0]:seg[1]]
        m = sm.OLS(dd["strat_ex"], sm.add_constant(dd[cols])).fit(
            cov_type="HAC", cov_kwds={"maxlags": 6})
        a, t, p = m.params["const"] * 12, m.tvalues["const"], m.pvalues["const"]
        betas = "  ".join(f"{c}={m.params[c]:+.2f}" for c in cols)
        print(f"\n  {label}")
        print(f"    alpha = {a:+.2%}/yr   t = {t:+.2f}   p = {p:.4f}   R2 = {m.rsquared:.2f}")
        print(f"    betas: {betas}")
        return a, t, p

    reg(["VFINX"], "[1] CAPM (market only)")
    reg(LH_ASSETS, "[2] FULL SPANNING (all 5 own assets — the strict test)")
    reg(LH_ASSETS, "[3] first half", seg=(None, d.index[len(d)//2]))
    reg(LH_ASSETS, "[4] second half", seg=(d.index[len(d)//2], None))

    print("\n  Interpretation: a significant intercept in [2] means NO constant-")
    print("  weight portfolio of the same five funds could replicate the strategy —")
    print("  the timing itself generated the excess return.")

if __name__ == "__main__":
    main()
