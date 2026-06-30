"""
Evaluate RP+Trend as a DIVERSIFYING SLEEVE, not a standalone strategy.

"Alpha = return uncorrelated with the market." A positive-return stream that is
only ~0.4 correlated with SPY can improve a SPY portfolio's Sharpe and drawdown
even if its standalone Sharpe is lower. This is the institutionally-correct way to
value it (vs the head-to-head Sharpe comparison, which is the wrong test).

Reports: CAPM alpha of RP+Trend vs the market, and the SPY+sleeve blend frontier.
"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, statsmodels.api as sm
import multi_asset as MA
from torture_test import block_bootstrap_sharpe_diff

sh = lambda r: r.mean() / r.std() * np.sqrt(12) if r.std() > 0 else np.nan
def dd(r): c = (1 + r).cumprod(); return (c / c.cummax() - 1).min()
def cagr(r): return (1 + r).prod() ** (12 / len(r)) - 1

def main():
    bt, _ = MA.run()
    rp, spy = bt["RP+Trend"], bt["SPY"]
    rf = MA.get_rf_monthly(bt.index)

    X = sm.add_constant((spy - rf).rename("mkt"))
    m = sm.OLS((rp - rf), X).fit(cov_type="HAC", cov_kwds={"maxlags": 6})
    print("CAPM:  RP+Trend(excess) ~ SPY(excess)")
    print(f"  alpha = {m.params['const']*12:+.2%}/yr   t = {m.tvalues['const']:.2f}"
          f"   beta = {m.params['mkt']:.2f}   corr = {rp.corr(spy):.2f}\n")

    print(f"Blend  w*RP+Trend + (1-w)*SPY (monthly rebal):")
    print(f"  {'wRP':>5}{'CAGR':>9}{'Vol':>8}{'Sharpe':>8}{'MaxDD':>9}")
    for w in [0, .2, .3, .5, .7, .8, 1.0]:
        b = w * rp + (1 - w) * spy
        print(f"  {w:>5.0%}{cagr(b):>8.1%}{b.std()*np.sqrt(12):>8.1%}{sh(b):>8.2f}{dd(b):>9.1%}")

    ws = np.linspace(0, 1, 101)
    wopt = ws[int(np.argmax([sh(w*rp+(1-w)*spy) for w in ws]))]
    bo = wopt*rp + (1-wopt)*spy
    d, lo, hi, p = block_bootstrap_sharpe_diff(bo, spy)
    print(f"\n  Sharpe-max: {wopt:.0%} RP+Trend / {1-wopt:.0%} SPY -> Sharpe={sh(bo):.2f}"
          f"  CAGR={cagr(bo):.1%}  MaxDD={dd(bo):.1%}")
    print(f"  vs SPY (Sharpe 0.71): ΔSharpe={d:+.2f}  95% CI [{lo:+.2f},{hi:+.2f}]  p={p:.2f}")

if __name__ == "__main__":
    main()
