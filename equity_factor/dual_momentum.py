"""
DUAL MOMENTUM on the 37-year fund-proxy history -- the "makes more money"
candidate. Everything so far either protects (trend/risk-parity) or is dead;
dual momentum (Antonacci) is the one published mechanism whose claim is higher
RAW return than equities, not just higher Sharpe.

PRE-COMMITTED SPECS (both straight from the literature, zero fitted params):
  GEM (classic Global Equities Momentum):
    Monthly. r12 = trailing 12-month return. If r12(US eq) > r12(T-bill):
    hold whichever of US/intl equity has higher r12. Else hold long Treasuries.
  RANK2 (relative momentum on our 5 proxies):
    Monthly. Rank all 5 assets by r12; hold top-2 equal-weight, but any
    top-2 asset with r12 < T-bill r12 goes to cash (absolute filter).

Signals use month t-1 and earlier closes; positions earn month t (1-month
implementation lag as everywhere else in this repo). Costs 10 bps/side on
turnover. Benchmarks: VFINX buy&hold, 60/40, and our RP+Trend.
"""
import warnings; warnings.filterwarnings("ignore")
import os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import multi_asset as MA
from backtest import metrics
from torture_test import block_bootstrap_sharpe_diff
from long_history import LH_ASSETS, LH_BENCH, START

COST = 10 / 1e4

def monthly_panel():
    px = MA.load_prices(assets=list(dict.fromkeys(LH_ASSETS + list(LH_BENCH))),
                        start=START)
    m = px.resample("ME").last()
    ret = m.pct_change()
    r12 = m.shift(1) / m.shift(13) - 1          # momentum through month t-1
    rf = MA.get_rf_monthly(ret.index)
    rf12 = (1 + rf).rolling(12).apply(np.prod, raw=True).shift(1) - 1
    return ret.dropna(how="all"), r12, rf, rf12

def run_gem(ret, r12, rf, rf12):
    us, intl, bond = "VFINX", "VWIGX", "VUSTX"
    rows, w_prev = [], None
    for t in ret.index:
        need = [r12.at[t, us], r12.at[t, intl], rf12.at[t]]
        if any(pd.isna(x) for x in need) or pd.isna(ret.at[t, bond]):
            continue
        if r12.at[t, us] > rf12.at[t]:
            hold = us if r12.at[t, us] >= r12.at[t, intl] else intl
        else:
            hold = bond
        r = ret.at[t, hold]
        turn = 0.0 if hold == w_prev else (1.0 if w_prev is None else 2.0)
        rows.append({"date": t, "ret": r - turn * COST, "hold": hold})
        w_prev = hold
    return pd.DataFrame(rows).set_index("date")

def run_rank2(ret, r12, rf, rf12):
    assets = LH_ASSETS
    rows, w_prev = [], pd.Series(dtype=float)
    for t in ret.index:
        sig = r12.loc[t, assets].dropna()
        if len(sig) < len(assets) or pd.isna(rf12.at[t]):
            continue
        top = sig.sort_values().index[-2:]
        w = pd.Series(0.0, index=assets)
        for a in top:
            if sig[a] > rf12.at[t]:
                w[a] = 0.5
        cash_w = 1 - w.sum()
        r = float((w * ret.loc[t, assets].fillna(0)).sum()) + cash_w * rf.at[t]
        allk = w.index.union(w_prev.index)
        turn = (w.reindex(allk).fillna(0) - w_prev.reindex(allk).fillna(0)).abs().sum()
        rows.append({"date": t, "ret": r - turn * COST})
        w_prev = w
    return pd.DataFrame(rows).set_index("date")

def show(name, r, bench):
    m = metrics(r)
    print(f"  {name:<28} CAGR={m['CAGR']:+.1%}  Vol={m['Vol']:.1%}  "
          f"Sharpe={m['Sharpe']:+.2f}  MaxDD={m['MaxDD']:.1%}")
    return m

def main():
    ret, r12, rf, rf12 = monthly_panel()
    gem = run_gem(ret, r12, rf, rf12)["ret"]
    rank2 = run_rank2(ret, r12, rf, rf12)["ret"]
    idx = gem.index.intersection(rank2.index)
    bt, _ = MA.run(assets=LH_ASSETS, bench=LH_BENCH, start=START)
    idx = idx.intersection(bt.index)
    gem, rank2 = gem.reindex(idx), rank2.reindex(idx)
    vfinx, sixty, rpt = (bt["SPY"].reindex(idx), bt["60/40"].reindex(idx),
                         bt["RP+Trend"].reindex(idx))
    print("=" * 78)
    print(f"  DUAL MOMENTUM, {idx.min():%Y-%m} -> {idx.max():%Y-%m}  ({len(idx)} months)")
    print("=" * 78)
    for n, s in [("GEM (US/intl/bond switch)", gem), ("RANK2 top-2 of 5", rank2),
                 ("VFINX buy & hold", vfinx), ("60/40", sixty), ("RP+Trend (ours)", rpt)]:
        show(n, s, vfinx)
    print("\nSIGNIFICANCE (block bootstrap Sharpe diff):")
    for n, s in [("GEM", gem), ("RANK2", rank2)]:
        d, lo, hi, p = block_bootstrap_sharpe_diff(s.dropna(), vfinx.dropna())
        tag = "SIGNIFICANT" if lo > 0 else "not significant"
        print(f"  {n} - VFINX: dSharpe={d:+.2f}  CI[{lo:+.2f},{hi:+.2f}]  p={p:.3f} -> {tag}")
    print("\nSUB-PERIODS (CAGR):")
    for lab, a, b in [("1987-1999", "1987", "1999"), ("2000-2009", "2000", "2009"),
                      ("2010-2024", "2010", "2024")]:
        seg = pd.DataFrame({"GEM": gem, "RANK2": rank2, "VFINX": vfinx}).loc[a:b]
        cag = {c: (1 + seg[c].dropna()).prod() ** (12 / max(len(seg[c].dropna()), 1)) - 1
               for c in seg}
        print(f"  {lab}: " + "  ".join(f"{c}={v:+.1%}" for c, v in cag.items()))

if __name__ == "__main__":
    main()
