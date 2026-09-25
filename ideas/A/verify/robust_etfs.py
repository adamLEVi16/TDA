"""POST-HOC ETF robustness (added after the one-time ETF test; listed as amendments in SPEC.md).
Not used to pick anything. Every variant printed here is counted in the final report.
 a) N in {3, 7} and k in {3, 10} for long-only NET1/NET12 (vs frozen N=5, k=5)
 b) placebo: k=5 RANDOM neighbours, 200 draws (seed 0), long-only top-5 net Sharpe
 c) earlier holdout 2000-01..2006-12 (ETFs available then; never looked at before), frozen parameters."""
import json

import numpy as np
import pandas as pd

import data
import engine as E

P = json.load(open("frozen_params.json"))
K, COST, N = P["k"], P["cost_etf"], P["etf_top_n"]
px = data.etf_prices().loc[:"2024-12-31"]
spy_px = px.pop("SPY")
px = px[data.ETFS]
daily = px.pct_change(fill_method=None)
mpx = px.resample("ME").last()
monthly = mpx.pct_change(fill_method=None)
monthly[mpx.isna()] = np.nan
spy_m = spy_px.resample("ME").last().pct_change(fill_method=None)
rf = data.factors_monthly()[1]["RF"]
ex = lambda bt: bt["net"] - rf.reindex(bt.index)
form = monthly.index[(monthly.index >= "2006-12-31") & (monthly.index <= "2024-11-30")]
S = E.build_signals(daily, monthly, [3, 5, 10], form)
EW = E.backtest(E.ew_weights(S["OWN1"]), monthly, COST)
print(f"EW all ETFs 2007-2024 net excess SR {E.sharpe(ex(EW)):+.2f}")

print("\n=== a) N and k variants, long-only, 2007-2024, 3 bp: net excess ann mean, Sharpe, dSR vs EW (p) ===")
for name in ["NET1", "NET12", "OWN12"]:
    for k in ([3, 5, 10] if name != "OWN12" else [None]):
        for n in [3, 5, 7]:
            sig = S[name] if k is None else S[(name, k)]
            bt = E.backtest(E.rank_weights(sig, n=n, long_only=True), monthly, COST)
            dsr, ci, p = E.boot_sharpe_diff(ex(bt), ex(EW))
            print(f"  {name:<6} k={str(k):<4} N={n}: {12*ex(bt).mean():+.2%} SR {E.sharpe(ex(bt)):+.2f} "
                  f"dSR vs EW {dsr:+.2f} [{ci[0]:+.2f},{ci[1]:+.2f}] p={p:.3f}")

print("\n=== b) placebo: random neighbours (k=5), 200 draws, LO top5 net excess Sharpe, 2007-2024 ===")
rng = np.random.default_rng(0)
pl = {"NET1": [], "NET12": []}
for b in range(200):
    s = E.build_signals(daily, monthly, [K], form, placebo_rng=rng)
    for name in pl:
        bt = E.backtest(E.rank_weights(s[(name, K)], n=N, long_only=True), monthly, COST)
        pl[name].append(E.sharpe(ex(bt)))
for name in pl:
    real = E.sharpe(ex(E.backtest(E.rank_weights(S[(name, K)], n=N, long_only=True), monthly, COST)))
    a = np.array(pl[name])
    print(f"  {name:<6}: real graph SR {real:+.2f} | random-neighbour mean {a.mean():+.2f} sd {a.std():.2f} "
          f"95th {np.percentile(a, 95):+.2f} | share >= real {np.mean(a >= real):.3f}")

print("\n=== c) earlier holdout, return months 2000-01..2006-12, frozen k=5, N=5, 3 bp ===")
form0 = monthly.index[(monthly.index >= "1999-12-31") & (monthly.index <= "2006-11-30")]
S0 = E.build_signals(daily, monthly, [K], form0)
nel = S0["OWN1"].notna().sum(axis=1)
print(f"  eligible ETFs per month: first {nel.iloc[0]} ({nel.index[0]:%Y-%m}), last {nel.iloc[-1]}, min {nel.min()}")
EW0 = E.backtest(E.ew_weights(S0["OWN1"]), monthly, COST)
SPY0 = E.backtest(pd.DataFrame({"SPY": 1.0}, index=form0), spy_m.to_frame("SPY"), P["cost_spy"])
print(E.summarize(EW0, rf=rf, label="EW all ETFs")[0])
print(E.summarize(SPY0, rf=rf, label="SPY buy&hold")[0])
LO0 = {}
for key in ["OWN1", "OWN12", ("NET1", K), ("NET12", K)]:
    LO0[key] = E.backtest(E.rank_weights(S0[key], n=N, long_only=True), monthly, COST)
    print(E.summarize(LO0[key], rf=rf, label=f"LO top5 {key}")[0])
for key in [("NET1", K), ("NET12", K)]:
    for bname, b in [("EW", EW0), ("OWN12 top5", LO0["OWN12"])]:
        dsr, ci, p = E.boot_sharpe_diff(ex(LO0[key]), ex(b))
        print(f"  LO {str(key):<12} - {bname:<10}: dSR {dsr:+.2f} CI [{ci[0]:+.2f},{ci[1]:+.2f}] p={p:.3f}")
for key in ["OWN1", "OWN12", ("NET1", K), ("NET12", K)]:
    print(E.summarize(E.backtest(E.rank_weights(S0[key], n=N), monthly, COST), label=f"L/S {key}", boot=False)[0])
