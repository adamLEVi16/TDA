"""TEST PERIOD (return months 1990-01..2024-12), run ONCE with frozen_params.json. Industries."""
import json
import pickle

import numpy as np
import pandas as pd

import data
import engine as E

P = json.load(open("frozen_params.json"))
K, COST = P["k"], P["cost_industry"]
print("frozen params:", P)
d, m = data.industries()
m = m.loc[:"2024-12-31"]
sigs = pickle.load(open("cache/ind_signals.pkl", "rb"))["sigs"]
S = {key: v.loc["1989-12-31":"2024-11-30"] for key, v in sigs.items()}   # formation 1989-12 -> return 1990-01
ff5, ff3 = data.factors_monthly()
rf = ff3["RF"]
SUB = [("1990-01", "2007-12"), ("2008-01", "2024-12")]
KEYS = ["OWN1", "OWN12", ("NET1", K), ("NET12", K), ("LAPGAP1", K), ("NET1_PURE", K)]
lab = lambda key: key if isinstance(key, str) else f"{key[0]}(k={key[1]})"

print("\n=== L/S quintile portfolios, 10 bp one-way, self-financing (no RF) ===")
LS = {}
for key in KEYS:
    bt = E.backtest(E.rank_weights(S[key], frac=P["quantile"]), m, COST)
    LS[key] = bt
    print(E.summarize(bt, label=f"L/S {lab(key)}")[0])
    for a, b in SUB:
        print("   " + E.summarize(bt.loc[a:b], label=f"  sub {a[:4]}-{b[:4]}", boot=False)[0])
print("missing held returns (earn 0):", {lab(k): v.attrs["n_missing_held"] for k, v in LS.items()})

print("\n=== Long-only top quintile vs EW all industries (excess of RF), 10 bp ===")
EW = E.backtest(E.ew_weights(S["OWN1"]), m, COST)
print(E.summarize(EW, rf=rf, label="EW all industries")[0])
mkt = (ff3["Mkt-RF"]).loc["1990-01":"2024-12"]
print(f"{'Market (FF Mkt-RF, no cost)':<34} ann excess {12*mkt.mean():+.2%} SR {E.sharpe(mkt):+.2f}")
LO = {}
for key in KEYS:
    bt = E.backtest(E.rank_weights(S[key], frac=P["quantile"], long_only=True), m, COST)
    LO[key] = bt
    print(E.summarize(bt, rf=rf, label=f"LO {lab(key)}")[0])
print("\nSharpe differences (net, excess of RF), paired stationary bootstrap:")
ex = lambda bt: bt["net"] - rf.reindex(bt.index)
for key, base in [(("NET1", K), "EW"), (("NET12", K), "EW"), (("NET1", K), "OWN12"), (("NET12", K), "OWN12"),
                  ("OWN12", "EW")]:
    b = ex(EW) if base == "EW" else ex(LO[base])
    dsr, ci, p = E.boot_sharpe_diff(ex(LO[key]), b)
    print(f"  LO {lab(key):<14} - {base:<6}: dSR {dsr:+.2f} CI [{ci[0]:+.2f},{ci[1]:+.2f}] p={p:.3f}")

print("\n=== Fama-MacBeth (test period), z-scored, mean slope %/month, NW(6) t ===")
fm = E.fama_macbeth(S, m, ["OWN1", "OWN12", ("NET1", K), ("NET12", K)])
for per in [("1990-01", "2024-12")] + SUB:
    f = fm.loc[per[0]:per[1]]
    print(f"  {per[0]}..{per[1]} n={len(f)}: " + "  ".join(
        f"{c}={100*E.nw_mean(f[c])[0]:+.3f}(t={E.nw_mean(f[c])[1]:+.2f})" for c in f))

print("\n=== Spanning regressions of NET L/S (net of cost) ===")
X1 = pd.concat([LS["OWN12"]["net"].rename("OWN12_LS"), LS["OWN1"]["net"].rename("OWN1_LS")], axis=1)
X2 = ff5.drop(columns="RF")
X3 = pd.concat([X2, X1], axis=1)
for key in [("NET1", K), ("NET12", K), ("LAPGAP1", K), ("NET1_PURE", K)]:
    for xn, X in [("own-mom L/S", X1), ("FF5+UMD", X2), ("FF5+UMD+own-mom", X3)]:
        for per in [("1990-01", "2024-12")] + SUB:
            for col in ["net", "gross"]:
                y = LS[key][col].loc[per[0]:per[1]]
                f = E.spanning(y, X.loc[per[0]:per[1]])
                ci = f.conf_int().loc["const"] * 12
                print(f"  {lab(key):<16} {col:<5} on {xn:<16} {per[0][:4]}-{per[1][:4]}: alpha {12*f.params['const']:+.2%} "
                      f"[{ci[0]:+.1%},{ci[1]:+.1%}] t={f.tvalues['const']:+.2f} R2={f.rsquared:.2f}")
    f = E.spanning(LS[key]["net"], X3)
    print("     loadings (full test, FF5+UMD+own):", " ".join(f"{c}={f.params[c]:+.2f}(t={f.tvalues[c]:+.1f})"
                                                       for c in X3.columns))

print("\ncorrelation of net L/S returns (test):")
print(pd.concat({lab(k): v["net"] for k, v in LS.items()}, axis=1).corr().round(2).to_string())
pickle.dump({"LS": LS, "LO": LO, "EW": EW}, open("cache/test_ind_results.pkl", "wb"))
