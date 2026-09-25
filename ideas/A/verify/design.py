"""DESIGN PERIOD ONLY (return months <= 1989-12). Evaluates the pre-registered signals for k in {3,5,10},
picks k by the SPEC rule, and freezes it to frozen_params.json. Does not touch 1990+ returns."""
import json
import pickle

import numpy as np
import pandas as pd

import data
import engine as E

DESIGN_END = pd.Timestamp("1989-12-31")
COST = 0.0010
KS = [3, 5, 10]

d, m = data.industries()
m = m.loc[:DESIGN_END]                         # hard cut: design code cannot see test returns
sigs = pickle.load(open("cache/ind_signals.pkl", "rb"))["sigs"]
form_last = m.index[-2]                        # formation 1989-11 -> return 1989-12
S = {key: v.loc[:form_last] for key, v in sigs.items()}
ff5, ff3 = data.factors_monthly()
rf = ff3["RF"]


def ls(sig):
    return E.backtest(E.rank_weights(sig, frac=0.2), m, COST)


def lo(sig):
    return E.backtest(E.rank_weights(sig, frac=0.2, long_only=True), m, COST)


print("=== DESIGN PERIOD (industries, 10 bp one-way) ===")
res = {}
for name in ["OWN1", "OWN12"]:
    bt = ls(S[name]); res[name] = bt
    print(E.summarize(bt, label=f"L/S {name}", boot=False)[0])
score = {}
for k in KS:
    for name in ["NET1", "NET12", "LAPGAP1", "NET1_PURE"]:
        bt = ls(S[(name, k)]); res[(name, k)] = bt
        print(E.summarize(bt, label=f"L/S {name} k={k}", boot=False)[0])
    score[k] = np.mean([E.sharpe(res[("NET1", k)]["net"]), E.sharpe(res[("NET12", k)]["net"])])
print("\nk-selection score (avg net Sharpe of NET1, NET12 L/S):", {k: round(v, 3) for k, v in score.items()})
k_star = max(score, key=score.get)
print("FROZEN k =", k_star)

print("\n--- long-only top quintile vs EW (excess of RF), design, k*=%d" % k_star)
ew = E.backtest(E.ew_weights(S["OWN1"]), m, COST)
print(E.summarize(ew, rf=rf, label="EW all industries", boot=False)[0])
for key in ["OWN1", "OWN12", ("NET1", k_star), ("NET12", k_star), ("LAPGAP1", k_star), ("NET1_PURE", k_star)]:
    print(E.summarize(lo(S[key]), rf=rf, label=f"LO {key}", boot=False)[0])

print("\n--- Fama-MacBeth, design period, z-scored regressors, k*=%d (mean slope in %%/month, NW t)" % k_star)
fm = E.fama_macbeth(S, m, ["OWN1", "OWN12", ("NET1", k_star), ("NET12", k_star)])
for c in fm:
    mu, t, _ = E.nw_mean(fm[c]); print(f"  {c:<14} {100*mu:+.3f}  t={t:+.2f}  n={len(fm)}")

print("\n--- spanning on own-momentum L/S (design), k*=%d; alpha %%/yr, NW t" % k_star)
X = pd.concat([res["OWN12"]["net"].rename("OWN12_LS"), res["OWN1"]["net"].rename("OWN1_LS")], axis=1)
for key in [("NET1", k_star), ("NET12", k_star), ("LAPGAP1", k_star), ("NET1_PURE", k_star)]:
    f = E.spanning(res[key]["net"], X)
    print(f"  {str(key):<18} alpha {12*f.params['const']:+.2%} t={f.tvalues['const']:+.2f}  "
          f"b(OWN12)={f.params['OWN12_LS']:+.2f} b(OWN1)={f.params['OWN1_LS']:+.2f} R2={f.rsquared:.2f}")

print("\ncorrelation of net L/S returns (design):")
cc = pd.concat({str(k): v["net"] for k, v in res.items() if k in
                ["OWN1", "OWN12", ("NET1", k_star), ("NET12", k_star), ("LAPGAP1", k_star), ("NET1_PURE", k_star)]},
               axis=1)
print(cc.corr().round(2).to_string())

json.dump({"k": int(k_star), "corr_window": E.WIN, "min_obs": E.MIN_OBS, "alpha": E.ALPHA, "T": E.T_STEPS,
           "quantile": 0.2, "etf_top_n": 5, "cost_industry": 0.0010, "cost_etf": 0.0003, "cost_spy": 0.0002,
           "selection_rule": "argmax_k mean(net Sharpe NET1 L/S, net Sharpe NET12 L/S), design 1927-07..1989-12",
           "score": {str(k): v for k, v in score.items()}}, open("frozen_params.json", "w"), indent=1)
print("wrote frozen_params.json")
