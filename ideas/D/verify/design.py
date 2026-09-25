"""Design period only (targets 1992-02 .. 1999-12). Chooses q for S2 and freezes parameters."""
import json, numpy as np, pandas as pd
from vrp_common import *

P = build_panel()
print("Panel", P.index.min().date(), "->", P.index.max().date(), "rows", len(P))
print(P[["VIX","IV","RV","VRP","mkt_rf","rf","spy_ret"]].describe().T.round(5))
print("Missing VRP months:", P["VRP"].isna().sum())
print(P[["VIX","IV","RV","VRP","mkt_rf"]].head(3)); print(P[["VIX","IV","RV","VRP","mkt_rf","spy_ret"]].tail(3))

DES_O1, DES_O2 = "1992-01-31", "1999-11-30"     # origins -> targets 1992-02..1999-12 (no overlap with test)
print("\n== Design-period OOS forecasts (expanding from 1990, min 24 obs)")
for h in (1, 3):
    last = "1999-09-30" if h == 3 else DES_O2   # targets must end by 1999-12
    F = oos_forecasts(P["VRP"], P["mkt_rf"], h, DES_O1, last)
    r2, t, p, n = clark_west(F, h)
    print(f"h={h}: n={n} OOS R2={r2:+.4f} CW t={t:+.2f} p={p:.3f} mean slope={F['slope'].mean():+.3f}")

print("\n== Design-period strategies on French Mkt (Mkt-RF+RF), cost 2 bp, returns 1992-02..1999-12")
risky, rf = P["mkt"], P["rf"]
F1 = oos_forecasts(P["VRP"], P["mkt_rf"], 1, "1992-01-31", DES_O2)
MKTRF_FULL = french_monthly()["Mkt-RF"]  # full history so the 60m variance exists from 1990
S1w = mv_weights(F1["f_model"], MKTRF_FULL); S0w = mv_weights(F1["f_mean"], MKTRF_FULL)
res = {}
res["B1 buy&hold"] = backtest(pd.Series(1.0, index=P.index), risky, rf, 2, "1992-02", "1999-12")
res["S1 VRP mean-var"] = backtest(S1w, risky, rf, 2, "1992-02", "1999-12")
res["S0 hist-mean mean-var"] = backtest(S0w, risky, rf, 2, "1992-02", "1999-12")
mkt_px = (1 + P["mkt"]).cumprod()
res["B2 SMA10"] = backtest(sma_rule(mkt_px), risky, rf, 2, "1992-02", "1999-12")
res["B3 voltarget15"] = backtest(voltarget(P["RV"]), risky, rf, 2, "1992-02", "1999-12")
best = None
for q in (0.10, 0.20, 0.30):
    w = expanding_quantile_rule(P["VRP"], q)
    bt = backtest(w, risky, rf, 2, "1992-02", "1999-12"); res[f"S2 VRP q={q}"] = bt
    sr = sharpe_lo(bt["ex"])[0]
    if best is None or sr > best[1]: best = (q, sr)
    res[f"B4 VIX q={q}"] = backtest(expanding_quantile_rule(P["IV"], q), risky, rf, 2, "1992-02", "1999-12")
rows = [summarize(k, v, 2) for k, v in res.items()]
pd.set_option("display.width", 200)
print(pd.DataFrame(rows).set_index("name").round(3))
print(f"\nChosen q (max design Sharpe of S2): {best[0]} (design Sharpe {best[1]:.3f})")
json.dump({"q": best[0], "gamma": 3.0, "var_window": 60, "voltarget": 0.15, "sma": 10,
           "cost_bp": 2, "frozen_on": "design period 1992-02..1999-12"},
          open("frozen_params.json", "w"), indent=1)
print("Frozen params written:", open("frozen_params.json").read())
