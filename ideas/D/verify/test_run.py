"""Single pre-registered TEST run: forecast origins 1999-12..2026-06, returns 2000-01..2026-07."""
import json, numpy as np, pandas as pd
from vrp_common import *
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
FP = json.load(open("frozen_params.json")); print("Frozen params:", FP)
P = build_panel(); MKTRF_FULL = french_monthly()["Mkt-RF"]
C = FP["cost_bp"]; S, E = "2000-01", "2026-07"

print("\n== H1: OOS forecasts of Mkt-RF (expanding window from 1990-02 targets)")
for h, last in ((1, "2026-06-30"), (3, "2026-04-30")):
    F = oos_forecasts(P["VRP"], P["mkt_rf"], h, "1999-12-31", last)
    for restrict in (False, True):
        r2, t, p, n = clark_west(F, h, restrict)
        print(f"h={h} {'CT-restricted' if restrict else 'unrestricted '} n={n} OOS R2={r2*100:+.2f}%  CW t={t:+.2f} one-sided p={p:.3f}")
    print(f"   slope at first/last origin {F['slope'].iloc[0]:+.3f}/{F['slope'].iloc[-1]:+.3f}; frac forecasts<0 {(F['f_model']<0).mean():.2%}")

print("\n== In-sample full-period regressions (IN-SAMPLE, for reference only)")
for h in (1, 3):
    y = sum(P["mkt_rf"].shift(-k) for k in range(1, h + 1))
    for sub, (a, b) in {"1990-2026": ("1990-01", "2026-12"), "2000-2026": ("1999-12", "2026-12")}.items():
        b_, t_, r2, T = nw_ols(y.loc[a:b], P["VRP"].loc[a:b], lags=max(h, 1) + 2)
        print(f"h={h} {sub}: slope={b_:+.3f} NW t={t_:+.2f} R2={r2*100:.2f}% T={T}")

print("\n== H2/H3: strategies on SPY, T-bills for cash, cost", C, "bp one-way, returns", S, "->", E)
F1 = oos_forecasts(P["VRP"], P["mkt_rf"], 1, "1999-12-31", "2026-06-30")
risky, rf = P["spy_ret"], P["rf"]
W = {"B1 SPY buy&hold": pd.Series(1.0, index=P.index),
     "S1 VRP mean-var (primary)": mv_weights(F1["f_model"], MKTRF_FULL, FP["gamma"], FP["var_window"]),
     "S0 hist-mean mean-var": mv_weights(F1["f_mean"], MKTRF_FULL, FP["gamma"], FP["var_window"]),
     f"S2 VRP q={FP['q']}": expanding_quantile_rule(P["VRP"], FP["q"]),
     "B2 SMA10": sma_rule(P["spy_px"].dropna(), FP["sma"]),
     "B3 voltarget15": voltarget(P["RV"], FP["voltarget"]),
     f"B4 VIX q={FP['q']}": expanding_quantile_rule(P["IV"], FP["q"])}
BT = {k: backtest(w, risky, rf, C, S, E) for k, w in W.items()}
tab = pd.DataFrame([summarize(k, v, C) for k, v in BT.items()]).set_index("name")
print(tab.round(3).to_string())
b1 = BT["B1 SPY buy&hold"]; b1_0 = backtest(W["B1 SPY buy&hold"], risky, rf, 0, S, E)
print("\nBreak-even one-way cost (bp): [own mean excess = 0] and [mean excess = B1 mean excess at that cost]")
for k, w in W.items():
    be0 = breakeven_cost(w, risky, rf, S, E)
    s0 = backtest(w, risky, rf, 0, S, E); s1 = backtest(w, risky, rf, 1, S, E)
    gap0 = s0["ex"].mean() - b1_0["ex"].mean()
    slope = (s0["ex"].mean() - s1["ex"].mean()) - (b1_0["ex"].mean() - backtest(W["B1 SPY buy&hold"], risky, rf, 1, S, E)["ex"].mean())
    bevb = "never (below B1 even at 0 cost)" if gap0 <= 0 else (f"{gap0/slope:.0f}" if slope > 0 else "inf")
    print(f"  {k:28s} zero-return BE={be0:9.0f}  match-B1 BE={bevb}   gross-of-cost ann ex gap vs B1={gap0*12*100:+.2f}%")

print("\nPaired stationary block bootstrap (block 6, 10000 reps, seed %d): A minus B" % SEED)
for a, b in [(k, "B1 SPY buy&hold") for k in W if k != "B1 SPY buy&hold"] + [("S1 VRP mean-var (primary)", "S0 hist-mean mean-var")]:
    r = paired_bootstrap(BT[a]["ex"].values, BT[b]["ex"].values)
    print(f"  {a:28s} - {b:22s}: dSharpe={r['d_sr']:+.3f} CI[{r['sr_ci'][0]:+.3f},{r['sr_ci'][1]:+.3f}] p(<=0)={r['p_sr']:.3f} | "
          f"dMean={r['d_mu']*100:+.2f}%/yr CI[{r['mu_ci'][0]*100:+.2f},{r['mu_ci'][1]*100:+.2f}] p(<=0)={r['p_mu']:.3f}")
S1 = BT["S1 VRP mean-var (primary)"]
print("\nS1 weight distribution:", S1["w"].describe().round(3).to_dict())
print("S1 months with w<1:", (S1["w"] < 0.999).sum(), "of", len(S1), "; months w=0:", (S1["w"] < 1e-9).sum())
low = S1[S1["w"] < 0.999]
print("S1 underweight months (weight, SPY return that month):")
print(pd.DataFrame({"w": low["w"], "spy_ret": risky.reindex(low.index)}).round(3).to_string())
for k, v in BT.items(): v.to_csv(f"bt_{k.split()[0]}.csv")
