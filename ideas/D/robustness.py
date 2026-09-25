"""Pre-registered robustness R1-R6 (SPEC.md) + one POST-HOC diagnostic (labelled)."""
import json, numpy as np, pandas as pd
from vrp_common import *
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
FP = json.load(open("frozen_params.json"))
P = build_panel(); MKTRF_FULL = french_monthly()["Mkt-RF"]
risky, rf = P["spy_ret"], P["rf"]

def fc(pred, h, last):
    return oos_forecasts(P[pred], P["mkt_rf"], h, "1999-12-31", last)
F = {("VRP", 1): fc("VRP", 1, "2026-06-30"), ("VRP", 3): fc("VRP", 3, "2026-04-30"),
     ("VRP21", 1): fc("VRP21", 1, "2026-06-30"), ("VRP21", 3): fc("VRP21", 3, "2026-04-30"),
     ("IV", 1): fc("IV", 1, "2026-06-30"), ("IV", 3): fc("IV", 3, "2026-04-30")}

print("== OOS R2 / Clark-West by predictor, horizon, sub-period, restriction (origins; test period only)")
subs = {"full 1999-12..": ("1999-12-31", "2026-12-31"), "R1a 1999-12..2012-11": ("1999-12-31", "2012-11-30"),
        "R1b 2012-12..": ("2012-12-31", "2026-12-31"), "R4 ex 2008-08..2009-05 origins": None}
for (pred, h), Fr in F.items():
    for sname, rng in subs.items():
        if rng is None:
            G = Fr.drop(Fr.loc["2008-08-31":"2009-05-31"].index)
        else:
            G = Fr.loc[rng[0]:rng[1]]
        for restrict in (False, True):
            r2, t, p, n = clark_west(G, h, restrict)
            print(f"  {pred:5s} h={h} {sname:32s} {'restr' if restrict else 'unres'} n={n:3d} R2={r2*100:+6.2f}% CW t={t:+.2f} p={p:.3f}")

def weights(pred="VRP"):
    Fr = F[(pred, 1)]
    return {"B1": pd.Series(1.0, index=P.index),
            "S1": mv_weights(Fr["f_model"], MKTRF_FULL, FP["gamma"], FP["var_window"]),
            "S0": mv_weights(Fr["f_mean"], MKTRF_FULL, FP["gamma"], FP["var_window"]),
            "S2": expanding_quantile_rule(P[pred], FP["q"]),
            "B2": sma_rule(P["spy_px"].dropna(), FP["sma"]),
            "B3": voltarget(P["RV"], FP["voltarget"])}

def compare(W, cost, S, E, drop=None, label=""):
    BT = {k: backtest(w, risky, rf, cost, S, E) for k, w in W.items()}
    if drop is not None:
        BT = {k: v.drop(v.loc[drop[0]:drop[1]].index) for k, v in BT.items()}
    print(f"\n-- {label}: cost {cost} bp, returns {S}..{E}" + (f", excluding {drop}" if drop else ""))
    for k in ["S1", "S2", "B2", "B3", "S0"]:
        s = summarize(k, BT[k], cost); b = summarize("B1", BT["B1"], cost)
        r = paired_bootstrap(BT[k]["ex"].values, BT["B1"]["ex"].values, reps=5000)
        print(f"  {k} Sharpe {s['sharpe']:+.3f} (B1 {b['sharpe']:+.3f}) dSR={r['d_sr']:+.3f} CI[{r['sr_ci'][0]:+.3f},{r['sr_ci'][1]:+.3f}] p={r['p_sr']:.3f}"
              f" | dMean={r['d_mu']*100:+.2f}%/yr p={r['p_mu']:.3f} | maxDD {s['maxdd']:.3f} vs {b['maxdd']:.3f}")
    r = paired_bootstrap(BT["S1"]["ex"].values, BT["S0"]["ex"].values, reps=5000)
    print(f"  S1-S0 dSR={r['d_sr']:+.3f} CI[{r['sr_ci'][0]:+.3f},{r['sr_ci'][1]:+.3f}] p={r['p_sr']:.3f}")

print("\n== Strategy robustness (bootstrap 5000 reps, block 6, seed %d)" % SEED)
W = weights("VRP")
compare(W, 2, "2000-01", "2012-12", label="R1a sub-period")
compare(W, 2, "2013-01", "2026-07", label="R1b sub-period")
compare(W, 10, "2000-01", "2026-07", label="R3 cost 10bp")
compare(W, 2, "2000-01", "2026-07", drop=("2008-09", "2009-06"), label="R4 ex-GFC months")
compare(weights("VRP21"), 2, "2000-01", "2026-07", label="R2 VRP with SPY 21d RV (S1,S2 use VRP21)")

print("\n== POST-HOC diagnostic (not pre-registered): where does the h=3 CT-restricted OOS gain come from?")
G = F[("VRP", 3)].dropna(subset=["y"])
fm = G["f_model"].clip(lower=0)
d = (G["y"] - G["f_mean"]) ** 2 - (G["y"] - fm) ** 2      # >0 = VRP model better
tot = d.sum()
by_year = d.groupby(d.index.year).sum() / ((G["y"] - G["f_mean"]) ** 2).sum() * 100
print("Contribution to OOS R2 (percentage points) by origin year:"); print(by_year.round(2).to_string())
top = d.sort_values(ascending=False).head(8) / ((G["y"] - G["f_mean"]) ** 2).sum() * 100
print("Top 8 origin months by contribution (pp):"); print(top.round(2).to_string())
print(f"OOS R2 without top 8 origins: {(1 - ((G['y']-fm)**2).drop(top.index).sum()/((G['y']-G['f_mean'])**2).drop(top.index).sum())*100:+.2f}%")
