"""ETF universe, return months 2007-01..2024-12, frozen parameters (no retuning). Run once."""
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
monthly = px.resample("ME").last().pct_change(fill_method=None)
monthly[px.resample("ME").last().isna()] = np.nan
spy_m = spy_px.resample("ME").last().pct_change(fill_method=None)
ff5, ff3 = data.factors_monthly()
rf = ff3["RF"]
form = monthly.index[(monthly.index >= "2006-12-31") & (monthly.index <= "2024-11-30")]
S = E.build_signals(daily, monthly, [K], form)
nel = S["OWN1"].notna().sum(axis=1)
print(f"frozen k={K}, top N={N}, cost {COST*1e4:.0f} bp. Eligible ETFs per month: min {nel.min()} "
      f"(first {nel.index[0]:%Y-%m}: {nel.iloc[0]}) max {nel.max()}")
print("first formation month each ETF is eligible:",
      {c: f"{S['OWN1'][c].first_valid_index():%Y-%m}" for c in S["OWN1"].columns})
KEYS = ["OWN1", "OWN12", ("NET1", K), ("NET12", K), ("LAPGAP1", K), ("NET1_PURE", K)]
lab = lambda key: key if isinstance(key, str) else f"{key[0]}(k={key[1]})"
SUB = [("2007-01", "2015-12"), ("2016-01", "2024-12")]
ex = lambda bt: bt["net"] - rf.reindex(bt.index)

print("\n=== Long-only, excess of RF, 3 bp one-way (SPY 2 bp) ===")
EW = E.backtest(E.ew_weights(S["OWN1"]), monthly, COST)
IV = E.backtest(E.invvol_weights(S["VOL"]), monthly, COST)
spyW = pd.DataFrame({"SPY": 1.0}, index=form)
SPY = E.backtest(spyW, spy_m.to_frame("SPY"), P["cost_spy"])
base = {"EW all ETFs": EW, "Inverse-vol all ETFs": IV, "SPY buy&hold": SPY}
for name, bt in base.items():
    print(E.summarize(bt, rf=rf, label=name)[0])
    for a, b in SUB:
        print("   " + E.summarize(bt.loc[a:b], rf=rf, label=f"  sub {a[:4]}-{b[:4]}", boot=False)[0])
LO = {}
for key in KEYS:
    bt = E.backtest(E.rank_weights(S[key], n=N, long_only=True), monthly, COST)
    LO[key] = bt
    print(E.summarize(bt, rf=rf, label=f"LO top{N} {lab(key)}")[0])
    for a, b in SUB:
        print("   " + E.summarize(bt.loc[a:b], rf=rf, label=f"  sub {a[:4]}-{b[:4]}", boot=False)[0])

print("\nSharpe differences (net, excess of RF), paired stationary bootstrap (block 6, 2000, seed 0):")
for key in [("NET1", K), ("NET12", K), "OWN12", "OWN1"]:
    for bname, b in [("EW", EW), ("InvVol", IV), ("SPY", SPY)] + ([("OWN12 top5", LO["OWN12"])] if key != "OWN12" else []):
        dsr, ci, p = E.boot_sharpe_diff(ex(LO[key]), ex(b))
        print(f"  LO {lab(key):<12} - {bname:<10}: dSR {dsr:+.2f} CI [{ci[0]:+.2f},{ci[1]:+.2f}] p={p:.3f}")

print("\n=== L/S top5 - bottom5, self-financing (no RF), 3 bp ===")
LS = {}
for key in KEYS:
    bt = E.backtest(E.rank_weights(S[key], n=N), monthly, COST)
    LS[key] = bt
    print(E.summarize(bt, label=f"L/S {lab(key)}")[0])

print("\n=== Spanning (net): ETF NET L/S on ETF OWN12 L/S + OWN1 L/S; and on FF5+UMD ===")
X1 = pd.concat([LS["OWN12"]["net"].rename("OWN12_LS"), LS["OWN1"]["net"].rename("OWN1_LS")], axis=1)
X2 = ff5.drop(columns="RF")
for key in [("NET1", K), ("NET12", K), ("NET1_PURE", K)]:
    for xn, X in [("own-mom L/S", X1), ("FF5+UMD", X2)]:
        f = E.spanning(LS[key]["net"], X)
        ci = f.conf_int().loc["const"] * 12
        print(f"  {lab(key):<16} on {xn:<12}: alpha {12*f.params['const']:+.2%} [{ci[0]:+.1%},{ci[1]:+.1%}] "
              f"t={f.tvalues['const']:+.2f} R2={f.rsquared:.2f}")
print("\n=== Long-only NET top5 excess over EW, regressed on OWN12-top5-minus-EW (NW t) ===")
for key in [("NET1", K), ("NET12", K)]:
    y = LO[key]["net"] - EW["net"]
    x = (LO["OWN12"]["net"] - EW["net"]).rename("OWN12top5_minus_EW")
    f = E.spanning(y, x.to_frame())
    print(f"  {lab(key):<12}: alpha {12*f.params['const']:+.2%} t={f.tvalues['const']:+.2f} beta {f.params.iloc[1]:+.2f} R2={f.rsquared:.2f}")

print("\n=== Fama-MacBeth on ETFs (z-scored; %/month; NW t) ===")
fm = E.fama_macbeth(S, monthly, ["OWN1", "OWN12", ("NET1", K), ("NET12", K)])
print(f"  n={len(fm)}: " + "  ".join(f"{c}={100*E.nw_mean(fm[c])[0]:+.3f}(t={E.nw_mean(fm[c])[1]:+.2f})" for c in fm))

print("\n=== Cost sensitivity, long-only top5 NET12 / OWN12 (net excess ann. mean, Sharpe) ===")
for key in [("NET1", K), ("NET12", K), "OWN12"]:
    W = E.rank_weights(S[key], n=N, long_only=True)
    cells = []
    for c in [0, 0.0003, 0.001, 0.0025]:
        bt = E.backtest(W, monthly, c)
        cells.append(f"{c*1e4:.0f}bp: {12*ex(bt).mean():+.2%} ({E.sharpe(ex(bt)):+.2f})")
    print(f"  {lab(key):<12} " + " | ".join(cells))
print("\ncalendar-year net total returns, LO top5 NET12 vs EW vs SPY vs LO top5 OWN12:")
yr = pd.concat({"NET12": LO[("NET12", K)]["net"], "NET1": LO[("NET1", K)]["net"], "OWN12": LO["OWN12"]["net"],
                "EW": EW["net"], "SPY": SPY["net"]}, axis=1)
print(((1 + yr).groupby(yr.index.year).prod() - 1).round(3).to_string())
