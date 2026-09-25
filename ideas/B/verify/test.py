"""Step 3: the single test-period evaluation (2015-2024) with signs frozen in frozen_signs.json."""
import json

import numpy as np
import pandas as pd
from statsmodels.stats.multitest import multipletests

from lib import (D, DESIGN, TEST, backtest, build_signals, factors, ff_alpha, fmt, nw_t, sr, sr_diff_ci,
                 summarize)
from data_load import universe

signs = json.load(open(D / "frozen_signs.json"))
R, spy, sectors, _ = universe()
sig, info = build_signals(R, sectors, "full")
print("graph stats (full sample):", {k: round(v, 4) for k, v in info.items()})

# causality check: signals built on data truncated at 2014 must equal the full-data signals on <=2014 dates
sig_d, _ = build_signals(R.loc[:DESIGN[1]], sectors, "design")
worst = max(float((sig[k][0].loc[:DESIGN[1]] - sig_d[k][0]).abs().max().max()) for k in sig)
print(f"max |full-sample signal - design-truncated signal| on design dates, all 14 signals: {worst:.2e}")

F = factors()
rf = F["RF"]
NQ = 16
rows, bts = [], {}
for name, (s, freq) in sig.items():
    bt = backtest(R, s, signs[name], NQ).loc[TEST[0]:TEST[1]]
    bts[name] = bt
    a = ff_alpha(bt["net"], F)
    rows.append({"signal": name, "freq": freq, "sign": signs[name], **summarize(bt),
                 "FF6_alpha": a["alpha_ann"], "t_alpha": a["t_alpha"], "b_mkt": a["beta_mkt"], "b_umd": a["beta_umd"]})
tab = pd.DataFrame(rows).set_index("signal")
rej, p_holm, _, _ = multipletests(tab["p_net"], alpha=0.05, method="holm")
tab["p_holm"] = p_holm
tab["survivor"] = (tab["ann_net"] > 0) & rej
print("\n== TEST period 2015-2024, quintile long-short (16/16 of 83), 5 bp one-way, frozen design signs")
print(fmt(tab.drop(columns=["start"])))
print("survivors (net mean > 0, Holm p < 0.05):", list(tab.index[tab["survivor"]]))
tab.to_csv(D / "test_longshort.csv")

# gross (pre-cost) FF6 alpha as well, for the horizon map
print("\n== TEST gross (pre-cost) FF5+UMD alpha")
for name, bt in bts.items():
    a = ff_alpha(bt["gross"], F)
    print(f"{name:<8} gross alpha {a['alpha_ann']:+.3f}/yr  t={a['t_alpha']:+.2f}   gross mean t={nw_t(bt['gross'])[0]:+.2f}")

# long-only variant vs equal-weight benchmark on the same schedule, same costs
lo_rows = []
for name, (s, freq) in sig.items():
    lo = backtest(R, s, signs[name], NQ, long_only=True).loc[TEST[0]:TEST[1]]
    ew = backtest(R, s, 1, NQ, ew=True).loc[TEST[0]:TEST[1]]
    a, b = summarize(lo, rf), summarize(ew, rf)
    diff = lo["net"] - ew["net"]
    t, p = nw_t(diff)
    x = rf.reindex(lo.index).fillna(0)
    (clo, chi), pboot = sr_diff_ci(lo["net"] - x, ew["net"] - x)
    al = ff_alpha(lo["net"], F, rf_sub=True)
    lo_rows.append({"signal": name, "sign": signs[name], "LO_ann_net": a["ann_net"], "LO_SR_xRF": a["SR_net"],
                    "EW_ann_net": b["ann_net"], "EW_SR_xRF": b["SR_net"], "excess_ann": diff.mean() * 252,
                    "t_excess": t, "p_excess": p, "dSR_CI": f"[{clo:+.2f},{chi:+.2f}]", "LO_turn_yr": a["turn_yr"],
                    "LO_breakeven_vs_EW_bp": (lo["gross"] - ew["gross"]).mean() / (lo["turnover"] - ew["turnover"]).mean() * 1e4,
                    "LO_FF6_alpha": al["alpha_ann"], "t_alpha": al["t_alpha"]})
lot = pd.DataFrame(lo_rows).set_index("signal")
lot["p_holm"] = multipletests(lot["p_excess"], method="holm")[1]
print("\n== TEST long-only top quintile vs EW-83 (same rebalance dates, 5 bp); Sharpe net of RF; excess = LO - EW daily net")
print(fmt(lot))
lot.to_csv(D / "test_longonly.csv")

s = spy.loc[TEST[0]:TEST[1]]
print(f"\nSPY buy-and-hold 2015-2024 (no cost; one 2 bp entry is negligible): ann mean {s.mean()*252:+.3f}, SR xRF {sr(s - rf.reindex(s.index).fillna(0)):+.3f}")
