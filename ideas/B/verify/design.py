"""Step 2: design period only (returns truncated at 2014-12-31 BEFORE any signal is computed).
Runs both signs for all 14 pre-registered signals, picks the sign by design gross mean, and writes frozen_signs.json."""
import json

import pandas as pd

from lib import D, DESIGN, backtest, build_signals, check_cube, fmt, summarize, factors
from data_load import universe

R_full, spy, sectors, _ = universe()
R = R_full.loc[:DESIGN[1]]          # nothing after 2014 is visible to this script
print(f"design data {R.index[0].date()}..{R.index[-1].date()}  {R.shape}")
print(f"max |cube - fractional_matrix_power| on 3 test days: {check_cube(R):.2e}")
sig, info = build_signals(R, sectors, "design")
print("graph stats (design):", {k: round(v, 4) for k, v in info.items()})
rf = factors()["RF"]

NQ = 16
rows, signs = [], {}
for name, (s, freq) in sig.items():
    res = {}
    for sign in (+1, -1):
        bt = backtest(R, s, sign, NQ).loc[DESIGN[0]:DESIGN[1]]
        res[sign] = bt
        rows.append({"signal": name, "freq": freq, "sign": sign, **summarize(bt)})
    signs[name] = +1 if res[+1]["gross"].mean() > 0 else -1

tab = pd.DataFrame(rows).set_index(["signal", "sign"])
print("\n== DESIGN period 2005-2014, quintile long-short (16/16 of 83), 5 bp one-way; sign +1 = continuation, -1 = reversal")
print(fmt(tab))
print("\nfrozen signs (by design gross mean):", signs)
json.dump(signs, open(D / "frozen_signs.json", "w"), indent=1)

lo_rows = []
for name, (s, freq) in sig.items():
    bt = backtest(R, s, signs[name], NQ, long_only=True).loc[DESIGN[0]:DESIGN[1]]
    ew = backtest(R, s, 1, NQ, ew=True).loc[bt.index[0]:DESIGN[1]]
    a, b = summarize(bt, rf), summarize(ew, rf)
    lo_rows.append({"signal": name, "sign": signs[name], "LO_ann_net": a["ann_net"], "LO_SR_net_xRF": a["SR_net"],
                    "EW_ann_net": b["ann_net"], "EW_SR_net_xRF": b["SR_net"], "LO_turn_yr": a["turn_yr"]})
print("\n== DESIGN long-only top quintile vs equal-weight 83 (same schedule, 5 bp), Sharpe net of RF")
print(fmt(pd.DataFrame(lo_rows).set_index("signal")))
