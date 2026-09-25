"""DESIGN period only (1929-01 .. 1989-12): choose direction/q for each binary rule, freeze parameters."""
import json

import numpy as np
import pandas as pd

from common_setup import DESIGN, F, M, ME, SIGMA_STAR, all_targets, md
from engine import INDS, fmt_table, metrics, simulate

COST = 2.0
s, e = DESIGN
r_m, rf = md["mkt"], md["rf"]
print(f"sigma* (annualised std of daily market total return 1927-1989) = {SIGMA_STAR:.4f}")

bh = simulate(pd.Series(1.0, index=ME), r_m, rf, s, e)
bh_m, _ = metrics(bh, 0)
print(f"design BH: SR {bh_m['SR']:+.3f}  CAGR {bh_m['CAGR']:.2%}  MDD {bh_m['MDD']:.1%}")

T = all_targets(None)
rows, sims = [], {}
for name, tgt in T.items():
    sim = simulate(tgt, r_m, rf, s, e)
    sims[name] = sim
    m, _ = metrics(sim, COST, bh_m["SR"])
    rows.append(dict(rule=name, **m))
D = pd.DataFrame(rows).set_index("rule")
pd.set_option("display.width", 250)
print("\n== Design period 1929-1989, 2 bp costs, monthly rebalance: ALL variants (42 binary grid + baselines)")
print(fmt_table(rows))

frozen = {}
print("\n== Frozen choice per binary indicator (max design net Sharpe over direction x q)")
for x in INDS:
    sub = D.loc[[i for i in D.index if i.startswith(f"B-{x}|")], "SR"]
    best = sub.idxmax()
    _, d, q = best.split("|")
    frozen[x] = {"direction": d, "q": float(q), "design_SR": float(sub.max())}
    print(f"{x:<8} -> {d:<4} q={q:<5} design SR {sub.max():+.3f} (BH {bh_m['SR']:+.3f});"
          f" all: " + ", ".join(f"{i.split('|',1)[1]}={v:+.2f}" for i, v in sub.items()))
frozen["_meta"] = {"sigma_star": SIGMA_STAR, "cap": 1.0, "cost_bp": COST, "rebalance": "monthly",
                   "design": DESIGN}
with open("frozen_params.json", "w") as f:
    json.dump(frozen, f, indent=2)
print("\nwrote frozen_params.json")

# timing diagnostic (SPEC 'Timing check'): same-day look-ahead version of B-RV frozen rule
tgt = T[f"B-RV|{frozen['RV']['direction']}|{frozen['RV']['q']}"]
idx = md.index
pos = idx.get_indexer(tgt.index)
cheat = pd.Series(tgt.values, index=idx[np.maximum(pos - 1, 0)])       # weight known at close d applied ON day d
mc, _ = metrics(simulate(cheat, r_m, rf, s, e), COST)
mh, _ = metrics(sims[f"B-RV|{frozen['RV']['direction']}|{frozen['RV']['q']}"], COST)
print(f"timing diagnostic, B-RV frozen, design: correct lag SR {mh['SR']:+.3f} vs look-ahead (1 day early) SR {mc['SR']:+.3f}")

# Design-period in-sample information: share of time out, per frozen binary rule
for x in INDS:
    k = f"B-{x}|{frozen[x]['direction']}|{frozen[x]['q']}"
    print(f"{k:<22} avg weight in design {sims[k]['w'].mean():.2f}")
