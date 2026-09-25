"""TEST period (1990-01 .. 2024-12), run once with frozen_params.json. Also used by robust.py via run_period()."""
import json
import sys

import numpy as np
import pandas as pd

from common_setup import ME, TEST, WK, all_targets, md
from engine import INDS, NET_INDS, boot_diff, fmt_table, jkm, metrics, simulate

FROZEN = json.load(open("frozen_params.json"))
UNDER_TEST = [f"B-{x}" for x in INDS] + [f"VT-RV+{x}" for x in NET_INDS]      # 13 rules for Holm
BASE = ["TREND10", "VT-naive", "VT-RV", "VT-RV x TREND10"]


def holm(pvals):
    names = list(pvals)
    order = np.argsort([pvals[n] for n in names])
    m, adj, running = len(names), {}, 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * pvals[names[i]]))
        adj[names[i]] = running
    return adj


def run_period(period, cost=2.0, cap=1.0, freq="M", borrow=0.0, label=""):
    s, e = period
    r_m, rf = md["mkt"], md["rf"]
    dates = ME if freq == "M" else WK
    T = all_targets(FROZEN, cap=cap, freq=freq)
    bh = simulate(pd.Series(1.0, index=dates), r_m, rf, s, e)
    mb, exb = metrics(bh, 0)
    rows, ex, sims = [dict(rule="BH", **mb)], {"BH": exb}, {"BH": bh}
    for name in BASE + UNDER_TEST:
        sim = simulate(T[name], r_m, rf, s, e, borrow_spread=borrow)
        m, x = metrics(sim, cost, mb["SR"])
        d, ci, p = boot_diff(x, exb)
        z, pj = jkm(x, exb)
        rows.append(dict(rule=name, **m, dSR_vs_BH=d, dSR_lo=ci[0], dSR_hi=ci[1], p_boot=p, p_JKM=pj))
        ex[name], sims[name] = x, sim
    adj = holm({r["rule"]: r["p_boot"] for r in rows if r["rule"] in UNDER_TEST})
    for r in rows:
        r["p_holm"] = adj.get(r["rule"], np.nan)
    rows = sorted(rows, key=lambda r: -r["SR"])
    print(f"\n== {label} {s}..{e}, cost {cost} bp, cap {cap}, rebalance {freq}, borrow spread {borrow:.2%}"
          f" (ranked by net excess Sharpe; dSR/p vs BH: paired 12m-block bootstrap and JK-Memmel; Holm over 13 rules)")
    print(fmt_table(rows))
    return rows, ex, sims


def incremental(ex, pairs, label=""):
    print(f"\n== Incremental tests {label} (Sharpe difference A - B, paired block bootstrap / JK-Memmel)")
    for a, b in pairs:
        d, ci, p = boot_diff(ex[a], ex[b])
        z, pj = jkm(ex[a], ex[b])
        print(f"{a:<18} vs {b:<16} dSR {d:+.3f}  95% CI [{ci[0]:+.2f}, {ci[1]:+.2f}]  p_boot {p:.3f}  JKM z {z:+.2f} p {pj:.3f}")


PAIRS = ([(f"VT-RV+{x}", "VT-RV") for x in NET_INDS] + [(f"B-{x}", "B-RV") for x in NET_INDS]
         + [(f"B-{x}", "TREND10") for x in NET_INDS] + [("VT-RV", "VT-naive"), ("B-RV", "TREND10"),
                                                        ("VT-RV x TREND10", "TREND10")])

if __name__ == "__main__":
    pd.set_option("display.width", 300)
    print("frozen:", {k: v for k, v in FROZEN.items()})
    rows, ex, sims = run_period(TEST, label="MAIN TEST")
    incremental(ex, PAIRS, "(test 1990-2024, 2 bp, monthly)")
    pd.DataFrame(rows).to_csv("test_main.csv", index=False)
    # crisis behaviour of the frozen rules
    print("\n== Average market weight in crisis windows (test period, monthly rebalance)")
    wins = {"GFC 2008-09..2009-03": ("2008-09-01", "2009-03-31"), "COVID 2020-02-20..04-30": ("2020-02-20", "2020-04-30"),
            "2022 bear 01..10": ("2022-01-01", "2022-10-31"), "calm 2017": ("2017-01-01", "2017-12-31")}
    for name, sim in sims.items():
        print(f"{name:<18}" + "  ".join(f"{k}: {sim['w'].loc[a:b].mean():.2f}" for k, (a, b) in wins.items())
              + "  | mkt return in window: " + "  ".join(f"{((1+sim['rm'].loc[a:b]).prod()-1):+.1%}" for a, b in wins.values()))
