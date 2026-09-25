"""Robustness (pre-registered in SPEC.md; never used to choose parameters). Industries.
1. every k in {3,5,10} on the test period; 2. terciles instead of quintiles; 3. cost sensitivity 0/5/10/20 bp;
4. placebo: NET1/NET12 with k*=5 RANDOM neighbours (200 draws, seed 0) vs the real correlation graph."""
import json
import os
import pickle
import time
from multiprocessing import Pool

import numpy as np
import pandas as pd

import data
import engine as E

P = json.load(open("frozen_params.json"))
K = P["k"]
d, m = data.industries()
d, m = d.loc[:"2024-12-31"], m.loc[:"2024-12-31"]
sigs = pickle.load(open("cache/ind_signals.pkl", "rb"))["sigs"]
TEST = ("1989-12-31", "2024-11-30")
DES = ("1927-06-30", "1989-11-30")
S = {key: v.loc[TEST[0]:TEST[1]] for key, v in sigs.items()}
lab = lambda key: key if isinstance(key, str) else f"{key[0]}(k={key[1]})"

if __name__ == "__main__":
    print("=== 1. all k, test period 1990-2024, quintile L/S, 10 bp ===")
    for k in [3, 5, 10]:
        for name in ["NET1", "NET12", "LAPGAP1", "NET1_PURE"]:
            bt = E.backtest(E.rank_weights(S[(name, k)], frac=0.2), m, 0.001)
            print(E.summarize(bt, label=f"L/S {lab((name, k))}", boot=False)[0])
    print("  FM with each k (test, NW t):")
    for k in [3, 5, 10]:
        fm = E.fama_macbeth(S, m, ["OWN1", "OWN12", ("NET1", k), ("NET12", k)])
        print(f"   k={k}: " + "  ".join(f"{c}={100*E.nw_mean(fm[c])[0]:+.3f}(t={E.nw_mean(fm[c])[1]:+.2f})" for c in fm))

    print("\n=== 2. terciles (1/3) instead of quintiles, test, 10 bp ===")
    for key in ["OWN1", "OWN12", ("NET1", K), ("NET12", K), ("NET1_PURE", K)]:
        bt = E.backtest(E.rank_weights(S[key], frac=1 / 3), m, 0.001)
        print(E.summarize(bt, label=f"L/S tercile {lab(key)}", boot=False)[0])

    print("\n=== 3. cost sensitivity, test, quintile L/S: net ann. mean (Sharpe) ===")
    for key in ["OWN12", ("NET1", K), ("NET12", K), ("NET1_PURE", K)]:
        W = E.rank_weights(S[key], frac=0.2)
        cells = []
        for c in [0, 0.0005, 0.001, 0.002]:
            bt = E.backtest(W, m, c)
            cells.append(f"{c*1e4:.0f}bp: {12*bt['net'].mean():+.2%} ({E.sharpe(bt['net']):+.2f})")
        print(f"  {lab(key):<16} " + " | ".join(cells))


def one_draw(seed):
    rng = np.random.default_rng(seed)
    s = E.build_signals(d, m, [K], m.index[m.index <= "2024-11-30"], placebo_rng=rng)
    out = {}
    for name in ["NET1", "NET12"]:
        bt = E.backtest(E.rank_weights(s[(name, K)], frac=0.2), m, 0.001)
        for per, (a, b) in {"design": ("1927-07", "1989-12"), "test": ("1990-01", "2024-12")}.items():
            out[(name, per)] = E.sharpe(bt["gross"].loc[a:b])
        # FM slope of placebo NET beyond own signals, test period
    return out


if __name__ == "__main__":
    print("\n=== 4. placebo: k=%d random neighbours, 200 draws (seeds 0..199), gross L/S Sharpe ===" % K)
    t0 = time.time()
    with Pool(min(8, os.cpu_count() or 1)) as pool:
        draws = pool.map(one_draw, range(200))
    print(f"  ({time.time()-t0:.0f}s, {os.cpu_count()} cpus)")
    for name in ["NET1", "NET12"]:
        bt = E.backtest(E.rank_weights(sigs[(name, K)], frac=0.2), m, 0.001)
        for per, (a, b) in {"design": ("1927-07", "1989-12"), "test": ("1990-01", "2024-12")}.items():
            real = E.sharpe(bt["gross"].loc[a:b])
            pl = np.array([x[(name, per)] for x in draws])
            print(f"  {name:<6} {per:<6}: real-graph gross SR {real:+.2f} | random-neighbour SR mean {pl.mean():+.2f} "
                  f"sd {pl.std():.2f} 95th pct {np.percentile(pl, 95):+.2f} | share of placebos >= real {np.mean(pl >= real):.3f}")
