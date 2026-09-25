"""Adversarial verification checks for Concept B (verifier's own script; builder files untouched).

V1 data: fresh Yahoo download vs the builder's cached prices; forward-fill count; large moves; stale runs.
V2 independent engine (dollar positions + NAV, written from scratch) vs lib.backtest on headline strategies.
V3 Newey-West lag sensitivity for the only positive headline numbers.
V4 placebo for the A3 "top-5 contributors" claim: how much does removing the top-5 leave-one-out contributors
   lower the long-only excess of RANDOM monthly stock picks? (seed 7, 200 placebos)
V5 multiple-testing haircut (Bonferroni over the builder's 70 variants) for the two t~1.85-2.05 numbers.
V6 practical-recipe claim: how many long-only variants beat SPY's Sharpe; SPY Sharpe recomputed.
V7 A1 industries: how often a held industry has a missing return that the builder filled with 0.
"""
import json
import time

import numpy as np
import pandas as pd

from data_load import CACHE, D, french, universe
from lib import TEST, backtest, build_signals, factors, nw_t, rebal_dates, sr, target_weights

t0 = time.time()
R, spy, sectors, _ = universe()
signs = json.load(open(D / "frozen_signs.json"))
sig, _ = build_signals(R, sectors, "full")
rf = factors()["RF"]
T0, T1 = TEST

# ------------------------------------------------------------------ V1
print("== V1 data checks")
p = CACHE / "px_fresh_verify.csv"
if not p.exists():
    import yfinance as yf
    px = yf.download(list(R.columns) + ["SPY"], start="2004-01-01", end="2025-01-01", auto_adjust=True,
                     progress=False)["Close"]
    px.to_csv(p)
fresh = pd.read_csv(p, index_col=0, parse_dates=True)
fr = fresh[list(R.columns)].loc["2004-12-31":"2024-12-31"].ffill(limit=5).pct_change().reindex(R.index)
fr.loc["2016-07-05", "DHR"] = 0.0
d = (fr - R).abs()
print(f"fresh Yahoo vs builder returns: max |diff| {d.max().max():.2e}; cells with |diff|>1e-6: {int((d > 1e-6).sum().sum())}"
      f" of {d.size}; worst {d.max().idxmax()} ")
fs = fresh["SPY"].loc["2004-12-31":"2024-12-31"].pct_change().reindex(R.index)
print(f"fresh SPY vs builder SPY: max |diff| {(fs - spy).abs().max():.2e}")
raw = pd.read_csv(CACHE / "px_sp100_2004_2024.csv", index_col=0, parse_dates=True)[list(R.columns)].loc["2005-01-03":"2024-12-31"]
print(f"forward-filled price cells in the 83-stock panel: {int(raw.isna().sum().sum())}")
big = [(R.index[i].date(), R.columns[j], round(R.iat[i, j], 3)) for i, j in zip(*np.where(R.abs().values > 0.2))]
print(f"daily |r|>20%: {len(big)} events: {big}")
z = (R == 0)
runs = {c: int(z[c].astype(int).groupby((~z[c]).cumsum()).sum().max()) for c in R.columns}
print("longest run of exactly-zero daily returns (top 5):", sorted(runs.items(), key=lambda kv: -kv[1])[:5])
print(f"DHR raw adjusted-close move on 2016-07-05 (builder set to 0): {raw['DHR'].pct_change().loc['2016-07-05']:+.3f}")


# ------------------------------------------------------------------ V2 independent engine
def indep(Rdf, s, sign, nq, cost_bp=5.0, long_only=False):
    """Dollar holdings h and NAV V. Targets decided at close t from s.loc[t]; P&L from day t+1."""
    X = Rdf.values
    pos = {dt: i for i, dt in enumerate(Rdf.index)}
    tgt = {}
    for dt, row in s.iterrows():
        v = (sign * row).dropna()
        if len(v) < 2 * nq:
            continue
        r = v.rank(method="first")
        w = np.zeros(X.shape[1])
        cols = [Rdf.columns.get_loc(c) for c in v.index]
        rr = r.values
        n = len(v)
        for k, c in enumerate(cols):
            if rr[k] > n - nq:
                w[c] = 1 / nq
            elif rr[k] <= nq and not long_only:
                w[c] = -1 / nq
        tgt[pos[dt]] = w
    first = min(tgt)
    V, h = 1.0, np.zeros(X.shape[1])
    g, to = [], []
    for i in range(first, len(X)):
        pnl = h @ X[i]
        ret = pnl / V
        h = h * (1 + X[i])
        V = V + pnl
        tv = 0.0
        if i in tgt:
            newh = V * tgt[i]
            tv = np.abs(newh - h).sum() / V
            h = newh
        g.append(ret); to.append(tv)
    out = pd.DataFrame({"gross": g, "turnover": to}, index=Rdf.index[first:])
    out["net"] = out["gross"] - out["turnover"] * cost_bp / 1e4
    return out


print("\n== V2 independent engine vs lib.backtest, TEST 2015-2024, 5 bp")
for name, lo in (("LR1", False), ("LR5", False), ("LR12_1", False), ("MOM12_1", False), ("REV5", False),
                 ("LR12_1", True)):
    s = sig[name][0]
    a = backtest(R, s, signs[name], 16, long_only=lo).loc[T0:T1]
    b = indep(R, s, signs[name], 16, long_only=lo).loc[T0:T1]
    x = rf.reindex(b.index).fillna(0) if lo else 0
    print(f"{name:<7} LO={lo!s:<5} builder SRg {sr(a['gross'] - x):+.3f} SRn {sr(a['net'] - x):+.3f} turn {a['turnover'].mean()*252:7.2f} | "
          f"independent SRg {sr(b['gross'] - x):+.3f} SRn {sr(b['net'] - x):+.3f} turn {b['turnover'].mean()*252:7.2f} "
          f"BE {b['gross'].mean()/b['turnover'].mean()*1e4:+.2f} bp | max|daily diff| {(a['net']-b['net']).abs().max():.1e}")

# ------------------------------------------------------------------ V3 NW lag sensitivity
print("\n== V3 Newey-West lag sensitivity (TEST)")
lr_ls = backtest(R, sig["LR12_1"][0], 1, 16).loc[T0:T1]["net"]
mo_ls = backtest(R, sig["MOM12_1"][0], 1, 16).loc[T0:T1]["net"]
lo = backtest(R, sig["LR12_1"][0], 1, 16, long_only=True).loc[T0:T1]["net"]
ew = backtest(R, sig["LR12_1"][0], 1, 16, ew=True).loc[T0:T1]["net"]
import statsmodels.api as sm
for L in (0, 5, 10, 21, 63):
    m = sm.OLS(lr_ls, sm.add_constant(mo_ls)).fit(cov_type="HAC", cov_kwds={"maxlags": L})
    print(f"lags {L:>2}: LR12_1 LS net t {nw_t(lr_ls, L)[0]:+.2f} | LO excess over EW t {nw_t(lo - ew, L)[0]:+.2f} | "
          f"A2 intercept vs MOM t {m.tvalues['const']:+.2f}")
# monthly (non-overlapping) version of the A2 regression and LO excess
mon = lambda x: np.expm1(np.log1p(x).groupby(x.index.to_period("M")).sum())
mlr, mmo, mex = mon(lr_ls), mon(mo_ls), mon(lo) - mon(ew)
m = sm.OLS(mlr, sm.add_constant(mmo)).fit(cov_type="HAC", cov_kwds={"maxlags": 3})
print(f"monthly returns (n={len(mlr)}): A2 intercept {m.params['const']*12:+.3f}/yr t {m.tvalues['const']:+.2f}; "
      f"LO excess {mex.mean()*12:+.3f}/yr t {nw_t(mex, 3)[0]:+.2f}")

# ------------------------------------------------------------------ V4 placebo for A3
print("\n== V4 placebo: effect of removing the top-5 leave-one-out contributors on long-only excess (monthly, 16 names)")
X = R.values
N = X.shape[1]
dM = rebal_dates(R.index, "M")
pos = {dt: i for i, dt in enumerate(R.index)}
start_i = pos[sig["LR12_1"][0].dropna(how="all").index[0]]
rb = [pos[dt] for dt in dM if pos[dt] >= start_i]
ti0, ti1 = R.index.get_indexer([R.loc[T0:].index[0]])[0], R.index.get_indexer([R.loc[:T1].index[-1]])[0]


def fast_excess(scores, keep, nq=16, cost=5e-4):
    """scores: (n_rebal, N) array; keep: bool mask of stocks in universe. Returns mean annual LO-minus-EW net excess
    over TEST. Same drift / cost convention as lib.backtest (cost booked on the rebalance day)."""
    idx = np.where(keep)[0]
    Xk = X[:, idx]
    n = len(idx)
    res = []
    for kind in ("lo", "ew"):
        net = np.zeros(len(X))
        w = np.zeros(n)
        for k, i in enumerate(rb):
            if kind == "lo":
                sc = scores[k, idx]
                top = np.argsort(sc, kind="mergesort")[-nq:]
                tw = np.zeros(n); tw[top] = 1 / nq
            else:
                tw = np.full(n, 1 / n)
            net[i] -= np.abs(tw - w).sum() * cost
            j1 = rb[k + 1] if k + 1 < len(rb) else len(X) - 1
            G = np.cumprod(1 + Xk[i + 1:j1 + 1], axis=0)
            V = G @ tw
            prev = np.concatenate([[1.0], V[:-1]])
            net[i + 1:j1 + 1] += V / prev - 1
            w = tw * G[-1] / V[-1] if len(V) else tw
        res.append(net[ti0:ti1 + 1])
    return (res[0] - res[1]).mean() * 252


def top5_drop(scores):
    full_keep = np.ones(N, bool)
    full = fast_excess(scores, full_keep)
    contrib = []
    for c in range(N):
        kp = full_keep.copy(); kp[c] = False
        contrib.append(full - fast_excess(scores, kp))
    top = np.argsort(contrib)[-5:]
    kp = full_keep.copy(); kp[top] = False
    return full, fast_excess(scores, kp), [R.columns[t] for t in top[::-1]]


S = sig["LR12_1"][0].reindex(dM[dM >= R.index[start_i]]).values
full, red, top = top5_drop(S)
print(f"fast engine on LR12_1: full excess {full:+.4f}/yr (builder A3: +0.057), without top-5 {red:+.4f} (builder: -0.002), top5 {top}")
rng = np.random.default_rng(7)
out = []
t1 = time.time()
for b in range(200):
    sc = rng.random((len(rb), N))
    f, r_, _ = top5_drop(sc)
    out.append((f, r_))
out = np.array(out)
drop = out[:, 0] - out[:, 1]
print(f"200 random-pick placebos ({time.time()-t1:.0f}s): full excess mean {out[:,0].mean():+.4f} sd {out[:,0].std():.4f}; "
      f"after removing own top-5 contributors mean {out[:,1].mean():+.4f}; drop mean {drop.mean():+.4f}, "
      f"5-95% [{np.percentile(drop,5):+.4f},{np.percentile(drop,95):+.4f}]")
print(f"LR12_1 drop {full-red:+.4f}; share of placebos with drop >= LR12_1's: {(drop >= full-red).mean():.3f}; "
      f"share of placebos with full excess >= LR12_1's {full:+.4f}: {(out[:,0] >= full).mean():.3f}")

# ------------------------------------------------------------------ V5 multiple testing
print("\n== V5 Bonferroni over the builder's 70 counted variants")
for lab, pval in (("LR12_1 LO excess (p=0.064)", 0.064), ("A2 intercept t=2.05", 2 * (1 - __import__('scipy').stats.norm.cdf(2.05))),
                  ("LR12_1 LS net (p=0.228)", 0.228)):
    print(f"{lab}: raw p {pval:.3f} -> Bonferroni(70) {min(1, pval*70):.2f}")

# ------------------------------------------------------------------ V6 practical recipe claim
print("\n== V6 long-only Sharpe (xRF) vs SPY, TEST")
lot = pd.read_csv(D / "test_longonly.csv", index_col=0)
s = spy.loc[T0:T1]
spysr = sr(s - rf.reindex(s.index).fillna(0))
print(f"SPY SR xRF {spysr:+.3f}; long-only variants with SR > SPY: {int((lot['LO_SR_xRF'] > spysr).sum())} of {len(lot)} "
      f"({', '.join(lot.index[lot['LO_SR_xRF'] > spysr])}); EW-83 SR {lot['EW_SR_xRF'].iloc[0]:+.3f}")
lsx = pd.read_csv(D / "test_longshort.csv", index_col=0)
print(f"long-short net SR range {lsx['SR_net'].min():+.2f}..{lsx['SR_net'].max():+.2f}")
ewd = R.mean(axis=1).loc[T0:T1]
print(f"EW-83 daily-rebalanced, no cost, SR xRF {sr(ewd - rf.reindex(ewd.index).fillna(0)):+.3f}")

# ------------------------------------------------------------------ V7 A1 industries missing-as-zero
print("\n== V7 A1: held industries with missing returns (filled with 0 by robust.py)")
ind = french("49_Industry_Portfolios_daily_CSV.zip").loc[:"2024-12-31"]
from lib import logcum
mom = logcum(ind, 21, 231).loc[rebal_dates(ind.index, "M")]
bt_days = 0
for k, dt in enumerate(mom.index[:-1]):
    v = mom.loc[dt].dropna()
    if len(v) < 20:
        continue
    held = list(v.sort_values().index[-10:]) + list(v.sort_values().index[:10])
    seg = ind.loc[dt:mom.index[k + 1], held].iloc[1:]
    bt_days += int(seg.isna().any(axis=1).sum())
print(f"days on which a MOM12_1 industry position had a missing return: {bt_days}")
print(f"\nruntime {time.time()-t0:.0f}s")
