"""VERIFIER: independent checks of Concept C (does not import the builder's engine.simulate / metrics).
Run from the verify/ directory (uses copies of the builder's data.py / cached raw files and rebuilt cache/*.pkl)."""
import json

import numpy as np
import pandas as pd
import statsmodels.api as sm
from ripser import ripser

from data import CACHE, french, industries_daily, market_daily, market_monthly

pd.set_option("display.width", 250)
md, mm = market_daily(), market_monthly()
F = pd.read_pickle(CACHE / "ind_daily.pkl")
M = pd.read_pickle(CACHE / "monthly.pkl")
FROZEN = json.load(open("frozen_params.json"))

# ------------------------------------------------------------------ 1. universe look-ahead: when does each industry start/end?
raw = french("49_Industry_Portfolios_daily_CSV.zip")
fv = raw.apply(lambda s: s.first_valid_index())
lv = raw.apply(lambda s: s.last_valid_index())
inner_nan = raw.apply(lambda s: s.loc[s.first_valid_index():s.last_valid_index()].isna().sum())
print("== 1. Industry universe")
print("industries whose data END before 2024-12-31:", list(lv[lv < pd.Timestamp("2024-12-31")].index))
print("industries with interior gaps:", list(inner_nan[inner_nan > 0].index))
for k in inner_nan[inner_nan > 0].index:
    s_ = raw[k].loc[raw[k].first_valid_index():]
    na = s_[s_.isna()].index
    print(f"   {k:<6} first {s_.index[0].date()}  interior-missing days {len(na):>5}  from {na.min().date()} to {na.max().date()}")
print("dropped industries' first valid dates:", {k: str(v.date()) for k, v in fv[fv > raw.index[0]].items()})
last_gap = max(raw[k].loc[raw[k].first_valid_index():].pipe(lambda s_: s_[s_.isna()].index.max()) for k in inner_nan[inner_nan > 0].index)
print(f"latest interior gap of any industry: {last_gap.date()}; no industry ends early: {(lv == raw.index[-1]).all()}")
print("=> the 'complete 1926-2024' universe is fully determined by data up to " + str(last_gap.date()) +
      ": uses future info only WITHIN the design period (drops e.g. Rubbr/Paper/Toys for pre-1946 gaps); no test-period look-ahead")

# ------------------------------------------------------------------ 2. spot-check the network indicators on random dates
print("\n== 2. Recompute AC60 / H1N / H1P / FIED on 8 random dates (seed 1) with independent code")
ind, _ = industries_daily()
rng = np.random.default_rng(1)
dates = [pd.Timestamp(t) for t in sorted(rng.choice(F.dropna().index.values, 8, replace=False))]
mx = 0.0
for d in dates:
    i = ind.index.get_loc(d)
    c = ind.iloc[i - 59:i + 1].corr().values
    ac = c[np.triu_indices(36, 1)].mean()
    dm = np.sqrt(np.maximum(2 - 2 * c, 0)); np.fill_diagonal(dm, 0)
    h = ripser(dm, distance_matrix=True, maxdim=1)["dgms"][1]
    h = h[np.isfinite(h[:, 1])]
    W = c * (c > 0.3); np.fill_diagonal(W, 0); dg = W.sum(1); dg[dg == 0] = 1
    fied = np.sort(np.linalg.eigvalsh(np.eye(36) - W / np.sqrt(np.outer(dg, dg))))[1]
    got = F.loc[d, ["AC60", "H1N", "H1P", "FIED"]].values.astype(float)
    mine = np.array([ac, len(h), (h[:, 1] - h[:, 0]).sum(), fied])
    mx = max(mx, np.abs(got - mine).max())
    print(f"{d.date()}  builder {np.round(got, 4)}  verifier {np.round(mine, 4)}")
print(f"max abs discrepancy {mx:.2e}")

# ------------------------------------------------------------------ 3. independent MONTHLY-algebra backtest
# Within a month with no rebalancing, a w/(1-w) market/T-bill portfolio earns exactly w*R_mkt + (1-w)*R_f (buy and hold).
# Turnover at the month-end = |w_new - w_old*(1+R_mkt)/(1+R_p)|.  Monthly returns compounded from the DAILY French file.
print("\n== 3. Independent monthly-algebra backtest (test 1990-01..2024-12, 2 bp)")
per = md.index.to_period("M")
Rm = (1 + md["mkt"]).groupby(per).prod() - 1
Rf = (1 + md["rf"]).groupby(per).prod() - 1
ME = pd.Series(md.index, index=md.index).groupby(per).last()           # month-end trading date per month


def exp_q(x, q):
    return x.expanding(min_periods=252).quantile(q)


def binary_w(x, d, q):
    thr = exp_q(x, q if d == "high" else 1 - q)
    off = (x > thr) if d == "high" else (x < thr)
    return (~off).astype(float).where(thr.notna())


def backtest(w_me, cost_bp, a="1990-01", b="2024-12"):
    """w_me: weight decided at close of month m (indexed by Period m). Returns monthly excess return series of months m+1."""
    w = w_me.shift(1)                                                  # weight in effect during month m+1
    months = pd.period_range(a, b, freq="M")
    w, rm, rf = w.reindex(months), Rm.reindex(months), Rf.reindex(months)
    assert w.notna().all()
    rp = w * rm + (1 - w) * rf
    drift = w * (1 + rm) / (1 + rp)
    w_next = w_me.reindex(months)                                      # new target set at the end of each month
    turn = (w_next - drift).abs()
    turn.iloc[-1] = 0.0                                                # last rebalance falls after the sample
    net = rp - turn.shift(1).fillna(0) * cost_bp / 1e4                 # cost of rebalance at end of m hits month m+1
    return net - rf, turn.sum() / (len(months) / 12)


def sr(x):
    return x.mean() / x.std() * np.sqrt(12)


Mi = F.reindex(ME.values)
Mi.index = ME.index
W = {"BH": pd.Series(1.0, index=Mi.index),
     "TREND10": M["TREND_IN"]}
for x in ["RV", "AC60", "AR60", "H1N", "H1P", "TOPOVOL", "FIED"]:
    W[f"B-{x}"] = binary_w(F[x], FROZEN[x]["direction"], FROZEN[x]["q"]).reindex(ME.values).set_axis(ME.index)
W["STUDENT TOPOVOL high|0.75"] = binary_w(F["TOPOVOL"], "high", 0.75).reindex(ME.values).set_axis(ME.index)
builder = pd.read_csv("test_main.csv").set_index("rule")["SR"]
EX = {}
for k, w in W.items():
    ex, to = backtest(w, 0 if k == "BH" else 2.0)
    EX[k] = ex
    print(f"{k:<28} verifier SR {sr(ex):+.3f}  turnover/yr {to:.2f}   builder SR {builder.get(k, np.nan):+.3f}")
x = mm.loc["1990-01":"2024-12", "mkt_rf"]
print(f"(BH from the French MONTHLY file directly: SR {sr(x):+.3f})")

# ------------------------------------------------------------------ 4. family-wise test of the BEST rule vs BH
# Max-statistic (White reality check style) over the 13 test rules would need the VT ones; here the 7 frozen binary rules +
# TREND10 (the rules whose |dSR| claims matter), paired stationary-block bootstrap of the MAX Sharpe difference.
print("\n== 4. Reality-check style family-wise p for 'best rule beats BH' (7 frozen binary rules + TREND10), test period")
names = [f"B-{x}" for x in ["RV", "AC60", "AR60", "H1N", "H1P", "TOPOVOL", "FIED"]] + ["TREND10"]
A = np.column_stack([EX[n].values for n in names]); b = EX["BH"].values; T = len(b)
d_obs = np.array([sr(pd.Series(A[:, j])) for j in range(A.shape[1])]) - sr(pd.Series(b))
rng = np.random.default_rng(20260925)
for L in (6, 12, 24):
    mx = []
    for _ in range(5000):
        st = rng.integers(0, T, int(np.ceil(T / L)))
        I = ((st[:, None] + np.arange(L)[None, :]).ravel()[:T]) % T
        a_, b_ = A[I], b[I]
        s_a = a_.mean(0) / a_.std(0, ddof=1) * np.sqrt(12); s_b = b_.mean() / b_.std(ddof=1) * np.sqrt(12)
        mx.append(np.max((s_a - s_b) - d_obs))                         # centred max statistic
    mx = np.array(mx)
    j = int(np.argmax(d_obs))
    print(f"block {L:>2}: best = {names[j]} dSR {d_obs[j]:+.3f}; family-wise p (max-stat) = {np.mean(mx >= d_obs[j]):.3f}")

# ------------------------------------------------------------------ 5. where does B-H1N's edge come from?
print("\n== 5. B-H1N attribution: months in cash in the test period and the market excess return in those months")
w = W["B-H1N"].shift(1).loc["1990-01":"2024-12"]
out = w[w == 0].index
exm = (Rm - Rf).reindex(out)
print(f"months in cash: {len(out)} of {len(w)}; mean market excess return in those months {exm.mean():+.3%} "
      f"(all test months {(Rm - Rf).loc['1990-01':'2024-12'].mean():+.3%})")
print("by year (months out, cumulative mkt excess return in those months):")
g = exm.groupby(exm.index.year).agg(["count", lambda s: (1 + s).prod() - 1])
g.columns = ["n_out", "mkt_ex_when_out"]
print(g.round(4).to_string())
# leave-one-year-out: drop each calendar year of the test and recompute dSR
dd = []
for y in range(1990, 2025):
    keep = EX["B-H1N"].index.year != y
    dd.append((y, sr(EX["B-H1N"][keep]) - sr(EX["BH"][keep])))
dd = pd.Series(dict(dd))
print(f"leave-one-year-out dSR(B-H1N - BH): min {dd.min():+.3f} (drop {dd.idxmin()}), max {dd.max():+.3f} (drop {dd.idxmax()})")
keep = ~EX["B-H1N"].index.year.isin([2000, 2001, 2002])
print(f"excluding 2000-2002: dSR {sr(EX['B-H1N'][keep]) - sr(EX['BH'][keep]):+.3f}")

# ------------------------------------------------------------------ 6. 10bp ranking claim
print("\n== 6. Ranking at 2 bp vs 10 bp (from the builder's own logs, re-run by the verifier)")


def ranking(path, start_marker):
    lines = open(path).read().splitlines()
    i = next(k for k, l in enumerate(lines) if l.startswith(start_marker))
    out = []
    for l in lines[i + 3:]:
        if not l.strip():
            break
        parts = l.split()
        name = " ".join(parts[:-19])
        out.append(name)
    return out


r2 = ranking("logs_test.txt", "== MAIN TEST")
r10 = ranking("logs_robust.txt", "== R1 10bp")
print("2 bp :", r2)
print("10 bp:", r10)
print("identical ranking:", r2 == r10)
print("moved:", [(n, r2.index(n) + 1, r10.index(n) + 1) for n in r2 if r2.index(n) != r10.index(n)])


# ------------------------------------------------------------------ 7. "In most years TREND10 earns less than buy-and-hold"?
print("\n== 7. Calendar-year comparison TREND10 (2 bp) vs BH, 1990-2024 (total returns incl. T-bill leg)")
tr = EX["TREND10"] + Rf.reindex(EX["TREND10"].index); bh = EX["BH"] + Rf.reindex(EX["BH"].index)
ya = (1 + tr).groupby(tr.index.year).prod() - 1; yb = (1 + bh).groupby(bh.index.year).prod() - 1
lag = (ya < yb - 1e-9).sum(); tie = (np.abs(ya - yb) <= 1e-9).sum()
print(f"years TREND10 < BH: {lag} of {len(ya)}; exactly equal (fully invested all year): {tie}; TREND10 > BH: {len(ya) - lag - tie}")
print(f"median annual shortfall in lagging years {(ya - yb)[ya < yb - 1e-9].median():+.2%}; median gain in leading years {(ya - yb)[ya > yb + 1e-9].median():+.2%}")

# ------------------------------------------------------------------ 8. block-length sensitivity of key paired p-values; Holm on JK-Memmel p
print("\n== 8. Paired Sharpe-difference bootstrap p vs BH for different block lengths (verifier's own bootstrap, seed 7)")
for name in ["B-TOPOVOL", "B-H1N", "TREND10", "STUDENT TOPOVOL high|0.75"]:
    a, b = EX[name].values, EX["BH"].values
    d0 = sr(pd.Series(a)) - sr(pd.Series(b))
    out = []
    for L in (1, 6, 12, 24):
        r = np.random.default_rng(7)
        ds = []
        for _ in range(5000):
            st = r.integers(0, T, int(np.ceil(T / L)))
            I = ((st[:, None] + np.arange(L)[None, :]).ravel()[:T]) % T
            ds.append(a[I].mean() / a[I].std(ddof=1) * np.sqrt(12) - b[I].mean() / b[I].std(ddof=1) * np.sqrt(12))
        ds = np.array(ds)
        out.append(f"L={L:>2}: p {np.mean(np.abs(ds - d0) >= abs(d0)):.3f}")
    print(f"{name:<26} dSR {d0:+.3f}  " + "  ".join(out))
tm = pd.read_csv("test_main.csv").set_index("rule")
UT = [f"B-{x}" for x in ["RV", "AC60", "AR60", "H1N", "H1P", "TOPOVOL", "FIED"]] + [f"VT-RV+{x}" for x in ["AC60", "AR60", "H1N", "H1P", "TOPOVOL", "FIED"]]
pj = tm.loc[UT, "p_JKM"].sort_values()
adj = np.maximum.accumulate([min(1, (len(pj) - i) * p) for i, p in enumerate(pj.values)])
print("Holm on JK-Memmel p (13 rules), smallest three:", {k: round(v, 3) for k, v in list(zip(pj.index, adj))[:3]})
