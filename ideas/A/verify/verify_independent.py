"""VERIFIER: independent re-implementation of the key Concept-A numbers (does NOT import engine.py), plus extra checks.
Uses only the builder's cached raw data (copied to ./cache): Ken French zips and etf_px.csv.
Checks:
 A. independent NET1/NET12 (k=5) signals, industry quintile L/S + LO, ETF top-5 LO (net, drift-aware costs)
 B. timing: explicit assertion + a deliberate look-ahead version to show how much a leak would inflate results
 C. LAPGAP1 in the student's ORIGINAL sign (long high residual x - h), since SPEC flipped it
 D. data: max |monthly file - compounded daily| (builder text says 'max about 1.8%')
 E. White (2000)-style reality check (stationary bootstrap, centred max statistic) over the 18 post-hoc
    ETF long-only NET cells (k x N) vs EW, and Holm adjustment of the pre-registered H4 p-values.
"""
import io
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

HERE = Path(__file__).parent
C = HERE / "cache"


def french(name):
    lines = zipfile.ZipFile(C / name).read(zipfile.ZipFile(C / name).namelist()[0]).decode("latin1").splitlines()
    h = [i for i, l in enumerate(lines) if l.startswith(",")][0]
    out = []
    for l in lines[h + 1:]:
        parts = [p.strip() for p in l.split(",")]
        if not parts[0].isdigit():
            if out:
                break
            continue
        out.append(parts)
    cols = [c.strip() for c in lines[h].split(",")[1:]]
    df = pd.DataFrame([o[1:] for o in out], index=[o[0] for o in out], columns=cols).astype(float)
    df = df.mask(df <= -99.99) / 100.0
    if len(df.index[0]) == 6:
        df.index = pd.PeriodIndex(df.index, freq="M").to_timestamp("M")
    else:
        df.index = pd.to_datetime(df.index)
    return df


def nw_t(x, lags=6):
    x = pd.Series(x).dropna().values
    f = sm.OLS(x, np.ones(len(x))).fit(cov_type="HAC", cov_kwds={"maxlags": lags})
    return f.params[0], f.tvalues[0]


def sr(x):
    x = np.asarray(x)
    return x.mean() / x.std(ddof=1) * np.sqrt(12)


def signals(daily, monthly, t, k, lookahead=False):
    """Independent signal code. lookahead=True deliberately uses the NEXT month's returns (bug demo)."""
    dwin = daily[daily.index <= t].iloc[-252:]
    assert dwin.index[-1] <= t
    mh = monthly[monthly.index <= t].iloc[-12:]
    if len(mh) < 12 or mh.index[-1] != t or len(dwin) < 252:
        return None
    ok = (dwin.count() >= 240) & mh.notna().all()
    names = ok.index[ok]
    if len(names) < 6:
        return None
    own1 = mh[names].iloc[-1]
    own12 = (1 + mh[names].iloc[:-1]).prod() - 1
    if lookahead:
        nxt = monthly[monthly.index > t]
        own1 = nxt[names].iloc[0].fillna(0)
    Cm = dwin[names].corr()
    net1, net12 = {}, {}
    for a in names:
        c = Cm.loc[a].drop(a)
        nb = c.sort_values(ascending=False, kind="mergesort").index[:k]
        w = c[nb].clip(lower=0)
        w = w / w.sum() if w.sum() > 0 else pd.Series(1 / k, nb)
        net1[a] = float((w * own1[nb]).sum())
        net12[a] = float((w * own12[nb]).sum())
    return pd.DataFrame({"OWN1": own1, "OWN12": own12, "NET1": pd.Series(net1), "NET12": pd.Series(net12)})


def run_book(weights_by_t, monthly, cost):
    """weights_by_t: {t: Series}; earns month after t; cost on |target - drifted|."""
    rows, prev = [], pd.Series(dtype=float)
    for t in sorted(weights_by_t):
        nxt = monthly.index[monthly.index > t]
        if len(nxt) == 0:
            break
        t1 = nxt[0]
        assert t1 > t
        w = weights_by_t[t]
        allk = w.index.union(prev.index)
        to = (w.reindex(allk, fill_value=0) - prev.reindex(allk, fill_value=0)).abs().sum()
        r = monthly.loc[t1, w.index].fillna(0)
        g = float((w * r).sum())
        prev = w * (1 + r) / (1 + g)
        rows.append((t1, g, g - cost * to, to))
    return pd.DataFrame(rows, columns=["m", "gross", "net", "to"]).set_index("m")


def pick(sig, frac=None, n=None, long_only=False):
    s = sig.dropna().sort_values(ascending=False, kind="mergesort")
    m = n if n is not None else max(1, int(round(frac * len(s))))
    w = pd.Series(0.0, s.index)
    w.iloc[:m] = 1 / m
    if not long_only:
        w.iloc[-m:] = -1 / m
    return w


def stat_idx(n, rng, block=6):
    idx = np.empty(n, int)
    idx[0] = rng.integers(n)
    for i in range(1, n):
        idx[i] = rng.integers(n) if rng.random() < 1 / block else (idx[i - 1] + 1) % n
    return idx


if __name__ == "__main__":
    ff3 = french("F-F_Research_Data_Factors_CSV.zip")
    rf = ff3["RF"]
    d = french("49_Industry_Portfolios_daily_CSV.zip").loc[:"2024-12-31"]
    m = french("49_Industry_Portfolios_CSV.zip").loc[:"2024-12-31"]

    # ---------------- D. data check
    comp = (1 + d).resample("ME").prod(min_count=15) - 1
    j = comp.index.intersection(m.index)
    diff = (comp.loc[j] - m.loc[j]).abs()
    print("D. |monthly - compounded daily|: overall max %.4f, median over industries of per-industry max %.4f, "
          "mean %.5f, share of industry-months > 0.5%%: %.4f" % (np.nanmax(diff.values), diff.max().median(),
                                                               np.nanmean(diff.values), np.nanmean(diff.values > 0.005)))
    print("   worst cell:", diff.stack().idxmax(), "; missing codes remaining (<=-0.9999):",
          int((m <= -0.9999).sum().sum()), int((d <= -0.9999).sum().sum()))

    # ---------------- A. independent industry replication (test 1990-2024)
    form = m.loc["1989-12-31":"2024-11-30"].index
    S = {t: signals(d, m, t, 5) for t in form}
    S = {t: s for t, s in S.items() if s is not None}
    print("\nA. INDEPENDENT industry replication, test 1990-01..2024-12, k=5, 10 bp")
    res = {}
    for name in ["OWN1", "OWN12", "NET1", "NET12"]:
        ls = run_book({t: pick(s[name], frac=0.2) for t, s in S.items()}, m, 0.001)
        lo = run_book({t: pick(s[name], frac=0.2, long_only=True) for t, s in S.items()}, m, 0.001)
        res[name] = (ls, lo)
        mu, t = nw_t(ls["net"])
        lox = lo["net"] - rf.reindex(lo.index)
        print(f"  L/S {name:<6} gross {12*ls['gross'].mean():+.2%} net {12*mu:+.2%} t={t:+.2f} SR {sr(ls['net']):+.2f} "
              f"TO {ls['to'].iloc[1:].mean():.2f} | LO net excess {12*lox.mean():+.2%} SR {sr(lox):+.2f}")
    ew = run_book({t: pd.Series(1 / len(s), s.index) for t, s in S.items()}, m, 0.001)
    ewx = ew["net"] - rf.reindex(ew.index)
    print(f"  EW all industries net excess {12*ewx.mean():+.2%} SR {sr(ewx):+.2f}")

    # ---------------- C. LAPGAP1 in the student's original sign = negation of builder's LAPGAP1 L/S
    import pickle, sys
    sys.path.insert(0, str(HERE))
    import engine as E
    sigs = pickle.load(open(C / "ind_signals.pkl", "rb"))["sigs"]
    lg = sigs[("LAPGAP1", 5)].loc["1989-12-31":"2024-11-30"]
    print("\nC. LAPGAP1, test 1990-2024, 10 bp, quintile L/S")
    for lab_, sig in [("builder sign (h - x, 'catch up')", lg), ("student's original sign (x - h, long high residual)", -lg)]:
        bt = E.backtest(E.rank_weights(sig, frac=0.2), m, 0.001)
        mu, t = nw_t(bt["net"])
        mg, tg = nw_t(bt["gross"])
        print(f"  {lab_:<52} gross {12*mg:+.2%} (t={tg:+.2f}) net {12*mu:+.2%} (t={t:+.2f}) SR {sr(bt['net']):+.2f} "
              f"TO {bt['turnover'].iloc[1:].mean():.2f}")
    for per in [("1927-07", "1989-12")]:
        lgd = sigs[("LAPGAP1", 5)].loc[:"1989-11-30"]
        bt = E.backtest(E.rank_weights(-lgd, frac=0.2), m.loc[:"1989-12-31"], 0.001)
        print(f"  original sign, DESIGN 1927-1989: gross {12*bt['gross'].mean():+.2%} net {12*bt['net'].mean():+.2%} "
              f"SR {sr(bt['net']):+.2f}")

    # ---------------- ETFs
    px = pd.read_csv(C / "etf_px.csv", index_col=0, parse_dates=True).loc[:"2024-12-31"]
    spy = px.pop("SPY")
    ed = px.pct_change(fill_method=None)
    mpx = px.resample("ME").last()
    em = mpx.pct_change(fill_method=None).mask(mpx.isna())
    spym = spy.resample("ME").last().pct_change(fill_method=None)
    eform = em.loc["2006-12-31":"2024-11-30"].index
    ES = {k: {t: signals(ed, em, t, k) for t in eform} for k in [3, 5, 10]}
    print("\nA2. INDEPENDENT ETF replication 2007-01..2024-12, 3 bp, long-only (excess RF)")
    eew = run_book({t: pd.Series(1 / len(s), s.index) for t, s in ES[5].items()}, em, 0.0003)
    eewx = eew["net"] - rf.reindex(eew.index)
    spyx = spym.loc["2007-01":"2024-12"] - rf.loc["2007-01":"2024-12"]
    print(f"  EW  net excess {12*eewx.mean():+.2%} SR {sr(eewx):+.2f};  SPY (no cost) excess {12*spyx.mean():+.2%} SR {sr(spyx):+.2f}")
    lo_cells = {}
    for name in ["NET1", "NET12"]:
        for k in [3, 5, 10]:
            for n in [3, 5, 7]:
                bt = run_book({t: pick(s[name], n=n, long_only=True) for t, s in ES[k].items()}, em, 0.0003)
                lo_cells[(name, k, n)] = bt["net"] - rf.reindex(bt.index)
    for name in ["OWN12"]:
        bt = run_book({t: pick(s[name], n=5, long_only=True) for t, s in ES[5].items()}, em, 0.0003)
        o12 = bt["net"] - rf.reindex(bt.index)
    for key in [("NET1", 5, 5), ("NET12", 5, 5)]:
        x = lo_cells[key]
        print(f"  LO top5 {key[0]:<6} net excess {12*x.mean():+.2%} SR {sr(x):+.2f} MDD "
              f"{float(((1+x+rf.reindex(x.index)).cumprod()/(1+x+rf.reindex(x.index)).cumprod().cummax()-1).min()):.0%}")
    print(f"  LO top5 OWN12 net excess {12*o12.mean():+.2%} SR {sr(o12):+.2f}")

    # ---------------- B. timing / leak demo
    Sleak = {t: signals(d, m, t, 5, lookahead=True) for t in list(S)[:120]}
    bl = run_book({t: pick(s["NET1"], frac=0.2) for t, s in Sleak.items() if s is not None}, m, 0.001)
    bo = run_book({t: pick(S[t]["NET1"], frac=0.2) for t in list(S)[:120]}, m, 0.001)
    print(f"\nB. leak demo (1990-1999, NET1 L/S net): correct timing SR {sr(bo['net']):+.2f}; "
          f"DELIBERATE look-ahead (neighbours' NEXT-month return) SR {sr(bl['net']):+.2f}  -> builder's numbers show no such inflation")

    # ---------------- E. multiple testing
    print("\nE1. White reality check: max over 18 post-hoc ETF LO NET cells of Sharpe(cell) - Sharpe(EW), 2007-2024")
    keys = list(lo_cells)
    M = pd.concat([lo_cells[k_] for k_ in keys] + [eewx], axis=1).dropna().values
    obs = np.array([sr(M[:, i]) - sr(M[:, -1]) for i in range(len(keys))])
    rng = np.random.default_rng(0)
    B = 2000
    maxc = np.empty(B)
    for b in range(B):
        ix = stat_idx(len(M), rng)
        Mb = M[ix]
        db = np.array([sr(Mb[:, i]) - sr(Mb[:, -1]) for i in range(len(keys))])
        maxc[b] = (db - obs).max()
    best = int(np.argmax(obs))
    print(f"  best cell {keys[best]} dSR {obs[best]:+.2f}; reality-check p = {np.mean(maxc >= obs.max()):.3f} "
          f"(n={len(M)} months, {B} stationary-bootstrap draws, block 6, seed 0)")
    print("  pre-registered cell (NET12,k=5,N=5) dSR %+.2f" % obs[keys.index(("NET12", 5, 5))])
    print("\nE2. Holm adjustment, pre-registered H4 tests (p from builder's logs_test_etfs.txt, 4 tests):")
    ps = {"NET1-EW": 0.281, "NET12-EW": 0.276, "NET1-OWN12": 0.338, "NET12-OWN12": 0.121}
    srt = sorted(ps.items(), key=lambda kv: kv[1])
    run = 0
    for i, (k_, p) in enumerate(srt):
        run = max(run, min(1, (len(ps) - i) * p))
        print(f"  {k_:<12} raw p {p:.3f} Holm p {run:.3f}")
    # Sharpe SE rule of thumb for the ETF sample
    T = len(eewx)
    print(f"\n  Approx. SE of an annualised Sharpe with T={T} months (Lo 2002 iid): {np.sqrt((1+0.5*0.59**2/12)/T*12):.2f}")
