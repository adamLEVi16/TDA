"""
PEAD x ATTENTION -- replicate a classic, extend it with novel data.

Base (replication): post-earnings-announcement drift. Events = EDGAR 8-K
Item-2.02 dates, 2005-2024. Surprise measured two pre-committed ways:
  (a) REACTION  = 2-day market-adjusted announcement return (close E-1 -> E+2),
      the market's own initial read of the news;
  (b) SUE       = Bernard-Thomas standardized unexpected earnings from SEC XBRL
      (seasonal random walk), point-in-time (earliest-filed values).
Drift = market-adjusted return from close E+2 to close E+21/+42/+61 trading
days (all three horizons reported -- no cherry-picking).

Extension (the new part): Hirshleifer/limited-attention theory predicts drift
is STRONGER when attention at the announcement is LOW (nobody notices the
news, so it gets priced slowly). Attention = abnormal Wikipedia pageviews in
the announcement window [E-1, E+1] vs the trailing 8-week median (observable
by the time the drift position opens at E+2). Available 2016+.

Point-in-time discipline for the TRADABLE test: an event's long/short-leg
assignment compares its surprise only to the PRIOR 60 events across the panel
(never to same-quarter events announced later). Calendar-time daily portfolio,
10 bps/side costs at entry/exit.
"""
import warnings; warnings.filterwarnings("ignore")
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "equity_factor"))
import numpy as np, pandas as pd
import statsmodels.api as sm
import price_history as PH
import earnings_data as ED
import xbrl_data as XB
import wiki_data as WD
from universe import TICKER_ARTICLE

E_MIN, E_MAX = "2005-01-01", "2024-09-30"
REACT_D, HORIZONS = 2, (21, 42, 61)
TRAIL_EVENTS = 60          # point-in-time ranking pool for the tradable test
COST_BPS = 10


# ---------------------------------------------------------------- events
def announcement_attention(pv_daily, E):
    """log(mean views E-1..E+1) - log(median views E-56..E-2). None if thin."""
    win = pv_daily.loc[E - pd.Timedelta(days=1): E + pd.Timedelta(days=1)]
    base = pv_daily.loc[E - pd.Timedelta(days=56): E - pd.Timedelta(days=2)]
    if len(win) < 2 or len(base) < 40:
        return np.nan
    return np.log(max(win.mean(), 1)) - np.log(max(base.median(), 1))


def build_events():
    tickers = list(TICKER_ARTICLE.keys())
    px = PH.get_prices_long(tickers + ["SPY"], verbose=False)
    spy = px["SPY"].dropna()
    earn = ED.get_earnings_dates(tickers, verbose=False)
    sue = XB.get_sue(tickers, verbose=False)
    pv = WD.get_pageviews(list(TICKER_ARTICLE.values()), verbose=False)
    art = {k: TICKER_ARTICLE[k] for k in tickers}

    rows = []
    for tk in tickers:
        if tk not in px.columns or tk not in earn:
            continue
        p = px[tk].dropna()
        idx = p.index
        s_tk = sue.get(tk, pd.Series(dtype=float))
        pv_tk = pv.get(art[tk])
        for E in earn[tk]:
            if not (pd.Timestamp(E_MIN) <= E <= pd.Timestamp(E_MAX)):
                continue
            prior = idx[idx < E]
            if len(prior) < 10:
                continue
            e0 = idx.get_loc(prior[-1])                      # close before E
            if e0 + max(HORIZONS) >= len(idx):
                continue
            # market-adjusted returns off matched SPY closes
            def madj(i, j):
                d0, d1 = idx[i], idx[j]
                if d0 not in spy.index or d1 not in spy.index:
                    return np.nan
                return (p.iloc[j] / p.iloc[i] - 1) - (spy.loc[d1] / spy.loc[d0] - 1)
            reaction = madj(e0, e0 + REACT_D)
            drifts = {f"drift{h}": madj(e0 + REACT_D, e0 + h) for h in HORIZONS}
            if np.isnan(reaction) or any(np.isnan(v) for v in drifts.values()):
                continue
            # match SUE: fiscal quarter ended before E, within 75 days.
            # TIMING NOTE: the EPS for that quarter is public AT E (the 8-K
            # Item-2.02 announcement itself contains it); the XBRL value we
            # use is the same number as later filed in the 10-Q/10-K
            # (earliest-filed dedup guards against restatements). Strictly,
            # press-release EPS and as-filed XBRL EPS can differ in rare
            # cases; we accept that proxy error and state it here rather
            # than lag SUE a full quarter (which would be economically wrong).
            su = np.nan
            if len(s_tk):
                cand = s_tk[(s_tk.index < E) & (s_tk.index > E - pd.Timedelta(days=75))]
                if len(cand):
                    su = float(cand.iloc[-1])
            attn = announcement_attention(pv_tk, E) if pv_tk is not None else np.nan
            rows.append({"ticker": tk, "E": E, "qtr": E.to_period("Q"),
                         "e0": e0, "reaction": reaction, "sue": su, "attn": attn,
                         **drifts})
    ev = pd.DataFrame(rows).sort_values("E").reset_index(drop=True)
    return ev, px, spy


# ------------------------------------------------- point-in-time leg labels
def pit_terciles(ev, sig):
    """Leg per event: surprise percentile vs the PRIOR TRAIL_EVENTS panel events
    (strictly earlier announcement dates). +1 top third, -1 bottom third."""
    vals = ev[sig].values
    legs = np.zeros(len(ev))
    for i in range(len(ev)):
        if np.isnan(vals[i]):
            continue
        pool = vals[max(0, i - TRAIL_EVENTS):i]
        pool = pool[~np.isnan(pool)]
        if len(pool) < 30:
            continue
        pct = (pool < vals[i]).mean()
        legs[i] = 1 if pct >= 2 / 3 else (-1 if pct <= 1 / 3 else 0)
    return legs


# ------------------------------------------------- calendar-time portfolio
def calendar_portfolio(ev, px, spy, sig, hold=61, cost_bps=COST_BPS):
    """Daily calendar-time L/S: long leg=+1 events, short leg=-1 events, active
    from close e0+REACT_D to close e0+hold. Equal weight within leg; daily
    ret = mean(long) - mean(short); costs charged at entry+exit."""
    ev = ev.copy()
    ev["leg"] = pit_terciles(ev, sig)
    cal = px.index
    active = {i: [] for i in range(len(cal))}          # day -> list of (ticker, leg)
    entries = np.zeros(len(cal))
    for _, r in ev[ev["leg"] != 0].iterrows():
        p = px[r["ticker"]].dropna()
        gi = px.index.get_indexer(p.index[[r["e0"] + REACT_D, min(r["e0"] + hold, len(p) - 1)]])
        a, b = gi[0], gi[1]
        for d in range(a + 1, b + 1):                  # earn returns after entry close
            active[d].append((r["ticker"], r["leg"]))
        entries[a] += 2 * cost_bps / 1e4               # entry + exit, per unit weight
    rets = px.pct_change()
    out = []
    for d in range(1, len(cal)):
        mem = active[d]
        if not mem:
            continue
        longs = [rets[t].iloc[d] for t, l in mem if l > 0 and np.isfinite(rets[t].iloc[d])]
        shorts = [rets[t].iloc[d] for t, l in mem if l < 0 and np.isfinite(rets[t].iloc[d])]
        r = (np.mean(longs) if longs else 0.0) - (np.mean(shorts) if shorts else 0.0)
        # amortize entry costs on the day they occur (approx: spread over legs)
        n = max(len(longs) + len(shorts), 1)
        out.append({"date": cal[d], "ret": r - entries[d] / n})
    s = pd.DataFrame(out).set_index("date")["ret"]
    return s


def perf(s, label, ppy=252):
    ann = s.mean() * ppy; vol = s.std() * np.sqrt(ppy)
    sh = ann / vol if vol > 0 else np.nan
    rng = np.random.default_rng(9)
    # circular block bootstrap p-value on the mean (10-day blocks)
    T, B = len(s), 10
    means = []
    v = s.values
    for _ in range(3000):
        starts = rng.integers(0, T, T // B + 1)
        idxs = np.concatenate([np.arange(x, x + B) % T for x in starts])[:T]
        means.append(v[idxs].mean())
    p = (np.array(means) <= 0).mean()
    print(f"  {label:<44} ann={ann:+7.2%}  Sharpe={sh:+5.2f}  p={p:.3f}  ({len(s)}d)")
    return sh, p


# ---------------------------------------------------------------- main
def main():
    ev, px, spy = build_events()
    ev.to_csv(os.path.join(os.path.dirname(__file__), "pead_events.csv"), index=False)
    print("=" * 84)
    print(f"  PEAD x ATTENTION -- {len(ev)} events, {ev['ticker'].nunique()} tickers, "
          f"{ev['E'].min().date()} -> {ev['E'].max().date()}")
    print(f"  with SUE: {ev['sue'].notna().sum()}   with attention: {ev['attn'].notna().sum()}")
    print("=" * 84)

    # [1] replication regressions: drift ~ surprise, cluster by quarter
    print("\n[1] DRIFT REGRESSIONS (cluster-robust by calendar quarter)")
    for sig in ["reaction", "sue"]:
        d = ev.dropna(subset=[sig]).copy()
        d["sz"] = d.groupby("qtr")[sig].transform(
            lambda x: (x - x.mean()) / x.std() if x.std() > 0 else x * 0)
        for h in HORIZONS:
            m = sm.OLS(d[f"drift{h}"], sm.add_constant(d[["sz"]])).fit(
                cov_type="cluster", cov_kwds={"groups": d["qtr"].astype(str)})
            print(f"  {sig:<9} -> drift+{h:<3} coef={m.params['sz']:+.4f}  "
                  f"t={m.tvalues['sz']:+.2f}  p={m.pvalues['sz']:.3f}  (n={len(d)})")

    # [2] tradable calendar-time portfolios
    print(f"\n[2] CALENDAR-TIME L/S PORTFOLIO (point-in-time terciles, {COST_BPS} bps/side)")
    for sig in ["reaction", "sue"]:
        s = calendar_portfolio(ev, px, spy, sig)
        perf(s, f"{sig}-sorted, hold to +61d, NET of costs")

    # [3] the extension: attention conditioning (2016+, where attn exists)
    print("\n[3] ATTENTION CONDITIONING (Hirshleifer: drift stronger when attention LOW)")
    d = ev.dropna(subset=["attn", "reaction"]).copy()
    d["sz"] = d.groupby("qtr")["reaction"].transform(
        lambda x: (x - x.mean()) / x.std() if x.std() > 0 else x * 0)
    d["az"] = d.groupby("qtr")["attn"].transform(
        lambda x: (x - x.mean()) / x.std() if x.std() > 0 else x * 0)
    d["sz_x_az"] = d["sz"] * d["az"]
    for h in HORIZONS:
        m = sm.OLS(d[f"drift{h}"], sm.add_constant(d[["sz", "az", "sz_x_az"]])).fit(
            cov_type="cluster", cov_kwds={"groups": d["qtr"].astype(str)})
        print(f"  drift+{h:<3}: surprise t={m.tvalues['sz']:+.2f}   "
              f"INTERACTION coef={m.params['sz_x_az']:+.4f} t={m.tvalues['sz_x_az']:+.2f} "
              f"p={m.pvalues['sz_x_az']:.3f}   (theory: negative)")
    # split-sample view
    med = d["attn"].median()
    print(f"\n  split at median announcement attention (n={len(d)}):")
    for lab, sub in [("LOW attention ", d[d["attn"] <= med]), ("HIGH attention", d[d["attn"] > med])]:
        m = sm.OLS(sub["drift61"], sm.add_constant(sub[["sz"]])).fit(
            cov_type="cluster", cov_kwds={"groups": sub["qtr"].astype(str)})
        print(f"    {lab}: drift61~surprise coef={m.params['sz']:+.4f} "
              f"t={m.tvalues['sz']:+.2f} p={m.pvalues['sz']:.3f} (n={len(sub)})")

    # [4] placebo: permute surprise across events
    print("\n[4] PLACEBO (permute reaction across events, 2000x, drift+61)")
    rng = np.random.default_rng(4)
    dd = ev.dropna(subset=["reaction"])
    real = np.polyfit(dd["reaction"].rank(pct=True), dd["drift61"], 1)[0]
    null = [np.polyfit(rng.permutation(dd["reaction"].rank(pct=True).values),
                       dd["drift61"].values, 1)[0] for _ in range(2000)]
    p = (np.abs(null) >= abs(real)).mean()
    print(f"  real slope={real:+.4f}  null95=[{np.percentile(null,2.5):+.4f},"
          f"{np.percentile(null,97.5):+.4f}]  p={p:.3f}")

    # [5] sub-periods for the tradable version
    print("\n[5] SUB-PERIOD STABILITY (reaction-sorted, net)")
    s = calendar_portfolio(ev, px, spy, "reaction")
    for lab, a, b in [("2005-2010", "2005", "2010"), ("2011-2016", "2011", "2016"),
                      ("2017-2020", "2017", "2020"), ("2021-2024", "2021", "2024")]:
        seg = s.loc[a:b]
        if len(seg) > 100:
            perf(seg, f"  {lab}")


if __name__ == "__main__":
    main()
