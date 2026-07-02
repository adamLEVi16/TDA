"""
INSIDER CLUSTER-BUY STUDY -- the most persistent documented anomaly reachable
with free data (Cohen-Malloy-Pomorski 2012 and the cluster-buying literature:
multiple insiders independently buying their own stock in the open market is
the strongest insider signal; routine/lone small buys are weak).

PRE-COMMITTED DESIGN (fixed before looking at any outcome):
  EVENT (cluster buy): >= 2 DISTINCT insiders make open-market purchases
    (Form 4, code P) of the same stock within 21 calendar days, combined value
    >= $100k. Event date = filing date of the purchase that COMPLETES the
    cluster (point-in-time: that's the day the market can know). One event per
    ticker per 63 trading days (no double-counting rolling windows).
  ENTRY: close of the first session AFTER the event date.
  HORIZONS: +21, +63, +126 trading days, market-adjusted (minus SPY).
  TESTS: [1] event-study mean abnormal return + t (cluster-robust by month),
         [2] circular-shift placebo within ticker,
         [3] calendar-time long portfolio, 10 bps/side, vs SPY (the tradable
             test: CAPM alpha + block-bootstrap p).
"""
import warnings; warnings.filterwarnings("ignore")
import os, sys, time
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "equity_factor"))
import numpy as np, pandas as pd
import statsmodels.api as sm
import data as D

CLUSTER_DAYS, MIN_INSIDERS, MIN_VALUE = 21, 2, 100_000
COOLDOWN_TD, HOLD_TD, COST_BPS = 63, 63, 10
PX_CACHE = os.path.join(HERE, "cache")


def get_px(tickers, start="2013-01-01", end="2024-12-31"):
    series = {}
    for tk in tickers:
        fp = os.path.join(PX_CACHE, f"px_{tk}.csv")
        if os.path.exists(fp):
            s = pd.read_csv(fp, index_col=0, parse_dates=True).iloc[:, 0]; s.name = tk
        else:
            s = D._fetch_one(tk, start, end); time.sleep(0.15)
            if s is not None and len(s) > 50:
                s.to_frame().to_csv(fp)
        if s is not None and len(s) > 50:
            series[tk] = s
    px = pd.concat(series.values(), axis=1, sort=True)
    px.columns = list(series.keys())
    return px


def detect_events(purch):
    """Cluster events per pre-committed rule; event date = completing filing."""
    events = []
    for tk, g in purch.groupby("ticker"):
        g = g.sort_values("filing_date")
        last_event = None
        for i, r in enumerate(g.itertuples()):
            w = g[(g["filing_date"] > r.filing_date - pd.Timedelta(days=CLUSTER_DAYS))
                  & (g["filing_date"] <= r.filing_date)]
            if w["owner"].nunique() >= MIN_INSIDERS and w["value"].sum() >= MIN_VALUE:
                if last_event is not None and \
                   (r.filing_date - last_event).days < COOLDOWN_TD * 1.6:
                    continue
                events.append({"ticker": tk, "date": r.filing_date,
                               "n_insiders": int(w["owner"].nunique()),
                               "value": float(w["value"].sum())})
                last_event = r.filing_date
    return pd.DataFrame(events).sort_values("date").reset_index(drop=True)


def abnormal_returns(events, px, spy):
    cal = px.index
    rows = []
    for r in events.itertuples():
        if r.ticker not in px.columns:
            continue
        p = px[r.ticker].dropna()
        pos = p.index.searchsorted(r.date, side="right")
        if pos >= len(p) or pos + 126 >= len(p):
            continue
        e = p.index[pos]                     # entry close
        if e not in spy.index:
            continue
        out = {"ticker": r.ticker, "date": r.date, "n_insiders": r.n_insiders,
               "value": r.value}
        for h in (21, 63, 126):
            j = min(pos + h, len(p) - 1)
            d1 = p.index[j]
            if d1 not in spy.index:
                out[f"ar{h}"] = np.nan; continue
            out[f"ar{h}"] = (p.iloc[j] / p.iloc[pos] - 1) - \
                            (spy.loc[d1] / spy.loc[e] - 1)
        rows.append(out)
    return pd.DataFrame(rows).dropna(subset=["ar63"])


def event_tests(ev):
    print(f"\n[1] EVENT STUDY -- {len(ev)} cluster-buy events, "
          f"{ev['ticker'].nunique()} tickers, {ev['date'].min().date()} -> "
          f"{ev['date'].max().date()}")
    ev = ev.copy(); ev["mon"] = ev["date"].dt.to_period("M").astype(str)
    for h in (21, 63, 126):
        d = ev.dropna(subset=[f"ar{h}"])
        m = sm.OLS(d[f"ar{h}"], np.ones(len(d))).fit(
            cov_type="cluster", cov_kwds={"groups": d["mon"]})
        print(f"  +{h:>3}td: mean AR={d[f'ar{h}'].mean():+.2%}  med={d[f'ar{h}'].median():+.2%}"
              f"  t={m.tvalues.iloc[0]:+.2f}  p={m.pvalues.iloc[0]:.3f}  hit={(d[f'ar{h}']>0).mean():.0%}")


def placebo(ev, px, spy, n=800, seed=42):
    """Same tickers, same number of events, dates circularly shifted within the
    sample -- keeps each name's risk profile, destroys the timing."""
    rng = np.random.default_rng(seed)
    lo, hi = ev["date"].min(), ev["date"].max()
    span = (hi - lo).days
    means = []
    for k in range(n):
        fake = ev.copy()
        shift = {tk: int(rng.integers(30, span - 30)) for tk in fake["ticker"].unique()}
        fake["date"] = [lo + pd.Timedelta(days=(((d - lo).days + shift[t]) % span))
                        for d, t in zip(fake["date"], fake["ticker"])]
        ar = abnormal_returns(fake, px, spy)
        if len(ar) > len(ev) * 0.5:
            means.append(ar["ar63"].mean())
    means = np.array(means)
    real = ev["ar63"].mean()
    p = (means >= real).mean()
    print(f"\n[2] PLACEBO (ticker-preserving date shift, n={len(means)}): "
          f"real={real:+.2%}  null95=[{np.percentile(means,2.5):+.2%},"
          f"{np.percentile(means,97.5):+.2%}]  p={p:.3f}")


def calendar_portfolio(ev, px, spy):
    cal = px.index
    active, costs = {}, {}
    for r in ev.itertuples():
        p = px[r.ticker].dropna()
        pos = p.index.searchsorted(r.date, side="right")
        if pos + HOLD_TD >= len(p):
            continue
        a = cal.searchsorted(p.index[pos]); b = cal.searchsorted(p.index[pos + HOLD_TD])
        for d in range(a + 1, b + 1):
            active.setdefault(d, []).append(r.ticker)
        costs[a + 1] = costs.get(a + 1, 0) + 1
        costs[b] = costs.get(b, 0) + 1
    rets = px.pct_change()
    rows = []
    for d in range(min(active), max(active) + 1):
        names = active.get(d, [])
        if names:
            rs = [rets[t].iloc[d] for t in names if np.isfinite(rets[t].iloc[d])]
            g = np.mean(rs) if rs else 0.0
            c = costs.get(d, 0) * (COST_BPS / 1e4) / max(len(names), 1)
            rows.append({"date": cal[d], "ret": g - c, "n": len(names)})
        else:
            rows.append({"date": cal[d], "ret": 0.0, "n": 0})
    s = pd.DataFrame(rows).set_index("date")
    ppy = 252
    full = s["ret"]; ann = full.mean() * ppy; vol = full.std() * np.sqrt(ppy)
    cum = (1 + full).cumprod(); mdd = (cum / cum.cummax() - 1).min()
    mm = pd.concat([full.rename("s"), spy.pct_change().rename("m")], axis=1).dropna()
    reg = sm.OLS(mm["s"], sm.add_constant(mm["m"])).fit(
        cov_type="HAC", cov_kwds={"maxlags": 10})
    a_, ta = reg.params["const"] * ppy, reg.tvalues["const"]
    rng = np.random.default_rng(7); v = full.values; T = len(v); B = 21
    boots = []
    for _ in range(4000):
        st = rng.integers(0, T, T // B + 1)
        idx = np.concatenate([np.arange(x, x + B) % T for x in st])[:T]
        boots.append(v[idx].mean())
    pmean = (np.array(boots) <= 0).mean()
    print(f"\n[3] CALENDAR-TIME LONG PORTFOLIO (hold {HOLD_TD}td, {COST_BPS}bps/side)")
    print(f"  ann={ann:+.2%}  Sharpe={ann/vol if vol>0 else np.nan:+.2f}  MaxDD={mdd:.1%}  "
          f"CAPM alpha={a_:+.1%}/yr (t={ta:+.2f}, beta={reg.params['m']:.2f})  "
          f"p(mean<=0)={pmean:.3f}")
    print(f"  invested {(s['n']>0).mean():.0%} of days, avg {s[s['n']>0]['n'].mean():.1f} names")
    return s


def main():
    purch = pd.read_csv(os.path.join(HERE, "purchases.csv"),
                        parse_dates=["trans_date", "filing_date"])
    print(f"purchases: {len(purch)} rows, {purch['ticker'].nunique()} tickers")
    events = detect_events(purch)
    print(f"cluster events: {len(events)} across {events['ticker'].nunique()} tickers")
    tickers = sorted(events["ticker"].unique())
    px = get_px(tickers + ["SPY"])
    spy = px["SPY"].dropna()
    ev = abnormal_returns(events, px, spy)
    ev.to_csv(os.path.join(HERE, "events.csv"), index=False)
    event_tests(ev)
    placebo(ev, px, spy)
    s = calendar_portfolio(ev, px, spy)
    # sub-periods
    print("\n[4] SUB-PERIODS (calendar-time)")
    for lab, a, b in [("2014-2018", "2014", "2018"), ("2019-2021", "2019", "2021"),
                      ("2022-2024", "2022", "2024")]:
        seg = s.loc[a:b, "ret"]
        if len(seg) > 100:
            sh = seg.mean() / seg.std() * np.sqrt(252) if seg.std() > 0 else np.nan
            print(f"  {lab}: ann={seg.mean()*252:+.2%}  Sharpe={sh:+.2f}")
    # size split (by combined cluster value)
    print("\n[5] BY CLUSTER SIZE / CONVICTION")
    for lab, sub in [("value >= $500k", ev[ev["value"] >= 5e5]),
                     ("3+ insiders", ev[ev["n_insiders"] >= 3])]:
        if len(sub) >= 15:
            print(f"  {lab:<16} n={len(sub):>3}  AR63={sub['ar63'].mean():+.2%}  "
                  f"med={sub['ar63'].median():+.2%}  hit={(sub['ar63']>0).mean():.0%}")


if __name__ == "__main__":
    main()
