"""
Convert the squeeze flag into an actual P&L -- the "is there alpha" test.

STRATEGY (mechanical, from the pre-committed V1 flag):
  When a name flags on Friday t (crowded short + attention spike), BUY it at the
  next session's close, hold 20 trading days, equal-weight across all active
  positions, 10 bps/side. Idle when nothing is flagged (cash earns 0 here --
  conservative). This is the LONG side of the tail statistic: flagged crowded
  shorts ripped >= +25% within 4 weeks 2.3x as often as unflagged ones.

EVALUATION:
  - full-period metrics (incl. idle cash days) AND when-invested metrics
  - CAPM alpha vs SPY (daily, HAC), block-bootstrap p on the mean
  - concentration checks: exclude GME/AMC (banned at the target fund),
    drop-best-single-name, drop-2021 (the meme year)
  - exposure: fraction of days invested, avg #positions
"""
import warnings; warnings.filterwarnings("ignore")
import os, sys
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "equity_factor"))
sys.path.insert(0, os.path.join(_HERE, "..", "attention_alpha"))
import numpy as np, pandas as pd
import statsmodels.api as sm
import price_history as PH
import monitor as MO
from sq_universe import TICKER_ARTICLE, BANNED

HOLD_D, COST_BPS = 20, 10


def build_pnl(panel, px, exclude=()):
    """Daily strategy returns from flag events. Entry: close of first session
    AFTER the flag Friday; hold HOLD_D sessions; equal weight actives daily.
    One position per name at a time: a re-flag while a position is already
    open is ignored (no window stacking -- a name that flags on consecutive
    weeks must NOT be averaged in twice on overlapping days)."""
    cal = px.index
    events = panel[panel["flag"] & ~panel["ticker"].isin(exclude)].sort_values("week")
    # active[day_index] -> list of tickers held that day
    active = {}
    entries = []
    open_until = {}                                     # ticker -> last held day idx
    for r in events.itertuples():
        pos = cal.searchsorted(r.week, side="right")   # first session after Friday t
        if pos + 1 + HOLD_D >= len(cal):
            continue
        if open_until.get(r.ticker, -1) >= pos:         # position already open
            continue
        entry = pos                                     # buy at close of cal[pos]
        for d in range(entry + 1, entry + 1 + HOLD_D):  # earn from next session
            active.setdefault(d, []).append(r.ticker)
        open_until[r.ticker] = entry + HOLD_D
        entries.append((entry, r.ticker))
    rets = px.pct_change()
    cost_by_day = {}
    for d, tk in entries:
        cost_by_day[d + 1] = cost_by_day.get(d + 1, 0) + 1     # entry cost day
        cost_by_day[d + HOLD_D] = cost_by_day.get(d + HOLD_D, 0) + 1  # exit
    rows = []
    lo = min(active) if active else 0
    hi = max(active) if active else 0
    for d in range(lo, hi + 1):
        names = active.get(d, [])
        if names:
            rs = [rets[t].iloc[d] for t in names if np.isfinite(rets[t].iloc[d])]
            gross = np.mean(rs) if rs else 0.0
            cost = cost_by_day.get(d, 0) * (COST_BPS / 1e4) / max(len(names), 1)
            rows.append({"date": cal[d], "ret": gross - cost, "n": len(names)})
        else:
            rows.append({"date": cal[d], "ret": 0.0, "n": 0})
    return pd.DataFrame(rows).set_index("date"), len(entries)


def perf_block(s, spy, label):
    ppy = 252
    full = s["ret"]
    inv = s[s["n"] > 0]["ret"]
    ann = full.mean() * ppy; vol = full.std() * np.sqrt(ppy)
    sh = ann / vol if vol > 0 else np.nan
    cum = (1 + full).cumprod(); mdd = (cum / cum.cummax() - 1).min()
    # CAPM alpha vs SPY on full period
    m = pd.concat([full.rename("s"), spy.pct_change().rename("m")], axis=1).dropna()
    reg = sm.OLS(m["s"], sm.add_constant(m["m"])).fit(cov_type="HAC", cov_kwds={"maxlags": 10})
    a, ta, beta = reg.params["const"] * ppy, reg.tvalues["const"], reg.params["m"]
    # block bootstrap p on mean (20d blocks)
    rng = np.random.default_rng(17); v = full.values; T = len(v); B = 20
    boots = []
    for _ in range(4000):
        st = rng.integers(0, T, T // B + 1)
        idx = np.concatenate([np.arange(x, x + B) % T for x in st])[:T]
        boots.append(v[idx].mean())
    p = (np.array(boots) <= 0).mean()
    print(f"  {label:<34} ann={ann:+7.2%}  Sharpe={sh:+5.2f}  MaxDD={mdd:6.1%}  "
          f"alpha={a:+6.1%}/yr (t={ta:+.2f}, beta={beta:.2f})  p(mean<=0)={p:.3f}")
    print(f"  {'':<34} invested {float((s['n']>0).mean()):.0%} of days, "
          f"avg {s[s['n']>0]['n'].mean():.1f} names when active, "
          f"when-invested ann={inv.mean()*ppy:+.1%}")
    return ann, sh, a, ta, p


def main():
    panel = MO.build_panel(verbose=False)
    px = PH.get_prices_long(list(TICKER_ARTICLE.keys()) + ["SPY"], verbose=False)
    spy = px["SPY"]
    print("=" * 96)
    print("  FLAG -> LONG P&L (buy flagged crowded shorts, hold 20 sessions, "
          f"{COST_BPS} bps/side)")
    print("=" * 96)

    s, n = build_pnl(panel, px)
    print(f"\n[1] FULL UNIVERSE ({n} events)")
    perf_block(s, spy, "long flagged, all names")

    s2, n2 = build_pnl(panel, px, exclude=BANNED)
    print(f"\n[2] EX GME/AMC -- the tradable version at the target fund ({n2} events)")
    perf_block(s2, spy, "long flagged, ex GME/AMC")

    print("\n[3] CONCENTRATION / ROBUSTNESS (ex GME/AMC base)")
    # drop 2021 (meme year)
    s3 = s2[(s2.index < "2021-01-01") | (s2.index > "2021-12-31")]
    perf_block(s3, spy, "ex GME/AMC, excluding 2021")
    # drop the single best contributing name
    ev = panel[panel["flag"] & ~panel["ticker"].isin(BANNED)]
    contrib = ev.groupby("ticker")["fwd4"].sum().sort_values()
    best = contrib.index[-1]
    s4, _ = build_pnl(panel, px, exclude=BANNED + (best,))
    perf_block(s4, spy, f"ex GME/AMC and best name ({best})")

    print("\n[4] EVENT-LEVEL VIEW (ex GME/AMC): per-event 4wk fwd returns")
    f = ev["fwd4"]
    print(f"  n={len(f)}  mean={f.mean():+.1%}  median={f.median():+.1%}  "
          f"hit(>0)={float((f>0).mean()):.0%}  P(>=+15%)={float((f>=.15).mean()):.0%}  "
          f"worst={f.min():+.1%}  best={f.max():+.1%}")
    yr = ev.groupby(ev["week"].dt.year)["fwd4"].agg(["count", "mean"])
    print("  by year:", {int(y): f"n={int(r['count'])},{r['mean']:+.1%}" for y, r in yr.iterrows()})


if __name__ == "__main__":
    main()
