"""
TURN-OF-MONTH (TOM) -- one of the only calendar anomalies that persisted for
decades AFTER publication (Lakonishok-Smidt 1988; McConnell-Xu 2008: all of the
equity premium sits in days -1..+3 around month-end; flow-driven -- payroll /
401k / pension contributions hit on a schedule).

PRE-COMMITTED (literature spec, no fitting):
  TOM window = last trading session of the month through the first 3 sessions
  of the next month (4 sessions). Strategy: hold equity only in the window,
  T-bills otherwise. 5 bps/side (SPY-class liquidity), 2 trades/month.
Tested on VFINX daily back to 1987 AND SPY 2004-2024 (two instruments).
"""
import warnings; warnings.filterwarnings("ignore")
import os, sys, time
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import data as D
import multi_asset as MA

COST = 5 / 1e4

def daily(tk, start):
    fp = os.path.join(HERE, "cache", f"tom_{tk}.csv")
    if os.path.exists(fp):
        s = pd.read_csv(fp, index_col=0, parse_dates=True).iloc[:, 0]
    else:
        s = D._fetch_one(tk, start, "2024-12-31"); time.sleep(0.2)
        s.to_frame().to_csv(fp)
    return s.dropna()

def tom_mask(idx):
    """True on the last session of each month and the first 3 of the next."""
    m = pd.Series(False, index=idx)
    mon = idx.to_period("M")
    for p in mon.unique():
        days = idx[mon == p]
        m[days[-1:]] = True          # last session of month p
        m[days[:3]] = True           # first 3 sessions of month p
    return m

def evaluate(tk, start):
    px = daily(tk, start)
    r = px.pct_change().dropna()
    mask = tom_mask(r.index)
    rf_m = MA.get_rf_monthly(r.index.to_period("M").to_timestamp("M").unique())
    rf_d = (r.index.to_period("M").to_timestamp("M")).map(rf_m) / 21.0
    ann = 252
    tom, rest = r[mask], r[~mask]
    print(f"\n{tk} {r.index.min().date()} -> {r.index.max().date()} "
          f"({len(r)} days; {mask.mean():.0%} in TOM window)")
    print(f"  mean daily ret  TOM={tom.mean()*1e4:+6.1f} bp   "
          f"rest={rest.mean()*1e4:+6.1f} bp")
    # difference-of-means t (Welch)
    se = np.sqrt(tom.var()/len(tom) + rest.var()/len(rest))
    t = (tom.mean() - rest.mean()) / se
    print(f"  TOM - rest = {(tom.mean()-rest.mean())*1e4:+.1f} bp/day  t={t:+.2f}")
    # cumulative share of total return
    tot = (1 + r).prod() - 1
    tom_only = (1 + tom).prod() - 1
    print(f"  cumulative: all days={tot:+,.0%}   TOM days only={tom_only:+,.0%}")
    # tradable version
    strat = r.where(mask, pd.Series(rf_d, index=r.index))
    # costs: enter at close before window, exit at close of last window day
    trades = mask.astype(int).diff().abs().fillna(0)
    strat = strat - trades * COST
    for name, s in [("TOM-only strategy (net)", strat), (f"{tk} buy & hold", r)]:
        cagr = (1 + s).prod() ** (ann / len(s)) - 1
        vol = s.std() * np.sqrt(ann)
        sh = (s.mean() * ann) / vol if vol > 0 else np.nan
        cum = (1 + s).cumprod(); mdd = (cum / cum.cummax() - 1).min()
        print(f"  {name:<26} CAGR={cagr:+.1%}  Vol={vol:.1%}  Sharpe={sh:+.2f}  MaxDD={mdd:.1%}")
    # sub-period stability of the daily edge
    print("  sub-period TOM-rest edge (bp/day):")
    for lab, a, b in [("first half", None, None), ("2010-2024", "2010", "2024")]:
        seg = r.loc[a:b] if a else r.loc[:r.index[len(r)//2]]
        mk = mask.reindex(seg.index)
        d_ = (seg[mk].mean() - seg[~mk].mean()) * 1e4
        se_ = np.sqrt(seg[mk].var()/mk.sum() + seg[~mk].var()/(~mk).sum())
        print(f"    {lab:<11} {d_:+.1f} bp  t={(seg[mk].mean()-seg[~mk].mean())/se_:+.2f}")

def main():
    print("=" * 74)
    print("  TURN-OF-MONTH TEST (pre-committed window: last session + first 3)")
    print("=" * 74)
    evaluate("VFINX", "1986-01-01")
    evaluate("SPY", "2004-01-01")

if __name__ == "__main__":
    main()
