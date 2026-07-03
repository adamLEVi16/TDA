"""
CASH-LEVEL CONDITIONING — does the AMOUNT of cash the strategy is holding
carry information, or create risk, beyond the trend signal itself?

Two questions, both new (not covered by any prior test):
  [1] Does the cash % level predict FORWARD returns — for the strategy itself,
      and for the equity market it stepped out of (i.e. is high cash usually
      "smart" timing or does it usually just sit through the recovery it
      missed)? Cash level here is a STATE variable, tested strictly
      out-of-sample: cash%% observed at month t predicts return t+1..t+h.
  [2] WHIPSAW COST: how often does the strategy re-enter cash within a few
      months of having just exited it (a "round trip")? Each round trip is
      pure cost with no crash avoided — the classic trend-following tax.
  [3] CORRELATION-REGIME risk (bonus, standard CTA objection): does the
      strategy do worse when its own asset universe's pairwise correlation
      spikes (everything falls together, trend signals stop diversifying)?
"""
import warnings; warnings.filterwarnings("ignore")
import os, sys
import numpy as np, pandas as pd, statsmodels.api as sm
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import multi_asset as MA
from backtest import metrics
from long_history import LH_ASSETS, LH_BENCH, START

sh = lambda r: r.mean() / r.std() * np.sqrt(12) if r.std() > 0 else np.nan


def main():
    bt, w = MA.run(assets=LH_ASSETS, bench=LH_BENCH, start=START)
    rp, eq = bt["RP+Trend"], bt["SPY"]
    cash = w["CASH"].clip(lower=0, upper=1).reindex(rp.index)

    print("=" * 82)
    print(f"  CASH-LEVEL CONDITIONING, {rp.index.min():%Y-%m} -> {rp.index.max():%Y-%m}  "
          f"({len(rp)} months)")
    print("=" * 82)
    print(f"  cash%% distribution: mean={cash.mean():.0%}  median={cash.median():.0%}  "
          f"p90={cash.quantile(.9):.0%}  max={cash.max():.0%}")

    # [1] forward returns by cash bin, strictly using cash%% at t to predict t+1..t+h
    print("\n[1] FORWARD RETURNS BY CASH LEVEL (cash%% at month t -> return t+1..t+h)")
    bins = [(0, .10, "0-10% (fully invested)"), (.10, .35, "10-35%"),
            (.35, .65, "35-65%"), (.65, .90, "65-90%"), (.90, 1.01, "90-100% (mostly cash)")]
    for h in (1, 3, 6, 12):
        eq_fwd = (1 + eq).rolling(h).apply(np.prod).shift(-h) - 1
        rp_fwd = (1 + rp).rolling(h).apply(np.prod).shift(-h) - 1
        print(f"\n  horizon +{h}mo:")
        for lo, hi, lab in bins:
            msk = (cash >= lo) & (cash < hi)
            n = msk.sum()
            if n < 8:
                continue
            print(f"    cash {lab:<24} n={n:>3}  strategy fwd={rp_fwd[msk].mean():+.1%}  "
                  f"equity fwd={eq_fwd[msk].mean():+.1%}  "
                  f"(equity forward Sharpe-ish t={eq_fwd[msk].mean()/eq_fwd[msk].std()*np.sqrt(12/h) if eq_fwd[msk].std()>0 else np.nan:+.2f})")

    # regression version: cash%% at t -> equity forward return (is cash level itself timing-useful?)
    print("\n  REGRESSION: cash%% at t -> equity's OWN forward return (cluster-robust monthly HAC)")
    for h in (1, 3, 6, 12):
        eq_fwd = (1 + eq).rolling(h).apply(np.prod).shift(-h) - 1
        d = pd.concat([cash.rename("cash"), eq_fwd.rename("fwd")], axis=1).dropna()
        m = sm.OLS(d["fwd"], sm.add_constant(d["cash"])).fit(cov_type="HAC", cov_kwds={"maxlags": h})
        print(f"    +{h:>2}mo: coef={m.params['cash']:+.3f}  t={m.tvalues['cash']:+.2f}  "
              f"p={m.pvalues['cash']:.3f}  (POSITIVE = high cash precedes strong equity -> a timing cost; "
              f"negative would mean cash rose before weak markets -> protective)")

    # [2] whipsaw: round trips into/out of cash within a short window
    print("\n[2] WHIPSAW COST (round trips into cash)")
    in_cash = cash >= 0.5
    entries = in_cash & ~in_cash.shift(1).fillna(False)
    exits = ~in_cash & in_cash.shift(1).fillna(False)
    entry_dates = cash.index[entries]
    exit_dates = cash.index[exits]
    short_stays = 0
    for i, ed in enumerate(entry_dates):
        nxt = exit_dates[exit_dates > ed]
        if len(nxt) and (nxt[0] - ed).days < 120:   # back out within ~4 months
            short_stays += 1
    print(f"  {len(entry_dates)} total entries into >=50% cash over {len(rp)/12:.0f} years "
          f"({len(entry_dates)/(len(rp)/12):.1f}/yr)")
    print(f"  {short_stays} were 'whipsaws' — back to <50% cash within 4 months "
          f"({short_stays/max(len(entry_dates),1):.0%} of entries)")
    # cost estimate: whipsaw months' strategy return vs what buy-and-hold-through would have done
    ws_dates = entry_dates[[i for i,ed in enumerate(entry_dates)
                            if len(exit_dates[exit_dates>ed]) and (exit_dates[exit_dates>ed][0]-ed).days<120]]
    if len(ws_dates):
        # return strategy earned during whipsaw episodes vs what staying invested would have earned
        seg_ret_strat, seg_ret_eq = [], []
        for ed in ws_dates:
            xd = exit_dates[exit_dates > ed][0]
            seg = rp.loc[ed:xd]; segE = eq.loc[ed:xd]
            seg_ret_strat.append((1+seg).prod()-1); seg_ret_eq.append((1+segE).prod()-1)
        print(f"  during whipsaw episodes: strategy avg {np.mean(seg_ret_strat):+.2%}, "
              f"equity avg {np.mean(seg_ret_eq):+.2%} (what staying invested would've earned)")

    # [3] correlation regime
    print("\n[3] CORRELATION-REGIME RISK (avg pairwise 12mo correlation of the 5 assets)")
    px = MA.load_prices(assets=LH_ASSETS, start=START)
    aret = px.resample("ME").last().pct_change()
    def avg_pairwise_corr(window):
        c = window.corr().values
        iu = np.triu_indices_from(c, k=1)
        return np.nanmean(c[iu])
    roll_corr = pd.Series({d: avg_pairwise_corr(aret.loc[:d].tail(12))
                           for d in aret.index[11::1]}).shift(1).reindex(rp.index)
    # .shift(1): month t is classified by correlation measured through t-1 only,
    # so a crash month cannot label its own regime (ex-ante classification)
    hi_corr = roll_corr >= roll_corr.median()
    for lab, msk in [("low correlation regime (bottom half)", ~hi_corr),
                     ("high correlation regime (top half)", hi_corr)]:
        seg = rp[msk.reindex(rp.index).fillna(False)]
        print(f"  {lab:<38} n={len(seg):>3}  Sharpe={sh(seg):+.2f}  "
              f"ann.ret={seg.mean()*12:+.1%}")
    print(f"  correlation regime range: {roll_corr.min():+.2f} to {roll_corr.max():+.2f}, "
          f"current-era vs full-sample median {roll_corr.median():+.2f}")


if __name__ == "__main__":
    main()
