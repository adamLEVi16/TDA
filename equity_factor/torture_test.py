"""
Torture tests for the multi-asset trend strategy — try hard to KILL it.

We do NOT trust a single backtest. Each test below attacks a different way the
0.86 vs 0.71 result could be a mirage:
  1. Sharpe-difference significance  (block bootstrap CI + p-value)
  2. Deflated Sharpe                  (haircut for trying many strategies)
  3. Drop-one-asset / drop-pillars    (is it just a bet on gold or bonds?)
  4. Rebalance-date robustness        (month-end luck?)
  5. Static-diversified benchmark     (does the trend TIMING add value, risk-matched?)
"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from scipy import stats
import multi_asset as MA
from backtest import metrics

def sharpe(r): return r.mean() / r.std() * np.sqrt(12) if r.std() > 0 else np.nan

def block_bootstrap_sharpe_diff(a, b, block=6, n=5000, seed=1):
    """Circular block bootstrap of Sharpe(a) - Sharpe(b), resampling months
    JOINTLY (preserves the cross-correlation between strategy and benchmark)."""
    rng = np.random.default_rng(seed)
    a, b = a.values, b.values
    T = len(a); nblocks = int(np.ceil(T / block))
    diffs = np.empty(n)
    for k in range(n):
        starts = rng.integers(0, T, nblocks)
        idx = np.concatenate([(np.arange(s, s + block) % T) for s in starts])[:T]
        sa = a[idx]; sb = b[idx]
        da = sa.mean()/sa.std()*np.sqrt(12) if sa.std()>0 else 0
        db = sb.mean()/sb.std()*np.sqrt(12) if sb.std()>0 else 0
        diffs[k] = da - db
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    p = (diffs <= 0).mean()           # one-sided: P(strategy not better)
    return diffs.mean(), lo, hi, p

def deflated_sharpe(sr, T, n_trials, skew=0, kurt=3):
    """Bailey & Lopez de Prado deflated Sharpe: prob the TRUE Sharpe > 0 given we
    picked the best of `n_trials` strategies. sr = observed (monthly-annualized)."""
    sr_m = sr / np.sqrt(12)                                  # to per-period
    # expected max of n_trials standard-normal Sharpes (under null SR=0)
    e_max = (1 - np.euler_gamma) * stats.norm.ppf(1 - 1.0/n_trials) + \
            np.euler_gamma * stats.norm.ppf(1 - 1.0/(n_trials*np.e))
    sr0_m = e_max / np.sqrt(T)                               # benchmark Sharpe threshold
    num = (sr_m - sr0_m) * np.sqrt(T - 1)
    den = np.sqrt(1 - skew*sr_m + (kurt-1)/4*sr_m**2)
    return stats.norm.cdf(num / den), sr0_m*np.sqrt(12)

def main():
    bt, _ = MA.run()
    rp, spy, sixty = bt["RP+Trend"], bt["SPY"], bt["60/40"]
    T = len(rp)
    print("="*72)
    print(f"  TORTURE TESTS — RP+Trend, {bt.index.min().date()}→{bt.index.max().date()} ({T} mo)")
    print("="*72)

    # 1. Significance of the Sharpe edge
    print("\n[1] SHARPE-DIFFERENCE SIGNIFICANCE (block bootstrap, 5000x)")
    for name, bench in [("SPY", spy), ("60/40", sixty)]:
        d, lo, hi, p = block_bootstrap_sharpe_diff(rp, bench)
        sig = "SIGNIFICANT" if (lo > 0 or hi < 0) else "NOT significant (CI spans 0)"
        print(f"  RP+Trend - {name:>5}: ΔSharpe={d:+.2f}  95% CI [{lo:+.2f}, {hi:+.2f}]"
              f"  p(not better)={p:.2f}  -> {sig}")

    # 2. Deflated Sharpe (we tried ~6 strategy families this project)
    print("\n[2] DEFLATED SHARPE (haircut for multiple testing)")
    for n_trials in [1, 6, 20]:
        dsr, thr = deflated_sharpe(sharpe(rp), T, max(n_trials,1))
        print(f"  if {n_trials:>2} strategies tried: P(true Sharpe>0)={dsr:.2f}"
              f"   (chance-Sharpe threshold ~{thr:.2f})")

    # 3. Drop-one-asset and drop-pillars
    print("\n[3] DROP-ONE-ASSET (does it depend on any single sleeve?)")
    base = sharpe(rp)
    res = []
    for a in MA.ASSETS:
        sub = [x for x in MA.ASSETS if x != a]
        b2, _ = MA.run(assets=sub); res.append((a, sharpe(b2["RP+Trend"])))
    for a, s in sorted(res, key=lambda x: x[1]):
        print(f"  without {a:<4}: Sharpe={s:.2f}  ({s-base:+.2f})")
    for label, drop in [("gold (GLD)", ["GLD"]),
                        ("all bonds (TLT,IEF)", ["TLT","IEF"]),
                        ("all commodities+gold (DBC,GLD)", ["DBC","GLD"])]:
        sub = [x for x in MA.ASSETS if x not in drop]
        b2, _ = MA.run(assets=sub)
        print(f"  without {label:<28}: Sharpe={sharpe(b2['RP+Trend']):.2f}")

    # 4. Rebalance-date robustness
    print("\n[4] REBALANCE-DATE ROBUSTNESS (shift rebalance ±days)")
    for sd in [-10, -5, 0, 5, 10]:
        b2, _ = MA.run(shift_days=sd)
        print(f"  shift {sd:+3d}d: Sharpe={sharpe(b2['RP+Trend']):.2f}  "
              f"MaxDD={metrics(b2['RP+Trend'])['MaxDD']:.1%}")

    # 5. Does the trend TIMING add value vs static-diversified, risk-matched?
    print("\n[5] TREND TIMING vs STATIC DIVERSIFICATION")
    print(f"  RP static (no trend): Sharpe={sharpe(bt['RP static']):.2f}, "
          f"MaxDD={metrics(bt['RP static'])['MaxDD']:.1%}")
    print(f"  RP + trend          : Sharpe={sharpe(rp):.2f}, "
          f"MaxDD={metrics(rp)['MaxDD']:.1%}")
    d, lo, hi, p = block_bootstrap_sharpe_diff(rp, bt["RP static"])
    print(f"  ΔSharpe (trend vs static) = {d:+.2f}  95% CI [{lo:+.2f},{hi:+.2f}]  p={p:.2f}")

if __name__ == "__main__":
    main()
