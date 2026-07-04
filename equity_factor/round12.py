"""
ROUND 12 GAUNTLET — implements EXACTLY the variants in
ROUND12_PREREGISTRATION.md (committed before this file existed). Any variant
or parameter not in that file is not tested here.

Engine note: re-implements the monthly loop from multi_asset.py with a
pluggable (scaler, weighting) pair so V0 reproduces multi_asset.run() exactly
(verified in main() as a parity check before any variant is evaluated).
"""
import warnings; warnings.filterwarnings("ignore")
import os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import multi_asset as MA
from backtest import metrics
from torture_test import block_bootstrap_sharpe_diff
from long_history import LH_ASSETS, LH_BENCH, START as LH_START

sh = lambda r: r.mean() / r.std() * np.sqrt(12) if len(r) and r.std() > 0 else np.nan


def erc_weights(cov, iters=200):
    """Equal-risk-contribution via multiplicative updates (prereg: 36m cov)."""
    n = cov.shape[0]
    w = np.ones(n) / n
    for _ in range(iters):
        mrc = cov @ w                       # marginal risk contributions
        rc = w * mrc
        target = rc.mean()
        w = w * np.sqrt(target / np.maximum(rc, 1e-12))
        w = np.clip(w, 1e-6, None)
        w = w / w.sum()
    return w


def run_variant(variant, assets=None, bench=("SPY", "IEF"), start="2005-01-01",
                sma_months=10, vol_win=12, cost_bps=10, shift_days=0):
    """variant in {'V0','V1','V2','V3'}. Returns (bt DataFrame, ann_turnover)."""
    assets = list(assets) if assets is not None else list(MA.ASSETS)
    mpx = MA.load_prices(assets=assets, start=start, shift_days=shift_days)
    bench_px = MA.load_prices(assets=list(bench), start=start, shift_days=shift_days)
    bench_px.columns = ["SPY", "IEF"]
    idx = mpx.index.intersection(bench_px.index)
    mpx, bench_px = mpx.loc[idx], bench_px.loc[idx]
    mret = mpx.pct_change()
    bret = bench_px.pct_change()
    sma10 = mpx.rolling(sma_months).mean()
    smas = {k: mpx.rolling(k).mean() for k in (3, 6, 9, 12)}   # V2 ensemble
    vol = mret.rolling(vol_win).std()
    rf = MA.get_rf_monthly(mpx.index)
    dates = mpx.index

    prev = pd.Series(0.0, index=assets)
    rows, turns = [], []
    for i in range(1, len(dates)):
        m, hold = dates[i - 1], dates[i]
        v = vol.loc[m]
        if v.isna().any() or sma10.loc[m].isna().any() or (variant == "V2" and smas[12].loc[m].isna().any()):
            continue
        # ----- base risk weights
        if variant == "V3":
            hist = mret.loc[:m].tail(36)
            if len(hist) >= 36:
                base = pd.Series(erc_weights(hist.cov().values), index=assets)
            else:
                inv = 1.0 / v; base = inv / inv.sum()
        else:
            inv = 1.0 / v; base = inv / inv.sum()
        # ----- trend scaler
        p = mpx.loc[m]
        if variant == "V1":
            s = ((p / sma10.loc[m] - 0.95) / 0.10).clip(0, 1)
        elif variant == "V2":
            s = sum((p > smas[k].loc[m]).astype(float) for k in (3, 6, 9, 12)) / 4.0
        else:                                   # V0, V3: binary 10m gate
            s = (p > sma10.loc[m]).astype(float)
        w = base * s

        r = mret.loc[hold]
        rf_h = float(rf.loc[hold])
        cash_w = max(0.0, 1.0 - w.sum())
        turn = (w - prev).abs().sum()
        ret = float((w * r).sum() + cash_w * rf_h - turn * cost_bps / 1e4)
        prev = w
        turns.append(turn)
        spy_r = float(bret.loc[hold, "SPY"])
        rows.append({"date": hold, "strat": ret, "SPY": spy_r,
                     "60/40": 0.6 * spy_r + 0.4 * float(bret.loc[hold, "IEF"])})
    bt = pd.DataFrame(rows).set_index("date")
    return bt, np.mean(turns) * 12 if turns else np.nan


def seg(r, a=None, b=None):
    return r.loc[a:b] if (a or b) else r


def main():
    print("=" * 86)
    print("  ROUND 12 GAUNTLET — specs per ROUND12_PREREGISTRATION.md (committed first)")
    print("=" * 86)

    # ---- parity check: V0 here must reproduce multi_asset.run() exactly
    bt0, _ = run_variant("V0", assets=LH_ASSETS, bench=LH_BENCH, start=LH_START)
    ref, _ = MA.run(assets=LH_ASSETS, bench=LH_BENCH, start=LH_START)
    diff = (bt0["strat"] - ref["RP+Trend"].reindex(bt0.index)).abs().max()
    print(f"\n[PARITY] V0 vs multi_asset.run(): max |diff| = {diff:.2e} "
          f"{'OK' if diff < 1e-12 else '*** MISMATCH — DO NOT TRUST RESULTS ***'}")

    # ---- V0-V3 on long history: selection / holdout
    print(f"\n{'variant':<8}{'SEL 87-05':>11}{'HOLD 06-24':>12}{'FULL':>8}{'MaxDD':>8}{'8-ETF':>8}{'8DD':>8}{'turn/yr':>9}")
    print("-" * 86)
    results = {}
    for vr in ("V0", "V1", "V2", "V3"):
        btl, turn_l = run_variant(vr, assets=LH_ASSETS, bench=LH_BENCH, start=LH_START)
        bt8, turn_8 = run_variant(vr)
        m_full = metrics(btl["strat"]); m8 = metrics(bt8["strat"])
        results[vr] = dict(btl=btl, bt8=bt8,
            sel=sh(seg(btl["strat"], None, "2005")), hold=sh(seg(btl["strat"], "2006", None)),
            full=m_full["Sharpe"], dd=m_full["MaxDD"], s8=m8["Sharpe"], dd8=m8["MaxDD"],
            turn=turn_l)
        r = results[vr]
        print(f"{vr:<8}{r['sel']:>11.2f}{r['hold']:>12.2f}{r['full']:>8.2f}"
              f"{r['dd']:>8.1%}{r['s8']:>8.2f}{r['dd8']:>8.1%}{r['turn']:>9.1%}")

    # ---- acceptance per prereg criteria
    print("\n[ACCEPTANCE] all of: sel>=V0, hold>=V0, 8ETF>=V0, DD within 3pts, turn<=1.5x")
    v0 = results["V0"]
    for vr in ("V1", "V2", "V3"):
        r = results[vr]
        checks = {
            "sel": r["sel"] >= v0["sel"], "hold": r["hold"] >= v0["hold"],
            "8etf": r["s8"] >= v0["s8"],
            "dd": (r["dd"] >= v0["dd"] - 0.03) and (r["dd8"] >= v0["dd8"] - 0.03),
            "turn": r["turn"] <= 1.5 * v0["turn"]}
        verdict = "ACCEPT (pending torture suite)" if all(checks.values()) else "REJECT"
        fails = [k for k, ok in checks.items() if not ok]
        print(f"  {vr}: {verdict}" + (f"  (failed: {', '.join(fails)})" if fails else ""))
        if all(checks.values()):
            d, lo, hi, p = block_bootstrap_sharpe_diff(r["btl"]["strat"], v0["btl"]["strat"])
            print(f"      ΔSharpe vs V0 (full 87-24): {d:+.2f}  CI[{lo:+.2f},{hi:+.2f}]  p={p:.3f}"
                  f"   [13 cumulative project trials — treat p accordingly]")

    # ---- V4: universe expansion, 8-ETF era, marginal contribution (exploratory)
    print("\n[V4] UNIVERSE EXPANSION (8-ETF era, exploratory — no long-history confirm possible)")
    for add in ("TIP", "IWM", "BWX"):
        try:
            btx, _ = run_variant("V0", assets=MA.ASSETS + [add])
            bt_base, _ = run_variant("V0")
            common = btx.index.intersection(bt_base.index)
            s_add = sh(btx["strat"].reindex(common)); s_bas = sh(bt_base["strat"].reindex(common))
            mx = metrics(btx["strat"].reindex(common)); mb = metrics(bt_base["strat"].reindex(common))
            print(f"  +{add}: Sharpe {s_bas:.2f} -> {s_add:.2f} ({s_add-s_bas:+.2f})   "
                  f"MaxDD {mb['MaxDD']:.1%} -> {mx['MaxDD']:.1%}   ({len(common)} common months)")
        except Exception as e:
            print(f"  +{add}: FAILED ({type(e).__name__}: {str(e)[:60]})")


if __name__ == "__main__":
    main()
