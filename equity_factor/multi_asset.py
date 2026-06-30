"""
Multi-asset trend + risk-parity ("Global Tactical Asset Allocation").

Universe: liquid ETFs across asset classes (survivorship-FREE — real instruments).
Each month-end m (decide using data through m, hold month m+1 — no look-ahead):
  - trend filter: an asset is eligible only if its price > its 10-month SMA;
    otherwise that sleeve goes to CASH (which earns the 1-month T-bill yield);
  - risk weighting: inverse-volatility (trailing 12m) across ALL assets,
    so eligible assets keep their risk-parity weight and ineligible ones are
    parked in cash. Risk-off months => largely cash, earning T-bills.

Portfolios produced:
  RP+Trend  = inverse-vol weights, trend-to-cash, cash earns T-bills   [the strategy]
  EW+Trend  = equal-weight eligible assets, trend-to-cash (Faber GTAA)
  RP static = inverse-vol, NO trend filter (control: isolates trend's value)
  60/40     = 60% SPY / 40% IEF, monthly rebalanced
  SPY       = 100% equity buy & hold

run() returns (returns_df, weights_panel) so the presentation can show what the
strategy actually holds over time.
"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import data as D
from backtest import metrics

# ── EXACT UNIVERSE: 8 liquid ETFs, one per major return driver ────────────────
ASSET_META = [
    ("SPY", "US large-cap equity",        "Growth / equity beta"),
    ("EFA", "Developed intl equity (ex-US)","Geographic equity diversification"),
    ("EEM", "Emerging-market equity",      "Higher-beta global growth"),
    ("TLT", "20+yr US Treasuries",         "Deflation / flight-to-quality hedge"),
    ("IEF", "7-10yr US Treasuries",        "Duration, lower-vol bond ballast"),
    ("GLD", "Gold",                        "Inflation / crisis hedge, low equity corr"),
    ("DBC", "Broad commodities",           "Inflation hedge, real-asset exposure"),
    ("VNQ", "US REITs (real estate)",      "Income / real-asset, distinct cycle"),
]
ASSETS = [a[0] for a in ASSET_META]
SMA_MONTHS = 10     # ~200-day trend filter (monthly)
VOL_WIN = 12        # trailing months for inverse-vol weights
COST_BPS = 10       # per side, on turnover

def load_prices(assets=None, shift_days=0):
    assets = list(assets) if assets is not None else list(ASSETS)
    px = D.get_prices(tickers=assets, start="2005-01-01", end="2024-12-31", verbose=False)
    if shift_days:                                  # rebalance-date robustness
        px = px.shift(shift_days)
    return px.resample("ME").last().dropna()        # common history only

_RF_CACHE = None
def get_rf_monthly(index):
    """1-month T-bill (risk-free) from Ken French, aligned to `index`. 0 if offline.
    Cached on disk + in memory so parameter sweeps don't re-hit the network."""
    global _RF_CACHE
    import os
    fp = os.path.join(D.CACHE_DIR, "_rf_monthly.csv")
    if _RF_CACHE is None and os.path.exists(fp):
        _RF_CACHE = pd.read_csv(fp, index_col=0, parse_dates=True).iloc[:, 0]
    if _RF_CACHE is None:
        try:
            import pandas_datareader.data as web
            ff = web.DataReader("F-F_Research_Data_5_Factors_2x3", "famafrench",
                                "2005-01-01", "2024-12-31")[0]
            rf = (ff["RF"] / 100.0)
            rf.index = rf.index.to_timestamp("M")
            rf.to_frame("RF").to_csv(fp)
            _RF_CACHE = rf
        except Exception as e:
            print(f"  [rf] T-bill unavailable ({type(e).__name__}); cash earns 0%")
            return pd.Series(0.0, index=index)
    return _RF_CACHE.reindex(index).fillna(0.0)

def run(sma_months=SMA_MONTHS, vol_win=VOL_WIN, cost_bps=COST_BPS,
        assets=None, shift_days=0):
    assets = list(assets) if assets is not None else list(ASSETS)
    mpx = load_prices(assets=assets, shift_days=shift_days)
    # Benchmarks (SPY, IEF) use full data regardless of the strategy universe,
    # so drop-one / subset tests don't change what we compare against.
    bench = load_prices(assets=["SPY", "IEF"], shift_days=shift_days)
    idx = mpx.index.intersection(bench.index)
    mpx, bench = mpx.loc[idx], bench.loc[idx]
    mret = mpx.pct_change()
    bret = bench.pct_change()
    sma = mpx.rolling(sma_months).mean()
    vol = mret.rolling(vol_win).std()
    rf = get_rf_monthly(mpx.index)
    dates = mpx.index

    prev_rp = pd.Series(0.0, index=assets)
    prev_ew = pd.Series(0.0, index=assets)
    prev_st = pd.Series(0.0, index=assets)
    rows, wlog = [], {}

    for i in range(1, len(dates)):
        m, hold = dates[i - 1], dates[i]
        v = vol.loc[m]; tr = mpx.loc[m] > sma.loc[m]
        if v.isna().any() or sma.loc[m].isna().any():
            continue
        inv = 1.0 / v
        rp_full = inv / inv.sum()                      # static risk parity (no trend)
        w_rp = rp_full.where(tr, 0.0)                  # trend-to-cash
        on = tr[tr].index
        w_ew = pd.Series(0.0, index=assets)
        if len(on) > 0:
            w_ew[on] = 1.0 / len(assets)               # 1/N eligible, rest cash

        r = mret.loc[hold]
        rf_h = float(rf.loc[hold])
        def sleeve(w, prev):
            cash_w = max(0.0, 1.0 - w.sum())           # uninvested -> T-bills
            turn = (w - prev).abs().sum()
            return float((w * r).sum() + cash_w * rf_h - turn * cost_bps / 1e4)
        rp = sleeve(w_rp, prev_rp); prev_rp = w_rp
        ew = sleeve(w_ew, prev_ew); prev_ew = w_ew
        st = sleeve(rp_full, prev_st); prev_st = rp_full

        spy_r = float(bret.loc[hold, "SPY"])
        rows.append({"date": hold, "RP+Trend": rp, "EW+Trend": ew,
                     "RP static": st, "SPY": spy_r,
                     "60/40": 0.6 * spy_r + 0.4 * float(bret.loc[hold, "IEF"])})
        wlog[hold] = w_rp.copy()

    bt = pd.DataFrame(rows).set_index("date")
    weights = pd.DataFrame(wlog).T
    weights["CASH"] = (1.0 - weights.sum(axis=1)).clip(lower=0)
    return bt, weights

if __name__ == "__main__":
    bt, w = run()
    print("=" * 78)
    print(f"  MULTI-ASSET TREND + RISK PARITY  —  {len(ASSETS)} ETFs, {SMA_MONTHS}m SMA, "
          f"inv-vol, cash=T-bills")
    print(f"  {bt.index.min().date()} → {bt.index.max().date()}  ({len(bt)} months)"
          f"   cost={COST_BPS}bps/side")
    print("=" * 78)
    print(f"{'Portfolio':<14}{'CAGR':>9}{'Vol':>8}{'Sharpe':>8}{'MaxDD':>9}{'Hit':>7}{'corrSPY':>9}")
    print("-" * 78)
    for col in ["RP+Trend", "EW+Trend", "RP static", "60/40", "SPY"]:
        mm = metrics(bt[col]); c = bt[col].corr(bt["SPY"])
        print(f"{col:<14}{mm['CAGR']:>8.1%}{mm['Vol']:>8.1%}{mm['Sharpe']:>8.2f}"
              f"{mm['MaxDD']:>9.1%}{mm['Hit']:>7.0%}{c:>9.2f}")
    print("-" * 78)
