"""
Cross-sectional momentum backtest on US single stocks — honest evaluation.

Strategy: classic 12-1 momentum (Jegadeesh & Titman 1993).
  - Monthly rebalance. Signal at month-end m = P[m-1] / P[m-13] - 1
    (12-month return, skipping the most recent month to avoid short-term
    reversal). Strictly uses only past data; positions held the FOLLOWING month.
  - Portfolios:
      LO  = long-only top quintile, equal weight   (vs SPY — the user's goal)
      LS  = long top quintile, short bottom quintile, dollar-neutral
      EW  = equal-weight whole universe             (survivorship control)
      SPY = buy & hold benchmark
  - Transaction costs charged on turnover each rebalance.

Honest controls:
  - EW-universe benchmark isolates *selection* skill from survivorship: if LO
    can't beat EW, momentum added nothing beyond "these survivors rose".
  - FF5 + UMD regression on LS: if alpha ~ 0, we're harvesting the momentum
    premium, not generating independent alpha (still fine for beating SPY).
"""
import os, warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import data as D

COST_BPS = 10          # per side, on turnover
N_QUINTILE = 5
START_TRADE = "2010-08-01"

def monthly_prices():
    px = D.get_prices(verbose=False)
    spy = px[D.BENCHMARK]
    stocks = px.drop(columns=[D.BENCHMARK])
    m_stocks = stocks.resample("ME").last()
    m_spy = spy.resample("ME").last()
    return m_stocks, m_spy

def momentum_signal(mpx):
    """12-1 momentum: P[m-1]/P[m-13]-1, available at end of month m."""
    return mpx.shift(1) / mpx.shift(13) - 1.0

def lowvol_signal(mret):
    """Low-volatility: -trailing 12-month stdev of monthly returns (higher = calmer).
    Uses returns through month m only; ranked long the calmest names."""
    return -mret.rolling(12).std()

def run():
    mpx, mspy = monthly_prices()
    mret = mpx.pct_change()
    spy_ret = mspy.pct_change()
    signal = momentum_signal(mpx)
    lvsig = lowvol_signal(mret)

    dates = mret.index
    w_lo_prev = pd.Series(0.0, index=mpx.columns)
    w_ls_prev = pd.Series(0.0, index=mpx.columns)
    w_ew_prev = pd.Series(0.0, index=mpx.columns)
    w_lv_prev = pd.Series(0.0, index=mpx.columns)
    rows = []

    for i in range(1, len(dates)):
        m = dates[i - 1]          # decision at end of month m
        hold = dates[i]           # earn returns during next month
        if hold < pd.Timestamp(START_TRADE):
            continue

        sig = signal.loc[m].dropna()
        # only names with a valid next-month return
        valid = mret.loc[hold].dropna().index
        sig = sig[sig.index.isin(valid)]
        if len(sig) < 2 * N_QUINTILE:
            continue

        ranked = sig.sort_values()
        q = max(1, len(ranked) // N_QUINTILE)
        losers = ranked.index[:q]
        winners = ranked.index[-q:]

        # target weights
        w_lo = pd.Series(0.0, index=mpx.columns); w_lo[winners] = 1.0 / len(winners)
        w_ls = pd.Series(0.0, index=mpx.columns)
        w_ls[winners] = 0.5 / len(winners); w_ls[losers] = -0.5 / len(losers)
        w_ew = pd.Series(0.0, index=mpx.columns); w_ew[valid] = 1.0 / len(valid)

        # low-vol long-only top quintile (calmest names), same valid set
        lv = lvsig.loc[m].dropna(); lv = lv[lv.index.isin(valid)]
        w_lv = pd.Series(0.0, index=mpx.columns)
        if len(lv) >= N_QUINTILE:
            calm = lv.sort_values().index[-max(1, len(lv) // N_QUINTILE):]
            w_lv[calm] = 1.0 / len(calm)

        r = mret.loc[hold].fillna(0.0)
        def net(w, w_prev):
            turn = (w - w_prev).abs().sum()
            return float((w * r).sum() - turn * COST_BPS / 1e4), w
        ret_lo, w_lo_prev = net(w_lo, w_lo_prev)
        ret_ls, w_ls_prev = net(w_ls, w_ls_prev)
        ret_ew, w_ew_prev = net(w_ew, w_ew_prev)
        ret_lv, w_lv_prev = net(w_lv, w_lv_prev)

        rows.append({"date": hold, "LO": ret_lo, "LS": ret_ls, "LV": ret_lv,
                     "EW": ret_ew, "SPY": float(spy_ret.loc[hold])})

    bt = pd.DataFrame(rows).set_index("date")
    return bt

def metrics(r):
    r = r.dropna()
    ann = (1 + r).prod() ** (12 / len(r)) - 1
    vol = r.std() * np.sqrt(12)
    sharpe = (r.mean() * 12) / vol if vol > 0 else np.nan
    cum = (1 + r).cumprod()
    dd = (cum / cum.cummax() - 1).min()
    return {"CAGR": ann, "Vol": vol, "Sharpe": sharpe,
            "MaxDD": dd, "Hit": (r > 0).mean()}

def factor_regression(strat_ret):
    """Regress monthly strategy returns on FF5 + UMD; report annualized alpha + t."""
    try:
        import statsmodels.api as sm
        import pandas_datareader.data as web
        ff5 = web.DataReader("F-F_Research_Data_5_Factors_2x3", "famafrench",
                             "2010-01-01", "2024-12-31")[0]
        mom = web.DataReader("F-F_Momentum_Factor", "famafrench",
                             "2010-01-01", "2024-12-31")[0]
        mom.columns = ["UMD"]
        f = ff5.join(mom, how="inner") / 100.0
        f.index = f.index.to_timestamp("M") if hasattr(f.index, "to_timestamp") \
            else pd.to_datetime(f.index)
    except Exception as e:
        return f"  (factor data unavailable: {type(e).__name__})"
    out = {}
    for name, r in strat_ret.items():
        y = r.copy(); y.index = y.index.to_period("M").to_timestamp("M")
        d = pd.concat([(y - f["RF"]).rename("ex"), f.drop(columns="RF")], axis=1).dropna()
        if len(d) < 24:
            continue
        X = sm.add_constant(d.drop(columns="ex"))
        mdl = sm.OLS(d["ex"], X).fit(cov_type="HAC", cov_kwds={"maxlags": 6})
        out[name] = (mdl.params["const"] * 12, mdl.tvalues["const"],
                     mdl.params.get("Mkt-RF", np.nan), mdl.rsquared)
    return out

if __name__ == "__main__":
    bt = run()
    bt.to_csv(os.path.join(os.path.dirname(__file__), "momentum_returns.csv"))
    print("=" * 74)
    print(f"  CROSS-SECTIONAL 12-1 MOMENTUM — US single stocks, {len(D.UNIVERSE)}-name universe")
    print(f"  {bt.index.min().date()} → {bt.index.max().date()}  ({len(bt)} months)"
          f"   cost={COST_BPS}bps/side")
    print("=" * 74)
    print(f"{'Portfolio':<34}{'CAGR':>9}{'Vol':>8}{'Sharpe':>8}{'MaxDD':>9}{'Hit':>7}")
    print("-" * 74)
    labels = {"LO": "Momentum long-only top quintile", "LV": "Low-vol long-only top quintile",
              "LS": "Momentum long-short (neutral)",
              "EW": "Equal-weight universe (control)", "SPY": "SPY buy & hold"}
    for col in ["LO", "LV", "LS", "EW", "SPY"]:
        m = metrics(bt[col])
        print(f"{labels[col]:<34}{m['CAGR']:>8.1%}{m['Vol']:>8.1%}"
              f"{m['Sharpe']:>8.2f}{m['MaxDD']:>9.1%}{m['Hit']:>7.0%}")
    print("-" * 74)
    print(f"Correlation of LO to SPY: {bt['LO'].corr(bt['SPY']):.2f}   "
          f"LS to SPY: {bt['LS'].corr(bt['SPY']):.2f}")
    print("\nFactor regression (FF5+UMD, HAC t-stats, annualized alpha):")
    fr = factor_regression(bt[["LO", "LV", "LS"]])
    if isinstance(fr, dict):
        print(f"{'':<14}{'Alpha':>9}{'t(alpha)':>10}{'Beta_Mkt':>10}{'R2':>7}")
        for name, (a, t, b, r2) in fr.items():
            print(f"  {name:<12}{a:>8.1%}{t:>10.2f}{b:>10.2f}{r2:>7.2f}")
    else:
        print(fr)
