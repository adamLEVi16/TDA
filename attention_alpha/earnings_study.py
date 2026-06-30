"""
EARNINGS-DATE EVENT STUDY -- the slow, low-turnover version of the attention
signal that the weekly test pointed to.

Economic idea: attention to a consumer brand during a quarter reflects consumer
demand; demand shows up in the quarterly numbers; so cumulative abnormal
attention over the quarter should predict the stock's move AT the earnings
announcement (a beat/miss reaction), and possibly the post-earnings drift.

This is far less turnover-sensitive than the weekly pop: each name trades ~4x/yr
with a 2-day hold, so transaction costs (which killed the weekly version) are
negligible here. If an edge exists it can actually be harvested.

No look-ahead: the signal uses ONLY attention weeks strictly before the
announcement date E; the reaction is measured from the close before E forward.

Tests: pooled regression (cluster by calendar quarter) -> placebo -> a tradable
long/short-into-earnings portfolio (the decisive test).
"""
import warnings; warnings.filterwarnings("ignore")
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "equity_factor"))
import numpy as np, pandas as pd
import statsmodels.api as sm
import data as D
import wiki_data as WD
import earnings_data as ED
from test_signal import asvi
from universe import TICKER_ARTICLE

START = "2015-06-01"
PRE_WEEKS = 12          # quarter of attention before earnings
REACT_DAYS = 2          # close(E-1) -> close(E+1): captures BMO or AMC release
DRIFT_DAYS = 21         # ~1 trading month of post-earnings drift


def build_events():
    tickers = list(TICKER_ARTICLE.keys())
    px = D.get_prices(tickers=tickers + ["SPY"], start=START, verbose=False)
    pv = WD.get_pageviews(list(TICKER_ARTICLE.values()), verbose=False)
    earn = ED.get_earnings_dates(tickers, verbose=False)
    art2tk = {v: k for k, v in TICKER_ARTICLE.items()}
    asvi_by_tk = {art2tk[a]: asvi(s) for a, s in pv.items()}
    # weekly log pageviews per ticker, for a year-over-year attention-GROWTH signal
    logv_by_tk = {art2tk[a]: np.log(s.resample("W-FRI").mean().clip(lower=1))
                  for a, s in pv.items()}
    spy = px["SPY"]

    rows = []
    for tk in tickers:
        if tk not in px.columns or tk not in asvi_by_tk or tk not in earn:
            continue
        p = px[tk].dropna()
        a = asvi_by_tk[tk].dropna()
        idx = p.index
        for E in earn[tk]:
            if E < pd.Timestamp("2016-01-01") or E > pd.Timestamp("2025-01-01"):
                continue
            # entry = last trading day strictly before E
            prior = idx[idx < E]
            if len(prior) == 0:
                continue
            e_pos = idx.get_loc(prior[-1])
            if e_pos + DRIFT_DAYS >= len(idx):
                continue
            # pre-earnings attention: weeks strictly before E
            pre = a[a.index < E].tail(PRE_WEEKS)
            if len(pre) < PRE_WEEKS // 2:
                continue
            # YoY attention GROWTH: this quarter's log-views vs same window a year ago
            lv = logv_by_tk[tk]
            now = lv[lv.index < E].tail(PRE_WEEKS)
            yr_ago = lv[lv.index < E - pd.Timedelta(weeks=52)].tail(PRE_WEEKS)
            yoy = now.mean() - yr_ago.mean() if len(yr_ago) >= PRE_WEEKS // 2 else np.nan
            entry, react, drift = p.iloc[e_pos], p.iloc[e_pos + REACT_DAYS], p.iloc[e_pos + DRIFT_DAYS]
            se, sr, sd = spy.iloc[e_pos], spy.iloc[e_pos + REACT_DAYS], spy.iloc[e_pos + DRIFT_DAYS]
            reaction = (react / entry - 1) - (sr / se - 1)             # mkt-adj 2-day reaction
            postdrift = (drift / react - 1) - (sd / sr - 1)            # mkt-adj drift AFTER reaction
            rows.append({"ticker": tk, "E": E, "qtr": E.to_period("Q"),
                         "pre_attn": pre.mean(), "attn_yoy": yoy,
                         "reaction": reaction, "drift": postdrift})
    ev = pd.DataFrame(rows)
    # cross-sectional standardize the signals within each calendar quarter
    for col in ["pre_attn", "attn_yoy"]:
        ev[col + "_z"] = ev.groupby("qtr")[col].transform(
            lambda x: (x - x.mean()) / x.std() if x.std() > 0 else x * 0)
    return ev.dropna(subset=["pre_attn_z", "reaction"])


def reg(ev, yvar, xvar="pre_attn_z"):
    d = ev.dropna(subset=[yvar, xvar])
    m = sm.OLS(d[yvar], sm.add_constant(d[[xvar]])).fit(
        cov_type="cluster", cov_kwds={"groups": d["qtr"].astype(str)})
    return m, len(d)


def placebo(ev, yvar, n=2000, seed=3):
    rng = np.random.default_rng(seed)
    d = ev.dropna(subset=[yvar, "pre_attn_z"])
    real = sm.OLS(d[yvar], sm.add_constant(d[["pre_attn_z"]])).fit().params["pre_attn_z"]
    null = np.empty(n)
    y = d[yvar].values
    for k in range(n):
        x = rng.permutation(d["pre_attn_z"].values)
        null[k] = np.polyfit(x, y, 1)[0]
    p = (np.abs(null) >= abs(real)).mean()
    return real, np.percentile(null, [2.5, 97.5]), p


def ls_portfolio(ev, yvar="reaction", frac=3):
    """Each calendar quarter: long the top-1/frac pre-earnings-attention names
    (held into their earnings), short the bottom-1/frac. Return = avg mkt-adj
    reaction of longs minus shorts, one observation per quarter."""
    rows = []
    for q, g in ev.groupby("qtr"):
        if len(g) < 2 * frac:
            continue
        gs = g.sort_values("pre_attn_z")
        k = max(1, len(g) // frac)
        rows.append({"qtr": q, "ret": gs[yvar].iloc[-k:].mean() - gs[yvar].iloc[:k].mean()})
    return pd.DataFrame(rows).set_index("qtr")["ret"].dropna()


def main():
    ev = build_events()
    ev.to_csv(os.path.join(os.path.dirname(__file__), "earnings_events.csv"), index=False)
    print("=" * 80)
    print(f"  EARNINGS-DATE ATTENTION EVENT STUDY")
    print(f"  {len(ev)} events, {ev['ticker'].nunique()} tickers, "
          f"{ev['qtr'].min()} -> {ev['qtr'].max()}")
    print("=" * 80)

    print("\n[1] DOES PRE-EARNINGS ATTENTION PREDICT THE EARNINGS REACTION?")
    print("    signal = attention LEVEL (mean abnormal attention over the quarter)")
    for xv, sig in [("pre_attn_z", "level"), ("attn_yoy_z", "YoY growth")]:
        for yv, lab in [("reaction", "2d reaction"), ("drift", "drift +2..+21d")]:
            m, n = reg(ev, yv, xv)
            b, t, p = m.params[xv], m.tvalues[xv], m.pvalues[xv]
            print(f"  [{sig:<10}] -> {lab:<16} coef={b:+.4f}  t={t:+.2f}  p={p:.3f}  (n={n})")

    print("\n[2] PLACEBO (permute the signal across events)")
    for yv in ["reaction", "drift"]:
        real, ci, p = placebo(ev, yv)
        print(f"  level -> {yv:<10} real={real:+.4f}  null95=[{ci[0]:+.4f},{ci[1]:+.4f}]  p={p:.3f}")

    print("\n[3] TRADABLE LONG/SHORT INTO EARNINGS (top vs bottom attention tercile)")
    for yv, lab in [("reaction", "level, hold E-1->E+1"), ("drift", "level, hold drift")]:
        s = ls_portfolio(ev, yv)
        ann, vol = s.mean() * 4, s.std() * np.sqrt(4)
        sh = ann / vol if vol > 0 else np.nan
        rng = np.random.default_rng(5)
        bs = [np.mean(rng.choice(s.values, len(s))) for _ in range(5000)]
        p = (np.array(bs) <= 0).mean()
        print(f"  {lab:<22} per-qtr mean={s.mean():+.4f}  ann={ann:+.2%}  "
              f"Sharpe={sh:+.2f}  hit={float((s>0).mean()):.0%}  p={p:.3f}  ({len(s)} qtrs)")


if __name__ == "__main__":
    main()
