"""
LIVE SQUEEZE WATCHLIST — runs the certified V2 monitor rule on TODAY's data.

This is the operational counterpart of monitor.py's backtest: same universe
(minus banned names), same pre-committed thresholds, but pointed at the most
recent published FINRA short-interest report and this week's Wikipedia
attention. Run it weekly (e.g. Monday); it prints the watchlist and writes
sq_live.html.

Rule (V2, from monitor.py — thresholds unchanged, nothing re-tuned):
  CROWDED : latest PUBLISHED short interest >= 75th percentile of the name's
            own trailing ~2y of published reports (volume-free).
  SPIKE   : weekly Wikipedia views >= 2x trailing 8-week median (weekly ASVI),
            OR daily views >= 3x trailing 56-day median (daily timing).
  FLAG    = CROWDED and SPIKE.    WATCH = CROWDED only.

Fresh data: caches are namespaced by run date, so each new day refetches.
Backtest caches in cache/ are untouched.
"""
import warnings; warnings.filterwarnings("ignore")
import os, sys, time, html
import datetime as dt
import numpy as np, pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_HERE, "..", "attention_alpha"))

from sq_universe import TICKER_ARTICLE, BANNED
from short_interest import _fetch as finra_fetch
from wiki_data import _fetch_article
from monitor import SI_PCTL, DAILY_SPIKE_X, ASVI_MIN, MED_WEEKS

LIVE_CACHE = os.path.join(_HERE, "cache", "live")
os.makedirs(LIVE_CACHE, exist_ok=True)
TODAY = dt.date.today()
OUT_HTML = os.path.join(_HERE, "sq_live.html")


def live_si(tk):
    fp = os.path.join(LIVE_CACHE, f"si_{tk}_{TODAY:%Y%m%d}.csv")
    if os.path.exists(fp):
        return pd.read_csv(fp, parse_dates=["settlementDate", "avail"])
    df = finra_fetch(tk)
    time.sleep(0.3)
    if df is not None:
        df.to_csv(fp, index=False)
    return df


def live_views(article, days=430):
    fn = article.replace("/", "_").replace("'", "").replace("&", "and")
    fp = os.path.join(LIVE_CACHE, f"pv_{fn}_{TODAY:%Y%m%d}.csv")
    if os.path.exists(fp):
        return pd.read_csv(fp, index_col=0, parse_dates=True).iloc[:, 0]
    start = (TODAY - dt.timedelta(days=days)).strftime("%Y%m%d")
    s = _fetch_article(article, start, TODAY.strftime("%Y%m%d"))
    time.sleep(0.2)
    if s is not None and len(s):
        s.to_frame().to_csv(fp)
    return s


def assess(tk, article):
    """One name -> dict of current crowding + attention readings, or None."""
    sdf = live_si(tk)
    if sdf is None or len(sdf) < 21:
        return None
    now = pd.Timestamp(TODAY)
    pub = sdf[sdf["avail"] <= now].tail(48)
    if len(pub) < 20:
        return None
    cur = pub["si"].iloc[-1]
    si_pctl = float((pub["si"].iloc[:-1] < cur).mean())
    dtc = float(pub["dtc"].iloc[-1])
    settle = pub["settlementDate"].iloc[-1].date()

    pv = live_views(article)
    if pv is None or len(pv) < 120:
        return None
    # weekly ASVI on the last COMPLETE week
    wk = pv.resample("W-FRI").mean()
    complete = wk[wk.index <= now]                     # current partial week excluded below
    if len(complete) and complete.index[-1].date() >= TODAY:
        complete = complete.iloc[:-1]
    logv = np.log(complete.clip(lower=1))
    asvi = float((logv - logv.shift(1).rolling(MED_WEEKS).median()).iloc[-1])
    # daily ratio on the most recent reported day
    ratio = (pv / pv.shift(1).rolling(56).median()).dropna()
    day_ratio = float(ratio.iloc[-1])
    day_date = ratio.index[-1].date()

    crowded = si_pctl >= SI_PCTL
    wk_spike = asvi >= ASVI_MIN
    dy_spike = day_ratio >= DAILY_SPIKE_X
    status = ("FLAG" if crowded and (wk_spike or dy_spike)
              else "WATCH" if crowded else "quiet")
    return dict(ticker=tk, article=article, status=status, si_pctl=si_pctl,
                dtc=dtc, settle=str(settle), asvi=asvi,
                views_x_weekly=float(np.exp(asvi)), day_ratio=day_ratio,
                day_date=str(day_date), crowded=crowded,
                wk_spike=wk_spike, dy_spike=dy_spike)


def render_html(rows, skipped):
    rank = {"FLAG": 0, "WATCH": 1, "quiet": 2}
    rows = sorted(rows, key=lambda r: (rank[r["status"]], -r["si_pctl"]))
    css = """
    :root{--sf:#fcfcfb;--pg:#f9f9f7;--ink:#0b0b0b;--ink2:#52514e;--mut:#898781;
    --grid:#e1e0d9;--ring:rgba(11,11,11,.1);--flag:#d03b3b;--watch:#eda100;--ok:#898781}
    @media (prefers-color-scheme:dark){:root{--sf:#1a1a19;--pg:#0d0d0d;--ink:#fff;
    --ink2:#c3c2b7;--grid:#2c2c2a;--ring:rgba(255,255,255,.1);--flag:#e66767;--watch:#c98500}}
    *{box-sizing:border-box;margin:0}
    body{background:var(--pg);color:var(--ink);font:14px/1.5 system-ui,sans-serif;padding:20px}
    h1{font-size:19px}.sub{color:var(--ink2);font-size:13px;margin:2px 0 14px}
    .card{background:var(--sf);border:1px solid var(--ring);border-radius:10px;padding:14px 16px}
    table{width:100%;border-collapse:collapse;font-variant-numeric:tabular-nums}
    th{color:var(--mut);font-size:11px;text-transform:uppercase;text-align:right;
       padding:6px 8px;border-bottom:1px solid var(--grid)}
    th:first-child,td:first-child,th:nth-child(2),td:nth-child(2){text-align:left}
    td{padding:6px 8px;text-align:right;border-bottom:1px solid var(--grid);font-size:13px}
    .b{display:inline-block;padding:1px 9px;border-radius:9px;color:#fff;font-weight:700;font-size:11px}
    .note{color:var(--mut);font-size:12px;margin-top:10px}
    """
    trs = ""
    for r in rows:
        bcol = {"FLAG": "var(--flag)", "WATCH": "var(--watch)", "quiet": "var(--ok)"}[r["status"]]
        spike_txt = (("weekly %.1fx " % r["views_x_weekly"]) if r["wk_spike"] else "") + \
                    (("daily %.1fx" % r["day_ratio"]) if r["dy_spike"] else "")
        trs += (f"<tr><td><b>{r['ticker']}</b></td>"
                f"<td><span class='b' style='background:{bcol}'>{r['status']}</span></td>"
                f"<td>{r['si_pctl']:.0%}</td><td>{r['dtc']:.1f}</td><td>{r['settle']}</td>"
                f"<td>{r['views_x_weekly']:.2f}x</td><td>{r['day_ratio']:.2f}x</td>"
                f"<td style='text-align:left;color:var(--ink2)'>{html.escape(spike_txt) or '—'}</td></tr>")
    body = f"""<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Squeeze Watchlist — {TODAY}</title><style>{css}</style>
<h1>Live squeeze watchlist — {TODAY}</h1>
<div class="sub">V2 rule on today's published FINRA short interest + Wikipedia attention ·
{len(rows)} names scored{f" · {len(skipped)} skipped (thin data): {', '.join(skipped)}" if skipped else ""}</div>
<div class="card"><table>
<tr><th>Name</th><th>Status</th><th>SI pctile</th><th>DTC</th><th>SI settle</th>
<th>Wk views</th><th>Day views</th><th>Spike detail</th></tr>{trs}</table>
<div class="note">FLAG = crowded (short interest ≥ 75th pctile of own trailing 2y of
published reports) AND attention spike (weekly ≥ 2x 8-wk median, or daily ≥ 3x 56-day
median). WATCH = crowded, attention normal. Historical base rates (2018–24 backtest,
banned names excluded): flagged crowded names were ~2.2x more likely to rally ≥ +25%
within 4 weeks (12.0% vs 5.5%, week-cluster bootstrap p = 0.032). A FLAG is a
risk-management prompt for anyone short the name, not a buy signal. Short-interest
data is twice-monthly with a 9-business-day publication lag — the SI settle column
shows how fresh the crowding reading is. GME and AMC excluded (restricted names).</div>
</div>"""
    with open(OUT_HTML, "w") as f:
        f.write("<!doctype html><html><head></head><body>" + body + "</body></html>")


def main():
    names = {tk: a for tk, a in TICKER_ARTICLE.items() if tk not in BANNED}
    print(f"live squeeze watchlist — {TODAY}  ({len(names)} names, banned excluded)")
    rows, skipped = [], []
    for tk, article in sorted(names.items()):
        r = assess(tk, article)
        if r is None:
            skipped.append(tk)
            continue
        rows.append(r)
        mark = {"FLAG": "**", "WATCH": " .", "quiet": "  "}[r["status"]]
        print(f" {mark} {tk:<6} {r['status']:<6} SIpctl={r['si_pctl']:>4.0%} "
              f"dtc={r['dtc']:>5.1f} settle={r['settle']}  "
              f"wk={r['views_x_weekly']:>5.2f}x  day={r['day_ratio']:>5.2f}x")
    n_flag = sum(r["status"] == "FLAG" for r in rows)
    n_watch = sum(r["status"] == "WATCH" for r in rows)
    print(f"\n{n_flag} FLAG, {n_watch} WATCH, {len(rows)-n_flag-n_watch} quiet"
          + (f"; skipped: {', '.join(skipped)}" if skipped else ""))
    render_html(rows, skipped)
    print(f"wrote {OUT_HTML}")


if __name__ == "__main__":
    main()
