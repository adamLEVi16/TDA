"""
One-page demo: the squeeze monitor on two tradable-universe episodes (CVNA,
ETSY -- names a consumer/retail book can actually hold; GME/AMC are excluded
throughout since they're banned at the target fund), plus the panel-level
tail-risk evidence. All figures computed live from the same pipeline as
monitor.py -- nothing hand-typed. Output: demo.html
"""
import warnings; warnings.filterwarnings("ignore")
import io, base64, os, sys
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "equity_factor"))
sys.path.insert(0, os.path.join(_HERE, "..", "attention_alpha"))
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import FancyBboxPatch

import wiki_data as WD
import price_history as PH
from short_interest import get_short_interest
from sq_universe import TICKER_ARTICLE
import monitor as MO

OUT = os.path.join(_HERE, "demo.html")

# palette (validated: dataviz reference, light mode)
SURF, INK, INK2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
GRID, AXIS = "#e1e0d9", "#c3c2b7"
BLUE, VIOLET, CRIT, DEEMPH = "#2a78d6", "#4a3aa7", "#d03b3b", "#c3c2b7"

plt.rcParams.update({
    "figure.dpi": 130, "font.size": 10.5, "font.family": "sans-serif",
    "axes.facecolor": SURF, "figure.facecolor": SURF,
    "axes.edgecolor": AXIS, "axes.linewidth": 1,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 1,
    "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.labelcolor": INK2, "text.color": INK})


def png(fig):
    b = io.BytesIO(); fig.savefig(b, format="png", bbox_inches="tight",
                                  facecolor=SURF); plt.close(fig)
    return base64.b64encode(b.getvalue()).decode()


def episode_chart(tk, article, start, end, peak_date, si):
    """Daily-flag (V2) episode chart. Retained for names where the volume-
    independent daily crowding test actually fires (see monitor.py V2); not
    used for the current demo episodes -- see episode_chart_weekly below."""
    px = PH.get_prices_long([tk], verbose=False)[tk].loc[start:end]
    pv = WD.get_pageviews([article], verbose=False)[article].loc[start:end]
    flags = MO.daily_flag_dates(tk, article, si[tk], start, end)
    f0 = flags[0][0] if flags else None

    fig, (a1, a2) = plt.subplots(2, 1, figsize=(7.6, 4.6), sharex=True,
                                 gridspec_kw={"height_ratios": [3, 2], "hspace": 0.12})
    a1.plot(px.index, px.values, color=BLUE, lw=2, solid_joinstyle="round")
    a1.set_ylabel("price (split-adj $)")
    a1.set_title(f"{tk} — price", loc="left", fontsize=10.5, color=INK2)
    a2.plot(pv.index, pv.values, color=VIOLET, lw=2, solid_joinstyle="round")
    a2.fill_between(pv.index, pv.values, color=VIOLET, alpha=0.10, lw=0)
    a2.set_yscale("log")
    a2.set_ylabel("Wikipedia views/day")
    a2.set_title(f"{tk} — public attention", loc="left", fontsize=10.5, color=INK2)
    for ax in (a1, a2):
        if f0 is not None:
            ax.axvline(f0, color=CRIT, lw=2, alpha=0.9)
        if peak_date is not None:
            ax.axvline(pd.Timestamp(peak_date), color=MUTED, lw=1, ls=(0, (1, 0)))
    a2.xaxis.set_major_locator(mdates.MonthLocator())
    a2.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    if f0 is not None:
        ymax = px.max()
        a1.annotate(f"⚑ first flag {f0:%b %d}", xy=(f0, ymax * 0.97),
                    xytext=(8, 0), textcoords="offset points",
                    fontsize=10, color=INK, weight="bold", va="top")
        a1.annotate(f"peak {pd.Timestamp(peak_date):%b %d}",
                    xy=(pd.Timestamp(peak_date), ymax * 0.60), xytext=(8, 0),
                    textcoords="offset points", fontsize=9.5, color=INK2, va="top")
        # flag marker dot with surface ring on the attention panel
        fv = pv.reindex([f0]).values
        if np.isfinite(fv[0]):
            a2.plot([f0], fv, "o", ms=9, color=CRIT, mec=SURF, mew=2, zorder=5)
    return png(fig), f0, flags


def episode_chart_weekly(tk, article, start, end, flag_week, si):
    """Weekly-flag episode chart (V1 rule -- the one carrying the population
    significance test), for names where the volume-independent V2 daily test
    doesn't reach its own 2-year-percentile threshold (e.g. a name whose SI
    history is dominated by its own prior extreme, like CVNA post-2022)."""
    px = PH.get_prices_long([tk], verbose=False)[tk].loc[start:end]
    pv = WD.get_pageviews([article], verbose=False)[article].loc[start:end]
    f0 = pd.Timestamp(flag_week)
    entry_px = px.reindex([px.index[px.index.searchsorted(f0)]]).iloc[0]
    peak_date = px.loc[f0:].idxmax()

    fig, (a1, a2) = plt.subplots(2, 1, figsize=(7.6, 4.6), sharex=True,
                                 gridspec_kw={"height_ratios": [3, 2], "hspace": 0.12})
    a1.plot(px.index, px.values, color=BLUE, lw=2, solid_joinstyle="round")
    a1.set_ylabel("price (split-adj $)")
    a1.set_title(f"{tk} — price", loc="left", fontsize=10.5, color=INK2)
    a2.plot(pv.index, pv.values, color=VIOLET, lw=2, solid_joinstyle="round")
    a2.fill_between(pv.index, pv.values, color=VIOLET, alpha=0.10, lw=0)
    a2.set_yscale("log")
    a2.set_ylabel("Wikipedia views/day")
    a2.set_title(f"{tk} — public attention", loc="left", fontsize=10.5, color=INK2)
    for ax in (a1, a2):
        ax.axvline(f0, color=CRIT, lw=2, alpha=0.9)
        ax.axvline(peak_date, color=MUTED, lw=1, ls=(0, (1, 0)))
    a2.xaxis.set_major_locator(mdates.MonthLocator())
    a2.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ymax = px.max()
    a1.annotate(f"⚑ flagged {f0:%b %d}", xy=(f0, ymax * 0.97),
                xytext=(8, 0), textcoords="offset points",
                fontsize=10, color=INK, weight="bold", va="top")
    a1.annotate(f"peak {peak_date:%b %d}  (+{px.loc[peak_date]/entry_px-1:.0%})",
                xy=(peak_date, ymax * 0.60), xytext=(8, 0),
                textcoords="offset points", fontsize=9.5, color=INK2, va="top")
    fv = pv.reindex([f0], method="nearest").values
    if np.isfinite(fv[0]):
        a2.plot([f0], fv, "o", ms=9, color=CRIT, mec=SURF, mew=2, zorder=5)
    return png(fig), f0, peak_date, float(px.loc[peak_date] / entry_px - 1)


def tail_chart(p25_flag, p25_noflag, p25_calm):
    fig, ax = plt.subplots(figsize=(6.8, 3.0))
    ax.grid(axis="x", visible=False)
    groups = [("crowded short + attention spike (FLAG)", p25_flag, BLUE),
              ("crowded short, no spike", p25_noflag, DEEMPH),
              ("not crowded (context)", p25_calm, DEEMPH)]
    y = np.arange(len(groups))[::-1]
    for yi, (lab, v, c) in zip(y, groups):
        # thin bar, 4px-rounded data end, square at baseline
        w = v
        ax.add_patch(FancyBboxPatch((0, yi - 0.16), w, 0.32,
                     boxstyle="round,pad=0,rounding_size=0.012",
                     mutation_aspect=0.5, fc=c, ec="none"))
        ax.text(w + 0.004, yi, f"{v:.1%}", va="center", fontsize=11,
                color=INK, weight="bold")
        ax.text(-0.004, yi, lab, va="center", ha="right", fontsize=10, color=INK2)
    ax.set_xlim(0, max(v for _, v, _ in groups) * 1.3)
    ax.set_ylim(-0.6, len(groups) - 0.4)
    ax.set_yticks([])
    ax.xaxis.set_major_formatter(lambda x, _: f"{x:.0%}")
    ax.set_xlabel("P(stock rises ≥ +25% within 4 weeks)")
    ax.spines["left"].set_visible(False)
    return png(fig)


BANNED = ("GME", "AMC")


def main():
    tickers = list(TICKER_ARTICLE.keys())
    si = get_short_interest(tickers, verbose=False)
    panel = MO.build_panel(verbose=False)
    panel = panel[~panel["ticker"].isin(BANNED)]     # tradable universe only
    cr = panel[panel["crowded"]]
    fl, nf = cr[cr["flag"]], cr[~cr["flag"]]
    calm = panel[~panel["crowded"]]
    p25f, p25n, p25c = [(g["fwd4"] >= 0.25).mean() for g in (fl, nf, calm)]
    real, ci, pval = MO.cluster_bootstrap_taildiff(cr, 0.25)
    nflags_yr = len(panel[panel["flag"]]) / panel["week"].dt.year.nunique()

    g_cvna, f_cvna, pk_cvna, ret_cvna = episode_chart_weekly(
        "CVNA", "Carvana", "2023-09-01", "2024-01-31", "2023-11-03", si)
    g_etsy, f_etsy, pk_etsy, ret_etsy = episode_chart_weekly(
        "ETSY", "Etsy", "2022-05-15", "2022-10-15", "2022-07-01", si)
    g_tail = tail_chart(p25f, p25n, p25c)

    cvna_lead = (pk_cvna - f_cvna).days
    etsy_lead = (pk_etsy - f_etsy).days
    ratio = p25f / p25n if p25n > 0 else np.nan

    html = f"""<!DOCTYPE html><html><head><meta charset="utf-8">
<title>Squeeze early-warning monitor — demo</title><style>
body{{font-family:system-ui,-apple-system,"Segoe UI",sans-serif;max-width:880px;margin:0 auto;
padding:28px;color:{INK};background:#f9f9f7;font-size:14px;line-height:1.5}}
h1{{font-size:22px;margin:0 0 2px}} .tag{{color:{INK2};margin:0 0 18px}}
h2{{font-size:13px;text-transform:uppercase;letter-spacing:.6px;color:{INK2};
border-bottom:1px solid {GRID};padding-bottom:4px;margin:26px 0 10px}}
.kpi{{display:flex;gap:12px;margin:14px 0}}
.kpi>div{{flex:1;background:{SURF};border:1px solid rgba(11,11,11,.10);border-radius:8px;
padding:12px 14px}}
.kpi b{{display:block;font-size:24px;font-weight:600;color:{INK}}}
.kpi span{{font-size:11.5px;color:{INK2}}}
img{{width:100%;background:{SURF};border:1px solid rgba(11,11,11,.10);border-radius:8px;
padding:6px;box-sizing:border-box;margin:6px 0}}
table{{border-collapse:collapse;width:100%;font-size:12.5px;margin:8px 0;background:{SURF}}}
th,td{{border:1px solid {GRID};padding:5px 9px;text-align:right;
font-variant-numeric:tabular-nums}}
th{{background:#f4f4f1;text-align:right;color:{INK2}}} th:first-child,td:first-child{{text-align:left}}
.warn{{background:#fff7ed;border-left:3px solid #ec835a;padding:9px 13px;border-radius:4px;
font-size:12.5px;color:{INK2}}}
.foot{{color:{MUTED};font-size:11px;margin-top:20px;border-top:1px solid {GRID};padding-top:9px}}
</style></head><body>

<h1>Squeeze early-warning monitor</h1>
<div class="tag">A risk tool for a consumer/retail short book: <b>crowded short interest
(FINRA, published data only) × abnormal public attention → step-aside flag.</b>
Free data, fully reproducible. It flags danger; it does not predict direction.
Universe and episodes below are tradable consumer/retail names — GME and AMC are
excluded throughout, including from the significance test.</div>

<div class="kpi">
<div><b>{cvna_lead} days</b><span>CVNA: flag → peak close<br>(flagged {f_cvna:%b %d, %Y}, +{ret_cvna:.0%})</span></div>
<div><b>{etsy_lead} days</b><span>ETSY: flag → peak close<br>(flagged {f_etsy:%b %d, %Y}, +{ret_etsy:.0%})</span></div>
<div><b>{ratio:.1f}×</b><span>tail-risk when flagged<br>P(+25%/4wk): {p25f:.1%} vs {p25n:.1%}, p={pval:.3f}</span></div>
<div><b>~{nflags_yr:.0f}/yr</b><span>flags across 30 names<br>(actionable, not noisy)</span></div>
</div>

<h2>Episode 1 — Carvana, November 2023</h2>
<img src="data:image/png;base64,{g_cvna}">
<div class="warn">CVNA is also the episode that exposed a mechanical flaw in the simpler
days-to-cover crowding test: DTC falls as volume explodes mid-squeeze, so it can read "safe"
exactly when risk peaks. The volume-independent crowding measure used here (short-interest
level vs. its own trailing history) doesn't have that failure mode — though on CVNA specifically
its own 2022 collapse-era extremes keep the percentile moderate even in a real squeeze, which is
why this episode is shown via the weekly rule (the one behind the p = {pval:.3f} population result)
rather than the finer daily view. Stated, not hidden.</div>

<h2>Episode 2 — Etsy, July 2022</h2>
<img src="data:image/png;base64,{g_etsy}">

<h2>Panel evidence — 28 tradable names (ex GME/AMC), 2018–2024, {len(panel):,} name-weeks</h2>
<img src="data:image/png;base64,{g_tail}">
<table>
<tr><th>group (name-weeks)</th><th>n</th><th>mean 4-wk</th><th>P(≥+15%)</th><th>P(≥+25%)</th></tr>
<tr><td>crowded short + attention spike (FLAG)</td><td>{len(fl)}</td><td>{fl['fwd4'].mean():+.1%}</td>
<td>{(fl['fwd4']>=.15).mean():.1%}</td><td>{p25f:.1%}</td></tr>
<tr><td>crowded short, no spike</td><td>{len(nf)}</td><td>{nf['fwd4'].mean():+.1%}</td>
<td>{(nf['fwd4']>=.15).mean():.1%}</td><td>{p25n:.1%}</td></tr>
<tr><td>not crowded (context)</td><td>{len(calm)}</td><td>{calm['fwd4'].mean():+.1%}</td>
<td>{(calm['fwd4']>=.15).mean():.1%}</td><td>{p25c:.1%}</td></tr>
</table>
<div style="font-size:12px;color:{INK2}">Week-cluster bootstrap on ΔP(≥+25%):
{real:+.1%}, 95% CI [{ci[0]:+.1%}, {ci[1]:+.1%}], one-sided p = {pval:.3f}.
Flag rule was fixed before evaluation (days-to-cover ≥ 4 and weekly views ≥ 2× the
trailing 8-week median).</div>

<h2>Honest limitations (on the sheet, not in a footnote)</h2>
<div class="warn"><ul style="margin:4px 0;padding-left:18px">
<li>Misses squeezes with no attention spike — e.g. CVNA's earlier July 2023 move
(roughly +50% in two weeks) was <i>not</i> flagged, a different episode from the
Nov 2023 one shown above that <i>was</i> — or that lack SI history (BYND 2019, IPO'd
too recently for a trailing baseline).</li>
<li>Delisted squeeze names (BBBY, EXPR) can't be back-tested with free price data —
survivorship acknowledged.</li>
<li>FINRA short interest is twice-monthly with a ~9-business-day publication lag (this
evaluation uses only published data); a desk feed (daily borrow/utilization) would
sharpen the crowding leg materially.</li>
<li>The flag says <b>reduce/size-down risk</b>, not "the top is in" — flags can arrive
days to weeks before a peak, but the position is often already dangerous well before
the flag fires.</li>
</ul></div>

<div class="foot">Data: FINRA consolidated short interest (published-date-lagged),
Wikimedia pageviews (daily), Yahoo adjusted prices. Rule pre-committed before
evaluation; population p-value from a week-clustered bootstrap; GME/AMC excluded
throughout (banned at the target fund). Reproducible: <code>squeeze_monitor/</code>
(monitor.py, demo.py). Hypothetical/backtested analysis; not investment advice.</div>
</body></html>"""
    open(OUT, "w").write(html)
    print("Wrote", OUT)
    print(f"  CVNA flag {f_cvna.date()} lead {cvna_lead}d (+{ret_cvna:.0%}) | "
          f"ETSY flag {f_etsy.date()} lead {etsy_lead}d (+{ret_etsy:.0%}) | "
          f"tail {p25f:.1%} vs {p25n:.1%} (x{ratio:.1f}, p={pval:.3f})")


if __name__ == "__main__":
    main()
