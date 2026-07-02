"""
Build a visual one-pager: the monthly decision WORKFLOW + real allocation PIE charts
(calm vs crisis vs today). Output: workflow.html (self-contained, base64 PNGs).
Everything shown is plain long-only ETFs + cash — no options, leverage, or shorting.
"""
import warnings; warnings.filterwarnings("ignore")
import io, base64, os
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import multi_asset as MA

OUT = os.path.join(os.path.dirname(__file__), "workflow.html")
NAMES = {"SPY":"US stocks","EFA":"Intl stocks","EEM":"EM stocks","TLT":"Long bonds",
         "IEF":"Mid bonds","GLD":"Gold","DBC":"Commodities","VNQ":"Real estate","CASH":"Cash (T-bills)"}
# validated reference palette (dataviz), fixed slot order; CASH = neutral gray
COL = {"SPY":"#2a78d6","EFA":"#1baf7a","EEM":"#eda100","TLT":"#008300","IEF":"#4a3aa7",
       "GLD":"#e34948","DBC":"#e87ba4","VNQ":"#eb6834","CASH":"#c3c2b7"}
SURF, INK, INK2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9"

def fig_timeline():
    """WHEN each step happens: the exact monthly cadence, with the data each
    step is allowed to see (no look-ahead)."""
    fig, ax = plt.subplots(figsize=(10.5, 3.1)); ax.set_xlim(0, 14); ax.set_ylim(0, 5)
    ax.axis("off"); fig.patch.set_facecolor(SURF)
    ax.add_patch(FancyArrowPatch((0.3, 1.5), (13.7, 1.5), arrowstyle="-|>",
                 mutation_scale=20, lw=2, color=INK2))
    pts = [(1.6, "T &minus; 10 months", "history window\nOPENS", "10 month-end closes\nfeed the trend average"),
           (5.6, "month-end T,\nat the close", "SIGNAL", "for each ETF:\nclose vs 10-mo average;\n12-mo vol for sizing"),
           (8.3, "T (close) /\nnext open", "TRADE", "rotate only what\nchanged; pay ~10 bps\non changes"),
           (11.2, "T+1 .. next\nmonth-end", "HOLD", "no intramonth action —\nnothing to watch,\nnothing to panic-sell"),
           (13.2, "next\nmonth-end", "REPEAT", "")]
    for x, when, what, detail in pts:
        ax.plot([x], [1.5], "o", ms=11, color="#2a78d6", mec=SURF, mew=2, zorder=5)
        ax.text(x, 2.15, what.replace("&minus;", "−"), ha="center", fontsize=11,
                weight="bold", color=INK)
        ax.text(x, 0.95, when.replace("&minus;", "−"), ha="center", va="top",
                fontsize=8.5, color=MUTED)
        if detail:
            ax.text(x, 2.75, detail, ha="center", va="bottom", fontsize=8.5, color=INK2)
    ax.text(0.3, 4.6, "One month in the life of the strategy — every input is a PAST close (no look-ahead)",
            fontsize=11.5, weight="bold", color=INK)
    return png(fig)

def fig_alloc_history(w):
    """WHAT it held WHEN: full monthly allocation history, 2007-2024."""
    cols = MA.ASSETS + ["CASH"]
    ww = w[cols].clip(lower=0)
    fig, ax = plt.subplots(figsize=(10.5, 3.6)); fig.patch.set_facecolor(SURF)
    ax.set_facecolor(SURF)
    ax.stackplot(ww.index, [ww[c].values for c in cols],
                 colors=[COL[c] for c in cols], labels=cols, lw=0)
    ax.set_ylim(0, 1); ax.set_xlim(ww.index[0], ww.index[-1])
    ax.yaxis.set_major_formatter(lambda y, _: f"{y:.0%}")
    ax.grid(False); [s.set_visible(False) for s in ax.spines.values()]
    ax.tick_params(colors=MUTED)
    for dt, lab in [("2008-10-31", "2008:\nall cash"), ("2020-03-31", "2020:\nmostly cash"),
                    ("2022-06-30", "2022:\nde-risked")]:
        d = pd.Timestamp(dt)
        ax.axvline(d, color=INK, lw=1, ls=(0, (2, 2)), alpha=0.6)
        ax.text(d, 1.03, lab, ha="center", va="bottom", fontsize=8.5, color=INK2)
    ax.legend([plt.Rectangle((0, 0), 1, 1, fc=COL[c]) for c in cols],
              [f"{c} — {NAMES[c]}" for c in cols], loc="upper center",
              bbox_to_anchor=(0.5, -0.08), ncol=5, fontsize=8, frameon=False)
    ax.set_title("What it actually held, month by month (2007–2024) — gray is cash",
                 fontsize=11.5, weight="bold", loc="left", color=INK, pad=24)
    return png(fig)

def png(fig):
    b = io.BytesIO(); fig.savefig(b, format="png", bbox_inches="tight", dpi=115); plt.close(fig)
    return base64.b64encode(b.getvalue()).decode()

def box(ax, xy, w, h, text, fc, fontsize=11, tc="black"):
    ax.add_patch(FancyBboxPatch((xy[0]-w/2, xy[1]-h/2), w, h,
        boxstyle="round,pad=0.02,rounding_size=0.04", fc=fc, ec="#333", lw=1.4))
    ax.text(xy[0], xy[1], text, ha="center", va="center", fontsize=fontsize,
            color=tc, weight="bold", wrap=True)

def arrow(ax, p1, p2, label="", color="#333"):
    ax.add_patch(FancyArrowPatch(p1, p2, arrowstyle="-|>", mutation_scale=18,
                 lw=1.6, color=color, shrinkA=2, shrinkB=2))
    if label:
        mx, my = (p1[0]+p2[0])/2, (p1[1]+p2[1])/2
        ax.text(mx+0.04, my, label, fontsize=10, weight="bold", color=color)

def fig_flow():
    fig, ax = plt.subplots(figsize=(8.2, 9.2)); ax.set_xlim(0,10); ax.set_ylim(0,12); ax.axis("off")
    box(ax,(5,11.2),7.2,0.9,"① Each month-end: pull prices for the 8 ETFs","#eef3ff")
    arrow(ax,(5,10.75),(5,10.15))
    box(ax,(5,9.6),8.4,1.0,"② For EACH ETF, ask one question:\nIs its price ABOVE its 10-month average? (uptrend?)","#fff3e6")
    arrow(ax,(3.2,9.1),(2.3,8.2),"YES","#1b7a3d"); arrow(ax,(6.8,9.1),(7.7,8.2),"NO","#b00020")
    box(ax,(2.3,7.6),3.8,1.05,"③a HOLD it.\nSize by inverse volatility\n(calm assets bigger,\njumpy assets smaller)","#e8f6ee",10)
    box(ax,(7.7,7.6),3.8,1.05,"③b SKIP it.\nThat slice goes to\nCASH, earning the\n1-month T-bill yield","#fdeaea",10)
    arrow(ax,(2.3,7.05),(4.3,6.2)); arrow(ax,(7.7,7.05),(5.7,6.2))
    box(ax,(5,5.7),7.6,1.0,"④ Combine into the month's portfolio\n(weights of held ETFs + whatever's left in cash)","#eef3ff")
    arrow(ax,(5,5.2),(5,4.5))
    box(ax,(5,4.0),7.6,1.0,"⑤ Hold for one month. Pay ~10 bps cost only on what changed.","#f3f3f3")
    arrow(ax,(8.8,4.0),(9.4,4.0)); ax.add_patch(FancyArrowPatch((9.4,4.0),(9.4,11.2),
        arrowstyle="-|>",mutation_scale=18,lw=1.6,color="#555",
        connectionstyle="arc3,rad=-0.32"))
    ax.text(9.65,7.6,"repeat\nnext\nmonth",fontsize=9.5,color="#555",weight="bold",rotation=90,va="center")
    box(ax,(5,2.4),9.0,1.5,"WHY IT WORKS:  in normal times you own a diversified, risk-balanced\n"
        "basket. When an asset rolls over (downtrend), you're already OUT of it and in\n"
        "cash — so big crashes (2008, 2022) mostly miss you. That's the whole edge.",
        "#fffbe6",10.5)
    box(ax,(5,0.7),9.0,0.95,"NO options · NO leverage · NO shorting · NO futures.\nPlain long-only ETFs + cash only.",
        "#e8f6ee",11,"#0b5d2e")
    return png(fig)

def fig_pies(w):
    cols = MA.ASSETS + ["CASH"]
    crisis = w["CASH"].idxmax(); calm = w["CASH"].idxmin(); latest = w.index[-1]
    picks = [(calm,"Calm month\n(%s)"%calm.strftime("%b %Y")),
             (latest,"Today\n(%s)"%latest.strftime("%b %Y")),
             (crisis,"Crisis month\n(%s)"%crisis.strftime("%b %Y"))]
    fig, axes = plt.subplots(1,3,figsize=(11.5,4.4))
    for ax,(dt,title) in zip(axes,picks):
        s = w.loc[dt,cols]; s = s[s>0.004]
        ax.pie(s.values, colors=[COL[k] for k in s.index], startangle=90,
               wedgeprops=dict(width=0.42,edgecolor="white",lw=1.5),
               autopct=lambda p:f"{p:.0f}%" if p>=7 else "", pctdistance=0.78, textprops={"fontsize":9})
        ax.set_title(title,fontsize=12,weight="bold")
    handles=[plt.Rectangle((0,0),1,1,fc=COL[k]) for k in cols]
    fig.legend(handles,[f"{k} — {NAMES[k]}" for k in cols],loc="lower center",
               ncol=5,fontsize=9.5,frameon=False,bbox_to_anchor=(0.5,-0.06))
    fig.suptitle("Same engine, totally different holdings by regime — it adapts automatically",
                 fontsize=12.5,weight="bold",y=1.02)
    return png(fig)

def main():
    _, w = MA.run()
    html = f"""<!DOCTYPE html><html><head><meta charset="utf-8"><title>How the strategy works</title>
<style>body{{font-family:-apple-system,Segoe UI,Arial,sans-serif;max-width:940px;margin:0 auto;padding:24px;color:#1a1a1a}}
h1{{font-size:24px}} h2{{border-bottom:2px solid #1b9e77;padding-bottom:4px;margin-top:30px}}
img{{width:100%;border:1px solid #eee;border-radius:8px;margin:10px 0}}
.box{{background:#f7faf8;border-left:4px solid #1b9e77;padding:10px 14px;border-radius:4px;margin:12px 0}}</style></head><body>
<h1>How the strategy works — one monthly loop</h1>
<div class="box"><b>Plain English:</b> once a month, check each of 8 boring index ETFs. Keep the ones
trending up (sized so no single one dominates the risk); dump the ones trending down into cash.
That's it. No options, no leverage, no shorting — just deciding <i>when</i> to own ordinary assets.</div>
<h2>When everything happens — the monthly cadence</h2>
<img src="data:image/png;base64,{fig_timeline()}">
<div class="box">The strategy touches the market <b>one day a month</b>. Signals use only month-end
closes that already happened; the position then sits untouched until the next month-end. Rebalance-date
tests show the exact day doesn't matter (trading 5, 10, even 21 sessions late still works).</div>
<h2>The monthly decision flow</h2>
<img src="data:image/png;base64,{fig_flow()}">
<h2>What it held, when — full 18-year allocation history</h2>
<img src="data:image/png;base64,{fig_alloc_history(w)}">
<div class="box">Reading the chart: in calm bull years the book is fully invested across all eight
assets. As trends break, slices flip to gray (cash) — fully by late 2008, mostly in the 2020 crash and
the 2022 bond-equity bear. That automatic de-risking is the entire source of the drawdown edge
(−12.8% worst vs −51% for equity).</div>
<h2>Snapshots — real allocations (auto-adapts by regime)</h2>
<img src="data:image/png;base64,{fig_pies(w)}">
<div class="box">Left: a calm month — fully invested, risk-balanced across everything. Middle: today.
Right: a risk-off month where every asset was below trend, so it sat <b>100% in cash</b> and waited.
The strategy never "predicts" — it just follows trends and steps aside when they break.</div>
</body></html>"""
    with open(OUT,"w") as f: f.write(html)
    print("Wrote", OUT)

if __name__ == "__main__":
    main()
