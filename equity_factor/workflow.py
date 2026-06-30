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
COL = {"SPY":"#d95f02","EFA":"#e6ab02","EEM":"#a6761d","TLT":"#1b9e77","IEF":"#66c2a5",
       "GLD":"#e7d020","DBC":"#8c6d31","VNQ":"#7570b3","CASH":"#cccccc"}

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
<h2>The monthly decision flow</h2>
<img src="data:image/png;base64,{fig_flow()}">
<h2>What it actually holds — real allocations (auto-adapts by regime)</h2>
<img src="data:image/png;base64,{fig_pies(w)}">
<div class="box">Left: a calm month — fully invested, risk-balanced across everything. Middle: today.
Right: a risk-off month where every asset was below trend, so it sat <b>100% in cash</b> and waited.
The strategy never "predicts" — it just follows trends and steps aside when they break.</div>
</body></html>"""
    with open(OUT,"w") as f: f.write(html)
    print("Wrote", OUT)

if __name__ == "__main__":
    main()
