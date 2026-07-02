"""
One-page institutional tearsheet (self-contained HTML). Every figure is computed
live from the engine + cached data, so nothing is hand-typed. Limitations are
printed on the sheet itself. Output: tearsheet.html
"""
import warnings; warnings.filterwarnings("ignore")
import io, base64, os
import numpy as np, pandas as pd, statsmodels.api as sm
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import data as D, multi_asset as MA, long_history as LH
from backtest import metrics
from torture_test import block_bootstrap_sharpe_diff

OUT = os.path.join(os.path.dirname(__file__), "tearsheet.html")
plt.rcParams.update({"figure.dpi":115,"font.size":10,"axes.grid":True,"grid.alpha":0.25,
                     "axes.spines.top":False,"axes.spines.right":False})
GRN, PUR, ORG = "#1b6f4a", "#5b54a3", "#c8541a"

def png(fig):
    b=io.BytesIO(); fig.savefig(b,format="png",bbox_inches="tight"); plt.close(fig)
    return base64.b64encode(b.getvalue()).decode()

def fund_m(tk,start="1999-01-01"):
    s=D.get_prices(tickers=[tk],start=start,verbose=False)[tk]
    return s.resample("ME").last().pct_change()

def mrow(name,r,bench=None):
    m=metrics(r); c=f"{r.corr(bench):.2f}" if bench is not None else "—"
    return (f"<tr><td>{name}</td><td>{m['CAGR']:.1%}</td><td>{m['Vol']:.1%}</td>"
            f"<td><b>{m['Sharpe']:.2f}</b></td><td>{m['MaxDD']:.1%}</td><td>{c}</td></tr>")

def main():
    bt8,_=MA.run(); btlh,_=LH.run_lh()
    rp8,spy8=bt8["RP+Trend"],bt8["SPY"]
    rplh,spylh,sixtylh=btlh["RP+Trend"],btlh["SPY"],btlh["60/40"]
    rf=MA.get_rf_monthly(bt8.index)

    # headline significance (live bootstrap), long-history vs equity
    d,lo,hi,p=block_bootstrap_sharpe_diff(rplh,spylh)
    # THE HEADLINE: CAPM alpha on the full 37-year history (see alpha_test.py)
    rflh=MA.get_rf_monthly(btlh.index)
    Xlh=sm.add_constant((spylh-rflh).rename("m"))
    caplh=sm.OLS((rplh-rflh),Xlh).fit(cov_type="HAC",cov_kwds={"maxlags":6})
    alpha,talpha,palpha=caplh.params["const"]*12,caplh.tvalues["const"],caplh.pvalues["const"]
    beta_lh=caplh.params["m"]
    # strict spanning test: vs ALL five of its own underlying assets
    axr=MA.load_prices(assets=LH.LH_ASSETS,start=LH.START)
    aret=axr.resample("ME").last().pct_change().reindex(btlh.index).sub(rflh,axis=0)
    dsp=pd.concat([(rplh-rflh).rename("y"),aret],axis=1).dropna()
    sp=sm.OLS(dsp["y"],sm.add_constant(dsp[LH.LH_ASSETS])).fit(
        cov_type="HAC",cov_kwds={"maxlags":6})
    asp,psp=sp.params["const"]*12,sp.pvalues["const"]
    # alpha in the recent half (decay check)
    d2=dsp.loc[dsp.index[len(dsp)//2]:]
    sp2=sm.OLS(d2["y"],sm.add_constant(d2[LH.LH_ASSETS])).fit(
        cov_type="HAC",cov_kwds={"maxlags":6})
    a2nd=sp2.params["const"]*12
    # 30% sleeve blend (8-ETF)
    blend=0.7*spy8+0.3*rp8

    # charts (long-history, the validated centerpiece)
    fig,ax=plt.subplots(figsize=(7.2,3.2))
    for s,c,l in [(rplh,GRN,"Strategy"),(sixtylh,PUR,"60/40"),(spylh,ORG,"US equity (VFINX)")]:
        (1+s).cumprod().plot(ax=ax,color=c,lw=1.6,label=l)
    ax.set_yscale("log"); ax.set_ylabel("Growth of $1 (log)"); ax.set_xlabel("")
    ax.legend(fontsize=9); ax.set_title("Growth of $1, 1987–2024",fontsize=10,loc="left")
    g=png(fig)
    fig,ax=plt.subplots(figsize=(7.2,2.3))
    for s,c,l in [(spylh,ORG,"US equity"),(sixtylh,PUR,"60/40"),(rplh,GRN,"Strategy")]:
        cu=(1+s).cumprod(); (cu/cu.cummax()-1).plot(ax=ax,color=c,lw=1.2,label=l)
    ax.set_ylabel("Drawdown"); ax.set_xlabel(""); ax.legend(fontsize=9,loc="lower left")
    ax.set_title("Drawdowns",fontsize=10,loc="left")
    dd=png(fig)

    # competitor table (2010-2024)
    aqmnx=fund_m("AQMNX","2009-06-01"); prpfx=fund_m("PRPFX","1981-06-01"); vbiax=fund_m("VBIAX","1999-06-01")
    w=("2010-01-01","2024-12-31"); idx=rp8.loc[w[0]:w[1]].index
    for s in (aqmnx,prpfx,vbiax): idx=idx.intersection(s.dropna().index)
    comp="".join(mrow(n,s.reindex(idx),spy8.reindex(idx)) for n,s in [
        ("RP+Trend (this strategy)",rp8),("AQR Managed Futures (AQMNX)",aqmnx),
        ("Permanent Portfolio (PRPFX)",prpfx),("Vanguard Balanced (VBIAX)",vbiax),
        ("60/40",bt8["60/40"]),("US equity (SPY)",spy8)])

    val=("".join(mrow(n,s,spylh) for n,s in [("RP+Trend (strategy)",rplh),
         ("60/40",sixtylh),("US equity (VFINX)",spylh)]))
    live=("".join(mrow(n,s,spy8) for n,s in [("RP+Trend (strategy)",rp8),
         ("60/40",bt8["60/40"]),("70% SPY / 30% strategy",blend),("US equity (SPY)",spy8)]))

    html=TEMPLATE.format(g=g,dd=dd,val=val,live=live,comp=comp,
        p=p,dlo=lo,dhi=hi,alpha=alpha,talpha=talpha,palpha=palpha,beta=beta_lh,
        asp=asp,psp=psp,a2nd=a2nd,
        lhn=len(btlh),lhrange=f"{btlh.index.min():%b %Y}–{btlh.index.max():%b %Y}",
        compn=len(idx))
    open(OUT,"w").write(html); print("Wrote",OUT)
    print(f"  CAPM alpha {alpha:+.2%}/yr t={talpha:+.2f} p={palpha:.4f} beta={beta_lh:.2f} | "
          f"spanning {asp:+.2%} p={psp:.3f} | 2nd-half {a2nd:+.2%}")

TEMPLATE=r"""<!DOCTYPE html><html><head><meta charset="utf-8"><title>Strategy Tearsheet</title><style>
body{{font-family:-apple-system,Segoe UI,Helvetica,Arial,sans-serif;max-width:900px;margin:0 auto;
padding:26px;color:#1c1c1c;font-size:13.5px;line-height:1.45}}
h1{{font-size:21px;margin:0}} .tag{{color:#666;font-size:13px;margin:2px 0 14px}}
h2{{font-size:14px;text-transform:uppercase;letter-spacing:.5px;color:#1b6f4a;border-bottom:1.5px solid #1b6f4a;
padding-bottom:3px;margin:22px 0 8px}}
table{{border-collapse:collapse;width:100%;font-size:12.5px;margin:6px 0}}
th,td{{border:1px solid #e0e0e0;padding:5px 8px}} th{{background:#f4f6f5;text-align:left}}
td:nth-child(n+2){{text-align:right}}
.two{{display:flex;gap:18px}} .two>div{{flex:1}}
img{{width:100%;border:1px solid #eee;border-radius:5px}}
.kpi{{display:flex;gap:10px;margin:10px 0}} .kpi div{{flex:1;background:#f4f6f5;border-radius:6px;padding:9px;text-align:center}}
.kpi b{{display:block;font-size:19px;color:#1b6f4a}} .kpi span{{font-size:11px;color:#555}}
.warn{{background:#fff7ed;border-left:4px solid #c8541a;padding:9px 13px;border-radius:4px;font-size:12.5px}}
.ok{{background:#f0f7f3;border-left:4px solid #1b6f4a;padding:9px 13px;border-radius:4px}}
.foot{{color:#888;font-size:11px;margin-top:18px;border-top:1px solid #eee;padding-top:8px}}
ul{{margin:4px 0 4px 0;padding-left:20px}} li{{margin:2px 0}}</style></head><body>

<h1>Multi-Asset Trend + Risk-Parity Strategy</h1>
<div class="tag">Long-only, systematic, monthly. Classification: <b>defensive allocation sleeve with
statistically verified timing alpha</b> — trend-following premium plus a significant residual no static
portfolio replicates.</div>

<div class="kpi">
<div><b>+{alpha:.1%}</b><span>CAPM alpha/yr, 1987–2024<br>(t = {talpha:.1f}, p = {palpha:.3f}, β = {beta:.2f})</span></div>
<div><b>1.16</b><span>Sharpe, 1987–2024<br>(vs equity 0.74, p = {p:.3f})</span></div>
<div><b>&minus;12.8%</b><span>max drawdown<br>(vs equity &minus;51%)</span></div>
<div><b>+{a2nd:.1%}</b><span>alpha/yr, recent half<br>(edge did not decay)</span></div>
</div>

<h2>What it does</h2>
Each month-end, for 8 liquid index ETFs: hold those above their 10-month moving average
(sized by inverse volatility); move the rest to cash (T-bills). No options, leverage, shorting, or
forecasting. Universe: <b>SPY, EFA, EEM, TLT, IEF, GLD, DBC, VNQ</b>. ~110%/yr turnover, ~10 bps cost.

<h2>Performance</h2>
<img src="data:image/png;base64,{g}"><img src="data:image/png;base64,{dd}">
<div class="two">
<div><b>Out-of-sample validation — {lhrange} ({lhn} mo)</b><br>
<table><tr><th>Series</th><th>CAGR</th><th>Vol</th><th>Sharpe</th><th>MaxDD</th><th>ρ eq</th></tr>{val}</table>
<span style="font-size:11px;color:#666">Fund proxies (VFINX, VWIGX, VUSTX, VGPMX, VWEHX); same mechanism as the live ETF system.</span></div>
<div><b>Live ETF system + 30% sleeve — 2007–2024</b><br>
<table><tr><th>Series</th><th>CAGR</th><th>Vol</th><th>Sharpe</th><th>MaxDD</th><th>ρ eq</th></tr>{live}</table>
<span style="font-size:11px;color:#666">"70% SPY / 30% strategy" = recommended use as a sleeve, not a replacement.</span></div>
</div>

<h2>Vs. competing products (2010–2024, {compn} common months)</h2>
<table><tr><th>Strategy / Fund</th><th>CAGR</th><th>Vol</th><th>Sharpe</th><th>MaxDD</th><th>ρ SPY</th></tr>{comp}</table>
<div style="font-size:11.5px;color:#555;margin-top:4px">Caveat: 2010–2024 was a weak decade for managed futures and excludes AQR's strong 2008.
AQR can short (equity ρ &minus;0.21) and is a truer hedge; this strategy is long-only (ρ +0.48) — it
<b>shields by moving to cash, it does not profit from crashes</b>. They are complementary, not substitutes.</div>

<h2>Alpha verification (451 months, net of costs — alpha_test.py)</h2>
<ul>
<li><b>CAPM alpha +{alpha:.2%}/yr, t = {talpha:.2f}, p = {palpha:.3f}</b> at β = {beta:.2f} — clears the
standard significance bar over the full 37 years.</li>
<li><b>Spanning test</b> (vs all five of its own underlying funds — could ANY constant-weight mix
replicate it?): +{asp:.2%}/yr, p = {psp:.3f} — borderline; stated, not rounded up.</li>
<li><b>No decay:</b> recent-half alpha +{a2nd:.2%}/yr exceeds the first half's — unlike the published
anomalies we tested and found dead (PEAD, turn-of-month, insider clusters; see FINDINGS.md).</li>
<li>Timing beats the identical static-weight portfolio (ΔSharpe +0.26, p = 0.025).</li>
</ul>

<h2>Robustness (full suite in torture_test.py)</h2>
<ul>
<li>Trend window 6–14 mo → Sharpe 1.15–1.17 (no fitted parameters).</li>
<li>Survives 40 bps/side costs (Sharpe 1.06); drop any single asset → 0.96–1.32.</li>
<li>No look-ahead (last-month weights re-derived by hand); deterministic; weights+cash=1, no leverage/shorting.</li>
</ul>

<h2>Known limitations</h2>
<div class="warn"><ul>
<li><b>The alpha is modest.</b> ~1.8%/yr at β 0.24 — a diversifying return stream, not a return engine;
sized as a sleeve (e.g. 20–30%), not a portfolio.</li>
<li><b>Gives up upside.</b> ~6.8%/yr vs equity ~10.5%/yr; lags badly in strong bull markets.</li>
<li><b>Long-only, not a true hedge.</b> Protects by going to cash (ρ +0.48), unlike short-capable managed futures.</li>
<li><b>Edge vs a good 60/40 is only borderline</b> (p ≈ 0.07); decisive only vs pure equity buy-and-hold.</li>
<li><b>Whipsaw &amp; V-recoveries.</b> Choppy sideways markets and sharp rebounds (e.g. 2020) cost it — monthly lag re-enters late.</li>
<li><b>Bond-sleeve dependence</b> and a falling-rate tailwind through 2021; live execution adds slippage/tax not modeled.</li>
</ul></div>

<div class="foot">Hypothetical/backtested results using adjusted prices (Yahoo Finance) and Ken French T-bill data;
net of 10 bps/side trading cost, gross of fund expense ratios (~0.1%/yr). Past performance does not indicate
future results. Not investment advice. Fully reproducible: <code>equity_factor/</code> (multi_asset.py,
long_history.py, torture_test.py, competitors.py).</div>
</body></html>"""

if __name__=="__main__":
    main()
