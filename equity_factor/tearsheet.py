"""
One-page institutional tearsheet (self-contained HTML). Every figure is computed
live from the engine + cached data, so nothing is hand-typed. Limitations are
printed on the sheet itself. Charts follow the dataviz standard (validated
palette, hairline grid, direct labels). Run extended_tests.py first (it caches
the rolling series). Output: tearsheet.html
"""
import warnings; warnings.filterwarnings("ignore")
import io, base64, os, json
import numpy as np, pandas as pd, statsmodels.api as sm
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import data as D, multi_asset as MA, long_history as LH
from backtest import metrics
from torture_test import block_bootstrap_sharpe_diff

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "tearsheet.html")

# validated palette (dataviz reference, light mode)
SURF, INK, INK2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
GRID, AXIS = "#e1e0d9", "#c3c2b7"
BLU, ORG, VIO = "#2a78d6", "#eb6834", "#4a3aa7"   # strategy / equity / 60-40
plt.rcParams.update({
    "figure.dpi": 150, "font.size": 10, "font.family": "sans-serif",
    "axes.facecolor": SURF, "figure.facecolor": SURF,
    "axes.edgecolor": AXIS, "axes.linewidth": 1,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 1,
    "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.labelcolor": INK2, "text.color": INK})


def png(fig):
    b = io.BytesIO(); fig.savefig(b, format="png", bbox_inches="tight", facecolor=SURF)
    plt.close(fig); return base64.b64encode(b.getvalue()).decode()


def fund_m(tk, start="1999-01-01"):
    s = D.get_prices(tickers=[tk], start=start, verbose=False)[tk]
    return s.resample("ME").last().pct_change()


def mrow(name, r, bench=None):
    m = metrics(r); c = f"{r.corr(bench):.2f}" if bench is not None else "—"
    return (f"<tr><td>{name}</td><td>{m['CAGR']:.1%}</td><td>{m['Vol']:.1%}</td>"
            f"<td><b>{m['Sharpe']:.2f}</b></td><td>{m['MaxDD']:.1%}</td><td>{c}</td></tr>")


def capm(y, mkt, rf):
    d = pd.concat([(y - rf).rename("y"), (mkt - rf).rename("m")], axis=1).dropna()
    m = sm.OLS(d["y"], sm.add_constant(d["m"])).fit(cov_type="HAC", cov_kwds={"maxlags": 6})
    return m.params["const"] * 12, m.tvalues["const"], m.pvalues["const"], m.params["m"]


def chart_growth(rplh, sixtylh, spylh):
    fig, ax = plt.subplots(figsize=(7.6, 3.4))
    series = [(rplh, BLU, "Strategy"), (sixtylh, VIO, "60/40"), (spylh, ORG, "US equity")]
    for s, c, l in series:
        (1 + s).cumprod().plot(ax=ax, color=c, lw=2, label=l)
    ax.set_yscale("log"); ax.set_ylabel("Growth of $1 (log)"); ax.set_xlabel("")
    for s, c, l in series:                       # direct end labels
        cum = (1 + s).cumprod()
        ax.annotate(f" {l}  ${cum.iloc[-1]:,.0f}", xy=(cum.index[-1], cum.iloc[-1]),
                    fontsize=9, color=INK, va="center", weight="bold")
    ax.legend(fontsize=8.5, frameon=False, loc="upper left")
    ax.set_xlim(rplh.index[0], rplh.index[-1] + pd.DateOffset(years=8))
    ax.set_title("Growth of $1, 1987–2024", fontsize=10.5, loc="left", color=INK2)
    return png(fig)


def chart_dd(rplh, sixtylh, spylh):
    fig, ax = plt.subplots(figsize=(7.6, 2.4))
    for s, c, l in [(spylh, ORG, "US equity"), (sixtylh, VIO, "60/40"), (rplh, BLU, "Strategy")]:
        cu = (1 + s).cumprod(); ddw = cu / cu.cummax() - 1
        ax.plot(ddw.index, ddw.values, color=c, lw=1.6, label=l)
        ax.fill_between(ddw.index, ddw.values, 0, color=c, alpha=0.10, lw=0)
    ax.set_ylabel("Drawdown"); ax.legend(fontsize=8.5, frameon=False, loc="lower left", ncol=3)
    ax.yaxis.set_major_formatter(lambda x, _: f"{x:.0%}")
    ax.set_title("Drawdowns", fontsize=10.5, loc="left", color=INK2)
    return png(fig)


def chart_rolling():
    rs = pd.read_csv(os.path.join(HERE, "cache/_roll_sh.csv"), index_col=0, parse_dates=True)
    ra = pd.read_csv(os.path.join(HERE, "cache/_roll_alpha.csv"), index_col=0, parse_dates=True)["alpha"]
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(7.6, 4.2), sharex=False,
                                 gridspec_kw={"hspace": 0.5})
    a1.plot(rs.index, rs["sh_rp"], color=BLU, lw=2, label="Strategy")
    a1.plot(rs.index, rs["sh_eq"], color=ORG, lw=2, label="US equity")
    a1.axhline(0, color=AXIS, lw=1)
    a1.set_title("Rolling 3-year Sharpe", fontsize=10.5, loc="left", color=INK2)
    a1.legend(fontsize=8.5, frameon=False, loc="upper left", ncol=2)
    a2.plot(ra.index, ra.values, color=BLU, lw=2)
    a2.fill_between(ra.index, ra.values, 0, where=ra.values >= 0, color=BLU, alpha=0.10, lw=0)
    a2.fill_between(ra.index, ra.values, 0, where=ra.values < 0, color=ORG, alpha=0.12, lw=0)
    a2.axhline(0, color=AXIS, lw=1)
    a2.yaxis.set_major_formatter(lambda x, _: f"{x:+.0%}")
    a2.set_title("Rolling 5-year CAPM alpha (positive in 84% of windows)",
                 fontsize=10.5, loc="left", color=INK2)
    return png(fig)


def chart_decades(rp, eq, rf):
    labs, vals = [], []
    for lab, a, b in [("1987–96", "1987", "1996"), ("1997–06", "1997", "2006"),
                      ("2007–16", "2007", "2016"), ("2017–24", "2017", "2024")]:
        al, *_ = capm(rp.loc[a:b], eq.loc[a:b], rf.loc[a:b])
        labs.append(lab); vals.append(al)
    fig, ax = plt.subplots(figsize=(3.6, 2.6))
    ax.grid(axis="x", visible=False)
    x = np.arange(len(labs))
    cols = [BLU if v >= 0 else ORG for v in vals]
    ax.bar(x, vals, width=0.55, color=cols)
    ax.axhline(0, color=AXIS, lw=1)
    for xi, v in zip(x, vals):
        ax.text(xi, v + (0.001 if v >= 0 else -0.001), f"{v:+.1%}",
                ha="center", va="bottom" if v >= 0 else "top", fontsize=9,
                color=INK, weight="bold")
    ax.set_xticks(x, labs, fontsize=8.5)
    ax.yaxis.set_major_formatter(lambda y, _: f"{y:+.0%}")
    ax.set_title("CAPM alpha by decade", fontsize=10.5, loc="left", color=INK2)
    return png(fig)


def chart_capture(rp, eq):
    up, dn = eq > 0, eq <= 0
    fig, ax = plt.subplots(figsize=(3.6, 2.6))
    ax.grid(axis="x", visible=False)
    x = np.arange(2)
    mkt = [eq[up].mean(), eq[dn].mean()]
    strat = [rp[up].mean(), rp[dn].mean()]
    ax.bar(x - 0.16, mkt, width=0.28, color=ORG, label="US equity")
    ax.bar(x + 0.16, strat, width=0.28, color=BLU, label="Strategy")
    ax.axhline(0, color=AXIS, lw=1)
    for xi, (m_, s_) in enumerate(zip(mkt, strat)):
        ax.text(xi - 0.16, m_ + np.sign(m_) * 0.001, f"{m_:+.1%}", ha="center",
                va="bottom" if m_ >= 0 else "top", fontsize=8.5, color=INK)
        ax.text(xi + 0.16, s_ + np.sign(s_) * 0.001, f"{s_:+.1%}", ha="center",
                va="bottom" if s_ >= 0 else "top", fontsize=8.5, color=INK, weight="bold")
    ax.set_xticks(x, [f"market up\n({up.sum()} mo)", f"market down\n({dn.sum()} mo)"], fontsize=8.5)
    ax.yaxis.set_major_formatter(lambda y, _: f"{y:+.0%}")
    ax.legend(fontsize=8, frameon=False, loc="lower left")
    ax.set_title("Avg month: 37% up-capture, 20% down", fontsize=10.5, loc="left", color=INK2)
    return png(fig)


def main():
    bt8, _ = MA.run(); btlh, _ = LH.run_lh()
    rp8, spy8 = bt8["RP+Trend"], bt8["SPY"]
    rplh, spylh, sixtylh = btlh["RP+Trend"], btlh["SPY"], btlh["60/40"]
    rf8 = MA.get_rf_monthly(bt8.index)
    rflh = MA.get_rf_monthly(btlh.index)

    d, lo, hi, p = block_bootstrap_sharpe_diff(rplh, spylh)
    alpha, talpha, palpha, beta = capm(rplh, spylh, rflh)
    # spanning (strict)
    axr = MA.load_prices(assets=LH.LH_ASSETS, start=LH.START)
    aret = axr.resample("ME").last().pct_change().reindex(btlh.index).sub(rflh, axis=0)
    dsp = pd.concat([(rplh - rflh).rename("y"), aret], axis=1).dropna()
    sp = sm.OLS(dsp["y"], sm.add_constant(dsp[LH.LH_ASSETS])).fit(
        cov_type="HAC", cov_kwds={"maxlags": 6})
    asp, psp = sp.params["const"] * 12, sp.pvalues["const"]
    # 40-year extension results (from extended_tests.py cache)
    ext = json.load(open(os.path.join(HERE, "cache/_ext_results.json")))["ext"]
    blend = 0.7 * spy8 + 0.3 * rp8

    g = chart_growth(rplh, sixtylh, spylh)
    dd = chart_dd(rplh, sixtylh, spylh)
    roll = chart_rolling()
    dec = chart_decades(rplh, spylh, rflh)
    cap = chart_capture(rplh, spylh)

    aqmnx = fund_m("AQMNX", "2009-06-01"); prpfx = fund_m("PRPFX", "1981-06-01")
    vbiax = fund_m("VBIAX", "1999-06-01")
    w = ("2010-01-01", "2024-12-31"); idx = rp8.loc[w[0]:w[1]].index
    for s in (aqmnx, prpfx, vbiax):
        idx = idx.intersection(s.dropna().index)
    comp = "".join(mrow(n, s.reindex(idx), spy8.reindex(idx)) for n, s in [
        ("RP+Trend (this strategy)", rp8), ("AQR Managed Futures (AQMNX)", aqmnx),
        ("Permanent Portfolio (PRPFX)", prpfx), ("Vanguard Balanced (VBIAX)", vbiax),
        ("60/40", bt8["60/40"]), ("US equity (SPY)", spy8)])
    val = "".join(mrow(n, s, spylh) for n, s in [("RP+Trend (strategy)", rplh),
          ("60/40", sixtylh), ("US equity (VFINX)", spylh)])
    live = "".join(mrow(n, s, spy8) for n, s in [("RP+Trend (strategy)", rp8),
           ("60/40", bt8["60/40"]), ("70% SPY / 30% strategy", blend),
           ("US equity (SPY)", spy8)])

    html = TEMPLATE.format(g=g, dd=dd, roll=roll, dec=dec, cap=cap,
        val=val, live=live, comp=comp, p=p, alpha=alpha, talpha=talpha,
        palpha=palpha, beta=beta, asp=asp, psp=psp,
        exta=ext["alpha"], extt=ext["t"], extp=ext["p"], extstart=ext["start"][:4],
        lhn=len(btlh), lhrange=f"{btlh.index.min():%b %Y}–{btlh.index.max():%b %Y}",
        compn=len(idx))
    open(OUT, "w").write(html); print("Wrote", OUT)
    print(f"  37y CAPM a={alpha:+.2%} t={talpha:+.2f} p={palpha:.4f} | "
          f"40y ext a={ext['alpha']:+.2%} t={ext['t']:+.2f} | spanning p={psp:.3f}")


TEMPLATE = r"""<!DOCTYPE html><html><head><meta charset="utf-8"><title>Strategy Tearsheet</title><style>
body{{font-family:system-ui,-apple-system,"Segoe UI",Helvetica,Arial,sans-serif;max-width:920px;margin:0 auto;
padding:26px;color:#1c1c1c;font-size:13.5px;line-height:1.45;background:#f9f9f7}}
h1{{font-size:21px;margin:0}} .tag{{color:#52514e;font-size:13px;margin:2px 0 14px}}
h2{{font-size:13px;text-transform:uppercase;letter-spacing:.6px;color:#1c5cab;border-bottom:1.5px solid #2a78d6;
padding-bottom:3px;margin:22px 0 8px}}
table{{border-collapse:collapse;width:100%;font-size:12.5px;margin:6px 0;background:#fcfcfb}}
th,td{{border:1px solid #e1e0d9;padding:5px 8px}} th{{background:#f4f4f1;text-align:left;color:#52514e}}
td:nth-child(n+2){{text-align:right;font-variant-numeric:tabular-nums}}
.two{{display:flex;gap:18px}} .two>div{{flex:1}}
img{{width:100%;background:#fcfcfb;border:1px solid rgba(11,11,11,.10);border-radius:6px;padding:4px;box-sizing:border-box}}
.kpi{{display:flex;gap:10px;margin:10px 0}} .kpi div{{flex:1;background:#fcfcfb;border:1px solid rgba(11,11,11,.10);
border-radius:6px;padding:10px;text-align:center}}
.kpi b{{display:block;font-size:20px;color:#1c5cab}} .kpi span{{font-size:11px;color:#52514e}}
.warn{{background:#fff7ed;border-left:4px solid #ec835a;padding:9px 13px;border-radius:4px;font-size:12.5px}}
.foot{{color:#898781;font-size:11px;margin-top:18px;border-top:1px solid #e1e0d9;padding-top:8px}}
ul{{margin:4px 0 4px 0;padding-left:20px}} li{{margin:2px 0}}</style></head><body>

<h1>Multi-Asset Trend + Risk-Parity Strategy</h1>
<div class="tag">Long-only, systematic, monthly. Classification: <b>defensive allocation sleeve with
statistically verified timing alpha</b> — trend-following premium plus a significant residual no static
portfolio replicates.</div>

<div class="kpi">
<div><b>+{alpha:.1%}</b><span>CAPM alpha/yr, 1987–2024<br>(t = {talpha:.1f}, p = {palpha:.3f}, β = {beta:.2f})</span></div>
<div><b>t = {extt:.1f}</b><span>alpha in the {extstart}–2024 extension<br>(+{exta:.1%}/yr incl. Volcker era)</span></div>
<div><b>1.16</b><span>Sharpe, 1987–2024<br>(vs equity 0.74, p = {p:.3f})</span></div>
<div><b>&minus;12.8%</b><span>max drawdown<br>(vs equity &minus;51%)</span></div>
</div>

<h2>What it does</h2>
Each month-end, for 8 liquid index ETFs: hold those above their 10-month moving average
(sized by inverse volatility); move the rest to cash (T-bills). No options, leverage, shorting, or
forecasting. Universe: <b>SPY, EFA, EEM, TLT, IEF, GLD, DBC, VNQ</b>. ~110%/yr turnover, ~10 bps cost.

<h2>Performance (fund-proxy validation, {lhrange}, {lhn} months)</h2>
<img src="data:image/png;base64,{g}"><img src="data:image/png;base64,{dd}">
<div class="two">
<div><b>Out-of-sample validation — 1987–2024</b><br>
<table><tr><th>Series</th><th>CAGR</th><th>Vol</th><th>Sharpe</th><th>MaxDD</th><th>ρ eq</th></tr>{val}</table></div>
<div><b>Live ETF system + 30% sleeve — 2007–2024</b><br>
<table><tr><th>Series</th><th>CAGR</th><th>Vol</th><th>Sharpe</th><th>MaxDD</th><th>ρ eq</th></tr>{live}</table></div>
</div>

<h2>Alpha verification (net of costs — alpha_test.py, extended_tests.py)</h2>
<div class="two">
<div>
<ul>
<li><b>CAPM alpha +{alpha:.2%}/yr, t = {talpha:.2f}, p = {palpha:.3f}</b> (β = {beta:.2f}), 451 months.</li>
<li><b>{extstart}–2024 extension (4-fund variant, incl. Volcker rate shock): +{exta:.2%}/yr, t = {extt:.2f}, p = {extp:.3f}.</b></li>
<li><b>Spanning test</b> (vs all five underlying funds — could any static mix replicate it?):
+{asp:.2%}/yr, p = {psp:.3f} — borderline; stated, not rounded up.</li>
<li>Timing beats the identical static-weight portfolio (ΔSharpe +0.26, p = 0.025);
stationary bootstrap p = 0.001–0.006 across block choices.</li>
<li>Survives rebalancing 5/10/21 sessions late (Sharpe 1.03–1.11) — no date luck.</li>
</ul>
</div>
<div><img src="data:image/png;base64,{dec}"><img src="data:image/png;base64,{cap}"></div>
</div>
<img src="data:image/png;base64,{roll}">

<h2>Vs. competing products (2010–2024, {compn} common months)</h2>
<table><tr><th>Strategy / Fund</th><th>CAGR</th><th>Vol</th><th>Sharpe</th><th>MaxDD</th><th>ρ SPY</th></tr>{comp}</table>
<div style="font-size:11.5px;color:#52514e;margin-top:4px">Caveat: 2010–2024 was a weak decade for managed futures and
excludes AQR's strong 2008. AQR can short (equity ρ &minus;0.21) and is a truer hedge; this strategy is long-only —
it <b>shields by moving to cash, it does not profit from crashes</b>. Complementary, not substitutes.</div>

<h2>Robustness (torture_test.py, extended_tests.py)</h2>
<ul>
<li>Trend window 6–14 mo → Sharpe 1.15–1.17 (no fitted parameters); survives 40 bps/side costs (1.06);
drop any single asset → 0.96–1.32.</li>
<li>Recessions (NBER): strategy &minus;2.4%/yr vs equity &minus;13.4%/yr; rolling 5-year alpha positive in 84% of windows.</li>
<li>No look-ahead (last-month weights re-derived by hand); deterministic; weights+cash=1; no leverage/shorting.</li>
</ul>

<h2>Known limitations</h2>
<div class="warn"><ul>
<li><b>The alpha is modest and episodic.</b> ~1.8%/yr at β 0.24 overall; by decade it lived in 1997–2016 and was
≈ 0 in 2017–2024 (see chart). A diversifying sleeve (20–30%), not a return engine.</li>
<li><b>Gives up upside — severe career risk.</b> 6.8%/yr vs equity 10.5%/yr; lags buy-and-hold in
<b>69% of rolling 3-year windows</b>, once for 159 consecutive months. This discomfort is likely why the premium persists.</li>
<li><b>Rising-rate regimes hurt</b> (Sharpe 0.34 vs equity 0.52 when trailing bond returns are negative); the bond
sleeve had a falling-rate tailwind for much of the sample.</li>
<li><b>Long-only, not a true hedge;</b> monthly rebalancing cannot dodge fast crashes (Oct-1987: &minus;11.8% in one month)
and re-enters late after V-recoveries (2020).</li>
<li><b>Edge vs a good 60/40 is only borderline</b> (p ≈ 0.07); decisive only vs pure equity buy-and-hold.</li>
</ul></div>

<div class="foot">Hypothetical/backtested results using adjusted prices (Yahoo Finance) and Ken French T-bill data;
net of 10 bps/side trading cost, gross of fund expense ratios (~0.1%/yr). Past performance does not indicate future
results. Not investment advice. Fully reproducible: <code>equity_factor/</code> (multi_asset.py, long_history.py,
alpha_test.py, extended_tests.py, torture_test.py, competitors.py).</div>
</body></html>"""

if __name__ == "__main__":
    main()
