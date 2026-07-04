"""
EXECUTED TRADE LOG + LIVE P&L -- turns the monthly weight history into what a
real account statement looks like: discrete trades (entry/exit price, hold
length, realized $ P&L), a per-asset holding timeline, an equity curve, and
average-win-vs-average-loss statistics. Runs on TWO eras:
  - 8-ETF live universe, 2007-2024 (ETF inception dates cap history here --
    GLD/VNQ/DBC didn't exist before ~2004-2006)
  - 5-fund proxy universe, 1987-2024 (37 years, the long_history.py universe --
    reaches through 2000-02, 2008, 2020, 2022 for a much larger trade sample)

TIMING (exact, matches multi_asset.run()'s own accounting -- not an
approximation): weights.loc[t, asset] is the position held DURING the return
realized at t, decided from data through t-1. So for a contiguous run of
months where weight>0 from row i to row j:
  entry price = close at the row BEFORE i (that's what M i's return compounds
                off of)
  exit price  = close at row j (the last held month's closing price)
Dollar P&L per trade is NOT price_return * notional (that would ignore months
where inverse-vol re-weighted the same holding) -- it's the sum of each held
month's actual weighted-dollar contribution to NAV, so it reconciles exactly
with the backtest's compounding.
"""
import warnings; warnings.filterwarnings("ignore")
import io, base64, os
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import FancyBboxPatch
import multi_asset as MA
import long_history as LH

HERE = os.path.dirname(os.path.abspath(__file__))
INITIAL_NAV = 1_000_000

# validated palette (dataviz reference) -- reused from workflow.py so asset
# colors are consistent across every artifact in this repo
SURF, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"

ERA_8ETF = dict(
    label="8-ETF live universe, 2007–2024", out="trade_log.html", csv="trade_log.csv",
    assets=MA.ASSETS, start="2005-01-01", bench=("SPY", "IEF"),
    names={"SPY": "US stocks", "EFA": "Intl stocks", "EEM": "EM stocks", "TLT": "Long bonds",
           "IEF": "Mid bonds", "GLD": "Gold", "DBC": "Commodities", "VNQ": "Real estate"},
    colors={"SPY": "#2a78d6", "EFA": "#1baf7a", "EEM": "#eda100", "TLT": "#008300",
            "IEF": "#4a3aa7", "GLD": "#e34948", "DBC": "#e87ba4", "VNQ": "#eb6834"})

ERA_LONGHIST = dict(
    label="5-fund proxy universe, 1987–2024 (37 years)",
    out="trade_log_1987.html", csv="trade_log_1987.csv",
    assets=LH.LH_ASSETS, start=LH.START, bench=LH.LH_BENCH,
    names={"VFINX": "US equity", "VWIGX": "Intl equity", "VUSTX": "Long Treasuries",
           "VGPMX": "Precious metals", "VWEHX": "High-yield credit"},
    colors={"VFINX": "#2a78d6", "VWIGX": "#1baf7a", "VUSTX": "#008300",
            "VGPMX": "#e34948", "VWEHX": "#4a3aa7"})

plt.rcParams.update({
    "figure.dpi": 140, "font.size": 10.5, "font.family": "sans-serif",
    "axes.facecolor": SURF, "figure.facecolor": SURF,
    "axes.edgecolor": AXIS, "axes.linewidth": 1,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 1,
    "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.labelcolor": INK2, "text.color": INK})


def png(fig):
    b = io.BytesIO(); fig.savefig(b, format="png", bbox_inches="tight", facecolor=SURF)
    plt.close(fig); return base64.b64encode(b.getvalue()).decode()


def build_trade_log(bt, w, mpx, assets):
    """One row per discrete holding run per asset."""
    nav_prev = (INITIAL_NAV * (1 + bt["RP+Trend"]).cumprod()).shift(1)
    nav_prev.iloc[0] = INITIAL_NAV
    mret = mpx.pct_change()
    all_dates = mpx.index          # includes the pre-history date before w's first row
    trades = []
    for asset in assets:
        held = (w[asset] > 1e-9).values
        idx = w.index
        i = 0
        while i < len(held):
            if not held[i]:
                i += 1; continue
            j = i
            while j + 1 < len(held) and held[j + 1]:
                j += 1
            entry_date = idx[i]
            exit_date = idx[j]
            entry_pos = all_dates.get_loc(entry_date)
            entry_price_date = all_dates[entry_pos - 1]     # close BEFORE the run
            entry_price = mpx.loc[entry_price_date, asset]
            exit_price = mpx.loc[exit_date, asset]
            price_ret = exit_price / entry_price - 1
            run_dates = idx[i:j + 1]
            dollar_pnl = float((w.loc[run_dates, asset] * nav_prev.loc[run_dates]
                               * mret.loc[run_dates, asset]).sum())
            avg_w = float(w.loc[run_dates, asset].mean())
            trades.append({"asset": asset, "entry": entry_price_date, "exit": exit_date,
                           "hold_mo": len(run_dates), "entry_px": entry_price,
                           "exit_px": exit_price, "price_ret": price_ret,
                           "avg_weight": avg_w, "pnl": dollar_pnl})
            i = j + 1
    return pd.DataFrame(trades).sort_values("entry").reset_index(drop=True)


def winloss_stats(trades):
    wins, losses = trades[trades["pnl"] > 0], trades[trades["pnl"] <= 0]
    avg_win_d = float(wins["pnl"].mean()) if len(wins) else 0.0
    avg_loss_d = float(losses["pnl"].mean()) if len(losses) else 0.0
    avg_win_pct = float(wins["price_ret"].mean()) if len(wins) else 0.0
    avg_loss_pct = float(losses["price_ret"].mean()) if len(losses) else 0.0
    ratio = abs(avg_win_d / avg_loss_d) if avg_loss_d != 0 else np.nan
    expectancy = float(trades["pnl"].mean())
    return {"n_win": len(wins), "n_loss": len(losses),
            "avg_win_d": avg_win_d, "avg_loss_d": avg_loss_d,
            "avg_win_pct": avg_win_pct, "avg_loss_pct": avg_loss_pct,
            "ratio": ratio, "expectancy": expectancy,
            "best": float(trades["pnl"].max()), "worst": float(trades["pnl"].min())}


def chart_equity(bt):
    nav = INITIAL_NAV * (1 + bt["RP+Trend"]).cumprod()
    spy_nav = INITIAL_NAV * (1 + bt["SPY"]).cumprod()
    fig, ax = plt.subplots(figsize=(9.0, 3.4))
    ax.plot(spy_nav.index, spy_nav.values, color="#eb6834", lw=1.6, label="US equity", alpha=0.85)
    ax.plot(nav.index, nav.values, color="#2a78d6", lw=2.2, label="Strategy (live P&L)")
    ax.fill_between(nav.index, nav.values, INITIAL_NAV, color="#2a78d6", alpha=0.08, lw=0)
    ax.axhline(INITIAL_NAV, color=AXIS, lw=1, ls=(0, (2, 2)))
    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(lambda v, _: f"${v/1e6:.1f}M" if v >= 1e6 else f"${v/1e3:.0f}K")
    ax.annotate(f" ${nav.iloc[-1]:,.0f}", xy=(nav.index[-1], nav.iloc[-1]),
                fontsize=10, color=INK, weight="bold", va="center")
    ax.annotate(f" ${spy_nav.iloc[-1]:,.0f}", xy=(spy_nav.index[-1], spy_nav.iloc[-1]),
                fontsize=9.5, color=INK2, va="center")
    ax.legend(fontsize=9, frameon=False, loc="upper left")
    ax.set_xlim(nav.index[0], nav.index[-1] + pd.DateOffset(years=2))
    ax.set_title(f"Live P&L on ${INITIAL_NAV:,.0f} starting capital (log scale)",
                 fontsize=11, loc="left", color=INK2)
    return png(fig)


def chart_timeline(trades, w, assets, names, colors):
    fig, ax = plt.subplots(figsize=(9.0, 3.6))
    ax.grid(axis="y", visible=False)
    y = {a: i for i, a in enumerate(reversed(assets))}
    for _, t in trades.iterrows():
        yi = y[t["asset"]]
        width_days = (t["exit"] - t["entry"]).days
        ax.add_patch(FancyBboxPatch((mdates.date2num(t["entry"]), yi - 0.32),
                     width_days, 0.64, boxstyle="round,pad=0,rounding_size=0",
                     mutation_aspect=1, fc=colors[t["asset"]], ec="none", alpha=0.9))
    ax.set_ylim(-0.6, len(assets) - 0.4)
    ax.set_yticks(list(y.values()), [f"{a} — {names[a]}" for a in y.keys()], fontsize=9)
    ax.xaxis_date()
    ax.set_xlim(w.index[0], w.index[-1])
    ax.set_title("Position timeline — when each asset was actually held",
                 fontsize=11, loc="left", color=INK2)
    return png(fig)


def chart_winloss(stats):
    fig, ax = plt.subplots(figsize=(5.4, 2.8))
    ax.grid(axis="x", visible=False)
    labs = ["avg WIN", "avg LOSS"]
    vals = [stats["avg_win_pct"], stats["avg_loss_pct"]]
    cols = ["#0ca30c", "#d03b3b"]
    y = np.arange(2)
    for yi, v, c, l in zip(y, vals, cols, labs):
        ax.barh(yi, v, height=0.5, color=c)
        ax.text(v + np.sign(v) * 0.006, yi, f"{v:+.1%}", va="center",
                ha="left" if v >= 0 else "right", fontsize=11, weight="bold", color=INK)
    pad = max(abs(v) for v in vals) * 0.55
    ax.set_xlim(min(vals) - pad, max(vals) + pad)
    ax.set_yticks(y, labs, fontsize=10)
    ax.axvline(0, color=AXIS, lw=1)
    ax.xaxis.set_major_formatter(lambda v, _: f"{v:+.0%}")
    ax.set_title("Avg win vs avg loss (price return per trade)", fontsize=10.5, loc="left", color=INK2)
    return png(fig)


def trade_table_html(trades):
    rows = []
    for _, t in trades.iterrows():
        pnl_color = "#0ca30c" if t["pnl"] >= 0 else "#d03b3b"
        rows.append(
            f"<tr><td>{t['asset']}</td><td>{t['entry']:%Y-%m-%d}</td>"
            f"<td>{t['exit']:%Y-%m-%d}</td><td>{t['hold_mo']}</td>"
            f"<td>${t['entry_px']:.2f}</td><td>${t['exit_px']:.2f}</td>"
            f"<td>{t['price_ret']:+.1%}</td><td>{t['avg_weight']:.0%}</td>"
            f"<td style='color:{pnl_color};font-weight:600'>{'+' if t['pnl']>=0 else ''}"
            f"${t['pnl']:,.0f}</td></tr>")
    return "".join(rows)


def run_era(era):
    bt, w = MA.run(assets=era["assets"], bench=era["bench"], start=era["start"])
    mpx = MA.load_prices(assets=era["assets"], start=era["start"])
    trades = build_trade_log(bt, w, mpx, era["assets"])
    trades.to_csv(os.path.join(HERE, era["csv"]), index=False)

    n = len(trades)
    s = winloss_stats(trades)
    win_rate = s["n_win"] / n
    total_pnl = float(trades["pnl"].sum())
    avg_hold = float(trades["hold_mo"].mean())
    final_nav = INITIAL_NAV * (1 + bt["RP+Trend"]).prod()
    unattrib = (final_nav - INITIAL_NAV) - total_pnl

    g_eq = chart_equity(bt)
    g_tl = chart_timeline(trades, w, era["assets"], era["names"], era["colors"])
    g_wl = chart_winloss(s)
    table_rows = trade_table_html(trades)

    html = f"""<!DOCTYPE html><html><head><meta charset="utf-8">
<title>Executed trade log — {era['label']}</title><style>
body{{font-family:system-ui,-apple-system,"Segoe UI",sans-serif;max-width:960px;margin:0 auto;
padding:28px;color:{INK};background:#f9f9f7;font-size:14px;line-height:1.5}}
h1{{font-size:22px;margin:0 0 2px}} .tag{{color:{INK2};margin:0 0 18px}}
h2{{font-size:13px;text-transform:uppercase;letter-spacing:.6px;color:{INK2};
border-bottom:1px solid {GRID};padding-bottom:4px;margin:26px 0 10px}}
.kpi{{display:flex;gap:12px;margin:14px 0;flex-wrap:wrap}}
.kpi>div{{flex:1;min-width:150px;background:{SURF};border:1px solid rgba(11,11,11,.10);
border-radius:8px;padding:12px 14px}}
.kpi b{{display:block;font-size:22px;font-weight:600;color:{INK}}}
.kpi span{{font-size:11.5px;color:{INK2}}}
.two{{display:flex;gap:16px;align-items:flex-start}} .two>div{{flex:1}}
img{{width:100%;background:{SURF};border:1px solid rgba(11,11,11,.10);border-radius:8px;
padding:6px;box-sizing:border-box;margin:6px 0}}
table{{border-collapse:collapse;width:100%;font-size:12px;margin:8px 0;background:{SURF}}}
th,td{{border:1px solid {GRID};padding:5px 8px;text-align:right;font-variant-numeric:tabular-nums}}
th{{background:#f4f4f1;color:{INK2}}} th:first-child,td:first-child{{text-align:left}}
tbody tr:nth-child(even){{background:#f7f7f5}}
.tblwrap{{max-height:520px;overflow-y:auto;border:1px solid {GRID};border-radius:8px}}
.tblwrap table{{margin:0;border:none}}
.foot{{color:{MUTED};font-size:11px;margin-top:20px;border-top:1px solid {GRID};padding-top:9px}}
</style></head><body>

<h1>Executed trade log — {era['label']}</h1>
<div class="tag">Simulated account starting at ${INITIAL_NAV:,.0f}. Every trade below is a real
discrete position derived from the strategy's actual monthly weight history — entry/exit prices
and dollar P&L, not illustrative numbers.</div>

<div class="kpi">
<div><b>${final_nav:,.0f}</b><span>ending NAV<br>from ${INITIAL_NAV:,.0f}</span></div>
<div><b>{n}</b><span>executed trades<br>({avg_hold:.1f} mo avg hold)</span></div>
<div><b>{win_rate:.0%}</b><span>win rate<br>({s['n_win']}W / {s['n_loss']}L)</span></div>
<div><b>{s['ratio']:.2f}×</b><span>avg win / avg loss<br>(magnitude, $ basis)</span></div>
</div>

<h2>Live P&L</h2>
<img src="data:image/png;base64,{g_eq}">

<h2>Position timeline</h2>
<img src="data:image/png;base64,{g_tl}">

<h2>Win / loss profile</h2>
<div class="two">
<div><img src="data:image/png;base64,{g_wl}"></div>
<div>
<table>
<tr><th>Metric</th><th>Value</th></tr>
<tr><td>Avg winning trade ($)</td><td style="color:#0ca30c;font-weight:600">+${s['avg_win_d']:,.0f}</td></tr>
<tr><td>Avg losing trade ($)</td><td style="color:#d03b3b;font-weight:600">−${abs(s['avg_loss_d']):,.0f}</td></tr>
<tr><td>Avg winning trade (%)</td><td style="color:#0ca30c;font-weight:600">{s['avg_win_pct']:+.1%}</td></tr>
<tr><td>Avg losing trade (%)</td><td style="color:#d03b3b;font-weight:600">{s['avg_loss_pct']:+.1%}</td></tr>
<tr><td>Win / loss ratio ($)</td><td>{s['ratio']:.2f}×</td></tr>
<tr><td>Expectancy / trade ($)</td><td>{'+' if s['expectancy']>=0 else ''}${s['expectancy']:,.0f}</td></tr>
<tr><td>Best single trade</td><td style="color:#0ca30c">+${s['best']:,.0f}</td></tr>
<tr><td>Worst single trade</td><td style="color:#d03b3b">−${abs(s['worst']):,.0f}</td></tr>
</table>
</div>
</div>

<h2>Executed trade log ({n} trades)</h2>
<div class="tblwrap">
<table>
<tr><th>Asset</th><th>Entry</th><th>Exit</th><th>Hold (mo)</th><th>Entry px</th>
<th>Exit px</th><th>Price Δ</th><th>Avg weight</th><th>$ P&L</th></tr>
{table_rows}
</table>
</div>

<div class="foot">Entry price = close before the position's first held month; exit price = close
ending its last held month (matches the backtest's own signal-lag accounting exactly, not
approximated). $ P&L per trade = sum of each held month's actual weight × prior-month NAV ×
that month's asset return — reconciles with the strategy's realized compounding. The
{'+' if unattrib>=0 else ''}${unattrib:,.0f} gap between summed trade P&L and total NAV growth is
cash-carry interest earned while sitting in T-bills, net of 10 bps/side trading costs — neither is
attributed to a single trade, both are real and embedded in the NAV curve above.
Reproducible: <code>trade_log.py</code>.</div>
</body></html>"""
    out_path = os.path.join(HERE, era["out"])
    open(out_path, "w").write(html)
    print(f"Wrote {out_path}  ({n} trades, {win_rate:.0%} win rate, final NAV ${final_nav:,.0f})")
    print(f"  avg win ${s['avg_win_d']:,.0f} ({s['avg_win_pct']:+.1%})  |  "
          f"avg loss ${s['avg_loss_d']:,.0f} ({s['avg_loss_pct']:+.1%})  |  "
          f"ratio {s['ratio']:.2f}x  |  expectancy ${s['expectancy']:,.0f}/trade")
    return trades, s


def main():
    for era in (ERA_8ETF, ERA_LONGHIST):
        print("=" * 78)
        run_era(era)


if __name__ == "__main__":
    main()
