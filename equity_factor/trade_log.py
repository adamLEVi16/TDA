"""
EXECUTED TRADE LOG + LIVE P&L -- turns the monthly weight history into what a
real account statement looks like: discrete trades (entry/exit price, hold
length, realized $ P&L), a per-asset holding timeline, and an equity curve.

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

OUT = os.path.join(os.path.dirname(__file__), "trade_log.html")
INITIAL_NAV = 1_000_000

# validated palette (dataviz reference) -- reused from workflow.py so asset
# colors are consistent across every artifact in this repo
SURF, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
NAMES = {"SPY": "US stocks", "EFA": "Intl stocks", "EEM": "EM stocks", "TLT": "Long bonds",
         "IEF": "Mid bonds", "GLD": "Gold", "DBC": "Commodities", "VNQ": "Real estate"}
COL = {"SPY": "#2a78d6", "EFA": "#1baf7a", "EEM": "#eda100", "TLT": "#008300", "IEF": "#4a3aa7",
       "GLD": "#e34948", "DBC": "#e87ba4", "VNQ": "#eb6834"}

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


def build_trade_log(bt, w, mpx):
    """One row per discrete holding run per asset."""
    nav_prev = (INITIAL_NAV * (1 + bt["RP+Trend"]).cumprod()).shift(1)
    nav_prev.iloc[0] = INITIAL_NAV
    mret = mpx.pct_change()
    all_dates = mpx.index          # includes the pre-history date before w's first row
    trades = []
    for asset in MA.ASSETS:
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


def chart_equity(bt):
    nav = INITIAL_NAV * (1 + bt["RP+Trend"]).cumprod()
    spy_nav = INITIAL_NAV * (1 + bt["SPY"]).cumprod()
    fig, ax = plt.subplots(figsize=(9.0, 3.4))
    ax.plot(spy_nav.index, spy_nav.values, color="#eb6834", lw=1.6, label="US equity (SPY)", alpha=0.85)
    ax.plot(nav.index, nav.values, color="#2a78d6", lw=2.2, label="Strategy (live P&L)")
    ax.fill_between(nav.index, nav.values, INITIAL_NAV, color="#2a78d6", alpha=0.08, lw=0)
    ax.axhline(INITIAL_NAV, color=AXIS, lw=1, ls=(0, (2, 2)))
    ax.yaxis.set_major_formatter(lambda v, _: f"${v/1e6:.1f}M" if v >= 1e6 else f"${v/1e3:.0f}K")
    ax.annotate(f" ${nav.iloc[-1]:,.0f}", xy=(nav.index[-1], nav.iloc[-1]),
                fontsize=10, color=INK, weight="bold", va="center")
    ax.annotate(f" ${spy_nav.iloc[-1]:,.0f}", xy=(spy_nav.index[-1], spy_nav.iloc[-1]),
                fontsize=9.5, color=INK2, va="center")
    ax.legend(fontsize=9, frameon=False, loc="upper left")
    ax.set_xlim(nav.index[0], nav.index[-1] + pd.DateOffset(years=2))
    ax.set_title(f"Live P&L on ${INITIAL_NAV:,.0f} starting capital",
                 fontsize=11, loc="left", color=INK2)
    return png(fig)


def chart_timeline(trades, w):
    assets = MA.ASSETS
    fig, ax = plt.subplots(figsize=(9.0, 3.6))
    ax.grid(axis="y", visible=False)
    y = {a: i for i, a in enumerate(reversed(assets))}
    for _, t in trades.iterrows():
        yi = y[t["asset"]]
        width_days = (t["exit"] - t["entry"]).days
        ax.add_patch(FancyBboxPatch((mdates.date2num(t["entry"]), yi - 0.32),
                     width_days, 0.64, boxstyle="round,pad=0,rounding_size=0",
                     mutation_aspect=1, fc=COL[t["asset"]], ec="none", alpha=0.9))
    ax.set_ylim(-0.6, len(assets) - 0.4)
    ax.set_yticks(list(y.values()), [f"{a} — {NAMES[a]}" for a in y.keys()], fontsize=9)
    ax.xaxis_date()
    ax.set_xlim(w.index[0], w.index[-1])
    ax.set_title("Position timeline — when each asset was actually held",
                 fontsize=11, loc="left", color=INK2)
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


def main():
    bt, w = MA.run()
    mpx = MA.load_prices(assets=MA.ASSETS, start="2005-01-01")
    trades = build_trade_log(bt, w, mpx)
    trades.to_csv(os.path.join(os.path.dirname(__file__), "trade_log.csv"), index=False)

    n = len(trades)
    win_rate = float((trades["pnl"] > 0).mean())
    total_pnl = float(trades["pnl"].sum())
    avg_hold = float(trades["hold_mo"].mean())
    final_nav = INITIAL_NAV * (1 + bt["RP+Trend"]).prod()
    unattrib = (final_nav - INITIAL_NAV) - total_pnl   # cash carry net of costs

    g_eq = chart_equity(bt)
    g_tl = chart_timeline(trades, w)
    table_rows = trade_table_html(trades)

    html = f"""<!DOCTYPE html><html><head><meta charset="utf-8">
<title>Executed trade log — multi-asset trend strategy</title><style>
body{{font-family:system-ui,-apple-system,"Segoe UI",sans-serif;max-width:960px;margin:0 auto;
padding:28px;color:{INK};background:#f9f9f7;font-size:14px;line-height:1.5}}
h1{{font-size:22px;margin:0 0 2px}} .tag{{color:{INK2};margin:0 0 18px}}
h2{{font-size:13px;text-transform:uppercase;letter-spacing:.6px;color:{INK2};
border-bottom:1px solid {GRID};padding-bottom:4px;margin:26px 0 10px}}
.kpi{{display:flex;gap:12px;margin:14px 0}}
.kpi>div{{flex:1;background:{SURF};border:1px solid rgba(11,11,11,.10);border-radius:8px;padding:12px 14px}}
.kpi b{{display:block;font-size:22px;font-weight:600;color:{INK}}}
.kpi span{{font-size:11.5px;color:{INK2}}}
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

<h1>Executed trade log — multi-asset trend + risk parity</h1>
<div class="tag">Simulated account starting at ${INITIAL_NAV:,.0f}, 2007–2024, 8-ETF live universe.
Every trade below is a real discrete position derived from the strategy's actual monthly weight
history — entry/exit prices and dollar P&L, not illustrative numbers.</div>

<div class="kpi">
<div><b>${final_nav:,.0f}</b><span>ending NAV<br>from ${INITIAL_NAV:,.0f}</span></div>
<div><b>{n}</b><span>executed trades<br>({avg_hold:.1f} mo avg hold)</span></div>
<div><b>{win_rate:.0%}</b><span>win rate<br>(trades with positive $ P&L)</span></div>
<div><b>{'+' if total_pnl>=0 else ''}${total_pnl:,.0f}</b><span>sum of trade P&L<br>(reconciles to NAV above cash carry)</span></div>
</div>

<h2>Live P&L</h2>
<img src="data:image/png;base64,{g_eq}">

<h2>Position timeline</h2>
<img src="data:image/png;base64,{g_tl}">

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
    open(OUT, "w").write(html)
    print(f"Wrote {OUT}  ({n} trades, {win_rate:.0%} win rate, final NAV ${final_nav:,.0f})")


if __name__ == "__main__":
    main()
