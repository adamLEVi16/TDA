"""Are bull put spreads on the S&P 500 a real edge?

Part 1 - real traded-price benchmarks from Cboe (1986/1988-2026):
  CNDR  iron condor: sells ~20-delta SPX put and call, buys ~5-delta wings, monthly, held to expiry
        (its put side is a 20/5-delta bull put spread)
  PUT   cash-secured at-the-money SPX put writing, monthly
  BXM   covered call (buy-write), for reference
  vs S&P 500 total return (^SP500TR). Cash / excess returns use the Ken French T-bill rate.

Part 2 - is the credit big enough? A mechanical ~1-month SPY bull put spread whose strikes sit the same number of
  VIX-implied standard deviations below spot as today's 20-delta / 5-delta strikes. Its payout at expiry is computed
  exactly from real S&P 500 prices (1990-2026, VIX needed for strike placement). The average payout is the break-even
  credit; compare it with the credit the market pays today (live SPY chain, sell at bid / buy at ask).
"""
import io, math, urllib.request, zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import norm

HERE = Path(__file__).parent
rng = np.random.default_rng(0)


def fetch(name, url):
    p = HERE / name
    if not p.exists():
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        p.write_bytes(urllib.request.urlopen(req).read())


for _t in ["CNDR", "PUT", "BXM"]:
    fetch(f"{_t}.csv", f"https://cdn-api.cboe.com/api/global/us_indices/daily_prices/{_t}_History.csv")
for _t in ["GSPC", "SP500TR", "VIX", "SPY"]:
    if not (HERE / f"{_t}.csv").exists():
        import yfinance as yf
        yf.download(("^" if _t != "SPY" else "") + _t, start="1950-01-01", end="2026-09-26", auto_adjust=True,
                    progress=False)["Close"].squeeze().rename(_t).to_csv(HERE / f"{_t}.csv")


def cboe(t):
    s = pd.read_csv(HERE / f"{t}.csv")
    return pd.Series(s[t].values, index=pd.to_datetime(s["DATE"]), name=t).sort_index()


def yf_csv(name):
    s = pd.read_csv(HERE / f"{name}.csv", index_col=0, parse_dates=True).squeeze()
    return s.sort_index()


def french_rf():
    url = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Research_Data_Factors_daily_CSV.zip"
    p = HERE / "ff3d.zip"
    if not p.exists():
        p.write_bytes(urllib.request.urlopen(url).read())
    with zipfile.ZipFile(p) as z:
        lines = z.read(z.namelist()[0]).decode("latin1").splitlines()
    rows = [l.split(",") for l in lines if l[:8].strip().isdigit() and len(l.split(",")) == 5]
    return pd.Series([float(r[4]) / 100 for r in rows], index=pd.to_datetime([r[0].strip() for r in rows]), name="RF")


rf_d = french_rf()
px = pd.concat([cboe("CNDR"), cboe("PUT"), cboe("BXM"), yf_csv("SP500TR").rename("SP500TR")], axis=1, sort=True)
END = rf_d.index[-1]
px = px.loc[:END]

# ---------------------------------------------------------------- Part 1: monthly stats
def stats(name, start, end=None):
    p = px[name].loc[start:end].dropna()
    m = p.resample("ME").last().pct_change().dropna()
    rf_m = (1 + rf_d).resample("ME").prod().reindex(m.index) - 1
    mk = px["SP500TR"].loc[start:end].resample("ME").last().pct_change().reindex(m.index)
    ex, mex = m - rf_m, mk - rf_m
    years = (p.index[-1] - p.index[0]).days / 365.25
    dd = (p / p.cummax() - 1)
    X = sm.add_constant(mex.values)
    reg = sm.OLS(ex.values, X).fit(cov_type="HAC", cov_kwds={"maxlags": 3})
    return {"series": name, "from": m.index[0].strftime("%Y-%m"), "CAGR %": 100 * ((p.iloc[-1] / p.iloc[0]) ** (1 / years) - 1),
            "vol %": 100 * m.std() * np.sqrt(12), "excess Sharpe": ex.mean() / ex.std() * np.sqrt(12),
            "max DD %": 100 * dd.min(), "worst month %": 100 * m.min(), "months up %": 100 * (m > 0).mean(),
            "beta": reg.params[1], "alpha %/yr": 100 * 12 * reg.params[0], "alpha t": reg.tvalues[0]}


def sharpe_diff_boot(a, b, reps=5000, block=6):
    a, b = a.values, b.values
    n = len(a)
    out = np.empty(reps)
    for r in range(reps):
        idx, i = np.empty(n, dtype=int), 0
        while i < n:
            L = rng.geometric(1 / block)
            seg = (rng.integers(n) + np.arange(L)) % n
            idx[i:i + L] = seg[: n - i]
            i += L
        sa, sb = a[idx], b[idx]
        out[r] = sa.mean() / sa.std() - sb.mean() / sb.std()
    out *= np.sqrt(12)
    return np.percentile(out, [2.5, 97.5]), 2 * min((out <= 0).mean(), (out >= 0).mean())


pd.set_option("display.width", 220)
for label, start, end in [("Full period", "1988-01-01", None), ("1988-2007", "1988-01-01", "2007-12-31"),
                          ("2008-2026", "2008-01-01", None), ("Last 10 years", "2016-08-01", None)]:
    rows = [stats(n, start, end) for n in ["CNDR", "PUT", "BXM", "SP500TR"]]
    print(f"\n=== {label}: through {min(pd.Timestamp(end) if end else END, END).date()} (monthly; excess over T-bills; "
          f"alpha/beta vs S&P 500 total return, NW t)")
    print(pd.DataFrame(rows).to_string(index=False, float_format=lambda v: f"{v:.2f}"))

m = px.loc["1988-01-01":].resample("ME").last().pct_change().dropna()
rf_m = (1 + rf_d).resample("ME").prod().reindex(m.index) - 1
for n in ["CNDR", "PUT"]:
    ci, p = sharpe_diff_boot(m[n] - rf_m, m["SP500TR"] - rf_m)
    print(f"Sharpe difference {n} minus S&P 500, 1988-{END.year}: CI [{ci[0]:+.2f}, {ci[1]:+.2f}], p {p:.2f}")

dd = px["CNDR"] / px["CNDR"].cummax() - 1
trough = dd.idxmin()
peak = px["CNDR"].loc[:trough].idxmax()
rec = px["CNDR"].loc[trough:][px["CNDR"].loc[trough:] >= px["CNDR"].loc[peak]]
print(f"CNDR worst drawdown {100 * dd.min():.1f}%: peak {peak.date()} -> trough {trough.date()}, "
      f"recovered {rec.index[0].date() if len(rec) else 'NOT YET (as of ' + str(px.index[-1].date()) + ')'}")
cm = px["CNDR"].resample("ME").last().pct_change().dropna()
print("CNDR worst 5 months:", {d.strftime('%Y-%m'): round(100 * v, 1) for d, v in cm.nsmallest(5).items()})
yr = px["CNDR"].resample("YE").last().pct_change().dropna().loc["2015":]
print("CNDR by calendar year (%):", {d.year: round(100 * v, 1) for d, v in yr.items()})

# ---------------------------------------------------------------- Part 2: break-even credit vs today's market credit
chain_file = HERE / "spy_puts_20261030.csv"
if not chain_file.exists():
    import yfinance as yf
    t = yf.Ticker("SPY")
    t.option_chain("2026-10-30").puts.to_csv(chain_file, index=False)
puts = pd.read_csv(chain_file)
spot = yf_csv("SPY").iloc[-1]
asof = yf_csv("SPY").index[-1]
T_cal = (pd.Timestamp("2026-10-30") - asof).days / 365.0
T_td = len(pd.bdate_range(asof + pd.Timedelta(days=1), "2026-10-30"))
vix_now = yf_csv("VIX").iloc[-1] / 100
puts = puts[(puts["bid"] > 0) & (puts["impliedVolatility"] > 0.01)].copy()
d1 = (np.log(spot / puts["strike"]) + 0.5 * puts["impliedVolatility"] ** 2 * T_cal) / (puts["impliedVolatility"] * np.sqrt(T_cal))
puts["delta"] = norm.cdf(d1) - 1
short = puts.iloc[(puts["delta"] + 0.20).abs().argmin()]
long_ = puts.iloc[(puts["delta"] + 0.05).abs().argmin()]
width = short["strike"] - long_["strike"]
credit_mid = (short["bid"] + short["ask"]) / 2 - (long_["bid"] + long_["ask"]) / 2
credit_fill = short["bid"] - long_["ask"]
z_s = math.log(short["strike"] / spot) / (vix_now * math.sqrt(T_cal))
z_l = math.log(long_["strike"] / spot) / (vix_now * math.sqrt(T_cal))
print(f"\n=== Live SPY chain, {asof.date()} close {spot:.2f}, expiry 2026-10-30 ({T_td} trading days), VIX {100 * vix_now:.1f}")
print(f"short put {short['strike']:.0f} (delta {short['delta']:.2f}, IV {100 * short['impliedVolatility']:.1f}%, bid/ask "
      f"{short['bid']:.2f}/{short['ask']:.2f}); long put {long_['strike']:.0f} (delta {long_['delta']:.2f}, IV "
      f"{100 * long_['impliedVolatility']:.1f}%, bid/ask {long_['bid']:.2f}/{long_['ask']:.2f})")
print(f"width {width:.0f}; credit at mid {credit_mid:.2f} ({100 * credit_mid / width:.1f}% of width); realistic fill "
      f"(sell bid, buy ask) {credit_fill:.2f} ({100 * credit_fill / width:.1f}% of width); strikes at z = {z_s:.2f} and "
      f"{z_l:.2f} VIX-sigmas")

# historical: same z-distances, every trading day from 1990, hold T_td trading days, payout at expiry from S&P 500 closes
spx = yf_csv("GSPC")
vix = yf_csv("VIX").reindex(spx.index).dropna() / 100
s = spx.loc[vix.index]
rows = []
vals = s.values
for i in range(len(s) - T_td):
    S0, v = vals[i], vix.values[i]
    Ks, Kl = S0 * math.exp(z_s * v * math.sqrt(T_cal)), S0 * math.exp(z_l * v * math.sqrt(T_cal))
    ST = vals[i + T_td]
    pay = max(0.0, Ks - ST) - max(0.0, Kl - ST)
    rows.append((s.index[i], v, pay / (Ks - Kl)))
h = pd.DataFrame(rows, columns=["date", "vix", "payout_frac"]).set_index("date")
print(f"\n=== Historical payout of the same spread (share of width), S&P 500, entries {h.index[0].date()}..{h.index[-1].date()} "
      f"(overlapping daily entries)")
print(f"average payout = break-even credit: {100 * h.payout_frac.mean():.1f}% of width | win rate (no payout): "
      f"{100 * (h.payout_frac == 0).mean():.1f}% | max loss (full width) frequency: {100 * (h.payout_frac >= 0.999).mean():.1f}%")
for lo, hi in [(0, 0.15), (0.15, 0.20), (0.20, 0.30), (0.30, 1.0)]:
    g = h[(h.vix >= lo) & (h.vix < hi)]
    print(f"  VIX {int(100 * lo)}-{int(100 * hi)}: n={len(g)}, break-even credit {100 * g.payout_frac.mean():.1f}% of width, "
          f"win rate {100 * (g.payout_frac == 0).mean():.1f}%")
for c, lab in [(credit_mid / width, "mid"), (credit_fill / width, "realistic fill")]:
    ev = c - h.payout_frac.mean()
    print(f"Expected profit per spread at today's {lab} credit ({100 * c:.1f}% of width): {100 * ev:+.1f}% of width = "
          f"{100 * ev / (1 - c):+.1f}% return on capital at risk per ~5-week trade, before commissions")
# month-by-month non-overlapping version, to show the path of a trader doing this every month at today's fill ratio
nonov = h.iloc[::T_td]
pnl = (credit_fill / width - nonov.payout_frac) / (1 - credit_fill / width)       # return on capital at risk
eq = (1 + pnl).cumprod()
print(f"\nNon-overlapping sequence ({len(nonov)} trades), all-in on capital at risk each trade, today's fill credit applied "
      f"historically: win rate {100 * (pnl > 0).mean():.1f}%, mean {100 * pnl.mean():+.2f}%/trade, worst trade "
      f"{100 * pnl.min():.1f}%, equity multiple {eq.iloc[-1]:.2f}, max drawdown {100 * (eq / eq.cummax() - 1).min():.1f}%")

# ---------------------------------------------------------------- Part 3: uncertainty, era split, call side, position sizing
se = nonov.payout_frac.std() / np.sqrt(len(nonov))
print(f"\n=== Precision: non-overlapping break-even estimate {100 * nonov.payout_frac.mean():.1f}% of width, SE {100 * se:.1f}% "
      f"-> 95% CI [{100 * (nonov.payout_frac.mean() - 1.96 * se):.1f}%, {100 * (nonov.payout_frac.mean() + 1.96 * se):.1f}%] "
      f"vs today's fill credit {100 * credit_fill / width:.1f}%")
for lab, a, b in [("1990-2007", "1990", "2007"), ("2008-2026", "2008", "2026")]:
    g = h.loc[a:b]
    print(f"  put-spread break-even {lab}: {100 * g.payout_frac.mean():.1f}% of width, win rate {100 * (g.payout_frac == 0).mean():.1f}%")

# call side of the condor (bear call spread at the mirror deltas), to see which leg hurt CNDR after 2008
cf = HERE / "spy_calls_20261030.csv"
if not cf.exists():
    import yfinance as yf
    yf.Ticker("SPY").option_chain("2026-10-30").calls.to_csv(cf, index=False)
calls = pd.read_csv(cf)
calls = calls[(calls["bid"] > 0) & (calls["impliedVolatility"] > 0.01)].copy()
d1c = (np.log(spot / calls["strike"]) + 0.5 * calls["impliedVolatility"] ** 2 * T_cal) / (calls["impliedVolatility"] * np.sqrt(T_cal))
calls["delta"] = norm.cdf(d1c)
cs = calls.iloc[(calls["delta"] - 0.20).abs().argmin()]
cl = calls.iloc[(calls["delta"] - 0.05).abs().argmin()]
cw = cl["strike"] - cs["strike"]
c_fill = cs["bid"] - cl["ask"]
zc_s = math.log(cs["strike"] / spot) / (vix_now * math.sqrt(T_cal))
zc_l = math.log(cl["strike"] / spot) / (vix_now * math.sqrt(T_cal))
pay_c = []
for i in range(len(s) - T_td):
    S0, v = vals[i], vix.values[i]
    Ks, Kl = S0 * math.exp(zc_s * v * math.sqrt(T_cal)), S0 * math.exp(zc_l * v * math.sqrt(T_cal))
    ST = vals[i + T_td]
    pay_c.append((max(0.0, ST - Ks) - max(0.0, ST - Kl)) / (Kl - Ks))
hc = pd.Series(pay_c, index=h.index)
print(f"\n=== Call side (bear call spread) today: sell {cs['strike']:.0f} / buy {cl['strike']:.0f}, fill credit {c_fill:.2f} = "
      f"{100 * c_fill / cw:.1f}% of width; historical break-even {100 * hc.mean():.1f}% "
      f"(1990-2007 {100 * hc.loc['1990':'2007'].mean():.1f}%, 2008-2026 {100 * hc.loc['2008':].mean():.1f}%)")

# position sizing: risk a fixed fraction f of the account per trade (max loss = f), non-overlapping monthly-ish trades
c = credit_fill / width
r_trade = (c - nonov.payout_frac) / (1 - c)            # return on capital at risk
years = (nonov.index[-1] - nonov.index[0]).days / 365.25
rf_ann = (1 + rf_d.loc[nonov.index[0]:]).prod() ** (1 / years) - 1
print(f"\n=== Position sizing (today's credit ratio applied to {len(nonov)} historical trades, {nonov.index[0].year}-"
      f"{nonov.index[-1].year}; unused cash earns nothing here; T-bills over the period ~{100 * rf_ann:.1f}%/yr)")
for f in [0.02, 0.05, 0.10, 0.20, 0.35, 0.50, 1.00]:
    eq = (1 + f * r_trade).cumprod()
    cagr = eq.iloc[-1] ** (1 / years) - 1 if eq.iloc[-1] > 0 else -1
    print(f"  risk {100 * f:>3.0f}% of account per trade: CAGR {100 * cagr:6.2f}%, max drawdown {100 * (eq / eq.cummax() - 1).min():6.1f}%, "
          f"worst single trade {100 * f * r_trade.min():6.1f}%")
