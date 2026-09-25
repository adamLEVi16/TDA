"""Pre-registered test of calendar-timed index exposure (see SPEC.md).

Replicates Cieslak-Morse-Vissing-Jorgensen (2019), Lucca-Moench (2015, close-to-close proxy) and
McConnell-Xu (2008) in their original samples, then evaluates the same fixed rules out of sample.
Run: python calendar_test.py  (needs fomc_dates.csv from fomc_dates.py)
"""
import io
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

HERE = Path(__file__).parent
CACHE = HERE / "cache"
COST_BP = 1.0
rng = np.random.default_rng(0)


# ------------------------------------------------------------------ data
def french_daily():
    CACHE.mkdir(exist_ok=True)
    z = CACHE / "ff3d.zip"
    if not z.exists():
        url = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Research_Data_Factors_daily_CSV.zip"
        z.write_bytes(urllib.request.urlopen(url).read())
    with zipfile.ZipFile(z) as f:
        lines = f.read(f.namelist()[0]).decode("latin1").splitlines()
    rows = [l.split(",") for l in lines if l[:8].strip().isdigit() and len(l.split(",")) == 5]
    df = pd.DataFrame([r[1:] for r in rows], index=pd.to_datetime([r[0].strip() for r in rows]),
                      columns=["MktRF", "SMB", "HML", "RF"]).astype(float) / 100
    assert (df.abs() < 0.5).all().all(), "missing-value codes present"
    return df[["MktRF", "RF"]]


def spy_returns(start, end):
    p = CACHE / "spy.csv"
    if not p.exists():
        import yfinance as yf
        yf.download("SPY", start="1993-01-01", end="2026-09-25", auto_adjust=True,
                    progress=False)["Close"].squeeze().rename("SPY").to_csv(p)
    px = pd.read_csv(p, index_col=0, parse_dates=True).squeeze()
    return px.pct_change().loc[start:end].dropna()


ff = french_daily()
fomc = pd.to_datetime(pd.read_csv(HERE / "fomc_dates.csv")["day0"])
print(f"Ken French daily: {ff.index[0].date()} .. {ff.index[-1].date()}; FOMC day-0 dates: {len(fomc)}")

# ------------------------------------------------------------------ FOMC cycle weeks (weekday clock, holidays included)
wd = pd.bdate_range("1993-06-01", "2027-12-31")                 # Mon-Fri, holidays kept (as in the paper)
pos = pd.Series(np.arange(len(wd)), index=wd)
day0_pos = np.sort(pos.reindex(fomc).dropna().astype(int).values)
def cycle_weeks(day0_pos, paper=True):
    """FOMC cycle week of every weekday. paper=True: CMVJ p.5 (day -1 -> week 0, days -6..-2 -> week -1,
    else floor((d+1)/5) from the last day 0). paper=False: the first-run spec (no week -1)."""
    out = pd.Series(np.nan, index=wd)
    for i, p in enumerate(pos.values):
        k = np.searchsorted(day0_pos, p, side="right") - 1      # most recent day 0 (<= today)
        nxt = day0_pos[k + 1] if k + 1 < len(day0_pos) else None
        if nxt is not None and nxt - p == 1:
            out.iloc[i] = 0
        elif paper and nxt is not None and 2 <= nxt - p <= 6:
            out.iloc[i] = -1
        elif k >= 0:
            out.iloc[i] = (p - day0_pos[k] + 1) // 5            # 0..3 -> 0, 4..8 -> 1, 9..13 -> 2, ...
    return out


cyc_week = cycle_weeks(day0_pos, paper=True)
cyc_week_v1 = cycle_weeks(day0_pos, paper=False)
even_wd = (cyc_week % 2 == 0)
even_wd_v1 = (cyc_week_v1 % 2 == 0)
is_day0 = pd.Series(wd.isin(fomc), index=wd)

# ------------------------------------------------------------------ turn of month on trading days
def tom_flags(idx):
    s = pd.Series(idx, index=idx)
    ym = idx.to_period("M")
    last = s.groupby(ym).transform("max") == s                  # last trading day of month
    rank = s.groupby(ym).cumcount()                             # 0 = first trading day
    return pd.Series(last.values | (rank.values <= 2), index=idx)   # last day + first 3 days


# ------------------------------------------------------------------ metrics
def nw_diff(ex, flag):
    X = sm.add_constant(flag.astype(float))
    m = sm.OLS(ex.values, X.values).fit(cov_type="HAC", cov_kwds={"maxlags": 5})
    return m.params[1], m.tvalues[1]


def strat(total, rf, w, cost_bp=COST_BP):
    w = w.astype(float)
    turn = w.diff().abs().fillna(w.iloc[0])
    return w * total + (1 - w) * rf - turn * cost_bp / 1e4


def sharpe(x):
    return x.mean() / x.std() * np.sqrt(252)


def cagr(total):
    years = (total.index[-1] - total.index[0]).days / 365.25
    return (1 + total).prod() ** (1 / years) - 1


def maxdd(total):
    c = (1 + total).cumprod()
    return (c / c.cummax() - 1).min()


def boot_sr_diff(a, b, reps=5000, block=20):
    a, b = a.values, b.values
    n = len(a)
    diffs = np.empty(reps)
    for r in range(reps):
        idx = np.empty(n, dtype=np.int64)
        i = 0
        while i < n:
            start = rng.integers(n)
            L = rng.geometric(1 / block)
            seg = (start + np.arange(L)) % n
            idx[i:i + L] = seg[: n - i]
            i += L
        sa, sb = a[idx], b[idx]
        diffs[r] = sa.mean() / sa.std() * np.sqrt(252) - sb.mean() / sb.std() * np.sqrt(252)
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    p = 2 * min((diffs <= 0).mean(), (diffs >= 0).mean())
    return lo, hi, min(p, 1.0)


def evaluate(name, total, rf, flag, cost_bp=COST_BP, do_boot=True):
    ex_mkt = total - rf
    diff, t = nw_diff(ex_mkt, flag)
    share = ex_mkt[flag].sum() / ex_mkt.sum() if ex_mkt.sum() != 0 else np.nan
    s = strat(total, rf, flag, cost_bp)
    sx, bx = s - rf, ex_mkt
    row = {"rule": name, "days": len(total), "in-mkt %": 100 * flag.mean(),
           "mean ex in (bp/day)": 1e4 * ex_mkt[flag].mean(), "mean ex out (bp/day)": 1e4 * ex_mkt[~flag].mean(),
           "diff t(NW)": t, "share of premium in window %": 100 * share,
           "strat SR": sharpe(sx), "BH SR": sharpe(bx),
           "strat CAGR %": 100 * cagr(s),
           "BH CAGR %": 100 * cagr(total),
           "strat MDD %": 100 * maxdd(s), "BH MDD %": 100 * maxdd(total)}
    if do_boot:
        lo, hi, p = boot_sr_diff(sx, bx)
        row.update({"dSR": sharpe(sx) - sharpe(bx), "dSR 95% CI": f"[{lo:+.2f}, {hi:+.2f}]", "p": p})
    return row


def run_period(label, start, end, rules, market="FF", cost_bp=COST_BP, do_boot=True):
    if market == "FF":
        d = ff.loc[start:end]
        total, rf = d["MktRF"] + d["RF"], d["RF"]
    else:
        r = spy_returns(start, end)
        rf = ff["RF"].reindex(r.index)
        keep = rf.notna()
        total, rf = r[keep], rf[keep]
    idx = total.index
    flags = {"EVEN": even_wd.reindex(idx).fillna(False).astype(bool),
             "EVEN_v1": even_wd_v1.reindex(idx).fillna(False).astype(bool),
             "FOMC0": is_day0.reindex(idx).fillna(False).astype(bool),
             "TOM": tom_flags(idx)}
    flags["UNION"] = flags["EVEN"] | flags["TOM"]
    out = [evaluate(k, total, rf, flags[k], cost_bp, do_boot) for k in rules]
    df = pd.DataFrame(out)
    print(f"\n=== {label}: {market} {idx[0].date()}..{idx[-1].date()}, cost {cost_bp:g} bp one-way")
    print(df.to_string(index=False, float_format=lambda v: f"{v:.2f}"))
    return df


pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 30)

# ------------------------------------------------------------------ replication checks against the papers
print("\n### Replication of CMVJ (1994-2016): 5-day excess return by FOMC week (paper: wk0 0.57%, wk2 0.33%, "
      "wk4 0.46%, wk6 0.60%; wk1 -0.18%, wk3 -0.18%, wk5 -0.09%) and day counts (paper: 920/924/831/120)")
d = ff.loc["1994-01-01":"2016-12-31"]
ex = (d["MktRF"]).reindex(pd.bdate_range("1994-01-01", "2016-12-31")).fillna(0.0)   # holidays -> 0 as in paper
for label, cw in [("paper definition", cyc_week), ("first-run definition (v1)", cyc_week_v1)]:
    wk = cw.reindex(ex.index)
    tab = pd.DataFrame({"days": wk.value_counts().sort_index(),
                        "mean 5-day excess %": (ex.groupby(wk).mean() * 5 * 100).round(2)})
    print(f"-- {label}"); print(tab.loc[-1:8].to_string())

rep = run_period("REPLICATION EVEN (CMVJ sample; paper: BH SR 0.45, even-week SR 0.92)", "1994-01-01", "2016-12-31",
                 ["EVEN", "EVEN_v1"], do_boot=False)
run_period("REPLICATION FOMC0 (Lucca-Moench sample, close-to-close proxy)", "1994-09-01", "2011-03-31", ["FOMC0"],
           do_boot=False)
run_period("REPLICATION TOM (McConnell-Xu sample)", "1926-07-01", "2005-12-31", ["TOM"], do_boot=False)

# ------------------------------------------------------------------ pre-registered out-of-sample tests
print("\n\n######## OUT OF SAMPLE (pre-registered) ########")
oos = [run_period("OOS EVEN + UNION (after CMVJ sample)", "2017-01-01", None, ["EVEN", "EVEN_v1", "UNION"]),
       run_period("OOS FOMC0 (after Lucca-Moench sample)", "2011-04-01", None, ["FOMC0"]),
       run_period("OOS TOM (after McConnell-Xu sample)", "2006-01-01", None, ["TOM"])]

print("\n######## ROBUSTNESS (pre-registered): 5 bp costs, SPY ########")
run_period("OOS at 5 bp", "2017-01-01", None, ["EVEN", "EVEN_v1", "UNION", "TOM"], cost_bp=5.0)
run_period("OOS TOM at 5 bp", "2006-01-01", None, ["TOM"], cost_bp=5.0)
run_period("SPY OOS", "2017-01-01", None, ["EVEN", "EVEN_v1", "UNION", "TOM"], market="SPY")
run_period("SPY OOS TOM", "2006-01-01", None, ["TOM"], market="SPY")

print("\n######## Sensitivity: drop the cancelled 2020-03-18 meeting ########")
fomc_alt = fomc[fomc != pd.Timestamp("2020-03-18")]
alt_pos = np.sort(pos.reindex(fomc_alt).dropna().astype(int).values)
even_keep, even_v1_keep = even_wd, even_wd_v1
even_wd = cycle_weeks(alt_pos, paper=True) % 2 == 0
even_wd_v1 = cycle_weeks(alt_pos, paper=False) % 2 == 0
run_period("OOS EVEN without cancelled meeting", "2017-01-01", None, ["EVEN", "EVEN_v1", "UNION"])
day0_keep = is_day0
is_day0 = pd.Series(wd.isin(fomc_alt), index=wd)
run_period("OOS FOMC0 without cancelled meeting (no announcement was made on 2020-03-18)", "2011-04-01", None, ["FOMC0"])
run_period("SPY OOS FOMC0 without cancelled meeting", "2011-04-01", None, ["FOMC0"], market="SPY")
is_day0 = day0_keep
even_wd, even_wd_v1 = even_keep, even_v1_keep

print("\n######## Descriptive: yearly excess return earned in/out of EVEN weeks (paper definition), 2017+ (not a test) ########")
d = ff.loc["2017-01-01":]
f = even_wd.reindex(d.index).fillna(False).astype(bool)
yr = pd.DataFrame({"even %": d["MktRF"][f].groupby(d.index[f].year).sum() * 100,
                   "odd %": d["MktRF"][~f].groupby(d.index[~f].year).sum() * 100}).round(1)
print(yr.to_string())

print("\n######## Descriptive (not pre-registered): timing rule vs holding the SAME average exposure every day ########")
print("Both sides WITHOUT trading costs. If the timing carries information, the rule should beat a constant mix with the")
print("same average market weight (daily-rebalanced).")
for label, start in [("EVEN", "2017-01-01"), ("UNION", "2017-01-01"), ("TOM", "2006-01-01")]:
    d = ff.loc[start:]
    total, rf = d["MktRF"] + d["RF"], d["RF"]
    idx = total.index
    fl = {"EVEN": even_wd.reindex(idx).fillna(False).astype(bool), "TOM": tom_flags(idx)}
    fl["UNION"] = fl["EVEN"] | fl["TOM"]
    f = fl[label]
    s0 = strat(total, rf, f, cost_bp=0.0)
    wbar = f.mean()
    const = wbar * total + (1 - wbar) * rf
    print(f"{label:<6} {idx[0].date()}..{idx[-1].date()}  avg weight {wbar:.2f} | timing rule: SR {sharpe(s0 - rf):.2f}, "
          f"CAGR {100 * cagr(s0):.2f}%, MDD {100 * maxdd(s0):.1f}% | constant mix: SR {sharpe(const - rf):.2f}, "
          f"CAGR {100 * cagr(const):.2f}%, MDD {100 * maxdd(const):.1f}%")

print("\n######## Replication on the paper's own metric: Sharpe of calendar-year excess returns, no costs, 1994-2016 ########")
print("(CMVJ Table 1 Panel C: strategy A always-in 0.45, B even weeks 0.92, C odd weeks negative)")
d = ff.loc["1994-01-01":"2016-12-31"]
f = even_wd.reindex(d.index).fillna(False).astype(bool)
for name, w in [("A always in", pd.Series(True, index=d.index)), ("B even weeks", f), ("C odd weeks", ~f)]:
    ex_d = np.where(w, d["MktRF"], 0.0)
    yr = pd.Series((1 + ex_d + d["RF"].values), index=d.index).groupby(d.index.year).prod() - \
        (1 + d["RF"]).groupby(d.index.year).prod()
    print(f"{name:<13} annual excess mean {100 * yr.mean():.2f}%, sd {100 * yr.std():.2f}%, Sharpe {yr.mean() / yr.std():.2f}")

print("\n######## Weeks beyond 6 in the out-of-sample EVEN definition (paper invests in weeks 0,2,4,6) ########")
w_oos = cyc_week.reindex(ff.loc["2017-01-01":].index)
print("OOS trading days by FOMC week:", w_oos.value_counts().sort_index().to_dict())
