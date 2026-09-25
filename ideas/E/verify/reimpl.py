"""Independent re-implementation of the core numbers (verifier). Written from the paper/SPEC definitions, not by
editing calendar_test.py. Differences in approach: cycle offsets via np.busday_count (weekday clock, no holiday
list = holidays included); own Newey-West; own stationary bootstrap with a different seed."""
import io
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
pd.set_option("display.width", 220)

# ---------------- data (own parser)
raw = zipfile.ZipFile(ROOT / "cache/ff3d.zip").read("F-F_Research_Data_Factors_daily.csv").decode("latin1")
recs = []
for line in raw.splitlines():
    parts = [p.strip() for p in line.split(",")]
    if len(parts) == 5 and parts[0].isdigit() and len(parts[0]) == 8:
        recs.append((pd.Timestamp(parts[0]), float(parts[1]) / 100, float(parts[4]) / 100))
ff = pd.DataFrame(recs, columns=["date", "mktrf", "rf"]).set_index("date")
ff["mkt"] = ff.mktrf + ff.rf

day0 = pd.to_datetime(pd.read_csv(ROOT / "fomc_dates.csv")["day0"]).sort_values().values.astype("datetime64[D]")


def fomc_week(dates, d0, paper=True):
    """CMVJ week for each date (weekday clock, holidays included). NaN if undefined."""
    dates = np.asarray(dates, dtype="datetime64[D]")
    wk = np.full(len(dates), np.nan)
    j_prev = np.searchsorted(d0, dates, side="right") - 1
    for i, (t, jp) in enumerate(zip(dates, j_prev)):
        nxt = d0[jp + 1] if jp + 1 < len(d0) else None
        to_next = np.busday_count(t, nxt) if nxt is not None else 999       # weekdays from t to next day 0
        if to_next == 1:
            wk[i] = 0
        elif paper and 2 <= to_next <= 6:
            wk[i] = -1
        elif jp >= 0:
            since = np.busday_count(d0[jp], t)                             # day index d >= 0
            wk[i] = 0 if since <= 3 else 1 + (since - 4) // 5              # 4-8 ->1, 9-13 ->2, ...
    return wk


def tom(idx):
    s = pd.Series(1, index=idx)
    g = s.groupby([idx.year, idx.month])
    first3 = g.cumcount() < 3
    last = g.cumcount(ascending=False) == 0
    return (first3 | last).values


def newey_west_t(y, x, L=5):
    X = np.column_stack([np.ones(len(y)), x.astype(float)])
    b = np.linalg.lstsq(X, y, rcond=None)[0]
    u = y - X @ b
    Xu = X * u[:, None]
    S = Xu.T @ Xu
    for l in range(1, L + 1):
        w = 1 - l / (L + 1)
        G = Xu[l:].T @ Xu[:-l]
        S += w * (G + G.T)
    XtXi = np.linalg.inv(X.T @ X)
    V = XtXi @ S @ XtXi
    return b[1], b[1] / np.sqrt(V[1, 1]), np.sqrt(V[1, 1])


def sr(x):
    return np.mean(x) / np.std(x, ddof=1) * np.sqrt(252)


def strategy(mkt, rf, w, cost_bp):
    w = np.asarray(w, dtype=float)
    dw = np.abs(np.diff(np.concatenate([[0.0], w])))
    return w * mkt + (1 - w) * rf - dw * cost_bp * 1e-4


def stat_boot_p(a, b, reps=5000, mean_block=20, seed=12345):
    """Politis-Romano stationary bootstrap of SR(a)-SR(b); returns observed diff, 95% CI, two-sided p
    (null-centred: share of |d* - d| >= |d|)."""
    r = np.random.default_rng(seed)
    n = len(a)
    d_obs = sr(a) - sr(b)
    out = np.empty(reps)
    p_new = 1.0 / mean_block
    pos = np.arange(n)
    for k in range(reps):
        newblk = r.random(n) < p_new
        newblk[0] = True
        starts = r.integers(n, size=n)
        bstart = np.maximum.accumulate(np.where(newblk, pos, 0))      # position where the current block began
        idx = (starts[bstart] + (pos - bstart)) % n                    # wrap-around (circular) blocks
        out[k] = sr(a[idx]) - sr(b[idx])
    lo, hi = np.percentile(out, [2.5, 97.5])
    p_centred = np.mean(np.abs(out - d_obs) >= abs(d_obs))
    p_pct = min(1.0, 2 * min(np.mean(out <= 0), np.mean(out >= 0)))
    return d_obs, lo, hi, p_centred, p_pct


def report(label, d, flag, cost_bp=1.0, boot=True):
    ex = (d.mkt - d.rf).values
    f = np.asarray(flag, dtype=bool)
    diff, t, se = newey_west_t(ex, f)
    s = strategy(d.mkt.values, d.rf.values, f, cost_bp)
    sx, bx = s - d.rf.values, ex
    row = dict(rule=label, n=len(d), pct_in=100 * f.mean(), ex_in_bp=1e4 * ex[f].mean(), ex_out_bp=1e4 * ex[~f].mean(),
               diff_bp=1e4 * diff, se_bp=1e4 * se, t_nw=t, share=100 * ex[f].sum() / ex.sum(), SR_strat=sr(sx),
               SR_bh=sr(bx), CAGR_s=100 * (np.prod(1 + s) ** (252 / len(s)) - 1),
               CAGR_bh=100 * (np.prod(1 + d.mkt.values) ** (252 / len(s)) - 1))
    if boot:
        dd, lo, hi, pc, pp = stat_boot_p(sx, bx)
        row.update(dSR=dd, CI=f"[{lo:+.2f},{hi:+.2f}]", p_centred=pc, p_pct=pp)
    return row


def show(rows, title):
    print(f"\n== {title}")
    print(pd.DataFrame(rows).to_string(index=False, float_format=lambda v: f"{v:.2f}"))


if __name__ == "__main__":
    import sys
    boot = "--noboot" not in sys.argv
    # ---- replication: day counts (weekdays incl. holidays) with and without the 1993-12-21 meeting
    wd = pd.bdate_range("1994-01-01", "2016-12-31")
    exwd = ff.mktrf.reindex(wd).fillna(0.0)
    for lab, d0 in [("analyst list", day0), ("+1993-12-21", np.sort(np.append(day0, np.datetime64("1993-12-21"))))]:
        wk = pd.Series(fomc_week(wd.values, d0), index=wd)
        tab = pd.DataFrame({"days": wk.value_counts().sort_index(), "5d_ex_%": exwd.groupby(wk).mean() * 500})
        print(f"\n-- CMVJ weeks 1994-2016, {lab}: weekdays={len(wd)}, NaN={wk.isna().sum()}")
        print(tab.T.round(2).to_string())
        # paper Panel C: hold in weeks 0,2,4,6 ; annual-return Sharpe, $1 value, no costs, holidays -> 0
        d = ff.loc["1994-01-01":"2016-12-31"]
        w = pd.Series(fomc_week(d.index.values, d0), index=d.index)
        even = (w % 2 == 0).values
        for nm, ff_flag in [("A all days", np.ones(len(d), bool)), ("B even", even), ("C odd", (~even) & w.notna().values)]:
            s = np.where(ff_flag, d.mkt.values, d.rf.values)
            yr = pd.Series(s, index=d.index).groupby(d.index.year).apply(lambda x: np.prod(1 + x) - 1)
            rfy = d.rf.groupby(d.index.year).apply(lambda x: np.prod(1 + x) - 1)
            exy = yr - rfy
            print(f"   {nm:<11} annual ex mean {100*exy.mean():6.2f}  sd {100*exy.std():6.2f}  SR(annual) {exy.mean()/exy.std():.2f}"
                  f"  daily-SR {sr(s - d.rf.values):.2f}  $1->{np.prod(1+s):.2f}")

    # ---- replications (papers' samples), 1 bp cost
    rows = []
    d = ff.loc["1994-01-01":"2016-12-31"]
    rows.append(report("EVEN 94-16", d, fomc_week(d.index.values, day0) % 2 == 0, boot=False))
    d = ff.loc["1994-09-01":"2011-03-31"]
    rows.append(report("FOMC0 94/09-11/03", d, np.isin(d.index.values.astype("datetime64[D]"), day0), boot=False))
    d = ff.loc["1926-07-01":"2005-12-31"]
    rows.append(report("TOM 26-05", d, tom(d.index), boot=False))
    show(rows, "Replication (1 bp)")

    # ---- out of sample
    rows = []
    d = ff.loc["2017-01-01":]
    ev = fomc_week(d.index.values, day0) % 2 == 0
    rows.append(report("EVEN 17-", d, ev, boot=boot))
    rows.append(report("EVEN_v1 17-", d, fomc_week(d.index.values, day0, paper=False) % 2 == 0, boot=boot))
    rows.append(report("UNION 17-", d, ev | tom(d.index), boot=boot))
    d = ff.loc["2011-04-01":]
    rows.append(report("FOMC0 11/04-", d, np.isin(d.index.values.astype("datetime64[D]"), day0), boot=boot))
    d = ff.loc["2006-01-01":]
    rows.append(report("TOM 06-", d, tom(d.index), boot=boot))
    show(rows, "Out of sample (1 bp)")
