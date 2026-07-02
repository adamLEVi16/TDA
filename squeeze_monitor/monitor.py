"""
SQUEEZE / CROWDING EARLY-WARNING MONITOR -- a risk tool for a short book,
not an alpha strategy. Question it answers: among names that are already
crowded shorts, does an abnormal PUBLIC-ATTENTION spike warn that the right
tail (a squeeze) is coming?

PRE-COMMITTED RULE (set before evaluation, from the two ingredients we already
validated/possess -- not tuned on outcomes):
  CROWDED  : latest FINRA days-to-cover >= 4, using only reports already
             PUBLISHED by that Friday (settlement + 9 business days).
  SPIKE    : weekly Wikipedia views >= 2x the trailing 8-week median
             (ASVI >= ln 2), fully observable that week.
  FLAG     : CROWDED and SPIKE.

Evaluation (also pre-committed):
  [1] Among CROWDED name-weeks only (the fair baseline for a short book):
      forward 1/2/4-week return distribution, flag vs no-flag -- especially the
      right tail P(fwd4 >= +15%) and P(fwd4 >= +25%), which is what hurts a short.
  [2] Episode table: known squeezes (GME Jan-21, AMC Jan/May-21) -- was the
      flag on BEFORE/DURING the explosive weeks?
  [3] Alarm budget: flags per year (a monitor nobody can act on is useless).
  [4] Significance: cluster-aware bootstrap on the tail-probability difference
      (flags cluster in time; naive iid p-values would overstate).

Survivorship note: delisted squeeze names (BBBY, EXPR) can't be fetched; the
episode evidence leans on still-listed GME/AMC. Stated, not hidden.
"""
import warnings; warnings.filterwarnings("ignore")
import os, sys
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "equity_factor"))
sys.path.insert(0, os.path.join(_HERE, "..", "attention_alpha"))
import numpy as np, pandas as pd

from sq_universe import TICKER_ARTICLE, SQUEEZE_COHORT
import wiki_data as WD
import price_history as PH
from short_interest import get_short_interest

DTC_MIN = 4.0
ASVI_MIN = np.log(2)
START, END = "2018-06-01", "2024-12-31"
MED_WEEKS = 8


def weekly_asvi(pv):
    wk = pv.resample("W-FRI").mean()
    logv = np.log(wk.clip(lower=1))
    return logv - logv.shift(1).rolling(MED_WEEKS).median()


def build_panel(verbose=True):
    tickers = list(TICKER_ARTICLE.keys())
    px = PH.get_prices_long(tickers + ["SPY"], verbose=False)
    pv = WD.get_pageviews(list(TICKER_ARTICLE.values()), verbose=False)
    si = get_short_interest(tickers, verbose=False)
    art2tk = {v: k for k, v in TICKER_ARTICLE.items()}

    wret = px.resample("W-FRI").last().pct_change()
    rows = []
    for article, s in pv.items():
        tk = art2tk[article]
        if tk not in px.columns or tk not in si:
            if verbose: print(f"  [panel] skip {tk}")
            continue
        a = weekly_asvi(s)
        sdf = si[tk]
        r = wret[tk]
        idx = a.index.intersection(r.index)
        idx = idx[(idx >= START) & (idx <= END)]
        for t in idx:
            pub = sdf[sdf["avail"] <= t]
            if not len(pub) or np.isnan(a.get(t, np.nan)):
                continue
            dtc = float(pub["dtc"].iloc[-1])
            rows.append({"ticker": tk, "week": t, "asvi": float(a[t]), "dtc": dtc,
                         "meme": tk in SQUEEZE_COHORT})
    panel = pd.DataFrame(rows)
    # forward returns
    fwd = {}
    for h in (1, 2, 4):
        f = (wret.shift(-h).rolling(h).apply(lambda x: (1 + x).prod() - 1, raw=False)
             if h > 1 else wret.shift(-1))
        fwd[h] = f
    for h in (1, 2, 4):
        panel[f"fwd{h}"] = [
            fwd[h][r.ticker].get(r.week, np.nan) for r in panel.itertuples()]
    panel["crowded"] = panel["dtc"] >= DTC_MIN
    panel["spike"] = panel["asvi"] >= ASVI_MIN
    panel["flag"] = panel["crowded"] & panel["spike"]
    return panel.dropna(subset=["fwd4"])


def tail_stats(df, label):
    f4 = df["fwd4"]
    return {"label": label, "n": len(df),
            "mean4": f4.mean(), "med4": f4.median(),
            "p15": (f4 >= 0.15).mean(), "p25": (f4 >= 0.25).mean(),
            "worst_for_short": f4.max()}


def cluster_bootstrap_taildiff(cr, thresh=0.15, n=4000, seed=21):
    """Difference in P(fwd4>=thresh) flag-vs-noflag among crowded weeks,
    resampling whole WEEKS (preserves the cross-name clustering of flags)."""
    rng = np.random.default_rng(seed)
    weeks = cr["week"].unique()
    by_week = {w: g for w, g in cr.groupby("week")}
    def taildiff(sample_weeks):
        d = pd.concat([by_week[w] for w in sample_weeks])
        a = d[d["flag"]]["fwd4"]; b = d[~d["flag"]]["fwd4"]
        if len(a) < 5 or len(b) < 5:
            return np.nan
        return (a >= thresh).mean() - (b >= thresh).mean()
    real = taildiff(weeks)
    null = []
    for _ in range(n):
        null.append(taildiff(rng.choice(weeks, len(weeks))))
    null = np.array([x for x in null if np.isfinite(x)])
    p = (null <= 0).mean()          # one-sided: flag tail NOT bigger
    return real, np.percentile(null, [2.5, 97.5]), p


# ---------------------------------------------------------------------------
# V2 -- post-hoc engineering iteration, clearly labeled. V1's episode check
# exposed a MECHANICAL failure: days-to-cover collapses during a squeeze
# because the volume denominator explodes (GME Jan-29-21: SI still 61M shares,
# but DTC "fell" to 2.1). V2 fixes crowding to be volume-independent and adds
# daily attention timing + max-adverse-excursion (what a short actually fears).
# ---------------------------------------------------------------------------
SI_PCTL = 0.75
DAILY_SPIKE_X = 3.0

def si_percentile(sdf, t, min_reports=20, window=48):
    """Published-only SI level vs its own trailing ~2y of reports (volume-free)."""
    pub = sdf[sdf["avail"] <= t].tail(window)
    if len(pub) < min_reports:
        return np.nan
    cur = pub["si"].iloc[-1]
    return (pub["si"].iloc[:-1] < cur).mean()

def add_v2(panel, px):
    si = get_short_interest(list(TICKER_ARTICLE.keys()), verbose=False)
    panel = panel.copy()
    panel["si_pctl"] = [si_percentile(si[r.ticker], r.week) if r.ticker in si
                        else np.nan for r in panel.itertuples()]
    panel["crowded2"] = panel["si_pctl"] >= SI_PCTL
    panel["flag2"] = panel["crowded2"] & panel["spike"]
    # max adverse excursion for a short: worst weekly close over the next 4 wks
    wpx = px.resample("W-FRI").last()
    def mae(tk, t):
        s = wpx[tk]
        if t not in s.index:
            return np.nan
        i = s.index.get_loc(t)
        if i + 4 >= len(s):
            return np.nan
        return s.iloc[i + 1:i + 5].max() / s.iloc[i] - 1
    panel["mae4"] = [mae(r.ticker, r.week) for r in panel.itertuples()]
    return panel

def daily_flag_dates(tk, article, sdf, start, end):
    """Daily-resolution flag timing: daily views >= 3x trailing 56d median AND
    volume-free crowding on the most recent published report."""
    pv = WD.get_pageviews([article], verbose=False).get(article)
    if pv is None:
        return []
    ratio = pv / pv.shift(1).rolling(56).median()
    out = []
    for d, x in ratio.loc[start:end].items():
        if np.isfinite(x) and x >= DAILY_SPIKE_X:
            if si_percentile(sdf, d) >= SI_PCTL:
                out.append((d, x))
    return out


def main():
    panel = build_panel()
    panel.to_csv(os.path.join(_HERE, "panel.csv"), index=False)
    cr = panel[panel["crowded"]]
    print("=" * 84)
    print(f"  SQUEEZE MONITOR EVALUATION  {panel['week'].min().date()} -> "
          f"{panel['week'].max().date()}   ({panel['ticker'].nunique()} names, "
          f"{len(panel)} name-weeks, {len(cr)} crowded)")
    print(f"  rule: DTC>={DTC_MIN:.0f} (published data only) AND views>=2x 8-wk median")
    print("=" * 84)

    print("\n[1] AMONG CROWDED SHORTS: does the attention spike warn of the right tail?")
    rows = [tail_stats(cr[cr["flag"]], "CROWDED + SPIKE (flag)"),
            tail_stats(cr[~cr["flag"]], "crowded, no spike"),
            tail_stats(panel[~panel["crowded"]], "not crowded (context)")]
    print(f"{'group':<26}{'n':>6}{'mean4w':>9}{'med4w':>8}{'P(+15%)':>9}{'P(+25%)':>9}{'max4w':>8}")
    for r in rows:
        print(f"{r['label']:<26}{r['n']:>6}{r['mean4']:>9.1%}{r['med4']:>8.1%}"
              f"{r['p15']:>9.1%}{r['p25']:>9.1%}{r['worst_for_short']:>8.0%}")

    print("\n[2] SIGNIFICANCE (week-cluster bootstrap on the tail difference)")
    for thr in (0.15, 0.25):
        real, ci, p = cluster_bootstrap_taildiff(cr, thr)
        print(f"  ΔP(fwd4>=+{thr:.0%}) flag-vs-noflag = {real:+.1%}   "
              f"boot95=[{ci[0]:+.1%},{ci[1]:+.1%}]   p(one-sided)={p:.3f}")

    print("\n[3] EPISODE CHECK — canonical squeezes (flag on before/during?)")
    for tk, eps in [("GME", ("2020-11-01", "2021-03-01")),
                    ("AMC", ("2021-01-01", "2021-07-01")),
                    ("BYND", ("2019-05-01", "2019-09-01")),
                    ("CVNA", ("2023-05-01", "2023-09-01"))]:
        sub = panel[(panel["ticker"] == tk) &
                    (panel["week"] >= eps[0]) & (panel["week"] <= eps[1])]
        if not len(sub):
            print(f"  {tk}: no data in window {eps}")
            continue
        fl = sub[sub["flag"]]
        print(f"  {tk} {eps[0][:7]}..{eps[1][:7]}: {len(fl)}/{len(sub)} weeks flagged"
              + (f"; first flag {fl['week'].min().date()}, fwd4 on flag weeks: "
                 f"{', '.join(f'{x:+.0%}' for x in fl['fwd4'].head(6))}" if len(fl) else ""))

    print("\n[4] ALARM BUDGET (can a PM actually act on this?)")
    flags = panel[panel["flag"]]
    per_yr = flags.groupby(flags["week"].dt.year).size()
    print(f"  flags/year: {dict(per_yr)}")
    print(f"  most-flagged names: {dict(flags['ticker'].value_counts().head(6))}")

    print("\n[5] SHORT-BOOK P&L VIEW (what stepping aside was worth)")
    a = cr[cr["flag"]]["fwd4"]; b = cr[~cr["flag"]]["fwd4"]
    print(f"  avg 4-wk move of a crowded short WITH flag : {a.mean():+.1%} "
          f"(a LOSS this size, on average, for a short holding through it)")
    print(f"  avg 4-wk move of a crowded short, no flag  : {b.mean():+.1%}")
    print(f"  -> average avoided loss per flagged episode ≈ {a.mean()-b.mean():+.1%} "
          f"of position, {len(a)} episodes")

    # ----------------------------------------------------------------- V2
    tickers = list(TICKER_ARTICLE.keys())
    px = PH.get_prices_long(tickers + ["SPY"], verbose=False)
    p2 = add_v2(panel, px)
    cr2 = p2[p2["crowded2"]]
    print("\n" + "=" * 84)
    print("  V2 (post-hoc iteration: volume-free crowding = SI >= 75th pctile of own"
          "\n  trailing 2y of PUBLISHED reports; fixes the DTC denominator failure)")
    print("=" * 84)
    print("\n[V2-1] AMONG CROWDED (v2): flag vs no-flag")
    rows = [tail_stats(cr2[cr2["flag2"]], "CROWDED2 + SPIKE (flag2)"),
            tail_stats(cr2[~cr2["flag2"]], "crowded2, no spike")]
    print(f"{'group':<26}{'n':>6}{'mean4w':>9}{'med4w':>8}{'P(+15%)':>9}{'P(+25%)':>9}{'max4w':>8}")
    for r in rows:
        print(f"{r['label']:<26}{r['n']:>6}{r['mean4']:>9.1%}{r['med4']:>8.1%}"
              f"{r['p15']:>9.1%}{r['p25']:>9.1%}{r['worst_for_short']:>8.0%}")
    # max adverse excursion (the margin-call number)
    ma = cr2[cr2["flag2"]]["mae4"].dropna(); mb = cr2[~cr2["flag2"]]["mae4"].dropna()
    print(f"\n  MAX ADVERSE EXCURSION (worst mark within 4 wks, short's view):")
    print(f"    flagged  : mean {ma.mean():+.1%}   P(MAE>=+25%)={float((ma>=.25).mean()):.1%}   n={len(ma)}")
    print(f"    unflagged: mean {mb.mean():+.1%}   P(MAE>=+25%)={float((mb>=.25).mean()):.1%}   n={len(mb)}")

    print("\n[V2-2] SIGNIFICANCE (week-cluster bootstrap, v2 flag)")
    cr2f = cr2.rename(columns={"flag": "_f1"}).rename(columns={"flag2": "flag"})
    for thr in (0.15, 0.25):
        real, ci, p = cluster_bootstrap_taildiff(cr2f, thr)
        print(f"  ΔP(fwd4>=+{thr:.0%}) = {real:+.1%}   boot95=[{ci[0]:+.1%},{ci[1]:+.1%}]   p={p:.3f}")

    print("\n[V2-3] EPISODES with DAILY timing (first daily flag vs the blow-off)")
    si = get_short_interest(tickers, verbose=False)
    for tk, article, start, end, note in [
            ("GME", "GameStop", "2021-01-01", "2021-03-01", "peak close Jan-27 $86.88"),
            ("AMC", "AMC_Theatres", "2021-05-01", "2021-07-01", "peak close Jun-2 $62.55"),
            ("BYND", "Beyond_Meat", "2019-06-01", "2019-08-15", "peak close Jul-26 $234.90"),
            ("CVNA", "Carvana", "2023-05-01", "2023-08-15", "+56% day Jul-19")]:
        fl = daily_flag_dates(tk, article, si.get(tk), start, end)
        if fl:
            d0, x0 = fl[0]
            print(f"  {tk:<5} first daily flag {d0.date()} (views {x0:.0f}x normal), "
                  f"{len(fl)} flag-days in window   [{note}]")
        else:
            print(f"  {tk:<5} NO daily flags in {start}..{end}   [{note}]")

    print("\n[V2-4] ALARM BUDGET (v2 weekly)")
    f2 = p2[p2["flag2"]]
    print(f"  flags/year: {f2.groupby(f2['week'].dt.year).size().to_dict()}")
    print(f"  most-flagged: {f2['ticker'].value_counts().head(6).to_dict()}")


if __name__ == "__main__":
    main()
