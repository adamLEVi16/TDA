"""Checks on main's risk_report.py, the source of the thesis's "verified" Sharpe -0.77 and alpha t-stats.

1. What the Fiedler value tracks, and whether "low Fiedler = stressed" (risk_report.py:291) is right.
2. Strategy D as coded vs. the regime label flipped vs. no regime at all.
3. The FF5+UMD regression as coded (subtracts RF from a self-financing long/short) vs. without RF,
   and before transaction costs.

Run from the repo root:  python audit/check_risk_report.py
"""
import io
import sys
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import risk_report as rr  # noqa: E402
from common import sharpe  # noqa: E402

FRENCH = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"


def french(name):
    raw = urllib.request.urlopen(FRENCH + name).read()
    with zipfile.ZipFile(io.BytesIO(raw)) as z:
        lines = z.read(z.namelist()[0]).decode("latin1").splitlines()
    header = next(i for i, line in enumerate(lines) if line.startswith(","))
    cols = [c.strip() for c in lines[header].split(",")[1:]]
    rows = [line.split(",") for line in lines[header + 1:] if line[:8].strip().isdigit()]
    return pd.DataFrame([r[1:] for r in rows], columns=cols,
                        index=pd.to_datetime([r[0].strip() for r in rows])).astype(float) / 100


prices, spy_prices = rr.download_prices()
R = rr.compute_returns(prices)
spy_vol = rr.compute_returns(spy_prices)["SPY"].rolling(20).std() * np.sqrt(252)

print("\n== 1. Fiedler value vs. market state (Spearman)")
signals = {}
for sector, tickers in rr.SECTORS.items():
    sig = rr.run_sector_signals(R, tickers)
    signals[sector] = sig
    avg_corr = [R[tickers].iloc[i - rr.LOOKBACK:i].corr().values[np.triu_indices(len(tickers), 1)].mean()
                for i in range(rr.LOOKBACK, len(R[tickers].dropna()))]
    fv = sig["fiedler"]
    low = fv < np.nanpercentile(fv.loc[:rr.TRAIN_END], 25)      # the code's "stressed" label
    sv = spy_vol.reindex(sig.index)
    print(f"{sector:<11} rho(Fiedler, avg corr)={pd.Series(avg_corr, index=sig.index).corr(fv, method='spearman'):+.2f}  "
          f"rho(Fiedler, SPY vol)={sv.corr(fv, method='spearman'):+.2f}  "
          f"SPY vol on 'stressed' (low-Fiedler) days {sv[low].mean():.1%} vs other days {sv[~low].mean():.1%}")

print("\n== 2. Strategy D, 2022-2024")
variants = {"as coded": lambda s: s,
            "regime label flipped": lambda s: s.assign(fiedler=-s["fiedler"]),
            "no regime (always short the high-beta spread)": lambda s: s.assign(fiedler=1e9)}
for name, transform in variants.items():
    legs = [rr.beta_spread_strategy(R, tk, transform(signals[sec]), rr.TRAIN_END) for sec, tk in rr.SECTORS.items()]
    d = pd.concat(legs, axis=1).dropna().mean(axis=1).loc[rr.TEST_START:]
    print(f"{name:<46} Sharpe {sharpe(d):+.2f}")
print("(post-hoc variants: shown to test the label, not as a strategy; none is significant over 3 years)")


def combined(tc_bps):
    rr.TC_BPS = tc_bps
    legs = {k: [] for k in "ABCD"}
    for sector, tickers in rr.SECTORS.items():
        sig = signals[sector]
        abc = rr.sector_returns(sig, tickers, R, rr.fiedler_filter(sig, rr.TRAIN_END), rr.vol_filter(sig, rr.TRAIN_END))
        for k in "ABC":
            legs[k].append(abc[k])
        legs["D"].append(rr.beta_spread_strategy(R, tickers, sig, rr.TRAIN_END))
    return {k: pd.concat(v, axis=1).dropna().mean(axis=1).loc[rr.TEST_START:] for k, v in legs.items()}


factors = french("F-F_Research_Data_5_Factors_2x3_daily_CSV.zip").join(
    french("F-F_Momentum_Factor_daily_CSV.zip").set_axis(["UMD"], axis=1), how="inner")
X = sm.add_constant(factors.drop(columns="RF"))


def alpha(y, subtract_rf):
    y = y - factors["RF"].reindex(y.index) if subtract_rf else y
    d = pd.concat([y.rename("y"), X], axis=1, join="inner").dropna()
    m = sm.OLS(d["y"], d.drop(columns="y")).fit(cov_type="HAC", cov_kwds={"maxlags": 5})
    return f"{m.params['const'] * 252:+.1%} (t={m.tvalues['const']:+.2f})"


net, gross = combined(5), combined(0)
print(f"\n== 3. FF5+UMD alpha, 2022-2024 (mean RF {factors['RF'].loc['2022':'2024'].mean() * 252:.2%}/yr)")
print(f"{'':<3}{'gross SR':>9}{'net SR':>8}   {'net alpha, as coded (y - RF)':<30}{'net alpha, no RF':<24}{'gross alpha, no RF'}")
for k in "ABCD":
    print(f"{k:<3}{sharpe(gross[k]):>+9.2f}{sharpe(net[k]):>+8.2f}   {alpha(net[k], True):<30}"
          f"{alpha(net[k], False):<24}{alpha(gross[k], False)}")
