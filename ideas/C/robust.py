"""Robustness checks listed in SPEC.md (10 bp; weekly; cap 1.5 + borrowing; subperiods), plus two POST-HOC additions
that are counted as extra variants in the report:
  (a) HAR-style vol baseline (log RV5, log RV21, log RV63) for the 'beyond realised vol' vol-forecast test;
  (b) EXPLORATORY mean-variance timing (gamma=3, w in [0,1]) using the return forecasts from horse_race.py,
      added AFTER seeing the test-period OOS R2 -> not a confirmatory result."""
import numpy as np
import pandas as pd
import statsmodels.api as sm

from common_setup import FIRST_X, M, ME, SIGMA_STAR, TEST, md, mm
from engine import NET_INDS, boot_diff, fmt_table, jkm, metrics, simulate, vt_forecasts
from horse_race import clark_west, expanding_fc
from test import PAIRS, incremental, run_period

pd.set_option("display.width", 300)
print("\n#################### R1: 10 bp costs")
run_period(TEST, cost=10.0, label="R1 10bp")
print("\n#################### R2: weekly rebalance (2 bp)")
_, exw, _ = run_period(TEST, freq="W", label="R2 weekly")
incremental(exw, PAIRS[:6], "(weekly)")
print("\n#################### R3: cap 1.5, borrowing at RF + 0.50%/yr (2 bp)")
_, exl, _ = run_period(TEST, cap=1.5, borrow=0.005, label="R3 cap1.5")
incremental(exl, PAIRS[:6], "(cap 1.5)")
print("\n#################### R4: subperiods (2 bp, monthly)")
for per in [("1990-01-01", "2007-12-31"), ("2008-01-01", "2024-12-31")]:
    _, exs, _ = run_period(per, label="R4 subperiod")

# ------------------------------------------------------------------------------ (a) HAR-style baseline, post-hoc
print("\n#################### POST-HOC (a): vol forecast with HAR-style base [1, log RV5, log RV21, log RV63]")
r = md["mkt"]
for k in (5, 63):
    s = np.sqrt(252 * (r ** 2).rolling(k).mean())
    M[f"logRV{k}"] = np.log(s.reindex(pd.DatetimeIndex(M["ME_DATE"].values)).values)
D = M.loc[FIRST_X:].copy()
y = D["logRVM"].shift(-1).values
out_month = D.index + 1
TST = (out_month >= pd.Period("1990-01", "M")) & (out_month <= pd.Period("2024-12", "M"))
har = ["logRV5", "logRV", "logRV63"]
Xh = sm.add_constant(D[har]).values
fh, _ = expanding_fc(y, Xh)
f1, _ = expanding_fc(y, sm.add_constant(D[["logRV"]]).values)
yt = y[TST]
print(f"HAR base vs log-RV21 base, test OOS R2: {1 - np.nansum((yt - fh[TST])**2) / np.nansum((yt - f1[TST])**2):.4f}")
for x in NET_INDS + ["AC60+topology"]:
    cols = har + (["AC60", "H1N", "H1P", "TOPOVOL", "FIED"] if x == "AC60+topology" else [x])
    fa, _ = expanding_fc(y, sm.add_constant(D[cols]).values)
    r2 = 1 - np.nansum((yt - fa[TST]) ** 2) / np.nansum((yt - fh[TST]) ** 2)
    t, p = clark_west(yt, fh[TST], fa[TST])
    print(f"HAR + {x:<14} OOS R2 vs HAR {r2:+.4f}  CW t {t:+.2f} p {p:.3f}")

# ------------------------------------------------------------------------------ (b) exploratory MV timing, post-hoc
print("\n#################### POST-HOC (b) EXPLORATORY: mean-variance timing w = clip(mu_hat / (3 * sigma_hat^2), 0, 1)")
mm_p = mm.copy()
mm_p.index = mm_p.index.to_period("M")
D["ret_next"] = mm_p["mkt_rf"].shift(-1).reindex(D.index)
yr = D["ret_next"].values
hm = pd.Series(yr).expanding(24).mean().shift(1).values
fv, s2 = vt_forecasts(M, ["logRV"], FIRST_X)
var_m = (np.exp(fv + s2 / 2) ** 2 / 12).values            # monthly variance forecast
dates = pd.DatetimeIndex(D["ME_DATE"].values)
tg = {"MV-histmean": hm}
for x in ["AC60", "FIED", "H1N"]:
    f, _ = expanding_fc(yr, sm.add_constant(D[[x]]).values)
    tg[f"MV-{x}"] = np.maximum(f, 0)
bh = simulate(pd.Series(1.0, index=ME), md["mkt"], md["rf"], *TEST)
mb, exb = metrics(bh, 0)
rows, exs = [], {}
for k, mu in tg.items():
    w = pd.Series(np.clip(mu / (3 * var_m), 0, 1), index=dates)
    sim = simulate(w, md["mkt"], md["rf"], *TEST)
    m, x = metrics(sim, 2.0, mb["SR"])
    d, ci, p = boot_diff(x, exb)
    rows.append(dict(rule=k, **m, dSR_vs_BH=d, dSR_lo=ci[0], dSR_hi=ci[1], p_boot=p, p_JKM=jkm(x, exb)[1]))
    exs[k] = x
rows.append(dict(rule="BH", **mb))
print(fmt_table(rows))
for k in ["MV-AC60", "MV-FIED", "MV-H1N"]:
    d, ci, p = boot_diff(exs[k], exs["MV-histmean"])
    print(f"{k} vs MV-histmean: dSR {d:+.3f} [{ci[0]:+.2f}, {ci[1]:+.2f}] p_boot {p:.3f}")
