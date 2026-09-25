"""POST-HOC (counted as an extra variant): the student's own filter direction, cash when TOPOVOL > expanding 75th pct,
evaluated on the test period (French market, monthly, 2 bp) and on SPY 1994-2024. Not the frozen rule (design chose 'low|0.5')."""
import pandas as pd
from common_setup import F, ME, TEST, md
from engine import binary_target, boot_diff, fmt_table, metrics, simulate
import spy as S

rows = []
for lab, r_m, rf, dates, per in [("FF market 1990-2024", md["mkt"], md["rf"], ME, TEST),
                                 ("SPY 1994-2024", S.r_spy, S.rf, S.ME_S, S.PER)]:
    bh = simulate(pd.Series(1.0, index=dates), r_m, rf, *per)
    mb, exb = metrics(bh, 0)
    for d, q in [("high", 0.75), ("low", 0.5)]:
        sim = simulate(binary_target(F["TOPOVOL"], d, q, dates), r_m, rf, *per)
        m, x = metrics(sim, 2.0, mb["SR"])
        dd, ci, p = boot_diff(x, exb)
        rows.append(dict(rule=f"{lab} TOPOVOL {d} q={q}", **m, dSR_vs_BH=dd, dSR_lo=ci[0], dSR_hi=ci[1], p_boot=p))
    rows.append(dict(rule=f"{lab} BH", **mb))
pd.set_option("display.width", 300)
print(fmt_table(rows))
w = binary_target(F["TOPOVOL"], "high", 0.75, ME)
for a, b in [("2008-09-01", "2009-03-31"), ("2020-02-20", "2020-04-30"), ("2022-01-01", "2022-10-31")]:
    print(f"student rule (high,0.75) share of month-ends in cash {a}..{b}: {1 - w.loc[a:b].mean():.2f}")
