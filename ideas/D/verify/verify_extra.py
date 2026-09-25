"""Extra data checks: SVXY/VXX suspicious daily moves, Oct/Nov 2008 realised vol claim, ^PUT DD with bad print removed."""
import os, numpy as np, pandas as pd, sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import vrp_common as vc
for t in ("SVXY", "VXX"):
    s = vc.yf_close(t); r = s.pct_change().dropna()
    print(f"{t} daily |ret|>15%:"); print(r[r.abs() > 0.15].round(4).to_string())
P = vc.build_panel()
print("\nFrench-market realised vol (annualised, sqrt(12*RV_month)) 2008-09..2008-12:")
print((np.sqrt(12 * P['RV']) * 100).loc['2008-09':'2008-12'].round(1).to_string())
put = vc.yf_close("^PUT").drop(pd.Timestamp("2020-03-13"))
dd = put / put.cummax() - 1
print(f"\n^PUT maxDD with 2020-03-13 bad print removed: {dd.min():.1%} (trough {dd.idxmin().date()}); 2020 worst DD {dd.loc['2020'].min():.1%}")
