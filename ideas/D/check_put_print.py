"""Data-quality check of ^PUT around March 2020 (flagged by tailrisk.log)."""
from vrp_common import yf_close
put = yf_close("^PUT"); spy = yf_close("SPY")
import pandas as pd
d = pd.DataFrame({"PUT": put, "PUT_ret": put.pct_change(), "SPY_ret": spy.pct_change()}).loc["2020-03-05":"2020-03-27"]
print(d.round(4).to_string())
