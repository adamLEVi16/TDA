"""
Weekly weather-anomaly signal, constructed with NO look-ahead.

For each metro: resample daily weather to weekly (Fri close), then compute
the anomaly vs a TRAILING climatology -- the mean of the same ISO week-of-year
over the prior N years only. A given week's "normal" never uses that week's
own year or any future year, so the climatology itself cannot leak information
from the future into the signal.
"""
import numpy as np, pandas as pd

CLIMATOLOGY_YEARS = 10
MIN_HISTORY_YEARS = 4   # need at least this many prior years before a week counts


def weekly_weather(df):
    w = df.resample("W-FRI").agg({"tmax": "mean", "tmin": "mean", "precip": "sum"})
    w["tavg"] = (w["tmax"] + w["tmin"]) / 2
    return w[["tavg", "precip"]].dropna()


def trailing_climatology(weekly, years=CLIMATOLOGY_YEARS, min_years=MIN_HISTORY_YEARS):
    """Normal for (week-of-year, year y) = mean of that same week-of-year over
    years [y-years, y-1] only -- strictly prior years, never the current or a
    future one."""
    df = weekly.copy()
    df["woy"] = df.index.isocalendar().week.astype(int)
    df["yr"] = df.index.year

    normals = pd.DataFrame(index=df.index, columns=["tavg_norm", "precip_norm"], dtype=float)
    for (w_, y_), grp in df.groupby(["woy", "yr"]):
        hist = df[(df["woy"] == w_) & (df["yr"] < y_) & (df["yr"] >= y_ - years)]
        if hist["yr"].nunique() >= min_years:
            normals.loc[grp.index, "tavg_norm"] = hist["tavg"].mean()
            normals.loc[grp.index, "precip_norm"] = hist["precip"].mean()

    out = df.join(normals)
    out["tanom"] = out["tavg"] - out["tavg_norm"]
    out["panom"] = out["precip"] - out["precip_norm"]
    return out[["tavg", "precip", "tanom", "panom"]].dropna()


def build_anomalies(wx_dict, **kw):
    """dict[metro] -> daily df  =>  dict[metro] -> weekly anomaly df"""
    return {m: trailing_climatology(weekly_weather(d), **kw) for m, d in wx_dict.items()}
