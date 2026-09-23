"""Free-data loader for the USD/JPY factor study.

Sources (no API keys):
  FRED    DEXJPUS (NY noon USD/JPY), DGS2, DGS10, DTB3, VIXCLS, DCOILWTICO, NIKKEI225
  MoF     JGB constant-maturity yields (Tokyo close)
  CFTC    Legacy COT, JPY futures (code 097741), speculative net position

Timing: FX is the New York noon fix. Every other input is taken as of the
last close strictly before the FX date, so no signal uses information
published after the FX observation it predicts from.
"""
import glob
import io
import os
import zipfile

import numpy as np
import pandas as pd
import requests

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
START = "1990-01-01"  # VIX starts 1990; earlier data only feeds look-back windows
FRED = ["DEXJPUS", "DGS2", "DGS10", "DTB3", "VIXCLS", "DCOILWTICO", "NIKKEI225"]
MOF = {
    "jgb_hist.csv": "https://www.mof.go.jp/english/policy/jgbs/reference/interest_rate/historical/jgbcme_all.csv",
    "jgb_cur.csv": "https://www.mof.go.jp/english/policy/jgbs/reference/interest_rate/jgbcme.csv",
}
COT_URL = "https://www.cftc.gov/files/dea/history/deacot{}.zip"


def download(cot=True):
    os.makedirs(DATA, exist_ok=True)
    for s in FRED:
        r = requests.get(f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={s}", timeout=60)
        r.raise_for_status()
        open(os.path.join(DATA, s + ".csv"), "wb").write(r.content)
    for name, url in MOF.items():
        r = requests.get(url, timeout=60)
        r.raise_for_status()
        open(os.path.join(DATA, name), "wb").write(r.content)
    if cot:
        years = ["1986_2016"] + [str(y) for y in range(2017, pd.Timestamp.today().year + 1)]
        for y in years:
            r = requests.get(COT_URL.format(y), timeout=120)
            r.raise_for_status()
            zipfile.ZipFile(io.BytesIO(r.content)).extractall(os.path.join(DATA, "cot", y))


def fred(s):
    df = pd.read_csv(os.path.join(DATA, s + ".csv"), index_col=0, parse_dates=True, na_values=".")
    return pd.to_numeric(df.iloc[:, 0], errors="coerce").rename(s).dropna()


def jgb():
    out = []
    for f in MOF:
        df = pd.read_csv(os.path.join(DATA, f), skiprows=1, na_values="-", encoding="latin1")
        dt = pd.to_datetime(df["Date"], format="%Y/%m/%d", errors="coerce")
        df = df[dt.notna()].set_index(dt[dt.notna()])
        out.append(df[["1Y", "2Y", "10Y"]].apply(pd.to_numeric, errors="coerce"))
    j = pd.concat(out)
    j = j[~j.index.duplicated(keep="last")]
    return j.rename(columns={"1Y": "jgb1", "2Y": "jgb2", "10Y": "jgb10"})


def cot_jpy():
    """Non-commercial net JPY long / open interest, indexed by the date it becomes usable
    (as-of Tuesday + 10 days: released Friday 15:30 ET, first used at the next Friday noon)."""
    frames = []
    for f in glob.glob(os.path.join(DATA, "cot", "*", "*.txt")):
        t = pd.read_csv(f, low_memory=False)
        t.columns = [c.strip() for c in t.columns]
        frames.append(t[t["CFTC Contract Market Code"].astype(str).str.strip() == "097741"])
    c = pd.concat(frames)
    c["date"] = pd.to_datetime(c["As of Date in Form YYYY-MM-DD"])
    c = c.drop_duplicates("date").set_index("date").sort_index()
    net = (c["Noncommercial Positions-Long (All)"] - c["Noncommercial Positions-Short (All)"]) \
        / c["Open Interest (All)"]
    return pd.Series(net.values, index=net.index + pd.Timedelta(days=10), name="cot_net")


def asof_before(s, dates, back=0, max_age_days=14):
    """Value of `s` at its last observation strictly before each date, stepped `back` more obs.
    NaN if that observation is older than `max_age_days` (data gaps)."""
    s = s.dropna()
    p = s.index.searchsorted(dates, side="left") - 1 - back
    ok = p >= 0
    pc = np.clip(p, 0, None)
    stale = (dates - s.index[pc]).days > max_age_days + 7 * back / 5
    return pd.Series(np.where(ok & ~stale, s.values[pc], np.nan), index=dates)


def panel(fx_dates):
    """Rates, VIX, oil and Nikkei as of the last close strictly before each FX date."""
    j = jgb()
    src = {"us2": fred("DGS2"), "us10": fred("DGS10"), "us3m": fred("DTB3"),
           "vix": fred("VIXCLS"), "oil": fred("DCOILWTICO"), "nky": fred("NIKKEI225"),
           "jgb1": j["jgb1"], "jgb2": j["jgb2"], "jgb10": j["jgb10"]}
    now = pd.DataFrame({k: asof_before(v, fx_dates) for k, v in src.items()})
    return now


def weekly_features():
    """One row per week, anchored on the last NY-noon FX print of the week (normally Friday).
    Returns X (signals known at t), y (log return t -> t+1), carry (short-rate carry for t -> t+1)."""
    fx = fred("DEXJPUS")["1985":]
    wk = fx.groupby(fx.index.to_period("W-FRI")).tail(1)
    p = panel(wk.index)
    lfx = np.log(wk)
    d2, d10 = p["us2"] - p["jgb2"], p["us10"] - p["jgb10"]
    X = pd.DataFrame(index=wk.index)
    X["carry_short"] = p["us3m"] - p["jgb1"]
    X["carry_2y"] = d2
    X["d_spread2y_1w"] = d2.diff()
    X["d_spread2y_4w"] = d2.diff(4)
    X["d_spread10y_1w"] = d10.diff()
    X["d_spread10y_4w"] = d10.diff(4)
    X["d_us10_1w"] = p["us10"].diff()
    X["d_jgb10_4w"] = p["jgb10"].diff(4)
    X["rev_1w"] = lfx.diff()
    X["mom_4w"] = lfx.diff(4)
    X["mom_12w"] = lfx.diff(12)
    X["mom_52w"] = lfx.diff(52)
    X["vix_log"] = np.log(p["vix"])
    X["d_vix_1w"] = np.log(p["vix"]).diff()
    X["oil_4w"] = np.log(p["oil"]).diff(4)
    X["nky_1w"] = np.log(p["nky"]).diff()
    X["nky_4w"] = np.log(p["nky"]).diff(4)
    X["misalign_z"] = _misalignment(lfx, d2, d10, 52)
    if glob.glob(os.path.join(DATA, "cot", "*", "*.txt")):
        cot = cot_jpy().reindex(wk.index, method="ffill")
        X["cot_net"] = cot
        X["cot_z3y"] = (cot - cot.rolling(156).mean()) / cot.rolling(156).std()
    y = lfx.diff().shift(-1).rename("ret_next")
    carry = ((p["us3m"] - p["jgb1"]) / 100 / 52).rename("carry_next")
    return X[START:], y[START:], carry[START:]


def _misalignment(lfx, d2, d10, win):
    """Z-scored residual of log USD/JPY on the 2y and 10y spreads over a trailing window.
    Positive = USD/JPY rich versus rates. Uses only data up to t."""
    Z = pd.concat([lfx, d2, d10], axis=1).dropna()
    v = Z.values
    res = pd.Series(np.nan, index=Z.index)
    for i in range(win, len(Z)):
        yy = v[i - win:i + 1, 0]
        A = np.column_stack([np.ones(win + 1), v[i - win:i + 1, 1:]])
        b, *_ = np.linalg.lstsq(A, yy, rcond=None)
        r = yy - A @ b
        res.iloc[i] = r[-1] / r.std()
    return res.reindex(lfx.index)
