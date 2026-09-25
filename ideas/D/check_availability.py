"""Availability check only: date ranges of series. No returns/performance evaluated."""
import yfinance as yf, pandas as pd, io, zipfile, urllib.request, os
for t in ["^VIX","^VIX3M","SPY","SVXY","XIV","VXX","ZIV","^PUT","^BXM","^SPVXSPID","^VXO"]:
    try:
        d = yf.download(t, start="1985-01-01", auto_adjust=True, progress=False)
        print(f"{t:10s} n={len(d):6d} first={d.index.min()} last={d.index.max()}")
    except Exception as e:
        print(t, "ERR", e)
FRENCH = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"
for name in ["F-F_Research_Data_Factors_daily_CSV.zip","F-F_Research_Data_Factors_CSV.zip"]:
    raw = urllib.request.urlopen(FRENCH+name).read()
    with zipfile.ZipFile(io.BytesIO(raw)) as z:
        lines = z.read(z.namelist()[0]).decode("latin1").splitlines()
    rows=[l for l in lines if l[:8].strip().isdigit()]
    print(name, "first", rows[0][:30], "last(monthly/daily part)", [r[:30] for r in rows if len(r.split(',')[0].strip()) in (6,8)][-1])
