"""
Download SEC 'Insider Transactions Data Sets' (quarterly zips, free) and build
one flat file of OPEN-MARKET PURCHASES (Form 4, code P, acquired) for the
pre-committed universe: symbol, trans_date, filing_date, owner_cik, shares,
price, value. FILING_DATE is the point-in-time availability date (insiders must
file within 2 business days; we only ever act on/after the filing date).
"""
import io, os, zipfile, urllib.request, time
import pandas as pd
from universe import UNIVERSE

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "cache"); os.makedirs(CACHE, exist_ok=True)
OUT = os.path.join(HERE, "purchases.csv")
UA = {"User-Agent": "tda-research adamwasnothere292@gmail.com"}
URL = ("https://www.sec.gov/files/structureddata/data/"
       "insider-transactions-data-sets/{q}_form345.zip")
QUARTERS = [f"{y}q{k}" for y in range(2014, 2025) for k in range(1, 5)]
UNIV = set(UNIVERSE)


def one_quarter(q):
    fp = os.path.join(CACHE, f"{q}_form345.zip")
    if not os.path.exists(fp):
        for attempt in range(3):
            try:
                req = urllib.request.Request(URL.format(q=q), headers=UA)
                data = urllib.request.urlopen(req, timeout=180).read()
                open(fp, "wb").write(data)
                break
            except Exception as e:
                if attempt == 2:
                    print(f"  [{q}] download FAILED: {type(e).__name__}")
                    return None
                time.sleep(3 * (attempt + 1))
    z = zipfile.ZipFile(fp)
    sub = pd.read_csv(io.BytesIO(z.read("SUBMISSION.tsv")), sep="\t",
                      usecols=["ACCESSION_NUMBER", "FILING_DATE", "DOCUMENT_TYPE",
                               "ISSUERTRADINGSYMBOL"], dtype=str)
    sub = sub[(sub["DOCUMENT_TYPE"] == "4") &
              sub["ISSUERTRADINGSYMBOL"].isin(UNIV)]
    if not len(sub):
        return None
    tr = pd.read_csv(io.BytesIO(z.read("NONDERIV_TRANS.tsv")), sep="\t",
                     usecols=["ACCESSION_NUMBER", "TRANS_DATE", "TRANS_CODE",
                              "TRANS_ACQUIRED_DISP_CD", "TRANS_SHARES",
                              "TRANS_PRICEPERSHARE"], dtype=str)
    tr = tr[(tr["TRANS_CODE"] == "P") & (tr["TRANS_ACQUIRED_DISP_CD"] == "A")]
    ro = pd.read_csv(io.BytesIO(z.read("REPORTINGOWNER.tsv")), sep="\t",
                     usecols=["ACCESSION_NUMBER", "RPTOWNERCIK"], dtype=str)
    d = tr.merge(sub, on="ACCESSION_NUMBER").merge(
        ro.drop_duplicates("ACCESSION_NUMBER"), on="ACCESSION_NUMBER")
    if not len(d):
        return None
    d["shares"] = pd.to_numeric(d["TRANS_SHARES"], errors="coerce")
    d["price"] = pd.to_numeric(d["TRANS_PRICEPERSHARE"], errors="coerce")
    d["value"] = d["shares"] * d["price"]
    d["trans_date"] = pd.to_datetime(d["TRANS_DATE"], format="%d-%b-%Y", errors="coerce")
    d["filing_date"] = pd.to_datetime(d["FILING_DATE"], format="%d-%b-%Y", errors="coerce")
    d = d.rename(columns={"ISSUERTRADINGSYMBOL": "ticker", "RPTOWNERCIK": "owner"})
    return d[["ticker", "trans_date", "filing_date", "owner", "shares",
              "price", "value"]].dropna(subset=["trans_date", "filing_date"])


def main():
    frames = []
    for q in QUARTERS:
        d = one_quarter(q)
        n = len(d) if d is not None else 0
        print(f"  [{q}] purchases in universe: {n}")
        if d is not None:
            frames.append(d)
        time.sleep(0.4)
    allp = pd.concat(frames).sort_values("filing_date")
    allp.to_csv(OUT, index=False)
    print(f"\nTotal open-market purchases: {len(allp)} across "
          f"{allp['ticker'].nunique()} tickers, "
          f"{allp['filing_date'].min().date()} -> {allp['filing_date'].max().date()}")


if __name__ == "__main__":
    main()
