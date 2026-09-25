"""Crossref bibliographic search for Pollet & Wilson (2010) and Memmel (2003)."""
import json, urllib.request, urllib.parse
for q in ["Average correlation and stock market returns Pollet Wilson", "Performance hypothesis testing with the Sharpe ratio Memmel Finance Letters 2003"]:
    u = "https://api.crossref.org/works?rows=4&query.bibliographic=" + urllib.parse.quote(q)
    for i in json.load(urllib.request.urlopen(u, timeout=60))["message"]["items"]:
        au = "; ".join(a.get("family", "") for a in i.get("author", []))
        yr = i.get("issued", {}).get("date-parts", [[None]])[0][0]
        print(f"{q[:40]:<40} | {i.get('DOI')} | {au} ({yr}) {i.get('title',[''])[0]} | {i.get('container-title',[''])[0]} {i.get('volume','')}({i.get('issue','')}):{i.get('page','')}")
