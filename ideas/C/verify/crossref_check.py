"""Resolve every cited DOI on Crossref; print authors/title/journal/year so they can be matched to the citation."""
import json, urllib.request, urllib.parse
DOIS = ["10.1111/jofi.12513", "10.1016/j.jfineco.2009.11.003", "10.1093/rfs/hhm055", "10.1016/j.jeconom.2006.05.023",
        "10.3905/jpm.2011.37.4.112", "10.3905/jwm.2007.674809", "10.1016/j.jfineco.2020.04.015",
        "10.1016/j.physa.2017.09.028", "10.1111/j.1540-6261.1981.tb04891.x", "10.1093/rfs/hhm014",
        "10.2469/faj.v58.n4.2453", "10.1093/jjfinec/nbp001"]
for d in DOIS:
    try:
        m = json.load(urllib.request.urlopen("https://api.crossref.org/works/" + d, timeout=60))["message"]
        au = "; ".join(f"{a.get('family','')}, {a.get('given','')}" for a in m.get("author", []))
        yr = (m.get("published-print") or m.get("published-online") or m.get("issued"))["date-parts"][0][0]
        print(f"OK  {d}\n    {au} ({yr}). {m['title'][0]}. {m.get('container-title',[''])[0]} {m.get('volume','')}({m.get('issue','')}):{m.get('page','')}")
    except Exception as e:
        print(f"FAIL {d}: {e}")
q = urllib.parse.quote("An alternative Sharpe ratio test Memmel 2003 Finance Letters")
m = json.load(urllib.request.urlopen(f"https://api.crossref.org/works?query.bibliographic={q}&rows=3", timeout=60))
print("Memmel search:", [(i.get("title"), i.get("DOI"), i.get("container-title")) for i in m["message"]["items"]])
