"""Resolve candidate citations on Crossref; print authors/title/journal/year so they can be matched."""
import json, urllib.request, urllib.parse, os
DOIS = ["10.1093/rfs/hhp008", "10.1093/rfs/hhn038", "10.1111/j.1540-6261.2009.01467.x", "10.1093/rfs/hhm055",
        "10.1016/j.jeconom.2006.05.023", "10.1093/rfs/hhm014", "10.2469/faj.v58.n4.2453", "10.1093/rfs/hhg002",
        "10.1111/jofi.12513", "10.1080/01621459.1994.10476870", "10.1093/rfs/hhy029", "10.1016/j.jeconom.2014.05.008",
        "10.1016/j.jfineco.2016.09.003", "10.2307/1913610"]
QUERIES = ["Volmageddon and the failure of short volatility products Augustin",
           "correlation risk premium dispersion trading index options",
           "Dispersion trading Deng", "Price of correlation risk dispersion Driessen Maenhout Vilkov review"]
def get(url):
    req = urllib.request.Request(url, headers={"User-Agent": "research-check (mailto:none@example.com)"})
    return json.loads(urllib.request.urlopen(req, timeout=30).read())
def fmt(m):
    au = "; ".join(f"{a.get('family','')}, {a.get('given','')}" for a in m.get("author", [])[:5])
    yr = (m.get("published-print") or m.get("published-online") or m.get("issued"))["date-parts"][0][0]
    return f"{au} ({yr}). {m.get('title',[''])[0]}. {m.get('container-title',[''])[0]} {m.get('volume','')}({m.get('issue','')}) {m.get('page','')}. DOI {m.get('DOI')}"
for d in DOIS:
    try: print("OK  ", fmt(get("https://api.crossref.org/works/" + d)["message"]))
    except Exception as e: print("FAIL", d, e)
for q in QUERIES:
    print("\nQUERY:", q)
    try:
        for it in get("https://api.crossref.org/works?rows=4&query.bibliographic=" + urllib.parse.quote(q))["message"]["items"]:
            print("   ", fmt(it))
    except Exception as e: print("FAIL", e)
