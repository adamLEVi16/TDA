"""Resolve each cited DOI on Crossref and print authors / title / journal / year."""
import json, urllib.request
DOIS = ["10.1111/j.1540-6261.1993.tb04702.x", "10.1111/j.1540-6261.1990.tb05110.x", "10.2307/2937816",
        "10.1016/j.jempfin.2011.01.003", "10.1111/0022-1082.00146", "10.1111/j.1540-6261.2008.01379.x",
        "10.2469/faj.v58.n4.2453", "10.2307/1913610", "10.1016/j.jfineco.2014.10.010", "10.1287/mnsc.2013.1766",
        "10.1093/rfs/hhv063", "10.1093/rfs/hhv059"]
for d in DOIS:
    try:
        m = json.load(urllib.request.urlopen(urllib.request.Request("https://api.crossref.org/works/" + d,
                      headers={"User-Agent": "research-check (mailto:none@example.com)"}), timeout=30))["message"]
        au = "; ".join(f"{a.get('family','')}, {a.get('given','')}" for a in m.get("author", []))
        yr = (m.get("published-print") or m.get("published-online") or m.get("issued"))["date-parts"][0][0]
        print(f"OK  {d} | {au} | {m['title'][0]} | {m.get('container-title',[''])[0]} | {yr} | vol {m.get('volume')} pp {m.get('page')}")
    except Exception as e:
        print(f"FAIL {d}: {e}")
