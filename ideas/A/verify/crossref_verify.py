"""VERIFIER: resolve every DOI cited in the builder report on Crossref and print authors/title/journal/year/vol/issue/pages."""
import json, urllib.request
DOIS = ["10.1111/0022-1082.00146", "10.1111/j.1540-6261.2008.01379.x", "10.1111/j.1540-6261.2010.01578.x",
        "10.1093/revfin/hhm003", "10.1016/j.jfineco.2019.10.007", "10.2139/ssrn.4540651", "10.1086/260061",
        "10.2307/1913610", "10.1080/01621459.1994.10476870", "10.1016/j.jfineco.2014.10.010",
        "10.1111/j.1540-6261.1997.tb03808.x", "10.1093/rfs/hhv059"]
for d in DOIS:
    try:
        req = urllib.request.Request("https://api.crossref.org/works/" + d, headers={"User-Agent": "verify (mailto:none@example.com)"})
        m = json.load(urllib.request.urlopen(req, timeout=60))["message"]
        au = "; ".join(f"{a.get('family','')}, {a.get('given','')}" for a in m.get("author", []))
        yr = (m.get("published-print") or m.get("published-online") or m.get("issued"))["date-parts"][0][0]
        print(f"OK  {d} | {au} | {m['title'][0]} | {(m.get('container-title') or [''])[0]} | {yr} | vol {m.get('volume')} "
              f"issue {m.get('issue')} pp {m.get('page')} | type {m.get('type')}")
    except Exception as e:
        print(f"ERR {d} {e}")
