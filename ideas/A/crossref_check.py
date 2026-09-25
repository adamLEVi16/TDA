"""Resolve each candidate citation's DOI on Crossref and print authors/title/journal/year."""
import json, urllib.request, urllib.parse
DOIS = ["10.1111/0022-1082.00146", "10.1111/j.1540-6261.2008.01379.x", "10.1111/j.1540-6261.2010.01567.x",
        "10.1016/j.jfineco.2019.10.007", "10.2307/1913610", "10.2469/faj.v58.n4.2453",
        "10.1080/01621459.1994.10476870", "10.1016/j.jfineco.2014.10.010", "10.1111/j.1540-6261.1997.tb03808.x",
        "10.1093/rfs/hhm008", "10.1086/260061", "10.1016/j.jempfin.2008.03.002", "10.1093/rfs/hhv059"]
def get(url):
    req = urllib.request.Request(url, headers={"User-Agent": "research-check (mailto:none@example.com)"})
    return json.load(urllib.request.urlopen(req, timeout=60))["message"]
for d in DOIS:
    try:
        m = get("https://api.crossref.org/works/" + d)
        au = "; ".join(f"{a.get('family','')}, {a.get('given','')}" for a in m.get("author", []))
        yr = (m.get("published-print") or m.get("published-online") or m.get("issued"))["date-parts"][0][0]
        print(f"OK  {d} | {au} | {m['title'][0]} | {(m.get('container-title') or [''])[0]} | {yr} | "
              f"vol {m.get('volume')} issue {m.get('issue')} pp {m.get('page')}")
    except Exception as e:
        print(f"ERR {d} {e}")
q = "Network momentum across asset classes Pu Roberts Dong Zohren"
for it in get("https://api.crossref.org/works?rows=3&query.bibliographic=" + urllib.parse.quote(q))["items"]:
    au = "; ".join(a.get("family", "") for a in it.get("author", []))
    print("SEARCH |", it.get("DOI"), "|", au, "|", (it.get("title") or [""])[0], "|",
          (it.get("container-title") or [""])[0], "|", it["issued"]["date-parts"][0][0])
