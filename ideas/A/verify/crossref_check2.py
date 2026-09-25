"""Title searches on Crossref for citations whose guessed DOI failed."""
import json, urllib.request, urllib.parse
def get(url):
    req = urllib.request.Request(url, headers={"User-Agent": "research-check (mailto:none@example.com)"})
    return json.load(urllib.request.urlopen(req, timeout=60))["message"]
for q in ["Market Segmentation and Cross-predictability of Returns Menzly Ozbas",
          "Industry Information Diffusion and the Lead-lag Effect in Stock Returns Hou",
          "Network Momentum across Asset Classes Pu Roberts Dong Zohren"]:
    for it in get("https://api.crossref.org/works?rows=3&query.bibliographic=" + urllib.parse.quote(q))["items"]:
        au = "; ".join(a.get("family", "") for a in it.get("author", []))
        print("SEARCH |", q[:30], "|", it.get("DOI"), "|", au, "|", (it.get("title") or [""])[0], "|",
              (it.get("container-title") or [""])[0], "|", it["issued"]["date-parts"][0][0], "| vol", it.get("volume"), "pp", it.get("page"))
