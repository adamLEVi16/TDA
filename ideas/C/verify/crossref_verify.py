"""VERIFIER: resolve every DOI cited in the builder's report on Crossref and print all date fields."""
import json
import urllib.parse
import urllib.request

CITED = {
    "10.1111/jofi.12513": "Moreira & Muir 2017 Volatility-Managed Portfolios, J Finance 72(4):1611-1644",
    "10.1016/j.jfineco.2020.04.015": "Cederburg, O'Doherty, Wang, Yan 2020, JFE 138(1):95-117",
    "10.1016/j.jfineco.2010.02.011": "Pollet & Wilson 2010 Average correlation and stock market returns, JFE 96(3):364-380",
    "10.1093/rfs/hhm055": "Campbell & Thompson 2008, RFS 21(4):1509-1531",
    "10.1016/j.jeconom.2006.05.023": "Clark & West 2007, J Econometrics 138(1):291-311",
    "10.3905/jpm.2011.37.4.112": "Kritzman, Li, Page, Rigobon 2011, JPM 37(4):112-126",
    "10.3905/jwm.2007.674809": "Faber 2007, J Wealth Management 9(4):69-79",
    "10.1016/j.physa.2017.09.028": "Gidea & Katz 2018, Physica A 491:820-834",
    "10.1111/j.1540-6261.1981.tb04891.x": "Jobson & Korkie 1981, J Finance 36(4):889-908",
    "10.1093/jjfinec/nbp001": "Corsi 2009, J Financial Econometrics 7(2):174-196",
}


def yr(m, k):
    v = m.get(k)
    return v["date-parts"][0] if v else None


for doi, claim in CITED.items():
    m = json.load(urllib.request.urlopen("https://api.crossref.org/works/" + urllib.parse.quote(doi), timeout=60))["message"]
    au = "; ".join(f"{a.get('family', '')}, {a.get('given', '')}" for a in m.get("author", []))
    print(f"{doi}\n  claimed : {claim}\n  crossref: {au} | {m['title'][0]} | {m.get('container-title', [''])[0]} "
          f"{m.get('volume', '')}({m.get('issue', '')}):{m.get('page', '')} | print {yr(m, 'published-print')} "
          f"online {yr(m, 'published-online')} issued {yr(m, 'issued')}")
