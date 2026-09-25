"""Resolve each DOI cited in the builder report on Crossref and print authors/title/journal/vol(issue)/pages/year."""
import json, urllib.request
CITED = {
 "10.1093/rfs/hhp008": "Bollerslev Tauchen Zhou 2009 RFS 22(11) 4463-4492",
 "10.1093/rfs/hhn038": "Carr Wu 2009 RFS 22(3) 1311-1341",
 "10.1111/j.1540-6261.2009.01467.x": "Driessen Maenhout Vilkov 2009 JF 64(3) 1377-1406",
 "10.1093/rfs/hhg002": "Bakshi Kapadia 2003 RFS 16(2) 527-566",
 "10.1016/j.jfineco.2016.04.003": "Dew-Becker Giglio Le Rodriguez 2017 JFE 123(2) 225-250",
 "10.1016/j.jeconom.2014.05.008": "Bekaert Hoerova 2014 JEconometrics 183(2) 181-192",
 "10.1093/rfs/hhy062": "Cheng 2019 RFS 32(1) 180-227",
 "10.1080/0015198x.2021.1913040": "Augustin Cheng Van den Bergen 2021 FAJ 77(3) 35-51",
 "10.1093/rfs/hhm055": "Campbell Thompson 2008 RFS 21(4) 1509-1531",
 "10.1016/j.jeconom.2006.05.023": "Clark West 2007 JEconometrics 138(1) 291-311",
 "10.1093/rfs/hhm014": "Welch Goyal 2008 RFS 21(4) 1455-1508",
 "10.2469/faj.v58.n4.2453": "Lo 2002 FAJ 58(4) 36-52",
 "10.1080/01621459.1994.10476870": "Politis Romano 1994 JASA 89(428) 1303-1313",
 "10.2307/1913610": "Newey West 1987 Econometrica 55(3) 703",
 "10.1111/jofi.12513": "Moreira Muir 2017 JF 72(4) 1611-1644",
}
for doi, claim in CITED.items():
    req = urllib.request.Request("https://api.crossref.org/works/" + doi, headers={"User-Agent": "verify (mailto:none@example.com)"})
    try:
        m = json.loads(urllib.request.urlopen(req, timeout=30).read())["message"]
        au = "; ".join(a.get("family", "") for a in m.get("author", []))
        yr = (m.get("published-print") or m.get("issued"))["date-parts"][0][0]
        print(f"OK   {doi}\n     claim : {claim}\n     crossref: {au} ({yr}) {m['title'][0]} | {m.get('container-title',[''])[0]} {m.get('volume','')}({m.get('issue','')}) {m.get('page','')}")
    except Exception as e:
        print("FAIL", doi, e)
