"""Second Crossref pass: title searches for references whose recalled DOI was wrong (see lit_check.log)."""
import urllib.parse
import json, urllib.request
exec(open("lit_check.py").read().split("for d in DOIS:")[0])  # reuse get/fmt definitions only
for q in ["The VIX Premium Cheng Review of Financial Studies", "The price of variance risk Dew-Becker Giglio Le Rodriguez",
          "Expected correlation and future market returns Buss Schonleber Vilkov"]:
    print("\nQUERY:", q)
    for it in get("https://api.crossref.org/works?rows=3&query.bibliographic=" + urllib.parse.quote(q))["message"]["items"]:
        print("   ", fmt(it))
