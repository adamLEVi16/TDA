"""
Fixed ticker -> Wikipedia-article mapping. Consumer-discretionary single names
(restaurants, apparel, specialty retail) where PUBLIC ATTENTION to the brand
plausibly leads consumer demand -- the kind of names a retail long/short PM
actually trades.

Committed BEFORE running any regression. No peeking at results then dropping
losers. If an article 404s on the pageviews API it is reported and excluded,
not silently swapped for a better-behaving one.
"""
TICKER_ARTICLE = {
    # restaurants
    "CMG":  "Chipotle_Mexican_Grill",
    "SBUX": "Starbucks",
    "MCD":  "McDonald's",
    "DPZ":  "Domino's_Pizza",
    "WEN":  "Wendy's",
    "CAKE": "The_Cheesecake_Factory",
    "TXRH": "Texas_Roadhouse",
    "SHAK": "Shake_Shack",
    "WING": "Wingstop",
    # apparel / footwear
    "LULU": "Lululemon_Athletica",
    "NKE":  "Nike,_Inc.",
    "ANF":  "Abercrombie_&_Fitch",
    "AEO":  "American_Eagle_Outfitters",
    "CROX": "Crocs",
    # specialty retail
    "HD":   "The_Home_Depot",
    "LOW":  "Lowe's",
    "TGT":  "Target_Corporation",
    "ULTA": "Ulta_Beauty",
    "BBY":  "Best_Buy",
    "DKS":  "Dick's_Sporting_Goods",
    "ELF":  "E.l.f._Beauty",
    # online / DTC consumer
    "W":    "Wayfair",
    "ETSY": "Etsy",
    "CHWY": "Chewy_(company)",
    "PTON": "Peloton_(company)",
}
