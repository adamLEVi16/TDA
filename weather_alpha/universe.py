"""
Fixed ticker -> HQ-metro mapping for the weather signal test.

Committed BEFORE running any regression (no peeking at results then picking
favorable names). Selection criteria, decided up front:
  - Consumer/retail/restaurant/apparel names where foot traffic or seasonal
    demand is plausibly weather-sensitive (a real, pre-existing hypothesis in
    the alt-data literature, not invented for this test).
  - Headquarters location used as a proxy for store-footprint/regional
    concentration. This is an ACKNOWLEDGED LIMITATION: HQ city is not the
    same as where most stores/customers actually are for a national chain
    (e.g. Home Depot is "Atlanta" but has stores nationwide). The test is
    honest about this approximation, not hidden.
  - No look-ahead in ticker selection: this list was written before any
    weather/return regression was run or inspected.
"""
TICKER_METRO = {
    "HD":    "ATLANTA",     # Home Depot - storm/cold-weather builder demand
    "LOW":   "CHARLOTTE",   # Lowe's
    "CBRL":  "NASHVILLE",   # Cracker Barrel - south/midwest highway diner traffic
    "TXRH":  "LOUISVILLE",  # Texas Roadhouse
    "EAT":   "DALLAS",      # Brinker (Chili's)
    "DRI":   "ORLANDO",     # Darden (Olive Garden, LongHorn, etc.)
    "CAKE":  "LOSANGELES",  # Cheesecake Factory
    "WEN":   "COLUMBUS",    # Wendy's
    "UAA":   "BALTIMORE",   # Under Armour - cold-weather athletic gear
    "SBUX":  "SEATTLE",     # Starbucks
    "JWN":   "SEATTLE",     # Nordstrom - department store foot traffic
    "LULU":  "SEATTLE",     # Lululemon (Vancouver HQ; Seattle = nearest covered metro)
    "COLM":  "PORTLAND",    # Columbia Sportswear - winter/outdoor gear
    "KSS":   "MILWAUKEE",   # Kohl's
}
