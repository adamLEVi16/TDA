"""
Sector Configuration for Phase 2 Analysis
==========================================

Defines stock universes for 7 sectors (20 stocks each).

Criteria for selection:
- Large cap (> $10B market cap)
- Liquid (average volume > 1M shares/day)
- Sector-pure (clear classification)
- Available data 2020-2024

Author: Adam Levine
Date: January 2026
"""

SECTORS = {
    'Technology': [
        'AAPL', 'MSFT', 'NVDA', 'GOOGL', 'META',
        'AMD', 'INTC', 'CSCO', 'ORCL', 'CRM',
        'ADBE', 'AVGO', 'TXN', 'QCOM', 'MU',
        'AMAT', 'LRCX', 'KLAC', 'SNPS', 'CDNS'
    ],

    'Healthcare': [
        'JNJ', 'UNH', 'PFE', 'ABBV', 'TMO',
        'ABT', 'MRK', 'LLY', 'DHR', 'BMY',
        'AMGN', 'GILD', 'CVS', 'CI', 'VRTX',
        'REGN', 'ISRG', 'SYK', 'BSX', 'EW'
    ],

    'Financials': [
        'JPM', 'BAC', 'WFC', 'C', 'GS',
        'MS', 'BLK', 'SPGI', 'AXP', 'BK',
        'USB', 'PNC', 'TFC', 'COF', 'SCHW',
        'CME', 'ICE', 'MMC', 'AON', 'AJG'
    ],

    'Energy': [
        'XOM', 'CVX', 'COP', 'SLB', 'EOG',
        'MPC', 'PSX', 'VLO', 'PXD', 'OXY',
        'KMI', 'WMB', 'HES', 'HAL', 'DVN',
        'FANG', 'MRO', 'APA', 'BKR', 'NOV'
    ],

    'Consumer': [
        'AMZN', 'TSLA', 'HD', 'MCD', 'NKE',
        'SBUX', 'TGT', 'LOW', 'DG', 'ROST',
        'YUM', 'CMG', 'ULTA', 'DPZ', 'ORLY',
        'AZO', 'BBY', 'EBAY', 'ETSY', 'W'
    ],

    'RealEstate': [
        'AMT', 'PLD', 'CCI', 'EQIX', 'PSA',
        'SPG', 'WELL', 'DLR', 'O', 'VICI',
        'EXR', 'AVB', 'EQR', 'VTR', 'SBAC',
        'ARE', 'INVH', 'MAA', 'UDR', 'ESS'
    ],

    'Industrials': [
        'GE', 'CAT', 'BA', 'HON', 'UNP',
        'RTX', 'LMT', 'UPS', 'DE', 'MMM',
        'GD', 'NOC', 'EMR', 'ETN', 'ITW',
        'CMI', 'PH', 'ROK', 'DOV', 'XYL'
    ],
}

# Validation
assert len(SECTORS) == 7, "Should have 7 sectors"
for sector, stocks in SECTORS.items():
    assert len(stocks) == 20, f"{sector} should have 20 stocks, has {len(stocks)}"
    assert len(stocks) == len(set(stocks)), f"{sector} has duplicate tickers"

print(f"✅ Sector configuration loaded: {len(SECTORS)} sectors, {sum(len(v) for v in SECTORS.values())} stocks")
