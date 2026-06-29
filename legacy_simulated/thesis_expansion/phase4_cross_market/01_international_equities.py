"""
Phase 4: International Equities Cross-Market Validation
========================================================

Tests if sector-specific topology generalizes to non-US markets.

Markets tested:
- FTSE 100 (UK) - European financial center
- DAX 40 (Germany) - European industrials/manufacturing
- Nikkei 225 (Japan) - Asian technology/automotive

Hypothesis: If correlation-topology relationship is universal, high-correlation
sectors in international markets should also produce positive Sharpe ratios.

Author: Adam Levine
Date: January 2026
"""

import yfinance as yf
import pandas as pd
import numpy as np
from pathlib import Path
from ripser import ripser
import time
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("PHASE 4: INTERNATIONAL EQUITIES CROSS-MARKET VALIDATION")
print("=" * 80)

# Configuration
START_DATE = '2020-01-01'
END_DATE = '2024-12-31'
DATA_DIR = Path(__file__).parent.parent / 'data'
DATA_DIR.mkdir(parents=True, exist_ok=True)

# International stock universes (15 stocks per market for efficiency)
INTERNATIONAL_MARKETS = {
    'FTSE': {
        'name': 'UK (FTSE 100)',
        'tickers': ['HSBA.L', 'BARC.L', 'LLOY.L', 'RIO.L', 'BP.L',
                   'SHEL.L', 'AZN.L', 'GSK.L', 'ULVR.L', 'DGE.L',
                   'VOD.L', 'BT-A.L', 'PRU.L', 'LSEG.L', 'REL.L']
    },
    'DAX': {
        'name': 'Germany (DAX 40)',
        'tickers': ['SAP.DE', 'SIE.DE', 'ALV.DE', 'DTE.DE', 'VOW3.DE',
                   'BMW.DE', 'BAS.DE', 'MUV2.DE', 'EOAN.DE', 'DB1.DE',
                   'DAI.DE', 'BAY.DE', 'FRE.DE', 'HEN3.DE', 'MBG.DE']
    },
    'Nikkei': {
        'name': 'Japan (Nikkei 225)',
        'tickers': ['7203.T', '6758.T', '9984.T', '6861.T', '8306.T',
                   '9433.T', '8035.T', '6902.T', '4063.T', '4502.T',
                   '4503.T', '8001.T', '8031.T', '9432.T', '7267.T']
    }
}

# ============================================================================
# DOWNLOAD INTERNATIONAL DATA
# ============================================================================

print("\n📥 Downloading international market data...")

international_data = {}

for market_code, market_info in INTERNATIONAL_MARKETS.items():
    print(f"\n🌍 {market_info['name']}:")

    returns_list = []
    successful_tickers = []

    for ticker in market_info['tickers']:
        try:
            data = yf.download(ticker, start=START_DATE, end=END_DATE,
                             progress=False, show_errors=False)

            if len(data) > 100:
                returns = data['Adj Close'].pct_change()
                returns.name = ticker
                returns_list.append(returns)
                successful_tickers.append(ticker)
                print(f"  ✅ {ticker}: {len(returns)} days")
            else:
                print(f"  ⚠️  {ticker}: Insufficient data")

        except Exception as e:
            print(f"  ❌ {ticker}: Failed")
            continue

        time.sleep(0.1)  # Rate limiting

    if len(returns_list) >= 10:
        market_df = pd.concat(returns_list, axis=1)
        market_df = market_df.dropna(how='all')

        international_data[market_code] = market_df

        # Save
        output_file = DATA_DIR / f'international_{market_code.lower()}_returns.csv'
        market_df.to_csv(output_file)

        print(f"  💾 Saved: {len(market_df)} days × {len(successful_tickers)} stocks")
    else:
        print(f"  ❌ Insufficient stocks for {market_code}")

if len(international_data) == 0:
    print("\n❌ No international data downloaded. Check internet connection or ticker symbols.")
    exit(1)

# ============================================================================
# CORRELATION ANALYSIS
# ============================================================================

print("\n" + "=" * 80)
print("CORRELATION ANALYSIS")
print("=" * 80)

correlation_stats = []

for market_code, df in international_data.items():
    corr_matrix = df.corr()

    # Get upper triangle
    upper_tri = corr_matrix.where(np.triu(np.ones_like(corr_matrix), k=1).astype(bool))
    correlations = upper_tri.stack().values

    mean_corr = np.mean(correlations)

    correlation_stats.append({
        'Market': INTERNATIONAL_MARKETS[market_code]['name'],
        'Mean Correlation': mean_corr,
        'Stocks': df.shape[1]
    })

    print(f"\n{INTERNATIONAL_MARKETS[market_code]['name']}:")
    print(f"  Mean correlation: {mean_corr:.3f}")
    print(f"  Stocks analyzed: {df.shape[1]}")

# Compare to US benchmark (Technology sector from Phase 2)
us_tech_file = DATA_DIR / 'sector_technology_returns.csv'
if us_tech_file.exists():
    us_tech = pd.read_csv(us_tech_file, index_col=0, parse_dates=True)
    us_corr_matrix = us_tech.corr()
    us_upper = us_corr_matrix.where(np.triu(np.ones_like(us_corr_matrix), k=1).astype(bool))
    us_mean_corr = us_upper.stack().mean()

    print(f"\n📊 Comparison to US Benchmark:")
    print(f"  US Technology: {us_mean_corr:.3f}")

    for market_code, df in international_data.items():
        market_corr = correlation_stats[[s['Market'] == INTERNATIONAL_MARKETS[market_code]['name']
                                        for s in correlation_stats]][0]['Mean Correlation']
        ratio = market_corr / us_mean_corr
        print(f"  {INTERNATIONAL_MARKETS[market_code]['name']}: {market_corr:.3f} ({ratio:.2f}x US)")

# ============================================================================
# COMPUTE TOPOLOGY
# ============================================================================

print("\n" + "=" * 80)
print("COMPUTING TOPOLOGY FOR INTERNATIONAL MARKETS")
print("=" * 80)

def compute_topology(returns_window):
    """Compute H0 and H1 features."""
    try:
        corr = returns_window.corr()
        dist = np.sqrt(2 * (1 - corr.values))

        result = ripser(dist, maxdim=1, distance_matrix=True, thresh=0.5)

        h1_dgm = result['dgms'][1]
        h1_count = len(h1_dgm)
        h1_persistence = (h1_dgm[:, 1] - h1_dgm[:, 0]).sum() if len(h1_dgm) > 0 else 0

        return {'h1_count': h1_count, 'h1_persistence': h1_persistence}
    except:
        return None

international_topology = {}

for market_code, df in international_data.items():
    print(f"\n🌍 Computing topology for {INTERNATIONAL_MARKETS[market_code]['name']}...")

    topology_features = []

    for i in range(60, len(df)):
        window = df.iloc[i-60:i]

        topo = compute_topology(window)
        if topo:
            topo['date'] = df.index[i]
            topology_features.append(topo)

        if (i - 60) % 100 == 0:
            pct = (i - 60) / (len(df) - 60) * 100
            print(f"  Progress: {pct:.1f}%", end='\r')

    print(f"  Progress: 100.0%")

    if topology_features:
        topo_df = pd.DataFrame(topology_features).set_index('date')
        international_topology[market_code] = topo_df

        # Save
        output_file = DATA_DIR / f'international_{market_code.lower()}_topology.csv'
        topo_df.to_csv(output_file)

        # Compute CV
        cv = topo_df['h1_count'].std() / topo_df['h1_count'].mean()
        mean_h1 = topo_df['h1_count'].mean()

        print(f"  ✅ Computed {len(topo_df)} snapshots")
        print(f"     Mean H₁: {mean_h1:.2f}, CV: {cv:.3f}")

# ============================================================================
# STABILITY COMPARISON
# ============================================================================

print("\n" + "=" * 80)
print("TOPOLOGY STABILITY COMPARISON")
print("=" * 80)

stability_results = []

for market_code, topo_df in international_topology.items():
    mean_h1 = topo_df['h1_count'].mean()
    std_h1 = topo_df['h1_count'].std()
    cv = std_h1 / mean_h1

    # Get correlation
    corr = [s['Mean Correlation'] for s in correlation_stats
            if s['Market'] == INTERNATIONAL_MARKETS[market_code]['name']][0]

    stability_results.append({
        'Market': INTERNATIONAL_MARKETS[market_code]['name'],
        'Mean H₁': mean_h1,
        'CV': cv,
        'Correlation': corr
    })

stability_df = pd.DataFrame(stability_results).sort_values('CV')

print(f"\n{'Market':<20} {'Mean H₁':<10} {'CV':<10} {'Correlation':<12}")
print("-" * 60)
for _, row in stability_df.iterrows():
    print(f"{row['Market']:<20} {row['Mean H₁']:>9.2f}  {row['CV']:>9.3f}  {row['Correlation']:>11.3f}")

# ============================================================================
# SAVE SUMMARY
# ============================================================================

summary_file = DATA_DIR / 'phase4_international_summary.csv'
stability_df.to_csv(summary_file, index=False)

print(f"\n💾 Summary saved: {summary_file}")

# ============================================================================
# KEY FINDINGS
# ============================================================================

print("\n" + "=" * 80)
print("KEY FINDINGS")
print("=" * 80)

# Test correlation-CV relationship
if len(stability_results) >= 2:
    corrs = [s['Correlation'] for s in stability_results]
    cvs = [s['CV'] for s in stability_results]

    # Simple correlation
    correlation_cv_relationship = np.corrcoef(corrs, cvs)[0, 1]

    print(f"\n📊 Correlation-CV Relationship: {correlation_cv_relationship:.3f}")

    if correlation_cv_relationship < -0.5:
        print("✅ VALIDATES! Higher correlation → Lower CV (more stable)")
        print("   Finding generalizes to international markets")
    else:
        print("⚠️  Relationship weaker in international markets")
        print("   May be due to different market structures")

print("\nNext: Run 02_cryptocurrency_analysis.py to test crypto markets")
