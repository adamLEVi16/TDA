"""
Phase 4: Cross-Market Comparison
=================================

Compares topology across all markets tested:
- US Sectors (7 sectors from Phase 2)
- International Equities (FTSE, DAX, Nikkei)
- Cryptocurrencies

Tests key hypothesis from Section 7:
- Does correlation-CV relationship (ρ = -0.87) generalize globally?
- Do findings hold across different asset classes?

This validates whether sector-specific topology is a universal phenomenon
or US-specific quirk.

Author: Adam Levine
Date: January 2026
"""

import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import sys

print("=" * 80)
print("PHASE 4: CROSS-MARKET COMPARISON")
print("=" * 80)

DATA_DIR = Path(__file__).parent.parent / 'data'

# ============================================================================
# LOAD ALL MARKET DATA
# ============================================================================

print("\n📂 Loading data from all markets...")

all_markets = []

# US Sectors (Phase 2)
print("\n🇺🇸 US Sectors:")

us_sectors = ['technology', 'healthcare', 'financials', 'energy',
              'consumer_discretionary', 'industrials', 'materials']

for sector in us_sectors:
    topology_file = DATA_DIR / f'sector_{sector}_topology.csv'
    returns_file = DATA_DIR / f'sector_{sector}_returns.csv'

    if topology_file.exists() and returns_file.exists():
        topology_df = pd.read_csv(topology_file, index_col=0, parse_dates=True)
        returns_df = pd.read_csv(returns_file, index_col=0, parse_dates=True)

        # Compute metrics
        mean_h1 = topology_df['h1_count'].mean()
        cv = topology_df['h1_count'].std() / mean_h1

        corr_matrix = returns_df.corr()
        upper_tri = corr_matrix.where(np.triu(np.ones_like(corr_matrix), k=1).astype(bool))
        mean_corr = upper_tri.stack().mean()

        all_markets.append({
            'Market': f'US {sector.title()}',
            'Asset Class': 'US Equities',
            'Region': 'North America',
            'Mean Correlation': mean_corr,
            'Mean H₁': mean_h1,
            'CV': cv,
            'N Assets': returns_df.shape[1]
        })

        print(f"  ✅ {sector.title()}: ρ={mean_corr:.3f}, CV={cv:.3f}")

# International Equities (Phase 4)
print("\n🌍 International Equities:")

intl_markets = {
    'ftse': ('UK (FTSE)', 'Europe'),
    'dax': ('Germany (DAX)', 'Europe'),
    'nikkei': ('Japan (Nikkei)', 'Asia')
}

for market_code, (market_name, region) in intl_markets.items():
    topology_file = DATA_DIR / f'international_{market_code}_topology.csv'
    returns_file = DATA_DIR / f'international_{market_code}_returns.csv'

    if topology_file.exists() and returns_file.exists():
        topology_df = pd.read_csv(topology_file, index_col=0, parse_dates=True)
        returns_df = pd.read_csv(returns_file, index_col=0, parse_dates=True)

        mean_h1 = topology_df['h1_count'].mean()
        cv = topology_df['h1_count'].std() / mean_h1

        corr_matrix = returns_df.corr()
        upper_tri = corr_matrix.where(np.triu(np.ones_like(corr_matrix), k=1).astype(bool))
        mean_corr = upper_tri.stack().mean()

        all_markets.append({
            'Market': market_name,
            'Asset Class': 'International Equities',
            'Region': region,
            'Mean Correlation': mean_corr,
            'Mean H₁': mean_h1,
            'CV': cv,
            'N Assets': returns_df.shape[1]
        })

        print(f"  ✅ {market_name}: ρ={mean_corr:.3f}, CV={cv:.3f}")

# Cryptocurrencies (Phase 4)
print("\n₿ Cryptocurrencies:")

crypto_topology_file = DATA_DIR / 'cryptocurrency_topology.csv'
crypto_returns_file = DATA_DIR / 'cryptocurrency_returns.csv'

if crypto_topology_file.exists() and crypto_returns_file.exists():
    topology_df = pd.read_csv(crypto_topology_file, index_col=0, parse_dates=True)
    returns_df = pd.read_csv(crypto_returns_file, index_col=0, parse_dates=True)

    mean_h1 = topology_df['h1_count'].mean()
    cv = topology_df['h1_count'].std() / mean_h1

    corr_matrix = returns_df.corr()
    upper_tri = corr_matrix.where(np.triu(np.ones_like(corr_matrix), k=1).astype(bool))
    mean_corr = upper_tri.stack().mean()

    all_markets.append({
        'Market': 'Cryptocurrency',
        'Asset Class': 'Cryptocurrency',
        'Region': 'Global',
        'Mean Correlation': mean_corr,
        'Mean H₁': mean_h1,
        'CV': cv,
        'N Assets': returns_df.shape[1]
    })

    print(f"  ✅ Cryptocurrency: ρ={mean_corr:.3f}, CV={cv:.3f}")

if len(all_markets) == 0:
    print("\n❌ No market data found. Run Phase 2 and Phase 4 scripts first!")
    sys.exit(1)

markets_df = pd.DataFrame(all_markets)

print(f"\n✅ Loaded {len(markets_df)} markets across {markets_df['Asset Class'].nunique()} asset classes")

# ============================================================================
# COMPARISON TABLE
# ============================================================================

print("\n" + "=" * 80)
print("CROSS-MARKET COMPARISON")
print("=" * 80)

# Sort by CV (most stable first)
markets_df = markets_df.sort_values('CV')

print(f"\n{'Market':<25} {'Asset Class':<20} {'Correlation':<12} {'CV':<8} {'Mean H₁':<10}")
print("-" * 90)

for _, row in markets_df.iterrows():
    print(f"{row['Market']:<25} {row['Asset Class']:<20} {row['Mean Correlation']:>11.3f}  "
          f"{row['CV']:>7.3f}  {row['Mean H₁']:>9.2f}")

# ============================================================================
# CORRELATION-CV RELATIONSHIP TEST
# ============================================================================

print("\n" + "=" * 80)
print("TESTING CORRELATION-CV RELATIONSHIP")
print("=" * 80)

# Compute global correlation
correlations = markets_df['Mean Correlation'].values
cvs = markets_df['CV'].values

if len(correlations) >= 3:
    global_corr = np.corrcoef(correlations, cvs)[0, 1]

    print(f"\n📊 Global Correlation-CV Relationship: {global_corr:.3f}")

    # Compare to Section 7 finding (ρ = -0.87 for US sectors)
    us_sector_data = markets_df[markets_df['Asset Class'] == 'US Equities']

    if len(us_sector_data) >= 3:
        us_corr = np.corrcoef(us_sector_data['Mean Correlation'].values,
                             us_sector_data['CV'].values)[0, 1]

        print(f"\n🇺🇸 US Sectors (Section 7):      ρ = {us_corr:.3f}")
        print(f"🌍 Global (All Markets):         ρ = {global_corr:.3f}")
        print(f"📊 Difference:                   {abs(global_corr - us_corr):.3f}")

        if abs(global_corr - us_corr) < 0.2:
            print("\n✅ GENERALIZES! Relationship holds globally.")
            print("   → Higher correlation → More stable topology (lower CV)")
        elif global_corr < -0.5:
            print("\n🟡 PARTIALLY GENERALIZES. Relationship still negative.")
        else:
            print("\n⚠️  DOES NOT GENERALIZE. Relationship weakens globally.")

# ============================================================================
# ASSET CLASS BREAKDOWN
# ============================================================================

print("\n" + "=" * 80)
print("ASSET CLASS BREAKDOWN")
print("=" * 80)

for asset_class in markets_df['Asset Class'].unique():
    subset = markets_df[markets_df['Asset Class'] == asset_class]

    print(f"\n{asset_class}:")
    print(f"  Markets:            {len(subset)}")
    print(f"  Mean Correlation:   {subset['Mean Correlation'].mean():.3f} "
          f"(range: {subset['Mean Correlation'].min():.3f} - {subset['Mean Correlation'].max():.3f})")
    print(f"  Mean CV:            {subset['CV'].mean():.3f} "
          f"(range: {subset['CV'].min():.3f} - {subset['CV'].max():.3f})")

    # Assess stability
    if subset['CV'].mean() < 0.5:
        print(f"  ✅ STABLE topology (CV < 0.5)")
    elif subset['CV'].mean() < 0.7:
        print(f"  🟡 MODERATE stability (0.5 < CV < 0.7)")
    else:
        print(f"  ❌ UNSTABLE topology (CV > 0.7)")

# ============================================================================
# REGION BREAKDOWN
# ============================================================================

print("\n" + "=" * 80)
print("GEOGRAPHIC BREAKDOWN")
print("=" * 80)

for region in markets_df['Region'].unique():
    subset = markets_df[markets_df['Region'] == region]

    print(f"\n{region}:")
    print(f"  Markets:            {len(subset)}")
    print(f"  Mean Correlation:   {subset['Mean Correlation'].mean():.3f}")
    print(f"  Mean CV:            {subset['CV'].mean():.3f}")

# ============================================================================
# TRADING VIABILITY ASSESSMENT
# ============================================================================

print("\n" + "=" * 80)
print("TRADING VIABILITY ASSESSMENT")
print("=" * 80)

print("\nBased on Section 7 criteria:")
print("  ✅ Good:     Correlation > 0.5 AND CV < 0.6")
print("  🟡 Marginal: Correlation > 0.4 OR CV < 0.7")
print("  ❌ Poor:     Correlation < 0.4 AND CV > 0.7")

viable_markets = []

for _, row in markets_df.iterrows():
    corr = row['Mean Correlation']
    cv = row['CV']

    if corr > 0.5 and cv < 0.6:
        status = '✅ Good'
        viable_markets.append(row['Market'])
    elif corr > 0.4 or cv < 0.7:
        status = '🟡 Marginal'
    else:
        status = '❌ Poor'

    print(f"  {status:12} {row['Market']:<25} (ρ={corr:.3f}, CV={cv:.3f})")

print(f"\n📊 Trading-Viable Markets: {len(viable_markets)}/{len(markets_df)}")

if len(viable_markets) > 0:
    print(f"\nRecommended for TDA-based trading:")
    for market in viable_markets:
        print(f"  ✅ {market}")

# ============================================================================
# CORRELATION VS CV SCATTER PLOT
# ============================================================================

print("\n📊 Generating correlation-CV scatter plot...")

sys.path.append(str(Path(__file__).parent.parent))
from plot_config import setup_plots, COLORS

setup_plots('publication')

fig, ax = plt.subplots(figsize=(12, 8))

# Color by asset class
color_map = {
    'US Equities': COLORS['blue'],
    'International Equities': COLORS['orange'],
    'Cryptocurrency': COLORS['purple']
}

for asset_class in markets_df['Asset Class'].unique():
    subset = markets_df[markets_df['Asset Class'] == asset_class]

    ax.scatter(subset['Mean Correlation'], subset['CV'],
              s=200, alpha=0.7, edgecolors='black', linewidths=2,
              color=color_map.get(asset_class, COLORS['green']),
              label=asset_class)

    # Add labels
    for _, row in subset.iterrows():
        ax.annotate(row['Market'].replace('US ', '').replace('(', '').replace(')', ''),
                   xy=(row['Mean Correlation'], row['CV']),
                   xytext=(5, 5), textcoords='offset points',
                   fontsize=9, alpha=0.8)

# Regression line (all markets)
if len(correlations) >= 3:
    z = np.polyfit(correlations, cvs, 1)
    p = np.poly1d(z)

    x_line = np.linspace(correlations.min(), correlations.max(), 100)
    ax.plot(x_line, p(x_line), 'k--', linewidth=2.5, alpha=0.5,
           label=f'Linear fit (ρ = {global_corr:.2f})')

# Threshold lines
ax.axhline(y=0.6, color='gray', linestyle=':', linewidth=1.5, alpha=0.5)
ax.axvline(x=0.5, color='gray', linestyle=':', linewidth=1.5, alpha=0.5)

ax.text(0.5, 0.61, 'CV = 0.6 (stability threshold)', fontsize=9,
       ha='left', va='bottom', alpha=0.7)
ax.text(0.51, 0.95, 'ρ = 0.5 (correlation threshold)', fontsize=9,
       ha='left', va='top', alpha=0.7, rotation=90)

ax.set_xlabel('Mean Pairwise Correlation', fontsize=12, fontweight='bold')
ax.set_ylabel('Coefficient of Variation (CV)', fontsize=12, fontweight='bold')
ax.set_title('Cross-Market Validation: Correlation-Stability Relationship',
            fontsize=14, fontweight='bold')
ax.legend(loc='best', fontsize=10, framealpha=0.9)
ax.grid(True, alpha=0.2)

plt.tight_layout()

FIG_DIR = Path(__file__).parent.parent / 'figures'
FIG_DIR.mkdir(parents=True, exist_ok=True)

fig_file = FIG_DIR / 'phase4_correlation_cv_scatter'
plt.savefig(f'{fig_file}.pdf', dpi=300, bbox_inches='tight')
plt.savefig(f'{fig_file}.png', dpi=300, bbox_inches='tight')
plt.close()

print(f"✅ Saved: {fig_file}.pdf/.png")

# ============================================================================
# SAVE SUMMARY
# ============================================================================

summary_file = DATA_DIR / 'phase4_cross_market_summary.csv'
markets_df.to_csv(summary_file, index=False)

print(f"\n💾 Summary saved: {summary_file}")

# ============================================================================
# KEY FINDINGS
# ============================================================================

print("\n" + "=" * 80)
print("KEY FINDINGS")
print("=" * 80)

print("\n1. Generalization Test:")
if len(correlations) >= 3:
    if abs(global_corr - us_corr) < 0.2:
        print(f"   ✅ Correlation-CV relationship GENERALIZES globally")
        print(f"   ✅ ρ_global = {global_corr:.3f} ≈ ρ_US = {us_corr:.3f}")
    else:
        print(f"   ⚠️  Relationship differs across markets")
        print(f"   📊 ρ_global = {global_corr:.3f} vs ρ_US = {us_corr:.3f}")

print("\n2. Most Stable Markets:")
top_3 = markets_df.nsmallest(3, 'CV')
for i, (_, row) in enumerate(top_3.iterrows(), 1):
    print(f"   {i}. {row['Market']}: CV = {row['CV']:.3f}")

print("\n3. Trading Viability:")
print(f"   {len(viable_markets)}/{len(markets_df)} markets suitable for TDA-based trading")

print("\n4. Asset Class Ranking (by stability):")
asset_class_cv = markets_df.groupby('Asset Class')['CV'].mean().sort_values()
for i, (asset_class, cv) in enumerate(asset_class_cv.items(), 1):
    print(f"   {i}. {asset_class}: CV = {cv:.3f}")

print("\n" + "=" * 80)
print("CROSS-MARKET COMPARISON COMPLETE")
print("=" * 80)

print("\nNext: Run 04_visualize_cross_market.py to create publication figures")
