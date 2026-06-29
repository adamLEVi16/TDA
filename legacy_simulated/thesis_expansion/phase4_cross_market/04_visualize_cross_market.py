"""
Phase 4: Cross-Market Visualizations
=====================================

Creates publication-quality figures for Section 9.

Figures:
1. Figure 9.1: Correlation-CV scatter plot (all markets)
2. Figure 9.2: Asset class comparison (bar charts)
3. Figure 9.3: Geographic heatmap
4. Figure 9.4: Topology stability by market

Author: Adam Levine
Date: January 2026
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys

# Add parent for imports
sys.path.append(str(Path(__file__).parent.parent))
from plot_config import setup_plots, COLORS, save_figure

print("=" * 80)
print("PHASE 4: CREATING CROSS-MARKET VISUALIZATIONS")
print("=" * 80)

# Setup
setup_plots('publication')

# Directories
DATA_DIR = Path(__file__).parent.parent / 'data'
FIG_DIR = Path(__file__).parent.parent / 'figures'
FIG_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================================
# LOAD DATA
# ============================================================================

print("\n📂 Loading cross-market summary...")

summary_file = DATA_DIR / 'phase4_cross_market_summary.csv'

if not summary_file.exists():
    print("❌ Summary file not found. Run 03_cross_market_comparison.py first!")
    sys.exit(1)

markets_df = pd.read_csv(summary_file)

print(f"✅ Loaded {len(markets_df)} markets")

# ============================================================================
# FIGURE 9.1: CORRELATION-CV SCATTER PLOT
# ============================================================================

print("\n📊 Creating Figure 9.1: Correlation-CV Relationship...")

fig, ax = plt.subplots(figsize=(14, 9))

# Color by asset class
color_map = {
    'US Equities': COLORS['blue'],
    'International Equities': COLORS['orange'],
    'Cryptocurrency': COLORS['purple']
}

marker_map = {
    'US Equities': 'o',
    'International Equities': 's',
    'Cryptocurrency': '^'
}

for asset_class in markets_df['Asset Class'].unique():
    subset = markets_df[markets_df['Asset Class'] == asset_class]

    ax.scatter(subset['Mean Correlation'], subset['CV'],
              s=250, alpha=0.8, edgecolors='black', linewidths=2.5,
              color=color_map.get(asset_class, COLORS['green']),
              marker=marker_map.get(asset_class, 'o'),
              label=asset_class, zorder=3)

    # Add market labels
    for _, row in subset.iterrows():
        label = row['Market'].replace('US ', '').replace('International ', '')
        label = label.replace('(', '').replace(')', '')

        ax.annotate(label,
                   xy=(row['Mean Correlation'], row['CV']),
                   xytext=(8, 8), textcoords='offset points',
                   fontsize=9, alpha=0.9, fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                            edgecolor='gray', alpha=0.7))

# Regression line (all markets)
correlations = markets_df['Mean Correlation'].values
cvs = markets_df['CV'].values

if len(correlations) >= 3:
    global_corr = np.corrcoef(correlations, cvs)[0, 1]

    z = np.polyfit(correlations, cvs, 1)
    p = np.poly1d(z)

    x_line = np.linspace(correlations.min() - 0.05, correlations.max() + 0.05, 100)
    ax.plot(x_line, p(x_line), 'k--', linewidth=3.0, alpha=0.6, zorder=2,
           label=f'Linear Fit (ρ = {global_corr:.2f})')

# Threshold boxes
# Good: Correlation > 0.5 AND CV < 0.6
ax.axhspan(0, 0.6, alpha=0.1, color='green', zorder=1)
ax.axvspan(0.5, 1.0, alpha=0.1, color='green', zorder=1)

# Add threshold labels
ax.text(0.95, 0.58, 'Stable Topology\n(CV < 0.6)', fontsize=10,
       ha='right', va='top', alpha=0.7, fontweight='bold',
       bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgreen', alpha=0.3))

ax.text(0.52, 0.95, 'High Correlation\n(ρ > 0.5)', fontsize=10,
       ha='left', va='top', alpha=0.7, fontweight='bold',
       bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgreen', alpha=0.3))

ax.set_xlabel('Mean Pairwise Correlation (ρ)', fontsize=13, fontweight='bold')
ax.set_ylabel('Coefficient of Variation (CV)', fontsize=13, fontweight='bold')
ax.set_title('Figure 9.1: Cross-Market Validation of Correlation-Stability Relationship',
            fontsize=14, fontweight='bold', pad=20)

ax.legend(loc='upper left', fontsize=11, framealpha=0.95, edgecolor='black', fancybox=True)
ax.grid(True, alpha=0.3, linestyle='--', linewidth=1.0)

ax.set_xlim(correlations.min() - 0.05, correlations.max() + 0.05)
ax.set_ylim(cvs.min() - 0.05, cvs.max() + 0.1)

plt.tight_layout()
save_figure('figure9_1_correlation_cv_scatter', fig, formats=['pdf', 'png'])
plt.close()

print("✅ Figure 9.1 saved")

# ============================================================================
# FIGURE 9.2: ASSET CLASS COMPARISON
# ============================================================================

print("\n📊 Creating Figure 9.2: Asset Class Comparison...")

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))

# Panel A: Mean Correlation by Asset Class
ax = ax1

asset_class_stats = markets_df.groupby('Asset Class').agg({
    'Mean Correlation': ['mean', 'std'],
    'CV': ['mean', 'std'],
    'N Assets': 'count'
})

asset_classes = asset_class_stats.index.tolist()
mean_corrs = asset_class_stats[('Mean Correlation', 'mean')].values
std_corrs = asset_class_stats[('Mean Correlation', 'std')].values

colors = [color_map.get(ac, COLORS['green']) for ac in asset_classes]

bars = ax.bar(asset_classes, mean_corrs, yerr=std_corrs, capsize=8,
             color=colors, alpha=0.7, edgecolor='black', linewidth=2.5,
             error_kw={'linewidth': 2.5, 'ecolor': 'black'})

ax.axhline(y=0.5, color='red', linestyle='--', linewidth=2.0, alpha=0.6,
          label='Trading Threshold (ρ = 0.5)')

ax.set_ylabel('Mean Pairwise Correlation', fontsize=12, fontweight='bold')
ax.set_title('A. Correlation by Asset Class', fontsize=13, fontweight='bold', loc='left')
ax.grid(True, alpha=0.3, axis='y')
ax.legend(loc='best', fontsize=10)

# Add value labels
for i, (bar, val, std) in enumerate(zip(bars, mean_corrs, std_corrs)):
    ax.text(i, val + std + 0.02, f'{val:.3f}', ha='center', va='bottom',
           fontsize=11, fontweight='bold')

# Panel B: Topology Stability (CV) by Asset Class
ax = ax2

mean_cvs = asset_class_stats[('CV', 'mean')].values
std_cvs = asset_class_stats[('CV', 'std')].values

bars = ax.bar(asset_classes, mean_cvs, yerr=std_cvs, capsize=8,
             color=colors, alpha=0.7, edgecolor='black', linewidth=2.5,
             error_kw={'linewidth': 2.5, 'ecolor': 'black'})

ax.axhline(y=0.6, color='red', linestyle='--', linewidth=2.0, alpha=0.6,
          label='Stability Threshold (CV = 0.6)')

ax.set_ylabel('Coefficient of Variation (CV)', fontsize=12, fontweight='bold')
ax.set_title('B. Topology Stability by Asset Class', fontsize=13, fontweight='bold', loc='left')
ax.grid(True, alpha=0.3, axis='y')
ax.legend(loc='best', fontsize=10)

# Add value labels
for i, (bar, val, std) in enumerate(zip(bars, mean_cvs, std_cvs)):
    ax.text(i, val + std + 0.02, f'{val:.3f}', ha='center', va='bottom',
           fontsize=11, fontweight='bold')

# Panel C: Number of Markets
ax = ax3

market_counts = asset_class_stats[('N Assets', 'count')].values

bars = ax.bar(asset_classes, market_counts,
             color=colors, alpha=0.7, edgecolor='black', linewidth=2.5)

ax.set_ylabel('Number of Markets', fontsize=12, fontweight='bold')
ax.set_title('C. Markets Tested by Asset Class', fontsize=13, fontweight='bold', loc='left')
ax.grid(True, alpha=0.3, axis='y')

# Add value labels
for i, (bar, val) in enumerate(zip(bars, market_counts)):
    ax.text(i, val + 0.2, f'{int(val)}', ha='center', va='bottom',
           fontsize=11, fontweight='bold')

plt.tight_layout()
save_figure('figure9_2_asset_class_comparison', fig, formats=['pdf', 'png'])
plt.close()

print("✅ Figure 9.2 saved")

# ============================================================================
# FIGURE 9.3: TRADING VIABILITY HEATMAP
# ============================================================================

print("\n📊 Creating Figure 9.3: Trading Viability Heatmap...")

fig, ax = plt.subplots(figsize=(14, 10))

# Create viability score: combine correlation and CV
# Score = (correlation - 0.3) - (CV - 0.4)
# Higher = better (high correlation, low CV)

markets_df['Viability Score'] = (markets_df['Mean Correlation'] - 0.3) - (markets_df['CV'] - 0.4)

# Sort by viability
markets_df_sorted = markets_df.sort_values('Viability Score', ascending=False)

# Prepare data for heatmap
metrics = ['Mean Correlation', 'CV', 'Mean H₁', 'Viability Score']
heatmap_data = markets_df_sorted[metrics].T

# Normalize each metric to 0-1 for visualization
heatmap_normalized = heatmap_data.copy()
for metric in metrics:
    row_data = heatmap_normalized.loc[metric]
    if metric == 'CV':  # Lower is better for CV
        heatmap_normalized.loc[metric] = 1 - (row_data - row_data.min()) / (row_data.max() - row_data.min())
    else:  # Higher is better
        heatmap_normalized.loc[metric] = (row_data - row_data.min()) / (row_data.max() - row_data.min())

# Plot heatmap
sns.heatmap(heatmap_normalized, annot=heatmap_data, fmt='.2f',
           cmap='RdYlGn', center=0.5, vmin=0, vmax=1,
           xticklabels=markets_df_sorted['Market'].values,
           yticklabels=['Correlation (ρ)', 'Stability (1-CV)', 'Mean H₁', 'Viability Score'],
           cbar_kws={'label': 'Normalized Score (0 = Worst, 1 = Best)'},
           linewidths=1.5, linecolor='black', ax=ax,
           annot_kws={'fontsize': 9, 'fontweight': 'bold'})

ax.set_title('Figure 9.3: Cross-Market Trading Viability Assessment',
            fontsize=14, fontweight='bold', pad=20)
ax.set_xlabel('Market', fontsize=12, fontweight='bold')
ax.set_ylabel('Metric', fontsize=12, fontweight='bold')

plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)

plt.tight_layout()
save_figure('figure9_3_trading_viability_heatmap', fig, formats=['pdf', 'png'])
plt.close()

print("✅ Figure 9.3 saved")

# ============================================================================
# FIGURE 9.4: REGIONAL COMPARISON
# ============================================================================

print("\n📊 Creating Figure 9.4: Regional Comparison...")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

# Panel A: Correlation by Region
ax = ax1

region_stats = markets_df.groupby('Region').agg({
    'Mean Correlation': ['mean', 'std', 'count'],
    'CV': ['mean', 'std']
})

regions = region_stats.index.tolist()
region_corrs = region_stats[('Mean Correlation', 'mean')].values
region_corr_stds = region_stats[('Mean Correlation', 'std')].values

region_colors = [COLORS['blue'], COLORS['orange'], COLORS['green'], COLORS['purple']][:len(regions)]

bars = ax.barh(regions, region_corrs, xerr=region_corr_stds, capsize=8,
              color=region_colors, alpha=0.7, edgecolor='black', linewidth=2.5,
              error_kw={'linewidth': 2.5, 'ecolor': 'black'})

ax.axvline(x=0.5, color='red', linestyle='--', linewidth=2.0, alpha=0.6,
          label='Trading Threshold')

ax.set_xlabel('Mean Pairwise Correlation', fontsize=12, fontweight='bold')
ax.set_title('A. Correlation by Geographic Region', fontsize=13, fontweight='bold', loc='left')
ax.grid(True, alpha=0.3, axis='x')
ax.legend(loc='best', fontsize=10)

# Add value labels
for i, (bar, val) in enumerate(zip(bars, region_corrs)):
    ax.text(val + 0.02, i, f'{val:.3f}', va='center', fontsize=11, fontweight='bold')

# Panel B: Topology Stability by Region
ax = ax2

region_cvs = region_stats[('CV', 'mean')].values
region_cv_stds = region_stats[('CV', 'std')].values

bars = ax.barh(regions, region_cvs, xerr=region_cv_stds, capsize=8,
              color=region_colors, alpha=0.7, edgecolor='black', linewidth=2.5,
              error_kw={'linewidth': 2.5, 'ecolor': 'black'})

ax.axvline(x=0.6, color='red', linestyle='--', linewidth=2.0, alpha=0.6,
          label='Stability Threshold')

ax.set_xlabel('Coefficient of Variation (CV)', fontsize=12, fontweight='bold')
ax.set_title('B. Topology Stability by Region', fontsize=13, fontweight='bold', loc='left')
ax.grid(True, alpha=0.3, axis='x')
ax.legend(loc='best', fontsize=10)

# Add value labels
for i, (bar, val) in enumerate(zip(bars, region_cvs)):
    ax.text(val + 0.02, i, f'{val:.3f}', va='center', fontsize=11, fontweight='bold')

plt.tight_layout()
save_figure('figure9_4_regional_comparison', fig, formats=['pdf', 'png'])
plt.close()

print("✅ Figure 9.4 saved")

# ============================================================================
# SUMMARY
# ============================================================================

print("\n" + "=" * 80)
print("VISUALIZATION COMPLETE")
print("=" * 80)

print(f"\n📊 Figures saved to: {FIG_DIR}")

print("\nFiles created:")
print("  - figure9_1_correlation_cv_scatter.pdf/.png")
print("  - figure9_2_asset_class_comparison.pdf/.png")
print("  - figure9_3_trading_viability_heatmap.pdf/.png")
print("  - figure9_4_regional_comparison.pdf/.png")

print("\nThese figures are ready for:")
print("  ✅ Thesis Section 9 (Cross-Market Validation)")
print("  ✅ Journal submission (300 DPI, vector PDF)")
print("  ✅ Presentations")

print("\nKey Insights:")

if len(markets_df) >= 3:
    # Most stable market
    most_stable = markets_df.nsmallest(1, 'CV').iloc[0]
    print(f"  🏆 Most Stable:   {most_stable['Market']} (CV = {most_stable['CV']:.3f})")

    # Highest correlation
    highest_corr = markets_df.nlargest(1, 'Mean Correlation').iloc[0]
    print(f"  🏆 Highest Corr:  {highest_corr['Market']} (ρ = {highest_corr['Mean Correlation']:.3f})")

    # Trading viable count
    viable = markets_df[(markets_df['Mean Correlation'] > 0.5) & (markets_df['CV'] < 0.6)]
    print(f"  ✅ Trading Viable: {len(viable)}/{len(markets_df)} markets")

print("\nNext: Create SECTION_9_TEXT.md with written content for thesis")
