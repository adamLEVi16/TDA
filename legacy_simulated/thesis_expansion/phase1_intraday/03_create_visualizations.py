"""
Phase 1: Create Professional Visualizations
============================================

This script generates publication-quality figures for Section 6 of the thesis.

Figures created:
1. Figure 6.1: Stability comparison (daily vs intraday)
2. Figure 6.2: H1 evolution comparison
3. Figure 6.3: Sample size vs stability analysis

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
from plot_config import setup_plots, COLORS, save_figure, add_crisis_shading

print("=" * 80)
print("PHASE 1: CREATING PUBLICATION-QUALITY VISUALIZATIONS")
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

print("\n📂 Loading data...")

# Intraday topology
intraday_file = DATA_DIR / 'intraday_topology_features.csv'
if not intraday_file.exists():
    print(f"❌ Missing: {intraday_file}")
    print("Run 02_compute_topology.py first!")
    sys.exit(1)

topology_intraday = pd.read_csv(intraday_file, index_col=0, parse_dates=True)
print(f"✅ Loaded intraday topology: {len(topology_intraday)} snapshots")

# Daily topology (if exists)
daily_file = Path('/home/user/TDA/topology_features.csv')
has_daily = daily_file.exists()

if has_daily:
    topology_daily = pd.read_csv(daily_file, index_col=0, parse_dates=True)
    print(f"✅ Loaded daily topology: {len(topology_daily)} snapshots")
else:
    print("⚠️  Daily topology not found - creating intraday-only figures")

# ============================================================================
# FIGURE 6.1: STABILITY COMPARISON
# ============================================================================

if has_daily:
    print("\n📊 Creating Figure 6.1: Stability Comparison...")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Panel A: Box plots
    ax = axes[0]

    data_for_plot = [
        topology_daily['h1_loops'].values,
        topology_intraday['h1_count'].values,
    ]

    bp = ax.boxplot(data_for_plot,
                     labels=['Daily\n(1,494 obs)', 'Intraday\n(~40,000 obs)'],
                     patch_artist=True,
                     widths=0.6)

    # Color boxes
    bp['boxes'][0].set_facecolor(COLORS['blue'])
    bp['boxes'][1].set_facecolor(COLORS['orange'])

    for box in bp['boxes']:
        box.set_alpha(0.7)
        box.set_edgecolor('black')
        box.set_linewidth(1.5)

    # Styling
    ax.set_ylabel('H₁ Loop Count', fontsize=12, fontweight='bold')
    ax.set_title('A. Distribution Comparison', fontsize=13, fontweight='bold', loc='left')
    ax.grid(True, alpha=0.2, axis='y')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Panel B: Coefficient of Variation
    ax = axes[1]

    daily_cv = topology_daily['h1_loops'].std() / topology_daily['h1_loops'].mean()
    intraday_cv = topology_intraday['h1_count'].std() / topology_intraday['h1_count'].mean()

    cvs = [daily_cv, intraday_cv]
    labels = ['Daily', 'Intraday']
    colors = [COLORS['blue'], COLORS['orange']]

    bars = ax.bar(labels, cvs, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)

    # Add value labels on bars
    for bar, cv in zip(bars, cvs):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{cv:.3f}',
                ha='center', va='bottom', fontsize=11, fontweight='bold')

    # Add improvement annotation
    improvement = (1 - intraday_cv/daily_cv) * 100
    ax.text(0.5, max(cvs) * 0.8,
            f'{improvement:.1f}%\nimprovement',
            ha='center', fontsize=12, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgreen', alpha=0.7))

    ax.set_ylabel('Coefficient of Variation', fontsize=12, fontweight='bold')
    ax.set_title('B. Stability Metric (Lower = More Stable)', fontsize=13, fontweight='bold', loc='left')
    ax.set_ylim(0, max(cvs) * 1.15)
    ax.grid(True, alpha=0.2, axis='y')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    save_figure('figure6_1_stability_comparison', fig, formats=['pdf', 'png'])
    plt.close()

    print("✅ Figure 6.1 saved")

# ============================================================================
# FIGURE 6.2: H1 EVOLUTION COMPARISON
# ============================================================================

if has_daily:
    print("\n📊 Creating Figure 6.2: H1 Evolution Comparison...")

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10), sharex=True)

    # Top panel: Daily
    topology_daily['h1_loops'].plot(ax=ax1, linewidth=1.8, color=COLORS['blue'],
                                     alpha=0.7, label='Daily Topology')

    # Add ±2σ bands
    mean_daily = topology_daily['h1_loops'].mean()
    std_daily = topology_daily['h1_loops'].std()

    ax1.axhline(y=mean_daily, color='black', linestyle='--', linewidth=1.5,
                alpha=0.5, label='Mean')
    ax1.fill_between(topology_daily.index,
                      mean_daily - 2*std_daily,
                      mean_daily + 2*std_daily,
                      alpha=0.15, color=COLORS['blue'], label='±2σ')

    # Crisis shading
    add_crisis_shading(ax1)

    ax1.set_ylabel('H₁ Loop Count', fontsize=12, fontweight='bold')
    ax1.set_title('A. Daily Data (1,494 observations)', fontsize=13, fontweight='bold', loc='left')
    ax1.legend(loc='upper left', fontsize=10)
    ax1.grid(True, alpha=0.2)
    ax1.set_xlim(topology_daily.index[0], topology_daily.index[-1])

    # Bottom panel: Intraday
    topology_intraday['h1_count'].plot(ax=ax2, linewidth=1.8, color=COLORS['orange'],
                                        alpha=0.7, label='Intraday Topology')

    # Add ±2σ bands
    mean_intra = topology_intraday['h1_count'].mean()
    std_intra = topology_intraday['h1_count'].std()

    ax2.axhline(y=mean_intra, color='black', linestyle='--', linewidth=1.5,
                alpha=0.5, label='Mean')
    ax2.fill_between(topology_intraday.index,
                      mean_intra - 2*std_intra,
                      mean_intra + 2*std_intra,
                      alpha=0.15, color=COLORS['orange'], label='±2σ')

    # Crisis shading
    add_crisis_shading(ax2)

    ax2.set_ylabel('H₁ Loop Count', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Date', fontsize=12, fontweight='bold')
    ax2.set_title(f'B. Intraday Data (~40,000 observations)', fontsize=13, fontweight='bold', loc='left')
    ax2.legend(loc='upper left', fontsize=10)
    ax2.grid(True, alpha=0.2)
    ax2.set_xlim(topology_intraday.index[0], topology_intraday.index[-1])

    plt.tight_layout()
    save_figure('figure6_2_h1_evolution', fig, formats=['pdf', 'png'])
    plt.close()

    print("✅ Figure 6.2 saved")

# ============================================================================
# FIGURE 6.3: ROLLING STATISTICS (Intraday only)
# ============================================================================

print("\n📊 Creating Figure 6.3: Rolling Statistics...")

fig, axes = plt.subplots(2, 2, figsize=(16, 10))

# Panel A: H1 loops time series
ax = axes[0, 0]
topology_intraday['h1_count'].plot(ax=ax, linewidth=2.0, color=COLORS['blue'], alpha=0.8)

threshold = topology_intraday['h1_count'].quantile(0.75)
ax.axhline(y=threshold, color=COLORS['red'], linestyle='--',
           linewidth=2.0, alpha=0.7, label='75th percentile')

ax.set_ylabel('H₁ Loop Count', fontsize=11, fontweight='bold')
ax.set_title('A. H₁ Loops Over Time', fontsize=12, fontweight='bold', loc='left')
ax.legend(loc='upper right')
ax.grid(True, alpha=0.2)

# Panel B: H1 persistence
ax = axes[0, 1]
topology_intraday['h1_persistence'].plot(ax=ax, linewidth=2.0,
                                          color=COLORS['green'], alpha=0.8)

ax.set_ylabel('Total Persistence', fontsize=11, fontweight='bold')
ax.set_title('B. H₁ Persistence Over Time', fontsize=12, fontweight='bold', loc='left')
ax.grid(True, alpha=0.2)

# Panel C: Histogram of H1 loops
ax = axes[1, 0]
ax.hist(topology_intraday['h1_count'], bins=30, color=COLORS['blue'],
        alpha=0.7, edgecolor='black', linewidth=1.2)

ax.axvline(x=topology_intraday['h1_count'].mean(), color='black',
           linestyle='--', linewidth=2.0, label='Mean')
ax.axvline(x=topology_intraday['h1_count'].median(), color=COLORS['red'],
           linestyle='--', linewidth=2.0, label='Median')

ax.set_xlabel('H₁ Loop Count', fontsize=11, fontweight='bold')
ax.set_ylabel('Frequency', fontsize=11, fontweight='bold')
ax.set_title('C. Distribution of H₁ Loops', fontsize=12, fontweight='bold', loc='left')
ax.legend(loc='upper right')
ax.grid(True, alpha=0.2, axis='y')

# Panel D: Rolling standard deviation
ax = axes[1, 1]

# Adaptive window size (use 1/3 of data length, min 3, max 30)
n_samples = len(topology_intraday)
window_size = max(3, min(30, n_samples // 3))

rolling_std = topology_intraday['h1_count'].rolling(window=window_size).std()

# Plot only non-NaN values
rolling_std.dropna().plot(ax=ax, linewidth=2.0, color=COLORS['purple'], alpha=0.8)

ax.set_xlabel('Date', fontsize=11, fontweight='bold')
ax.set_ylabel(f'{window_size}-Period Rolling Std Dev', fontsize=11, fontweight='bold')
ax.set_title('D. Topology Volatility', fontsize=12, fontweight='bold', loc='left')
ax.grid(True, alpha=0.2)

plt.tight_layout()
save_figure('figure6_3_rolling_stats', fig, formats=['pdf', 'png'])
plt.close()

print("✅ Figure 6.3 saved")

# ============================================================================
# SUMMARY
# ============================================================================

print("\n" + "=" * 80)
print("VISUALIZATION COMPLETE")
print("=" * 80)

print(f"\n📊 Figures saved to: {FIG_DIR}")
print("\nFiles created:")
print("  - figure6_1_stability_comparison.pdf")
print("  - figure6_2_h1_evolution.pdf")
print("  - figure6_3_rolling_stats.pdf")
print("  (+ PNG versions)")

print("\nThese figures are ready for:")
print("  ✅ Journal submission (300 DPI, vector PDF)")
print("  ✅ Thesis inclusion (high resolution)")
print("  ✅ Presentations (clear, readable)")

print("\nNext: Copy Section 6 text from section6_text.md into your thesis")
