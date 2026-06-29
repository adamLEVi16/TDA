"""
Phase 2: Create Sector Analysis Visualizations
===============================================

Generates publication-quality figures for Section 7 (Sector-Specific Topology).

Figures created:
1. Figure 7.1: Sector topology stability comparison
2. Figure 7.2: Sector strategy performance (equity curves)
3. Figure 7.3: Multi-sector portfolio vs individuals
4. Figure 7.4: Within-sector vs cross-sector correlations

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
from sector_config import SECTORS

print("=" * 80)
print("PHASE 2: CREATING SECTOR ANALYSIS VISUALIZATIONS")
print("=" * 80)

# Setup
setup_plots('publication')

# Directories
DATA_DIR = Path(__file__).parent.parent / 'data'
FIG_DIR = Path(__file__).parent.parent / 'figures'
FIG_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================================
# FIGURE 7.1: SECTOR TOPOLOGY STABILITY
# ============================================================================

print("\n📊 Creating Figure 7.1: Sector Topology Stability...")

stability_file = DATA_DIR / 'sector_topology_stability.csv'

if stability_file.exists():
    stability_df = pd.read_csv(stability_file)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Panel A: H1 Loop Count CV
    ax = ax1

    sectors = stability_df['Sector'].values
    cvs = stability_df['H1_CV'].astype(float).values

    # Sort by CV
    sorted_idx = np.argsort(cvs)
    sectors_sorted = sectors[sorted_idx]
    cvs_sorted = cvs[sorted_idx]

    # Bar chart
    colors_list = [COLORS['green'] if cv < 0.5 else COLORS['orange'] if cv < 0.7 else COLORS['red']
                   for cv in cvs_sorted]

    bars = ax.barh(sectors_sorted, cvs_sorted, color=colors_list, alpha=0.7, edgecolor='black', linewidth=1.2)

    ax.set_xlabel('Coefficient of Variation (Lower = More Stable)', fontsize=11, fontweight='bold')
    ax.set_title('A. Topology Stability by Sector', fontsize=12, fontweight='bold', loc='left')
    ax.axvline(x=0.5, color='green', linestyle='--', linewidth=1.5, alpha=0.5, label='Good (< 0.5)')
    ax.axvline(x=0.7, color='orange', linestyle='--', linewidth=1.5, alpha=0.5, label='Fair (< 0.7)')
    ax.legend(loc='lower right', fontsize=9)
    ax.grid(True, alpha=0.2, axis='x')

    # Panel B: Mean H1 Loops
    ax = ax2

    h1_means = stability_df['H1_Mean'].astype(float).values[sorted_idx]

    bars = ax.barh(sectors_sorted, h1_means, color=COLORS['blue'], alpha=0.7, edgecolor='black', linewidth=1.2)

    ax.set_xlabel('Mean H₁ Loop Count', fontsize=11, fontweight='bold')
    ax.set_title('B. Average Topological Complexity', fontsize=12, fontweight='bold', loc='left')
    ax.grid(True, alpha=0.2, axis='x')

    plt.tight_layout()
    save_figure('figure7_1_sector_stability', fig, formats=['pdf', 'png'])
    plt.close()

    print("✅ Figure 7.1 saved")
else:
    print("⚠️  Stability file not found, skipping Figure 7.1")

# ============================================================================
# FIGURE 7.2: SECTOR STRATEGY PERFORMANCE
# ============================================================================

print("\n📊 Creating Figure 7.2: Sector Strategy Performance...")

# Load all sector strategies
sector_strategies = {}

for sector_name in SECTORS.keys():
    strategy_file = DATA_DIR / f'sector_{sector_name.lower()}_strategy.csv'

    if strategy_file.exists():
        strategy_df = pd.read_csv(strategy_file, index_col=0, parse_dates=True)
        sector_strategies[sector_name] = strategy_df

if sector_strategies:
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))

    # Panel A: Cumulative returns
    ax = ax1

    color_cycle = [COLORS['blue'], COLORS['orange'], COLORS['green'], COLORS['red'],
                   COLORS['purple'], COLORS['brown'], COLORS['gray']]

    for i, (sector_name, strategy_df) in enumerate(sector_strategies.items()):
        cum_returns = (1 + strategy_df['return']).cumprod()
        cum_returns.plot(ax=ax, label=sector_name, linewidth=2.0, alpha=0.8,
                        color=color_cycle[i % len(color_cycle)])

    ax.axhline(y=1.0, color='black', linestyle='--', linewidth=1.5, alpha=0.5)
    ax.set_ylabel('Cumulative Return (1.0 = Breakeven)', fontsize=11, fontweight='bold')
    ax.set_title('A. Sector Strategy Equity Curves', fontsize=12, fontweight='bold', loc='left')
    ax.legend(loc='best', fontsize=9, ncol=2)
    ax.grid(True, alpha=0.2)

    # Panel B: Drawdown
    ax = ax2

    for i, (sector_name, strategy_df) in enumerate(sector_strategies.items()):
        cum_returns = (1 + strategy_df['return']).cumprod()
        running_max = cum_returns.expanding().max()
        drawdown = (cum_returns - running_max) / running_max

        drawdown.plot(ax=ax, label=sector_name, linewidth=2.0, alpha=0.8,
                     color=color_cycle[i % len(color_cycle)])

    ax.axhline(y=0.0, color='black', linestyle='--', linewidth=1.5, alpha=0.5)
    ax.set_ylabel('Drawdown', fontsize=11, fontweight='bold')
    ax.set_xlabel('Date', fontsize=11, fontweight='bold')
    ax.set_title('B. Strategy Drawdowns', fontsize=12, fontweight='bold', loc='left')
    ax.legend(loc='best', fontsize=9, ncol=2)
    ax.grid(True, alpha=0.2)

    # Format y-axis as percentage
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.1f}'))

    plt.tight_layout()
    save_figure('figure7_2_sector_performance', fig, formats=['pdf', 'png'])
    plt.close()

    print("✅ Figure 7.2 saved")
else:
    print("⚠️  No sector strategies found, skipping Figure 7.2")

# ============================================================================
# FIGURE 7.3: MULTI-SECTOR PORTFOLIO
# ============================================================================

print("\n📊 Creating Figure 7.3: Multi-Sector Portfolio vs Individuals...")

portfolio_file = DATA_DIR / 'multi_sector_portfolio.csv'
comparison_file = DATA_DIR / 'sector_performance_comparison.csv'

if portfolio_file.exists() and comparison_file.exists():
    portfolio_df = pd.read_csv(portfolio_file, index_col=0, parse_dates=True)
    comparison_df = pd.read_csv(comparison_file)

    # Get top 3 sectors
    top_sectors = comparison_df.nlargest(3, 'Sharpe')['Sector'].tolist()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Panel A: Cumulative returns comparison
    ax = ax1

    # Plot top 3 individual sectors
    for i, sector in enumerate(top_sectors):
        strategy_file = DATA_DIR / f'sector_{sector.lower()}_strategy.csv'
        if strategy_file.exists():
            strategy_df = pd.read_csv(strategy_file, index_col=0, parse_dates=True)
            cum_returns = (1 + strategy_df['return']).cumprod()
            cum_returns.plot(ax=ax, label=sector, linewidth=1.5, alpha=0.6,
                           linestyle='--', color=COLORS[['blue', 'orange', 'green'][i]])

    # Plot multi-sector portfolio
    portfolio_cum = (1 + portfolio_df['return']).cumprod()
    portfolio_cum.plot(ax=ax, label='Multi-Sector Portfolio', linewidth=3.0,
                      color='black', alpha=0.9)

    ax.axhline(y=1.0, color='gray', linestyle=':', linewidth=1.5, alpha=0.5)
    ax.set_ylabel('Cumulative Return', fontsize=11, fontweight='bold')
    ax.set_xlabel('Date', fontsize=11, fontweight='bold')
    ax.set_title('A. Multi-Sector vs Individual Sectors', fontsize=12, fontweight='bold', loc='left')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.2)

    # Panel B: Sharpe ratio comparison
    ax = ax2

    # Sharpe ratios
    sharpes = []
    labels = []

    for sector in top_sectors:
        sharpe = comparison_df[comparison_df['Sector'] == sector]['Sharpe'].values[0]
        sharpes.append(sharpe)
        labels.append(sector)

    # Portfolio Sharpe
    portfolio_returns = portfolio_df['return']
    portfolio_sharpe = portfolio_returns.mean() / portfolio_returns.std() * np.sqrt(252)
    sharpes.append(portfolio_sharpe)
    labels.append('Multi-Sector')

    # Bar chart
    colors_bars = [COLORS['blue'], COLORS['orange'], COLORS['green'], 'black']
    bars = ax.barh(labels, sharpes, color=colors_bars, alpha=0.7, edgecolor='black', linewidth=1.5)

    # Highlight multi-sector
    bars[-1].set_linewidth(2.5)

    ax.axvline(x=0, color='gray', linestyle='-', linewidth=1.5)
    ax.set_xlabel('Sharpe Ratio', fontsize=11, fontweight='bold')
    ax.set_title('B. Risk-Adjusted Performance', fontsize=12, fontweight='bold', loc='left')
    ax.grid(True, alpha=0.2, axis='x')

    # Add value labels
    for i, (bar, sharpe) in enumerate(zip(bars, sharpes)):
        ax.text(sharpe, i, f'  {sharpe:.3f}', va='center', fontsize=10, fontweight='bold')

    plt.tight_layout()
    save_figure('figure7_3_multi_sector_portfolio', fig, formats=['pdf', 'png'])
    plt.close()

    print("✅ Figure 7.3 saved")
else:
    print("⚠️  Portfolio or comparison file not found, skipping Figure 7.3")

# ============================================================================
# FIGURE 7.4: CORRELATION HEATMAP
# ============================================================================

print("\n📊 Creating Figure 7.4: Within-Sector vs Cross-Sector Correlations...")

# Load returns for all sectors
sector_returns = {}

for sector_name in SECTORS.keys():
    returns_file = DATA_DIR / f'sector_{sector_name.lower()}_returns.csv'
    if returns_file.exists():
        returns_df = pd.read_csv(returns_file, index_col=0, parse_dates=True)
        # Use first 5 stocks from each sector for visualization
        sector_returns[sector_name] = returns_df.iloc[:, :5]

if len(sector_returns) >= 2:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Panel A: Within-sector correlation example (first sector)
    ax = ax1

    first_sector = list(sector_returns.keys())[0]
    first_returns = sector_returns[first_sector]

    corr_within = first_returns.corr()

    sns.heatmap(corr_within, annot=True, fmt='.2f', cmap='RdYlGn', center=0.5,
                vmin=0, vmax=1, square=True, ax=ax, cbar_kws={'label': 'Correlation'},
                linewidths=0.5, linecolor='gray')

    ax.set_title(f'A. Within-Sector Correlations ({first_sector})', fontsize=12, fontweight='bold', loc='left')

    # Panel B: Cross-sector correlation (sample from 2 sectors)
    ax = ax2

    if len(sector_returns) >= 2:
        sector_names = list(sector_returns.keys())
        sector1_returns = sector_returns[sector_names[0]]
        sector2_returns = sector_returns[sector_names[1]]

        # Align dates
        common_dates = sector1_returns.index.intersection(sector2_returns.index)
        sector1_aligned = sector1_returns.loc[common_dates]
        sector2_aligned = sector2_returns.loc[common_dates]

        # Compute cross-correlation
        cross_corr = pd.DataFrame(index=sector1_aligned.columns, columns=sector2_aligned.columns)

        for col1 in sector1_aligned.columns:
            for col2 in sector2_aligned.columns:
                cross_corr.loc[col1, col2] = sector1_aligned[col1].corr(sector2_aligned[col2])

        cross_corr = cross_corr.astype(float)

        sns.heatmap(cross_corr, annot=True, fmt='.2f', cmap='RdYlGn', center=0.5,
                    vmin=0, vmax=1, square=True, ax=ax, cbar_kws={'label': 'Correlation'},
                    linewidths=0.5, linecolor='gray')

        ax.set_title(f'B. Cross-Sector Correlations ({sector_names[0]} vs {sector_names[1]})',
                    fontsize=12, fontweight='bold', loc='left')
        ax.set_xlabel(sector_names[1], fontsize=10, fontweight='bold')
        ax.set_ylabel(sector_names[0], fontsize=10, fontweight='bold')

    plt.tight_layout()
    save_figure('figure7_4_correlation_heatmaps', fig, formats=['pdf', 'png'])
    plt.close()

    print("✅ Figure 7.4 saved")
else:
    print("⚠️  Insufficient sector data for correlation heatmap")

# ============================================================================
# SUMMARY
# ============================================================================

print("\n" + "=" * 80)
print("VISUALIZATION COMPLETE")
print("=" * 80)

print(f"\n📊 Figures saved to: {FIG_DIR}")

print("\nFiles created:")
print("  - figure7_1_sector_stability.pdf/.png")
print("  - figure7_2_sector_performance.pdf/.png")
print("  - figure7_3_multi_sector_portfolio.pdf/.png")
print("  - figure7_4_correlation_heatmaps.pdf/.png")

print("\nThese figures are ready for:")
print("  ✅ Thesis Section 7 (Sector-Specific Topology)")
print("  ✅ Journal submission (300 DPI, vector PDF)")
print("  ✅ Presentations")

print("\nNext: Copy Section 7 text from SECTION_7_TEXT.md into your thesis")
