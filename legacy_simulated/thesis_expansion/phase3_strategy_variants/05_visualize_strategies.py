"""
Phase 3: Visualize Strategy Variants
=====================================

Creates publication-quality figures for Section 8.

Figures:
1. Figure 8.1: Equity curves for all strategy variants
2. Figure 8.2: Performance comparison (Sharpe, returns, drawdowns)
3. Figure 8.3: Ensemble portfolio vs best individual

Author: Adam Levine
Date: January 2026
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# Add parent for imports
sys.path.append(str(Path(__file__).parent.parent))
from plot_config import setup_plots, COLORS, save_figure

print("=" * 80)
print("PHASE 3: CREATING STRATEGY VARIANT VISUALIZATIONS")
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

print("\n📂 Loading strategy results...")

strategies = {}

# Baseline
baseline_file = DATA_DIR / 'sector_technology_strategy.csv'
if baseline_file.exists():
    strategies['Baseline (Mean Rev)'] = pd.read_csv(baseline_file, index_col=0, parse_dates=True)

# Momentum + TDA
hybrid_file = DATA_DIR / 'phase3_momentum_tda_hybrid.csv'
if hybrid_file.exists():
    strategies['Momentum + TDA'] = pd.read_csv(hybrid_file, index_col=0, parse_dates=True)

# Scale-Consistent
scale_file = DATA_DIR / 'phase3_scale_consistent.csv'
if scale_file.exists():
    strategies['Scale-Consistent'] = pd.read_csv(scale_file, index_col=0, parse_dates=True)

# Adaptive Threshold
adaptive_file = DATA_DIR / 'phase3_adaptive_threshold.csv'
if adaptive_file.exists():
    strategies['Adaptive Threshold'] = pd.read_csv(adaptive_file, index_col=0, parse_dates=True)

if len(strategies) == 0:
    print("❌ No strategy results found. Exiting.")
    sys.exit(1)

print(f"✅ Loaded {len(strategies)} strategies")

# Align dates
common_dates = None
for name, df in strategies.items():
    if common_dates is None:
        common_dates = set(df.index)
    else:
        common_dates = common_dates.intersection(set(df.index))

common_dates = sorted(list(common_dates))

# ============================================================================
# FIGURE 8.1: EQUITY CURVES
# ============================================================================

print("\n📊 Creating Figure 8.1: Equity Curves...")

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))

# Panel A: Cumulative returns
ax = ax1

colors = [COLORS['blue'], COLORS['orange'], COLORS['green'], COLORS['purple']]

for i, (name, df) in enumerate(strategies.items()):
    aligned_returns = df.loc[common_dates, 'return']
    cum_returns = (1 + aligned_returns).cumprod()

    cum_returns.plot(ax=ax, label=name, linewidth=2.0, alpha=0.8,
                    color=colors[i % len(colors)])

ax.axhline(y=1.0, color='black', linestyle='--', linewidth=1.5, alpha=0.5, label='Breakeven')
ax.set_ylabel('Cumulative Return', fontsize=11, fontweight='bold')
ax.set_title('A. Strategy Equity Curves', fontsize=12, fontweight='bold', loc='left')
ax.legend(loc='best', fontsize=10)
ax.grid(True, alpha=0.2)

# Panel B: Drawdowns
ax = ax2

for i, (name, df) in enumerate(strategies.items()):
    aligned_returns = df.loc[common_dates, 'return']
    cum_returns = (1 + aligned_returns).cumprod()

    running_max = cum_returns.expanding().max()
    drawdown = (cum_returns - running_max) / running_max

    drawdown.plot(ax=ax, label=name, linewidth=2.0, alpha=0.8,
                 color=colors[i % len(colors)])

ax.axhline(y=0, color='black', linestyle='--', linewidth=1.5, alpha=0.5)
ax.set_ylabel('Drawdown', fontsize=11, fontweight='bold')
ax.set_xlabel('Date', fontsize=11, fontweight='bold')
ax.set_title('B. Strategy Drawdowns', fontsize=12, fontweight='bold', loc='left')
ax.legend(loc='best', fontsize=10)
ax.grid(True, alpha=0.2)

ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))

plt.tight_layout()
save_figure('figure8_1_strategy_equity_curves', fig, formats=['pdf', 'png'])
plt.close()

print("✅ Figure 8.1 saved")

# ============================================================================
# FIGURE 8.2: PERFORMANCE COMPARISON
# ============================================================================

print("\n📊 Creating Figure 8.2: Performance Comparison...")

# Calculate metrics
def calc_metrics(returns_series):
    cum_ret = (1 + returns_series).cumprod()
    total_ret = cum_ret.iloc[-1] - 1
    n_years = len(returns_series) / 252
    annual_ret = (1 + total_ret) ** (1 / n_years) - 1
    sharpe = returns_series.mean() / returns_series.std() * np.sqrt(252)
    running_max = cum_ret.expanding().max()
    dd = (cum_ret - running_max) / running_max
    max_dd = dd.min()

    return {
        'annual_return': annual_ret,
        'sharpe': sharpe,
        'max_drawdown': max_dd
    }

metrics_data = []
for name, df in strategies.items():
    aligned_returns = df.loc[common_dates, 'return']
    metrics = calc_metrics(aligned_returns)
    metrics['strategy'] = name
    metrics_data.append(metrics)

metrics_df = pd.DataFrame(metrics_data)

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16, 5))

# Panel A: Sharpe Ratio
ax = ax1

sharpes = metrics_df['sharpe'].values
strategies_names = metrics_df['strategy'].values

bars = ax.barh(strategies_names, sharpes, color=colors[:len(sharpes)],
              alpha=0.7, edgecolor='black', linewidth=1.5)

ax.axvline(x=0, color='gray', linestyle='-', linewidth=1.5)
ax.set_xlabel('Sharpe Ratio', fontsize=11, fontweight='bold')
ax.set_title('A. Sharpe Ratio Comparison', fontsize=12, fontweight='bold', loc='left')
ax.grid(True, alpha=0.2, axis='x')

# Add value labels
for i, (bar, sharpe) in enumerate(zip(bars, sharpes)):
    ax.text(sharpe, i, f'  {sharpe:.3f}', va='center', fontsize=10, fontweight='bold')

# Panel B: Annual Return
ax = ax2

annual_rets = metrics_df['annual_return'].values

bars = ax.barh(strategies_names, annual_rets, color=colors[:len(annual_rets)],
              alpha=0.7, edgecolor='black', linewidth=1.5)

ax.axvline(x=0, color='gray', linestyle='-', linewidth=1.5)
ax.set_xlabel('Annual Return', fontsize=11, fontweight='bold')
ax.set_title('B. Annual Return Comparison', fontsize=12, fontweight='bold', loc='left')
ax.grid(True, alpha=0.2, axis='x')
ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.0%}'))

# Add value labels
for i, (bar, ret) in enumerate(zip(bars, annual_rets)):
    ax.text(ret, i, f'  {ret:.1%}', va='center', fontsize=10, fontweight='bold')

# Panel C: Max Drawdown
ax = ax3

max_dds = metrics_df['max_drawdown'].values

bars = ax.barh(strategies_names, max_dds, color=colors[:len(max_dds)],
              alpha=0.7, edgecolor='black', linewidth=1.5)

ax.axvline(x=0, color='gray', linestyle='-', linewidth=1.5)
ax.set_xlabel('Max Drawdown (Lower = Better)', fontsize=11, fontweight='bold')
ax.set_title('C. Max Drawdown Comparison', fontsize=12, fontweight='bold', loc='left')
ax.grid(True, alpha=0.2, axis='x')
ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.0%}'))

# Add value labels
for i, (bar, dd) in enumerate(zip(bars, max_dds)):
    ax.text(dd, i, f'  {dd:.1%}', va='center', fontsize=10, fontweight='bold')

plt.tight_layout()
save_figure('figure8_2_performance_comparison', fig, formats=['pdf', 'png'])
plt.close()

print("✅ Figure 8.2 saved")

# ============================================================================
# FIGURE 8.3: ENSEMBLE PORTFOLIO
# ============================================================================

print("\n📊 Creating Figure 8.3: Ensemble Portfolio...")

# Build ensemble
return_matrix = pd.DataFrame({
    name: df.loc[common_dates, 'return']
    for name, df in strategies.items()
})

ensemble_returns = return_matrix.mean(axis=1)

# Find best individual
best_strategy_name = metrics_df.sort_values('sharpe', ascending=False).iloc[0]['strategy']
best_returns = strategies[best_strategy_name].loc[common_dates, 'return']

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# Panel A: Cumulative returns comparison
ax = ax1

best_cum = (1 + best_returns).cumprod()
ensemble_cum = (1 + ensemble_returns).cumprod()

best_cum.plot(ax=ax, label=f'Best Individual ({best_strategy_name})',
             linewidth=2.5, color=COLORS['blue'], alpha=0.8, linestyle='--')

ensemble_cum.plot(ax=ax, label='Ensemble (Equal-Weight)',
                 linewidth=3.0, color='black', alpha=0.9)

ax.axhline(y=1.0, color='gray', linestyle=':', linewidth=1.5, alpha=0.5)
ax.set_ylabel('Cumulative Return', fontsize=11, fontweight='bold')
ax.set_xlabel('Date', fontsize=11, fontweight='bold')
ax.set_title('A. Ensemble vs Best Individual', fontsize=12, fontweight='bold', loc='left')
ax.legend(loc='best', fontsize=10)
ax.grid(True, alpha=0.2)

# Panel B: Performance metrics
ax = ax2

best_metrics = calc_metrics(best_returns)
ensemble_metrics = calc_metrics(ensemble_returns)

metrics_names = ['Sharpe', 'Annual Return', 'Max DD']
best_values = [best_metrics['sharpe'], best_metrics['annual_return'], best_metrics['max_drawdown']]
ensemble_values = [ensemble_metrics['sharpe'], ensemble_metrics['annual_return'], ensemble_metrics['max_drawdown']]

x = np.arange(len(metrics_names))
width = 0.35

bars1 = ax.bar(x - width/2, best_values, width, label=f'Best Individual',
              color=COLORS['blue'], alpha=0.7, edgecolor='black', linewidth=1.5)

bars2 = ax.bar(x + width/2, ensemble_values, width, label='Ensemble',
              color='black', alpha=0.7, edgecolor='black', linewidth=1.5)

ax.set_ylabel('Value', fontsize=11, fontweight='bold')
ax.set_title('B. Performance Metrics Comparison', fontsize=12, fontweight='bold', loc='left')
ax.set_xticks(x)
ax.set_xticklabels(metrics_names)
ax.legend(loc='best', fontsize=10)
ax.grid(True, alpha=0.2, axis='y')
ax.axhline(y=0, color='gray', linestyle='-', linewidth=1.5)

# Add value labels
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{height:.2f}', ha='center', va='bottom' if height > 0 else 'top',
               fontsize=9, fontweight='bold')

plt.tight_layout()
save_figure('figure8_3_ensemble_portfolio', fig, formats=['pdf', 'png'])
plt.close()

print("✅ Figure 8.3 saved")

# ============================================================================
# SUMMARY
# ============================================================================

print("\n" + "=" * 80)
print("VISUALIZATION COMPLETE")
print("=" * 80)

print(f"\n📊 Figures saved to: {FIG_DIR}")

print("\nFiles created:")
print("  - figure8_1_strategy_equity_curves.pdf/.png")
print("  - figure8_2_performance_comparison.pdf/.png")
print("  - figure8_3_ensemble_portfolio.pdf/.png")

print("\nThese figures are ready for:")
print("  ✅ Thesis Section 8 (Strategy Variants)")
print("  ✅ Journal submission (300 DPI, vector PDF)")
print("  ✅ Presentations")

print("\nNext: Copy Section 8 text from SECTION_8_TEXT.md into your thesis")
