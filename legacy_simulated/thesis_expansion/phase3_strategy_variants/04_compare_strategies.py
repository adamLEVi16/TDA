"""
Phase 3: Compare All Strategy Variants
=======================================

Compares performance across all strategy variants:
1. Baseline (from Phase 2 - mean reversion with static threshold)
2. Momentum + TDA Hybrid
3. Scale-Consistent Architecture
4. Adaptive Threshold

Determines which variant(s) best address the original failure modes.

Author: Adam Levine
Date: January 2026
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys

print("=" * 80)
print("PHASE 3: STRATEGY VARIANT COMPARISON")
print("=" * 80)

DATA_DIR = Path(__file__).parent.parent / 'data'

# ============================================================================
# LOAD ALL STRATEGIES
# ============================================================================

print("\n📂 Loading strategy results...")

strategies = {}

# Baseline (Phase 2 sector strategy)
baseline_file = DATA_DIR / 'sector_technology_strategy.csv'
if baseline_file.exists():
    strategies['Baseline (Mean Rev)'] = pd.read_csv(baseline_file, index_col=0, parse_dates=True)
    print("  ✅ Baseline (Mean Reversion)")
else:
    print("  ⚠️  Baseline not found")

# Momentum + TDA Hybrid
hybrid_file = DATA_DIR / 'phase3_momentum_tda_hybrid.csv'
if hybrid_file.exists():
    strategies['Momentum + TDA'] = pd.read_csv(hybrid_file, index_col=0, parse_dates=True)
    print("  ✅ Momentum + TDA Hybrid")
else:
    print("  ⚠️  Momentum + TDA not found")

# Scale-Consistent
scale_file = DATA_DIR / 'phase3_scale_consistent.csv'
if scale_file.exists():
    strategies['Scale-Consistent'] = pd.read_csv(scale_file, index_col=0, parse_dates=True)
    print("  ✅ Scale-Consistent Architecture")
else:
    print("  ⚠️  Scale-Consistent not found")

# Adaptive Threshold
adaptive_file = DATA_DIR / 'phase3_adaptive_threshold.csv'
if adaptive_file.exists():
    strategies['Adaptive Threshold'] = pd.read_csv(adaptive_file, index_col=0, parse_dates=True)
    print("  ✅ Adaptive Threshold")
else:
    print("  ⚠️  Adaptive Threshold not found")

if len(strategies) == 0:
    print("\n❌ No strategy results found. Run Phase 2 and Phase 3 scripts first!")
    sys.exit(1)

print(f"\nLoaded {len(strategies)} strategies")

# ============================================================================
# ALIGN DATES
# ============================================================================

print("\n📊 Aligning dates across strategies...")

# Find common dates
common_dates = None

for name, df in strategies.items():
    if common_dates is None:
        common_dates = set(df.index)
    else:
        common_dates = common_dates.intersection(set(df.index))

common_dates = sorted(list(common_dates))

print(f"  Common trading days: {len(common_dates)}")

# ============================================================================
# COMPUTE PERFORMANCE METRICS
# ============================================================================

print("\n📊 Computing performance metrics...")

def calc_metrics(returns_series):
    """Calculate performance metrics."""
    cum_ret = (1 + returns_series).cumprod()
    total_ret = cum_ret.iloc[-1] - 1

    n_years = len(returns_series) / 252
    annual_ret = (1 + total_ret) ** (1 / n_years) - 1

    sharpe = returns_series.mean() / returns_series.std() * np.sqrt(252) if returns_series.std() > 0 else 0

    running_max = cum_ret.expanding().max()
    dd = (cum_ret - running_max) / running_max
    max_dd = dd.min()

    win_rate = (returns_series > 0).mean()

    calmar = annual_ret / abs(max_dd) if max_dd != 0 else 0

    return {
        'total_return': total_ret,
        'annual_return': annual_ret,
        'sharpe': sharpe,
        'max_drawdown': max_dd,
        'win_rate': win_rate,
        'calmar': calmar,
        'volatility': returns_series.std() * np.sqrt(252)
    }

# Compute for each strategy
results = []

for name, df in strategies.items():
    aligned_returns = df.loc[common_dates, 'return']

    metrics = calc_metrics(aligned_returns)
    metrics['strategy'] = name

    results.append(metrics)

results_df = pd.DataFrame(results)

# ============================================================================
# COMPARISON TABLE
# ============================================================================

print("\n" + "=" * 80)
print("STRATEGY PERFORMANCE COMPARISON")
print("=" * 80)

# Sort by Sharpe ratio
results_df = results_df.sort_values('sharpe', ascending=False)

print(f"\n{'Strategy':<25} {'Sharpe':<8} {'Annual Ret':<12} {'Max DD':<10} {'Win Rate':<10}")
print("-" * 80)

for _, row in results_df.iterrows():
    print(f"{row['strategy']:<25} {row['sharpe']:>7.3f}  {row['annual_return']:>10.1%}  "
          f"{row['max_drawdown']:>9.1%}  {row['win_rate']:>9.1%}")

# ============================================================================
# BEST STRATEGY
# ============================================================================

print("\n" + "=" * 80)
print("BEST PERFORMING STRATEGY")
print("=" * 80)

best_strategy = results_df.iloc[0]

print(f"\n🏆 Winner: {best_strategy['strategy']}")
print(f"   Sharpe Ratio:    {best_strategy['sharpe']:.3f}")
print(f"   Annual Return:   {best_strategy['annual_return']:.1%}")
print(f"   Max Drawdown:    {best_strategy['max_drawdown']:.1%}")
print(f"   Win Rate:        {best_strategy['win_rate']:.1%}")

# Improvement over baseline
if 'Baseline (Mean Rev)' in results_df['strategy'].values:
    baseline_sharpe = results_df[results_df['strategy'] == 'Baseline (Mean Rev)']['sharpe'].values[0]
    best_sharpe = best_strategy['sharpe']

    improvement_pct = (best_sharpe - baseline_sharpe) / abs(baseline_sharpe) * 100

    print(f"\n📊 Improvement over Baseline:")
    print(f"   Baseline Sharpe: {baseline_sharpe:.3f}")
    print(f"   Best Sharpe:     {best_sharpe:.3f}")
    print(f"   Improvement:     {improvement_pct:+.1f}%")

# ============================================================================
# CORRELATION ANALYSIS
# ============================================================================

print("\n" + "=" * 80)
print("STRATEGY RETURN CORRELATIONS")
print("=" * 80)

# Build return matrix
return_matrix = pd.DataFrame({
    name: df.loc[common_dates, 'return']
    for name, df in strategies.items()
})

corr_matrix = return_matrix.corr()

print("\n" + corr_matrix.to_string())

avg_corr = corr_matrix.values[np.triu_indices_from(corr_matrix.values, k=1)].mean()

print(f"\nAverage pairwise correlation: {avg_corr:.3f}")

if avg_corr < 0.5:
    print("✅ Low correlation → Good diversification potential")
    print("   Could combine strategies in ensemble portfolio")
elif avg_corr < 0.7:
    print("🟡 Moderate correlation → Some diversification benefit")
else:
    print("❌ High correlation → Limited diversification benefit")

# ============================================================================
# ENSEMBLE PORTFOLIO
# ============================================================================

print("\n" + "=" * 80)
print("ENSEMBLE PORTFOLIO (EQUAL-WEIGHT)")
print("=" * 80)

# Equal-weight combination
ensemble_returns = return_matrix.mean(axis=1)

ensemble_metrics = calc_metrics(ensemble_returns)

print(f"\n📊 Ensemble Performance:")
print(f"   Sharpe Ratio:    {ensemble_metrics['sharpe']:.3f}")
print(f"   Annual Return:   {ensemble_metrics['annual_return']:.1%}")
print(f"   Max Drawdown:    {ensemble_metrics['max_drawdown']:.1%}")
print(f"   Win Rate:        {ensemble_metrics['win_rate']:.1%}")

# Compare to best individual
if ensemble_metrics['sharpe'] > best_sharpe:
    improvement = (ensemble_metrics['sharpe'] - best_sharpe) / best_sharpe * 100
    print(f"\n✅ Ensemble BEATS best individual by {improvement:.1f}%!")
    print("   Diversification provides value")
else:
    print(f"\n🟡 Ensemble underperforms best individual")
    print("   Stick with single best strategy")

# ============================================================================
# SAVE RESULTS
# ============================================================================

summary_file = DATA_DIR / 'phase3_strategy_comparison.csv'
results_df.to_csv(summary_file, index=False)

ensemble_file = DATA_DIR / 'phase3_ensemble_portfolio.csv'
pd.DataFrame({'return': ensemble_returns}).to_csv(ensemble_file)

print(f"\n💾 Comparison saved: {summary_file}")
print(f"💾 Ensemble saved: {ensemble_file}")

# ============================================================================
# RECOMMENDATIONS
# ============================================================================

print("\n" + "=" * 80)
print("RECOMMENDATIONS")
print("=" * 80)

print("\n📊 Based on performance:")

if best_strategy['sharpe'] > 0.5:
    print(f"   ✅ {best_strategy['strategy']} achieves good risk-adjusted returns")
    print(f"   ✅ Sharpe {best_strategy['sharpe']:.3f} > 0.5 threshold")
    print("   → Use this variant for live trading")

elif best_strategy['sharpe'] > 0:
    print(f"   🟡 {best_strategy['strategy']} achieves positive Sharpe")
    print(f"   🟡 Sharpe {best_strategy['sharpe']:.3f} is modest")
    print("   → Needs further refinement before live trading")

else:
    print(f"   ❌ All variants have negative Sharpe")
    print("   ❌ TDA-based trading not viable for this sector")
    print("   → Try different sectors or abandon approach")

print("\n📊 Failure mode analysis:")

failure_modes = {
    'Baseline (Mean Rev)': 'Mean reversion in trending markets',
    'Momentum + TDA': 'Addresses trending markets',
    'Scale-Consistent': 'Addresses scale mismatch',
    'Adaptive Threshold': 'Addresses static threshold'
}

for name, mode in failure_modes.items():
    if name in results_df['strategy'].values:
        sharpe = results_df[results_df['strategy'] == name]['sharpe'].values[0]
        status = "✅" if sharpe > best_sharpe - 0.1 else "❌"
        print(f"   {status} {name}: {mode}")

print("\n" + "=" * 80)
print("COMPARISON COMPLETE")
print("=" * 80)

print("\nNext: Run 05_visualize_strategies.py to create publication figures")
