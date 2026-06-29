"""
Phase 2: Sector Comparison & Multi-Sector Portfolio
====================================================

Compares performance across sectors and builds multi-sector portfolio.

Multi-sector portfolio combines top-performing sectors with equal weights
for diversification benefits.

Author: Adam Levine
Date: January 2026
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sector_config import SECTORS
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("PHASE 2: SECTOR COMPARISON & MULTI-SECTOR PORTFOLIO")
print("=" * 80)

DATA_DIR = Path(__file__).parent.parent / 'data'

# ============================================================================
# LOAD SECTOR RESULTS
# ============================================================================

print(f"\n📂 Loading sector strategy results...")

sector_strategies = {}

for sector_name in SECTORS.keys():
    strategy_file = DATA_DIR / f'sector_{sector_name.lower()}_strategy.csv'

    if strategy_file.exists():
        strategy_df = pd.read_csv(strategy_file, index_col=0, parse_dates=True)
        sector_strategies[sector_name] = strategy_df
        print(f"  ✅ {sector_name}: {len(strategy_df)} days")
    else:
        print(f"  ⚠️  {sector_name}: No strategy file found")

if not sector_strategies:
    print("\n❌ No sector strategies found. Run 03_sector_pairs_trading.py first!")
    exit(1)

# ============================================================================
# SECTOR PERFORMANCE METRICS
# ============================================================================

print("\n" + "=" * 80)
print("SECTOR PERFORMANCE ANALYSIS")
print("=" * 80)

def calculate_metrics(returns_series):
    """Calculate performance metrics."""
    cum_returns = (1 + returns_series).cumprod()
    total_return = cum_returns.iloc[-1] - 1

    n_days = len(returns_series)
    n_years = n_days / 252
    annual_return = (1 + total_return) ** (1 / n_years) - 1

    sharpe = returns_series.mean() / returns_series.std() * np.sqrt(252) if returns_series.std() > 0 else 0

    running_max = cum_returns.expanding().max()
    drawdown = (cum_returns - running_max) / running_max
    max_drawdown = drawdown.min()

    return {
        'total_return': total_return,
        'annual_return': annual_return,
        'sharpe': sharpe,
        'max_drawdown': max_drawdown,
        'volatility': returns_series.std() * np.sqrt(252),
        'win_rate': (returns_series > 0).mean(),
    }

sector_metrics = {}

for sector_name, strategy_df in sector_strategies.items():
    metrics = calculate_metrics(strategy_df['return'])
    sector_metrics[sector_name] = metrics

# Create comparison table
comparison_data = []

for sector_name, metrics in sector_metrics.items():
    comparison_data.append({
        'Sector': sector_name,
        'Sharpe': metrics['sharpe'],
        'Annual Return': metrics['annual_return'],
        'Volatility': metrics['volatility'],
        'Max DD': metrics['max_drawdown'],
        'Win Rate': metrics['win_rate'],
    })

comparison_df = pd.DataFrame(comparison_data).sort_values('Sharpe', ascending=False)

print("\n📊 Sector Performance Ranked by Sharpe:")
print(f"\n{'Sector':<15} {'Sharpe':<8} {'Annual Return':<15} {'Volatility':<12} {'Max DD':<10} {'Win Rate':<10}")
print("-" * 80)

for _, row in comparison_df.iterrows():
    print(f"{row['Sector']:<15} {row['Sharpe']:>7.3f}  {row['Annual Return']:>13.1%}  "
          f"{row['Volatility']:>11.1%}  {row['Max DD']:>9.1%}  {row['Win Rate']:>9.1%}")

# Save comparison
comparison_file = DATA_DIR / 'sector_performance_comparison.csv'
comparison_df.to_csv(comparison_file, index=False)

print(f"\n💾 Comparison saved: {comparison_file}")

# ============================================================================
# IDENTIFY TOP SECTORS
# ============================================================================

print("\n" + "=" * 80)
print("TOP PERFORMING SECTORS")
print("=" * 80)

# Select sectors with positive Sharpe
positive_sectors = comparison_df[comparison_df['Sharpe'] > 0]['Sector'].tolist()

if positive_sectors:
    print(f"\n✅ {len(positive_sectors)} sector(s) with positive Sharpe:")
    for sector in positive_sectors:
        sharpe = comparison_df[comparison_df['Sector'] == sector]['Sharpe'].values[0]
        ann_ret = comparison_df[comparison_df['Sector'] == sector]['Annual Return'].values[0]
        print(f"   {sector:12s}: Sharpe = {sharpe:.3f}, Annual Return = {ann_ret:>6.1%}")

    # Select top 3 for portfolio
    top_n = min(3, len(positive_sectors))
    top_sectors = positive_sectors[:top_n]

    print(f"\n🎯 Selected top {top_n} for multi-sector portfolio:")
    for sector in top_sectors:
        print(f"   - {sector}")

else:
    print("\n⚠️  No sectors with positive Sharpe")
    print("   Using all sectors for multi-sector portfolio (diversification may help)")
    top_sectors = list(sector_strategies.keys())[:3]

# ============================================================================
# BUILD MULTI-SECTOR PORTFOLIO
# ============================================================================

print("\n" + "=" * 80)
print("MULTI-SECTOR PORTFOLIO CONSTRUCTION")
print("=" * 80)

print(f"\nCombining {len(top_sectors)} sectors with equal weights:")

# Align dates across selected sectors
common_dates = None

for sector in top_sectors:
    sector_dates = sector_strategies[sector].index

    if common_dates is None:
        common_dates = set(sector_dates)
    else:
        common_dates = common_dates.intersection(set(sector_dates))

common_dates = sorted(list(common_dates))

print(f"   Common trading days: {len(common_dates)}")

# Construct multi-sector portfolio
portfolio_returns = []

for date in common_dates:
    # Equal-weight average of sector returns
    sector_rets = []

    for sector in top_sectors:
        ret = sector_strategies[sector].loc[date, 'return']
        sector_rets.append(ret)

    portfolio_ret = np.mean(sector_rets)

    portfolio_returns.append({
        'date': date,
        'return': portfolio_ret,
    })

portfolio_df = pd.DataFrame(portfolio_returns).set_index('date')

# Calculate portfolio metrics
portfolio_metrics = calculate_metrics(portfolio_df['return'])

print("\n📊 Multi-Sector Portfolio Performance:")
print(f"   Total Return:    {portfolio_metrics['total_return']:>8.2%}")
print(f"   Annual Return:   {portfolio_metrics['annual_return']:>8.2%}")
print(f"   Sharpe Ratio:    {portfolio_metrics['sharpe']:>8.3f}")
print(f"   Volatility:      {portfolio_metrics['volatility']:>8.2%}")
print(f"   Max Drawdown:    {portfolio_metrics['max_drawdown']:>8.2%}")
print(f"   Win Rate:        {portfolio_metrics['win_rate']:>8.1%}")

# Save portfolio returns
portfolio_file = DATA_DIR / 'multi_sector_portfolio.csv'
portfolio_df.to_csv(portfolio_file)

print(f"\n💾 Portfolio saved: {portfolio_file}")

# ============================================================================
# COMPARISON: INDIVIDUAL VS PORTFOLIO
# ============================================================================

print("\n" + "=" * 80)
print("INDIVIDUAL SECTORS VS MULTI-SECTOR PORTFOLIO")
print("=" * 80)

print(f"\n{'Strategy':<20} {'Sharpe':<10} {'Annual Return':<15} {'Max DD':<10}")
print("-" * 60)

# Individual sectors
for sector in top_sectors:
    metrics = sector_metrics[sector]
    print(f"{sector:<20} {metrics['sharpe']:>9.3f}  {metrics['annual_return']:>13.1%}  "
          f"{metrics['max_drawdown']:>9.1%}")

# Portfolio
print("-" * 60)
print(f"{'Multi-Sector Portfolio':<20} {portfolio_metrics['sharpe']:>9.3f}  "
      f"{portfolio_metrics['annual_return']:>13.1%}  {portfolio_metrics['max_drawdown']:>9.1%}")

# ============================================================================
# DIVERSIFICATION BENEFIT
# ============================================================================

print("\n" + "=" * 80)
print("DIVERSIFICATION ANALYSIS")
print("=" * 80)

# Average Sharpe of individual sectors
avg_individual_sharpe = np.mean([sector_metrics[s]['sharpe'] for s in top_sectors])

# Portfolio Sharpe
portfolio_sharpe = portfolio_metrics['sharpe']

# Diversification benefit
sharpe_improvement = portfolio_sharpe - avg_individual_sharpe

print(f"\nAverage Individual Sharpe: {avg_individual_sharpe:.3f}")
print(f"Multi-Sector Sharpe:       {portfolio_sharpe:.3f}")
print(f"Improvement:               {sharpe_improvement:+.3f}")

if sharpe_improvement > 0:
    print("\n✅ Diversification provides benefit!")
    print("   Multi-sector portfolio has higher risk-adjusted returns.")
else:
    print("\n⚠️  No diversification benefit observed.")
    print("   Sectors may be too correlated.")

# Correlation between sector strategies
if len(top_sectors) >= 2:
    print("\n📊 Correlation between sector strategies:")

    sector_returns_matrix = pd.DataFrame({
        sector: sector_strategies[sector].loc[common_dates, 'return']
        for sector in top_sectors
    })

    corr_matrix = sector_returns_matrix.corr()

    print(f"\n{corr_matrix.to_string()}")

    avg_corr = corr_matrix.values[np.triu_indices_from(corr_matrix.values, k=1)].mean()
    print(f"\nAverage correlation: {avg_corr:.3f}")

    if avg_corr < 0.5:
        print("✅ Low correlation → Good diversification potential")
    elif avg_corr < 0.7:
        print("🟡 Moderate correlation → Some diversification benefit")
    else:
        print("❌ High correlation → Limited diversification benefit")

# ============================================================================
# KEY FINDINGS
# ============================================================================

print("\n" + "=" * 80)
print("KEY FINDINGS")
print("=" * 80)

if portfolio_metrics['sharpe'] > 0:
    print("\n✅ SUCCESS: Sector-specific topology produces POSITIVE Sharpe!")
    print(f"   Multi-sector portfolio Sharpe: {portfolio_metrics['sharpe']:.3f}")
    print(f"   Annual return: {portfolio_metrics['annual_return']:.1%}")

    print("\n💡 Why it works:")
    print("   1. Sector-homogeneous correlations → Cleaner topology")
    print("   2. Pairs trading within sector → Better mean reversion")
    print("   3. Multi-sector diversification → Reduces sector-specific risk")

else:
    print(f"\n❌ Multi-sector portfolio still negative (Sharpe: {portfolio_metrics['sharpe']:.3f})")
    print("   But likely BETTER than original cross-sector strategy")
    print("   Check original paper results for comparison")

# ============================================================================
# SUMMARY
# ============================================================================

print("\n" + "=" * 80)
print("ANALYSIS COMPLETE")
print("=" * 80)

print(f"\n📊 Analyzed {len(sector_strategies)} sector strategies")
print(f"📁 Results saved to: {DATA_DIR}")

print("\nFiles created:")
print(f"  - sector_performance_comparison.csv")
print(f"  - multi_sector_portfolio.csv")

print("\nNext step:")
print("  Run 05_visualize_sectors.py to create publication-quality figures")
