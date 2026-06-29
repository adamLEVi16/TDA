"""
Phase 3: Momentum + TDA Hybrid Strategy
========================================

Addresses the "mean reversion in trending markets" failure mode.

Original problem: Mean-reversion strategy (long losers, short winners) fails
in 2022-2024 bull market where trends persist.

Solution: Use topology to detect regime, then apply appropriate strategy:
- High H₁ (stressed) → Mean reversion (overreactions correct)
- Low H₁ (calm) → Momentum (trends persist)

This is the OPPOSITE logic from the sector pairs trading strategy.

Author: Adam Levine
Date: January 2026
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add parent for imports
sys.path.append(str(Path(__file__).parent.parent))

print("=" * 80)
print("PHASE 3: MOMENTUM + TDA HYBRID STRATEGY")
print("=" * 80)

# Configuration
DATA_DIR = Path(__file__).parent.parent / 'data'
TRAIN_END = '2022-12-31'
TEST_START = '2023-01-01'
TRANSACTION_COST = 0.0005  # 5 bps
LOOKBACK_MOMENTUM = 20  # 20-day momentum
REBALANCE_FREQ = 5  # Every 5 days

# ============================================================================
# LOAD DATA
# ============================================================================

print("\n📂 Loading data...")

# Load sector returns (use Technology sector as example)
returns_file = DATA_DIR / 'sector_technology_returns.csv'
topology_file = DATA_DIR / 'sector_technology_topology.csv'

if not returns_file.exists() or not topology_file.exists():
    print("⚠️  Technology sector data not found. Run Phase 2 first!")
    print(f"   Looking for: {returns_file}")
    sys.exit(1)

returns_df = pd.read_csv(returns_file, index_col=0, parse_dates=True)
topology_df = pd.read_csv(topology_file, index_col=0, parse_dates=True)

print(f"✅ Returns: {len(returns_df)} days × {returns_df.shape[1]} stocks")
print(f"✅ Topology: {len(topology_df)} snapshots")

# ============================================================================
# MOMENTUM + TDA HYBRID STRATEGY
# ============================================================================

print("\n🎯 Running Momentum + TDA Hybrid Strategy...")

# Split train/test
train_topology = topology_df[topology_df.index <= TRAIN_END]
test_topology = topology_df[topology_df.index > TEST_START]

# Determine H₁ threshold on training data
threshold = train_topology['h1_count'].quantile(0.75)

print(f"\nTraining period: {train_topology.index[0].strftime('%Y-%m-%d')} to "
      f"{train_topology.index[-1].strftime('%Y-%m-%d')}")
print(f"Testing period:  {test_topology.index[0].strftime('%Y-%m-%d')} to "
      f"{test_topology.index[-1].strftime('%Y-%m-%d')}")
print(f"H₁ threshold (75th percentile): {threshold:.2f}")

# Generate signals
strategy_returns = []

for i, date in enumerate(test_topology.index):
    if date not in returns_df.index:
        continue

    # Get topology regime
    h1_loops = test_topology.loc[date, 'h1_count']

    # Compute 20-day momentum for all stocks
    date_idx = returns_df.index.get_loc(date)

    if date_idx < LOOKBACK_MOMENTUM:
        continue  # Skip early period

    momentum_window = returns_df.iloc[date_idx - LOOKBACK_MOMENTUM:date_idx]
    momentum_scores = momentum_window.sum()  # Total return over 20 days

    # Select top 5 and bottom 5 by momentum
    top_momentum = momentum_scores.nlargest(5).index
    bottom_momentum = momentum_scores.nsmallest(5).index

    # Strategy logic (OPPOSITE of mean reversion):
    # High H₁ (stressed) → Mean reversion → Long losers, short winners
    # Low H₁ (calm) → Momentum → Long winners, short losers

    if h1_loops > threshold:
        # Stressed regime: Mean reversion
        long_stocks = bottom_momentum  # Losers will bounce
        short_stocks = top_momentum    # Winners will correct
        regime = 'mean_reversion'
    else:
        # Calm regime: Momentum
        long_stocks = top_momentum     # Winners continue
        short_stocks = bottom_momentum  # Losers continue falling
        regime = 'momentum'

    # Compute strategy return
    long_return = returns_df.loc[date, long_stocks].mean()
    short_return = returns_df.loc[date, short_stocks].mean()

    strategy_return = (long_return - short_return) / 2  # 50% long, 50% short

    # Transaction costs on rebalances
    if i % REBALANCE_FREQ == 0:
        n_trades = 10  # 5 long + 5 short
        cost = n_trades * TRANSACTION_COST
        strategy_return -= cost

    strategy_returns.append({
        'date': date,
        'return': strategy_return,
        'h1_loops': h1_loops,
        'regime': regime,
        'long_ret': long_return,
        'short_ret': short_return
    })

strategy_df = pd.DataFrame(strategy_returns).set_index('date')

# ============================================================================
# PERFORMANCE METRICS
# ============================================================================

print("\n📊 Performance Metrics:")

cum_returns = (1 + strategy_df['return']).cumprod()
total_return = cum_returns.iloc[-1] - 1

n_days = len(strategy_df)
n_years = n_days / 252
annual_return = (1 + total_return) ** (1 / n_years) - 1

returns_series = strategy_df['return']
sharpe = returns_series.mean() / returns_series.std() * np.sqrt(252) if returns_series.std() > 0 else 0

running_max = cum_returns.expanding().max()
drawdown = (cum_returns - running_max) / running_max
max_drawdown = drawdown.min()

win_rate = (returns_series > 0).mean()

# Regime breakdown
regime_counts = strategy_df['regime'].value_counts()
mean_rev_days = regime_counts.get('mean_reversion', 0)
momentum_days = regime_counts.get('momentum', 0)

print(f"\nTotal Return:      {total_return * 100:>8.2f}%")
print(f"Annual Return:     {annual_return * 100:>8.2f}%")
print(f"Sharpe Ratio:      {sharpe:>8.3f}")
print(f"Max Drawdown:      {max_drawdown * 100:>8.2f}%")
print(f"Win Rate:          {win_rate * 100:>8.1f}%")
print(f"\nRegime Breakdown:")
print(f"  Mean Reversion days: {mean_rev_days} ({mean_rev_days/len(strategy_df)*100:.1f}%)")
print(f"  Momentum days:       {momentum_days} ({momentum_days/len(strategy_df)*100:.1f}%)")

# Compare to baseline (momentum-only)
momentum_only_returns = []

for date in test_topology.index:
    if date not in returns_df.index:
        continue

    date_idx = returns_df.index.get_loc(date)
    if date_idx < LOOKBACK_MOMENTUM:
        continue

    momentum_window = returns_df.iloc[date_idx - LOOKBACK_MOMENTUM:date_idx]
    momentum_scores = momentum_window.sum()

    top_momentum = momentum_scores.nlargest(5).index
    bottom_momentum = momentum_scores.nsmallest(5).index

    # Pure momentum: Always long winners, short losers
    long_return = returns_df.loc[date, top_momentum].mean()
    short_return = returns_df.loc[date, bottom_momentum].mean()

    mom_return = (long_return - short_return) / 2
    momentum_only_returns.append(mom_return)

mom_only_df = pd.DataFrame({'return': momentum_only_returns})
mom_only_sharpe = mom_only_df['return'].mean() / mom_only_df['return'].std() * np.sqrt(252)

print(f"\n📊 Comparison:")
print(f"  Hybrid (Momentum + TDA) Sharpe:  {sharpe:>6.3f}")
print(f"  Momentum-Only Sharpe:            {mom_only_sharpe:>6.3f}")
print(f"  Improvement:                     {(sharpe - mom_only_sharpe):>+6.3f}")

if sharpe > mom_only_sharpe:
    print("\n✅ Hybrid strategy BEATS momentum-only!")
    print("   TDA regime detection adds value")
else:
    print("\n❌ Hybrid strategy underperforms momentum-only")
    print("   TDA regime detection not helpful for momentum")

# ============================================================================
# SAVE RESULTS
# ============================================================================

output_file = DATA_DIR / 'phase3_momentum_tda_hybrid.csv'
strategy_df.to_csv(output_file)

print(f"\n💾 Results saved: {output_file}")

# ============================================================================
# SUMMARY
# ============================================================================

print("\n" + "=" * 80)
print("MOMENTUM + TDA HYBRID COMPLETE")
print("=" * 80)

print("\n📊 Key Findings:")
print(f"  - Hybrid Sharpe: {sharpe:.3f}")
print(f"  - Uses mean reversion {mean_rev_days/(mean_rev_days+momentum_days)*100:.0f}% of time")
print(f"  - Uses momentum {momentum_days/(mean_rev_days+momentum_days)*100:.0f}% of time")

if sharpe > 0:
    print(f"\n✅ POSITIVE Sharpe! Addresses trending market failure mode.")
else:
    print(f"\n⚠️  Still negative, but may be better than pure mean-reversion")

print("\nNext: Run 02_scale_consistent_architecture.py")
