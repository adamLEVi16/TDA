"""
Phase 3: Scale-Consistent Architecture
=======================================

Addresses the "scale mismatch" failure mode.

Original problem: Daily trading signals filtered by monthly topology
- Graph Laplacian diffusion uses 60-day windows (monthly scale)
- Trading signals generated daily
- Mismatch: Local daily fluctuations vs global monthly structure

Solution: Generate topology at SAME scale as trading signals
- Use 5-day topology (weekly structure)
- Generate 5-day trading signals
- Both aligned at weekly timescale

This tests if scale consistency improves performance.

Author: Adam Levine
Date: January 2026
"""

import pandas as pd
import numpy as np
from pathlib import Path
from ripser import ripser
import sys
import warnings
warnings.filterwarnings('ignore')

# Add parent for imports
sys.path.append(str(Path(__file__).parent.parent))

print("=" * 80)
print("PHASE 3: SCALE-CONSISTENT ARCHITECTURE")
print("=" * 80)

# Configuration
DATA_DIR = Path(__file__).parent.parent / 'data'
TRAIN_END = '2022-12-31'
TEST_START = '2023-01-01'
TRANSACTION_COST = 0.0005

# Scale-consistent parameters
SHORT_WINDOW = 5  # 5-day (weekly) topology
REBALANCE_FREQ = 5  # Rebalance weekly to match topology scale

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def compute_topology_short_window(returns_window):
    """
    Compute topology on SHORT window (5 days instead of 60).

    This creates weekly-scale topology aligned with weekly trading signals.
    """
    try:
        corr = returns_window.corr()
        dist = np.sqrt(2 * (1 - corr.values))

        result = ripser(dist, maxdim=1, distance_matrix=True, thresh=0.5)

        h1_dgm = result['dgms'][1]
        h1_count = len(h1_dgm)
        h1_persistence = (h1_dgm[:, 1] - h1_dgm[:, 0]).sum() if len(h1_dgm) > 0 else 0

        return {
            'h1_count': h1_count,
            'h1_persistence': h1_persistence
        }
    except:
        return None


# ============================================================================
# LOAD DATA
# ============================================================================

print("\n📂 Loading data...")

# Use Technology sector as example
returns_file = DATA_DIR / 'sector_technology_returns.csv'

if not returns_file.exists():
    print("⚠️  Technology sector data not found. Run Phase 2 first!")
    sys.exit(1)

returns_df = pd.read_csv(returns_file, index_col=0, parse_dates=True)

print(f"✅ Returns: {len(returns_df)} days × {returns_df.shape[1]} stocks")

# ============================================================================
# COMPUTE SHORT-WINDOW TOPOLOGY
# ============================================================================

print(f"\n📊 Computing {SHORT_WINDOW}-day topology...")

topology_features = []

for i in range(SHORT_WINDOW, len(returns_df)):
    window = returns_df.iloc[i - SHORT_WINDOW:i]

    topo = compute_topology_short_window(window)

    if topo:
        topo['date'] = returns_df.index[i]
        topology_features.append(topo)

    if (i - SHORT_WINDOW) % 100 == 0:
        pct = (i - SHORT_WINDOW) / (len(returns_df) - SHORT_WINDOW) * 100
        print(f"  Progress: {pct:.1f}%", end='\r')

print(f"  Progress: 100.0%")

topology_df = pd.DataFrame(topology_features).set_index('date')

print(f"✅ Computed {len(topology_df)} topology snapshots")

# ============================================================================
# SCALE-CONSISTENT STRATEGY
# ============================================================================

print("\n🎯 Running Scale-Consistent Strategy...")

# Split train/test
train_topology = topology_df[topology_df.index <= TRAIN_END]
test_topology = topology_df[topology_df.index > TEST_START]

# Determine threshold on training data
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

    # Get 5-day topology
    h1_loops = test_topology.loc[date, 'h1_count']

    # Compute 5-day returns (SAME scale as topology)
    date_idx = returns_df.index.get_loc(date)

    if date_idx < SHORT_WINDOW:
        continue

    past_5day_returns = returns_df.iloc[date_idx - SHORT_WINDOW:date_idx].sum()

    # Select top 5 and bottom 5
    top_performers = past_5day_returns.nlargest(5).index
    bottom_performers = past_5day_returns.nsmallest(5).index

    # Strategy: Mean reversion when high H₁
    if h1_loops > threshold:
        # High loops → stressed → mean reversion
        long_stocks = bottom_performers
        short_stocks = top_performers
    else:
        # Low loops → calm → no trade (flat)
        long_stocks = []
        short_stocks = []

    # Compute return
    if len(long_stocks) > 0:
        long_return = returns_df.loc[date, long_stocks].mean()
        short_return = returns_df.loc[date, short_stocks].mean()
        strategy_return = (long_return - short_return) / 2
    else:
        strategy_return = 0  # No position

    # Transaction costs on rebalances (every 5 days)
    if i % REBALANCE_FREQ == 0 and len(long_stocks) > 0:
        n_trades = 10
        cost = n_trades * TRANSACTION_COST
        strategy_return -= cost

    strategy_returns.append({
        'date': date,
        'return': strategy_return,
        'h1_loops': h1_loops,
        'position': 1 if h1_loops > threshold else 0
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

# Position analysis
position_pct = strategy_df['position'].mean()

print(f"\nTotal Return:      {total_return * 100:>8.2f}%")
print(f"Annual Return:     {annual_return * 100:>8.2f}%")
print(f"Sharpe Ratio:      {sharpe:>8.3f}")
print(f"Max Drawdown:      {max_drawdown * 100:>8.2f}%")
print(f"Win Rate:          {win_rate * 100:>8.1f}%")
print(f"Position %:        {position_pct * 100:>8.1f}%")

# Compare to 60-day topology baseline
baseline_file = DATA_DIR / 'sector_technology_strategy.csv'

if baseline_file.exists():
    baseline_df = pd.read_csv(baseline_file, index_col=0, parse_dates=True)

    # Align dates
    common_dates = strategy_df.index.intersection(baseline_df.index)

    if len(common_dates) > 10:
        baseline_returns = baseline_df.loc[common_dates, 'return']
        baseline_sharpe = baseline_returns.mean() / baseline_returns.std() * np.sqrt(252)

        print(f"\n📊 Comparison to 60-Day Topology Baseline:")
        print(f"  Scale-Consistent (5-day) Sharpe:  {sharpe:>6.3f}")
        print(f"  Baseline (60-day) Sharpe:         {baseline_sharpe:>6.3f}")
        print(f"  Improvement:                      {(sharpe - baseline_sharpe):>+6.3f}")

        if sharpe > baseline_sharpe:
            print("\n✅ Scale-consistent architecture IMPROVES performance!")
        else:
            print("\n❌ Scale-consistent architecture does not improve performance")
else:
    print("\n⚠️  No baseline comparison (run Phase 2 first)")

# ============================================================================
# TOPOLOGY STABILITY ANALYSIS
# ============================================================================

print("\n📊 Topology Stability Comparison:")

# 5-day topology CV
short_cv = topology_df['h1_count'].std() / topology_df['h1_count'].mean()

# 60-day topology CV (if available)
long_topology_file = DATA_DIR / 'sector_technology_topology.csv'

if long_topology_file.exists():
    long_topology_df = pd.read_csv(long_topology_file, index_col=0, parse_dates=True)
    long_cv = long_topology_df['h1_count'].std() / long_topology_df['h1_count'].mean()

    print(f"  5-day topology CV:   {short_cv:.3f}")
    print(f"  60-day topology CV:  {long_cv:.3f}")

    if short_cv < long_cv:
        print(f"\n✅ Short-window topology is MORE stable (lower CV)")
    else:
        print(f"\n⚠️  Short-window topology is LESS stable (higher CV)")
        print(f"     This may explain performance issues")
else:
    print(f"  5-day topology CV: {short_cv:.3f}")

# ============================================================================
# SAVE RESULTS
# ============================================================================

output_file = DATA_DIR / 'phase3_scale_consistent.csv'
strategy_df.to_csv(output_file)

print(f"\n💾 Results saved: {output_file}")

# ============================================================================
# SUMMARY
# ============================================================================

print("\n" + "=" * 80)
print("SCALE-CONSISTENT ARCHITECTURE COMPLETE")
print("=" * 80)

print("\n📊 Key Findings:")
print(f"  - Short-window (5-day) Sharpe: {sharpe:.3f}")
print(f"  - Position taken {position_pct * 100:.0f}% of trading days")
print(f"  - Addresses scale mismatch (aligned 5-day signals + 5-day topology)")

if sharpe > 0:
    print(f"\n✅ POSITIVE Sharpe! Scale consistency helps.")
else:
    print(f"\n⚠️  Negative Sharpe suggests other issues dominate")
    print(f"     (5-day windows may be too short for robust topology)")

print("\nNext: Run 03_adaptive_threshold_strategy.py")
