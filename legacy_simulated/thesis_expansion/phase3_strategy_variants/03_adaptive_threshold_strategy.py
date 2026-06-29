"""
Phase 3: Adaptive Threshold Strategy
=====================================

Addresses the "static threshold" limitation.

Original approach: Use fixed 75th percentile threshold from training data
- Market volatility changes over time
- What's "high stress" in calm 2020 ≠ "high stress" in volatile 2022
- Static threshold becomes miscalibrated

Solution: Adapt threshold based on recent market volatility
- Use rolling Z-score: z = (H₁_t - μ_recent) / σ_recent
- Threshold adapts to current market conditions
- More robust to regime shifts

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
print("PHASE 3: ADAPTIVE THRESHOLD STRATEGY")
print("=" * 80)

# Configuration
DATA_DIR = Path(__file__).parent.parent / 'data'
TEST_START = '2023-01-01'
TRANSACTION_COST = 0.0005
REBALANCE_FREQ = 5

# Adaptive parameters
ROLLING_WINDOW = 60  # 60-day rolling mean/std for Z-score
Z_THRESHOLD = 1.0  # Trade when |z| > 1.0 (1 std dev from mean)

# ============================================================================
# LOAD DATA
# ============================================================================

print("\n📂 Loading data...")

# Use Technology sector
returns_file = DATA_DIR / 'sector_technology_returns.csv'
topology_file = DATA_DIR / 'sector_technology_topology.csv'

if not returns_file.exists() or not topology_file.exists():
    print("⚠️  Technology sector data not found. Run Phase 2 first!")
    sys.exit(1)

returns_df = pd.read_csv(returns_file, index_col=0, parse_dates=True)
topology_df = pd.read_csv(topology_file, index_col=0, parse_dates=True)

print(f"✅ Returns: {len(returns_df)} days × {returns_df.shape[1]} stocks")
print(f"✅ Topology: {len(topology_df)} snapshots")

# ============================================================================
# COMPUTE ADAPTIVE Z-SCORE
# ============================================================================

print(f"\n📊 Computing {ROLLING_WINDOW}-day rolling Z-scores...")

# Rolling mean and std
rolling_mean = topology_df['h1_count'].rolling(window=ROLLING_WINDOW).mean()
rolling_std = topology_df['h1_count'].rolling(window=ROLLING_WINDOW).std()

# Z-score
topology_df['h1_zscore'] = (topology_df['h1_count'] - rolling_mean) / rolling_std

# Drop NaN from initial window
topology_df = topology_df.dropna()

print(f"✅ Z-scores computed for {len(topology_df)} dates")

# ============================================================================
# ADAPTIVE THRESHOLD STRATEGY
# ============================================================================

print("\n🎯 Running Adaptive Threshold Strategy...")

test_topology = topology_df[topology_df.index > TEST_START]

print(f"Testing period: {test_topology.index[0].strftime('%Y-%m-%d')} to "
      f"{test_topology.index[-1].strftime('%Y-%m-%d')}")
print(f"Z-score threshold: ±{Z_THRESHOLD}")

# Generate signals
strategy_returns = []

for i, date in enumerate(test_topology.index):
    if date not in returns_df.index:
        continue

    # Get adaptive Z-score
    z_score = test_topology.loc[date, 'h1_zscore']
    h1_loops = test_topology.loc[date, 'h1_count']

    # Compute momentum
    date_idx = returns_df.index.get_loc(date)

    if date_idx < 20:
        continue

    past_20day_returns = returns_df.iloc[date_idx - 20:date_idx].sum()

    top_performers = past_20day_returns.nlargest(5).index
    bottom_performers = past_20day_returns.nsmallest(5).index

    # Adaptive strategy:
    # z > +1.0: Very high stress → Strong mean reversion signal
    # -1.0 < z < +1.0: Normal → No trade
    # z < -1.0: Very low stress → Could indicate complacency → Fade momentum

    if z_score > Z_THRESHOLD:
        # High stress → Mean reversion
        long_stocks = bottom_performers
        short_stocks = top_performers
        signal_strength = min(abs(z_score) / 2, 1.0)  # Scale by Z-score magnitude
        regime = 'high_stress'

    elif z_score < -Z_THRESHOLD:
        # Very low stress → Fade (contrarian)
        long_stocks = bottom_performers
        short_stocks = top_performers
        signal_strength = min(abs(z_score) / 2, 1.0)
        regime = 'low_stress'

    else:
        # Normal regime → No trade
        long_stocks = []
        short_stocks = []
        signal_strength = 0
        regime = 'normal'

    # Compute return
    if len(long_stocks) > 0:
        long_return = returns_df.loc[date, long_stocks].mean()
        short_return = returns_df.loc[date, short_stocks].mean()

        strategy_return = (long_return - short_return) / 2 * signal_strength
    else:
        strategy_return = 0

    # Transaction costs
    if i % REBALANCE_FREQ == 0 and len(long_stocks) > 0:
        n_trades = 10
        cost = n_trades * TRANSACTION_COST
        strategy_return -= cost

    strategy_returns.append({
        'date': date,
        'return': strategy_return,
        'h1_loops': h1_loops,
        'z_score': z_score,
        'regime': regime,
        'signal_strength': signal_strength
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
high_stress_days = regime_counts.get('high_stress', 0)
low_stress_days = regime_counts.get('low_stress', 0)
normal_days = regime_counts.get('normal', 0)

print(f"\nTotal Return:      {total_return * 100:>8.2f}%")
print(f"Annual Return:     {annual_return * 100:>8.2f}%")
print(f"Sharpe Ratio:      {sharpe:>8.3f}")
print(f"Max Drawdown:      {max_drawdown * 100:>8.2f}%")
print(f"Win Rate:          {win_rate * 100:>8.1f}%")

print(f"\nRegime Distribution:")
print(f"  High Stress:  {high_stress_days} days ({high_stress_days/n_days*100:.1f}%)")
print(f"  Low Stress:   {low_stress_days} days ({low_stress_days/n_days*100:.1f}%)")
print(f"  Normal (flat): {normal_days} days ({normal_days/n_days*100:.1f}%)")

# Average Z-score by regime
avg_z_high = strategy_df[strategy_df['regime'] == 'high_stress']['z_score'].mean()
avg_z_low = strategy_df[strategy_df['regime'] == 'low_stress']['z_score'].mean()

print(f"\nAverage Z-score:")
print(f"  High Stress regime: {avg_z_high:>6.2f}")
print(f"  Low Stress regime:  {avg_z_low:>6.2f}")

# Compare to static threshold baseline
baseline_file = DATA_DIR / 'sector_technology_strategy.csv'

if baseline_file.exists():
    baseline_df = pd.read_csv(baseline_file, index_col=0, parse_dates=True)

    common_dates = strategy_df.index.intersection(baseline_df.index)

    if len(common_dates) > 10:
        baseline_returns = baseline_df.loc[common_dates, 'return']
        baseline_sharpe = baseline_returns.mean() / baseline_returns.std() * np.sqrt(252)

        print(f"\n📊 Comparison to Static Threshold Baseline:")
        print(f"  Adaptive Threshold Sharpe:  {sharpe:>6.3f}")
        print(f"  Static Threshold Sharpe:    {baseline_sharpe:>6.3f}")
        print(f"  Improvement:                {(sharpe - baseline_sharpe):>+6.3f}")

        if sharpe > baseline_sharpe:
            print("\n✅ Adaptive threshold IMPROVES performance!")
        else:
            print("\n❌ Adaptive threshold does not improve performance")

# ============================================================================
# SAVE RESULTS
# ============================================================================

output_file = DATA_DIR / 'phase3_adaptive_threshold.csv'
strategy_df.to_csv(output_file)

print(f"\n💾 Results saved: {output_file}")

# ============================================================================
# SUMMARY
# ============================================================================

print("\n" + "=" * 80)
print("ADAPTIVE THRESHOLD STRATEGY COMPLETE")
print("=" * 80)

print("\n📊 Key Findings:")
print(f"  - Adaptive Sharpe: {sharpe:.3f}")
print(f"  - Trades only in extreme regimes (|z| > {Z_THRESHOLD})")
print(f"  - Active {(high_stress_days + low_stress_days)/n_days*100:.0f}% of time")

if sharpe > 0:
    print(f"\n✅ POSITIVE Sharpe! Adaptive approach adds value.")
else:
    print(f"\n⚠️  Negative Sharpe - static threshold may be sufficient")

print("\nNext: Run 04_compare_strategies.py to evaluate all variants")
