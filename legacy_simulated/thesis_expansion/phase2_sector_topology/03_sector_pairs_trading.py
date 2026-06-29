"""
Phase 2: Sector-Specific Pairs Trading Strategy
================================================

Tests pairs trading strategy within each sector using sector-specific topology.

Strategy logic:
1. Compute topology for sector (already done)
2. When H1 loops > threshold: Market stressed → Long volatility → Long high-beta pairs
3. When H1 loops < threshold: Market calm → Short volatility → Short high-beta pairs

This is sector-specific mean-reversion based on topological regime detection.

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
print("PHASE 2: SECTOR-SPECIFIC PAIRS TRADING")
print("=" * 80)

# Configuration
TRAIN_START = '2020-01-01'
TRAIN_END = '2022-12-31'
TEST_START = '2023-01-01'
TEST_END = '2024-12-31'

TRANSACTION_COST = 0.0005  # 5 bps per trade
REBALANCE_FREQ = 5  # Rebalance every 5 days

DATA_DIR = Path(__file__).parent.parent / 'data'

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def compute_beta(returns_df, market_returns):
    """Compute beta for each stock vs sector average."""
    betas = {}

    for col in returns_df.columns:
        stock_returns = returns_df[col].dropna()
        aligned_market = market_returns.loc[stock_returns.index]

        # Simple regression beta
        covariance = stock_returns.cov(aligned_market)
        market_variance = aligned_market.var()

        beta = covariance / market_variance if market_variance > 0 else 1.0
        betas[col] = beta

    return pd.Series(betas)


def pairs_trading_strategy(returns_df, topology_df, train_end_date):
    """
    Sector-specific pairs trading strategy.

    Parameters
    ----------
    returns_df : DataFrame
        Daily returns for sector stocks
    topology_df : DataFrame
        Topology features (H1 loops, persistence)
    train_end_date : str
        End of training period

    Returns
    -------
    DataFrame : Strategy returns
    """
    # Split train/test
    train_mask = topology_df.index <= train_end_date
    test_mask = topology_df.index > train_end_date

    train_topology = topology_df[train_mask]
    test_topology = topology_df[test_mask]

    # Determine threshold on training data
    threshold = train_topology['h1_count'].quantile(0.75)

    print(f"    Training period: {train_topology.index[0].strftime('%Y-%m-%d')} to "
          f"{train_topology.index[-1].strftime('%Y-%m-%d')}")
    print(f"    Testing period:  {test_topology.index[0].strftime('%Y-%m-%d')} to "
          f"{test_topology.index[-1].strftime('%Y-%m-%d')}")
    print(f"    H1 threshold (75th percentile): {threshold:.2f}")

    # Compute sector market returns (equal-weight average)
    market_returns = returns_df.mean(axis=1)

    # Compute betas on training data
    train_returns = returns_df.loc[:train_end_date]
    train_market = market_returns.loc[:train_end_date]

    betas = compute_beta(train_returns, train_market)

    # Select high/low beta stocks
    n_pairs = 3  # Top 3 high-beta, top 3 low-beta
    high_beta_stocks = betas.nlargest(n_pairs).index.tolist()
    low_beta_stocks = betas.nsmallest(n_pairs).index.tolist()

    print(f"    High-beta stocks: {high_beta_stocks}")
    print(f"    Low-beta stocks:  {low_beta_stocks}")

    # Generate signals
    signals = []

    for date in test_topology.index:
        if date not in returns_df.index:
            continue

        h1_loops = test_topology.loc[date, 'h1_count']

        # Signal logic:
        # High loops (> threshold) = stressed market = long high-beta (expect mean reversion)
        # Low loops (< threshold) = calm market = short high-beta (expect continuation)

        if h1_loops > threshold:
            # Stressed regime: Long high-beta, short low-beta
            position = 1.0
        else:
            # Calm regime: Short high-beta, long low-beta
            position = -1.0

        signals.append({
            'date': date,
            'h1_loops': h1_loops,
            'position': position
        })

    signals_df = pd.DataFrame(signals).set_index('date')

    # Compute strategy returns
    strategy_returns = []

    for i, date in enumerate(signals_df.index):
        if date not in returns_df.index:
            continue

        position = signals_df.loc[date, 'position']

        # Portfolio return: position × (high_beta - low_beta)
        high_beta_return = returns_df.loc[date, high_beta_stocks].mean()
        low_beta_return = returns_df.loc[date, low_beta_stocks].mean()

        spread_return = high_beta_return - low_beta_return
        strategy_return = position * spread_return

        # Transaction costs on rebalances
        if i % REBALANCE_FREQ == 0:
            # Trade all pairs = 2 × n_pairs stocks
            n_trades = 2 * n_pairs
            cost = n_trades * TRANSACTION_COST
            strategy_return -= cost

        strategy_returns.append({
            'date': date,
            'return': strategy_return,
            'position': position,
            'h1_loops': signals_df.loc[date, 'h1_loops']
        })

    return pd.DataFrame(strategy_returns).set_index('date')


def calculate_performance_metrics(returns_series):
    """Calculate Sharpe, max drawdown, win rate, etc."""
    # Cumulative returns
    cum_returns = (1 + returns_series).cumprod()

    # Total return
    total_return = cum_returns.iloc[-1] - 1

    # Annualized return
    n_days = len(returns_series)
    n_years = n_days / 252
    annual_return = (1 + total_return) ** (1 / n_years) - 1

    # Sharpe ratio
    sharpe = returns_series.mean() / returns_series.std() * np.sqrt(252) if returns_series.std() > 0 else 0

    # Max drawdown
    running_max = cum_returns.expanding().max()
    drawdown = (cum_returns - running_max) / running_max
    max_drawdown = drawdown.min()

    # Win rate
    win_rate = (returns_series > 0).mean()

    # Calmar ratio
    calmar = annual_return / abs(max_drawdown) if max_drawdown != 0 else 0

    return {
        'Total Return': f"{total_return * 100:.2f}%",
        'Annual Return': f"{annual_return * 100:.2f}%",
        'Sharpe Ratio': f"{sharpe:.3f}",
        'Max Drawdown': f"{max_drawdown * 100:.2f}%",
        'Calmar Ratio': f"{calmar:.3f}",
        'Win Rate': f"{win_rate * 100:.1f}%",
        'Observations': len(returns_series),
    }


# ============================================================================
# RUN STRATEGIES FOR EACH SECTOR
# ============================================================================

print(f"\n🎯 Running sector-specific strategies...")
print(f"   Train: {TRAIN_START} to {TRAIN_END}")
print(f"   Test:  {TEST_START} to {TEST_END}")

sector_results = {}

for sector_name in SECTORS.keys():
    print(f"\n{'='*60}")
    print(f"  {sector_name}")
    print(f"{'='*60}")

    # Load data
    returns_file = DATA_DIR / f'sector_{sector_name.lower()}_returns.csv'
    topology_file = DATA_DIR / f'sector_{sector_name.lower()}_topology.csv'

    if not returns_file.exists() or not topology_file.exists():
        print(f"    ❌ Missing data files")
        continue

    returns_df = pd.read_csv(returns_file, index_col=0, parse_dates=True)
    topology_df = pd.read_csv(topology_file, index_col=0, parse_dates=True)

    # Run strategy
    try:
        strategy_df = pairs_trading_strategy(returns_df, topology_df, TRAIN_END)

        if len(strategy_df) > 0:
            # Calculate metrics
            metrics = calculate_performance_metrics(strategy_df['return'])

            print(f"\n    📊 Performance Metrics:")
            for key, value in metrics.items():
                print(f"       {key:15s}: {value}")

            sector_results[sector_name] = {
                'strategy_df': strategy_df,
                'metrics': metrics
            }

            # Save strategy returns
            output_file = DATA_DIR / f'sector_{sector_name.lower()}_strategy.csv'
            strategy_df.to_csv(output_file)

            # Extract Sharpe for summary
            sharpe_str = metrics['Sharpe Ratio']
            sharpe_val = float(sharpe_str)

            if sharpe_val > 0.5:
                print(f"    ✅ POSITIVE Sharpe! ({sharpe_val:.2f})")
            elif sharpe_val > 0:
                print(f"    🟡 Slightly positive Sharpe ({sharpe_val:.2f})")
            else:
                print(f"    ❌ Negative Sharpe ({sharpe_val:.2f})")

        else:
            print(f"    ⚠️  No test period data")

    except Exception as e:
        print(f"    ❌ Strategy failed: {str(e)[:60]}")
        continue

# ============================================================================
# SUMMARY COMPARISON
# ============================================================================

print("\n" + "=" * 80)
print("SECTOR STRATEGY COMPARISON")
print("=" * 80)

summary_data = []

for sector_name, result in sector_results.items():
    metrics = result['metrics']

    # Extract numeric values
    sharpe = float(metrics['Sharpe Ratio'])
    total_ret = float(metrics['Total Return'].rstrip('%')) / 100
    win_rate = float(metrics['Win Rate'].rstrip('%')) / 100

    summary_data.append({
        'Sector': sector_name,
        'Sharpe': sharpe,
        'Total Return': f"{total_ret*100:.1f}%",
        'Win Rate': f"{win_rate*100:.0f}%",
        'Max DD': metrics['Max Drawdown'],
    })

summary_df = pd.DataFrame(summary_data).sort_values('Sharpe', ascending=False)

print("\n📊 Ranked by Sharpe Ratio:")
print("\n" + summary_df.to_string(index=False))

# Save summary
summary_file = DATA_DIR / 'sector_strategy_summary.csv'
summary_df.to_csv(summary_file, index=False)

print(f"\n💾 Summary saved: {summary_file}")

# ============================================================================
# IDENTIFY WINNING SECTORS
# ============================================================================

print("\n" + "=" * 80)
print("WINNING SECTORS")
print("=" * 80)

positive_sharpe_sectors = [s for s in summary_data if s['Sharpe'] > 0]

if positive_sharpe_sectors:
    print(f"\n✅ {len(positive_sharpe_sectors)} sector(s) with POSITIVE Sharpe:")
    for sector_data in sorted(positive_sharpe_sectors, key=lambda x: x['Sharpe'], reverse=True):
        print(f"   {sector_data['Sector']:12s}: Sharpe = {sector_data['Sharpe']:.3f}, "
              f"Return = {sector_data['Total Return']}, Win Rate = {sector_data['Win Rate']}")

    print("\n💡 Key Insight:")
    print("   Sector-specific topology CAN produce positive returns!")
    print("   Focus on these sectors for multi-sector portfolio.")

else:
    print("\n❌ No sectors with positive Sharpe")
    print("   Need to refine strategy logic or parameters")

# ============================================================================
# NEXT STEPS
# ============================================================================

print("\n" + "=" * 80)
print("STRATEGY TESTING COMPLETE")
print("=" * 80)

print(f"\n📊 Tested {len(sector_results)} sector strategies")
print(f"📁 Results saved to: {DATA_DIR}")

print("\nNext steps:")
print("  1. Run 04_compare_sectors.py to build multi-sector portfolio")
print("  2. Run 05_visualize_sectors.py to create publication figures")
print("  3. Analyze why some sectors work better than others")
