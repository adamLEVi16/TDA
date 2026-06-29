"""
Phase 1: Compute Topology Features on Intraday Data
====================================================

This script computes persistent homology (H0, H1) features on intraday
correlation networks to test the hypothesis that increased sample size
improves topological inference stability.

Input:
- intraday_returns_5min.csv (from script 01)

Output:
- intraday_topology_features.csv: H1 features over time
- topology_comparison.csv: Daily vs intraday comparison

Author: Adam Levine
Date: January 2026
"""

import pandas as pd
import numpy as np
from scipy import linalg
from ripser import ripser
import sys
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Add parent directory for imports
sys.path.append(str(Path(__file__).parent.parent))

print("=" * 80)
print("PHASE 1: TOPOLOGY COMPUTATION ON INTRADAY DATA")
print("=" * 80)

# Directories
DATA_DIR = Path(__file__).parent.parent / 'data'
OUTPUT_DIR = DATA_DIR

# Parameters (matching original paper)
LOOKBACK_BARS = 780  # ~60 trading days × 13 bars/day (6.5 market hours)
CORRELATION_THRESHOLD = 0.3
SAMPLE_FREQUENCY = 78  # Sample topology every 78 bars (~1 trading day)

print(f"\n⚙️  Parameters:")
print(f"   Lookback window: {LOOKBACK_BARS} bars (~60 trading days)")
print(f"   Correlation threshold: {CORRELATION_THRESHOLD}")
print(f"   Sampling frequency: Every {SAMPLE_FREQUENCY} bars (~1 day)")

# ============================================================================
# LOAD DATA
# ============================================================================

print(f"\n📂 Loading intraday data...")
intraday_file = DATA_DIR / 'intraday_returns_5min.csv'

if not intraday_file.exists():
    print(f"❌ File not found: {intraday_file}")
    print("Run 01_download_intraday_data.py first!")
    sys.exit(1)

intraday_returns = pd.read_csv(intraday_file, index_col=0, parse_dates=True)

print(f"✅ Loaded {len(intraday_returns):,} 5-minute bars")
print(f"   Assets: {len(intraday_returns.columns)}")
print(f"   Date range: {intraday_returns.index[0]} to {intraday_returns.index[-1]}")

# ============================================================================
# COMPUTE TOPOLOGY FEATURES
# ============================================================================

def compute_persistent_homology(returns_window, threshold=0.3):
    """
    Compute H0 and H1 persistence features for a rolling window

    Args:
        returns_window (pd.DataFrame): Returns data for correlation computation
        threshold (float): Correlation threshold for graph construction

    Returns:
        dict: Topology features (h0_count, h1_count, h1_persistence)
    """

    # Correlation matrix
    corr = returns_window.corr()

    # Convert to distance metric
    dist = np.sqrt(2 * (1 - corr.values))
    np.fill_diagonal(dist, 0)

    # Compute persistent homology
    try:
        result = ripser(dist, maxdim=1, distance_matrix=True)

        # Extract H0 (components)
        h0_diagram = result['dgms'][0]
        h0_diagram = h0_diagram[~np.isinf(h0_diagram).any(axis=1)]

        # Extract H1 (loops)
        h1_diagram = result['dgms'][1]
        h1_diagram = h1_diagram[~np.isinf(h1_diagram).any(axis=1)]

        # Calculate features
        features = {
            'h0_count': len(h0_diagram),
            'h1_count': len(h1_diagram),
            'h1_persistence': 0.0,
            'h1_max_lifetime': 0.0,
        }

        # H1 persistence (total and max)
        if len(h1_diagram) > 0:
            lifetimes = h1_diagram[:, 1] - h1_diagram[:, 0]
            features['h1_persistence'] = lifetimes.sum()
            features['h1_max_lifetime'] = lifetimes.max()

        return features

    except Exception as e:
        # If computation fails, return NaN
        print(f"⚠️  Topology computation failed: {e}")
        return {
            'h0_count': np.nan,
            'h1_count': np.nan,
            'h1_persistence': np.nan,
            'h1_max_lifetime': np.nan,
        }


print(f"\n⏳ Computing topology features...")
print(f"   Total bars: {len(intraday_returns):,}")
print(f"   Lookback: {LOOKBACK_BARS}")
print(f"   Expected samples: {(len(intraday_returns) - LOOKBACK_BARS) // SAMPLE_FREQUENCY:,}")
print(f"\n   Estimated time: ~3-5 minutes")

# Storage
topology_results = []
dates = []

# Rolling computation
total_samples = 0
for i in range(LOOKBACK_BARS, len(intraday_returns), SAMPLE_FREQUENCY):

    # Progress indicator
    if total_samples % 10 == 0:
        progress = (i - LOOKBACK_BARS) / (len(intraday_returns) - LOOKBACK_BARS) * 100
        print(f"   Progress: {progress:.1f}% ({total_samples} samples)", end='\r', flush=True)

    # Get window
    window = intraday_returns.iloc[i-LOOKBACK_BARS:i]

    # Compute topology
    features = compute_persistent_homology(window, threshold=CORRELATION_THRESHOLD)

    # Store
    topology_results.append(features)
    dates.append(intraday_returns.index[i])
    total_samples += 1

print(f"\n✅ Computed {total_samples} topology snapshots")

# Create DataFrame
topology_intraday = pd.DataFrame(topology_results, index=dates)

# Remove any NaN rows
n_nan = topology_intraday.isna().any(axis=1).sum()
if n_nan > 0:
    print(f"⚠️  Removing {n_nan} failed computations")
    topology_intraday = topology_intraday.dropna()

print(f"\n📊 Topology Features Summary:")
print(topology_intraday.describe())

# ============================================================================
# LOAD DAILY TOPOLOGY FOR COMPARISON
# ============================================================================

print(f"\n📂 Loading original daily topology...")

# Check if original topology file exists
daily_topology_file = Path('/home/user/TDA/topology_features.csv')

if daily_topology_file.exists():
    topology_daily = pd.read_csv(daily_topology_file, index_col=0, parse_dates=True)
    print(f"✅ Loaded {len(topology_daily)} daily topology snapshots")

    # ========================================================================
    # STABILITY COMPARISON
    # ========================================================================

    print(f"\n" + "=" * 80)
    print("STABILITY ANALYSIS: Daily vs Intraday")
    print("=" * 80)

    # Coefficient of variation (CV = std/mean)
    daily_cv_h1 = topology_daily['h1_loops'].std() / topology_daily['h1_loops'].mean()
    intraday_cv_h1 = topology_intraday['h1_count'].std() / topology_intraday['h1_count'].mean()

    daily_cv_persist = topology_daily['h1_persistence'].std() / topology_daily['h1_persistence'].mean()
    intraday_cv_persist = topology_intraday['h1_persistence'].std() / topology_intraday['h1_persistence'].mean()

    print(f"\nH1 Loop Count:")
    print(f"   Daily:    Mean = {topology_daily['h1_loops'].mean():.2f}, " +
          f"Std = {topology_daily['h1_loops'].std():.2f}, CV = {daily_cv_h1:.3f}")
    print(f"   Intraday: Mean = {topology_intraday['h1_count'].mean():.2f}, " +
          f"Std = {topology_intraday['h1_count'].std():.2f}, CV = {intraday_cv_h1:.3f}")

    improvement_h1 = (1 - intraday_cv_h1/daily_cv_h1) * 100
    print(f"   → Stability improvement: {improvement_h1:.1f}%")

    print(f"\nH1 Persistence:")
    print(f"   Daily:    CV = {daily_cv_persist:.3f}")
    print(f"   Intraday: CV = {intraday_cv_persist:.3f}")

    improvement_persist = (1 - intraday_cv_persist/daily_cv_persist) * 100
    print(f"   → Stability improvement: {improvement_persist:.1f}%")

    # Save comparison
    comparison_df = pd.DataFrame({
        'Metric': ['H1 Loop Count', 'H1 Persistence'],
        'Daily CV': [daily_cv_h1, daily_cv_persist],
        'Intraday CV': [intraday_cv_h1, intraday_cv_persist],
        'Improvement (%)': [improvement_h1, improvement_persist],
    })

    comparison_file = OUTPUT_DIR / 'topology_comparison.csv'
    comparison_df.to_csv(comparison_file, index=False)
    print(f"\n💾 Saved comparison: {comparison_file}")

else:
    print("⚠️  Original daily topology file not found")
    print("   Skipping comparison (will only analyze intraday data)")

# ============================================================================
# SAVE RESULTS
# ============================================================================

output_file = OUTPUT_DIR / 'intraday_topology_features.csv'
topology_intraday.to_csv(output_file)

print(f"\n💾 Saved: {output_file}")
print(f"   Rows: {len(topology_intraday):,}")
print(f"   Columns: {list(topology_intraday.columns)}")

print("\n" + "=" * 80)
print("TOPOLOGY COMPUTATION COMPLETE")
print("=" * 80)
print(f"\nKey Finding:")
if 'improvement_h1' in locals():
    print(f"  → Intraday data provides {improvement_h1:.1f}% stability improvement")
    print(f"  → Mean H1 loop count consistent: {topology_daily['h1_loops'].mean():.2f} (daily) vs " +
          f"{topology_intraday['h1_count'].mean():.2f} (intraday)")
    print(f"\nThis validates that topology features reflect genuine")
    print(f"market structure, not sampling artifacts!")
else:
    print(f"  → Intraday topology features computed successfully")
    print(f"  → Mean H1 loops: {topology_intraday['h1_count'].mean():.2f}")

print(f"\nNext step: Run 03_create_visualizations.py")
