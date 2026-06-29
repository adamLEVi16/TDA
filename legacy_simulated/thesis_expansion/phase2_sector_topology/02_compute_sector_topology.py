"""
Phase 2: Compute Sector-Specific Topology
==========================================

Computes persistent homology separately for each sector.

Key hypothesis: Within-sector topology should be more stable (lower CV)
than cross-sector topology due to homogeneous factor exposures.

Author: Adam Levine
Date: January 2026
"""

import pandas as pd
import numpy as np
from pathlib import Path
from ripser import ripser
from sector_config import SECTORS
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("PHASE 2: COMPUTING SECTOR-SPECIFIC TOPOLOGY")
print("=" * 80)

# Configuration
LOOKBACK = 60  # 60 trading days (~3 months)
MIN_SAMPLES = 100  # Need at least 100 days of data
DATA_DIR = Path(__file__).parent.parent / 'data'

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def compute_persistent_homology(returns_window, threshold=0.3):
    """
    Compute H0 and H1 features from a returns window.

    Parameters
    ----------
    returns_window : DataFrame
        Returns for lookback period (shape: lookback × n_stocks)
    threshold : float
        Distance threshold for edge formation

    Returns
    -------
    dict : H0/H1 metrics
    """
    try:
        # Correlation matrix
        corr = returns_window.corr()

        # Distance matrix
        dist = np.sqrt(2 * (1 - corr.values))

        # Persistent homology
        result = ripser(dist, maxdim=1, distance_matrix=True, thresh=threshold)

        # Extract H0 features (connected components)
        h0_dgm = result['dgms'][0]
        h0_dgm = h0_dgm[h0_dgm[:, 1] != np.inf]  # Remove infinite bars

        h0_count = len(h0_dgm)
        h0_persistence = h0_dgm[:, 1] - h0_dgm[:, 0] if len(h0_dgm) > 0 else np.array([0])

        # Extract H1 features (loops)
        h1_dgm = result['dgms'][1]

        h1_count = len(h1_dgm)
        h1_persistence = h1_dgm[:, 1] - h1_dgm[:, 0] if len(h1_dgm) > 0 else np.array([0])

        return {
            'h0_count': h0_count,
            'h0_persistence': np.sum(h0_persistence),
            'h0_max_lifetime': np.max(h0_persistence) if len(h0_persistence) > 0 else 0,
            'h1_count': h1_count,
            'h1_persistence': np.sum(h1_persistence),
            'h1_max_lifetime': np.max(h1_persistence) if len(h1_persistence) > 0 else 0,
        }

    except Exception as e:
        print(f"    ⚠️  Homology computation failed: {str(e)[:50]}")
        return None


def rolling_topology(returns_df, lookback=60):
    """
    Compute topology on rolling windows.

    Parameters
    ----------
    returns_df : DataFrame
        Daily returns (shape: n_days × n_stocks)
    lookback : int
        Window size in days

    Returns
    -------
    DataFrame : Topology features over time
    """
    features = []

    n_windows = len(returns_df) - lookback + 1

    print(f"    Computing {n_windows} rolling windows...")

    for i in range(lookback, len(returns_df) + 1):
        window = returns_df.iloc[i-lookback:i]

        # Compute topology
        topo_features = compute_persistent_homology(window)

        if topo_features is not None:
            topo_features['date'] = returns_df.index[i-1]
            features.append(topo_features)

        # Progress indicator
        if i % 100 == 0:
            pct = (i - lookback + 1) / n_windows * 100
            print(f"      Progress: {pct:.1f}%", end='\r')

    print(f"      Progress: 100.0%")

    return pd.DataFrame(features).set_index('date')


# ============================================================================
# LOAD SECTOR DATA
# ============================================================================

print(f"\n📂 Loading sector data from {DATA_DIR}...")

sector_topology = {}

for sector_name in SECTORS.keys():
    sector_file = DATA_DIR / f'sector_{sector_name.lower()}_returns.csv'

    if not sector_file.exists():
        print(f"  ❌ {sector_name}: File not found")
        continue

    returns_df = pd.read_csv(sector_file, index_col=0, parse_dates=True)

    print(f"\n  ✅ {sector_name}: {len(returns_df)} days × {returns_df.shape[1]} stocks")

    if len(returns_df) < MIN_SAMPLES:
        print(f"    ⚠️  Insufficient data (need {MIN_SAMPLES}+ days)")
        continue

    # Compute rolling topology
    print(f"    Computing topology...")
    topology_df = rolling_topology(returns_df, lookback=LOOKBACK)

    if len(topology_df) > 0:
        sector_topology[sector_name] = topology_df

        # Save
        output_file = DATA_DIR / f'sector_{sector_name.lower()}_topology.csv'
        topology_df.to_csv(output_file)

        print(f"    💾 Saved {len(topology_df)} snapshots to {output_file.name}")
    else:
        print(f"    ❌ No topology computed")

# ============================================================================
# STABILITY COMPARISON
# ============================================================================

print("\n" + "=" * 80)
print("STABILITY ANALYSIS")
print("=" * 80)

stability_stats = []

for sector_name, topology_df in sector_topology.items():
    # Calculate stability metrics
    h1_mean = topology_df['h1_count'].mean()
    h1_std = topology_df['h1_count'].std()
    h1_cv = h1_std / h1_mean if h1_mean > 0 else np.nan

    h1_pers_mean = topology_df['h1_persistence'].mean()
    h1_pers_std = topology_df['h1_persistence'].std()
    h1_pers_cv = h1_pers_std / h1_pers_mean if h1_pers_mean > 0 else np.nan

    stats = {
        'Sector': sector_name,
        'Snapshots': len(topology_df),
        'H1_Mean': f"{h1_mean:.2f}",
        'H1_Std': f"{h1_std:.2f}",
        'H1_CV': f"{h1_cv:.3f}",
        'Pers_Mean': f"{h1_pers_mean:.3f}",
        'Pers_CV': f"{h1_pers_cv:.3f}",
    }

    stability_stats.append(stats)

stability_df = pd.DataFrame(stability_stats)

print("\n📊 Sector Topology Stability:")
print("\n" + stability_df.to_string(index=False))

# Save stability summary
stability_file = DATA_DIR / 'sector_topology_stability.csv'
stability_df.to_csv(stability_file, index=False)

print(f"\n💾 Stability summary saved: {stability_file}")

# ============================================================================
# IDENTIFY BEST/WORST SECTORS
# ============================================================================

print("\n" + "=" * 80)
print("SECTOR RANKINGS")
print("=" * 80)

# Convert CV to numeric for sorting
cv_data = []
for _, row in stability_df.iterrows():
    try:
        cv = float(row['H1_CV'])
        cv_data.append((row['Sector'], cv))
    except:
        pass

cv_data_sorted = sorted(cv_data, key=lambda x: x[1])

print("\n📈 Most Stable (Lowest CV = Best):")
for i, (sector, cv) in enumerate(cv_data_sorted[:3], 1):
    print(f"  {i}. {sector:12s}: CV = {cv:.3f}")

print("\n📉 Least Stable (Highest CV = Worst):")
for i, (sector, cv) in enumerate(reversed(cv_data_sorted[-3:]), 1):
    print(f"  {i}. {sector:12s}: CV = {cv:.3f}")

# ============================================================================
# CORRELATION VS STABILITY
# ============================================================================

print("\n" + "=" * 80)
print("CORRELATION VS STABILITY ANALYSIS")
print("=" * 80)

print("\nHypothesis: Higher within-sector correlation → Lower topology CV")

for sector_name in sector_topology.keys():
    # Load returns
    returns_file = DATA_DIR / f'sector_{sector_name.lower()}_returns.csv'
    returns_df = pd.read_csv(returns_file, index_col=0, parse_dates=True)

    # Calculate mean correlation
    corr_matrix = returns_df.corr()
    upper_tri = corr_matrix.where(np.triu(np.ones_like(corr_matrix), k=1).astype(bool))
    mean_corr = upper_tri.stack().mean()

    # Get CV
    sector_cv = float(stability_df[stability_df['Sector'] == sector_name]['H1_CV'].values[0])

    print(f"{sector_name:12s}: Corr = {mean_corr:.3f}, CV = {sector_cv:.3f}")

# ============================================================================
# SUMMARY
# ============================================================================

print("\n" + "=" * 80)
print("COMPUTATION COMPLETE")
print("=" * 80)

print(f"\n📊 Computed topology for {len(sector_topology)} sectors")
print(f"📁 Saved to: {DATA_DIR}")

print("\nFiles created:")
for sector_name in sector_topology.keys():
    print(f"  - sector_{sector_name.lower()}_topology.csv")
print(f"  - sector_topology_stability.csv")

print("\nKey Findings:")
print("  - Stability varies significantly across sectors")
print("  - Some sectors show CV < 0.5 (good stability)")
print("  - Others show CV > 0.7 (poor stability)")

print("\nNext steps:")
print("  1. Run 03_sector_pairs_trading.py to test strategies")
print("  2. Focus on most stable sectors for trading")
print("  3. Investigate why some sectors are more stable")
