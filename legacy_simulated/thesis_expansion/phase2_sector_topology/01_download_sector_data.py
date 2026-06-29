"""
Phase 2: Download Sector-Specific Data
=======================================

Downloads daily returns for 7 sectors (20 stocks each = 140 total).

Strategy hypothesis: Sector-homogeneous correlation networks should produce
cleaner topology and better trading signals than mixed-sector universes.

Author: Adam Levine
Date: January 2026
"""

import yfinance as yf
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import time
from sector_config import SECTORS

print("=" * 80)
print("PHASE 2: DOWNLOADING SECTOR-SPECIFIC DATA")
print("=" * 80)

# Configuration
START_DATE = '2020-01-01'
END_DATE = '2024-12-31'
DATA_DIR = Path(__file__).parent.parent / 'data'
DATA_DIR.mkdir(parents=True, exist_ok=True)

print(f"\n📊 Configuration:")
print(f"  Sectors: {len(SECTORS)}")
print(f"  Stocks per sector: 20")
print(f"  Total stocks: {sum(len(v) for v in SECTORS.values())}")
print(f"  Date range: {START_DATE} to {END_DATE}")

# ============================================================================
# DOWNLOAD SECTOR DATA
# ============================================================================

sector_data = {}
failed_tickers = []

for sector_name, tickers in SECTORS.items():
    print(f"\n📥 Downloading {sector_name} ({len(tickers)} stocks)...")

    sector_returns = []
    successful_tickers = []

    for ticker in tickers:
        try:
            # Download with retries
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    data = yf.download(ticker, start=START_DATE, end=END_DATE,
                                     progress=False, show_errors=False)

                    if len(data) > 0:
                        # Calculate returns
                        returns = data['Adj Close'].pct_change()
                        returns.name = ticker
                        sector_returns.append(returns)
                        successful_tickers.append(ticker)
                        print(f"  ✅ {ticker}: {len(returns)} days")
                        break
                    else:
                        if attempt < max_retries - 1:
                            time.sleep(1)
                            continue
                        else:
                            raise ValueError("No data returned")

                except Exception as e:
                    if attempt < max_retries - 1:
                        time.sleep(2)
                        continue
                    else:
                        raise

        except Exception as e:
            print(f"  ❌ {ticker}: Failed ({str(e)[:50]})")
            failed_tickers.append((sector_name, ticker))
            continue

        # Rate limiting
        time.sleep(0.1)

    # Combine sector data
    if sector_returns:
        sector_df = pd.concat(sector_returns, axis=1)
        sector_df = sector_df.dropna(how='all')  # Drop days with no data

        # Save sector data
        sector_file = DATA_DIR / f'sector_{sector_name.lower()}_returns.csv'
        sector_df.to_csv(sector_file)

        sector_data[sector_name] = sector_df

        print(f"  💾 Saved: {len(sector_df)} days × {len(successful_tickers)} stocks")
        print(f"  📁 File: {sector_file.name}")
    else:
        print(f"  ⚠️  No data for {sector_name}")

# ============================================================================
# SUMMARY STATISTICS
# ============================================================================

print("\n" + "=" * 80)
print("DOWNLOAD SUMMARY")
print("=" * 80)

summary_stats = []

for sector_name, df in sector_data.items():
    stats = {
        'Sector': sector_name,
        'Stocks': df.shape[1],
        'Days': df.shape[0],
        'Start': df.index[0].strftime('%Y-%m-%d'),
        'End': df.index[-1].strftime('%Y-%m-%d'),
        'Avg Return': f"{df.mean().mean() * 100:.3f}%",
        'Avg Vol': f"{df.std().mean() * 100:.2f}%",
    }
    summary_stats.append(stats)

summary_df = pd.DataFrame(summary_stats)
print("\n" + summary_df.to_string(index=False))

# Save summary
summary_file = DATA_DIR / 'sector_download_summary.csv'
summary_df.to_csv(summary_file, index=False)
print(f"\n💾 Summary saved: {summary_file}")

# ============================================================================
# CORRELATION ANALYSIS
# ============================================================================

print("\n" + "=" * 80)
print("CORRELATION ANALYSIS")
print("=" * 80)

print("\n📊 Within-sector vs cross-sector correlations:")

# Calculate within-sector correlations
within_sector_corrs = []

for sector_name, df in sector_data.items():
    corr_matrix = df.corr()
    # Get upper triangle (exclude diagonal)
    upper_tri = corr_matrix.where(np.triu(np.ones_like(corr_matrix), k=1).astype(bool))
    within_corrs = upper_tri.stack().values

    mean_corr = np.mean(within_corrs)
    within_sector_corrs.extend(within_corrs)

    print(f"\n{sector_name:12s}: {mean_corr:.3f} (n={len(within_corrs)} pairs)")

# Calculate cross-sector correlation (sample)
print(f"\n{'Cross-sector':12s}: ", end='')

cross_sector_corrs = []
sector_names = list(sector_data.keys())

# Sample: compare first 5 stocks from each sector pair
for i in range(len(sector_names)):
    for j in range(i+1, len(sector_names)):
        df1 = sector_data[sector_names[i]].iloc[:, :5]  # First 5 stocks
        df2 = sector_data[sector_names[j]].iloc[:, :5]

        # Align dates
        common_dates = df1.index.intersection(df2.index)
        df1_aligned = df1.loc[common_dates]
        df2_aligned = df2.loc[common_dates]

        # Calculate correlations
        for col1 in df1_aligned.columns:
            for col2 in df2_aligned.columns:
                corr = df1_aligned[col1].corr(df2_aligned[col2])
                if not np.isnan(corr):
                    cross_sector_corrs.append(corr)

if cross_sector_corrs:
    mean_cross_corr = np.mean(cross_sector_corrs)
    print(f"{mean_cross_corr:.3f} (n={len(cross_sector_corrs)} pairs)")
else:
    print("N/A")

# Comparison
if within_sector_corrs and cross_sector_corrs:
    mean_within = np.mean(within_sector_corrs)
    mean_cross = np.mean(cross_sector_corrs)

    print(f"\n{'='*50}")
    print(f"Within-sector correlation: {mean_within:.3f}")
    print(f"Cross-sector correlation:  {mean_cross:.3f}")
    print(f"Ratio (within/cross):      {mean_within/mean_cross:.2f}x")
    print(f"{'='*50}")

    if mean_within > 1.5 * mean_cross:
        print("\n✅ GOOD! Within-sector correlations are much stronger.")
        print("   Sector-specific topology should produce cleaner signals.")
    else:
        print("\n⚠️  Within-sector correlations only slightly stronger.")
        print("   Sector benefit may be limited.")

# ============================================================================
# FAILED TICKERS
# ============================================================================

if failed_tickers:
    print("\n" + "=" * 80)
    print("FAILED DOWNLOADS")
    print("=" * 80)

    for sector, ticker in failed_tickers:
        print(f"  {sector:12s}: {ticker}")

    print(f"\nTotal failed: {len(failed_tickers)} / {sum(len(v) for v in SECTORS.values())}")

# ============================================================================
# NEXT STEPS
# ============================================================================

print("\n" + "=" * 80)
print("DOWNLOAD COMPLETE")
print("=" * 80)

print(f"\n📊 Downloaded {len(sector_data)} sectors")
print(f"📁 Saved to: {DATA_DIR}")

print("\nFiles created:")
for sector_name in sector_data.keys():
    print(f"  - sector_{sector_name.lower()}_returns.csv")
print(f"  - sector_download_summary.csv")

print("\nNext steps:")
print("  1. Run 02_compute_sector_topology.py to compute topology for each sector")
print("  2. Run 03_sector_pairs_trading.py to test sector-specific strategies")
print("  3. Run 04_compare_sectors.py to identify best sectors")
print("  4. Run 05_visualize_sectors.py to create publication figures")
