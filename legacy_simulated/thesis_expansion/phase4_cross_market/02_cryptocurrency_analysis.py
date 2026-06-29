"""
Phase 4: Cryptocurrency Market Analysis
========================================

Tests if topology works in decentralized, 24/7 crypto markets.

Hypothesis: Crypto has different characteristics than equities:
- Higher volatility (3-5× equity markets)
- Different correlation structure (driven by BTC, not fundamentals)
- 24/7 trading (no market hours)

If topology generalizes, high-correlation crypto groups should still
produce stable features despite extreme volatility.

Cryptocurrencies tested:
- BTC, ETH (large cap, established)
- Top 10 altcoins by market cap

Author: Adam Levine
Date: January 2026
"""

import yfinance as yf
import pandas as pd
import numpy as np
from pathlib import Path
from ripser import ripser
import time
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("PHASE 4: CRYPTOCURRENCY MARKET ANALYSIS")
print("=" * 80)

# Configuration
START_DATE = '2020-01-01'
END_DATE = '2024-12-31'
DATA_DIR = Path(__file__).parent.parent / 'data'

# Cryptocurrency universe (use -USD tickers from yfinance)
CRYPTO_TICKERS = [
    'BTC-USD',   # Bitcoin
    'ETH-USD',   # Ethereum
    'BNB-USD',   # Binance Coin
    'XRP-USD',   # Ripple
    'ADA-USD',   # Cardano
    'DOGE-USD',  # Dogecoin
    'SOL-USD',   # Solana
    'MATIC-USD', # Polygon
    'DOT-USD',   # Polkadot
    'LTC-USD',   # Litecoin
    'AVAX-USD',  # Avalanche
    'LINK-USD',  # Chainlink
]

# ============================================================================
# DOWNLOAD CRYPTO DATA
# ============================================================================

print("\n₿ Downloading cryptocurrency data...")

crypto_returns = []
successful_tickers = []

for ticker in CRYPTO_TICKERS:
    try:
        print(f"  Downloading {ticker}...", end=' ')

        data = yf.download(ticker, start=START_DATE, end=END_DATE,
                         progress=False, show_errors=False)

        if len(data) > 100:
            returns = data['Adj Close'].pct_change()
            returns.name = ticker.replace('-USD', '')
            crypto_returns.append(returns)
            successful_tickers.append(ticker)
            print(f"✅ {len(returns)} days")
        else:
            print(f"⚠️  Insufficient data")

    except Exception as e:
        print(f"❌ Failed: {str(e)[:30]}")
        continue

    time.sleep(0.1)

if len(crypto_returns) < 5:
    print("\n❌ Insufficient crypto data. Need at least 5 cryptos.")
    exit(1)

# Combine
crypto_df = pd.concat(crypto_returns, axis=1)
crypto_df = crypto_df.dropna(how='all')

print(f"\n✅ Downloaded {crypto_df.shape[1]} cryptos, {len(crypto_df)} days")

# Save
crypto_file = DATA_DIR / 'cryptocurrency_returns.csv'
crypto_df.to_csv(crypto_file)

# ============================================================================
# CORRELATION ANALYSIS
# ============================================================================

print("\n" + "=" * 80)
print("CRYPTO CORRELATION ANALYSIS")
print("=" * 80)

corr_matrix = crypto_df.corr()
upper_tri = corr_matrix.where(np.triu(np.ones_like(corr_matrix), k=1).astype(bool))
correlations = upper_tri.stack().values

mean_corr = np.mean(correlations)
std_corr = np.std(correlations)

print(f"\nCrypto Market Statistics:")
print(f"  Mean correlation: {mean_corr:.3f}")
print(f"  Std correlation:  {std_corr:.3f}")
print(f"  Min correlation:  {correlations.min():.3f}")
print(f"  Max correlation:  {correlations.max():.3f}")

# Compare to equities
us_tech_file = DATA_DIR / 'sector_technology_returns.csv'
if us_tech_file.exists():
    us_tech = pd.read_csv(us_tech_file, index_col=0, parse_dates=True)
    us_corr_matrix = us_tech.corr()
    us_upper = us_corr_matrix.where(np.triu(np.ones_like(us_corr_matrix), k=1).astype(bool))
    us_mean_corr = us_upper.stack().mean()

    print(f"\n📊 Comparison to US Tech Equities:")
    print(f"  Equities mean correlation: {us_mean_corr:.3f}")
    print(f"  Crypto mean correlation:   {mean_corr:.3f}")
    print(f"  Ratio (crypto/equity):     {mean_corr/us_mean_corr:.2f}x")

    if mean_corr < us_mean_corr:
        print("\n⚠️  Crypto correlations WEAKER than equities")
        print("   This may lead to noisier topology (Section 7 prediction)")
    else:
        print("\n✅ Crypto correlations COMPARABLE to equities")

# Volatility analysis
crypto_vols = crypto_df.std() * np.sqrt(365)  # Annualize (crypto trades 365 days)
mean_vol = crypto_vols.mean()

print(f"\nCrypto Volatility:")
print(f"  Mean annualized volatility: {mean_vol:.1%}")

if us_tech_file.exists():
    us_vols = us_tech.std() * np.sqrt(252)
    us_mean_vol = us_vols.mean()
    print(f"  US tech volatility:         {us_mean_vol:.1%}")
    print(f"  Ratio (crypto/equity):      {mean_vol/us_mean_vol:.1f}x")

# ============================================================================
# COMPUTE CRYPTO TOPOLOGY
# ============================================================================

print("\n" + "=" * 80)
print("COMPUTING CRYPTOCURRENCY TOPOLOGY")
print("=" * 80)

def compute_topology(returns_window):
    """Compute H0 and H1 features."""
    try:
        corr = returns_window.corr()
        dist = np.sqrt(2 * (1 - corr.values))

        result = ripser(dist, maxdim=1, distance_matrix=True, thresh=0.5)

        h1_dgm = result['dgms'][1]
        h1_count = len(h1_dgm)
        h1_persistence = (h1_dgm[:, 1] - h1_dgm[:, 0]).sum() if len(h1_dgm) > 0 else 0

        return {'h1_count': h1_count, 'h1_persistence': h1_persistence}
    except:
        return None

print(f"\n₿ Computing topology for crypto market...")

topology_features = []

for i in range(60, len(crypto_df)):
    window = crypto_df.iloc[i-60:i]

    topo = compute_topology(window)
    if topo:
        topo['date'] = crypto_df.index[i]
        topology_features.append(topo)

    if (i - 60) % 100 == 0:
        pct = (i - 60) / (len(crypto_df) - 60) * 100
        print(f"  Progress: {pct:.1f}%", end='\r')

print(f"  Progress: 100.0%")

crypto_topology = pd.DataFrame(topology_features).set_index('date')

# Save
topo_file = DATA_DIR / 'cryptocurrency_topology.csv'
crypto_topology.to_csv(topo_file)

# ============================================================================
# STABILITY ANALYSIS
# ============================================================================

print("\n" + "=" * 80)
print("CRYPTO TOPOLOGY STABILITY")
print("=" * 80)

mean_h1 = crypto_topology['h1_count'].mean()
std_h1 = crypto_topology['h1_count'].std()
cv_h1 = std_h1 / mean_h1

print(f"\nCrypto Topology Statistics:")
print(f"  Mean H₁ loops:    {mean_h1:.2f}")
print(f"  Std H₁ loops:     {std_h1:.2f}")
print(f"  CV:               {cv_h1:.3f}")

# Compare to US tech
us_topo_file = DATA_DIR / 'sector_technology_topology.csv'
if us_topo_file.exists():
    us_topo = pd.read_csv(us_topo_file, index_col=0, parse_dates=True)
    us_cv = us_topo['h1_count'].std() / us_topo['h1_count'].mean()

    print(f"\n📊 Comparison to US Tech:")
    print(f"  US tech CV:       {us_cv:.3f}")
    print(f"  Crypto CV:        {cv_h1:.3f}")
    print(f"  Difference:       {(cv_h1 - us_cv)/us_cv*100:+.1f}%")

    if cv_h1 > us_cv * 1.2:
        print("\n⚠️  Crypto topology is LESS stable (higher CV)")
        print("   Consistent with lower correlations (Section 7 finding)")
    elif cv_h1 < us_cv * 0.8:
        print("\n✅ Crypto topology is MORE stable (lower CV)")
        print("   Surprising given higher volatility!")
    else:
        print("\n🟡 Crypto topology stability COMPARABLE to equities")

# ============================================================================
# CORRELATION-CV RELATIONSHIP TEST
# ============================================================================

print("\n" + "=" * 80)
print("CORRELATION-CV RELATIONSHIP TEST")
print("=" * 80)

print(f"\nCrypto market:")
print(f"  Mean correlation: {mean_corr:.3f}")
print(f"  Topology CV:      {cv_h1:.3f}")

# Predict CV based on Section 7 relationship (ρ = -0.87)
# CV ≈ a - b × correlation (from linear regression on sector data)
# Using US tech as calibration: corr = 0.58, CV = 0.451
# Simple prediction: CV decreases as correlation increases

predicted_cv_change = (us_mean_corr - mean_corr) * 0.5  # Rough estimate
predicted_crypto_cv = us_cv + predicted_cv_change

print(f"\n📊 Prediction based on Section 7:")
print(f"  Predicted crypto CV: {predicted_crypto_cv:.3f}")
print(f"  Actual crypto CV:    {cv_h1:.3f}")
print(f"  Error:               {abs(cv_h1 - predicted_crypto_cv):.3f}")

if abs(cv_h1 - predicted_crypto_cv) < 0.2:
    print("\n✅ Prediction ACCURATE! Relationship generalizes to crypto.")
else:
    print("\n⚠️  Prediction less accurate for crypto")
    print("   Crypto may have unique characteristics (24/7 trading, different drivers)")

# ============================================================================
# SAVE SUMMARY
# ============================================================================

summary_data = {
    'Asset Class': 'Cryptocurrency',
    'Mean Correlation': mean_corr,
    'Mean H₁': mean_h1,
    'CV': cv_h1,
    'Cryptos': crypto_df.shape[1],
    'Days': len(crypto_df)
}

summary_df = pd.DataFrame([summary_data])
summary_file = DATA_DIR / 'phase4_crypto_summary.csv'
summary_df.to_csv(summary_file, index=False)

print(f"\n💾 Summary saved: {summary_file}")

# ============================================================================
# KEY FINDINGS
# ============================================================================

print("\n" + "=" * 80)
print("KEY FINDINGS")
print("=" * 80)

print("\n1. Correlation Structure:")
if mean_corr < 0.5:
    print(f"   ⚠️  Crypto correlations ({mean_corr:.3f}) below 0.5 threshold")
    print("   Section 7 recommends AGAINST using TDA for trading")
else:
    print(f"   ✅ Crypto correlations ({mean_corr:.3f}) sufficient for TDA")

print("\n2. Topology Stability:")
if cv_h1 < 0.6:
    print(f"   ✅ CV ({cv_h1:.3f}) indicates stable topology")
elif cv_h1 < 0.8:
    print(f"   🟡 CV ({cv_h1:.3f}) indicates moderate stability")
else:
    print(f"   ❌ CV ({cv_h1:.3f}) indicates unstable topology")

print("\n3. Generalization:")
print("   Testing if correlation-CV relationship from Section 7 holds...")

print("\nNext: Run 03_cross_market_comparison.py to compare all asset classes")
