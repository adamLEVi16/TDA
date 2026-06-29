"""
Phase 1: Download Intraday Data
================================

This script downloads 5-minute bar data for the same equity universe
used in the original paper, providing ~27x more observations for
robust topological inference.

Data sources:
1. Alpha Vantage API (free tier: 500 calls/day, 2 years history)
2. yfinance (limited to 60 days intraday)

Output:
- intraday_returns_5min.csv: 5-minute returns matrix

Author: Adam Levine
Date: January 2026
"""

import pandas as pd
import numpy as np
import time
import requests
from datetime import datetime, timedelta
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

print("=" * 80)
print("PHASE 1: INTRADAY DATA DOWNLOAD")
print("=" * 80)

# Same universe as original paper
UNIVERSE = [
    'AAPL', 'MSFT', 'AMZN', 'NVDA', 'META', 'GOOG', 'TSLA',
    'NFLX', 'JPM', 'PEP', 'CSCO', 'ORCL', 'DIS', 'BAC',
    'XOM', 'IBM', 'INTC', 'AMD', 'KO', 'WMT'
]

# Output directory
OUTPUT_DIR = Path(__file__).parent.parent / 'data'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print(f"\n📊 Universe: {len(UNIVERSE)} stocks")
print(f"Tickers: {', '.join(UNIVERSE[:5])}... (+ {len(UNIVERSE)-5} more)")

# ============================================================================
# METHOD 1: yfinance (Quick Test - 60 days only)
# ============================================================================

def download_yfinance_intraday(universe, days_back=60):
    """
    Download intraday data using yfinance (limited to 60 days)

    This is faster but only gives recent data. Good for testing.
    """
    print("\n" + "=" * 80)
    print("METHOD 1: yfinance (Last 60 days)")
    print("=" * 80)

    try:
        import yfinance as yf
    except ImportError:
        print("❌ yfinance not installed. Run: pip install yfinance")
        return None

    end_date = datetime.now()
    start_date = end_date - timedelta(days=days_back)

    print(f"\nDownloading {days_back}-day history...")
    print(f"Date range: {start_date.date()} to {end_date.date()}")

    prices_dict = {}
    failed = []

    for ticker in universe:
        try:
            print(f"  {ticker}...", end=" ", flush=True)
            stock = yf.Ticker(ticker)
            data = stock.history(start=start_date, end=end_date, interval='5m')

            if not data.empty and len(data) > 0:
                prices_dict[ticker] = data['Close']
                print(f"✅ {len(data)} bars")
            else:
                print("❌ No data")
                failed.append(ticker)

        except Exception as e:
            print(f"❌ Error: {str(e)[:30]}")
            failed.append(ticker)

    if len(prices_dict) == 0:
        print("\n⚠️  No data downloaded!")
        return None

    # Combine into DataFrame
    prices = pd.DataFrame(prices_dict)

    # Forward fill gaps (missing bars during market hours)
    prices = prices.ffill()

    # Calculate returns
    returns = prices.pct_change().dropna()

    print(f"\n✅ Downloaded {len(returns)} 5-minute bars")
    print(f"   Successful: {len(prices_dict)} stocks")
    if failed:
        print(f"   Failed: {failed}")

    return returns


# ============================================================================
# METHOD 2: Alpha Vantage (Full History - 2 years)
# ============================================================================

def download_alphavantage_intraday(universe, api_key=None):
    """
    Download intraday data using Alpha Vantage API

    Get free API key at: https://www.alphavantage.co/support/#api-key

    Free tier limitations:
    - 500 calls/day
    - 5 calls/minute
    - 2 years of intraday history

    Args:
        universe (list): List of ticker symbols
        api_key (str): Alpha Vantage API key

    Returns:
        pd.DataFrame: Intraday returns
    """

    print("\n" + "=" * 80)
    print("METHOD 2: Alpha Vantage (Full 2-year history)")
    print("=" * 80)

    if api_key is None:
        print("\n⚠️  No API key provided!")
        print("Get free key at: https://www.alphavantage.co/support/#api-key")
        print("\nThen run:")
        print("  python 01_download_intraday_data.py YOUR_API_KEY_HERE")
        return None

    print(f"\nAPI Key: {api_key[:8]}...")
    print(f"Downloading {len(universe)} stocks...")
    print("⏱️  This will take ~4 minutes (rate limit: 5 calls/min)")

    prices_dict = {}
    failed = []

    for idx, ticker in enumerate(universe, 1):
        try:
            print(f"\n[{idx}/{len(universe)}] {ticker}...", end=" ", flush=True)

            # API request
            url = (f'https://www.alphavantage.co/query?'
                   f'function=TIME_SERIES_INTRADAY&'
                   f'symbol={ticker}&'
                   f'interval=5min&'
                   f'outputsize=full&'
                   f'apikey={api_key}&'
                   f'datatype=csv')

            response = requests.get(url)
            response.raise_for_status()

            # Parse CSV
            from io import StringIO
            data = pd.read_csv(StringIO(response.text))

            if 'timestamp' not in data.columns:
                print(f"❌ Invalid response (check API key)")
                failed.append(ticker)
                continue

            # Process
            data['timestamp'] = pd.to_datetime(data['timestamp'])
            data = data.set_index('timestamp').sort_index()

            prices_dict[ticker] = data['close']
            print(f"✅ {len(data)} bars")

            # Rate limit: 5 calls/min
            if idx < len(universe):
                time.sleep(12)  # 12 seconds between calls = 5/min

        except Exception as e:
            print(f"❌ Error: {str(e)[:40]}")
            failed.append(ticker)

    if len(prices_dict) == 0:
        print("\n⚠️  No data downloaded!")
        return None

    # Combine
    prices = pd.DataFrame(prices_dict)

    # Align timestamps (remove partial days)
    prices = prices.ffill().dropna()

    # Calculate returns
    returns = prices.pct_change().dropna()

    print(f"\n✅ Downloaded {len(returns)} 5-minute bars")
    print(f"   Date range: {returns.index[0]} to {returns.index[-1]}")
    print(f"   Successful: {len(prices_dict)} stocks")
    if failed:
        print(f"   Failed: {failed}")

    return returns


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == '__main__':

    # Check for API key argument
    if len(sys.argv) > 1:
        api_key = sys.argv[1]
        print(f"\n🔑 Using provided API key")

        # Download full history
        intraday_returns = download_alphavantage_intraday(UNIVERSE, api_key)

    else:
        print("\n⚡ No API key provided, using yfinance (last 60 days only)")
        print("For full 2-year history, run:")
        print("  python 01_download_intraday_data.py YOUR_ALPHAVANTAGE_KEY")

        # Download recent data only
        intraday_returns = download_yfinance_intraday(UNIVERSE, days_back=60)

    # Save results
    if intraday_returns is not None:
        output_file = OUTPUT_DIR / 'intraday_returns_5min.csv'
        intraday_returns.to_csv(output_file)

        print(f"\n💾 Saved: {output_file}")
        print(f"   Shape: {intraday_returns.shape}")
        print(f"   Size: {output_file.stat().st_size / 1024:.1f} KB")

        # Summary statistics
        print(f"\n📊 Summary:")
        print(f"   Observations: {len(intraday_returns):,}")
        print(f"   Assets: {len(intraday_returns.columns)}")
        print(f"   Date range: {intraday_returns.index[0].date()} to {intraday_returns.index[-1].date()}")
        print(f"   Missing values: {intraday_returns.isna().sum().sum()}")

        # Compare to daily data
        days_equivalent = len(intraday_returns) / 78  # ~78 5-min bars per trading day
        print(f"\n   Equivalent to ~{days_equivalent:.0f} trading days")
        print(f"   vs. 1,494 days in original paper")

        if days_equivalent > 1494:
            improvement = (days_equivalent / 1494 - 1) * 100
            print(f"   ✅ {improvement:.0f}% more effective observations!")
        else:
            print(f"   ⚠️  Limited by yfinance 60-day constraint")
            print(f"   Use Alpha Vantage for full 2-year history")

    else:
        print("\n❌ Download failed - no data saved")

    print("\n" + "=" * 80)
    print("DOWNLOAD COMPLETE")
    print("=" * 80)
    print("\nNext step: Run 02_compute_topology.py")
