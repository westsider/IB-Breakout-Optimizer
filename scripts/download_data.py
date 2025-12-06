#!/usr/bin/env python
"""
Download historical minute bar data using Polygon.io API.

Usage:
    python download_data.py --ticker AAPL --days 365
    python download_data.py --ticker MSFT,NVDA --days 365
"""

import sys
import argparse
from pathlib import Path
from datetime import datetime, timedelta
import time
import os

sys.path.insert(0, str(Path(__file__).parent.parent))


def download_polygon_data(ticker: str, days: int, output_dir: str, api_key: str = None):
    """
    Download minute bar data from Polygon.io.

    Note: Free tier has rate limits and limited historical access.
    For production use, consider a paid API key.
    """
    try:
        import requests
    except ImportError:
        print("Error: requests library not installed. Run: pip install requests")
        return

    # Try to get API key from environment or argument
    api_key = api_key or os.environ.get('POLYGON_API_KEY')

    if not api_key:
        print("Error: Polygon API key required.")
        print("Set POLYGON_API_KEY environment variable or pass --api-key argument.")
        print("\nYou can get a free API key at: https://polygon.io/")
        return

    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)

    print(f"Downloading {ticker} data from {start_date.date()} to {end_date.date()}")

    # Polygon.io API endpoint for aggregates
    base_url = "https://api.polygon.io/v2/aggs/ticker"

    all_results = []
    current_date = start_date

    # Download in chunks to handle rate limits
    while current_date < end_date:
        chunk_end = min(current_date + timedelta(days=30), end_date)

        url = (
            f"{base_url}/{ticker}/range/1/minute/"
            f"{current_date.strftime('%Y-%m-%d')}/{chunk_end.strftime('%Y-%m-%d')}"
            f"?adjusted=true&sort=asc&limit=50000&apiKey={api_key}"
        )

        try:
            response = requests.get(url)
            data = response.json()

            if data.get('status') == 'OK' and data.get('results'):
                all_results.extend(data['results'])
                print(f"  Downloaded {len(data['results'])} bars for "
                      f"{current_date.date()} to {chunk_end.date()}")
            elif data.get('status') == 'ERROR':
                print(f"  API Error: {data.get('error', 'Unknown error')}")
            else:
                print(f"  No data for {current_date.date()} to {chunk_end.date()}")

        except Exception as e:
            print(f"  Request error: {e}")

        current_date = chunk_end + timedelta(days=1)

        # Rate limiting - free tier is 5 calls/minute, so wait 12+ seconds between calls
        time.sleep(13)

    if not all_results:
        print(f"No data downloaded for {ticker}")
        return

    # Convert to NinjaTrader format and save
    output_path = Path(output_dir) / f"{ticker}_1min_{start_date.strftime('%Y%m%d')}_to_{end_date.strftime('%Y%m%d')}_NT.txt"

    with open(output_path, 'w') as f:
        for bar in all_results:
            # Polygon timestamp is in milliseconds
            ts = datetime.fromtimestamp(bar['t'] / 1000)
            dt_str = ts.strftime('%Y%m%d %H%M%S')

            f.write(f"{dt_str};{bar['o']};{bar['h']};{bar['l']};{bar['c']};{int(bar['v'])}\n")

    print(f"\nSaved {len(all_results)} bars to {output_path}")
    return output_path


def download_from_firstrate(ticker: str, output_dir: str):
    """
    Download free sample data from FirstRate Data.

    Note: FirstRate provides paid historical data services.
    This function downloads any available free samples.
    """
    print(f"FirstRate Data requires a subscription for historical data.")
    print(f"Visit https://firstratedata.com for pricing.")


def main():
    parser = argparse.ArgumentParser(description='Download historical minute bar data')
    parser.add_argument('--ticker', type=str, required=True,
                        help='Ticker symbol(s), comma-separated for multiple')
    parser.add_argument('--days', type=int, default=365,
                        help='Number of days of history to download')
    parser.add_argument('--output-dir', type=str, default=r'C:\Users\Warren\Downloads',
                        help='Output directory for data files')
    parser.add_argument('--api-key', type=str, default=None,
                        help='Polygon.io API key')
    parser.add_argument('--source', type=str, default='polygon',
                        choices=['polygon', 'firstrate'],
                        help='Data source')

    args = parser.parse_args()

    tickers = [t.strip().upper() for t in args.ticker.split(',')]

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    for ticker in tickers:
        print(f"\n{'='*50}")
        print(f"Processing {ticker}")
        print(f"{'='*50}")

        if args.source == 'polygon':
            download_polygon_data(ticker, args.days, args.output_dir, args.api_key)
        elif args.source == 'firstrate':
            download_from_firstrate(ticker, args.output_dir)


if __name__ == '__main__':
    main()
