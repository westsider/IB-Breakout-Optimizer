"""
Data loader for IB Breakout Optimizer.

Loads 1-minute bar data from various formats:
- NinjaTrader export format: yyyyMMdd HHmmss;open;high;low;close;volume[;openinterest]
- FirstRate Data CSV: yyyy-mm-dd HH:MM,open,high,low,close,volume
- Polygon.io converted format (NT format with 7 columns)
"""

import os
from datetime import datetime
from typing import List, Optional, Dict, Tuple
from pathlib import Path
import pandas as pd
import numpy as np

from .data_types import Bar


class DataLoader:
    """
    Loads and manages historical bar data.

    Supports multiple data formats and provides caching for performance.
    """

    def __init__(self, data_dir: str, timezone: str = "America/New_York"):
        """
        Initialize the data loader.

        Args:
            data_dir: Directory containing data files
            timezone: Timezone for timestamps (default: Eastern Time)
        """
        self.data_dir = Path(data_dir)
        self.timezone = timezone
        self._cache: Dict[str, pd.DataFrame] = {}

    def load_ninjatrader_file(self, filepath: str, ticker: str = "") -> pd.DataFrame:
        """
        Load NinjaTrader format file.

        Format: yyyyMMdd HHmmss;open;high;low;close;volume[;openinterest]
        Example: 20231205 093000;185.12;185.15;185.10;185.14;1523;0

        Args:
            filepath: Path to the data file
            ticker: Ticker symbol (used if not inferrable from filename)

        Returns:
            DataFrame with columns: timestamp, open, high, low, close, volume, ticker
        """
        filepath = Path(filepath)

        if not filepath.exists():
            raise FileNotFoundError(f"Data file not found: {filepath}")

        # Infer ticker from filename if not provided
        if not ticker:
            ticker = filepath.stem.split("_")[0].upper()

        # Read the file
        rows = []
        with open(filepath, 'r') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue

                try:
                    # Split by semicolon
                    parts = line.split(';')
                    # Accept 6 columns (standard) or 7 columns (with open interest)
                    if len(parts) < 6:
                        continue

                    datetime_str = parts[0]
                    open_price = float(parts[1])
                    high_price = float(parts[2])
                    low_price = float(parts[3])
                    close_price = float(parts[4])
                    volume = int(float(parts[5]))  # Handle decimal volumes

                    # Parse datetime: "yyyyMMdd HHmmss"
                    dt = datetime.strptime(datetime_str, "%Y%m%d %H%M%S")

                    rows.append({
                        'timestamp': dt,
                        'open': open_price,
                        'high': high_price,
                        'low': low_price,
                        'close': close_price,
                        'volume': volume,
                        'ticker': ticker
                    })

                except (ValueError, IndexError) as e:
                    # Skip malformed lines
                    continue

        if not rows:
            raise ValueError(f"No valid data found in {filepath}")

        df = pd.DataFrame(rows)
        df = df.sort_values('timestamp').reset_index(drop=True)

        # Cache the data
        self._cache[ticker] = df

        print(f"Loaded {len(df):,} bars for {ticker} from {filepath.name}")
        print(f"  Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")

        return df

    def load_firstrate_csv(self, filepath: str, ticker: str = "") -> pd.DataFrame:
        """
        Load FirstRate Data CSV format.

        Format: yyyy-mm-dd HH:MM,open,high,low,close,volume
        Example: 2024-01-02 04:00,185.12,185.15,185.10,185.14,1523

        Args:
            filepath: Path to the CSV file
            ticker: Ticker symbol

        Returns:
            DataFrame with standard columns
        """
        filepath = Path(filepath)

        if not filepath.exists():
            raise FileNotFoundError(f"Data file not found: {filepath}")

        # Infer ticker from filename
        if not ticker:
            ticker = filepath.stem.split("_")[0].upper()

        # Read CSV
        df = pd.read_csv(filepath, header=None)

        # Assign column names based on number of columns
        if len(df.columns) == 6:
            df.columns = ['datetime', 'open', 'high', 'low', 'close', 'volume']
        elif len(df.columns) == 7:
            # Some formats include ticker
            df.columns = ['datetime', 'open', 'high', 'low', 'close', 'volume', 'ticker_col']
        else:
            raise ValueError(f"Unexpected number of columns: {len(df.columns)}")

        # Parse datetime with multiple format attempts
        datetime_formats = [
            '%Y-%m-%d %H:%M',
            '%Y-%m-%d %H:%M:%S',
            '%m/%d/%Y %H:%M',
            '%m/%d/%Y %H:%M:%S'
        ]

        parsed_dates = None
        for fmt in datetime_formats:
            try:
                parsed_dates = pd.to_datetime(df['datetime'], format=fmt)
                break
            except (ValueError, TypeError):
                continue

        if parsed_dates is None:
            # Fall back to pandas auto-detection
            parsed_dates = pd.to_datetime(df['datetime'])

        df['timestamp'] = parsed_dates
        df['ticker'] = ticker

        # Select and order columns
        df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume', 'ticker']]
        df = df.sort_values('timestamp').reset_index(drop=True)

        # Ensure numeric types
        for col in ['open', 'high', 'low', 'close']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        df['volume'] = pd.to_numeric(df['volume'], errors='coerce').fillna(0).astype(int)

        # Cache
        self._cache[ticker] = df

        print(f"Loaded {len(df):,} bars for {ticker} from {filepath.name}")
        print(f"  Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")

        return df

    def load_auto_detect(self, filepath: str, ticker: str = "") -> pd.DataFrame:
        """
        Auto-detect file format and load.

        Args:
            filepath: Path to data file
            ticker: Ticker symbol

        Returns:
            DataFrame with standard columns
        """
        filepath = Path(filepath)

        if not filepath.exists():
            raise FileNotFoundError(f"Data file not found: {filepath}")

        # Read first line to detect format
        with open(filepath, 'r') as f:
            first_line = f.readline().strip()

        # Check for semicolon delimiter (NinjaTrader format)
        if ';' in first_line:
            return self.load_ninjatrader_file(filepath, ticker)
        # Check for comma delimiter (CSV format)
        elif ',' in first_line:
            return self.load_firstrate_csv(filepath, ticker)
        else:
            raise ValueError(f"Unknown file format: {filepath}")

    def get_bars(self, ticker: str) -> List[Bar]:
        """
        Get bars as a list of Bar objects.

        Args:
            ticker: Ticker symbol

        Returns:
            List of Bar objects
        """
        if ticker not in self._cache:
            raise KeyError(f"Data for {ticker} not loaded. Call load_* method first.")

        df = self._cache[ticker]
        bars = []

        for _, row in df.iterrows():
            bar = Bar(
                timestamp=row['timestamp'].to_pydatetime() if hasattr(row['timestamp'], 'to_pydatetime') else row['timestamp'],
                open=row['open'],
                high=row['high'],
                low=row['low'],
                close=row['close'],
                volume=int(row['volume']),
                ticker=row['ticker']
            )
            bars.append(bar)

        return bars

    def get_dataframe(self, ticker: str) -> pd.DataFrame:
        """
        Get cached DataFrame for a ticker.

        Args:
            ticker: Ticker symbol

        Returns:
            DataFrame with bar data
        """
        if ticker not in self._cache:
            raise KeyError(f"Data for {ticker} not loaded.")
        return self._cache[ticker].copy()

    def get_date_range(self, ticker: str) -> Tuple[datetime, datetime]:
        """
        Get date range for loaded data.

        Args:
            ticker: Ticker symbol

        Returns:
            Tuple of (start_date, end_date)
        """
        if ticker not in self._cache:
            raise KeyError(f"Data for {ticker} not loaded.")

        df = self._cache[ticker]
        return df['timestamp'].min(), df['timestamp'].max()

    def get_trading_days(self, ticker: str) -> List[datetime]:
        """
        Get list of unique trading days.

        Args:
            ticker: Ticker symbol

        Returns:
            List of trading dates
        """
        if ticker not in self._cache:
            raise KeyError(f"Data for {ticker} not loaded.")

        df = self._cache[ticker]
        dates = df['timestamp'].dt.date.unique()
        return sorted([datetime.combine(d, datetime.min.time()) for d in dates])

    def filter_regular_hours(self, ticker: str, market_open: str = "09:30",
                             market_close: str = "16:00") -> pd.DataFrame:
        """
        Filter data to regular trading hours only.

        Args:
            ticker: Ticker symbol
            market_open: Market open time (HH:MM)
            market_close: Market close time (HH:MM)

        Returns:
            Filtered DataFrame
        """
        if ticker not in self._cache:
            raise KeyError(f"Data for {ticker} not loaded.")

        df = self._cache[ticker].copy()

        # Parse time strings
        open_time = datetime.strptime(market_open, "%H:%M").time()
        close_time = datetime.strptime(market_close, "%H:%M").time()

        # Filter by time
        df['time'] = df['timestamp'].dt.time
        mask = (df['time'] >= open_time) & (df['time'] < close_time)
        df = df[mask].drop(columns=['time'])

        return df.reset_index(drop=True)

    def get_bars_for_date(self, ticker: str, date: datetime) -> List[Bar]:
        """
        Get all bars for a specific date.

        Args:
            ticker: Ticker symbol
            date: Date to filter

        Returns:
            List of Bar objects for that date
        """
        if ticker not in self._cache:
            raise KeyError(f"Data for {ticker} not loaded.")

        df = self._cache[ticker]
        target_date = date.date() if isinstance(date, datetime) else date

        mask = df['timestamp'].dt.date == target_date
        day_df = df[mask]

        bars = []
        for _, row in day_df.iterrows():
            bar = Bar(
                timestamp=row['timestamp'].to_pydatetime() if hasattr(row['timestamp'], 'to_pydatetime') else row['timestamp'],
                open=row['open'],
                high=row['high'],
                low=row['low'],
                close=row['close'],
                volume=int(row['volume']),
                ticker=row['ticker']
            )
            bars.append(bar)

        return bars

    def is_loaded(self, ticker: str) -> bool:
        """Check if data for ticker is loaded."""
        return ticker in self._cache

    def clear_cache(self, ticker: Optional[str] = None):
        """
        Clear cached data.

        Args:
            ticker: Specific ticker to clear, or None to clear all
        """
        if ticker:
            if ticker in self._cache:
                del self._cache[ticker]
        else:
            self._cache.clear()

    def summary(self) -> Dict:
        """
        Get summary of loaded data.

        Returns:
            Dict with ticker -> {bars, start, end} mapping
        """
        summary = {}
        for ticker, df in self._cache.items():
            summary[ticker] = {
                'bars': len(df),
                'start': df['timestamp'].min(),
                'end': df['timestamp'].max(),
                'trading_days': df['timestamp'].dt.date.nunique()
            }
        return summary


def load_multiple_tickers(data_dir: str, tickers: List[str],
                          file_pattern: str = "{ticker}_1min_NT.txt") -> DataLoader:
    """
    Convenience function to load multiple tickers.

    Args:
        data_dir: Directory containing data files
        tickers: List of ticker symbols
        file_pattern: Pattern for filenames, with {ticker} placeholder

    Returns:
        DataLoader with all tickers loaded
    """
    loader = DataLoader(data_dir)

    for ticker in tickers:
        filename = file_pattern.format(ticker=ticker)
        filepath = os.path.join(data_dir, filename)

        if os.path.exists(filepath):
            try:
                loader.load_auto_detect(filepath, ticker)
            except Exception as e:
                print(f"Warning: Failed to load {ticker}: {e}")
        else:
            print(f"Warning: File not found for {ticker}: {filepath}")

    return loader


if __name__ == "__main__":
    # Test the data loader
    import sys

    # Default test path
    test_dir = r"C:\Users\Warren\Downloads"

    # Try to load any available data
    loader = DataLoader(test_dir)

    # Look for NinjaTrader format files
    for filename in os.listdir(test_dir):
        if filename.endswith("_NT.txt"):
            try:
                filepath = os.path.join(test_dir, filename)
                df = loader.load_ninjatrader_file(filepath)
                print(f"\nSample data from {filename}:")
                print(df.head())
                print(f"\nDate range: {df['timestamp'].min()} to {df['timestamp'].max()}")
                break
            except Exception as e:
                print(f"Error loading {filename}: {e}")
