"""
Data Updater - Automated daily data updates from Polygon.io.

Fetches new market data and appends to existing files,
tracking when data was last updated.
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta, date
from typing import List, Optional, Dict, Tuple
from pathlib import Path
import json
import os
import time


@dataclass
class DataUpdateConfig:
    """Configuration for data updates."""
    api_key: Optional[str] = None  # Polygon.io API key
    data_dir: str = ""  # Directory for market data files
    tickers: List[str] = field(default_factory=list)  # Tickers to update
    update_time: str = "18:00"  # Time to run daily updates (after market close)
    rate_limit_seconds: float = 13.0  # Seconds between API calls (free tier)
    max_retries: int = 3  # Retries on failure
    backfill_days: int = 5  # Days to look back for missing data


@dataclass
class UpdateResult:
    """Result of a data update attempt."""
    ticker: str
    success: bool
    bars_added: int
    start_date: Optional[date]
    end_date: Optional[date]
    error_message: str = ""
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'ticker': self.ticker,
            'success': self.success,
            'bars_added': self.bars_added,
            'start_date': self.start_date.isoformat() if self.start_date else None,
            'end_date': self.end_date.isoformat() if self.end_date else None,
            'error_message': self.error_message,
            'timestamp': self.timestamp.isoformat(),
        }


class DataUpdater:
    """
    Automated data updater for market data files.

    Fetches new data from Polygon.io and appends to existing
    NinjaTrader format files.
    """

    def __init__(
        self,
        config: Optional[DataUpdateConfig] = None,
        status_file: Optional[str] = None,
    ):
        """
        Initialize the data updater.

        Args:
            config: Update configuration
            status_file: Optional file to persist update status
        """
        self.config = config or DataUpdateConfig()

        # Get API key from config or environment
        if not self.config.api_key:
            self.config.api_key = os.environ.get('POLYGON_API_KEY')

        # Default data directory
        if not self.config.data_dir:
            self.config.data_dir = str(
                Path(__file__).parent.parent / "market_data"
            )

        self.status_file = status_file or str(
            Path(__file__).parent.parent / "output" / "data_update_status.json"
        )

        # Track update history
        self.update_history: List[UpdateResult] = []
        self.last_update_by_ticker: Dict[str, datetime] = {}

        # Load persisted status
        self._load_status()

    def _load_status(self):
        """Load persisted update status."""
        try:
            if os.path.exists(self.status_file):
                with open(self.status_file, 'r') as f:
                    data = json.load(f)
                    for ticker, ts in data.get('last_update', {}).items():
                        self.last_update_by_ticker[ticker] = datetime.fromisoformat(ts)
        except Exception:
            pass

    def _save_status(self):
        """Save update status to file."""
        try:
            Path(self.status_file).parent.mkdir(parents=True, exist_ok=True)
            data = {
                'last_update': {
                    t: ts.isoformat()
                    for t, ts in self.last_update_by_ticker.items()
                },
                'recent_updates': [
                    r.to_dict() for r in self.update_history[-20:]
                ],
            }
            with open(self.status_file, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception:
            pass

    def get_last_data_date(self, ticker: str) -> Optional[date]:
        """
        Get the last date of data in the file.

        Args:
            ticker: Ticker symbol

        Returns:
            Last date with data, or None if file doesn't exist
        """
        # Find the data file
        data_path = self._find_data_file(ticker)
        if not data_path or not data_path.exists():
            return None

        # Read last line(s) to find the latest date
        try:
            with open(data_path, 'rb') as f:
                # Seek to end and read backwards to find last line
                f.seek(0, 2)  # End of file
                file_size = f.tell()

                if file_size == 0:
                    return None

                # Read last 1KB or so
                read_size = min(1024, file_size)
                f.seek(file_size - read_size)
                last_chunk = f.read().decode('utf-8', errors='ignore')

            lines = last_chunk.strip().split('\n')
            for line in reversed(lines):
                if ';' in line:
                    # Parse NT format: YYYYMMDD HHMMSS;O;H;L;C;V
                    parts = line.split(';')
                    if len(parts) >= 5:
                        dt_str = parts[0].split()[0]  # Get date part
                        try:
                            return datetime.strptime(dt_str, '%Y%m%d').date()
                        except ValueError:
                            continue
        except Exception:
            pass

        return None

    def _find_data_file(self, ticker: str) -> Optional[Path]:
        """Find the data file for a ticker."""
        data_dir = Path(self.config.data_dir)

        # Check for NT format file first
        nt_file = data_dir / f"{ticker}_NT.txt"
        if nt_file.exists():
            return nt_file

        # Check for other patterns
        for pattern in [f"{ticker}.txt", f"{ticker}_1min*.txt", f"{ticker}.csv"]:
            matches = list(data_dir.glob(pattern))
            if matches:
                return max(matches, key=lambda p: p.stat().st_mtime)

        return None

    def update_ticker(
        self,
        ticker: str,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
        progress_callback=None,
    ) -> UpdateResult:
        """
        Update data for a single ticker.

        Args:
            ticker: Ticker symbol
            start_date: Start date (defaults to last data date + 1)
            end_date: End date (defaults to today)
            progress_callback: Optional callback for progress updates

        Returns:
            UpdateResult with details
        """
        try:
            import requests
        except ImportError:
            return UpdateResult(
                ticker=ticker,
                success=False,
                bars_added=0,
                start_date=None,
                end_date=None,
                error_message="requests library not installed",
            )

        if not self.config.api_key:
            return UpdateResult(
                ticker=ticker,
                success=False,
                bars_added=0,
                start_date=None,
                end_date=None,
                error_message="No Polygon API key configured",
            )

        # Determine date range
        if start_date is None:
            last_date = self.get_last_data_date(ticker)
            if last_date:
                start_date = last_date + timedelta(days=1)
            else:
                # No existing data - download recent history
                start_date = (datetime.now() - timedelta(days=30)).date()

        if end_date is None:
            end_date = datetime.now().date()

        # Skip if start is after end (data is already current)
        if start_date > end_date:
            return UpdateResult(
                ticker=ticker,
                success=True,
                bars_added=0,
                start_date=start_date,
                end_date=end_date,
                error_message="Data already up to date",
            )

        if progress_callback:
            progress_callback(f"Updating {ticker}: {start_date} to {end_date}")

        # Fetch data from Polygon
        bars = self._fetch_polygon_data(
            ticker, start_date, end_date, progress_callback
        )

        if not bars:
            return UpdateResult(
                ticker=ticker,
                success=True,
                bars_added=0,
                start_date=start_date,
                end_date=end_date,
                error_message="No new data available",
            )

        # Append to file
        success = self._append_to_file(ticker, bars)

        result = UpdateResult(
            ticker=ticker,
            success=success,
            bars_added=len(bars),
            start_date=start_date,
            end_date=end_date,
            error_message="" if success else "Failed to write data",
        )

        if success:
            self.last_update_by_ticker[ticker] = datetime.now()
            self._save_status()

        self.update_history.append(result)
        return result

    def _fetch_polygon_data(
        self,
        ticker: str,
        start_date: date,
        end_date: date,
        progress_callback=None,
    ) -> List[Dict]:
        """Fetch data from Polygon.io API."""
        import requests

        all_bars = []
        current_date = datetime.combine(start_date, datetime.min.time())
        end_datetime = datetime.combine(end_date, datetime.max.time())

        base_url = "https://api.polygon.io/v2/aggs/ticker"

        while current_date < end_datetime:
            chunk_end = min(
                current_date + timedelta(days=30),
                end_datetime
            )

            url = (
                f"{base_url}/{ticker}/range/1/minute/"
                f"{current_date.strftime('%Y-%m-%d')}/{chunk_end.strftime('%Y-%m-%d')}"
                f"?adjusted=true&sort=asc&limit=50000&apiKey={self.config.api_key}"
            )

            for attempt in range(self.config.max_retries):
                try:
                    response = requests.get(url, timeout=30)
                    data = response.json()

                    if data.get('status') == 'OK' and data.get('results'):
                        all_bars.extend(data['results'])
                        if progress_callback:
                            progress_callback(
                                f"  Downloaded {len(data['results'])} bars for "
                                f"{current_date.date()} to {chunk_end.date()}"
                            )
                        break
                    elif data.get('status') == 'ERROR':
                        error = data.get('error', 'Unknown error')
                        if progress_callback:
                            progress_callback(f"  API Error: {error}")
                        break
                    else:
                        # No data for this period
                        break

                except Exception as e:
                    if attempt < self.config.max_retries - 1:
                        time.sleep(5)  # Wait before retry
                    else:
                        if progress_callback:
                            progress_callback(f"  Request error: {e}")

            current_date = chunk_end + timedelta(days=1)

            # Rate limiting
            time.sleep(self.config.rate_limit_seconds)

        return all_bars

    def _append_to_file(self, ticker: str, bars: List[Dict]) -> bool:
        """Append new bars to the data file."""
        data_path = self._find_data_file(ticker)

        if data_path is None:
            # Create new file
            data_path = Path(self.config.data_dir) / f"{ticker}_NT.txt"

        try:
            # Get existing last date to avoid duplicates
            last_existing_date = self.get_last_data_date(ticker)

            with open(data_path, 'a') as f:
                for bar in bars:
                    # Polygon timestamp is in milliseconds
                    ts = datetime.fromtimestamp(bar['t'] / 1000)

                    # Skip if this is on or before the last existing date
                    if last_existing_date and ts.date() <= last_existing_date:
                        continue

                    dt_str = ts.strftime('%Y%m%d %H%M%S')
                    line = f"{dt_str};{bar['o']};{bar['h']};{bar['l']};{bar['c']};{int(bar['v'])}\n"
                    f.write(line)

            return True
        except Exception:
            return False

    def update_all_tickers(
        self,
        progress_callback=None,
    ) -> List[UpdateResult]:
        """
        Update all configured tickers.

        Args:
            progress_callback: Optional callback for progress updates

        Returns:
            List of UpdateResult for each ticker
        """
        results = []

        for ticker in self.config.tickers:
            if progress_callback:
                progress_callback(f"\nUpdating {ticker}...")

            result = self.update_ticker(ticker, progress_callback=progress_callback)
            results.append(result)

            if progress_callback:
                if result.success:
                    progress_callback(
                        f"  {ticker}: Added {result.bars_added} bars"
                    )
                else:
                    progress_callback(
                        f"  {ticker}: FAILED - {result.error_message}"
                    )

        return results

    def needs_update(self, ticker: str, max_age_hours: int = 24) -> bool:
        """Check if a ticker needs updating."""
        last_update = self.last_update_by_ticker.get(ticker)
        if last_update is None:
            return True

        age = datetime.now() - last_update
        return age.total_seconds() > max_age_hours * 3600

    def get_update_summary(self) -> Dict:
        """Get summary of data update status."""
        now = datetime.now()

        ticker_status = {}
        for ticker in self.config.tickers:
            last_update = self.last_update_by_ticker.get(ticker)
            last_data = self.get_last_data_date(ticker)

            ticker_status[ticker] = {
                'last_update': last_update.isoformat() if last_update else None,
                'last_data_date': last_data.isoformat() if last_data else None,
                'needs_update': self.needs_update(ticker),
                'hours_since_update': (
                    (now - last_update).total_seconds() / 3600
                    if last_update else None
                ),
            }

        return {
            'tickers': ticker_status,
            'api_key_configured': bool(self.config.api_key),
            'data_dir': self.config.data_dir,
            'recent_updates': [
                r.to_dict() for r in self.update_history[-10:]
            ],
        }

    def discover_tickers(self) -> List[str]:
        """Discover tickers from existing data files."""
        data_dir = Path(self.config.data_dir)
        if not data_dir.exists():
            return []

        tickers = set()

        for f in data_dir.glob("*_NT.txt"):
            # Extract ticker from filename like "TSLA_NT.txt"
            ticker = f.stem.replace("_NT", "")
            tickers.add(ticker)

        for f in data_dir.glob("*.txt"):
            if "_NT" not in f.stem:
                # Try to extract ticker
                parts = f.stem.split("_")
                if parts:
                    tickers.add(parts[0])

        return sorted(tickers)


if __name__ == "__main__":
    # Test the data updater
    print("Testing Data Updater")
    print("=" * 50)

    config = DataUpdateConfig(
        tickers=["TSLA", "AAPL", "QQQ"],
    )

    updater = DataUpdater(config)

    # Check for API key
    if not updater.config.api_key:
        print("Note: No Polygon API key found. Set POLYGON_API_KEY environment variable.")
        print("Testing with mock data instead...\n")
    else:
        print(f"API key configured: {updater.config.api_key[:8]}...")

    # Discover existing tickers
    discovered = updater.discover_tickers()
    print(f"\nDiscovered tickers in data directory: {discovered}")

    # Check last data dates
    print("\nLast data dates:")
    for ticker in discovered[:5]:
        last_date = updater.get_last_data_date(ticker)
        needs = updater.needs_update(ticker)
        print(f"  {ticker}: {last_date} {'(needs update)' if needs else ''}")

    # Get summary
    print("\nUpdate Summary:")
    summary = updater.get_update_summary()
    print(f"  API key configured: {summary['api_key_configured']}")
    print(f"  Data directory: {summary['data_dir']}")

    if updater.config.api_key and discovered:
        print(f"\nWould you like to update {discovered[0]}? (requires API calls)")
        # Uncomment to actually run:
        # result = updater.update_ticker(discovered[0], progress_callback=print)
        # print(f"\nResult: {result.to_dict()}")
