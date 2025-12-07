"""
Distribution Statistics Calculator - Pre-compute statistical distributions for filters.

Calculates and persists:
- Gap % distribution (today's open vs yesterday's close)
- Daily range % distribution (high-low / low)
- Prior days trend statistics

These stats are cached per ticker and regenerated when new data is downloaded.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, asdict
from typing import Dict, Optional, Tuple
from pathlib import Path
import json
from datetime import datetime


@dataclass
class GapStats:
    """Statistics for gap % distribution."""
    mean: float
    std: float
    median: float
    # Percentile boundaries
    p16: float  # -1 std dev (lower bound of 68%)
    p84: float  # +1 std dev (upper bound of 68%)
    p5: float   # 5th percentile (extreme low)
    p95: float  # 95th percentile (extreme high)
    p25: float  # 25th percentile
    p75: float  # 75th percentile
    n_samples: int

    def in_middle_68(self, gap_pct: float) -> bool:
        """Check if gap is within middle 68% (1 std dev)."""
        return self.p16 <= gap_pct <= self.p84

    def in_lower_tail(self, gap_pct: float) -> bool:
        """Check if gap is in lower 16% tail."""
        return gap_pct < self.p16

    def in_upper_tail(self, gap_pct: float) -> bool:
        """Check if gap is in upper 16% tail."""
        return gap_pct > self.p84


@dataclass
class RangeStats:
    """Statistics for daily range % distribution."""
    mean: float
    std: float
    median: float
    # Percentile boundaries
    p16: float  # -1 std dev (lower bound of 68%)
    p84: float  # +1 std dev (upper bound of 68%)
    p50: float  # Median
    p68: float  # 68th percentile
    p90: float  # 90th percentile (high volatility)
    n_samples: int

    def in_middle_68(self, range_pct: float) -> bool:
        """Check if range is within middle 68% (1 std dev)."""
        return self.p16 <= range_pct <= self.p84

    def above_68th_percentile(self, range_pct: float) -> bool:
        """Check if range is above 68th percentile (higher volatility)."""
        return range_pct > self.p68

    def below_median(self, range_pct: float) -> bool:
        """Check if range is below median (lower volatility)."""
        return range_pct < self.p50


@dataclass
class TickerDistributionStats:
    """All distribution statistics for a ticker."""
    ticker: str
    gap_stats: GapStats
    range_stats: RangeStats
    computed_date: str
    data_file_mtime: float  # Modification time of source data file
    n_trading_days: int

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            'ticker': self.ticker,
            'gap_stats': asdict(self.gap_stats),
            'range_stats': asdict(self.range_stats),
            'computed_date': self.computed_date,
            'data_file_mtime': self.data_file_mtime,
            'n_trading_days': self.n_trading_days
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'TickerDistributionStats':
        """Create from dictionary."""
        return cls(
            ticker=data['ticker'],
            gap_stats=GapStats(**data['gap_stats']),
            range_stats=RangeStats(**data['range_stats']),
            computed_date=data['computed_date'],
            data_file_mtime=data['data_file_mtime'],
            n_trading_days=data['n_trading_days']
        )


class DistributionStatsCalculator:
    """
    Calculate and cache distribution statistics for tickers.

    Stats are persisted to JSON and only recomputed when:
    1. No cached stats exist
    2. Source data file has been modified (new data downloaded)
    """

    def __init__(self, data_dir: str, cache_dir: Optional[str] = None):
        """
        Initialize calculator.

        Args:
            data_dir: Directory containing market data files
            cache_dir: Directory to store cached stats (defaults to data_dir/stats_cache)
        """
        self.data_dir = Path(data_dir)
        self.cache_dir = Path(cache_dir) if cache_dir else self.data_dir / "stats_cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # In-memory cache
        self._stats_cache: Dict[str, TickerDistributionStats] = {}

    def get_stats(self, ticker: str, force_recalc: bool = False) -> Optional[TickerDistributionStats]:
        """
        Get distribution stats for a ticker, computing if necessary.

        Args:
            ticker: Ticker symbol
            force_recalc: Force recalculation even if cache is valid

        Returns:
            TickerDistributionStats or None if data not available
        """
        # Check memory cache first
        if ticker in self._stats_cache and not force_recalc:
            cached = self._stats_cache[ticker]
            if self._is_cache_valid(ticker, cached):
                return cached

        # Check file cache
        cache_file = self.cache_dir / f"{ticker}_dist_stats.json"
        if cache_file.exists() and not force_recalc:
            try:
                with open(cache_file, 'r') as f:
                    data = json.load(f)
                stats = TickerDistributionStats.from_dict(data)
                if self._is_cache_valid(ticker, stats):
                    self._stats_cache[ticker] = stats
                    return stats
            except (json.JSONDecodeError, KeyError):
                pass  # Cache corrupted, recalculate

        # Calculate fresh stats
        stats = self._calculate_stats(ticker)
        if stats:
            self._stats_cache[ticker] = stats
            self._save_to_cache(ticker, stats)

        return stats

    def _is_cache_valid(self, ticker: str, stats: TickerDistributionStats) -> bool:
        """Check if cached stats are still valid (data file hasn't changed)."""
        data_file = self._find_data_file(ticker)
        if not data_file:
            return False

        current_mtime = data_file.stat().st_mtime
        return abs(current_mtime - stats.data_file_mtime) < 1.0  # 1 second tolerance

    def _find_data_file(self, ticker: str) -> Optional[Path]:
        """Find the data file for a ticker."""
        # Try NinjaTrader format first
        nt_file = self.data_dir / f"{ticker}_NT.txt"
        if nt_file.exists():
            return nt_file

        # Try CSV format
        csv_file = self.data_dir / f"{ticker}.csv"
        if csv_file.exists():
            return csv_file

        return None

    def _calculate_stats(self, ticker: str) -> Optional[TickerDistributionStats]:
        """Calculate distribution statistics from market data."""
        data_file = self._find_data_file(ticker)
        if not data_file:
            return None

        # Load data
        try:
            if data_file.suffix == '.txt':
                # NinjaTrader format
                df = self._load_ninjatrader(data_file)
            else:
                df = pd.read_csv(data_file)

            if df.empty:
                return None

            # Calculate daily OHLC
            daily_df = self._compute_daily_ohlc(df)
            if len(daily_df) < 20:  # Need enough data
                return None

            # Calculate gap % (today's open vs yesterday's close)
            daily_df['prev_close'] = daily_df['close'].shift(1)
            daily_df['gap_pct'] = (daily_df['open'] - daily_df['prev_close']) / daily_df['prev_close'] * 100
            daily_df = daily_df.dropna()

            # Calculate daily range %
            daily_df['range_pct'] = (daily_df['high'] - daily_df['low']) / daily_df['low'] * 100

            # Compute gap statistics
            gaps = daily_df['gap_pct'].values
            gap_stats = GapStats(
                mean=float(np.mean(gaps)),
                std=float(np.std(gaps)),
                median=float(np.median(gaps)),
                p16=float(np.percentile(gaps, 16)),
                p84=float(np.percentile(gaps, 84)),
                p5=float(np.percentile(gaps, 5)),
                p95=float(np.percentile(gaps, 95)),
                p25=float(np.percentile(gaps, 25)),
                p75=float(np.percentile(gaps, 75)),
                n_samples=len(gaps)
            )

            # Compute range statistics
            ranges = daily_df['range_pct'].values
            range_stats = RangeStats(
                mean=float(np.mean(ranges)),
                std=float(np.std(ranges)),
                median=float(np.median(ranges)),
                p16=float(np.percentile(ranges, 16)),
                p84=float(np.percentile(ranges, 84)),
                p50=float(np.percentile(ranges, 50)),
                p68=float(np.percentile(ranges, 68)),
                p90=float(np.percentile(ranges, 90)),
                n_samples=len(ranges)
            )

            return TickerDistributionStats(
                ticker=ticker,
                gap_stats=gap_stats,
                range_stats=range_stats,
                computed_date=datetime.now().isoformat(),
                data_file_mtime=data_file.stat().st_mtime,
                n_trading_days=len(daily_df)
            )

        except Exception as e:
            print(f"Error calculating stats for {ticker}: {e}")
            return None

    def _load_ninjatrader(self, filepath: Path) -> pd.DataFrame:
        """Load NinjaTrader format file."""
        df = pd.read_csv(
            filepath,
            sep=';',
            header=None,
            names=['datetime', 'open', 'high', 'low', 'close', 'volume', 'oi']
        )
        df['timestamp'] = pd.to_datetime(df['datetime'], format='%Y%m%d %H%M%S')
        return df

    def _compute_daily_ohlc(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Aggregate minute bars to daily OHLC using RTH only.

        Gap is calculated as: (today's 9:30 open - yesterday's 16:00 close) / yesterday's close
        This ignores extended hours trading.
        """
        if 'timestamp' not in df.columns:
            df['timestamp'] = pd.to_datetime(df['datetime'])

        df = df.copy()
        df['date'] = df['timestamp'].dt.date
        df['time'] = df['timestamp'].dt.time

        from datetime import time as dt_time

        # Filter to regular trading hours only (9:30-16:00 ET)
        rth_df = df[(df['time'] >= dt_time(9, 30)) & (df['time'] <= dt_time(16, 0))]

        # Aggregate to daily
        daily = rth_df.groupby('date').agg({
            'open': 'first',   # First bar of RTH (9:30)
            'high': 'max',
            'low': 'min',
            'close': 'last',   # Last bar of RTH (16:00)
            'volume': 'sum'
        }).reset_index()

        return daily

    def _save_to_cache(self, ticker: str, stats: TickerDistributionStats):
        """Save stats to file cache."""
        cache_file = self.cache_dir / f"{ticker}_dist_stats.json"
        with open(cache_file, 'w') as f:
            json.dump(stats.to_dict(), f, indent=2)

    def invalidate_cache(self, ticker: str):
        """Invalidate cache for a ticker (call after new data download)."""
        if ticker in self._stats_cache:
            del self._stats_cache[ticker]

        cache_file = self.cache_dir / f"{ticker}_dist_stats.json"
        if cache_file.exists():
            cache_file.unlink()

    def get_all_cached_tickers(self) -> list:
        """Get list of tickers with cached stats."""
        return [f.stem.replace('_dist_stats', '')
                for f in self.cache_dir.glob('*_dist_stats.json')]


# Filter functions for use in backtester

def check_gap_filter(
    gap_pct: float,
    gap_stats: GapStats,
    filter_mode: str,
    trade_direction: str
) -> bool:
    """
    Check if trade passes gap filter.

    Args:
        gap_pct: Today's gap percentage
        gap_stats: Precomputed gap statistics
        filter_mode: One of:
            - 'middle_68': Only trade gaps within 1 std dev
            - 'exclude_middle_68': Only trade extreme gaps
            - 'directional': Gap up = longs only, gap down = shorts only
            - 'reverse_directional': Gap up = shorts only, gap down = longs only
            - 'any': No filter
        trade_direction: 'long' or 'short'

    Returns:
        True if trade passes filter
    """
    if filter_mode == 'any':
        return True

    if filter_mode == 'middle_68':
        return gap_stats.in_middle_68(gap_pct)

    if filter_mode == 'exclude_middle_68':
        return not gap_stats.in_middle_68(gap_pct)

    if filter_mode == 'directional':
        # Gap up -> longs only, Gap down -> shorts only
        if gap_pct > 0 and trade_direction == 'short':
            return False
        if gap_pct < 0 and trade_direction == 'long':
            return False
        return True

    if filter_mode == 'reverse_directional':
        # Gap up -> shorts only (fade the gap), Gap down -> longs only
        if gap_pct > 0 and trade_direction == 'long':
            return False
        if gap_pct < 0 and trade_direction == 'short':
            return False
        return True

    return True


def check_range_filter(
    range_pct: float,
    range_stats: RangeStats,
    filter_mode: str
) -> bool:
    """
    Check if trade passes range (volatility) filter.

    Args:
        range_pct: Today's daily range percentage
        range_stats: Precomputed range statistics
        filter_mode: One of:
            - 'middle_68': Only trade when range is within 1 std dev
            - 'above_68': Only trade high volatility days (> 68th percentile)
            - 'below_median': Only trade low volatility days
            - 'middle_68_or_below': Trade if in middle 68% OR below
            - 'any': No filter

    Returns:
        True if trade passes filter
    """
    if filter_mode == 'any':
        return True

    if filter_mode == 'middle_68':
        return range_stats.in_middle_68(range_pct)

    if filter_mode == 'above_68':
        return range_stats.above_68th_percentile(range_pct)

    if filter_mode == 'below_median':
        return range_stats.below_median(range_pct)

    if filter_mode == 'middle_68_or_below':
        # Trade if normal volatility OR below normal
        return range_pct <= range_stats.p84

    return True


def check_trend_filter(
    prior_days_bullish: int,
    prior_days_lookback: int,
    filter_mode: str,
    trade_direction: str
) -> bool:
    """
    Check if trade passes trend filter.

    Args:
        prior_days_bullish: Number of bullish days in lookback period
        prior_days_lookback: Total lookback days
        filter_mode: One of:
            - 'with_trend': Bullish trend = longs only, Bearish = shorts only
            - 'counter_trend': Bullish trend = shorts only (mean reversion)
            - 'any': No filter
        trade_direction: 'long' or 'short'

    Returns:
        True if trade passes filter
    """
    if filter_mode == 'any':
        return True

    # Determine trend
    bullish_ratio = prior_days_bullish / prior_days_lookback if prior_days_lookback > 0 else 0.5
    is_bullish_trend = bullish_ratio > 0.5
    is_bearish_trend = bullish_ratio < 0.5

    if filter_mode == 'with_trend':
        # Trade with the trend
        if is_bullish_trend and trade_direction == 'short':
            return False
        if is_bearish_trend and trade_direction == 'long':
            return False
        return True

    if filter_mode == 'counter_trend':
        # Trade against the trend (mean reversion)
        if is_bullish_trend and trade_direction == 'long':
            return False
        if is_bearish_trend and trade_direction == 'short':
            return False
        return True

    return True
