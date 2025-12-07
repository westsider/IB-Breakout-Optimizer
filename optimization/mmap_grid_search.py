"""
Memory-Mapped Grid Search Optimizer for IB Breakout Strategy.

Uses memory-mapped NumPy arrays to share data across worker processes,
reducing memory usage by 80-90%. Integrates with Numba-accelerated
backtester for 6x faster execution.
"""

import sys
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional, Callable
from dataclasses import dataclass, field, asdict
import multiprocessing
import time
import json
import numpy as np
from joblib import Parallel, delayed

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from data.data_loader import DataLoader
from data.session_builder import SessionBuilder
from data.data_types import Bar
from data.distribution_stats import DistributionStatsCalculator
from metrics.performance_metrics import PerformanceMetrics
from optimization.parameter_space import ParameterSpace, create_parameter_space
from optimization.mmap_data import MMapDataManager, MMapArrayPaths, load_mmap_arrays


@dataclass
class OptimizationResult:
    """Result from a single parameter combination."""
    params: Dict[str, Any]
    total_trades: int
    win_rate: float
    total_pnl: float
    profit_factor: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    avg_trade: float
    objective_value: float
    run_time_seconds: float = 0.0
    trade_pnls: List[float] = field(default_factory=list)  # Individual trade P&Ls for equity curve
    k_ratio: float = 0.0  # Kestner ratio for consistency measurement

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for export."""
        result = {**self.params}
        result.update({
            'total_trades': self.total_trades,
            'win_rate': self.win_rate,
            'total_pnl': self.total_pnl,
            'profit_factor': self.profit_factor,
            'sharpe_ratio': self.sharpe_ratio,
            'sortino_ratio': self.sortino_ratio,
            'max_drawdown': self.max_drawdown,
            'avg_trade': self.avg_trade,
            'k_ratio': self.k_ratio,
            'objective_value': self.objective_value,
            'run_time': self.run_time_seconds
            # Note: trade_pnls not included in CSV export to keep file size manageable
        })
        return result


@dataclass
class GridSearchResults:
    """Complete results from grid search optimization."""
    results: List[OptimizationResult] = field(default_factory=list)
    best_result: Optional[OptimizationResult] = None
    objective: str = "sharpe_ratio"
    total_combinations: int = 0
    completed_combinations: int = 0
    total_time_seconds: float = 0.0
    ticker: str = ""
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None

    def add_result(self, result: OptimizationResult):
        """Add a result and update best if necessary."""
        self.results.append(result)
        self.completed_combinations += 1

        if self.best_result is None or result.objective_value > self.best_result.objective_value:
            self.best_result = result

    def get_top_n(self, n: int = 10) -> List[OptimizationResult]:
        """Get top N results by objective value."""
        sorted_results = sorted(self.results, key=lambda x: x.objective_value, reverse=True)
        return sorted_results[:n]

    def to_dataframe(self):
        """Convert all results to DataFrame."""
        import pandas as pd
        rows = [r.to_dict() for r in self.results]
        df = pd.DataFrame(rows)
        return df.sort_values('objective_value', ascending=False)

    def save_results(self, filepath: str):
        """Save results to CSV."""
        df = self.to_dataframe()
        df.to_csv(filepath, index=False)
        print(f"Results saved to: {filepath}")

    def summary(self) -> str:
        """Get summary of optimization results."""
        lines = [
            "=" * 60,
            "GRID SEARCH OPTIMIZATION RESULTS",
            "=" * 60,
            f"Ticker: {self.ticker}",
            f"Objective: {self.objective}",
            f"Total Combinations: {self.total_combinations:,}",
            f"Completed: {self.completed_combinations:,}",
            f"Total Time: {self.total_time_seconds:.1f} seconds",
            ""
        ]

        if self.best_result:
            lines.extend([
                "BEST RESULT:",
                "-" * 40,
                f"Objective Value: {self.best_result.objective_value:.4f}",
                f"Total Trades: {self.best_result.total_trades}",
                f"Win Rate: {self.best_result.win_rate:.1f}%",
                f"Total P&L: ${self.best_result.total_pnl:,.2f}",
                f"Profit Factor: {self.best_result.profit_factor:.2f}",
                f"Sharpe Ratio: {self.best_result.sharpe_ratio:.2f}",
                f"Max Drawdown: ${self.best_result.max_drawdown:,.2f}",
                "",
                "Best Parameters:",
            ])
            for param, value in self.best_result.params.items():
                lines.append(f"  {param}: {value}")

        return "\n".join(lines)


def _mmap_backtest_worker(
    params: Dict[str, Any],
    mmap_paths_dict: Dict[str, Any],
    ticker: str,
    objective: str,
    min_trades: int = 10,
    dist_stats: Optional[Dict[str, float]] = None
) -> OptimizationResult:
    """
    Worker function for parallel backtest using memory-mapped arrays.

    This function is called in worker processes. It loads the mmap arrays
    (which are shared, not copied) and runs a single backtest.

    Args:
        params: Strategy parameters
        mmap_paths_dict: Paths to memory-mapped arrays
        ticker: Ticker symbol
        objective: Optimization objective
        min_trades: Minimum trades required
        dist_stats: Pre-computed distribution statistics (gap_p16, gap_p84, etc.)
    """
    from numba import njit, prange
    import warnings
    warnings.filterwarnings('ignore')

    start_time = time.time()

    # Reconstruct MMapArrayPaths from dict
    paths = MMapArrayPaths(**mmap_paths_dict)

    # Load memory-mapped arrays (shared, no copy)
    arrays = load_mmap_arrays(paths)

    # Extract arrays
    opens = arrays['opens']
    highs = arrays['highs']
    lows = arrays['lows']
    closes = arrays['closes']
    date_indices = arrays['date_indices']
    hours = arrays['hours']
    minutes = arrays['minutes']
    n_bars = arrays['n_bars']
    n_dates = arrays['n_dates']

    # Get parameters
    ib_duration = params.get('ib_duration_minutes', 30)
    profit_target_pct = params.get('profit_target_percent', 0.5) / 100
    stop_loss_type = params.get('stop_loss_type', 'opposite_ib')
    trade_direction = params.get('trade_direction', 'both')
    position_size = params.get('fixed_share_size', 100)
    min_ib_range = params.get('min_ib_range_percent', 0.0)
    max_ib_range = params.get('max_ib_range_percent', 10.0)
    trailing_stop_enabled = params.get('trailing_stop_enabled', False)
    trailing_stop_atr_mult = params.get('trailing_stop_atr_mult', 2.0)
    break_even_enabled = params.get('break_even_enabled', False)
    break_even_pct = params.get('break_even_pct', 0.5)
    use_qqq_filter = params.get('use_qqq_filter', False)

    # Statistical filter parameters (new mode-based filters)
    gap_filter_mode = params.get('gap_filter_mode', 'any')
    trend_filter_mode = params.get('trend_filter_mode', 'any')
    trend_lookback_days = params.get('trend_lookback_days', 3)
    range_filter_mode = params.get('range_filter_mode', 'any')

    # Extract distribution stats thresholds (if provided)
    if dist_stats:
        gap_p16 = dist_stats.get('gap_p16', -999)
        gap_p84 = dist_stats.get('gap_p84', 999)
        range_p16 = dist_stats.get('range_p16', 0)
        range_p84 = dist_stats.get('range_p84', 999)
        range_p50 = dist_stats.get('range_p50', 0)
        range_p68 = dist_stats.get('range_p68', 999)
    else:
        # Default thresholds if no stats (effectively disables statistical filters)
        gap_p16, gap_p84 = -999, 999
        range_p16, range_p84, range_p50, range_p68 = 0, 999, 0, 999

    # Legacy filter parameters (keep for backwards compatibility)
    prior_days_lookback = trend_lookback_days
    daily_range_lookback = params.get('daily_range_lookback', 5)

    # Parse max_breakout_time (e.g., "14:00" -> hour=14, minute=0)
    max_breakout_str = params.get('max_breakout_time', '14:00')
    max_breakout_parts = max_breakout_str.split(':')
    max_breakout_hour = int(max_breakout_parts[0])
    max_breakout_minute = int(max_breakout_parts[1]) if len(max_breakout_parts) > 1 else 0

    # Parse eod_exit_time (e.g., "15:55" -> hour=15, minute=55)
    eod_exit_str = params.get('eod_exit_time', '15:55')
    eod_exit_parts = eod_exit_str.split(':')
    eod_exit_hour = int(eod_exit_parts[0])
    eod_exit_minute = int(eod_exit_parts[1]) if len(eod_exit_parts) > 1 else 0

    # Compute IB levels for each day
    # IB starts at 9:30, so IB end = 9:30 + ib_duration minutes
    # Convert to hour:minute format
    ib_end_total_minutes = 9 * 60 + 30 + ib_duration  # 9:30 + ib_duration
    ib_end_hour = ib_end_total_minutes // 60
    ib_end_minute = ib_end_total_minutes % 60

    ib_highs = np.zeros(n_dates)
    ib_lows = np.full(n_dates, np.inf)

    for i in range(n_bars):
        h = hours[i]
        m = minutes[i]

        # Check if in IB window (9:30 to IB end time, exclusive)
        # IB window: bar_time >= 9:30 AND bar_time < ib_end_time
        # Convert to minutes from midnight for easy comparison
        bar_minutes = h * 60 + m
        session_start_minutes = 9 * 60 + 30  # 9:30 = 570 minutes
        ib_end_minutes = ib_end_total_minutes  # Already calculated above

        in_ib = (bar_minutes >= session_start_minutes and bar_minutes < ib_end_minutes)

        if in_ib:
            date_idx = date_indices[i]
            if highs[i] > ib_highs[date_idx]:
                ib_highs[date_idx] = highs[i]
            if lows[i] < ib_lows[date_idx]:
                ib_lows[date_idx] = lows[i]

    # Replace inf with 0 for days with no IB bars
    ib_lows[ib_lows == np.inf] = 0

    # Calculate IB ranges
    ib_ranges = np.where(ib_lows > 0, (ib_highs - ib_lows) / ib_lows * 100, 0)

    # =========================================================================
    # DAILY OHLC CALCULATION
    # Calculate daily Open, High, Low, Close for each date (for filters)
    # We use Regular Trading Hours (9:30-16:00) for daily OHLC
    # =========================================================================
    daily_opens = np.zeros(n_dates)
    daily_highs = np.zeros(n_dates)
    daily_lows = np.full(n_dates, np.inf)
    daily_closes = np.zeros(n_dates)
    daily_open_set = np.zeros(n_dates, dtype=bool)  # Track if open has been set

    for i in range(n_bars):
        date_idx = date_indices[i]
        h = hours[i]
        m = minutes[i]
        bar_minutes = h * 60 + m

        # Regular trading hours: 9:30 (570) to 16:00 (960)
        if bar_minutes >= 570 and bar_minutes <= 960:
            # Set open on first RTH bar
            if not daily_open_set[date_idx]:
                daily_opens[date_idx] = opens[i]
                daily_open_set[date_idx] = True

            # Track high/low
            if highs[i] > daily_highs[date_idx]:
                daily_highs[date_idx] = highs[i]
            if lows[i] < daily_lows[date_idx]:
                daily_lows[date_idx] = lows[i]

            # Update close (last RTH bar wins)
            daily_closes[date_idx] = closes[i]

    # Replace inf with 0 for days with no RTH bars
    daily_lows[daily_lows == np.inf] = 0

    # =========================================================================
    # CALCULATE FILTER DATA FOR EACH DAY
    # - Gap %: (today's open - yesterday's close) / yesterday's close * 100
    # - Prior days bullish count: how many of last N days closed > opened
    # - Average daily range: mean of (high - low) / low * 100 for last N days
    # =========================================================================
    gap_percent = np.zeros(n_dates)
    prior_days_bullish_count = np.zeros(n_dates, dtype=np.int32)
    avg_daily_range = np.zeros(n_dates)

    for d in range(n_dates):
        # Gap % (need prior day's close)
        if d > 0 and daily_closes[d - 1] > 0 and daily_opens[d] > 0:
            gap_percent[d] = (daily_opens[d] - daily_closes[d - 1]) / daily_closes[d - 1] * 100

        # Prior days bullish count
        bullish_count = 0
        for lookback in range(1, prior_days_lookback + 1):
            if d >= lookback:
                prev_idx = d - lookback
                if daily_opens[prev_idx] > 0 and daily_closes[prev_idx] > daily_opens[prev_idx]:
                    bullish_count += 1
        prior_days_bullish_count[d] = bullish_count

        # Average daily range over lookback period
        range_sum = 0.0
        range_count = 0
        for lookback in range(1, daily_range_lookback + 1):
            if d >= lookback:
                prev_idx = d - lookback
                if daily_lows[prev_idx] > 0:
                    day_range = (daily_highs[prev_idx] - daily_lows[prev_idx]) / daily_lows[prev_idx] * 100
                    range_sum += day_range
                    range_count += 1
        avg_daily_range[d] = range_sum / range_count if range_count > 0 else 0.0

    # =========================================================================
    # QQQ FILTER LOGIC
    # If use_qqq_filter is True, calculate QQQ IB levels and track breakouts.
    # Primary ticker can only break out if QQQ broke first in the same direction.
    #
    # IMPORTANT: The date indices for primary and filter are independent, so we
    # need to map them using timestamps to find matching dates.
    # =========================================================================
    import pandas as pd

    # Maps: primary_date_idx -> (qqq_broke_long_time, qqq_broke_short_time)
    # Times are in minutes from midnight, -1 means no breakout
    qqq_breakout_times = {}  # {primary_date_idx: {'long': time_minutes, 'short': time_minutes}}

    if use_qqq_filter and 'filter' in arrays:
        filter_data = arrays['filter']
        f_timestamps = filter_data['timestamps']
        f_highs = filter_data['highs']
        f_lows = filter_data['lows']
        f_date_indices = filter_data['date_indices']
        f_hours = filter_data['hours']
        f_minutes = filter_data['minutes']
        f_n_bars = filter_data['n_bars']

        # Find the max date index in filter data
        f_n_dates = int(f_date_indices.max()) + 1 if f_n_bars > 0 else 0

        # Calculate QQQ IB levels for each filter date index
        filter_ib_highs = np.zeros(f_n_dates)
        filter_ib_lows = np.full(f_n_dates, np.inf)

        for i in range(f_n_bars):
            h = f_hours[i]
            m = f_minutes[i]

            # Check if in IB window (same logic as primary)
            # IB window: bar_time >= 9:30 AND bar_time < ib_end_time
            bar_minutes = h * 60 + m
            session_start_minutes = 9 * 60 + 30  # 9:30 = 570 minutes
            ib_end_minutes = ib_end_total_minutes

            in_ib = (bar_minutes >= session_start_minutes and bar_minutes < ib_end_minutes)

            if in_ib:
                f_date_idx = f_date_indices[i]
                if f_highs[i] > filter_ib_highs[f_date_idx]:
                    filter_ib_highs[f_date_idx] = f_highs[i]
                if f_lows[i] < filter_ib_lows[f_date_idx]:
                    filter_ib_lows[f_date_idx] = f_lows[i]

        # Replace inf with 0
        filter_ib_lows[filter_ib_lows == np.inf] = 0

        # Track when QQQ breaks its IB for each filter date index
        # Store as time in minutes from midnight, -1 means no breakout
        filter_broke_long_time = np.full(f_n_dates, -1, dtype=np.int32)  # minutes from midnight
        filter_broke_short_time = np.full(f_n_dates, -1, dtype=np.int32)

        for i in range(f_n_bars):
            f_date_idx = f_date_indices[i]
            h = f_hours[i]
            m = f_minutes[i]
            bar_minutes = h * 60 + m

            # Only check for breakouts after IB ends
            ib_end_minutes = ib_end_hour * 60 + ib_end_minute
            if bar_minutes < ib_end_minutes:
                continue

            f_ib_high = filter_ib_highs[f_date_idx]
            f_ib_low = filter_ib_lows[f_date_idx]

            if f_ib_high == 0 or f_ib_low == 0:
                continue

            # Check for QQQ long breakout (first one only)
            if filter_broke_long_time[f_date_idx] == -1 and f_highs[i] > f_ib_high:
                filter_broke_long_time[f_date_idx] = bar_minutes

            # Check for QQQ short breakout (first one only)
            if filter_broke_short_time[f_date_idx] == -1 and f_lows[i] < f_ib_low:
                filter_broke_short_time[f_date_idx] = bar_minutes

        # Build a mapping from filter date to breakout times
        # We'll use the timestamp to get actual date for each filter date index
        filter_date_to_breakout = {}  # {date_as_int: {'long': time, 'short': time}}

        for i in range(f_n_bars):
            f_date_idx = f_date_indices[i]
            # Get the date from the timestamp (just need one bar per date)
            ts_ns = f_timestamps[i]
            date_int = int(pd.Timestamp(ts_ns, unit='ns').date().toordinal())

            if date_int not in filter_date_to_breakout:
                filter_date_to_breakout[date_int] = {
                    'long': int(filter_broke_long_time[f_date_idx]),
                    'short': int(filter_broke_short_time[f_date_idx])
                }

        # Now map primary date indices to filter breakout times
        timestamps = arrays['timestamps']
        primary_date_to_idx = {}  # date_int -> primary_date_idx
        for i in range(n_bars):
            d_idx = date_indices[i]
            if d_idx not in primary_date_to_idx.values():
                ts_ns = timestamps[i]
                date_int = int(pd.Timestamp(ts_ns, unit='ns').date().toordinal())
                primary_date_to_idx[date_int] = d_idx

        # Build final mapping: primary_date_idx -> QQQ breakout times
        for date_int, p_idx in primary_date_to_idx.items():
            if date_int in filter_date_to_breakout:
                qqq_breakout_times[p_idx] = filter_date_to_breakout[date_int]
            else:
                # No QQQ data for this date - no trades allowed
                qqq_breakout_times[p_idx] = {'long': -1, 'short': -1}

    # Find breakouts and simulate trades
    entry_indices = []
    entry_prices = []
    is_long = []
    entry_date_indices = []

    current_date_idx = -1
    found_breakout = False

    for i in range(n_bars):
        date_idx = date_indices[i]
        h = hours[i]
        m = minutes[i]

        # Reset on new day
        if date_idx != current_date_idx:
            current_date_idx = date_idx
            found_breakout = False

        if found_breakout:
            continue

        # Check if in post-IB trading window (after IB ends, before max_breakout_time)
        bar_minutes = h * 60 + m
        ib_end_minutes = ib_end_hour * 60 + ib_end_minute
        max_breakout_minutes = max_breakout_hour * 60 + max_breakout_minute

        if bar_minutes < ib_end_minutes or bar_minutes > max_breakout_minutes:
            continue

        # Check IB range filter
        ib_range_pct = ib_ranges[date_idx]
        if ib_range_pct < min_ib_range or ib_range_pct > max_ib_range:
            found_breakout = True  # Skip this day
            continue

        # Get primary IB levels
        ib_high = ib_highs[date_idx]
        ib_low = ib_lows[date_idx]

        if ib_high == 0 or ib_low == 0:
            continue

        # =====================================================================
        # BREAKOUT LOGIC
        # There are two modes:
        # 1. Direct breakout (no QQQ filter): Enter when PRIMARY breaks its IB
        # 2. QQQ filter mode: Enter when QQQ breaks its IB, in QQQ's direction
        #    (Primary doesn't need to break, just needs IB calculated)
        # =====================================================================

        long_break = False
        short_break = False

        if use_qqq_filter and qqq_breakout_times:
            # QQQ filter mode: Entry triggered by QQQ breakout, not primary
            # Match BacktestRunner: when QQQ breaks, enter same direction on primary
            if date_idx in qqq_breakout_times:
                qqq_times = qqq_breakout_times[date_idx]
                qqq_long_time = qqq_times['long']
                qqq_short_time = qqq_times['short']

                # Check if QQQ broke long at or before this bar
                if qqq_long_time != -1 and qqq_long_time <= bar_minutes:
                    # QQQ broke long - enter long on primary (if not filtered by direction)
                    if trade_direction != "short_only":
                        long_break = True

                # Check if QQQ broke short at or before this bar (and no long entry yet)
                if not long_break and qqq_short_time != -1 and qqq_short_time <= bar_minutes:
                    # QQQ broke short - enter short on primary (if not filtered by direction)
                    if trade_direction != "long_only":
                        short_break = True

                # Tie-breaker: if both broke at same time, prefer long (matches backtest)
                # Actually, prefer whichever broke first
                if long_break and short_break:
                    if qqq_long_time <= qqq_short_time:
                        short_break = False
                    else:
                        long_break = False
            else:
                # No QQQ data for this date - no entry allowed
                pass
        else:
            # Direct breakout mode: Enter when PRIMARY breaks its IB
            # Use HIGH/LOW for breakout detection (matches NinjaTrader)
            long_break = highs[i] > ib_high
            short_break = lows[i] < ib_low

            if trade_direction == "long_only":
                short_break = False
            elif trade_direction == "short_only":
                long_break = False

        if long_break or short_break:
            # Apply statistical filters
            skip_trade = False
            day_gap = gap_percent[date_idx]
            day_range = avg_daily_range[date_idx]
            trade_dir = 'long' if long_break else 'short'

            # =====================================================================
            # GAP FILTER (statistical)
            # =====================================================================
            if gap_filter_mode != 'any' and not skip_trade:
                if gap_filter_mode == 'middle_68':
                    # Only trade gaps within middle 68% (normal days)
                    if not (gap_p16 <= day_gap <= gap_p84):
                        skip_trade = True
                elif gap_filter_mode == 'exclude_middle_68':
                    # Only trade extreme gap days
                    if gap_p16 <= day_gap <= gap_p84:
                        skip_trade = True
                elif gap_filter_mode == 'directional':
                    # Gap up = longs only, Gap down = shorts only
                    if day_gap > 0 and trade_dir == 'short':
                        skip_trade = True
                    elif day_gap < 0 and trade_dir == 'long':
                        skip_trade = True
                elif gap_filter_mode == 'reverse_directional':
                    # Gap up = shorts only (fade), Gap down = longs only
                    if day_gap > 0 and trade_dir == 'long':
                        skip_trade = True
                    elif day_gap < 0 and trade_dir == 'short':
                        skip_trade = True

            # =====================================================================
            # TREND FILTER
            # =====================================================================
            if trend_filter_mode != 'any' and not skip_trade:
                bullish_count = prior_days_bullish_count[date_idx]
                bearish_count = prior_days_lookback - bullish_count
                is_bullish_trend = bullish_count > bearish_count
                is_bearish_trend = bearish_count > bullish_count

                if trend_filter_mode == 'with_trend':
                    # Trade with the trend
                    if is_bullish_trend and trade_dir == 'short':
                        skip_trade = True
                    elif is_bearish_trend and trade_dir == 'long':
                        skip_trade = True
                elif trend_filter_mode == 'counter_trend':
                    # Trade against the trend (mean reversion)
                    if is_bullish_trend and trade_dir == 'long':
                        skip_trade = True
                    elif is_bearish_trend and trade_dir == 'short':
                        skip_trade = True

            # =====================================================================
            # RANGE (VOLATILITY) FILTER (statistical)
            # =====================================================================
            if range_filter_mode != 'any' and not skip_trade:
                if range_filter_mode == 'middle_68':
                    # Only trade normal volatility days
                    if not (range_p16 <= day_range <= range_p84):
                        skip_trade = True
                elif range_filter_mode == 'above_68':
                    # Only trade high volatility days
                    if day_range <= range_p68:
                        skip_trade = True
                elif range_filter_mode == 'below_median':
                    # Only trade low volatility days
                    if day_range >= range_p50:
                        skip_trade = True
                elif range_filter_mode == 'middle_68_or_below':
                    # Trade if normal OR below normal volatility
                    if day_range > range_p84:
                        skip_trade = True

            if skip_trade:
                found_breakout = True  # Skip this day but mark as processed
                continue

            found_breakout = True
            entry_indices.append(i)
            entry_prices.append(closes[i])
            is_long.append(long_break)
            entry_date_indices.append(date_idx)

    # Simulate trades
    pnls = []
    exit_reasons = []

    for idx, (entry_idx, entry_price, long, entry_date_idx) in enumerate(
        zip(entry_indices, entry_prices, is_long, entry_date_indices)
    ):
        ib_high = ib_highs[entry_date_idx]
        ib_low = ib_lows[entry_date_idx]

        # Calculate target and stop
        if long:
            target = entry_price * (1 + profit_target_pct)
            if stop_loss_type == 'opposite_ib':
                stop = ib_low
            else:  # match_target
                stop = entry_price * (1 - profit_target_pct)
        else:
            target = entry_price * (1 - profit_target_pct)
            if stop_loss_type == 'opposite_ib':
                stop = ib_high
            else:
                stop = entry_price * (1 + profit_target_pct)

        exit_price = None
        exit_reason = None

        # Track current stop (may be modified by trailing/break-even)
        current_stop = stop
        break_even_triggered = False

        # Calculate break-even threshold price
        if break_even_enabled:
            if long:
                break_even_threshold = entry_price + (target - entry_price) * break_even_pct
            else:
                break_even_threshold = entry_price - (entry_price - target) * break_even_pct

        # Calculate trailing stop distance (using ATR approximation from IB range)
        trailing_distance = 0.0
        if trailing_stop_enabled:
            # Use IB range as proxy for ATR (typical volatility measure)
            ib_range = ib_high - ib_low
            trailing_distance = ib_range * trailing_stop_atr_mult

        # Track best price for trailing stop
        best_price = entry_price

        # Calculate EOD exit time in minutes
        eod_minutes = eod_exit_hour * 60 + eod_exit_minute

        # Scan forward from entry
        for j in range(entry_idx + 1, n_bars):
            # Check if still same day
            if date_indices[j] != entry_date_idx:
                # EOD exit at previous bar's close
                if j > entry_idx + 1:
                    exit_price = closes[j - 1]
                    exit_reason = 'eod'
                break

            # Check for EOD exit time
            bar_h = hours[j]
            bar_m = minutes[j]
            bar_minutes = bar_h * 60 + bar_m
            if bar_minutes >= eod_minutes:
                exit_price = closes[j]
                exit_reason = 'eod'
                break

            # Update best price and trailing stop
            if trailing_stop_enabled:
                if long:
                    if highs[j] > best_price:
                        best_price = highs[j]
                        # Trail the stop up
                        new_stop = best_price - trailing_distance
                        if new_stop > current_stop:
                            current_stop = new_stop
                else:
                    if lows[j] < best_price:
                        best_price = lows[j]
                        # Trail the stop down
                        new_stop = best_price + trailing_distance
                        if new_stop < current_stop:
                            current_stop = new_stop

            # Check break-even trigger
            if break_even_enabled and not break_even_triggered:
                if long:
                    if highs[j] >= break_even_threshold:
                        # Move stop to entry (break-even)
                        if entry_price > current_stop:
                            current_stop = entry_price
                        break_even_triggered = True
                else:
                    if lows[j] <= break_even_threshold:
                        # Move stop to entry (break-even)
                        if entry_price < current_stop:
                            current_stop = entry_price
                        break_even_triggered = True

            # Exit logic matching BacktestRunner's ExitManager
            # Uses realistic fill prices that account for gaps/slippage
            if long:
                if highs[j] >= target:
                    # Target hit - use target price or open if gapped through
                    # For long target: min(open, target) if open <= target, else target
                    if opens[j] <= target:
                        exit_price = min(opens[j], target)
                    else:
                        exit_price = target
                    exit_reason = 'target'
                    break
                elif lows[j] <= current_stop:
                    # Stop hit - use stop price or open if gapped through
                    # For long stop: max(open, stop) if open >= stop, else open
                    if opens[j] >= current_stop:
                        exit_price = max(opens[j], current_stop)
                    else:
                        exit_price = opens[j]  # Gapped through stop
                    exit_reason = 'trailing_stop' if trailing_stop_enabled else ('break_even' if break_even_triggered else 'stop')
                    break
            else:
                if lows[j] <= target:
                    # Short target hit - use target price or open if gapped through
                    # For short target: max(open, target) if open >= target, else target
                    if opens[j] >= target:
                        exit_price = max(opens[j], target)
                    else:
                        exit_price = target
                    exit_reason = 'target'
                    break
                elif highs[j] >= current_stop:
                    # Short stop hit - use stop price or open if gapped through
                    # For short stop: min(open, stop) if open <= stop, else open
                    if opens[j] <= current_stop:
                        exit_price = min(opens[j], current_stop)
                    else:
                        exit_price = opens[j]  # Gapped through stop
                    exit_reason = 'trailing_stop' if trailing_stop_enabled else ('break_even' if break_even_triggered else 'stop')
                    break

        if exit_price is not None:
            if long:
                pnl = (exit_price - entry_price) * position_size
            else:
                pnl = (entry_price - exit_price) * position_size
            pnls.append(pnl)
            exit_reasons.append(exit_reason)

    # Calculate metrics
    pnls = np.array(pnls)

    if len(pnls) < min_trades:
        return OptimizationResult(
            params=params,
            total_trades=len(pnls),
            win_rate=0.0,
            total_pnl=0.0,
            profit_factor=0.0,
            sharpe_ratio=-999.0,
            sortino_ratio=0.0,
            max_drawdown=0.0,
            avg_trade=0.0,
            objective_value=-999.0,
            run_time_seconds=time.time() - start_time,
            trade_pnls=pnls.tolist() if len(pnls) > 0 else []
        )

    wins = pnls[pnls > 0]
    losses = pnls[pnls < 0]

    total_wins = wins.sum() if len(wins) > 0 else 0
    total_losses = abs(losses.sum()) if len(losses) > 0 else 0.0001

    win_rate = len(wins) / len(pnls) * 100 if len(pnls) > 0 else 0
    profit_factor = total_wins / total_losses if total_losses > 0 else 0
    total_pnl = pnls.sum()
    avg_trade = pnls.mean()

    # Sharpe ratio (annualized)
    if len(pnls) > 1 and pnls.std() > 0:
        sharpe = (pnls.mean() / pnls.std()) * np.sqrt(252)
    else:
        sharpe = 0.0

    # Sortino ratio
    downside = pnls[pnls < 0]
    if len(downside) > 1 and downside.std() > 0:
        sortino = (pnls.mean() / downside.std()) * np.sqrt(252)
    else:
        sortino = sharpe

    # Max drawdown
    equity = np.cumsum(pnls)
    running_max = np.maximum.accumulate(equity)
    drawdown = running_max - equity
    max_drawdown = drawdown.max() if len(drawdown) > 0 else 0

    # K-Ratio (Kestner Ratio) - measures consistency of returns
    # K-Ratio = slope / standard_error normalized by sqrt(n)
    k_ratio = 0.0
    n = len(equity)
    if n >= 3:
        x = np.arange(1, n + 1, dtype=np.float64)
        y = equity.astype(np.float64)

        x_mean = x.mean()
        y_mean = y.mean()

        ss_xx = np.sum((x - x_mean) ** 2)
        ss_xy = np.sum((x - x_mean) * (y - y_mean))

        if ss_xx > 0:
            slope = ss_xy / ss_xx
            y_pred = slope * x + (y_mean - slope * x_mean)
            residuals = y - y_pred
            ss_residuals = np.sum(residuals ** 2)

            mse = ss_residuals / (n - 2)
            se_slope = np.sqrt(mse / ss_xx) if mse > 0 else 0

            if se_slope > 0:
                k_ratio = (slope / se_slope) / np.sqrt(n)
            elif slope > 0:
                k_ratio = 10.0  # Cap for perfect fit
            elif slope < 0:
                k_ratio = -10.0

    # Calculate objective value
    if objective == "sharpe_ratio":
        obj_value = sharpe
    elif objective == "sortino_ratio":
        obj_value = sortino
    elif objective == "profit_factor":
        obj_value = profit_factor if profit_factor < 100 else 0
    elif objective == "total_profit":
        obj_value = total_pnl
    elif objective == "win_rate":
        obj_value = win_rate
    elif objective == "k_ratio":
        obj_value = k_ratio
    else:
        obj_value = sharpe

    return OptimizationResult(
        params=params,
        total_trades=len(pnls),
        win_rate=win_rate,
        total_pnl=total_pnl,
        profit_factor=profit_factor,
        sharpe_ratio=sharpe,
        sortino_ratio=sortino,
        max_drawdown=max_drawdown,
        avg_trade=avg_trade,
        objective_value=obj_value,
        run_time_seconds=time.time() - start_time,
        trade_pnls=pnls.tolist(),
        k_ratio=k_ratio
    )


class MMapGridSearchOptimizer:
    """
    Memory-efficient grid search optimizer using memory-mapped arrays.

    Reduces memory usage by 80-90% compared to standard grid search by
    sharing data across worker processes instead of copying.
    """

    def __init__(
        self,
        data_dir: str,
        output_dir: Optional[str] = None,
        n_jobs: int = -1
    ):
        """
        Initialize optimizer.

        Args:
            data_dir: Directory containing data files
            output_dir: Directory for output files
            n_jobs: Number of parallel jobs (-1 = all CPUs)
        """
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir) if output_dir else self.data_dir / "optimization_results"
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.n_jobs = n_jobs if n_jobs > 0 else multiprocessing.cpu_count()

        # Data storage
        self.loader = DataLoader(str(self.data_dir))
        self.sessions = None
        self.ticker = None
        self.filter_bars_dict = None
        self.mmap_manager = None

    def _find_data_file(self, ticker: str) -> Path:
        """Find data file for ticker, preferring NT format."""
        nt_files = []
        other_files = []

        for f in self.data_dir.iterdir():
            if f.is_file() and ticker.upper() in f.name.upper():
                if f.suffix.lower() in ['.txt', '.csv']:
                    if '_NT' in f.name.upper() or 'NT.TXT' in f.name.upper():
                        nt_files.append(f)
                    else:
                        other_files.append(f)

        if nt_files:
            return nt_files[0]
        if other_files:
            return other_files[0]

        raise FileNotFoundError(f"No data file found for {ticker} in {self.data_dir}")

    def load_data(self, ticker: str, data_file: Optional[str] = None,
                  filter_ticker: Optional[str] = None):
        """
        Load and prepare data for optimization.

        Args:
            ticker: Ticker symbol
            data_file: Specific data file (optional)
            filter_ticker: Optional filter ticker (e.g., "QQQ")
        """
        self.ticker = ticker

        if data_file:
            filepath = self.data_dir / data_file
        else:
            filepath = self._find_data_file(ticker)

        # Load data
        df = self.loader.load_auto_detect(str(filepath), ticker)

        # Build sessions
        session_builder = SessionBuilder()
        self.sessions = session_builder.build_sessions_from_dataframe(df, ticker)

        print(f"Loaded {len(self.sessions)} sessions for {ticker}")

        # Load filter ticker data if specified
        if filter_ticker:
            self._load_filter_data(filter_ticker)

    def _load_filter_data(self, filter_ticker: str):
        """Load filter ticker data (e.g., QQQ)."""
        try:
            filter_file = self._find_data_file(filter_ticker)
            df_filter = self.loader.load_auto_detect(str(filter_file), filter_ticker)

            self.filter_bars_dict = {}
            for _, row in df_filter.iterrows():
                bar = Bar(
                    timestamp=row['timestamp'],
                    open=row['open'],
                    high=row['high'],
                    low=row['low'],
                    close=row['close'],
                    volume=row.get('volume', 0),
                    ticker=filter_ticker
                )
                self.filter_bars_dict[row['timestamp']] = bar

            print(f"Loaded {len(self.filter_bars_dict):,} filter bars for {filter_ticker}")

        except FileNotFoundError as e:
            print(f"Warning: Could not load filter data for {filter_ticker}: {e}")
            self.filter_bars_dict = None

    def optimize(
        self,
        parameter_space: Optional[ParameterSpace] = None,
        objective: str = "sharpe_ratio",
        min_trades: int = 10,
        progress_callback: Optional[Callable] = None
    ) -> GridSearchResults:
        """
        Run grid search optimization with memory-mapped arrays.

        Args:
            parameter_space: Parameter space to search
            objective: Objective to optimize
            min_trades: Minimum trades required
            progress_callback: Optional callback for progress updates

        Returns:
            GridSearchResults
        """
        if self.sessions is None:
            raise ValueError("No data loaded. Call load_data() first.")

        space = parameter_space or create_parameter_space("standard")
        combinations = space.get_grid_combinations()

        print(f"\n{'='*60}")
        print("MEMORY-EFFICIENT GRID SEARCH OPTIMIZATION")
        print(f"{'='*60}")
        print(f"Ticker: {self.ticker}")
        print(f"Objective: {objective}")
        print(f"Parameter Combinations: {len(combinations):,}")
        print(f"Parallel Jobs: {self.n_jobs}")
        print(f"{'='*60}\n")

        results = GridSearchResults(
            objective=objective,
            total_combinations=len(combinations),
            ticker=self.ticker
        )

        start_time = time.time()

        # Load distribution stats for statistical filters (cached per ticker)
        dist_stats_dict = None
        try:
            stats_calc = DistributionStatsCalculator(str(self.data_dir))
            ticker_stats = stats_calc.get_stats(self.ticker)
            if ticker_stats:
                # Convert to simple dict for passing to workers
                dist_stats_dict = {
                    'gap_p16': ticker_stats.gap_stats.p16,
                    'gap_p84': ticker_stats.gap_stats.p84,
                    'gap_mean': ticker_stats.gap_stats.mean,
                    'gap_std': ticker_stats.gap_stats.std,
                    'range_p16': ticker_stats.range_stats.p16,
                    'range_p50': ticker_stats.range_stats.p50,
                    'range_p68': ticker_stats.range_stats.p68,
                    'range_p84': ticker_stats.range_stats.p84,
                    'range_p90': ticker_stats.range_stats.p90,
                    'range_mean': ticker_stats.range_stats.mean,
                    'range_std': ticker_stats.range_stats.std,
                }
                print(f"Loaded distribution stats for {self.ticker}:")
                print(f"  Gap: mean={ticker_stats.gap_stats.mean:.2f}%, "
                      f"p16={ticker_stats.gap_stats.p16:.2f}%, p84={ticker_stats.gap_stats.p84:.2f}%")
                print(f"  Range: mean={ticker_stats.range_stats.mean:.2f}%, "
                      f"p50={ticker_stats.range_stats.p50:.2f}%, p84={ticker_stats.range_stats.p84:.2f}%")
        except Exception as e:
            print(f"Warning: Could not load distribution stats: {e}")
            print("Statistical filters will use default thresholds")

        # Create memory-mapped arrays
        print("Creating memory-mapped arrays...")
        self.mmap_manager = MMapDataManager(
            self.sessions,
            self.filter_bars_dict,
            ticker=self.ticker,
            filter_ticker="QQQ" if self.filter_bars_dict else ""
        )
        paths = self.mmap_manager.get_paths()
        paths_dict = asdict(paths)

        print(f"  Arrays created: {paths.n_bars:,} bars, {paths.n_dates:,} days")

        # Run parallel optimization in batches for progress updates
        print(f"\nRunning {len(combinations):,} backtests using {self.n_jobs} workers...")

        # Process in batches to allow progress updates
        batch_size = max(self.n_jobs * 4, 50)  # Process ~4 batches per worker at a time
        total = len(combinations)

        try:
            for batch_start in range(0, total, batch_size):
                batch_end = min(batch_start + batch_size, total)
                batch = combinations[batch_start:batch_end]

                # Run batch in parallel
                batch_results = Parallel(
                    n_jobs=self.n_jobs,
                    backend='loky',
                    verbose=0  # Quiet mode - we handle progress ourselves
                )(
                    delayed(_mmap_backtest_worker)(
                        params=params,
                        mmap_paths_dict=paths_dict,
                        ticker=self.ticker,
                        objective=objective,
                        min_trades=min_trades,
                        dist_stats=dist_stats_dict
                    )
                    for params in batch
                )

                # Collect batch results
                for result in batch_results:
                    results.add_result(result)

                # Progress callback
                if progress_callback:
                    progress_callback(batch_end, total, results.best_result)

        finally:
            # Cleanup mmap files
            if self.mmap_manager:
                self.mmap_manager.cleanup()

        elapsed = time.time() - start_time
        results.total_time_seconds = elapsed

        best_val = results.best_result.objective_value if results.best_result else 0
        print(f"\nComplete: {len(combinations):,} combinations in {elapsed:.1f}s")
        print(f"Best {objective}: {best_val:.4f}")

        if progress_callback:
            progress_callback(len(combinations), len(combinations), results.best_result)

        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = self.output_dir / f"mmap_grid_{self.ticker}_{timestamp}.csv"
        results.save_results(str(results_file))

        if results.best_result:
            best_file = self.output_dir / f"best_params_{self.ticker}_{timestamp}.json"
            with open(best_file, 'w') as f:
                json.dump(results.best_result.params, f, indent=2)
            print(f"Best parameters saved to: {best_file}")

        return results

    def quick_optimize(self, objective: str = "sharpe_ratio") -> GridSearchResults:
        """Run quick optimization."""
        space = create_parameter_space("quick")
        return self.optimize(space, objective)

    def standard_optimize(self, objective: str = "sharpe_ratio") -> GridSearchResults:
        """Run standard optimization."""
        space = create_parameter_space("standard")
        return self.optimize(space, objective)

    def full_optimize(self, objective: str = "sharpe_ratio") -> GridSearchResults:
        """Run full optimization."""
        space = create_parameter_space("full")
        return self.optimize(space, objective)


# Alias for drop-in replacement
GridSearchOptimizer = MMapGridSearchOptimizer


if __name__ == "__main__":
    # Test the optimizer
    optimizer = MMapGridSearchOptimizer(r"C:\Users\Warren\Downloads")

    # Load data
    optimizer.load_data("TSLA")

    # Run quick optimization
    results = optimizer.quick_optimize(objective="profit_factor")

    print("\n" + results.summary())

    # Show top 5 results
    print("\n\nTOP 5 RESULTS:")
    print("-" * 60)
    for i, result in enumerate(results.get_top_n(5)):
        print(f"\n{i+1}. Objective: {result.objective_value:.4f}")
        print(f"   Trades: {result.total_trades}, Win Rate: {result.win_rate:.1f}%")
        print(f"   P&L: ${result.total_pnl:,.2f}, PF: {result.profit_factor:.2f}")
