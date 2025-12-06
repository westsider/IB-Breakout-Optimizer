"""
Fully NumPy-Vectorized Backtester for IB Breakout Strategy.

Uses pure NumPy array operations - no Python loops for trade simulation.
Should be 50-100x faster than loop-based approaches.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, List
from dataclasses import dataclass
import time
from numba import njit, prange
import warnings
warnings.filterwarnings('ignore')


@njit(parallel=True)
def simulate_trades_numba(
    entry_prices: np.ndarray,
    entry_indices: np.ndarray,
    is_long: np.ndarray,
    targets: np.ndarray,
    stops: np.ndarray,
    highs: np.ndarray,
    lows: np.ndarray,
    closes: np.ndarray,
    date_indices: np.ndarray,
    position_size: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Numba-accelerated trade simulation.

    Runs in parallel across trades using all CPU cores.
    """
    n_trades = len(entry_prices)
    pnls = np.zeros(n_trades)
    exit_reasons = np.zeros(n_trades, dtype=np.int32)  # 0=none, 1=target, 2=stop, 3=eod

    for i in prange(n_trades):
        entry_idx = entry_indices[i]
        entry_date = date_indices[entry_idx]
        entry_price = entry_prices[i]
        target = targets[i]
        stop = stops[i]
        long = is_long[i]

        exit_price = 0.0
        exit_reason = 0

        # Scan forward from entry
        for j in range(entry_idx + 1, len(highs)):
            # Check if still same day
            if date_indices[j] != entry_date:
                # EOD exit at previous bar's close
                if j > entry_idx + 1:
                    exit_price = closes[j - 1]
                    exit_reason = 3
                break

            if long:
                if highs[j] >= target:
                    exit_price = target
                    exit_reason = 1
                    break
                elif lows[j] <= stop:
                    exit_price = stop
                    exit_reason = 2
                    break
            else:
                if lows[j] <= target:
                    exit_price = target
                    exit_reason = 1
                    break
                elif highs[j] >= stop:
                    exit_price = stop
                    exit_reason = 2
                    break

        # Calculate P&L
        if exit_reason > 0:
            if long:
                pnls[i] = (exit_price - entry_price) * position_size
            else:
                pnls[i] = (entry_price - exit_price) * position_size

        exit_reasons[i] = exit_reason

    return pnls, exit_reasons


class NumbaBacktester:
    """
    Ultra-fast backtester using Numba JIT compilation.

    First call will be slow (JIT compilation), subsequent calls are very fast.
    """

    def __init__(self, df: pd.DataFrame):
        """
        Initialize with price data.

        Args:
            df: DataFrame with columns: timestamp, open, high, low, close
        """
        # Convert to numpy arrays for speed
        self.timestamps = df['timestamp'].values
        self.opens = df['open'].values.astype(np.float64)
        self.highs = df['high'].values.astype(np.float64)
        self.lows = df['low'].values.astype(np.float64)
        self.closes = df['close'].values.astype(np.float64)

        # Extract date and time info
        ts = pd.to_datetime(df['timestamp'])
        self.dates = ts.dt.date.values
        self.hours = ts.dt.hour.values.astype(np.int32)
        self.minutes = ts.dt.minute.values.astype(np.int32)

        # Create date index (unique dates mapped to integers)
        unique_dates = np.unique(self.dates)
        date_to_idx = {d: i for i, d in enumerate(unique_dates)}
        self.date_indices = np.array([date_to_idx[d] for d in self.dates], dtype=np.int32)
        self.unique_dates = unique_dates
        self.n_bars = len(df)

        # Pre-compute IB levels for common durations
        self._ib_cache = {}

    def _compute_ib_levels(self, ib_duration: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute IB high/low for each day.

        Returns: (ib_highs, ib_lows) arrays indexed by date_idx
        """
        if ib_duration in self._ib_cache:
            return self._ib_cache[ib_duration]

        n_dates = len(self.unique_dates)
        ib_highs = np.zeros(n_dates)
        ib_lows = np.full(n_dates, np.inf)

        # IB window: 9:30 to 9:30 + duration
        ib_end_minutes = 30 + ib_duration
        ib_end_hour = 9 + (ib_end_minutes // 60)
        ib_end_minute = ib_end_minutes % 60

        for i in range(self.n_bars):
            h = self.hours[i]
            m = self.minutes[i]

            # Check if in IB window
            in_ib = False
            if h == 9 and m >= 30:
                in_ib = True
            elif h == ib_end_hour and m < ib_end_minute:
                in_ib = True
            elif h > 9 and h < ib_end_hour:
                in_ib = True

            if in_ib:
                date_idx = self.date_indices[i]
                if self.highs[i] > ib_highs[date_idx]:
                    ib_highs[date_idx] = self.highs[i]
                if self.lows[i] < ib_lows[date_idx]:
                    ib_lows[date_idx] = self.lows[i]

        # Replace inf with 0 for days with no IB bars
        ib_lows[ib_lows == np.inf] = 0

        self._ib_cache[ib_duration] = (ib_highs, ib_lows)
        return ib_highs, ib_lows

    def _find_breakouts(
        self,
        ib_highs: np.ndarray,
        ib_lows: np.ndarray,
        ib_duration: int,
        trade_direction: str,
        max_breakout_hour: int = 14
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Find first breakout for each day.

        Returns: (entry_indices, entry_prices, is_long)
        """
        # Post-IB window start
        ib_end_minutes = 30 + ib_duration
        ib_end_hour = 9 + (ib_end_minutes // 60)
        ib_end_minute = ib_end_minutes % 60

        entry_indices = []
        entry_prices = []
        is_long = []

        current_date_idx = -1
        found_breakout = False

        for i in range(self.n_bars):
            date_idx = self.date_indices[i]
            h = self.hours[i]
            m = self.minutes[i]

            # Reset on new day
            if date_idx != current_date_idx:
                current_date_idx = date_idx
                found_breakout = False

            if found_breakout:
                continue

            # Check if in post-IB trading window
            in_window = False
            if h == ib_end_hour and m >= ib_end_minute:
                in_window = True
            elif h > ib_end_hour and h < max_breakout_hour:
                in_window = True
            elif h == max_breakout_hour and m == 0:
                in_window = True

            if not in_window:
                continue

            # Check for breakout
            ib_high = ib_highs[date_idx]
            ib_low = ib_lows[date_idx]

            if ib_high == 0 or ib_low == 0:
                continue

            long_break = self.highs[i] > ib_high
            short_break = self.lows[i] < ib_low

            if trade_direction == "long_only":
                short_break = False
            elif trade_direction == "short_only":
                long_break = False

            if long_break or short_break:
                found_breakout = True
                entry_indices.append(i)
                entry_prices.append(self.closes[i])
                is_long.append(long_break)

        return (
            np.array(entry_indices, dtype=np.int64),
            np.array(entry_prices, dtype=np.float64),
            np.array(is_long, dtype=np.bool_)
        )

    def run_backtest(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run backtest with given parameters.

        Returns dict with metrics.
        """
        ib_duration = params.get('ib_duration_minutes', 30)
        profit_target_pct = params.get('profit_target_percent', 0.5) / 100
        stop_loss_type = params.get('stop_loss_type', 'opposite_ib')
        trade_direction = params.get('trade_direction', 'both')
        position_size = params.get('fixed_share_size', 100)

        # Get IB levels
        ib_highs, ib_lows = self._compute_ib_levels(ib_duration)

        # Find breakouts
        entry_indices, entry_prices, is_long = self._find_breakouts(
            ib_highs, ib_lows, ib_duration, trade_direction
        )

        if len(entry_indices) == 0:
            return {
                'total_trades': 0,
                'profit_factor': 0.0,
                'sharpe_ratio': 0.0,
                'total_pnl': 0.0,
                'win_rate': 0.0
            }

        # Calculate targets and stops
        targets = np.where(
            is_long,
            entry_prices * (1 + profit_target_pct),
            entry_prices * (1 - profit_target_pct)
        )

        if stop_loss_type == 'opposite_ib':
            stops = np.where(
                is_long,
                ib_lows[self.date_indices[entry_indices]],
                ib_highs[self.date_indices[entry_indices]]
            )
        else:  # match_target
            stops = np.where(
                is_long,
                entry_prices * (1 - profit_target_pct),
                entry_prices * (1 + profit_target_pct)
            )

        # Run simulation
        pnls, exit_reasons = simulate_trades_numba(
            entry_prices,
            entry_indices,
            is_long,
            targets,
            stops,
            self.highs,
            self.lows,
            self.closes,
            self.date_indices,
            position_size
        )

        # Filter valid trades (exit_reason > 0)
        valid = exit_reasons > 0
        pnls = pnls[valid]

        if len(pnls) == 0:
            return {
                'total_trades': 0,
                'profit_factor': 0.0,
                'sharpe_ratio': 0.0,
                'total_pnl': 0.0,
                'win_rate': 0.0
            }

        # Calculate metrics
        wins = pnls[pnls > 0]
        losses = pnls[pnls < 0]

        total_wins = wins.sum() if len(wins) > 0 else 0
        total_losses = abs(losses.sum()) if len(losses) > 0 else 0.0001

        profit_factor = total_wins / total_losses

        if len(pnls) > 1 and pnls.std() > 0:
            sharpe = (pnls.mean() / pnls.std()) * np.sqrt(252)
        else:
            sharpe = 0.0

        return {
            'total_trades': len(pnls),
            'profit_factor': profit_factor,
            'sharpe_ratio': sharpe,
            'total_pnl': pnls.sum(),
            'win_rate': len(wins) / len(pnls) * 100 if len(pnls) > 0 else 0
        }


def run_numba_optimization(
    df: pd.DataFrame,
    param_combinations: List[Dict[str, Any]],
    objective: str = 'profit_factor',
    min_trades: int = 10
) -> Tuple[Dict[str, Any], Dict[str, Any], float]:
    """
    Run optimization using Numba-accelerated backtester.

    Returns: (best_params, best_result, elapsed_time)
    """
    backtester = NumbaBacktester(df)

    # Warmup JIT
    print("Warming up JIT compiler...")
    _ = backtester.run_backtest(param_combinations[0])

    print(f"Running {len(param_combinations)} backtests...")
    start = time.time()

    best_params = None
    best_result = None
    best_value = float('-inf')

    for i, params in enumerate(param_combinations):
        result = backtester.run_backtest(params)

        value = result.get(objective, 0)

        if value > best_value and result['total_trades'] >= min_trades:
            best_value = value
            best_params = params
            best_result = result

        if (i + 1) % 10 == 0:
            print(f"  {i+1}/{len(param_combinations)} complete")

    elapsed = time.time() - start
    return best_params, best_result, elapsed


if __name__ == "__main__":
    import sys
    sys.path.insert(0, '.')

    from data.data_loader import DataLoader
    from optimization.parameter_space import create_parameter_space

    print("="*60)
    print("NUMBA-ACCELERATED BACKTEST BENCHMARK")
    print("="*60)

    print("\nLoading data...")
    loader = DataLoader(r"C:\Users\Warren\Downloads")
    df = loader.load_auto_detect(
        r"C:\Users\Warren\Downloads\TSLA_1min_20231204_to_20241204_NT.txt",
        "TSLA"
    )
    print(f"Loaded {len(df)} bars")

    # Test single backtest
    print("\n--- Single Backtest Test ---")
    backtester = NumbaBacktester(df)

    params = {
        'ib_duration_minutes': 30,
        'profit_target_percent': 0.5,
        'stop_loss_type': 'opposite_ib',
        'trade_direction': 'both'
    }

    # Warmup
    print("JIT warmup...")
    _ = backtester.run_backtest(params)

    # Timed run
    start = time.time()
    result = backtester.run_backtest(params)
    elapsed = time.time() - start

    print(f"Time: {elapsed*1000:.1f}ms")
    print(f"Trades: {result['total_trades']}")
    print(f"P&L: ${result['total_pnl']:.2f}")
    print(f"Profit Factor: {result['profit_factor']:.4f}")

    # Optimization benchmark
    print("\n--- Optimization Benchmark ---")
    for preset in ['turbo', 'quick', 'standard']:
        space = create_parameter_space(preset)
        combos = space.get_grid_combinations()

        best_params, best_result, elapsed = run_numba_optimization(
            df, combos, 'profit_factor'
        )

        print(f"\n{preset.upper()}: {len(combos)} combos in {elapsed:.2f}s "
              f"({len(combos)/elapsed:.1f}/sec)")
        if best_result:
            print(f"  Best PF: {best_result['profit_factor']:.4f}, "
                  f"Trades: {best_result['total_trades']}, "
                  f"P&L: ${best_result['total_pnl']:.2f}")
