"""
Grid Search Optimizer for IB Breakout Strategy.

Performs exhaustive search over parameter combinations.
Useful for initial exploration and understanding parameter sensitivity.
"""

import sys
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple, Callable
from dataclasses import dataclass, field
import pandas as pd
import numpy as np
import multiprocessing
import time
import json
from joblib import Parallel, delayed

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from data.data_loader import DataLoader
from data.session_builder import SessionBuilder
from data.data_types import Trade, Bar
from strategy.ib_breakout import IBBreakoutStrategy, StrategyParams
from metrics.performance_metrics import calculate_metrics, PerformanceMetrics
from optimization.parameter_space import ParameterSpace, create_parameter_space


@dataclass
class OptimizationResult:
    """Result from a single parameter combination."""
    params: Dict[str, Any]
    metrics: PerformanceMetrics
    trades: List[Trade]
    objective_value: float
    run_time_seconds: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for export."""
        return {
            **self.params,
            'total_trades': self.metrics.total_trades,
            'win_rate': self.metrics.percent_profitable,
            'total_pnl': self.metrics.total_net_profit,
            'profit_factor': self.metrics.profit_factor,
            'sharpe_ratio': self.metrics.sharpe_ratio,
            'sortino_ratio': self.metrics.sortino_ratio,
            'max_drawdown': self.metrics.max_drawdown,
            'avg_trade': self.metrics.avg_trade,
            'objective_value': self.objective_value,
            'run_time': self.run_time_seconds
        }


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

    def to_dataframe(self) -> pd.DataFrame:
        """Convert all results to DataFrame."""
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
                f"Total Trades: {self.best_result.metrics.total_trades}",
                f"Win Rate: {self.best_result.metrics.percent_profitable:.1f}%",
                f"Total P&L: ${self.best_result.metrics.total_net_profit:,.2f}",
                f"Profit Factor: {self.best_result.metrics.profit_factor:.2f}",
                f"Sharpe Ratio: {self.best_result.metrics.sharpe_ratio:.2f}",
                f"Max Drawdown: ${self.best_result.metrics.max_drawdown:,.2f}",
                "",
                "Best Parameters:",
            ])
            for param, value in self.best_result.params.items():
                lines.append(f"  {param}: {value}")

        return "\n".join(lines)


def calculate_objective(metrics: PerformanceMetrics, objective: str = "sharpe_ratio") -> float:
    """
    Calculate objective value for optimization.

    Args:
        metrics: Performance metrics
        objective: Objective to optimize

    Returns:
        Objective value (higher is better)
    """
    if metrics.total_trades < 10:
        # Penalize strategies with too few trades
        return -999.0

    if objective == "sharpe_ratio":
        return metrics.sharpe_ratio

    elif objective == "sortino_ratio":
        return metrics.sortino_ratio

    elif objective == "profit_factor":
        return metrics.profit_factor if metrics.profit_factor < 100 else 0.0

    elif objective == "total_profit":
        return metrics.total_net_profit

    elif objective == "calmar_ratio":
        return metrics.calmar_ratio

    elif objective == "risk_adjusted_return":
        # Custom: Sharpe * sqrt(trades) to favor more trades
        return metrics.sharpe_ratio * np.sqrt(metrics.total_trades / 100)

    elif objective == "profit_per_trade":
        return metrics.avg_trade

    elif objective == "win_rate":
        return metrics.percent_profitable

    else:
        return metrics.sharpe_ratio


def _run_backtest_worker(
    params: Dict[str, Any],
    sessions_data: List[Dict],
    ticker: str,
    objective: str,
    filter_bars_data: Optional[Dict] = None
) -> OptimizationResult:
    """
    Worker function for parallel backtest execution.
    Reconstructs objects from serialized data.
    """
    from data.data_types import Bar, TradingSession

    start_time = time.time()

    # Reconstruct sessions from serialized data
    sessions = []
    for session_dict in sessions_data:
        bars = [Bar(**bar_data) for bar_data in session_dict['bars']]
        session = TradingSession(
            date=session_dict['date'],
            ticker=session_dict['ticker'],
            session_start=session_dict['session_start'],
            session_end=session_dict['session_end'],
            bars=bars
        )
        sessions.append(session)

    # Reconstruct filter bars dict
    filter_bars_dict = None
    if filter_bars_data:
        filter_bars_dict = {ts: Bar(**bar_data) for ts, bar_data in filter_bars_data.items()}

    # Determine if QQQ filter should be used
    use_qqq_filter = params.get('use_qqq_filter', False) and filter_bars_dict is not None

    # Create strategy params
    strategy_params = StrategyParams(
        ib_duration_minutes=params.get('ib_duration_minutes', 30),
        ib_proximity_percent=params.get('ib_proximity_percent', 0.0),
        trade_direction=params.get('trade_direction', 'both'),
        trading_start_time=params.get('trading_start_time', '09:00'),
        trading_end_time=params.get('trading_end_time', '15:00'),
        fixed_share_size=params.get('fixed_share_size', 100),
        profit_target_percent=params.get('profit_target_percent', 0.5),
        stop_loss_type=params.get('stop_loss_type', 'opposite_ib'),
        trailing_stop_enabled=params.get('trailing_stop_enabled', False),
        trailing_stop_atr_mult=params.get('trailing_stop_atr_mult', 2.0),
        break_even_enabled=params.get('break_even_enabled', False),
        break_even_pct=params.get('break_even_pct', 0.7),
        max_bars_enabled=params.get('max_bars_enabled', False),
        max_bars=params.get('max_bars', 60),
        eod_exit_time=params.get('eod_exit_time', '15:55'),
        use_qqq_filter=use_qqq_filter,
        min_ib_range_percent=params.get('min_ib_range_percent', 0.0),
        max_ib_range_percent=params.get('max_ib_range_percent', 10.0),
        max_breakout_time=params.get('max_breakout_time', '14:00'),
        trade_monday=params.get('trade_monday', True),
        trade_tuesday=params.get('trade_tuesday', True),
        trade_wednesday=params.get('trade_wednesday', True),
        trade_thursday=params.get('trade_thursday', True),
        trade_friday=params.get('trade_friday', True)
    )

    # Run backtest
    strategy = IBBreakoutStrategy(strategy_params)

    for session in sessions:
        is_first = True
        for bar in session.bars:
            filter_bar = filter_bars_dict.get(bar.timestamp) if use_qqq_filter else None
            strategy.process_bar(bar, is_first, qqq_bar=filter_bar)
            is_first = False

    # Calculate metrics
    trades = strategy.get_trades()
    metrics = calculate_metrics(trades)
    obj_value = calculate_objective(metrics, objective)

    run_time = time.time() - start_time

    return OptimizationResult(
        params=params,
        metrics=metrics,
        trades=trades,
        objective_value=obj_value,
        run_time_seconds=run_time
    )


def run_single_backtest(
    params: Dict[str, Any],
    sessions: List,
    ticker: str,
    objective: str = "sharpe_ratio",
    filter_bars_dict: Optional[Dict] = None
) -> OptimizationResult:
    """
    Run a single backtest with given parameters.

    Args:
        params: Strategy parameters
        sessions: Pre-built trading sessions
        ticker: Ticker symbol
        objective: Objective to optimize
        filter_bars_dict: Optional dict mapping timestamp -> Bar for QQQ filter

    Returns:
        OptimizationResult
    """
    start_time = time.time()

    # Determine if QQQ filter should be used
    use_qqq_filter = params.get('use_qqq_filter', False) and filter_bars_dict is not None

    # Create strategy params
    strategy_params = StrategyParams(
        ib_duration_minutes=params.get('ib_duration_minutes', 30),
        ib_proximity_percent=params.get('ib_proximity_percent', 0.0),
        trade_direction=params.get('trade_direction', 'both'),
        trading_start_time=params.get('trading_start_time', '09:00'),
        trading_end_time=params.get('trading_end_time', '15:00'),
        fixed_share_size=params.get('fixed_share_size', 100),
        profit_target_percent=params.get('profit_target_percent', 0.5),
        stop_loss_type=params.get('stop_loss_type', 'opposite_ib'),
        trailing_stop_enabled=params.get('trailing_stop_enabled', False),
        trailing_stop_atr_mult=params.get('trailing_stop_atr_mult', 2.0),
        break_even_enabled=params.get('break_even_enabled', False),
        break_even_pct=params.get('break_even_pct', 0.7),
        max_bars_enabled=params.get('max_bars_enabled', False),
        max_bars=params.get('max_bars', 60),
        eod_exit_time=params.get('eod_exit_time', '15:55'),
        use_qqq_filter=use_qqq_filter,
        min_ib_range_percent=params.get('min_ib_range_percent', 0.0),
        max_ib_range_percent=params.get('max_ib_range_percent', 10.0),
        max_breakout_time=params.get('max_breakout_time', '14:00'),
        trade_monday=params.get('trade_monday', True),
        trade_tuesday=params.get('trade_tuesday', True),
        trade_wednesday=params.get('trade_wednesday', True),
        trade_thursday=params.get('trade_thursday', True),
        trade_friday=params.get('trade_friday', True)
    )

    # Run backtest
    strategy = IBBreakoutStrategy(strategy_params)

    for session in sessions:
        is_first = True
        for bar in session.bars:
            # Get filter bar if QQQ filter is enabled
            filter_bar = filter_bars_dict.get(bar.timestamp) if use_qqq_filter else None
            strategy.process_bar(bar, is_first, qqq_bar=filter_bar)
            is_first = False

    # Calculate metrics
    trades = strategy.get_trades()
    metrics = calculate_metrics(trades)

    # Calculate objective
    obj_value = calculate_objective(metrics, objective)

    run_time = time.time() - start_time

    return OptimizationResult(
        params=params,
        metrics=metrics,
        trades=trades,
        objective_value=obj_value,
        run_time_seconds=run_time
    )


class GridSearchOptimizer:
    """
    Grid search optimizer for IB Breakout strategy.

    Performs exhaustive search over all parameter combinations.
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
        self.filter_bars_dict = None  # For QQQ filter

    def _find_data_file(self, ticker: str) -> Path:
        """Find data file for ticker, preferring NT format."""
        # Prefer NT format files
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
            filter_ticker: Optional filter ticker (e.g., "QQQ") for filter optimization
        """
        self.ticker = ticker

        if data_file:
            filepath = self.data_dir / data_file
        else:
            filepath = self._find_data_file(ticker)

        # Load data
        df = self.loader.load_auto_detect(str(filepath), ticker)

        # Build sessions (using default IB duration - will be recalculated per test)
        session_builder = SessionBuilder()
        self.sessions = session_builder.build_sessions_from_dataframe(df, ticker)

        print(f"Loaded {len(self.sessions)} sessions for {ticker}")

        # Load filter ticker data if specified
        if filter_ticker:
            self._load_filter_data(filter_ticker)

    def _load_filter_data(self, filter_ticker: str):
        """Load filter ticker data (e.g., QQQ) for filter optimization."""
        try:
            filter_file = self._find_data_file(filter_ticker)
            df_filter = self.loader.load_auto_detect(str(filter_file), filter_ticker)

            # Build bar dictionary indexed by timestamp
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
            print("  QQQ filter optimization will be disabled.")
            self.filter_bars_dict = None

    def optimize(
        self,
        parameter_space: Optional[ParameterSpace] = None,
        objective: str = "sharpe_ratio",
        min_trades: int = 10,
        progress_callback: Optional[Callable] = None
    ) -> GridSearchResults:
        """
        Run grid search optimization.

        Args:
            parameter_space: Parameter space to search (uses standard if None)
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
        print("GRID SEARCH OPTIMIZATION")
        print(f"{'='*60}")
        print(f"Ticker: {self.ticker}")
        print(f"Objective: {objective}")
        print(f"Parameter Combinations: {len(combinations):,}")
        print(f"Parallel Jobs: {self.n_jobs}")
        print(f"QQQ Filter Data: {'Available' if self.filter_bars_dict else 'Not loaded'}")
        print(f"{'='*60}\n")

        results = GridSearchResults(
            objective=objective,
            total_combinations=len(combinations),
            ticker=self.ticker
        )

        start_time = time.time()

        # Serialize session data for parallel workers
        sessions_data = []
        for session in self.sessions:
            session_bars = [{
                'timestamp': bar.timestamp,
                'open': bar.open,
                'high': bar.high,
                'low': bar.low,
                'close': bar.close,
                'volume': bar.volume,
                'ticker': bar.ticker
            } for bar in session.bars]
            sessions_data.append({
                'bars': session_bars,
                'date': session.date,
                'ticker': session.ticker,
                'session_start': session.session_start,
                'session_end': session.session_end
            })

        # Serialize filter bars if available
        filter_bars_data = None
        if self.filter_bars_dict:
            filter_bars_data = {
                ts: {
                    'timestamp': bar.timestamp,
                    'open': bar.open,
                    'high': bar.high,
                    'low': bar.low,
                    'close': bar.close,
                    'volume': bar.volume,
                    'ticker': bar.ticker
                }
                for ts, bar in self.filter_bars_dict.items()
            }

        print(f"Starting parallel optimization ({len(combinations):,} combinations) using {self.n_jobs} workers...")

        # Use joblib with a single parallel call for maximum efficiency
        # verbose=10 shows progress every 10% or so
        all_results = Parallel(
            n_jobs=self.n_jobs,
            backend='loky',
            verbose=10
        )(
            delayed(_run_backtest_worker)(
                params=params,
                sessions_data=sessions_data,
                ticker=self.ticker,
                objective=objective,
                filter_bars_data=filter_bars_data
            )
            for params in combinations
        )

        # Add all results
        for result in all_results:
            results.add_result(result)

        # Final progress update
        elapsed = time.time() - start_time
        best_val = results.best_result.objective_value if results.best_result else 0

        print(f"Complete: {len(combinations):,} combinations in {elapsed:.1f}s "
              f"| Best {objective}: {best_val:.4f}")

        if progress_callback:
            progress_callback(len(combinations), len(combinations), results.best_result)

        results.total_time_seconds = time.time() - start_time

        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = self.output_dir / f"grid_search_{self.ticker}_{timestamp}.csv"
        results.save_results(str(results_file))

        # Save best parameters
        if results.best_result:
            best_file = self.output_dir / f"best_params_{self.ticker}_{timestamp}.json"
            with open(best_file, 'w') as f:
                json.dump(results.best_result.params, f, indent=2)
            print(f"Best parameters saved to: {best_file}")

        return results

    def quick_optimize(self, objective: str = "sharpe_ratio") -> GridSearchResults:
        """
        Run quick optimization with minimal parameters.

        Args:
            objective: Objective to optimize

        Returns:
            GridSearchResults
        """
        space = create_parameter_space("quick")
        return self.optimize(space, objective)

    def standard_optimize(self, objective: str = "sharpe_ratio") -> GridSearchResults:
        """
        Run standard optimization.

        Args:
            objective: Objective to optimize

        Returns:
            GridSearchResults
        """
        space = create_parameter_space("standard")
        return self.optimize(space, objective)

    def full_optimize(self, objective: str = "sharpe_ratio") -> GridSearchResults:
        """
        Run full optimization with all parameters.

        WARNING: This can take a very long time!

        Args:
            objective: Objective to optimize

        Returns:
            GridSearchResults
        """
        space = create_parameter_space("full")
        print(f"WARNING: Full optimization with {space.get_grid_size():,} combinations!")
        return self.optimize(space, objective)


def sensitivity_analysis(
    optimizer: GridSearchOptimizer,
    base_params: Dict[str, Any],
    param_to_analyze: str,
    values: List[Any],
    objective: str = "sharpe_ratio"
) -> pd.DataFrame:
    """
    Perform sensitivity analysis on a single parameter.

    Args:
        optimizer: GridSearchOptimizer instance
        base_params: Base parameter values
        param_to_analyze: Parameter to vary
        values: Values to test
        objective: Objective to measure

    Returns:
        DataFrame with results
    """
    results = []

    for value in values:
        params = base_params.copy()
        params[param_to_analyze] = value

        result = run_single_backtest(
            params=params,
            sessions=optimizer.sessions,
            ticker=optimizer.ticker,
            objective=objective
        )

        results.append({
            param_to_analyze: value,
            'objective_value': result.objective_value,
            'total_trades': result.metrics.total_trades,
            'win_rate': result.metrics.percent_profitable,
            'total_pnl': result.metrics.total_net_profit,
            'sharpe_ratio': result.metrics.sharpe_ratio,
            'profit_factor': result.metrics.profit_factor
        })

    return pd.DataFrame(results)


if __name__ == "__main__":
    # Test grid search optimizer
    optimizer = GridSearchOptimizer(r"C:\Users\Warren\Downloads")

    # Load data
    optimizer.load_data("QQQ", "QQQ_1min_20231204_to_20241204_NT.txt")

    # Run quick optimization
    results = optimizer.quick_optimize(objective="sharpe_ratio")

    print("\n" + results.summary())

    # Show top 5 results
    print("\n\nTOP 5 RESULTS:")
    print("-" * 60)
    for i, result in enumerate(results.get_top_n(5)):
        print(f"\n{i+1}. Objective: {result.objective_value:.4f}")
        print(f"   Trades: {result.metrics.total_trades}, Win Rate: {result.metrics.percent_profitable:.1f}%")
        print(f"   P&L: ${result.metrics.total_net_profit:,.2f}, PF: {result.metrics.profit_factor:.2f}")
        print(f"   Params: {result.params}")
