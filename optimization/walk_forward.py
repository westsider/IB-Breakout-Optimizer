"""
Walk-Forward Analysis for IB Breakout Strategy.

Implements robust out-of-sample testing using:
- Rolling window walk-forward
- Anchored (expanding window) walk-forward
- Multi-period validation
"""

import sys
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
import pandas as pd
import numpy as np
import json

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from data.data_loader import DataLoader
from data.session_builder import SessionBuilder
from data.data_types import TradingSession, Trade, Bar
from strategy.ib_breakout import IBBreakoutStrategy, StrategyParams
from metrics.performance_metrics import calculate_metrics, PerformanceMetrics
from optimization.parameter_space import ParameterSpace, create_parameter_space
from optimization.grid_search import GridSearchOptimizer, run_single_backtest, calculate_objective


@dataclass
class WalkForwardPeriod:
    """Single walk-forward period result."""
    period_id: int
    in_sample_start: datetime
    in_sample_end: datetime
    out_sample_start: datetime
    out_sample_end: datetime
    best_params: Dict[str, Any]
    in_sample_metrics: Optional[PerformanceMetrics] = None
    out_sample_metrics: Optional[PerformanceMetrics] = None
    in_sample_objective: float = 0.0
    out_sample_objective: float = 0.0
    optimization_time_seconds: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'period_id': self.period_id,
            'in_sample_start': self.in_sample_start,
            'in_sample_end': self.in_sample_end,
            'out_sample_start': self.out_sample_start,
            'out_sample_end': self.out_sample_end,
            'in_sample_objective': self.in_sample_objective,
            'out_sample_objective': self.out_sample_objective,
            'is_trades': self.in_sample_metrics.total_trades if self.in_sample_metrics else 0,
            'is_win_rate': self.in_sample_metrics.percent_profitable if self.in_sample_metrics else 0,
            'is_pnl': self.in_sample_metrics.total_net_profit if self.in_sample_metrics else 0,
            'os_trades': self.out_sample_metrics.total_trades if self.out_sample_metrics else 0,
            'os_win_rate': self.out_sample_metrics.percent_profitable if self.out_sample_metrics else 0,
            'os_pnl': self.out_sample_metrics.total_net_profit if self.out_sample_metrics else 0,
            **{f'param_{k}': v for k, v in self.best_params.items()}
        }


@dataclass
class WalkForwardResults:
    """Complete walk-forward analysis results."""
    periods: List[WalkForwardPeriod] = field(default_factory=list)
    objective: str = "sharpe_ratio"
    in_sample_days: int = 0
    out_sample_days: int = 0
    anchored: bool = False
    ticker: str = ""
    total_time_seconds: float = 0.0

    # Aggregated metrics
    combined_out_sample_trades: List[Trade] = field(default_factory=list)
    combined_metrics: Optional[PerformanceMetrics] = None

    def add_period(self, period: WalkForwardPeriod):
        """Add a walk-forward period."""
        self.periods.append(period)

    def calculate_combined_metrics(self):
        """Calculate metrics for combined out-of-sample periods."""
        if self.combined_out_sample_trades:
            self.combined_metrics = calculate_metrics(self.combined_out_sample_trades)

    def get_efficiency_ratio(self) -> float:
        """
        Calculate walk-forward efficiency ratio.

        Efficiency = Avg(OOS Objective) / Avg(IS Objective)
        Higher is better - indicates parameters generalize well.
        """
        if not self.periods:
            return 0.0

        is_values = [p.in_sample_objective for p in self.periods if p.in_sample_objective > -900]
        os_values = [p.out_sample_objective for p in self.periods if p.out_sample_objective > -900]

        if not is_values or not os_values:
            return 0.0

        avg_is = np.mean(is_values)
        avg_os = np.mean(os_values)

        if avg_is <= 0:
            return 0.0

        return avg_os / avg_is

    def to_dataframe(self) -> pd.DataFrame:
        """Convert all periods to DataFrame."""
        rows = [p.to_dict() for p in self.periods]
        return pd.DataFrame(rows)

    def save_results(self, filepath: str):
        """Save results to CSV."""
        df = self.to_dataframe()
        df.to_csv(filepath, index=False)

    def summary(self) -> str:
        """Get summary of results."""
        lines = [
            "=" * 60,
            "WALK-FORWARD ANALYSIS RESULTS",
            "=" * 60,
            f"Ticker: {self.ticker}",
            f"Objective: {self.objective}",
            f"Mode: {'Anchored' if self.anchored else 'Rolling'}",
            f"In-Sample Days: {self.in_sample_days}",
            f"Out-of-Sample Days: {self.out_sample_days}",
            f"Total Periods: {len(self.periods)}",
            f"Total Time: {self.total_time_seconds:.1f} seconds",
            "",
            f"Walk-Forward Efficiency: {self.get_efficiency_ratio():.2%}",
            ""
        ]

        # Period-by-period summary
        lines.append("PERIOD SUMMARY:")
        lines.append("-" * 60)
        lines.append(f"{'Period':<8} {'IS Obj':>10} {'OS Obj':>10} {'IS Trades':>10} {'OS Trades':>10} {'OS P&L':>12}")
        lines.append("-" * 60)

        for p in self.periods:
            is_trades = p.in_sample_metrics.total_trades if p.in_sample_metrics else 0
            os_trades = p.out_sample_metrics.total_trades if p.out_sample_metrics else 0
            os_pnl = p.out_sample_metrics.total_net_profit if p.out_sample_metrics else 0

            lines.append(f"{p.period_id:<8} {p.in_sample_objective:>10.4f} {p.out_sample_objective:>10.4f} "
                        f"{is_trades:>10} {os_trades:>10} ${os_pnl:>10,.2f}")

        # Combined out-of-sample
        if self.combined_metrics:
            lines.extend([
                "",
                "COMBINED OUT-OF-SAMPLE RESULTS:",
                "-" * 40,
                f"Total Trades: {self.combined_metrics.total_trades}",
                f"Win Rate: {self.combined_metrics.percent_profitable:.1f}%",
                f"Total P&L: ${self.combined_metrics.total_net_profit:,.2f}",
                f"Profit Factor: {self.combined_metrics.profit_factor:.2f}",
                f"Sharpe Ratio: {self.combined_metrics.sharpe_ratio:.2f}",
                f"Max Drawdown: ${self.combined_metrics.max_drawdown:,.2f}",
            ])

        return "\n".join(lines)


class WalkForwardAnalyzer:
    """
    Walk-forward analysis for strategy robustness testing.

    Tests parameters on out-of-sample data after optimizing on in-sample.
    """

    def __init__(
        self,
        data_dir: str,
        output_dir: Optional[str] = None
    ):
        """
        Initialize analyzer.

        Args:
            data_dir: Directory containing data files
            output_dir: Directory for output files
        """
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir) if output_dir else self.data_dir / "walk_forward_results"
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.loader = DataLoader(str(self.data_dir))
        self.sessions = None
        self.df = None
        self.ticker = None
        self.filter_bars_dict = None  # For QQQ filter

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

        raise FileNotFoundError(f"No data file found for {ticker}")

    def load_data(self, ticker: str, data_file: Optional[str] = None,
                  filter_ticker: Optional[str] = None):
        """
        Load and prepare data.

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

        self.df = self.loader.load_auto_detect(str(filepath), ticker)
        print(f"Loaded data for {ticker}: {self.df['timestamp'].min()} to {self.df['timestamp'].max()}")

        # Load filter ticker data if specified
        if filter_ticker:
            self._load_filter_data(filter_ticker)

    def _load_filter_data(self, filter_ticker: str):
        """Load filter ticker data (e.g., QQQ) for filter optimization."""
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

    def _get_sessions_for_period(
        self,
        start_date: datetime,
        end_date: datetime
    ) -> List[TradingSession]:
        """Get sessions for a date range."""
        mask = (self.df['timestamp'] >= start_date) & (self.df['timestamp'] <= end_date)
        period_df = self.df[mask].copy()

        if period_df.empty:
            return []

        session_builder = SessionBuilder()
        return session_builder.build_sessions_from_dataframe(period_df, self.ticker)

    def _optimize_period(
        self,
        sessions: List[TradingSession],
        parameter_space: ParameterSpace,
        objective: str
    ) -> Tuple[Dict[str, Any], float, PerformanceMetrics]:
        """
        Optimize parameters on a set of sessions.

        Returns:
            Tuple of (best_params, best_objective, best_metrics)
        """
        combinations = parameter_space.get_grid_combinations()

        best_params = None
        best_objective = -float('inf')
        best_metrics = None

        for params in combinations:
            result = run_single_backtest(
                params=params,
                sessions=sessions,
                ticker=self.ticker,
                objective=objective,
                filter_bars_dict=self.filter_bars_dict
            )

            if result.objective_value > best_objective:
                best_objective = result.objective_value
                best_params = params
                best_metrics = result.metrics

        return best_params, best_objective, best_metrics

    def _test_period(
        self,
        sessions: List[TradingSession],
        params: Dict[str, Any],
        objective: str
    ) -> Tuple[float, PerformanceMetrics, List[Trade]]:
        """
        Test parameters on out-of-sample sessions.

        Returns:
            Tuple of (objective_value, metrics, trades)
        """
        result = run_single_backtest(
            params=params,
            sessions=sessions,
            ticker=self.ticker,
            objective=objective,
            filter_bars_dict=self.filter_bars_dict
        )

        return result.objective_value, result.metrics, result.trades

    def analyze(
        self,
        in_sample_days: int = 180,
        out_sample_days: int = 30,
        anchored: bool = False,
        parameter_space: Optional[ParameterSpace] = None,
        objective: str = "sharpe_ratio",
        min_trades_per_period: int = 10
    ) -> WalkForwardResults:
        """
        Run walk-forward analysis.

        Args:
            in_sample_days: Number of days for in-sample (optimization)
            out_sample_days: Number of days for out-of-sample (testing)
            anchored: If True, use expanding window; if False, use rolling window
            parameter_space: Parameter space to optimize
            objective: Objective to optimize
            min_trades_per_period: Minimum trades required per period

        Returns:
            WalkForwardResults
        """
        if self.df is None:
            raise ValueError("No data loaded. Call load_data() first.")

        space = parameter_space or create_parameter_space("quick")

        print(f"\n{'='*60}")
        print("WALK-FORWARD ANALYSIS")
        print(f"{'='*60}")
        print(f"Ticker: {self.ticker}")
        print(f"Mode: {'Anchored' if anchored else 'Rolling'}")
        print(f"In-Sample: {in_sample_days} days")
        print(f"Out-of-Sample: {out_sample_days} days")
        print(f"Objective: {objective}")
        print(f"Parameter Combinations: {space.get_grid_size()}")
        print(f"{'='*60}\n")

        results = WalkForwardResults(
            objective=objective,
            in_sample_days=in_sample_days,
            out_sample_days=out_sample_days,
            anchored=anchored,
            ticker=self.ticker
        )

        # Get date range
        data_start = self.df['timestamp'].min()
        data_end = self.df['timestamp'].max()

        # Calculate periods
        total_days = (data_end - data_start).days
        period_length = in_sample_days + out_sample_days

        if total_days < period_length:
            print(f"Error: Not enough data. Need {period_length} days, have {total_days} days")
            return results

        # Generate walk-forward periods
        import time
        start_time = time.time()

        period_id = 1
        anchor_start = data_start

        if anchored:
            # Anchored: IS window grows, OOS window slides
            current_is_end = data_start + timedelta(days=in_sample_days)
        else:
            # Rolling: Both windows slide
            current_is_start = data_start

        while True:
            if anchored:
                is_start = anchor_start
                is_end = current_is_end
                os_start = is_end
                os_end = os_start + timedelta(days=out_sample_days)

                if os_end > data_end:
                    break

                current_is_end = os_end  # Expand IS window

            else:  # Rolling
                is_start = current_is_start
                is_end = is_start + timedelta(days=in_sample_days)
                os_start = is_end
                os_end = os_start + timedelta(days=out_sample_days)

                if os_end > data_end:
                    break

                current_is_start = current_is_start + timedelta(days=out_sample_days)  # Slide

            print(f"\nPeriod {period_id}:")
            print(f"  In-Sample: {is_start.date()} to {is_end.date()}")
            print(f"  Out-of-Sample: {os_start.date()} to {os_end.date()}")

            # Get sessions
            is_sessions = self._get_sessions_for_period(is_start, is_end)
            os_sessions = self._get_sessions_for_period(os_start, os_end)

            print(f"  IS Sessions: {len(is_sessions)}, OS Sessions: {len(os_sessions)}")

            if not is_sessions or not os_sessions:
                print("  Skipping - insufficient sessions")
                period_id += 1
                continue

            # Optimize on in-sample
            opt_start = time.time()
            best_params, is_objective, is_metrics = self._optimize_period(
                is_sessions, space, objective
            )
            opt_time = time.time() - opt_start

            print(f"  IS Objective: {is_objective:.4f} ({is_metrics.total_trades} trades)")

            # Test on out-of-sample
            os_objective, os_metrics, os_trades = self._test_period(
                os_sessions, best_params, objective
            )

            print(f"  OS Objective: {os_objective:.4f} ({os_metrics.total_trades} trades)")

            # Create period result
            period = WalkForwardPeriod(
                period_id=period_id,
                in_sample_start=is_start,
                in_sample_end=is_end,
                out_sample_start=os_start,
                out_sample_end=os_end,
                best_params=best_params,
                in_sample_metrics=is_metrics,
                out_sample_metrics=os_metrics,
                in_sample_objective=is_objective,
                out_sample_objective=os_objective,
                optimization_time_seconds=opt_time
            )

            results.add_period(period)
            results.combined_out_sample_trades.extend(os_trades)

            period_id += 1

        results.total_time_seconds = time.time() - start_time
        results.calculate_combined_metrics()

        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = self.output_dir / f"wf_{self.ticker}_{timestamp}.csv"
        results.save_results(str(results_file))
        print(f"\nResults saved to: {results_file}")

        return results


def quick_walk_forward(
    ticker: str,
    data_dir: str = r"C:\Users\Warren\Downloads",
    in_sample_days: int = 120,
    out_sample_days: int = 30
) -> WalkForwardResults:
    """
    Quick walk-forward analysis helper.

    Args:
        ticker: Ticker symbol
        data_dir: Data directory
        in_sample_days: In-sample period length
        out_sample_days: Out-of-sample period length

    Returns:
        WalkForwardResults
    """
    analyzer = WalkForwardAnalyzer(data_dir)
    analyzer.load_data(ticker)

    space = create_parameter_space("quick")
    return analyzer.analyze(
        in_sample_days=in_sample_days,
        out_sample_days=out_sample_days,
        parameter_space=space
    )


if __name__ == "__main__":
    # Test walk-forward analysis
    analyzer = WalkForwardAnalyzer(r"C:\Users\Warren\Downloads")
    analyzer.load_data("QQQ", "QQQ_1min_20231204_to_20241204_NT.txt")

    results = analyzer.analyze(
        in_sample_days=90,
        out_sample_days=30,
        anchored=False,
        parameter_space=create_parameter_space("quick"),
        objective="sharpe_ratio"
    )

    print("\n" + results.summary())
