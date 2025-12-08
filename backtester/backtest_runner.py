"""
Backtest Runner for IB Breakout Optimizer.

Main orchestrator for running backtests with the IB Breakout strategy.
Handles data loading, session building, strategy execution, and metrics calculation.
"""

import os
import sys
from datetime import datetime, date
from typing import List, Dict, Optional, Tuple
from pathlib import Path
import pandas as pd

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from data.data_loader import DataLoader
from data.session_builder import SessionBuilder, MultiTickerSessionBuilder
from data.data_types import Bar, TradingSession, Trade, BacktestResult
from strategy.ib_breakout import IBBreakoutStrategy, StrategyParams
from metrics.performance_metrics import calculate_metrics, PerformanceMetrics


class BacktestRunner:
    """
    Runs backtests for the IB Breakout strategy.

    Handles:
    - Data loading and session building
    - Single-ticker and multi-ticker backtests
    - Strategy execution
    - Performance metrics calculation
    - Results storage
    """

    def __init__(
        self,
        data_dir: str,
        output_dir: Optional[str] = None,
        initial_capital: float = 100000.0
    ):
        """
        Initialize backtest runner.

        Args:
            data_dir: Directory containing data files
            output_dir: Directory for output files (optional)
            initial_capital: Starting capital
        """
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir) if output_dir else self.data_dir / "output"
        self.initial_capital = initial_capital

        # Create output directory if needed
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize components
        self.loader = DataLoader(str(self.data_dir))
        self.session_builder = None  # Created per-backtest based on params

        # Results storage
        self.last_result: Optional[BacktestResult] = None
        self.last_metrics: Optional[PerformanceMetrics] = None

    def run_backtest(
        self,
        ticker: str,
        params: Optional[StrategyParams] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        data_file: Optional[str] = None,
        verbose: bool = True
    ) -> Tuple[BacktestResult, PerformanceMetrics]:
        """
        Run a backtest for a single ticker.

        Args:
            ticker: Ticker symbol (e.g., "TSLA")
            params: Strategy parameters (uses defaults if None)
            start_date: Filter start date (optional)
            end_date: Filter end date (optional)
            data_file: Specific data file to use (optional)
            verbose: Print progress

        Returns:
            Tuple of (BacktestResult, PerformanceMetrics)
        """
        params = params or StrategyParams()

        if verbose:
            print(f"\n{'='*60}")
            print(f"Running Backtest: {ticker}")
            print(f"{'='*60}")

        # Load data
        if data_file:
            filepath = self.data_dir / data_file
        else:
            filepath = self._find_data_file(ticker)

        if verbose:
            print(f"Loading data from: {filepath.name}")

        df = self.loader.load_auto_detect(str(filepath), ticker)

        # Filter by date range
        if start_date:
            df = df[df['timestamp'] >= start_date]
        if end_date:
            df = df[df['timestamp'] <= end_date]

        if df.empty:
            raise ValueError(f"No data in specified date range for {ticker}")

        if verbose:
            print(f"Data range: {df['timestamp'].min()} to {df['timestamp'].max()}")
            print(f"Total bars: {len(df):,}")

        # Build sessions
        self.session_builder = SessionBuilder(
            market_open="09:30",
            market_close="16:00",
            ib_duration_minutes=params.ib_duration_minutes
        )
        sessions = self.session_builder.build_sessions_from_dataframe(df, ticker)

        if verbose:
            print(f"Trading sessions: {len(sessions)}")

        # Create and run strategy
        strategy = IBBreakoutStrategy(params)

        # Calculate statistical filter data if any filter mode is not 'any'
        if (params.gap_filter_mode != 'any' or
            params.trend_filter_mode != 'any' or
            params.range_filter_mode != 'any'):
            strategy.calculate_filter_data_from_df(
                df,
                trend_lookback=params.prior_days_lookback,
                range_lookback=params.daily_range_lookback
            )
            if verbose:
                print(f"  Statistical filters: gap={params.gap_filter_mode}, trend={params.trend_filter_mode}, range={params.range_filter_mode}")

        if verbose:
            print(f"\nRunning strategy...")
            print(f"  IB Duration: {params.ib_duration_minutes} min")
            print(f"  Target: {params.profit_target_percent}%")
            print(f"  Stop: {params.stop_loss_type}")
            print(f"  Direction: {params.trade_direction}")

        # Process each session
        signals = []
        for session in sessions:
            is_first = True
            for bar in session.bars:
                signal = strategy.process_bar(bar, is_first)
                is_first = False

                if signal:
                    signals.append(signal)
                    if verbose:
                        print(f"  [{bar.timestamp}] {signal.signal_type.value}: {signal.reason}")

        # Get trades
        trades = strategy.get_trades()

        if verbose:
            print(f"\nTotal signals: {len(signals)}")
            print(f"Total trades: {len(trades)}")

        # Calculate metrics
        metrics = calculate_metrics(trades, self.initial_capital)

        # Create result object
        result = BacktestResult(
            start_date=df['timestamp'].min().date() if hasattr(df['timestamp'].min(), 'date') else df['timestamp'].min(),
            end_date=df['timestamp'].max().date() if hasattr(df['timestamp'].max(), 'date') else df['timestamp'].max(),
            tickers=[ticker],
            trades=trades,
            parameters=vars(params)
        )

        # Store results
        self.last_result = result
        self.last_metrics = metrics

        if verbose:
            print(f"\n{'='*60}")
            print("RESULTS SUMMARY")
            print(f"{'='*60}")
            summary = strategy.get_trade_summary()
            print(f"Total Trades: {summary['total_trades']}")
            print(f"Win Rate: {summary['win_rate']:.1f}%")
            print(f"Total P&L: ${summary['total_pnl']:.2f}")
            print(f"Profit Factor: {summary['profit_factor']:.2f}")
            print(f"Max Drawdown: ${metrics.max_drawdown:.2f}")
            print(f"Sharpe Ratio: {metrics.sharpe_ratio:.2f}")

        return result, metrics

    def run_backtest_with_filter(
        self,
        ticker: str,
        filter_ticker: str = "QQQ",
        params: Optional[StrategyParams] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        verbose: bool = True
    ) -> Tuple[BacktestResult, PerformanceMetrics]:
        """
        Run a backtest with QQQ (or other) filter enabled.

        Loads both the primary ticker and filter ticker data, synchronizes
        bars by timestamp, and passes both to the strategy.

        Args:
            ticker: Primary ticker symbol (e.g., "TSLA")
            filter_ticker: Filter ticker symbol (default "QQQ")
            params: Strategy parameters (use_qqq_filter should be True)
            start_date: Filter start date (optional)
            end_date: Filter end date (optional)
            verbose: Print progress

        Returns:
            Tuple of (BacktestResult, PerformanceMetrics)
        """
        params = params or StrategyParams(use_qqq_filter=True)

        # Ensure QQQ filter is enabled
        if not params.use_qqq_filter:
            params.use_qqq_filter = True

        if verbose:
            print(f"\n{'='*60}")
            print(f"Running Backtest with {filter_ticker} Filter: {ticker}")
            print(f"{'='*60}")

        # Find data files for both tickers
        primary_file = self._find_data_file(ticker)
        filter_file = self._find_data_file(filter_ticker)

        if verbose:
            print(f"Primary data: {primary_file.name}")
            print(f"Filter data: {filter_file.name}")

        # Load both datasets
        df_primary = self.loader.load_auto_detect(str(primary_file), ticker)
        df_filter = self.loader.load_auto_detect(str(filter_file), filter_ticker)

        # Filter by date range
        if start_date:
            df_primary = df_primary[df_primary['timestamp'] >= start_date]
            df_filter = df_filter[df_filter['timestamp'] >= start_date]
        if end_date:
            df_primary = df_primary[df_primary['timestamp'] <= end_date]
            df_filter = df_filter[df_filter['timestamp'] <= end_date]

        if df_primary.empty:
            raise ValueError(f"No data in specified date range for {ticker}")
        if df_filter.empty:
            raise ValueError(f"No data in specified date range for {filter_ticker}")

        if verbose:
            print(f"Primary range: {df_primary['timestamp'].min()} to {df_primary['timestamp'].max()}")
            print(f"Filter range: {df_filter['timestamp'].min()} to {df_filter['timestamp'].max()}")
            print(f"Primary bars: {len(df_primary):,}")
            print(f"Filter bars: {len(df_filter):,}")

        # Build synchronized bar dictionary for filter ticker
        # Key: timestamp, Value: Bar object
        filter_bars_dict = {}
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
            filter_bars_dict[row['timestamp']] = bar

        if verbose:
            print(f"Filter bars indexed: {len(filter_bars_dict):,}")

        # Build sessions for primary ticker
        self.session_builder = SessionBuilder(
            market_open="09:30",
            market_close="16:00",
            ib_duration_minutes=params.ib_duration_minutes
        )
        sessions = self.session_builder.build_sessions_from_dataframe(df_primary, ticker)

        if verbose:
            print(f"Trading sessions: {len(sessions)}")

        # Create and run strategy
        strategy = IBBreakoutStrategy(params)

        if verbose:
            print(f"\nRunning strategy with {filter_ticker} filter...")
            print(f"  IB Duration: {params.ib_duration_minutes} min")
            print(f"  Target: {params.profit_target_percent}%")
            print(f"  Stop: {params.stop_loss_type}")
            print(f"  Direction: {params.trade_direction}")
            print(f"  {filter_ticker} Filter: ON")

        # Process each session
        signals = []
        bars_with_filter = 0
        bars_without_filter = 0

        for session in sessions:
            is_first = True
            for bar in session.bars:
                # Look up corresponding filter bar
                filter_bar = filter_bars_dict.get(bar.timestamp)

                if filter_bar:
                    bars_with_filter += 1
                else:
                    bars_without_filter += 1

                # Process bar with filter data
                signal = strategy.process_bar(bar, is_first, qqq_bar=filter_bar)
                is_first = False

                if signal:
                    signals.append(signal)
                    if verbose:
                        print(f"  [{bar.timestamp}] {signal.signal_type.value}: {signal.reason}")

        if verbose:
            print(f"\nBar synchronization:")
            print(f"  Bars with {filter_ticker} data: {bars_with_filter:,}")
            print(f"  Bars missing {filter_ticker} data: {bars_without_filter:,}")
            sync_rate = bars_with_filter / (bars_with_filter + bars_without_filter) * 100 if (bars_with_filter + bars_without_filter) > 0 else 0
            print(f"  Sync rate: {sync_rate:.1f}%")

        # Get trades
        trades = strategy.get_trades()

        if verbose:
            print(f"\nTotal signals: {len(signals)}")
            print(f"Total trades: {len(trades)}")

        # Calculate metrics
        metrics = calculate_metrics(trades, self.initial_capital)

        # Create result object
        result = BacktestResult(
            start_date=df_primary['timestamp'].min().date() if hasattr(df_primary['timestamp'].min(), 'date') else df_primary['timestamp'].min(),
            end_date=df_primary['timestamp'].max().date() if hasattr(df_primary['timestamp'].max(), 'date') else df_primary['timestamp'].max(),
            tickers=[ticker],
            trades=trades,
            parameters=vars(params)
        )

        # Store results
        self.last_result = result
        self.last_metrics = metrics

        if verbose:
            print(f"\n{'='*60}")
            print("RESULTS SUMMARY")
            print(f"{'='*60}")
            summary = strategy.get_trade_summary()
            print(f"Total Trades: {summary['total_trades']}")
            print(f"Win Rate: {summary['win_rate']:.1f}%")
            print(f"Total P&L: ${summary['total_pnl']:.2f}")
            print(f"Profit Factor: {summary['profit_factor']:.2f}")
            print(f"Max Drawdown: ${metrics.max_drawdown:.2f}")
            print(f"Sharpe Ratio: {metrics.sharpe_ratio:.2f}")

        return result, metrics

    def _find_data_file(self, ticker: str) -> Path:
        """
        Find data file for a ticker.

        Prefers NinjaTrader format files (_NT.txt) over other formats.

        Args:
            ticker: Ticker symbol

        Returns:
            Path to data file

        Raises:
            FileNotFoundError if no file found
        """
        # Try wildcard match first, preferring NT format
        nt_files = []
        other_files = []

        for f in self.data_dir.iterdir():
            if f.is_file() and ticker.upper() in f.name.upper():
                if f.suffix.lower() in ['.txt', '.csv']:
                    if '_NT' in f.name.upper() or 'NT.TXT' in f.name.upper():
                        nt_files.append(f)
                    else:
                        other_files.append(f)

        # Prefer NT format files
        if nt_files:
            return nt_files[0]
        if other_files:
            return other_files[0]

        # Try exact patterns as fallback
        patterns = [
            f"{ticker}_1min_NT.txt",
            f"{ticker}_NT.txt",
            f"{ticker}.txt",
            f"{ticker}_1min.csv",
            f"{ticker}.csv"
        ]

        for pattern in patterns:
            test_path = self.data_dir / pattern
            if test_path.exists():
                return test_path

        raise FileNotFoundError(f"No data file found for {ticker} in {self.data_dir}")

    def run_multi_ticker_backtest(
        self,
        tickers: List[str],
        params: Optional[StrategyParams] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        verbose: bool = True
    ) -> Dict[str, Tuple[BacktestResult, PerformanceMetrics]]:
        """
        Run backtests for multiple tickers.

        Args:
            tickers: List of ticker symbols
            params: Strategy parameters
            start_date: Filter start date
            end_date: Filter end date
            verbose: Print progress

        Returns:
            Dict mapping ticker -> (BacktestResult, PerformanceMetrics)
        """
        results = {}

        for ticker in tickers:
            try:
                result, metrics = self.run_backtest(
                    ticker=ticker,
                    params=params,
                    start_date=start_date,
                    end_date=end_date,
                    verbose=verbose
                )
                results[ticker] = (result, metrics)
            except Exception as e:
                print(f"Error running backtest for {ticker}: {e}")
                continue

        return results

    def print_full_report(self):
        """Print detailed performance report."""
        if self.last_metrics:
            self.last_metrics.print_report()

    def export_trades(self, filename: str = "trades.csv"):
        """Export trades to CSV."""
        if not self.last_result or not self.last_result.trades:
            print("No trades to export")
            return

        trades_data = []
        for trade in self.last_result.trades:
            trades_data.append({
                'trade_id': trade.trade_id,
                'ticker': trade.ticker,
                'direction': trade.direction.value,
                'entry_time': trade.entry_time,
                'entry_price': trade.entry_price,
                'exit_time': trade.exit_time,
                'exit_price': trade.exit_price,
                'quantity': trade.quantity,
                'pnl': trade.pnl,
                'pnl_pct': trade.pnl_pct,
                'exit_reason': trade.exit_reason.value if trade.exit_reason else None,
                'bars_held': trade.bars_held,
                'ib_high': trade.ib.ib_high if trade.ib else None,
                'ib_low': trade.ib.ib_low if trade.ib else None,
                'ib_range_pct': trade.ib.ib_range_pct if trade.ib else None
            })

        df = pd.DataFrame(trades_data)
        output_path = self.output_dir / filename
        df.to_csv(output_path, index=False)
        print(f"Trades exported to: {output_path}")

    def export_metrics(self, filename: str = "metrics.csv"):
        """Export metrics to CSV."""
        if not self.last_metrics:
            print("No metrics to export")
            return

        metrics_dict = self.last_metrics.to_dict()

        # Flatten nested dict
        rows = []
        for category, items in metrics_dict.items():
            for metric, value in items.items():
                rows.append({
                    'Category': category,
                    'Metric': metric,
                    'Value': value
                })

        df = pd.DataFrame(rows)
        output_path = self.output_dir / filename
        df.to_csv(output_path, index=False)
        print(f"Metrics exported to: {output_path}")


def main():
    """
    Main function to run a backtest from command line.
    """
    import argparse

    parser = argparse.ArgumentParser(description="Run IB Breakout Strategy Backtest")
    parser.add_argument("ticker", help="Ticker symbol (e.g., TSLA)")
    parser.add_argument("--data-dir", default=r"C:\Users\Warren\Downloads",
                        help="Directory containing data files")
    parser.add_argument("--ib-duration", type=int, default=30,
                        help="IB duration in minutes")
    parser.add_argument("--target-pct", type=float, default=0.5,
                        help="Profit target percentage")
    parser.add_argument("--stop-type", choices=["opposite_ib", "match_target"],
                        default="opposite_ib", help="Stop loss type")
    parser.add_argument("--direction", choices=["both", "long_only", "short_only"],
                        default="both", help="Trade direction")
    parser.add_argument("--start-date", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end-date", help="End date (YYYY-MM-DD)")
    parser.add_argument("--export", action="store_true",
                        help="Export trades and metrics to CSV")

    args = parser.parse_args()

    # Create params
    params = StrategyParams(
        ib_duration_minutes=args.ib_duration,
        profit_target_percent=args.target_pct,
        stop_loss_type=args.stop_type,
        trade_direction=args.direction
    )

    # Parse dates
    start_date = datetime.strptime(args.start_date, "%Y-%m-%d") if args.start_date else None
    end_date = datetime.strptime(args.end_date, "%Y-%m-%d") if args.end_date else None

    # Run backtest
    runner = BacktestRunner(args.data_dir)
    result, metrics = runner.run_backtest(
        ticker=args.ticker,
        params=params,
        start_date=start_date,
        end_date=end_date
    )

    # Print report
    runner.print_full_report()

    # Export if requested
    if args.export:
        runner.export_trades()
        runner.export_metrics()


if __name__ == "__main__":
    # Default test run
    runner = BacktestRunner(r"C:\Users\Warren\Downloads")

    # Look for any available data file
    data_dir = Path(r"C:\Users\Warren\Downloads")
    for f in data_dir.iterdir():
        if f.suffix in ['.txt', '.csv'] and any(t in f.name.upper() for t in ['TSLA', 'QQQ', 'AAPL']):
            ticker = f.stem.split("_")[0].upper()
            print(f"Found data file for {ticker}: {f.name}")

            try:
                result, metrics = runner.run_backtest(
                    ticker=ticker,
                    data_file=f.name,
                    params=StrategyParams(
                        use_qqq_filter=False,
                        profit_target_percent=0.5,
                        stop_loss_type="opposite_ib"
                    )
                )

                print("\n" + "="*60)
                print("FULL PERFORMANCE REPORT")
                print("="*60)
                runner.print_full_report()

                # Export results
                runner.export_trades(f"{ticker}_trades.csv")
                runner.export_metrics(f"{ticker}_metrics.csv")

                break
            except Exception as e:
                print(f"Error: {e}")
                import traceback
                traceback.print_exc()
