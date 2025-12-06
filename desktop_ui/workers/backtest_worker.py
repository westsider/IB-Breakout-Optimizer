"""
Backtest Worker - Run full backtests in a background thread.

Uses BacktestRunner to generate actual Trade objects with all details
needed for Trade Browser, Equity Curve, and IB Analysis tabs.
"""

from PySide6.QtCore import QThread, Signal
import traceback
from pathlib import Path


class BacktestWorker(QThread):
    """Worker thread for running full backtests with trade objects."""

    finished = Signal(object, object)  # trades list, metrics
    error = Signal(str)
    status = Signal(str)

    def __init__(self, data_dir: str, params: dict, ticker: str):
        super().__init__()
        self.data_dir = data_dir
        self.params = params
        self.ticker = ticker

    def run(self):
        """Execute the full backtest using BacktestRunner."""
        try:
            use_qqq = self.params.get('use_qqq_filter', False)
            self.status.emit(f"Running backtest for {self.ticker} (QQQ filter: {use_qqq})...")

            from backtester.backtest_runner import BacktestRunner
            from strategy.ib_breakout import StrategyParams

            # Build StrategyParams from the optimization result params
            strategy_params = StrategyParams(
                ib_duration_minutes=self.params.get('ib_duration_minutes', 30),
                profit_target_percent=self.params.get('profit_target_percent', 0.5),
                stop_loss_type=self.params.get('stop_loss_type', 'opposite_ib'),
                trade_direction=self.params.get('trade_direction', 'both'),
                use_qqq_filter=use_qqq,
                min_ib_range_percent=self.params.get('min_ib_range_percent', 0.0),
                max_ib_range_percent=self.params.get('max_ib_range_percent', 100.0),
                max_breakout_time=self.params.get('max_breakout_time', '14:00'),
                eod_exit_time=self.params.get('eod_exit_time', '15:55'),
                trailing_stop_enabled=self.params.get('trailing_stop_enabled', False),
                trailing_stop_atr_mult=self.params.get('trailing_stop_atr_mult', 2.0),
                break_even_enabled=self.params.get('break_even_enabled', False),
                break_even_pct=self.params.get('break_even_pct', 0.5),
                max_bars_enabled=self.params.get('max_bars_enabled', False),
                max_bars=self.params.get('max_bars', 60),
                fixed_share_size=self.params.get('fixed_share_size', 100),
            )

            # Create runner and run backtest
            runner = BacktestRunner(self.data_dir)

            # Check if QQQ filter is needed
            if strategy_params.use_qqq_filter:
                result, metrics = runner.run_backtest_with_filter(
                    ticker=self.ticker,
                    filter_ticker="QQQ",
                    params=strategy_params,
                    verbose=False
                )
            else:
                result, metrics = runner.run_backtest(
                    ticker=self.ticker,
                    params=strategy_params,
                    verbose=False
                )

            self.status.emit(f"Backtest complete: {len(result.trades)} trades")
            self.finished.emit(result.trades, metrics)

        except Exception as e:
            self.error.emit(f"{str(e)}\n{traceback.format_exc()}")
