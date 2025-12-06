"""
Backtest Worker - Run backtests in a background thread.
"""

from PySide6.QtCore import QThread, Signal
import traceback


class BacktestWorker(QThread):
    """Worker thread for running backtests."""

    finished = Signal(object, object)  # result, metrics
    error = Signal(str)

    def __init__(self, data_dir: str, params: dict):
        super().__init__()
        self.data_dir = data_dir
        self.params = params

    def run(self):
        """Execute the backtest."""
        try:
            from backtester.backtest_runner import BacktestRunner
            from strategy.ib_breakout import StrategyParams

            runner = BacktestRunner(self.data_dir)

            # Build strategy params
            strategy_params = StrategyParams(
                ib_duration_minutes=self.params['ib_duration_minutes'],
                profit_target_percent=self.params['profit_target_percent'],
                stop_loss_type=self.params['stop_loss_type'],
                trade_direction=self.params['trade_direction'],
                use_qqq_filter=self.params['use_qqq_filter'],
                min_ib_range_percent=self.params['min_ib_range_percent'],
                max_ib_range_percent=self.params['max_ib_range_percent'],
                max_breakout_time=self.params['max_breakout_time'],
                eod_exit_time=self.params['eod_exit_time'],
                trailing_stop_enabled=self.params['trailing_stop_enabled'],
                trailing_stop_atr_mult=self.params['trailing_stop_atr_mult'],
                break_even_enabled=self.params['break_even_enabled'],
                break_even_pct=self.params['break_even_pct'],
                max_bars_enabled=self.params['max_bars_enabled'],
                max_bars=self.params['max_bars'],
            )

            ticker = self.params['ticker']
            use_qqq_filter = self.params['use_qqq_filter']

            # Run backtest
            if use_qqq_filter and ticker != 'QQQ':
                result, metrics = runner.run_backtest_with_filter(
                    ticker=ticker,
                    filter_ticker='QQQ',
                    params=strategy_params,
                    verbose=False
                )
            else:
                result, metrics = runner.run_backtest(
                    ticker=ticker,
                    params=strategy_params,
                    verbose=False
                )

            self.finished.emit(result, metrics)

        except Exception as e:
            self.error.emit(f"{str(e)}\n{traceback.format_exc()}")
