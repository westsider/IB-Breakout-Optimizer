"""
Walk-Forward Analysis Tab - Rolling optimization with out-of-sample testing.

Performs walk-forward analysis by:
1. Training on a rolling window of historical data
2. Selecting the best parameters based on in-sample performance
3. Testing those parameters on the following out-of-sample period
4. Rolling forward and repeating
"""

from datetime import datetime, timedelta
from dataclasses import dataclass
from typing import List, Optional
import json
from pathlib import Path

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QLabel, QComboBox, QPushButton, QGroupBox,
    QTableWidget, QTableWidgetItem, QHeaderView, QProgressBar,
    QSplitter, QFrame
)
from PySide6.QtCore import Qt, Signal, QThread, QSettings
from PySide6.QtGui import QColor
from PySide6.QtWebEngineWidgets import QWebEngineView

import plotly.graph_objects as go
from plotly.subplots import make_subplots

from desktop_ui.widgets.metrics_panel import MetricCard


@dataclass
class WalkForwardPeriod:
    """Results from a single walk-forward period."""
    period_num: int
    train_start: str
    train_end: str
    test_start: str
    test_end: str
    best_params: dict
    # In-sample (training) metrics
    is_trades: int
    is_pnl: float
    is_pf: float
    is_win_rate: float
    # Out-of-sample (test) metrics
    oos_trades: int
    oos_pnl: float
    oos_pf: float
    oos_win_rate: float
    oos_equity_curve: List[float]


class WalkForwardWorker(QThread):
    """Worker thread for running walk-forward analysis."""
    progress = Signal(int, int, str)  # current, total, message
    period_complete = Signal(dict)  # Single period results
    finished = Signal(dict)  # Final aggregated results
    error = Signal(str)

    def __init__(self, data_dir: str, ticker: str, train_months: int,
                 test_weeks: int, preset: str, objective: str,
                 gap_filter: str, trend_filter: str, range_filter: str):
        super().__init__()
        self.data_dir = data_dir
        self.ticker = ticker
        self.train_months = train_months
        self.test_weeks = test_weeks
        self.preset = preset
        self.objective = objective
        self.gap_filter = gap_filter
        self.trend_filter = trend_filter
        self.range_filter = range_filter
        self._cancelled = False

    def cancel(self):
        self._cancelled = True

    def run(self):
        try:
            from optimization.mmap_grid_search import MMapGridSearchOptimizer
            from optimization.parameter_space import create_parameter_space
            from backtester.backtest_runner import BacktestRunner
            from pathlib import Path
            import pandas as pd

            # Load data directly to determine date range
            data_path = Path(self.data_dir) / f"{self.ticker}_NT.txt"
            if not data_path.exists():
                self.error.emit(f"Data file not found: {data_path}")
                return

            df = pd.read_csv(data_path, sep=';', header=None,
                names=['datetime', 'open', 'high', 'low', 'close', 'volume', 'oi'])

            # Get data date range
            dates = pd.to_datetime(df['datetime'].astype(str).str[:8], format='%Y%m%d')
            data_start = dates.min()
            data_end = dates.max()

            # Calculate periods
            train_days = self.train_months * 30
            test_days = self.test_weeks * 7

            periods = []
            current_train_start = data_start

            # Calculate how many periods we can fit
            while True:
                train_end = current_train_start + timedelta(days=train_days)
                test_start = train_end + timedelta(days=1)
                test_end = test_start + timedelta(days=test_days)

                if test_end > data_end:
                    break

                periods.append({
                    'train_start': current_train_start,
                    'train_end': train_end,
                    'test_start': test_start,
                    'test_end': test_end
                })

                # Roll forward by test period
                current_train_start = current_train_start + timedelta(days=test_days)

            if not periods:
                self.error.emit("Not enough data for walk-forward analysis with these settings")
                return

            total_periods = len(periods)
            self.progress.emit(0, total_periods, f"Starting walk-forward analysis ({total_periods} periods)")

            # Create optimizer once - it will be used for all training periods
            optimizer = MMapGridSearchOptimizer(self.data_dir)
            optimizer.load_data(self.ticker)

            results = []
            all_oos_equity = []
            cumulative_oos_pnl = 0

            for i, period in enumerate(periods):
                if self._cancelled:
                    self.error.emit("Cancelled by user")
                    return

                self.progress.emit(i, total_periods,
                    f"Period {i+1}/{total_periods}: Training on {period['train_start'].date()} to {period['train_end'].date()}")

                # Create parameter space
                preset_name = self.preset.split(" ")[0]
                space = create_parameter_space(preset_name)

                # Apply statistical filters - fix specific filter value by setting choices
                if self.gap_filter != "any":
                    space.parameters['gap_filter_mode'].choices = [self.gap_filter]
                    space.parameters['gap_filter_mode'].enabled = True

                if self.trend_filter != "any":
                    space.parameters['trend_filter_mode'].choices = [self.trend_filter]
                    space.parameters['trend_filter_mode'].enabled = True

                if self.range_filter != "any":
                    space.parameters['range_filter_mode'].choices = [self.range_filter]
                    space.parameters['range_filter_mode'].enabled = True

                # Run optimization on training period
                train_result = optimizer.optimize(
                    space,
                    objective=self.objective,
                    start_date=period['train_start'],
                    end_date=period['train_end']
                )

                if not train_result or not train_result.best_result:
                    continue

                best_params = train_result.best_result.params
                is_metrics = {
                    'trades': train_result.best_result.total_trades,
                    'pnl': train_result.best_result.total_pnl,
                    'pf': train_result.best_result.profit_factor,
                    'win_rate': train_result.best_result.win_rate
                }

                # Test on out-of-sample period
                self.progress.emit(i, total_periods,
                    f"Period {i+1}/{total_periods}: Testing on {period['test_start'].date()} to {period['test_end'].date()}")

                runner = BacktestRunner(self.data_dir)

                # Convert params dict to StrategyParams - filter to only valid fields
                from strategy.ib_breakout import StrategyParams
                from dataclasses import fields
                valid_fields = {f.name for f in fields(StrategyParams)}
                filtered_params = {k: v for k, v in best_params.items() if k in valid_fields}
                strategy_params = StrategyParams(**filtered_params)

                backtest_result, oos_metrics = runner.run_backtest(
                    ticker=self.ticker,
                    params=strategy_params,
                    start_date=period['test_start'],
                    end_date=period['test_end'],
                    verbose=False
                )

                trades = backtest_result.trades if backtest_result else []

                # Build equity curve from trades
                oos_equity = [0]
                for trade in trades:
                    oos_equity.append(oos_equity[-1] + trade.pnl)

                # Shift equity curve to cumulative
                shifted_equity = [e + cumulative_oos_pnl for e in oos_equity]
                cumulative_oos_pnl = shifted_equity[-1] if shifted_equity else cumulative_oos_pnl
                all_oos_equity.extend(shifted_equity[1:])  # Skip first 0

                period_result = WalkForwardPeriod(
                    period_num=i + 1,
                    train_start=period['train_start'].strftime('%Y-%m-%d'),
                    train_end=period['train_end'].strftime('%Y-%m-%d'),
                    test_start=period['test_start'].strftime('%Y-%m-%d'),
                    test_end=period['test_end'].strftime('%Y-%m-%d'),
                    best_params=best_params,
                    is_trades=is_metrics['trades'],
                    is_pnl=is_metrics['pnl'],
                    is_pf=is_metrics['pf'],
                    is_win_rate=is_metrics['win_rate'],
                    oos_trades=len(trades),
                    oos_pnl=oos_metrics.total_net_profit if oos_metrics else 0,
                    oos_pf=oos_metrics.profit_factor if oos_metrics else 0,
                    oos_win_rate=oos_metrics.percent_profitable if oos_metrics else 0,
                    oos_equity_curve=oos_equity
                )

                results.append(period_result)

                # Emit period result for live updates
                self.period_complete.emit({
                    'period': period_result,
                    'cumulative_equity': all_oos_equity.copy()
                })

            self.progress.emit(total_periods, total_periods, "Complete")

            # Calculate aggregate statistics
            total_is_pnl = sum(r.is_pnl for r in results)
            total_oos_pnl = sum(r.oos_pnl for r in results)
            total_is_trades = sum(r.is_trades for r in results)
            total_oos_trades = sum(r.oos_trades for r in results)

            # Calculate aggregate profit factor
            oos_wins = sum(r.oos_pnl for r in results if r.oos_pnl > 0)
            oos_losses = abs(sum(r.oos_pnl for r in results if r.oos_pnl < 0))
            oos_pf = oos_wins / oos_losses if oos_losses > 0 else 0

            # Calculate efficiency ratio (OOS/IS performance)
            efficiency = (total_oos_pnl / total_is_pnl * 100) if total_is_pnl > 0 else 0

            # Win rate across all OOS trades
            profitable_periods = sum(1 for r in results if r.oos_pnl > 0)
            period_win_rate = (profitable_periods / len(results) * 100) if results else 0

            self.finished.emit({
                'periods': results,
                'cumulative_equity': all_oos_equity,
                'total_is_pnl': total_is_pnl,
                'total_oos_pnl': total_oos_pnl,
                'total_is_trades': total_is_trades,
                'total_oos_trades': total_oos_trades,
                'oos_profit_factor': oos_pf,
                'efficiency_ratio': efficiency,
                'period_win_rate': period_win_rate,
                'num_periods': len(results)
            })

        except Exception as e:
            import traceback
            self.error.emit(f"{str(e)}\n{traceback.format_exc()}")


class WalkForwardTab(QWidget):
    """Tab for walk-forward analysis."""

    def __init__(self, data_dir: str, output_dir: str):
        super().__init__()
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.worker = None
        self.results = None

        # Load settings
        self.settings = QSettings("TradingTools", "IBBreakoutOptimizer")

        self._setup_ui()
        self._load_settings()
        self._update_date_range()

    def _setup_ui(self):
        """Create the UI layout."""
        layout = QVBoxLayout(self)
        layout.setSpacing(8)
        layout.setContentsMargins(8, 8, 8, 8)

        # Settings group
        settings_group = QGroupBox("Walk-Forward Settings")
        settings_grid = QGridLayout(settings_group)
        settings_grid.setSpacing(6)

        # Row 0: Ticker, Train Window, Test Window, Objective, Preset
        settings_grid.addWidget(QLabel("Ticker:"), 0, 0)
        self.ticker_combo = QComboBox()
        self.ticker_combo.addItems(["TSLA", "QQQ", "AAPL", "NVDA", "MSFT", "SPY", "AMD", "AMZN", "GOOGL", "META"])
        self.ticker_combo.setMinimumWidth(70)
        self.ticker_combo.currentTextChanged.connect(self._update_date_range)
        settings_grid.addWidget(self.ticker_combo, 0, 1)

        self.date_range_label = QLabel("")
        self.date_range_label.setStyleSheet("color: #888888; font-size: 10px;")
        settings_grid.addWidget(self.date_range_label, 0, 2)

        settings_grid.addWidget(QLabel("Train:"), 0, 3)
        self.train_combo = QComboBox()
        self.train_combo.addItems(["3 months", "6 months", "9 months", "12 months"])
        self.train_combo.setCurrentText("6 months")
        self.train_combo.currentTextChanged.connect(self._update_period_estimate)
        settings_grid.addWidget(self.train_combo, 0, 4)

        settings_grid.addWidget(QLabel("Test:"), 0, 5)
        self.test_combo = QComboBox()
        self.test_combo.addItems(["1 week", "2 weeks", "4 weeks"])
        self.test_combo.setCurrentText("2 weeks")
        self.test_combo.currentTextChanged.connect(self._update_period_estimate)
        settings_grid.addWidget(self.test_combo, 0, 6)

        settings_grid.addWidget(QLabel("Objective:"), 0, 7)
        self.objective_combo = QComboBox()
        self.objective_combo.addItems([
            "profit_factor", "sharpe_ratio", "sortino_ratio",
            "total_profit", "calmar_ratio", "win_rate", "k_ratio"
        ])
        settings_grid.addWidget(self.objective_combo, 0, 8)

        settings_grid.addWidget(QLabel("Preset:"), 0, 9)
        self.preset_combo = QComboBox()
        self.preset_combo.addItems(["quick (96)", "standard (288)", "full (1,152)"])
        self.preset_combo.setCurrentText("quick (96)")
        settings_grid.addWidget(self.preset_combo, 0, 10)

        # Run/Cancel buttons
        self.run_button = QPushButton("Run")
        self.run_button.setObjectName("primary")
        self.run_button.setMinimumHeight(28)
        self.run_button.clicked.connect(self._run_walk_forward)
        settings_grid.addWidget(self.run_button, 0, 11)

        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.setMinimumHeight(28)
        self.cancel_button.setEnabled(False)
        self.cancel_button.clicked.connect(self._cancel)
        settings_grid.addWidget(self.cancel_button, 0, 12)

        self.save_button = QPushButton("Save")
        self.save_button.setMinimumHeight(28)
        self.save_button.setEnabled(False)
        self.save_button.setToolTip("Save the most recent period's parameters to Saved Tests")
        self.save_button.clicked.connect(self._save_result)
        settings_grid.addWidget(self.save_button, 0, 13)

        # Row 1: Statistical filters + progress
        filters_label = QLabel("Filters:")
        filters_label.setStyleSheet("color: #888888;")
        settings_grid.addWidget(filters_label, 1, 0)

        gap_label = QLabel("Gap:")
        gap_label.setStyleSheet("color: #888888; font-size: 10px;")
        settings_grid.addWidget(gap_label, 1, 1)
        self.gap_filter_combo = QComboBox()
        self.gap_filter_combo.addItems([
            "any", "middle_68", "exclude_middle_68", "directional", "reverse_directional"
        ])
        self.gap_filter_combo.setMinimumWidth(110)
        settings_grid.addWidget(self.gap_filter_combo, 1, 2)

        trend_label = QLabel("Trend:")
        trend_label.setStyleSheet("color: #888888; font-size: 10px;")
        settings_grid.addWidget(trend_label, 1, 3)
        self.trend_filter_combo = QComboBox()
        self.trend_filter_combo.addItems(["any", "with_trend", "counter_trend"])
        settings_grid.addWidget(self.trend_filter_combo, 1, 4)

        range_label = QLabel("Range:")
        range_label.setStyleSheet("color: #888888; font-size: 10px;")
        settings_grid.addWidget(range_label, 1, 5)
        self.range_filter_combo = QComboBox()
        self.range_filter_combo.addItems([
            "any", "middle_68", "above_68", "below_median", "middle_68_or_below"
        ])
        self.range_filter_combo.setMinimumWidth(110)
        settings_grid.addWidget(self.range_filter_combo, 1, 6)

        self.period_estimate_label = QLabel("")
        self.period_estimate_label.setStyleSheet("color: #888888; font-size: 10px;")
        settings_grid.addWidget(self.period_estimate_label, 1, 7, 1, 2)

        self.progress_bar = QProgressBar()
        self.progress_bar.setMinimumHeight(22)
        self.progress_bar.setTextVisible(True)
        self.progress_bar.setFormat("%v / %m periods")
        settings_grid.addWidget(self.progress_bar, 1, 9, 1, 4)

        layout.addWidget(settings_group)

        # Status and metrics row
        metrics_row = QHBoxLayout()
        metrics_row.setSpacing(6)

        self.status_label = QLabel("Ready")
        self.status_label.setStyleSheet("color: #888888;")
        self.status_label.setMinimumWidth(200)
        self.status_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        metrics_row.addWidget(self.status_label)

        # In-sample metrics
        is_label = QLabel("In-Sample:")
        is_label.setStyleSheet("color: #666666; font-weight: bold;")
        metrics_row.addWidget(is_label)

        self.is_pnl_card = MetricCard("IS P&L", compact=True)
        metrics_row.addWidget(self.is_pnl_card)

        self.is_trades_card = MetricCard("IS Trades", compact=True)
        metrics_row.addWidget(self.is_trades_card)

        sep1 = QLabel("|")
        sep1.setStyleSheet("color: #444444;")
        metrics_row.addWidget(sep1)

        # Out-of-sample metrics
        oos_label = QLabel("Out-of-Sample:")
        oos_label.setStyleSheet("color: #00aa00; font-weight: bold;")
        metrics_row.addWidget(oos_label)

        self.oos_pnl_card = MetricCard("OOS P&L", compact=True)
        metrics_row.addWidget(self.oos_pnl_card)

        self.oos_pf_card = MetricCard("OOS PF", compact=True)
        metrics_row.addWidget(self.oos_pf_card)

        self.oos_trades_card = MetricCard("OOS Trades", compact=True)
        metrics_row.addWidget(self.oos_trades_card)

        sep2 = QLabel("|")
        sep2.setStyleSheet("color: #444444;")
        metrics_row.addWidget(sep2)

        self.efficiency_card = MetricCard("Efficiency", compact=True)
        self.efficiency_card.setToolTip("OOS P&L / IS P&L - measures how well optimization generalizes")
        metrics_row.addWidget(self.efficiency_card)

        self.period_wr_card = MetricCard("Period Win%", compact=True)
        self.period_wr_card.setToolTip("% of periods that were profitable out-of-sample")
        metrics_row.addWidget(self.period_wr_card)

        sep3 = QLabel("|")
        sep3.setStyleSheet("color: #444444;")
        metrics_row.addWidget(sep3)

        self.oos_length_card = MetricCard("OOS Length", compact=True)
        self.oos_length_card.setToolTip("Total out-of-sample testing period")
        metrics_row.addWidget(self.oos_length_card)

        self.monthly_pnl_card = MetricCard("$/Month", compact=True)
        self.monthly_pnl_card.setToolTip("Average monthly P&L based on OOS results")
        metrics_row.addWidget(self.monthly_pnl_card)

        metrics_row.addStretch()
        layout.addLayout(metrics_row)

        # Main content - splitter with table and charts
        main_splitter = QSplitter(Qt.Vertical)

        # Periods table (top)
        table_frame = QFrame()
        table_layout = QVBoxLayout(table_frame)
        table_layout.setContentsMargins(0, 0, 0, 0)

        table_label = QLabel("Walk-Forward Periods (click row to highlight on chart)")
        table_label.setStyleSheet("font-weight: bold; font-size: 13px;")
        table_layout.addWidget(table_label)

        self.periods_table = QTableWidget()
        self.periods_table.setColumnCount(12)
        self.periods_table.setHorizontalHeaderLabels([
            "#", "Train Start", "Train End", "Test Start", "Test End",
            "IS Trades", "IS P&L", "IS PF",
            "OOS Trades", "OOS P&L", "OOS PF", "Direction"
        ])
        self.periods_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.periods_table.setAlternatingRowColors(True)
        self.periods_table.setSelectionBehavior(QTableWidget.SelectRows)
        self.periods_table.verticalHeader().setVisible(False)
        self.periods_table.itemSelectionChanged.connect(self._on_period_selected)
        table_layout.addWidget(self.periods_table)

        main_splitter.addWidget(table_frame)

        # Charts (bottom) - horizontal splitter for equity curve and comparison
        charts_splitter = QSplitter(Qt.Horizontal)

        # OOS Equity curve (left)
        equity_frame = QFrame()
        equity_frame.setStyleSheet("""
            QFrame {
                border: 1px solid #333333;
                border-radius: 4px;
                background-color: #1e1e1e;
            }
        """)
        equity_layout = QVBoxLayout(equity_frame)
        equity_layout.setContentsMargins(2, 2, 2, 2)

        self.equity_chart = QWebEngineView()
        self.equity_chart.setMinimumHeight(200)
        equity_layout.addWidget(self.equity_chart)

        charts_splitter.addWidget(equity_frame)

        # IS vs OOS comparison bar chart (right)
        compare_frame = QFrame()
        compare_frame.setStyleSheet("""
            QFrame {
                border: 1px solid #333333;
                border-radius: 4px;
                background-color: #1e1e1e;
            }
        """)
        compare_layout = QVBoxLayout(compare_frame)
        compare_layout.setContentsMargins(2, 2, 2, 2)

        self.compare_chart = QWebEngineView()
        self.compare_chart.setMinimumHeight(200)
        compare_layout.addWidget(self.compare_chart)

        charts_splitter.addWidget(compare_frame)
        charts_splitter.setSizes([500, 400])

        main_splitter.addWidget(charts_splitter)
        main_splitter.setSizes([250, 350])

        layout.addWidget(main_splitter, 1)

        # Initial period estimate
        self._update_period_estimate()

    def _update_date_range(self):
        """Update date range label for selected ticker."""
        ticker = self.ticker_combo.currentText()
        try:
            import pandas as pd
            from pathlib import Path

            data_path = Path(self.data_dir) / f"{ticker}_NT.txt"
            if data_path.exists():
                df = pd.read_csv(data_path, sep=';', header=None,
                    names=['datetime', 'open', 'high', 'low', 'close', 'volume', 'oi'])
                df['date'] = pd.to_datetime(df['datetime'].astype(str).str[:8], format='%Y%m%d')
                start = df['date'].min()
                end = df['date'].max()
                days = (end - start).days
                self.date_range_label.setText(f"{start.date()} to {end.date()} ({days//30}mo)")
            else:
                self.date_range_label.setText("No data file found")
        except Exception as e:
            self.date_range_label.setText(f"Error: {str(e)[:30]}")

        self._update_period_estimate()

    def _update_period_estimate(self):
        """Estimate number of walk-forward periods based on settings."""
        try:
            import pandas as pd
            from pathlib import Path

            ticker = self.ticker_combo.currentText()
            data_path = Path(self.data_dir) / f"{ticker}_NT.txt"

            if not data_path.exists():
                self.period_estimate_label.setText("No data")
                return

            df = pd.read_csv(data_path, sep=';', header=None,
                names=['datetime', 'open', 'high', 'low', 'close', 'volume', 'oi'])
            df['date'] = pd.to_datetime(df['datetime'].astype(str).str[:8], format='%Y%m%d')
            total_days = (df['date'].max() - df['date'].min()).days

            train_months = int(self.train_combo.currentText().split()[0])
            test_weeks = int(self.test_combo.currentText().split()[0])

            train_days = train_months * 30
            test_days = test_weeks * 7

            # Estimate periods
            usable_days = total_days - train_days
            if usable_days > 0:
                periods = usable_days // test_days
                self.period_estimate_label.setText(f"~{periods} periods")
            else:
                self.period_estimate_label.setText("Not enough data")

        except Exception:
            self.period_estimate_label.setText("")

    def _run_walk_forward(self):
        """Start walk-forward analysis."""
        self.run_button.setEnabled(False)
        self.cancel_button.setEnabled(True)
        self.status_label.setText("Starting walk-forward analysis...")
        self.status_label.setStyleSheet("color: #2a82da;")

        # Clear previous results
        self.periods_table.setRowCount(0)
        self.equity_chart.setHtml("")
        self.compare_chart.setHtml("")

        # Parse settings
        train_months = int(self.train_combo.currentText().split()[0])
        test_weeks = int(self.test_combo.currentText().split()[0])

        self.worker = WalkForwardWorker(
            data_dir=self.data_dir,
            ticker=self.ticker_combo.currentText(),
            train_months=train_months,
            test_weeks=test_weeks,
            preset=self.preset_combo.currentText(),
            objective=self.objective_combo.currentText(),
            gap_filter=self.gap_filter_combo.currentText(),
            trend_filter=self.trend_filter_combo.currentText(),
            range_filter=self.range_filter_combo.currentText()
        )

        self.worker.progress.connect(self._on_progress)
        self.worker.period_complete.connect(self._on_period_complete)
        self.worker.finished.connect(self._on_finished)
        self.worker.error.connect(self._on_error)
        self.worker.start()

    def _cancel(self):
        """Cancel walk-forward analysis."""
        if self.worker:
            self.worker.cancel()
        self.status_label.setText("Cancelling...")

    def _on_progress(self, current: int, total: int, message: str):
        """Handle progress update."""
        self.progress_bar.setMaximum(total)
        self.progress_bar.setValue(current)
        self.status_label.setText(message)

    def _on_period_complete(self, data: dict):
        """Handle single period completion - update table and charts live."""
        period = data['period']

        # Add row to table
        row = self.periods_table.rowCount()
        self.periods_table.insertRow(row)

        # Get direction from params
        direction = period.best_params.get('trade_direction', 'both')

        values = [
            str(period.period_num),
            period.train_start,
            period.train_end,
            period.test_start,
            period.test_end,
            str(period.is_trades),
            f"${period.is_pnl:,.0f}",
            f"{period.is_pf:.2f}",
            str(period.oos_trades),
            f"${period.oos_pnl:,.0f}",
            f"{period.oos_pf:.2f}",
            direction
        ]

        for col, value in enumerate(values):
            item = QTableWidgetItem(value)
            item.setFlags(item.flags() & ~Qt.ItemIsEditable)

            # Color OOS P&L
            if col == 9:
                color = "#00ff00" if period.oos_pnl >= 0 else "#ff4444"
                item.setForeground(QColor(color))

            self.periods_table.setItem(row, col, item)

        # Update equity chart with cumulative data
        self._update_equity_chart(data['cumulative_equity'])

    def _on_finished(self, results: dict):
        """Handle walk-forward completion."""
        self.run_button.setEnabled(True)
        self.cancel_button.setEnabled(False)
        self.save_button.setEnabled(True)
        self.results = results

        num_periods = results['num_periods']
        oos_pnl = results['total_oos_pnl']

        self.status_label.setText(f"Complete: {num_periods} periods, OOS P&L: ${oos_pnl:,.0f}")
        self.status_label.setStyleSheet("color: #00ff00;" if oos_pnl >= 0 else "color: #ff4444;")

        # Update metric cards
        is_pnl = results['total_is_pnl']
        is_color = "#00ff00" if is_pnl >= 0 else "#ff4444"
        self.is_pnl_card.set_value(f"${is_pnl:,.0f}", is_color)
        self.is_trades_card.set_value(str(results['total_is_trades']))

        oos_color = "#00ff00" if oos_pnl >= 0 else "#ff4444"
        self.oos_pnl_card.set_value(f"${oos_pnl:,.0f}", oos_color)
        self.oos_pf_card.set_value(f"{results['oos_profit_factor']:.2f}")
        self.oos_trades_card.set_value(str(results['total_oos_trades']))

        efficiency = results['efficiency_ratio']
        eff_color = "#00ff00" if efficiency >= 50 else "#ffaa00" if efficiency >= 25 else "#ff4444"
        self.efficiency_card.set_value(f"{efficiency:.0f}%", eff_color)

        period_wr = results['period_win_rate']
        wr_color = "#00ff00" if period_wr >= 60 else "#ffaa00" if period_wr >= 40 else "#ff4444"
        self.period_wr_card.set_value(f"{period_wr:.0f}%", wr_color)

        # Calculate OOS length and monthly P&L from periods
        periods = results.get('periods', [])
        if periods:
            # Get first and last OOS dates
            first_oos_start = periods[0].test_start
            last_oos_end = periods[-1].test_end

            # Parse dates and calculate total OOS days
            try:
                from datetime import datetime
                start_date = datetime.strptime(first_oos_start, '%Y-%m-%d')
                end_date = datetime.strptime(last_oos_end, '%Y-%m-%d')
                total_oos_days = (end_date - start_date).days
                total_oos_months = total_oos_days / 30.0

                # Display OOS length
                if total_oos_months >= 1:
                    self.oos_length_card.set_value(f"{total_oos_months:.1f} mo")
                else:
                    self.oos_length_card.set_value(f"{total_oos_days} days")

                # Calculate monthly P&L
                if total_oos_months > 0:
                    monthly_pnl = oos_pnl / total_oos_months
                    monthly_color = "#00ff00" if monthly_pnl >= 0 else "#ff4444"
                    self.monthly_pnl_card.set_value(f"${monthly_pnl:,.0f}", monthly_color)
                else:
                    self.monthly_pnl_card.set_value("N/A")
            except Exception:
                self.oos_length_card.set_value(f"{num_periods} wks")
                self.monthly_pnl_card.set_value("N/A")
        else:
            self.oos_length_card.set_value("N/A")
            self.monthly_pnl_card.set_value("N/A")

        # Update comparison chart
        self._update_comparison_chart(results['periods'])

    def _on_error(self, error: str):
        """Handle error."""
        self.run_button.setEnabled(True)
        self.cancel_button.setEnabled(False)
        self.status_label.setText(f"Error: {error[:80]}")
        self.status_label.setStyleSheet("color: #ff4444;")
        print(f"Walk-forward error: {error}")

    def _update_equity_chart(self, equity_curve: List[float]):
        """Update the OOS equity curve chart."""
        if not equity_curve:
            return

        fig = go.Figure()

        # Add equity line
        final_pnl = equity_curve[-1] if equity_curve else 0
        line_color = '#00ff00' if final_pnl >= 0 else '#ff4444'

        fig.add_trace(go.Scatter(
            y=equity_curve,
            mode='lines',
            name='OOS Equity',
            line=dict(color=line_color, width=2),
            fill='tozeroy',
            fillcolor=f'rgba({"0, 255, 0" if final_pnl >= 0 else "255, 68, 68"}, 0.1)'
        ))

        fig.add_hline(y=0, line_dash="dash", line_color="#555555", line_width=1)

        fig.update_layout(
            title=dict(
                text=f"Out-of-Sample Equity Curve | Total: ${final_pnl:,.0f}",
                font=dict(size=12, color='#aaaaaa'),
                x=0.5
            ),
            height=250,
            margin=dict(l=60, r=20, t=40, b=30),
            paper_bgcolor='#1e1e1e',
            plot_bgcolor='#252525',
            font=dict(color='#cccccc', size=10),
            showlegend=False,
            xaxis=dict(title="Trade #", gridcolor='#333333'),
            yaxis=dict(title="P&L ($)", gridcolor='#333333', tickformat='$,.0f')
        )

        html = fig.to_html(include_plotlyjs='cdn', full_html=True)
        html = html.replace('<body>', '<body style="background-color: #1e1e1e; margin: 0; padding: 0;">')
        self.equity_chart.setHtml(html)

    def _update_comparison_chart(self, periods: List[WalkForwardPeriod]):
        """Update the IS vs OOS comparison bar chart."""
        if not periods:
            return

        period_nums = [f"P{p.period_num}" for p in periods]
        is_pnls = [p.is_pnl for p in periods]
        oos_pnls = [p.oos_pnl for p in periods]

        fig = go.Figure()

        # In-sample bars
        fig.add_trace(go.Bar(
            name='In-Sample',
            x=period_nums,
            y=is_pnls,
            marker_color='#2a82da',
            opacity=0.7
        ))

        # Out-of-sample bars
        oos_colors = ['#00ff00' if p >= 0 else '#ff4444' for p in oos_pnls]
        fig.add_trace(go.Bar(
            name='Out-of-Sample',
            x=period_nums,
            y=oos_pnls,
            marker_color=oos_colors
        ))

        fig.update_layout(
            title=dict(
                text="In-Sample vs Out-of-Sample P&L by Period",
                font=dict(size=12, color='#aaaaaa'),
                x=0.5
            ),
            height=250,
            margin=dict(l=60, r=20, t=40, b=30),
            paper_bgcolor='#1e1e1e',
            plot_bgcolor='#252525',
            font=dict(color='#cccccc', size=10),
            barmode='group',
            legend=dict(
                orientation='h',
                yanchor='bottom',
                y=1.02,
                xanchor='right',
                x=1
            ),
            xaxis=dict(title="Period", gridcolor='#333333'),
            yaxis=dict(title="P&L ($)", gridcolor='#333333', tickformat='$,.0f')
        )

        html = fig.to_html(include_plotlyjs='cdn', full_html=True)
        html = html.replace('<body>', '<body style="background-color: #1e1e1e; margin: 0; padding: 0;">')
        self.compare_chart.setHtml(html)

    def _on_period_selected(self):
        """Handle period selection - could highlight on chart."""
        # Future enhancement: highlight selected period on chart
        pass

    def set_data_dir(self, path: str):
        """Update data directory."""
        self.data_dir = path
        self._update_date_range()

    def set_output_dir(self, path: str):
        """Update output directory."""
        self.output_dir = path

    def _save_result(self):
        """Save the walk-forward results to disk."""
        if not self.results:
            self.status_label.setText("No results to save")
            return

        try:
            # Create output directory
            save_dir = Path(self.output_dir) / "walk_forward_results"
            save_dir.mkdir(parents=True, exist_ok=True)

            # Get the last period's parameters (most recent optimization)
            periods = self.results.get('periods', [])
            if not periods:
                self.status_label.setText("No periods to save")
                return

            # Get ticker and create filename with timestamp
            ticker = self.ticker_combo.currentText()
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"{ticker}_wf_{timestamp}.json"
            filepath = save_dir / filename

            # Build comprehensive save data
            save_data = {
                'ticker': ticker,
                'timestamp': timestamp,
                'settings': {
                    'train_months': int(self.train_combo.currentText().split()[0]),
                    'test_weeks': int(self.test_combo.currentText().split()[0]),
                    'preset': self.preset_combo.currentText(),
                    'objective': self.objective_combo.currentText(),
                    'gap_filter': self.gap_filter_combo.currentText(),
                    'trend_filter': self.trend_filter_combo.currentText(),
                    'range_filter': self.range_filter_combo.currentText()
                },
                'summary': {
                    'num_periods': self.results.get('num_periods', 0),
                    'total_is_pnl': self.results.get('total_is_pnl', 0),
                    'total_oos_pnl': self.results.get('total_oos_pnl', 0),
                    'oos_profit_factor': self.results.get('oos_profit_factor', 0),
                    'total_oos_trades': self.results.get('total_oos_trades', 0),
                    'efficiency_ratio': self.results.get('efficiency_ratio', 0),
                    'period_win_rate': self.results.get('period_win_rate', 0)
                },
                'periods': [
                    {
                        'period_num': p.period_num,
                        'train_start': p.train_start,
                        'train_end': p.train_end,
                        'test_start': p.test_start,
                        'test_end': p.test_end,
                        'best_params': p.best_params,
                        'is_trades': p.is_trades,
                        'is_pnl': p.is_pnl,
                        'is_pf': p.is_pf,
                        'is_win_rate': p.is_win_rate,
                        'oos_trades': p.oos_trades,
                        'oos_pnl': p.oos_pnl,
                        'oos_pf': p.oos_pf,
                        'oos_win_rate': p.oos_win_rate
                    }
                    for p in periods
                ],
                'final_equity_curve': self.results.get('oos_equity_curve', [])
            }

            # Save to file
            with open(filepath, 'w') as f:
                json.dump(save_data, f, indent=2)

            self.status_label.setText(f"Saved to {filename}")
            self.status_label.setStyleSheet("color: #00ff00;")

        except Exception as e:
            self.status_label.setText(f"Save error: {str(e)[:50]}")
            self.status_label.setStyleSheet("color: #ff4444;")

    def _load_settings(self):
        """Load saved settings from QSettings."""
        # Ticker
        ticker = self.settings.value("wf_ticker", "TSLA")
        idx = self.ticker_combo.findText(ticker)
        if idx >= 0:
            self.ticker_combo.setCurrentIndex(idx)

        # Train window
        train = self.settings.value("wf_train", "12 months")
        idx = self.train_combo.findText(train)
        if idx >= 0:
            self.train_combo.setCurrentIndex(idx)

        # Test window
        test = self.settings.value("wf_test", "1 week")
        idx = self.test_combo.findText(test)
        if idx >= 0:
            self.test_combo.setCurrentIndex(idx)

        # Objective
        objective = self.settings.value("wf_objective", "profit_factor")
        idx = self.objective_combo.findText(objective)
        if idx >= 0:
            self.objective_combo.setCurrentIndex(idx)

        # Preset
        preset = self.settings.value("wf_preset", "quick (96)")
        idx = self.preset_combo.findText(preset)
        if idx >= 0:
            self.preset_combo.setCurrentIndex(idx)

        # Filters
        gap = self.settings.value("wf_gap_filter", "any")
        idx = self.gap_filter_combo.findText(gap)
        if idx >= 0:
            self.gap_filter_combo.setCurrentIndex(idx)

        trend = self.settings.value("wf_trend_filter", "any")
        idx = self.trend_filter_combo.findText(trend)
        if idx >= 0:
            self.trend_filter_combo.setCurrentIndex(idx)

        range_f = self.settings.value("wf_range_filter", "any")
        idx = self.range_filter_combo.findText(range_f)
        if idx >= 0:
            self.range_filter_combo.setCurrentIndex(idx)

        # Connect signals to save on change (after loading)
        self.ticker_combo.currentTextChanged.connect(self._save_settings)
        self.train_combo.currentTextChanged.connect(self._save_settings)
        self.test_combo.currentTextChanged.connect(self._save_settings)
        self.objective_combo.currentTextChanged.connect(self._save_settings)
        self.preset_combo.currentTextChanged.connect(self._save_settings)
        self.gap_filter_combo.currentTextChanged.connect(self._save_settings)
        self.trend_filter_combo.currentTextChanged.connect(self._save_settings)
        self.range_filter_combo.currentTextChanged.connect(self._save_settings)

    def _save_settings(self):
        """Save current settings to QSettings."""
        self.settings.setValue("wf_ticker", self.ticker_combo.currentText())
        self.settings.setValue("wf_train", self.train_combo.currentText())
        self.settings.setValue("wf_test", self.test_combo.currentText())
        self.settings.setValue("wf_objective", self.objective_combo.currentText())
        self.settings.setValue("wf_preset", self.preset_combo.currentText())
        self.settings.setValue("wf_gap_filter", self.gap_filter_combo.currentText())
        self.settings.setValue("wf_trend_filter", self.trend_filter_combo.currentText())
        self.settings.setValue("wf_range_filter", self.range_filter_combo.currentText())
