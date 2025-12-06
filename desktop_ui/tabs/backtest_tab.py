"""
Backtest Tab - Run backtests with custom parameters.
"""

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QLabel, QComboBox, QDoubleSpinBox, QSpinBox, QCheckBox,
    QPushButton, QGroupBox, QTableWidget, QTableWidgetItem,
    QHeaderView, QProgressBar, QSplitter, QFrame
)
from PySide6.QtCore import Qt, Signal, QThread
from PySide6.QtGui import QColor

from desktop_ui.workers.backtest_worker import BacktestWorker
from desktop_ui.widgets.metrics_panel import MetricsPanel


class BacktestTab(QWidget):
    """Tab for running backtests with custom parameters."""

    backtest_complete = Signal(object, object)  # result, metrics

    def __init__(self, data_dir: str, output_dir: str):
        super().__init__()

        self.data_dir = data_dir
        self.output_dir = output_dir
        self.worker = None
        self.last_result = None
        self.last_metrics = None

        self._setup_ui()

    def _setup_ui(self):
        """Create the UI layout."""
        layout = QVBoxLayout(self)
        layout.setSpacing(12)

        # Parameters section
        params_layout = QHBoxLayout()

        # Left column - Basic parameters
        basic_group = QGroupBox("Basic Parameters")
        basic_layout = QGridLayout(basic_group)
        basic_layout.setSpacing(8)

        # Ticker
        basic_layout.addWidget(QLabel("Ticker:"), 0, 0)
        self.ticker_combo = QComboBox()
        self.ticker_combo.addItems(["TSLA", "QQQ", "AAPL", "NVDA", "MSFT", "SPY", "AMD", "AMZN", "GOOGL", "META"])
        basic_layout.addWidget(self.ticker_combo, 0, 1)

        # Trade direction
        basic_layout.addWidget(QLabel("Direction:"), 1, 0)
        self.direction_combo = QComboBox()
        self.direction_combo.addItems(["long_only", "short_only", "both"])
        basic_layout.addWidget(self.direction_combo, 1, 1)

        # Profit target
        basic_layout.addWidget(QLabel("Profit Target %:"), 2, 0)
        self.profit_target_spin = QDoubleSpinBox()
        self.profit_target_spin.setRange(0.3, 5.0)
        self.profit_target_spin.setValue(1.0)
        self.profit_target_spin.setSingleStep(0.1)
        basic_layout.addWidget(self.profit_target_spin, 2, 1)

        # Stop loss type
        basic_layout.addWidget(QLabel("Stop Loss:"), 3, 0)
        self.stop_loss_combo = QComboBox()
        self.stop_loss_combo.addItems(["opposite_ib", "match_target"])
        basic_layout.addWidget(self.stop_loss_combo, 3, 1)

        # IB Duration
        basic_layout.addWidget(QLabel("IB Duration:"), 4, 0)
        self.ib_duration_combo = QComboBox()
        self.ib_duration_combo.addItems(["15", "30", "45", "60"])
        self.ib_duration_combo.setCurrentText("30")
        basic_layout.addWidget(self.ib_duration_combo, 4, 1)

        params_layout.addWidget(basic_group)

        # Right column - Filters
        filter_group = QGroupBox("Filters & Options")
        filter_layout = QGridLayout(filter_group)
        filter_layout.setSpacing(8)

        # QQQ Filter
        self.qqq_filter_check = QCheckBox("Use QQQ Filter")
        self.qqq_filter_check.setChecked(True)
        filter_layout.addWidget(self.qqq_filter_check, 0, 0, 1, 2)

        # Min IB Range
        filter_layout.addWidget(QLabel("Min IB Range %:"), 1, 0)
        self.min_ib_spin = QDoubleSpinBox()
        self.min_ib_spin.setRange(0.0, 5.0)
        self.min_ib_spin.setValue(0.0)
        self.min_ib_spin.setSingleStep(0.1)
        filter_layout.addWidget(self.min_ib_spin, 1, 1)

        # Max IB Range
        filter_layout.addWidget(QLabel("Max IB Range %:"), 2, 0)
        self.max_ib_spin = QDoubleSpinBox()
        self.max_ib_spin.setRange(1.0, 20.0)
        self.max_ib_spin.setValue(10.0)
        self.max_ib_spin.setSingleStep(0.5)
        filter_layout.addWidget(self.max_ib_spin, 2, 1)

        # Max breakout time
        filter_layout.addWidget(QLabel("Max Breakout Time:"), 3, 0)
        self.max_breakout_combo = QComboBox()
        self.max_breakout_combo.addItems(["12:00", "13:00", "14:00", "15:00"])
        self.max_breakout_combo.setCurrentText("14:00")
        filter_layout.addWidget(self.max_breakout_combo, 3, 1)

        # EOD exit time
        filter_layout.addWidget(QLabel("EOD Exit Time:"), 4, 0)
        self.eod_exit_combo = QComboBox()
        self.eod_exit_combo.addItems(["15:30", "15:45", "15:55"])
        self.eod_exit_combo.setCurrentText("15:55")
        filter_layout.addWidget(self.eod_exit_combo, 4, 1)

        params_layout.addWidget(filter_group)

        # Advanced exits column
        advanced_group = QGroupBox("Advanced Exits")
        advanced_layout = QGridLayout(advanced_group)
        advanced_layout.setSpacing(8)

        # Trailing stop
        self.trailing_check = QCheckBox("Trailing Stop")
        advanced_layout.addWidget(self.trailing_check, 0, 0)
        self.trailing_atr_spin = QDoubleSpinBox()
        self.trailing_atr_spin.setRange(1.0, 5.0)
        self.trailing_atr_spin.setValue(2.0)
        self.trailing_atr_spin.setSingleStep(0.5)
        self.trailing_atr_spin.setEnabled(False)
        advanced_layout.addWidget(self.trailing_atr_spin, 0, 1)
        self.trailing_check.toggled.connect(self.trailing_atr_spin.setEnabled)

        # Break-even stop
        self.breakeven_check = QCheckBox("Break-Even Stop")
        advanced_layout.addWidget(self.breakeven_check, 1, 0)
        self.breakeven_pct_spin = QDoubleSpinBox()
        self.breakeven_pct_spin.setRange(0.3, 0.9)
        self.breakeven_pct_spin.setValue(0.7)
        self.breakeven_pct_spin.setSingleStep(0.1)
        self.breakeven_pct_spin.setEnabled(False)
        advanced_layout.addWidget(self.breakeven_pct_spin, 1, 1)
        self.breakeven_check.toggled.connect(self.breakeven_pct_spin.setEnabled)

        # Max bars
        self.maxbars_check = QCheckBox("Max Bars Exit")
        advanced_layout.addWidget(self.maxbars_check, 2, 0)
        self.maxbars_spin = QSpinBox()
        self.maxbars_spin.setRange(10, 200)
        self.maxbars_spin.setValue(60)
        self.maxbars_spin.setEnabled(False)
        advanced_layout.addWidget(self.maxbars_spin, 2, 1)
        self.maxbars_check.toggled.connect(self.maxbars_spin.setEnabled)

        params_layout.addWidget(advanced_group)

        layout.addLayout(params_layout)

        # Run button and progress
        run_layout = QHBoxLayout()

        self.run_button = QPushButton("Run Backtest")
        self.run_button.setObjectName("primary")
        self.run_button.setMinimumHeight(40)
        self.run_button.clicked.connect(self._run_backtest)
        run_layout.addWidget(self.run_button)

        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        run_layout.addWidget(self.progress_bar)

        self.status_label = QLabel("")
        run_layout.addWidget(self.status_label)
        run_layout.addStretch()

        layout.addLayout(run_layout)

        # Results section
        splitter = QSplitter(Qt.Vertical)

        # Metrics panel
        self.metrics_panel = MetricsPanel()
        splitter.addWidget(self.metrics_panel)

        # Trade table
        table_frame = QFrame()
        table_layout = QVBoxLayout(table_frame)
        table_layout.setContentsMargins(0, 0, 0, 0)

        table_label = QLabel("Trade List")
        table_label.setStyleSheet("font-weight: bold; font-size: 14px;")
        table_layout.addWidget(table_label)

        self.trade_table = QTableWidget()
        self.trade_table.setColumnCount(9)
        self.trade_table.setHorizontalHeaderLabels([
            "Entry Time", "Exit Time", "Direction", "Entry", "Exit",
            "P&L", "P&L %", "Exit Reason", "Bars"
        ])
        self.trade_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.trade_table.setAlternatingRowColors(True)
        self.trade_table.setSelectionBehavior(QTableWidget.SelectRows)
        table_layout.addWidget(self.trade_table)

        splitter.addWidget(table_frame)
        splitter.setSizes([200, 400])

        layout.addWidget(splitter)

    def set_data_dir(self, path: str):
        """Update data directory."""
        self.data_dir = path

    def set_output_dir(self, path: str):
        """Update output directory."""
        self.output_dir = path

    def _run_backtest(self):
        """Execute the backtest."""
        self.run_button.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 0)  # Indeterminate
        self.status_label.setText("Running backtest...")

        # Collect parameters
        params = {
            'ticker': self.ticker_combo.currentText(),
            'trade_direction': self.direction_combo.currentText(),
            'profit_target_percent': self.profit_target_spin.value(),
            'stop_loss_type': self.stop_loss_combo.currentText(),
            'ib_duration_minutes': int(self.ib_duration_combo.currentText()),
            'use_qqq_filter': self.qqq_filter_check.isChecked(),
            'min_ib_range_percent': self.min_ib_spin.value(),
            'max_ib_range_percent': self.max_ib_spin.value(),
            'max_breakout_time': self.max_breakout_combo.currentText(),
            'eod_exit_time': self.eod_exit_combo.currentText(),
            'trailing_stop_enabled': self.trailing_check.isChecked(),
            'trailing_stop_atr_mult': self.trailing_atr_spin.value(),
            'break_even_enabled': self.breakeven_check.isChecked(),
            'break_even_pct': self.breakeven_pct_spin.value(),
            'max_bars_enabled': self.maxbars_check.isChecked(),
            'max_bars': self.maxbars_spin.value(),
        }

        # Start worker thread
        self.worker = BacktestWorker(self.data_dir, params)
        self.worker.finished.connect(self._on_backtest_complete)
        self.worker.error.connect(self._on_backtest_error)
        self.worker.start()

    def _on_backtest_complete(self, result, metrics):
        """Handle backtest completion."""
        self.run_button.setEnabled(True)
        self.progress_bar.setVisible(False)
        self.status_label.setText(
            f"Complete: {metrics.total_trades} trades, "
            f"P&L: ${metrics.total_net_profit:,.2f}"
        )

        self.last_result = result
        self.last_metrics = metrics

        # Update metrics panel
        self.metrics_panel.update_metrics(metrics)

        # Update trade table
        self._populate_trade_table(result.trades if result else [])

        # Emit signal for other tabs
        self.backtest_complete.emit(result, metrics)

    def _on_backtest_error(self, error_msg: str):
        """Handle backtest error."""
        self.run_button.setEnabled(True)
        self.progress_bar.setVisible(False)
        self.status_label.setText(f"Error: {error_msg}")
        self.status_label.setStyleSheet("color: #ff4444;")

    def _populate_trade_table(self, trades: list):
        """Fill the trade table with results."""
        self.trade_table.setRowCount(len(trades))

        for row, trade in enumerate(trades):
            # Entry time
            entry_item = QTableWidgetItem(
                trade.entry_time.strftime('%Y-%m-%d %H:%M')
            )
            self.trade_table.setItem(row, 0, entry_item)

            # Exit time
            exit_time = trade.exit_time.strftime('%Y-%m-%d %H:%M') if trade.exit_time else ""
            self.trade_table.setItem(row, 1, QTableWidgetItem(exit_time))

            # Direction
            direction = "LONG" if trade.direction.value == "long" else "SHORT"
            dir_item = QTableWidgetItem(direction)
            dir_item.setForeground(
                QColor("#00ff00") if direction == "LONG" else QColor("#ff4444")
            )
            self.trade_table.setItem(row, 2, dir_item)

            # Entry price
            self.trade_table.setItem(row, 3, QTableWidgetItem(f"${trade.entry_price:.2f}"))

            # Exit price
            exit_price = f"${trade.exit_price:.2f}" if trade.exit_price else ""
            self.trade_table.setItem(row, 4, QTableWidgetItem(exit_price))

            # P&L
            pnl_item = QTableWidgetItem(f"${trade.pnl:.2f}")
            pnl_item.setForeground(
                QColor("#00ff00") if trade.pnl >= 0 else QColor("#ff4444")
            )
            self.trade_table.setItem(row, 5, pnl_item)

            # P&L %
            pnl_pct_item = QTableWidgetItem(f"{trade.pnl_pct:.2f}%")
            pnl_pct_item.setForeground(
                QColor("#00ff00") if trade.pnl_pct >= 0 else QColor("#ff4444")
            )
            self.trade_table.setItem(row, 6, pnl_pct_item)

            # Exit reason
            exit_reason = trade.exit_reason.value if trade.exit_reason else ""
            self.trade_table.setItem(row, 7, QTableWidgetItem(exit_reason))

            # Bars held
            self.trade_table.setItem(row, 8, QTableWidgetItem(str(trade.bars_held)))

    def load_params_from_json(self, params: dict):
        """Load parameters from optimization results."""
        if 'trade_direction' in params:
            idx = self.direction_combo.findText(params['trade_direction'])
            if idx >= 0:
                self.direction_combo.setCurrentIndex(idx)

        if 'profit_target_percent' in params:
            self.profit_target_spin.setValue(params['profit_target_percent'])

        if 'stop_loss_type' in params:
            idx = self.stop_loss_combo.findText(params['stop_loss_type'])
            if idx >= 0:
                self.stop_loss_combo.setCurrentIndex(idx)

        if 'ib_duration_minutes' in params:
            idx = self.ib_duration_combo.findText(str(params['ib_duration_minutes']))
            if idx >= 0:
                self.ib_duration_combo.setCurrentIndex(idx)

        if 'use_qqq_filter' in params:
            self.qqq_filter_check.setChecked(params['use_qqq_filter'])

        if 'min_ib_range_percent' in params:
            self.min_ib_spin.setValue(params['min_ib_range_percent'])

        if 'max_ib_range_percent' in params:
            self.max_ib_spin.setValue(params['max_ib_range_percent'])

        if 'trailing_stop_enabled' in params:
            self.trailing_check.setChecked(params['trailing_stop_enabled'])
            if params['trailing_stop_enabled'] and 'trailing_stop_atr_mult' in params:
                self.trailing_atr_spin.setValue(params['trailing_stop_atr_mult'])

        if 'break_even_enabled' in params:
            self.breakeven_check.setChecked(params['break_even_enabled'])
            if params['break_even_enabled'] and 'break_even_pct' in params:
                self.breakeven_pct_spin.setValue(params['break_even_pct'])

        if 'max_bars_enabled' in params:
            self.maxbars_check.setChecked(params['max_bars_enabled'])
            if params['max_bars_enabled'] and 'max_bars' in params:
                self.maxbars_spin.setValue(params['max_bars'])
