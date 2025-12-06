"""
Optimization Tab - Run parameter optimization with live progress updates.
"""

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QLabel, QComboBox, QCheckBox, QPushButton, QGroupBox,
    QTableWidget, QTableWidgetItem, QHeaderView, QProgressBar,
    QSplitter, QFrame, QTextEdit
)
from PySide6.QtCore import Qt, Signal, QSettings
from PySide6.QtWebEngineWidgets import QWebEngineView

from desktop_ui.workers.optimization_worker import OptimizationWorker
from desktop_ui.widgets.metrics_panel import MetricCard


class OptimizationTab(QWidget):
    """Tab for running parameter optimization."""

    optimization_complete = Signal(dict)  # results dict

    def __init__(self, data_dir: str, output_dir: str):
        super().__init__()

        self.data_dir = data_dir
        self.output_dir = output_dir
        self.worker = None
        self.last_results = None
        self.previous_results = None  # For delta comparison
        self.best_params = None  # Store best params for double-click handler
        self.top_results_data = []  # Store results for double-click handler
        self.settings = QSettings("TradingTools", "IBBreakoutOptimizer")

        self._setup_ui()
        self._load_settings()
        self._load_persisted_results()
        self._update_date_range()  # Show date range for initial ticker

    def _setup_ui(self):
        """Create the UI layout."""
        layout = QVBoxLayout(self)
        layout.setSpacing(12)

        # Settings section
        settings_layout = QHBoxLayout()

        # Data settings
        data_group = QGroupBox("Data Settings")
        data_layout = QGridLayout(data_group)
        data_layout.setSpacing(8)

        data_layout.addWidget(QLabel("Ticker:"), 0, 0)
        self.ticker_combo = QComboBox()
        self.ticker_combo.addItems(["TSLA", "QQQ", "AAPL", "NVDA", "MSFT", "SPY", "AMD", "AMZN", "GOOGL", "META"])
        self.ticker_combo.currentTextChanged.connect(self._update_date_range)
        data_layout.addWidget(self.ticker_combo, 0, 1)

        # Date range label
        self.date_range_label = QLabel("")
        self.date_range_label.setStyleSheet("color: #888888; font-size: 11px;")
        data_layout.addWidget(self.date_range_label, 0, 2)

        self.qqq_filter_check = QCheckBox("Include QQQ Filter")
        self.qqq_filter_check.setChecked(True)
        data_layout.addWidget(self.qqq_filter_check, 1, 0, 1, 3)

        data_layout.addWidget(QLabel("Objective:"), 2, 0)
        self.objective_combo = QComboBox()
        self.objective_combo.addItems([
            "profit_factor", "sharpe_ratio", "sortino_ratio",
            "total_profit", "calmar_ratio", "win_rate", "k_ratio"
        ])
        self.objective_combo.setToolTip(
            "profit_factor: Gross profit / gross loss\n"
            "sharpe_ratio: Risk-adjusted returns\n"
            "sortino_ratio: Downside risk-adjusted returns\n"
            "total_profit: Total P&L\n"
            "calmar_ratio: Return / max drawdown\n"
            "win_rate: Percentage of winning trades\n"
            "k_ratio: Smooth equity curve (consistency)"
        )
        data_layout.addWidget(self.objective_combo, 2, 1)

        # Add objective description label
        self.objective_desc = QLabel("")
        self.objective_desc.setStyleSheet("color: #888888; font-size: 11px;")
        data_layout.addWidget(self.objective_desc, 2, 2)
        self.objective_combo.currentTextChanged.connect(self._update_objective_desc)
        self._update_objective_desc(self.objective_combo.currentText())

        # Connect change signals for saving
        self.ticker_combo.currentTextChanged.connect(self._save_settings)
        self.qqq_filter_check.stateChanged.connect(self._save_settings)
        self.objective_combo.currentTextChanged.connect(self._save_settings)

        settings_layout.addWidget(data_group)

        # Optimization settings
        opt_group = QGroupBox("Optimization Settings")
        opt_layout = QGridLayout(opt_group)
        opt_layout.setSpacing(8)

        opt_layout.addWidget(QLabel("Preset:"), 0, 0)
        self.preset_combo = QComboBox()
        self.preset_combo.addItems(["quick (96)", "standard (288)", "full (1,152)", "thorough (2,592)"])
        self.preset_combo.setCurrentText("standard (288)")  # Default to standard
        opt_layout.addWidget(self.preset_combo, 0, 1)

        # Preset info
        self.preset_info = QLabel("All parameters with coarse grid, ~200 combinations (5x faster than full)")
        self.preset_info.setStyleSheet("color: #888888; font-style: italic;")
        opt_layout.addWidget(self.preset_info, 1, 0, 1, 2)
        self.preset_combo.currentTextChanged.connect(self._update_preset_info)

        # Connect change signals for saving
        self.preset_combo.currentTextChanged.connect(self._save_settings)

        settings_layout.addWidget(opt_group)

        layout.addLayout(settings_layout)

        # Run button and progress section - compact layout
        run_frame = QFrame()
        run_frame.setStyleSheet("""
            QFrame {
                background-color: #252525;
                border: 1px solid #333333;
                border-radius: 6px;
                padding: 8px;
            }
        """)
        run_layout = QVBoxLayout(run_frame)
        run_layout.setSpacing(6)
        run_layout.setContentsMargins(8, 8, 8, 8)

        # Top row: Button + Progress bar + Status (all horizontal)
        top_row = QHBoxLayout()
        top_row.setSpacing(8)

        self.run_button = QPushButton("Run Optimization")
        self.run_button.setObjectName("primary")
        self.run_button.setMinimumHeight(32)
        self.run_button.setMinimumWidth(140)
        self.run_button.clicked.connect(self._run_optimization)
        top_row.addWidget(self.run_button)

        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.setMinimumHeight(32)
        self.cancel_button.setMinimumWidth(70)
        self.cancel_button.setEnabled(False)
        self.cancel_button.clicked.connect(self._cancel_optimization)
        top_row.addWidget(self.cancel_button)

        # Progress bar inline
        self.progress_bar = QProgressBar()
        self.progress_bar.setMinimumHeight(24)
        self.progress_bar.setMinimumWidth(200)
        self.progress_bar.setTextVisible(True)
        self.progress_bar.setFormat("%v / %m (%p%)")
        top_row.addWidget(self.progress_bar, 1)  # stretch

        # Status text inline
        self.status_label = QLabel("Ready")
        self.status_label.setStyleSheet("color: #888888;")
        self.status_label.setMinimumWidth(200)
        top_row.addWidget(self.status_label)

        run_layout.addLayout(top_row)

        # Metrics row - compact inline cards
        metrics_row = QHBoxLayout()
        metrics_row.setSpacing(4)

        self.completed_card = MetricCard("Done", compact=True)
        metrics_row.addWidget(self.completed_card)

        self.best_objective_card = MetricCard("Best Obj", compact=True)
        metrics_row.addWidget(self.best_objective_card)

        self.best_trades_card = MetricCard("Trades", compact=True)
        metrics_row.addWidget(self.best_trades_card)

        self.best_pnl_card = MetricCard("P&L", compact=True)
        metrics_row.addWidget(self.best_pnl_card)

        self.speed_card = MetricCard("Speed", compact=True)
        metrics_row.addWidget(self.speed_card)

        # Separator
        sep = QLabel("|")
        sep.setStyleSheet("color: #555555;")
        metrics_row.addWidget(sep)

        # Delta cards (comparison to previous run)
        self.delta_pnl_card = MetricCard("ΔP&L", compact=True)
        metrics_row.addWidget(self.delta_pnl_card)

        self.delta_wr_card = MetricCard("ΔWin%", compact=True)
        metrics_row.addWidget(self.delta_wr_card)

        self.delta_pf_card = MetricCard("ΔPF", compact=True)
        metrics_row.addWidget(self.delta_pf_card)

        metrics_row.addStretch()
        run_layout.addLayout(metrics_row)

        layout.addWidget(run_frame)

        # Results section - horizontal split for better space usage
        main_splitter = QSplitter(Qt.Horizontal)

        # Left side: Best parameters (narrow)
        best_frame = QFrame()
        best_layout = QVBoxLayout(best_frame)
        best_layout.setContentsMargins(0, 0, 0, 0)
        best_layout.setSpacing(4)

        best_label = QLabel("Best Parameters")
        best_label.setStyleSheet("font-weight: bold; font-size: 13px;")
        best_layout.addWidget(best_label)

        self.best_params_table = QTableWidget()
        self.best_params_table.setColumnCount(2)
        self.best_params_table.setHorizontalHeaderLabels(["Parameter", "Value"])
        self.best_params_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self.best_params_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeToContents)
        self.best_params_table.setAlternatingRowColors(True)
        self.best_params_table.verticalHeader().setDefaultSectionSize(22)  # Compact rows
        self.best_params_table.verticalHeader().setVisible(False)
        best_layout.addWidget(self.best_params_table)

        main_splitter.addWidget(best_frame)

        # Right side: Vertical splitter with results table and equity curve
        right_splitter = QSplitter(Qt.Vertical)

        # Top results table
        top_frame = QFrame()
        top_layout = QVBoxLayout(top_frame)
        top_layout.setContentsMargins(0, 0, 0, 0)
        top_layout.setSpacing(4)

        top_label = QLabel("Top 10 Results (click row to view equity curve)")
        top_label.setStyleSheet("font-weight: bold; font-size: 13px;")
        top_layout.addWidget(top_label)

        self.results_table = QTableWidget()
        self.results_table.setColumnCount(9)  # Added PF column
        self.results_table.setHorizontalHeaderLabels([
            "Direction", "Target", "Stop", "QQQ",
            "Objective", "PF", "Trades", "Win%", "P&L"
        ])
        self.results_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.results_table.setAlternatingRowColors(True)
        self.results_table.setSelectionBehavior(QTableWidget.SelectRows)
        self.results_table.verticalHeader().setDefaultSectionSize(24)  # Compact rows
        self.results_table.verticalHeader().setVisible(False)
        self.results_table.itemSelectionChanged.connect(self._on_result_selected)
        top_layout.addWidget(self.results_table)

        right_splitter.addWidget(top_frame)

        # Equity curve chart (bottom) - no title label for more chart space
        chart_frame = QFrame()
        chart_frame.setStyleSheet("""
            QFrame {
                border: 1px solid #333333;
                border-radius: 4px;
                background-color: #1e1e1e;
            }
        """)
        chart_layout = QVBoxLayout(chart_frame)
        chart_layout.setContentsMargins(2, 2, 2, 2)
        chart_layout.setSpacing(0)

        self.equity_chart = QWebEngineView()
        self.equity_chart.setMinimumHeight(120)
        chart_layout.addWidget(self.equity_chart)

        right_splitter.addWidget(chart_frame)
        right_splitter.setSizes([210, 190])  # More space for equity curve

        main_splitter.addWidget(right_splitter)
        main_splitter.setSizes([220, 680])  # Narrow left, wide right

        layout.addWidget(main_splitter, 1)  # Give splitter stretch priority

    def _update_preset_info(self, preset: str):
        """Update preset description."""
        # Extract preset name (remove combo count suffix)
        preset_name = preset.split(" ")[0]
        info = {
            "quick": "~3 sec - IB duration, direction, profit target, stop type",
            "standard": "~5 sec - Core params + IB range filter (recommended)",
            "full": "~15 sec - All params including trailing stop & break-even",
            "thorough": "~30 sec - Finer profit target grid (0.2% steps)"
        }
        self.preset_info.setText(info.get(preset_name, ""))

    def _update_objective_desc(self, objective: str):
        """Update objective description label."""
        descriptions = {
            "profit_factor": "gross profit / gross loss",
            "sharpe_ratio": "risk-adjusted returns",
            "sortino_ratio": "downside risk-adjusted",
            "total_profit": "total P&L",
            "calmar_ratio": "return / max drawdown",
            "win_rate": "% winning trades",
            "k_ratio": "smooth equity curve"
        }
        self.objective_desc.setText(descriptions.get(objective, ""))

    def set_data_dir(self, path: str):
        """Update data directory."""
        self.data_dir = path
        self._update_date_range()  # Refresh date range when data dir changes

    def set_output_dir(self, path: str):
        """Update output directory."""
        self.output_dir = path

    def _update_date_range(self):
        """Update the date range label based on available data for selected ticker."""
        from pathlib import Path

        ticker = self.ticker_combo.currentText()
        data_path = Path(self.data_dir)

        # Try to find data file
        possible_files = [
            data_path / f"{ticker}_NT.txt",
            data_path / f"{ticker}.txt",
            data_path / f"{ticker}.csv",
        ]

        data_file = None
        for f in possible_files:
            if f.exists():
                data_file = f
                break

        if not data_file:
            self.date_range_label.setText("(no data)")
            return

        try:
            # Read first and last lines to get date range
            with open(data_file, 'r') as f:
                first_line = f.readline().strip()
                # Skip header if CSV
                if first_line.startswith('timestamp') or first_line.startswith('date'):
                    first_line = f.readline().strip()

                # Seek to end and read last line
                f.seek(0, 2)  # Go to end
                file_size = f.tell()
                # Read last ~200 bytes to find last line
                f.seek(max(0, file_size - 200))
                lines = f.readlines()
                last_line = lines[-1].strip() if lines else ""

            # Parse dates based on format
            if ';' in first_line:
                # NT format: 20240102 093000;...
                first_date = first_line.split(';')[0].split()[0]
                last_date = last_line.split(';')[0].split()[0]
                # Format: YYYYMMDD -> MM/DD/YY
                first_formatted = f"{first_date[4:6]}/{first_date[6:8]}/{first_date[2:4]}"
                last_formatted = f"{last_date[4:6]}/{last_date[6:8]}/{last_date[2:4]}"
            else:
                # CSV format: 2024-01-02 09:30:00,...
                first_date = first_line.split(',')[0].split()[0]
                last_date = last_line.split(',')[0].split()[0]
                # Format: YYYY-MM-DD -> MM/DD/YY
                parts = first_date.split('-')
                first_formatted = f"{parts[1]}/{parts[2]}/{parts[0][2:]}"
                parts = last_date.split('-')
                last_formatted = f"{parts[1]}/{parts[2]}/{parts[0][2:]}"

            self.date_range_label.setText(f"({first_formatted} - {last_formatted})")

        except Exception as e:
            self.date_range_label.setText("(error reading dates)")

    def _run_optimization(self):
        """Start the optimization."""
        import time
        self._start_time = time.time()

        self.run_button.setEnabled(False)
        self.cancel_button.setEnabled(True)
        self.progress_bar.setValue(0)
        self.status_label.setText("Starting optimization...")
        self.status_label.setStyleSheet("color: #2a82da;")

        # Clear previous results
        self.best_params_table.setRowCount(0)
        self.results_table.setRowCount(0)

        # Reset metric cards
        self.completed_card.set_value("0")
        self.best_objective_card.set_value("--")
        self.best_trades_card.set_value("--")
        self.best_pnl_card.set_value("--")
        self.speed_card.set_value("--")

        # Collect settings
        # Extract preset name (remove combo count suffix like "standard (288)" -> "standard")
        preset_text = self.preset_combo.currentText()
        preset_name = preset_text.split(" ")[0]

        settings = {
            'ticker': self.ticker_combo.currentText(),
            'use_qqq_filter': self.qqq_filter_check.isChecked(),
            'objective': self.objective_combo.currentText(),
            'preset': preset_name,
            'mode': 'two_phase',  # Always use two-phase (Grid + Bayesian)
        }

        # Start worker
        self.worker = OptimizationWorker(self.data_dir, self.output_dir, settings)
        self.worker.progress.connect(self._on_progress)
        self.worker.status_update.connect(self._on_status_update)
        self.worker.finished.connect(self._on_finished)
        self.worker.error.connect(self._on_error)
        self.worker.start()

    def _cancel_optimization(self):
        """Cancel the running optimization."""
        if self.worker:
            self.worker.cancel()
            self.status_label.setText("Cancelling...")
            self.status_label.setStyleSheet("color: #ffaa00;")

    def _on_status_update(self, message: str):
        """Handle status update from worker."""
        self.status_label.setText(message)

    def _on_progress(self, current: int, total: int, best_objective: float,
                     best_trades: int, best_pnl: float, speed: float):
        """Handle progress update from worker."""
        # Update progress bar
        self.progress_bar.setMaximum(total)
        self.progress_bar.setValue(current)

        # Update metric cards
        self.completed_card.set_value(f"{current:,}")

        if best_objective is not None:
            color = "#00ff00" if best_objective > 0.5 else "#ffaa00"
            self.best_objective_card.set_value(f"{best_objective:.4f}", color)

        if best_trades is not None:
            self.best_trades_card.set_value(str(best_trades))

        if best_pnl is not None:
            color = "#00ff00" if best_pnl > 0 else "#ff4444"
            self.best_pnl_card.set_value(f"${best_pnl:,.0f}", color)

        if speed > 0:
            self.speed_card.set_value(f"{speed:.1f}/s")

        # Calculate time estimates
        pct = (current / total * 100) if total > 0 else 0

        # Get elapsed time from worker start
        if hasattr(self, '_start_time') and self._start_time:
            import time
            elapsed = time.time() - self._start_time
            elapsed_str = self._format_time(elapsed)

            # Estimate remaining time
            if current > 0 and speed > 0:
                remaining = (total - current) / speed
                remaining_str = self._format_time(remaining)
                time_info = f" | Elapsed: {elapsed_str} | ETA: {remaining_str}"
            else:
                time_info = f" | Elapsed: {elapsed_str}"
        else:
            time_info = ""

        self.status_label.setText(
            f"Processing: {current:,} / {total:,} ({pct:.1f}%){time_info}"
        )

    def _format_time(self, seconds: float) -> str:
        """Format seconds into human-readable time."""
        if seconds < 60:
            return f"{seconds:.0f}s"
        elif seconds < 3600:
            mins = int(seconds // 60)
            secs = int(seconds % 60)
            return f"{mins}m {secs}s"
        else:
            hours = int(seconds // 3600)
            mins = int((seconds % 3600) // 60)
            return f"{hours}h {mins}m"

    def _on_finished(self, results: dict):
        """Handle optimization completion."""
        self.run_button.setEnabled(True)
        self.cancel_button.setEnabled(False)
        self.progress_bar.setValue(self.progress_bar.maximum())

        # Calculate deltas from previous run
        self._update_delta_metrics(results)

        # Store current as previous for next run
        self.previous_results = self.last_results
        self.last_results = results

        self.status_label.setText(
            f"Complete! {results.get('completed', 0):,} combinations tested in "
            f"{results.get('total_time', 0):.1f}s"
        )
        self.status_label.setStyleSheet("color: #00ff00;")

        # Populate best parameters table
        best_params = results.get('best_params', {})
        self.best_params = best_params  # Store for double-click handler
        self._populate_best_params(best_params)

        # Populate top results table
        top_results = results.get('top_results', [])
        self._populate_top_results(top_results)

        # Persist results for next launch
        self._save_persisted_results(results)

        # Emit signal
        self.optimization_complete.emit(results)

    def _update_delta_metrics(self, results: dict):
        """Update delta metric cards comparing to previous run."""
        best_metrics = results.get('best_metrics', {})
        curr_pnl = best_metrics.get('total_pnl', 0)
        curr_wr = best_metrics.get('win_rate', 0)
        curr_pf = best_metrics.get('profit_factor', 0)

        # Get previous values
        if self.previous_results:
            prev_metrics = self.previous_results.get('best_metrics', {})
            prev_pnl = prev_metrics.get('total_pnl', 0)
            prev_wr = prev_metrics.get('win_rate', 0)
            prev_pf = prev_metrics.get('profit_factor', 0)
        else:
            prev_pnl, prev_wr, prev_pf = 0, 0, 0

        # Calculate deltas
        delta_pnl = curr_pnl - prev_pnl
        delta_wr = curr_wr - prev_wr
        delta_pf = curr_pf - prev_pf

        # Update cards with color coding
        if delta_pnl != 0 or self.previous_results:
            color = "#00ff00" if delta_pnl >= 0 else "#ff4444"
            sign = "+" if delta_pnl >= 0 else ""
            self.delta_pnl_card.set_value(f"{sign}${delta_pnl:,.0f}", color)
        else:
            self.delta_pnl_card.set_value("--")

        if delta_wr != 0 or self.previous_results:
            color = "#00ff00" if delta_wr >= 0 else "#ff4444"
            sign = "+" if delta_wr >= 0 else ""
            self.delta_wr_card.set_value(f"{sign}{delta_wr:.1f}%", color)
        else:
            self.delta_wr_card.set_value("--")

        if delta_pf != 0 or self.previous_results:
            color = "#00ff00" if delta_pf >= 0 else "#ff4444"
            sign = "+" if delta_pf >= 0 else ""
            self.delta_pf_card.set_value(f"{sign}{delta_pf:.2f}", color)
        else:
            self.delta_pf_card.set_value("--")

    def _on_error(self, error_msg: str):
        """Handle optimization error."""
        self.run_button.setEnabled(True)
        self.cancel_button.setEnabled(False)
        self.status_label.setText(f"Error: {error_msg}")
        self.status_label.setStyleSheet("color: #ff4444;")

    def _populate_best_params(self, params: dict):
        """Fill the best parameters table."""
        # Filter out day-of-week params for cleaner display
        display_params = {
            k: v for k, v in params.items()
            if not k.startswith('trade_') or k == 'trade_direction'
        }

        self.best_params_table.setRowCount(len(display_params))

        for row, (key, value) in enumerate(display_params.items()):
            self.best_params_table.setItem(row, 0, QTableWidgetItem(key))

            # Format value
            if isinstance(value, float):
                value_str = f"{value:.2f}"
            elif isinstance(value, bool):
                value_str = "Yes" if value else "No"
            else:
                value_str = str(value)

            self.best_params_table.setItem(row, 1, QTableWidgetItem(value_str))

    def _populate_top_results(self, results: list):
        """Fill the top results table."""
        self.top_results_data = results  # Store for equity curve and double-click handler
        self.results_table.setRowCount(len(results))

        for row, result in enumerate(results):
            # Direction
            self.results_table.setItem(
                row, 0, QTableWidgetItem(result.get('trade_direction', ''))
            )

            # Profit target
            pt = result.get('profit_target_percent', 0)
            self.results_table.setItem(row, 1, QTableWidgetItem(f"{pt:.1f}%"))

            # Stop type
            self.results_table.setItem(
                row, 2, QTableWidgetItem(result.get('stop_loss_type', ''))
            )

            # QQQ filter
            qqq = "Yes" if result.get('use_qqq_filter', False) else "No"
            self.results_table.setItem(row, 3, QTableWidgetItem(qqq))

            # Objective value
            obj = result.get('objective_value', 0)
            self.results_table.setItem(row, 4, QTableWidgetItem(f"{obj:.4f}"))

            # Profit factor
            pf = result.get('profit_factor', 0)
            self.results_table.setItem(row, 5, QTableWidgetItem(f"{pf:.2f}"))

            # Trades
            trades = result.get('total_trades', 0)
            self.results_table.setItem(row, 6, QTableWidgetItem(str(trades)))

            # Win rate
            wr = result.get('win_rate', 0)
            self.results_table.setItem(row, 7, QTableWidgetItem(f"{wr:.1f}%"))

            # P&L
            pnl = result.get('total_pnl', 0)
            self.results_table.setItem(row, 8, QTableWidgetItem(f"${pnl:,.2f}"))

    def _on_result_selected(self):
        """Handle result row selection - show equity curve."""
        selected_rows = self.results_table.selectionModel().selectedRows()
        if not selected_rows or not hasattr(self, 'top_results_data'):
            return

        row = selected_rows[0].row()
        if row >= len(self.top_results_data):
            return

        result = self.top_results_data[row]
        self._show_equity_curve(result)

    def _show_equity_curve(self, result: dict):
        """Generate and display equity curve for selected result."""
        import plotly.graph_objects as go

        # Get trade P&Ls and build cumulative equity
        trade_pnls = result.get('trade_pnls', [])

        if trade_pnls:
            # Build cumulative equity from actual trade P&Ls
            equity_data = []
            cumulative = 0
            for pnl in trade_pnls:
                cumulative += pnl
                equity_data.append(cumulative)
        else:
            # No trade data - show empty chart
            equity_data = [0]

        # Create figure
        fig = go.Figure()

        final_pnl = equity_data[-1] if equity_data else 0
        is_profitable = final_pnl >= 0

        fig.add_trace(go.Scatter(
            y=equity_data,
            mode='lines',
            name='Equity',
            line=dict(color='#00ff00' if is_profitable else '#ff4444', width=2),
            fill='tozeroy',
            fillcolor='rgba(0, 255, 0, 0.15)' if is_profitable else 'rgba(255, 68, 68, 0.15)'
        ))

        # Add zero line
        fig.add_hline(y=0, line_dash="dash", line_color="#555555", line_width=1)

        # Build title from result info
        direction = result.get('trade_direction', '')
        target = result.get('profit_target_percent', 0)
        pnl = result.get('total_pnl', 0)
        trades = result.get('total_trades', 0)
        pf = result.get('profit_factor', 0)

        fig.update_layout(
            title=dict(
                text=f"{direction} | {target:.1f}% | {trades} trades | PF {pf:.2f} | ${pnl:,.0f}",
                font=dict(size=11, color='#aaaaaa'),
                x=0.5,
                xanchor='center'
            ),
            paper_bgcolor='#1e1e1e',
            plot_bgcolor='#1e1e1e',
            font=dict(color='#cccccc', size=10),
            margin=dict(l=60, r=10, t=30, b=25),
            xaxis=dict(
                title=None,
                gridcolor='#333333',
                showgrid=True,
                zeroline=False
            ),
            yaxis=dict(
                title=None,
                gridcolor='#333333',
                showgrid=True,
                tickformat='$,.0f',
                zeroline=False
            ),
            showlegend=False
        )

        # Render to HTML and display
        html = fig.to_html(include_plotlyjs='cdn', full_html=True)
        self.equity_chart.setHtml(html)

    def _save_settings(self):
        """Save current settings to QSettings."""
        self.settings.setValue("opt/ticker", self.ticker_combo.currentText())
        self.settings.setValue("opt/qqq_filter", self.qqq_filter_check.isChecked())
        self.settings.setValue("opt/objective", self.objective_combo.currentText())
        # Save just the preset name (e.g., "standard" not "standard (288)")
        preset_name = self.preset_combo.currentText().split(" ")[0]
        self.settings.setValue("opt/preset", preset_name)

    def _load_settings(self):
        """Load saved settings from QSettings."""
        # Ticker
        ticker = self.settings.value("opt/ticker", "TSLA")
        idx = self.ticker_combo.findText(ticker)
        if idx >= 0:
            self.ticker_combo.setCurrentIndex(idx)

        # QQQ filter
        qqq_filter = self.settings.value("opt/qqq_filter", True, type=bool)
        self.qqq_filter_check.setChecked(qqq_filter)

        # Objective (default to profit_factor)
        objective = self.settings.value("opt/objective", "profit_factor")
        idx = self.objective_combo.findText(objective)
        if idx >= 0:
            self.objective_combo.setCurrentIndex(idx)

        # Preset (default to standard) - find by prefix since combo has "(count)" suffix
        preset = self.settings.value("opt/preset", "standard")
        for i in range(self.preset_combo.count()):
            if self.preset_combo.itemText(i).startswith(preset):
                self.preset_combo.setCurrentIndex(i)
                break
        self._update_preset_info(self.preset_combo.currentText())

    def _save_persisted_results(self, results: dict):
        """Save optimization results to disk for persistence across launches."""
        import json
        from pathlib import Path

        # Create results file path
        results_file = Path(self.output_dir) / "optimization" / "last_optimization.json"
        results_file.parent.mkdir(parents=True, exist_ok=True)

        # Strip trade_pnls from top_results to keep file size small
        save_results = results.copy()
        if 'top_results' in save_results:
            save_results['top_results'] = [
                {k: v for k, v in r.items() if k != 'trade_pnls'}
                for r in save_results['top_results']
            ]

        try:
            with open(results_file, 'w') as f:
                json.dump(save_results, f, indent=2, default=str)
        except Exception as e:
            print(f"Warning: Could not save optimization results: {e}")

    def _load_persisted_results(self):
        """Load previously saved optimization results."""
        import json
        from pathlib import Path

        results_file = Path(self.output_dir) / "optimization" / "last_optimization.json"

        if not results_file.exists():
            return

        try:
            with open(results_file, 'r') as f:
                results = json.load(f)

            self.last_results = results
            self.previous_results = results  # Use as baseline for first delta comparison

            # Populate UI with saved results
            best_params = results.get('best_params', {})
            self._populate_best_params(best_params)

            top_results = results.get('top_results', [])
            self._populate_top_results(top_results)

            # Update status to show loaded results
            completed = results.get('completed', 0)
            total_time = results.get('total_time', 0)
            self.status_label.setText(f"Loaded: {completed:,} combinations ({total_time:.1f}s)")
            self.status_label.setStyleSheet("color: #888888;")

            # Update metric cards with best metrics
            best_metrics = results.get('best_metrics', {})
            if best_metrics:
                self.completed_card.set_value(f"{completed:,}")

                obj = results.get('best_objective', 0)
                color = "#00ff00" if obj > 0.5 else "#ffaa00"
                self.best_objective_card.set_value(f"{obj:.4f}", color)

                trades = best_metrics.get('total_trades', 0)
                self.best_trades_card.set_value(str(trades))

                pnl = best_metrics.get('total_pnl', 0)
                color = "#00ff00" if pnl > 0 else "#ff4444"
                self.best_pnl_card.set_value(f"${pnl:,.0f}", color)

        except Exception as e:
            print(f"Warning: Could not load optimization results: {e}")
