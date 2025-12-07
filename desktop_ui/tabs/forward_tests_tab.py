"""
Forward Tests Tab - Run forward tests on saved optimizations with new market data.
"""

import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QLabel, QPushButton, QGroupBox, QSplitter,
    QTableWidget, QTableWidgetItem, QHeaderView, QFrame,
    QComboBox, QScrollArea, QMessageBox, QDateEdit
)
from PySide6.QtCore import Qt, Signal, QDate, QThread
from PySide6.QtGui import QColor
from PySide6.QtWebEngineWidgets import QWebEngineView

import plotly.graph_objects as go
from plotly.subplots import make_subplots

from desktop_ui.widgets.metrics_panel import MetricCard


@dataclass
class ForwardTestResult:
    """Single forward test run result."""
    run_date: str  # When this forward test was run
    test_start_date: str  # Start date of the test period
    test_end_date: str  # End date of the test period
    trades: int
    pnl: float
    win_rate: float
    profit_factor: float


@dataclass
class ForwardTest:
    """A forward test configuration and results."""
    ticker: str
    params: dict
    created_date: str
    backtest_pnl: float  # Original backtest P&L
    backtest_trades: int
    backtest_pf: float
    backtest_win_rate: float
    forward_results: List[ForwardTestResult]  # List of forward test runs
    notes: str = ""

    def total_forward_pnl(self) -> float:
        return sum(r.pnl for r in self.forward_results)

    def total_forward_trades(self) -> int:
        return sum(r.trades for r in self.forward_results)


class ForwardTestWorker(QThread):
    """Worker thread for running forward tests."""
    status = Signal(str)
    finished = Signal(dict)
    error = Signal(str)

    def __init__(self, data_dir: str, params: dict, ticker: str,
                 start_date: datetime, end_date: datetime):
        super().__init__()
        self.data_dir = data_dir
        self.params = params
        self.ticker = ticker
        self.start_date = start_date
        self.end_date = end_date

    def run(self):
        try:
            self.status.emit(f"Loading data for {self.ticker}...")

            from backtester.backtest_runner import BacktestRunner

            # Create backtest runner
            runner = BacktestRunner(self.data_dir)

            # Run backtest with date filter
            self.status.emit(f"Running forward test {self.start_date.date()} to {self.end_date.date()}...")

            result = runner.run(
                ticker=self.ticker,
                params=self.params,
                start_date=self.start_date,
                end_date=self.end_date
            )

            trades = result.get('trades', [])
            metrics = result.get('metrics', {})

            self.finished.emit({
                'trades': len(trades),
                'pnl': metrics.get('total_pnl', 0),
                'win_rate': metrics.get('win_rate', 0),
                'profit_factor': metrics.get('profit_factor', 0),
                'start_date': self.start_date.strftime('%Y-%m-%d'),
                'end_date': self.end_date.strftime('%Y-%m-%d')
            })

        except Exception as e:
            self.error.emit(str(e))


class ForwardTestsTab(QWidget):
    """Tab for managing and running forward tests."""

    def __init__(self, data_dir: str, output_dir: str):
        super().__init__()
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.forward_tests: Dict[str, ForwardTest] = {}  # ticker -> ForwardTest
        self.saved_results: Dict[str, List[dict]] = {}  # From saved tests
        self.selected_ticker: Optional[str] = None
        self.worker = None

        self._setup_ui()
        self._load_saved_results()
        self._load_forward_tests()

    def _setup_ui(self):
        """Create the UI layout."""
        layout = QVBoxLayout(self)
        layout.setSpacing(8)

        # Header
        header_layout = QHBoxLayout()
        header_label = QLabel("Forward Tests")
        header_label.setStyleSheet("font-size: 16px; font-weight: bold;")
        header_layout.addWidget(header_label)
        header_layout.addStretch()

        self.refresh_btn = QPushButton("Refresh")
        self.refresh_btn.clicked.connect(self._refresh_all)
        header_layout.addWidget(self.refresh_btn)

        layout.addLayout(header_layout)

        # Main splitter
        main_splitter = QSplitter(Qt.Horizontal)

        # Left panel - Test list and controls
        left_frame = QFrame()
        left_layout = QVBoxLayout(left_frame)
        left_layout.setContentsMargins(0, 0, 0, 0)

        # New test section
        new_test_group = QGroupBox("Start New Forward Test")
        new_test_layout = QVBoxLayout(new_test_group)

        # Ticker selection from saved tests
        ticker_layout = QHBoxLayout()
        ticker_layout.addWidget(QLabel("Ticker:"))
        self.ticker_combo = QComboBox()
        self.ticker_combo.currentTextChanged.connect(self._on_ticker_selected)
        ticker_layout.addWidget(self.ticker_combo)
        new_test_layout.addLayout(ticker_layout)

        # Optimization selection (when multiple saved for same ticker)
        opt_layout = QHBoxLayout()
        opt_layout.addWidget(QLabel("Optimization:"))
        self.optimization_combo = QComboBox()
        self.optimization_combo.currentIndexChanged.connect(self._on_optimization_selected)
        opt_layout.addWidget(self.optimization_combo)
        new_test_layout.addLayout(opt_layout)

        # Show selected params summary
        self.params_label = QLabel("Select a ticker to see parameters")
        self.params_label.setWordWrap(True)
        self.params_label.setStyleSheet("color: #888888; font-size: 11px;")
        new_test_layout.addWidget(self.params_label)

        # Start new forward test button
        self.start_new_btn = QPushButton("Start Forward Test")
        self.start_new_btn.setObjectName("primary")
        self.start_new_btn.clicked.connect(self._start_new_forward_test)
        self.start_new_btn.setEnabled(False)
        new_test_layout.addWidget(self.start_new_btn)

        left_layout.addWidget(new_test_group)

        # Active forward tests list
        tests_group = QGroupBox("Active Forward Tests")
        tests_layout = QVBoxLayout(tests_group)

        self.tests_table = QTableWidget()
        self.tests_table.setColumnCount(5)
        self.tests_table.setHorizontalHeaderLabels([
            "Ticker", "Fwd P&L", "Fwd Trades", "Runs", "Status"
        ])
        self.tests_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.tests_table.setSelectionBehavior(QTableWidget.SelectRows)
        self.tests_table.setSelectionMode(QTableWidget.SingleSelection)
        self.tests_table.itemSelectionChanged.connect(self._on_test_selected)
        self.tests_table.verticalHeader().setVisible(False)
        tests_layout.addWidget(self.tests_table)

        # Test actions
        actions_layout = QHBoxLayout()
        self.run_test_btn = QPushButton("Run Forward Test")
        self.run_test_btn.clicked.connect(self._run_forward_test)
        self.run_test_btn.setEnabled(False)
        actions_layout.addWidget(self.run_test_btn)

        self.delete_test_btn = QPushButton("Delete")
        self.delete_test_btn.clicked.connect(self._delete_forward_test)
        self.delete_test_btn.setEnabled(False)
        actions_layout.addWidget(self.delete_test_btn)

        tests_layout.addLayout(actions_layout)

        left_layout.addWidget(tests_group)

        main_splitter.addWidget(left_frame)

        # Right panel - Details and results
        right_frame = QFrame()
        right_layout = QVBoxLayout(right_frame)
        right_layout.setContentsMargins(0, 0, 0, 0)

        # Metrics row
        metrics_layout = QHBoxLayout()
        metrics_layout.setSpacing(8)

        self.backtest_pnl_card = MetricCard("Backtest P&L", compact=True)
        metrics_layout.addWidget(self.backtest_pnl_card)

        self.forward_pnl_card = MetricCard("Forward P&L", compact=True)
        metrics_layout.addWidget(self.forward_pnl_card)

        self.forward_trades_card = MetricCard("Fwd Trades", compact=True)
        metrics_layout.addWidget(self.forward_trades_card)

        self.total_runs_card = MetricCard("Test Runs", compact=True)
        metrics_layout.addWidget(self.total_runs_card)

        self.consistency_card = MetricCard("Consistency", compact=True)
        metrics_layout.addWidget(self.consistency_card)

        metrics_layout.addStretch()
        right_layout.addLayout(metrics_layout)

        # Chart splitter
        chart_splitter = QSplitter(Qt.Vertical)

        # Forward test equity chart
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

        self.chart_view = QWebEngineView()
        self.chart_view.setMinimumHeight(200)
        chart_layout.addWidget(self.chart_view)

        chart_splitter.addWidget(chart_frame)

        # Forward test runs table
        runs_frame = QFrame()
        runs_layout = QVBoxLayout(runs_frame)
        runs_layout.setContentsMargins(0, 0, 0, 0)

        runs_label = QLabel("Forward Test Runs")
        runs_label.setStyleSheet("font-weight: bold; font-size: 13px;")
        runs_layout.addWidget(runs_label)

        self.runs_table = QTableWidget()
        self.runs_table.setColumnCount(6)
        self.runs_table.setHorizontalHeaderLabels([
            "Run Date", "Period Start", "Period End", "Trades", "P&L", "Win%"
        ])
        self.runs_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.runs_table.setAlternatingRowColors(True)
        self.runs_table.verticalHeader().setVisible(False)
        runs_layout.addWidget(self.runs_table)

        chart_splitter.addWidget(runs_frame)
        chart_splitter.setSizes([300, 200])

        right_layout.addWidget(chart_splitter)

        # Status bar
        self.status_label = QLabel("Ready")
        self.status_label.setStyleSheet("color: #888888;")
        right_layout.addWidget(self.status_label)

        main_splitter.addWidget(right_frame)
        main_splitter.setSizes([300, 700])

        layout.addWidget(main_splitter)

    def _get_saved_results_path(self) -> Path:
        """Get path to saved results file."""
        return self.output_dir / "saved_tests" / "saved_results.json"

    def _get_forward_tests_path(self) -> Path:
        """Get path to forward tests file."""
        path = self.output_dir / "forward_tests"
        path.mkdir(parents=True, exist_ok=True)
        return path / "forward_tests.json"

    def _load_saved_results(self):
        """Load saved results from disk."""
        storage_path = self._get_saved_results_path()

        if not storage_path.exists():
            self.saved_results = {}
            return

        try:
            with open(storage_path, 'r') as f:
                self.saved_results = json.load(f)
            self._update_ticker_combo()
        except Exception as e:
            print(f"Error loading saved results: {e}")
            self.saved_results = {}

    def _load_forward_tests(self):
        """Load forward tests from disk."""
        storage_path = self._get_forward_tests_path()

        if not storage_path.exists():
            self.forward_tests = {}
            return

        try:
            with open(storage_path, 'r') as f:
                data = json.load(f)

            # Reconstruct ForwardTest objects
            self.forward_tests = {}
            for ticker, test_data in data.items():
                forward_results = [
                    ForwardTestResult(**r) for r in test_data.get('forward_results', [])
                ]
                test_data['forward_results'] = forward_results
                self.forward_tests[ticker] = ForwardTest(**test_data)

            self._update_tests_table()
        except Exception as e:
            print(f"Error loading forward tests: {e}")
            self.forward_tests = {}

    def _save_forward_tests(self):
        """Save forward tests to disk."""
        storage_path = self._get_forward_tests_path()

        try:
            # Convert to serializable format
            data = {}
            for ticker, test in self.forward_tests.items():
                test_dict = asdict(test)
                data[ticker] = test_dict

            with open(storage_path, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            print(f"Error saving forward tests: {e}")

    def _update_ticker_combo(self):
        """Update ticker combo with saved test tickers."""
        self.ticker_combo.clear()
        self.ticker_combo.addItem("-- Select Ticker --")

        for ticker in sorted(self.saved_results.keys()):
            # Check if already has forward test
            already_active = ticker in self.forward_tests
            display = f"{ticker}" + (" (active)" if already_active else "")
            self.ticker_combo.addItem(display, ticker)

    def _update_tests_table(self):
        """Update the forward tests table."""
        self.tests_table.setRowCount(len(self.forward_tests))

        for row, (ticker, test) in enumerate(sorted(self.forward_tests.items())):
            fwd_pnl = test.total_forward_pnl()
            fwd_trades = test.total_forward_trades()
            runs = len(test.forward_results)

            # Determine status
            if runs == 0:
                status = "New"
                status_color = "#ffaa00"
            elif fwd_pnl > 0:
                status = "Profitable"
                status_color = "#00ff00"
            else:
                status = "Losing"
                status_color = "#ff4444"

            items = [
                ticker,
                f"${fwd_pnl:,.0f}",
                str(fwd_trades),
                str(runs),
                status
            ]

            for col, value in enumerate(items):
                item = QTableWidgetItem(value)
                item.setFlags(item.flags() & ~Qt.ItemIsEditable)

                # Color P&L
                if col == 1:
                    item.setForeground(QColor("#00ff00") if fwd_pnl >= 0 else QColor("#ff4444"))

                # Color status
                if col == 4:
                    item.setForeground(QColor(status_color))

                self.tests_table.setItem(row, col, item)

    def _on_ticker_selected(self, text: str):
        """Handle ticker selection."""
        if text == "-- Select Ticker --" or not text:
            self.selected_ticker = None
            self.optimization_combo.clear()
            self.start_new_btn.setEnabled(False)
            self.params_label.setText("Select a ticker to see parameters")
            return

        # Extract ticker from display text
        ticker = self.ticker_combo.currentData()
        if not ticker:
            return

        self.selected_ticker = ticker

        # Populate optimization combo with all saved results for this ticker
        results_list = self.saved_results.get(ticker, [])
        self.optimization_combo.clear()

        if results_list:
            # Sort by P&L descending (best first)
            sorted_results = sorted(results_list, key=lambda r: r.get('total_pnl', 0), reverse=True)

            for i, result in enumerate(sorted_results):
                pnl = result.get('total_pnl', 0)
                pf = result.get('profit_factor', 0)
                params = result.get('params', {})
                direction = params.get('trade_direction', 'both')

                # Create descriptive label
                label = f"${pnl:,.0f} | PF {pf:.2f} | {direction}"
                if i == 0:
                    label += " (best)"

                self.optimization_combo.addItem(label, i)  # Store index as data

            # Select best by default
            self._on_optimization_selected(0)
        else:
            self.params_label.setText("No saved results found")
            self.start_new_btn.setEnabled(False)

    def _on_optimization_selected(self, index: int):
        """Handle optimization selection from combo."""
        if index < 0 or not self.selected_ticker:
            return

        results_list = self.saved_results.get(self.selected_ticker, [])
        if not results_list:
            return

        # Get sorted results (same order as combo)
        sorted_results = sorted(results_list, key=lambda r: r.get('total_pnl', 0), reverse=True)

        if index >= len(sorted_results):
            return

        result = sorted_results[index]
        params = result.get('params', {})

        # Show key params
        direction = params.get('trade_direction', 'both')
        ib_dur = params.get('ib_duration_minutes', 30)
        target = params.get('profit_target_percent', 1.0)
        pnl = result.get('total_pnl', 0)
        pf = result.get('profit_factor', 0)
        win_rate = result.get('win_rate', 0)
        trades = result.get('total_trades', 0)

        self.params_label.setText(
            f"Direction: {direction} | IB: {ib_dur}min | Target: {target:.1f}%\n"
            f"P&L: ${pnl:,.0f} | PF: {pf:.2f} | Win: {win_rate:.0f}% | Trades: {trades}"
        )
        self.start_new_btn.setEnabled(self.selected_ticker not in self.forward_tests)

    def _on_test_selected(self):
        """Handle forward test selection."""
        selected = self.tests_table.selectedItems()
        if not selected:
            self.run_test_btn.setEnabled(False)
            self.delete_test_btn.setEnabled(False)
            self._clear_details()
            return

        row = selected[0].row()
        ticker = self.tests_table.item(row, 0).text()

        if ticker in self.forward_tests:
            self.run_test_btn.setEnabled(True)
            self.delete_test_btn.setEnabled(True)
            self._show_test_details(ticker)

    def _start_new_forward_test(self):
        """Start a new forward test from saved optimization."""
        if not self.selected_ticker:
            return

        ticker = self.selected_ticker

        # Get selected result from combo box
        results_list = self.saved_results.get(ticker, [])
        if not results_list:
            return

        # Get the selected optimization index
        opt_index = self.optimization_combo.currentIndex()
        if opt_index < 0:
            opt_index = 0

        # Get sorted results (same order as combo)
        sorted_results = sorted(results_list, key=lambda r: r.get('total_pnl', 0), reverse=True)

        if opt_index >= len(sorted_results):
            return

        selected_result = sorted_results[opt_index]

        # Create new forward test
        forward_test = ForwardTest(
            ticker=ticker,
            params=selected_result.get('params', {}),
            created_date=datetime.now().isoformat(),
            backtest_pnl=selected_result.get('total_pnl', 0),
            backtest_trades=selected_result.get('total_trades', 0),
            backtest_pf=selected_result.get('profit_factor', 0),
            backtest_win_rate=selected_result.get('win_rate', 0),
            forward_results=[]
        )

        self.forward_tests[ticker] = forward_test
        self._save_forward_tests()
        self._update_tests_table()
        self._update_ticker_combo()

        self.status_label.setText(f"Created forward test for {ticker}")
        self.status_label.setStyleSheet("color: #00ff00;")

    def _run_forward_test(self):
        """Run forward test on selected ticker."""
        selected = self.tests_table.selectedItems()
        if not selected:
            return

        row = selected[0].row()
        ticker = self.tests_table.item(row, 0).text()

        if ticker not in self.forward_tests:
            return

        test = self.forward_tests[ticker]

        # Determine date range for forward test
        # Use last run's end date as start, or backtest end date
        if test.forward_results:
            last_run = test.forward_results[-1]
            start_date = datetime.strptime(last_run.test_end_date, '%Y-%m-%d') + timedelta(days=1)
        else:
            # Start from a week ago if new test
            start_date = datetime.now() - timedelta(days=7)

        end_date = datetime.now()

        # Ensure we have a valid date range
        if start_date >= end_date:
            QMessageBox.information(
                self, "No New Data",
                f"No new data available since last forward test run.\n"
                f"Last test ended: {test.forward_results[-1].test_end_date if test.forward_results else 'N/A'}"
            )
            return

        # Run the forward test
        self.run_test_btn.setEnabled(False)
        self.status_label.setText(f"Running forward test for {ticker}...")
        self.status_label.setStyleSheet("color: #2a82da;")

        self.worker = ForwardTestWorker(
            str(self.data_dir),
            test.params,
            ticker,
            start_date,
            end_date
        )
        self.worker.status.connect(self._on_worker_status)
        self.worker.finished.connect(lambda r: self._on_forward_test_complete(ticker, r))
        self.worker.error.connect(self._on_worker_error)
        self.worker.start()

    def _on_worker_status(self, message: str):
        """Handle worker status update."""
        self.status_label.setText(message)

    def _on_forward_test_complete(self, ticker: str, result: dict):
        """Handle forward test completion."""
        self.run_test_btn.setEnabled(True)

        if ticker not in self.forward_tests:
            return

        test = self.forward_tests[ticker]

        # Add result
        forward_result = ForwardTestResult(
            run_date=datetime.now().isoformat(),
            test_start_date=result['start_date'],
            test_end_date=result['end_date'],
            trades=result['trades'],
            pnl=result['pnl'],
            win_rate=result['win_rate'],
            profit_factor=result['profit_factor']
        )

        test.forward_results.append(forward_result)

        self._save_forward_tests()
        self._update_tests_table()
        self._show_test_details(ticker)

        pnl = result['pnl']
        trades = result['trades']
        self.status_label.setText(
            f"Forward test complete: {trades} trades, ${pnl:,.0f} P&L"
        )
        self.status_label.setStyleSheet("color: #00ff00;" if pnl >= 0 else "color: #ff4444;")

    def _on_worker_error(self, error: str):
        """Handle worker error."""
        self.run_test_btn.setEnabled(True)
        self.status_label.setText(f"Error: {error[:80]}")
        self.status_label.setStyleSheet("color: #ff4444;")
        print(f"Forward test error: {error}")

    def _delete_forward_test(self):
        """Delete selected forward test."""
        selected = self.tests_table.selectedItems()
        if not selected:
            return

        row = selected[0].row()
        ticker = self.tests_table.item(row, 0).text()

        reply = QMessageBox.question(
            self, "Delete Forward Test",
            f"Delete forward test for {ticker}?\nThis will remove all forward test history.",
            QMessageBox.Yes | QMessageBox.No
        )

        if reply == QMessageBox.Yes:
            if ticker in self.forward_tests:
                del self.forward_tests[ticker]
                self._save_forward_tests()
                self._update_tests_table()
                self._update_ticker_combo()
                self._clear_details()
                self.status_label.setText(f"Deleted forward test for {ticker}")

    def _show_test_details(self, ticker: str):
        """Show details for selected forward test."""
        if ticker not in self.forward_tests:
            return

        test = self.forward_tests[ticker]

        # Update metric cards
        pnl_color = "#00ff00" if test.backtest_pnl >= 0 else "#ff4444"
        self.backtest_pnl_card.set_value(f"${test.backtest_pnl:,.0f}", pnl_color)

        fwd_pnl = test.total_forward_pnl()
        fwd_pnl_color = "#00ff00" if fwd_pnl >= 0 else "#ff4444"
        self.forward_pnl_card.set_value(f"${fwd_pnl:,.0f}", fwd_pnl_color)

        self.forward_trades_card.set_value(str(test.total_forward_trades()))
        self.total_runs_card.set_value(str(len(test.forward_results)))

        # Calculate consistency (% of profitable runs)
        if test.forward_results:
            profitable_runs = sum(1 for r in test.forward_results if r.pnl > 0)
            consistency = profitable_runs / len(test.forward_results) * 100
            cons_color = "#00ff00" if consistency >= 60 else "#ffaa00" if consistency >= 40 else "#ff4444"
            self.consistency_card.set_value(f"{consistency:.0f}%", cons_color)
        else:
            self.consistency_card.set_value("--")

        # Update runs table
        self._update_runs_table(test)

        # Update chart
        self._update_chart(test)

    def _update_runs_table(self, test: ForwardTest):
        """Update the forward test runs table."""
        self.runs_table.setRowCount(len(test.forward_results))

        for row, result in enumerate(reversed(test.forward_results)):  # Most recent first
            run_date = datetime.fromisoformat(result.run_date).strftime('%Y-%m-%d %H:%M')

            items = [
                run_date,
                result.test_start_date,
                result.test_end_date,
                str(result.trades),
                f"${result.pnl:,.0f}",
                f"{result.win_rate:.1f}%"
            ]

            for col, value in enumerate(items):
                item = QTableWidgetItem(value)
                item.setFlags(item.flags() & ~Qt.ItemIsEditable)

                # Color P&L
                if col == 4:
                    item.setForeground(QColor("#00ff00") if result.pnl >= 0 else QColor("#ff4444"))

                self.runs_table.setItem(row, col, item)

    def _update_chart(self, test: ForwardTest):
        """Update the forward test chart."""
        if not test.forward_results:
            self.chart_view.setHtml("")
            return

        # Build cumulative P&L
        cumulative_pnl = []
        dates = []
        running_total = 0

        for result in test.forward_results:
            running_total += result.pnl
            cumulative_pnl.append(running_total)
            dates.append(result.test_end_date)

        # Create figure
        fig = make_subplots(
            rows=2, cols=1,
            row_heights=[0.6, 0.4],
            shared_xaxes=True,
            vertical_spacing=0.1,
            subplot_titles=("Cumulative Forward P&L", "P&L per Run")
        )

        # Cumulative P&L line
        is_profitable = running_total >= 0
        line_color = '#00ff00' if is_profitable else '#ff4444'

        fig.add_trace(
            go.Scatter(
                x=dates,
                y=cumulative_pnl,
                mode='lines+markers',
                name='Cumulative P&L',
                line=dict(color=line_color, width=2),
                marker=dict(size=8),
                fill='tozeroy',
                fillcolor=f'rgba({"0, 255, 0" if is_profitable else "255, 68, 68"}, 0.1)'
            ),
            row=1, col=1
        )

        # Per-run P&L bars
        run_pnls = [r.pnl for r in test.forward_results]
        bar_colors = ['#00ff00' if p >= 0 else '#ff4444' for p in run_pnls]

        fig.add_trace(
            go.Bar(
                x=dates,
                y=run_pnls,
                name='Run P&L',
                marker_color=bar_colors
            ),
            row=2, col=1
        )

        # Add backtest reference line
        fig.add_hline(
            y=0, line_dash="dash", line_color="#555555",
            line_width=1, row=1, col=1
        )

        # Update layout
        fig.update_layout(
            height=300,
            margin=dict(l=60, r=20, t=40, b=20),
            paper_bgcolor='#1e1e1e',
            plot_bgcolor='#252525',
            font=dict(color='#cccccc', size=10),
            showlegend=False,
            title=dict(
                text=f"{test.ticker} Forward Test | Total: ${running_total:,.0f}",
                font=dict(size=12, color='#aaaaaa'),
                x=0.5
            )
        )

        fig.update_xaxes(gridcolor='#333333', showgrid=True)
        fig.update_yaxes(gridcolor='#333333', showgrid=True, tickformat='$,.0f')

        # Render to HTML
        html = fig.to_html(include_plotlyjs='cdn', full_html=True)
        html = html.replace('<body>', '<body style="background-color: #1e1e1e; margin: 0; padding: 0;">')
        self.chart_view.setHtml(html)

    def _clear_details(self):
        """Clear the details panel."""
        self.backtest_pnl_card.set_value("--")
        self.forward_pnl_card.set_value("--")
        self.forward_trades_card.set_value("--")
        self.total_runs_card.set_value("--")
        self.consistency_card.set_value("--")
        self.runs_table.setRowCount(0)
        self.chart_view.setHtml("")

    def _refresh_all(self):
        """Refresh all data."""
        self._load_saved_results()
        self._load_forward_tests()
        self._clear_details()
        self.status_label.setText("Refreshed")

    def set_data_dir(self, path: str):
        """Update data directory."""
        self.data_dir = Path(path)

    def set_output_dir(self, path: str):
        """Update output directory."""
        self.output_dir = Path(path)
        self._refresh_all()
