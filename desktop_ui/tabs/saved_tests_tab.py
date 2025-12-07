"""
Saved Tests Tab - Store and display the best test results per instrument.
"""

import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QLabel, QPushButton, QGroupBox, QSplitter,
    QTableWidget, QTableWidgetItem, QHeaderView, QFrame,
    QMessageBox, QComboBox, QTextEdit
)
from PySide6.QtCore import Qt, Signal
from PySide6.QtWebEngineWidgets import QWebEngineView

import plotly.graph_objects as go
from plotly.subplots import make_subplots


@dataclass
class SavedTestResult:
    """A saved test result with parameters and metrics."""
    ticker: str
    run_date: str  # ISO format datetime
    total_pnl: float
    profit_factor: float
    win_rate: float
    total_trades: int
    max_drawdown: float
    sharpe_ratio: float
    params: Dict
    equity_curve: List[float]  # Running P&L values

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> 'SavedTestResult':
        return cls(**data)


class SavedTestsTab(QWidget):
    """Tab for viewing and managing saved test results."""

    # Signal emitted when user wants to load a saved test's parameters
    load_params_requested = Signal(dict, str)  # params, ticker

    def __init__(self, output_dir: str):
        super().__init__()
        self.output_dir = Path(output_dir)
        self.saved_results: Dict[str, List[SavedTestResult]] = {}  # ticker -> list of results
        self._current_result: Optional[SavedTestResult] = None

        self._setup_ui()
        self._load_all_results()
        self._refresh_table()

    def _setup_ui(self):
        """Create the UI layout."""
        layout = QVBoxLayout(self)
        layout.setSpacing(12)

        # Header with action buttons
        header_frame = QFrame()
        header_layout = QHBoxLayout(header_frame)
        header_layout.setContentsMargins(0, 0, 0, 0)

        header_label = QLabel("Saved Test Results")
        header_label.setStyleSheet("font-size: 16px; font-weight: bold;")
        header_layout.addWidget(header_label)

        header_layout.addStretch()

        # Filter by ticker
        header_layout.addWidget(QLabel("Filter:"))
        self.filter_combo = QComboBox()
        self.filter_combo.addItem("All Tickers")
        self.filter_combo.setMinimumWidth(100)
        self.filter_combo.currentTextChanged.connect(self._refresh_table)
        header_layout.addWidget(self.filter_combo)

        # Refresh button
        self.refresh_btn = QPushButton("Refresh")
        self.refresh_btn.clicked.connect(self._refresh_all)
        header_layout.addWidget(self.refresh_btn)

        # Delete button
        self.delete_btn = QPushButton("Delete Selected")
        self.delete_btn.clicked.connect(self._delete_selected)
        self.delete_btn.setEnabled(False)
        header_layout.addWidget(self.delete_btn)

        layout.addWidget(header_frame)

        # Main splitter - table on left, details on right
        splitter = QSplitter(Qt.Horizontal)

        # Left side - results table
        table_frame = QFrame()
        table_layout = QVBoxLayout(table_frame)
        table_layout.setContentsMargins(0, 0, 0, 0)

        self.results_table = QTableWidget()
        self.results_table.setColumnCount(7)
        self.results_table.setHorizontalHeaderLabels([
            "Ticker", "Run Date", "P&L", "PF", "Win%", "Trades", "Sharpe"
        ])
        self.results_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.results_table.setAlternatingRowColors(True)
        self.results_table.verticalHeader().setVisible(False)
        self.results_table.setSelectionBehavior(QTableWidget.SelectRows)
        self.results_table.setSelectionMode(QTableWidget.SingleSelection)
        self.results_table.itemSelectionChanged.connect(self._on_selection_changed)
        self.results_table.setSortingEnabled(True)

        table_layout.addWidget(self.results_table)
        splitter.addWidget(table_frame)

        # Right side - details panel
        details_frame = QFrame()
        details_layout = QVBoxLayout(details_frame)

        # Equity curve chart
        chart_group = QGroupBox("Equity Curve")
        chart_layout = QVBoxLayout(chart_group)

        self.chart_view = QWebEngineView()
        self.chart_view.setMinimumHeight(250)
        chart_layout.addWidget(self.chart_view)

        details_layout.addWidget(chart_group)

        # Parameters display
        params_group = QGroupBox("Parameters")
        params_layout = QVBoxLayout(params_group)

        self.params_text = QTextEdit()
        self.params_text.setReadOnly(True)
        self.params_text.setMaximumHeight(200)
        self.params_text.setStyleSheet("""
            QTextEdit {
                background-color: #252525;
                color: #cccccc;
                font-family: Consolas, monospace;
                font-size: 11px;
            }
        """)
        params_layout.addWidget(self.params_text)

        # Load params button
        self.load_params_btn = QPushButton("Load Parameters to Optimizer")
        self.load_params_btn.setEnabled(False)
        self.load_params_btn.clicked.connect(self._load_params_to_optimizer)
        params_layout.addWidget(self.load_params_btn)

        details_layout.addWidget(params_group)

        splitter.addWidget(details_frame)

        # Set splitter proportions (40% table, 60% details)
        splitter.setSizes([400, 600])

        layout.addWidget(splitter)

    def _get_storage_path(self) -> Path:
        """Get path to saved results file."""
        storage_dir = self.output_dir / "saved_tests"
        storage_dir.mkdir(parents=True, exist_ok=True)
        return storage_dir / "saved_results.json"

    def _load_all_results(self):
        """Load all saved results from disk."""
        storage_path = self._get_storage_path()

        if not storage_path.exists():
            self.saved_results = {}
            return

        try:
            with open(storage_path, 'r') as f:
                data = json.load(f)

            self.saved_results = {}
            for ticker, results_list in data.items():
                self.saved_results[ticker] = [
                    SavedTestResult.from_dict(r) for r in results_list
                ]

            # Update filter combo
            self._update_filter_combo()

        except Exception as e:
            print(f"Error loading saved results: {e}")
            self.saved_results = {}

    def _save_all_results(self):
        """Save all results to disk."""
        storage_path = self._get_storage_path()

        try:
            data = {}
            for ticker, results_list in self.saved_results.items():
                data[ticker] = [r.to_dict() for r in results_list]

            with open(storage_path, 'w') as f:
                json.dump(data, f, indent=2)

        except Exception as e:
            print(f"Error saving results: {e}")

    def _update_filter_combo(self):
        """Update filter combo with available tickers."""
        current = self.filter_combo.currentText()
        self.filter_combo.blockSignals(True)
        self.filter_combo.clear()
        self.filter_combo.addItem("All Tickers")

        for ticker in sorted(self.saved_results.keys()):
            self.filter_combo.addItem(ticker)

        # Restore selection if possible
        idx = self.filter_combo.findText(current)
        if idx >= 0:
            self.filter_combo.setCurrentIndex(idx)

        self.filter_combo.blockSignals(False)

    def _refresh_table(self):
        """Refresh the results table."""
        self.results_table.setRowCount(0)

        filter_ticker = self.filter_combo.currentText()

        # Collect all results to display
        all_results = []
        for ticker, results_list in self.saved_results.items():
            if filter_ticker == "All Tickers" or filter_ticker == ticker:
                for result in results_list:
                    all_results.append(result)

        # Sort by run date (newest first)
        all_results.sort(key=lambda r: r.run_date, reverse=True)

        self.results_table.setRowCount(len(all_results))

        for row, result in enumerate(all_results):
            # Format values
            run_date = result.run_date[:16].replace('T', ' ')  # YYYY-MM-DD HH:MM
            pnl_str = f"${result.total_pnl:,.0f}"
            pf_str = f"{result.profit_factor:.2f}"
            win_str = f"{result.win_rate:.1f}%"
            trades_str = str(result.total_trades)
            sharpe_str = f"{result.sharpe_ratio:.2f}"

            # Color P&L
            pnl_color = "#00ff00" if result.total_pnl > 0 else "#ff4444"

            items = [
                result.ticker,
                run_date,
                pnl_str,
                pf_str,
                win_str,
                trades_str,
                sharpe_str
            ]

            for col, value in enumerate(items):
                item = QTableWidgetItem(value)
                item.setFlags(item.flags() & ~Qt.ItemIsEditable)

                # Store result reference in first column
                if col == 0:
                    item.setData(Qt.UserRole, result)

                # Color P&L column
                if col == 2:
                    item.setForeground(Qt.GlobalColor.green if result.total_pnl > 0 else Qt.GlobalColor.red)

                self.results_table.setItem(row, col, item)

    def _on_selection_changed(self):
        """Handle selection change in table."""
        selected = self.results_table.selectedItems()

        if not selected:
            self._current_result = None
            self.delete_btn.setEnabled(False)
            self.load_params_btn.setEnabled(False)
            self.params_text.clear()
            self.chart_view.setHtml("")
            return

        # Get the result from the first column's user data
        row = selected[0].row()
        first_item = self.results_table.item(row, 0)
        result = first_item.data(Qt.UserRole)

        if result:
            self._current_result = result
            self.delete_btn.setEnabled(True)
            self.load_params_btn.setEnabled(True)
            self._display_result_details(result)

    def _display_result_details(self, result: SavedTestResult):
        """Display details for selected result."""
        # Update params text
        params_lines = []
        for key, value in sorted(result.params.items()):
            params_lines.append(f"{key}: {value}")
        self.params_text.setText("\n".join(params_lines))

        # Update equity curve chart
        self._update_equity_chart(result)

    def _update_equity_chart(self, result: SavedTestResult):
        """Update equity curve chart for result."""
        if not result.equity_curve:
            self.chart_view.setHtml("<p style='color:#888;text-align:center;padding:50px;'>No equity curve data</p>")
            return

        # Create equity curve figure
        fig = make_subplots(
            rows=2, cols=1,
            row_heights=[0.7, 0.3],
            shared_xaxes=True,
            vertical_spacing=0.05,
            subplot_titles=("Equity Curve", "Drawdown")
        )

        # Equity curve
        equity = result.equity_curve
        x = list(range(len(equity)))

        fig.add_trace(
            go.Scatter(
                x=x, y=equity,
                mode='lines',
                name='Equity',
                line=dict(color='#2a82da', width=2),
                fill='tozeroy',
                fillcolor='rgba(42, 130, 218, 0.1)'
            ),
            row=1, col=1
        )

        # Calculate drawdown
        peak = 0
        drawdown = []
        for val in equity:
            peak = max(peak, val)
            dd = (val - peak) / peak * 100 if peak > 0 else 0
            drawdown.append(dd)

        fig.add_trace(
            go.Scatter(
                x=x, y=drawdown,
                mode='lines',
                name='Drawdown',
                line=dict(color='#ff4444', width=1),
                fill='tozeroy',
                fillcolor='rgba(255, 68, 68, 0.2)'
            ),
            row=2, col=1
        )

        # Update layout
        fig.update_layout(
            height=250,
            margin=dict(l=50, r=20, t=30, b=20),
            paper_bgcolor='#1e1e1e',
            plot_bgcolor='#252525',
            font=dict(color='#cccccc', size=10),
            showlegend=False,
            title=dict(
                text=f"{result.ticker} - ${result.total_pnl:,.0f}",
                font=dict(size=12)
            )
        )

        fig.update_xaxes(gridcolor='#333333', showgrid=True)
        fig.update_yaxes(gridcolor='#333333', showgrid=True)

        # Render to HTML
        html = fig.to_html(include_plotlyjs='cdn', full_html=True)
        self.chart_view.setHtml(html)

    def save_test_result(
        self,
        ticker: str,
        total_pnl: float,
        profit_factor: float,
        win_rate: float,
        total_trades: int,
        max_drawdown: float,
        sharpe_ratio: float,
        params: Dict,
        equity_curve: List[float]
    ) -> bool:
        """
        Save a test result.

        Returns True if this is a new best P&L for this ticker.
        """
        result = SavedTestResult(
            ticker=ticker,
            run_date=datetime.now().isoformat(),
            total_pnl=total_pnl,
            profit_factor=profit_factor,
            win_rate=win_rate,
            total_trades=total_trades,
            max_drawdown=max_drawdown,
            sharpe_ratio=sharpe_ratio,
            params=params,
            equity_curve=equity_curve
        )

        # Initialize ticker list if needed
        if ticker not in self.saved_results:
            self.saved_results[ticker] = []

        # Check if this is a new best
        existing_pnls = [r.total_pnl for r in self.saved_results[ticker]]
        is_new_best = not existing_pnls or total_pnl > max(existing_pnls)

        # Add result
        self.saved_results[ticker].append(result)

        # Save to disk
        self._save_all_results()

        # Update UI
        self._update_filter_combo()
        self._refresh_table()

        return is_new_best

    def _delete_selected(self):
        """Delete the selected result."""
        if not self._current_result:
            return

        result = self._current_result

        # Confirm deletion
        reply = QMessageBox.question(
            self,
            "Delete Result",
            f"Delete saved result for {result.ticker} from {result.run_date[:10]}?\n\n"
            f"P&L: ${result.total_pnl:,.0f}",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )

        if reply != QMessageBox.Yes:
            return

        # Remove from saved results
        if result.ticker in self.saved_results:
            self.saved_results[result.ticker] = [
                r for r in self.saved_results[result.ticker]
                if r.run_date != result.run_date
            ]

            # Remove ticker if no results left
            if not self.saved_results[result.ticker]:
                del self.saved_results[result.ticker]

        # Save and refresh
        self._save_all_results()
        self._update_filter_combo()
        self._refresh_table()

        self._current_result = None
        self.delete_btn.setEnabled(False)
        self.load_params_btn.setEnabled(False)
        self.params_text.clear()
        self.chart_view.setHtml("")

    def _load_params_to_optimizer(self):
        """Emit signal to load current result's params to optimizer."""
        if self._current_result:
            self.load_params_requested.emit(
                self._current_result.params,
                self._current_result.ticker
            )

    def _refresh_all(self):
        """Reload all results from disk and refresh."""
        self._load_all_results()
        self._refresh_table()

    def set_output_dir(self, path: str):
        """Update output directory."""
        self.output_dir = Path(path)
        self._load_all_results()
        self._refresh_table()

    def get_best_pnl(self, ticker: str) -> Optional[float]:
        """Get the best P&L for a ticker, or None if no results."""
        if ticker not in self.saved_results:
            return None

        pnls = [r.total_pnl for r in self.saved_results[ticker]]
        return max(pnls) if pnls else None
