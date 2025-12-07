"""
Portfolio Tab - Combine multiple saved tests to view combined equity and stats.
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QLabel, QPushButton, QGroupBox, QSplitter,
    QTableWidget, QTableWidgetItem, QHeaderView, QFrame,
    QCheckBox, QScrollArea
)
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor
from PySide6.QtWebEngineWidgets import QWebEngineView

import plotly.graph_objects as go
from plotly.subplots import make_subplots

from desktop_ui.widgets.metrics_panel import MetricCard


class PortfolioTab(QWidget):
    """Tab for combining multiple saved tests into a portfolio view."""

    def __init__(self, output_dir: str):
        super().__init__()
        self.output_dir = Path(output_dir)
        self.saved_results: Dict[str, List[dict]] = {}  # ticker -> list of results
        self.selected_results: Dict[str, dict] = {}  # ticker -> selected result
        self.ticker_checkboxes: Dict[str, QCheckBox] = {}

        self._setup_ui()
        self._load_saved_results()

    def _setup_ui(self):
        """Create the UI layout."""
        layout = QVBoxLayout(self)
        layout.setSpacing(8)

        # Header
        header_layout = QHBoxLayout()
        header_label = QLabel("Portfolio Simulation")
        header_label.setStyleSheet("font-size: 16px; font-weight: bold;")
        header_layout.addWidget(header_label)
        header_layout.addStretch()

        self.refresh_btn = QPushButton("Refresh")
        self.refresh_btn.clicked.connect(self._refresh_all)
        header_layout.addWidget(self.refresh_btn)

        self.calculate_btn = QPushButton("Calculate Portfolio")
        self.calculate_btn.setObjectName("primary")
        self.calculate_btn.clicked.connect(self._calculate_portfolio)
        header_layout.addWidget(self.calculate_btn)

        layout.addLayout(header_layout)

        # Main splitter - left (selection), right (results)
        main_splitter = QSplitter(Qt.Horizontal)

        # Left panel - ticker selection
        selection_frame = QFrame()
        selection_layout = QVBoxLayout(selection_frame)
        selection_layout.setContentsMargins(0, 0, 0, 0)

        selection_label = QLabel("Select Tickers")
        selection_label.setStyleSheet("font-weight: bold; font-size: 13px;")
        selection_layout.addWidget(selection_label)

        # Scroll area for checkboxes
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        self.checkbox_container = QWidget()
        self.checkbox_layout = QVBoxLayout(self.checkbox_container)
        self.checkbox_layout.setSpacing(4)
        self.checkbox_layout.addStretch()

        scroll.setWidget(self.checkbox_container)
        selection_layout.addWidget(scroll)

        # Select all / none buttons
        btn_layout = QHBoxLayout()
        self.select_all_btn = QPushButton("Select All")
        self.select_all_btn.clicked.connect(self._select_all)
        btn_layout.addWidget(self.select_all_btn)

        self.select_none_btn = QPushButton("Select None")
        self.select_none_btn.clicked.connect(self._select_none)
        btn_layout.addWidget(self.select_none_btn)

        selection_layout.addLayout(btn_layout)

        main_splitter.addWidget(selection_frame)

        # Right panel - results
        results_frame = QFrame()
        results_layout = QVBoxLayout(results_frame)
        results_layout.setContentsMargins(0, 0, 0, 0)

        # Metrics row
        metrics_layout = QHBoxLayout()
        metrics_layout.setSpacing(8)

        self.total_pnl_card = MetricCard("Total P&L", compact=True)
        metrics_layout.addWidget(self.total_pnl_card)

        self.total_trades_card = MetricCard("Total Trades", compact=True)
        metrics_layout.addWidget(self.total_trades_card)

        self.avg_win_rate_card = MetricCard("Avg Win%", compact=True)
        metrics_layout.addWidget(self.avg_win_rate_card)

        self.combined_pf_card = MetricCard("Combined PF", compact=True)
        metrics_layout.addWidget(self.combined_pf_card)

        self.combined_sharpe_card = MetricCard("Combined Sharpe", compact=True)
        metrics_layout.addWidget(self.combined_sharpe_card)

        self.max_drawdown_card = MetricCard("Max DD", compact=True)
        metrics_layout.addWidget(self.max_drawdown_card)

        self.ticker_count_card = MetricCard("Tickers", compact=True)
        metrics_layout.addWidget(self.ticker_count_card)

        metrics_layout.addStretch()
        results_layout.addLayout(metrics_layout)

        # Vertical splitter for chart and table
        chart_splitter = QSplitter(Qt.Vertical)

        # Combined equity chart
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
        self.chart_view.setMinimumHeight(250)
        chart_layout.addWidget(self.chart_view)

        chart_splitter.addWidget(chart_frame)

        # Per-ticker breakdown table
        table_frame = QFrame()
        table_layout = QVBoxLayout(table_frame)
        table_layout.setContentsMargins(0, 0, 0, 0)

        table_label = QLabel("Per-Ticker Breakdown")
        table_label.setStyleSheet("font-weight: bold; font-size: 13px;")
        table_layout.addWidget(table_label)

        self.breakdown_table = QTableWidget()
        self.breakdown_table.setColumnCount(8)
        self.breakdown_table.setHorizontalHeaderLabels([
            "Ticker", "P&L", "Trades", "Win%", "PF", "Sharpe", "Max DD", "Contribution"
        ])
        self.breakdown_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.breakdown_table.setAlternatingRowColors(True)
        self.breakdown_table.verticalHeader().setVisible(False)
        table_layout.addWidget(self.breakdown_table)

        chart_splitter.addWidget(table_frame)
        chart_splitter.setSizes([350, 250])

        results_layout.addWidget(chart_splitter)

        main_splitter.addWidget(results_frame)
        main_splitter.setSizes([200, 800])

        layout.addWidget(main_splitter)

    def _get_storage_path(self) -> Path:
        """Get path to saved results file."""
        return self.output_dir / "saved_tests" / "saved_results.json"

    def _load_saved_results(self):
        """Load saved results from disk."""
        storage_path = self._get_storage_path()

        if not storage_path.exists():
            self.saved_results = {}
            return

        try:
            with open(storage_path, 'r') as f:
                self.saved_results = json.load(f)
            self._update_checkboxes()
        except Exception as e:
            print(f"Error loading saved results: {e}")
            self.saved_results = {}

    def _update_checkboxes(self):
        """Update checkbox list based on saved results."""
        # Clear existing checkboxes
        for cb in self.ticker_checkboxes.values():
            cb.deleteLater()
        self.ticker_checkboxes.clear()

        # Remove stretch
        while self.checkbox_layout.count() > 0:
            item = self.checkbox_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        # Add checkbox for each ticker with saved results
        for ticker in sorted(self.saved_results.keys()):
            results_list = self.saved_results[ticker]
            if not results_list:
                continue

            # Get best result for this ticker
            best_result = max(results_list, key=lambda r: r.get('total_pnl', 0))
            pnl = best_result.get('total_pnl', 0)
            trades = best_result.get('total_trades', 0)

            # Create checkbox with details
            cb = QCheckBox(f"{ticker}")
            cb.setToolTip(
                f"Best P&L: ${pnl:,.0f}\n"
                f"Trades: {trades}\n"
                f"PF: {best_result.get('profit_factor', 0):.2f}\n"
                f"Win Rate: {best_result.get('win_rate', 0):.1f}%"
            )
            cb.setStyleSheet(f"""
                QCheckBox {{
                    padding: 4px;
                    color: {'#00ff00' if pnl > 0 else '#ff4444'};
                }}
            """)
            cb.stateChanged.connect(self._on_checkbox_changed)

            # Store reference to best result
            cb.setProperty("result", best_result)
            cb.setProperty("ticker", ticker)

            self.ticker_checkboxes[ticker] = cb
            self.checkbox_layout.addWidget(cb)

        self.checkbox_layout.addStretch()

    def _on_checkbox_changed(self, state):
        """Handle checkbox state change."""
        sender = self.sender()
        if not sender:
            return

        ticker = sender.property("ticker")
        result = sender.property("result")

        if state == Qt.Checked:
            self.selected_results[ticker] = result
        else:
            if ticker in self.selected_results:
                del self.selected_results[ticker]

        # Update ticker count
        self.ticker_count_card.set_value(str(len(self.selected_results)))

    def _select_all(self):
        """Select all tickers."""
        for cb in self.ticker_checkboxes.values():
            cb.setChecked(True)

    def _select_none(self):
        """Deselect all tickers."""
        for cb in self.ticker_checkboxes.values():
            cb.setChecked(False)

    def _calculate_portfolio(self):
        """Calculate combined portfolio metrics and display."""
        if not self.selected_results:
            self._clear_results()
            return

        # Collect all equity curves and metrics
        all_equity_curves = []
        ticker_data = []

        total_pnl = 0
        total_trades = 0
        total_winners = 0
        total_losers = 0
        gross_profit = 0
        gross_loss = 0

        for ticker, result in self.selected_results.items():
            equity_curve = result.get('equity_curve', [])
            pnl = result.get('total_pnl', 0)
            trades = result.get('total_trades', 0)
            win_rate = result.get('win_rate', 0)
            pf = result.get('profit_factor', 0)
            sharpe = result.get('sharpe_ratio', 0)
            max_dd = result.get('max_drawdown', 0)

            # Track for portfolio calculations
            total_pnl += pnl
            total_trades += trades

            # Estimate winners/losers from win rate
            winners = int(trades * win_rate / 100)
            losers = trades - winners
            total_winners += winners
            total_losers += losers

            # Estimate gross profit/loss from PF
            if pf > 0 and pnl != 0:
                # PF = gross_profit / gross_loss
                # pnl = gross_profit - gross_loss
                # gross_profit = pf * gross_loss
                # pnl = pf * gross_loss - gross_loss = gross_loss * (pf - 1)
                if pf != 1:
                    ticker_gross_loss = abs(pnl / (pf - 1)) if pf > 1 else abs(pnl * pf / (1 - pf))
                    ticker_gross_profit = ticker_gross_loss * pf if pf > 1 else pnl + ticker_gross_loss
                else:
                    ticker_gross_profit = max(0, pnl)
                    ticker_gross_loss = abs(min(0, pnl))
                gross_profit += ticker_gross_profit
                gross_loss += ticker_gross_loss

            all_equity_curves.append({
                'ticker': ticker,
                'curve': equity_curve,
                'pnl': pnl
            })

            ticker_data.append({
                'ticker': ticker,
                'pnl': pnl,
                'trades': trades,
                'win_rate': win_rate,
                'pf': pf,
                'sharpe': sharpe,
                'max_dd': max_dd
            })

        # Calculate combined metrics
        combined_win_rate = (total_winners / total_trades * 100) if total_trades > 0 else 0
        combined_pf = (gross_profit / gross_loss) if gross_loss > 0 else float('inf')

        # Calculate combined equity curve (sum all curves)
        max_len = max(len(ec['curve']) for ec in all_equity_curves) if all_equity_curves else 0
        combined_curve = [0] * max_len

        for ec in all_equity_curves:
            curve = ec['curve']
            for i, val in enumerate(curve):
                if i < max_len:
                    combined_curve[i] += val
            # Extend with final value for shorter curves
            if curve:
                final_val = curve[-1]
                for i in range(len(curve), max_len):
                    combined_curve[i] += final_val

        # Calculate combined drawdown and max DD
        if combined_curve:
            peak = 0
            max_dd = 0
            for val in combined_curve:
                peak = max(peak, val)
                dd = val - peak
                max_dd = min(max_dd, dd)
        else:
            max_dd = 0

        # Calculate combined Sharpe (approximate from individual Sharpes weighted by trades)
        if total_trades > 0:
            weighted_sharpe = sum(
                td['sharpe'] * td['trades'] / total_trades
                for td in ticker_data
            )
        else:
            weighted_sharpe = 0

        # Update metric cards
        pnl_color = "#00ff00" if total_pnl > 0 else "#ff4444"
        self.total_pnl_card.set_value(f"${total_pnl:,.0f}", pnl_color)
        self.total_trades_card.set_value(str(total_trades))
        self.avg_win_rate_card.set_value(f"{combined_win_rate:.1f}%")

        pf_color = "#00ff00" if combined_pf >= 1.5 else "#ffaa00" if combined_pf >= 1.0 else "#ff4444"
        self.combined_pf_card.set_value(f"{combined_pf:.2f}" if combined_pf < float('inf') else "N/A", pf_color)

        sharpe_color = "#00ff00" if weighted_sharpe >= 1.0 else "#ffaa00" if weighted_sharpe >= 0.5 else "#ff4444"
        self.combined_sharpe_card.set_value(f"{weighted_sharpe:.2f}", sharpe_color)

        self.max_drawdown_card.set_value(f"${max_dd:,.0f}", "#ff4444")
        self.ticker_count_card.set_value(str(len(self.selected_results)))

        # Update chart
        self._update_chart(all_equity_curves, combined_curve, total_pnl)

        # Update breakdown table
        self._update_breakdown_table(ticker_data, total_pnl)

    def _update_chart(self, equity_curves: List[dict], combined_curve: List[float], total_pnl: float):
        """Update the equity curve chart."""
        fig = make_subplots(
            rows=2, cols=1,
            row_heights=[0.7, 0.3],
            shared_xaxes=True,
            vertical_spacing=0.05,
            subplot_titles=("Combined Portfolio Equity", "Drawdown")
        )

        # Add individual ticker curves
        colors = ['#2a82da', '#ff4444', '#00ff00', '#ffaa00', '#aa00ff',
                  '#00aaff', '#ff00aa', '#aaff00', '#ff8800', '#8800ff']

        for i, ec in enumerate(equity_curves):
            ticker = ec['ticker']
            curve = ec['curve']
            color = colors[i % len(colors)]

            if curve:
                fig.add_trace(
                    go.Scatter(
                        x=list(range(len(curve))),
                        y=curve,
                        mode='lines',
                        name=ticker,
                        line=dict(color=color, width=1),
                        opacity=0.5
                    ),
                    row=1, col=1
                )

        # Add combined curve (bold)
        if combined_curve:
            is_profitable = total_pnl >= 0
            combined_color = '#00ff00' if is_profitable else '#ff4444'

            fig.add_trace(
                go.Scatter(
                    x=list(range(len(combined_curve))),
                    y=combined_curve,
                    mode='lines',
                    name='Combined',
                    line=dict(color=combined_color, width=3),
                    fill='tozeroy',
                    fillcolor=f'rgba({"0, 255, 0" if is_profitable else "255, 68, 68"}, 0.1)'
                ),
                row=1, col=1
            )

            # Calculate and plot drawdown
            peak = 0
            drawdown = []
            for val in combined_curve:
                peak = max(peak, val)
                dd = val - peak
                drawdown.append(dd)

            fig.add_trace(
                go.Scatter(
                    x=list(range(len(drawdown))),
                    y=drawdown,
                    mode='lines',
                    name='Drawdown',
                    line=dict(color='#ff4444', width=1),
                    fill='tozeroy',
                    fillcolor='rgba(255, 68, 68, 0.2)'
                ),
                row=2, col=1
            )

        # Add zero line
        fig.add_hline(y=0, line_dash="dash", line_color="#555555", line_width=1, row=1, col=1)

        # Update layout
        fig.update_layout(
            height=350,
            margin=dict(l=60, r=20, t=40, b=20),
            paper_bgcolor='#1e1e1e',
            plot_bgcolor='#252525',
            font=dict(color='#cccccc', size=10),
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1,
                font=dict(size=9)
            ),
            title=dict(
                text=f"Portfolio: {len(equity_curves)} Tickers | Total: ${total_pnl:,.0f}",
                font=dict(size=12, color='#aaaaaa'),
                x=0.5
            )
        )

        fig.update_xaxes(gridcolor='#333333', showgrid=True)
        fig.update_yaxes(gridcolor='#333333', showgrid=True, tickformat='$,.0f', row=1, col=1)
        fig.update_yaxes(gridcolor='#333333', showgrid=True, tickformat='$,.0f', row=2, col=1)

        # Render to HTML
        html = fig.to_html(include_plotlyjs='cdn', full_html=True)
        html = html.replace('<body>', '<body style="background-color: #1e1e1e; margin: 0; padding: 0;">')
        self.chart_view.setHtml(html)

    def _update_breakdown_table(self, ticker_data: List[dict], total_pnl: float):
        """Update the per-ticker breakdown table."""
        # Sort by P&L descending
        ticker_data.sort(key=lambda x: x['pnl'], reverse=True)

        self.breakdown_table.setRowCount(len(ticker_data))

        for row, td in enumerate(ticker_data):
            ticker = td['ticker']
            pnl = td['pnl']
            trades = td['trades']
            win_rate = td['win_rate']
            pf = td['pf']
            sharpe = td['sharpe']
            max_dd = td['max_dd']
            contribution = (pnl / total_pnl * 100) if total_pnl != 0 else 0

            items = [
                ticker,
                f"${pnl:,.0f}",
                str(trades),
                f"{win_rate:.1f}%",
                f"{pf:.2f}",
                f"{sharpe:.2f}",
                f"${max_dd:,.0f}",
                f"{contribution:.1f}%"
            ]

            for col, value in enumerate(items):
                item = QTableWidgetItem(value)
                item.setFlags(item.flags() & ~Qt.ItemIsEditable)

                # Color P&L column
                if col == 1:
                    item.setForeground(QColor("#00ff00") if pnl > 0 else QColor("#ff4444"))

                # Color contribution
                if col == 7:
                    item.setForeground(QColor("#00ff00") if contribution > 0 else QColor("#ff4444"))

                self.breakdown_table.setItem(row, col, item)

    def _clear_results(self):
        """Clear all results display."""
        self.total_pnl_card.set_value("--")
        self.total_trades_card.set_value("--")
        self.avg_win_rate_card.set_value("--")
        self.combined_pf_card.set_value("--")
        self.combined_sharpe_card.set_value("--")
        self.max_drawdown_card.set_value("--")
        self.ticker_count_card.set_value("0")
        self.breakdown_table.setRowCount(0)
        self.chart_view.setHtml("")

    def _refresh_all(self):
        """Refresh saved results from disk."""
        self._load_saved_results()
        self.selected_results.clear()
        self._clear_results()

    def set_output_dir(self, path: str):
        """Update output directory."""
        self.output_dir = Path(path)
        self._refresh_all()
