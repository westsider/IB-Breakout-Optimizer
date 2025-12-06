"""
Equity Curve Tab - Display equity curve and drawdown charts.
"""

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QGroupBox,
    QSplitter, QFrame
)
from PySide6.QtCore import Qt

import pyqtgraph as pg
import numpy as np

from desktop_ui.widgets.metrics_panel import MetricCard


class EquityCurveTab(QWidget):
    """Tab for displaying equity curve and drawdown analysis."""

    def __init__(self):
        super().__init__()
        self.trades = []
        self._setup_ui()

    def _setup_ui(self):
        """Create the UI layout."""
        layout = QVBoxLayout(self)
        layout.setSpacing(12)

        # Summary metrics
        metrics_layout = QHBoxLayout()

        self.total_pnl_card = MetricCard("Total P&L")
        metrics_layout.addWidget(self.total_pnl_card)

        self.max_drawdown_card = MetricCard("Max Drawdown")
        metrics_layout.addWidget(self.max_drawdown_card)

        self.max_runup_card = MetricCard("Max Run-up")
        metrics_layout.addWidget(self.max_runup_card)

        self.recovery_factor_card = MetricCard("Recovery Factor")
        metrics_layout.addWidget(self.recovery_factor_card)

        self.avg_drawdown_card = MetricCard("Avg Drawdown")
        metrics_layout.addWidget(self.avg_drawdown_card)

        layout.addLayout(metrics_layout)

        # Charts splitter
        splitter = QSplitter(Qt.Vertical)

        # Equity curve chart
        equity_frame = QFrame()
        equity_layout = QVBoxLayout(equity_frame)
        equity_layout.setContentsMargins(0, 0, 0, 0)

        equity_label = QLabel("Equity Curve")
        equity_label.setStyleSheet("font-weight: bold; font-size: 14px;")
        equity_layout.addWidget(equity_label)

        self.equity_chart = pg.PlotWidget()
        self.equity_chart.setBackground('#1e1e1e')
        self.equity_chart.showGrid(x=True, y=True, alpha=0.3)
        self.equity_chart.setLabel('left', 'Equity ($)')
        self.equity_chart.setLabel('bottom', 'Trade #')
        self.equity_chart.addLegend()
        equity_layout.addWidget(self.equity_chart)

        splitter.addWidget(equity_frame)

        # Drawdown chart
        dd_frame = QFrame()
        dd_layout = QVBoxLayout(dd_frame)
        dd_layout.setContentsMargins(0, 0, 0, 0)

        dd_label = QLabel("Drawdown")
        dd_label.setStyleSheet("font-weight: bold; font-size: 14px;")
        dd_layout.addWidget(dd_label)

        self.drawdown_chart = pg.PlotWidget()
        self.drawdown_chart.setBackground('#1e1e1e')
        self.drawdown_chart.showGrid(x=True, y=True, alpha=0.3)
        self.drawdown_chart.setLabel('left', 'Drawdown ($)')
        self.drawdown_chart.setLabel('bottom', 'Trade #')
        dd_layout.addWidget(self.drawdown_chart)

        splitter.addWidget(dd_frame)

        # Monthly returns chart
        monthly_frame = QFrame()
        monthly_layout = QVBoxLayout(monthly_frame)
        monthly_layout.setContentsMargins(0, 0, 0, 0)

        monthly_label = QLabel("Monthly Returns")
        monthly_label.setStyleSheet("font-weight: bold; font-size: 14px;")
        monthly_layout.addWidget(monthly_label)

        self.monthly_chart = pg.PlotWidget()
        self.monthly_chart.setBackground('#1e1e1e')
        self.monthly_chart.showGrid(x=True, y=True, alpha=0.3)
        self.monthly_chart.setLabel('left', 'P&L ($)')
        self.monthly_chart.setLabel('bottom', 'Month')
        monthly_layout.addWidget(self.monthly_chart)

        splitter.addWidget(monthly_frame)

        splitter.setSizes([300, 200, 200])
        layout.addWidget(splitter)

    def load_trades(self, trades: list):
        """Load trades and update charts."""
        self.trades = trades

        if not trades:
            self._clear_charts()
            return

        self._update_equity_curve()
        self._update_drawdown_chart()
        self._update_monthly_chart()
        self._update_metrics()

    def _clear_charts(self):
        """Clear all charts."""
        self.equity_chart.clear()
        self.drawdown_chart.clear()
        self.monthly_chart.clear()

        for card in [self.total_pnl_card, self.max_drawdown_card,
                     self.max_runup_card, self.recovery_factor_card,
                     self.avg_drawdown_card]:
            card.set_value("--")

    def _update_equity_curve(self):
        """Update the equity curve chart."""
        self.equity_chart.clear()

        if not self.trades:
            return

        # Calculate cumulative P&L
        pnls = [t.pnl for t in self.trades]
        cumulative = np.cumsum(pnls)

        # Add starting point at 0
        x = list(range(len(cumulative) + 1))
        y = [0] + list(cumulative)

        # Plot equity curve
        pen = pg.mkPen(color='#2a82da', width=2)
        self.equity_chart.plot(x, y, pen=pen, name='Equity')

        # Add zero line
        self.equity_chart.addLine(y=0, pen=pg.mkPen('w', style=Qt.DashLine, width=1))

        # Fill above/below zero
        # Green fill for positive equity
        pos_y = np.maximum(y, 0)
        fill_pos = pg.FillBetweenItem(
            pg.PlotDataItem(x, pos_y),
            pg.PlotDataItem(x, [0] * len(x)),
            brush=pg.mkBrush(0, 255, 0, 30)
        )
        self.equity_chart.addItem(fill_pos)

        # Red fill for negative equity
        neg_y = np.minimum(y, 0)
        fill_neg = pg.FillBetweenItem(
            pg.PlotDataItem(x, neg_y),
            pg.PlotDataItem(x, [0] * len(x)),
            brush=pg.mkBrush(255, 0, 0, 30)
        )
        self.equity_chart.addItem(fill_neg)

    def _update_drawdown_chart(self):
        """Update the drawdown chart."""
        self.drawdown_chart.clear()

        if not self.trades:
            return

        # Calculate drawdown
        pnls = [t.pnl for t in self.trades]
        cumulative = np.cumsum(pnls)

        # Running maximum
        running_max = np.maximum.accumulate(cumulative)
        drawdown = cumulative - running_max

        # Add starting point
        x = list(range(len(drawdown) + 1))
        y = [0] + list(drawdown)

        # Plot drawdown (as negative values)
        pen = pg.mkPen(color='#ff4444', width=2)
        self.drawdown_chart.plot(x, y, pen=pen, fillLevel=0,
                                  brush=pg.mkBrush(255, 68, 68, 50))

        # Add zero line
        self.drawdown_chart.addLine(y=0, pen=pg.mkPen('w', style=Qt.DashLine, width=1))

    def _update_monthly_chart(self):
        """Update the monthly returns chart."""
        self.monthly_chart.clear()

        if not self.trades:
            return

        # Group trades by month
        monthly_pnl = {}
        for trade in self.trades:
            month_key = trade.entry_time.strftime('%Y-%m')
            if month_key not in monthly_pnl:
                monthly_pnl[month_key] = 0
            monthly_pnl[month_key] += trade.pnl

        if not monthly_pnl:
            return

        # Sort by month
        sorted_months = sorted(monthly_pnl.keys())
        x = list(range(len(sorted_months)))
        y = [monthly_pnl[m] for m in sorted_months]

        # Create bar chart with colors based on P&L
        for i, pnl in enumerate(y):
            color = '#00ff00' if pnl >= 0 else '#ff4444'
            bar = pg.BarGraphItem(
                x=[i], height=[pnl], width=0.6,
                brush=color
            )
            self.monthly_chart.addItem(bar)

        # Add zero line
        self.monthly_chart.addLine(y=0, pen=pg.mkPen('w', style=Qt.DashLine, width=1))

        # Set x-axis labels
        ax = self.monthly_chart.getAxis('bottom')
        ax.setTicks([[(i, m[-5:]) for i, m in enumerate(sorted_months)]])

    def _update_metrics(self):
        """Update the summary metrics."""
        if not self.trades:
            return

        pnls = [t.pnl for t in self.trades]
        cumulative = np.cumsum(pnls)

        # Total P&L
        total_pnl = sum(pnls)
        color = "#00ff00" if total_pnl >= 0 else "#ff4444"
        self.total_pnl_card.set_value(f"${total_pnl:,.2f}", color)

        # Max drawdown
        running_max = np.maximum.accumulate(cumulative)
        drawdowns = cumulative - running_max
        max_dd = min(drawdowns) if len(drawdowns) > 0 else 0
        self.max_drawdown_card.set_value(f"${max_dd:,.2f}", "#ff4444")

        # Max run-up
        max_runup = max(cumulative) if len(cumulative) > 0 else 0
        self.max_runup_card.set_value(f"${max_runup:,.2f}", "#00ff00")

        # Recovery factor (total profit / max drawdown)
        if max_dd != 0:
            recovery = abs(total_pnl / max_dd)
            rf_color = "#00ff00" if recovery >= 2 else "#ffaa00" if recovery >= 1 else "#ff4444"
            self.recovery_factor_card.set_value(f"{recovery:.2f}", rf_color)
        else:
            self.recovery_factor_card.set_value("∞", "#00ff00")

        # Average drawdown
        negative_dd = drawdowns[drawdowns < 0]
        if len(negative_dd) > 0:
            avg_dd = np.mean(negative_dd)
            self.avg_drawdown_card.set_value(f"${avg_dd:,.2f}", "#ffaa00")
        else:
            self.avg_drawdown_card.set_value("$0.00", "#00ff00")
