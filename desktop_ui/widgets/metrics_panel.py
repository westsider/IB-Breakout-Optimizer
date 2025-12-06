"""
Metrics Panel - Display performance metrics in a grid layout.
"""

from PySide6.QtWidgets import (
    QWidget, QGridLayout, QVBoxLayout, QLabel, QFrame
)
from PySide6.QtCore import Qt


class MetricCard(QFrame):
    """A single metric display card."""

    def __init__(self, label: str, value: str = "--", parent=None, compact: bool = False):
        super().__init__(parent)
        self.compact = compact

        if compact:
            # Compact inline style: "Label: Value"
            self.setStyleSheet("""
                MetricCard {
                    background-color: #2a2a2a;
                    border: 1px solid #3a3a3a;
                    border-radius: 4px;
                    padding: 2px 6px;
                }
            """)
            from PySide6.QtWidgets import QHBoxLayout
            layout = QHBoxLayout(self)
            layout.setSpacing(4)
            layout.setContentsMargins(6, 2, 6, 2)

            # Label first
            self.label = QLabel(f"{label}:")
            self.label.setStyleSheet("color: #888888; font-size: 11px;")
            layout.addWidget(self.label)

            # Value inline
            self.value_label = QLabel(value)
            self.value_label.setStyleSheet("font-size: 12px; font-weight: bold;")
            layout.addWidget(self.value_label)
        else:
            # Standard vertical card style
            self.setStyleSheet("""
                MetricCard {
                    background-color: #252525;
                    border: 1px solid #333333;
                    border-radius: 6px;
                    padding: 8px;
                }
            """)

            layout = QVBoxLayout(self)
            layout.setSpacing(4)
            layout.setContentsMargins(12, 8, 12, 8)

            # Value label
            self.value_label = QLabel(value)
            self.value_label.setObjectName("metric-value")
            self.value_label.setAlignment(Qt.AlignCenter)
            layout.addWidget(self.value_label)

            # Description label
            self.label = QLabel(label)
            self.label.setObjectName("metric-label")
            self.label.setAlignment(Qt.AlignCenter)
            layout.addWidget(self.label)

    def set_value(self, value: str, color: str = None):
        """Update the metric value."""
        self.value_label.setText(value)
        if self.compact:
            if color:
                self.value_label.setStyleSheet(f"color: {color}; font-size: 12px; font-weight: bold;")
            else:
                self.value_label.setStyleSheet("font-size: 12px; font-weight: bold;")
        else:
            if color:
                self.value_label.setStyleSheet(f"color: {color}; font-size: 18px; font-weight: bold;")
            else:
                self.value_label.setStyleSheet("font-size: 18px; font-weight: bold;")


class MetricsPanel(QWidget):
    """Panel displaying all performance metrics."""

    def __init__(self, parent=None):
        super().__init__(parent)

        self._setup_ui()

    def _setup_ui(self):
        """Create the metrics grid."""
        layout = QGridLayout(self)
        layout.setSpacing(12)

        # Row 1
        self.total_pnl = MetricCard("Total P&L")
        layout.addWidget(self.total_pnl, 0, 0)

        self.total_trades = MetricCard("Total Trades")
        layout.addWidget(self.total_trades, 0, 1)

        self.win_rate = MetricCard("Win Rate")
        layout.addWidget(self.win_rate, 0, 2)

        self.profit_factor = MetricCard("Profit Factor")
        layout.addWidget(self.profit_factor, 0, 3)

        self.sharpe = MetricCard("Sharpe Ratio")
        layout.addWidget(self.sharpe, 0, 4)

        # Row 2
        self.max_drawdown = MetricCard("Max Drawdown")
        layout.addWidget(self.max_drawdown, 1, 0)

        self.avg_trade = MetricCard("Avg Trade")
        layout.addWidget(self.avg_trade, 1, 1)

        self.avg_winner = MetricCard("Avg Winner")
        layout.addWidget(self.avg_winner, 1, 2)

        self.avg_loser = MetricCard("Avg Loser")
        layout.addWidget(self.avg_loser, 1, 3)

        self.sortino = MetricCard("Sortino Ratio")
        layout.addWidget(self.sortino, 1, 4)

    def update_metrics(self, metrics):
        """Update all metrics from a PerformanceMetrics object."""
        # Total P&L
        pnl = metrics.total_net_profit
        color = "#00ff00" if pnl >= 0 else "#ff4444"
        self.total_pnl.set_value(f"${pnl:,.2f}", color)

        # Total trades
        self.total_trades.set_value(str(metrics.total_trades))

        # Win rate
        wr = metrics.percent_profitable
        wr_color = "#00ff00" if wr >= 50 else "#ffaa00" if wr >= 40 else "#ff4444"
        self.win_rate.set_value(f"{wr:.1f}%", wr_color)

        # Profit factor
        pf = metrics.profit_factor
        pf_color = "#00ff00" if pf >= 1.5 else "#ffaa00" if pf >= 1.0 else "#ff4444"
        self.profit_factor.set_value(f"{pf:.2f}", pf_color)

        # Sharpe ratio
        sr = metrics.sharpe_ratio
        sr_color = "#00ff00" if sr >= 1.0 else "#ffaa00" if sr >= 0.5 else "#ff4444"
        self.sharpe.set_value(f"{sr:.2f}", sr_color)

        # Max drawdown
        dd = metrics.max_drawdown
        dd_color = "#ff4444" if dd < -1000 else "#ffaa00" if dd < -500 else "#00ff00"
        self.max_drawdown.set_value(f"${dd:,.2f}", dd_color)

        # Average trade
        avg = metrics.avg_trade
        avg_color = "#00ff00" if avg >= 0 else "#ff4444"
        self.avg_trade.set_value(f"${avg:.2f}", avg_color)

        # Average winner
        self.avg_winner.set_value(f"${metrics.avg_winning_trade:.2f}", "#00ff00")

        # Average loser
        self.avg_loser.set_value(f"${metrics.avg_losing_trade:.2f}", "#ff4444")

        # Sortino ratio
        sortino = metrics.sortino_ratio
        sortino_color = "#00ff00" if sortino >= 1.0 else "#ffaa00" if sortino >= 0.5 else "#ff4444"
        self.sortino.set_value(f"{sortino:.2f}", sortino_color)

    def clear(self):
        """Reset all metrics to default."""
        for card in [self.total_pnl, self.total_trades, self.win_rate,
                     self.profit_factor, self.sharpe, self.max_drawdown,
                     self.avg_trade, self.avg_winner, self.avg_loser, self.sortino]:
            card.set_value("--")
