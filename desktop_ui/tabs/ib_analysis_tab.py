"""
IB Analysis Tab - Analyze performance by IB characteristics.
"""

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QLabel, QGroupBox, QTableWidget, QTableWidgetItem,
    QHeaderView, QSplitter, QFrame
)
from PySide6.QtCore import Qt
from PySide6.QtGui import QColor

import pyqtgraph as pg


class IBAnalysisTab(QWidget):
    """Tab for analyzing performance by IB characteristics."""

    def __init__(self):
        super().__init__()
        self.trades = []
        self._setup_ui()

    def _setup_ui(self):
        """Create the UI layout."""
        layout = QVBoxLayout(self)
        layout.setSpacing(12)

        # Info label
        info_label = QLabel(
            "Analyze trade performance bucketed by Initial Balance (IB) characteristics. "
            "Run a backtest first to populate this analysis."
        )
        info_label.setWordWrap(True)
        info_label.setStyleSheet("color: #888888; padding: 8px;")
        layout.addWidget(info_label)

        # Main splitter
        splitter = QSplitter(Qt.Horizontal)

        # Left side - IB Size analysis
        size_frame = QFrame()
        size_layout = QVBoxLayout(size_frame)
        size_layout.setContentsMargins(0, 0, 0, 0)

        size_label = QLabel("Performance by IB Size")
        size_label.setStyleSheet("font-weight: bold; font-size: 14px;")
        size_layout.addWidget(size_label)

        self.size_table = QTableWidget()
        self.size_table.setColumnCount(6)
        self.size_table.setHorizontalHeaderLabels([
            "IB Size Bucket", "Trades", "Win Rate", "Avg P&L", "Total P&L", "Profit Factor"
        ])
        self.size_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.size_table.setAlternatingRowColors(True)
        size_layout.addWidget(self.size_table)

        # Size chart
        self.size_chart = pg.PlotWidget()
        self.size_chart.setBackground('#1e1e1e')
        self.size_chart.showGrid(x=True, y=True, alpha=0.3)
        self.size_chart.setLabel('left', 'Win Rate %')
        self.size_chart.setLabel('bottom', 'IB Size Bucket')
        self.size_chart.setMaximumHeight(200)
        size_layout.addWidget(self.size_chart)

        splitter.addWidget(size_frame)

        # Right side - Day of Week analysis
        dow_frame = QFrame()
        dow_layout = QVBoxLayout(dow_frame)
        dow_layout.setContentsMargins(0, 0, 0, 0)

        dow_label = QLabel("Performance by Day of Week")
        dow_label.setStyleSheet("font-weight: bold; font-size: 14px;")
        dow_layout.addWidget(dow_label)

        self.dow_table = QTableWidget()
        self.dow_table.setColumnCount(6)
        self.dow_table.setHorizontalHeaderLabels([
            "Day", "Trades", "Win Rate", "Avg P&L", "Total P&L", "Profit Factor"
        ])
        self.dow_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.dow_table.setAlternatingRowColors(True)
        dow_layout.addWidget(self.dow_table)

        # DOW chart
        self.dow_chart = pg.PlotWidget()
        self.dow_chart.setBackground('#1e1e1e')
        self.dow_chart.showGrid(x=True, y=True, alpha=0.3)
        self.dow_chart.setLabel('left', 'Total P&L')
        self.dow_chart.setLabel('bottom', 'Day')
        self.dow_chart.setMaximumHeight(200)
        dow_layout.addWidget(self.dow_chart)

        splitter.addWidget(dow_frame)

        layout.addWidget(splitter)

        # Time of day analysis
        time_group = QGroupBox("Performance by Entry Time")
        time_layout = QVBoxLayout(time_group)

        self.time_table = QTableWidget()
        self.time_table.setColumnCount(6)
        self.time_table.setHorizontalHeaderLabels([
            "Time Bucket", "Trades", "Win Rate", "Avg P&L", "Total P&L", "Profit Factor"
        ])
        self.time_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.time_table.setAlternatingRowColors(True)
        self.time_table.setMaximumHeight(200)
        time_layout.addWidget(self.time_table)

        layout.addWidget(time_group)

    def load_trades(self, trades: list):
        """Load trades and compute analysis."""
        self.trades = trades

        if not trades:
            self._clear_analysis()
            return

        self._analyze_by_size()
        self._analyze_by_dow()
        self._analyze_by_time()

    def _clear_analysis(self):
        """Clear all analysis tables and charts."""
        self.size_table.setRowCount(0)
        self.dow_table.setRowCount(0)
        self.time_table.setRowCount(0)
        self.size_chart.clear()
        self.dow_chart.clear()

    def _analyze_by_size(self):
        """Analyze performance by IB size buckets."""
        # Define buckets (as percentage of price)
        buckets = [
            ("Small (< 0.5%)", lambda r: r < 0.5),
            ("Medium (0.5-1%)", lambda r: 0.5 <= r < 1.0),
            ("Large (1-2%)", lambda r: 1.0 <= r < 2.0),
            ("Very Large (> 2%)", lambda r: r >= 2.0)
        ]

        self.size_table.setRowCount(len(buckets))
        bucket_data = []

        for row, (name, condition) in enumerate(buckets):
            # Filter trades by IB size
            # Note: This requires IB range info to be stored in trades
            # For now, we'll use a placeholder based on pnl_pct as a proxy
            bucket_trades = [
                t for t in self.trades
                if condition(abs(t.pnl_pct) if hasattr(t, 'pnl_pct') else 0)
            ]

            stats = self._compute_stats(bucket_trades)
            bucket_data.append((name, stats))

            self._fill_stats_row(self.size_table, row, name, stats)

        # Update chart
        self._update_size_chart(bucket_data)

    def _analyze_by_dow(self):
        """Analyze performance by day of week."""
        days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
        self.dow_table.setRowCount(5)
        day_data = []

        for row, day_name in enumerate(days):
            day_num = row  # Monday = 0
            day_trades = [
                t for t in self.trades
                if t.entry_time.weekday() == day_num
            ]

            stats = self._compute_stats(day_trades)
            day_data.append((day_name, stats))

            self._fill_stats_row(self.dow_table, row, day_name, stats)

        # Update chart
        self._update_dow_chart(day_data)

    def _analyze_by_time(self):
        """Analyze performance by entry time."""
        time_buckets = [
            ("09:30-10:00", 9.5, 10.0),
            ("10:00-11:00", 10.0, 11.0),
            ("11:00-12:00", 11.0, 12.0),
            ("12:00-13:00", 12.0, 13.0),
            ("13:00-14:00", 13.0, 14.0),
            ("14:00-15:00", 14.0, 15.0),
            ("15:00-16:00", 15.0, 16.0)
        ]

        self.time_table.setRowCount(len(time_buckets))

        for row, (name, start, end) in enumerate(time_buckets):
            bucket_trades = [
                t for t in self.trades
                if start <= (t.entry_time.hour + t.entry_time.minute / 60) < end
            ]

            stats = self._compute_stats(bucket_trades)
            self._fill_stats_row(self.time_table, row, name, stats)

    def _compute_stats(self, trades: list) -> dict:
        """Compute statistics for a group of trades."""
        if not trades:
            return {
                'count': 0,
                'win_rate': 0,
                'avg_pnl': 0,
                'total_pnl': 0,
                'profit_factor': 0
            }

        winners = [t for t in trades if t.pnl >= 0]
        losers = [t for t in trades if t.pnl < 0]

        total_pnl = sum(t.pnl for t in trades)
        gross_profit = sum(t.pnl for t in winners) if winners else 0
        gross_loss = abs(sum(t.pnl for t in losers)) if losers else 0

        return {
            'count': len(trades),
            'win_rate': (len(winners) / len(trades) * 100) if trades else 0,
            'avg_pnl': total_pnl / len(trades) if trades else 0,
            'total_pnl': total_pnl,
            'profit_factor': (gross_profit / gross_loss) if gross_loss > 0 else float('inf')
        }

    def _fill_stats_row(self, table: QTableWidget, row: int, name: str, stats: dict):
        """Fill a table row with statistics."""
        table.setItem(row, 0, QTableWidgetItem(name))
        table.setItem(row, 1, QTableWidgetItem(str(stats['count'])))

        # Win rate with color
        wr_item = QTableWidgetItem(f"{stats['win_rate']:.1f}%")
        if stats['win_rate'] >= 50:
            wr_item.setForeground(QColor("#00ff00"))
        elif stats['win_rate'] >= 40:
            wr_item.setForeground(QColor("#ffaa00"))
        else:
            wr_item.setForeground(QColor("#ff4444"))
        table.setItem(row, 2, wr_item)

        # Avg P&L with color
        avg_item = QTableWidgetItem(f"${stats['avg_pnl']:.2f}")
        avg_item.setForeground(
            QColor("#00ff00") if stats['avg_pnl'] >= 0 else QColor("#ff4444")
        )
        table.setItem(row, 3, avg_item)

        # Total P&L with color
        total_item = QTableWidgetItem(f"${stats['total_pnl']:.2f}")
        total_item.setForeground(
            QColor("#00ff00") if stats['total_pnl'] >= 0 else QColor("#ff4444")
        )
        table.setItem(row, 4, total_item)

        # Profit factor
        pf = stats['profit_factor']
        pf_str = f"{pf:.2f}" if pf != float('inf') else "∞"
        pf_item = QTableWidgetItem(pf_str)
        if pf >= 1.5:
            pf_item.setForeground(QColor("#00ff00"))
        elif pf >= 1.0:
            pf_item.setForeground(QColor("#ffaa00"))
        else:
            pf_item.setForeground(QColor("#ff4444"))
        table.setItem(row, 5, pf_item)

    def _update_size_chart(self, bucket_data: list):
        """Update the IB size chart."""
        self.size_chart.clear()

        if not bucket_data:
            return

        x = list(range(len(bucket_data)))
        y = [stats['win_rate'] for _, stats in bucket_data]

        # Create bar chart
        bar = pg.BarGraphItem(
            x=x, height=y, width=0.6,
            brush='#2a82da'
        )
        self.size_chart.addItem(bar)

        # Add 50% reference line
        self.size_chart.addLine(y=50, pen=pg.mkPen('r', style=Qt.DashLine))

        # Set x-axis labels
        ax = self.size_chart.getAxis('bottom')
        ax.setTicks([[(i, name[:8]) for i, (name, _) in enumerate(bucket_data)]])

    def _update_dow_chart(self, day_data: list):
        """Update the day of week chart."""
        self.dow_chart.clear()

        if not day_data:
            return

        x = list(range(len(day_data)))
        y = [stats['total_pnl'] for _, stats in day_data]

        # Create bar chart with colors based on P&L
        for i, pnl in enumerate(y):
            color = '#00ff00' if pnl >= 0 else '#ff4444'
            bar = pg.BarGraphItem(
                x=[i], height=[pnl], width=0.6,
                brush=color
            )
            self.dow_chart.addItem(bar)

        # Add zero reference line
        self.dow_chart.addLine(y=0, pen=pg.mkPen('w', style=Qt.DashLine))

        # Set x-axis labels
        ax = self.dow_chart.getAxis('bottom')
        ax.setTicks([[(i, name[:3]) for i, (name, _) in enumerate(day_data)]])
