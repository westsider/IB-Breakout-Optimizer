"""
Trade Browser Tab - Browse and filter trades with charts.
"""

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QLabel, QComboBox, QCheckBox, QPushButton, QGroupBox,
    QTableWidget, QTableWidgetItem, QHeaderView, QSplitter,
    QFrame, QDateEdit
)
from PySide6.QtCore import Qt, QDate
from PySide6.QtGui import QColor

import pyqtgraph as pg


class TradeBrowserTab(QWidget):
    """Tab for browsing and filtering trades."""

    def __init__(self):
        super().__init__()
        self.trades = []
        self._setup_ui()

    def _setup_ui(self):
        """Create the UI layout."""
        layout = QVBoxLayout(self)
        layout.setSpacing(12)

        # Filters section
        filter_group = QGroupBox("Filters")
        filter_layout = QHBoxLayout(filter_group)

        # Direction filter
        filter_layout.addWidget(QLabel("Direction:"))
        self.direction_filter = QComboBox()
        self.direction_filter.addItems(["All", "Long", "Short"])
        self.direction_filter.currentTextChanged.connect(self._apply_filters)
        filter_layout.addWidget(self.direction_filter)

        # Result filter
        filter_layout.addWidget(QLabel("Result:"))
        self.result_filter = QComboBox()
        self.result_filter.addItems(["All", "Winners", "Losers"])
        self.result_filter.currentTextChanged.connect(self._apply_filters)
        filter_layout.addWidget(self.result_filter)

        # Exit reason filter
        filter_layout.addWidget(QLabel("Exit:"))
        self.exit_filter = QComboBox()
        self.exit_filter.addItems(["All", "Target", "Stop", "EOD", "Trailing", "Break-Even"])
        self.exit_filter.currentTextChanged.connect(self._apply_filters)
        filter_layout.addWidget(self.exit_filter)

        # Date range
        filter_layout.addWidget(QLabel("From:"))
        self.date_from = QDateEdit()
        self.date_from.setCalendarPopup(True)
        self.date_from.setDate(QDate.currentDate().addYears(-1))
        self.date_from.dateChanged.connect(self._apply_filters)
        filter_layout.addWidget(self.date_from)

        filter_layout.addWidget(QLabel("To:"))
        self.date_to = QDateEdit()
        self.date_to.setCalendarPopup(True)
        self.date_to.setDate(QDate.currentDate())
        self.date_to.dateChanged.connect(self._apply_filters)
        filter_layout.addWidget(self.date_to)

        filter_layout.addStretch()

        # Reset button
        reset_btn = QPushButton("Reset Filters")
        reset_btn.clicked.connect(self._reset_filters)
        filter_layout.addWidget(reset_btn)

        layout.addWidget(filter_group)

        # Main splitter
        splitter = QSplitter(Qt.Vertical)

        # Trade table
        table_frame = QFrame()
        table_layout = QVBoxLayout(table_frame)
        table_layout.setContentsMargins(0, 0, 0, 0)

        self.trade_count_label = QLabel("0 trades")
        self.trade_count_label.setStyleSheet("font-weight: bold;")
        table_layout.addWidget(self.trade_count_label)

        self.trade_table = QTableWidget()
        self.trade_table.setColumnCount(9)
        self.trade_table.setHorizontalHeaderLabels([
            "Entry Time", "Exit Time", "Direction", "Entry", "Exit",
            "P&L", "P&L %", "Exit Reason", "Bars"
        ])
        self.trade_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.trade_table.setAlternatingRowColors(True)
        self.trade_table.setSelectionBehavior(QTableWidget.SelectRows)
        self.trade_table.itemSelectionChanged.connect(self._on_trade_selected)
        table_layout.addWidget(self.trade_table)

        splitter.addWidget(table_frame)

        # Chart area
        chart_frame = QFrame()
        chart_layout = QVBoxLayout(chart_frame)
        chart_layout.setContentsMargins(0, 0, 0, 0)

        chart_label = QLabel("Trade Chart (select a trade above)")
        chart_label.setStyleSheet("font-weight: bold;")
        chart_layout.addWidget(chart_label)

        # PyQtGraph chart
        self.chart_widget = pg.PlotWidget()
        self.chart_widget.setBackground('#1e1e1e')
        self.chart_widget.showGrid(x=True, y=True, alpha=0.3)
        self.chart_widget.setLabel('left', 'Price')
        self.chart_widget.setLabel('bottom', 'Bar')
        chart_layout.addWidget(self.chart_widget)

        splitter.addWidget(chart_frame)
        splitter.setSizes([400, 300])

        layout.addWidget(splitter)

    def load_trades(self, trades: list):
        """Load trades into the browser."""
        self.trades = trades

        # Set date range based on actual trade dates
        if trades:
            min_date = min(t.entry_time.date() for t in trades)
            max_date = max(t.entry_time.date() for t in trades)
            self.date_from.setDate(QDate(min_date.year, min_date.month, min_date.day))
            self.date_to.setDate(QDate(max_date.year, max_date.month, max_date.day))

        self._apply_filters()

    def _apply_filters(self):
        """Apply filters and update the table."""
        if not self.trades:
            self.trade_table.setRowCount(0)
            self.trade_count_label.setText("0 trades")
            return

        filtered = self.trades.copy()

        # Direction filter
        direction = self.direction_filter.currentText()
        if direction != "All":
            dir_value = "long" if direction == "Long" else "short"
            filtered = [t for t in filtered if t.direction.value == dir_value]

        # Result filter
        result = self.result_filter.currentText()
        if result == "Winners":
            filtered = [t for t in filtered if t.pnl >= 0]
        elif result == "Losers":
            filtered = [t for t in filtered if t.pnl < 0]

        # Exit reason filter
        exit_type = self.exit_filter.currentText()
        if exit_type != "All":
            exit_map = {
                "Target": "profit_target",
                "Stop": "stop_loss",
                "EOD": "eod_exit",
                "Trailing": "trailing_stop",
                "Break-Even": "break_even"
            }
            exit_value = exit_map.get(exit_type, "")
            filtered = [
                t for t in filtered
                if t.exit_reason and t.exit_reason.value == exit_value
            ]

        # Date filter
        date_from = self.date_from.date().toPython()
        date_to = self.date_to.date().toPython()
        filtered = [
            t for t in filtered
            if date_from <= t.entry_time.date() <= date_to
        ]

        # Update table
        self._populate_table(filtered)
        self.trade_count_label.setText(f"{len(filtered)} trades")

    def _populate_table(self, trades: list):
        """Fill the table with trades."""
        self.trade_table.setRowCount(len(trades))

        for row, trade in enumerate(trades):
            # Store trade reference in first item
            entry_item = QTableWidgetItem(
                trade.entry_time.strftime('%Y-%m-%d %H:%M')
            )
            entry_item.setData(Qt.UserRole, trade)
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

    def _reset_filters(self):
        """Reset all filters to default."""
        self.direction_filter.setCurrentText("All")
        self.result_filter.setCurrentText("All")
        self.exit_filter.setCurrentText("All")
        self.date_from.setDate(QDate.currentDate().addYears(-1))
        self.date_to.setDate(QDate.currentDate())

    def _on_trade_selected(self):
        """Handle trade selection - show chart."""
        selected = self.trade_table.selectedItems()
        if not selected:
            return

        # Get trade from first column
        row = selected[0].row()
        trade_item = self.trade_table.item(row, 0)
        trade = trade_item.data(Qt.UserRole)

        if trade:
            self._show_trade_chart(trade)

    def _show_trade_chart(self, trade):
        """Display a chart for the selected trade."""
        self.chart_widget.clear()

        # For now, show a simple representation
        # In a full implementation, we'd load the actual bar data

        entry_bar = 0
        exit_bar = trade.bars_held

        # Plot entry and exit points
        self.chart_widget.plot(
            [entry_bar], [trade.entry_price],
            pen=None, symbol='t',
            symbolPen='g' if trade.direction.value == 'long' else 'r',
            symbolBrush='g' if trade.direction.value == 'long' else 'r',
            symbolSize=15,
            name="Entry"
        )

        if trade.exit_price:
            self.chart_widget.plot(
                [exit_bar], [trade.exit_price],
                pen=None, symbol='t1',
                symbolPen='g' if trade.pnl >= 0 else 'r',
                symbolBrush='g' if trade.pnl >= 0 else 'r',
                symbolSize=15,
                name="Exit"
            )

            # Draw line between entry and exit
            color = 'g' if trade.pnl >= 0 else 'r'
            self.chart_widget.plot(
                [entry_bar, exit_bar],
                [trade.entry_price, trade.exit_price],
                pen=pg.mkPen(color, width=2, style=Qt.DashLine)
            )

        # Add labels
        entry_text = pg.TextItem(
            f"Entry: ${trade.entry_price:.2f}",
            color='w', anchor=(0, 1)
        )
        entry_text.setPos(entry_bar, trade.entry_price)
        self.chart_widget.addItem(entry_text)

        if trade.exit_price:
            exit_text = pg.TextItem(
                f"Exit: ${trade.exit_price:.2f}\nP&L: ${trade.pnl:.2f}",
                color='g' if trade.pnl >= 0 else 'r',
                anchor=(1, 0)
            )
            exit_text.setPos(exit_bar, trade.exit_price)
            self.chart_widget.addItem(exit_text)
