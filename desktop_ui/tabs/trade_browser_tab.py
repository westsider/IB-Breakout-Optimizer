"""
Trade Browser Tab - Browse and filter trades with candlestick charts.
"""

from pathlib import Path
from datetime import datetime, timedelta

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QLabel, QComboBox, QCheckBox, QPushButton, QGroupBox,
    QTableWidget, QTableWidgetItem, QHeaderView, QSplitter,
    QFrame, QDateEdit
)
from PySide6.QtCore import Qt, QDate
from PySide6.QtGui import QColor

import pyqtgraph as pg
import numpy as np


class CandlestickItem(pg.GraphicsObject):
    """Custom candlestick graphics item for pyqtgraph."""

    def __init__(self, data):
        """
        data: list of tuples (index, open, high, low, close)
        """
        pg.GraphicsObject.__init__(self)
        self.data = data
        self.generatePicture()

    def generatePicture(self):
        from PySide6.QtGui import QPainter, QPicture
        from PySide6.QtCore import QRectF

        self.picture = QPicture()
        p = QPainter(self.picture)

        w = 0.4  # Half width of candle body

        for (i, open_price, high, low, close) in self.data:
            if close >= open_price:
                # Bullish - green
                p.setPen(pg.mkPen('#00aa00', width=1))
                p.setBrush(pg.mkBrush('#00aa00'))
            else:
                # Bearish - red
                p.setPen(pg.mkPen('#cc0000', width=1))
                p.setBrush(pg.mkBrush('#cc0000'))

            # Draw wick (high-low line)
            p.drawLine(pg.QtCore.QPointF(i, low), pg.QtCore.QPointF(i, high))

            # Draw body
            p.drawRect(QRectF(i - w, open_price, w * 2, close - open_price))

        p.end()

    def paint(self, p, *args):
        p.drawPicture(0, 0, self.picture)

    def boundingRect(self):
        from PySide6.QtCore import QRectF
        return QRectF(self.picture.boundingRect())


class TradeBrowserTab(QWidget):
    """Tab for browsing and filtering trades with candlestick charts."""

    def __init__(self, data_dir: str = ""):
        super().__init__()
        self.trades = []
        self.data_dir = data_dir
        self.bar_cache = {}  # Cache loaded bar data by ticker
        self._setup_ui()

    def set_data_dir(self, path: str):
        """Update data directory."""
        self.data_dir = path
        self.bar_cache = {}  # Clear cache when data dir changes

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

        self.chart_label = QLabel("Trade Chart (select a trade above)")
        self.chart_label.setStyleSheet("font-weight: bold;")
        chart_layout.addWidget(self.chart_label)

        # PyQtGraph chart
        self.chart_widget = pg.PlotWidget()
        self.chart_widget.setBackground('#1e1e1e')
        self.chart_widget.showGrid(x=True, y=True, alpha=0.3)
        self.chart_widget.setLabel('left', 'Price')
        self.chart_widget.setLabel('bottom', 'Time')
        chart_layout.addWidget(self.chart_widget)

        splitter.addWidget(chart_frame)
        splitter.setSizes([350, 350])

        layout.addWidget(splitter)

    def load_trades(self, trades: list):
        """Load trades into the browser."""
        self.trades = trades

        # Set date range based on actual trade dates
        if trades:
            min_date = min(t.entry_time.date() for t in trades if t.entry_time)
            max_date = max(t.entry_time.date() for t in trades if t.entry_time)
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
            if t.entry_time and date_from <= t.entry_time.date() <= date_to
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
                trade.entry_time.strftime('%Y-%m-%d %H:%M') if trade.entry_time else ""
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

    def _load_bars_for_date(self, ticker: str, trade_date) -> list:
        """Load bar data for a specific date."""
        if not self.data_dir:
            return []

        # Check cache first
        cache_key = f"{ticker}_{trade_date}"
        if cache_key in self.bar_cache:
            return self.bar_cache[cache_key]

        try:
            from data.data_loader import DataLoader

            # Find data file
            data_path = Path(self.data_dir)
            data_file = None

            for f in data_path.iterdir():
                if f.is_file() and ticker.upper() in f.name.upper():
                    if f.suffix.lower() in ['.txt', '.csv']:
                        data_file = f
                        if '_NT' in f.name.upper():
                            break  # Prefer NT format

            if not data_file:
                return []

            # Load data
            loader = DataLoader(str(data_path))
            df = loader.load_auto_detect(str(data_file), ticker)

            # Filter to trade date
            df['date'] = df['timestamp'].dt.date
            day_df = df[df['date'] == trade_date].copy()

            if day_df.empty:
                return []

            # Convert to list of tuples for candlestick
            bars = []
            for idx, (_, row) in enumerate(day_df.iterrows()):
                bars.append({
                    'index': idx,
                    'timestamp': row['timestamp'],
                    'open': row['open'],
                    'high': row['high'],
                    'low': row['low'],
                    'close': row['close']
                })

            # Cache result
            self.bar_cache[cache_key] = bars
            return bars

        except Exception as e:
            print(f"Error loading bars: {e}")
            return []

    def _show_trade_chart(self, trade):
        """Display a candlestick chart for the selected trade."""
        self.chart_widget.clear()

        if not trade.entry_time:
            return

        # Update chart label
        direction = "LONG" if trade.direction.value == "long" else "SHORT"
        pnl_str = f"${trade.pnl:+.2f}" if trade.pnl >= 0 else f"-${abs(trade.pnl):.2f}"
        self.chart_label.setText(
            f"{trade.ticker} | {trade.entry_time.strftime('%Y-%m-%d')} | "
            f"{direction} | {pnl_str}"
        )

        # Load bar data for the trade date
        bars = self._load_bars_for_date(trade.ticker, trade.entry_time.date())

        if not bars:
            # Fall back to simple display if no bar data
            self._show_simple_chart(trade)
            return

        # Find entry and exit bar indices
        entry_idx = None
        exit_idx = None

        for bar in bars:
            if entry_idx is None and bar['timestamp'] >= trade.entry_time:
                entry_idx = bar['index']
            if trade.exit_time and exit_idx is None and bar['timestamp'] >= trade.exit_time:
                exit_idx = bar['index']

        # If exact match not found, use closest
        if entry_idx is None and bars:
            entry_idx = 0
        if exit_idx is None and trade.exit_time and bars:
            exit_idx = len(bars) - 1

        # Create candlestick data
        candle_data = [
            (bar['index'], bar['open'], bar['high'], bar['low'], bar['close'])
            for bar in bars
        ]

        # Add candlesticks
        candlestick = CandlestickItem(candle_data)
        self.chart_widget.addItem(candlestick)

        # Determine y-axis range
        all_highs = [bar['high'] for bar in bars]
        all_lows = [bar['low'] for bar in bars]
        y_min = min(all_lows) * 0.999
        y_max = max(all_highs) * 1.001

        # Include IB levels in y-axis range calculation first
        ib_start_idx = None
        ib_end_idx = None
        if trade.ib and trade.ib.ib_high > 0 and trade.ib.ib_low < float('inf'):
            y_min = min(y_min, trade.ib.ib_low * 0.998)
            y_max = max(y_max, trade.ib.ib_high * 1.002)

            # Find IB period bar indices
            if trade.ib.session_start and trade.ib.ib_end_time:
                for bar in bars:
                    if ib_start_idx is None and bar['timestamp'] >= trade.ib.session_start:
                        ib_start_idx = bar['index']
                    if ib_end_idx is None and bar['timestamp'] >= trade.ib.ib_end_time:
                        ib_end_idx = bar['index']
                        break

            # Default to start of day if not found
            if ib_start_idx is None:
                ib_start_idx = 0
            if ib_end_idx is None and bars:
                ib_end_idx = min(30, len(bars) - 1)

        # Draw entry marker
        if entry_idx is not None:
            entry_color = '#00ff00' if trade.direction.value == 'long' else '#ff4444'

            # Entry arrow: up triangle for long (buying), down triangle for short (selling)
            # pyqtgraph: 't1' = triangle pointing up, 't' = triangle pointing down
            entry_symbol = 't1' if trade.direction.value == 'long' else 't'
            self.chart_widget.plot(
                [entry_idx], [trade.entry_price],
                pen=None, symbol=entry_symbol,
                symbolPen=entry_color,
                symbolBrush=entry_color,
                symbolSize=14
            )

            # Entry label
            entry_text = pg.TextItem(
                f"Entry: ${trade.entry_price:.2f}",
                color=entry_color, anchor=(0, 1)
            )
            entry_text.setPos(entry_idx + 1, trade.entry_price)
            self.chart_widget.addItem(entry_text)

        # Draw exit marker
        if exit_idx is not None and trade.exit_price:
            exit_color = '#00ff00' if trade.pnl >= 0 else '#ff4444'

            # Exit arrow: down triangle for long (selling to close), up triangle for short (buying to cover)
            exit_symbol = 't' if trade.direction.value == 'long' else 't1'
            self.chart_widget.plot(
                [exit_idx], [trade.exit_price],
                pen=None, symbol=exit_symbol,
                symbolPen=exit_color,
                symbolBrush=exit_color,
                symbolSize=14
            )

            # Exit label
            exit_text = pg.TextItem(
                f"Exit: ${trade.exit_price:.2f}\nP&L: ${trade.pnl:+.2f}",
                color=exit_color, anchor=(1, 0)
            )
            exit_text.setPos(exit_idx - 1, trade.exit_price)
            self.chart_widget.addItem(exit_text)

            # Shade the trade region
            if entry_idx is not None:
                region = pg.LinearRegionItem(
                    values=[entry_idx, exit_idx],
                    orientation='vertical',
                    brush=pg.mkBrush(exit_color + '20'),  # Semi-transparent
                    pen=pg.mkPen(None),
                    movable=False
                )
                self.chart_widget.addItem(region)

        # Set axis ranges
        self.chart_widget.setYRange(y_min, y_max, padding=0.02)
        self.chart_widget.setXRange(0, len(bars), padding=0.02)

        # Draw IB High/Low lines and IB period AFTER setting axis range
        if trade.ib and trade.ib.ib_high > 0 and trade.ib.ib_low < float('inf'):
            ib = trade.ib

            # IB High line (white dashed) - use InfiniteLine for full-width line
            ib_high_line = pg.InfiniteLine(
                pos=ib.ib_high,
                angle=0,  # horizontal
                pen=pg.mkPen('#ffffff', width=1.5, style=Qt.DashLine)
            )
            self.chart_widget.addItem(ib_high_line)

            # IB High label
            ib_high_text = pg.TextItem(
                f"IB High ${ib.ib_high:.2f}",
                color='#ffffff', anchor=(1, 1)
            )
            ib_high_text.setPos(len(bars) - 2, ib.ib_high)
            self.chart_widget.addItem(ib_high_text)

            # IB Low line (white dashed)
            ib_low_line = pg.InfiniteLine(
                pos=ib.ib_low,
                angle=0,  # horizontal
                pen=pg.mkPen('#ffffff', width=1.5, style=Qt.DashLine)
            )
            self.chart_widget.addItem(ib_low_line)

            # IB Low label
            ib_low_text = pg.TextItem(
                f"IB Low ${ib.ib_low:.2f}",
                color='#ffffff', anchor=(1, 0)
            )
            ib_low_text.setPos(len(bars) - 2, ib.ib_low)
            self.chart_widget.addItem(ib_low_text)

            # Shade the IB period (light cyan)
            if ib_start_idx is not None and ib_end_idx is not None:
                ib_region = pg.LinearRegionItem(
                    values=[ib_start_idx, ib_end_idx],
                    orientation='vertical',
                    brush=pg.mkBrush('#00ffff20'),  # Light cyan, more visible
                    pen=pg.mkPen('#00ffff', width=1, style=Qt.DotLine),
                    movable=False
                )
                self.chart_widget.addItem(ib_region)

                # IB period label at top
                ib_label = pg.TextItem(
                    "IB Period",
                    color='#00ffff', anchor=(0.5, 1)
                )
                ib_label.setPos((ib_start_idx + ib_end_idx) / 2, y_max * 0.999)
                self.chart_widget.addItem(ib_label)

        # Add time axis labels (show every N bars)
        if bars:
            # Create custom axis labels
            time_labels = {}
            step = max(1, len(bars) // 10)  # Show ~10 labels
            for i in range(0, len(bars), step):
                time_labels[i] = bars[i]['timestamp'].strftime('%H:%M')

            axis = self.chart_widget.getAxis('bottom')
            axis.setTicks([list(time_labels.items())])

    def _show_simple_chart(self, trade):
        """Fallback simple chart when bar data not available."""
        entry_bar = 0
        exit_bar = trade.bars_held if trade.bars_held > 0 else 10

        # Entry arrow: up triangle for long (buying), down triangle for short (selling)
        # pyqtgraph: 't1' = triangle pointing up, 't' = triangle pointing down
        entry_symbol = 't1' if trade.direction.value == 'long' else 't'
        self.chart_widget.plot(
            [entry_bar], [trade.entry_price],
            pen=None, symbol=entry_symbol,
            symbolPen='g' if trade.direction.value == 'long' else 'r',
            symbolBrush='g' if trade.direction.value == 'long' else 'r',
            symbolSize=15,
            name="Entry"
        )

        if trade.exit_price:
            # Exit arrow: down triangle for long (selling), up triangle for short (buying to cover)
            exit_symbol = 't' if trade.direction.value == 'long' else 't1'
            self.chart_widget.plot(
                [exit_bar], [trade.exit_price],
                pen=None, symbol=exit_symbol,
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

        # Note about missing data
        note = pg.TextItem(
            "Bar data not available - showing simplified view",
            color='#888888', anchor=(0.5, 0)
        )
        note.setPos(exit_bar / 2, trade.entry_price * 1.01)
        self.chart_widget.addItem(note)
