"""
Filter Analysis Tab for IB Breakout Optimizer.

Provides interactive visualization of all filter calculations:
- Gap % filter (today's open vs yesterday's close)
- Prior days trend filter (bullish/bearish count)
- Daily range % filter (volatility)

Features:
- Full-width candlestick chart with daily bars
- Horizontal scrollbar for date navigation
- Crosshair with real-time data display
- Visual annotations showing filter values
"""

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QComboBox,
    QScrollBar, QFrame, QSplitter, QGroupBox
)
from PySide6.QtCore import Qt, Signal, QTimer
from PySide6.QtGui import QPainter, QColor, QPen, QBrush, QFont, QFontMetrics
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Optional, Dict, List, Tuple


class CandlestickChartWidget(QWidget):
    """
    Custom widget for drawing candlestick charts with filter annotations.

    Features:
    - Daily candlesticks
    - Gap visualization (shaded area between prior close and today's open)
    - Prior days trend indicators (colored dots showing bullish/bearish days)
    - Daily range annotations
    - Crosshair following mouse
    - Data panel at top showing crosshair values
    """

    crosshair_moved = Signal(dict)  # Emits data at crosshair position

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMouseTracking(True)
        self.setMinimumHeight(400)

        # Data
        self.daily_data: pd.DataFrame = pd.DataFrame()
        self.filter_data: Dict = {}
        self.gap_percentiles: Tuple[float, float] = (0, 0)
        self.range_percentiles: Tuple[float, float, float, float] = (0, 0, 0, 0)

        # View settings
        self.visible_days = 10  # Number of days visible at once
        self.scroll_offset = 0  # Start index
        self.bar_width = 60  # Pixels per day
        self.margin_left = 80
        self.margin_right = 20
        self.margin_top = 120  # Space for data panel
        self.margin_bottom = 80  # Space for date labels and trend dots

        # Crosshair
        self.mouse_x = 0
        self.mouse_y = 0
        self.crosshair_enabled = True

        # Colors
        self.bg_color = QColor("#1e1e1e")
        self.grid_color = QColor("#333333")
        self.text_color = QColor("#cccccc")
        self.bullish_color = QColor("#00cc66")
        self.bearish_color = QColor("#ff4444")
        self.gap_up_color = QColor(0, 200, 100, 60)
        self.gap_down_color = QColor(255, 68, 68, 60)
        self.crosshair_color = QColor("#ffaa00")
        self.range_color = QColor("#4488ff")

    def set_data(self, daily_df: pd.DataFrame, filter_data: Dict,
                 gap_percentiles: Tuple, range_percentiles: Tuple):
        """Set the data to display."""
        self.daily_data = daily_df.copy()
        self.filter_data = filter_data
        self.gap_percentiles = gap_percentiles
        self.range_percentiles = range_percentiles

        # Scroll to end (most recent data)
        if len(self.daily_data) > self.visible_days:
            self.scroll_offset = len(self.daily_data) - self.visible_days
        else:
            self.scroll_offset = 0

        self.update()

    def set_scroll_offset(self, offset: int):
        """Set scroll position."""
        max_offset = max(0, len(self.daily_data) - self.visible_days)
        self.scroll_offset = max(0, min(offset, max_offset))
        self.update()

    def get_max_scroll(self) -> int:
        """Get maximum scroll value."""
        return max(0, len(self.daily_data) - self.visible_days)

    def wheelEvent(self, event):
        """Handle mouse wheel for zooming."""
        delta = event.angleDelta().y()
        if delta > 0:
            # Zoom in (fewer days visible)
            self.visible_days = max(5, self.visible_days - 1)
        else:
            # Zoom out (more days visible)
            self.visible_days = min(30, self.visible_days + 1)

        # Recalculate bar width
        chart_width = self.width() - self.margin_left - self.margin_right
        self.bar_width = chart_width / self.visible_days

        # Adjust scroll to keep current view centered
        max_offset = max(0, len(self.daily_data) - self.visible_days)
        self.scroll_offset = min(self.scroll_offset, max_offset)

        self.update()

    def mouseMoveEvent(self, event):
        """Track mouse for crosshair."""
        self.mouse_x = event.pos().x()
        self.mouse_y = event.pos().y()

        # Emit data at crosshair position
        data = self._get_data_at_crosshair()
        if data:
            self.crosshair_moved.emit(data)

        self.update()

    def _get_data_at_crosshair(self) -> Optional[Dict]:
        """Get data at current crosshair position."""
        if self.daily_data.empty:
            return None

        # Calculate which day the crosshair is over
        chart_width = self.width() - self.margin_left - self.margin_right
        self.bar_width = chart_width / self.visible_days

        rel_x = self.mouse_x - self.margin_left
        if rel_x < 0 or rel_x > chart_width:
            return None

        day_index = int(rel_x / self.bar_width) + self.scroll_offset

        if day_index < 0 or day_index >= len(self.daily_data):
            return None

        row = self.daily_data.iloc[day_index]
        date = row['date']

        # Get filter data for this day
        fd = self.filter_data.get(date, {})

        return {
            'date': date,
            'open': row['open'],
            'high': row['high'],
            'low': row['low'],
            'close': row['close'],
            'gap_pct': fd.get('gap_pct'),
            'gap_points': row.get('gap_points'),
            'bullish_count': fd.get('bullish_count'),
            'trend_lookback': fd.get('trend_lookback', 3),
            'avg_range_pct': fd.get('avg_range_pct'),
            'range_pct': row.get('range_pct'),
            'prior_close': row.get('prior_close'),
            'bullish': row.get('bullish')
        }

    def paintEvent(self, event):
        """Draw the chart."""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        # Background
        painter.fillRect(self.rect(), self.bg_color)

        if self.daily_data.empty:
            painter.setPen(self.text_color)
            painter.drawText(self.rect(), Qt.AlignCenter, "No data loaded.\nSelect a ticker above.")
            return

        # Calculate dimensions
        chart_width = self.width() - self.margin_left - self.margin_right
        chart_height = self.height() - self.margin_top - self.margin_bottom
        self.bar_width = chart_width / self.visible_days

        # Get visible data
        start_idx = self.scroll_offset
        end_idx = min(start_idx + self.visible_days, len(self.daily_data))
        visible_data = self.daily_data.iloc[start_idx:end_idx]

        if visible_data.empty:
            return

        # Calculate price range for visible data
        price_min = visible_data['low'].min() * 0.995
        price_max = visible_data['high'].max() * 1.005
        price_range = price_max - price_min

        if price_range == 0:
            price_range = 1

        def price_to_y(price):
            return self.margin_top + chart_height - (price - price_min) / price_range * chart_height

        def x_for_day(idx):
            return self.margin_left + (idx - start_idx) * self.bar_width + self.bar_width / 2

        # Draw grid
        self._draw_grid(painter, price_min, price_max, chart_height)

        # Draw gap areas first (behind candles)
        self._draw_gaps(painter, visible_data, start_idx, price_to_y, x_for_day)

        # Draw candlesticks
        self._draw_candles(painter, visible_data, start_idx, price_to_y, x_for_day)

        # Draw trend dots below chart
        self._draw_trend_dots(painter, visible_data, start_idx, x_for_day, chart_height)

        # Draw range bars
        self._draw_range_bars(painter, visible_data, start_idx, x_for_day, chart_height)

        # Draw date labels
        self._draw_date_labels(painter, visible_data, start_idx, x_for_day, chart_height)

        # Draw crosshair
        if self.crosshair_enabled:
            self._draw_crosshair(painter, price_min, price_range, chart_height)

        # Draw data panel at top
        self._draw_data_panel(painter)

    def _draw_grid(self, painter, price_min, price_max, chart_height):
        """Draw price grid lines."""
        painter.setPen(QPen(self.grid_color, 1, Qt.DotLine))

        # Horizontal grid lines (price levels)
        num_lines = 6
        for i in range(num_lines + 1):
            y = self.margin_top + i * chart_height / num_lines
            painter.drawLine(self.margin_left, int(y),
                           self.width() - self.margin_right, int(y))

            # Price label
            price = price_max - i * (price_max - price_min) / num_lines
            painter.setPen(self.text_color)
            painter.drawText(5, int(y) + 5, f"${price:.2f}")
            painter.setPen(QPen(self.grid_color, 1, Qt.DotLine))

    def _draw_gaps(self, painter, visible_data, start_idx, price_to_y, x_for_day):
        """Draw gap areas between prior close and today's open."""
        for i, (idx, row) in enumerate(visible_data.iterrows()):
            prior_close = row.get('prior_close')
            if pd.isna(prior_close):
                continue

            open_price = row['open']
            gap = open_price - prior_close

            if abs(gap) < 0.01:
                continue

            x = x_for_day(start_idx + i)
            y1 = price_to_y(prior_close)
            y2 = price_to_y(open_price)

            # Draw shaded gap area
            if gap > 0:
                painter.fillRect(int(x - self.bar_width/3), int(y2),
                               int(self.bar_width * 2/3), int(y1 - y2),
                               self.gap_up_color)
            else:
                painter.fillRect(int(x - self.bar_width/3), int(y1),
                               int(self.bar_width * 2/3), int(y2 - y1),
                               self.gap_down_color)

            # Draw gap annotation
            gap_pct = row.get('gap_pct', 0)
            if not pd.isna(gap_pct):
                mid_y = (y1 + y2) / 2
                painter.setPen(self.text_color)
                font = painter.font()
                font.setPointSize(8)
                painter.setFont(font)

                gap_text = f"{gap_pct:+.2f}%"
                painter.drawText(int(x - 25), int(mid_y) + 4, gap_text)

    def _draw_candles(self, painter, visible_data, start_idx, price_to_y, x_for_day):
        """Draw candlestick bars."""
        candle_width = max(4, self.bar_width * 0.6)

        for i, (idx, row) in enumerate(visible_data.iterrows()):
            x = x_for_day(start_idx + i)

            o, h, l, c = row['open'], row['high'], row['low'], row['close']

            is_bullish = c >= o
            color = self.bullish_color if is_bullish else self.bearish_color

            # Draw wick
            painter.setPen(QPen(color, 1))
            painter.drawLine(int(x), int(price_to_y(h)), int(x), int(price_to_y(l)))

            # Draw body
            body_top = price_to_y(max(o, c))
            body_bottom = price_to_y(min(o, c))
            body_height = max(1, body_bottom - body_top)

            if is_bullish:
                painter.fillRect(int(x - candle_width/2), int(body_top),
                               int(candle_width), int(body_height),
                               color)
            else:
                painter.fillRect(int(x - candle_width/2), int(body_top),
                               int(candle_width), int(body_height),
                               color)

    def _draw_trend_dots(self, painter, visible_data, start_idx, x_for_day, chart_height):
        """Draw trend indicator dots below chart."""
        dot_y = self.margin_top + chart_height + 20
        dot_radius = 6

        # Draw label
        painter.setPen(self.text_color)
        font = painter.font()
        font.setPointSize(9)
        painter.setFont(font)
        painter.drawText(5, int(dot_y) + 4, "Trend:")

        for i, (idx, row) in enumerate(visible_data.iterrows()):
            x = x_for_day(start_idx + i)

            # Get trend data for this day
            date = row['date']
            fd = self.filter_data.get(date, {})
            bullish_count = fd.get('bullish_count')
            lookback = fd.get('trend_lookback', 3)

            if bullish_count is None:
                continue

            bearish_count = lookback - bullish_count

            # Draw small dots for each prior day
            for j in range(lookback):
                dot_x = x - (lookback - 1) * 5 / 2 + j * 5
                # Determine if this day was bullish
                # We show the count visually: first N dots are bearish, rest bullish
                if j < bearish_count:
                    painter.setBrush(self.bearish_color)
                else:
                    painter.setBrush(self.bullish_color)

                painter.setPen(Qt.NoPen)
                painter.drawEllipse(int(dot_x) - 3, int(dot_y) - 3, 6, 6)

            # Draw count label
            painter.setPen(self.text_color)
            count_text = f"{int(bullish_count)}/{lookback}"
            painter.drawText(int(x) - 10, int(dot_y) + 15, count_text)

    def _draw_range_bars(self, painter, visible_data, start_idx, x_for_day, chart_height):
        """Draw daily range % indicator bars."""
        bar_y = self.margin_top + chart_height + 45
        bar_height = 15
        max_bar_width = self.bar_width * 0.7

        # Draw label
        painter.setPen(self.text_color)
        font = painter.font()
        font.setPointSize(9)
        painter.setFont(font)
        painter.drawText(5, int(bar_y) + 12, "Range%:")

        # Get max range for scaling
        ranges = []
        for idx, row in visible_data.iterrows():
            date = row['date']
            fd = self.filter_data.get(date, {})
            avg_range = fd.get('avg_range_pct')
            if avg_range is not None:
                ranges.append(avg_range)

        if not ranges:
            return

        max_range = max(ranges) if ranges else 1

        for i, (idx, row) in enumerate(visible_data.iterrows()):
            x = x_for_day(start_idx + i)

            date = row['date']
            fd = self.filter_data.get(date, {})
            avg_range = fd.get('avg_range_pct')

            if avg_range is None:
                continue

            # Scale bar width by range value
            bar_w = (avg_range / max_range) * max_bar_width if max_range > 0 else 0

            # Color based on percentile
            range_p50 = self.range_percentiles[1] if len(self.range_percentiles) > 1 else 5
            if avg_range > range_p50:
                color = QColor("#ff8844")  # High volatility
            else:
                color = self.range_color  # Normal/low volatility

            painter.fillRect(int(x - bar_w/2), int(bar_y), int(bar_w), bar_height, color)

            # Draw value
            painter.setPen(self.text_color)
            painter.drawText(int(x) - 12, int(bar_y) + bar_height + 12, f"{avg_range:.1f}%")

    def _draw_date_labels(self, painter, visible_data, start_idx, x_for_day, chart_height):
        """Draw date labels at bottom."""
        painter.setPen(self.text_color)
        font = painter.font()
        font.setPointSize(8)
        painter.setFont(font)

        label_y = self.height() - 5

        for i, (idx, row) in enumerate(visible_data.iterrows()):
            x = x_for_day(start_idx + i)
            date = row['date']

            if isinstance(date, datetime):
                date_str = date.strftime("%m/%d")
            else:
                date_str = str(date)[-5:]  # MM-DD

            painter.drawText(int(x) - 15, label_y, date_str)

    def _draw_crosshair(self, painter, price_min, price_range, chart_height):
        """Draw crosshair at mouse position."""
        if self.mouse_x < self.margin_left or self.mouse_x > self.width() - self.margin_right:
            return
        if self.mouse_y < self.margin_top or self.mouse_y > self.margin_top + chart_height:
            return

        painter.setPen(QPen(self.crosshair_color, 1, Qt.DashLine))

        # Vertical line
        painter.drawLine(self.mouse_x, self.margin_top,
                        self.mouse_x, self.margin_top + chart_height)

        # Horizontal line
        painter.drawLine(self.margin_left, self.mouse_y,
                        self.width() - self.margin_right, self.mouse_y)

        # Price at crosshair
        price = price_min + (self.margin_top + chart_height - self.mouse_y) / chart_height * price_range
        painter.setPen(self.crosshair_color)
        painter.fillRect(self.width() - self.margin_right, self.mouse_y - 10,
                        self.margin_right, 20, self.bg_color)
        painter.drawText(self.width() - self.margin_right + 2, self.mouse_y + 4, f"${price:.2f}")

    def _draw_data_panel(self, painter):
        """Draw data panel at top showing crosshair values."""
        data = self._get_data_at_crosshair()

        # Panel background
        panel_height = self.margin_top - 10
        painter.fillRect(0, 0, self.width(), panel_height, QColor("#252525"))

        painter.setPen(self.text_color)
        font = painter.font()
        font.setPointSize(11)
        font.setBold(True)
        painter.setFont(font)

        if not data:
            painter.drawText(20, 25, "Hover over a candlestick to see filter data")
            return

        # Row 1: Date and OHLC
        date_str = str(data['date'])
        o, h, l, c = data['open'], data['high'], data['low'], data['close']

        # Change color based on day direction
        if c >= o:
            painter.setPen(self.bullish_color)
        else:
            painter.setPen(self.bearish_color)

        text1 = f"Date: {date_str}   O: ${o:.2f}   H: ${h:.2f}   L: ${l:.2f}   C: ${c:.2f}"
        painter.drawText(20, 25, text1)

        # Row 2: Gap info
        font.setBold(False)
        painter.setFont(font)
        painter.setPen(self.text_color)

        gap_pct = data.get('gap_pct')
        prior_close = data.get('prior_close')

        if gap_pct is not None and prior_close is not None:
            gap_points = o - prior_close
            gap_color = self.bullish_color if gap_pct > 0 else self.bearish_color
            painter.setPen(gap_color)
            text2 = f"GAP: {gap_pct:+.2f}%  ({gap_points:+.2f} pts)   Prior Close: ${prior_close:.2f}"
        else:
            text2 = "GAP: N/A (first day)"

        painter.drawText(20, 50, text2)

        # Row 3: Trend info
        painter.setPen(self.text_color)
        bullish_count = data.get('bullish_count')
        lookback = data.get('trend_lookback', 3)

        if bullish_count is not None:
            bearish_count = lookback - bullish_count
            if bullish_count > bearish_count:
                trend_str = "BULLISH"
                painter.setPen(self.bullish_color)
            elif bearish_count > bullish_count:
                trend_str = "BEARISH"
                painter.setPen(self.bearish_color)
            else:
                trend_str = "NEUTRAL"
                painter.setPen(QColor("#ffaa00"))

            text3 = f"TREND: {trend_str} ({int(bullish_count)}/{lookback} bullish days prior)"
        else:
            text3 = "TREND: N/A (insufficient data)"

        painter.drawText(20, 75, text3)

        # Row 4: Range info - show BOTH prior 5-day average (used for filter) AND today's range
        painter.setPen(self.text_color)
        avg_range = data.get('avg_range_pct')
        day_range = data.get('range_pct')

        if avg_range is not None:
            range_p50 = self.range_percentiles[1] if len(self.range_percentiles) > 1 else 0
            range_p84 = self.range_percentiles[3] if len(self.range_percentiles) > 3 else 0

            # Prior 5-day classification (used by filter)
            if avg_range > range_p84:
                prior_vol_str = "HIGH"
            elif avg_range < range_p50:
                prior_vol_str = "LOW"
            else:
                prior_vol_str = "NORMAL"

            # Today's range classification
            if day_range is not None and day_range > 0:
                if day_range > range_p84:
                    today_vol_str = "HIGH"
                    today_color = QColor("#ff8844")
                elif day_range < range_p50:
                    today_vol_str = "LOW"
                    today_color = self.range_color
                else:
                    today_vol_str = "NORMAL"
                    today_color = self.text_color

                # Color based on prior (that's what filter uses)
                if prior_vol_str == "HIGH":
                    painter.setPen(QColor("#ff8844"))
                elif prior_vol_str == "LOW":
                    painter.setPen(self.range_color)
                else:
                    painter.setPen(self.text_color)

                text4 = f"PRIOR 5-DAY AVG: {prior_vol_str} ({avg_range:.2f}%)   TODAY'S RANGE: {today_vol_str} ({day_range:.2f}%)"
            else:
                text4 = f"PRIOR 5-DAY AVG: {prior_vol_str} ({avg_range:.2f}%)"
        else:
            text4 = "VOLATILITY: N/A (insufficient data)"

        painter.drawText(20, 100, text4)


class FilterAnalysisTab(QWidget):
    """
    Tab for analyzing and visualizing filter calculations.

    Shows an interactive daily chart with:
    - Gap % visualization
    - Prior days trend indicators
    - Daily range % bars
    - Crosshair with detailed data display
    """

    def __init__(self, data_dir: str = "market_data", parent=None):
        super().__init__(parent)
        self.data_dir = data_dir
        self.daily_data = pd.DataFrame()
        self.filter_data = {}

        self._setup_ui()

    def _setup_ui(self):
        """Set up the user interface."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)
        layout.setSpacing(5)

        # Top control bar
        control_bar = QHBoxLayout()

        # Ticker selector
        control_bar.addWidget(QLabel("Ticker:"))
        self.ticker_combo = QComboBox()
        self.ticker_combo.setMinimumWidth(100)
        control_bar.addWidget(self.ticker_combo)

        # Days selector (create BEFORE populating tickers since _load_data needs it)
        control_bar.addWidget(QLabel("   Show:"))
        self.days_combo = QComboBox()
        self.days_combo.addItems(["1 Week", "2 Weeks", "1 Month", "3 Months", "All"])
        self.days_combo.setCurrentText("1 Month")
        self.days_combo.currentTextChanged.connect(self._on_days_changed)
        control_bar.addWidget(self.days_combo)

        control_bar.addStretch()

        # Legend
        legend_label = QLabel(
            "<span style='color:#00cc66'>Green</span>=Bullish  "
            "<span style='color:#ff4444'>Red</span>=Bearish  "
            "<span style='color:#4488ff'>Blue</span>=Range  "
            "<span style='color:#ffaa00'>Orange</span>=Crosshair"
        )
        control_bar.addWidget(legend_label)

        layout.addLayout(control_bar)

        # Chart widget (takes most of space)
        self.chart = CandlestickChartWidget()
        layout.addWidget(self.chart, stretch=1)

        # Horizontal scrollbar
        self.scrollbar = QScrollBar(Qt.Horizontal)
        self.scrollbar.setMinimum(0)
        self.scrollbar.valueChanged.connect(self._on_scroll)
        layout.addWidget(self.scrollbar)

        # Instructions
        instructions = QLabel(
            "Mouse wheel: Zoom in/out   |   Hover: See filter details   |   "
            "Scroll: Navigate dates"
        )
        instructions.setStyleSheet("color: #888888; font-size: 10px;")
        layout.addWidget(instructions)

        # Now populate tickers (AFTER chart is created since _load_data needs it)
        self._populate_tickers()
        self.ticker_combo.currentTextChanged.connect(self._on_ticker_changed)

    def _populate_tickers(self):
        """Find available data files and populate ticker combo."""
        self.ticker_combo.clear()

        data_path = Path(self.data_dir)
        if not data_path.exists():
            return

        tickers = set()
        for f in data_path.iterdir():
            if f.is_file() and f.suffix.lower() in ['.txt', '.csv']:
                # Extract ticker from filename
                name = f.stem.upper()
                for known in ['TSLA', 'QQQ', 'AAPL', 'NVDA', 'MSFT', 'SPY']:
                    if known in name:
                        tickers.add(known)
                        break

        for ticker in sorted(tickers):
            self.ticker_combo.addItem(ticker)

        if tickers:
            self._load_data(list(tickers)[0])

    def _on_ticker_changed(self, ticker: str):
        """Handle ticker selection change."""
        if ticker:
            self._load_data(ticker)

    def _on_days_changed(self, days_text: str):
        """Handle days selection change."""
        self._update_view()

    def _on_scroll(self, value: int):
        """Handle scrollbar movement."""
        self.chart.set_scroll_offset(value)

    def _load_data(self, ticker: str):
        """Load data for the selected ticker."""
        data_path = Path(self.data_dir)

        # Find data file
        data_file = None
        for f in data_path.iterdir():
            if f.is_file() and ticker.upper() in f.name.upper():
                if f.suffix.lower() in ['.txt', '.csv']:
                    data_file = f
                    if '_NT' in f.name.upper():
                        break

        if not data_file:
            return

        try:
            # Load NinjaTrader format data
            df = pd.read_csv(data_file, sep=';', header=None,
                names=['datetime', 'open', 'high', 'low', 'close', 'volume', 'oi'])

            # Parse date
            df['date'] = pd.to_datetime(df['datetime'].astype(str).str[:8], format='%Y%m%d')

            # Aggregate to daily OHLC
            daily = df.groupby('date').agg({
                'open': 'first',
                'high': 'max',
                'low': 'min',
                'close': 'last',
                'volume': 'sum'
            }).reset_index()

            # Calculate filter values
            daily = self._calculate_filter_values(daily)

            self.daily_data = daily
            self._update_view()

        except Exception as e:
            print(f"Error loading data: {e}")

    def _calculate_filter_values(self, daily: pd.DataFrame,
                                  trend_lookback: int = 3,
                                  range_lookback: int = 5) -> pd.DataFrame:
        """Calculate all filter values."""
        # Gap calculation
        daily['prior_close'] = daily['close'].shift(1)
        daily['gap_pct'] = (daily['open'] - daily['prior_close']) / daily['prior_close'] * 100
        daily['gap_points'] = daily['open'] - daily['prior_close']

        # Trend calculation
        daily['bullish'] = (daily['close'] > daily['open']).astype(int)
        daily['bullish_count'] = daily['bullish'].shift(1).rolling(trend_lookback).sum()

        # Range calculation
        daily['range_pct'] = (daily['high'] - daily['low']) / daily['low'] * 100
        daily['avg_range_pct'] = daily['range_pct'].shift(1).rolling(range_lookback).mean()

        # Build filter_data dict
        self.filter_data = {}
        for _, row in daily.iterrows():
            date_key = row['date']
            self.filter_data[date_key] = {
                'gap_pct': row['gap_pct'] if pd.notna(row['gap_pct']) else None,
                'bullish_count': row['bullish_count'] if pd.notna(row['bullish_count']) else None,
                'trend_lookback': trend_lookback,
                'avg_range_pct': row['avg_range_pct'] if pd.notna(row['avg_range_pct']) else None,
            }

        # Calculate percentiles
        gap_values = daily['gap_pct'].dropna()
        if len(gap_values) > 0:
            self.gap_percentiles = (
                float(np.percentile(gap_values, 16)),
                float(np.percentile(gap_values, 84))
            )
        else:
            self.gap_percentiles = (0, 0)

        range_values = daily['avg_range_pct'].dropna()
        if len(range_values) > 0:
            self.range_percentiles = (
                float(np.percentile(range_values, 16)),
                float(np.percentile(range_values, 50)),
                float(np.percentile(range_values, 68)),
                float(np.percentile(range_values, 84))
            )
        else:
            self.range_percentiles = (0, 0, 0, 0)

        return daily

    def _update_view(self):
        """Update the chart with current data and settings."""
        if self.daily_data.empty:
            return

        # Filter by date range
        days_text = self.days_combo.currentText()
        if days_text == "1 Week":
            days = 7
        elif days_text == "2 Weeks":
            days = 14
        elif days_text == "1 Month":
            days = 30
        elif days_text == "3 Months":
            days = 90
        else:
            days = len(self.daily_data)

        # Get last N days
        view_data = self.daily_data.tail(days).copy()

        # Update chart
        self.chart.set_data(view_data, self.filter_data,
                          self.gap_percentiles, self.range_percentiles)

        # Update scrollbar
        max_scroll = self.chart.get_max_scroll()
        self.scrollbar.setMaximum(max_scroll)
        self.scrollbar.setValue(max_scroll)  # Scroll to end
