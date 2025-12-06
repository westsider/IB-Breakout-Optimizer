"""
Monitoring Tab - Continuous learning dashboard for performance tracking.

Displays rolling metrics, regime detection, degradation alerts,
and re-optimization status.
"""

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QLabel, QGroupBox, QTableWidget, QTableWidgetItem,
    QHeaderView, QSplitter, QFrame, QPushButton,
    QProgressBar, QTextEdit, QComboBox, QSpinBox,
    QScrollArea
)
from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtGui import QColor, QFont

import pyqtgraph as pg
from datetime import datetime, timedelta
from typing import List, Dict, Optional

from monitoring import (
    PerformanceMonitor,
    RollingMetrics,
    TradeRecord,
    RegimeDetector,
    MarketRegime,
    DegradationDetector,
    DegradationThresholds,
    AlertType,
    AlertSeverity,
    ReoptimizationTrigger,
    TriggerConfig,
    DataUpdater,
    DataUpdateConfig,
)


class MetricCard(QFrame):
    """A card displaying a single metric with label and value."""

    def __init__(self, title: str, value: str = "-", subtitle: str = ""):
        super().__init__()
        self.setFrameStyle(QFrame.StyledPanel | QFrame.Raised)
        self.setStyleSheet("""
            MetricCard {
                background-color: #2d2d2d;
                border: 1px solid #3d3d3d;
                border-radius: 8px;
                padding: 8px;
            }
        """)

        layout = QVBoxLayout(self)
        layout.setSpacing(4)
        layout.setContentsMargins(12, 8, 12, 8)

        # Title
        self.title_label = QLabel(title)
        self.title_label.setStyleSheet("color: #888888; font-size: 11px;")
        layout.addWidget(self.title_label)

        # Value
        self.value_label = QLabel(value)
        self.value_label.setStyleSheet("font-size: 22px; font-weight: bold;")
        layout.addWidget(self.value_label)

        # Subtitle
        self.subtitle_label = QLabel(subtitle)
        self.subtitle_label.setStyleSheet("color: #666666; font-size: 10px;")
        layout.addWidget(self.subtitle_label)

    def set_value(self, value: str, color: str = "#ffffff"):
        """Update the displayed value."""
        self.value_label.setText(value)
        self.value_label.setStyleSheet(f"font-size: 22px; font-weight: bold; color: {color};")

    def set_subtitle(self, text: str):
        """Update the subtitle."""
        self.subtitle_label.setText(text)


class MonitoringTab(QWidget):
    """Tab for continuous learning and performance monitoring."""

    # Signal when re-optimization is requested
    reoptimization_requested = Signal(str, dict)  # ticker, params

    def __init__(self, data_dir: str):
        super().__init__()
        self.data_dir = data_dir

        # Initialize monitoring components
        self.perf_monitor = PerformanceMonitor(windows=[20, 50, 100])
        self.regime_detector = RegimeDetector()
        self.degradation_detector = DegradationDetector(
            self.perf_monitor,
            self.regime_detector,
            DegradationThresholds(
                min_sharpe=0.5,
                min_win_rate=40.0,
                min_profit_factor=1.0,
                max_drawdown=15.0,
                max_consecutive_losses=5,
            )
        )
        self.trigger = ReoptimizationTrigger(
            self.degradation_detector,
            self.regime_detector,
            TriggerConfig(
                critical_alert_threshold=2,
                sustained_degradation_days=5,
                scheduled_interval_days=30,
            )
        )
        self.data_updater = DataUpdater(DataUpdateConfig(data_dir=data_dir))

        # Track loaded trades
        self.trades: List = []

        self._setup_ui()

    def _setup_ui(self):
        """Create the UI layout."""
        layout = QVBoxLayout(self)
        layout.setSpacing(12)

        # Header with refresh button
        header_layout = QHBoxLayout()
        header_label = QLabel("Performance Monitoring Dashboard")
        header_label.setStyleSheet("font-size: 16px; font-weight: bold;")
        header_layout.addWidget(header_label)
        header_layout.addStretch()

        self.refresh_btn = QPushButton("Refresh")
        self.refresh_btn.clicked.connect(self._refresh_metrics)
        header_layout.addWidget(self.refresh_btn)

        layout.addLayout(header_layout)

        # Main content in a scroll area
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        content_widget = QWidget()
        content_layout = QVBoxLayout(content_widget)
        content_layout.setSpacing(16)

        # Health status section
        content_layout.addWidget(self._create_health_section())

        # Rolling metrics section
        content_layout.addWidget(self._create_metrics_section())

        # Regime detection section
        content_layout.addWidget(self._create_regime_section())

        # Alerts section
        content_layout.addWidget(self._create_alerts_section())

        # Data update section
        content_layout.addWidget(self._create_data_section())

        content_layout.addStretch()
        scroll.setWidget(content_widget)
        layout.addWidget(scroll)

    def _create_health_section(self) -> QGroupBox:
        """Create the strategy health status section."""
        group = QGroupBox("Strategy Health")
        layout = QHBoxLayout(group)

        # Health score card
        self.health_card = MetricCard("Health Score", "—", "Overall strategy health")
        layout.addWidget(self.health_card)

        # Status indicator
        self.status_card = MetricCard("Status", "—", "No trades loaded")
        layout.addWidget(self.status_card)

        # Active alerts
        self.alerts_card = MetricCard("Active Alerts", "0", "0 critical, 0 warning")
        layout.addWidget(self.alerts_card)

        # Trades loaded
        self.trades_card = MetricCard("Trades Loaded", "0", "Load trades to monitor")
        layout.addWidget(self.trades_card)

        return group

    def _create_metrics_section(self) -> QGroupBox:
        """Create the rolling metrics section."""
        group = QGroupBox("Rolling Performance Metrics")
        layout = QVBoxLayout(group)

        # Window selector
        window_layout = QHBoxLayout()
        window_layout.addWidget(QLabel("Window Size:"))
        self.window_combo = QComboBox()
        self.window_combo.addItems(["20 trades", "50 trades", "100 trades"])
        self.window_combo.currentIndexChanged.connect(self._on_window_changed)
        window_layout.addWidget(self.window_combo)
        window_layout.addStretch()
        layout.addLayout(window_layout)

        # Metric cards row
        cards_layout = QHBoxLayout()

        self.sharpe_card = MetricCard("Sharpe Ratio", "—", "Risk-adjusted return")
        cards_layout.addWidget(self.sharpe_card)

        self.winrate_card = MetricCard("Win Rate", "—%", "Percentage of winning trades")
        cards_layout.addWidget(self.winrate_card)

        self.pf_card = MetricCard("Profit Factor", "—", "Gross profit / gross loss")
        cards_layout.addWidget(self.pf_card)

        self.drawdown_card = MetricCard("Max Drawdown", "$—", "Largest peak-to-trough")
        cards_layout.addWidget(self.drawdown_card)

        layout.addLayout(cards_layout)

        # Second row
        cards_layout2 = QHBoxLayout()

        self.pnl_card = MetricCard("Total P&L", "$—", "Rolling window total")
        cards_layout2.addWidget(self.pnl_card)

        self.avg_pnl_card = MetricCard("Avg Trade", "$—", "Average P&L per trade")
        cards_layout2.addWidget(self.avg_pnl_card)

        self.streak_card = MetricCard("Current Streak", "—", "Consecutive wins/losses")
        cards_layout2.addWidget(self.streak_card)

        self.comparison_card = MetricCard("Trend", "—", "Recent vs long-term")
        cards_layout2.addWidget(self.comparison_card)

        layout.addLayout(cards_layout2)

        # Equity curve chart
        self.equity_chart = pg.PlotWidget()
        self.equity_chart.setBackground('#1e1e1e')
        self.equity_chart.showGrid(x=True, y=True, alpha=0.3)
        self.equity_chart.setLabel('left', 'Cumulative P&L ($)')
        self.equity_chart.setLabel('bottom', 'Trade #')
        self.equity_chart.setMaximumHeight(200)
        layout.addWidget(self.equity_chart)

        return group

    def _create_regime_section(self) -> QGroupBox:
        """Create the market regime section."""
        group = QGroupBox("Market Regime Detection")
        layout = QHBoxLayout(group)

        # Volatility regime
        self.vol_card = MetricCard("Volatility", "—", "ATR percentile: —")
        layout.addWidget(self.vol_card)

        # Trend regime
        self.trend_card = MetricCard("Trend", "—", "Strength: —")
        layout.addWidget(self.trend_card)

        # Correlation regime
        self.corr_card = MetricCard("Correlation", "—", "With market: —")
        layout.addWidget(self.corr_card)

        # ATR value
        self.atr_card = MetricCard("ATR", "—", "— % of price")
        layout.addWidget(self.atr_card)

        return group

    def _create_alerts_section(self) -> QGroupBox:
        """Create the degradation alerts section."""
        group = QGroupBox("Degradation Alerts")
        layout = QVBoxLayout(group)

        # Alerts table
        self.alerts_table = QTableWidget()
        self.alerts_table.setColumnCount(5)
        self.alerts_table.setHorizontalHeaderLabels([
            "Time", "Severity", "Type", "Message", "Value"
        ])
        self.alerts_table.horizontalHeader().setSectionResizeMode(3, QHeaderView.Stretch)
        self.alerts_table.setMaximumHeight(150)
        self.alerts_table.setAlternatingRowColors(True)
        layout.addWidget(self.alerts_table)

        # Action buttons
        btn_layout = QHBoxLayout()
        self.acknowledge_btn = QPushButton("Acknowledge Selected")
        self.acknowledge_btn.clicked.connect(self._acknowledge_alert)
        btn_layout.addWidget(self.acknowledge_btn)

        self.trigger_reopt_btn = QPushButton("Trigger Re-optimization")
        self.trigger_reopt_btn.clicked.connect(self._trigger_reoptimization)
        self.trigger_reopt_btn.setStyleSheet("background-color: #3a5a8a;")
        btn_layout.addWidget(self.trigger_reopt_btn)

        btn_layout.addStretch()
        layout.addLayout(btn_layout)

        return group

    def _create_data_section(self) -> QGroupBox:
        """Create the data update section."""
        group = QGroupBox("Data Updates")
        layout = QVBoxLayout(group)

        # Status row
        status_layout = QHBoxLayout()

        self.data_status_label = QLabel("Data status: Not checked")
        status_layout.addWidget(self.data_status_label)

        status_layout.addStretch()

        self.check_data_btn = QPushButton("Check Updates")
        self.check_data_btn.clicked.connect(self._check_data_updates)
        status_layout.addWidget(self.check_data_btn)

        self.update_data_btn = QPushButton("Update Data")
        self.update_data_btn.clicked.connect(self._update_data)
        status_layout.addWidget(self.update_data_btn)

        layout.addLayout(status_layout)

        # Data table
        self.data_table = QTableWidget()
        self.data_table.setColumnCount(4)
        self.data_table.setHorizontalHeaderLabels([
            "Ticker", "Last Data", "Last Update", "Status"
        ])
        self.data_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.data_table.setMaximumHeight(120)
        layout.addWidget(self.data_table)

        return group

    def load_trades(self, trades: List):
        """Load trades from a backtest for monitoring."""
        self.trades = trades

        # Clear and reload performance monitor
        self.perf_monitor.clear()

        for trade in trades:
            record = TradeRecord(
                pnl=trade.pnl,
                timestamp=trade.exit_time if hasattr(trade, 'exit_time') else datetime.now(),
                ticker=trade.ticker if hasattr(trade, 'ticker') else "",
                direction=trade.direction if hasattr(trade, 'direction') else "",
                entry_price=trade.entry_price if hasattr(trade, 'entry_price') else 0,
                exit_price=trade.exit_price if hasattr(trade, 'exit_price') else 0,
            )
            self.perf_monitor.add_trade(record)

        # Check for degradation
        self.degradation_detector.check_degradation()

        # Update UI
        self._refresh_metrics()

    def _refresh_metrics(self):
        """Refresh all displayed metrics."""
        # Health status
        health = self.degradation_detector.get_health_status()
        score = health['score']
        status = health['status']

        # Color based on health
        if score >= 80:
            color = "#00ff00"
        elif score >= 60:
            color = "#ffff00"
        elif score >= 40:
            color = "#ff8800"
        else:
            color = "#ff0000"

        self.health_card.set_value(f"{score:.0f}", color)
        self.status_card.set_value(status.upper(), color)

        # Alerts
        active_alerts = self.degradation_detector.get_active_alerts()
        critical = sum(1 for a in active_alerts if a.severity == AlertSeverity.CRITICAL)
        warning = sum(1 for a in active_alerts if a.severity == AlertSeverity.WARNING)
        self.alerts_card.set_value(str(len(active_alerts)),
                                   "#ff0000" if critical > 0 else "#ffff00" if warning > 0 else "#00ff00")
        self.alerts_card.set_subtitle(f"{critical} critical, {warning} warning")

        # Trade count
        self.trades_card.set_value(str(len(self.trades)))
        self.trades_card.set_subtitle(f"{self.perf_monitor.get_trade_count()} in monitor")

        # Rolling metrics
        self._update_rolling_metrics()

        # Regime
        self._update_regime_display()

        # Alerts table
        self._update_alerts_table()

        # Equity chart
        self._update_equity_chart()

    def _update_rolling_metrics(self):
        """Update the rolling metrics cards."""
        windows = [20, 50, 100]
        window = windows[self.window_combo.currentIndex()]
        metrics = self.perf_monitor.get_current_metrics(window)

        if metrics.total_trades == 0:
            return

        # Sharpe
        sharpe = metrics.sharpe_ratio
        sharpe_color = "#00ff00" if sharpe > 1.0 else "#ffff00" if sharpe > 0.5 else "#ff4444"
        self.sharpe_card.set_value(f"{sharpe:.2f}", sharpe_color)

        # Win rate
        wr = metrics.win_rate
        wr_color = "#00ff00" if wr > 55 else "#ffff00" if wr > 45 else "#ff4444"
        self.winrate_card.set_value(f"{wr:.1f}%", wr_color)

        # Profit factor
        pf = metrics.profit_factor
        pf_color = "#00ff00" if pf > 1.5 else "#ffff00" if pf > 1.0 else "#ff4444"
        self.pf_card.set_value(f"{pf:.2f}", pf_color)

        # Drawdown
        dd = metrics.max_drawdown
        dd_color = "#00ff00" if dd < 500 else "#ffff00" if dd < 1000 else "#ff4444"
        self.drawdown_card.set_value(f"${dd:.0f}", dd_color)

        # P&L
        pnl = metrics.total_pnl
        pnl_color = "#00ff00" if pnl > 0 else "#ff4444"
        self.pnl_card.set_value(f"${pnl:,.0f}", pnl_color)

        # Average
        avg = metrics.avg_pnl
        avg_color = "#00ff00" if avg > 0 else "#ff4444"
        self.avg_pnl_card.set_value(f"${avg:.0f}", avg_color)

        # Streak
        if metrics.consecutive_wins > 0:
            self.streak_card.set_value(f"+{metrics.consecutive_wins}", "#00ff00")
            self.streak_card.set_subtitle("consecutive wins")
        elif metrics.consecutive_losses > 0:
            self.streak_card.set_value(f"-{metrics.consecutive_losses}", "#ff4444")
            self.streak_card.set_subtitle("consecutive losses")
        else:
            self.streak_card.set_value("0", "#888888")

        # Comparison
        comparison = self.perf_monitor.get_metrics_comparison()
        if comparison:
            if comparison.get('is_degrading'):
                self.comparison_card.set_value("DEGRADING", "#ff4444")
                self.comparison_card.set_subtitle(
                    f"Win rate: {comparison.get('win_rate_diff', 0):+.1f}%"
                )
            else:
                wr_diff = comparison.get('win_rate_diff', 0)
                if wr_diff > 0:
                    self.comparison_card.set_value("IMPROVING", "#00ff00")
                else:
                    self.comparison_card.set_value("STABLE", "#888888")
                self.comparison_card.set_subtitle(
                    f"Win rate: {wr_diff:+.1f}%"
                )

    def _update_regime_display(self):
        """Update the regime detection display."""
        regime = self.regime_detector.get_current_regime()

        if regime is None:
            return

        # Volatility
        vol = regime.volatility.value
        vol_colors = {"low": "#00ff00", "medium": "#888888", "high": "#ff8800", "extreme": "#ff0000"}
        self.vol_card.set_value(vol.upper(), vol_colors.get(vol, "#888888"))
        self.vol_card.set_subtitle(f"ATR percentile: {regime.volatility_percentile:.0f}")

        # Trend
        trend = regime.trend.value.replace("_", " ")
        trend_colors = {
            "strong_up": "#00ff00", "weak_up": "#88ff88",
            "ranging": "#888888",
            "weak_down": "#ff8888", "strong_down": "#ff0000"
        }
        self.trend_card.set_value(trend.upper(), trend_colors.get(regime.trend.value, "#888888"))
        self.trend_card.set_subtitle(f"Strength: {regime.trend_strength:.0f}")

        # Correlation
        corr = regime.correlation.value.replace("_", " ")
        self.corr_card.set_value(corr.upper(), "#888888")
        self.corr_card.set_subtitle(f"With market: {regime.correlation_value:.2f}")

        # ATR
        self.atr_card.set_value(f"${regime.atr_value:.2f}", "#888888")
        self.atr_card.set_subtitle(f"{regime.atr_percent:.2f}% of price")

    def _update_alerts_table(self):
        """Update the alerts table."""
        alerts = self.degradation_detector.get_active_alerts()

        self.alerts_table.setRowCount(len(alerts))

        for row, alert in enumerate(alerts):
            # Time
            self.alerts_table.setItem(row, 0, QTableWidgetItem(
                alert.timestamp.strftime("%Y-%m-%d %H:%M")
            ))

            # Severity
            severity_item = QTableWidgetItem(alert.severity.value.upper())
            if alert.severity == AlertSeverity.CRITICAL:
                severity_item.setBackground(QColor("#660000"))
            elif alert.severity == AlertSeverity.WARNING:
                severity_item.setBackground(QColor("#665500"))
            self.alerts_table.setItem(row, 1, severity_item)

            # Type
            self.alerts_table.setItem(row, 2, QTableWidgetItem(
                alert.alert_type.value.replace("_", " ")
            ))

            # Message
            self.alerts_table.setItem(row, 3, QTableWidgetItem(alert.message))

            # Value
            self.alerts_table.setItem(row, 4, QTableWidgetItem(
                f"{alert.current_value:.2f}"
            ))

    def _update_equity_chart(self):
        """Update the rolling equity chart."""
        self.equity_chart.clear()

        if not self.trades:
            return

        # Calculate cumulative P&L
        pnls = [t.pnl for t in self.trades]
        cumulative = []
        total = 0
        for pnl in pnls:
            total += pnl
            cumulative.append(total)

        x = list(range(len(cumulative)))

        # Plot
        pen = pg.mkPen(color='#2a82da', width=2)
        self.equity_chart.plot(x, cumulative, pen=pen)

        # Add zero line
        self.equity_chart.addLine(y=0, pen=pg.mkPen('#444444', width=1))

    def _on_window_changed(self, index: int):
        """Handle window size change."""
        self._update_rolling_metrics()

    def _acknowledge_alert(self):
        """Acknowledge selected alert."""
        row = self.alerts_table.currentRow()
        if row >= 0:
            alerts = self.degradation_detector.get_active_alerts()
            if row < len(alerts):
                self.degradation_detector.acknowledge_alert(alerts[row].alert_type)
                self._update_alerts_table()

    def _trigger_reoptimization(self):
        """Manually trigger re-optimization."""
        request = self.trigger.trigger_manual("TSLA", "Manual trigger from monitoring tab")
        self.reoptimization_requested.emit("TSLA", {})

    def _check_data_updates(self):
        """Check for available data updates."""
        # Discover tickers
        tickers = self.data_updater.discover_tickers()

        self.data_table.setRowCount(len(tickers))

        for row, ticker in enumerate(tickers):
            # Ticker
            self.data_table.setItem(row, 0, QTableWidgetItem(ticker))

            # Last data date
            last_data = self.data_updater.get_last_data_date(ticker)
            self.data_table.setItem(row, 1, QTableWidgetItem(
                last_data.isoformat() if last_data else "N/A"
            ))

            # Last update
            last_update = self.data_updater.last_update_by_ticker.get(ticker)
            self.data_table.setItem(row, 2, QTableWidgetItem(
                last_update.strftime("%Y-%m-%d %H:%M") if last_update else "Never"
            ))

            # Status
            needs_update = self.data_updater.needs_update(ticker)
            status_item = QTableWidgetItem("Needs Update" if needs_update else "Up to Date")
            if needs_update:
                status_item.setBackground(QColor("#665500"))
            self.data_table.setItem(row, 3, status_item)

        self.data_status_label.setText(f"Found {len(tickers)} tickers")

    def _update_data(self):
        """Trigger data update for all tickers."""
        if not self.data_updater.config.api_key:
            self.data_status_label.setText("No Polygon API key configured")
            self.data_status_label.setStyleSheet("color: #ff4444;")
            return

        self.data_status_label.setText("Updating data...")
        self.update_data_btn.setEnabled(False)

        # This should be done in a worker thread for production
        # For now, just update status
        tickers = self.data_updater.discover_tickers()
        self.data_updater.config.tickers = tickers

        # Note: actual update would block UI - should use QThread
        self.data_status_label.setText(
            "Data update requires Polygon API. Set POLYGON_API_KEY environment variable."
        )
        self.update_data_btn.setEnabled(True)

    def set_data_dir(self, path: str):
        """Update the data directory."""
        self.data_dir = path
        self.data_updater.config.data_dir = path
