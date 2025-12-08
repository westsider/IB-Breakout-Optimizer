"""
Main Window - Tabbed interface for the IB Breakout Optimizer.
"""

from PySide6.QtWidgets import (
    QMainWindow, QTabWidget, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QStatusBar, QMenuBar, QMenu, QMessageBox, QFileDialog
)
from PySide6.QtCore import Qt, QSettings
from PySide6.QtGui import QAction

from desktop_ui.tabs.optimization_tab import OptimizationTab
from desktop_ui.tabs.portfolio_tab import PortfolioTab
from desktop_ui.tabs.forward_tests_tab import ForwardTestsTab
from desktop_ui.tabs.walk_forward_tab import WalkForwardTab
from desktop_ui.tabs.trade_browser_tab import TradeBrowserTab
from desktop_ui.tabs.ib_analysis_tab import IBAnalysisTab
from desktop_ui.tabs.filter_analysis_tab import FilterAnalysisTab
from desktop_ui.tabs.equity_curve_tab import EquityCurveTab
from desktop_ui.tabs.download_tab import DownloadTab
from desktop_ui.tabs.monitoring_tab import MonitoringTab
from desktop_ui.tabs.ml_filter_tab import MLFilterTab
from desktop_ui.tabs.saved_tests_tab import SavedTestsTab
from desktop_ui.workers.backtest_worker import BacktestWorker


class MainWindow(QMainWindow):
    """Main application window with tabbed interface."""

    def __init__(self):
        super().__init__()

        self.setWindowTitle("IB Breakout Optimizer")
        self.setMinimumSize(1200, 800)

        # Load settings
        self.settings = QSettings("TradingTools", "IBBreakoutOptimizer")
        self._load_window_geometry()

        # Default paths - use app data folder
        from pathlib import Path
        app_dir = Path(__file__).parent.parent
        default_data_dir = app_dir / "market_data"
        default_output_dir = app_dir / "output"

        # Create directories if they don't exist
        default_data_dir.mkdir(parents=True, exist_ok=True)
        default_output_dir.mkdir(parents=True, exist_ok=True)

        self.data_dir = self.settings.value(
            "data_dir",
            str(default_data_dir)
        )
        self.output_dir = self.settings.value(
            "output_dir",
            str(default_output_dir)
        )

        # Setup UI
        self._setup_menu_bar()
        self._setup_central_widget()
        self._setup_status_bar()

    def _setup_menu_bar(self):
        """Create the menu bar."""
        menubar = self.menuBar()

        # File menu
        file_menu = menubar.addMenu("&File")

        # Data directory action
        data_dir_action = QAction("Set &Data Directory...", self)
        data_dir_action.triggered.connect(self._set_data_directory)
        file_menu.addAction(data_dir_action)

        # Output directory action
        output_dir_action = QAction("Set &Output Directory...", self)
        output_dir_action.triggered.connect(self._set_output_directory)
        file_menu.addAction(output_dir_action)

        file_menu.addSeparator()

        # Exit action
        exit_action = QAction("E&xit", self)
        exit_action.setShortcut("Alt+F4")
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)

        # Help menu
        help_menu = menubar.addMenu("&Help")

        wf_guide_action = QAction("How to &Walk-Forward Test and Trade", self)
        wf_guide_action.triggered.connect(self._show_walk_forward_guide)
        help_menu.addAction(wf_guide_action)

        help_menu.addSeparator()

        about_action = QAction("&About", self)
        about_action.triggered.connect(self._show_about)
        help_menu.addAction(about_action)

    def _setup_central_widget(self):
        """Create the central widget with tabs."""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        layout = QVBoxLayout(central_widget)
        layout.setContentsMargins(8, 8, 8, 8)

        # Header
        header = QLabel("IB Breakout Strategy Optimizer")
        header.setStyleSheet("""
            font-size: 20px;
            font-weight: bold;
            color: #2a82da;
            padding: 8px 0;
        """)
        layout.addWidget(header)

        # Tab widget
        self.tabs = QTabWidget()
        layout.addWidget(self.tabs)

        # Create tabs
        self.optimization_tab = OptimizationTab(self.data_dir, self.output_dir)
        self.walk_forward_tab = WalkForwardTab(self.data_dir, self.output_dir)
        self.portfolio_tab = PortfolioTab(self.output_dir)
        self.forward_tests_tab = ForwardTestsTab(self.data_dir, self.output_dir)
        self.trade_browser_tab = TradeBrowserTab(self.data_dir)
        self.ib_analysis_tab = IBAnalysisTab()
        self.filter_analysis_tab = FilterAnalysisTab(self.data_dir)
        self.equity_curve_tab = EquityCurveTab()
        self.monitoring_tab = MonitoringTab(self.data_dir)
        self.ml_filter_tab = MLFilterTab(self.data_dir, self.output_dir)
        self.download_tab = DownloadTab(self.data_dir)
        self.saved_tests_tab = SavedTestsTab(self.output_dir)

        # Connect tabs for sharing data
        self.optimization_tab.optimization_complete.connect(self._on_optimization_complete)
        self.optimization_tab.result_double_clicked.connect(self._on_result_double_clicked)
        self.optimization_tab.save_test_requested.connect(self._on_save_test_requested)
        self.saved_tests_tab.load_params_requested.connect(self._on_load_saved_params)

        # Backtest worker for populating other tabs
        self.backtest_worker = None

        # Add tabs
        self.tabs.addTab(self.optimization_tab, "Optimization")
        self.tabs.addTab(self.walk_forward_tab, "Walk-Forward")
        self.tabs.addTab(self.portfolio_tab, "Portfolio")
        self.tabs.addTab(self.forward_tests_tab, "Forward Tests")
        self.tabs.addTab(self.equity_curve_tab, "Equity Curve")
        self.tabs.addTab(self.trade_browser_tab, "Trade Browser")
        self.tabs.addTab(self.ib_analysis_tab, "IB Analysis")
        self.tabs.addTab(self.filter_analysis_tab, "Filter Analysis")
        self.tabs.addTab(self.monitoring_tab, "Monitoring")
        self.tabs.addTab(self.ml_filter_tab, "ML Filter")
        self.tabs.addTab(self.saved_tests_tab, "Saved Tests")
        self.tabs.addTab(self.download_tab, "Download")

        # Restore last active tab
        last_tab = self.settings.value("last_tab", 0, type=int)
        if 0 <= last_tab < self.tabs.count():
            self.tabs.setCurrentIndex(last_tab)

        # Save tab changes
        self.tabs.currentChanged.connect(self._on_tab_changed)

    def _setup_status_bar(self):
        """Create the status bar."""
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)

        self.status_label = QLabel("Ready")
        self.status_bar.addWidget(self.status_label)

        # Data directory indicator
        self.data_dir_label = QLabel(f"Data: {self.data_dir}")
        self.data_dir_label.setStyleSheet("color: #888888;")
        self.status_bar.addPermanentWidget(self.data_dir_label)

    def _load_window_geometry(self):
        """Restore window size and position."""
        geometry = self.settings.value("geometry")
        if geometry:
            self.restoreGeometry(geometry)
        else:
            # Center on screen
            self.resize(1400, 900)

    def _save_window_geometry(self):
        """Save window size and position."""
        self.settings.setValue("geometry", self.saveGeometry())

    def _set_data_directory(self):
        """Let user select data directory."""
        directory = QFileDialog.getExistingDirectory(
            self,
            "Select Data Directory",
            self.data_dir
        )
        if directory:
            self.data_dir = directory
            self.settings.setValue("data_dir", directory)
            self.data_dir_label.setText(f"Data: {directory}")

            # Update tabs
            self.optimization_tab.set_data_dir(directory)
            self.download_tab.set_data_dir(directory)
            self.trade_browser_tab.set_data_dir(directory)
            self.monitoring_tab.set_data_dir(directory)
            self.ml_filter_tab.set_data_dir(directory)
            self.forward_tests_tab.set_data_dir(directory)
            self.walk_forward_tab.set_data_dir(directory)
            self.filter_analysis_tab.set_data_dir(directory)

            self.status_label.setText(f"Data directory set to: {directory}")

    def _set_output_directory(self):
        """Let user select output directory."""
        directory = QFileDialog.getExistingDirectory(
            self,
            "Select Output Directory",
            self.output_dir
        )
        if directory:
            self.output_dir = directory
            self.settings.setValue("output_dir", directory)

            # Update tabs
            self.optimization_tab.set_output_dir(directory)
            self.ml_filter_tab.set_output_dir(directory)
            self.saved_tests_tab.set_output_dir(directory)
            self.portfolio_tab.set_output_dir(directory)
            self.forward_tests_tab.set_output_dir(directory)
            self.walk_forward_tab.set_output_dir(directory)

            self.status_label.setText(f"Output directory set to: {directory}")

    def _on_optimization_complete(self, results):
        """Handle optimization completion."""
        self.status_label.setText(
            f"Optimization complete: {results.get('completed', 0)} combinations tested"
        )

        # Send best result to ML Filter tab for training
        best_params = results.get('best_params', {})
        ticker = results.get('ticker', 'TSLA')

        if best_params:
            # Add win rate from best_metrics if available
            best_metrics = results.get('best_metrics', {})
            best_params['win_rate'] = best_metrics.get('win_rate', 0)

            # Send to ML tab
            self.ml_filter_tab.set_optimizer_params(best_params, ticker)

    def _on_result_double_clicked(self, params: dict, ticker: str):
        """Handle double-click on optimization result - run full backtest."""
        self.status_label.setText(f"Running backtest for {ticker}...")
        self.status_label.setStyleSheet("color: #2a82da;")

        # Start backtest worker
        self.backtest_worker = BacktestWorker(self.data_dir, params, ticker)
        self.backtest_worker.status.connect(self._on_backtest_status)
        self.backtest_worker.finished.connect(self._on_backtest_complete)
        self.backtest_worker.error.connect(self._on_backtest_error)
        self.backtest_worker.start()

    def _on_backtest_status(self, message: str):
        """Handle backtest status update."""
        self.status_label.setText(message)

    def _on_backtest_complete(self, trades, metrics):
        """Handle backtest completion - populate all tabs with trade data."""
        trade_count = len(trades) if trades else 0
        self.status_label.setText(f"Backtest complete: {trade_count} trades loaded")
        self.status_label.setStyleSheet("color: #00ff00;")

        if trades:
            # Populate Trade Browser
            self.trade_browser_tab.load_trades(trades)

            # Populate Equity Curve
            self.equity_curve_tab.load_trades(trades)

            # Populate IB Analysis
            self.ib_analysis_tab.load_trades(trades)

            # Populate Monitoring tab
            self.monitoring_tab.load_trades(trades)

            # Switch to Equity Curve tab to show results
            self.tabs.setCurrentWidget(self.equity_curve_tab)

    def _on_backtest_error(self, error_msg: str):
        """Handle backtest error."""
        self.status_label.setText(f"Backtest error: {error_msg[:100]}")
        self.status_label.setStyleSheet("color: #ff4444;")
        print(f"Backtest error: {error_msg}")

    def _on_load_saved_params(self, params: dict, ticker: str):
        """Handle request to load saved params to optimizer."""
        # Switch to Optimization tab and load params
        self.tabs.setCurrentWidget(self.optimization_tab)
        self.optimization_tab.load_params(params, ticker)
        self.status_label.setText(f"Loaded saved parameters for {ticker}")
        self.status_label.setStyleSheet("color: #00ff00;")

    def _on_save_test_requested(self, result_dict: dict):
        """Handle request to save current test result."""
        ticker = result_dict.get('ticker', 'UNKNOWN')

        is_new_best = self.saved_tests_tab.save_test_result(
            ticker=ticker,
            total_pnl=result_dict.get('total_pnl', 0),
            profit_factor=result_dict.get('profit_factor', 0),
            win_rate=result_dict.get('win_rate', 0),
            total_trades=result_dict.get('total_trades', 0),
            max_drawdown=result_dict.get('max_drawdown', 0),
            sharpe_ratio=result_dict.get('sharpe_ratio', 0),
            params=result_dict.get('params', {}),
            equity_curve=result_dict.get('equity_curve', [])
        )

        if is_new_best:
            self.status_label.setText(f"NEW BEST for {ticker}! Saved to Saved Tests")
            self.status_label.setStyleSheet("color: #00ff00;")
        else:
            self.status_label.setText(f"Saved test for {ticker}")
            self.status_label.setStyleSheet("color: #2a82da;")

    def _show_walk_forward_guide(self):
        """Show walk-forward testing and trading guide."""
        guide_text = """
        <h2>Walk-Forward Testing & Trading Guide</h2>

        <h3>What is Walk-Forward Analysis?</h3>
        <p>Walk-forward analysis validates your strategy by simulating how it would perform
        if you re-optimized periodically. It divides your data into rolling windows:</p>
        <ul>
            <li><b>In-Sample (IS)</b>: Training period where parameters are optimized</li>
            <li><b>Out-of-Sample (OOS)</b>: Forward test period using IS parameters</li>
        </ul>

        <h3>Recommended Settings</h3>
        <table border="1" cellpadding="5" style="border-collapse: collapse;">
            <tr><th>Setting</th><th>Value</th><th>Rationale</th></tr>
            <tr><td>Train Window</td><td>12 months</td><td>Enough data for statistical significance</td></tr>
            <tr><td>Test Window</td><td>1 week</td><td>Matches weekly re-optimization cycle</td></tr>
            <tr><td>Preset</td><td>Standard or Quick</td><td>Balance between thoroughness and speed</td></tr>
        </table>

        <h3>Weekly Re-optimization Workflow</h3>
        <p>Once walk-forward validates your approach, use this weekly cycle:</p>
        <table border="1" cellpadding="5" style="border-collapse: collapse;">
            <tr><th>Day</th><th>Action</th></tr>
            <tr><td>Friday Close</td><td>Week's trading ends</td></tr>
            <tr><td>Weekend</td><td>Run optimization with last 12 months of data</td></tr>
            <tr><td>Monday</td><td>Use new optimized parameters for live trading</td></tr>
            <tr><td>Mon-Fri</td><td>Trade with these parameters</td></tr>
            <tr><td>Repeat</td><td>Next weekend, re-optimize with fresh data</td></tr>
        </table>

        <h3>How Parameters Roll Forward</h3>
        <pre>
Week 1: Optimize months 1-12  → Trade week 13
Week 2: Optimize months 1.25-12.25 → Trade week 14
Week 3: Optimize months 1.5-12.5 → Trade week 15
        </pre>

        <h3>Key Metrics to Watch</h3>
        <ul>
            <li><b>Period Win Rate > 60%</b>: Most OOS weeks are profitable</li>
            <li><b>Efficiency Ratio > 50%</b>: OOS captures at least half of IS edge</li>
            <li><b>OOS Profit Factor > 1.5</b>: Good risk/reward in forward tests</li>
        </ul>

        <h3>When to Go Live</h3>
        <p>Your strategy is ready for live trading when:</p>
        <ol>
            <li>Walk-forward shows consistent OOS profitability across multiple periods</li>
            <li>Period Win Rate exceeds 50% (preferably 60%+)</li>
            <li>Efficiency Ratio shows IS edge transfers to OOS</li>
            <li>OOS metrics (P&L, PF, Win Rate) meet your trading requirements</li>
        </ol>

        <h3>Tips</h3>
        <ul>
            <li>Use the <b>Save</b> button to store successful walk-forward results</li>
            <li>Compare different filter combinations to find robust setups</li>
            <li>Longer IS periods (12mo) are generally more robust than shorter ones</li>
            <li>1-week OOS matches a practical weekly re-optimization schedule</li>
        </ul>
        """

        msg = QMessageBox(self)
        msg.setWindowTitle("Walk-Forward Testing & Trading Guide")
        msg.setTextFormat(Qt.RichText)
        msg.setText(guide_text)
        msg.setStandardButtons(QMessageBox.Ok)
        msg.setMinimumWidth(700)
        msg.exec()

    def _show_about(self):
        """Show about dialog."""
        QMessageBox.about(
            self,
            "About IB Breakout Optimizer",
            """<h3>IB Breakout Optimizer</h3>
            <p>Version 1.0</p>
            <p>A tool for backtesting and optimizing the Initial Balance
            Breakout trading strategy.</p>
            <p>Features:</p>
            <ul>
                <li>Multi-ticker backtesting</li>
                <li>Parallel grid search optimization</li>
                <li>Walk-forward analysis</li>
                <li>Trade visualization</li>
                <li>NinjaTrader integration</li>
            </ul>
            """
        )

    def _on_tab_changed(self, index: int):
        """Save current tab when changed."""
        self.settings.setValue("last_tab", index)

    def closeEvent(self, event):
        """Handle window close."""
        self._save_window_geometry()
        self.settings.setValue("last_tab", self.tabs.currentIndex())
        event.accept()
