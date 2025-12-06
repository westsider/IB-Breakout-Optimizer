"""
Main Window - Tabbed interface for the IB Breakout Optimizer.
"""

from PySide6.QtWidgets import (
    QMainWindow, QTabWidget, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QStatusBar, QMenuBar, QMenu, QMessageBox, QFileDialog
)
from PySide6.QtCore import Qt, QSettings
from PySide6.QtGui import QAction

from desktop_ui.tabs.backtest_tab import BacktestTab
from desktop_ui.tabs.optimization_tab import OptimizationTab
from desktop_ui.tabs.trade_browser_tab import TradeBrowserTab
from desktop_ui.tabs.ib_analysis_tab import IBAnalysisTab
from desktop_ui.tabs.equity_curve_tab import EquityCurveTab
from desktop_ui.tabs.download_tab import DownloadTab


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
        self.backtest_tab = BacktestTab(self.data_dir, self.output_dir)
        self.optimization_tab = OptimizationTab(self.data_dir, self.output_dir)
        self.trade_browser_tab = TradeBrowserTab()
        self.ib_analysis_tab = IBAnalysisTab()
        self.equity_curve_tab = EquityCurveTab()
        self.download_tab = DownloadTab(self.data_dir)

        # Connect tabs for sharing data
        self.backtest_tab.backtest_complete.connect(self._on_backtest_complete)
        self.optimization_tab.optimization_complete.connect(self._on_optimization_complete)

        # Connect optimization result double-click to run backtest
        self.optimization_tab.results_table.doubleClicked.connect(self._on_optimization_result_double_clicked)
        self.optimization_tab.best_params_table.doubleClicked.connect(self._on_best_params_double_clicked)

        # Add tabs
        self.tabs.addTab(self.backtest_tab, "Backtest")
        self.tabs.addTab(self.optimization_tab, "Optimization")
        self.tabs.addTab(self.equity_curve_tab, "Equity Curve")
        self.tabs.addTab(self.trade_browser_tab, "Trade Browser")
        self.tabs.addTab(self.ib_analysis_tab, "IB Analysis")
        self.tabs.addTab(self.download_tab, "Download")

        # Restore last active tab
        last_tab = self.settings.value("last_tab", 0, type=int)
        if 0 <= last_tab < self.tabs.count():
            self.tabs.setCurrentIndex(last_tab)

        # Save tab changes
        self.tabs.currentChanged.connect(self._on_tab_changed)

        # Load saved trades on startup
        self._load_saved_trades()

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
            self.backtest_tab.set_data_dir(directory)
            self.optimization_tab.set_data_dir(directory)
            self.download_tab.set_data_dir(directory)

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
            self.backtest_tab.set_output_dir(directory)
            self.optimization_tab.set_output_dir(directory)

            self.status_label.setText(f"Output directory set to: {directory}")

    def _on_backtest_complete(self, result, metrics):
        """Handle backtest completion."""
        trades = result.trades if result else []
        self.trade_browser_tab.load_trades(trades)
        self.equity_curve_tab.load_trades(trades)
        self.ib_analysis_tab.load_trades(trades)
        self.status_label.setText(
            f"Backtest complete: {metrics.total_trades} trades, "
            f"P&L: ${metrics.total_net_profit:,.2f}"
        )

    def _on_optimization_complete(self, results):
        """Handle optimization completion."""
        self.status_label.setText(
            f"Optimization complete: {results.get('completed', 0)} combinations tested"
        )

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

    def _load_saved_trades(self):
        """Load saved trades on startup."""
        trades = self.backtest_tab.load_saved_trades()
        if trades:
            self.trade_browser_tab.load_trades(trades)
            self.equity_curve_tab.load_trades(trades)
            self.ib_analysis_tab.load_trades(trades)
            self.status_label.setText(f"Loaded {len(trades)} saved trades")

    def _on_optimization_result_double_clicked(self, index):
        """Handle double-click on optimization result to run backtest with those params."""
        row = index.row()
        if not hasattr(self.optimization_tab, 'top_results_data'):
            return
        if row >= len(self.optimization_tab.top_results_data):
            return

        result = self.optimization_tab.top_results_data[row]
        self._run_backtest_with_params(result)

    def _on_best_params_double_clicked(self, index):
        """Handle double-click on best params to run backtest."""
        if hasattr(self.optimization_tab, 'best_params') and self.optimization_tab.best_params:
            self._run_backtest_with_params(self.optimization_tab.best_params)

    def _run_backtest_with_params(self, params: dict):
        """Load params into backtest tab and run."""
        # Set ticker if available
        ticker = self.optimization_tab.ticker_combo.currentText()
        idx = self.backtest_tab.ticker_combo.findText(ticker)
        if idx >= 0:
            self.backtest_tab.ticker_combo.setCurrentIndex(idx)

        # Load the parameters
        self.backtest_tab.load_params_from_json(params)

        # Switch to backtest tab and run
        self.tabs.setCurrentWidget(self.backtest_tab)
        self.backtest_tab._run_backtest()

        self.status_label.setText(f"Running backtest with optimization params...")

    def closeEvent(self, event):
        """Handle window close."""
        self._save_window_geometry()
        self.settings.setValue("last_tab", self.tabs.currentIndex())
        event.accept()
