"""
ML Filter Tab - Train and evaluate ML trade filter models.

Features:
- Run backtest to generate training data
- Train LightGBM or Ensemble classifier
- View model metrics (accuracy, precision, recall, ROC AUC)
- View feature importance chart
- View ML insights and recommendations
- Save/load trained models
- Integration with optimization results
"""

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QLabel, QComboBox, QPushButton, QGroupBox,
    QTableWidget, QTableWidgetItem, QHeaderView, QProgressBar,
    QSplitter, QFrame, QSpinBox, QDoubleSpinBox, QFileDialog,
    QMessageBox, QSlider, QTextEdit, QToolTip
)
from PySide6.QtCore import Qt, Signal, QThread
from PySide6.QtWebEngineWidgets import QWebEngineView
from PySide6.QtGui import QFont

from desktop_ui.widgets.metrics_panel import MetricCard
from pathlib import Path
import numpy as np


# Metric definitions for tooltips
METRIC_TOOLTIPS = {
    'accuracy': (
        "Accuracy: % of all predictions that were correct.\n"
        "Formula: (TP + TN) / Total\n"
        "Note: Can be misleading with imbalanced data."
    ),
    'precision': (
        "Precision: When model predicts WIN, how often is it right?\n"
        "Formula: TP / (TP + FP)\n"
        "High precision = fewer false alarms (costly bad trades)."
    ),
    'recall': (
        "Recall: Of all actual wins, how many did the model catch?\n"
        "Formula: TP / (TP + FN)\n"
        "High recall = fewer missed opportunities."
    ),
    'f1': (
        "F1 Score: Balance between precision and recall (0-1).\n"
        "Formula: 2 * (Precision * Recall) / (Precision + Recall)\n"
        "Use when you need balance between false alarms and missed trades."
    ),
    'roc_auc': (
        "ROC AUC: Model's ability to distinguish wins from losses.\n"
        "Range: 0.5 (random) to 1.0 (perfect)\n"
        "Values 0.6-0.7 are decent, >0.7 is good for trading."
    ),
    'cv_mean': (
        "CV Mean: Average accuracy across 5 time-based validation folds.\n"
        "Uses TimeSeriesSplit to preserve temporal order.\n"
        "Lower than train accuracy indicates some overfitting."
    )
}


class TrainingWorker(QThread):
    """Background worker for ML training."""

    progress = Signal(str)  # status message
    finished = Signal(dict)  # results dict
    error = Signal(str)

    def __init__(self, data_dir: str, ticker: str, params: dict,
                 use_ensemble: bool = True, threshold: float = 0.55):
        super().__init__()
        self.data_dir = data_dir
        self.ticker = ticker
        self.params = params
        self.use_ensemble = use_ensemble
        self.threshold = threshold

    def run(self):
        """Run training in background thread."""
        try:
            import sys
            sys.path.insert(0, str(Path(__file__).parent.parent.parent))

            from data.data_loader import DataLoader
            from backtester.backtest_runner import BacktestRunner
            from strategy.ib_breakout import StrategyParams
            from ml_filter.feature_builder import FeatureBuilder
            from ml_filter.model_trainer import MLTradeFilter

            # Step 1: Load data
            self.progress.emit("Loading market data...")
            loader = DataLoader(self.data_dir)
            bars_df = loader.load_ninjatrader_file(
                str(Path(self.data_dir) / f"{self.ticker}_NT.txt"),
                self.ticker
            )

            if bars_df.empty:
                self.error.emit(f"No data found for {self.ticker}")
                return

            # Step 2: Run backtest to get trades
            self.progress.emit("Running backtest to generate trades...")
            runner = BacktestRunner(self.data_dir)

            # Always use QQQ filter for non-QQQ tickers
            use_qqq = self.ticker != 'QQQ'

            strategy_params = StrategyParams(
                ib_duration_minutes=self.params.get('ib_duration', 30),
                profit_target_percent=self.params.get('profit_target', 1.0),
                trade_direction=self.params.get('direction', 'both'),
                use_qqq_filter=use_qqq
            )

            # Run backtest
            filter_ticker = 'QQQ' if use_qqq else None
            if filter_ticker:
                # run_backtest_with_filter returns (BacktestResult, PerformanceMetrics) tuple
                backtest_result, _ = runner.run_backtest_with_filter(
                    self.ticker, filter_ticker, strategy_params
                )
                trades = backtest_result.trades if hasattr(backtest_result, 'trades') else []
            else:
                result = runner.run_backtest(self.ticker, strategy_params)
                trades = result.trades if hasattr(result, 'trades') else []

            if len(trades) < 20:
                self.error.emit(f"Not enough trades for training: {len(trades)} (need at least 20)")
                return

            self.progress.emit(f"Found {len(trades)} trades, extracting features...")

            # Step 3: Build features
            feature_builder = FeatureBuilder(
                prior_days_lookback=self.params.get('prior_days_lookback', 3),
                daily_range_lookback=self.params.get('daily_range_lookback', 5)
            )

            # Convert bars to numpy array for feature extraction
            bars_array = bars_df[['open', 'high', 'low', 'close', 'volume']].values
            # Use 'timestamp' column, not index (index is just row numbers)
            timestamps = bars_df['timestamp'].values

            # Convert trades to list of dicts
            trade_dicts = []
            for trade in trades:
                # Get IB high/low from trade.ib object if available
                ib_high = None
                ib_low = None
                if hasattr(trade, 'ib') and trade.ib:
                    ib_high = trade.ib.ib_high
                    ib_low = trade.ib.ib_low

                # Get direction as string (trade.direction is an enum)
                direction = trade.direction.value if hasattr(trade.direction, 'value') else str(trade.direction)

                trade_dict = {
                    'entry_time': trade.entry_time,
                    'entry_price': trade.entry_price,
                    'direction': direction,
                    'pnl': trade.pnl,
                    'ib_high': ib_high,
                    'ib_low': ib_low
                }
                trade_dicts.append(trade_dict)

            # Build strategy params dict for feature extraction
            strategy_params_for_features = {
                'profit_target_percent': self.params.get('profit_target', 1.0),
                'stop_loss_type': self.params.get('stop_loss_type', 'opposite_ib'),
                'trailing_stop_enabled': self.params.get('trailing_stop_enabled', False),
                'break_even_enabled': self.params.get('break_even_enabled', False),
            }

            features_df = feature_builder.build_features_from_backtest(
                trade_dicts, bars_array, timestamps,
                ib_duration_minutes=self.params.get('ib_duration', 30),
                qqq_filter_used=use_qqq,
                strategy_params=strategy_params_for_features
            )

            if len(features_df) < 20:
                self.error.emit(f"Could not extract enough features: {len(features_df)} samples")
                return

            model_type = "ensemble" if self.use_ensemble else "LightGBM"
            self.progress.emit(f"Training {model_type} model on {len(features_df)} samples...")

            # Step 4: Train model
            ml_filter = MLTradeFilter(use_ensemble=self.use_ensemble)
            result = ml_filter.train(features_df, ticker=self.ticker)

            self.progress.emit("Training complete!")

            # Return results
            self.finished.emit({
                'ml_filter': ml_filter,
                'training_result': result,
                'n_trades': len(trades),
                'n_samples': len(features_df),
                'features_df': features_df,
                'threshold': self.threshold
            })

        except Exception as e:
            import traceback
            self.error.emit(f"{str(e)}\n{traceback.format_exc()}")


class MLFilterTab(QWidget):
    """Tab for training and evaluating ML trade filters."""

    model_trained = Signal(object)  # MLTradeFilter instance

    def __init__(self, data_dir: str, output_dir: str):
        super().__init__()

        self.data_dir = data_dir
        self.output_dir = output_dir
        self.worker = None
        self.current_model = None
        self.optimizer_params = None  # Params from optimization tab
        self.features_df = None  # Store for insights

        self._setup_ui()

    def _setup_ui(self):
        """Create the UI layout."""
        layout = QVBoxLayout(self)
        layout.setSpacing(12)

        # Top section: Training controls (simplified)
        top_layout = QHBoxLayout()

        # Hidden fields for params (still used internally but not shown)
        self.ticker_combo = QComboBox()
        self.ticker_combo.addItems(["TSLA", "QQQ", "AAPL", "NVDA", "MSFT", "SPY", "AMD"])
        self.ticker_combo.setVisible(False)

        self.ib_duration_spin = QSpinBox()
        self.ib_duration_spin.setRange(15, 60)
        self.ib_duration_spin.setValue(30)
        self.ib_duration_spin.setVisible(False)

        self.direction_combo = QComboBox()
        self.direction_combo.addItems(["both", "long_only", "short_only"])
        self.direction_combo.setVisible(False)

        self.profit_target_spin = QDoubleSpinBox()
        self.profit_target_spin.setRange(0.3, 3.0)
        self.profit_target_spin.setValue(1.0)
        self.profit_target_spin.setVisible(False)

        # Model settings (the only visible config)
        model_group = QGroupBox("Model Settings")
        model_layout = QGridLayout(model_group)
        model_layout.setSpacing(8)

        model_layout.addWidget(QLabel("Model Type:"), 0, 0)
        self.model_type_combo = QComboBox()
        self.model_type_combo.addItems(["Ensemble", "LightGBM"])
        self.model_type_combo.setToolTip(
            "Ensemble: LightGBM + Random Forest + Logistic Regression (more robust)\n"
            "LightGBM: Single model (faster, simpler)"
        )
        model_layout.addWidget(self.model_type_combo, 0, 1)

        # Probability threshold slider
        model_layout.addWidget(QLabel("Threshold:"), 1, 0)
        threshold_layout = QHBoxLayout()
        self.threshold_slider = QSlider(Qt.Horizontal)
        self.threshold_slider.setRange(50, 70)  # 0.50 to 0.70
        self.threshold_slider.setValue(55)  # Default 0.55
        self.threshold_slider.setTickPosition(QSlider.TicksBelow)
        self.threshold_slider.setTickInterval(5)
        self.threshold_slider.valueChanged.connect(self._on_threshold_changed)
        threshold_layout.addWidget(self.threshold_slider)
        self.threshold_label = QLabel("0.55")
        self.threshold_label.setMinimumWidth(35)
        self.threshold_label.setToolTip(
            "Probability threshold for filtering trades.\n"
            "Higher = more selective (fewer trades, higher precision)\n"
            "Lower = more trades but may include weaker signals"
        )
        threshold_layout.addWidget(self.threshold_label)
        model_layout.addLayout(threshold_layout, 1, 1)

        top_layout.addWidget(model_group)

        # Actions
        actions_group = QGroupBox("Actions")
        actions_layout = QVBoxLayout(actions_group)

        # Hidden train button (kept for internal use)
        self.train_button = QPushButton("Train Model")
        self.train_button.setVisible(False)
        self.train_button.clicked.connect(self._train_model)
        actions_layout.addWidget(self.train_button)

        self.train_from_best_button = QPushButton("Train from Best")
        self.train_from_best_button.setObjectName("primary")
        self.train_from_best_button.setToolTip(
            "Train using best parameters from optimization.\n"
            "Run optimization first, then click this button."
        )
        self.train_from_best_button.setEnabled(False)
        self.train_from_best_button.clicked.connect(self._train_from_optimizer)
        actions_layout.addWidget(self.train_from_best_button)

        self.save_button = QPushButton("Save Model")
        self.save_button.setEnabled(False)
        self.save_button.clicked.connect(self._save_model)
        actions_layout.addWidget(self.save_button)

        self.load_button = QPushButton("Load Model")
        self.load_button.clicked.connect(self._load_model)
        actions_layout.addWidget(self.load_button)

        top_layout.addWidget(actions_group)

        layout.addLayout(top_layout)

        # Optimizer params display (shows when params received from optimization)
        self.optimizer_params_frame = QFrame()
        self.optimizer_params_frame.setStyleSheet("""
            QFrame {
                background-color: #1a3a1a;
                border: 1px solid #2d5a2d;
                border-radius: 6px;
                padding: 8px;
            }
        """)
        optimizer_params_layout = QHBoxLayout(self.optimizer_params_frame)
        optimizer_params_layout.setContentsMargins(8, 4, 8, 4)
        self.optimizer_params_label = QLabel("")
        self.optimizer_params_label.setStyleSheet("color: #88ff88;")
        optimizer_params_layout.addWidget(self.optimizer_params_label)
        self.optimizer_params_frame.setVisible(False)
        layout.addWidget(self.optimizer_params_frame)

        # Status bar
        status_frame = QFrame()
        status_frame.setStyleSheet("""
            QFrame {
                background-color: #252525;
                border: 1px solid #333333;
                border-radius: 6px;
                padding: 8px;
            }
        """)
        status_layout = QHBoxLayout(status_frame)
        status_layout.setContentsMargins(8, 4, 8, 4)

        self.status_label = QLabel("Run optimization first, then use 'Train from Best' to train ML model")
        self.status_label.setStyleSheet("color: #888888;")
        status_layout.addWidget(self.status_label)

        layout.addWidget(status_frame)

        # Results section
        results_splitter = QSplitter(Qt.Horizontal)

        # Left: Model metrics
        metrics_frame = QFrame()
        metrics_layout = QVBoxLayout(metrics_frame)
        metrics_layout.setContentsMargins(0, 0, 0, 0)

        metrics_label = QLabel("Model Performance")
        metrics_label.setStyleSheet("font-weight: bold; font-size: 14px;")
        metrics_layout.addWidget(metrics_label)

        # Metric cards grid with tooltips
        cards_grid = QGridLayout()
        cards_grid.setSpacing(8)

        self.accuracy_card = MetricCard("Accuracy (?)")
        self.accuracy_card.setToolTip(METRIC_TOOLTIPS['accuracy'])
        cards_grid.addWidget(self.accuracy_card, 0, 0)

        self.precision_card = MetricCard("Precision (?)")
        self.precision_card.setToolTip(METRIC_TOOLTIPS['precision'])
        cards_grid.addWidget(self.precision_card, 0, 1)

        self.recall_card = MetricCard("Recall (?)")
        self.recall_card.setToolTip(METRIC_TOOLTIPS['recall'])
        cards_grid.addWidget(self.recall_card, 1, 0)

        self.f1_card = MetricCard("F1 Score (?)")
        self.f1_card.setToolTip(METRIC_TOOLTIPS['f1'])
        cards_grid.addWidget(self.f1_card, 1, 1)

        self.auc_card = MetricCard("ROC AUC (?)")
        self.auc_card.setToolTip(METRIC_TOOLTIPS['roc_auc'])
        cards_grid.addWidget(self.auc_card, 2, 0)

        self.cv_card = MetricCard("CV Mean (?)")
        self.cv_card.setToolTip(METRIC_TOOLTIPS['cv_mean'])
        cards_grid.addWidget(self.cv_card, 2, 1)

        metrics_layout.addLayout(cards_grid)

        # Confusion matrix display with explanation
        cm_label = QLabel("Confusion Matrix (?)")
        cm_label.setStyleSheet("font-weight: bold; margin-top: 12px;")
        cm_label.setToolTip(
            "Confusion Matrix Explained:\n\n"
            "TN (Pred Loss, Actual Loss) = Correctly avoided losing trade (GREEN)\n"
            "FP (Pred Win, Actual Loss) = Took trade expecting win, got loss - COSTLY! (RED)\n"
            "FN (Pred Loss, Actual Win) = Skipped trade that would have won - opportunity cost (YELLOW)\n"
            "TP (Pred Win, Actual Win) = Correctly took winning trade (GREEN)\n\n"
            "For trading, minimizing FP (false positives) is critical - these are costly bad trades."
        )
        metrics_layout.addWidget(cm_label)

        self.cm_table = QTableWidget(2, 2)
        self.cm_table.setHorizontalHeaderLabels(["Pred Loss", "Pred Win"])
        self.cm_table.setVerticalHeaderLabels(["Actual Loss", "Actual Win"])
        self.cm_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.cm_table.verticalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.cm_table.setMinimumHeight(120)
        self.cm_table.setMaximumHeight(150)
        self.cm_table.setStyleSheet("""
            QTableWidget {
                background-color: #252525;
                gridline-color: #444444;
                font-size: 14px;
                font-weight: bold;
            }
            QTableWidget::item {
                padding: 8px;
                text-align: center;
            }
            QHeaderView::section {
                background-color: #333333;
                color: #cccccc;
                padding: 6px;
                border: 1px solid #444444;
                font-weight: bold;
            }
        """)
        metrics_layout.addWidget(self.cm_table)

        # Training info
        self.info_label = QLabel("")
        self.info_label.setStyleSheet("color: #888888; font-size: 11px;")
        metrics_layout.addWidget(self.info_label)

        metrics_layout.addStretch()

        results_splitter.addWidget(metrics_frame)

        # Right side: Feature importance chart + Insights panel (vertical split)
        right_splitter = QSplitter(Qt.Vertical)

        # Feature importance chart
        chart_frame = QFrame()
        chart_frame.setStyleSheet("""
            QFrame {
                border: 1px solid #333333;
                border-radius: 4px;
                background-color: #1e1e1e;
            }
        """)
        chart_layout = QVBoxLayout(chart_frame)
        chart_layout.setContentsMargins(4, 4, 4, 4)

        chart_label = QLabel("Feature Importance")
        chart_label.setStyleSheet("font-weight: bold; font-size: 14px; color: #cccccc;")
        chart_layout.addWidget(chart_label)

        self.importance_chart = QWebEngineView()
        self.importance_chart.setStyleSheet("background-color: #1e1e1e;")
        chart_layout.addWidget(self.importance_chart)

        right_splitter.addWidget(chart_frame)

        # Insights panel
        insights_frame = QFrame()
        insights_frame.setStyleSheet("""
            QFrame {
                border: 1px solid #333333;
                border-radius: 4px;
                background-color: #1e1e1e;
            }
        """)
        insights_layout = QVBoxLayout(insights_frame)
        insights_layout.setContentsMargins(8, 8, 8, 8)

        insights_header = QLabel("ML Insights & Recommendations")
        insights_header.setStyleSheet("font-weight: bold; font-size: 14px; color: #ffcc00;")
        insights_layout.addWidget(insights_header)

        self.insights_text = QTextEdit()
        self.insights_text.setReadOnly(True)
        self.insights_text.setStyleSheet("""
            QTextEdit {
                background-color: #252525;
                color: #cccccc;
                border: 1px solid #444444;
                border-radius: 4px;
                padding: 8px;
                font-size: 12px;
            }
        """)
        self.insights_text.setPlaceholderText(
            "Train a model to see insights and recommendations based on the data."
        )
        self.insights_text.setMinimumHeight(100)
        insights_layout.addWidget(self.insights_text)

        right_splitter.addWidget(insights_frame)
        right_splitter.setSizes([350, 150])

        results_splitter.addWidget(right_splitter)
        results_splitter.setSizes([300, 500])

        layout.addWidget(results_splitter, 1)

    def set_data_dir(self, path: str):
        """Update data directory."""
        self.data_dir = path

    def set_output_dir(self, path: str):
        """Update output directory."""
        self.output_dir = path

    def _train_model(self):
        """Start model training."""
        self.train_button.setEnabled(False)
        self.train_from_best_button.setEnabled(False)
        self.status_label.setText("Starting training...")
        self.status_label.setStyleSheet("color: #2a82da;")

        # Clear previous results
        self._clear_results()

        params = {
            'ib_duration': self.ib_duration_spin.value(),
            'profit_target': self.profit_target_spin.value(),
            'direction': self.direction_combo.currentText(),
            'use_qqq_filter': False,  # Could add checkbox for this
            'prior_days_lookback': 3,
            'daily_range_lookback': 5
        }

        use_ensemble = self.model_type_combo.currentText() == "Ensemble"
        threshold = self.threshold_slider.value() / 100.0

        self.worker = TrainingWorker(
            self.data_dir,
            self.ticker_combo.currentText(),
            params,
            use_ensemble=use_ensemble,
            threshold=threshold
        )
        self.worker.progress.connect(self._on_progress)
        self.worker.finished.connect(self._on_finished)
        self.worker.error.connect(self._on_error)
        self.worker.start()

    def _train_from_optimizer(self):
        """Train using best parameters from optimization."""
        if self.optimizer_params is None:
            QMessageBox.warning(
                self, "No Optimizer Results",
                "Run optimization first to get best parameters.\n"
                "Go to the Optimization tab and run an optimization, "
                "then return here to train from those results."
            )
            return

        # Apply optimizer params to UI
        ticker = self.optimizer_params.get('ticker', 'TSLA')
        idx = self.ticker_combo.findText(ticker)
        if idx >= 0:
            self.ticker_combo.setCurrentIndex(idx)

        ib_duration = self.optimizer_params.get('ib_duration_minutes', 30)
        self.ib_duration_spin.setValue(ib_duration)

        direction = self.optimizer_params.get('trade_direction', 'both')
        idx = self.direction_combo.findText(direction)
        if idx >= 0:
            self.direction_combo.setCurrentIndex(idx)

        profit_target = self.optimizer_params.get('profit_target_percent', 1.0)
        self.profit_target_spin.setValue(profit_target)

        # Now train with these params
        self._train_model()

    def _on_threshold_changed(self, value: int):
        """Update threshold label when slider changes."""
        self.threshold_label.setText(f"{value / 100:.2f}")

    def _on_progress(self, message: str):
        """Handle progress update."""
        self.status_label.setText(message)

    def _on_finished(self, results: dict):
        """Handle training completion."""
        self.train_button.setEnabled(True)
        self.save_button.setEnabled(True)
        if self.optimizer_params:
            self.train_from_best_button.setEnabled(True)

        self.current_model = results['ml_filter']
        training_result = results['training_result']
        self.features_df = results.get('features_df')

        # Show model type in status
        model_type = getattr(training_result, 'model_type', 'lightgbm')
        self.status_label.setText(
            f"Training complete! {results['n_samples']} samples, "
            f"{training_result.accuracy:.1%} accuracy ({model_type})"
        )
        self.status_label.setStyleSheet("color: #00ff00;")

        # Update metric cards
        self._update_metrics(training_result)

        # Update feature importance chart
        self._update_importance_chart(training_result.feature_importance)

        # Update insights panel
        self._update_insights(training_result)

        # Update info label
        self.info_label.setText(
            f"Trained on {training_result.n_samples} samples "
            f"({training_result.n_winners} wins, {training_result.n_losers} losses)\n"
            f"Train date: {training_result.train_date}"
        )

        # Emit signal
        self.model_trained.emit(self.current_model)

    def _on_error(self, error_msg: str):
        """Handle training error."""
        self.train_button.setEnabled(True)
        if self.optimizer_params:
            self.train_from_best_button.setEnabled(True)
        self.status_label.setText(f"Error: {error_msg[:100]}")
        self.status_label.setStyleSheet("color: #ff4444;")

    def _clear_results(self):
        """Clear previous results."""
        for card in [self.accuracy_card, self.precision_card, self.recall_card,
                     self.f1_card, self.auc_card, self.cv_card]:
            card.set_value("--")

        for i in range(2):
            for j in range(2):
                self.cm_table.setItem(i, j, QTableWidgetItem("--"))

        self.info_label.setText("")
        self.importance_chart.setHtml("")
        self.insights_text.clear()

    def _update_metrics(self, result):
        """Update metric cards with training results."""
        # Color based on value quality
        def get_color(value, good_threshold=0.6, great_threshold=0.75):
            if value >= great_threshold:
                return "#00ff00"
            elif value >= good_threshold:
                return "#ffaa00"
            else:
                return "#ff4444"

        self.accuracy_card.set_value(
            f"{result.accuracy:.1%}",
            get_color(result.accuracy)
        )
        self.precision_card.set_value(
            f"{result.precision:.1%}",
            get_color(result.precision)
        )
        self.recall_card.set_value(
            f"{result.recall:.1%}",
            get_color(result.recall)
        )
        self.f1_card.set_value(
            f"{result.f1:.3f}",
            get_color(result.f1)
        )
        self.auc_card.set_value(
            f"{result.roc_auc:.3f}",
            get_color(result.roc_auc, 0.55, 0.65)
        )
        self.cv_card.set_value(
            f"{result.cv_mean:.1%} (±{result.cv_std:.1%})",
            get_color(result.cv_mean)
        )

        # Update confusion matrix with color coding
        cm = result.confusion_matrix

        # TN (True Negative) - Correctly predicted loss - green
        tn_item = QTableWidgetItem(str(cm[0, 0]))
        tn_item.setTextAlignment(Qt.AlignCenter)
        tn_item.setBackground(Qt.darkGreen)
        self.cm_table.setItem(0, 0, tn_item)

        # FP (False Positive) - Predicted win but was loss - red
        fp_item = QTableWidgetItem(str(cm[0, 1]))
        fp_item.setTextAlignment(Qt.AlignCenter)
        fp_item.setBackground(Qt.darkRed)
        self.cm_table.setItem(0, 1, fp_item)

        # FN (False Negative) - Predicted loss but was win - orange/yellow
        fn_item = QTableWidgetItem(str(cm[1, 0]))
        fn_item.setTextAlignment(Qt.AlignCenter)
        fn_item.setBackground(Qt.darkYellow)
        self.cm_table.setItem(1, 0, fn_item)

        # TP (True Positive) - Correctly predicted win - green
        tp_item = QTableWidgetItem(str(cm[1, 1]))
        tp_item.setTextAlignment(Qt.AlignCenter)
        tp_item.setBackground(Qt.darkGreen)
        self.cm_table.setItem(1, 1, tp_item)

    def _update_importance_chart(self, importance: dict):
        """Update feature importance bar chart."""
        import plotly.graph_objects as go

        # Sort by importance
        sorted_items = sorted(importance.items(), key=lambda x: x[1], reverse=True)
        names = [item[0] for item in sorted_items]
        values = [item[1] for item in sorted_items]

        # Create horizontal bar chart
        fig = go.Figure(go.Bar(
            x=values,
            y=names,
            orientation='h',
            marker_color='#2a82da'
        ))

        fig.update_layout(
            paper_bgcolor='#1e1e1e',
            plot_bgcolor='#1e1e1e',
            font=dict(color='#cccccc', size=10),
            margin=dict(l=150, r=20, t=10, b=30),
            xaxis=dict(
                title="Importance",
                gridcolor='#333333',
                showgrid=True
            ),
            yaxis=dict(
                gridcolor='#333333',
                showgrid=False,
                autorange='reversed'  # Highest at top
            ),
            height=350
        )

        html = fig.to_html(include_plotlyjs='cdn', full_html=True)
        html = html.replace('<body>', '<body style="background-color: #1e1e1e; margin: 0;">')
        self.importance_chart.setHtml(html)

    def _update_insights(self, training_result):
        """Update insights panel with actionable recommendations."""
        insights = getattr(training_result, 'insights', None)

        if not insights:
            self.insights_text.setPlainText(
                "No insights available. The model may need more data to generate recommendations."
            )
            return

        # Format insights with bullet points
        formatted = []
        for insight in insights:
            formatted.append(f"• {insight}")

        self.insights_text.setPlainText("\n\n".join(formatted))

    def set_optimizer_params(self, params: dict, ticker: str):
        """
        Receive best parameters from optimization tab.

        This is called by main_window when optimization completes.
        Enables the 'Train from Best' button and shows params.
        """
        self.optimizer_params = params.copy()
        self.optimizer_params['ticker'] = ticker

        # Show optimizer params in the UI
        ib_dur = params.get('ib_duration_minutes', 30)
        direction = params.get('trade_direction', 'both')
        target = params.get('profit_target_percent', 1.0)
        win_rate = params.get('win_rate', 0)

        # win_rate comes as 0-100 (percent), not 0-1, so don't use :.0%
        self.optimizer_params_label.setText(
            f"Params from optimization: {ticker}, {ib_dur}min IB, "
            f"{direction}, {target:.1f}% target, {win_rate:.0f}% win rate"
        )
        self.optimizer_params_frame.setVisible(True)

        # Enable the Train from Best button
        self.train_from_best_button.setEnabled(True)

        # Update status
        self.status_label.setText(
            f"Optimization results received. Click 'Train from Best' to train ML model."
        )
        self.status_label.setStyleSheet("color: #88ff88;")

    def _save_model(self):
        """Save trained model to file."""
        if self.current_model is None:
            return

        # Create models directory
        models_dir = Path(self.output_dir) / "models"
        models_dir.mkdir(parents=True, exist_ok=True)

        # Generate filename
        ticker = self.ticker_combo.currentText()
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        default_name = f"ml_filter_{ticker}_{timestamp}.pkl"

        filepath, _ = QFileDialog.getSaveFileName(
            self, "Save Model",
            str(models_dir / default_name),
            "Pickle Files (*.pkl)"
        )

        if filepath:
            try:
                self.current_model.save(filepath)
                self.status_label.setText(f"Model saved to {Path(filepath).name}")
                self.status_label.setStyleSheet("color: #00ff00;")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to save model: {e}")

    def _load_model(self):
        """Load model from file."""
        models_dir = Path(self.output_dir) / "models"

        filepath, _ = QFileDialog.getOpenFileName(
            self, "Load Model",
            str(models_dir) if models_dir.exists() else "",
            "Pickle Files (*.pkl)"
        )

        if filepath:
            try:
                from ml_filter.model_trainer import MLTradeFilter
                self.current_model = MLTradeFilter.load(filepath)

                # Update UI with loaded model info
                if self.current_model.training_result:
                    self._update_metrics(self.current_model.training_result)
                    self._update_importance_chart(
                        self.current_model.training_result.feature_importance
                    )
                    self.info_label.setText(
                        f"Loaded model for {self.current_model.ticker}\n"
                        f"Train date: {self.current_model.train_date}"
                    )

                self.save_button.setEnabled(True)
                self.status_label.setText(f"Loaded model from {Path(filepath).name}")
                self.status_label.setStyleSheet("color: #00ff00;")

                # Emit signal
                self.model_trained.emit(self.current_model)

            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to load model: {e}")

    def get_current_model(self):
        """Get the currently loaded/trained model."""
        return self.current_model
