"""
ML Filter Tab - Train and evaluate ML trade filter models.

Features:
- Run backtest to generate training data
- Train LightGBM classifier
- View model metrics (accuracy, precision, recall, ROC AUC)
- View feature importance chart
- Save/load trained models
"""

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QLabel, QComboBox, QPushButton, QGroupBox,
    QTableWidget, QTableWidgetItem, QHeaderView, QProgressBar,
    QSplitter, QFrame, QSpinBox, QDoubleSpinBox, QFileDialog,
    QMessageBox
)
from PySide6.QtCore import Qt, Signal, QThread
from PySide6.QtWebEngineWidgets import QWebEngineView

from desktop_ui.widgets.metrics_panel import MetricCard
from pathlib import Path
import numpy as np


class TrainingWorker(QThread):
    """Background worker for ML training."""

    progress = Signal(str)  # status message
    finished = Signal(dict)  # results dict
    error = Signal(str)

    def __init__(self, data_dir: str, ticker: str, params: dict):
        super().__init__()
        self.data_dir = data_dir
        self.ticker = ticker
        self.params = params

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

            features_df = feature_builder.build_features_from_backtest(
                trade_dicts, bars_array, timestamps,
                ib_duration_minutes=self.params.get('ib_duration', 30),
                qqq_filter_used=self.params.get('use_qqq_filter', False)
            )

            if len(features_df) < 20:
                self.error.emit(f"Could not extract enough features: {len(features_df)} samples")
                return

            self.progress.emit(f"Training ML model on {len(features_df)} samples...")

            # Step 4: Train model
            ml_filter = MLTradeFilter()
            result = ml_filter.train(features_df, ticker=self.ticker)

            self.progress.emit("Training complete!")

            # Return results
            self.finished.emit({
                'ml_filter': ml_filter,
                'training_result': result,
                'n_trades': len(trades),
                'n_samples': len(features_df)
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

        self._setup_ui()

    def _setup_ui(self):
        """Create the UI layout."""
        layout = QVBoxLayout(self)
        layout.setSpacing(12)

        # Top section: Training controls
        top_layout = QHBoxLayout()

        # Data settings
        data_group = QGroupBox("Training Data")
        data_layout = QGridLayout(data_group)
        data_layout.setSpacing(8)

        data_layout.addWidget(QLabel("Ticker:"), 0, 0)
        self.ticker_combo = QComboBox()
        self.ticker_combo.addItems(["TSLA", "QQQ", "AAPL", "NVDA", "MSFT", "SPY", "AMD"])
        data_layout.addWidget(self.ticker_combo, 0, 1)

        data_layout.addWidget(QLabel("IB Duration:"), 1, 0)
        self.ib_duration_spin = QSpinBox()
        self.ib_duration_spin.setRange(15, 60)
        self.ib_duration_spin.setValue(30)
        self.ib_duration_spin.setSuffix(" min")
        data_layout.addWidget(self.ib_duration_spin, 1, 1)

        data_layout.addWidget(QLabel("Direction:"), 2, 0)
        self.direction_combo = QComboBox()
        self.direction_combo.addItems(["both", "long_only", "short_only"])
        data_layout.addWidget(self.direction_combo, 2, 1)

        top_layout.addWidget(data_group)

        # Training settings
        train_group = QGroupBox("Training Settings")
        train_layout = QGridLayout(train_group)
        train_layout.setSpacing(8)

        train_layout.addWidget(QLabel("Profit Target:"), 0, 0)
        self.profit_target_spin = QDoubleSpinBox()
        self.profit_target_spin.setRange(0.3, 3.0)
        self.profit_target_spin.setValue(1.0)
        self.profit_target_spin.setSingleStep(0.1)
        self.profit_target_spin.setSuffix("%")
        train_layout.addWidget(self.profit_target_spin, 0, 1)

        train_layout.addWidget(QLabel("Prior Days:"), 1, 0)
        self.prior_days_spin = QSpinBox()
        self.prior_days_spin.setRange(1, 10)
        self.prior_days_spin.setValue(3)
        train_layout.addWidget(self.prior_days_spin, 1, 1)

        train_layout.addWidget(QLabel("Range Days:"), 2, 0)
        self.range_days_spin = QSpinBox()
        self.range_days_spin.setRange(1, 20)
        self.range_days_spin.setValue(5)
        train_layout.addWidget(self.range_days_spin, 2, 1)

        top_layout.addWidget(train_group)

        # Actions
        actions_group = QGroupBox("Actions")
        actions_layout = QVBoxLayout(actions_group)

        self.train_button = QPushButton("Train Model")
        self.train_button.setObjectName("primary")
        self.train_button.clicked.connect(self._train_model)
        actions_layout.addWidget(self.train_button)

        self.save_button = QPushButton("Save Model")
        self.save_button.setEnabled(False)
        self.save_button.clicked.connect(self._save_model)
        actions_layout.addWidget(self.save_button)

        self.load_button = QPushButton("Load Model")
        self.load_button.clicked.connect(self._load_model)
        actions_layout.addWidget(self.load_button)

        top_layout.addWidget(actions_group)

        layout.addLayout(top_layout)

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

        self.status_label = QLabel("Ready - Select ticker and click 'Train Model'")
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

        # Metric cards grid
        cards_grid = QGridLayout()
        cards_grid.setSpacing(8)

        self.accuracy_card = MetricCard("Accuracy")
        cards_grid.addWidget(self.accuracy_card, 0, 0)

        self.precision_card = MetricCard("Precision")
        cards_grid.addWidget(self.precision_card, 0, 1)

        self.recall_card = MetricCard("Recall")
        cards_grid.addWidget(self.recall_card, 1, 0)

        self.f1_card = MetricCard("F1 Score")
        cards_grid.addWidget(self.f1_card, 1, 1)

        self.auc_card = MetricCard("ROC AUC")
        cards_grid.addWidget(self.auc_card, 2, 0)

        self.cv_card = MetricCard("CV Mean")
        cards_grid.addWidget(self.cv_card, 2, 1)

        metrics_layout.addLayout(cards_grid)

        # Confusion matrix display
        cm_label = QLabel("Confusion Matrix")
        cm_label.setStyleSheet("font-weight: bold; margin-top: 12px;")
        metrics_layout.addWidget(cm_label)

        self.cm_table = QTableWidget(2, 2)
        self.cm_table.setHorizontalHeaderLabels(["Pred Loss", "Pred Win"])
        self.cm_table.setVerticalHeaderLabels(["Actual Loss", "Actual Win"])
        self.cm_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.cm_table.verticalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.cm_table.setMaximumHeight(100)
        metrics_layout.addWidget(self.cm_table)

        # Training info
        self.info_label = QLabel("")
        self.info_label.setStyleSheet("color: #888888; font-size: 11px;")
        metrics_layout.addWidget(self.info_label)

        metrics_layout.addStretch()

        results_splitter.addWidget(metrics_frame)

        # Right: Feature importance chart
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

        results_splitter.addWidget(chart_frame)
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
        self.status_label.setText("Starting training...")
        self.status_label.setStyleSheet("color: #2a82da;")

        # Clear previous results
        self._clear_results()

        params = {
            'ib_duration': self.ib_duration_spin.value(),
            'profit_target': self.profit_target_spin.value(),
            'direction': self.direction_combo.currentText(),
            'use_qqq_filter': False,  # Could add checkbox for this
            'prior_days_lookback': self.prior_days_spin.value(),
            'daily_range_lookback': self.range_days_spin.value()
        }

        self.worker = TrainingWorker(
            self.data_dir,
            self.ticker_combo.currentText(),
            params
        )
        self.worker.progress.connect(self._on_progress)
        self.worker.finished.connect(self._on_finished)
        self.worker.error.connect(self._on_error)
        self.worker.start()

    def _on_progress(self, message: str):
        """Handle progress update."""
        self.status_label.setText(message)

    def _on_finished(self, results: dict):
        """Handle training completion."""
        self.train_button.setEnabled(True)
        self.save_button.setEnabled(True)

        self.current_model = results['ml_filter']
        training_result = results['training_result']

        self.status_label.setText(
            f"Training complete! {results['n_samples']} samples, "
            f"{training_result.accuracy:.1%} accuracy"
        )
        self.status_label.setStyleSheet("color: #00ff00;")

        # Update metric cards
        self._update_metrics(training_result)

        # Update feature importance chart
        self._update_importance_chart(training_result.feature_importance)

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

        # Update confusion matrix
        cm = result.confusion_matrix
        self.cm_table.setItem(0, 0, QTableWidgetItem(str(cm[0, 0])))  # TN
        self.cm_table.setItem(0, 1, QTableWidgetItem(str(cm[0, 1])))  # FP
        self.cm_table.setItem(1, 0, QTableWidgetItem(str(cm[1, 0])))  # FN
        self.cm_table.setItem(1, 1, QTableWidgetItem(str(cm[1, 1])))  # TP

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
