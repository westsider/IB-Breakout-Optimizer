"""
Download Tab - Download and update market data from Polygon.io API.
"""

import os
from pathlib import Path
from datetime import datetime, timedelta
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QLabel, QComboBox, QLineEdit, QPushButton, QGroupBox,
    QTableWidget, QTableWidgetItem, QHeaderView, QFrame,
    QMessageBox
)
from PySide6.QtCore import Qt, QThread, Signal, QSettings


class DownloadWorker(QThread):
    """Worker thread for downloading data from Polygon.io."""

    progress = Signal(str)  # Status message
    finished = Signal(bool, str)  # Success, message

    def __init__(self, ticker: str, api_key: str, output_dir: str,
                 start_date: datetime, end_date: datetime, mode: str = "download"):
        super().__init__()
        self.ticker = ticker
        self.api_key = api_key
        self.output_dir = output_dir
        self.start_date = start_date
        self.end_date = end_date
        self.mode = mode  # "download" or "update"
        self._cancelled = False

    def cancel(self):
        self._cancelled = True

    def run(self):
        """Download data from Polygon.io API."""
        import requests
        import time

        all_results = []
        current_date = self.start_date

        total_days = (self.end_date - self.start_date).days
        chunks_total = (total_days // 30) + 1
        chunks_done = 0
        start_time = time.time()

        self.progress.emit(f"Starting download for {self.ticker} ({total_days} days, ~{chunks_total} API calls)...")

        while current_date < self.end_date and not self._cancelled:
            chunk_end = min(current_date + timedelta(days=30), self.end_date)

            url = (
                f"https://api.polygon.io/v2/aggs/ticker/{self.ticker}/range/1/minute/"
                f"{current_date.strftime('%Y-%m-%d')}/{chunk_end.strftime('%Y-%m-%d')}"
                f"?adjusted=true&sort=asc&limit=50000&apiKey={self.api_key}"
            )

            try:
                response = requests.get(url, timeout=30)
                data = response.json()

                if data.get('status') in ['OK', 'DELAYED'] and data.get('results'):
                    all_results.extend(data['results'])
                    bars_count = len(data['results'])
                    chunks_done += 1
                    elapsed = time.time() - start_time

                    # Calculate ETA
                    if chunks_done > 0:
                        avg_time_per_chunk = elapsed / chunks_done
                        chunks_remaining = chunks_total - chunks_done
                        eta_seconds = chunks_remaining * avg_time_per_chunk
                        eta_str = self._format_time(eta_seconds)
                        elapsed_str = self._format_time(elapsed)
                    else:
                        eta_str = "calculating..."
                        elapsed_str = "0s"

                    self.progress.emit(
                        f"[{chunks_done}/{chunks_total}] Downloaded {bars_count:,} bars "
                        f"({current_date.strftime('%m/%d/%y')} - {chunk_end.strftime('%m/%d/%y')}) | "
                        f"{len(all_results):,} total | Elapsed: {elapsed_str} | ETA: {eta_str}"
                    )
                elif data.get('status') == 'ERROR':
                    error_msg = data.get('error', 'Unknown error')
                    self.progress.emit(f"API Error: {error_msg}")
                    chunks_done += 1
                else:
                    self.progress.emit(
                        f"No data for {current_date.strftime('%m/%d/%y')} to {chunk_end.strftime('%m/%d/%y')}"
                    )
                    chunks_done += 1

            except requests.exceptions.Timeout:
                self.progress.emit(f"Request timeout - retrying...")
                time.sleep(5)
                continue
            except Exception as e:
                self.progress.emit(f"Request error: {e}")
                chunks_done += 1

            current_date = chunk_end + timedelta(days=1)

            # Rate limiting - free tier is 5 calls/minute
            # Break sleep into 1-second chunks for responsive UI
            if current_date < self.end_date and not self._cancelled:
                for i in range(13, 0, -1):
                    if self._cancelled:
                        break
                    elapsed = time.time() - start_time
                    elapsed_str = self._format_time(elapsed)
                    self.progress.emit(
                        f"[{chunks_done}/{chunks_total}] Rate limit pause ({i}s)... "
                        f"{len(all_results):,} bars | Elapsed: {elapsed_str}"
                    )
                    time.sleep(1)

        # After the while loop - handle completion
        if self._cancelled:
            self.finished.emit(False, "Download cancelled")
            return

        if not all_results:
            self.finished.emit(False, f"No data downloaded for {self.ticker}")
            return

        # Save to file with simple naming convention (TICKER_NT.txt)
        self.progress.emit(f"Saving {len(all_results):,} bars to file...")

        output_path = Path(self.output_dir) / f"{self.ticker}_NT.txt"

        # If updating, we need to merge with existing data
        if self.mode == "update" and output_path.exists():
            self.progress.emit("Merging with existing data...")
            existing_data = self._load_existing_data(output_path)
            all_results = self._merge_data(existing_data, all_results)

        # Sort by timestamp and write
        all_results.sort(key=lambda x: x['t'])

        with open(output_path, 'w') as f:
            for bar in all_results:
                ts = datetime.fromtimestamp(bar['t'] / 1000)
                dt_str = ts.strftime('%Y%m%d %H%M%S')
                f.write(f"{dt_str};{bar['o']};{bar['h']};{bar['l']};{bar['c']};{int(bar['v'])};0\n")

        self.finished.emit(True, f"Saved {len(all_results):,} bars to {output_path.name}")

    def _format_time(self, seconds: float) -> str:
        """Format seconds into human-readable time."""
        if seconds < 60:
            return f"{seconds:.0f}s"
        elif seconds < 3600:
            mins = int(seconds // 60)
            secs = int(seconds % 60)
            return f"{mins}m {secs}s"
        else:
            hours = int(seconds // 3600)
            mins = int((seconds % 3600) // 60)
            return f"{hours}h {mins}m"

    def _load_existing_data(self, filepath: Path) -> list:
        """Load existing NT format data as list of dicts."""
        results = []
        try:
            with open(filepath, 'r') as f:
                for line in f:
                    parts = line.strip().split(';')
                    if len(parts) >= 6:
                        dt_str = parts[0]
                        dt = datetime.strptime(dt_str, '%Y%m%d %H%M%S')
                        results.append({
                            't': int(dt.timestamp() * 1000),
                            'o': float(parts[1]),
                            'h': float(parts[2]),
                            'l': float(parts[3]),
                            'c': float(parts[4]),
                            'v': int(parts[5])
                        })
        except Exception as e:
            self.progress.emit(f"Error loading existing data: {e}")
        return results

    def _merge_data(self, existing: list, new: list) -> list:
        """Merge existing and new data, removing duplicates."""
        # Use timestamp as key
        existing_timestamps = {bar['t'] for bar in existing}
        merged = existing.copy()

        new_count = 0
        for bar in new:
            if bar['t'] not in existing_timestamps:
                merged.append(bar)
                new_count += 1

        self.progress.emit(f"Added {new_count:,} new bars to existing {len(existing):,} bars")
        return merged


class DownloadTab(QWidget):
    """Tab for downloading and updating market data."""

    def __init__(self, data_dir: str):
        super().__init__()
        self.data_dir = data_dir
        self.worker = None
        self.settings = QSettings("TradingTools", "IBBreakoutOptimizer")

        self._setup_ui()
        self._load_settings()
        self._update_data_info()

    def _setup_ui(self):
        """Create the UI layout."""
        layout = QVBoxLayout(self)
        layout.setSpacing(12)

        # Settings section
        settings_group = QGroupBox("Download Settings")
        settings_layout = QGridLayout(settings_group)
        settings_layout.setSpacing(8)

        # Ticker selection
        settings_layout.addWidget(QLabel("Ticker:"), 0, 0)
        self.ticker_combo = QComboBox()
        self.ticker_combo.addItems(["TSLA", "QQQ", "AAPL", "NVDA", "MSFT", "SPY", "AMD", "AMZN", "GOOGL", "META"])
        self.ticker_combo.setEditable(True)
        self.ticker_combo.currentTextChanged.connect(self._update_data_info)
        settings_layout.addWidget(self.ticker_combo, 0, 1)

        # Data info label
        self.data_info_label = QLabel("")
        self.data_info_label.setStyleSheet("color: #888888; font-size: 11px;")
        settings_layout.addWidget(self.data_info_label, 0, 2)

        # API Key
        settings_layout.addWidget(QLabel("API Key:"), 1, 0)
        self.api_key_input = QLineEdit()
        self.api_key_input.setEchoMode(QLineEdit.Password)
        self.api_key_input.setPlaceholderText("Enter Polygon.io API key")
        self.api_key_input.editingFinished.connect(self._save_settings)  # Save when focus leaves
        settings_layout.addWidget(self.api_key_input, 1, 1, 1, 2)

        # Show/hide API key button
        self.show_key_btn = QPushButton("Show")
        self.show_key_btn.setMaximumWidth(60)
        self.show_key_btn.clicked.connect(self._toggle_api_key_visibility)
        settings_layout.addWidget(self.show_key_btn, 1, 3)

        # API key help
        api_help = QLabel('<a href="https://polygon.io/">Get free API key at polygon.io</a>')
        api_help.setOpenExternalLinks(True)
        api_help.setStyleSheet("color: #888888; font-size: 10px;")
        settings_layout.addWidget(api_help, 2, 1, 1, 2)

        layout.addWidget(settings_group)

        # Action buttons
        button_frame = QFrame()
        button_layout = QHBoxLayout(button_frame)
        button_layout.setContentsMargins(0, 0, 0, 0)

        self.download_btn = QPushButton("Download (1 Year)")
        self.download_btn.setObjectName("primary")
        self.download_btn.setMinimumHeight(36)
        self.download_btn.setMinimumWidth(150)
        self.download_btn.clicked.connect(self._start_download)
        button_layout.addWidget(self.download_btn)

        self.update_btn = QPushButton("Update (Latest)")
        self.update_btn.setMinimumHeight(36)
        self.update_btn.setMinimumWidth(150)
        self.update_btn.clicked.connect(self._start_update)
        button_layout.addWidget(self.update_btn)

        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.setMinimumHeight(36)
        self.cancel_btn.setEnabled(False)
        self.cancel_btn.clicked.connect(self._cancel_download)
        button_layout.addWidget(self.cancel_btn)

        # Separator
        sep = QLabel("|")
        sep.setStyleSheet("color: #444444;")
        button_layout.addWidget(sep)

        # Rebuild stats button
        self.rebuild_stats_btn = QPushButton("Rebuild Stats")
        self.rebuild_stats_btn.setMinimumHeight(36)
        self.rebuild_stats_btn.setToolTip(
            "Rebuild distribution statistics (gap %, range %) for this ticker.\n"
            "Stats are used by statistical filters in the optimizer."
        )
        self.rebuild_stats_btn.clicked.connect(self._rebuild_stats)
        button_layout.addWidget(self.rebuild_stats_btn)

        button_layout.addStretch()
        layout.addWidget(button_frame)

        # Distribution Stats section
        stats_group = QGroupBox("Distribution Statistics")
        stats_layout = QVBoxLayout(stats_group)

        self.stats_label = QLabel("No stats cached for this ticker")
        self.stats_label.setStyleSheet("color: #888888;")
        self.stats_label.setWordWrap(True)
        stats_layout.addWidget(self.stats_label)

        layout.addWidget(stats_group)

        # Progress section
        progress_frame = QFrame()
        progress_frame.setStyleSheet("""
            QFrame {
                background-color: #252525;
                border: 1px solid #333333;
                border-radius: 6px;
                padding: 8px;
            }
        """)
        progress_layout = QVBoxLayout(progress_frame)
        progress_layout.setContentsMargins(12, 8, 12, 8)

        self.progress_label = QLabel("Ready to download")
        self.progress_label.setStyleSheet("color: #888888;")
        self.progress_label.setWordWrap(True)
        progress_layout.addWidget(self.progress_label)

        layout.addWidget(progress_frame)

        # Data preview section
        preview_group = QGroupBox("Data Preview (Latest 10 Days)")
        preview_layout = QVBoxLayout(preview_group)

        self.preview_table = QTableWidget()
        self.preview_table.setColumnCount(6)
        self.preview_table.setHorizontalHeaderLabels([
            "Date", "Bars", "Open", "High", "Low", "Close"
        ])
        self.preview_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.preview_table.setAlternatingRowColors(True)
        self.preview_table.verticalHeader().setVisible(False)
        self.preview_table.setMaximumHeight(300)
        preview_layout.addWidget(self.preview_table)

        layout.addWidget(preview_group)
        layout.addStretch()

    def _toggle_api_key_visibility(self):
        """Toggle API key visibility."""
        if self.api_key_input.echoMode() == QLineEdit.Password:
            self.api_key_input.setEchoMode(QLineEdit.Normal)
            self.show_key_btn.setText("Hide")
        else:
            self.api_key_input.setEchoMode(QLineEdit.Password)
            self.show_key_btn.setText("Show")

    def _update_data_info(self):
        """Update the data info label and preview table."""
        ticker = self.ticker_combo.currentText().strip().upper()
        if not ticker:
            self.data_info_label.setText("")
            self.preview_table.setRowCount(0)
            return

        # Try multiple file name patterns
        possible_files = [
            Path(self.data_dir) / f"{ticker}_NT.txt",
            Path(self.data_dir) / f"{ticker}.txt",
            Path(self.data_dir) / f"{ticker}.csv",
        ]

        data_file = None
        for f in possible_files:
            if f.exists():
                data_file = f
                break

        if not data_file:
            self.data_info_label.setText("(no data)")
            self.preview_table.setRowCount(0)
            return

        try:
            # Get file info
            file_size = data_file.stat().st_size / (1024 * 1024)  # MB

            # Read first and last lines for date range
            with open(data_file, 'r') as f:
                first_line = f.readline().strip()
                f.seek(0, 2)
                file_size_bytes = f.tell()
                f.seek(max(0, file_size_bytes - 200))
                lines = f.readlines()
                last_line = lines[-1].strip() if lines else ""

            # Parse dates
            first_date = first_line.split(';')[0].split()[0]
            last_date = last_line.split(';')[0].split()[0]
            first_formatted = f"{first_date[4:6]}/{first_date[6:8]}/{first_date[2:4]}"
            last_formatted = f"{last_date[4:6]}/{last_date[6:8]}/{last_date[2:4]}"

            self.data_info_label.setText(
                f"({first_formatted} - {last_formatted}, {file_size:.1f} MB)"
            )

            # Update preview table with daily summaries
            self._update_preview_table(data_file)

            # Update distribution stats display
            self._update_stats_display()

        except Exception as e:
            self.data_info_label.setText(f"(error: {e})")

    def _update_preview_table(self, data_file: Path):
        """Update the preview table with latest 10 days of data."""
        try:
            # Read last portion of file to get recent data
            daily_data = {}

            with open(data_file, 'r') as f:
                # Seek to last ~500KB for recent data
                f.seek(0, 2)
                file_size = f.tell()
                f.seek(max(0, file_size - 500000))

                # Skip partial line
                if file_size > 500000:
                    f.readline()

                for line in f:
                    parts = line.strip().split(';')
                    if len(parts) >= 6:
                        dt_str = parts[0]
                        date = dt_str.split()[0]  # YYYYMMDD

                        o, h, l, c = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])

                        if date not in daily_data:
                            daily_data[date] = {
                                'bars': 0, 'open': o, 'high': h, 'low': l, 'close': c
                            }

                        daily_data[date]['bars'] += 1
                        daily_data[date]['high'] = max(daily_data[date]['high'], h)
                        daily_data[date]['low'] = min(daily_data[date]['low'], l)
                        daily_data[date]['close'] = c

            # Get latest 10 days
            sorted_dates = sorted(daily_data.keys(), reverse=True)[:10]

            self.preview_table.setRowCount(len(sorted_dates))

            for row, date in enumerate(sorted_dates):
                data = daily_data[date]

                # Format date
                formatted_date = f"{date[4:6]}/{date[6:8]}/{date[0:4]}"

                self.preview_table.setItem(row, 0, QTableWidgetItem(formatted_date))
                self.preview_table.setItem(row, 1, QTableWidgetItem(str(data['bars'])))
                self.preview_table.setItem(row, 2, QTableWidgetItem(f"${data['open']:.2f}"))
                self.preview_table.setItem(row, 3, QTableWidgetItem(f"${data['high']:.2f}"))
                self.preview_table.setItem(row, 4, QTableWidgetItem(f"${data['low']:.2f}"))
                self.preview_table.setItem(row, 5, QTableWidgetItem(f"${data['close']:.2f}"))

        except Exception as e:
            self.preview_table.setRowCount(1)
            self.preview_table.setItem(0, 0, QTableWidgetItem(f"Error: {e}"))

    def _start_download(self):
        """Start downloading 1 year of data."""
        if not self._validate_inputs():
            return

        ticker = self.ticker_combo.currentText().strip().upper()
        api_key = self.api_key_input.text().strip()

        end_date = datetime.now()
        start_date = end_date - timedelta(days=365)

        self._start_worker(ticker, api_key, start_date, end_date, "download")

    def _start_update(self):
        """Start updating with latest data."""
        if not self._validate_inputs():
            return

        ticker = self.ticker_combo.currentText().strip().upper()
        api_key = self.api_key_input.text().strip()

        # Find the last date in existing data
        data_file = Path(self.data_dir) / f"{ticker}_NT.txt"

        if data_file.exists():
            try:
                with open(data_file, 'r') as f:
                    f.seek(0, 2)
                    file_size = f.tell()
                    f.seek(max(0, file_size - 200))
                    lines = f.readlines()
                    last_line = lines[-1].strip() if lines else ""

                last_date_str = last_line.split(';')[0].split()[0]
                start_date = datetime.strptime(last_date_str, '%Y%m%d') + timedelta(days=1)
            except:
                start_date = datetime.now() - timedelta(days=30)
        else:
            # No existing data, download last 30 days
            start_date = datetime.now() - timedelta(days=30)

        end_date = datetime.now()

        if start_date >= end_date:
            self.progress_label.setText("Data is already up to date!")
            self.progress_label.setStyleSheet("color: #00ff00;")
            return

        days_to_fetch = (end_date - start_date).days
        self.progress_label.setText(f"Updating {days_to_fetch} days of data...")

        self._start_worker(ticker, api_key, start_date, end_date, "update")

    def _start_worker(self, ticker: str, api_key: str, start_date: datetime,
                      end_date: datetime, mode: str):
        """Start the download worker thread."""
        self.download_btn.setEnabled(False)
        self.update_btn.setEnabled(False)
        self.cancel_btn.setEnabled(True)
        self.progress_label.setStyleSheet("color: #2a82da;")

        self.worker = DownloadWorker(
            ticker, api_key, self.data_dir, start_date, end_date, mode
        )
        self.worker.progress.connect(self._on_progress)
        self.worker.finished.connect(self._on_finished)
        self.worker.start()

    def _validate_inputs(self) -> bool:
        """Validate user inputs."""
        ticker = self.ticker_combo.currentText().strip()
        api_key = self.api_key_input.text().strip()

        if not ticker:
            QMessageBox.warning(self, "Missing Ticker", "Please enter a ticker symbol.")
            return False

        if not api_key:
            QMessageBox.warning(
                self, "Missing API Key",
                "Please enter your Polygon.io API key.\n\n"
                "You can get a free key at https://polygon.io/"
            )
            return False

        return True

    def _cancel_download(self):
        """Cancel the current download."""
        if self.worker:
            self.worker.cancel()
            self.progress_label.setText("Cancelling...")
            self.progress_label.setStyleSheet("color: #ffaa00;")

    def _on_progress(self, message: str):
        """Handle progress updates from worker."""
        self.progress_label.setText(message)

    def _on_finished(self, success: bool, message: str):
        """Handle download completion."""
        self.download_btn.setEnabled(True)
        self.update_btn.setEnabled(True)
        self.cancel_btn.setEnabled(False)

        if success:
            self.progress_label.setText(message)
            self.progress_label.setStyleSheet("color: #00ff00;")
            self._update_data_info()  # Refresh preview
        else:
            self.progress_label.setText(message)
            self.progress_label.setStyleSheet("color: #ff4444;")

    def _save_settings(self):
        """Save settings to QSettings."""
        self.settings.setValue("download/api_key", self.api_key_input.text())
        self.settings.setValue("download/ticker", self.ticker_combo.currentText())

    def _load_settings(self):
        """Load settings from QSettings."""
        api_key = self.settings.value("download/api_key", "")
        self.api_key_input.setText(api_key)

        ticker = self.settings.value("download/ticker", "TSLA")
        idx = self.ticker_combo.findText(ticker)
        if idx >= 0:
            self.ticker_combo.setCurrentIndex(idx)

    def set_data_dir(self, path: str):
        """Update data directory."""
        self.data_dir = path
        self._update_data_info()

    def _rebuild_stats(self):
        """Rebuild distribution statistics for the current ticker."""
        from data.distribution_stats import DistributionStatsCalculator

        ticker = self.ticker_combo.currentText().strip().upper()
        if not ticker:
            return

        self.progress_label.setText(f"Rebuilding distribution stats for {ticker}...")
        self.progress_label.setStyleSheet("color: #2a82da;")
        self.rebuild_stats_btn.setEnabled(False)

        # Process events to update UI
        from PySide6.QtWidgets import QApplication
        QApplication.processEvents()

        try:
            calc = DistributionStatsCalculator(self.data_dir)
            calc.invalidate_cache(ticker)  # Force recalculation
            stats = calc.get_stats(ticker, force_recalc=True)

            if stats:
                self.progress_label.setText(f"Stats rebuilt for {ticker} ({stats.n_trading_days} trading days)")
                self.progress_label.setStyleSheet("color: #00ff00;")
                self._update_stats_display(stats)
            else:
                self.progress_label.setText(f"Could not compute stats for {ticker} (no data?)")
                self.progress_label.setStyleSheet("color: #ff4444;")

        except Exception as e:
            self.progress_label.setText(f"Error rebuilding stats: {e}")
            self.progress_label.setStyleSheet("color: #ff4444;")

        self.rebuild_stats_btn.setEnabled(True)

    def _update_stats_display(self, stats=None):
        """Update the distribution stats display."""
        from data.distribution_stats import DistributionStatsCalculator

        ticker = self.ticker_combo.currentText().strip().upper()
        if not ticker:
            self.stats_label.setText("Select a ticker to view stats")
            return

        if stats is None:
            try:
                calc = DistributionStatsCalculator(self.data_dir)
                stats = calc.get_stats(ticker)
            except Exception:
                stats = None

        if stats is None:
            self.stats_label.setText(
                f"No stats cached for {ticker}. Click 'Rebuild Stats' to compute."
            )
            self.stats_label.setStyleSheet("color: #888888;")
            return

        # Format the stats display
        gap = stats.gap_stats
        rng = stats.range_stats

        stats_text = (
            f"<b>{ticker}</b> - {stats.n_trading_days} trading days (computed {stats.computed_date[:10]})<br><br>"
            f"<b>Gap % Distribution</b> (open vs prior close):<br>"
            f"&nbsp;&nbsp;Mean: {gap.mean:.2f}% | Std: {gap.std:.2f}%<br>"
            f"&nbsp;&nbsp;Middle 68%: {gap.p16:.2f}% to {gap.p84:.2f}%<br>"
            f"&nbsp;&nbsp;Quartiles: {gap.p25:.2f}% (25th) | {gap.median:.2f}% (50th) | {gap.p75:.2f}% (75th)<br>"
            f"&nbsp;&nbsp;Extremes: {gap.p5:.2f}% (5th) | {gap.p95:.2f}% (95th)<br><br>"
            f"<b>Daily Range % Distribution</b> (high-low):<br>"
            f"&nbsp;&nbsp;Mean: {rng.mean:.2f}% | Std: {rng.std:.2f}%<br>"
            f"&nbsp;&nbsp;Middle 68%: {rng.p16:.2f}% to {rng.p84:.2f}%<br>"
            f"&nbsp;&nbsp;Percentiles: {rng.p50:.2f}% (50th) | {rng.p68:.2f}% (68th) | {rng.p90:.2f}% (90th)"
        )

        self.stats_label.setText(stats_text)
        self.stats_label.setStyleSheet("color: #cccccc;")
