"""
Optimization Worker - Run grid search optimization in a background thread with live progress updates.

Uses subprocess to launch the parallel optimizer script for full CPU utilization,
while providing real-time progress updates via Qt signals.
"""

import subprocess
import json
import time
import sys
from pathlib import Path
import traceback
import uuid
import os
import logging
from datetime import datetime

from PySide6.QtCore import QThread, Signal


PROJECT_ROOT = Path(__file__).parent.parent.parent
LOG_DIR = PROJECT_ROOT / "logs"
LOG_DIR.mkdir(exist_ok=True)

# Setup logging
log_file = LOG_DIR / f"optimization_{datetime.now().strftime('%Y%m%d')}.log"
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
OUTPUT_DIR = PROJECT_ROOT / "output"


class OptimizationWorker(QThread):
    """Worker thread for running optimization with live progress updates."""

    # Signals for thread-safe UI updates
    progress = Signal(int, int, float, int, float, float)  # current, total, best_obj, best_trades, best_pnl, speed
    status_update = Signal(str)  # status message for debugging
    finished = Signal(dict)  # results dictionary
    error = Signal(str)

    def __init__(self, data_dir: str, output_dir: str, settings: dict):
        super().__init__()
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.settings = settings
        self._cancelled = False
        self.process = None

    def cancel(self):
        """Cancel the optimization."""
        self._cancelled = True
        if self.process:
            self.process.terminate()

    def run(self):
        """Execute the optimization - directly in thread for better compatibility."""
        logger.info(f"Starting optimization: {self.settings}")
        try:
            self._run_direct()
        except Exception as e:
            error_msg = f"{str(e)}\n{traceback.format_exc()}"
            logger.error(f"Optimization failed: {error_msg}")
            self.error.emit(error_msg)

    def _run_direct(self):
        """Run optimization directly in the worker thread."""
        import sys
        sys.path.insert(0, str(PROJECT_ROOT))

        from optimization.parameter_space import create_parameter_space

        ticker = self.settings['ticker']
        preset = self.settings['preset']
        objective = self.settings['objective']
        use_qqq_filter = self.settings['use_qqq_filter']
        mode = self.settings.get('mode', 'grid')

        self.status_update.emit(f"Loading data for {ticker} (mode: {mode})...")

        # Select optimizer based on mode
        filter_ticker = 'QQQ' if use_qqq_filter and ticker != 'QQQ' else None

        if mode == 'two_phase':
            # Two-phase optimizer (coarse grid + Bayesian)
            from optimization.two_phase_optimizer import TwoPhaseOptimizer

            optimizer = TwoPhaseOptimizer(
                data_dir=self.data_dir,
                output_dir=f"{self.output_dir}/optimization",
                n_jobs=-1
            )

            self.status_update.emit("Running two-phase optimization (Phase 1: Grid Search)...")

            # For two-phase, use the selected preset for Phase 1
            # Valid presets: quick, standard, full, thorough
            phase1_preset = preset if preset in ['quick', 'standard', 'full', 'thorough'] else 'standard'

            # Track timing for progress
            start_time = time.time()

            def two_phase_progress(current, total_combos, best):
                if self._cancelled:
                    raise InterruptedError("Cancelled by user")

                now = time.time()
                elapsed = now - start_time
                speed = current / elapsed if elapsed > 0 else 0

                # Handle both old (metrics.X) and new (direct X) result formats
                if best:
                    obj_val = best.objective_value
                    trades = getattr(best, 'total_trades', None)
                    pnl = getattr(best, 'total_pnl', None)
                    if trades is None and hasattr(best, 'metrics'):
                        trades = best.metrics.total_trades
                        pnl = best.metrics.total_net_profit
                else:
                    obj_val, trades, pnl = 0.0, 0, 0.0

                self.progress.emit(
                    current, total_combos,
                    obj_val,
                    trades or 0,
                    pnl or 0.0,
                    speed
                )

            def status_update(msg: str):
                self.status_update.emit(msg)

            try:
                results = optimizer.optimize(
                    ticker=ticker,
                    filter_ticker=filter_ticker,
                    objective=objective,
                    phase1_preset=phase1_preset,
                    phase2_trials=50,
                    top_n_for_phase2=10,
                    progress_callback=two_phase_progress,
                    status_callback=status_update
                )

                # Convert TwoPhaseResults to standard format
                final_results = self._build_results_dict_from_two_phase(results)
                self.finished.emit(final_results)
                return

            except InterruptedError:
                logger.info("Optimization cancelled by user")
                self.error.emit("Optimization cancelled by user")
                return
            except Exception as e:
                error_msg = f"Two-phase optimization failed: {e}\n{traceback.format_exc()}"
                logger.error(error_msg)
                self.error.emit(f"Two-phase optimization failed: {e}")
                return

        else:
            # Standard grid search - use memory-efficient mmap optimizer
            from optimization.mmap_grid_search import MMapGridSearchOptimizer

            optimizer = MMapGridSearchOptimizer(
                data_dir=self.data_dir,
                output_dir=f"{self.output_dir}/optimization",
                n_jobs=-1
            )

        # Load data
        try:
            optimizer.load_data(ticker, filter_ticker=filter_ticker)
        except FileNotFoundError as e:
            self.error.emit(f"Data file not found for {ticker}: {e}")
            return
        except Exception as e:
            self.error.emit(f"Error loading data: {e}")
            return

        self.status_update.emit(f"Loaded {len(optimizer.sessions)} sessions")

        # Create parameter space
        space = create_parameter_space(preset)

        # Enable QQQ filter parameter if requested
        if use_qqq_filter and 'use_qqq_filter' in space.parameters:
            space.parameters['use_qqq_filter'].enabled = True

        combinations = space.get_grid_combinations()
        total = len(combinations)

        self.status_update.emit(f"Starting optimization: {total} combinations...")

        # Emit initial progress (0/total) so UI shows the total
        self.progress.emit(0, total, 0.0, 0, 0.0, 0.0)

        # Track state for progress callback
        start_time = time.time()
        last_time = [start_time]
        last_count = [0]

        def progress_callback(current, total_combos, best):
            if self._cancelled:
                raise InterruptedError("Cancelled by user")

            # Calculate speed
            now = time.time()
            elapsed = now - start_time
            speed = current / elapsed if elapsed > 0 else 0

            # Handle both old (metrics.X) and new (direct X) result formats
            if best:
                obj_val = best.objective_value
                # New mmap optimizer uses direct attributes
                trades = getattr(best, 'total_trades', None)
                pnl = getattr(best, 'total_pnl', None)
                # Fall back to old metrics format
                if trades is None and hasattr(best, 'metrics'):
                    trades = best.metrics.total_trades
                    pnl = best.metrics.total_net_profit
            else:
                obj_val, trades, pnl = 0.0, 0, 0.0

            # Always emit progress - batches already provide natural throttling
            self.progress.emit(
                current, total_combos,
                obj_val,
                trades or 0,
                pnl or 0.0,
                speed
            )

        # Run optimization
        try:
            results = optimizer.optimize(
                parameter_space=space,
                objective=objective,
                progress_callback=progress_callback
            )
        except InterruptedError:
            self.error.emit("Optimization cancelled by user")
            return

        # Build results dict - handle both old and new result formats
        top_results = []
        for r in results.get_top_n(10):
            # New mmap optimizer uses direct attributes
            trades = getattr(r, 'total_trades', None)
            win_rate = getattr(r, 'win_rate', None)
            pnl = getattr(r, 'total_pnl', None)
            pf = getattr(r, 'profit_factor', None)
            trade_pnls = getattr(r, 'trade_pnls', [])
            # Fall back to old metrics format
            if trades is None and hasattr(r, 'metrics'):
                trades = r.metrics.total_trades
                win_rate = r.metrics.percent_profitable
                pnl = r.metrics.total_net_profit
                pf = r.metrics.profit_factor
                trade_pnls = []

            # Include ALL params from the optimization result
            result_dict = dict(r.params)  # Start with all params
            result_dict.update({
                'objective_value': r.objective_value,
                'profit_factor': pf or 0,
                'total_trades': trades or 0,
                'win_rate': win_rate or 0,
                'total_pnl': pnl or 0,
                'trade_pnls': trade_pnls  # Individual trade P&Ls for equity curve
            })
            top_results.append(result_dict)

        # Get best result metrics
        best = results.best_result
        if best:
            # Try new format first (direct attributes)
            best_trades = getattr(best, 'total_trades', None)
            best_win_rate = getattr(best, 'win_rate', None)
            best_pnl = getattr(best, 'total_pnl', None)
            best_pf = getattr(best, 'profit_factor', None)
            best_sharpe = getattr(best, 'sharpe_ratio', None)
            best_dd = getattr(best, 'max_drawdown', None)
            # Fall back to old metrics format
            if best_trades is None and hasattr(best, 'metrics'):
                best_trades = best.metrics.total_trades
                best_win_rate = best.metrics.percent_profitable
                best_pnl = best.metrics.total_net_profit
                best_pf = best.metrics.profit_factor
                best_sharpe = best.metrics.sharpe_ratio
                best_dd = best.metrics.max_drawdown

            best_metrics = {
                'total_trades': best_trades or 0,
                'win_rate': best_win_rate or 0,
                'total_pnl': best_pnl or 0,
                'profit_factor': best_pf or 0,
                'sharpe_ratio': best_sharpe or 0,
                'max_drawdown': best_dd or 0
            }
        else:
            best_metrics = {}

        final_results = {
            'total_combinations': results.total_combinations,
            'completed': results.completed_combinations,
            'total_time': results.total_time_seconds,
            'best_params': results.best_result.params if results.best_result else {},
            'best_metrics': best_metrics,
            'best_objective': results.best_result.objective_value if results.best_result else 0,
            'top_results': top_results
        }

        self.finished.emit(final_results)

    def _run_subprocess(self):
        """Execute the optimization via subprocess (alternative approach)."""
        try:
            # Create unique progress file
            run_id = uuid.uuid4().hex[:8]
            progress_file = OUTPUT_DIR / f"optimization_progress_{run_id}.json"
            OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

            # Build command
            script_path = SCRIPTS_DIR / "run_optimization.py"

            # Verify script exists
            if not script_path.exists():
                self.error.emit(f"Script not found: {script_path}")
                return

            cmd = [
                sys.executable,
                str(script_path),
                "--ticker", self.settings['ticker'],
                "--preset", self.settings['preset'],
                "--objective", self.settings['objective'],
                "--data-dir", self.data_dir,
                "--output-dir", self.output_dir,
                "--progress-file", str(progress_file)
            ]

            if self.settings['use_qqq_filter']:
                cmd.append("--use-qqq-filter")

            self.status_update.emit(f"Starting: {' '.join(cmd[:3])}...")

            # Start subprocess
            self.process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                cwd=str(PROJECT_ROOT)
            )

            self.status_update.emit(f"Process started (PID: {self.process.pid})")

            # Track timing for speed calculation
            start_time = time.time()
            last_count = 0
            last_time = start_time

            # Poll for progress
            poll_count = 0
            while self.process.poll() is None and not self._cancelled:
                poll_count += 1

                # Check for progress file
                if progress_file.exists():
                    try:
                        with open(progress_file, 'r') as f:
                            content = f.read()
                            if content.strip():
                                data = json.loads(content)

                                # Check if complete
                                if data.get('status') == 'complete':
                                    self.status_update.emit("Optimization complete")
                                    break

                                # Extract progress
                                current = data.get('current', 0)
                                total = data.get('total', 1)
                                best_obj = data.get('best_objective')
                                best_trades = data.get('best_trades')
                                best_pnl = data.get('best_pnl')

                                # Calculate speed
                                now = time.time()
                                if now - last_time >= 1.0:
                                    speed = (current - last_count) / (now - last_time)
                                    last_count = current
                                    last_time = now
                                else:
                                    speed = 0

                                # Emit progress signal
                                self.progress.emit(
                                    current, total,
                                    best_obj if best_obj is not None else 0.0,
                                    best_trades if best_trades is not None else 0,
                                    best_pnl if best_pnl is not None else 0.0,
                                    speed
                                )

                    except (json.JSONDecodeError, FileNotFoundError, ValueError) as e:
                        if poll_count % 50 == 0:  # Log occasionally
                            self.status_update.emit(f"Waiting for progress data... ({poll_count})")
                else:
                    if poll_count % 25 == 0:  # Every 5 seconds
                        elapsed = time.time() - start_time
                        self.status_update.emit(f"Waiting for optimization to start... ({elapsed:.0f}s)")

                time.sleep(0.2)  # Poll every 200ms

            if self._cancelled:
                self.error.emit("Optimization cancelled by user")
                self._cleanup(progress_file)
                return

            # Wait for process to complete and get output
            stdout, stderr = self.process.communicate(timeout=30)

            # Log any stderr output
            if stderr and stderr.strip():
                self.status_update.emit(f"Script output: {stderr[:500]}")

            if self.process.returncode != 0:
                error_msg = stderr[:1000] if stderr else "Unknown error"
                self.error.emit(f"Optimization failed (code {self.process.returncode}): {error_msg}")
                self._cleanup(progress_file)
                return

            # Read final results
            if progress_file.exists():
                with open(progress_file, 'r') as f:
                    results = json.load(f)

                # Extract top results for the table
                top_results = self._get_top_results()

                # Build results dict
                final_results = {
                    'total_combinations': results.get('total_combinations', 0),
                    'completed': results.get('completed', 0),
                    'total_time': results.get('total_time', 0),
                    'best_params': results.get('best_params', {}),
                    'best_metrics': results.get('best_metrics', {}),
                    'best_objective': results.get('best_objective', 0),
                    'top_results': top_results
                }

                self.finished.emit(final_results)

                # Cleanup
                self._cleanup(progress_file)

            else:
                self.error.emit("Results file not found")

        except Exception as e:
            self.error.emit(f"{str(e)}\n{traceback.format_exc()}")

    def _get_top_results(self) -> list:
        """Load top results from the full results CSV if available."""
        # Look for the most recent results file
        results_dir = Path(self.output_dir) / "optimization"

        if not results_dir.exists():
            return []

        # Find most recent results file for this ticker
        ticker = self.settings['ticker']
        pattern = f"grid_search_{ticker}_*.csv"

        csv_files = list(results_dir.glob(pattern))
        if not csv_files:
            return []

        # Get most recent
        latest_csv = max(csv_files, key=lambda p: p.stat().st_mtime)

        try:
            import pandas as pd
            df = pd.read_csv(latest_csv)

            # Sort by objective value
            objective = self.settings['objective']
            if objective in df.columns:
                df = df.sort_values(objective, ascending=False)

            # Get top 10
            top_10 = df.head(10)

            results = []
            for _, row in top_10.iterrows():
                # Include ALL columns from the CSV as params
                result_dict = row.to_dict()
                # Ensure objective_value is set
                result_dict['objective_value'] = row.get(objective, 0)
                # Map total_net_profit to total_pnl for consistency
                if 'total_net_profit' in result_dict:
                    result_dict['total_pnl'] = result_dict['total_net_profit']
                results.append(result_dict)

            return results

        except Exception:
            return []

    def _cleanup(self, progress_file: Path):
        """Clean up temporary files."""
        try:
            if progress_file.exists():
                progress_file.unlink()
        except:
            pass

    def _build_results_dict_from_two_phase(self, results) -> dict:
        """Convert TwoPhaseResults to standard results dictionary."""
        # Helper to get metrics from result (handles both old and new formats)
        def get_metrics(r):
            if hasattr(r, 'total_trades') and r.total_trades is not None:
                return {
                    'total_trades': r.total_trades,
                    'win_rate': r.win_rate,
                    'total_pnl': r.total_pnl,
                    'profit_factor': r.profit_factor,
                    'sharpe_ratio': r.sharpe_ratio,
                    'max_drawdown': r.max_drawdown
                }
            elif hasattr(r, 'metrics'):
                return {
                    'total_trades': r.metrics.total_trades,
                    'win_rate': r.metrics.percent_profitable,
                    'total_pnl': r.metrics.total_net_profit,
                    'profit_factor': r.metrics.profit_factor,
                    'sharpe_ratio': r.metrics.sharpe_ratio,
                    'max_drawdown': r.metrics.max_drawdown
                }
            return {'total_trades': 0, 'win_rate': 0, 'total_pnl': 0,
                    'profit_factor': 0, 'sharpe_ratio': 0, 'max_drawdown': 0}

        # Get top results from Phase 1
        top_results = []
        for r in results.phase1_results.get_top_n(10):
            m = get_metrics(r)
            trade_pnls = getattr(r, 'trade_pnls', [])
            # Include ALL params from the optimization result
            result_dict = dict(r.params)
            result_dict.update({
                'objective_value': r.objective_value,
                'profit_factor': m['profit_factor'],
                'total_trades': m['total_trades'],
                'win_rate': m['win_rate'],
                'total_pnl': m['total_pnl'],
                'trade_pnls': trade_pnls
            })
            top_results.append(result_dict)

        # Get best result metrics
        best_metrics = get_metrics(results.best_result) if results.best_result else {}

        return {
            'total_combinations': results.phase1_combinations + results.phase2_trials,
            'completed': results.phase1_combinations + results.phase2_trials,
            'total_time': results.total_time_seconds,
            'best_params': results.best_result.params if results.best_result else {},
            'best_metrics': best_metrics,
            'best_objective': results.best_result.objective_value if results.best_result else 0,
            'top_results': top_results
        }
