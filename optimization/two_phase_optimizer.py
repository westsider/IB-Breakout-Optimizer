"""
Two-Phase Optimizer for IB Breakout Strategy.

Phase 1: Coarse grid search to identify promising regions
Phase 2: Bayesian optimization to refine the best parameters

This approach provides the thoroughness of full optimization
in a fraction of the time (typically 10-20x faster).
"""

import sys
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional, Callable
from dataclasses import dataclass
import time

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from optimization.mmap_grid_search import (
    MMapGridSearchOptimizer, GridSearchResults, OptimizationResult,
    _mmap_backtest_worker
)
from optimization.mmap_data import MMapDataManager, load_mmap_arrays
from optimization.parameter_space import ParameterSpace, create_parameter_space, ParameterConfig, ParameterType
from dataclasses import asdict


@dataclass
class TwoPhaseResults:
    """Results from two-phase optimization."""
    phase1_results: GridSearchResults
    phase2_results: Optional[Any]  # Optuna study results
    best_result: OptimizationResult
    total_time_seconds: float
    phase1_time_seconds: float
    phase2_time_seconds: float
    phase1_combinations: int
    phase2_trials: int

    def summary(self) -> str:
        """Get summary of two-phase optimization."""
        lines = [
            "=" * 60,
            "TWO-PHASE OPTIMIZATION RESULTS",
            "=" * 60,
            "",
            f"Phase 1 (Coarse Grid): {self.phase1_combinations} combinations in {self.phase1_time_seconds:.1f}s",
            f"Phase 2 (Bayesian):    {self.phase2_trials} trials in {self.phase2_time_seconds:.1f}s",
            f"Total Time:            {self.total_time_seconds:.1f}s",
            "",
            "BEST RESULT:",
            "-" * 40,
        ]

        if self.best_result:
            # Handle both old (metrics.X) and new (direct X) result formats
            r = self.best_result
            # Check for direct attributes first (new format)
            if hasattr(r, 'total_trades') and r.total_trades is not None:
                trades = r.total_trades
                win_rate = r.win_rate
                pnl = r.total_pnl
                sharpe = r.sharpe_ratio
                pf = r.profit_factor
            elif hasattr(r, 'metrics'):
                # Old format with metrics object
                trades = r.metrics.total_trades
                win_rate = r.metrics.percent_profitable
                pnl = r.metrics.total_net_profit
                sharpe = r.metrics.sharpe_ratio
                pf = r.metrics.profit_factor
            else:
                trades = win_rate = pnl = sharpe = pf = 0

            lines.extend([
                f"Objective Value: {r.objective_value:.4f}",
                f"Total Trades:    {trades}",
                f"Win Rate:        {win_rate:.1f}%",
                f"Total P&L:       ${pnl:,.2f}",
                f"Sharpe Ratio:    {sharpe:.2f}",
                f"Profit Factor:   {pf:.2f}",
                "",
                "Best Parameters:",
            ])
            for param, value in r.params.items():
                lines.append(f"  {param}: {value}")

        return "\n".join(lines)


class TwoPhaseOptimizer:
    """
    Two-phase optimizer combining grid search and Bayesian optimization.

    Phase 1: Run coarse grid search (fast_full preset) to identify
             promising parameter regions.

    Phase 2: Use Bayesian optimization (Optuna) to refine parameters
             around the top results from Phase 1.
    """

    def __init__(
        self,
        data_dir: str,
        output_dir: Optional[str] = None,
        n_jobs: int = -1
    ):
        """
        Initialize two-phase optimizer.

        Args:
            data_dir: Directory containing data files
            output_dir: Directory for output files
            n_jobs: Number of parallel jobs (-1 = all CPUs)
        """
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir) if output_dir else self.data_dir / "optimization_results"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.n_jobs = n_jobs

        # Will be set during optimization
        self.grid_optimizer = None
        self.bayesian_optimizer = None
        self.ticker = None
        self.filter_ticker = None

    def optimize(
        self,
        ticker: str,
        filter_ticker: Optional[str] = "QQQ",
        objective: str = "sharpe_ratio",
        phase1_preset: str = "fast_full",
        phase2_trials: int = 50,
        top_n_for_phase2: int = 10,
        min_trades: int = 10,
        progress_callback: Optional[Callable] = None,
        status_callback: Optional[Callable[[str], None]] = None,
        parameter_overrides: Optional[Dict[str, Any]] = None
    ) -> TwoPhaseResults:
        """
        Run two-phase optimization.

        Args:
            ticker: Primary ticker symbol
            filter_ticker: Filter ticker (e.g., "QQQ") or None to disable
            objective: Objective to optimize
            phase1_preset: Preset for Phase 1 grid search
            phase2_trials: Number of Bayesian optimization trials in Phase 2
            top_n_for_phase2: Number of top results from Phase 1 to seed Phase 2
            min_trades: Minimum trades required
            progress_callback: Optional progress callback for grid search progress
            status_callback: Optional callback for status messages (phase changes)
            parameter_overrides: Optional dict of parameter overrides (e.g., filter settings)

        Returns:
            TwoPhaseResults
        """
        self.ticker = ticker
        self.filter_ticker = filter_ticker

        total_start = time.time()

        print("\n" + "=" * 60)
        print("TWO-PHASE OPTIMIZATION")
        print("=" * 60)
        print(f"Ticker: {ticker}")
        print(f"Filter: {filter_ticker or 'None'}")
        print(f"Objective: {objective}")
        print(f"Phase 1: {phase1_preset} grid search")
        print(f"Phase 2: {phase2_trials} Bayesian trials")
        print("=" * 60)

        # Helper to emit status
        def emit_status(msg: str):
            print(msg)
            if status_callback:
                status_callback(msg)

        # ========== PHASE 1: Coarse Grid Search (with mmap) ==========
        emit_status("Phase 1: Starting coarse grid search (memory-mapped)...")

        phase1_start = time.time()

        self.grid_optimizer = MMapGridSearchOptimizer(
            data_dir=str(self.data_dir),
            output_dir=str(self.output_dir),
            n_jobs=self.n_jobs
        )

        # Load data
        self.grid_optimizer.load_data(ticker, filter_ticker=filter_ticker)

        # Run coarse grid search
        phase1_space = create_parameter_space(phase1_preset)

        # Apply parameter overrides (e.g., filter settings from UI)
        if parameter_overrides:
            for param_name, value in parameter_overrides.items():
                if param_name in phase1_space.parameters:
                    phase1_space.parameters[param_name].enabled = True
                    phase1_space.parameters[param_name].values = [value]

        phase1_results = self.grid_optimizer.optimize(
            parameter_space=phase1_space,
            objective=objective,
            min_trades=min_trades,
            progress_callback=progress_callback
        )

        phase1_time = time.time() - phase1_start

        emit_status(f"Phase 1 complete: {phase1_results.completed_combinations} combinations in {phase1_time:.1f}s")

        # Get top results to seed Phase 2
        top_results = phase1_results.get_top_n(top_n_for_phase2)

        print(f"\nTop {len(top_results)} results from Phase 1:")
        for i, r in enumerate(top_results[:5]):
            trades = getattr(r, 'total_trades', 0)
            pnl = getattr(r, 'total_pnl', 0)
            print(f"  {i+1}. {objective}={r.objective_value:.4f}, "
                  f"trades={trades}, P&L=${pnl:.2f}")

        # ========== PHASE 2: Bayesian Refinement ==========
        emit_status(f"Phase 2: Starting Bayesian refinement ({phase2_trials} trials)...")

        phase2_start = time.time()

        # Create refined parameter space based on top results
        refined_space = self._create_refined_space(top_results, phase1_space)

        print(f"Refined parameter ranges based on top {len(top_results)} results:")
        for name, config in refined_space.get_enabled_parameters().items():
            if config.min_value is not None:
                print(f"  {name}: [{config.min_value}, {config.max_value}]")
            elif config.choices:
                print(f"  {name}: {config.choices}")

        # Run Bayesian optimization
        phase2_results = self._run_bayesian_phase(
            refined_space=refined_space,
            objective=objective,
            n_trials=phase2_trials,
            top_results=top_results,
            min_trades=min_trades
        )

        phase2_time = time.time() - phase2_start

        print(f"\nPhase 2 complete: {phase2_trials} trials in {phase2_time:.1f}s")

        # ========== Combine Results ==========
        total_time = time.time() - total_start

        # Find overall best result - only use Phase 2 if it has enough trades
        best_result = phase1_results.best_result
        if phase2_results:
            p2_trades = getattr(phase2_results, 'total_trades', 0)
            if p2_trades >= min_trades and phase2_results.objective_value > best_result.objective_value:
                best_result = phase2_results
            else:
                print(f"Phase 2 result rejected (trades={p2_trades}, min={min_trades}), using Phase 1 best")

        results = TwoPhaseResults(
            phase1_results=phase1_results,
            phase2_results=phase2_results,
            best_result=best_result,
            total_time_seconds=total_time,
            phase1_time_seconds=phase1_time,
            phase2_time_seconds=phase2_time,
            phase1_combinations=phase1_results.completed_combinations,
            phase2_trials=phase2_trials
        )

        # Save results
        self._save_results(results, objective)

        print("\n" + results.summary())

        return results

    def _create_refined_space(
        self,
        top_results: List[OptimizationResult],
        original_space: ParameterSpace
    ) -> ParameterSpace:
        """
        Create refined parameter space based on top results.

        Narrows ranges around the best-performing parameter values.
        """
        refined = ParameterSpace()
        refined.disable_all()

        # Get parameter values from top results
        param_values = {}
        for result in top_results:
            for param, value in result.params.items():
                if param not in param_values:
                    param_values[param] = []
                param_values[param].append(value)

        # Refine each enabled parameter
        for name, config in original_space.get_enabled_parameters().items():
            if name not in param_values:
                continue

            values = param_values[name]
            new_config = ParameterConfig(
                name=name,
                param_type=config.param_type,
                default=config.default,
                enabled=True
            )

            if config.param_type == ParameterType.FLOAT:
                # Narrow range around min/max of top values
                min_val = min(values)
                max_val = max(values)
                range_width = max_val - min_val

                # Expand range slightly to explore nearby
                margin = max(range_width * 0.3, config.step or 0.1)
                new_config.min_value = max(config.min_value, min_val - margin)
                new_config.max_value = min(config.max_value, max_val + margin)
                new_config.step = (config.step or 0.1) / 2  # Finer step

            elif config.param_type == ParameterType.INT:
                min_val = min(values)
                max_val = max(values)
                margin = max((max_val - min_val) * 0.3, config.step or 1)
                new_config.min_value = max(config.min_value, min_val - margin)
                new_config.max_value = min(config.max_value, max_val + margin)
                new_config.step = config.step

            elif config.param_type == ParameterType.CATEGORICAL:
                # Keep only values that appeared in top results
                unique_values = list(set(values))
                new_config.choices = unique_values if len(unique_values) > 1 else config.choices

            elif config.param_type == ParameterType.BOOL:
                # Check if both True and False are in top results
                unique_values = list(set(values))
                if len(unique_values) == 1:
                    # All top results have same value - fix it
                    new_config.choices = unique_values
                    new_config.param_type = ParameterType.CATEGORICAL
                else:
                    pass  # Keep as bool

            refined.add_parameter(new_config)

        return refined

    def _run_bayesian_phase(
        self,
        refined_space: ParameterSpace,
        objective: str,
        n_trials: int,
        top_results: List[OptimizationResult],
        min_trades: int
    ) -> Optional[OptimizationResult]:
        """
        Run Bayesian optimization phase.

        Uses Optuna with initial points seeded from Phase 1 results.
        """
        try:
            import optuna
            from optuna.samplers import TPESampler
        except ImportError:
            print("Warning: Optuna not installed. Skipping Phase 2.")
            return None

        # Suppress Optuna logging
        optuna.logging.set_verbosity(optuna.logging.WARNING)

        # Create mmap arrays for fast backtesting
        sessions = self.grid_optimizer.sessions
        filter_bars_dict = self.grid_optimizer.filter_bars_dict

        mmap_manager = MMapDataManager(
            sessions,
            filter_bars_dict,
            ticker=self.ticker,
            filter_ticker=self.filter_ticker or ""
        )
        mmap_paths = mmap_manager.get_paths()
        mmap_paths_dict = asdict(mmap_paths)

        def objective_func(trial: optuna.Trial) -> float:
            """Optuna objective function."""
            params = {}

            for name, config in refined_space.get_enabled_parameters().items():
                if config.param_type == ParameterType.FLOAT:
                    params[name] = trial.suggest_float(
                        name, config.min_value, config.max_value
                    )
                elif config.param_type == ParameterType.INT:
                    params[name] = trial.suggest_int(
                        name, int(config.min_value), int(config.max_value)
                    )
                elif config.param_type == ParameterType.BOOL:
                    params[name] = trial.suggest_categorical(name, [True, False])
                elif config.param_type == ParameterType.CATEGORICAL:
                    params[name] = trial.suggest_categorical(name, config.choices)

            # Add defaults for non-optimized parameters
            for name, config in refined_space.parameters.items():
                if name not in params:
                    params[name] = config.default

            # Run backtest using mmap worker
            result = _mmap_backtest_worker(
                params=params,
                mmap_paths_dict=mmap_paths_dict,
                ticker=self.ticker,
                objective=objective,
                min_trades=min_trades
            )

            return result.objective_value

        # Create study with TPE sampler
        sampler = TPESampler(seed=42, n_startup_trials=min(10, n_trials // 3))
        study = optuna.create_study(direction="maximize", sampler=sampler)

        # Enqueue top results from Phase 1 as starting points
        for result in top_results[:5]:
            # Filter to only include parameters in refined space
            enqueue_params = {
                k: v for k, v in result.params.items()
                if k in refined_space.get_enabled_parameters()
            }
            if enqueue_params:
                try:
                    study.enqueue_trial(enqueue_params)
                except Exception:
                    pass  # Skip if parameters don't match

        try:
            # Run optimization
            study.optimize(objective_func, n_trials=n_trials, show_progress_bar=True)

            # Get best result
            best_params = study.best_params

            # Add defaults for missing params
            for name, config in refined_space.parameters.items():
                if name not in best_params:
                    best_params[name] = config.default

            # Run final backtest with best params
            best_result = _mmap_backtest_worker(
                params=best_params,
                mmap_paths_dict=mmap_paths_dict,
                ticker=self.ticker,
                objective=objective,
                min_trades=min_trades
            )

            print(f"Phase 2 best {objective}: {best_result.objective_value:.4f}")

            return best_result

        finally:
            # Cleanup mmap files
            mmap_manager.cleanup()

    def _save_results(self, results: TwoPhaseResults, objective: str):
        """Save two-phase optimization results."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Save Phase 1 results
        phase1_file = self.output_dir / f"twophase_p1_{self.ticker}_{timestamp}.csv"
        results.phase1_results.save_results(str(phase1_file))

        # Get metrics from best result (handle both formats)
        r = results.best_result
        trades = getattr(r, 'total_trades', 0)
        win_rate = getattr(r, 'win_rate', 0)
        pnl = getattr(r, 'total_pnl', 0)
        sharpe = getattr(r, 'sharpe_ratio', 0)
        pf = getattr(r, 'profit_factor', 0)

        # Save best parameters
        import json
        best_file = self.output_dir / f"twophase_best_{self.ticker}_{timestamp}.json"
        with open(best_file, 'w') as f:
            json.dump({
                'params': r.params,
                'objective': objective,
                'objective_value': r.objective_value,
                'metrics': {
                    'total_trades': trades,
                    'win_rate': win_rate,
                    'total_pnl': pnl,
                    'sharpe_ratio': sharpe,
                    'profit_factor': pf
                },
                'timing': {
                    'total_seconds': results.total_time_seconds,
                    'phase1_seconds': results.phase1_time_seconds,
                    'phase2_seconds': results.phase2_time_seconds
                }
            }, f, indent=2)

        print(f"\nResults saved to: {self.output_dir}")


if __name__ == "__main__":
    # Test two-phase optimizer
    optimizer = TwoPhaseOptimizer(r"C:\Users\Warren\Downloads")

    results = optimizer.optimize(
        ticker="TSLA",
        filter_ticker="QQQ",
        objective="sharpe_ratio",
        phase1_preset="fast_full",
        phase2_trials=30
    )

    print("\n" + results.summary())
