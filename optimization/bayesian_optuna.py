"""
Bayesian Optimization with Optuna for IB Breakout Strategy.

Uses Optuna for efficient parameter search using:
- Tree-structured Parzen Estimator (TPE)
- Pruning of unpromising trials
- Multi-objective optimization support
"""

import sys
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple, Callable
from dataclasses import dataclass, field
import pandas as pd
import numpy as np
import json
import time

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import optuna
    from optuna.samplers import TPESampler
    from optuna.pruners import MedianPruner
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    print("Warning: Optuna not installed. Run: pip install optuna")

from data.data_loader import DataLoader
from data.session_builder import SessionBuilder
from data.data_types import Trade, Bar
from strategy.ib_breakout import IBBreakoutStrategy, StrategyParams
from metrics.performance_metrics import calculate_metrics, PerformanceMetrics
from optimization.parameter_space import ParameterSpace, create_parameter_space
from optimization.grid_search import OptimizationResult, calculate_objective


@dataclass
class BayesianOptimizationResults:
    """Results from Bayesian optimization."""
    best_params: Dict[str, Any] = field(default_factory=dict)
    best_value: float = 0.0
    best_metrics: Optional[PerformanceMetrics] = None
    all_trials: List[Dict[str, Any]] = field(default_factory=list)
    n_trials: int = 0
    objective: str = "sharpe_ratio"
    total_time_seconds: float = 0.0
    ticker: str = ""

    def to_dataframe(self) -> pd.DataFrame:
        """Convert trials to DataFrame."""
        return pd.DataFrame(self.all_trials)

    def save_results(self, filepath: str):
        """Save results to CSV."""
        df = self.to_dataframe()
        df.to_csv(filepath, index=False)

    def summary(self) -> str:
        """Get summary of results."""
        lines = [
            "=" * 60,
            "BAYESIAN OPTIMIZATION RESULTS (Optuna)",
            "=" * 60,
            f"Ticker: {self.ticker}",
            f"Objective: {self.objective}",
            f"Total Trials: {self.n_trials}",
            f"Total Time: {self.total_time_seconds:.1f} seconds",
            "",
            "BEST RESULT:",
            "-" * 40,
            f"Objective Value: {self.best_value:.4f}",
        ]

        if self.best_metrics:
            lines.extend([
                f"Total Trades: {self.best_metrics.total_trades}",
                f"Win Rate: {self.best_metrics.percent_profitable:.1f}%",
                f"Total P&L: ${self.best_metrics.total_net_profit:,.2f}",
                f"Profit Factor: {self.best_metrics.profit_factor:.2f}",
                f"Sharpe Ratio: {self.best_metrics.sharpe_ratio:.2f}",
                f"Max Drawdown: ${self.best_metrics.max_drawdown:,.2f}",
            ])

        lines.extend(["", "Best Parameters:"])
        for param, value in self.best_params.items():
            lines.append(f"  {param}: {value}")

        return "\n".join(lines)


class BayesianOptimizer:
    """
    Bayesian optimizer using Optuna.

    More efficient than grid search for large parameter spaces.
    """

    def __init__(
        self,
        data_dir: str,
        output_dir: Optional[str] = None
    ):
        """
        Initialize optimizer.

        Args:
            data_dir: Directory containing data files
            output_dir: Directory for output files
        """
        if not OPTUNA_AVAILABLE:
            raise ImportError("Optuna is required. Install with: pip install optuna")

        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir) if output_dir else self.data_dir / "optimization_results"
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Data storage
        self.loader = DataLoader(str(self.data_dir))
        self.sessions = None
        self.ticker = None
        self.filter_bars_dict = None  # For QQQ filter

        # Optimization state
        self.study = None
        self.parameter_space = None
        self.objective_name = "sharpe_ratio"

    def _find_data_file(self, ticker: str) -> Path:
        """Find data file for ticker, preferring NT format."""
        nt_files = []
        other_files = []

        for f in self.data_dir.iterdir():
            if f.is_file() and ticker.upper() in f.name.upper():
                if f.suffix.lower() in ['.txt', '.csv']:
                    if '_NT' in f.name.upper() or 'NT.TXT' in f.name.upper():
                        nt_files.append(f)
                    else:
                        other_files.append(f)

        if nt_files:
            return nt_files[0]
        if other_files:
            return other_files[0]

        raise FileNotFoundError(f"No data file found for {ticker}")

    def load_data(self, ticker: str, data_file: Optional[str] = None,
                  filter_ticker: Optional[str] = None):
        """
        Load and prepare data.

        Args:
            ticker: Ticker symbol
            data_file: Specific data file (optional)
            filter_ticker: Optional filter ticker (e.g., "QQQ") for filter optimization
        """
        self.ticker = ticker

        if data_file:
            filepath = self.data_dir / data_file
        else:
            filepath = self._find_data_file(ticker)

        df = self.loader.load_auto_detect(str(filepath), ticker)
        session_builder = SessionBuilder()
        self.sessions = session_builder.build_sessions_from_dataframe(df, ticker)

        print(f"Loaded {len(self.sessions)} sessions for {ticker}")

        # Load filter ticker data if specified
        if filter_ticker:
            self._load_filter_data(filter_ticker)

    def _load_filter_data(self, filter_ticker: str):
        """Load filter ticker data (e.g., QQQ) for filter optimization."""
        try:
            filter_file = self._find_data_file(filter_ticker)
            df_filter = self.loader.load_auto_detect(str(filter_file), filter_ticker)

            self.filter_bars_dict = {}
            for _, row in df_filter.iterrows():
                bar = Bar(
                    timestamp=row['timestamp'],
                    open=row['open'],
                    high=row['high'],
                    low=row['low'],
                    close=row['close'],
                    volume=row.get('volume', 0),
                    ticker=filter_ticker
                )
                self.filter_bars_dict[row['timestamp']] = bar

            print(f"Loaded {len(self.filter_bars_dict):,} filter bars for {filter_ticker}")

        except FileNotFoundError as e:
            print(f"Warning: Could not load filter data for {filter_ticker}: {e}")
            self.filter_bars_dict = None

    def _suggest_params(self, trial: 'optuna.Trial') -> Dict[str, Any]:
        """
        Suggest parameters using Optuna trial.

        Args:
            trial: Optuna trial object

        Returns:
            Dict of suggested parameters
        """
        params = {}

        for name, config in self.parameter_space.parameters.items():
            if not config.enabled:
                params[name] = config.default
                continue

            optuna_config = config.get_optuna_config()

            if optuna_config["type"] == "categorical":
                params[name] = trial.suggest_categorical(name, optuna_config["choices"])

            elif optuna_config["type"] == "int":
                params[name] = trial.suggest_int(
                    name,
                    optuna_config["low"],
                    optuna_config["high"],
                    step=optuna_config.get("step", 1)
                )

            elif optuna_config["type"] == "float":
                if optuna_config.get("step"):
                    # Discrete float
                    params[name] = trial.suggest_float(
                        name,
                        optuna_config["low"],
                        optuna_config["high"],
                        step=optuna_config["step"]
                    )
                else:
                    params[name] = trial.suggest_float(
                        name,
                        optuna_config["low"],
                        optuna_config["high"]
                    )

        # Apply conditional logic
        if not params.get("trailing_stop_enabled", False):
            params["trailing_stop_atr_mult"] = self.parameter_space.parameters["trailing_stop_atr_mult"].default

        if not params.get("break_even_enabled", False):
            params["break_even_pct"] = self.parameter_space.parameters["break_even_pct"].default

        if not params.get("max_bars_enabled", False):
            params["max_bars"] = self.parameter_space.parameters["max_bars"].default

        return params

    def _objective(self, trial: 'optuna.Trial') -> float:
        """
        Objective function for Optuna.

        Args:
            trial: Optuna trial

        Returns:
            Objective value to maximize
        """
        # Suggest parameters
        params = self._suggest_params(trial)

        # Determine if QQQ filter should be used
        use_qqq_filter = params.get('use_qqq_filter', False) and self.filter_bars_dict is not None

        # Create strategy
        strategy_params = StrategyParams(
            ib_duration_minutes=params.get('ib_duration_minutes', 30),
            ib_proximity_percent=params.get('ib_proximity_percent', 0.0),
            trade_direction=params.get('trade_direction', 'both'),
            trading_start_time=params.get('trading_start_time', '09:00'),
            trading_end_time=params.get('trading_end_time', '15:00'),
            fixed_share_size=params.get('fixed_share_size', 100),
            profit_target_percent=params.get('profit_target_percent', 0.5),
            stop_loss_type=params.get('stop_loss_type', 'opposite_ib'),
            trailing_stop_enabled=params.get('trailing_stop_enabled', False),
            trailing_stop_atr_mult=params.get('trailing_stop_atr_mult', 2.0),
            break_even_enabled=params.get('break_even_enabled', False),
            break_even_pct=params.get('break_even_pct', 0.7),
            max_bars_enabled=params.get('max_bars_enabled', False),
            max_bars=params.get('max_bars', 60),
            eod_exit_time=params.get('eod_exit_time', '15:55'),
            use_qqq_filter=use_qqq_filter,
            min_ib_range_percent=params.get('min_ib_range_percent', 0.0),
            max_ib_range_percent=params.get('max_ib_range_percent', 10.0),
            max_breakout_time=params.get('max_breakout_time', '14:00'),
            trade_monday=params.get('trade_monday', True),
            trade_tuesday=params.get('trade_tuesday', True),
            trade_wednesday=params.get('trade_wednesday', True),
            trade_thursday=params.get('trade_thursday', True),
            trade_friday=params.get('trade_friday', True)
        )

        # Run backtest
        strategy = IBBreakoutStrategy(strategy_params)

        for session in self.sessions:
            is_first = True
            for bar in session.bars:
                # Get filter bar if QQQ filter is enabled
                filter_bar = self.filter_bars_dict.get(bar.timestamp) if use_qqq_filter else None
                strategy.process_bar(bar, is_first, qqq_bar=filter_bar)
                is_first = False

        # Calculate metrics
        trades = strategy.get_trades()
        metrics = calculate_metrics(trades)

        # Calculate objective
        obj_value = calculate_objective(metrics, self.objective_name)

        # Store additional info in trial
        trial.set_user_attr("total_trades", metrics.total_trades)
        trial.set_user_attr("win_rate", metrics.percent_profitable)
        trial.set_user_attr("total_pnl", metrics.total_net_profit)
        trial.set_user_attr("profit_factor", metrics.profit_factor)
        trial.set_user_attr("sharpe_ratio", metrics.sharpe_ratio)
        trial.set_user_attr("max_drawdown", metrics.max_drawdown)

        return obj_value

    def optimize(
        self,
        n_trials: int = 100,
        parameter_space: Optional[ParameterSpace] = None,
        objective: str = "sharpe_ratio",
        timeout: Optional[int] = None,
        n_startup_trials: int = 10,
        show_progress: bool = True
    ) -> BayesianOptimizationResults:
        """
        Run Bayesian optimization.

        Args:
            n_trials: Number of trials to run
            parameter_space: Parameter space (uses standard if None)
            objective: Objective to optimize
            timeout: Timeout in seconds (optional)
            n_startup_trials: Random trials before TPE kicks in
            show_progress: Show progress bar

        Returns:
            BayesianOptimizationResults
        """
        if self.sessions is None:
            raise ValueError("No data loaded. Call load_data() first.")

        self.parameter_space = parameter_space or create_parameter_space("standard")
        self.objective_name = objective

        print(f"\n{'='*60}")
        print("BAYESIAN OPTIMIZATION (Optuna)")
        print(f"{'='*60}")
        print(f"Ticker: {self.ticker}")
        print(f"Objective: {objective}")
        print(f"Trials: {n_trials}")
        print(f"Enabled Parameters: {len(self.parameter_space.get_enabled_parameters())}")
        print(f"{'='*60}\n")

        # Create study
        sampler = TPESampler(n_startup_trials=n_startup_trials, seed=42)
        pruner = MedianPruner(n_startup_trials=5, n_warmup_steps=10)

        self.study = optuna.create_study(
            direction="maximize",
            sampler=sampler,
            pruner=pruner,
            study_name=f"ib_breakout_{self.ticker}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )

        # Optimization callback
        def callback(study, trial):
            if show_progress and trial.number % 10 == 0:
                print(f"Trial {trial.number}: value={trial.value:.4f}, "
                      f"best={study.best_value:.4f}")

        start_time = time.time()

        # Run optimization
        self.study.optimize(
            self._objective,
            n_trials=n_trials,
            timeout=timeout,
            callbacks=[callback],
            show_progress_bar=show_progress
        )

        total_time = time.time() - start_time

        # Get best trial
        best_trial = self.study.best_trial

        # Run best params again to get full metrics
        best_params = self._suggest_params_from_dict(best_trial.params)

        strategy_params = StrategyParams(**{
            k: v for k, v in best_params.items()
            if hasattr(StrategyParams, k) or k in StrategyParams.__dataclass_fields__
        })

        strategy = IBBreakoutStrategy(strategy_params)
        for session in self.sessions:
            is_first = True
            for bar in session.bars:
                strategy.process_bar(bar, is_first)
                is_first = False

        best_metrics = calculate_metrics(strategy.get_trades())

        # Compile all trials
        all_trials = []
        for trial in self.study.trials:
            if trial.state == optuna.trial.TrialState.COMPLETE:
                trial_data = {
                    **trial.params,
                    'objective_value': trial.value,
                    'total_trades': trial.user_attrs.get('total_trades', 0),
                    'win_rate': trial.user_attrs.get('win_rate', 0),
                    'total_pnl': trial.user_attrs.get('total_pnl', 0),
                    'profit_factor': trial.user_attrs.get('profit_factor', 0),
                    'sharpe_ratio': trial.user_attrs.get('sharpe_ratio', 0),
                    'max_drawdown': trial.user_attrs.get('max_drawdown', 0),
                }
                all_trials.append(trial_data)

        results = BayesianOptimizationResults(
            best_params=best_params,
            best_value=self.study.best_value,
            best_metrics=best_metrics,
            all_trials=all_trials,
            n_trials=len(self.study.trials),
            objective=objective,
            total_time_seconds=total_time,
            ticker=self.ticker
        )

        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = self.output_dir / f"optuna_{self.ticker}_{timestamp}.csv"
        results.save_results(str(results_file))
        print(f"Results saved to: {results_file}")

        # Save best parameters
        best_file = self.output_dir / f"optuna_best_{self.ticker}_{timestamp}.json"
        with open(best_file, 'w') as f:
            json.dump(best_params, f, indent=2)
        print(f"Best parameters saved to: {best_file}")

        return results

    def _suggest_params_from_dict(self, trial_params: Dict[str, Any]) -> Dict[str, Any]:
        """Convert Optuna trial params to full params dict."""
        params = {}

        for name, config in self.parameter_space.parameters.items():
            if name in trial_params:
                params[name] = trial_params[name]
            else:
                params[name] = config.default

        return params

    def get_param_importances(self) -> Dict[str, float]:
        """
        Get parameter importance scores.

        Returns:
            Dict mapping parameter name to importance score
        """
        if self.study is None:
            return {}

        try:
            importances = optuna.importance.get_param_importances(self.study)
            return dict(importances)
        except Exception:
            return {}

    def plot_optimization_history(self, filepath: Optional[str] = None):
        """Plot optimization history."""
        if self.study is None:
            return

        try:
            fig = optuna.visualization.plot_optimization_history(self.study)
            if filepath:
                fig.write_html(filepath)
            return fig
        except Exception as e:
            print(f"Could not create plot: {e}")

    def plot_param_importances(self, filepath: Optional[str] = None):
        """Plot parameter importances."""
        if self.study is None:
            return

        try:
            fig = optuna.visualization.plot_param_importances(self.study)
            if filepath:
                fig.write_html(filepath)
            return fig
        except Exception as e:
            print(f"Could not create plot: {e}")

    def plot_contour(self, params: List[str], filepath: Optional[str] = None):
        """Plot contour for two parameters."""
        if self.study is None or len(params) != 2:
            return

        try:
            fig = optuna.visualization.plot_contour(self.study, params=params)
            if filepath:
                fig.write_html(filepath)
            return fig
        except Exception as e:
            print(f"Could not create plot: {e}")


def quick_optimize(
    ticker: str,
    data_dir: str = r"C:\Users\Warren\Downloads",
    n_trials: int = 50,
    objective: str = "sharpe_ratio"
) -> BayesianOptimizationResults:
    """
    Quick Bayesian optimization helper function.

    Args:
        ticker: Ticker symbol
        data_dir: Data directory
        n_trials: Number of trials
        objective: Objective to optimize

    Returns:
        BayesianOptimizationResults
    """
    optimizer = BayesianOptimizer(data_dir)
    optimizer.load_data(ticker)

    space = create_parameter_space("quick")
    return optimizer.optimize(n_trials=n_trials, parameter_space=space, objective=objective)


if __name__ == "__main__":
    if not OPTUNA_AVAILABLE:
        print("Install Optuna to run this example: pip install optuna")
        sys.exit(1)

    # Test Bayesian optimization
    optimizer = BayesianOptimizer(r"C:\Users\Warren\Downloads")
    optimizer.load_data("QQQ", "QQQ_1min_20231204_to_20241204_NT.txt")

    # Run optimization
    results = optimizer.optimize(
        n_trials=50,
        parameter_space=create_parameter_space("standard"),
        objective="sharpe_ratio"
    )

    print("\n" + results.summary())

    # Show parameter importances
    importances = optimizer.get_param_importances()
    if importances:
        print("\n\nPARAMETER IMPORTANCES:")
        print("-" * 40)
        for param, importance in sorted(importances.items(), key=lambda x: x[1], reverse=True):
            print(f"  {param}: {importance:.4f}")
