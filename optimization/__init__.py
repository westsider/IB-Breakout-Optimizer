"""
Optimization module for IB Breakout Optimizer.

Provides parameter optimization using:
- Grid search for exhaustive exploration
- Bayesian optimization with Optuna
- Walk-forward analysis for robustness testing
"""

from .parameter_space import ParameterSpace, ParameterConfig
from .grid_search import GridSearchOptimizer
from .bayesian_optuna import BayesianOptimizer
from .walk_forward import WalkForwardAnalyzer

__all__ = [
    "ParameterSpace",
    "ParameterConfig",
    "GridSearchOptimizer",
    "BayesianOptimizer",
    "WalkForwardAnalyzer",
]
