"""
Fast Grid Search Optimizer for IB Breakout Strategy.

This is now essentially an alias for the standard GridSearchOptimizer
since we fixed the parallelization there. Kept for backwards compatibility.
"""

from optimization.grid_search import GridSearchOptimizer

# FastGridSearchOptimizer is now just an alias
FastGridSearchOptimizer = GridSearchOptimizer
